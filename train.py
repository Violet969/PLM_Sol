import argparse
import yaml
from models import *  # For loading classes specified in config
from torch.optim import *  # For loading optimizer specified in config
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from datasets.embeddings_dataset import EmbeddingsDataset
from datasets.transforms import *
import os
from solver import Solver
from utils.general import padded_permuted_collate, seed_all


def train(args):
    
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/'+args.exp_name):
        os.makedirs('outputs/'+args.exp_name)
    if not os.path.exists('outputs/'+args.exp_name+'/'+'models'):
        os.makedirs('outputs/'+args.exp_name+'/'+'models')

    seed_all(args.seed)
    transform = transforms.Compose([SolubilityToInt(), ToTensor()])
    train_set = EmbeddingsDataset(args.train_embeddings, args.train_remapping, args.unknown_solubility,
                                               max_length=args.max_length, key_format=args.key_format,
                                              embedding_mode=args.embedding_mode, transform=transform)
    val_set = EmbeddingsDataset(args.val_embeddings, args.val_remapping, args.unknown_solubility,
                                            key_format=args.key_format, max_length=args.max_length,
                                            embedding_mode=args.embedding_mode, transform=transform)
    
    if len(train_set[0][0].shape) == 2:  # if we have per residue embeddings they have an additional length dim
        collate_function = padded_permuted_collate
    else:  # if we have reduced sequence wise embeddings use the default collate function by passing None
        collate_function = None

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_function,drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,shuffle=True, collate_fn=collate_function,drop_last=True)

    # Needs "from models import *" to work
    model = globals()[args.model_type](embeddings_dim=train_set[0][0].shape[-1], **args.model_parameters)
    print('trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Needs "from torch.optim import *" and "from models import *" to work
    solver = Solver(model, args, globals()[args.optimizer])
    solver.train(train_loader, val_loader, eval_data=val_set)

    if args.eval_on_test:
        test_set = EmbeddingsDataset(args.test_embeddings, args.test_remapping, args.unknown_solubility,
                                                 key_format=args.key_format, embedding_mode=args.embedding_mode,
                                                 transform=transform)
        solver.evaluation(test_set, filename='test_set_after_train')


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=argparse.FileType(mode='r'), default='configs/inference2.yaml')
    p.add_argument('--num_epochs', type=int, default=2500, help='number of times to iterate through all samples')
    p.add_argument('--batch_size', type=int, default=1024, help='samples that will be processed in parallel')
    p.add_argument('--patience', type=int, default=50, help='stop training after no improvement in this many epochs')
    p.add_argument('--min_train_acc', type=int, default=0, help='dont stop training before reaching this acc')
    p.add_argument('--n_draws', type=int, default=200, help='number of times to sample for estimation of stderr')
    p.add_argument('--seed', type=int, default=123, help='seed for reproducibility')
    p.add_argument('--optimizer', type=str, default='Adam', help='Class name of torch.optim like [Adam, SGD, AdamW]')
    p.add_argument('--optimizer_parameters', type=dict, help='parameters with keywords of the chosen optimizer like lr')
    p.add_argument('--log_iterations', type=int, default=-1,
                   help='log every log_iterations iterations (-1 for only logging after each epoch)')
    p.add_argument('--checkpoint', type=str, help='path to directory that contains a checkpoint')

    p.add_argument('--model_type', type=str, default='FFN', help='Classname of one of the models in the models dir')
    p.add_argument('--model_parameters', type=dict, help='dictionary of model parameters')
    p.add_argument('--loss_function', type=str, default='LocCrossEntropy',
                   help='Classname of one of the loss functions models/loss_functions.py')
    p.add_argument('--target', type=str, default='loc', help='to predict solubility or localization [loc,sol]')
    
    p.add_argument('--solubility_loss', type=float, default=0,
                   help='how much the loss of the solubility will be weighted')
    p.add_argument('--unknown_solubility', type=bool, default=True,
                   help='whether or not to include sequences with unknown solubility in the dataset')
    p.add_argument('--max_length', type=int, default=6000, help='maximum lenght of sequences that will be used for '
                                                                'training when using embedddings of variable length')
    p.add_argument('--embedding_mode', type=str, default='lm',
                   help='type of embedding to use (lm means Language model) [lm, onehot, profile]')

    p.add_argument('--eval_on_test', type=bool, default=True, help='runs evaluation on test set if true')
    p.add_argument('--train_embeddings', type=str, default='data/embeddings/train.h5',
                   help='.h5 or .h5py file with keys fitting the ids in the corresponding fasta remapping file')
    p.add_argument('--train_remapping', type=str, default='data/embeddings/train_remapped.fasta',
                   help='fasta file with remappings by bio_embeddings for the keys in the corresponding .h5 file')
    p.add_argument('--val_embeddings', type=str, default='data/embeddings/val.h5',
                   help='.h5 or .h5py file with keys fitting the ids in the corresponding fasta remapping file')
    p.add_argument('--val_remapping', type=str, default='data/embeddings/val_remapped.fasta',
                   help='fasta file with remappings by bio_embeddings for the keys in the corresponding .h5 file')
    p.add_argument('--test_embeddings', type=str, default='data/embeddings/test.h5',
                   help='.h5 or .h5py file with keys fitting the ids in the corresponding fasta remapping file')
    p.add_argument('--test_remapping', type=str, default='data/embeddings/test_remapped.fasta',
                   help='fasta file with remappings by bio_embeddings for the keys in the corresponding .h5 file')
    p.add_argument('--key_format', type=str, default='hash',
                   help='the formatting of the keys in the h5 file [fasta_descriptor_old, fasta_descriptor, hash]')
    p.add_argument('--exp_name', type=str, default='exp', metavar='N',help='Name of the experiment')
    args = p.parse_args()
    
    if args.config:
        data = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in data.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
    return args


if __name__ == '__main__':
    train(parse_arguments())
    