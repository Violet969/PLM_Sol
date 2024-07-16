import copy

from models import *  # For loading classes specified in config
# from models.legacy import *  # For loading classes specified in config
from torch.optim import *  # For loading optimizer class that was used in the checkpoint
import os
import argparse
import yaml
import torch.nn as nn
from torchvision.transforms import transforms
from datasets.embeddings_dataset import Embeddings_predict_Dataset
from datasets.transforms import *
from solver import Solver


def inference(args):
    transform = transforms.Compose([Solubility_predict_ToInt(), predict_ToTensor()])

    data_set = Embeddings_predict_Dataset(args.embeddings, args.remapping,
                                             key_format=args.key_format,
                                             embedding_mode=args.embedding_mode,
                                             transform=transform)
    
    model: nn.Module = globals()[args.model_type](embeddings_dim=data_set[0][0].shape[-1], **args.model_parameters)

    # Needs "from torch.optim import *" and "from models import *" to work
    solver = Solver(model, args, globals()[args.optimizer])
    return solver.predict_evaluation(data_set)


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=argparse.FileType(mode='r'), default='configs/inference.yaml')
    p.add_argument('--checkpoints_list', default=[],
                   help='if there are paths specified here, they all are evaluated')
    p.add_argument('--batch_size', type=int, default=16, help='samples that will be processed in parallel')
    p.add_argument('--log_iterations', type=int, default=100, help='log every log_iterations (-1 for no logging)')
    p.add_argument('--embeddings', type=str, default='data/embeddings/val_reduced.h5',
                   help='.h5 or .h5py file with keys fitting the ids in the corresponding fasta remapping file')
    p.add_argument('--remapping', type=str, default='data/embeddings/val_remapped.fasta',
                   help='fasta file with remappings by bio_embeddings for the keys in the corresponding .h5 file')
    p.add_argument('--distance_threshold', type=float, default=-1.0,
                   help='cutoff similarity for when to do lookup and when to use denovo predictions. If negative, denovo predictions will always be used.')
    p.add_argument('--key_format', type=str, default='hash',
                   help='the formatting of the keys in the h5 file [fasta_descriptor_old, fasta_descriptor, hash]')


    args = p.parse_args()
    arg_dict = args.__dict__
    if args.config:
        data = yaml.load(args.config, Loader=yaml.FullLoader)
        for key, value in data.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
    return args


if __name__ == '__main__':
    original_args = copy.copy(parse_arguments())
    
    for checkpoint in original_args.checkpoints_list:
        args = copy.copy(original_args)
        arg_dict = args.__dict__
        arg_dict['checkpoint'] = checkpoint
        # get the arguments from the yaml config file that is saved in the runs checkpoint
        data = yaml.load(open(os.path.join('./model_param/train_arguments.yml'), 'r'), Loader=yaml.FullLoader)
        for key, value in data.items():
            if key not in args.__dict__.keys():
                if isinstance(value, list):
                    for v in value:
                        arg_dict[key].append(v)
                else:
                    arg_dict[key] = value
        # call teh actual inference
        inference(args)
   
