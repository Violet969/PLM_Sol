import copy
import inspect
import os
import shutil
import pandas as pd
import pyaml
import torch
import numpy as np
from models import *
import sklearn.metrics as metrics
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.general import padded_permuted_collate,predict_padded_permuted_collate

class Solver():
    def __init__(self, model, args, optim=torch.optim.Adam, eval=False):
        self.optim = optim(list(model.parameters()), **args.optimizer_parameters)
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        if args.checkpoint and not eval:
            checkpoint = torch.load(os.path.join(args.checkpoint), map_location=self.device)
            
            self.model.load_state_dict(checkpoint)
            
        elif not eval:
            self.start_epoch = 0
            self.max_val_acc = 0  # running accuracy to decide whether or not a new model should be saved
        

    def train(self, train_loader: DataLoader, val_loader: DataLoader, eval_data=None):
        """
        Train and simultaneously evaluate on the val_loader and then estimate the stderr on eval_data if it is provided
        Args:
            train_loader: For training
            val_loader: For validation during training
            eval_data: For evaluation and estimating stderr after training

        Returns:

        """
        args = self.args
        io = IOStream('outputs/' + self.args.exp_name + '/run.log')
        
        lr_scheduler = ReduceLROnPlateau(self.optim, mode='min', factor=0.1, patience=1, verbose=True)
        best_test_acc = 0
        for epoch in range(self.start_epoch, args.num_epochs):  # loop over the dataset multiple times
            self.model.train()
            args = self.args
            train_pred = []
            train_true = []
            train_loss = 0.0
            count = 0.0
            for i, batch in enumerate(train_loader):
                embedding, sol, metadata = batch  # print('sol',sol)
                
                embedding, solubility, sol_known = embedding.to(self.device), sol.to(self.device), metadata['solubility_known'].to(self.device)
                # print('sol_known',sol_known)
                sequence_lengths = metadata['length'][:, None].to(self.device) 
                frequencies = metadata['frequencies'].to(self.device)  
                # create mask corresponding to the zero padding used for the shorter sequecnes in the batch. All values corresponding to padding are False and the rest is True.
                # print(metadata)
                mask = torch.arange(metadata['length'].max())[None, :] < metadata['length'][:,None]  # [batchsize, seq_len]
                outputs = self.model(embedding, mask=mask.to(self.device), sequence_lengths=sequence_lengths,
                                        frequencies=frequencies)
                # print('outputs',outputs)
                loss = F.binary_cross_entropy(outputs.squeeze(1), solubility.float())
                # print('loss',loss)
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()

                count += self.args.batch_size
                train_loss += loss.item() * self.args.batch_size
                train_true.append(solubility.cpu().numpy())

                train_pred.append((outputs.detach().cpu().numpy() >= 0.5))



            train_true = np.concatenate(train_true)
            train_pred = np.concatenate(train_pred)
            outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                     train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
            
            
            io.cprint(outstr)
            ####################
            # Val
            ####################

            self.model.eval()
            with torch.no_grad():  
                test_loss = 0.0
                count = 0.0
                test_pred = []
                test_true = []
                for batch in val_loader:
                    embedding, sol, metadata = batch  # print('sol',sol)
                
                    embedding, solubility, sol_known = embedding.to(self.device), sol.to(self.device), metadata['solubility_known'].to(self.device)
                    sequence_lengths = metadata['length'][:, None].to(self.device) 
                    frequencies = metadata['frequencies'].to(self.device)  

                    mask = torch.arange(metadata['length'].max())[None, :] < metadata['length'][:,None]  
                    outputs = self.model(embedding, mask=mask.to(self.device), sequence_lengths=sequence_lengths,
                                            frequencies=frequencies)
                    
                    loss = F.binary_cross_entropy(outputs.squeeze(1), solubility.float())

                    lr_scheduler.step(loss)
                    count += self.args.batch_size
                    test_loss += loss.item() * self.args.batch_size
                    test_true.append(solubility.cpu().numpy())
                    test_pred.append((outputs.detach().cpu().numpy() >= 0.5))
                test_true = np.concatenate(test_true)
                test_pred = np.concatenate(test_pred)
                test_acc = metrics.accuracy_score(test_true, test_pred)
                avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
                outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                                      test_loss*1.0/count,
                                                                                      test_acc,
                                                                                      avg_per_class_acc)
                io.cprint(outstr)
                if test_acc >= best_test_acc:
                    best_test_acc = test_acc
                    best_epoch = epoch
                    torch.save(self.model.state_dict(), 'outputs/{exp}/models/model-{epoch}.t7'.format(exp=args.exp_name,epoch=str(epoch)))
                    print("save weights!!!")
                    io.cprint("save weights!!!")
                    self.save_checkpoint(epoch + 1)



        if eval_data:  # do evaluation on the test data if a eval_data is provided
            # load checkpoint of best model to do evaluation
            checkpoint = torch.load(os.path.join('outputs/{exp}/models/model-{epoch}.t7'.format(exp=args.exp_name,epoch=str(best_epoch))), map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.evaluation(eval_data, filename='val_data_after_training')

            
    def evaluation(self, eval_dataset: Dataset):
        """
        Estimate the standard error on the provided dataset and write it to evaluation_val.txt in the run directory
        Args:
            eval_dataset: the dataset for which to estimate the stderr
            filename: string to append to the produced visualizations
            lookup_dataset: dataset used for embedding space similarity annotation transfer. If it is none, no annotation transfer will be done
            accuracy_threshold: accuracy to determine the distance below which the annotation transfer is used.

        Returns:

        """
       
        self.model.eval()
        if len(eval_dataset[0][0].shape) == 2:  # if we have per residue embeddings they have an additional length dim
            collate_function = padded_permuted_collate
        else:  # if we have reduced sequence wise embeddings use the default collate function by passing None
            collate_function = None
        io = IOStream('outputs/' + self.args.exp_name + '/run.log')
        data_loader = DataLoader(eval_dataset, batch_size=self.args.batch_size, collate_fn=collate_function)
        
        with torch.no_grad():  
            
            count = 0.0
            test_pred = []
            test_true = []
            for batch in data_loader:
                embedding, sol, metadata = batch  # print('sol',sol)

                embedding, solubility, sol_known = embedding.to(self.device), sol.to(self.device), metadata['solubility_known'].to(self.device)
                sequence_lengths = metadata['length'][:, None].to(self.device) 
                frequencies = metadata['frequencies'].to(self.device)  

                mask = torch.arange(metadata['length'].max())[None, :] < metadata['length'][:,None]  
                outputs = self.model(embedding, mask=mask.to(self.device), sequence_lengths=sequence_lengths,
                                        frequencies=frequencies)

                
                
                count += self.args.batch_size
                
                test_true.append(solubility.cpu().numpy())
                test_pred.append((outputs.detach().cpu().numpy() >= 0.5))
            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            test_acc = metrics.accuracy_score(test_true, test_pred)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
            outstr = 'Test acc: %.6f, Test avg acc: %.6f' % (test_acc,avg_per_class_acc)
            io.cprint(outstr)
                

    def predict_evaluation(self, eval_dataset: Dataset):
        """
        Estimate the standard error on the provided dataset and write it to evaluation_val.txt in the run directory
        Args:
            eval_dataset: the dataset for which to estimate the stderr
            filename: string to append to the produced visualizations
            lookup_dataset: dataset used for embedding space similarity annotation transfer. If it is none, no annotation transfer will be done
            accuracy_threshold: accuracy to determine the distance below which the annotation transfer is used.

        Returns:

        """
        
        self.model.eval()
        if len(eval_dataset[0][0].shape) == 2:  # if we have per residue embeddings they have an additional length dim
            collate_function = predict_padded_permuted_collate
        else:  # if we have reduced sequence wise embeddings use the default collate function by passing None
            collate_function = None

        data_loader = DataLoader(eval_dataset, batch_size=self.args.batch_size, collate_fn=collate_function)
        identifiers =[]
        sequences = []
        predictions = []
        
        with torch.no_grad():  
                
                count = 0.0
                test_pred = []
                
                for batch in data_loader:
                    # print(batch)
                    
                    embedding, metadata = batch  # print('sol',sol)
                
                    embedding = embedding.to(self.device)
                    sequence_lengths = metadata['length'][:, None].to(self.device) 
                    frequencies = metadata['frequencies'].to(self.device)  
                    
                    mask = torch.arange(metadata['length'].max())[None, :] < metadata['length'][:,None]  
                    outputs = self.model(embedding, mask=mask.to(self.device), sequence_lengths=sequence_lengths,
                                            frequencies=frequencies)
                    
                    identifiers.append(metadata['id'])
                    sequences.append(metadata['sequence'])
                    predictions.append(outputs.detach().cpu().numpy())

                    
                    count += self.args.batch_size
                    
                    test_pred.append((outputs.detach().cpu().numpy() >= 0.5))
                
                test_pred = np.concatenate(test_pred)
        
        prediction_result = pd.DataFrame(columns=['protein_ID','sequence','predict_result'])
        # print('identifiers',identifiers)
        prediction_result['protein_ID'] = [s for i in identifiers for s in i]
        prediction_result['sequence'] = [s for i in sequences for s in i]
        prediction_result['predict_result'] = [a for i in predictions for s in i for a in s]
        
        prediction_result.to_csv('protTrans_prediction_result.csv')
       
    def save_checkpoint(self, epoch: int):
        """
        Saves checkpoint of model in the logdir of the summarywriter/ in the used rundir
        Args:
            epoch: current epoch from which the run will be continued if it is loaded

        Returns:

        """
        run_dir = 'outputs/{exp}/models/'.format(exp=self.args.exp_name)
        
        train_args = copy.copy(self.args)
        train_args.config = train_args.config.name
        pyaml.dump(train_args.__dict__, open(os.path.join(run_dir, 'train_arguments.yaml'), 'w'))
        shutil.copyfile(self.args.config.name, os.path.join(run_dir, os.path.basename(self.args.config.name)))

        # Get the class of the used model (works because of the "from models import *" calling the init.py in the models dir)
        model_class = globals()[type(self.model).__name__]
        source_code = inspect.getsource(model_class)  # Get the sourcecode of the class of the model.
        file_name = os.path.basename(inspect.getfile(model_class))
        with open(os.path.join(run_dir, file_name), "w") as f:
            f.write(source_code)
            
            
class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()
