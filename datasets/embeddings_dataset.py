from typing import Tuple

import h5py
import torch
from Bio import SeqIO
from torch.utils.data import Dataset
import torch.nn.functional as F

from utils.general import AMINO_ACIDS


class EmbeddingsDataset(Dataset):
    def __init__(self, embeddings_path: str, remapped_sequences: str, unknown_solubility: bool = True,
                 key_format:str = 'hash',
                 max_length: int = float('inf'),
                 embedding_mode: str = 'lm',
                 transform=lambda x: x) -> None:
        """Create dataset.
        Args:
            embeddings_path:  path to .hdf5 .h5 file with embeddings as generated by the bio_embeddings pipeline, or the profiles (pssms) in an h5 file, or None if embedding_mode is 'onehot'
                https://github.com/sacdallago/bio_embeddings. Can either be a file of reduced fixed length embeddings or of
                variable length embeddings.
            remapped_sequences: remapped_sequences_file.fasta as generated by bio_embeddings where the ids in the
                annotations are the keys for the .h5 file in the embeddings path
            unknown_solubility: Whether or not to include sequences with unknown solubility in the dataset
            transform: Pytorch torchvision transforms that should be applied to each sample
            max_length: bigger sequences wont be taken into the dataset
            embedding_mode: ['lm', 'onehot', 'profiles'] what type of protein encoding to return (lm stands for language model) the embeddings_file needs to be either the lm embeddings or the profiles or none if embedding_mode is 'onehot'
        """
        super().__init__()
        self.transform = transform
        self.embedding_mode = embedding_mode
        if self.embedding_mode == 'lm' or self.embedding_mode == 'profiles':
            self.embeddings_file = h5py.File(embeddings_path, 'r')
        self.solubility_metadata_list = []
        # self.class_weights = torch.zeros(10)
        self.one_hot_enc = []
        for record in SeqIO.parse(open(remapped_sequences), 'fasta'):
            if key_format == 'hash':
                solubility = record.description.split(' ')[2].split('-')[-1]
                id = str(record.id)
            elif key_format == 'fasta_descriptor':
                solubility = record.description.split(' ')[2].split('-')[-1]
                id = str(record.description.split(' ')[0]).replace('.','_').replace('/','_')
            elif key_format == 'fasta_descriptor_old':
                solubility = record.description.split(' ')[1].split('-')[-1]
                id = str(record.description)
            else:
                raise Exception('Unknown key_format: ', key_format)
            if len(record.seq) <= max_length:
                if self.embedding_mode == 'onehot':
                    amino_acid_ids = []
                    for char in record.seq:
                        amino_acid_ids.append(AMINO_ACIDS[char])
                    one_hot_enc = F.one_hot(torch.tensor(amino_acid_ids), num_classes=len(AMINO_ACIDS))
                    self.one_hot_enc.append(one_hot_enc)
                frequencies = torch.zeros(25)
                for i, aa in enumerate(AMINO_ACIDS):
                    frequencies[i] = str(record.seq).count(aa)
                frequencies /= len(record.seq)
                metadata = {'id': id,
                            'sequence': str(record.seq),
                            'length': len(record.seq),
                            'frequencies': frequencies,
                            'solubility_known': not (solubility == 'U')}

                # if unknown solubility is false only the sequences with known solubility are included
                if unknown_solubility or not (solubility == 'U'):
                    self.solubility_metadata_list.append(
                        {'solubility': solubility, 'metadata': metadata})


    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """retrieve single sample from the dataset

        Args:
            index: index of sample to retrieve

        Returns:
            embedding: either a one dimensional Tensor [embedding_size] if the provided embeddings_path is of reduced
            embeddings or [length_of_sequence, embeddings_size] if the h5 file contains non reduced embeddings
            localization: localization in the format specified by the given transform.
            solubility: solubility as specified by a transform.
        """
        solubility_metadata = self.solubility_metadata_list[index]
        if self.embedding_mode == 'lm':
            embedding = self.embeddings_file[solubility_metadata['metadata']['id']][:]
        elif self.embedding_mode == 'profiles':
            embedding = self.embeddings_file[solubility_metadata['metadata']['sequence']][:]
        elif self.embedding_mode == 'onehot':
            embedding = self.one_hot_enc[index]
        else:
            raise Exception('embedding_mode {} not supported'.format(self.embedding_mode))

        embedding, solubility = self.transform(
            (embedding,solubility_metadata['solubility']))

        return embedding, solubility, solubility_metadata['metadata']

    def __len__(self) -> int:
        return len(self.solubility_metadata_list)

    
class Embeddings_predict_Dataset(Dataset):
    def __init__(self, embeddings_path: str, remapped_sequences: str,
                 key_format:str = 'hash',
                 max_length: int = float('inf'),
                 embedding_mode: str = 'lm',
                 transform=lambda x: x) -> None:
        
        super().__init__()
        self.transform = transform
        self.embedding_mode = embedding_mode
        if self.embedding_mode == 'lm' or self.embedding_mode == 'profiles':
            self.embeddings_file = h5py.File(embeddings_path, 'r')
        self.solubility_metadata_list = []
        # self.class_weights = torch.zeros(10)
        self.one_hot_enc = []
        for record in SeqIO.parse(open(remapped_sequences), 'fasta'):
            if key_format == 'hash':
                id = str(record.id)
            elif key_format == 'fasta_descriptor':
                id = str(record.description.split(' ')[0]).replace('.','_').replace('/','_')
            elif key_format == 'fasta_descriptor_old':
                id = str(record.description)
            else:
                raise Exception('Unknown key_format: ', key_format)
            if len(record.seq) <= max_length:
                if self.embedding_mode == 'onehot':
                    amino_acid_ids = []
                    for char in record.seq:
                        amino_acid_ids.append(AMINO_ACIDS[char])
                    one_hot_enc = F.one_hot(torch.tensor(amino_acid_ids), num_classes=len(AMINO_ACIDS))
                    self.one_hot_enc.append(one_hot_enc)
                frequencies = torch.zeros(25)
                for i, aa in enumerate(AMINO_ACIDS):
                    frequencies[i] = str(record.seq).count(aa)
                frequencies /= len(record.seq)
                metadata = {'id': id,
                            'sequence': str(record.seq),
                            'length': len(record.seq),
                            'frequencies': frequencies}
                self.solubility_metadata_list.append({'metadata': metadata})
                

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """retrieve single sample from the dataset

        Args:
            index: index of sample to retrieve

        Returns:
            embedding: either a one dimensional Tensor [embedding_size] if the provided embeddings_path is of reduced
            embeddings or [length_of_sequence, embeddings_size] if the h5 file contains non reduced embeddings
            localization: localization in the format specified by the given transform.
            solubility: solubility as specified by a transform.
        """
        solubility_metadata = self.solubility_metadata_list[index]
        # print(solubility_metadata)
        if self.embedding_mode == 'lm':
            embedding = self.embeddings_file[solubility_metadata['metadata']['id']][:]
        elif self.embedding_mode == 'profiles':
            embedding = self.embeddings_file[solubility_metadata['metadata']['sequence']][:]
        elif self.embedding_mode == 'onehot':
            embedding = self.one_hot_enc[index]
        else:
            raise Exception('embedding_mode {} not supported'.format(self.embedding_mode))

        embedding = self.transform(embedding)

        return embedding, solubility_metadata['metadata']

    def __len__(self) -> int:
        return len(self.solubility_metadata_list)
