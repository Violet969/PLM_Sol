import random
from typing import List, Tuple
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
from models import *  # imports all classes in the models directory

SOLUBILITY = ['0', '1']

AMINO_ACIDS = {'A': 0,
               'R': 1,
               'N': 2,
               'D': 3,
               'C': 4,
               'Q': 5,
               'E': 6,
               'G': 7,
               'H': 8,
               'I': 9,
               'L': 10,
               'K': 11,
               'M': 12,
               'F': 13,
               'P': 14,
               'S': 15,
               'T': 16,
               'W': 17,
               'Y': 18,
               'V': 19,
               'U': 20,
               'X': 21,
               'B': 22,
               'J': 23,
               'Z': 24}


def seed_all(seed):
    if not seed:
        seed = 0

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

def padded_permuted_collate(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]]) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    Takes list of tuples with embeddings of variable sizes and pads them with zeros
    Args:
        batch: list of tuples with embeddings and the corresponding label

    Returns: tuple of tensor of embeddings with [batchsize, length_of_longest_sequence, embeddings_dim]
    and tensor of labels [batchsize, labels_dim] and metadate collated according to default collate

    """
    embeddings = [item[0] for item in batch]
    # localization = torch.tensor([item[1] for item in batch])
    solubility = torch.tensor([item[1] for item in batch])
    metadata = [item[2] for item in batch]
    metadata = torch.utils.data.dataloader.default_collate(metadata)
    embeddings = pad_sequence(embeddings, batch_first=True)
    return embeddings.permute(0, 2, 1), solubility, metadata

def predict_padded_permuted_collate(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]]) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    Takes list of tuples with embeddings of variable sizes and pads them with zeros
    Args:
        batch: list of tuples with embeddings and the corresponding label

    Returns: tuple of tensor of embeddings with [batchsize, length_of_longest_sequence, embeddings_dim]
    and tensor of labels [batchsize, labels_dim] and metadate collated according to default collate

    """
    embeddings = [item[0] for item in batch]
    # localization = torch.tensor([item[1] for item in batch])
    metadata = [item[1] for item in batch]
    metadata = torch.utils.data.dataloader.default_collate(metadata)
    embeddings = pad_sequence(embeddings, batch_first=True)
    return embeddings.permute(0, 2, 1), metadata


def numpy_collate_to_reduced(batch: List[Tuple[np.array, np.array, np.array, dict]]) -> Tuple[
    np.array, np.array, np.array, dict]:
    """
    Takes list of tuples with embeddings of variable sizes and takes the mean over the length dimension
    Args:
        batch: list of tuples with embeddings and the corresponding label

    Returns: tuple of np.arrays of embeddings with [batchsize, embeddings_dim] and the rest in batched form

    """
    embeddings = [np.array(item[0]).mean(axis=-2) for item in batch]  # take mean over lenght dimension
    localization = [np.array(item[1]) for item in batch]
    solubility = [item[2] for item in batch]
    metadata = [item[3] for item in batch]
    metadata = torch.utils.data.dataloader.default_collate(metadata)
    return embeddings, localization, solubility, metadata


def numpy_collate_for_reduced(batch: List[Tuple[np.array, np.array, np.array, dict]]) -> Tuple[
    np.array, np.array, np.array, dict]:
    """
    Collate function for reduced per protein embedding that returns numpy arrays intead of tensors
    Args:
        batch: list of tuples with embeddings and the corresponding label

    Returns: tuple of np.arrays of embeddings with [batchsize, embeddings_dim] and the rest in batched form

    """
    embeddings = [np.array(item[0]) for item in batch]
    localization = [np.array(item[1]) for item in batch]
    solubility = [item[2] for item in batch]
    metadata = [item[3] for item in batch]
    metadata = torch.utils.data.dataloader.default_collate(metadata)
    return embeddings, localization, solubility, metadata


def packed_padded_collate(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Takes list of tuples with embeddings of variable sizes and pads them with zeros
    Args:
        batch: list of tuples with embeddings and the corresponding label

    Returns: tuple of tensor of embeddings with [batchsize, length_of_longest_sequence, embeddings_dim]
    and tensor of labels [batchsize, labels_dim]

    """
    embeddings = [item[0] for item in batch]
    localization = torch.tensor([item[1] for item in batch])
    solubility = torch.tensor([item[1] for item in batch]).float()
    embeddings = pad_sequence(embeddings, batch_first=True)
    return embeddings.permute(0, 2, 1), localization, solubility


def normalize(arr):
    arr = arr - arr.min()
    arr = arr / arr.max()
    return arr
