from torch.utils.data import Dataset
from torch import nn
import torch

from timemachine.timemachine_process import Vocabulary


class TimeDataset(Dataset):
    def __init__(self, data, size=640):
        self.data = data
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx:idx + 6], self.data[idx + 6]


class TimeMachineDataset(Dataset):
    def __init__(self, corpus: list[int], seq_len=5):
        self.corpus = corpus
        self.seq_len = seq_len

    def __len__(self):
        return len(self.corpus) - self.seq_len

    def __getitem__(self, idx):
        e = idx + self.seq_len
        return (
            torch.tensor(self.corpus[idx:e], dtype=torch.long),
            torch.tensor(self.corpus[idx + 1:e + 1], dtype=torch.long),
        )
