from torch.utils.data import Dataset
from torch import nn


class TimeDataset(Dataset):
    def __init__(self, data, size=640):
        self.data = data
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx:idx + 6], self.data[idx + 6]
