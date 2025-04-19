from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    def __init__(self, src_tensor, tar_tensor):
        super().__init__()
        self.src_tensor = src_tensor
        self.tar_tensor = tar_tensor
        self.length = len(src_tensor)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.src_tensor[idx], self.tar_tensor[idx]
