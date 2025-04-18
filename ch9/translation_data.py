from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        pass
        return 1

    def __getitem__(self, idx):
        pass
        return idx