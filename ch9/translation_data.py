from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    def __init__(self, src_tensor, tar_tensor, src_len, tar_len):
        super().__init__()
        self.src_tensor = src_tensor
        self.tar_tensor = tar_tensor
        self.src_len = src_len
        self.tar_len = tar_len
        self.length = len(src_tensor)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # x y valid_y_len
        return self.src_tensor[idx], self.tar_tensor[idx], self.tar_len[idx]
