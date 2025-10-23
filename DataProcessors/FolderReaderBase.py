from torch.utils.data import Dataset

class FolderReaderBase(Dataset):
    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
