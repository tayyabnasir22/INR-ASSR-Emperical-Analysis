from torch.utils.data import Dataset

class SRDataProcessorBase(Dataset):
    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass