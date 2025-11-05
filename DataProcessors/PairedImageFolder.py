from DataProcessors.FolderReaderBase import FolderReaderBase
from DataProcessors.ImageFolder import ImageFolder

class PairedImageFolders(FolderReaderBase):

    def __init__(self, scaled_down_path, hr_path, **kwargs):
        self.dataset_1 = ImageFolder(scaled_down_path, **kwargs)
        self.dataset_2 = ImageFolder(hr_path, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]