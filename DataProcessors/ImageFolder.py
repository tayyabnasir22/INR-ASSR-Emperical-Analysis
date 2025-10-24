import os
import json
from PIL import Image
from DataProcessors.FolderReaderBase import FolderReaderBase
from torchvision import transforms

class ImageFolder(FolderReaderBase):

    def __init__(self, root_path,
                 repeat=1):
        self.repeat = repeat

        file_names = sorted(os.listdir(root_path))
        
        self.files = []
        for file_name in file_names:
            file = os.path.join(root_path, file_name)

            self.files.append(file)

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        return transforms.ToTensor()(Image.open(x).convert('RGB'))