from DataProcessors.FolderReaderBase import FolderReaderBase
from DataProcessors.SRDataProcessorBase import SRDataProcessorBase
from torch.utils.data import DataLoader

class DataLoaders:
    @staticmethod
    def GetTrainingDataLoader(data_processor: SRDataProcessorBase, batch_size: int = 16):
        return DataLoader(data_processor, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    @staticmethod
    def GetValidationDataLoader(data_processor: SRDataProcessorBase, batch_size: int = 16):
        return DataLoader(data_processor, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    @staticmethod
    def GetTestingDataLoader(data_processor: SRDataProcessorBase, batch_size: int = 16):
        return DataLoader(data_processor, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)