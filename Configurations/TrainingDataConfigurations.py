from dataclasses import dataclass
from Configurations.NormalizerRange import NormalizerRange
from DataProcessors.FolderReaderBase import FolderReaderBase
from DataProcessors.SRDataProcessorBase import SRDataProcessorBase

@dataclass
class TrainingDataConfigurations:
    file_reader: FolderReaderBase
    file_processor: SRDataProcessorBase
    patch_size: int
    augment: bool
    batch_size: int
    base_folder: str
    repeat: int
    scale_range: tuple[int, int]
    input_nomrlizer_range: NormalizerRange
