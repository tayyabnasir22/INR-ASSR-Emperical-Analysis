from dataclasses import dataclass
from Models.NormalizerRange import NormalizerRange

@dataclass
class TrainingDataConfigurations:
    patch_size: int
    augment: bool
    batch_size: int
    base_folder: str
    repeat: int
    scale_range: tuple[int, int]
    input_nomrlizer_range: NormalizerRange
    total_examples: int