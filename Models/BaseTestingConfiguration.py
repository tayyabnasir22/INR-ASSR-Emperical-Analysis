from dataclasses import dataclass
from Models.BenchmarkType import BenchmarkType
from Models.DecoderType import DecoderType
from Models.EncoderType import EncoderType
from Models.RecipeType import RecipeType
from Models.SavedModelType import SavedModelType
from Models.TestingStrategy import TestingStrategy
from typing import Optional

@dataclass
class BaseTestingConfiguration:
    encoder: EncoderType = EncoderType.EDSR
    decoder: DecoderType = DecoderType.LIIF
    # Recipe args
    recipe: RecipeType = RecipeType.Simple
    input_patch: int = 48 
    scale_range: list[int, int] = [1,4] 
    # Scale args
    eval_scale: int = 2
    # Dataset args
    benchmark: BenchmarkType = BenchmarkType.DIV2K
    valid_data_path: str = './datasets/DIV2K/DIV2K_valid_HR'
    valid_data_pathScale: Optional[str] = None
    total_example: int = 100
    # Model args
    eval_batch_size: int = 300
    model_name: SavedModelType = SavedModelType.Last
    test_strategy: TestingStrategy = TestingStrategy.Simple

    def to_dict(self):
        return self.__dict__.copy()