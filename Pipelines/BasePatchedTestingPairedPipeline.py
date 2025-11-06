from Configurations.PatchedValidationConfigurations import PatchedValidationConfigurations
from Configurations.ValidationDataConfigurations import ValidationDataConfigurations
from Models.BenchmarkType import BenchmarkType
from DataProcessors.PairedImageFolder import PairedImageFolders
from DataProcessors.SRImplicitPairedPatched import SRImplicitPairedPatched
from Models.NormalizerRange import NormalizerRange
from Pipelines.BasePatchedTestingPipeline import BasePatchedTestingPipeline
from Utilities.DataLoaders import DataLoaders
import os

class BasePatchedTestingPairedPipeline(BasePatchedTestingPipeline):
    def __init__(
        self,
        valid_data_path: str = './datasets/DIV2K/DIV2K_valid_HR',
        valid_data_pathScale: str = './datasets/DIV2K/DIV2K_valid_LRbicx2',
        model_load_path: str = './model_states',
        model_name: str = 'last.pth',
        total_example: int = 100,
        eval_scale: int = 4,
        eval_batch_size: int = 300,
        patch_size_valid: int = 48,
        benchmark: BenchmarkType = BenchmarkType.DIV2K,
        breakdown_patch_size: int = 100,
    ):
        self._valid_data_path = valid_data_path
        self._valid_data_pathScale = valid_data_pathScale
        self._model_load_path = model_load_path
        self._model_name = model_name
        self._total_example = total_example
        self._eval_scale = eval_scale
        self._eval_batch_Size = eval_batch_size
        self._patch_size_valid = patch_size_valid
        self._benchmark = benchmark
        
        self._breakdown_patch_size = breakdown_patch_size

    def CreateDataLoaders(self,):
        self.validation_data_loader = DataLoaders.GetTestingDataLoader(
            SRImplicitPairedPatched(
                dataset=PairedImageFolders(
                    # The scaled down version is to be passed as the first argument
                    self.configurations.data_configurations.base_folder2, 
                    self.configurations.data_configurations.base_folder,
                    self.configurations.breakdown_patch_size,
                ),
                inp_size=None,
            ),
            self.configurations.data_configurations.batch_size
        )
