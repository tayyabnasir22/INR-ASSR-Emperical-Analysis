from Models.BenchmarkType import BenchmarkType
from Models.NormalizerRange import NormalizerRange
from Configurations.PatchedValidationConfigurations import PatchedValidationConfigurations
from Configurations.ValidationDataConfigurations import ValidationDataConfigurations
from DataProcessors.ImageFolder import ImageFolder
from DataProcessors.SRImplicitDownsampledPatched import SRImplicitDownsampledPatched
from Pipelines.BaseTestingPipeline import BaseTestingPipeline
from Utilities.DataLoaders import DataLoaders
import os

class BasePatchedTestingPipeline(BaseTestingPipeline):
    def __init__(
        self,
        valid_data_path: str = './datasets/DIV2K/DIV2K_valid_HR',
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
        self._valid_data_pathScale = ''
        self._model_load_path = model_load_path
        self._model_name = model_name
        self._total_example = total_example
        self._eval_scale = eval_scale
        self._eval_batch_Size = eval_batch_size
        self._patch_size_valid = patch_size_valid
        self._benchmark = benchmark
        self._breakdown_patch_size = breakdown_patch_size


    def LoadConfigurations(self,):
        self.configurations = PatchedValidationConfigurations(
            model_path=os.path.join(self._model_load_path, self._model_name),
            data_configurations=ValidationDataConfigurations(
                patch_size=self._patch_size_valid, 
                augment=False, 
                batch_size=1, 
                base_folder=self._valid_data_path, 
                repeat=1, 
                scale_range=[4,4], 
                input_nomrlizer_range=NormalizerRange(), 
                total_examples=self._total_example, 
                benchmark_type=self._benchmark, 
                eval_batch_size=self._eval_batch_Size, 
                eval_scale=self._eval_scale,
                base_folder2=self._valid_data_pathScale,
            ),
            save_path=self._model_load_path,
            breakdown_patch_size=self._breakdown_patch_size
        )

    def CreateDataLoaders(self,):
        self.validation_data_loader = DataLoaders.GetTestingDataLoader(
            SRImplicitDownsampledPatched(
                dataset=ImageFolder(
                    self.configurations.data_configurations.base_folder, 
                ),
                inp_size=None,
                scale_min=self.configurations.data_configurations.eval_scale,
                scale_max=self.configurations.data_configurations.eval_scale,
                patch_size=self.configurations.breakdown_patch_size,
                augment=self.configurations.data_configurations.augment,
            ),
            self.configurations.data_configurations.batch_size
        )

    