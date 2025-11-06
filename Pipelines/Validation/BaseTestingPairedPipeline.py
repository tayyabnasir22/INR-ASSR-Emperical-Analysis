from Models.BenchmarkType import BenchmarkType
from DataProcessors.PairedImageFolder import PairedImageFolders
from DataProcessors.SRImplicitPaired import SRImplicitPaired
from Pipelines.Validation.BaseTestingPipeline import BaseTestingPipeline
from Utilities.DataLoaders import DataLoaders

class BaseTestingPairedPipeline(BaseTestingPipeline):
    def __init__(
        self,
        valid_data_path: str = './datasets/DIV2K/DIV2K_valid_HR',
        valid_data_pathScale: str = './datasets/DIV2K/DIV2K_valid_LRbicx4', # Make sure the directory matches the scale value of eval_scale
        model_load_path: str = './model_states',
        model_name: str = 'last.pth',
        total_example: int = 100,
        eval_scale: int = 4,
        eval_batch_size: int = 300,
        patch_size_valid: int = 48,
        benchmark: BenchmarkType = BenchmarkType.DIV2K,
    ):
        super().__init__(
            valid_data_path=valid_data_path,
            model_load_path=model_load_path,
            model_name=model_name,
            total_example=total_example,
            eval_scale=eval_scale,
            eval_batch_size=eval_batch_size,
            patch_size_valid=patch_size_valid,
            benchmark=benchmark,
        )
        self._valid_data_pathScale = valid_data_pathScale


    def CreateDataLoaders(self,):
        self.validation_data_loader = DataLoaders.GetTestingDataLoader(
            SRImplicitPaired(
                dataset=PairedImageFolders(
                    # The scaled down version is to be passed as the first argument
                    self.configurations.data_configurations.base_folder2, 
                    self.configurations.data_configurations.base_folder,
                ),
                inp_size=None,
            ),
            self.configurations.data_configurations.batch_size
        )
