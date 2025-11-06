from Models.BenchmarkType import BenchmarkType
from DataProcessors.PairedImageFolder import PairedImageFolders
from DataProcessors.SRImplicitPaired import SRImplicitPaired
from Pipelines.BaseTestingPipeline import BaseTestingPipeline
from Utilities.DataLoaders import DataLoaders

class BaseTestingPairedPipeline(BaseTestingPipeline):
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
