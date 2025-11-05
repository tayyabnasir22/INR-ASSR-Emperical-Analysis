from Models.BenchmarkType import BenchmarkType
from DataProcessors.PairedImageFolder import PairedImageFolders
from DataProcessors.SRImplicitPaired import SRImplicitPaired
from Pipelines.BaseTestingPipeline import BaseTestingPipeline
from Utilities.DataLoaders import DataLoaders

class BaseTestingPairedPipeline(BaseTestingPipeline):
    def __init__(self):
        self._valid_data_path = './datasets/DIV2K_valid_HR'
        self._valid_data_pathScale = './datasets/DIV2K_valid_HR'
        self._model_load_path = './model_states'
        self._model_name = 'last.pth'
        self._total_example = 100
        self._eval_scale = 4
        self._eval_batch_Size = 300
        self._patch_size_valid = 48
        self._benchmark = BenchmarkType.DIV2K

    def CreateDataLoaders(self,):
        self.validation_data_loader = DataLoaders.GetTestingDataLoader(
            SRImplicitPaired(
                dataset=PairedImageFolders(
                    self.configurations.data_configurations.base_folder, 
                    self.configurations.data_configurations.base_folder2,
                ),
                inp_size=None,
            ),
            self.configurations.data_configurations.batch_size
        )
