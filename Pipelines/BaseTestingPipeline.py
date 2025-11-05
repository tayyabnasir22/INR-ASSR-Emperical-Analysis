from Models.BenchmarkType import BenchmarkType
from Models.NormalizerRange import NormalizerRange
from Configurations.ValidationConfigurations import ValidationConfigurations
from Configurations.ValidationDataConfigurations import ValidationDataConfigurations
from DataProcessors.ImageFolder import ImageFolder
from DataProcessors.SRImplicitDownsampled import SRImplicitDownsampled
from Pipelines.PipelineBase import PipelineBase
import torch.nn as nn
import torch
from Utilities.DataLoaders import DataLoaders
from Utilities.PathManager import PathManager
import os

class BaseTestingPipeline(PipelineBase):
    def __init__(self):
        self._valid_data_path = './datasets/DIV2K_valid_HR'
        self._valid_data_pathScale = ''
        self._model_load_path = './model_states'
        self._model_name = 'last.pth'
        self._total_example = 100
        self._eval_scale = 4
        self._eval_batch_Size = 300
        self._patch_size_valid = 48
        self._benchmark = BenchmarkType.DIV2K

    def InitModel(self, model: nn.Module):
        self.model = model

    def LoadConfigurations(self,):
        self.configurations = ValidationConfigurations(
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
            logger_path=self._model_load_path
        )

    def CreateDataLoaders(self,):
        self.validation_data_loader = DataLoaders.GetTestingDataLoader(
            SRImplicitDownsampled(
                dataset=ImageFolder(
                    self.configurations.data_configurations.base_folder, 
                ),
                inp_size=None,
                scale_min=self.configurations.data_configurations.eval_scale,
                scale_max=self.configurations.data_configurations.eval_scale,
                augment=self.configurations.data_configurations.augment,
            ),
            self.configurations.data_configurations.batch_size
        )

    def LoadModelWeights(self, ):
        self.saved_model = torch.load(self.configurations.model_path)
        self.model.load_state_dict(self.saved_model['model'])
        self.model = torch.compile(self.model)

    def InitTrainingRecipe(self, ):
        pass

    def InitModelObjectives(self, ):
        pass