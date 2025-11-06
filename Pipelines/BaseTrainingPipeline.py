import os
from Models.BenchmarkType import BenchmarkType
from Models.NormalizerRange import NormalizerRange
from Configurations.TrainingConfigurations import TrainingConfigurations
from Configurations.TrainingDataConfigurations import TrainingDataConfigurations
from Configurations.ValidationDataConfigurations import ValidationDataConfigurations
from DataProcessors.ImageFolder import ImageFolder
from DataProcessors.SRImplicitDownsampled import SRImplicitDownsampled
from Pipelines.PipelineBase import PipelineBase
from Utilities.DataLoaders import DataLoaders
from Utilities.Evaluation import Evalutaion
from Utilities.ModelAttributesManager import ModelAttributesManager
from torch.optim.optimizer import Optimizer
import torch.nn as nn
import torch

from Utilities.StatsHelpers import StatsHelpers

class BaseTrainingPipeline(PipelineBase):
    def __init__(
        self,
        train_data_path: str = './datasets/DIV2K_train_HR',
        valid_data_path: str = './datasets/DIV2K/DIV2K_valid_HR',
        model_save_path: str = './model_states',
        model_load_path: str = './model_states/last.pth',
        batch_size: int = 32,
        train_repeat: int = 40,
        patch_size_train: int = 48,
        patch_size_valid: int = 48,
        start_learning_rate: float = 4.e-4,
        scale_range: list[int, int] = [4,4],
        total_examples: int = 800,
        epochs: int = 100,
        milestones_count: int = 4,
        epoch_val: int = 1,
        epoch_save: int = 1,
        gamma_schedular: float = 0.5,
        benchmark_type: BenchmarkType = BenchmarkType.DIV2K,
    ):
        self._train_data_path = train_data_path
        self._valid_data_path = valid_data_path
        self._model_save_path = model_save_path
        self._model_load_path = model_load_path
        self._batch_Size = batch_size
        self._train_repeat = train_repeat
        self._patch_size_train = patch_size_train
        self._patch_size_valid = patch_size_valid

        # -1 to when starting fresh for the optimizers and schedulars to work to indicate 0->1 epoch
        # Last successfully completed epoch for example 50 if the 50th epoch was completed to mark 50->51 epoch
        # Make sure to add + 1 before training to save epoch completed number properly
        self.start_epoch = -1



        self._start_learning_rate = start_learning_rate
        self._scale_range = scale_range
        self._total_examples = total_examples
        self._epochs = epochs
        self._milestones_count = milestones_count
        self._epoch_val = epoch_val
        self._epoch_save = epoch_save
        self._gamma_schedular = gamma_schedular
        self._benchmark_type = benchmark_type

    def InitModel(self, model: nn.Module):
        self.model = model

    def LoadConfigurations(self,):
        self.configurations = TrainingConfigurations(
            optimizer={'learning_rate': self._start_learning_rate},
            data_configurations=TrainingDataConfigurations(
                patch_size=self._patch_size_train, 
                augment=True, 
                batch_size=self._batch_Size, 
                base_folder=self._train_data_path, 
                repeat=self._train_repeat, 
                scale_range=self._scale_range, 
                input_nomrlizer_range=NormalizerRange(), 
                total_examples=self._total_examples,
            ),
            validation_data_configurations=ValidationDataConfigurations(
                patch_size=self._patch_size_valid, 
                augment=False, 
                batch_size=1, 
                base_folder=self._valid_data_path, 
                repeat=1, 
                scale_range=[4,4], 
                input_nomrlizer_range=NormalizerRange(), 
                total_examples=100, 
                benchmark_type=self._benchmark_type, 
                eval_batch_size=100, 
                eval_scale=4,
                base_folder2='',
            ),
            lr_scheduler={'milestones': StatsHelpers.GetMulitStepMilestones(self._epochs, self._milestones_count), 'gamma': self._gamma_schedular},
            epochs=self._epochs,
            save_path=self._model_save_path,
            resume_path=self._model_load_path,
            epoch_val=self._epoch_val,
            epoch_save=self._epoch_save,
            monitor_metric='psnr',
        )

    def CreateDataLoaders(self,):
        self.training_data_loader = DataLoaders.GetTrainingDataLoader(
            SRImplicitDownsampled(
                dataset=ImageFolder(
                    self.configurations.data_configurations.base_folder, 
                    self.configurations.data_configurations.repeat
                ),
                inp_size=self.configurations.data_configurations.patch_size,
                scale_min=self.configurations.data_configurations.scale_range[0],
                scale_max=self.configurations.data_configurations.scale_range[1],
                augment=self.configurations.data_configurations.augment
            ),
            self.configurations.data_configurations.batch_size,
        )
        self.validation_data_loader = DataLoaders.GetValidationDataLoader(
            SRImplicitDownsampled(
                dataset=ImageFolder(
                    self.configurations.validation_data_configurations.base_folder, 
                    self.configurations.validation_data_configurations.repeat
                ),
                inp_size=self.configurations.validation_data_configurations.patch_size,
                scale_min=self.configurations.validation_data_configurations.scale_range[0],
                scale_max=self.configurations.validation_data_configurations.scale_range[1],
                augment=self.configurations.validation_data_configurations.augment
            ),
            self.configurations.validation_data_configurations.batch_size
        )

    def LoadModelWeights(self, ):
        self.saved_model = None
        if os.path.exists(self.configurations.resume_path):
            self.saved_model = torch.load(self.configurations.resume_path)
            self.model.load_state_dict(self.saved_model['model'])

    def InitTrainingRecipe(self, ):
        # 1. Set the start epoch
        self.start_epoch = self.start_epoch if self.saved_model is None else self.saved_model['epoch']

        # 2. Create/Load the optimizer
        if self.saved_model is not None:
            self.optimizer: Optimizer = ModelAttributesManager.CreateAdamOptimizer(
                self.model.parameters(), 
                self.saved_model['optimizer'], 
                self.configurations.optimizer['learning_rate'], 
                load_sd=True
            )
        else:
            self.optimizer: Optimizer = ModelAttributesManager.CreateAdamOptimizer(
                self.model.parameters(), 
                None,
                self.configurations.optimizer['learning_rate'], 
                load_sd=False
            )

        # 3. Create/Load the LR Scheduler
        self.lr_scheduler = ModelAttributesManager.CreateMultiStepLRScheduler(
            self.optimizer, 
            self.configurations.lr_scheduler['milestones'], 
            self.configurations.lr_scheduler['gamma'], 
            self.start_epoch
        )

    def InitModelObjectives(self, ):
        self.loss = nn.L1Loss()
        self.metrics = [Evalutaion.PSNRTrain, Evalutaion.SSIMTrain]
