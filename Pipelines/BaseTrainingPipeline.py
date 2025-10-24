import os
from Configurations.BenchmarkType import BenchmarkType
from Configurations.NormalizerRange import NormalizerRange
from Configurations.TrainingConfigurations import TrainingConfigurations
from Configurations.TrainingDataConfigurations import TrainingDataConfigurations
from Configurations.ValidationDataConfigurations import ValidationDataConfigurations
from DataProcessors.ImageFolder import ImageFolder
from DataProcessors.SRImplicitDownsampled import SRImplicitDownsampled
from Pipelines.PipelineBase import PipelineBase
from Utilities.Evaluation import Evalutaion
from Utilities.ModelAttributesManager import ModelAttributesManager
from torch.optim.optimizer import Optimizer
import torch.nn as nn

class BaseTrainingPipeline(PipelineBase):
    def InitModel(self, model: nn.Module):
        self.model = model

    def LoadConfigurations(self,):
        self.configurations = TrainingConfigurations(
            optimizer={'learning_rate': 4.e-4},
            data_configurations=TrainingDataConfigurations(
                patch_size=48, 
                augment=True, 
                batch_size=32, 
                base_folder='./datasets/DIV2K_trin_HR', 
                repeat=40, 
                scale_range=[1,4], 
                input_nomrlizer_range=NormalizerRange(), 
                total_examples=800,
            ),
            validation_data_configurations=ValidationDataConfigurations(
                patch_size=48, 
                augment=False, 
                batch_size=1, 
                base_folder='./datasets/DIV2K_valid_HR', 
                repeat=1, 
                scale_range=[4,4], 
                input_nomrlizer_range=NormalizerRange(), 
                total_examples=100, 
                benchmark_type=BenchmarkType.DIV2K, 
                eval_batch_size=100, 
                eval_scale=4,
            ),
            lr_schedular={'milestones': [200, 400, 600, 800], 'gamma': 0.5},
            epochs=1000,
            save_path='./model_states',
            resume_path='./model_states',
            epoch_val=10,
            epoch_save=10,
            monitor_metric='psnr',
        )

    def CreateDataLoaders(self,):
        self.training_data_loader = SRImplicitDownsampled(
            dataset=ImageFolder(
                self.configurations.data_configurations.base_folder, 
                self.configurations.data_configurations.repeat
            ),
            inp_size=self.configurations.data_configurations.patch_size,
            scale_min=self.configurations.data_configurations.scale_range[0],
            scale_max=self.configurations.data_configurations.scale_range[1],
            augment=self.configurations.data_configurations.augment
        )
        self.validation_data_loader = SRImplicitDownsampled(
            dataset=ImageFolder(
                self.configurations.validation_data_configurations.base_folder, 
                self.configurations.validation_data_configurations.repeat
            ),
            inp_size=self.configurations.validation_data_configurations.patch_size,
            scale_min=self.configurations.validation_data_configurations.scale_range[0],
            scale_max=self.configurations.validation_data_configurations.scale_range[1],
            augment=self.configurations.validation_data_configurations.augment
        )

    def LoadModelWeights(self, ):
        self.saved_model = None
        if os.path.exists(self.configurations.resume_path):
            self.saved_model = torch.load(self.configurations.resume_path)
            self.model.load_state_dict(self.saved_model['model'])

    def InitTrainingRecipe(self, ):
        # 1. Set the start epoch
        self.start_epoch = 1 if self.saved_model is None else self.saved_model['epoch'] + 1

        # 2. Create/Load the optimizer
        if self.saved_model is not None:
            self.optimizer: Optimizer = ModelAttributesManager.CreateAdamOptimizer(self.model.parameters(), self.saved_model['optimizer'], self.configurations.optimizer['learning_rate'], load_sd=True)
        else:
            self.optimizer: Optimizer = ModelAttributesManager.CreateAdamOptimizer(self.model.parameters(), None ,self.configurations.optimizer['learning_rate'], load_sd=False)

        # 3. Create/Load the LR Schedular
        self.lr_schedular = ModelAttributesManager.CreateMultiStepLRSchedular(self.optimizer, self.configurations.lr_schedular['milestones'], self.configurations.lr_schedular['gamma'], self.start_epoch)

    def InitModelObjectives(self, ):
        self.loss = nn.L1Loss()
        self.metrics = [Evalutaion.PSNRTrain, Evalutaion.SSIMTrain]
