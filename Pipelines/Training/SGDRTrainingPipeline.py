from Models.BenchmarkType import BenchmarkType
from Models.NormalizerRange import NormalizerRange
from Configurations.TrainingConfigurations import TrainingConfigurations
from Configurations.TrainingDataConfigurations import TrainingDataConfigurations
from Configurations.ValidationDataConfigurations import ValidationDataConfigurations
from Pipelines.Training.BaseTrainingPipeline import BaseTrainingPipeline
from Utilities.ModelAttributesManager import ModelAttributesManager
from torch.optim.optimizer import Optimizer
import torch.nn as nn

class SGDRTrainingPipeline(BaseTrainingPipeline):
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
        scale_range: list[int, int] = [1,4],
        total_examples: int = 800,
        epochs: int = 100,
        epoch_val: int = 1,
        epoch_save: int = 1,
        warmup_epochs: int = 50,
        lr_min: float = 2.e-6,
        benchmark_type: BenchmarkType = BenchmarkType.DIV2K,
    ):
        super().__init__(
            train_data_path=train_data_path,
            valid_data_path=valid_data_path,
            model_save_path=model_save_path,
            model_load_path=model_load_path,
            batch_size=batch_size,
            train_repeat=train_repeat,
            patch_size_train=patch_size_train,
            patch_size_valid=patch_size_valid,
            start_learning_rate=start_learning_rate,
            scale_range=scale_range,
            total_examples=total_examples,
            epochs=epochs,
            milestones_count=[],
            epoch_val=epoch_val,
            epoch_save=epoch_save,
            gamma_schedular=None,
            benchmark_type=benchmark_type,
        )

        self._warmup_epochs = warmup_epochs
        self._lr_min = lr_min

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
            lr_scheduler={'warmup_epochs': self._warmup_epochs, 'lr_min': self._lr_min},
            epochs=self._epochs,
            save_path=self._model_save_path,
            resume_path=self._model_load_path,
            epoch_val=self._epoch_val,
            epoch_save=self._epoch_save,
            monitor_metric='psnr',
        )

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
        self.lr_scheduler = ModelAttributesManager.CreateSGDRScheduler(
            optimizer=self.optimizer, 
            t_zero=self.configurations.lr_scheduler['warmup_epochs'],
            lr_min=self.configurations.lr_scheduler['lr_min'],
            last_epoch=self.start_epoch
        )
