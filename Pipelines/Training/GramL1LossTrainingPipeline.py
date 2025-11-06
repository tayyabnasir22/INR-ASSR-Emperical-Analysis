from Components.GramL1Loss import GramL1Loss
from Models.BenchmarkType import BenchmarkType
from Pipelines.Training.BaseTrainingPipeline import BaseTrainingPipeline
from Utilities.Evaluation import Evalutaion

class GramL1LossTrainingPipeline(BaseTrainingPipeline):
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
        milestones_count: int = 4,
        epoch_val: int = 1,
        epoch_save: int = 1,
        gamma_schedular: float = 0.5,
        benchmark_type: BenchmarkType = BenchmarkType.DIV2K,
        lmbda: float = 0.5,
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
            milestones_count=milestones_count,
            epoch_val=epoch_val,
            epoch_save=epoch_save,
            gamma_schedular=gamma_schedular,
            benchmark_type=benchmark_type,
        )
        self._lmbda = lmbda


    def InitModelObjectives(self, ):
        self.loss = GramL1Loss(lambda_texture=self._lmbda)
        self.metrics = [Evalutaion.PSNRTrain, Evalutaion.SSIMTrain]

