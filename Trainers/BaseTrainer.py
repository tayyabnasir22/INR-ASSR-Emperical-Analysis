from ModelFactories.BaseModelFactory import BaseModelFactory
from Models.DecoderType import DecoderType
from Models.EncoderType import EncoderType
from Models.RecipeType import RecipeType
from Pipelines.Training.BaseTrainingPipeline import BaseTrainingPipeline
from Pipelines.Training.GradientLossTrainingPipeline import GradientLossTrainingPipeline
from Pipelines.Training.GramL1LossTrainingPipeline import GramL1LossTrainingPipeline
from Pipelines.Training.SGDRTrainingPipeline import SGDRTrainingPipeline
from Utilities.PathManager import PathManager
from Utilities.TrainingHelpers import TrainingHelpers

class BaseTrainer:
    def __init__(self, encoder: EncoderType, decoder: DecoderType, recipe: RecipeType, input_patch: int = 48, scale_range: list[int, int] = [1, 4]):
        self._encoder = encoder
        self._decoder = decoder
        self._recipe = recipe
        self._input_patch = input_patch
        self._scale_range = scale_range

    def _GetPipeline(self,):
        if self._recipe == RecipeType.Simple:
            return BaseTrainingPipeline(
                scale_range=self._scale_range,
                patch_size_train=self._input_patch,
                patch_size_valid=self._input_patch,
            )
        elif self._recipe == RecipeType.GradLoss:
            return GradientLossTrainingPipeline(
                scale_range=self._scale_range,
                patch_size_train=self._input_patch,
                patch_size_valid=self._input_patch,
            )
        elif self._recipe == RecipeType.GradLoss:
            return GramL1LossTrainingPipeline(
                scale_range=self._scale_range,
                patch_size_train=self._input_patch,
                patch_size_valid=self._input_patch,
            )
        elif self._recipe == RecipeType.GradLoss:
            return SGDRTrainingPipeline(
                scale_range=self._scale_range,
                patch_size_train=self._input_patch,
                patch_size_valid=self._input_patch,
            )
        else:
            raise NotImplemented('Pipeline not found')

    def _GetModelFactory(self, pipeline: BaseTrainingPipeline):
        return BaseModelFactory().BuildModel(pipeline, self._encoder, self._decoder)

    def _RunTrain(self, pipeline: BaseTrainingPipeline, factory: BaseModelFactory):
        factory.BuildModel(pipeline)

        # 3. Call the training
        logger, writer = PathManager.SetModelSavePath(
            pipeline.configurations.save_path, False
        )

        TrainingHelpers.Train(pipeline, logger, writer, 0, False)

    def TrainModel(self,):
        # 1. Init thre required Pipeline
        pipeline = self._GetPipeline()

        # 2. Build the model
        factory = self._GetModelFactory(pipeline)

        # 3. Train the model using the factory
        self._RunTrain(pipeline, factory)
    