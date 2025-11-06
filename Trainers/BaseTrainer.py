from abc import ABC, abstractmethod
from ModelFactories.BaseModelFactory import BaseModelFactory
from Pipelines.Training.BaseTrainingPipeline import BaseTrainingPipeline
from Utilities.PathManager import PathManager
from Utilities.TrainingHelpers import TrainingHelpers

class BaseTrainer(ABC):
    @abstractmethod
    def TrainModel(self,):
        pass

    def _RunTrain(self, pipeline: BaseTrainingPipeline, factory: BaseModelFactory):
        factory.BuildModel(pipeline)

        # 3. Call the training
        logger, writer = PathManager.SetModelSavePath(
            pipeline.configurations.save_path, False
        )

        TrainingHelpers.Train(pipeline, logger, writer, 0, False)