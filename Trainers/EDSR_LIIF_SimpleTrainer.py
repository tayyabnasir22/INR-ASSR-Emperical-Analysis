from ModelFactories.EDSR_LIIF_Factory import EDSR_LIIF_Factory
from Pipelines.BaseTrainingPipeline import BaseTrainingPipeline
from Trainers.BaseTrainer import BaseTrainer
from Utilities.PathManager import PathManager
from Utilities.TrainingHelpers import TrainingHelpers

class EDSR_LIIF_SimpleTrainer(BaseTrainer):
    def TrainModel(self):
        # 1. Init thre required Pipeline
        pipeline = BaseTrainingPipeline()

        # 2. Build the model
        factory = EDSR_LIIF_Factory()
        factory.BuildModel(pipeline)

        # 3. Call the training
        logger, writer = PathManager.SetModelSavePath(
            pipeline.configurations.save_path, False
        )

        TrainingHelpers.Train(pipeline, logger, writer, 0, False)