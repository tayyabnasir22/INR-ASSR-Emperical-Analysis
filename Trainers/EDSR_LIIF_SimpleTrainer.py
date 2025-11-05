from ModelFactories.EDSR_LIIF_Factory import EDSR_LIIF_Factory
from Pipelines.BaseTrainingPipeline import BaseTrainingPipeline
from Trainers.BaseTrainer import BaseTrainer

class EDSR_LIIF_SimpleTrainer(BaseTrainer):
    def TrainModel(self):
        # 1. Init thre required Pipeline
        pipeline = BaseTrainingPipeline()

        # 2. Build the model
        factory = EDSR_LIIF_Factory()
        
        self._RunTrain(pipeline, factory)