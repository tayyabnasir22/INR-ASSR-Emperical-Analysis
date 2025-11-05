from ModelFactories.EDSR_LIIF_Factory import EDSR_LIIF_Factory
from Models.TestingStrategy import TestingStrategy
from Pipelines.BaseTestingPipeline import BaseTestingPipeline
from Validators.BaseValidator import BaseValidator

class EDSR_LIIF_SimpleValidator(BaseValidator):
    def TestModel(self,):
        # 1. Init thre required Pipeline
        pipeline = BaseTestingPipeline()

        # 2. Build the model
        factory = EDSR_LIIF_Factory()
        
        self._RunTests(pipeline, factory, TestingStrategy.Simple)