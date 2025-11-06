from ModelFactories.EDSR_LIIF_Factory import EDSR_LIIF_Factory
from Models.TestingStrategy import TestingStrategy
from Pipelines.Validation.BaseOverlapPatchedTestingPipeline import BaseOverlapPatchedTestingPipeline
from Validators.BaseValidator import BaseValidator

class EDSR_LIIF_OverlapPatchedValidator(BaseValidator):
    def TestModel(self,):
        # 1. Init thre required Pipeline
        pipeline = BaseOverlapPatchedTestingPipeline()

        # 2. Build the model
        factory = EDSR_LIIF_Factory()
        
        self._RunTests(pipeline, factory, TestingStrategy.OverlappingPatched)