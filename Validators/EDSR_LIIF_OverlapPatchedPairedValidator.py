from ModelFactories.EDSR_LIIF_Factory import EDSR_LIIF_Factory
from Models.TestingStrategy import TestingStrategy
from Pipelines.BaseOverlapPatchedTestingPairedPipeline import BaseOverlapPatchedTestingPairedPipeline
from Validators.BaseValidator import BaseValidator

class EDSR_LIIF_OverlapPatchedPairedValidator(BaseValidator):

    def TestModel(self,):
        # 1. Init thre required Pipeline
        pipeline = BaseOverlapPatchedTestingPairedPipeline()

        # 2. Build the model
        factory = EDSR_LIIF_Factory()

        self._RunTests(pipeline, factory, TestingStrategy.OverlappingPatched)