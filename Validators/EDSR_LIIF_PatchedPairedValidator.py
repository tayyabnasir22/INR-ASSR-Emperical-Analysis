from ModelFactories.BaseModelFactory import BaseModelFactory
from Models.DecoderType import DecoderType
from Models.EncoderType import EncoderType
from Models.TestingStrategy import TestingStrategy
from Pipelines.Validation.BasePatchedTestingPairedPipeline import BasePatchedTestingPairedPipeline
from Validators.BaseValidator import BaseValidator

class EDSR_LIIF_PatchedPairedValidator(BaseValidator):
    def __init__(self, encoder: EncoderType, decoder: DecoderType):
        super().__init__(encoder, decoder)

    def TestModel(self,):
        # 1. Init thre required Pipeline
        pipeline = BasePatchedTestingPairedPipeline()

        # 2. Build the model
        factory = BaseModelFactory()

        self._RunTests(pipeline, factory, TestingStrategy.Patched)