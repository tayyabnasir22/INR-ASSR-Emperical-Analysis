from ModelFactories.BaseModelFactory import BaseModelFactory
from Models.DecoderType import DecoderType
from Models.EncoderType import EncoderType
from Models.TestingStrategy import TestingStrategy
from Pipelines.Validation.BaseTestingPairedPipeline import BaseTestingPairedPipeline
from Validators.BaseValidator import BaseValidator

class EDSR_LIIF_SimplePairedValidator(BaseValidator):
    def __init__(self, encoder: EncoderType, decoder: DecoderType):
        super().__init__(encoder, decoder)

    def TestModel(self,):
        # 1. Init thre required Pipeline
        pipeline = BaseTestingPairedPipeline()

        # 2. Build the model
        factory = BaseModelFactory()

        self._RunTests(pipeline, factory, TestingStrategy.Simple)