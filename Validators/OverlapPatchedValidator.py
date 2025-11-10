from ModelFactories.BaseModelFactory import BaseModelFactory
from Models.BenchmarkType import BenchmarkType
from Models.DecoderType import DecoderType
from Models.EncoderType import EncoderType
from Models.TestingStrategy import TestingStrategy
from Pipelines.Validation.BaseOverlapPatchedTestingPipeline import BaseOverlapPatchedTestingPipeline
from Validators.BaseValidator import BaseValidator

class OverlapPatchedValidator(BaseValidator):
    def __init__(
            self, 
            encoder: EncoderType, 
            decoder: DecoderType,
            valid_data_path: str,
            model_load_path: str,
            model_name: str,
            total_example: int,
            eval_scale: int,
            eval_batch_size: int,
            patch_size_valid: int,
            benchmark: BenchmarkType,
            breakdown_patch_size: int,
            overlap_size: int,
        ):
        super().__init__(encoder, decoder)
    
        self._valid_data_path = valid_data_path
        self._model_load_path = model_load_path
        self._model_name = model_name
        self._total_example = total_example
        self._eval_scale = eval_scale
        self._eval_batch_size = eval_batch_size
        self._patch_size_valid = patch_size_valid
        self._benchmark = benchmark
        self._breakdown_patch_size = breakdown_patch_size
        self._overlap_size = overlap_size
        
    def TestModel(self,):
        # 1. Init thre required Pipeline
        pipeline = BaseOverlapPatchedTestingPipeline(
            valid_data_path=self._valid_data_path,
            model_load_path=self._model_load_path,
            model_name=self._model_name,
            total_example=self._total_example,
            eval_scale=self._eval_scale,
            eval_batch_size=self._eval_batch_size,
            patch_size_valid=self._patch_size_valid,
            benchmark=self._benchmark,
            breakdown_patch_size=self._breakdown_patch_size,
            overlap_size=self._overlap_size,
        )

        # 2. Build the model
        factory = BaseModelFactory()
        
        self._RunTests(pipeline, factory, TestingStrategy.OverlappingPatched)