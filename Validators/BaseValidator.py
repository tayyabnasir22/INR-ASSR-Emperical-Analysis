from abc import ABC, abstractmethod
from ModelFactories.BaseModelFactory import BaseModelFactory
from Models.RunningAverage import RunningAverage
from Models.TestingStrategy import TestingStrategy
from Models.Timer import Timer
from Pipelines.BaseOverlapPatchedTestingPipeline import BaseOverlapPatchedTestingPipeline
from Pipelines.BasePatchedTestingPipeline import BasePatchedTestingPipeline
from Pipelines.BaseTestingPipeline import BaseTestingPipeline
from Utilities.LPIPSManager import LPIPSManager
from Utilities.Logger import Logger
from Utilities.PredictionHelpers import PredictionHelpers
import os
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

class BaseValidator(ABC):
    @abstractmethod
    def TestModel(self,):
        pass


    def _GetPredictionForSimple(self, pipeline: BaseTestingPipeline, lpips_model: LearnedPerceptualImagePatchSimilarity):
        return PredictionHelpers.EvaluteForTesting(
                pipeline.validation_data_loader, 
                pipeline.model, 
                lpips_model, 
                pipeline.configurations.data_configurations.input_nomrlizer_range, 
                pipeline.configurations.data_configurations.eval_batch_size, 
                pipeline.configurations.data_configurations.eval_scale, 
                pipeline.configurations.data_configurations.benchmark_type
            )

    def _GetPredictionForPatched(self, pipeline: BasePatchedTestingPipeline, lpips_model: LearnedPerceptualImagePatchSimilarity):
        return PredictionHelpers.EvaluteForPatchedTesting(
                pipeline.validation_data_loader, 
                pipeline.model, 
                lpips_model, 
                pipeline.configurations.data_configurations.input_nomrlizer_range, 
                pipeline.configurations.data_configurations.eval_batch_size, 
                pipeline.configurations.breakdown_patch_size,
                pipeline.configurations.data_configurations.eval_scale, 
                pipeline.configurations.data_configurations.benchmark_type
            )
    
    def _GetPredictionForOverlapPatched(self, pipeline: BaseOverlapPatchedTestingPipeline, lpips_model: LearnedPerceptualImagePatchSimilarity):
        return PredictionHelpers.EvaluteForOverlapPatchedTesting(
                pipeline.validation_data_loader, 
                pipeline.model, 
                lpips_model, 
                pipeline.configurations.data_configurations.input_nomrlizer_range, 
                pipeline.configurations.data_configurations.eval_batch_size,
                pipeline.configurations.breakdown_patch_size,
                pipeline.configurations.overlap_size,
                pipeline.configurations.data_configurations.eval_scale, 
                pipeline.configurations.data_configurations.benchmark_type
            )

    def _GetPrediction(self, pipeline: BaseTestingPipeline, lpips_model: LearnedPerceptualImagePatchSimilarity, test_type: TestingStrategy):
        if test_type == TestingStrategy.Simple:
            return self._GetPredictionForSimple(pipeline, lpips_model)
        elif test_type == TestingStrategy.Patched:
            return self._GetPredictionForPatched(pipeline, lpips_model)
        elif test_type == TestingStrategy.OverlappingPatched:
            return self._GetPredictionForOverlapPatched(pipeline, lpips_model)
        else:
            raise NotImplemented('Testing strategy not recognized')

    def _RunTests(self, pipeline: BaseTestingPipeline, factory: BaseModelFactory, test_type: TestingStrategy):
        factory.BuildModel(pipeline)

        # 3. Get the model for LPIPS eval
        lpips_model = LPIPSManager.GetBaseModelForLPIPS()

        # 4. Call the testing
        timer = Timer()
        results: dict[str, RunningAverage] = self._GetPrediction(pipeline, lpips_model, test_type)

        # 5. Pass the result to writer
        out_path = Logger.LogTestResultsToCSV(
            results, 
            pipeline.configurations.data_configurations.benchmark_type, 
            pipeline.configurations.data_configurations.eval_scale, 
            os.path.dirname(pipeline.configurations.model_path), 
            timer, 
            pipeline.configurations.model_path, 
            test_type
        )

        Logger.Log('Test results saved at: ', out_path)