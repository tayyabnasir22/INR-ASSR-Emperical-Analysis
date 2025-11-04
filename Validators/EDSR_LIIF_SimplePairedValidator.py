from ModelFactories.EDSR_LIIF_Factory import EDSR_LIIF_Factory
from Models.RunningAverage import RunningAverage
from Models.TestingStrategy import TestingStrategy
from Models.Timer import Timer
from Pipelines.BaseTestingPairedPipeline import BaseTestingPairedPipeline
from Utilities.LPIPSManager import LPIPSManager
from Utilities.Logger import Logger
from Validators.BaseValidator import BaseValidator
from Utilities.PredictionHelpers import PredictionHelpers
import os

class EDSR_LIIF_SimplePairedValidator(BaseValidator):

    def TestModel(self,):
        # 1. Init thre required Pipeline
        pipeline = BaseTestingPairedPipeline()

        # 2. Build the model
        factory = EDSR_LIIF_Factory()
        factory.BuildModel(pipeline)

        # 3. Get the model for LPIPS eval
        lpips_model = LPIPSManager.GetBaseModelForLPIPS()

        # 4. Call the testing
        timer = Timer()
        results: dict[str, RunningAverage] = PredictionHelpers.EvaluteForTesting(
            pipeline.validation_data_loader, 
            pipeline.model, 
            lpips_model, 
            pipeline.configurations.data_configurations.input_nomrlizer_range, 
            pipeline.configurations.data_configurations.eval_batch_size, 
            pipeline.configurations.data_configurations.eval_scale, 
            pipeline.configurations.data_configurations.benchmark_type
        )

        # 5. Pass the result to writer
        out_path = Logger.LogTestResultsToCSV(
            results, 
            pipeline.configurations.data_configurations.benchmark_type, 
            pipeline.configurations.data_configurations.eval_scale, 
            os.path.dirname(pipeline.configurations.model_path), 
            timer, 
            pipeline.configurations.model_path, 
            TestingStrategy.Simple
        )

        Logger.Log('Test results saved at: ', out_path)