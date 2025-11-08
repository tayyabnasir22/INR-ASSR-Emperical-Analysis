from Models.BenchmarkType import BenchmarkType
from Models.DecoderType import DecoderType
from Models.EncoderType import EncoderType
from Models.RecipeType import RecipeType
from Models.SavedModelType import SavedModelType
from Models.TestingStrategy import TestingStrategy
from TestingOrchestrators.ValidatorBuilder import ValidatorBuilder

class EDSR_HIIF_Validator:
    
    @staticmethod
    def Test(
        # Recipe args
        recipe: RecipeType,
        input_patch: int, 
        scale_range: list[int], 
        # Scale args
        eval_scale=2,
        # Dataset args
        benchmark=BenchmarkType.DIV2K,
        valid_data_path='./datasets/DIV2K/DIV2K_valid_HR',
        valid_data_pathScale=None,
        total_example=100,
        # Model args
        eval_batch_size=300,
        model_name: SavedModelType = SavedModelType.Last,
        test_strategy: TestingStrategy = TestingStrategy.Simple,
        # Patch information
        breakdown_patch_size: int = None,
        overlap: int = None,
    ):
        # 1. Define the encoder and decoder
        encoder = EncoderType.EDSR
        decoder = DecoderType.HIIF

        ValidatorBuilder.BuildAndRunValidation(
            encoder,
            decoder,
            recipe,
            input_patch,
            scale_range,
            eval_scale,
            benchmark,
            valid_data_path,
            valid_data_pathScale,
            total_example,
            eval_batch_size,
            model_name,
            test_strategy,
            breakdown_patch_size,
            overlap,
        )

        