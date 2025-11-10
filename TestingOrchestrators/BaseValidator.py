from Models.BenchmarkType import BenchmarkType
from Models.RecipeType import RecipeType
from Models.SavedModelType import SavedModelType
from Models.TestingStrategy import TestingStrategy
from Utilities.PathManager import PathManager
from Validators.OverlapPatchedPairedValidator import OverlapPatchedPairedValidator
from Validators.OverlapPatchedValidator import OverlapPatchedValidator
from Validators.PatchedPairedValidator import PatchedPairedValidator
from Validators.PatchedValidator import PatchedValidator
from Validators.SimpleValidator import SimpleValidator
from Validators.SimplePairedValidator import SimplePairedValidator
from Models.DecoderType import DecoderType
from Models.EncoderType import EncoderType

class BaseValidator:
    @staticmethod
    def GetValidator(
        encoder: EncoderType,
        decoder: DecoderType,
        eval_scale,
        # Dataset args
        benchmark,
        valid_data_path,
        valid_data_pathScale,
        total_example,
        eval_batch_size,
        model_path: str,
        model_file: str,
    ):
        if valid_data_pathScale is None:
            return SimpleValidator(
                        encoder=encoder, 
                        decoder=decoder,
                        valid_data_path=valid_data_path,
                        model_load_path=model_path,
                        model_name=model_file,
                        total_example=total_example,
                        eval_scale=eval_scale,
                        eval_batch_size=eval_batch_size,
                        patch_size_valid=48, # don't care
                        benchmark=benchmark
                    )
        else:
            return SimplePairedValidator(
                    encoder=encoder,
                    decoder=decoder,
                    valid_data_path=valid_data_path,
                    valid_data_pathScale=valid_data_pathScale,
                    model_load_path=model_path,
                    model_name=model_file,
                    total_example=total_example,
                    eval_scale=eval_scale,
                    eval_batch_size=eval_batch_size,
                    patch_size_valid=48, # don't care
                    benchmark=benchmark
                )

    @staticmethod
    def GetPatchedValiator(
        encoder: EncoderType,
        decoder: DecoderType,
        eval_scale,
        # Dataset args
        benchmark,
        valid_data_path,
        valid_data_pathScale,
        total_example,
        eval_batch_size,
        model_path: str,
        model_file: str,
        # Patch information
        breakdown_patch_size: int = None,
    ):
        if valid_data_pathScale is None:
            return PatchedValidator(
                    encoder=encoder, 
                    decoder=decoder,
                    valid_data_path=valid_data_path,
                    model_load_path=model_path,
                    model_name=model_file,
                    total_example=total_example,
                    eval_scale=eval_scale,
                    eval_batch_size=eval_batch_size,
                    patch_size_valid=48, # don't care
                    benchmark=benchmark,
                    breakdown_patch_size=breakdown_patch_size,
                )
        else:
            return PatchedPairedValidator(
                    encoder=encoder,
                    decoder=decoder,
                    valid_data_path=valid_data_path,
                    valid_data_pathScale=valid_data_pathScale,
                    model_load_path=model_path,
                    model_name=model_file,
                    total_example=total_example,
                    eval_scale=eval_scale,
                    eval_batch_size=eval_batch_size,
                    patch_size_valid=48, # don't care
                    benchmark=benchmark,
                    breakdown_patch_size=breakdown_patch_size,
                )

    @staticmethod
    def GetPatchedOverlapValidator(
        encoder: EncoderType,
        decoder: DecoderType,
        eval_scale,
        # Dataset args
        benchmark,
        valid_data_path,
        valid_data_pathScale,
        total_example,
        eval_batch_size,
        model_path: str,
        model_file: str,
        # Patch information
        breakdown_patch_size: int = None,
        overlap: int = None,
    ):
        if valid_data_pathScale is None:
            return OverlapPatchedValidator(
                    encoder=encoder, 
                    decoder=decoder,
                    valid_data_path=valid_data_path,
                    model_load_path=model_path,
                    model_name=model_file,
                    total_example=total_example,
                    eval_scale=eval_scale,
                    eval_batch_size=eval_batch_size,
                    patch_size_valid=48, # don't care
                    benchmark=benchmark,
                    breakdown_patch_size=breakdown_patch_size,
                    overlap=overlap,
                )
        else:
            return OverlapPatchedPairedValidator(
                    encoder=encoder,
                    decoder=decoder,
                    valid_data_path=valid_data_path,
                    valid_data_pathScale=valid_data_pathScale,
                    model_load_path=model_path,
                    model_name=model_file,
                    total_example=total_example,
                    eval_scale=eval_scale,
                    eval_batch_size=eval_batch_size,
                    patch_size_valid=48, # don't care
                    benchmark=benchmark,
                    breakdown_patch_size=breakdown_patch_size,
                    overlap=overlap,
                )


    @staticmethod
    def BuildAndValidate(
        encoder: EncoderType,
        decoder: DecoderType,
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
        # 2. Get model path and save name
        model_path = PathManager.GetModelSavePath(
                        encoder,
                        decoder,
                        recipe,
                        input_patch,
                        scale_range
                    )
        model_file = model_name.value + '.pth'

        # 3. Build the required validator
        validator = None

        if test_strategy == TestingStrategy.Simple:
            validator = BaseValidator.GetValidator(
                encoder,
                decoder,
                eval_scale,
                benchmark,
                valid_data_path,
                valid_data_pathScale,
                total_example,
                eval_batch_size,
                model_path,
                model_file,
            )
        elif test_strategy == TestingStrategy.Patched:
            if breakdown_patch_size is None:
                raise Exception('Provide patch size')
            validator = BaseValidator.GetPatchedValiator(
                encoder,
                decoder,
                eval_scale,
                benchmark,
                valid_data_path,
                valid_data_pathScale,
                total_example,
                eval_batch_size,
                model_path,
                model_file,
                breakdown_patch_size,
            )
        elif test_strategy == TestingStrategy.OverlappingPatched:
            if breakdown_patch_size is None or overlap is None:
                raise Exception('Provide patch size, and overlap')
            validator = BaseValidator.GetPatchedOverlapValidator(
                encoder,
                decoder,
                eval_scale,
                benchmark,
                valid_data_path,
                valid_data_pathScale,
                total_example,
                eval_batch_size,
                model_path,
                model_file,
                breakdown_patch_size,
                overlap,
            )
        else:
            raise Exception('Testing strategy not defined')


        # 4. Run tests
        validator.TestModel()