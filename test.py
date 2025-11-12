from Models.BaseTestingConfiguration import BaseTestingConfiguration
from Models.DecoderType import DecoderType
from Models.EncoderType import EncoderType
from Models.RecipeType import RecipeType
from Models.SavedModelType import SavedModelType
from Models.TestingStrategy import TestingStrategy
from TestingOrchestrators.TestingOrchestrator import TestingOrchestrator

def main():
    base_config = BaseTestingConfiguration()

    # Encoder-Decoder details
    base_config.encoder = EncoderType.EDSR
    base_config.decoder = DecoderType.LIIF
    
    # Recipe args
    base_config.recipe = RecipeType.Simple
    base_config.input_patch = 48 
    base_config.scale_range = [1, 4]
    
    # Model args
    base_config.eval_batch_size = 300
    base_config.model_name = SavedModelType.Last
    base_config.test_strategy = TestingStrategy.Simple
    base_config.breakdown_patch_size = None
    base_config.overlap = None

    TestingOrchestrator.TestDIV2K(
        EncoderType.EDSR,
        DecoderType.HIIF,
        base_config
    )

    TestingOrchestrator.TestSet5(
        EncoderType.EDSR,
        DecoderType.HIIF,
        base_config
    )

    TestingOrchestrator.TestSet14(
        EncoderType.EDSR,
        DecoderType.HIIF,
        base_config
    )

    TestingOrchestrator.TestB100(
        EncoderType.EDSR,
        DecoderType.HIIF,
        base_config
    )

    TestingOrchestrator.TestUrban100(
        EncoderType.EDSR,
        DecoderType.HIIF,
        base_config
    )

    TestingOrchestrator.TestSVT(
        EncoderType.EDSR,
        DecoderType.HIIF,
        base_config
    )

    TestingOrchestrator.TestCelebA_HQ(
        EncoderType.EDSR,
        DecoderType.HIIF,
        base_config
    )


if __name__ == '__main__':
    main()