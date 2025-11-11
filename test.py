from Models.DecoderType import DecoderType
from Models.EncoderType import EncoderType
from TestingOrchestrators.TestingOrchestrator import TestingOrchestrator

def main():
    TestingOrchestrator.TestDIV2K(
        EncoderType.EDSR,
        DecoderType.HIIF
    )

    TestingOrchestrator.TestSet5(
        EncoderType.EDSR,
        DecoderType.HIIF
    )

    TestingOrchestrator.TestSet14(
        EncoderType.EDSR,
        DecoderType.HIIF
    )

    TestingOrchestrator.TestB100(
        EncoderType.EDSR,
        DecoderType.HIIF
    )

    TestingOrchestrator.TestUrban100(
        EncoderType.EDSR,
        DecoderType.HIIF
    )

    TestingOrchestrator.TestSVT(
        EncoderType.EDSR,
        DecoderType.HIIF
    )

    TestingOrchestrator.TestCelebA_HQ(
        EncoderType.EDSR,
        DecoderType.HIIF
    )


if __name__ == '__main__':
    main()