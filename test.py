from Models.BaseTestingConfiguration import BaseTestingConfiguration
from Models.DecoderType import DecoderType
from Models.EncoderType import EncoderType
from TestingOrchestrators.TestingOrchestrator import TestingOrchestrator

def main():
    TestingOrchestrator.TestDIV2K(
        EncoderType.EDSR,
        DecoderType.HIIF,
        BaseTestingConfiguration()
    )

    TestingOrchestrator.TestSet5(
        EncoderType.EDSR,
        DecoderType.HIIF,
        BaseTestingConfiguration()
    )

    TestingOrchestrator.TestSet14(
        EncoderType.EDSR,
        DecoderType.HIIF,
        BaseTestingConfiguration()
    )

    TestingOrchestrator.TestB100(
        EncoderType.EDSR,
        DecoderType.HIIF,
        BaseTestingConfiguration()
    )

    TestingOrchestrator.TestUrban100(
        EncoderType.EDSR,
        DecoderType.HIIF,
        BaseTestingConfiguration()
    )

    TestingOrchestrator.TestSVT(
        EncoderType.EDSR,
        DecoderType.HIIF,
        BaseTestingConfiguration()
    )

    TestingOrchestrator.TestCelebA_HQ(
        EncoderType.EDSR,
        DecoderType.HIIF,
        BaseTestingConfiguration()
    )


if __name__ == '__main__':
    main()