from Models.DecoderType import DecoderType
from Models.EncoderType import EncoderType
from TestingOrchestrators.TestingOrchestrator import TestingOrchestrator

def main():
    TestingOrchestrator.TestDIV2K(
        EncoderType.EDSR,
        DecoderType.HIIF
    )


if __name__ == '__main__':
    main()