from Models.DecoderType import DecoderType
from Models.EncoderType import EncoderType
from TrainingOrchestrators.TrainingOrchestrator import TrainingOrchestrator


def main():
    TrainingOrchestrator.TrainSimple(
        EncoderType.EDSR,
        DecoderType.HIIF
    )    

if __name__ == '__main__':
    main()