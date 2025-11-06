from Models.DecoderType import DecoderType
from Models.EncoderType import EncoderType
from Models.RecipeType import RecipeType
from Trainers.BaseTrainer import BaseTrainer

def main():
    # EDSR-LIIF with L1 loss, 48 patch, Multi-step LR schedular, and 1-4 scale range
    BaseTrainer(
        encoder=EncoderType.EDSR,
        decoder=DecoderType.LIIF,
        recipe=RecipeType.Simple,
        input_patch=48,
        scale_range=[1, 4]
    ).TrainModel()

if __name__ == '__main__':
    main()