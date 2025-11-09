from Models.DecoderType import DecoderType
from Models.EncoderType import EncoderType
from Models.RecipeType import RecipeType
from Trainers.BaseTrainer import BaseTrainer

class TrainingOrchestrator:
    @staticmethod
    def TrainSimple(encoder: EncoderType, decoder: DecoderType):
        BaseTrainer(
            encoder=encoder,
            decoder=decoder,
            recipe=RecipeType.Simple,
            input_patch=48,
            scale_range=[1, 4]
        ).TrainModel()

    @staticmethod
    def TrainSimpleLargerPatch(encoder: EncoderType, decoder: DecoderType):
        BaseTrainer(
            encoder=encoder,
            decoder=decoder,
            recipe=RecipeType.Simple,
            input_patch=64,
            scale_range=[1, 4]
        ).TrainModel()

    @staticmethod
    def TrainSimpleLargerPatchScale(encoder: EncoderType, decoder: DecoderType):
        BaseTrainer(
            encoder=encoder,
            decoder=decoder,
            recipe=RecipeType.Simple,
            input_patch=64,
            scale_range=[1, 6]
        ).TrainModel()

    @staticmethod
    def TrainSGDR(encoder: EncoderType, decoder: DecoderType):
        BaseTrainer(
            encoder=encoder,
            decoder=decoder,
            recipe=RecipeType.SGDR,
            input_patch=48,
            scale_range=[1, 4]
        ).TrainModel()

    @staticmethod
    def TrainGradLoss(encoder: EncoderType, decoder: DecoderType):
        BaseTrainer(
            encoder=encoder,
            decoder=decoder,
            recipe=RecipeType.GradLoss,
            input_patch=48,
            scale_range=[1, 4]
        ).TrainModel()

    @staticmethod
    def TrainGramL1Loss(encoder: EncoderType, decoder: DecoderType):
        BaseTrainer(
            encoder=encoder,
            decoder=decoder,
            recipe=RecipeType.GramL1Loss,
            input_patch=48,
            scale_range=[1, 4]
        ).TrainModel()