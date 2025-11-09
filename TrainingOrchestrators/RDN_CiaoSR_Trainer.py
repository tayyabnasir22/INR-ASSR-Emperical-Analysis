from Models.DecoderType import DecoderType
from Models.EncoderType import EncoderType
from Models.RecipeType import RecipeType
from Trainers.BaseTrainer import BaseTrainer

class RDN_CiaoSR_Trainer:
    @staticmethod
    def TrainSimple():
        BaseTrainer(
            encoder=EncoderType.RDN,
            decoder=DecoderType.CiaoSR,
            recipe=RecipeType.Simple,
            input_patch=48,
            scale_range=[1, 4]
        ).TrainModel()

    @staticmethod
    def TrainSimpleLargerPatch():
        BaseTrainer(
            encoder=EncoderType.RDN,
            decoder=DecoderType.CiaoSR,
            recipe=RecipeType.Simple,
            input_patch=64,
            scale_range=[1, 4]
        ).TrainModel()

    @staticmethod
    def TrainSimpleLargerPatchScale():
        BaseTrainer(
            encoder=EncoderType.RDN,
            decoder=DecoderType.CiaoSR,
            recipe=RecipeType.Simple,
            input_patch=64,
            scale_range=[1, 6]
        ).TrainModel()

    @staticmethod
    def TrainSGDR():
        BaseTrainer(
            encoder=EncoderType.RDN,
            decoder=DecoderType.CiaoSR,
            recipe=RecipeType.SGDR,
            input_patch=48,
            scale_range=[1, 4]
        ).TrainModel()

    @staticmethod
    def TrainGradLoss():
        BaseTrainer(
            encoder=EncoderType.RDN,
            decoder=DecoderType.CiaoSR,
            recipe=RecipeType.GradLoss,
            input_patch=48,
            scale_range=[1, 4]
        ).TrainModel()

    @staticmethod
    def TrainGramL1Loss():
        BaseTrainer(
            encoder=EncoderType.RDN,
            decoder=DecoderType.CiaoSR,
            recipe=RecipeType.GramL1Loss,
            input_patch=48,
            scale_range=[1, 4]
        ).TrainModel()
    
