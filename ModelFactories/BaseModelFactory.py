from Decoders.CiaoSR import CiaoSR
from Decoders.HIIF import HIIF
from Decoders.LIIF import LIIF
from Decoders.LINF import LINF
from Decoders.LTE import LTE
from Decoders.MetaSR import MetaSR
from Decoders.SRNO import SRNO
from Encoders.EDSR import EDSR
from Encoders.RDN import RDN
from Models.DecoderType import DecoderType
from Models.EncoderType import EncoderType
from Pipelines.PipelineBase import PipelineBase

class BaseModelFactory:
    def __init__(self):
        self._encoders = {
            EncoderType.EDSR: EDSR,
            EncoderType.RDN: RDN
        }

        self._decoders = {
            DecoderType.CiaoSR: CiaoSR,
            DecoderType.HIIF: HIIF,
            DecoderType.LIIF: LIIF,
            DecoderType.LTE: LTE,
            DecoderType.MetaSR: MetaSR,
            DecoderType.SRNO: SRNO,
            DecoderType.LINF: LINF,
        }

    def BuildModel(self, pipeline: PipelineBase, encoder: EncoderType, decoder: DecoderType):
        # Create the ecnoder and decoder
        encoder = self._encoders[encoder]().cuda()
        model = self._decoders[decoder](encoder).cuda()

        # Call the Pipeline chains
        pipeline.LoadConfigurations()
        pipeline.InitModel(model)
        pipeline.CreateDataLoaders()
        pipeline.LoadModelWeights()
        pipeline.InitTrainingRecipe()
        pipeline.InitModelObjectives()