from Decoders.LIIF import LIIF
from Encoders.EDSR import EDSR
from TrainingFactories.BaseTrainingFactory import BaseTrainingFactory
from Pipelines.PipelineBase import PipelineBase

class EDSR_LIIF_Factory(BaseTrainingFactory):
    def BuildModel(self, pipeline: PipelineBase) -> PipelineBase:
        # Create the ecnoder and decoder
        encoder = EDSR()
        model = LIIF(encoder)

        # Call the Pipeline chains
        pipeline.LoadConfigurations()
        pipeline.InitModel(model)
        pipeline.CreateDataLoaders()
        pipeline.LoadModelWeights()
        pipeline.InitTrainingRecipe()
        pipeline.InitModelObjectives()
