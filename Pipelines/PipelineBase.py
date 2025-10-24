from abc import ABC, abstractmethod
import torch.nn as nn

class PipelineBase(ABC):
    @abstractmethod
    def InitModel(self, model: nn.Module):
        self.model = model

    @abstractmethod
    def LoadConfigurations(self,):
        pass

    @abstractmethod
    def CreateDataLoaders(self,):
        pass

    @abstractmethod
    def LoadModelWeights(self, ):
        pass

    @abstractmethod
    def InitTrainingRecipe(self, ):
        pass

    @abstractmethod
    def InitModelObjectives(self, ):
        pass