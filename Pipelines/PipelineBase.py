from abc import ABC, abstractmethod

class PipelineBase(ABC):
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
    def InitTrainingSettings(self, ):
        pass

    @abstractmethod
    def InitModelObjectives(self, ):
        pass