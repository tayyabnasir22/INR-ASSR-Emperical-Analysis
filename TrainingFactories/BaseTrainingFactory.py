from abc import ABC, abstractmethod
from Pipelines.PipelineBase import PipelineBase

class BaseTrainingFactory(ABC):
    @abstractmethod
    def BuildModel(self,) -> PipelineBase:
        pass