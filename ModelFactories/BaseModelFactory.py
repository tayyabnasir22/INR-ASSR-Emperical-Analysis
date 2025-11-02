from abc import ABC, abstractmethod
from Pipelines.PipelineBase import PipelineBase

class BaseModelFactory(ABC):
    @abstractmethod
    def BuildModel(self,) -> PipelineBase:
        pass