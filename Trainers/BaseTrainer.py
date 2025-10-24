from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    @abstractmethod
    def TrainModel(self,):
        pass