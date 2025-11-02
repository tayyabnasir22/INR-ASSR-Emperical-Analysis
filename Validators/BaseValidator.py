from abc import ABC, abstractmethod

class BaseValidator(ABC):
    @abstractmethod
    def TestModel(self,):
        pass