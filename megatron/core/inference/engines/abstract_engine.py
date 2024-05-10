from abc import ABC, abstractmethod
from typing import List


class AbstractEngine(ABC):
    @staticmethod
    @abstractmethod
    def generate(self) -> dict:
        """The abstarct backends generate function. 

        To define your own backend, make sure you implement this and return the outputs as a dictionary . 

        Returns:
            dict: The output dictionary which will have as keys mostly the generated tokens, text and log probabilitites. 
        """
        pass
