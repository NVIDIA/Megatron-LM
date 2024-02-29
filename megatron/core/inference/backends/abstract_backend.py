from abc import ABC, abstractmethod
from typing import List
from megatron.core.inference.common_inference_params import CommonInferenceParams

class AbstractBackend(ABC):
    
    @staticmethod
    @abstractmethod
    def generate(prompts:List[str], common_inference_params: CommonInferenceParams):
        pass