from abc import ABC, abstractmethod
from typing import List 

class AbstractTextGenerationStrategy(ABC):
    def __init__(self, model, common_inference_params, tokenizer):
        pass