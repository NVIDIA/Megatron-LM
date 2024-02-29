from typing import List
from megatron.core.inference.backends.abstract_backend import AbstractBackend
from megatron.core.inference.common_inference_params import CommonInferenceParams
from megatron.core.models.common.language_module.language_module import LanguageModule

class TRTLLMBackend(AbstractBackend):
    def __init__(self, model: LanguageModule, tokenizer = None):
        self.model = model
        self.tokenizer = tokenizer

    # TODO : Implement this
    def generate(self, prompts:List[str], common_inference_params: CommonInferenceParams):
        return prompts

    # TODO : Implement this
    @staticmethod
    def is_model_trt_llm_exportable(model: LanguageModule):
        return False