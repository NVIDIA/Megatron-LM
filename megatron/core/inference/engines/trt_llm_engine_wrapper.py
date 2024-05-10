from typing import List

from megatron.core.inference.common_inference_params import CommonInferenceParams
from megatron.core.inference.engines.abstract_engine import AbstractEngine
from megatron.core.models.common.language_module.language_module import LanguageModule


class TRTLLMEngineWrapper(AbstractEngine):
    def __init__(self, model: LanguageModule, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer

    # TODO : Will use high level apis to implement this
    def generate(self, prompts: List[str], common_inference_params: CommonInferenceParams):
        return prompts

    # TODO : Need to implement this
    @staticmethod
    def is_model_trt_llm_exportable(model: LanguageModule):
        return False
