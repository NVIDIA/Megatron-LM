# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from megatron.core.export.model_type import ModelType
from megatron.core.export.trtllm.model_to_trllm_mapping.falcon_model import FALCON_DICT
from megatron.core.export.trtllm.model_to_trllm_mapping.gemma_model import GEMMA_DICT
from megatron.core.export.trtllm.model_to_trllm_mapping.gpt_model import GPT_DICT
from megatron.core.export.trtllm.model_to_trllm_mapping.gpt_next_model import GPT_NEXT_DICT
from megatron.core.export.trtllm.model_to_trllm_mapping.llama_model import LLAMA_DICT
from megatron.core.export.trtllm.model_to_trllm_mapping.starcoder_model import STARCODER_DICT

DEFAULT_CONVERSION_DICT = {
    ModelType.llama: LLAMA_DICT,
    ModelType.falcon: FALCON_DICT,
    ModelType.gemma: GEMMA_DICT,
    ModelType.starcoder: STARCODER_DICT,
    ModelType.gpt: GPT_DICT,
    ModelType.gptnext: GPT_NEXT_DICT,
}
