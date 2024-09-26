# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from megatron.core.export.model_type import ModelType

TRT_MODEL_TYPE_STRING = {
    ModelType.gpt: 'GPTForCausalLM',
    ModelType.gptnext: 'GPTForCausalLM',
    ModelType.starcoder: 'GPTForCausalLM',
    ModelType.mixtral: 'LlamaForCausalLM',
    ModelType.llama: 'LlamaForCausalLM',
    ModelType.gemma: 'GemmaForCausalLM',
    ModelType.falcon: 'FalconForCausalLM',
}
