# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import tensorrt_llm

from megatron.core.export.model_type import ModelType

TRT_MODEL_CONFIG = {
    ModelType.gpt: tensorrt_llm.models.gpt.config.GPTConfig,
    ModelType.gptnext: tensorrt_llm.models.gpt.config.GPTConfig,
    ModelType.starcoder: tensorrt_llm.models.gpt.config.GPTConfig,
    ModelType.mixtral: tensorrt_llm.models.llama.config.LLaMAConfig,
    ModelType.llama: tensorrt_llm.models.llama.config.LLaMAConfig,
    ModelType.gemma: tensorrt_llm.models.GemmaConfig,
    ModelType.falcon: tensorrt_llm.models.falcon.config.FalconConfig,
    ModelType.nemotron_nas: tensorrt_llm.models.nemotron_nas.config.DeciConfig,
}
