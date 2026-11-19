# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Register the supported DeepSeek-V3, GPT-OSS, Llama, Nemotron and Qwen conversion families."""

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    GatedMLPMapping,
    MegatronParamMapping,
    QKVMapping,
    ReplicatedMapping,
    RowParallelMapping,
)
from megatron.bridge.models.deepseek.deepseek_v3_bridge import DeepSeekV3Bridge
from megatron.bridge.models.gpt_oss.gpt_oss_bridge import GPTOSSBridge
from megatron.bridge.models.llama.llama_bridge import LlamaBridge
from megatron.bridge.models.nemotron_3.nemotron_3_bridge import Nemotron3Bridge
from megatron.bridge.models.nemotron_3.super_vl_bridge import Nemotron35SuperVLBridge
from megatron.bridge.models.qwen.qwen35_bridge import Qwen35Bridge, Qwen35MoEBridge
from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import Qwen35VLBridge, Qwen35VLMoEBridge
