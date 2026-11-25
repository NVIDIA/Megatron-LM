# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Register the supported Nemotron and Qwen model conversion families."""

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
from megatron.bridge.models.nemotron_3.nemotron_3_bridge import Nemotron3Bridge
from megatron.bridge.models.nemotron_3.super_vl_bridge import Nemotron35SuperVLBridge
from megatron.bridge.models.qwen.qwen35_bridge import Qwen35Bridge, Qwen35MoEBridge
from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import Qwen35VLBridge, Qwen35VLMoEBridge
