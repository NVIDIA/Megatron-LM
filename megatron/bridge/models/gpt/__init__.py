# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Builder-backed GPT model configs shared by the GPT conversion families."""

from megatron.bridge.models.gpt.model_config import BridgeGPTMLAModelConfig, BridgeGPTModelConfig

__all__ = ["BridgeGPTMLAModelConfig", "BridgeGPTModelConfig"]
