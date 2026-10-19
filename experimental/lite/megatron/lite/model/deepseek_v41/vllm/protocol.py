# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Native Lite lifecycle/DistOpt; deployment operator model selection only."""
from megatron.lite.model.deepseek_v41.lite import protocol as native
from megatron.lite.model.deepseek_v41.lite.protocol import (
    ImplConfig,
    build_model_config,
    forward_step,
    load_hf_weights,
    validate_impl_cfg_keys,
)


def build_model(model_cfg, *, impl_cfg):
    from .model import DeepseekV41Model

    return native.build_model(model_cfg, impl_cfg=impl_cfg, _model_class=DeepseekV41Model)
