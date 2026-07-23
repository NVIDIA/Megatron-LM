# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Model registration helpers for Megatron Lite."""

from megatron.lite.model.registry import (
    get_model_package,
    get_train_runtime_module,
    list_models,
    register_model,
    resolve_model_type_from_hf,
    resolve_runtime_model_name,
)

__all__ = [
    "get_model_package",
    "get_train_runtime_module",
    "list_models",
    "register_model",
    "resolve_model_type_from_hf",
    "resolve_runtime_model_name",
]
