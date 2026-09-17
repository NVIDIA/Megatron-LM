# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compatibility exports; use megatron.determinism for setup before Core imports."""

from megatron.determinism import ACCEPTED_ENV_VAR_VALUES as ACCEPTED_ENV_VAR_VALUES
from megatron.determinism import ACCEPTED_NCCL_ALGO_TOKENS as ACCEPTED_NCCL_ALGO_TOKENS
from megatron.determinism import (
    ARG_VALUES_REQUIRED_FOR_DETERMINISM as ARG_VALUES_REQUIRED_FOR_DETERMINISM,
)
from megatron.determinism import AUX_LOSS_FUSION_ARG as AUX_LOSS_FUSION_ARG
from megatron.determinism import DETERMINISM_ENV_VAR_DEFAULTS as DETERMINISM_ENV_VAR_DEFAULTS
from megatron.determinism import apply_determinism_env as apply_determinism_env
from megatron.determinism import configure_determinism as configure_determinism
from megatron.determinism import is_determinism_configured as is_determinism_configured
from megatron.determinism import validate_determinism_config as validate_determinism_config

__all__ = [
    "ACCEPTED_ENV_VAR_VALUES",
    "ACCEPTED_NCCL_ALGO_TOKENS",
    "ARG_VALUES_REQUIRED_FOR_DETERMINISM",
    "AUX_LOSS_FUSION_ARG",
    "DETERMINISM_ENV_VAR_DEFAULTS",
    "apply_determinism_env",
    "configure_determinism",
    "is_determinism_configured",
    "validate_determinism_config",
]
