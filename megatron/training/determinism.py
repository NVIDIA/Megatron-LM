# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Training compatibility entrypoints for the shared MCore startup policy."""

from megatron.determinism import ACCEPTED_ENV_VAR_VALUES as ACCEPTED_ENV_VAR_VALUES
from megatron.determinism import ACCEPTED_NCCL_ALGO_TOKENS as ACCEPTED_NCCL_ALGO_TOKENS
from megatron.determinism import (
    ARG_VALUES_REQUIRED_FOR_DETERMINISM as ARG_VALUES_REQUIRED_FOR_DETERMINISM,
)
from megatron.determinism import AUX_LOSS_FUSION_ARG as AUX_LOSS_FUSION_ARG
from megatron.determinism import DETERMINISM_ENV_VAR_DEFAULTS as DETERMINISM_ENV_VAR_DEFAULTS
from megatron.determinism import apply_determinism_env as apply_determinism_env
from megatron.determinism import configure_determinism

__all__ = [
    "ACCEPTED_ENV_VAR_VALUES",
    "ACCEPTED_NCCL_ALGO_TOKENS",
    "ARG_VALUES_REQUIRED_FOR_DETERMINISM",
    "AUX_LOSS_FUSION_ARG",
    "DETERMINISM_ENV_VAR_DEFAULTS",
    "apply_determinism_env",
    "configure_determinism",
    "apply_determinism_to_args",
]


def apply_determinism_to_args(args: object) -> dict:
    """Validate parsed training options and apply the shared process-wide policy.

    The caller selects this path when deterministic mode is enabled. Preserve
    compatibility with helpers that supply only the fields checked here, and
    leave the Namespace untouched. Return the effective policy for logging.
    """
    options = {name: getattr(args, name) for name in ARG_VALUES_REQUIRED_FOR_DETERMINISM}
    options.update(
        deterministic_mode=getattr(args, "deterministic_mode", True),
        moe_router_fusion=getattr(args, "moe_router_fusion", False),
        moe_router_aux_loss_fusion=getattr(args, AUX_LOSS_FUSION_ARG, None),
    )
    return configure_determinism(options)
