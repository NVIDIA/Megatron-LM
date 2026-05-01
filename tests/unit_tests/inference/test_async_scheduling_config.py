# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""
Unit tests for ``InferenceConfig.enable_async_scheduling`` and its
config-time guards (no speculative MTP, no hybrid/Mamba).
"""

import pytest

from megatron.core.inference.config import InferenceConfig, MambaInferenceStateConfig


def _base_kwargs(**overrides):
    """Minimal keyword args for constructing an InferenceConfig."""
    kwargs = dict(max_sequence_length=128)
    kwargs.update(overrides)
    return kwargs


def test_default_enable_async_scheduling_is_false():
    cfg = InferenceConfig(**_base_kwargs())
    assert cfg.enable_async_scheduling is False


def test_enable_async_scheduling_alone_is_accepted():
    cfg = InferenceConfig(**_base_kwargs(enable_async_scheduling=True))
    assert cfg.enable_async_scheduling is True


def test_enable_async_scheduling_with_speculative_mtp_raises():
    with pytest.raises(ValueError, match="speculative MTP"):
        InferenceConfig(**_base_kwargs(enable_async_scheduling=True, num_speculative_tokens=2))


def test_enable_async_scheduling_with_hybrid_mamba_raises():
    import torch

    mamba_cfg = MambaInferenceStateConfig(
        layer_type_list=["M"],
        conv_states_shape=(4, 32),
        ssm_states_shape=(1, 32, 16),
        conv_states_dtype=torch.bfloat16,
        ssm_states_dtype=torch.bfloat16,
    )
    with pytest.raises(ValueError, match="hybrid \\(Mamba\\)"):
        InferenceConfig(
            **_base_kwargs(enable_async_scheduling=True, mamba_inference_state_config=mamba_cfg)
        )


def test_async_off_with_mtp_or_mamba_is_accepted():
    """Guards must NOT trip when async is disabled."""
    cfg_mtp = InferenceConfig(**_base_kwargs(num_speculative_tokens=2))
    assert cfg_mtp.enable_async_scheduling is False
    import torch

    mamba_cfg = MambaInferenceStateConfig(
        layer_type_list=["M"],
        conv_states_shape=(4, 32),
        ssm_states_shape=(1, 32, 16),
        conv_states_dtype=torch.bfloat16,
        ssm_states_dtype=torch.bfloat16,
    )
    cfg_mamba = InferenceConfig(**_base_kwargs(mamba_inference_state_config=mamba_cfg))
    assert cfg_mamba.enable_async_scheduling is False
