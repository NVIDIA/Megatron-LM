# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared helpers for the SSM dynamic-inference engine tests.

The prefix-caching and CUDA-graph suites in this directory each run against
both hybrid SSM mixers, so they need the same optional-dependency gate, the
same skip logic, and the same model-construction knobs. Keeping one copy here
means the reasoning written down below -- particularly `ssm_state_config`'s --
cannot drift between files.
"""

import pytest
import torch

from megatron.core.inference.config import MambaInferenceStateConfig
from megatron.core.models.hybrid.hybrid_layer_specs import (
    gated_delta_product_stack_spec,
    hybrid_stack_spec,
)
from megatron.core.ssm.mamba_mixer import _check_mamba_sequence_packing_support
from megatron.core.ssm.packed_seq_helpers import check_fla_sequence_packing_support

try:
    import einops  # noqa: F401
    import fla  # noqa: F401
    import mamba_ssm  # noqa: F401

    HAVE_GDP_DEPS = True
except ImportError:
    HAVE_GDP_DEPS = False


def skip_if_sequence_packing_not_available(ssm_mixer="mamba"):
    """Skip unless the packing support the given mixer's kernels need is present."""
    if ssm_mixer == "gdp":
        if not HAVE_GDP_DEPS:
            pytest.skip("GDP requires fla + mamba_ssm + einops")
        available, reason = check_fla_sequence_packing_support()
    else:
        available, reason = _check_mamba_sequence_packing_support()
    if not available:
        pytest.skip(reason)


def hybrid_mixer_kwargs(ssm_mixer):
    """TransformerConfig kwargs selecting the hybrid stack's linear-attention mixer.

    GDP needs its head/group/state dims spelled out, plus the Householder count
    that sizes its chunk descriptors.
    """
    if ssm_mixer == "gdp":
        return dict(
            gdp_num_householder=2,
            mamba_num_heads=8,
            mamba_head_dim=32,
            mamba_num_groups=8,
            mamba_state_dim=64,
        )
    return dict(mamba_num_heads=16)


def hybrid_stack_spec_for(ssm_mixer):
    """The stack spec that builds the requested mixer."""
    return gated_delta_product_stack_spec if ssm_mixer == "gdp" else hybrid_stack_spec


def ssm_state_config(model):
    """Inference state config with an FP32 recurrent state.

    Prefix caching round trips a request's recurrent state through the cache,
    where the uncached baseline these tests compare against keeps it in the
    kernel's FP32 accumulator the whole way. With a BF16 cache that round trip
    rounds the state and the two runs can genuinely diverge in their last
    generated tokens -- the same reason batch-invariant mode forces FP32 (see
    MambaInferenceStateConfig.from_model). Pinning FP32 keeps these tests
    measuring the caching logic rather than cache precision.

    Serving still defaults to the model dtype; this is a test-side choice.
    """
    return MambaInferenceStateConfig.from_model(model, ssm_states_dtype=torch.float32)
