# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests that the LLaVA local decoder spec honours qk_layernorm and normalization."""

from megatron.core.models.multimodal.llava_spec import decoder_model_with_local_default_spec
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.torch_norm import WrappedTorchNorm


def test_local_spec_qk_layernorm():
    spec = decoder_model_with_local_default_spec(qk_layernorm=True)
    attention = spec.submodules.self_attention.submodules

    assert attention.q_layernorm not in (None, IdentityOp)
    assert attention.k_layernorm not in (None, IdentityOp)


def test_local_spec_rms_norm_uses_torch_norm():
    # Apex FusedLayerNorm implements LayerNorm only, so RMSNorm must come from Torch.
    spec = decoder_model_with_local_default_spec(qk_layernorm=True, normalization="RMSNorm")
    attention = spec.submodules.self_attention.submodules

    assert spec.submodules.input_layernorm is WrappedTorchNorm
    assert spec.submodules.pre_mlp_layernorm is WrappedTorchNorm
    assert attention.q_layernorm is WrappedTorchNorm
    assert attention.k_layernorm is WrappedTorchNorm
