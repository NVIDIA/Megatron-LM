# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Test the inter-sublayer dependency introduced by V4.1 mHC."""

import pytest
import torch

from megatron.core.transformer.single_pass_mhc import SinglePassHyperConnection, contract_streams
from megatron.core.transformer.transformer_config import TransformerConfig


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("fused", [False, True])
def test_previous_mix_controls_input_and_receives_gradients(dtype, fused):
    config = TransformerConfig(
        num_layers=2,
        hidden_size=16,
        num_attention_heads=2,
        layernorm_epsilon=1e-20,
        enable_mhc_connections=True,
        use_fused_mhc=fused,
    )
    module = SinglePassHyperConnection(config, 1).cuda()
    x = torch.randn(5, 2, 64, device="cuda", dtype=dtype, requires_grad=True)
    previous = torch.randn(5, 2, 4, device="cuda", requires_grad=True)
    branch, following, post, residual = module(x, previous)
    oracle = torch.einsum("sbn,sbnc->sbc", previous, x.float().unflatten(-1, (4, 16)))
    torch.testing.assert_close(branch, oracle.to(dtype))
    assert following.dtype == post.dtype == residual.dtype == torch.float32
    output = module.combine(branch, x, post, residual)
    reference = torch.matmul(residual.transpose(-1, -2), x.float().unflatten(-1, (4, 16)))
    reference = reference + post.unsqueeze(-1) * branch.float().unsqueeze(-2)
    torch.testing.assert_close(output, reference.flatten(-2).to(dtype))
    contract_streams(output, following, 4).square().mean().backward()
    assert previous.grad is not None and previous.grad.abs().sum() > 0
    assert module.mapping_proj.weight.grad is not None
    assert torch.isfinite(x.grad).all()


def test_entry_selects_first_stream_and_exit_uses_last_mix():
    x = torch.arange(24.0, device="cuda").reshape(2, 1, 12).requires_grad_()
    torch.testing.assert_close(contract_streams(x, None, 3), x[..., :4])
    mix = torch.tensor([0.0, 0.0, 1.0], device="cuda").expand(2, 1, 3)
    torch.testing.assert_close(contract_streams(x, mix, 3), x[..., -4:])
    with pytest.raises(ValueError, match="preceding"):
        contract_streams(x, mix[:, :, :2], 3)
