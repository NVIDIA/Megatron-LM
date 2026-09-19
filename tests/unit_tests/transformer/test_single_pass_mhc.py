# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Test the inter-sublayer dependency introduced by V4.1 mHC."""

import pytest
import torch

from megatron.core.transformer.hyper_connection import HyperConnectionModule, SinglePassMHCState
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
        mhc_single_pass=True,
        use_fused_mhc=fused,
    )
    module = HyperConnectionModule(config, 1).cuda()
    x = torch.randn(5, 2, 64, device="cuda", dtype=dtype, requires_grad=True)
    previous = torch.randn(5, 2, 4, device="cuda", requires_grad=True)
    state = SinglePassMHCState(previous)
    branch, residual, post, residual_input = module(x, mhc_state=state, return_residual=True)
    following = state.pre_mix
    oracle = torch.einsum("sbn,sbnc->sbc", previous, x.float().unflatten(-1, (4, 16)))
    torch.testing.assert_close(branch, oracle.to(dtype))
    assert following.dtype == post.dtype == residual.dtype == torch.float32
    output = module.fused_h_res_h_post_bda(
        residual, residual_input, post, (branch, None), 0.0, True, False
    )
    reference = torch.matmul(residual.transpose(-1, -2), x.float().unflatten(-1, (4, 16)))
    reference = reference + post.unsqueeze(-1) * branch.float().unsqueeze(-2)
    torch.testing.assert_close(output, reference.flatten(-2).to(dtype))
    state.contract(output, 4).square().mean().backward()
    assert previous.grad is not None and previous.grad.abs().sum() > 0
    assert module.mapping_proj.weight.grad is not None
    assert torch.isfinite(x.grad).all()


def test_entry_selects_first_stream_and_exit_uses_last_mix():
    x = torch.arange(24.0, device="cuda").reshape(2, 1, 12).requires_grad_()
    torch.testing.assert_close(SinglePassMHCState().contract(x, 3), x[..., :4])
    mix = torch.tensor([0.0, 0.0, 1.0], device="cuda").expand(2, 1, 3)
    torch.testing.assert_close(SinglePassMHCState(mix).contract(x, 3), x[..., -4:])
    with pytest.raises(ValueError, match="preceding"):
        SinglePassMHCState(mix[:, :, :2]).contract(x, 3)


@pytest.mark.parametrize("backend", ["none", "native", "triton"])
def test_fp32_mixing_is_preserved_under_outer_autocast(backend):
    """AMP must not round the residual-mixing matmul to BF16 for FP32 activations."""
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=2,
        enable_mhc_connections=True,
        mhc_single_pass=True,
        use_fused_mhc=backend != "none",
        mhc_fused_backend="auto" if backend == "none" else backend,
    )
    module = HyperConnectionModule(config, 1).cuda()
    x = torch.randn(5, 2, 64, device="cuda")
    previous = torch.randn(5, 2, 4, device="cuda")
    outputs = []
    for enabled in (False, True):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled):
            state = SinglePassMHCState(previous)
            branch, residual_mix, post, residual = module(x, mhc_state=state, return_residual=True)
            outputs.append(
                module.fused_h_res_h_post_bda(
                    residual_mix, residual, post, (branch, None), 0.0, False, False
                )
            )
    assert all(output.dtype == torch.float32 for output in outputs)
    torch.testing.assert_close(outputs[1], outputs[0], rtol=1e-5, atol=1e-6)
