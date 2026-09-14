# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.ssm import causal_conv1d as causal_conv1d_module
from megatron.core.utils import is_causal_conv1d_min_version
from tests.unit_tests.test_utilities import Utils

try:
    from causal_conv1d import causal_conv1d_fn

    HAVE_CAUSAL_CONV1D = True
except ImportError:
    HAVE_CAUSAL_CONV1D = False


def _contiguous_slice(tensor, cp_rank, local_seq_len):
    return tensor[:, cp_rank * local_seq_len : (cp_rank + 1) * local_seq_len].contiguous()


@pytest.mark.internal
@pytest.mark.skipif(
    not HAVE_CAUSAL_CONV1D or not torch.cuda.is_available() or Utils.world_size < 2,
    reason="CP causal convolution parity requires causal-conv1d and at least two GPUs",
)
@pytest.mark.parametrize(
    ("batch_size", "packed_sequence_len"),
    [(1, None), (2, None), (1, 64), (1, 63)],
    ids=["bshd-b1", "bshd-b2", "thd-cp-boundary", "thd-cp-partial"],
)
def test_causal_conv1d_cp_matches_full_sequence(batch_size, packed_sequence_len):
    if packed_sequence_len is not None and not is_causal_conv1d_min_version("1.7.0"):
        pytest.skip("THD CP causal convolution requires causal-conv1d >= 1.7.0")
    Utils.initialize_model_parallel(context_parallel_size=Utils.world_size)
    try:
        cp_group = parallel_state.get_context_parallel_group()
        cp_size = dist.get_world_size(group=cp_group)
        cp_rank = dist.get_rank(group=cp_group)
        device = torch.device("cuda", torch.cuda.current_device())
        dtype = torch.float32
        rtol, atol = 3e-4, 1e-3
        local_seq_len = 64
        global_seq_len = cp_size * local_seq_len

        torch.manual_seed(1234)

        channels, width = 16, 4
        x_global = torch.randn(batch_size, global_seq_len, channels, device=device, dtype=dtype)
        weight_global = torch.randn(channels, width, device=device, dtype=dtype)
        bias_global = torch.randn(channels, device=device, dtype=dtype)
        dy_global = torch.randn_like(x_global)
        global_seq_idx = (
            torch.arange(global_seq_len, device=device, dtype=torch.int32).unsqueeze(0)
            // packed_sequence_len
            if packed_sequence_len is not None
            else None
        )
        x_ref = x_global.detach().clone().requires_grad_(True)
        weight_ref = weight_global.detach().clone().requires_grad_(True)
        bias_ref = bias_global.detach().clone().requires_grad_(True)
        output_ref = causal_conv1d_fn(
            x=x_ref.transpose(1, 2),
            weight=weight_ref,
            bias=bias_ref,
            seq_idx=global_seq_idx,
            activation="silu",
        ).transpose(1, 2)
        output_ref.backward(dy_global)

        x_local = _contiguous_slice(x_global, cp_rank, local_seq_len).detach().requires_grad_(True)
        weight_local = weight_global.detach().clone().requires_grad_(True)
        bias_local = bias_global.detach().clone().requires_grad_(True)
        output_local = causal_conv1d_module.causal_conv1d_cp(
            x=x_local,
            weight=weight_local,
            bias=bias_local,
            activation="silu",
            cp_group=cp_group,
            global_seq_idx=global_seq_idx,
        )
        output_local.backward(_contiguous_slice(dy_global, cp_rank, local_seq_len))
        dist.all_reduce(weight_local.grad, group=cp_group)
        dist.all_reduce(bias_local.grad, group=cp_group)

        expected_output = _contiguous_slice(output_ref, cp_rank, local_seq_len)
        expected_dx = _contiguous_slice(x_ref.grad, cp_rank, local_seq_len)
        torch.testing.assert_close(output_local, expected_output, rtol=rtol, atol=atol)
        torch.testing.assert_close(x_local.grad, expected_dx, rtol=rtol, atol=atol)
        torch.testing.assert_close(weight_local.grad, weight_ref.grad, rtol=rtol, atol=atol)
        torch.testing.assert_close(bias_local.grad, bias_ref.grad, rtol=rtol, atol=atol)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not HAVE_CAUSAL_CONV1D, reason="causal_conv1d is not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_causal_conv1d_channel_contiguous_matches_sequence_contiguous(dtype):
    """Verify forward and backward equivalence across causal_conv1d layouts.

    Both inputs have shape (batch, dim, seqlen) and identical values. Only
    their strides differ: a unit dim stride selects the channel-last kernel,
    while a unit seqlen stride selects the standard kernel.
    """
    torch.manual_seed(42)
    batch, seq_len, dim, width = 2, 128, 64, 4
    padding = 16
    storage_dim = dim + 2 * padding

    # Slice a larger allocation so the channel-last input is non-dense and
    # has gaps between consecutive sequence elements.
    channel_last_storage = torch.randn(batch, seq_len, storage_dim, device="cuda", dtype=dtype)
    x_channel_contiguous = (
        channel_last_storage[:, :, padding : padding + dim]
        .transpose(1, 2)
        .detach()
        .requires_grad_()
    )
    x_sequence_contiguous = x_channel_contiguous.detach().contiguous().requires_grad_()

    assert x_channel_contiguous.shape == x_sequence_contiguous.shape == (batch, dim, seq_len)
    assert torch.equal(x_channel_contiguous, x_sequence_contiguous)
    assert x_channel_contiguous.stride() == (seq_len * storage_dim, 1, storage_dim)
    assert x_sequence_contiguous.stride() == (dim * seq_len, seq_len, 1)

    weight = torch.randn(dim, width, device="cuda", dtype=torch.float32, requires_grad=True)
    bias = torch.randn(dim, device="cuda", dtype=torch.float32, requires_grad=True)

    out_channel_contiguous = causal_conv1d_fn(x_channel_contiguous, weight, bias, activation="silu")
    out_sequence_contiguous = causal_conv1d_fn(
        x_sequence_contiguous, weight, bias, activation="silu"
    )

    assert out_channel_contiguous.stride() == (seq_len * dim, 1, dim)
    assert out_sequence_contiguous.stride() == (dim * seq_len, seq_len, 1)

    # The forward kernels should be bitwise identical.
    assert torch.equal(out_channel_contiguous, out_sequence_contiguous)

    grad_channel_contiguous = torch.randn_like(out_channel_contiguous)
    grad_sequence_contiguous = grad_channel_contiguous.contiguous()
    assert torch.equal(grad_channel_contiguous, grad_sequence_contiguous)
    assert grad_channel_contiguous.stride() == (seq_len * dim, 1, dim)
    assert grad_sequence_contiguous.stride() == (dim * seq_len, seq_len, 1)

    dx_channel, dweight_channel, dbias_channel = torch.autograd.grad(
        out_channel_contiguous,
        (x_channel_contiguous, weight, bias),
        grad_outputs=grad_channel_contiguous,
    )
    dx_sequence, dweight_sequence, dbias_sequence = torch.autograd.grad(
        out_sequence_contiguous,
        (x_sequence_contiguous, weight, bias),
        grad_outputs=grad_sequence_contiguous,
    )

    assert dx_channel.stride() == (seq_len * dim, 1, dim)
    assert dx_sequence.stride() == (dim * seq_len, seq_len, 1)

    # dx is returned in the input dtype, so BF16 needs looser tolerances. Weight
    # and bias gradients are accumulated in FP32; their small differences come
    # from the kernels using different parallel reduction orders.
    input_grad_rtol, input_grad_atol = (3e-4, 1e-3) if dtype == torch.float32 else (1e-2, 5e-2)
    param_grad_rtol, param_grad_atol = 1e-3, 1e-3

    torch.testing.assert_close(dx_channel, dx_sequence, rtol=input_grad_rtol, atol=input_grad_atol)
    torch.testing.assert_close(
        dweight_channel, dweight_sequence, rtol=param_grad_rtol, atol=param_grad_atol
    )
    torch.testing.assert_close(
        dbias_channel, dbias_sequence, rtol=param_grad_rtol, atol=param_grad_atol
    )
