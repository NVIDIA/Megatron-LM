# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.ssm import causal_conv1d as causal_conv1d_module
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
def test_causal_conv1d_cp_matches_full_sequence():
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
        x_global = torch.randn(1, global_seq_len, channels, device=device, dtype=dtype)
        weight_global = torch.randn(channels, width, device=device, dtype=dtype)
        bias_global = torch.randn(channels, device=device, dtype=dtype)
        dy_global = torch.randn_like(x_global)

        x_ref = x_global.detach().clone().requires_grad_(True)
        weight_ref = weight_global.detach().clone().requires_grad_(True)
        bias_ref = bias_global.detach().clone().requires_grad_(True)
        output_ref = causal_conv1d_fn(
            x=x_ref.transpose(1, 2), weight=weight_ref, bias=bias_ref, activation="silu"
        ).transpose(1, 2)
        output_ref.backward(dy_global)

        x_local = _contiguous_slice(x_global, cp_rank, local_seq_len).detach().requires_grad_(True)
        weight_local = weight_global.detach().clone().requires_grad_(True)
        bias_local = bias_global.detach().clone().requires_grad_(True)
        output_local = causal_conv1d_module.causal_conv1d_cp(
            x=x_local, weight=weight_local, bias=bias_local, activation="silu", cp_group=cp_group
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
