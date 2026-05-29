# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.ssm.ops.mamba_ssm import selective_state_update


def _requires_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.mark.internal
class TestSelectiveStateUpdate:

    def setup_method(self, method):
        _requires_cuda()

    def test_write_indices_preserve_read_state_and_update_destination(self):
        """Out-of-place SSM updates read committed banks and write candidate banks."""
        torch.manual_seed(42)
        batch, seq_len, dim, dstate = 2, 3, 8, 16
        num_states = 6
        state = torch.randn(num_states, dim, dstate, device="cuda", dtype=torch.float32)
        x = torch.randn(batch, seq_len, dim, device="cuda", dtype=torch.float32)
        dt = torch.rand(batch, seq_len, dim, device="cuda", dtype=torch.float32) * 0.01
        A = -torch.rand(dim, dstate, device="cuda", dtype=torch.float32)
        B = torch.randn(batch, seq_len, dstate, device="cuda", dtype=torch.float32)
        C = torch.randn(batch, seq_len, dstate, device="cuda", dtype=torch.float32)
        read_indices = torch.tensor([0, 2], device="cuda", dtype=torch.int64)
        write_indices = torch.tensor([1, 3], device="cuda", dtype=torch.int64)
        intermediate = torch.zeros(
            batch, seq_len, dim, dstate, device="cuda", dtype=torch.float32
        )

        state_triton = state.clone()
        result = selective_state_update(
            state_triton,
            x,
            dt,
            A,
            B,
            C,
            state_batch_indices=read_indices,
            state_batch_write_indices=write_indices,
            intermediate_ssm_states=intermediate,
            state_bank_count=2,
        )

        state_ref = state[read_indices].clone()
        expected = selective_state_update(state_ref, x, dt, A, B, C)

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(state_triton[read_indices], state[read_indices])
        torch.testing.assert_close(state_triton[write_indices], state_ref)
        torch.testing.assert_close(intermediate[:, -1], state_ref)
