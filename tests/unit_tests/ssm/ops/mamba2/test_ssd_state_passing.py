# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import contextlib
import unittest
from unittest import mock

import torch

try:
    import triton

    from megatron.core.ssm.ops.mamba2 import ssd_state_passing
    from megatron.core.ssm.ops.mamba2.ssd_state_passing import _state_passing_fwd

    HAVE_SSD_OPS = True
except (ImportError, Exception):
    HAVE_SSD_OPS = False


@unittest.skipIf(not HAVE_SSD_OPS, "SSD ops (Triton 3+) not available")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA required for SSD ops")
class TestStatePassingFwd(unittest.TestCase):
    """Tests for _state_passing_fwd: recurrence out = exp(dA_cs_last) * prev + new_states."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda")
        self.nchunks = 4
        self.nheads = 2
        self.chunk_size = 16
        self.dim = self.chunk_size * 8  # headdim * dstate flattened
        self.cu_chunk_seqlens = torch.tensor(
            [0, 16, 32, 48, 64], dtype=torch.int32, device=self.device
        )

    def test_state_passing_fwd_shape(self):
        states = torch.randn(
            self.nchunks, self.nheads, self.dim, device=self.device, dtype=torch.float32
        )
        dA_cumsum = torch.randn(
            self.nheads, self.nchunks, self.chunk_size, device=self.device, dtype=torch.float32
        )
        seq_idx = torch.zeros(self.nchunks, dtype=torch.int32, device=self.device)

        out = _state_passing_fwd(
            states, dA_cumsum, self.cu_chunk_seqlens, seq_idx, initial_states=None
        )

        self.assertEqual(out.shape, (self.nchunks, self.nheads, self.dim))
        self.assertFalse(torch.isnan(out).any())

    def test_state_passing_fwd_with_initial_states(self):
        states = torch.randn(
            self.nchunks, self.nheads, self.dim, device=self.device, dtype=torch.float32
        )
        dA_cumsum = torch.randn(
            self.nheads, self.nchunks, self.chunk_size, device=self.device, dtype=torch.float32
        )
        seq_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=self.device)
        initial_states = torch.randn(
            2, self.nheads, self.dim, device=self.device, dtype=torch.float32
        )

        out = _state_passing_fwd(
            states, dA_cumsum, self.cu_chunk_seqlens, seq_idx, initial_states=initial_states
        )

        self.assertEqual(out.shape, (self.nchunks, self.nheads, self.dim))
        self.assertFalse(torch.isnan(out).any())

    def test_state_passing_fwd_recurrence_single_head_single_dim(self):
        """Sanity: single head, small dim, check recurrence manually for first elements."""
        dim = 4
        nchunks = 2
        nheads = 1
        chunk_size = 2
        cu_chunk_seqlens = torch.tensor([0, 2, 4], dtype=torch.int32, device=self.device)
        seq_idx = torch.zeros(nchunks, dtype=torch.int32, device=self.device)

        states = torch.randn(nchunks, nheads, dim, device=self.device, dtype=torch.float32)
        dA_cumsum = torch.randn(
            nheads, nchunks, chunk_size, device=self.device, dtype=torch.float32
        )

        out = _state_passing_fwd(states, dA_cumsum, cu_chunk_seqlens, seq_idx)

        # Chunk 0: out[0] = exp(dA_cumsum[0,-1]) * 0 + states[0] = states[0] (no initial state)
        # So out[0] should equal states[0]
        torch.testing.assert_close(out[0], states[0], rtol=1e-4, atol=1e-4)
        self.assertEqual(out.shape, (nchunks, nheads, dim))

    @contextlib.contextmanager
    def _timed_autotuning(self):
        """Time two BLOCK_SIZE candidates from a cold autotune cache, whatever the environment
        pins, and check that the launch inside really benchmarked them."""
        tuner = ssd_state_passing._state_passing_fwd_kernel
        candidates = [triton.Config({"BLOCK_SIZE": 64}), triton.Config({"BLOCK_SIZE": 128})]
        benchmarked = []
        do_bench = tuner.do_bench

        def counting_do_bench(kernel_call, quantiles):
            benchmarked.append(kernel_call)
            return do_bench(kernel_call, quantiles=quantiles)

        with (
            mock.patch.object(tuner, "configs", candidates),
            mock.patch.object(tuner, "cache", {}),
            mock.patch.object(tuner, "cache_results", False, create=True),
            mock.patch.object(tuner, "do_bench", counting_do_bench),
        ):
            yield
        self.assertEqual(len(benchmarked), len(candidates))

    def test_dst_states_commit_once_under_timed_autotuning(self):
        """dst mode in place, as batch-invariant decode calls it: the slot states are both the
        initial and the destination states. Each benchmark launch of a cold autotune cache must
        leave them as it found them, so that the step commits each flagged entry once."""
        num_slots, entries = 4, 3
        slots = torch.tensor([1, 0, 3], dtype=torch.int32, device=self.device)
        flags = torch.tensor([1, 0, 1], dtype=torch.int32, device=self.device)
        # Integer states and no decay keep every sum exact.
        states = torch.randint(-4, 5, (entries, self.nheads, self.dim), device=self.device).float()
        dA_cumsum = torch.zeros(self.nheads, entries, self.chunk_size, device=self.device)
        ssm_state = torch.randint(
            -4, 5, (num_slots, self.nheads, self.dim), device=self.device
        ).float()
        expected = ssm_state.clone()
        expected[1] += states[0]
        expected[3] += states[2]

        with self._timed_autotuning():
            _state_passing_fwd(
                states,
                dA_cumsum,
                None,
                slots,
                initial_states=ssm_state,
                dst_states=ssm_state,
                dst_indices=slots,
                dst_flags=flags,
            )

        torch.testing.assert_close(ssm_state, expected, rtol=0, atol=0)

    def test_chunk_states_under_timed_autotuning(self):
        """The mode without dst states has none to restore while the autotuner times it."""
        states = torch.randint(
            -4, 5, (self.nchunks, self.nheads, self.dim), device=self.device
        ).float()
        dA_cumsum = torch.zeros(self.nheads, self.nchunks, self.chunk_size, device=self.device)
        seq_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=self.device)
        initial_states = torch.randint(
            -4, 5, (2, self.nheads, self.dim), device=self.device
        ).float()

        with self._timed_autotuning():
            out = _state_passing_fwd(
                states, dA_cumsum, self.cu_chunk_seqlens, seq_idx, initial_states=initial_states
            )

        first = initial_states[0] + states[0]
        second = initial_states[1] + states[2]
        expected = torch.stack([first, first + states[1], second, second + states[3]])
        torch.testing.assert_close(out, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
