# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import math
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

# Assume the provided class is in mamba_mixer.py
from megatron.core.ssm.mamba_mixer import MambaMixer


class MockContextParallel:
    """
    Mocks the MambaContextParallel helper.
    """

    def __init__(self, d_inner, ngroups, nheads, d_state, device):
        self.d_inner_local_tpcp = d_inner
        self.ngroups_local_tpcp = ngroups
        self.nheads_local_tpcp = nheads
        self.cp_size = 1

        # Random weights for the mock
        self.conv1d_weight = torch.randn(d_inner + 2 * ngroups * d_state, 1, 4, device=device)
        self.conv1d_bias = torch.randn(d_inner + 2 * ngroups * d_state, device=device)
        self.A_log = torch.randn(nheads, device=device)
        self.D = torch.ones(nheads, device=device)
        self.dt_bias = torch.randn(nheads, device=device)

        # Simple conv1d layer for the fallback path if needed
        self.conv1d_layer = nn.Conv1d(
            in_channels=self.conv1d_weight.shape[0],
            out_channels=self.conv1d_weight.shape[0],
            kernel_size=4,
            groups=self.conv1d_weight.shape[0],
            padding=3,
        ).to(device)

    def get_A_log(self):
        return self.A_log

    def get_D(self):
        return self.D

    def get_dt_bias(self):
        return self.dt_bias

    def get_conv1d_weight(self):
        return self.conv1d_weight

    def get_conv1d_bias(self):
        return self.conv1d_bias

    def conv1d(self, x):
        return self.conv1d_layer(x)

    def pre_conv_ssm(self, x):
        return x

    def post_conv_ssm(self, x):
        return x


class TestMambaDynamicInference(unittest.TestCase):

    # Heads on this rank.
    nheads = 8

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == 'cpu':
            self.skipTest("Mamba Triton kernels require CUDA")

        # --- Configuration ---
        self.d_model = 256
        self.d_state = 16
        self.headdim = 64
        self.d_conv = 4
        self.ngroups = 1
        self.d_inner = self.nheads * self.headdim

        # Create the Mixer instance directly
        self.mixer = MagicMock(spec=MambaMixer)
        self.mixer.config = SimpleNamespace(batch_invariant_mode=False)
        self.mixer.d_state = self.d_state
        self.mixer.d_conv = self.d_conv
        self.mixer.headdim = self.headdim
        self.mixer.chunk_size = 256
        self.mixer.activation = "silu"
        self.mixer.act = nn.SiLU()
        self.mixer.D_has_hdim = False
        self.mixer.rmsnorm = True

        # Mock the Context Parallel wrapper (used by ssm_prefill)
        self.mixer.cp = MockContextParallel(
            d_inner=self.d_inner,
            ngroups=self.ngroups,
            nheads=self.nheads,
            d_state=self.d_state,
            device=self.device,
        )

        # --- Setup for ssm_decode ---
        # ssm_decode accesses attributes directly from self, not self.cp
        self.mixer.d_inner_local_tp = self.d_inner
        self.mixer.ngroups_local_tp = self.ngroups
        self.mixer.nheads_local_tp = self.nheads

        # Create real parameters for ssm_decode to access
        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.mixer.conv1d_weight = nn.Parameter(
            torch.randn(conv_dim, 1, self.d_conv, device=self.device)
        )
        self.mixer.conv1d_bias = nn.Parameter(torch.randn(conv_dim, device=self.device))
        self.mixer.dt_bias = nn.Parameter(torch.randn(self.nheads, device=self.device))
        self.mixer.A_log = nn.Parameter(torch.randn(self.nheads, device=self.device))
        self.mixer.D = nn.Parameter(torch.ones(self.nheads, device=self.device))

        # Bind methods
        self.mixer.ssm_prefill = MambaMixer.ssm_prefill.__get__(self.mixer, MambaMixer)
        self.mixer.ssm_decode = MambaMixer.ssm_decode.__get__(self.mixer, MambaMixer)

    def test_ssm_prefill_padding_isolation(self):
        """
        Tests that ssm_prefill only updates states for the real request
        and that padding request states remain untouched.

        ssm_prefill reads all varlen metadata from the DynamicInferenceContext
        and expects `zxBCdt` pre-stripped to real tokens only (stripping is
        done upstream). This test passes only the real tokens, wires the
        metadata through a mock context, and verifies that only the active
        request's state is modified.
        """
        num_requests = 48
        real_seq_len = 6

        # Inputs: only real tokens (padding is stripped upstream)
        dim_inputs = self.d_inner * 2 + 2 * self.ngroups * self.d_state + self.nheads
        zxBCdt = torch.randn(real_seq_len, 1, dim_inputs, device=self.device, dtype=torch.float32)

        # Metadata: single real request
        seq_idx = torch.zeros((1, real_seq_len), dtype=torch.int32, device=self.device)

        cu_seqlens = torch.tensor([0, real_seq_len], dtype=torch.int32, device=self.device)

        batch_indices = torch.tensor([0], dtype=torch.long, device=self.device)

        # States
        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        conv_state = torch.zeros(num_requests, conv_dim, self.d_conv, device=self.device)
        ssm_state = torch.zeros(
            num_requests, self.nheads, self.headdim, self.d_state, device=self.device
        )

        # Mock the dynamic inference context. Leaving the chunk metadata (and
        # extraction buffers) unset exercises the non-precomputed fallback path,
        # which rebuilds chunk boundaries from cu_seqlens; no slot allocator means
        # intermediate-state extraction (prefix caching) is disabled.
        mamba_metadata = SimpleNamespace(
            seq_idx=seq_idx,
            cu_seqlens=cu_seqlens,
            batch_indices_prefill=batch_indices,
            intermediate_chunk_indices=None,
            intermediate_abs_positions=None,
            intermediate_real_count=None,
            cu_chunk_seqlens=None,
            last_chunk_indices=None,
            seq_idx_for_varlen=None,
            conv_seq_idx=None,
            conv_seq_start=None,
        )
        context = SimpleNamespace(mamba_metadata=mamba_metadata, mamba_slot_allocator=None)

        # Run
        self.mixer.norm = MagicMock(side_effect=lambda x, z: x * z)
        output = self.mixer.ssm_prefill(
            zxBCdt=zxBCdt, conv_state=conv_state, ssm_state=ssm_state, context=context
        )

        # Output should have real_seq_len tokens
        self.assertEqual(output.shape[0], real_seq_len)
        self.assertTrue(conv_state[0].abs().max() > 0, "Real request conv_state should be modified")

        # Verify isolation of padding states
        remaining_conv_states = conv_state[1:num_requests]
        remaining_ssm_states = ssm_state[1:num_requests]

        self.assertTrue(
            torch.allclose(remaining_conv_states, torch.zeros_like(remaining_conv_states)),
            "Conv states for padding requests (indices 1 to N-1) should remain 0",
        )
        self.assertTrue(
            torch.allclose(remaining_ssm_states, torch.zeros_like(remaining_ssm_states)),
            "SSM states for padding requests (indices 1 to N-1) should remain 0",
        )


def _head0_channels(mixer):
    """Conv channels of head 0: its own x channels, then the B and C channels all heads share."""
    conv_dim = mixer.conv1d_weight.shape[0]
    channels = torch.cat(
        [torch.arange(mixer.headdim), torch.arange(mixer.d_inner_local_tp, conv_dim)]
    )
    return channels.to(mixer.conv1d_weight.device)


def _proj_width(mixer):
    """Width of the projected zxBCdt input: z, x, B, C, then one dt per head."""
    d_inner, nheads = mixer.d_inner_local_tp, mixer.nheads_local_tp
    return 2 * d_inner + 2 * mixer.ngroups_local_tp * mixer.d_state + nheads


def _head0_columns(mixer):
    """Columns of zxBCdt that head 0 reads: its z, its x, the shared B and C, and its dt."""
    z = torch.arange(mixer.headdim, device=mixer.conv1d_weight.device)
    dt = torch.tensor([_proj_width(mixer) - mixer.nheads_local_tp], device=z.device)
    return torch.cat([z, mixer.d_inner_local_tp + _head0_channels(mixer), dt])


def _assert_head_matches(actual, expected):
    """On GPU the kernels may tile one head and several heads differently, so allow rounding
    error relative to the tensor's scale; a head reading the wrong parameters is off by O(1)."""
    scale = expected.abs().max().item()
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3 * scale)


class TestMambaDynamicInferenceSingleHead(TestMambaDynamicInference):
    """The same tests on a rank that holds one head (nheads_local_tp == 1), plus a check that
    the head computes what head 0 of a multi-head rank computes from the same inputs."""

    nheads = 1

    def setUp(self):
        # The multi-head stub first, then this one with head 0's parameters.
        with patch.object(type(self), "nheads", TestMambaDynamicInference.nheads):
            super().setUp()
        self.reference = self.mixer
        super().setUp()

        channels = _head0_channels(self.reference)
        with torch.no_grad():
            # ssm_prefill reads the context-parallel copies, ssm_decode the mixer's own.
            for src, dst in [(self.reference.cp, self.mixer.cp), (self.reference, self.mixer)]:
                dst.conv1d_weight.copy_(src.conv1d_weight[channels])
                dst.conv1d_bias.copy_(src.conv1d_bias[channels])
                for name in ("A_log", "D", "dt_bias"):
                    getattr(dst, name).copy_(getattr(src, name)[:1])

        for mixer in (self.reference, self.mixer):
            mixer.norm = lambda y, z: y * z
            # Decode reads -exp(A_log) from the inference cache.
            mixer._get_decode_A_neg_exp = MambaMixer._get_decode_A_neg_exp.__get__(
                mixer, MambaMixer
            )
            mixer.training = False
            mixer._A_neg_exp_cache = -torch.exp(mixer.A_log.float())
            mixer._A_neg_exp_cache_stale = False

    def _states(self, *leading):
        """Random conv and SSM states, or per-token snapshots of them, for the multi-head stub.

        Nonzero, as a chunked-prefill or decode step continues from.
        """
        conv = torch.randn(
            *leading, self.reference.conv1d_weight.shape[0], self.d_conv, device=self.device
        )
        ssm = torch.randn(
            *leading, self.reference.nheads_local_tp, self.headdim, self.d_state, device=self.device
        )
        return conv, ssm

    def _head0(self, conv, ssm):
        """Head 0's part of multi-head conv and SSM states."""
        return conv[..., _head0_channels(self.reference), :], ssm[..., :1, :, :]

    def test_ssm_prefill_matches_head0_of_multi_head(self):
        """Prefill gives the head the output and states head 0 of a multi-head rank gets."""
        seq_lens = [300, 37]  # the first spans two chunks
        batch_indices = torch.tensor([2, 0], dtype=torch.long, device=self.device)
        # Unset chunk metadata takes the non-precomputed path, as in the test above.
        mamba_metadata = SimpleNamespace(
            seq_idx=torch.repeat_interleave(
                torch.arange(len(seq_lens), dtype=torch.int32, device=self.device),
                torch.tensor(seq_lens, device=self.device),
            ).unsqueeze(0),
            cu_seqlens=torch.tensor([0, 300, 337], dtype=torch.int32, device=self.device),
            batch_indices_prefill=batch_indices,
            intermediate_chunk_indices=None,
            intermediate_abs_positions=None,
            intermediate_real_count=None,
            cu_chunk_seqlens=None,
            last_chunk_indices=None,
            seq_idx_for_varlen=None,
            conv_seq_idx=None,
            conv_seq_start=None,
        )
        context = SimpleNamespace(mamba_metadata=mamba_metadata, mamba_slot_allocator=None)
        zxBCdt = torch.randn(sum(seq_lens), 1, _proj_width(self.reference), device=self.device)
        conv_ref, ssm_ref = self._states(4)
        conv_state, ssm_state = (s.clone() for s in self._head0(conv_ref, ssm_ref))

        y_ref = self.reference.ssm_prefill(zxBCdt, conv_ref, ssm_ref, context)
        y = self.mixer.ssm_prefill(
            zxBCdt[..., _head0_columns(self.reference)], conv_state, ssm_state, context
        )

        _assert_head_matches(y, y_ref[..., : self.headdim])
        for state, state_ref in zip((conv_state, ssm_state), self._head0(conv_ref, ssm_ref)):
            _assert_head_matches(state, state_ref)

    def test_ssm_decode_matches_head0_of_multi_head(self):
        """So does decode, for one token and for a speculative step with per-token states."""
        batch_indices = torch.tensor([2, 0], dtype=torch.int32, device=self.device)
        for seq_len in (1, 3):
            with self.subTest(seq_len=seq_len):
                zxBCdt = torch.randn(
                    len(batch_indices), seq_len, _proj_width(self.reference), device=self.device
                )
                states_ref = self._states(4)
                snapshots_ref = self._states(4, seq_len) if seq_len > 1 else (None, None)
                states = [s.clone() for s in self._head0(*states_ref)]
                snapshots = snapshots_ref
                if seq_len > 1:
                    snapshots = [s.clone() for s in self._head0(*snapshots_ref)]

                with torch.no_grad():
                    y_ref = self.reference.ssm_decode(
                        zxBCdt, *states_ref, batch_indices, *snapshots_ref
                    )
                    y = self.mixer.ssm_decode(
                        zxBCdt[..., _head0_columns(self.reference)],
                        *states,
                        batch_indices,
                        *snapshots,
                    )

                _assert_head_matches(y, y_ref[..., : self.headdim])
                for state, state_ref in zip(states, self._head0(*states_ref)):
                    _assert_head_matches(state, state_ref)
                if seq_len > 1:
                    for snapshot, snapshot_ref in zip(snapshots, self._head0(*snapshots_ref)):
                        _assert_head_matches(snapshot, snapshot_ref)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
