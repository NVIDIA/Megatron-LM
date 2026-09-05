# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the ReplaySSM speculative-decode kernels (mamba_ssm_replay.py).

Oracle layering (mirrors the ReplaySSM upstream test suite):
  * the baseline `selective_state_update` kernel, stepped over the same token
    window from the same checkpoint, provides per-position ground-truth outputs
    and states;
  * a non-flush verify must leave the checkpoint BITWISE untouched;
  * a multi-step random-acceptance rollback drives the scatter/verify/flush and
    commit kernels through ring wraparound and flush boundaries, checking
    accepted-position outputs against the baseline and the checkpoint against
    baseline snapshots keyed by the cumulative folded-token count.
"""

import pytest
import torch

from megatron.core.ssm.ops.mamba2.mamba_ssm import selective_state_update
from megatron.core.ssm.ops.mamba2.mamba_ssm_replay import (
    commit_replayssm_spec,
    selective_state_update_replayssm_spec,
)

DEVICE = "cuda"


def _requires_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def _tolerances(state_dtype, act_dtype):
    if state_dtype == torch.float32 and act_dtype == torch.float32:
        return 1e-4, 1e-3
    return 6e-2, 2e-1


def _geometry(name):
    # (nheads, headdim, dstate, ngroups)
    return {"small": (8, 64, 64, 4), "tiny": (4, 64, 16, 1), "super120b": (128, 64, 128, 8)}[name]


def _make_tied_inputs(nheads, headdim, dstate, device, dtype=torch.float32):
    """Per-head scalar A (< 0) and dt_bias with the TIE_HDIM broadcast layout."""
    A_head = -torch.rand(nheads, device=device, dtype=torch.float32) - 1.0
    A = A_head.view(nheads, 1, 1).expand(nheads, headdim, dstate)
    dt_bias = torch.rand(nheads, device=device, dtype=torch.float32) - 4.0
    D = torch.randn(nheads, device=device, dtype=torch.float32)
    return A, dt_bias, D


def _pack_conv_out(x, B, C):
    """Pack (S, H, P), (S, G, N), (S, G, N) -> (S, H*P + 2*G*N) channel-last."""
    S = x.shape[0]
    return torch.cat([x.reshape(S, -1), B.reshape(S, -1), C.reshape(S, -1)], dim=-1)


def _baseline_window(state, x, dt, A, B, C, D, dt_bias, headdim):
    """Run the baseline selective_state_update over a (1, S, ...) window in place
    on `state` and return the per-position outputs (S, H, P)."""
    S, nheads = dt.shape
    dt_b = dt.unsqueeze(0).unsqueeze(-1).expand(1, S, nheads, headdim)
    dt_bias_b = dt_bias.unsqueeze(-1).expand(nheads, headdim)
    D_b = D.unsqueeze(-1).expand(nheads, headdim)
    # Always provide a (throwaway) intermediate buffer: without one the baseline
    # launcher passes x as a zero-stride dummy pointer for the intermediate-state
    # dump, which scribbles state values onto x[0, 0, 0, 0] and races other
    # programs' loads of x, corrupting the oracle.
    dstate = state.shape[-1]
    scratch = torch.empty(
        1, S, nheads, headdim, dstate, device=x.device, dtype=state.dtype
    )
    out = selective_state_update(
        state,
        x.unsqueeze(0),
        dt_b,
        A,
        B.unsqueeze(0),
        C.unsqueeze(0),
        D_b,
        z=None,
        dt_bias=dt_bias_b,
        dt_softplus=True,
        state_batch_indices=torch.zeros(1, dtype=torch.int64, device=x.device),
        intermediate_ssm_states=scratch,
    )
    return out.squeeze(0).float()


def _scatter_history(conv_cache, dt_cache, slot, x_hist, B_hist, dt_hist, d_inner):
    """Write committed history [0, wp) into the ring at origin 0 (test setup)."""
    wp = x_hist.shape[0]
    if wp == 0:
        return
    conv_cache[slot, :wp, :d_inner] = x_hist.reshape(wp, -1).to(conv_cache.dtype)
    conv_cache[slot, :wp, d_inner:] = B_hist.reshape(wp, -1).to(conv_cache.dtype)
    dt_cache[slot, :, :wp] = dt_hist.t().to(dt_cache.dtype)


class _ReplayRig:
    """Single-layer ReplaySSM buffers + geometry for kernel-level tests."""

    def __init__(self, num_slots, nheads, headdim, dstate, ngroups, buffer_len, window,
                 state_dtype=torch.float32, act_dtype=torch.float32):
        self.nheads, self.headdim, self.dstate, self.ngroups = nheads, headdim, dstate, ngroups
        self.d_inner = nheads * headdim
        self.window = window
        self.L = buffer_len + window
        self.buf = 1 << (self.L - 1).bit_length()
        self.block_spec = 1 << (window - 1).bit_length()
        self.state = torch.randn(
            num_slots, nheads, headdim, dstate, device=DEVICE, dtype=state_dtype
        ) * 0.1
        cache_conv_dim = self.d_inner + ngroups * dstate
        self.conv_cache = torch.zeros(
            num_slots, self.buf, cache_conv_dim, device=DEVICE, dtype=act_dtype
        )
        self.dt_cache = torch.zeros(num_slots, nheads, self.buf, device=DEVICE, dtype=torch.float32)
        self.write_pos = torch.zeros(num_slots, dtype=torch.int32, device=DEVICE)
        self.origin = torch.zeros(num_slots, dtype=torch.int32, device=DEVICE)
        self.is_flush = torch.zeros(num_slots, dtype=torch.int8, device=DEVICE)

    def run_step(self, conv_out, dt_raw, A, D, dt_bias, sbi, qsl, bc_pre=None):
        return selective_state_update_replayssm_spec(
            state_checkpoint=self.state,
            post_conv_cache=self.conv_cache,
            dt_cache=self.dt_cache,
            conv_out=conv_out,
            dt_spec=dt_raw,
            A=A,
            write_pos=self.write_pos,
            post_conv_state_pos=self.origin,
            is_flush=self.is_flush,
            query_start_loc=qsl,
            state_batch_indices=sbi,
            max_cache_len=self.L,
            max_spec_len=self.window,
            d_inner=self.d_inner,
            ngroups=self.ngroups,
            dstate=self.dstate,
            D=D.unsqueeze(-1).expand(self.nheads, self.headdim),
            z=None,
            dt_bias=dt_bias,
            dt_softplus=True,
            bc_pre=bc_pre,
        )


@pytest.mark.internal
class TestReplaySpecVerify:
    """Single-step verify: outputs match the baseline recurrence; the checkpoint
    is bitwise untouched on non-flush rows; padded rows produce zeros."""

    def setup_method(self, method):
        _requires_cuda()

    @pytest.mark.parametrize("geometry", ["small", "tiny", "super120b"])
    @pytest.mark.parametrize("window", [2, 3, 5])
    @pytest.mark.parametrize(
        "state_dtype,act_dtype",
        [(torch.float32, torch.float32), (torch.float32, torch.bfloat16),
         (torch.bfloat16, torch.bfloat16)],
        ids=["s32_a32", "s32_a16", "s16_a16"],
    )
    def test_verify_matches_baseline(self, geometry, window, state_dtype, act_dtype):
        torch.manual_seed(0)
        nheads, headdim, dstate, ngroups = _geometry(geometry)
        buffer_len = 16
        rig = _ReplayRig(2, nheads, headdim, dstate, ngroups, buffer_len, window,
                         state_dtype, act_dtype)
        A, dt_bias, D = _make_tied_inputs(nheads, headdim, dstate, DEVICE)
        slot = 1
        C_fill = buffer_len - window  # max non-flush committed fill
        for wp in sorted({0, C_fill // 2, C_fill}):
            rig.write_pos.zero_(); rig.origin.zero_(); rig.is_flush.zero_()
            rig.write_pos[slot] = wp
            # committed history + fresh window (round-tripped through the activation
            # dtype so oracle and kernel consume identical values)
            x_h = torch.randn(wp, nheads, headdim, device=DEVICE).to(act_dtype).float()
            B_h = torch.randn(wp, ngroups, dstate, device=DEVICE).to(act_dtype).float()
            dt_h = torch.randn(wp, nheads, device=DEVICE) * 0.5
            _scatter_history(rig.conv_cache, rig.dt_cache, slot, x_h, B_h, dt_h, rig.d_inner)
            x_w = torch.randn(window, nheads, headdim, device=DEVICE).to(act_dtype).float()
            B_w = torch.randn(window, ngroups, dstate, device=DEVICE).to(act_dtype).float()
            C_w = torch.randn(window, ngroups, dstate, device=DEVICE).to(act_dtype).float()
            dt_w = torch.randn(window, nheads, device=DEVICE) * 0.5

            state_before = rig.state.clone()
            conv_out = _pack_conv_out(x_w, B_w, C_w).to(act_dtype)
            sbi = torch.tensor([slot], dtype=torch.int64, device=DEVICE)
            qsl = torch.tensor([0, window], dtype=torch.int32, device=DEVICE)
            out = rig.run_step(conv_out, dt_w, A, D, dt_bias, sbi, qsl)

            # Oracle: baseline over history + window from the same checkpoint.
            st = state_before[slot : slot + 1].float().clone()
            x_all = torch.cat([x_h, x_w]); B_all = torch.cat([B_h, B_w])
            dt_all = torch.cat([dt_h, dt_w])
            # History C is irrelevant for the window outputs; use zeros.
            C_all = torch.cat([torch.zeros(wp, ngroups, dstate, device=DEVICE), C_w])
            ref = _baseline_window(st, x_all, dt_all, A, B_all, C_all, D, dt_bias, headdim)
            rtol, atol = _tolerances(state_dtype, act_dtype)
            torch.testing.assert_close(
                out[:window].float(), ref[wp:].float(), rtol=rtol, atol=atol
            )
            # Non-flush verify must not touch the checkpoint (bitwise).
            assert torch.equal(rig.state, state_before)

    def test_padded_rows_zeroed_and_unused_slots_untouched(self):
        torch.manual_seed(1)
        nheads, headdim, dstate, ngroups = _geometry("tiny")
        window = 3
        rig = _ReplayRig(6, nheads, headdim, dstate, ngroups, 8, window)
        A, dt_bias, D = _make_tied_inputs(nheads, headdim, dstate, DEVICE)
        batch = 3  # rows: slot 2, padding (-1), slot 4
        sbi = torch.tensor([2, -1, 4], dtype=torch.int64, device=DEVICE)
        qsl = torch.arange(0, (batch + 1) * window, window, dtype=torch.int32, device=DEVICE)
        x_w = torch.randn(batch * window, nheads, headdim, device=DEVICE)
        B_w = torch.randn(batch * window, ngroups, dstate, device=DEVICE)
        C_w = torch.randn(batch * window, ngroups, dstate, device=DEVICE)
        dt_w = torch.randn(batch * window, nheads, device=DEVICE) * 0.5
        conv_out = _pack_conv_out(x_w, B_w, C_w)
        state_before = rig.state.clone()
        out = rig.run_step(conv_out, dt_w, A, D, dt_bias, sbi, qsl)
        # Padding row -> zeros.
        assert torch.all(out[window : 2 * window] == 0)
        # Unused state slots bitwise untouched.
        for s in (0, 1, 3, 5):
            assert torch.equal(rig.state[s], state_before[s])


@pytest.mark.internal
class TestReplayCommit:
    """Commit kernel semantics: pointer math, early-flush flag, prefill gating."""

    def setup_method(self, method):
        _requires_cuda()

    def _commit(self, rig, accepted, prefill, sbi):
        commit_replayssm_spec(
            write_pos=rig.write_pos,
            post_conv_state_pos=rig.origin,
            is_flush=rig.is_flush,
            accepted_draft_counts=accepted,
            prefill_status=prefill,
            state_batch_indices=sbi,
            max_cache_len=rig.L,
            max_spec_len=rig.window,
            cache_buf_len=rig.buf,
        )

    def test_pointer_math_and_early_flush(self):
        nheads, headdim, dstate, ngroups = _geometry("tiny")
        window = 3  # W; L = 8 + 3 = 11, buf = 16
        rig = _ReplayRig(4, nheads, headdim, dstate, ngroups, 8, window)
        sbi = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device=DEVICE)
        prefill = torch.tensor([0, 0, 1, 0], dtype=torch.int32, device=DEVICE)
        rig.write_pos.copy_(torch.tensor([0, 4, 5, 2], dtype=torch.int32))
        rig.origin.copy_(torch.tensor([0, 6, 1, 2], dtype=torch.int32))
        rig.is_flush.copy_(torch.tensor([0, 1, 0, 0], dtype=torch.int8))
        accepted = torch.tensor([2, 1, 2, 0], dtype=torch.int64, device=DEVICE)
        self._commit(rig, accepted, prefill, sbi)
        # Row 0: non-flush -> wp = 0 + (2+1) = 3; origin unchanged.
        # Row 1: flush -> origin = (6+4) & 15 = 10; wp = (1+1) = 2.
        # Row 2: prefill -> untouched.
        # Row 3: non-flush -> wp = 2 + 1 = 3.
        assert rig.write_pos.cpu().tolist() == [3, 2, 5, 3]
        assert rig.origin.cpu().tolist() == [0, 10, 1, 2]
        # next_is_flush = (wp + 2W > L) = (wp + 6 > 11)
        assert rig.is_flush.cpu().tolist() == [0, 0, 0, 0]
        # Push row 0 over the early-flush threshold: wp 3 -> 6, 6 + 6 > 11.
        self._commit(rig, accepted, prefill, sbi)
        assert rig.write_pos.cpu().tolist()[0] == 6
        assert rig.is_flush.cpu().tolist()[0] == 1

    def test_null_slot_skipped(self):
        nheads, headdim, dstate, ngroups = _geometry("tiny")
        rig = _ReplayRig(2, nheads, headdim, dstate, ngroups, 8, 2)
        sbi = torch.tensor([-1, 1], dtype=torch.int64, device=DEVICE)
        prefill = torch.zeros(2, dtype=torch.int32, device=DEVICE)
        accepted = torch.ones(2, dtype=torch.int64, device=DEVICE)
        self._commit(rig, accepted, prefill, sbi)
        assert rig.write_pos.cpu().tolist() == [0, 2]


@pytest.mark.internal
class TestReplayMultiStepRollback:
    """40 steps of scatter/verify/flush + commit with random per-step acceptance,
    tracking the baseline recurrence over the accepted token stream. Exercises
    ring wraparound, flush cadence, and rollback."""

    def setup_method(self, method):
        _requires_cuda()

    @pytest.mark.parametrize("buffer_len", [8, 16])
    @pytest.mark.parametrize("window", [2, 3, 5])
    @pytest.mark.parametrize(
        "state_dtype,act_dtype",
        [(torch.float32, torch.float32), (torch.float32, torch.bfloat16)],
        ids=["s32_a32", "s32_a16"],
    )
    def test_rollback_tracks_baseline(self, buffer_len, window, state_dtype, act_dtype):
        torch.manual_seed(42)
        gen = torch.Generator().manual_seed(43)
        nheads, headdim, dstate, ngroups = _geometry("small")
        rig = _ReplayRig(2, nheads, headdim, dstate, ngroups, buffer_len, window,
                         state_dtype, act_dtype)
        A, dt_bias, D = _make_tied_inputs(nheads, headdim, dstate, DEVICE)
        slot = 1
        sbi = torch.tensor([slot], dtype=torch.int64, device=DEVICE)
        qsl = torch.tensor([0, window], dtype=torch.int32, device=DEVICE)
        prefill = torch.zeros(1, dtype=torch.int32, device=DEVICE)
        rtol, atol = _tolerances(state_dtype, act_dtype)

        state_base = rig.state[slot : slot + 1].float().clone()
        snapshots = {0: state_base.clone()}
        total_accepted = 0
        n_folded = 0

        for step in range(40):
            # Round-trip window inputs through the activation dtype so the oracle
            # consumes exactly the values the kernel caches (otherwise the input
            # rounding gap compounds across flush folds over 40 steps).
            x_w = torch.randn(window, nheads, headdim, device=DEVICE).to(act_dtype).float()
            B_w = torch.randn(window, ngroups, dstate, device=DEVICE).to(act_dtype).float()
            C_w = torch.randn(window, ngroups, dstate, device=DEVICE).to(act_dtype).float()
            dt_w = torch.randn(window, nheads, device=DEVICE) * 0.5
            conv_out = _pack_conv_out(x_w, B_w, C_w).to(act_dtype)

            wp_before = int(rig.write_pos[slot].item())
            flush_before = int(rig.is_flush[slot].item())

            out = rig.run_step(conv_out, dt_w, A, D, dt_bias, sbi, qsl)
            if flush_before and wp_before > 0:
                n_folded += wp_before

            # accepted drafts in [0, window - 1]; total commit = accepted + 1
            k_drafts = int(torch.randint(0, window, (1,), generator=gen).item())
            n_commit = k_drafts + 1

            # Baseline consumes exactly the committed tokens.
            ref = _baseline_window(
                state_base, x_w[:n_commit], dt_w[:n_commit], A, B_w[:n_commit],
                C_w[:n_commit], D, dt_bias, headdim,
            )
            for _ in range(n_commit):
                total_accepted += 1
            snapshots[total_accepted] = state_base.clone()
            torch.testing.assert_close(
                out[:n_commit].float(), ref, rtol=rtol, atol=atol
            )

            accepted = torch.tensor([k_drafts], dtype=torch.int64, device=DEVICE)
            commit_replayssm_spec(
                write_pos=rig.write_pos,
                post_conv_state_pos=rig.origin,
                is_flush=rig.is_flush,
                accepted_draft_counts=accepted,
                prefill_status=prefill,
                state_batch_indices=sbi,
                max_cache_len=rig.L,
                max_spec_len=rig.window,
                cache_buf_len=rig.buf,
            )

            if n_folded in snapshots:
                torch.testing.assert_close(
                    rig.state[slot].float(),
                    snapshots[n_folded].squeeze(0),
                    rtol=rtol,
                    atol=atol,
                )

        assert n_folded > 0, "the ring never flushed; the test is vacuous"

    def test_perturbation_is_detected(self):
        """Anti-vacuity: corrupting one cached history value must break parity."""
        torch.manual_seed(7)
        nheads, headdim, dstate, ngroups = _geometry("small")
        window = 3
        rig = _ReplayRig(2, nheads, headdim, dstate, ngroups, 16, window)
        A, dt_bias, D = _make_tied_inputs(nheads, headdim, dstate, DEVICE)
        slot = 1
        wp = 5
        rig.write_pos[slot] = wp
        x_h = torch.randn(wp, nheads, headdim, device=DEVICE)
        B_h = torch.randn(wp, ngroups, dstate, device=DEVICE)
        dt_h = torch.randn(wp, nheads, device=DEVICE) * 0.5
        _scatter_history(rig.conv_cache, rig.dt_cache, slot, x_h, B_h, dt_h, rig.d_inner)
        rig.conv_cache[slot, wp - 1, 0] += 5.0  # corrupt one cached x value
        x_w = torch.randn(window, nheads, headdim, device=DEVICE)
        B_w = torch.randn(window, ngroups, dstate, device=DEVICE)
        C_w = torch.randn(window, ngroups, dstate, device=DEVICE)
        dt_w = torch.randn(window, nheads, device=DEVICE) * 0.5
        sbi = torch.tensor([slot], dtype=torch.int64, device=DEVICE)
        qsl = torch.tensor([0, window], dtype=torch.int32, device=DEVICE)
        out = rig.run_step(_pack_conv_out(x_w, B_w, C_w), dt_w, A, D, dt_bias, sbi, qsl)
        st = rig.state[slot : slot + 1].float().clone()
        x_all = torch.cat([x_h, x_w]); B_all = torch.cat([B_h, B_w])
        dt_all = torch.cat([dt_h, dt_w])
        C_all = torch.cat([torch.zeros(wp, ngroups, dstate, device=DEVICE), C_w])
        ref = _baseline_window(st, x_all, dt_all, A, B_all, C_all, D, dt_bias, headdim)
        with pytest.raises(AssertionError):
            torch.testing.assert_close(out[:window].float(), ref[wp:], rtol=1e-4, atol=1e-3)
