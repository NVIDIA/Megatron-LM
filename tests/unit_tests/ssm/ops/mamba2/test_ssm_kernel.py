# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.inference.contexts.attention_context.mamba_ssd_metadata import compute_ssd_tiling

# Assume the provided class is in mamba_mixer.py
from megatron.core.ssm.mamba_mixer import MambaMixer
from megatron.core.ssm.ops.ssd_backend import cutedsl_ssd_available


def _build_varlen_ssd_inputs(seq_lens, chunk_size, nheads, headdim, ngroups, dstate, device, dtype):
    """Construct token-packed (THD) varlen inputs for the SSD scan.

    All sequence lengths must be multiples of ``chunk_size`` (the regime the
    CuteDSL kernel accelerates). Returns a kwargs dict shared by both backends.
    """
    total = sum(seq_lens)
    x = torch.randn(total, nheads, headdim, device=device, dtype=dtype)
    dt = torch.randn(total, nheads, device=device, dtype=dtype)
    A = -torch.exp(torch.randn(nheads, device=device, dtype=torch.float32))
    B = torch.randn(total, ngroups, dstate, device=device, dtype=dtype)
    C = torch.randn(total, ngroups, dstate, device=device, dtype=dtype)
    D = torch.ones(nheads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(nheads, device=device, dtype=torch.float32)

    chunk_boundaries = [0]
    last_chunk_indices = []
    seq_idx_per_chunk = []
    cu = [0]
    for s in seq_lens:
        cu.append(cu[-1] + s)
    for i, s in enumerate(seq_lens):
        start, end = cu[i], cu[i + 1]
        pos = start + chunk_size
        while pos < end:
            chunk_boundaries.append(pos)
            seq_idx_per_chunk.append(i)
            pos += chunk_size
        chunk_boundaries.append(end)
        seq_idx_per_chunk.append(i)
        last_chunk_indices.append(len(chunk_boundaries) - 2)

    return dict(
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        D=D,
        dt_bias=dt_bias,
        chunk_size=chunk_size,
        cu_chunk_seqlens=torch.tensor(chunk_boundaries, dtype=torch.int32, device=device),
        last_chunk_indices=torch.tensor(last_chunk_indices, dtype=torch.int64, device=device),
        seq_idx=torch.tensor(seq_idx_per_chunk, dtype=torch.int32, device=device),
    )


def _emit_per_seq_indices(seq_lens, chunk_size, device):
    """One intermediate-emit chunk per multi-chunk sequence (global chunk indices).

    The offset ``min(i + 1, n_chunks - 1)`` varies the emit position across
    sequences (interior chunks for early sequences, up to the last chunk for
    later ones); single-chunk and empty sequences emit nothing.
    """
    # CEIL chunk counts: a ragged tail owns a (partial) chunk of its own, which
    # is how both the caller's cu_chunk_seqlens and the kernel number chunks.
    starts = [0]
    for s in seq_lens:
        starts.append(starts[-1] + -(-s // chunk_size))
    emit = []
    for i, s in enumerate(seq_lens):
        n_chunks = -(-s // chunk_size)
        if n_chunks > 1:
            emit.append(starts[i] + min(i + 1, n_chunks - 1))
    return torch.tensor(emit, dtype=torch.int64, device=device)


def _ssd_case(seq_lens, **overrides):
    """One CuteDSL-vs-Triton parity case; unset knobs take the defaults below."""
    case = dict(
        seq_lens=seq_lens,
        chunk_size=128,  # intermediate emission requires chunk_size == kernel L (128)
        nheads=8,
        dstate=128,
        with_initial=False,  # chunked prefill: non-zero carried SSM state
        inter=None,  # None | "per_seq" | explicit index list (may contain duplicates)
        pad_to=None,  # trailing token-buffer padding (fixed-size CUDA-graph buffers)
        expect_fallback=False,  # cutedsl_unsupported_reason must reject (Triton fallback)
    )
    assert set(overrides) <= set(case), f"unknown case keys: {set(overrides) - set(case)}"
    case.update(overrides)
    return case


_PARITY_CASES = [
    pytest.param(_ssd_case([256] * 4, chunk_size=256, nheads=16), id="divisible-equal"),
    pytest.param(
        _ssd_case([512, 256, 768, 256], chunk_size=256, nheads=16), id="divisible-unequal"
    ),
    pytest.param(
        _ssd_case([256, 512, 256], chunk_size=256, nheads=16, dstate=16), id="divisible-d16"
    ),
    pytest.param(_ssd_case([256, 512], chunk_size=256, pad_to=1024), id="padded-buffer"),
    pytest.param(_ssd_case([256, 512, 256], chunk_size=256, with_initial=True), id="prefill"),
    pytest.param(_ssd_case([2048, 0, 0, 0], with_initial=True), id="empty-slots-prefill"),
    pytest.param(
        _ssd_case([384, 256, 512], inter="per_seq", with_initial=True), id="prefix-prefill"
    ),
    pytest.param(_ssd_case([384, 256, 512], inter=[0] * 12), id="inter-duplicate-padding"),
    pytest.param(
        _ssd_case([256, 200], chunk_size=128, pad_to=512, with_initial=True),
        id="tail-ragged-prefill",
    ),
    pytest.param(_ssd_case([384, 127], chunk_size=128, pad_to=512), id="tail-ragged-l-minus-1"),
    pytest.param(_ssd_case([200, 256], chunk_size=128, pad_to=512), id="ragged-interior"),
    pytest.param(
        _ssd_case([300, 500, 700, 0, 0], chunk_size=128, pad_to=1536, with_initial=True),
        id="ragged-all-empty-prefill",
    ),
    pytest.param(_ssd_case([1, 1, 1, 1], chunk_size=128, pad_to=128), id="ragged-single-tokens"),
    pytest.param(
        _ssd_case([256, 200], chunk_size=128, expect_fallback=True), id="ragged-unpadded-fallback"
    ),
    pytest.param(
        _ssd_case([200, 256], chunk_size=128, pad_to=512, inter="per_seq", expect_fallback=True),
        id="ragged-inter-fallback",
    ),
    pytest.param(
        _ssd_case([0, 256, 0], with_initial=True, inter="per_seq", expect_fallback=True),
        id="empty-interleaved-inter-fallback",
    ),
]


def _torch_reference_varlen(kernel_args: dict, real_num_tokens: int):
    """Exact fp32 per-token recurrence for the varlen SSD op contract.

    Recurrence compute
    ``state = exp(dt*A) * state + dt * outer(x, B); y = state @ C + D * x``,
    seeding from ``initial_states``.
    """
    ccs = kernel_args["cu_chunk_seqlens"].long()
    lci = kernel_args["last_chunk_indices"].long()
    x = kernel_args["x"].float()
    B, C = kernel_args["B"].float(), kernel_args["C"].float()
    A, D = kernel_args["A"].float(), kernel_args["D"].float()
    dt = kernel_args["dt"].float() + kernel_args["dt_bias"].float()
    if kernel_args["dt_softplus"]:
        dt = F.softplus(dt)
    lo, hi = kernel_args["dt_limit"]
    if lo != 0.0 or hi != float("inf"):
        dt = torch.clamp(dt, lo, hi)

    H, P = x.shape[1], x.shape[2]
    G, N = B.shape[1], B.shape[2]
    ratio = H // G
    n_chunks = ccs.numel() - 1
    # chunk -> owning sequence (empty sequences own one zero-length chunk).
    chunk_seq, prev = [], -1
    for s, last in enumerate(lci.tolist()):
        chunk_seq.extend([s] * (last - prev))
        prev = last
    S = len(lci)

    init = kernel_args["initial_states"]
    seq_state = [
        (init[s].float().clone() if init is not None else x.new_zeros(H, P, N)) for s in range(S)
    ]
    out = x.new_zeros(real_num_tokens, H, P)
    state_after_chunk = x.new_zeros(n_chunks, H, P, N)
    for c in range(n_chunks):
        s = chunk_seq[c]
        state = seq_state[s]
        for t in range(int(ccs[c]), int(ccs[c + 1])):
            decay = torch.exp(dt[t] * A)  # (H,)
            Bh = B[t].repeat_interleave(ratio, 0)  # (H, N)
            Ch = C[t].repeat_interleave(ratio, 0)
            state = state * decay[:, None, None] + dt[t][:, None, None] * (
                x[t][:, :, None] * Bh[:, None, :]
            )
            out[t] = (state * Ch[:, None, :]).sum(-1) + D[:, None] * x[t]
        seq_state[s] = state
        state_after_chunk[c] = state

    final = torch.stack(seq_state)
    idx = kernel_args["intermediate_chunk_indices"]
    inter = state_after_chunk[idx] if idx is not None else None
    return out, final, inter


def _dt_preprocess_reference(dt, A, dt_bias, dt_softplus, dt_limit):
    """Token-major ``(T, H)`` fp32 delta: bias, softplus and clamp, in that order."""
    dt_f = dt.float()
    if dt_bias is not None:
        dt_f = dt_f + dt_bias.float()
    if dt_softplus:
        dt_f = torch.nn.functional.softplus(dt_f)
    lo, hi = dt_limit
    if lo != 0.0 or hi != float("inf"):
        dt_f = torch.clamp(dt_f, lo, hi)
    return dt_f


def _repack_reference(src, N, N_pad, TC, L):
    """The strided view copy the tiled B/C repack kernel replaces."""
    G = src.shape[1]
    GN = G * N
    dst = torch.zeros(1, G, N_pad, TC, L, device=src.device, dtype=src.dtype)
    dst[:, :, :N].copy_(src.as_strided((1, G, N, TC, L), (0, N, 1, L * GN, GN)))
    return dst


@pytest.mark.skipif(
    not cutedsl_ssd_available(),
    reason="CuteDSL SSD backend requires Blackwell (SM 10.0+) and the cutlass DSL runtime",
)
class TestCuteDSLAccuracy:
    """The CuteDSL SSD kernel must produce near-identical results to BOTH
    references — the Triton kernel and the exact torch recurrence — across every
    supported varlen prefill shape (divisible sequence lengths) and inference
    feature (chunked prefill, prefix caching, padded/empty engine batches), so
    all three implementations are pinned to the same op contract. Unsupported
    shapes must be rejected by cutedsl_unsupported_reason so the dispatcher
    falls back to Triton. All scenarios share one skeleton (_PARITY_CASES).

    The two Triton helpers the CuteDSL path launches before its own kernel —
    the fused dt preprocessing and the tiled B/C repack — are pinned here too,
    each against its own torch reference. Testing them next to the kernel they
    feed keeps a failure attributable: a helper test failing alongside the
    parity cases says the input staging broke, not the SSD math."""

    @pytest.mark.parametrize("ref_backend", ["triton", "pytorch"])
    @pytest.mark.parametrize("params", _PARITY_CASES)
    def test_cutedsl_matches_ref(self, params: dict, ref_backend: str):
        from megatron.core.ssm.ops.cutedsl_mamba2_ssd import (
            cutedsl_unsupported_reason,
            mamba_chunk_scan_combined_varlen_cutedsl_thd,
        )
        from megatron.core.ssm.ops.mamba2.ssd_combined import (
            _mamba_chunk_scan_combined_varlen_triton,
        )

        torch.manual_seed(42)
        device = torch.device("cuda")
        seq_lens, chunk_size = params["seq_lens"], params["chunk_size"]
        headdim, ngroups = 64, 2
        real_num_tokens = sum(seq_lens)

        # The torch reference walks tokens one by one, so it handles a partial
        # final chunk natively (which pins the tail-ragged math against a third
        # implementation); it is only restricted to equal-length batches.
        if ref_backend == "pytorch" and (len(set(seq_lens)) != 1 or seq_lens[0] == 0):
            pytest.skip("torch reference supports only equal sequence lengths")

        common_kernel_args = _build_varlen_ssd_inputs(
            seq_lens,
            chunk_size,
            params["nheads"],
            headdim,
            ngroups,
            params["dstate"],
            device,
            torch.bfloat16,
        )
        if params["pad_to"] is not None:
            assert params["pad_to"] > real_num_tokens, "test must actually pad"
            pad = params["pad_to"] - real_num_tokens
            for k in ("x", "dt", "B", "C"):
                t = common_kernel_args[k]
                common_kernel_args[k] = torch.cat(
                    [t, torch.randn(pad, *t.shape[1:], device=device, dtype=t.dtype)], dim=0
                )
        if params["inter"] == "per_seq":
            idx = _emit_per_seq_indices(seq_lens, chunk_size, device)
        elif params["inter"] is not None:
            idx = torch.tensor(params["inter"], dtype=torch.int64, device=device)
        else:
            idx = None
        initial_states = (
            torch.randn(
                len(seq_lens),
                params["nheads"],
                headdim,
                params["dstate"],
                device=device,
                dtype=torch.float32,
            )
            if params["with_initial"]
            else None
        )
        kernel_args = dict(
            z=None,
            initial_states=initial_states,
            dt_softplus=True,
            dt_limit=(0.0, float("inf")),
            state_dtype=torch.float32,
            intermediate_chunk_indices=idx,
            **common_kernel_args,
        )

        if ref_backend == "triton":
            # Triton no longer gathers flagged chunks itself: it returns the
            # dense all-chunk states via return_raw_states, and the caller
            # indexes them. That gather IS the contract the CuteDSL wrapper's
            # sparse emission must reproduce.
            triton_args = {
                k: v for k, v in kernel_args.items() if k != "intermediate_chunk_indices"
            }
            out_ref = torch.empty_like(common_kernel_args["x"])
            ref = _mamba_chunk_scan_combined_varlen_triton(
                out=out_ref, return_raw_states=idx is not None, **triton_args
            )
            if idx is not None:
                final_ref, raw_ref = ref
                inter_ref = raw_ref[idx]
            else:
                final_ref, inter_ref = ref, None
            out_ref = out_ref[:real_num_tokens]
        else:
            out_ref, final_ref, inter_ref = _torch_reference_varlen(kernel_args, real_num_tokens)

        # The engine derives this once per step; op-level callers build it here.
        cu_list = list(itertools.accumulate(seq_lens, initial=0))
        tiling = _ssd_tiling_for_test(cu_list, 128, device)
        reason = cutedsl_unsupported_reason(
            x=kernel_args["x"],
            chunk_size=kernel_args["chunk_size"],
            tiling=tiling,
            z=kernel_args["z"],
            return_raw_states=False,
            intermediate_chunk_indices=kernel_args["intermediate_chunk_indices"],
        )
        if params["expect_fallback"]:
            assert reason is not None, "eligibility check should reject this case"
            pytest.skip(reason)
        assert reason is None, f"unexpected fallback: {reason}"

        out_cute = torch.empty_like(common_kernel_args["x"])
        cute = mamba_chunk_scan_combined_varlen_cutedsl_thd(
            out=out_cute, tiling=tiling, **kernel_args
        )

        if idx is not None:
            final_cute, inter_cute = cute
            # Every emitted slot must match, including duplicated/padding slots.
            torch.testing.assert_close(
                inter_cute, inter_ref.to(inter_cute.dtype), rtol=3e-2, atol=3e-2
            )
        else:
            final_cute = cute
        # bf16 MMAs + padded dstate are not bitwise equal; compare every REAL
        # token elementwise (padding outputs are undefined for both backends).
        # atol is one bf16 ulp at the typical accumulated-term magnitude (~64):
        # where y is a small difference of large terms, no two independently
        # tiled bf16 kernels agree below ulp(|terms|), even vs an fp64 reference.
        torch.testing.assert_close(
            out_cute[:real_num_tokens], out_ref.to(out_cute.dtype), rtol=2e-2, atol=0.25
        )
        torch.testing.assert_close(final_cute, final_ref.to(final_cute.dtype), rtol=3e-2, atol=3e-2)

    @pytest.mark.parametrize("use_bias", [True, False])
    @pytest.mark.parametrize("layout", ["divisible", "aligned"])
    def test_fused_softplus_cumsum_matches_torch(self, layout, use_bias):
        """The Triton kernel that fuses dt preprocessing (bias + softplus + clamp +
        cumsum) must match the torch reference in both the divisible
        (B=1, C=total_chunks) and aligned (B=S, C=Cmax) output layouts."""
        from megatron.core.ssm.ops.cutedsl_mamba2_ssd._fused_cumsum import fused_softplus_cumsum

        torch.manual_seed(0)
        device, dtype, L, H = "cuda", torch.bfloat16, 128, 6
        A = -torch.exp(torch.randn(H, device=device))
        dt_bias = torch.randn(H, device=device) if use_bias else None

        if layout == "divisible":
            B, C = 1, 5  # (1, H, total_chunks, L)
            T = C * L
        else:
            B, C = 3, 4  # aligned: S sequences, Cmax chunks each; seqlen0 = Cmax * L
            T = B * C * L
        dt = torch.randn(T, H, device=device, dtype=dtype) * 0.5 - 2.0

        delta = torch.zeros(B, H, C, L, device=device, dtype=dtype)
        cumsum = torch.zeros(B, H, C, L, device=device, dtype=torch.float32)
        fused_softplus_cumsum(dt, A, dt_bias, True, (0.0, float("inf")), delta, cumsum, B, H, C)

        dt_f = _dt_preprocess_reference(dt, A, dt_bias, True, (0.0, float("inf")))
        ref_delta = dt_f.view(B, C, L, H).permute(0, 3, 1, 2)  # (B, H, C, L)
        ref_cumsum = torch.cumsum(ref_delta.float() * A.view(1, H, 1, 1), dim=-1)

        torch.testing.assert_close(delta.float(), ref_delta, rtol=5e-3, atol=5e-3)
        torch.testing.assert_close(cumsum, ref_cumsum, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize(
        "G,N,N_pad,trim",
        [
            # production dims, reading a trimmed slice of a padded token buffer
            (8, 128, 128, True),
            # small dstate: the padding rows n >= 16 must stay zero
            (2, 16, 128, False),
        ],
    )
    def test_bc_repack_matches_strided_copy(self, G, N, N_pad, trim):
        """The tiled B/C repack (token-major -> dense chunk-major) must be bitwise
        identical to the strided copy_ it replaces, and must leave the N_pad
        zero-padding rows of the destination untouched."""
        from megatron.core.ssm.ops.cutedsl_mamba2_ssd._bc_repack import repack_bc_chunk_major

        torch.manual_seed(0)
        device, L, TC = "cuda", 128, 6
        real = TC * L
        # trim=True mirrors dynamic inference: B[:real] slices a padded token
        # buffer, keeping the original strides — the kernel must honor them.
        rows = real + 300 if trim else real
        B = torch.randn(rows, G, N, device=device, dtype=torch.bfloat16)[:real]
        C = torch.randn(rows, G, N, device=device, dtype=torch.bfloat16)[:real]

        B_dst = torch.zeros(1, G, N_pad, TC, L, device=device, dtype=torch.bfloat16)
        C_dst = torch.zeros(1, G, N_pad, TC, L, device=device, dtype=torch.bfloat16)
        repack_bc_chunk_major(B, C, B_dst, C_dst, N, TC, L)

        torch.testing.assert_close(B_dst, _repack_reference(B, N, N_pad, TC, L), rtol=0, atol=0)
        torch.testing.assert_close(C_dst, _repack_reference(C, N, N_pad, TC, L), rtol=0, atol=0)


@pytest.mark.skipif(
    not cutedsl_ssd_available(),
    reason="CuteDSL SSD backend requires Blackwell (SM 10.0+) and the cutlass DSL runtime",
)
class TestCuteDSLRawStates:
    """``return_raw_states=True`` must return the SSM state after EVERY caller
    chunk, elementwise-matching Triton -- including the zero-length chunks that
    empty (padded) sequences contribute to the caller's chunk numbering."""

    @pytest.mark.parametrize(
        "seq_lens,with_initial",
        [
            ([256, 512, 128], False),  # unequal, divisible
            ([256, 200], True),  # tail-ragged (partial final chunk) + prefill
            ([2048, 0, 0, 0], True),  # engine shape: the appended empty-slot rows
        ],
    )
    def test_matches_triton(self, seq_lens, with_initial):
        from megatron.core.ssm.ops.cutedsl_mamba2_ssd import (
            cutedsl_unsupported_reason,
            mamba_chunk_scan_combined_varlen_cutedsl_thd,
        )
        from megatron.core.ssm.ops.mamba2.ssd_combined import (
            _mamba_chunk_scan_combined_varlen_triton,
        )

        torch.manual_seed(42)
        device = torch.device("cuda")
        chunk_size = 128  # raw states require chunk_size == kernel L
        nheads, headdim, ngroups, dstate = 8, 64, 2, 128
        real = sum(seq_lens)

        common = _build_varlen_ssd_inputs(
            seq_lens, chunk_size, nheads, headdim, ngroups, dstate, device, torch.bfloat16
        )
        if real % chunk_size:  # ragged tail needs a padded token buffer
            pad = -real % chunk_size
            for k in ("x", "dt", "B", "C"):
                t = common[k]
                common[k] = torch.cat(
                    [t, torch.randn(pad, *t.shape[1:], device=device, dtype=t.dtype)], dim=0
                )
        initial_states = (
            torch.randn(len(seq_lens), nheads, headdim, dstate, device=device, dtype=torch.float32)
            if with_initial
            else None
        )
        call = dict(
            z=None,
            initial_states=initial_states,
            dt_softplus=True,
            dt_limit=(0.0, float("inf")),
            state_dtype=torch.float32,
            **common,
        )

        out_tri = torch.empty_like(common["x"])
        final_tri, raw_tri = _mamba_chunk_scan_combined_varlen_triton(
            out=out_tri, return_raw_states=True, **call
        )

        tiling = _ssd_tiling_for_test(list(itertools.accumulate(seq_lens, initial=0)), 128, device)
        reason = cutedsl_unsupported_reason(
            x=call["x"],
            chunk_size=call["chunk_size"],
            tiling=tiling,
            z=None,
            return_raw_states=True,
        )
        assert reason is None, f"unexpected fallback: {reason}"

        out_cute = torch.empty_like(common["x"])
        final_cute, raw_cute = mamba_chunk_scan_combined_varlen_cutedsl_thd(
            out=out_cute, return_raw_states=True, tiling=tiling, **call
        )

        # One row per caller chunk, in the caller's numbering.
        assert raw_cute.shape == raw_tri.shape, (raw_cute.shape, raw_tri.shape)
        torch.testing.assert_close(raw_cute, raw_tri, rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(final_cute, final_tri, rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(out_cute[:real], out_tri[:real], rtol=2e-2, atol=0.25)
        # Triton derives the final states from the raw ones; so must we.
        torch.testing.assert_close(
            final_cute, raw_cute[call["last_chunk_indices"]], rtol=3e-2, atol=3e-2
        )


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

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu":
            self.skipTest("Mamba Triton kernels require CUDA")

        # --- Configuration ---
        self.d_model = 256
        self.d_state = 16
        self.headdim = 64
        self.d_conv = 4
        self.ngroups = 1
        self.d_inner = self.d_model * 2  # expand=2
        self.nheads = self.d_inner // self.headdim

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


SSD_KERNEL_L = 128
"""The CuteDSL SSD kernel's tile length; the tiling arrays must be built at it."""


def _ssd_metadata_for_test(cu, chunk_size, device):
    """Stand-in for the `MambaSSDMetadata` a step publishes, as device tensors.

    Delegates the derivation to `compute_ssd_tiling` so a stand-in can never
    drift from what production computes.

    Args:
        cu: Cumulative token counts, one entry per slot plus one.
        chunk_size: The kernel's L.
        device: Where the index arrays live.

    Returns:
        A namespace carrying the `ssd_*` members and the scalars `SSDTiling` reads.
    """
    tiling = compute_ssd_tiling(cu, len(cu) - 1, chunk_size)
    members = {
        f"ssd_{name}": torch.tensor(values, dtype=torch.int32, device=device)
        for name, values in tiling.arrays.items()
    }
    return SimpleNamespace(
        **members,
        ssd_starts_aligned=tiling.starts_aligned,
        ssd_active_is_prefix=tiling.active_is_prefix,
        mamba_chunk_size=chunk_size,
        cu_seqlens=torch.empty(len(cu), dtype=torch.int32, device=device),
        real_prefill_token_count=cu[-1],
    )


def _prefill_metadata_for_test(cu_seqlens, seq_idx, chunk_size, batch_indices=None):
    """Stand-in for MambaMetadata carrying the fields ``ssm_prefill`` reads.

    Args:
        cu_seqlens: Per-sequence cumulative token counts.
        seq_idx: Per-token request id, or None.
        chunk_size: The caller's chunk size.
        batch_indices: Prefill slot map, or None.

    Returns:
        A ``SimpleNamespace`` shaped like MambaMetadata.
    """
    cu = cu_seqlens.tolist()
    chunk_boundaries = [0]
    last_chunk_indices = []
    for i in range(len(cu) - 1):
        pos = cu[i] + chunk_size
        while pos < cu[i + 1]:
            chunk_boundaries.append(pos)
            pos += chunk_size
        chunk_boundaries.append(cu[i + 1])
        last_chunk_indices.append(len(chunk_boundaries) - 2)
    cu_chunk_seqlens = cu_seqlens.new_tensor(chunk_boundaries)
    ssd = _ssd_metadata_for_test(cu, SSD_KERNEL_L, cu_seqlens.device)
    ssd.cu_seqlens = cu_seqlens
    return SimpleNamespace(
        ssd=ssd,
        mamba_chunk_size=SSD_KERNEL_L,
        real_prefill_token_count=cu[-1],
        seq_idx=seq_idx,
        cu_seqlens=cu_seqlens,
        cu_chunk_seqlens=cu_chunk_seqlens,
        last_chunk_indices=cu_seqlens.new_tensor(last_chunk_indices),
        seq_idx_for_varlen=(
            seq_idx[0, cu_chunk_seqlens[:-1]].contiguous() if seq_idx is not None else None
        ),
        batch_indices_prefill=batch_indices,
        conv_seq_idx=None,
        conv_seq_start=None,
        intermediate_chunk_indices=None,
        intermediate_abs_positions=None,
        intermediate_real_count=None,
    )


def _ssd_tiling_for_test(cu, chunk_size, device):
    """Build the op-layer SSDTiling the kernel wrapper takes.

    Args:
        cu: Cumulative token counts, one entry per slot plus one.
        chunk_size: The kernel's L.
        device: Where the index arrays live.

    Returns:
        An `SSDTiling` built from a `MambaSSDMetadata`-shaped stand-in.
    """
    from megatron.core.ssm.ops.cutedsl_mamba2_ssd import SSDTiling

    return SSDTiling(_ssd_metadata_for_test(cu, chunk_size, device))
