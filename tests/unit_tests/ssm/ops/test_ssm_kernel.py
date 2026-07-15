# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import math
import unittest
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

# Assume the provided class is in mamba_mixer.py
from megatron.core.ssm.mamba_mixer import MambaMixer


def _cutedsl_available() -> bool:
    """True if the CuteDSL SSD backend can run here (Blackwell + cutlass DSL)."""
    if not torch.cuda.is_available():
        return False
    try:
        from megatron.core.ssm.ops.cutedsl_mamba2_ssd import is_cutedsl_ssd_available

        return torch.cuda.get_device_capability()[0] >= 10 and is_cutedsl_ssd_available()
    except Exception:
        return False


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
    starts = [0]
    for s in seq_lens:
        starts.append(starts[-1] + s // chunk_size)
    emit = []
    for i, s in enumerate(seq_lens):
        n_chunks = s // chunk_size
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


# Every supported dispatch shape and inference feature in one table. Grouped by
# the scenario each case exercises; ids mirror the grouping.
_PARITY_CASES = [
    # Plain divisible batches (varlen tile scheduler), nheads=16 and dstate=16
    # variants (dstate 16 exercises N padding + the workspace-cache-collision
    # regression where dstate 16 and 128 both pad to N_pad=128).
    pytest.param(_ssd_case([256] * 4, chunk_size=256, nheads=16), id="basic-equal"),
    pytest.param(_ssd_case([256, 512, 256], chunk_size=256, nheads=16), id="basic-unequal"),
    pytest.param(_ssd_case([512, 256, 768, 256], chunk_size=256, nheads=16), id="basic-unequal4"),
    pytest.param(_ssd_case([256] * 4, chunk_size=256, nheads=16, dstate=16), id="basic-d16"),
    pytest.param(
        _ssd_case([256, 512, 256], chunk_size=256, nheads=16, dstate=16), id="basic-unequal-d16"
    ),
    pytest.param(
        _ssd_case([512, 256, 768, 256], chunk_size=256, nheads=16, dstate=16),
        id="basic-unequal4-d16",
    ),
    # Trailing token-buffer padding: dynamic inference hands fixed-size
    # CUDA-graph token buffers with x.shape[0] > sum(seq_lens); the wrapper
    # must trim to the real tokens (cu_chunk_seqlens[-1]). A ragged
    # (non-divisible) batch must be rejected by cutedsl_unsupported_reason.
    pytest.param(_ssd_case([256, 256], chunk_size=256, pad_to=768), id="padded-equal"),
    pytest.param(_ssd_case([256, 512], chunk_size=256, pad_to=1024), id="padded-unequal"),
    pytest.param(
        _ssd_case([200, 300], chunk_size=256, pad_to=768, expect_fallback=True),
        id="padded-ragged-fallback",
    ),
    # Chunked prefill: non-zero carried SSM state (CuteDSL seeds the state on
    # the divisible path).
    pytest.param(_ssd_case([256, 512, 256], chunk_size=256, with_initial=True), id="prefill"),
    pytest.param(_ssd_case([128, 256, 384], chunk_size=256, with_initial=True), id="prefill-mixed"),
    # Prefix caching: intermediate states emitted at flagged chunk boundaries
    # must match Triton's states[indices] — alone and combined with chunked
    # prefill.
    pytest.param(_ssd_case([384, 256, 512], inter="per_seq"), id="prefix"),
    pytest.param(
        _ssd_case([384, 256, 512], inter="per_seq", with_initial=True), id="prefix-prefill"
    ),
    # Duplicate / padded intermediate indices must follow gather semantics: the
    # dynamic engine pads intermediate_chunk_indices to a fixed size with chunk
    # 0, and the CuteDSL emit-slot scatter collides on duplicates — the wrapper
    # must resolve them so EVERY slot holds its chunk's state.
    pytest.param(_ssd_case([384, 256, 512], inter=[0] * 12), id="inter-dup-all-padding"),
    pytest.param(
        _ssd_case([384, 256, 512], inter=[0, 4, 7] + [0] * 9), id="inter-dup-real-plus-padding"
    ),
    pytest.param(_ssd_case([384, 256, 512], inter=[5, 5, 2, 2, 2, 0]), id="inter-dup-arbitrary"),
    # Empty (0-length) padded sequences — the dynamic engine pads batches to a
    # fixed slot count — must NOT deadlock the varlen tile scheduler (a 0-chunk
    # work-item hangs the pipeline); the wrapper compacts empties out of the
    # launch and scatters per-seq final states back. Interleaved empties +
    # intermediate emission is unsupported (chunk-numbering mismatch) and must
    # fall back to Triton; trailing empties stay on CuteDSL.
    pytest.param(_ssd_case([2048, 0, 0, 0], with_initial=True), id="empty-trailing"),
    pytest.param(_ssd_case([256, 512, 0, 0], with_initial=True), id="empty-trailing2"),
    pytest.param(_ssd_case([1024] + [0] * 7, with_initial=True), id="empty-many"),
    pytest.param(_ssd_case([0, 256, 0], with_initial=True), id="empty-interleaved"),
    pytest.param(
        _ssd_case([2048, 0, 0, 0], with_initial=True, inter="per_seq"), id="empty-trailing-inter"
    ),
    pytest.param(
        _ssd_case([256, 512, 0, 0], with_initial=True, inter="per_seq"), id="empty-trailing2-inter"
    ),
    pytest.param(
        _ssd_case([1024] + [0] * 7, with_initial=True, inter="per_seq"), id="empty-many-inter"
    ),
    pytest.param(
        _ssd_case([0, 256, 0], with_initial=True, inter="per_seq", expect_fallback=True),
        id="empty-interleaved-inter-fallback",
    ),
]


@pytest.mark.skipif(
    not _cutedsl_available(),
    reason="CuteDSL SSD backend requires Blackwell (SM 10.0+) and the cutlass DSL runtime",
)
class TestCuteDSLMatchesTriton:
    """The CuteDSL SSD kernel must produce near-identical results to Triton across
    every supported varlen prefill shape (divisible sequence lengths), inference
    feature (chunked prefill, prefix caching, padded/empty engine batches), and
    must raise NotImplementedError on unsupported shapes so the dispatcher falls
    back to Triton. All scenarios share one skeleton, driven by _PARITY_CASES."""

    @pytest.mark.parametrize("case", _PARITY_CASES)
    def test_cutedsl_matches_triton(self, case):
        from megatron.core.ssm.ops.cutedsl_mamba2_ssd import (
            cutedsl_unsupported_reason,
            mamba_chunk_scan_combined_varlen_cutedsl_thd,
        )
        from megatron.core.ssm.ops.ssd_combined import _mamba_chunk_scan_combined_varlen_triton

        torch.manual_seed(0)
        device = torch.device("cuda")
        seq_lens, chunk_size = case["seq_lens"], case["chunk_size"]
        headdim, ngroups = 64, 2
        real = sum(seq_lens)

        common = _build_varlen_ssd_inputs(
            seq_lens,
            chunk_size,
            case["nheads"],
            headdim,
            ngroups,
            case["dstate"],
            device,
            torch.bfloat16,
        )
        if case["pad_to"] is not None:
            # Extend x/dt/B/C with trailing padding rows (cu_chunk_seqlens still
            # describes only the `real` tokens, exactly as the engine feeds them).
            assert case["pad_to"] > real, "test must actually pad"
            pad = case["pad_to"] - real
            for k in ("x", "dt", "B", "C"):
                t = common[k]
                common[k] = torch.cat(
                    [t, torch.randn(pad, *t.shape[1:], device=device, dtype=t.dtype)], dim=0
                )
        if case["inter"] == "per_seq":
            idx = _emit_per_seq_indices(seq_lens, chunk_size, device)
        elif case["inter"] is not None:
            idx = torch.tensor(case["inter"], dtype=torch.int64, device=device)
        else:
            idx = None
        initial_states = (
            torch.randn(
                len(seq_lens),
                case["nheads"],
                headdim,
                case["dstate"],
                device=device,
                dtype=torch.float32,
            )
            if case["with_initial"]
            else None
        )
        call = dict(
            z=None,
            initial_states=initial_states,
            dt_softplus=True,
            dt_limit=(0.0, float("inf")),
            state_dtype=torch.float32,
            intermediate_chunk_indices=idx,
            **common,
        )

        out_tri = torch.empty_like(common["x"])
        tri = _mamba_chunk_scan_combined_varlen_triton(out=out_tri, **call)
        out_cute = torch.empty_like(common["x"])

        # The wrapper assumes eligibility (it no longer raises); callers must
        # consult cutedsl_unsupported_reason first, exactly like the dispatcher.
        # NOTE: kernel_chunk_size is the kernel's fixed L (default 128), NOT the
        # caller's chunk_size — do not override it here.
        reason = cutedsl_unsupported_reason(
            x=call["x"],
            chunk_size=call["chunk_size"],
            cu_chunk_seqlens=call["cu_chunk_seqlens"],
            last_chunk_indices=call["last_chunk_indices"],
            z=call["z"],
            return_intermediate_states=False,
            intermediate_chunk_indices=call["intermediate_chunk_indices"],
        )
        if case["expect_fallback"]:
            assert reason is not None, "eligibility check should reject this case"
            return  # the dispatcher runs Triton for this batch; nothing to compare
        assert reason is None, f"unexpected fallback for a supported case: {reason}"

        cute = mamba_chunk_scan_combined_varlen_cutedsl_thd(out=out_cute, **call)

        if idx is not None:
            final_tri, inter_tri = tri
            final_cute, inter_cute = cute
            # Every emitted slot must match, including duplicated/padding slots.
            torch.testing.assert_close(inter_cute, inter_tri, rtol=3e-2, atol=3e-2)
        else:
            final_tri, final_cute = tri, cute
        # bf16 MMAs + padded dstate are not bitwise equal; compare every REAL
        # token elementwise (padding outputs are undefined for both backends).
        # atol is one bf16 ulp at the typical accumulated-term magnitude (~64):
        # where y is a small difference of large terms, no two independently
        # tiled bf16 kernels agree below ulp(|terms|), even vs an fp64 reference.
        torch.testing.assert_close(out_cute[:real], out_tri[:real], rtol=2e-2, atol=0.25)
        torch.testing.assert_close(final_cute, final_tri, rtol=3e-2, atol=3e-2)


def _fused_cumsum_available() -> bool:
    """True if the standalone fused softplus+cumsum Triton kernel can run here."""
    if not torch.cuda.is_available():
        return False
    try:
        import megatron.core.ssm.ops.cutedsl_mamba2_ssd._fused_cumsum  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _fused_cumsum_available(), reason="fused cumsum kernel requires CUDA + Triton"
)
class TestFusedSoftplusCumsum:
    """The standalone Triton kernel that fuses dt preprocessing (softplus + bias +
    clamp + cumsum) for the CuteDSL hot paths must match the torch reference in both
    the divisible (B=1, C=total_chunks) and aligned (B=S, C=Cmax) output layouts."""

    def _reference(self, dt, A, dt_bias, dt_softplus, dt_limit):
        dt_f = dt.float()
        if dt_bias is not None:
            dt_f = dt_f + dt_bias.float()
        if dt_softplus:
            dt_f = torch.nn.functional.softplus(dt_f)
        lo, hi = dt_limit
        if lo != 0.0 or hi != float("inf"):
            dt_f = torch.clamp(dt_f, lo, hi)
        return dt_f  # (T, H)

    @pytest.mark.parametrize("use_bias", [True, False])
    @pytest.mark.parametrize("layout", ["divisible", "aligned"])
    def test_matches_torch(self, layout, use_bias):
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

        dt_f = self._reference(dt, A, dt_bias, True, (0.0, float("inf")))
        ref_delta = dt_f.view(B, C, L, H).permute(0, 3, 1, 2)  # (B, H, C, L)
        ref_cumsum = torch.cumsum(ref_delta.float() * A.view(1, H, 1, 1), dim=-1)

        torch.testing.assert_close(delta.float(), ref_delta, rtol=5e-3, atol=5e-3)
        torch.testing.assert_close(cumsum, ref_cumsum, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
    not _fused_cumsum_available(), reason="B/C repack kernel requires CUDA + Triton"
)
class TestBCRepack:
    """The tiled B/C repack (token-major -> dense chunk-major) must be bitwise
    identical to the strided copy_ it replaces, and must leave the N_pad
    zero-padding rows of the destination untouched."""

    @staticmethod
    def _ref(src, N, N_pad, TC, L):
        G = src.shape[1]
        GN = G * N
        dst = torch.zeros(1, G, N_pad, TC, L, device=src.device, dtype=src.dtype)
        dst[:, :, :N].copy_(src.as_strided((1, G, N, TC, L), (0, N, 1, L * GN, GN)))
        return dst

    @pytest.mark.parametrize(
        "G,N,N_pad",
        [
            (8, 128, 128),  # production dims (mamba-num-groups 8, dstate 128)
            (2, 16, 128),  # small dstate: padding rows n >= 16 must stay zero
            (1, 128, 128),  # single group
        ],
    )
    @pytest.mark.parametrize("trim", [False, True])
    def test_matches_strided_copy(self, G, N, N_pad, trim):
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

        torch.testing.assert_close(B_dst, self._ref(B, N, N_pad, TC, L), rtol=0, atol=0)
        torch.testing.assert_close(C_dst, self._ref(C, N, N_pad, TC, L), rtol=0, atol=0)


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
        if self.device.type == 'cpu':
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
        self.mixer._ssm_prefill = MambaMixer._ssm_prefill.__get__(self.mixer, MambaMixer)
        self.mixer._ssm_decode = MambaMixer._ssm_decode.__get__(self.mixer, MambaMixer)

    def test_ssm_prefill_padding_isolation(self):
        """
        Tests that ssm_prefill only updates states for the real request
        and that padding request states remain untouched.

        _ssm_prefill expects inputs pre-stripped to real tokens only
        (stripping is done by _dynamic_inference_prefill). This test
        passes only the real tokens and verifies that only the active
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

        # Run
        self.mixer.norm = MagicMock(side_effect=lambda x, z: x * z)
        output = self.mixer._ssm_prefill(
            zxBCdt=zxBCdt,
            conv_state=conv_state,
            ssm_state=ssm_state,
            seq_idx=seq_idx,
            cu_seqlens=cu_seqlens,
            batch_indices=batch_indices,
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


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
