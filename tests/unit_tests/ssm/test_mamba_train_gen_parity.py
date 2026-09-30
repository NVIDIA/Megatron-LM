# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Bitwise parity between the Mamba training forward and the generation kernels.

The scan comparison spans two implementations, so both have to pick the same Triton
tiling: export MAMBA_DETERMINISTIC=1 before pytest, or that test skips.
"""

import os

import pytest
import torch

try:
    from mamba_ssm.ops.triton.ssd_combined import (
        mamba_chunk_scan_combined,
        mamba_split_conv1d_scan_combined,
    )

    from megatron.core.ssm.ops.mamba2.ssd_combined import mamba_chunk_scan_combined_varlen

    HAVE_MAMBA = True
except ImportError:
    HAVE_MAMBA = False

try:
    from causal_conv1d import causal_conv1d_fn

    HAVE_CAUSAL_CONV1D = causal_conv1d_fn is not None
except ImportError:
    HAVE_CAUSAL_CONV1D = False


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(not HAVE_MAMBA, reason="mamba_ssm required"),
]

# Reading the environment is the whole check: the autotune config lists are fixed at
# import, long before any test code runs, so only the launcher can pin them.
requires_pinned_autotune = pytest.mark.skipif(
    not os.environ.get("MAMBA_DETERMINISTIC", "").startswith("1"),
    reason="needs MAMBA_DETERMINISTIC=1 so both implementations pick the same tiling",
)

NHEADS = 8
HEADDIM = 32
NGROUPS = 1
DSTATE = 16
CHUNK_SIZE = 32
SEQLEN = 96  # three chunks, so the state-passing carry is exercised
D_CONV = 4


def _report(label, a, b):
    """Log a pair's agreement and return whether it is bitwise identical."""
    same = torch.equal(a, b)
    delta = (a.float() - b.float()).pow(2).mean().sqrt()
    scale = b.float().pow(2).mean().sqrt().clamp_min(1e-12)
    print(f"[mamba-parity] {label}: bitwise={same} rel_rms={(delta / scale).item():.3e}")
    return same


@pytest.fixture(name="inputs")
def _inputs():
    """One set of SSM inputs, shared by every path under test."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(0)

    def randn(*shape, dt=dtype):
        return torch.randn(*shape, device=device, dtype=dt, generator=generator)

    return dict(
        device=device,
        dtype=dtype,
        x=randn(1, SEQLEN, NHEADS, HEADDIM),
        z=randn(1, SEQLEN, NHEADS, HEADDIM),
        dt=randn(1, SEQLEN, NHEADS, dt=torch.float32),
        B=randn(1, SEQLEN, NGROUPS, DSTATE),
        C=randn(1, SEQLEN, NGROUPS, DSTATE),
        A=-torch.rand(NHEADS, device=device, dtype=torch.float32, generator=generator).exp(),
        D=torch.rand(NHEADS, device=device, dtype=torch.float32, generator=generator),
        dt_bias=torch.rand(NHEADS, device=device, dtype=torch.float32, generator=generator),
    )


class TestScanParity:
    """Training's chunk scan against the varlen scan dynamic prefill runs."""

    @requires_pinned_autotune
    def test_training_scan_matches_varlen_prefill(self, inputs):
        """Bitwise, since the prefill state seeds every token decoded after it."""
        x, z, dt, B, C = (inputs[k] for k in ("x", "z", "dt", "B", "C"))
        A, D, dt_bias = inputs["A"], inputs["D"], inputs["dt_bias"]

        # Batch-invariant convention: z through the scan.
        y_train, _ = mamba_chunk_scan_combined(
            x,
            dt,
            A,
            B,
            C,
            CHUNK_SIZE,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=True,
            return_final_states=True,
        )

        # One request spanning the sequence, chunked the same way, so the kernel is the
        # only difference. The varlen entry point writes into a preallocated output.
        device = inputs["device"]
        num_chunks = SEQLEN // CHUNK_SIZE
        cu_chunk_seqlens = torch.arange(0, SEQLEN + 1, CHUNK_SIZE, dtype=torch.int32, device=device)
        last_chunk_indices = torch.tensor([num_chunks - 1], dtype=torch.int64, device=device)
        seq_idx = torch.zeros(num_chunks, dtype=torch.int32, device=device)
        y_prefill = torch.empty_like(x.squeeze(0))
        mamba_chunk_scan_combined_varlen(
            x=x.squeeze(0),
            dt=dt.squeeze(0),
            A=A,
            B=B.squeeze(0),
            C=C.squeeze(0),
            chunk_size=CHUNK_SIZE,
            cu_chunk_seqlens=cu_chunk_seqlens,
            last_chunk_indices=last_chunk_indices,
            seq_idx=seq_idx,
            out=y_prefill,
            D=D,
            z=z.squeeze(0),
            dt_bias=dt_bias,
            dt_softplus=True,
        )

        same = _report(
            "training chunk scan vs varlen prefill scan",
            y_train.squeeze(0).reshape(-1),
            y_prefill.reshape(-1),
        )
        assert same, (
            "dynamic prefill does not reproduce the training scan bitwise, so a zero-KL "
            "run on this model is not possible as configured"
        )


@pytest.mark.skipif(not HAVE_CAUSAL_CONV1D, reason="causal-conv1d required")
class TestConvParity:
    """The depthwise conv, whose existing test only requires atol=1e-2 for bf16."""

    def test_training_conv_matches_varlen_prefill_conv(self, inputs):
        """Bitwise, not close."""
        from megatron.core.ssm.ops.common.causal_conv1d_varlen import causal_conv1d_varlen_fn

        device, dtype = inputs["device"], inputs["dtype"]
        conv_dim = NHEADS * HEADDIM
        generator = torch.Generator(device=device).manual_seed(1)
        xBC = torch.randn(SEQLEN, conv_dim, device=device, dtype=dtype, generator=generator)
        weight = torch.randn(conv_dim, D_CONV, device=device, dtype=dtype, generator=generator)
        bias = torch.randn(conv_dim, device=device, dtype=dtype, generator=generator)
        cu_seqlens = torch.tensor([0, SEQLEN], dtype=torch.int32, device=device)

        y_train = causal_conv1d_fn(
            x=xBC.t().unsqueeze(0).contiguous(), weight=weight, bias=bias, activation="silu"
        )
        y_prefill = causal_conv1d_varlen_fn(
            x=xBC, weight=weight, bias=bias, cu_seqlens=cu_seqlens, activation="silu"
        )

        same = _report(
            "training conv vs varlen prefill conv",
            y_train.squeeze(0).t().reshape(-1),
            y_prefill.reshape(-1),
        )
        assert same, (
            "the varlen conv fork is not bitwise-equal to the pip conv the training "
            "forward runs; its own test only requires atol=1e-2"
        )


@pytest.mark.skipif(not HAVE_CAUSAL_CONV1D, reason="causal-conv1d required")
class TestFusedTrainingParity:
    """The memory-efficient training kernel against generation's varlen conv + varlen scan.

    ``MambaMixer._ssm_training`` calls ``mamba_split_conv1d_scan_combined`` with no
    ``rmsnorm_weight``: the gate goes through the scan and the norm afterwards is ungated. That
    is generation's arrangement in batch-invariant mode, so the fused kernel should equal the
    conv followed by the scan, which is what ``ssm_prefill`` runs.
    """

    @requires_pinned_autotune
    def test_fused_training_kernel_matches_varlen_prefill(self, inputs):
        from megatron.core.ssm.ops.common.causal_conv1d_varlen import causal_conv1d_varlen_fn

        device, dtype = inputs["device"], inputs["dtype"]
        x, z, B, C = (inputs[k] for k in ("x", "z", "B", "C"))
        A, D, dt_bias = inputs["A"], inputs["D"], inputs["dt_bias"]
        dt = inputs["dt"].to(dtype)  # the mixer's dt is a bf16 slice of the projection
        d_inner = NHEADS * HEADDIM
        conv_dim = d_inner + 2 * NGROUPS * DSTATE
        generator = torch.Generator(device=device).manual_seed(1)
        conv_weight = torch.randn(conv_dim, D_CONV, device=device, dtype=dtype, generator=generator)
        conv_bias = torch.randn(conv_dim, device=device, dtype=dtype, generator=generator)

        xBC = torch.cat([x.flatten(2), B.flatten(2), C.flatten(2)], dim=-1)  # [1, L, conv_dim]
        zxbcdt = torch.cat([z.flatten(2), xBC, dt], dim=-1)

        # Training: one fused kernel, exactly as _ssm_training calls it.
        y_train = mamba_split_conv1d_scan_combined(
            zxbcdt,
            conv_weight,
            conv_bias,
            dt_bias,
            A,
            D=D,
            chunk_size=CHUNK_SIZE,
            activation="silu",
            headdim=HEADDIM,
            ngroups=NGROUPS,
            norm_before_gate=True,
        )

        # Generation: varlen conv, then varlen scan with the gate inside, as ssm_prefill does.
        cu_seqlens = torch.tensor([0, SEQLEN], dtype=torch.int32, device=device)
        conv_out = causal_conv1d_varlen_fn(
            x=xBC.squeeze(0).contiguous(),
            weight=conv_weight,
            bias=conv_bias,
            cu_seqlens=cu_seqlens,
            activation="silu",
        )
        x_c, B_c, C_c = torch.split(conv_out, [d_inner, NGROUPS * DSTATE, NGROUPS * DSTATE], dim=-1)
        num_chunks = SEQLEN // CHUNK_SIZE
        y_prefill = torch.zeros(SEQLEN, NHEADS, HEADDIM, device=device, dtype=dtype)
        mamba_chunk_scan_combined_varlen(
            x=x_c.reshape(SEQLEN, NHEADS, HEADDIM).contiguous(),
            dt=dt.squeeze(0),
            A=A,
            B=B_c.reshape(SEQLEN, NGROUPS, DSTATE).contiguous(),
            C=C_c.reshape(SEQLEN, NGROUPS, DSTATE).contiguous(),
            chunk_size=CHUNK_SIZE,
            cu_chunk_seqlens=torch.arange(
                0, SEQLEN + 1, CHUNK_SIZE, dtype=torch.int32, device=device
            ),
            last_chunk_indices=torch.tensor([num_chunks - 1], dtype=torch.int64, device=device),
            seq_idx=torch.zeros(num_chunks, dtype=torch.int32, device=device),
            out=y_prefill,
            D=D,
            z=z.squeeze(0),
            dt_bias=dt_bias,
            dt_softplus=True,
        )

        a, b = y_train.squeeze(0).reshape(SEQLEN, -1), y_prefill.reshape(SEQLEN, -1)
        same = _report("fused training kernel vs varlen conv + varlen scan", a, b)
        if not same:
            bad = (a != b).nonzero()
            print(
                f"[mamba-parity] {bad.shape[0]} elements differ; "
                f"first (token, channel): {bad[:6].tolist()}"
            )
        assert same, "the fused training kernel does not match generation's varlen conv + scan"
