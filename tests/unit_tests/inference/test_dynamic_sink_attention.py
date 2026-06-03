# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit tests for the sink (off-by-one / learnable) softmax post-correction
used by the dynamic-batching inference path in :class:`Attention`.

The dynamic-batching inference path bypasses ``self.core_attention`` and calls
flash-attention kernels directly. To support ``config.softmax_type`` of
``"off-by-one"`` or ``"learnable"`` we apply the sink correction as a post-hoc
rescale of the flash-attention output using its log-sum-exp tensor:

    out_sink = out_vanilla * sigmoid(lse - softmax_offset)

These tests validate that the rescale matches the canonical sink-softmax
definition used by the static path (``SoftmaxOne``) — i.e.

    softmax_with_sink(s)_i = exp(s_i) / (exp(sink) + sum_j exp(s_j))
"""
import pytest
import torch

from megatron.core.transformer.attention import Attention


def _vanilla_attention_with_lse(q, k, v, softmax_scale):
    """Compute vanilla causal attention and return (out, lse) per token, per head.

    Args:
        q (Tensor): ``(B, S_q, H, D)``.
        k (Tensor): ``(B, S_k, H, D)``.
        v (Tensor): ``(B, S_k, H, D)``.

    Returns:
        out (Tensor): ``(B, S_q, H, D)`` attention output (vanilla softmax).
        lse (Tensor): ``(B, H, S_q)`` log-sum-exp matching the flash-attn layout.
    """
    # (B, H, S_q, D) @ (B, H, D, S_k) -> (B, H, S_q, S_k)
    qh = q.transpose(1, 2).to(torch.float32)
    kh = k.transpose(1, 2).to(torch.float32)
    vh = v.transpose(1, 2).to(torch.float32)
    scores = torch.matmul(qh, kh.transpose(-1, -2)) * softmax_scale

    # Apply causal mask aligned to the bottom-right corner (matches flash-attn
    # decode-style attention where S_q <= S_k and queries see only the most
    # recent S_q keys plus all preceding ones).
    s_q = qh.size(-2)
    s_k = kh.size(-2)
    causal = torch.tril(
        torch.ones(s_q, s_k, device=q.device, dtype=torch.bool), diagonal=s_k - s_q
    )
    scores = scores.masked_fill(~causal, float("-inf"))

    lse = torch.logsumexp(scores, dim=-1)  # (B, H, S_q)
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, vh)  # (B, H, S_q, D)
    return out.transpose(1, 2), lse  # (B, S_q, H, D), (B, H, S_q)


def _sink_attention_reference(q, k, v, softmax_scale, softmax_offset):
    """Reference sink-attention output computed via the canonical SoftmaxOne path."""
    qh = q.transpose(1, 2).to(torch.float32)
    kh = k.transpose(1, 2).to(torch.float32)
    vh = v.transpose(1, 2).to(torch.float32)
    scores = torch.matmul(qh, kh.transpose(-1, -2)) * softmax_scale

    s_q = qh.size(-2)
    s_k = kh.size(-2)
    causal = torch.tril(
        torch.ones(s_q, s_k, device=q.device, dtype=torch.bool), diagonal=s_k - s_q
    )
    scores = scores.masked_fill(~causal, float("-inf"))

    # Append per-head sink logit, softmax, drop the extra slot — mirrors
    # SoftmaxOne in megatron/core/fusions/fused_softmax.py.
    sink = softmax_offset.reshape(1, -1, 1, 1).expand(
        scores.size(0), -1, scores.size(2), 1
    ).to(scores)
    qk = torch.cat([scores, sink], dim=-1)
    probs = torch.softmax(qk, dim=-1)[..., :-1]
    out = torch.matmul(probs, vh)
    return out.transpose(1, 2)


class TestSinkSoftmaxCorrection:
    """Math-only tests; no flash-attn dependency."""

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("offset_kind", ["off-by-one", "learnable"])
    def test_bshd_correction_matches_sink_softmax(self, dtype, offset_kind):
        """``_apply_sink_softmax_correction_bshd`` must match SoftmaxOne semantics."""
        b, s_q, s_k, h, d = 2, 4, 8, 3, 16
        softmax_scale = d ** -0.5

        q = torch.randn(b, s_q, h, d, device=self.device, dtype=dtype)
        k = torch.randn(b, s_k, h, d, device=self.device, dtype=dtype)
        v = torch.randn(b, s_k, h, d, device=self.device, dtype=dtype)

        if offset_kind == "off-by-one":
            softmax_offset = torch.zeros(h, device=self.device, dtype=dtype)
        else:
            softmax_offset = torch.randn(h, device=self.device, dtype=dtype) * 0.5

        # Vanilla flash-attn-like output + LSE.
        out_vanilla, lse = _vanilla_attention_with_lse(q, k, v, softmax_scale)
        out_vanilla = out_vanilla.to(dtype)

        # Apply correction.
        out_corrected = Attention._apply_sink_softmax_correction_bshd(
            out_vanilla, lse, softmax_offset
        )

        # Reference: full recompute with SoftmaxOne semantics.
        out_ref = _sink_attention_reference(q, k, v, softmax_scale, softmax_offset).to(
            dtype
        )

        rtol = 1e-2 if dtype == torch.bfloat16 else 1e-5
        atol = 1e-2 if dtype == torch.bfloat16 else 1e-5
        assert torch.allclose(out_corrected, out_ref, rtol=rtol, atol=atol), (
            f"Sink-corrected output diverges from reference "
            f"(max abs diff = {(out_corrected.float() - out_ref.float()).abs().max():.3e})"
        )

    @pytest.mark.parametrize("offset_kind", ["off-by-one", "learnable"])
    def test_varlen_correction_matches_sink_softmax(self, offset_kind):
        """``_apply_sink_softmax_correction_varlen`` must match SoftmaxOne semantics.

        Constructs a single packed sequence (B=1) so the varlen and bshd layouts
        give identical numerical results — we can reuse the (B,S,H,D) reference.
        """
        s_q, s_k, h, d = 6, 6, 4, 8  # square so causal mask is trivial diag
        softmax_scale = d ** -0.5
        dtype = torch.float32

        q = torch.randn(1, s_q, h, d, device=self.device, dtype=dtype)
        k = torch.randn(1, s_k, h, d, device=self.device, dtype=dtype)
        v = torch.randn(1, s_k, h, d, device=self.device, dtype=dtype)

        if offset_kind == "off-by-one":
            softmax_offset = torch.zeros(h, device=self.device, dtype=dtype)
        else:
            softmax_offset = torch.randn(h, device=self.device, dtype=dtype) * 0.5

        out_vanilla_bshd, lse_bshd = _vanilla_attention_with_lse(q, k, v, softmax_scale)
        # Reshape to varlen layout: (total_q, H, D) and (H, total_q)
        out_vanilla_varlen = out_vanilla_bshd.reshape(-1, h, d)
        lse_varlen = lse_bshd.reshape(h, -1)

        out_corrected_varlen = Attention._apply_sink_softmax_correction_varlen(
            out_vanilla_varlen, lse_varlen, softmax_offset
        )
        out_corrected = out_corrected_varlen.reshape(1, s_q, h, d)

        out_ref = _sink_attention_reference(q, k, v, softmax_scale, softmax_offset)

        assert torch.allclose(out_corrected, out_ref, rtol=1e-5, atol=1e-5), (
            "Varlen sink-corrected output diverges from reference."
        )

    def test_off_by_one_with_zero_logit_equals_plus_one_denominator(self):
        """With ``softmax_offset == 0``, the sink contributes ``exp(0) == 1`` to
        the denominator — the canonical Miller off-by-one softmax."""
        b, s, h, d = 1, 3, 2, 4
        dtype = torch.float32

        # Construct trivial attention with zero scores -> uniform probs over s
        # vanilla, and uniform over s+1 (with sink) under sink.
        out_vanilla = torch.full((b, s, h, d), 1.0, device=self.device, dtype=dtype)
        # logsumexp of s zeros == log(s)
        lse = torch.full((b, h, s), float(torch.tensor(float(s)).log()), device=self.device)
        softmax_offset = torch.zeros(h, device=self.device, dtype=dtype)

        out_corrected = Attention._apply_sink_softmax_correction_bshd(
            out_vanilla, lse, softmax_offset
        )

        # Scale factor: sigmoid(log(s) - 0) = s / (s + 1).
        expected_scale = s / (s + 1.0)
        torch.testing.assert_close(
            out_corrected,
            out_vanilla * expected_scale,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_nan_lse_rows_unmodified(self):
        """Rows with NaN LSE (e.g. kernel artifacts on padded queries) must be
        left alone so NaNs do not propagate through the inference pipeline.

        Note: ``-inf`` LSE is a legitimate "no attended keys" signal that maps
        to ``sigmoid(-inf - sink) == 0`` — this correctly zeroes the output
        for that row, which matches the static path's behavior.
        """
        b, s, h, d = 1, 3, 1, 2
        dtype = torch.float32

        out_vanilla = torch.tensor(
            [[[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]],
            device=self.device,
            dtype=dtype,
        )
        # Row 0: finite lse=0     -> sigmoid(0)  = 0.5    -> scale by 0.5
        # Row 1: lse=-inf         -> sigmoid(-inf) = 0    -> zero the row
        # Row 2: lse=NaN          -> NaN          (guard) -> keep row unchanged
        lse = torch.tensor(
            [[[0.0, float("-inf"), float("nan")]]], device=self.device
        )
        softmax_offset = torch.zeros(h, device=self.device, dtype=dtype)

        out_corrected = Attention._apply_sink_softmax_correction_bshd(
            out_vanilla, lse, softmax_offset
        )

        torch.testing.assert_close(
            out_corrected[0, 0, 0],
            torch.tensor([0.5, 1.0], device=self.device),
            rtol=1e-6,
            atol=1e-6,
        )
        torch.testing.assert_close(
            out_corrected[0, 1, 0],
            torch.tensor([0.0, 0.0], device=self.device),
            rtol=1e-6,
            atol=1e-6,
        )
        # NaN-LSE row preserved (guarded by torch.where(isfinite, ..., 1)).
        torch.testing.assert_close(
            out_corrected[0, 2, 0],
            torch.tensor([5.0, 6.0], device=self.device),
            rtol=1e-6,
            atol=1e-6,
        )
