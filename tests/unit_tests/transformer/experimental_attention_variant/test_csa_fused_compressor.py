# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for the cudnn-frontend-backed fused CSA Compressor dispatch.

The kernels themselves are validated in cudnn-frontend (PR #427: numerics vs fp32/fp64
references, backward zero-write coverage, CUDA-graph capture, determinism). These tests
cover the Megatron-side wiring only:

  - numerics of the dispatched fused region vs the eager region it replaces, with the
    original PR #5984 gates: ``dKV``/``dScore`` bitwise vs an fp32-intermediate eager
    reference, forward within one bf16 rounding step, tolerance vs the verbatim
    upstream eager numerics;
  - static-capacity padding rows (``fixed_total_comp``) through the dispatch;
  - dispatch gating / eager fallback of ``maybe_compress_thd_fused``: the caller's
    ``enabled`` switch, deterministic mode, unsupported
    configurations, and a missing/old cudnn-frontend (no ``cudnn.csa``);
  - ``Compressor._forward_thd`` integration: the fused dispatch engages and matches
    eager, gradients flow, and the module falls back to the bitwise-identical eager
    path when the frontend is unavailable.

Without a cudnn-frontend that provides ``cudnn.csa`` (or without CUDA / below
compute-capability major 10) every kernel test skips.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import csa as csa_module
from megatron.core.transformer.experimental_attention_variant.csa import (
    Compressor,
    CompressorSubmodules,
    batch_of_row,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils import (
    fused_compressor as cfc,
)

# Run this module on GB200 hardware in CI (marker-driven selection, see
# tests/unit_tests/find_test_cases.py); everywhere else the tests skip via
# _require_fused().
pytestmark = pytest.mark.launch_on_gb200


def _require_fused():
    if not torch.cuda.is_available():
        pytest.skip("fused CSA compressor tests require CUDA")
    if cfc._get_frontend() is None:
        pytest.skip(
            "cudnn-frontend with the CSA compressor API (cudnn.csa, cudnn-frontend #427) "
            f"is not available: {cfc._frontend_error!r}"
        )
    if not cfc.fused_compressor_available():
        pytest.skip("fused CSA compressor requires compute-capability major >= 10 (SM100+)")


# ---------------------------------------------------------------------------
# Eager reference: verbatim replica of the region of ``Compressor._forward_thd``
# (non-pre-grouped THD path) that the fused dispatch replaces, from the projection
# outputs (kv, score) to the pre-RMSNorm pooled output. ``mode`` selects the
# numerics: "upstream" reproduces the current eager code exactly (softmax weights
# rounded to bf16, bf16 multiply); "fp32" keeps all intermediates fp32 with a
# single final bf16 rounding (the fused kernels' numerics). The overlap-window
# transform is the real upstream implementation.
# ---------------------------------------------------------------------------


def _eager_pool(kv, score, ape, cu_seqlens, cu_seqlens_comp, total_comp, ratio, d, coff, mode):
    device = kv.device
    row_idx = torch.arange(total_comp, device=device, dtype=cu_seqlens_comp.dtype)
    batch_ids = batch_of_row(cu_seqlens_comp, total_q=total_comp)
    valid_comp = row_idx < cu_seqlens_comp[-1]
    local_pos = row_idx - cu_seqlens_comp[batch_ids]
    local_pos = torch.where(valid_comp, local_pos, torch.zeros_like(local_pos))
    base = cu_seqlens[batch_ids].unsqueeze(1) + local_pos.unsqueeze(1) * ratio
    base = torch.where(valid_comp.unsqueeze(1), base, torch.zeros_like(base))
    offsets = torch.arange(ratio, device=device, dtype=base.dtype).unsqueeze(0)
    gather_idx = base + offsets  # (total_comp, ratio)

    if mode == "fp32":
        kv = kv.float()
        score = score.float()

    kv_grouped = kv[gather_idx]  # (total_comp, ratio, 1, coff * d)
    score_grouped = score[gather_idx]
    score_grouped = score_grouped + ape.view(1, ratio, 1, -1)

    if coff == 2:
        is_first = local_pos == 0
        stub = SimpleNamespace(head_dim=d)
        kv_grouped = Compressor._overlap_transform_thd(stub, kv_grouped, is_first, fill_value=0)
        score_grouped = Compressor._overlap_transform_thd(
            stub, score_grouped, is_first, fill_value=float("-inf")
        )

    if mode == "upstream":
        weights = torch.softmax(score_grouped, dim=1, dtype=torch.float32).to(kv_grouped.dtype)
        out = (kv_grouped * weights).sum(dim=1)
    else:  # fp32 intermediates, single final bf16 rounding
        weights = torch.softmax(score_grouped, dim=1, dtype=torch.float32)
        out = (kv_grouped * weights).sum(dim=1).to(torch.bfloat16)
    return out  # (total_comp, 1, d)


def _make_inputs(lens, d, ratio, coff, seed=1234, device="cuda"):
    total = sum(lens)
    w = coff * d
    gen = torch.Generator(device="cpu").manual_seed(seed)
    kv = torch.randn(total, 1, w, generator=gen, dtype=torch.float32).to(torch.bfloat16)
    score = (torch.randn(total, 1, w, generator=gen, dtype=torch.float32).mul_(1.5)).to(
        torch.bfloat16
    )
    ape = torch.randn(ratio, w, generator=gen, dtype=torch.float32).mul_(0.25)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device=device)
    seg_comp = torch.tensor([seg_len // ratio for seg_len in lens])
    cuc = torch.tensor([0] + list(seg_comp.cumsum(0)), dtype=torch.int32, device=device)
    total_comp = int(cuc[-1].item())
    go = torch.randn(total_comp, 1, d, generator=gen, dtype=torch.float32).to(torch.bfloat16)
    return kv.to(device), score.to(device), ape.to(device), cu, cuc, total_comp, go.to(device)


def _run_eager(kv, score, ape, cu, cuc, total_comp, ratio, d, coff, go, mode):
    """Forward + backward through the eager reference; returns (out, dKV, dScore, dAPE)."""
    kv_l = kv.clone().requires_grad_(True)
    score_l = score.clone().requires_grad_(True)
    ape_l = ape.clone().requires_grad_(True)
    out = _eager_pool(kv_l, score_l, ape_l, cu, cuc, total_comp, ratio, d, coff, mode)
    out.backward(go.to(out.dtype))
    torch.cuda.synchronize()
    return out.detach(), kv_l.grad.detach(), score_l.grad.detach(), ape_l.grad.detach()


def _run_fused(kv, score, ape, cu, cuc, total_comp, ratio, d, coff, go):
    """Forward + backward through the dispatch; returns (out, dKV, dScore, dAPE)."""
    kv_l = kv.clone().requires_grad_(True)
    score_l = score.clone().requires_grad_(True)
    ape_l = ape.clone().requires_grad_(True)
    out = cfc.maybe_compress_thd_fused(
        kv_l, score_l, ape_l, cu, cuc, total_comp, ratio=ratio, head_dim=d, coff=coff
    )
    assert out is not None, "fused dispatch did not engage"
    out.backward(go)
    torch.cuda.synchronize()
    return out.detach(), kv_l.grad.detach(), score_l.grad.detach(), ape_l.grad.detach()


_SHAPES = [
    # (lens, head_dim); ratio = 4, coff = 2 (the only dispatched configuration)
    pytest.param([2048], 128, id="b1-d128"),
    pytest.param([1023, 2048, 509], 128, id="ragged3-d128"),
    pytest.param([2048], 512, id="b1-d512"),
    pytest.param([3, 515, 1024, 129], 128, id="short-seg-d128"),
]


@pytest.mark.parametrize("lens,d", _SHAPES)
def test_numerics_vs_eager(lens, d):
    """Dispatched fused fwd+bwd vs fp32-eager (bitwise dKV/dScore) and upstream eager."""
    _require_fused()
    ratio, coff = 4, 2
    kv, score, ape, cu, cuc, total_comp, go = _make_inputs(lens, d, ratio, coff)

    r_fused = _run_fused(kv, score, ape, cu, cuc, total_comp, ratio, d, coff, go)
    r_fp32 = _run_eager(kv, score, ape, cu, cuc, total_comp, ratio, d, coff, go, mode="fp32")
    r_up = _run_eager(kv, score, ape, cu, cuc, total_comp, ratio, d, coff, go, mode="upstream")

    # vs fp32-intermediate eager reference (the fused kernels' numerics contract):
    # dKV / dScore bit-identical; forward within one bf16 rounding step on a tiny
    # fraction of elements; dAPE within fp32 atomics reorder noise.
    assert torch.equal(r_fused[1], r_fp32[1]), "dKV must be bit-identical to the fp32 reference"
    assert torch.equal(r_fused[2], r_fp32[2]), "dScore must be bit-identical to the fp32 reference"
    fwd_diff = (r_fused[0].float() - r_fp32[0].float()).abs()
    n_diff = (r_fused[0] != r_fp32[0]).sum().item()
    assert n_diff <= max(1, int(0.001 * r_fused[0].numel())), n_diff
    assert fwd_diff.max().item() <= 1.6e-2
    assert (r_fused[3] - r_fp32[3]).abs().max().item() <= 1e-3

    # vs the verbatim upstream eager numerics: not bit-identical (the eager path rounds
    # softmax weights to bf16 and multiplies in bf16), but close.
    for fused_t, up_t in zip(r_fused, r_up):
        assert torch.allclose(fused_t.float(), up_t.float(), rtol=0, atol=0.1)


def test_fixed_total_comp_padding():
    """Static-capacity padding rows: eager-matching forward, ignored padding gradients."""
    _require_fused()
    # Leading segment shorter than ratio (0 compressed blocks), so padding rows gather
    # tokens [0, ratio) that span a segment boundary — exactly like the eager gather.
    lens, d, ratio, coff, pad = [3, 515, 1024, 129], 128, 4, 2, 8
    kv, score, ape, cu, cuc, total_true, _ = _make_inputs(lens, d, ratio, coff)
    capacity = total_true + pad
    gen = torch.Generator(device="cpu").manual_seed(7)
    go = torch.randn(capacity, 1, d, generator=gen, dtype=torch.float32)
    go = go.to(torch.bfloat16).cuda()
    go_zero_pad = go.clone()
    go_zero_pad[total_true:] = 0

    r_fused = _run_fused(kv, score, ape, cu, cuc, capacity, ratio, d, coff, go)
    r_fp32 = _run_eager(kv, score, ape, cu, cuc, capacity, ratio, d, coff, go_zero_pad, mode="fp32")

    # Forward: padding rows replicate row 0's window exactly like the eager code, so the
    # full padded output (valid + padding rows) obeys the same criteria as the unpadded
    # comparison.
    assert (r_fused[0] != r_fp32[0]).sum().item() <= max(1, int(0.001 * r_fused[0].numel()))
    assert (r_fused[0].float() - r_fp32[0].float()).abs().max().item() <= 1.6e-2

    # Backward: incoming gradients on padding rows are ignored by design — the fused
    # gradients (computed with NONZERO padding-row grads) match the eager reference run
    # with zeroed padding-row grads bit-for-bit on dKV/dScore.
    assert torch.equal(r_fused[1], r_fp32[1])
    assert torch.equal(r_fused[2], r_fp32[2])
    assert (r_fused[3] - r_fp32[3]).abs().max().item() <= 1e-3


def test_dispatch_gating_and_fallback():
    """``maybe_compress_thd_fused`` returns None for every unsupported configuration."""
    _require_fused()
    kv, score, ape, cu, cuc, total_comp, _ = _make_inputs([512, 256], 128, 4, 2)
    kwargs = dict(ratio=4, head_dim=128, coff=2)

    supported = cfc.maybe_compress_thd_fused(kv, score, ape, cu, cuc, total_comp, **kwargs)
    assert supported is not None and supported.shape == (total_comp, 1, 128)

    # Disabled by the caller (``use_fused_dsa_kernels(config)`` is False).
    assert (
        cfc.maybe_compress_thd_fused(kv, score, ape, cu, cuc, total_comp, enabled=False, **kwargs)
        is None
    )

    # Missing/old cudnn-frontend (no ``cudnn.csa``): the probe caches None and the
    # dispatch keeps eager.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cfc, "_frontend", None)
        assert not cfc.fused_compressor_available()
        assert cfc.maybe_compress_thd_fused(kv, score, ape, cu, cuc, total_comp, **kwargs) is None

    # Deterministic mode keeps the (deterministic) eager path — dAPE uses fp32 atomics.
    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        assert cfc.maybe_compress_thd_fused(kv, score, ape, cu, cuc, total_comp, **kwargs) is None
    finally:
        torch.use_deterministic_algorithms(prev_det, warn_only=prev_warn)

    # Enabling deterministic mode between forward and backward: the frontend backward
    # raises instead of silently returning a nondeterministic dAPE.
    kv_l = kv.clone().requires_grad_(True)
    score_l = score.clone().requires_grad_(True)
    ape_l = ape.clone().requires_grad_(True)
    out = cfc.maybe_compress_thd_fused(kv_l, score_l, ape_l, cu, cuc, total_comp, **kwargs)
    assert out is not None
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        with pytest.raises(RuntimeError, match="not deterministic"):
            out.backward(torch.ones_like(out))
    finally:
        torch.use_deterministic_algorithms(prev_det, warn_only=prev_warn)

    # compress_ratio 128 / coff 1 (the non-overlapping form) dispatches as well.
    kv1, score1, ape1, cu1, cuc1, tc1, _ = _make_inputs([1024], 128, 128, 1)
    out_r128 = cfc.maybe_compress_thd_fused(
        kv1, score1, ape1, cu1, cuc1, tc1, ratio=128, head_dim=128, coff=1
    )
    assert out_r128 is not None and out_r128.shape == (tc1, 1, 128)

    # Head dims outside the r128 kernels' validated set stay on eager.
    assert (
        cfc.maybe_compress_thd_fused(
            kv1, score1, ape1, cu1, cuc1, tc1, ratio=128, head_dim=64, coff=1
        )
        is None
    )

    # The gate follows the frontend envelope, not the Compressor's ratio -> coff
    # derivation: the other two validated combinations dispatch as well.
    kv2, score2, ape2, cu2, cuc2, tc2, _ = _make_inputs([1024], 128, 128, 2)
    out_r128_c2 = cfc.maybe_compress_thd_fused(
        kv2, score2, ape2, cu2, cuc2, tc2, ratio=128, head_dim=128, coff=2
    )
    assert out_r128_c2 is not None and out_r128_c2.shape == (tc2, 1, 128)

    kv3, score3, ape3, cu3, cuc3, tc3, _ = _make_inputs([1024], 128, 4, 1)
    out_r4_c1 = cfc.maybe_compress_thd_fused(
        kv3, score3, ape3, cu3, cuc3, tc3, ratio=4, head_dim=128, coff=1
    )
    assert out_r4_c1 is not None and out_r4_c1.shape == (tc3, 1, 128)

    # Ratios outside the frontend envelope stay on eager.
    assert (
        cfc.maybe_compress_thd_fused(
            kv3, score3, ape3, cu3, cuc3, tc3, ratio=8, head_dim=128, coff=1
        )
        is None
    )

    # Non-bf16 inputs fall back.
    assert (
        cfc.maybe_compress_thd_fused(kv.float(), score.float(), ape, cu, cuc, total_comp, **kwargs)
        is None
    )

    # Unexpected layout falls back.
    assert (
        cfc.maybe_compress_thd_fused(
            kv.view(kv.shape[0], -1),
            score.view(score.shape[0], -1),
            ape,
            cu,
            cuc,
            total_comp,
            **kwargs,
        )
        is None
    )

    # Empty output falls back (nothing to compute).
    assert cfc.maybe_compress_thd_fused(kv, score, ape, cu, cuc, 0, **kwargs) is None


class TestCompressorFusedIntegration:
    """``Compressor._forward_thd`` level: fused dispatch engages and matches eager."""

    @pytest.fixture(scope='class', autouse=True)
    def class_environment(self, request):
        # Skip (do not crash) on machines without CUDA / the frontend / SM100+ before
        # touching model-parallel state.
        _require_fused()

        from megatron.core.process_groups_config import ProcessGroupCollection
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        from tests.unit_tests.test_utilities import Utils

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        cls = request.cls
        from megatron.core.transformer.transformer_config import MLATransformerConfig

        cls.config = MLATransformerConfig(
            num_layers=4,
            hidden_size=256,
            num_attention_heads=16,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            q_lora_rank=64,
            kv_lora_rank=64,
            qk_head_dim=32,
            qk_pos_emb_head_dim=32,
            v_head_dim=64,
            rope_type='rope',
            rotary_base=10000,
            rotary_percent=1.0,
            multi_latent_attention=True,
            experimental_attention_variant='dsv4_hybrid',
            csa_compress_ratios=[4, 128, 4, 128],
            csa_window_size=8,
            dsa_indexer_n_heads=8,
            dsa_indexer_head_dim=64,
            dsa_indexer_topk=8,
            dsa_indexer_loss_coeff=0.0,
            # The fused compressor follows the same switch as the other optional
            # CSA/DSA fused kernels (use_fused_dsa_kernels).
            dsa_kernel_backend='cudnn',
        )
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])

        from megatron.core.models.common.embeddings import RotaryEmbedding

        cls.rotary_pos_emb = RotaryEmbedding(
            cls.config.qk_pos_emb_head_dim,
            rotary_percent=cls.config.rotary_percent,
            rotary_base=cls.config.rotary_base,
            cp_group=cls.pg_collection.cp,
        )

        yield
        Utils.destroy_model_parallel()

    def _make_compressor(self):
        from megatron.core.extensions.transformer_engine import TELinear, TENorm
        from megatron.core.transformer.spec_utils import ModuleSpec

        return Compressor(
            config=self.config,
            submodules=CompressorSubmodules(
                linear_wkv=ModuleSpec(module=TELinear),
                linear_wgate=ModuleSpec(module=TELinear),
                norm=ModuleSpec(module=TENorm),
            ),
            compress_ratio=4,
            head_dim=self.config.v_head_dim,
            rotate=False,
            rotary_pos_emb=self.rotary_pos_emb,
            pg_collection=self.pg_collection,
        ).cuda()

    def test_forward_thd_fused_matches_eager(self):
        """THD forward: the fused dispatch engages and matches the eager path closely."""
        _require_fused()
        compressor = self._make_compressor()
        lens = [255, 512, 129]
        total = sum(lens)
        x = torch.randn(total, 1, self.config.hidden_size, dtype=torch.bfloat16, device="cuda")
        cu_seqlens = torch.tensor(
            [0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device="cuda"
        )

        returns = []
        orig = csa_module.maybe_compress_thd_fused

        def _spy(*args, **kwargs):
            result = orig(*args, **kwargs)
            returns.append(result)
            return result

        with patch.object(csa_module, "maybe_compress_thd_fused", side_effect=_spy):
            out_fused, cuc_fused = compressor._forward_thd(
                x, cu_seqlens, max_seqlen_q=max(lens), fixed_total_comp=total // 4
            )
        assert len(returns) == 1
        assert returns[0] is not None, "fused fast path did not engage"

        # Scope the disable to this call only: the missing-frontend block below must
        # reach the frontend probe, not exit early at the ``enabled`` gate.
        with pytest.MonkeyPatch.context() as mp_off:
            mp_off.setattr(compressor, "use_fused_compressor", False)
            out_eager, cuc_eager = compressor._forward_thd(
                x, cu_seqlens, max_seqlen_q=max(lens), fixed_total_comp=total // 4
            )

        assert out_fused.shape == out_eager.shape
        assert torch.equal(cuc_fused, cuc_eager)
        assert torch.allclose(out_fused.float(), out_eager.float(), rtol=0, atol=0.1)

        # Missing/old cudnn-frontend: bitwise the same eager path as the kill switch.
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(cfc, "_frontend", None)
            out_fb, cuc_fb = compressor._forward_thd(
                x, cu_seqlens, max_seqlen_q=max(lens), fixed_total_comp=total // 4
            )
        assert torch.equal(out_fb, out_eager)
        assert torch.equal(cuc_fb, cuc_eager)

    def test_forward_thd_gradients_flow(self):
        """Gradients flow through the fused fast path to inputs and parameters."""
        _require_fused()
        compressor = self._make_compressor()
        lens = [256, 512]
        total = sum(lens)
        x = torch.randn(
            total, 1, self.config.hidden_size, dtype=torch.bfloat16, device="cuda"
        ).requires_grad_(True)
        cu_seqlens = torch.tensor(
            [0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device="cuda"
        )
        out, _ = compressor._forward_thd(x, cu_seqlens, max_seqlen_q=max(lens))
        out.sum().backward()
        assert x.grad is not None
        assert compressor.ape.grad is not None
        assert compressor.ape.grad.abs().sum().item() > 0
