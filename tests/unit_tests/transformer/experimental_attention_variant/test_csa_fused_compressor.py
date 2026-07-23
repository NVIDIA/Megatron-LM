# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for the fused CuTe-DSL CSA Compressor kernels.

Covers, per https://github.com/NVIDIA/Megatron-LM/issues/5968:
  - numerics of the fused region vs an fp32-intermediate eager reference (``dKV``/``dScore``
    bit-identical, forward within one bf16 rounding step), vs the verbatim upstream eager
    numerics (tolerance), and vs an fp64 oracle (fused error <= eager error);
  - ragged THD packs (segment lengths not multiples of ``ratio``, segments shorter than
    ``ratio``);
  - ``coff == 1`` (``compress_ratio == 128``) functional correctness through the explicit
    op API;
  - static-capacity padding rows (``fixed_total_comp``);
  - run-to-run determinism of forward / ``dKV`` / ``dScore`` (``dAPE`` uses fp32 atomics
    and is exempt by design);
  - CUDA graph capture: warmup -> capture fwd+bwd -> replay (including replay with new
    data and a smaller device-side true row count), and the loud error when the first
    call for a configuration would JIT under capture;
  - dispatch gating / eager fallback of ``maybe_compress_thd_fused`` and the
    ``Compressor._forward_thd`` integration.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import csa as csa_module
from megatron.core.transformer.experimental_attention_variant import csa_fused_compressor as cfc
from megatron.core.transformer.experimental_attention_variant.csa import (
    Compressor,
    CompressorSubmodules,
    batch_of_row,
)

# Run this module on GB200 hardware in CI (marker-driven selection, see
# tests/unit_tests/find_test_cases.py); everywhere else the tests skip via
# _require_fused().
pytestmark = pytest.mark.launch_on_gb200


def _require_fused():
    if not torch.cuda.is_available():
        pytest.skip("fused CSA compressor tests require CUDA")
    if not cfc._CUTE_AVAILABLE:
        pytest.skip("nvidia-cutlass-dsl is not available in this environment")
    if not cfc.fused_compressor_available():
        pytest.skip("fused CSA compressor requires compute capability 10.0")


# ---------------------------------------------------------------------------
# Eager reference: verbatim replica of the region of ``Compressor._forward_thd``
# (non-pre-grouped THD path) that the fused kernels replace, from the projection
# outputs (kv, score) to the pre-RMSNorm pooled output. ``mode`` selects the
# numerics: "upstream" reproduces the current eager code exactly (softmax weights
# rounded to bf16, bf16 multiply); "fp32" keeps all intermediates fp32 with a
# single final bf16 rounding (the fused kernels' numerics); "fp64" is an oracle.
# The overlap-window transform is the real upstream implementation.
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
    elif mode == "fp64":
        kv = kv.double()
        score = score.double()
        ape = ape.double()

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
    elif mode == "fp32":
        weights = torch.softmax(score_grouped, dim=1, dtype=torch.float32)
        out = (kv_grouped * weights).sum(dim=1).to(torch.bfloat16)
    else:  # fp64 oracle
        weights = torch.softmax(score_grouped, dim=1, dtype=torch.float64)
        out = (kv_grouped * weights).sum(dim=1)
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
    dtype = torch.float64 if mode == "fp64" else None
    kv_l = (kv.to(dtype) if dtype else kv.clone()).requires_grad_(True)
    score_l = (score.to(dtype) if dtype else score.clone()).requires_grad_(True)
    ape_l = (ape.to(dtype) if dtype else ape.clone()).requires_grad_(True)
    out = _eager_pool(kv_l, score_l, ape_l, cu, cuc, total_comp, ratio, d, coff, mode)
    out.backward(go.to(out.dtype))
    torch.cuda.synchronize()
    return out.detach(), kv_l.grad.detach(), score_l.grad.detach(), ape_l.grad.detach()


def _run_fused(kv, score, ape, cu, cuc, total_comp, ratio, d, coff, go):
    """Forward + backward through the fused op; returns (out, dKV, dScore, dAPE)."""
    kv_l = kv.clone().requires_grad_(True)
    score_l = score.clone().requires_grad_(True)
    ape_l = ape.clone().requires_grad_(True)
    out = cfc.compress_thd_fused(
        kv_l.view(kv.shape[0], -1),
        score_l.view(score.shape[0], -1),
        ape_l,
        cu,
        cuc,
        ratio,
        d,
        coff,
        total_comp=total_comp,
    ).view(total_comp, 1, d)
    out.backward(go)
    torch.cuda.synchronize()
    return out.detach(), kv_l.grad.detach(), score_l.grad.detach(), ape_l.grad.detach()


_SHAPES = [
    # (lens, head_dim, ratio, coff)
    pytest.param([2048], 128, 4, 2, id="b1-d128-r4"),
    pytest.param([1023, 2048, 509], 128, 4, 2, id="ragged3-d128-r4"),
    pytest.param([2048], 512, 4, 2, id="b1-d512-r4"),
    pytest.param([3, 515, 1024, 129], 128, 4, 2, id="short-seg-d128-r4"),
    pytest.param([1024, 300], 128, 128, 1, id="b2-d128-r128-coff1"),
]


@pytest.mark.parametrize("lens,d,ratio,coff", _SHAPES)
def test_numerics_vs_references(lens, d, ratio, coff):
    """Fused fwd+bwd vs fp32-eager (bitwise dKV/dScore), upstream eager, and fp64 oracle."""
    _require_fused()
    kv, score, ape, cu, cuc, total_comp, go = _make_inputs(lens, d, ratio, coff)

    r_fused = _run_fused(kv, score, ape, cu, cuc, total_comp, ratio, d, coff, go)
    r_fp32 = _run_eager(kv, score, ape, cu, cuc, total_comp, ratio, d, coff, go, mode="fp32")
    r_up = _run_eager(kv, score, ape, cu, cuc, total_comp, ratio, d, coff, go, mode="upstream")
    r_fp64 = _run_eager(kv, score, ape, cu, cuc, total_comp, ratio, d, coff, go, mode="fp64")

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

    # vs the fp64 oracle: the fused kernel is at least as accurate as the current eager
    # code on every output.
    for i in range(4):
        err_fused = (r_fused[i].double() - r_fp64[i].double()).abs().max().item()
        err_up = (r_up[i].double() - r_fp64[i].double()).abs().max().item()
        assert err_fused <= err_up * (1 + 1e-6) + 1e-4, (i, err_fused, err_up)


def test_replay_determinism():
    """Forward, dKV and dScore replay bitwise identically run to run (dAPE is exempt)."""
    _require_fused()
    kv, score, ape, cu, cuc, total_comp, go = _make_inputs([1023, 2048, 509], 128, 4, 2)
    runs = [_run_fused(kv, score, ape, cu, cuc, total_comp, 4, 128, 2, go) for _ in range(3)]
    for other in runs[1:]:
        assert torch.equal(runs[0][0], other[0])
        assert torch.equal(runs[0][1], other[1])
        assert torch.equal(runs[0][2], other[2])
        # dAPE is accumulated with fp32 atomics; equality is not guaranteed, closeness is.
        assert torch.allclose(runs[0][3], other[3], rtol=0, atol=1e-3)


_PADDING_SHAPES = [
    # (lens, head_dim, ratio, coff, pad); the first case has a LEADING segment shorter
    # than ratio (0 compressed blocks), so padding rows gather tokens [0, ratio) that
    # span a segment boundary — exactly like the eager gather.
    ([3, 515, 1024, 129], 128, 4, 2, 8),
    ([1023, 2048, 509], 128, 4, 2, 8),
    ([1024, 300], 128, 128, 1, 5),
]


@pytest.mark.parametrize("lens,d,ratio,coff,pad", _PADDING_SHAPES)
def test_fixed_total_comp_padding(lens, d, ratio, coff, pad):
    """Static-capacity padding rows: eager-matching forward, ignored padding gradients."""
    _require_fused()
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

    # And explicitly: nonzero vs zero padding-row grads produce identical fused grads.
    r_fused_zero = _run_fused(kv, score, ape, cu, cuc, capacity, ratio, d, coff, go_zero_pad)
    assert torch.equal(r_fused[1], r_fused_zero[1])
    assert torch.equal(r_fused[2], r_fused_zero[2])


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_cuda_graph_capture():
    """Warmup -> capture fwd+bwd -> replay; JIT under capture raises a clear error."""
    _require_fused()
    ratio, d, coff = 4, 128, 2
    lens = [512, 256]
    kv, score, ape, cu, cuc, total_true, _ = _make_inputs(lens, d, ratio, coff)
    capacity = total_true + 8  # static capacity, as with fixed_total_comp

    kv_s = kv.view(kv.shape[0], -1).clone().requires_grad_(True)
    score_s = score.view(score.shape[0], -1).clone().requires_grad_(True)
    ape_s = ape.clone().requires_grad_(True)
    go_s = torch.zeros(capacity, d, device="cuda", dtype=torch.bfloat16)
    go_s[:total_true] = torch.randn(total_true, d, device="cuda").to(torch.bfloat16)

    def _fused_fwd_bwd():
        out = cfc.compress_thd_fused(
            kv_s, score_s, ape_s, cu, cuc, ratio, d, coff, total_comp=capacity
        )
        grads = torch.autograd.grad(out, [kv_s, score_s, ape_s], grad_outputs=go_s)
        return out, grads

    # One eager warmup per configuration on a side stream (JIT-compiles both kernels).
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _fused_fwd_bwd()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_c, (gkv_c, gscore_c, gape_c) = _fused_fwd_bwd()

    # Replay on the same data must reproduce the direct (non-captured) fused results
    # bitwise on forward/dKV/dScore.
    graph.replay()
    torch.cuda.synchronize()
    ref_out, (ref_gkv, ref_gscore, ref_gape) = _fused_fwd_bwd()
    torch.cuda.synchronize()
    assert torch.equal(out_c, ref_out)
    assert torch.equal(gkv_c, ref_gkv)
    assert torch.equal(gscore_c, ref_gscore)
    assert torch.allclose(gape_c, ref_gape, rtol=0, atol=1e-3)

    # Replay with new data and a SMALLER device-side true row count (the fixed capacity
    # stays static, cu/cuc contents change) — the graph-replayed gradients must match the
    # fp32 eager reference bitwise on dKV/dScore.
    lens2 = [384, 128]
    kv2, score2, ape2, cu2, cuc2, total2, _ = _make_inputs(lens2, d, ratio, coff, seed=99)
    n2 = sum(lens2)
    kv_s.data.zero_()
    score_s.data.zero_()
    kv_s.data[:n2] = kv2.view(n2, -1)
    score_s.data[:n2] = score2.view(n2, -1)
    ape_s.data.copy_(ape2)
    cu.copy_(cu2)
    cuc.copy_(cuc2)
    go_s.zero_()
    go_s[:total2] = torch.randn(total2, d, device="cuda").to(torch.bfloat16)
    graph.replay()
    torch.cuda.synchronize()
    r_fp32 = _run_eager(
        kv_s.data.view(-1, 1, coff * d),
        score_s.data.view(-1, 1, coff * d),
        ape_s.data,
        cu,
        cuc,
        capacity,
        ratio,
        d,
        coff,
        go_s.view(capacity, 1, d),
        mode="fp32",
    )
    assert torch.equal(gkv_c.view_as(r_fp32[1]), r_fp32[1])
    assert torch.equal(gscore_c.view_as(r_fp32[2]), r_fp32[2])

    # A first call for a NEW configuration under capture must raise loudly instead of
    # JIT-compiling (which is not capture-safe). head_dim 192 is used by no other test
    # in this module, so this configuration is guaranteed to be uncompiled regardless of
    # test execution order.
    d_new = 192
    kv3 = torch.randn(256, coff * d_new, device="cuda").to(torch.bfloat16)
    score3 = torch.randn(256, coff * d_new, device="cuda").to(torch.bfloat16)
    ape3 = torch.randn(ratio, coff * d_new, device="cuda")
    cu3 = torch.tensor([0, 256], dtype=torch.int32, device="cuda")
    cuc3 = torch.tensor([0, 64], dtype=torch.int32, device="cuda")
    graph2 = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="CUDA graph capture"):
        with torch.cuda.graph(graph2):
            cfc.compress_thd_fused(kv3, score3, ape3, cu3, cuc3, ratio, d_new, coff, total_comp=64)
    # The CUDA context must remain usable after the aborted capture.
    torch.cuda.synchronize()
    probe = torch.ones(8, device="cuda")
    assert probe.sum().item() == 8


def test_dispatch_gating_and_fallback(monkeypatch):
    """``maybe_compress_thd_fused`` returns None for every unsupported configuration."""
    _require_fused()
    kv, score, ape, cu, cuc, total_comp, _ = _make_inputs([512, 256], 128, 4, 2)
    kwargs = dict(ratio=4, head_dim=128, coff=2)

    supported = cfc.maybe_compress_thd_fused(kv, score, ape, cu, cuc, total_comp, **kwargs)
    assert supported is not None and supported.shape == (total_comp, 1, 128)
    direct = cfc.compress_thd_fused(
        kv.view(kv.shape[0], -1),
        score.view(score.shape[0], -1),
        ape,
        cu,
        cuc,
        4,
        128,
        2,
        total_comp=total_comp,
    )
    assert torch.equal(supported.squeeze(1), direct)

    # Kill switch.
    monkeypatch.setenv("MCORE_CSA_FUSED_COMPRESSOR", "0")
    assert cfc.maybe_compress_thd_fused(kv, score, ape, cu, cuc, total_comp, **kwargs) is None
    monkeypatch.delenv("MCORE_CSA_FUSED_COMPRESSOR")

    # Deterministic mode keeps the (deterministic) eager path — dAPE uses fp32 atomics.
    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        assert cfc.maybe_compress_thd_fused(kv, score, ape, cu, cuc, total_comp, **kwargs) is None
        # The explicit op API raises in backward instead of silently returning a
        # nondeterministic dAPE.
        kv_l = kv.view(kv.shape[0], -1).clone().requires_grad_(True)
        score_l = score.view(score.shape[0], -1).clone().requires_grad_(True)
        ape_l = ape.clone().requires_grad_(True)
        out = cfc.compress_thd_fused(
            kv_l, score_l, ape_l, cu, cuc, 4, 128, 2, total_comp=total_comp
        )
        with pytest.raises(RuntimeError, match="not deterministic"):
            out.backward(torch.ones_like(out))
    finally:
        torch.use_deterministic_algorithms(prev_det, warn_only=prev_warn)

    # compress_ratio 128 stays on eager in the dispatch (functionally supported by the
    # explicit op, not yet a wall-clock win).
    kv1, score1, ape1, cu1, cuc1, tc1, _ = _make_inputs([1024], 128, 128, 1)
    assert (
        cfc.maybe_compress_thd_fused(
            kv1, score1, ape1, cu1, cuc1, tc1, ratio=128, head_dim=128, coff=1
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
        # Skip (do not crash) on machines without CUDA / the DSL / CC 10.0 before
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

    def test_forward_thd_fused_matches_eager(self, monkeypatch):
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

        monkeypatch.setenv("MCORE_CSA_FUSED_COMPRESSOR", "0")
        out_eager, cuc_eager = compressor._forward_thd(
            x, cu_seqlens, max_seqlen_q=max(lens), fixed_total_comp=total // 4
        )
        monkeypatch.delenv("MCORE_CSA_FUSED_COMPRESSOR")

        assert out_fused.shape == out_eager.shape
        assert torch.equal(cuc_fused, cuc_eager)
        assert torch.allclose(out_fused.float(), out_eager.float(), rtol=0, atol=0.1)

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
