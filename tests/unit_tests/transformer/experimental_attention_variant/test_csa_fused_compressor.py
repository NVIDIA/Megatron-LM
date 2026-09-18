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
    eager for regular and CP pre-grouped inputs, gradients flow, precomputed RoPE
    positions are reused, and the module falls back to the bitwise-identical eager
    path when the frontend is unavailable;
  - CP projection before compaction: input/weight gradients inside MXFP8, saved
    hidden storage, single-projection fallbacks, FSDP late overwrite_main_grad
    on the first microbatch, and changing-pack CUDA graph replay.

Without a cudnn-frontend that provides ``cudnn.csa`` (or without CUDA / below
compute-capability major 10) every kernel test skips.
"""

import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from megatron.core.fp8_utils import get_fp8_context
from megatron.core.transformer.experimental_attention_variant import csa as csa_module
from megatron.core.transformer.experimental_attention_variant.csa import (
    Compressor,
    CompressorSubmodules,
    batch_of_row,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils
from megatron.core.transformer.experimental_attention_variant.csa_utils import (
    fused_compressor as cfc,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils import thd_layout_kernels

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


def _attach_late_fsdp_overwrite(compressor, *, strategy, fine_grained):
    """Mark FSDP wrapping without setting overwrite_main_grad until linear.forward."""
    fsdp = SimpleNamespace(
        data_parallel_sharding_strategy=strategy, enable_fine_grained_param_gather_hook=fine_grained
    )
    for linear in (compressor.linear_wkv, compressor.linear_wgate):
        linear.weight._megatron_fsdp_model = fsdp
        linear.register_forward_pre_hook(
            lambda _mod, _inp, weight=linear.weight: setattr(weight, "overwrite_main_grad", True)
        )


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

    def _make_compressor(self, ratio=4, head_dim=None, config=None):
        from megatron.core.extensions.transformer_engine import TELinear, TENorm
        from megatron.core.transformer.spec_utils import ModuleSpec

        return Compressor(
            config=config or self.config,
            submodules=CompressorSubmodules(
                linear_wkv=ModuleSpec(module=TELinear),
                linear_wgate=ModuleSpec(module=TELinear),
                norm=ModuleSpec(module=TENorm),
            ),
            compress_ratio=ratio,
            head_dim=head_dim or self.config.v_head_dim,
            rotate=False,
            rotary_pos_emb=self.rotary_pos_emb,
            pg_collection=self.pg_collection,
        ).cuda()

    @pytest.mark.parametrize("ratio,head_dim", [(4, 128), (4, 512), (128, 512)])
    @pytest.mark.parametrize("detach_input", [False, True])
    @pytest.mark.parametrize("fuse_wgrad", [False, True])
    def test_cp_projects_before_compaction_forward_backward_and_saved_storage(
        self, ratio, head_dim, detach_input, fuse_wgrad
    ):
        """BF16 compressor parity inside MXFP8, including first-microbatch fused wgrad.

        The detached case is the indexer contract: its weights train without
        adding gradients to local or exchanged hidden states. Saved-storage
        checks guard the memory benefit rather than just a reordered forward.
        """
        if not thd_layout_kernels._CUTE_AVAILABLE:
            pytest.skip("CP compressor requires CuTeDSL")
        config = copy.copy(self.config)
        config.hidden_size = 2048
        config.gradient_accumulation_fusion = fuse_wgrad
        config.fp8 = "e4m3"
        config.fp8_recipe = "mxfp8"
        config.fp8_param = True
        reference = self._make_compressor(ratio, head_dim, config)
        actual = self._make_compressor(ratio, head_dim, config)
        actual.load_state_dict(reference.state_dict())
        # Cross-rank groups, multiple packed sequences, and an incomplete tail.
        cu = torch.tensor([0, 129, 1801, 2048], dtype=torch.int32, device="cuda")
        local_rows, start, cp_size = 1024, 1024, 2
        x_base = torch.randn(local_rows, 1, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        boundary_base = torch.randn(128, 1, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        layout = thd_layout_kernels.build_cp_compressor_layout(
            cu, start, local_rows, cp_size, ratio
        )
        grad_out = torch.randn(
            layout.group_ids.numel(), 1, head_dim, device="cuda", dtype=torch.bfloat16
        )
        # Only gathered canonical rows receive gradients. Halo and padding must
        # not contribute, although halo tokens feed a canonical overlap window.
        physical = torch.arange(grad_out.shape[0], device="cuda") + grad_out.shape[0]
        canonical = torch.isin(physical, layout.seq_to_rank_row)
        grad_out[~canonical] = 0
        # Emphasize the first crossing group so losing the halo's wgrad cannot
        # hide under the tolerance for BF16 reduction-order differences.
        grad_out[torch.nonzero(canonical, as_tuple=True)[0][0]] *= 16
        projection_runs = []

        def run(module, projected):
            x = x_base.detach().clone().requires_grad_(True)
            boundary = boundary_base.detach().clone().requires_grad_(True)
            x_input = x.detach() if detach_input else x
            boundary_input = boundary.detach() if detach_input else boundary
            for linear in (module.linear_wkv, module.linear_wgate):
                if fuse_wgrad:
                    linear.weight.main_grad = torch.zeros_like(linear.weight, dtype=torch.float32)
            linears = (module.linear_wkv, module.linear_wgate)
            projections = {linear: [] for linear in linears}

            def capture_projection(linear, inputs, output):
                call = {"input": inputs[0].detach(), "output": output[0].detach()}
                projections[linear].append(call)

                def capture_gradient(grad):
                    call["grad"] = grad.detach()

                output[0].register_hook(capture_gradient)

            handles = [linear.register_forward_hook(capture_projection) for linear in linears]
            saved = []

            def pack(tensor):
                saved.append((tuple(tensor.shape), tensor.untyped_storage().data_ptr()))
                return tensor

            with (
                get_fp8_context(config),
                torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor),
            ):
                if projected:
                    out, _ = module._forward_thd_cp(x_input, boundary_input, cu, layout, 2048)
                else:
                    compact, group_ids, positions, local_cu, local_cuc, _, _ = (
                        cp_utils.prepare_cp_compressor_input(
                            x_input, boundary_input, cu, start, cp_size, ratio
                        )
                    )
                    out, _ = module._forward_thd(
                        compact,
                        cu,
                        max_seqlen_q=2048,
                        compressed_group_ids=group_ids,
                        compressed_position_ids=positions,
                        pre_grouped_cu_seqlens=local_cu,
                        pre_grouped_cu_seqlens_compressed=local_cuc,
                    )
            out.backward(grad_out)
            for handle in handles:
                handle.remove()
            for index, linear in enumerate(linears):
                calls = projections[linear]
                assert len(calls) == (2 if projected else 1)
                partials = []
                for call in calls:
                    values, dy = call["input"], call["grad"]
                    # CPU FP32 avoids TF32 in this independent wgrad reference.
                    values = values.reshape(-1, values.shape[-1]).float().cpu()
                    dy = dy.reshape(-1, dy.shape[-1]).float().cpu()
                    partials.append(dy.T @ values)
                expected_wgrad = sum(partials)
                wgrad = linear.weight.main_grad if fuse_wgrad else linear.weight.grad
                tolerance = 1e-3 + 1e-4 * expected_wgrad.abs()
                if not fuse_wgrad:
                    # Each GEMM rounds to BF16; the split path also rounds their
                    # sum. Bound each rounding before cancellation of the terms.
                    u = torch.finfo(torch.bfloat16).eps / 2
                    tolerance += u * sum(part.abs() for part in partials)
                    if projected:
                        tolerance += u * sum(part.bfloat16().float() for part in partials).abs()
                error = (wgrad.float().cpu() - expected_wgrad).abs()
                assert torch.all(error <= tolerance), (
                    f"Projection {index} wgrad exceeds FP32 reference tolerance: "
                    f"max normalized error {(error / tolerance).max().item()}"
                )
            projection_runs.append([projections[linear] for linear in linears])
            compact_rows = layout.compact_to_source.numel()
            if projected:
                assert not any(
                    shape
                    in ((compact_rows, config.hidden_size), (compact_rows, 1, config.hidden_size))
                    for shape, _ in saved
                )
                assert any(ptr == x.untyped_storage().data_ptr() for _, ptr in saved)
            else:
                assert any(ptr == compact.untyped_storage().data_ptr() for _, ptr in saved)
            if detach_input:
                assert x.grad is None and boundary.grad is None
            grads = [module.ape.grad, module.norm.weight.grad]
            if not detach_input:
                grads += [x.grad, boundary.grad]
            return out.detach(), grads

        ref_out, ref_grads = run(reference, False)
        out, grads = run(actual, True)
        torch.testing.assert_close(out[canonical], ref_out[canonical], rtol=2e-2, atol=2e-2)
        # Different GEMM shapes can change BF16 projection rounding and hence dY.
        # Compare those directly; each wgrad above uses its own actual dY oracle.
        source = layout.compact_to_source
        valid = source >= 0
        used = torch.zeros(
            local_rows + layout.boundary_rows, dtype=torch.bool, device=source.device
        )
        used[source[valid].long()] = True
        for ref_calls, calls in zip(*projection_runs):
            for key in ("output", "grad"):
                values = torch.cat((calls[1][key], calls[0][key]))
                if key == "grad":
                    assert torch.count_nonzero(values[~used]) == 0
                compact = values[source[valid].long()]
                torch.testing.assert_close(compact, ref_calls[0][key][valid], rtol=3e-2, atol=5e-2)
        for result, expected in zip(grads, ref_grads):
            torch.testing.assert_close(result.float(), expected.float(), rtol=3e-2, atol=5e-2)

    @pytest.mark.parametrize(
        "fallback",
        ["delay_wgrad_compute", "overwrite_main_grad", "fsdp_optim_grads", "fsdp_fine_grained"],
    )
    @pytest.mark.parametrize("shared_input", [False, True])
    def test_cp_preserves_single_projection_contract(self, fallback, shared_input):
        """Deferred wgrad/FSDP still enqueue exactly one call per linear."""
        compressor = self._make_compressor()
        compressor.config = copy.copy(compressor.config)
        if fallback == "delay_wgrad_compute":
            compressor.config.delay_wgrad_compute = True
        elif fallback == "overwrite_main_grad":
            compressor.linear_wkv.weight.overwrite_main_grad = True
            compressor.linear_wgate.weight.overwrite_main_grad = True
        elif fallback == "fsdp_optim_grads":
            _attach_late_fsdp_overwrite(compressor, strategy="optim_grads", fine_grained=False)
        elif fallback == "fsdp_fine_grained":
            _attach_late_fsdp_overwrite(
                compressor, strategy="optim_grads_params", fine_grained=True
            )
        else:
            raise AssertionError(f"unknown fallback {fallback}")
        if fallback.startswith("fsdp_"):
            assert compressor._cp_requires_single_projection()
            assert not any(
                getattr(linear.weight, "overwrite_main_grad", False)
                for linear in (compressor.linear_wkv, compressor.linear_wgate)
            )
        x = torch.randn(64, 1, self.config.hidden_size, device="cuda", dtype=torch.bfloat16)
        boundary = torch.randn_like(x[:8])
        cu = torch.tensor([0, 128], device="cuda", dtype=torch.int32)
        layout = thd_layout_kernels.build_cp_compressor_layout(cu, 64, 64, 2, 4)
        compact = None
        if shared_input:
            compact, *_ = cp_utils.prepare_cp_compressor_input(x, boundary, cu, 64, 2, 4)
        with patch.object(
            compressor.linear_wkv, "forward", wraps=compressor.linear_wkv.forward
        ) as kv:
            with patch.object(
                compressor.linear_wgate, "forward", wraps=compressor.linear_wgate.forward
            ) as gate:
                compressor._forward_thd_cp(
                    x, boundary, cu, layout, 128, pre_compacted_hidden=compact
                )
        assert kv.call_count == gate.call_count == 1
        if shared_input:
            assert kv.call_args.args[0] is compact
            assert gate.call_args.args[0] is compact

    def test_cp_fsdp_unit_unshard_waits_for_overwrite_flag(self):
        """Layer-level FSDP unshard already sets overwrite_main_grad before CSA."""
        compressor = self._make_compressor()
        fsdp = SimpleNamespace(
            data_parallel_sharding_strategy="optim_grads_params",
            enable_fine_grained_param_gather_hook=False,
        )
        for linear in (compressor.linear_wkv, compressor.linear_wgate):
            linear.weight._megatron_fsdp_model = fsdp
        assert not compressor._cp_requires_single_projection()
        compressor.linear_wkv.weight.overwrite_main_grad = True
        assert compressor._cp_requires_single_projection()

    def test_cp_fsdp_no_shard_does_not_force_single_projection(self):
        """no_shard never overwrites main_grad, even with fine-grained hooks enabled."""
        compressor = self._make_compressor()
        fsdp = SimpleNamespace(
            data_parallel_sharding_strategy="no_shard", enable_fine_grained_param_gather_hook=True
        )
        for linear in (compressor.linear_wkv, compressor.linear_wgate):
            linear.weight._megatron_fsdp_model = fsdp
        assert not compressor._cp_requires_single_projection()

    @pytest.mark.parametrize(
        "strategy,fine_grained", [("optim_grads", False), ("optim_grads_params", True)]
    )
    def test_cp_fsdp_late_overwrite_first_microbatch_wgrad(self, strategy, fine_grained):
        """First FSDP fwd/bwd keeps halo wgrad when overwrite_main_grad is late."""
        if not thd_layout_kernels._CUTE_AVAILABLE:
            pytest.skip("CP compressor requires CuTeDSL")
        config = copy.copy(self.config)
        config.hidden_size = 2048
        config.gradient_accumulation_fusion = True
        config.fp8 = "e4m3"
        config.fp8_recipe = "mxfp8"
        config.fp8_param = True
        reference = self._make_compressor(4, 128, config)
        actual = self._make_compressor(4, 128, config)
        actual.load_state_dict(reference.state_dict())
        _attach_late_fsdp_overwrite(actual, strategy=strategy, fine_grained=fine_grained)
        assert actual._cp_requires_single_projection()
        assert not any(
            getattr(linear.weight, "overwrite_main_grad", False)
            for linear in (actual.linear_wkv, actual.linear_wgate)
        )
        cu = torch.tensor([0, 129, 1801, 2048], dtype=torch.int32, device="cuda")
        local_rows, start, cp_size, ratio = 1024, 1024, 2, 4
        x_base = torch.randn(local_rows, 1, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        boundary_base = torch.randn(128, 1, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        layout = thd_layout_kernels.build_cp_compressor_layout(
            cu, start, local_rows, cp_size, ratio
        )
        grad_out = torch.randn(
            layout.group_ids.numel(), 1, 128, device="cuda", dtype=torch.bfloat16
        )
        physical = torch.arange(grad_out.shape[0], device="cuda") + grad_out.shape[0]
        canonical = torch.isin(physical, layout.seq_to_rank_row)
        grad_out[~canonical] = 0
        grad_out[torch.nonzero(canonical, as_tuple=True)[0][0]] *= 16

        def run(module, projected):
            x = x_base.detach().clone().requires_grad_(True)
            boundary = boundary_base.detach().clone().requires_grad_(True)
            for linear in (module.linear_wkv, module.linear_wgate):
                linear.weight.main_grad = torch.zeros_like(linear.weight, dtype=torch.float32)
            with get_fp8_context(config):
                if projected:
                    out, _ = module._forward_thd_cp(x, boundary, cu, layout, 2048)
                else:
                    compact, group_ids, positions, local_cu, local_cuc, _, _ = (
                        cp_utils.prepare_cp_compressor_input(x, boundary, cu, start, cp_size, ratio)
                    )
                    out, _ = module._forward_thd(
                        compact,
                        cu,
                        max_seqlen_q=2048,
                        compressed_group_ids=group_ids,
                        compressed_position_ids=positions,
                        pre_grouped_cu_seqlens=local_cu,
                        pre_grouped_cu_seqlens_compressed=local_cuc,
                    )
            out.backward(grad_out)
            weight_grads = [
                linear.weight.main_grad.clone()
                for linear in (module.linear_wkv, module.linear_wgate)
            ]
            return out.detach(), weight_grads + [
                module.ape.grad,
                module.norm.weight.grad,
                x.grad,
                boundary.grad,
            ]

        ref_out, ref_grads = run(reference, False)
        out, grads = run(actual, True)
        torch.testing.assert_close(out[canonical], ref_out[canonical], rtol=2e-2, atol=2e-2)
        for result, expected in zip(grads, ref_grads):
            torch.testing.assert_close(result.float(), expected.float(), rtol=3e-2, atol=5e-2)
        assert all(
            getattr(linear.weight, "overwrite_main_grad", False)
            for linear in (actual.linear_wkv, actual.linear_wgate)
        )

    @pytest.mark.parametrize("ratio,head_dim", [(4, 128), (128, 512)])
    def test_cp_compressor_graph_replay_with_fused_wgrad(self, ratio, head_dim):
        """Replay both GEMMs and pooling backward as packed sequence boundaries change."""
        config = copy.copy(self.config)
        config.gradient_accumulation_fusion = True
        compressor = self._make_compressor(ratio, head_dim, config)
        for linear in (compressor.linear_wkv, compressor.linear_wgate):
            linear.weight.main_grad = torch.zeros_like(linear.weight, dtype=torch.float32)
        x = torch.randn(
            1024, 1, config.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        boundary = torch.randn(
            128, 1, config.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        cu = torch.tensor([0, 129, 1801, 2048], dtype=torch.int32, device="cuda")
        layout = thd_layout_kernels.build_cp_compressor_layout(cu, 1024, 1024, 2, ratio)
        grad_out = torch.zeros(
            layout.group_ids.numel(), 1, head_dim, device="cuda", dtype=torch.bfloat16
        )

        def step():
            for tensor in (x, boundary, *compressor.parameters()):
                if tensor.grad is not None:
                    tensor.grad.zero_()
            for linear in (compressor.linear_wkv, compressor.linear_wgate):
                linear.weight.main_grad.zero_()
            local_layout = thd_layout_kernels.build_cp_compressor_layout(cu, 1024, 1024, 2, ratio)
            out, _ = compressor._forward_thd_cp(x, boundary, cu, local_layout, 2048)
            out.backward(grad_out)
            return (
                out,
                x.grad,
                boundary.grad,
                compressor.linear_wkv.weight.main_grad,
                compressor.linear_wgate.weight.main_grad,
                compressor.ape.grad,
                compressor.norm.weight.grad,
            )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = step()
        for prefixes in ([0, 129, 1801, 2048], [0, 997, 1025, 2048], [0, 0, 2048, 2048]):
            cu.copy_(torch.tensor(prefixes, device="cuda", dtype=torch.int32))
            layout = thd_layout_kernels.build_cp_compressor_layout(cu, 1024, 1024, 2, ratio)
            physical = torch.arange(grad_out.shape[0], device="cuda") + grad_out.shape[0]
            with torch.no_grad():
                x.normal_()
                boundary.normal_()
                grad_out.normal_()
                grad_out[~torch.isin(physical, layout.seq_to_rank_row)] = 0
            expected = tuple(t.detach().clone() for t in step())
            graph.replay()
            for actual, eager in zip(captured, expected):
                torch.testing.assert_close(actual.float(), eager.float(), rtol=2e-3, atol=2e-3)

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

    def test_pre_grouped_cp_uses_local_prefixes_and_precomputed_positions(self):
        """CP compact inputs dispatch through the fused pool on their canonical rows."""
        _require_fused()
        compressor = self._make_compressor()
        capacity, ratio = 8, 4
        x_base = torch.randn(
            capacity * ratio, 1, self.config.hidden_size, dtype=torch.bfloat16, device="cuda"
        )
        cu_global = torch.tensor([0, 32, 64], dtype=torch.int32, device="cuda")
        group_ids = torch.tensor([5, 6, 7, 0, 1, -1, -1, -1], dtype=torch.int32, device="cuda")
        position_ids = torch.tensor([20, 24, 28, 0, 4, 0, 0, 0], dtype=torch.int32, device="cuda")
        local_cu = torch.tensor([0, 12, 20], dtype=torch.int32, device="cuda")
        local_cuc = torch.tensor([0, 3, 5], dtype=torch.int32, device="cuda")
        canonical_rows = torch.tensor([1, 2, 3, 4], device="cuda")
        grad_out = torch.zeros(
            capacity, 1, self.config.v_head_dim, dtype=torch.bfloat16, device="cuda"
        )
        grad_out[canonical_rows] = torch.randn_like(grad_out[canonical_rows])

        def run(use_fused):
            x = x_base.detach().clone().requires_grad_(True)
            seen_positions = []

            def fake_rope(tensor, *_args, **kwargs):
                seen_positions.append(kwargs["position_ids"])
                return tensor

            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(compressor, "use_fused_compressor", use_fused)
                mp.setattr(compressor.config, "apply_rope_fusion", True)
                with patch.object(csa_module, "fused_mla_rope_inplace", side_effect=fake_rope):
                    with patch.object(
                        csa_module,
                        "maybe_compress_thd_fused",
                        wraps=csa_module.maybe_compress_thd_fused,
                    ) as dispatch:
                        out, returned_cuc = compressor._forward_thd(
                            x,
                            cu_global,
                            max_seqlen_q=64,
                            compressed_group_ids=group_ids,
                            compressed_position_ids=position_ids,
                            pre_grouped_cu_seqlens=local_cu,
                            pre_grouped_cu_seqlens_compressed=local_cuc,
                        )
                    grad_x, grad_ape = torch.autograd.grad(
                        out, (x, compressor.ape), grad_outputs=grad_out
                    )
            assert returned_cuc is None
            assert dispatch.call_count == 1
            assert dispatch.call_args.args[3] is local_cu
            assert dispatch.call_args.args[4] is local_cuc
            assert seen_positions[0].data_ptr() == position_ids.data_ptr()
            return out.detach(), grad_x.detach(), grad_ape.detach()

        fused = run(True)
        eager = run(False)
        assert torch.allclose(
            fused[0].index_select(0, canonical_rows).float(),
            eager[0].index_select(0, canonical_rows).float(),
            rtol=0,
            atol=0.1,
        )
        assert torch.allclose(fused[1].float(), eager[1].float(), rtol=0, atol=0.1)
        assert torch.allclose(fused[2], eager[2], rtol=0, atol=0.1)
