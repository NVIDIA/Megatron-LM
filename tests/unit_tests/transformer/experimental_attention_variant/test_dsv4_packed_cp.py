# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Packed DSv4 CP parity against per-document SBHD; run on at least 4 GPUs."""

import json
from copy import copy
from dataclasses import replace

import pytest
import torch
import torch.distributed as dist

from megatron.core.extensions.transformer_engine import HAVE_TE, TELinear
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_dsv4_hybrid_attention import (
    HAVE_HADAMARD,
    _build_attention,
    _make_config,
)

pytestmark = pytest.mark.launch_on_gb200


class _CP1:
    @staticmethod
    def size():
        return 1

    @staticmethod
    def rank():
        return 0


def _assert_match(actual, expected):
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    # Keep both relative-norm and elementwise gates while the diagnostic
    # controls separate BF16 accumulation, backend and CP effects.
    delta = (actual.float() - expected.float()).square().sum()
    energy = expected.float().square().sum().clamp_min(1e-12)
    assert delta / energy < 2e-3
    torch.testing.assert_close(actual, expected, atol=0.06, rtol=0.06)


def _reference_per_document(model, hidden, physical, real):
    """Use the existing native SBHD backend without packed layout or CP helpers."""
    outputs = []
    for i, (start, end) in enumerate(zip(physical, physical[1:])):
        length = real[i + 1] - real[i]
        if length:
            output, _ = model(hidden[start : start + length], attention_mask=None)
            outputs.append(output)
        if start + length < end:
            # Keep padding gradients explicitly zero in the reference input.
            outputs.append(hidden[start + length : end] * 0)
    return torch.cat(outputs, dim=0)


def _run_attention(model, hidden, grad, packed, *, documents=None, cp_group=None):
    """Collect real TE FP32 wgrads, or ordinary grads when fusion is disabled."""
    fused_weights = set()
    for module in model.modules():
        if isinstance(module, TELinear) and module.fuse_wgrad_accumulation:
            module.weight.main_grad = torch.zeros_like(module.weight, dtype=torch.float32)
            fused_weights.add(id(module.weight))
    hidden = hidden.detach().clone().requires_grad_()
    if documents is None:
        output, _ = model(hidden, attention_mask=None, packed_seq_params=packed)
    else:
        output = _reference_per_document(model, hidden, *documents)
    output.backward(grad)
    result = {"output": output.detach(), "input_grad": hidden.grad.detach()}
    for name, param in model.named_parameters():
        value = param.main_grad if id(param) in fused_weights else param.grad
        assert value is not None, f"Missing gradient: {name}"
        value = value.detach().to(dtype=torch.float32, copy=True)
        if cp_group is not None:
            dist.all_reduce(value, group=cp_group)
        result[f"param:{name}"] = value
    return result


def _compare_results(actual, expected, label, failures, rows=None):
    """Report every tensor before failing, keeping both existing error gates."""
    assert actual.keys() == expected.keys(), label
    for name, value in actual.items():
        reference = expected[name]
        if rows is not None and not name.startswith("param:"):
            reference = reference[rows]
        a, b = value.float(), reference.float()
        delta = a - b
        energy = b.square().sum().clamp_min(1e-12)
        stats = {
            "comparison": label,
            "tensor": name,
            "dtype": str(value.dtype),
            "reference_dtype": str(reference.dtype),
            "relative_l2": (delta.square().sum() / energy).sqrt().item(),
            "max_abs": delta.abs().max().item(),
            "mismatch_fraction": (delta.abs() > 0.06 + 0.06 * b.abs()).float().mean().item(),
            "reference_norm": b.norm().item(),
            "cosine": (
                1.0
                if torch.equal(a, b)
                else torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
            ),
        }
        try:
            _assert_match(value, reference)
        except AssertionError as error:
            failures.append(f"{label}/{name}: {error}")
            stats["passed"] = False
        else:
            stats["passed"] = True
        print("DSV4_CP_PARITY " + json.dumps(stats), flush=True)


@pytest.mark.skipif(
    not (torch.cuda.is_available() and HAVE_TE and HAVE_HADAMARD),
    reason="needs CUDA, TE and the real Hadamard kernel",
)
@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize(
    "ratio,sparse,coeff,recompute",
    [
        (0, True, 0.0, False),
        (4, True, 0.2, True),
        (4, False, 0.0, False),
        (4, False, 0.2, False),
        (128, True, 0.0, True),
    ],
)
def test_packed_cp_matches_full_attention_and_gradients(cp_size, ratio, sparse, coeff, recompute):
    if Utils.world_size < cp_size:
        pytest.skip(f"requires {cp_size} ranks")
    pytest.importorskip("flash_mla")
    pytest.importorskip("cudnn.deepseek_sparse_attention")
    # CI builds FlashMLA with FLASH_MLA_DISABLE_SM90=1.
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("Fused CSA CP tests require the SM100 kernels included in the CI image")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=cp_size
    )
    try:
        pg = ProcessGroupCollection.use_mpu_process_groups()
        ref_pg = copy(pg)
        ref_pg.cp = _CP1()
        torch.manual_seed(198)
        model_parallel_cuda_manual_seed(198)
        # A rank boundary cuts a document and a ratio-4 group. Real lengths
        # exclude internal padding; a repeated prefix exercises empty documents.
        physical_cu = [0, 133, 133, 1157, 2048]
        real_cu = [0, 129, 129, 1141, 2016]
        cfg = _make_config(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=64,
            v_head_dim=512,
            qk_pos_emb_head_dim=64,
            q_lora_rank=128,
            csa_compress_ratios=[ratio],
            csa_window_size=128,
            dsa_indexer_n_heads=64,
            dsa_indexer_head_dim=128,
            dsa_indexer_topk=512,
            dsa_indexer_loss_coeff=coeff,
            dsa_indexer_use_sparse_loss=sparse,
            dsa_kernel_backend="cudnn",
            context_parallel_size=cp_size,
            attention_cp_layout="contiguous",
            linear_cp_layout="contiguous",
            qk_layernorm=True,
            apply_rope_fusion=True,
            gradient_accumulation_fusion=True,
            recompute_granularity="selective" if recompute else None,
            recompute_modules=["mla_up_proj"] if recompute else [],
        )
        # Keep RoPE arithmetic identical. Sum each document's auxiliary loss
        # with one global divisor, matching the packed mean over real tokens.
        ref_cfg = replace(
            cfg,
            context_parallel_size=1,
            dsa_kernel_backend="none",
            calculate_per_token_loss=True,
            dsa_indexer_loss_coeff=coeff / real_cu[-1],
            recompute_granularity=None,
            recompute_modules=[],
        )
        model = _build_attention(cfg, 1, pg).cuda()
        reference = _build_attention(ref_cfg, 1, ref_pg).cuda()
        reference.load_state_dict(model.state_dict())
        physical = torch.tensor(physical_cu, dtype=torch.int32, device="cuda")
        real = torch.tensor(real_cu, dtype=torch.int32, device="cuda")
        packed = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=real,
            cu_seqlens_kv=real,
            cu_seqlens_q_padded=physical,
            cu_seqlens_kv_padded=physical,
            max_seqlen_q=1024,
            max_seqlen_kv=1024,
        )
        whole = torch.randn(2048, 1, 256, dtype=torch.bfloat16, device="cuda")
        grad = torch.randn_like(whole)
        count = whole.shape[0] // cp_size
        rows = slice(pg.cp.rank() * count, (pg.cp.rank() + 1) * count)
        documents = (physical_cu, real_cu)
        actual = _run_attention(model, whole[rows], grad[rows], packed, cp_group=pg.cp)
        expected = _run_attention(reference, whole, grad, packed, documents=documents)
        failures = []
        label = f"ratio={ratio}:cp={cp_size}:fp32_wgrad"
        _compare_results(actual, expected, f"{label}:cp_vs_native", failures, rows)
        if ratio == 4 and coeff == 0:
            for name, value in actual.items():
                if ".indexer." in name:
                    assert torch.count_nonzero(value) == 0, name

        # Limit extra controls to the problematic sparse/recompute case. They
        # separate CP, recompute and backend effects without expanding the matrix.
        if ratio == 4 and sparse and recompute:
            state = model.state_dict()

            def run_control(config, groups, *, native=False):
                attention = _build_attention(config, 1, groups).cuda()
                attention.load_state_dict(state)
                is_cp = config.context_parallel_size > 1
                return _run_attention(
                    attention,
                    whole[rows] if is_cp else whole,
                    grad[rows] if is_cp else grad,
                    packed,
                    documents=documents if native else None,
                    cp_group=pg.cp if is_cp else None,
                )

            eager_cfg = replace(cfg, recompute_granularity=None, recompute_modules=[])
            eager = run_control(eager_cfg, pg)
            fused_cp1 = run_control(replace(eager_cfg, context_parallel_size=1), ref_pg)
            _compare_results(actual, eager, f"{label}:recompute_vs_eager", failures)
            _compare_results(eager, fused_cp1, f"{label}:cp_vs_fused_cp1", failures, rows)
            _compare_results(fused_cp1, expected, f"{label}:fused_cp1_vs_native", failures)

            # Keep the old BF16-wgrad result visible as a diagnostic baseline;
            # correctness gates above use FP32 accumulation from the TE GEMM onward.
            bf16_actual = run_control(replace(cfg, gradient_accumulation_fusion=False), pg)
            bf16_expected = run_control(
                replace(
                    ref_cfg,
                    gradient_accumulation_fusion=False,
                    recompute_granularity=cfg.recompute_granularity,
                    recompute_modules=cfg.recompute_modules,
                ),
                ref_pg,
                native=True,
            )
            baseline_failures = []
            _compare_results(
                bf16_actual,
                bf16_expected,
                f"ratio={ratio}:cp={cp_size}:bf16_wgrad_baseline",
                baseline_failures,
                rows,
            )
            for result in (bf16_actual, bf16_expected):
                for name, value in result.items():
                    assert torch.isfinite(value).all(), f"Non-finite BF16 baseline: {name}"
        assert not failures, "\n\n".join(failures)
    finally:
        Utils.destroy_model_parallel()
