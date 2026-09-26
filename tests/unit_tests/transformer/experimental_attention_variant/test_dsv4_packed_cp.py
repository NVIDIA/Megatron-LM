# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""DSv4 backend, packed-layout and CP parity; run on at least 4 GPUs."""

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


def _similarities(actual, expected):
    a, b = actual.flatten().double(), expected.flatten().double()
    cosine = 1.0 if torch.equal(a, b) else torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    denominator = (a.square() + b.square()).sum()
    tensor_sim = (2 * (a * b).sum() / denominator).item() if denominator else 1.0
    return cosine, tensor_sim


def _assert_match(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    # Match dev's CP criteria: direction and magnitude, reduced in FP64.
    cosine, tensor_sim = _similarities(actual, expected)
    assert cosine > 0.999, f"cosine similarity {cosine:.8f} must exceed 0.999"
    assert tensor_sim > 0.999, f"tensor similarity {tensor_sim:.8f} must exceed 0.999"


def _forward_per_document(model, hidden, physical, real):
    """Run the configured SBHD backend without packed layout or CP helpers."""
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
        output = _forward_per_document(model, hidden, *documents)
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
    """Report every tensor before failing either of dev's similarity gates."""
    assert actual.keys() == expected.keys(), label
    for name, value in actual.items():
        reference = expected[name]
        if rows is not None and not name.startswith("param:"):
            reference = reference[rows]
        a, b = value.float(), reference.float()
        delta = a - b
        energy = b.square().sum().clamp_min(1e-12)
        cosine, tensor_sim = _similarities(value, reference)
        stats = {
            "comparison": label,
            "tensor": name,
            "dtype": str(value.dtype),
            "reference_dtype": str(reference.dtype),
            "relative_l2": (delta.square().sum() / energy).sqrt().item(),
            "max_abs": delta.abs().max().item(),
            "mismatch_fraction": (delta.abs() > 0.06 + 0.06 * b.abs()).float().mean().item(),
            "reference_norm": b.norm().item(),
            "actual_norm": a.norm().item(),
            "cosine": cosine,
            "tensor_sim": tensor_sim,
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
        # Isolate CP: keep the packed layout, backend and recompute settings identical.
        ref_cfg = replace(cfg, context_parallel_size=1)
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
        expected = _run_attention(reference, whole, grad, packed)
        failures = []
        label = f"ratio={ratio}:cp={cp_size}:fp32_wgrad"
        _compare_results(actual, expected, f"{label}:cp_vs_fused_cp1", failures, rows)
        if ratio == 4 and coeff == 0:
            for name, value in actual.items():
                if ".indexer." in name:
                    assert torch.count_nonzero(value) == 0, name

        # Check the preceding stages once per configuration, not again for CP4.
        if cp_size == 2:
            state = model.state_dict()

            def run_control(config, *, sbhd=False):
                attention = _build_attention(config, 1, ref_pg).cuda()
                attention.load_state_dict(state)
                return _run_attention(
                    attention, whole, grad, packed, documents=documents if sbhd else None
                )

            # Sum each document's auxiliary loss using the packed global real-token count.
            sbhd_cfg = replace(
                ref_cfg, calculate_per_token_loss=True, dsa_indexer_loss_coeff=coeff / real_cu[-1]
            )
            native_sbhd = run_control(replace(sbhd_cfg, dsa_kernel_backend="none"), sbhd=True)
            fused_sbhd = run_control(sbhd_cfg, sbhd=True)
            _compare_results(fused_sbhd, native_sbhd, f"{label}:sbhd_fused_vs_native", failures)
            _compare_results(expected, fused_sbhd, f"{label}:thd_cp1_vs_fused_sbhd", failures)
            if recompute:
                eager = run_control(
                    replace(ref_cfg, recompute_granularity=None, recompute_modules=[])
                )
                _compare_results(expected, eager, f"{label}:recompute_vs_eager", failures)
        assert not failures, "\n\n".join(failures)
    finally:
        Utils.destroy_model_parallel()
