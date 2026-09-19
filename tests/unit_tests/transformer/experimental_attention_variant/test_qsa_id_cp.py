# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Two-rank opt-in QSA selected-ID parity across CP zigzag and packed VL RoPE."""

import os
import hashlib
from pathlib import Path

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_qsa_module_spec_for_backend,
)
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core import parallel_state
from megatron.core.ssm.mamba_context_parallel import reconstruct_tensor_cp, split_tensor_cp
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.spec_utils import build_module
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_attention_variant_qsa import (
    _make_config,
    _rotary,
)


def _emit(*parts):
    """Write one atomic record when both torchrun ranks share a log file."""
    os.write(1, (" ".join(map(str, parts)) + "\n").encode())


def _weight_sha(weight):
    return hashlib.sha256(
        weight.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


@pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 2, reason="requires two ranks")
@pytest.mark.parametrize("packed,mrope", [(False, False), (True, False), (True, True)])
def test_qsa_id_cp2_matches_dense_masked_local_output_and_gradients(packed, mrope):
    Utils.initialize_model_parallel(1, 1, context_parallel_size=2)
    torch.cuda.set_per_process_memory_fraction(0.06)
    model_parallel_cuda_manual_seed(123)
    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    os.environ["NVTE_UNFUSED_ATTN"] = "1"
    try:
        config = _make_config(context_parallel_size=2)
        if mrope:
            config.mrope_section = [2, 1, 1]
        spec = get_qsa_module_spec_for_backend(config)
        attention = build_module(spec, config=config, layer_number=1).cuda().eval()
        seq_len = 52
        cu = torch.tensor([0, 24, seq_len], dtype=torch.int32, device="cuda")
        packed_params = (
            PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu,
                cu_seqlens_kv=cu,
                cu_seqlens_q_padded=cu,
                cu_seqlens_kv_padded=cu,
                max_seqlen_q=28,
                max_seqlen_kv=28,
            )
            if packed
            else None
        )
        torch.manual_seed(131)
        global_hidden = torch.randn(
            seq_len, 1, config.hidden_size, dtype=torch.bfloat16, device="cuda"
        )
        local_hidden = split_tensor_cp(global_hidden, packed_params, dim=0)
        if mrope:
            global_freqs = torch.randn(seq_len, 1, 1, 8, device="cuda")
            local_freqs = split_tensor_cp(global_freqs, packed_params, dim=0)
        elif packed:
            # Packed standard RoPE takes the full max-document position table.
            local_freqs = RotaryEmbedding(
                kv_channels=config.kv_channels, rotary_percent=0.5, rotary_base=10000
            ).get_emb(28)
        else:
            local_freqs = _rotary(config, seq_len).cuda()
        grad_output = torch.randn_like(local_hidden)

        def run(backend):
            attention.core_attention.sparse_backend = backend
            hidden = local_hidden.detach().clone().requires_grad_()
            output, _ = attention(
                hidden,
                attention_mask=None,
                rotary_pos_emb=local_freqs,
                packed_seq_params=packed_params,
            )
            gradients = torch.autograd.grad(
                output, (hidden, attention.linear_qkv.weight), grad_output
            )
            return output.detach(), tuple(grad.detach() for grad in gradients)

        dense = run("dense_masked")
        sparse = run("id_sparse")
        torch.testing.assert_close(sparse[0].float(), dense[0].float(), atol=3e-2, rtol=3e-2)
        for actual, reference in zip(sparse[1], dense[1]):
            torch.testing.assert_close(actual.float(), reference.float(), atol=6e-2, rtol=6e-2)
        assert attention.core_attention._selection.selected_bits is None
        assert attention.core_attention._selection.selected_ids is not None
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 1, reason="requires one rank")
def test_qsa_id_cp1_real_vl_geometry_sparse_packed_mrope_gradient_baseline():
    """CP1 baseline for the same real geometry and rank-0-only loss as the CP2 gate."""
    Utils.initialize_model_parallel(1, 1)
    torch.cuda.set_per_process_memory_fraction(0.06)
    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    os.environ["NVTE_UNFUSED_ATTN"] = "1"
    try:
        torch.manual_seed(137)
        config = _make_config(
            hidden_size=2560,
            num_attention_heads=24,
            num_query_groups=2,
            kv_channels=256,
            qsa_indexer_n_heads=4,
            qsa_indexer_kv_heads=1,
            qsa_indexer_head_dim=128,
            qsa_indexer_budget=2048,
            qsa_indexer_compress_ratio=4,
            mrope_section=[11, 11, 10],
        )
        attention = (
            build_module(get_qsa_module_spec_for_backend(config), config=config, layer_number=1)
            .cuda()
            .eval()
        )
        total = 2064
        cu = torch.tensor([0, 2056, total], dtype=torch.int32, device="cuda")
        packed_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            cu_seqlens_q_padded=cu,
            cu_seqlens_kv_padded=cu,
            max_seqlen_q=2056,
            max_seqlen_kv=2056,
        )
        torch.manual_seed(151)
        hidden_base = torch.randn(total, 1, config.hidden_size, dtype=torch.bfloat16, device="cuda")
        global_freqs = torch.randn(total, 1, 1, 64, device="cuda")
        rank0_positions = torch.cat(
            (
                torch.arange(0, 514, device="cuda"),
                torch.arange(1542, 2056, device="cuda"),
                torch.arange(2056, 2058, device="cuda"),
                torch.arange(2062, 2064, device="cuda"),
            )
        )
        upstream_grad = torch.zeros_like(hidden_base)
        upstream_grad[rank0_positions] = torch.randn(
            1032, 1, config.hidden_size, dtype=torch.bfloat16, device="cuda"
        )

        def run(backend):
            attention.core_attention.sparse_backend = backend
            hidden = hidden_base.detach().clone().requires_grad_()
            output, _ = attention(
                hidden,
                attention_mask=None,
                rotary_pos_emb=global_freqs,
                packed_seq_params=packed_params,
            )
            assert attention.core_attention._selection.all_selected is False
            gradients = torch.autograd.grad(
                output, (hidden, attention.linear_qkv.weight), upstream_grad
            )
            return output.detach(), tuple(grad.detach() for grad in gradients)

        dense, sparse = run("dense_masked"), run("id_sparse")
        if save_dir := os.environ.get("QSA_CP_SAVE_DIR"):
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            for backend, result in (("dense", dense), ("id", sparse)):
                torch.save(
                    {
                        "output": result[0].cpu(),
                        "hidden_grad": result[1][0].cpu(),
                        "weight_grad": result[1][1].cpu(),
                        "weight_sha": _weight_sha(attention.linear_qkv.weight),
                    },
                    Path(save_dir) / f"cp1-{backend}.pt",
                )
        torch.testing.assert_close(sparse[0].float(), dense[0].float(), atol=7e-2, rtol=7e-2)
        for label, actual, reference in zip(("hidden", "qkv_weight"), sparse[1], dense[1]):
            difference = (actual.float() - reference.float()).abs()
            mismatch = difference > (8e-2 + 8e-2 * reference.float().abs())
            _emit(
                "QSA_CP1_REAL_STATS",
                label,
                "max_abs",
                difference.max().item(),
                "mismatch_008",
                mismatch.sum().item(),
                "count",
                reference.numel(),
                "relative_l2",
                (difference.norm() / reference.float().norm()).item(),
            )
            assert mismatch.sum().item() <= 8
            torch.testing.assert_close(actual.float(), reference.float(), atol=1.2e-1, rtol=8e-2)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 2, reason="requires two ranks")
def test_qsa_id_cp2_real_vl_geometry_sparse_packed_mrope_gradients():
    """Exercise D256/24Q/2KV/K512 and a sparse 2056-token document across CP ranks."""
    Utils.initialize_model_parallel(1, 1, context_parallel_size=2)
    torch.cuda.set_per_process_memory_fraction(0.06)
    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    os.environ["NVTE_UNFUSED_ATTN"] = "1"
    try:
        torch.manual_seed(137)
        config = _make_config(
            context_parallel_size=2,
            hidden_size=2560,
            num_attention_heads=24,
            num_query_groups=2,
            kv_channels=256,
            qsa_indexer_n_heads=4,
            qsa_indexer_kv_heads=1,
            qsa_indexer_head_dim=128,
            qsa_indexer_budget=2048,
            qsa_indexer_compress_ratio=4,
            mrope_section=[11, 11, 10],
        )
        attention = (
            build_module(get_qsa_module_spec_for_backend(config), config=config, layer_number=1)
            .cuda()
            .eval()
        )
        total = 2064
        cu = torch.tensor([0, 2056, total], dtype=torch.int32, device="cuda")
        packed_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            cu_seqlens_q_padded=cu,
            cu_seqlens_kv_padded=cu,
            max_seqlen_q=2056,
            max_seqlen_kv=2056,
        )
        torch.manual_seed(151)
        global_hidden = torch.randn(
            total, 1, config.hidden_size, dtype=torch.bfloat16, device="cuda"
        )
        hidden_base = split_tensor_cp(global_hidden, packed_params, dim=0)
        global_freqs = torch.randn(total, 1, 1, 64, device="cuda")
        local_freqs = split_tensor_cp(global_freqs, packed_params, dim=0)
        upstream_grad = (
            torch.randn_like(hidden_base)
            if parallel_state.get_context_parallel_rank() == 0
            else torch.zeros_like(hidden_base)
        )

        def run(backend, recompute=False):
            attention.core_attention.sparse_backend = backend
            hidden = hidden_base.detach().clone().requires_grad_()

            def attention_forward(input_hidden):
                output, _ = attention(
                    input_hidden,
                    attention_mask=None,
                    rotary_pos_emb=local_freqs,
                    packed_seq_params=packed_params,
                )
                return output

            output = (
                checkpoint(attention_forward, hidden, use_reentrant=False)
                if recompute
                else attention_forward(hidden)
            )
            assert attention.core_attention._selection.all_selected is False
            gradients = torch.autograd.grad(
                output, (hidden, attention.linear_qkv.weight), upstream_grad
            )
            return output.detach(), tuple(grad.detach() for grad in gradients)

        dense = run("dense_masked")
        sparse = run("id_sparse")
        recomputed = run("id_sparse", recompute=True)
        if save_dir := os.environ.get("QSA_CP_SAVE_DIR"):
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            for backend, result in (("dense", dense), ("id", sparse)):
                full_output = reconstruct_tensor_cp(result[0], packed_params, dim=0)
                full_hidden_grad = reconstruct_tensor_cp(result[1][0], packed_params, dim=0)
                torch.save(
                    {
                        "output": full_output.cpu(),
                        "hidden_grad": full_hidden_grad.cpu(),
                        "weight_grad": result[1][1].cpu(),
                        "weight_sha": _weight_sha(attention.linear_qkv.weight),
                    },
                    Path(save_dir)
                    / f"cp2-{backend}-rank{parallel_state.get_context_parallel_rank()}.pt",
                )
        torch.testing.assert_close(recomputed[0].float(), sparse[0].float(), atol=0, rtol=0)
        for actual, reference in zip(recomputed[1], sparse[1]):
            difference = (actual.float() - reference.float()).abs()
            _emit(
                "QSA_CP_REAL_RECOMPUTE",
                parallel_state.get_context_parallel_rank(),
                "max_abs",
                difference.max().item(),
                "relative_l2",
                (difference.norm() / reference.float().norm()).item(),
            )
            torch.testing.assert_close(actual.float(), reference.float(), atol=8e-2, rtol=8e-2)
        torch.testing.assert_close(sparse[0].float(), dense[0].float(), atol=7e-2, rtol=7e-2)
        for index, (actual, reference) in enumerate(zip(sparse[1], dense[1])):
            actual_fp32, reference_fp32 = actual.float(), reference.float()
            difference = (actual_fp32 - reference_fp32).abs()
            strict_mismatch = difference > (8e-2 + 8e-2 * reference_fp32.abs())
            worst_flat = difference.flatten().argmax()
            worst_position = torch.unravel_index(worst_flat, difference.shape)
            _emit(
                "QSA_CP_REAL_STATS",
                parallel_state.get_context_parallel_rank(),
                "hidden" if index == 0 else "qkv_weight",
                "max_abs",
                difference.max().item(),
                "mismatch_008",
                strict_mismatch.sum().item(),
                "count",
                reference.numel(),
                "relative_l2",
                (difference.norm() / reference_fp32.norm()).item(),
                "nonzero_grad",
                torch.count_nonzero(actual).item(),
                "worst_position",
                tuple(position.item() for position in worst_position),
                "worst_actual",
                actual_fp32.flatten()[worst_flat].item(),
                "worst_reference",
                reference_fp32.flatten()[worst_flat].item(),
                "reference_l2",
                reference_fp32.norm().item(),
            )
            if index == 1:
                bad_coordinates = torch.nonzero(strict_mismatch)[:8]
                _emit(
                    "QSA_CP_REAL_BAD",
                    parallel_state.get_context_parallel_rank(),
                    [
                        (
                            row.item(),
                            col.item(),
                            actual_fp32[row, col].item(),
                            reference_fp32[row, col].item(),
                        )
                        for row, col in bad_coordinates
                    ],
                )
                group_rows = actual.shape[0] // config.num_query_groups
                heads_per_group = config.num_attention_heads // config.num_query_groups
                spans = (
                    ("query", 0, heads_per_group * config.kv_channels),
                    (
                        "gate",
                        heads_per_group * config.kv_channels,
                        2 * heads_per_group * config.kv_channels,
                    ),
                    (
                        "key",
                        2 * heads_per_group * config.kv_channels,
                        (2 * heads_per_group + 1) * config.kv_channels,
                    ),
                    ("value", (2 * heads_per_group + 1) * config.kv_channels, group_rows),
                )
                for name, begin, end in spans:
                    view = difference.reshape(config.num_query_groups, group_rows, -1)[:, begin:end]
                    mismatches = strict_mismatch.reshape(config.num_query_groups, group_rows, -1)[
                        :, begin:end
                    ]
                    _emit(
                        "QSA_CP_REAL_SLICE",
                        parallel_state.get_context_parallel_rank(),
                        name,
                        "max_abs",
                        view.max().item(),
                        "mismatch_008",
                        mismatches.sum().item(),
                        "count",
                        view.numel(),
                    )
            if index == 0:
                torch.testing.assert_close(actual_fp32, reference_fp32, atol=8e-2, rtol=8e-2)
            else:
                # BF16 QKV weight accumulation differs in a few of 34M cells.
                assert strict_mismatch.sum().item() <= 8
                torch.testing.assert_close(actual_fp32, reference_fp32, atol=1.2e-1, rtol=8e-2)
        if parallel_state.get_context_parallel_rank() == 1:
            # Rank 1 has zero upstream gradient; its nonzero input gradient is
            # entirely due to K/V consumed by rank 0 queries after CP gather.
            assert torch.count_nonzero(upstream_grad) == 0
            assert torch.count_nonzero(sparse[1][0]) > 0
            grouped_weight_grad = sparse[1][1].view(config.num_query_groups, -1, config.hidden_size)
            assert torch.count_nonzero(grouped_weight_grad[:, : 24 * 256]) == 0
            assert torch.count_nonzero(grouped_weight_grad[:, 24 * 256 :]) > 0
        assert attention.core_attention._selection.selected_bits is None
        assert attention.core_attention._selection.selected_ids is not None
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not os.environ.get("QSA_CP_SAVE_DIR"), reason="requires saved CP1/CP2 tensors")
def test_qsa_id_cp1_cp2_saved_real_geometry_parity():
    """Compare full outputs/hidden grads and summed weight grads after both GPU probes."""
    directory = Path(os.environ["QSA_CP_SAVE_DIR"])
    for backend in ("dense", "id"):
        cp1 = torch.load(directory / f"cp1-{backend}.pt", map_location="cpu", weights_only=True)
        cp2 = [
            torch.load(
                directory / f"cp2-{backend}-rank{rank}.pt", map_location="cpu", weights_only=True
            )
            for rank in range(2)
        ]
        assert cp1["weight_sha"] == cp2[0]["weight_sha"] == cp2[1]["weight_sha"]
        for rank in range(2):
            torch.testing.assert_close(cp2[rank]["output"], cp1["output"], atol=0, rtol=0)
            torch.testing.assert_close(
                cp2[rank]["hidden_grad"].float(), cp1["hidden_grad"].float(), atol=1e-2, rtol=1e-2
            )
        summed_weight_grad = cp2[0]["weight_grad"].float() + cp2[1]["weight_grad"].float()
        torch.testing.assert_close(
            summed_weight_grad, cp1["weight_grad"].float(), atol=8e-2, rtol=8e-2
        )
