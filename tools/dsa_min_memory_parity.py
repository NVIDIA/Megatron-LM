#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compare DSA reference math against the min-memory backend on one synthetic layer.

This is a diagnostic harness, not a benchmark.  It prints component-level parity checks in the
order that data flows through sparse DSA, then compares end-to-end forward/loss/grads.  Use it to
find the first tensor that diverges before drawing conclusions from long convergence runs.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _import_dsa_modules() -> None:
    """Import Megatron DSA modules after CUDA preflight has validated the process context."""
    global DSAMinMemoryGQAFn
    global dsa_dense_indexer_loss
    global dsa_min_memory_gqa
    global _project_k_index_block
    global _project_q_index_tile
    global _selected_index_scores_tile
    global _sparse_attention_tile
    global _teacher_scores_tile
    global _topk_index_tile
    global _triton_dispatch_enabled
    global compute_gqa_dsa_indexer_loss
    global fused_qk_topk_naive
    global unfused_grouped_dsa_fn

    from megatron.core.transformer.experimental_attention_variant.dsa import (
        fused_qk_topk_naive as _fused_qk_topk_naive,
    )
    from megatron.core.transformer.experimental_attention_variant.dsa_gqa import (
        compute_gqa_dsa_indexer_loss as _compute_gqa_dsa_indexer_loss,
        unfused_grouped_dsa_fn as _unfused_grouped_dsa_fn,
    )
    from megatron.core.transformer.experimental_attention_variant.dsa_min_memory import (
        DSAMinMemoryGQAFn as _DSAMinMemoryGQAFn,
        dsa_dense_indexer_loss as _dsa_dense_indexer_loss,
        dsa_min_memory_gqa as _dsa_min_memory_gqa,
        _project_k_index_block as _project_k_index_block_imported,
        _project_q_index_tile as _project_q_index_tile_imported,
        _selected_index_scores_tile as _selected_index_scores_tile_imported,
        _sparse_attention_tile as _sparse_attention_tile_imported,
        _teacher_scores_tile as _teacher_scores_tile_imported,
        _topk_index_tile as _topk_index_tile_imported,
        _triton_dispatch_enabled as _triton_dispatch_enabled_imported,
    )

    DSAMinMemoryGQAFn = _DSAMinMemoryGQAFn
    dsa_dense_indexer_loss = _dsa_dense_indexer_loss
    dsa_min_memory_gqa = _dsa_min_memory_gqa
    _project_k_index_block = _project_k_index_block_imported
    _project_q_index_tile = _project_q_index_tile_imported
    _selected_index_scores_tile = _selected_index_scores_tile_imported
    _sparse_attention_tile = _sparse_attention_tile_imported
    _teacher_scores_tile = _teacher_scores_tile_imported
    _topk_index_tile = _topk_index_tile_imported
    _triton_dispatch_enabled = _triton_dispatch_enabled_imported
    compute_gqa_dsa_indexer_loss = _compute_gqa_dsa_indexer_loss
    fused_qk_topk_naive = _fused_qk_topk_naive
    unfused_grouped_dsa_fn = _unfused_grouped_dsa_fn


def _cuda_summary() -> str:
    lines = [
        f"python={sys.executable}",
        f"torch={torch.__version__}",
        f"torch.version.cuda={torch.version.cuda}",
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
        f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH')}",
    ]
    try:
        available = torch.cuda.is_available()
        lines.append(f"torch.cuda.is_available={available}")
        lines.append(f"torch.cuda.device_count={torch.cuda.device_count()}")
        if available:
            lines.append(f"current_device={torch.cuda.current_device()}")
            lines.append(f"device_name={torch.cuda.get_device_name(0)}")
    except Exception as error:
        lines.append(f"cuda_probe_error={error!r}")
    return "\n".join(lines)


def _selected_kv_advanced(
    tensor: torch.Tensor,
    group_idx: int,
    topk_indices: torch.Tensor,
) -> torch.Tensor:
    tensor = tensor[:, :, group_idx, :].permute(1, 0, 2)
    batch_size = topk_indices.size(0)
    batch_index = torch.arange(batch_size, device=topk_indices.device).view(batch_size, 1, 1)
    return tensor[batch_index, topk_indices]


def _selected_kv_gather(
    tensor: torch.Tensor,
    group_idx: int,
    topk_indices: torch.Tensor,
) -> torch.Tensor:
    tensor = tensor[:, :, group_idx, :].permute(1, 0, 2)
    batch_size, query_length, topk = topk_indices.shape
    gather_index = topk_indices[:, :, :, None].expand(
        batch_size, query_length, topk, tensor.size(-1)
    )
    return torch.gather(
        tensor[:, None, :, :].expand(batch_size, query_length, tensor.size(1), tensor.size(2)),
        2,
        gather_index,
    )


def _scatter_add_oracle(
    topk_indices: torch.Tensor,
    selected_grad: torch.Tensor,
    sequence_length: int,
    num_groups: int,
    group_idx: int,
    input_dtype: torch.dtype,
) -> torch.Tensor:
    batch_size, query_length, topk = topk_indices.shape
    head_dim = selected_grad.size(-1)
    grad_group = torch.zeros(
        batch_size, sequence_length, head_dim, device=selected_grad.device, dtype=torch.float32
    )
    scatter_index = topk_indices[:, :, :, None].expand(
        batch_size, query_length, topk, head_dim
    )
    grad_group.scatter_add_(
        1,
        scatter_index.reshape(batch_size, query_length * topk, head_dim),
        selected_grad.to(dtype=input_dtype).float().reshape(batch_size, query_length * topk, head_dim),
    )
    grad = torch.zeros(
        sequence_length,
        batch_size,
        num_groups,
        head_dim,
        device=selected_grad.device,
        dtype=torch.float32,
    )
    grad[:, :, group_idx, :] = grad_group.permute(1, 0, 2)
    return grad


def _tensor_diff_line(name: str, actual: torch.Tensor, expected: torch.Tensor) -> str:
    diff = (actual.float() - expected.float()).abs()
    return (
        f"{name}: actual_dtype={actual.dtype} expected_dtype={expected.dtype} "
        f"max_abs={float(diff.max().item()):.6e} mean_abs={float(diff.mean().item()):.6e}"
    )


def _run_gather_backward_diagnostic(args, device: torch.device) -> int:
    if device.type != "cuda":
        print("Gather backward diagnostic is intended for CUDA.", flush=True)
    torch.manual_seed(args.seed)
    dtype = _dtype(args.dtype, device)
    if dtype == torch.float16:
        print("Using fp16 diagnostic; bf16 is usually the relevant DSA training dtype.", flush=True)

    sequence_length = args.seq_len
    query_length = args.query_block_size
    batch_size = args.batch_size
    num_groups = args.num_query_groups
    head_dim = args.head_dim
    topk = args.topk
    group_idx = 0
    hot_keys = min(sequence_length, max(1, args.gather_diag_hot_keys))
    base_key = torch.randn(
        sequence_length, batch_size, num_groups, head_dim, device=device, dtype=dtype
    )
    topk_indices = torch.randint(
        0, hot_keys, (batch_size, query_length, topk), device=device, dtype=torch.long
    )
    selected_grad = torch.randn(
        batch_size, query_length, topk, head_dim, device=device, dtype=torch.float32
    )

    def run_once(select_fn, key_dtype: torch.dtype) -> torch.Tensor:
        key = base_key.to(dtype=key_dtype).detach().clone().requires_grad_(True)
        selected = select_fn(key, group_idx, topk_indices)
        loss = (selected.float() * selected_grad).sum()
        (grad_key,) = torch.autograd.grad(loss, (key,))
        return grad_key.detach()

    advanced_grad = run_once(_selected_kv_advanced, dtype)
    gather_grad = run_once(_selected_kv_gather, dtype)
    fp32_advanced_grad = run_once(_selected_kv_advanced, torch.float32)
    fp32_gather_grad = run_once(_selected_kv_gather, torch.float32)
    oracle = _scatter_add_oracle(
        topk_indices, selected_grad, sequence_length, num_groups, group_idx, dtype
    )
    oracle_cast = oracle.to(dtype=dtype).float()

    counts = torch.zeros(
        batch_size, sequence_length, device=device, dtype=torch.int32
    ).scatter_add_(
        1,
        topk_indices.reshape(batch_size, query_length * topk),
        torch.ones(batch_size, query_length * topk, device=device, dtype=torch.int32),
    )
    print(
        f"Gather backward diagnostic dtype={dtype} S={sequence_length} Q={query_length} "
        f"B={batch_size} G={num_groups} D={head_dim} topk={topk} hot_keys={hot_keys} "
        f"max_collision_count={int(counts.max().item())}",
        flush=True,
    )
    print(_tensor_diff_line("bf16_or_dtype advanced_vs_gather", advanced_grad, gather_grad), flush=True)
    print(_tensor_diff_line("bf16_or_dtype advanced_vs_fp32_scatter_cast", advanced_grad.float(), oracle_cast), flush=True)
    print(_tensor_diff_line("bf16_or_dtype gather_vs_fp32_scatter_cast", gather_grad.float(), oracle_cast), flush=True)
    print(_tensor_diff_line("fp32 advanced_vs_gather", fp32_advanced_grad, fp32_gather_grad), flush=True)

    bf16_diff = (advanced_grad.float() - gather_grad.float()).abs().max().item()
    fp32_diff = (fp32_advanced_grad.float() - fp32_gather_grad.float()).abs().max().item()
    if dtype in (torch.bfloat16, torch.float16) and bf16_diff > max(fp32_diff * 10, 1e-3):
        print(
            "RESULT: advanced indexing and torch.gather have dtype-sensitive repeated-index "
            "backward accumulation differences in this runtime.",
            flush=True,
        )
    else:
        print(
            "RESULT: this diagnostic did not isolate a dtype-sensitive accumulation difference.",
            flush=True,
        )
    return 0


class _DummyTPGroup:
    def size(self) -> int:
        return 1


class _DummyPGCollection:
    tp = _DummyTPGroup()


@dataclass
class _RuntimePGCollection:
    tp: object


class _SimpleRotary:
    def __init__(self, rotary_dim: int, device: torch.device, rotary_interleaved: bool = False):
        self.inv_freq = 1.0 / (
            10000 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim)
        )
        self.rotary_interleaved = rotary_interleaved
        self.seq_len_interpolation_factor = None


@dataclass
class Case:
    hidden_states: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    linear_q_weight: torch.Tensor
    linear_k_weight: torch.Tensor
    k_norm_weight: torch.Tensor
    k_norm_bias: torch.Tensor
    linear_weights_weight: torch.Tensor


def _causal_mask(seqlen: int, device: torch.device) -> torch.Tensor:
    return torch.triu(
        torch.full((seqlen, seqlen), float("-inf"), dtype=torch.float32, device=device),
        diagonal=1,
    )


def _dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[name]


def _distributed_env_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _configure_distributed_for_dense_mode(device: torch.device, backend: Optional[str]):
    world_size = _distributed_env_world_size()
    if world_size <= 1:
        return device, 0, 1, _DummyPGCollection()

    if device.type == "cuda":
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if device.index is None:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            torch.cuda.set_device(device.index)

    if not torch.distributed.is_initialized():
        if backend is None:
            backend = "nccl" if device.type == "cuda" else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    return device, rank, world_size, _RuntimePGCollection(torch.distributed.group.WORLD)


def _all_gather_concat(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return tensor
    world_size = torch.distributed.get_world_size()
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, tensor.contiguous())
    return torch.cat(gathered, dim=dim)


def _make_leaf(shape: Tuple[int, ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(shape, device=device, dtype=dtype).requires_grad_(True)


def _make_case(args, device: torch.device, dtype: torch.dtype) -> Case:
    torch.manual_seed(args.seed)
    hidden_states = torch.randn(
        args.seq_len, args.batch_size, args.hidden_size, device=device, dtype=dtype
    )
    query = _make_leaf(
        (args.seq_len, args.batch_size, args.num_query_heads, args.head_dim), device, dtype
    )
    key = _make_leaf(
        (args.seq_len, args.batch_size, args.num_query_groups, args.head_dim), device, dtype
    )
    value = _make_leaf(
        (args.seq_len, args.batch_size, args.num_query_groups, args.value_head_dim), device, dtype
    )
    linear_q_weight = _make_leaf(
        (args.indexer_heads * args.indexer_head_dim, args.hidden_size), device, dtype
    )
    linear_k_weight = _make_leaf((args.indexer_head_dim, args.hidden_size), device, dtype)
    k_norm_weight = (1.0 + 0.02 * torch.randn(args.indexer_head_dim, device=device, dtype=dtype))
    k_norm_weight.requires_grad_(True)
    k_norm_bias = (0.02 * torch.randn(args.indexer_head_dim, device=device, dtype=dtype))
    k_norm_bias.requires_grad_(True)
    linear_weights_weight = _make_leaf((args.indexer_heads, args.hidden_size), device, dtype)
    return Case(
        hidden_states,
        query,
        key,
        value,
        linear_q_weight,
        linear_k_weight,
        k_norm_weight,
        k_norm_bias,
        linear_weights_weight,
    )


def _make_dense_local_case(args, device: torch.device, dtype: torch.dtype, rank: int) -> Case:
    torch.manual_seed(args.seed)
    hidden_states = torch.randn(
        args.seq_len, args.batch_size, args.hidden_size, device=device, dtype=dtype
    )
    linear_q_weight = _make_leaf(
        (args.indexer_heads * args.indexer_head_dim, args.hidden_size), device, dtype
    )
    linear_k_weight = _make_leaf((args.indexer_head_dim, args.hidden_size), device, dtype)
    k_norm_weight = (1.0 + 0.02 * torch.randn(args.indexer_head_dim, device=device, dtype=dtype))
    k_norm_weight.requires_grad_(True)
    k_norm_bias = (0.02 * torch.randn(args.indexer_head_dim, device=device, dtype=dtype))
    k_norm_bias.requires_grad_(True)
    linear_weights_weight = _make_leaf((args.indexer_heads, args.hidden_size), device, dtype)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 1009 + rank)
    query = torch.randn(
        args.seq_len,
        args.batch_size,
        args.num_query_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    ).requires_grad_(True)
    key = torch.randn(
        args.seq_len,
        args.batch_size,
        args.num_query_groups,
        args.head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    ).requires_grad_(True)
    value = torch.randn(
        args.seq_len,
        args.batch_size,
        args.num_query_groups,
        args.value_head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    ).requires_grad_(True)

    return Case(
        hidden_states,
        query,
        key,
        value,
        linear_q_weight,
        linear_k_weight,
        k_norm_weight,
        k_norm_bias,
        linear_weights_weight,
    )


def _clone_case(case: Case) -> Case:
    values = {}
    for name, tensor in case.__dict__.items():
        clone = tensor.detach().clone()
        if tensor.requires_grad:
            clone.requires_grad_(True)
        values[name] = clone
    return Case(**values)


def _case_tensors(case: Case) -> Iterable[Tuple[str, torch.Tensor]]:
    for name in (
        "query",
        "key",
        "value",
        "linear_q_weight",
        "linear_k_weight",
        "k_norm_weight",
        "k_norm_bias",
        "linear_weights_weight",
    ):
        yield name, getattr(case, name)


class _WeightOnlyModule:
    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, eps: float = 1e-5):
        self.weight = weight
        self.bias = bias
        self.eps = eps


def _indexer_from_case(case: Case, args, pg_collection):
    return type(
        "_DenseWarmupIndexer",
        (),
        {
            "linear_q": _WeightOnlyModule(case.linear_q_weight),
            "linear_k": _WeightOnlyModule(case.linear_k_weight),
            "k_norm": _WeightOnlyModule(case.k_norm_weight, case.k_norm_bias, args.layernorm_eps),
            "linear_weights_proj": _WeightOnlyModule(case.linear_weights_weight),
            "index_n_heads": args.indexer_heads,
            "index_head_dim": args.indexer_head_dim,
            "index_topk": args.topk,
            "index_rotary_dim": args.indexer_rotary_dim,
            "rotary_pos_emb": (
                _SimpleRotary(
                    args.indexer_rotary_dim,
                    case.hidden_states.device,
                    args.rotary_interleaved,
                )
                if args.indexer_rotary_dim > 0
                else None
            ),
            "pg_collection": pg_collection,
            "config": type(
                "_DenseWarmupIndexerConfig",
                (),
                {
                    "layernorm_epsilon": args.layernorm_eps,
                    "dsa_indexer_use_hadamard": args.hadamard,
                    "rotary_interleaved": args.rotary_interleaved,
                },
            )(),
        },
    )()


def _project_indexer(case: Case, args, rotary_pos_emb):
    q_index, weights = _project_q_index_tile(
        case.hidden_states,
        0,
        args.seq_len,
        case.linear_q_weight,
        case.linear_weights_weight,
        args.indexer_heads,
        args.indexer_head_dim,
        args.indexer_rotary_dim,
        rotary_pos_emb,
        args.rotary_interleaved,
        args.indexer_rotary_dim > 0,
        args.hadamard,
    )
    k_index = _project_k_index_block(
        case.hidden_states,
        0,
        args.seq_len,
        case.linear_k_weight,
        case.k_norm_weight,
        case.k_norm_bias,
        True,
        args.layernorm_eps,
        args.indexer_head_dim,
        args.indexer_rotary_dim,
        rotary_pos_emb,
        args.rotary_interleaved,
        args.indexer_rotary_dim > 0,
        args.hadamard,
    )
    return q_index, k_index, weights


def _reference_teacher_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    sq, batch_size, num_query_heads, head_dim = query.shape
    sk = key.size(0)
    num_query_groups = key.size(2)
    repeat_factor = num_query_heads // num_query_groups
    repeated_key = key.repeat_interleave(repeat_factor, dim=2)
    query_bhsd = query.permute(1, 2, 0, 3)
    key_bhkd = repeated_key.permute(1, 2, 0, 3)
    gather_index = topk_indices[:, None, :, :, None].expand(
        batch_size, num_query_heads, sq, topk_indices.size(-1), head_dim
    )
    selected_key = torch.gather(
        key_bhkd[:, :, None, :, :].expand(batch_size, num_query_heads, sq, sk, head_dim),
        3,
        gather_index,
    )
    scores = torch.einsum("bnsh,bnskh->bnsk", query_bhsd.float(), selected_key.float())
    scores = scores * softmax_scale
    query_positions = torch.arange(sq, device=topk_indices.device, dtype=topk_indices.dtype)
    invalid = topk_indices > query_positions.view(1, sq, 1)
    scores = scores.masked_fill(invalid[:, None, :, :], float("-inf"))
    probs = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32)
    teacher = probs.sum(dim=1)
    return teacher / teacher.sum(dim=-1, keepdim=True)


def _reference_run(case: Case, args, rotary_pos_emb, topk_override: Optional[torch.Tensor] = None):
    q_index, k_index, weights = _project_indexer(case, args, rotary_pos_emb)
    index_scores, natural_topk_indices = fused_qk_topk_naive(
        q_index, k_index, weights, args.topk, _causal_mask(args.seq_len, case.query.device)
    )
    topk_indices = (
        natural_topk_indices
        if topk_override is None
        else topk_override.to(device=natural_topk_indices.device)
    )
    output = unfused_grouped_dsa_fn(
        case.query,
        case.key,
        case.value,
        topk_indices,
        args.head_dim**-0.5,
        use_gather=True,
    )
    selected_scores = index_scores.gather(-1, topk_indices)
    loss = compute_gqa_dsa_indexer_loss(
        index_scores=None,
        topk_indices=topk_indices,
        query=case.query.detach(),
        key=case.key.detach(),
        softmax_scale=args.head_dim**-0.5,
        loss_coeff=args.loss_coeff,
        sparse_loss=True,
        pg_collection=_DummyPGCollection(),
        sparse_loss_use_topk_only=True,
        query_chunk_size=args.query_block_size,
        selected_index_scores=selected_scores,
    )
    (output.float().sum() + loss.float()).backward()
    return {
        "q_index": q_index.detach(),
        "k_index": k_index.detach(),
        "weights": weights.detach(),
        "natural_topk_indices": natural_topk_indices.detach(),
        "topk_indices": topk_indices.detach(),
        "index_scores": index_scores.detach(),
        "selected_scores": selected_scores.detach(),
        "teacher_scores": _reference_teacher_scores(
            case.query.detach(), case.key.detach(), topk_indices.detach(), args.head_dim**-0.5
        ),
        "output": output.detach(),
        "loss": loss.detach(),
        "grads": {name: tensor.grad.detach().clone() for name, tensor in _case_tensors(case)},
    }


def _min_memory_run(case: Case, args, rotary_pos_emb, use_triton: bool):
    output, loss = DSAMinMemoryGQAFn.apply(
        case.query,
        case.key,
        case.value,
        case.hidden_states,
        case.linear_q_weight,
        case.linear_k_weight,
        case.k_norm_weight,
        case.k_norm_bias,
        case.linear_weights_weight,
        True,
        args.layernorm_eps,
        args.indexer_heads,
        args.indexer_head_dim,
        args.topk,
        args.indexer_rotary_dim,
        rotary_pos_emb,
        args.indexer_rotary_dim > 0,
        args.hadamard,
        args.head_dim**-0.5,
        args.loss_coeff,
        args.query_block_size,
        args.key_block_size,
        _DummyPGCollection(),
        args.rotary_interleaved,
        False,
        0,
        "",
        args.cache_routing,
        args.cache_indexer_k,
        args.cache_selected_scores,
        use_triton,
    )
    (output.float().sum() + loss.float()).backward()
    return {
        "output": output.detach(),
        "loss": loss.detach(),
        "grads": {name: tensor.grad.detach().clone() for name, tensor in _case_tensors(case)},
    }


def _min_memory_components(case: Case, args, rotary_pos_emb, use_triton: bool):
    key_block = args.key_block_size
    with _triton_dispatch_enabled(use_triton), torch.no_grad():
        topk_scores, topk_indices, q_index, weights = _topk_index_tile(
            case.hidden_states,
            0,
            args.seq_len,
            case.linear_q_weight,
            case.linear_k_weight,
            case.k_norm_weight,
            case.k_norm_bias,
            True,
            case.linear_weights_weight,
            args.layernorm_eps,
            args.indexer_heads,
            args.indexer_head_dim,
            args.topk,
            args.indexer_rotary_dim,
            rotary_pos_emb,
            args.rotary_interleaved,
            args.indexer_rotary_dim > 0,
            args.hadamard,
            key_block,
        )
        selected_scores = _selected_index_scores_tile(
            case.hidden_states,
            0,
            args.seq_len,
            topk_indices,
            q_index,
            weights,
            case.linear_k_weight,
            case.k_norm_weight,
            case.k_norm_bias,
            True,
            args.layernorm_eps,
            args.indexer_head_dim,
            args.indexer_rotary_dim,
            rotary_pos_emb,
            args.rotary_interleaved,
            args.indexer_rotary_dim > 0,
            args.hadamard,
        )
        teacher_scores = _teacher_scores_tile(
            case.query,
            case.key,
            topk_indices,
            args.head_dim**-0.5,
            0,
            _DummyPGCollection(),
        )
        sparse_output = _sparse_attention_tile(
            case.query,
            case.key,
            case.value,
            topk_indices,
            args.head_dim**-0.5,
            0,
        )
    return {
        "q_index": q_index,
        "weights": weights,
        "topk_scores": topk_scores,
        "topk_indices": topk_indices,
        "selected_scores": selected_scores,
        "teacher_scores": teacher_scores,
        "sparse_output": sparse_output,
    }


def _reference_sparse_attention_grads(
    case: Case,
    args,
    topk_indices: torch.Tensor,
    grad_output: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    query = case.query.detach().clone().requires_grad_(True)
    key = case.key.detach().clone().requires_grad_(True)
    value = case.value.detach().clone().requires_grad_(True)
    output = unfused_grouped_dsa_fn(
        query,
        key,
        value,
        topk_indices,
        args.head_dim**-0.5,
        use_gather=True,
    ).view(args.seq_len, args.batch_size, args.num_query_heads, args.value_head_dim)
    grads = torch.autograd.grad(output, (query, key, value), grad_outputs=grad_output)
    return {"attn_grad_query": grads[0], "attn_grad_key": grads[1], "attn_grad_value": grads[2]}


def _min_memory_sparse_attention_grads(
    case: Case,
    args,
    topk_indices: torch.Tensor,
    grad_output: torch.Tensor,
    use_triton: bool,
) -> Dict[str, torch.Tensor]:
    grad_query = torch.zeros(case.query.shape, device=case.query.device, dtype=torch.float32)
    grad_key = torch.zeros(case.key.shape, device=case.key.device, dtype=torch.float32)
    grad_value = torch.zeros(case.value.shape, device=case.value.device, dtype=torch.float32)

    with _triton_dispatch_enabled(use_triton):
        for q_start in range(0, args.seq_len, args.query_block_size):
            q_end = min(q_start + args.query_block_size, args.seq_len)
            query_tile = case.query[q_start:q_end].detach().clone().requires_grad_(True)
            key_leaf = case.key.detach().clone().requires_grad_(True)
            value_leaf = case.value.detach().clone().requires_grad_(True)
            output_tile = _sparse_attention_tile(
                query_tile,
                key_leaf,
                value_leaf,
                topk_indices[:, q_start:q_end, :],
                args.head_dim**-0.5,
                q_start,
            )
            grads = torch.autograd.grad(
                output_tile,
                (query_tile, key_leaf, value_leaf),
                grad_outputs=grad_output[q_start:q_end],
            )
            grad_query[q_start:q_end].add_(grads[0].float())
            grad_key.add_(grads[1].float())
            grad_value.add_(grads[2].float())

    return {
        "attn_grad_query": grad_query,
        "attn_grad_key": grad_key,
        "attn_grad_value": grad_value,
    }


def _dense_reference_loss_and_grads(
    case: Case,
    args,
    rotary_pos_emb,
    full_query: torch.Tensor,
    full_key: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    q_index, k_index, weights = _project_indexer(case, args, rotary_pos_emb)
    index_scores, topk_indices = fused_qk_topk_naive(
        q_index, k_index, weights, args.topk, _causal_mask(args.seq_len, case.query.device)
    )
    loss = compute_gqa_dsa_indexer_loss(
        index_scores=index_scores,
        topk_indices=topk_indices,
        query=full_query.detach(),
        key=full_key.detach(),
        softmax_scale=args.head_dim**-0.5,
        loss_coeff=args.loss_coeff,
        sparse_loss=False,
        pg_collection=_DummyPGCollection(),
    )
    params = (
        case.linear_q_weight,
        case.linear_k_weight,
        case.k_norm_weight,
        case.k_norm_bias,
        case.linear_weights_weight,
    )
    grads = torch.autograd.grad(loss, params)
    return loss.detach(), {
        "linear_q_weight": grads[0].detach(),
        "linear_k_weight": grads[1].detach(),
        "k_norm_weight": grads[2].detach(),
        "k_norm_bias": grads[3].detach(),
        "linear_weights_weight": grads[4].detach(),
    }


def _dense_min_memory_loss_and_grads(
    case: Case,
    args,
    pg_collection,
    use_triton: bool,
) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]], Dict[str, Optional[torch.Tensor]]]:
    indexer = _indexer_from_case(case, args, pg_collection)
    loss = dsa_dense_indexer_loss(
        query=case.query,
        key=case.key,
        hidden_states=case.hidden_states.requires_grad_(True),
        indexer=indexer,
        softmax_scale=args.head_dim**-0.5,
        loss_coeff=args.loss_coeff,
        use_indexer_rope=args.indexer_rotary_dim > 0,
        query_chunk_size=args.query_block_size,
        key_chunk_size=args.key_block_size,
        use_triton=use_triton,
    )
    grad_inputs = (
        case.linear_q_weight,
        case.linear_k_weight,
        case.k_norm_weight,
        case.k_norm_bias,
        case.linear_weights_weight,
        case.query,
        case.key,
        case.hidden_states,
    )
    grads = torch.autograd.grad(loss, grad_inputs, allow_unused=True)
    param_grads = {
        "linear_q_weight": grads[0],
        "linear_k_weight": grads[1],
        "k_norm_weight": grads[2],
        "k_norm_bias": grads[3],
        "linear_weights_weight": grads[4],
    }
    input_grads = {
        "query": grads[5],
        "key": grads[6],
        "hidden_states": grads[7],
    }
    return loss.detach(), param_grads, input_grads


def _full_support_indices(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    return (
        torch.arange(seq_len, device=device, dtype=torch.long)
        .view(1, 1, seq_len)
        .expand(batch_size, seq_len, seq_len)
        .contiguous()
    )


def _sparse_full_topk_run(case: Case, args, rotary_pos_emb, pg_collection):
    q_index, k_index, weights = _project_indexer(case, args, rotary_pos_emb)
    index_scores, topk_indices = fused_qk_topk_naive(
        q_index, k_index, weights, args.seq_len, _causal_mask(args.seq_len, case.query.device)
    )
    output = unfused_grouped_dsa_fn(
        case.query,
        case.key,
        case.value,
        topk_indices,
        args.head_dim**-0.5,
        query_chunk_size=args.query_block_size,
        use_gather=True,
    )
    selected_scores = index_scores.gather(-1, topk_indices)
    loss = compute_gqa_dsa_indexer_loss(
        index_scores=None,
        topk_indices=topk_indices,
        query=case.query.detach(),
        key=case.key.detach(),
        softmax_scale=args.head_dim**-0.5,
        loss_coeff=args.loss_coeff,
        sparse_loss=True,
        pg_collection=pg_collection,
        sparse_loss_use_topk_only=True,
        query_chunk_size=args.query_block_size,
        selected_index_scores=selected_scores,
    )
    grad_inputs = (
        case.query,
        case.key,
        case.value,
        case.linear_q_weight,
        case.linear_k_weight,
        case.k_norm_weight,
        case.k_norm_bias,
        case.linear_weights_weight,
    )
    grads = torch.autograd.grad(output.float().sum() + loss.float(), grad_inputs)
    return {
        "topk_indices": topk_indices.detach(),
        "output": output.detach(),
        "loss": loss.detach(),
        "grads": {
            "query": grads[0].detach(),
            "key": grads[1].detach(),
            "value": grads[2].detach(),
            "linear_q_weight": grads[3].detach(),
            "linear_k_weight": grads[4].detach(),
            "k_norm_weight": grads[5].detach(),
            "k_norm_bias": grads[6].detach(),
            "linear_weights_weight": grads[7].detach(),
        },
    }


def _dense_full_support_run(case: Case, args, pg_collection, use_triton: bool):
    full_support = _full_support_indices(args.batch_size, args.seq_len, case.query.device)
    output = unfused_grouped_dsa_fn(
        case.query,
        case.key,
        case.value,
        full_support,
        args.head_dim**-0.5,
        use_gather=False,
    )
    indexer = _indexer_from_case(case, args, pg_collection)
    loss = dsa_dense_indexer_loss(
        query=case.query,
        key=case.key,
        hidden_states=case.hidden_states.requires_grad_(True),
        indexer=indexer,
        softmax_scale=args.head_dim**-0.5,
        loss_coeff=args.loss_coeff,
        use_indexer_rope=args.indexer_rotary_dim > 0,
        query_chunk_size=args.query_block_size,
        key_chunk_size=args.key_block_size,
        use_triton=use_triton,
    )
    grad_inputs = (
        case.query,
        case.key,
        case.value,
        case.linear_q_weight,
        case.linear_k_weight,
        case.k_norm_weight,
        case.k_norm_bias,
        case.linear_weights_weight,
        case.hidden_states,
    )
    grads = torch.autograd.grad(output.float().sum() + loss.float(), grad_inputs, allow_unused=True)
    return {
        "output": output.detach(),
        "loss": loss.detach(),
        "grads": {
            "query": grads[0].detach() if grads[0] is not None else None,
            "key": grads[1].detach() if grads[1] is not None else None,
            "value": grads[2].detach() if grads[2] is not None else None,
            "linear_q_weight": grads[3].detach() if grads[3] is not None else None,
            "linear_k_weight": grads[4].detach() if grads[4] is not None else None,
            "k_norm_weight": grads[5].detach() if grads[5] is not None else None,
            "k_norm_bias": grads[6].detach() if grads[6] is not None else None,
            "linear_weights_weight": grads[7].detach() if grads[7] is not None else None,
            "hidden_states": grads[8].detach() if grads[8] is not None else None,
        },
    }


def _reference_sparse_fwd_dense_loss_run(
    case: Case,
    args,
    rotary_pos_emb,
    topk_indices: torch.Tensor,
    full_query: torch.Tensor,
    full_key: torch.Tensor,
):
    q_index, k_index, weights = _project_indexer(case, args, rotary_pos_emb)
    index_scores, natural_topk_indices = fused_qk_topk_naive(
        q_index, k_index, weights, args.topk, _causal_mask(args.seq_len, case.query.device)
    )
    output = unfused_grouped_dsa_fn(
        case.query,
        case.key,
        case.value,
        topk_indices,
        args.head_dim**-0.5,
        query_chunk_size=args.query_block_size,
        use_gather=True,
    )
    loss = compute_gqa_dsa_indexer_loss(
        index_scores=index_scores,
        topk_indices=natural_topk_indices,
        query=full_query.detach(),
        key=full_key.detach(),
        softmax_scale=args.head_dim**-0.5,
        loss_coeff=args.loss_coeff,
        sparse_loss=False,
        pg_collection=_DummyPGCollection(),
    )
    grad_inputs = (
        case.query,
        case.key,
        case.value,
        case.linear_q_weight,
        case.linear_k_weight,
        case.k_norm_weight,
        case.k_norm_bias,
        case.linear_weights_weight,
    )
    grads = torch.autograd.grad(output.float().sum() + loss.float(), grad_inputs)
    return {
        "natural_topk_indices": natural_topk_indices.detach(),
        "output": output.detach(),
        "loss": loss.detach(),
        "grads": {
            "query": grads[0].detach(),
            "key": grads[1].detach(),
            "value": grads[2].detach(),
            "linear_q_weight": grads[3].detach(),
            "linear_k_weight": grads[4].detach(),
            "k_norm_weight": grads[5].detach(),
            "k_norm_bias": grads[6].detach(),
            "linear_weights_weight": grads[7].detach(),
        },
    }


def _min_memory_sparse_fwd_dense_loss_run(
    case: Case,
    args,
    pg_collection,
    use_triton: bool,
):
    indexer = _indexer_from_case(case, args, pg_collection)
    output, sparse_loss = dsa_min_memory_gqa(
        query=case.query,
        key=case.key,
        value=case.value,
        hidden_states=case.hidden_states.detach(),
        indexer=indexer,
        softmax_scale=args.head_dim**-0.5,
        loss_coeff=0.0,
        use_indexer_rope=args.indexer_rotary_dim > 0,
        query_chunk_size=args.query_block_size,
        key_chunk_size=args.key_block_size,
        cache_routing=args.cache_routing,
        cache_indexer_k=args.cache_indexer_k,
        cache_selected_scores=False,
        use_triton=use_triton,
    )
    hidden_for_loss = case.hidden_states.detach().requires_grad_(True)
    loss = dsa_dense_indexer_loss(
        query=case.query.detach(),
        key=case.key.detach(),
        hidden_states=hidden_for_loss,
        indexer=indexer,
        softmax_scale=args.head_dim**-0.5,
        loss_coeff=args.loss_coeff,
        use_indexer_rope=args.indexer_rotary_dim > 0,
        query_chunk_size=args.query_block_size,
        key_chunk_size=args.key_block_size,
        use_triton=use_triton,
    )
    grad_inputs = (
        case.query,
        case.key,
        case.value,
        case.linear_q_weight,
        case.linear_k_weight,
        case.k_norm_weight,
        case.k_norm_bias,
        case.linear_weights_weight,
        hidden_for_loss,
    )
    grads = torch.autograd.grad(output.float().sum() + loss.float(), grad_inputs, allow_unused=True)
    return {
        "output": output.detach(),
        "sparse_loss": sparse_loss.detach(),
        "loss": loss.detach(),
        "grads": {
            "query": grads[0].detach() if grads[0] is not None else None,
            "key": grads[1].detach() if grads[1] is not None else None,
            "value": grads[2].detach() if grads[2] is not None else None,
            "linear_q_weight": grads[3].detach() if grads[3] is not None else None,
            "linear_k_weight": grads[4].detach() if grads[4] is not None else None,
            "k_norm_weight": grads[5].detach() if grads[5] is not None else None,
            "k_norm_bias": grads[6].detach() if grads[6] is not None else None,
            "linear_weights_weight": grads[7].detach() if grads[7] is not None else None,
            "hidden_states": grads[8].detach() if grads[8] is not None else None,
        },
    }


def _check_dtype(name: str, actual: torch.dtype, expected: torch.dtype, fail_fast: bool) -> bool:
    ok = actual == expected
    status = "OK " if ok else "BAD"
    print(f"{status} {name:<36} actual={actual} expected={expected}", flush=True)
    if fail_fast and not ok:
        raise SystemExit(1)
    return ok


def _check_no_grad(name: str, grad: Optional[torch.Tensor], fail_fast: bool) -> bool:
    ok = grad is None or bool((grad.float() == 0).all().item())
    status = "OK " if ok else "BAD"
    if grad is None:
        detail = "grad=None"
    else:
        detail = f"grad_dtype={grad.dtype} max_abs={float(grad.float().abs().max().item()):.6e}"
    print(f"{status} no_indexer_input_grad_{name:<12} {detail}", flush=True)
    if fail_fast and not ok:
        raise SystemExit(1)
    return ok


def _run_dense_warmup_parity(args, device: torch.device, dtype: torch.dtype) -> int:
    use_triton = args.backend == "triton-min-memory"
    device, rank, world_size, pg_collection = _configure_distributed_for_dense_mode(
        device, args.distributed_backend
    )
    rotary_pos_emb = (
        _SimpleRotary(args.indexer_rotary_dim, device, args.rotary_interleaved)
        if args.indexer_rotary_dim > 0
        else None
    )
    atol = args.atol if args.atol is not None else (4e-2 if dtype != torch.float32 else 3e-4)
    rtol = args.rtol if args.rtol is not None else (4e-2 if dtype != torch.float32 else 3e-4)

    print(
        f"Dense warmup parity backend={args.backend} rank={rank}/{world_size} device={device} "
        f"dtype={dtype} local_Hq={args.num_query_heads} local_G={args.num_query_groups} "
        f"global_Hq={args.num_query_heads * world_size} "
        f"global_G={args.num_query_groups * world_size} S={args.seq_len} "
        f"QBLOCK={args.query_block_size} KBLOCK={args.key_block_size} "
        f"hadamard={args.hadamard} rotary_dim={args.indexer_rotary_dim} "
        f"rotary_interleaved={args.rotary_interleaved}",
        flush=True,
    )

    base = _make_dense_local_case(args, device, dtype, rank)
    min_case = _clone_case(base)
    ref_case = _clone_case(base)

    full_query = _all_gather_concat(base.query.detach(), dim=2)
    full_key = _all_gather_concat(base.key.detach(), dim=2)
    ref_loss, ref_grads = _dense_reference_loss_and_grads(
        ref_case, args, rotary_pos_emb, full_query, full_key
    )
    min_loss, min_grads, input_grads = _dense_min_memory_loss_and_grads(
        min_case, args, pg_collection, use_triton
    )

    with _triton_dispatch_enabled(use_triton), torch.no_grad():
        q_index, k_index, weights = _project_indexer(min_case, args, rotary_pos_emb)

    failures = 0
    failures += not _check_dtype("dense_q_index_dtype", q_index.dtype, dtype, args.fail_fast)
    failures += not _check_dtype("dense_k_index_dtype", k_index.dtype, dtype, args.fail_fast)
    failures += not _check_dtype("dense_weights_dtype", weights.dtype, dtype, args.fail_fast)
    failures += not _check_dtype(
        "dense_indexer_loss_dtype", min_loss.dtype, torch.float32, args.fail_fast
    )
    failures += not _check_tensor(
        "dense_indexer_loss",
        min_loss,
        ref_loss,
        atol,
        rtol,
        args.fail_fast,
    )
    for name, expected in ref_grads.items():
        actual = min_grads[name]
        if actual is None:
            print(f"BAD dense_grad_{name:<24} actual=None", flush=True)
            failures += 1
            if args.fail_fast:
                raise SystemExit(1)
            continue
        print(
            f"INFO dense_grad_{name:<24} actual_dtype={actual.dtype} "
            f"reference_dtype={expected.dtype}",
            flush=True,
        )
        failures += not _check_tensor(
            f"dense_grad_{name}",
            actual,
            expected,
            atol,
            rtol,
            args.fail_fast,
        )

    for name, grad in input_grads.items():
        failures += not _check_no_grad(name, grad, args.fail_fast)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        fail_tensor = torch.tensor([failures], device=device, dtype=torch.int32)
        torch.distributed.all_reduce(fail_tensor, op=torch.distributed.ReduceOp.SUM)
        failures = int(fail_tensor.item())

    if failures:
        print(f"\nFAIL: {failures} dense warmup parity checks failed.", flush=True)
        return 1
    print("\nPASS: all dense warmup parity checks passed.", flush=True)
    return 0


def _run_dense_vs_full_topk_parity(args, device: torch.device, dtype: torch.dtype) -> int:
    use_triton = args.backend == "triton-min-memory"
    device, rank, world_size, pg_collection = _configure_distributed_for_dense_mode(
        device, args.distributed_backend
    )
    rotary_pos_emb = (
        _SimpleRotary(args.indexer_rotary_dim, device, args.rotary_interleaved)
        if args.indexer_rotary_dim > 0
        else None
    )
    atol = args.atol if args.atol is not None else (5e-2 if dtype != torch.float32 else 5e-4)
    rtol = args.rtol if args.rtol is not None else (5e-2 if dtype != torch.float32 else 5e-4)

    print(
        f"Dense-vs-full-topk parity backend={args.backend} rank={rank}/{world_size} "
        f"device={device} dtype={dtype} local_Hq={args.num_query_heads} "
        f"local_G={args.num_query_groups} global_Hq={args.num_query_heads * world_size} "
        f"global_G={args.num_query_groups * world_size} S={args.seq_len} "
        f"full_topk={args.seq_len} QBLOCK={args.query_block_size} "
        f"KBLOCK={args.key_block_size} hadamard={args.hadamard} "
        f"rotary_dim={args.indexer_rotary_dim} rotary_interleaved={args.rotary_interleaved}",
        flush=True,
    )

    base = _make_dense_local_case(args, device, dtype, rank)
    sparse_case = _clone_case(base)
    dense_case = _clone_case(base)

    sparse = _sparse_full_topk_run(sparse_case, args, rotary_pos_emb, pg_collection)
    dense = _dense_full_support_run(dense_case, args, pg_collection, use_triton)

    failures = 0
    failures += not _check_indices(
        "full_topk_support",
        sparse["topk_indices"],
        _full_support_indices(args.batch_size, args.seq_len, device),
        args.fail_fast,
    )
    failures += not _check_tensor(
        "full_topk_sparse_vs_dense_output",
        sparse["output"],
        dense["output"],
        atol,
        rtol,
        args.fail_fast,
    )
    failures += not _check_tensor(
        "full_topk_sparse_vs_dense_loss",
        sparse["loss"],
        dense["loss"],
        atol,
        rtol,
        args.fail_fast,
    )

    for name, sparse_grad in sparse["grads"].items():
        dense_grad = dense["grads"].get(name)
        if dense_grad is None:
            print(f"BAD full_topk_grad_{name:<24} dense_grad=None", flush=True)
            failures += 1
            if args.fail_fast:
                raise SystemExit(1)
            continue
        print(
            f"INFO full_topk_grad_{name:<24} sparse_dtype={sparse_grad.dtype} "
            f"dense_dtype={dense_grad.dtype}",
            flush=True,
        )
        failures += not _check_tensor(
            f"full_topk_grad_{name}",
            sparse_grad,
            dense_grad,
            atol,
            rtol,
            args.fail_fast,
        )
    failures += not _check_no_grad(
        "hidden_states",
        dense["grads"].get("hidden_states"),
        args.fail_fast,
    )

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        fail_tensor = torch.tensor([failures], device=device, dtype=torch.int32)
        torch.distributed.all_reduce(fail_tensor, op=torch.distributed.ReduceOp.SUM)
        failures = int(fail_tensor.item())

    if failures:
        print(f"\nFAIL: {failures} dense-vs-full-topk parity checks failed.", flush=True)
        return 1
    print("\nPASS: all dense-vs-full-topk parity checks passed.", flush=True)
    return 0


def _run_sparse_fwd_dense_loss_parity(args, device: torch.device, dtype: torch.dtype) -> int:
    if args.cache_selected_scores:
        raise SystemExit(
            "--cache-selected-scores is invalid for sparse-forward dense-loss mode; "
            "there is no selected-score sparse loss to cache."
        )
    use_triton = args.backend == "triton-min-memory"
    device, rank, world_size, pg_collection = _configure_distributed_for_dense_mode(
        device, args.distributed_backend
    )
    rotary_pos_emb = (
        _SimpleRotary(args.indexer_rotary_dim, device, args.rotary_interleaved)
        if args.indexer_rotary_dim > 0
        else None
    )
    atol = args.atol if args.atol is not None else (5e-2 if dtype != torch.float32 else 5e-4)
    rtol = args.rtol if args.rtol is not None else (5e-2 if dtype != torch.float32 else 5e-4)

    print(
        f"Sparse-forward dense-loss parity backend={args.backend} rank={rank}/{world_size} "
        f"device={device} dtype={dtype} local_Hq={args.num_query_heads} "
        f"local_G={args.num_query_groups} global_Hq={args.num_query_heads * world_size} "
        f"global_G={args.num_query_groups * world_size} S={args.seq_len} topk={args.topk} "
        f"QBLOCK={args.query_block_size} KBLOCK={args.key_block_size} "
        f"cache_routing={args.cache_routing} cache_indexer_k={args.cache_indexer_k} "
        f"hadamard={args.hadamard} rotary_dim={args.indexer_rotary_dim} "
        f"rotary_interleaved={args.rotary_interleaved}",
        flush=True,
    )

    base = _make_dense_local_case(args, device, dtype, rank)
    comp_case = _clone_case(base)
    ref_case = _clone_case(base)
    min_case = _clone_case(base)

    components = _min_memory_components(comp_case, args, rotary_pos_emb, use_triton)
    full_query = _all_gather_concat(base.query.detach(), dim=2)
    full_key = _all_gather_concat(base.key.detach(), dim=2)
    reference = _reference_sparse_fwd_dense_loss_run(
        ref_case,
        args,
        rotary_pos_emb,
        components["topk_indices"],
        full_query,
        full_key,
    )
    min_memory = _min_memory_sparse_fwd_dense_loss_run(
        min_case,
        args,
        pg_collection,
        use_triton,
    )

    failures = 0
    failures += not _check_causal_support_indices(
        "sparse_dense_loss_topk_support",
        components["topk_indices"],
        reference["natural_topk_indices"],
        args.fail_fast,
    )
    failures += not _check_tensor(
        "sparse_dense_loss_forward_output",
        min_memory["output"],
        reference["output"],
        atol,
        rtol,
        args.fail_fast,
    )
    failures += not _check_tensor(
        "sparse_dense_loss_internal_sparse_loss_zero",
        min_memory["sparse_loss"],
        torch.zeros_like(min_memory["sparse_loss"]),
        0.0,
        0.0,
        args.fail_fast,
    )
    failures += not _check_tensor(
        "sparse_dense_loss_indexer_loss",
        min_memory["loss"],
        reference["loss"],
        atol,
        rtol,
        args.fail_fast,
    )

    for name, expected in reference["grads"].items():
        actual = min_memory["grads"].get(name)
        if actual is None:
            print(f"BAD sparse_dense_loss_grad_{name:<18} actual=None", flush=True)
            failures += 1
            if args.fail_fast:
                raise SystemExit(1)
            continue
        print(
            f"INFO sparse_dense_loss_grad_{name:<18} actual_dtype={actual.dtype} "
            f"reference_dtype={expected.dtype}",
            flush=True,
        )
        failures += not _check_tensor(
            f"sparse_dense_loss_grad_{name}",
            actual,
            expected,
            atol,
            rtol,
            args.fail_fast,
        )
    failures += not _check_no_grad(
        "hidden_states",
        min_memory["grads"].get("hidden_states"),
        args.fail_fast,
    )

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        fail_tensor = torch.tensor([failures], device=device, dtype=torch.int32)
        torch.distributed.all_reduce(fail_tensor, op=torch.distributed.ReduceOp.SUM)
        failures = int(fail_tensor.item())

    if failures:
        print(f"\nFAIL: {failures} sparse-forward dense-loss parity checks failed.", flush=True)
        return 1
    print("\nPASS: all sparse-forward dense-loss parity checks passed.", flush=True)
    return 0


def _max_stats(actual: torch.Tensor, expected: torch.Tensor) -> Tuple[float, float]:
    actual_f = actual.float()
    expected_f = expected.float()
    same_nonfinite = (
        ~torch.isfinite(actual_f)
        & ~torch.isfinite(expected_f)
        & (torch.signbit(actual_f) == torch.signbit(expected_f))
    )
    diff = (actual_f - expected_f).abs()
    diff = torch.where(same_nonfinite, torch.zeros_like(diff), diff)
    return float(diff.max().item()), float(diff.mean().item())


def _check_tensor(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    rtol: float,
    fail_fast: bool,
) -> bool:
    actual = actual.detach()
    expected = expected.detach()
    ok = torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol, equal_nan=True)
    max_abs, mean_abs = _max_stats(actual, expected)
    status = "OK " if ok else "BAD"
    print(
        f"{status} {name:<36} shape={tuple(actual.shape)!s:<18} "
        f"max_abs={max_abs:.6e} mean_abs={mean_abs:.6e}",
        flush=True,
    )
    if not ok and actual.shape == expected.shape:
        diff = (actual.float() - expected.float()).abs()
        flat_idx = int(torch.nan_to_num(diff, nan=float("inf")).argmax().item())
        index = tuple(int(v) for v in torch.unravel_index(torch.tensor(flat_idx), diff.shape))
        print(
            f"    max mismatch at {index}: actual={actual[index].item()} "
            f"expected={expected[index].item()}",
            flush=True,
        )
    if fail_fast and not ok:
        raise SystemExit(1)
    return ok


def _check_indices(name: str, actual: torch.Tensor, expected: torch.Tensor, fail_fast: bool) -> bool:
    actual_sorted = actual.sort(dim=-1).values
    expected_sorted = expected.sort(dim=-1).values
    ok = torch.equal(actual_sorted, expected_sorted)
    status = "OK " if ok else "BAD"
    print(f"{status} {name:<36} shape={tuple(actual.shape)}", flush=True)
    if not ok:
        mismatch = (actual_sorted != expected_sorted).nonzero()
        if mismatch.numel() > 0:
            index = tuple(int(v) for v in mismatch[0])
            print(
                f"    first mismatch at {index}: actual={actual_sorted[index].item()} "
                f"expected={expected_sorted[index].item()}",
                flush=True,
            )
    if fail_fast and not ok:
        raise SystemExit(1)
    return ok


def _check_causal_support_indices(
    name: str, actual: torch.Tensor, expected: torch.Tensor, fail_fast: bool
) -> bool:
    query_positions = torch.arange(actual.size(1), device=actual.device).view(1, actual.size(1), 1)
    actual_valid = actual <= query_positions
    expected_valid = expected <= query_positions
    ok = bool(actual_valid.sum().item() == expected_valid.sum().item())
    if ok:
        actual_valid_sorted = actual.masked_fill(~actual_valid, -1).sort(dim=-1).values
        expected_valid_sorted = expected.masked_fill(~expected_valid, -1).sort(dim=-1).values
        ok = torch.equal(actual_valid_sorted, expected_valid_sorted)
    status = "OK " if ok else "BAD"
    print(f"{status} {name:<36} shape={tuple(actual.shape)}", flush=True)
    if not ok:
        actual_valid_sorted = actual.masked_fill(~actual_valid, -1).sort(dim=-1).values
        expected_valid_sorted = expected.masked_fill(~expected_valid, -1).sort(dim=-1).values
        mismatch = (actual_valid_sorted != expected_valid_sorted).nonzero()
        if mismatch.numel() > 0:
            index = tuple(int(v) for v in mismatch[0])
            print(
                f"    first valid-support mismatch at {index}: "
                f"actual={actual_valid_sorted[index].item()} "
                f"expected={expected_valid_sorted[index].item()}",
                flush=True,
            )
    if fail_fast and not ok:
        raise SystemExit(1)
    return ok


def _report_exact_indices(name: str, actual: torch.Tensor, expected: torch.Tensor) -> bool:
    ok = torch.equal(actual, expected)
    status = "OK " if ok else "INFO"
    print(f"{status} {name:<36} shape={tuple(actual.shape)}", flush=True)
    if not ok:
        mismatch = (actual != expected).nonzero()
        if mismatch.numel() > 0:
            index = tuple(int(v) for v in mismatch[0])
            print(
                f"    first order mismatch at {index}: actual={actual[index].item()} "
                f"natural_reference={expected[index].item()}",
                flush=True,
            )
    return ok


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("sparse", "dense-warmup", "dense-vs-full-topk", "sparse-fwd-dense-loss"),
        default="sparse",
        help="Which parity surface to run.",
    )
    parser.add_argument("--backend", choices=("torch-min-memory", "triton-min-memory"), default="torch-min-memory")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-query-heads", type=int, default=16)
    parser.add_argument("--num-query-groups", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--value-head-dim", type=int, default=None)
    parser.add_argument("--indexer-heads", type=int, default=4)
    parser.add_argument("--indexer-head-dim", type=int, default=64)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--indexer-rotary-dim", type=int, default=0)
    parser.add_argument("--rotary-interleaved", action="store_true")
    parser.add_argument("--hadamard", action="store_true")
    parser.add_argument("--loss-coeff", type=float, default=0.7)
    parser.add_argument("--layernorm-eps", type=float, default=1e-5)
    parser.add_argument("--query-block-size", type=int, default=512)
    parser.add_argument("--key-block-size", type=int, default=1024)
    parser.add_argument("--cache-routing", action="store_true")
    parser.add_argument("--cache-indexer-k", action="store_true")
    parser.add_argument("--cache-selected-scores", action="store_true")
    parser.add_argument(
        "--skip-attention-grad-check",
        action="store_true",
        help="Skip standalone sparse-attention backward parity before end-to-end parity.",
    )
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--distributed-backend",
        choices=("nccl", "gloo"),
        default=None,
        help="Distributed backend for dense-warmup TP checks. Defaults to nccl on CUDA.",
    )
    parser.add_argument(
        "--cuda-preflight-only",
        action="store_true",
        help="Print CUDA/PyTorch process diagnostics without importing Megatron DSA modules.",
    )
    parser.add_argument(
        "--gather-backward-diagnostic",
        action="store_true",
        help="Isolate advanced-indexing vs torch.gather backward accumulation precision.",
    )
    parser.add_argument(
        "--gather-diag-hot-keys",
        type=int,
        default=128,
        help="Number of key positions sampled by the gather backward diagnostic.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.value_head_dim is None:
        args.value_head_dim = args.head_dim

    if args.cuda_preflight_only:
        print(_cuda_summary(), flush=True)
        return 0

    requested_cuda = args.device == "cuda" or args.device.startswith("cuda:")
    if requested_cuda and not torch.cuda.is_available():
        print(
            "CUDA preflight failed before importing Megatron DSA modules. "
            "Launch this harness with the same GPU-visible wrapper used for training.\n",
            flush=True,
        )
        print(_cuda_summary(), flush=True)
        return 2

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.gather_backward_diagnostic:
        return _run_gather_backward_diagnostic(args, device)

    if args.backend == "triton-min-memory" and device.type != "cuda":
        print("triton-min-memory parity requires --device cuda.", flush=True)
        print(_cuda_summary(), flush=True)
        return 2

    _import_dsa_modules()

    dtype = _dtype(args.dtype, device)
    if args.hadamard and dtype != torch.bfloat16:
        raise SystemExit("--hadamard requires --dtype bf16 because rotate_activation requires bf16.")
    if args.num_query_heads % args.num_query_groups != 0:
        raise SystemExit("--num-query-heads must be divisible by --num-query-groups.")
    if args.indexer_rotary_dim < 0 or args.indexer_rotary_dim > args.indexer_head_dim:
        raise SystemExit("--indexer-rotary-dim must be in [0, --indexer-head-dim].")
    args.indexer_rotary_dim -= args.indexer_rotary_dim % 2

    if args.mode == "dense-warmup":
        return _run_dense_warmup_parity(args, device, dtype)
    if args.mode == "dense-vs-full-topk":
        return _run_dense_vs_full_topk_parity(args, device, dtype)
    if args.mode == "sparse-fwd-dense-loss":
        return _run_sparse_fwd_dense_loss_parity(args, device, dtype)

    use_triton = args.backend == "triton-min-memory"
    atol = args.atol if args.atol is not None else (3e-2 if dtype != torch.float32 else 2e-4)
    rtol = args.rtol if args.rtol is not None else (3e-2 if dtype != torch.float32 else 2e-4)
    rotary_pos_emb = (
        _SimpleRotary(args.indexer_rotary_dim, device, args.rotary_interleaved)
        if args.indexer_rotary_dim > 0
        else None
    )

    print(
        f"DSA parity backend={args.backend} device={device} dtype={dtype} "
        f"S={args.seq_len} B={args.batch_size} Hq={args.num_query_heads} "
        f"G={args.num_query_groups} topk={args.topk} index_dim={args.indexer_head_dim} "
        f"hadamard={args.hadamard} rotary_dim={args.indexer_rotary_dim}",
        flush=True,
    )

    base = _make_case(args, device, dtype)
    min_case = _clone_case(base)
    comp_case = _clone_case(base)

    components = _min_memory_components(comp_case, args, rotary_pos_emb, use_triton)
    ref_case = _clone_case(base)
    with _triton_dispatch_enabled(False):
        reference = _reference_run(
            ref_case, args, rotary_pos_emb, topk_override=components["topk_indices"]
        )
    min_memory = _min_memory_run(min_case, args, rotary_pos_emb, use_triton)

    failures = 0
    failures += not _check_tensor("q_index", components["q_index"], reference["q_index"], atol, rtol, args.fail_fast)
    failures += not _check_tensor("weights", components["weights"], reference["weights"], atol, rtol, args.fail_fast)
    failures += not _check_causal_support_indices(
        "topk_support",
        components["topk_indices"],
        reference["natural_topk_indices"],
        args.fail_fast,
    )
    _report_exact_indices(
        "topk_order_vs_natural_reference",
        components["topk_indices"],
        reference["natural_topk_indices"],
    )

    ref_selected_for_min_support = reference["index_scores"].gather(-1, components["topk_indices"])
    failures += not _check_tensor(
        "selected_index_scores",
        components["selected_scores"],
        ref_selected_for_min_support,
        atol,
        rtol,
        args.fail_fast,
    )
    ref_teacher_for_min_support = _reference_teacher_scores(
        comp_case.query.detach(),
        comp_case.key.detach(),
        components["topk_indices"],
        args.head_dim**-0.5,
    )
    failures += not _check_tensor(
        "teacher_scores",
        components["teacher_scores"],
        ref_teacher_for_min_support,
        atol,
        rtol,
        args.fail_fast,
    )
    ref_sparse_for_min_support = unfused_grouped_dsa_fn(
        comp_case.query.detach(),
        comp_case.key.detach(),
        comp_case.value.detach(),
        components["topk_indices"],
        args.head_dim**-0.5,
        use_gather=True,
    ).view(
        args.seq_len,
        args.batch_size,
        args.num_query_heads,
        args.value_head_dim,
    )
    failures += not _check_tensor(
        "sparse_attention_output",
        components["sparse_output"],
        ref_sparse_for_min_support,
        atol,
        rtol,
        args.fail_fast,
    )
    if not args.skip_attention_grad_check and use_triton:
        print(
            "INFO sparse_attention_grad_check skipped for triton-min-memory; "
            "raw Triton tile forward is validated through DSAMinMemoryGQAFn.backward.",
            flush=True,
        )
    elif not args.skip_attention_grad_check:
        attention_grad_output = torch.ones_like(components["sparse_output"])
        reference_attention_grads = _reference_sparse_attention_grads(
            comp_case, args, components["topk_indices"], attention_grad_output
        )
        min_attention_grads = _min_memory_sparse_attention_grads(
            comp_case, args, components["topk_indices"], attention_grad_output, use_triton
        )
        for name in ("attn_grad_query", "attn_grad_key", "attn_grad_value"):
            failures += not _check_tensor(
                name,
                min_attention_grads[name],
                reference_attention_grads[name],
                atol,
                rtol,
                args.fail_fast,
            )
    failures += not _check_tensor("forward_output", min_memory["output"], reference["output"], atol, rtol, args.fail_fast)
    failures += not _check_tensor("indexer_loss", min_memory["loss"], reference["loss"], atol, rtol, args.fail_fast)

    for name in min_memory["grads"]:
        failures += not _check_tensor(
            f"grad_{name}",
            min_memory["grads"][name],
            reference["grads"][name],
            atol,
            rtol,
            args.fail_fast,
        )

    if failures:
        print(f"\nFAIL: {failures} parity checks failed.", flush=True)
        return 1
    print("\nPASS: all parity checks passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
