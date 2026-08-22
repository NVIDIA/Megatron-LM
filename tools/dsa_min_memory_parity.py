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
from types import SimpleNamespace
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
    global dsa_main_attention_aux_loss
    global dsa_min_memory_gqa
    global dsa_min_memory_gqa_forward_only
    global _project_k_index_block
    global _project_q_index_tile
    global _indexer_input_norm_stats
    global _selected_index_scores_tile
    global _project_simplified_q_index_tile
    global _routing_key_chunk_size
    global _simplified_index_scores_block
    global _simplified_topk_index_tile
    global _sparse_attention_tile
    global _teacher_scores_tile
    global _topk_index_tile
    global _triton_dispatch_enabled
    global compute_gqa_dsa_indexer_loss
    global fused_qk_topk_naive
    global rotate_activation
    global unfused_grouped_dsa_fn

    from megatron.core.transformer.experimental_attention_variant.dsa import (
        fused_qk_topk_naive as _fused_qk_topk_naive,
        rotate_activation as _rotate_activation,
    )
    from megatron.core.transformer.experimental_attention_variant.dsa_gqa import (
        compute_gqa_dsa_indexer_loss as _compute_gqa_dsa_indexer_loss,
        unfused_grouped_dsa_fn as _unfused_grouped_dsa_fn,
    )
    from megatron.core.transformer.experimental_attention_variant.dsa_min_memory import (
        DSAMinMemoryGQAFn as _DSAMinMemoryGQAFn,
        dsa_dense_indexer_loss as _dsa_dense_indexer_loss,
        dsa_main_attention_aux_loss as _dsa_main_attention_aux_loss,
        dsa_min_memory_gqa as _dsa_min_memory_gqa,
        dsa_min_memory_gqa_forward_only as _dsa_min_memory_gqa_forward_only,
        _project_k_index_block as _project_k_index_block_imported,
        _project_q_index_tile as _project_q_index_tile_imported,
        _indexer_input_norm_stats as _indexer_input_norm_stats_imported,
        _project_simplified_q_index_tile as _project_simplified_q_index_tile_imported,
        _routing_key_chunk_size as _routing_key_chunk_size_imported,
        _selected_index_scores_tile as _selected_index_scores_tile_imported,
        _simplified_index_scores_block as _simplified_index_scores_block_imported,
        _simplified_topk_index_tile as _simplified_topk_index_tile_imported,
        _sparse_attention_tile as _sparse_attention_tile_imported,
        _teacher_scores_tile as _teacher_scores_tile_imported,
        _topk_index_tile as _topk_index_tile_imported,
        _triton_dispatch_enabled as _triton_dispatch_enabled_imported,
    )

    DSAMinMemoryGQAFn = _DSAMinMemoryGQAFn
    dsa_dense_indexer_loss = _dsa_dense_indexer_loss
    dsa_main_attention_aux_loss = _dsa_main_attention_aux_loss
    dsa_min_memory_gqa = _dsa_min_memory_gqa
    dsa_min_memory_gqa_forward_only = _dsa_min_memory_gqa_forward_only
    _project_k_index_block = _project_k_index_block_imported
    _project_q_index_tile = _project_q_index_tile_imported
    _indexer_input_norm_stats = _indexer_input_norm_stats_imported
    _project_simplified_q_index_tile = _project_simplified_q_index_tile_imported
    _routing_key_chunk_size = _routing_key_chunk_size_imported
    _selected_index_scores_tile = _selected_index_scores_tile_imported
    _simplified_index_scores_block = _simplified_index_scores_block_imported
    _simplified_topk_index_tile = _simplified_topk_index_tile_imported
    _sparse_attention_tile = _sparse_attention_tile_imported
    _teacher_scores_tile = _teacher_scores_tile_imported
    _topk_index_tile = _topk_index_tile_imported
    _triton_dispatch_enabled = _triton_dispatch_enabled_imported
    compute_gqa_dsa_indexer_loss = _compute_gqa_dsa_indexer_loss
    fused_qk_topk_naive = _fused_qk_topk_naive
    rotate_activation = _rotate_activation
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


def _standard_input_norm(args, device: torch.device, dtype: torch.dtype):
    if args.standard_input_norm == "none":
        return None
    weight = torch.linspace(0.75, 1.25, args.hidden_size, device=device, dtype=dtype)
    if args.zero_centered_gamma:
        weight = weight - 1.0
    bias = (
        torch.linspace(-0.1, 0.1, args.hidden_size, device=device, dtype=dtype)
        if args.standard_input_norm == "layernorm"
        else None
    )
    return SimpleNamespace(
        normalization=(
            "LayerNorm" if args.standard_input_norm == "layernorm" else "RMSNorm"
        ),
        weight=weight,
        bias=bias,
        eps=args.layernorm_eps,
        zero_centered_gamma=args.zero_centered_gamma,
    )


def _standard_indexer_input_oracle(
    hidden_states: torch.Tensor,
    args,
    norm_stats: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    norm = _standard_input_norm(args, hidden_states.device, hidden_states.dtype)
    if norm is None:
        return hidden_states
    weight = norm.weight + 1.0 if norm.zero_centered_gamma else norm.weight
    if norm.normalization == "RMSNorm":
        hidden_float = hidden_states.float()
        rstd = norm_stats
        if rstd is None:
            rstd = torch.rsqrt(
                hidden_float.square().mean(dim=-1, keepdim=True) + norm.eps
            )
        elif rstd.dim() == hidden_states.dim() - 1:
            rstd = rstd.unsqueeze(-1)
        return (hidden_float * rstd * weight.float()).to(hidden_states.dtype)
    return F.layer_norm(
        hidden_states,
        (hidden_states.size(-1),),
        weight,
        norm.bias,
        norm.eps,
    )


def _simplified_indexer_from_case(case: Case, args, pg_collection):
    indexer = type(
        "_SimplifiedIndexer",
        (),
        {
            "linear_q": _WeightOnlyModule(case.linear_q_weight),
            "linear_k": (
                _WeightOnlyModule(case.linear_k_weight) if args.simplified_learned_k else None
            ),
            "index_n_heads": 1,
            "index_head_dim": args.indexer_head_dim,
            "index_topk": args.topk,
            "softmax_scale": args.indexer_head_dim**-0.5,
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
                "_SimplifiedIndexerConfig",
                (),
                {
                    "dsa_indexer_mode": "simplified",
                    "dsa_simplified_use_learned_k": args.simplified_learned_k,
                    "rotary_interleaved": args.rotary_interleaved,
                },
            )(),
        },
    )()
    return indexer


def _apply_rope_oracle(
    tensor: torch.Tensor,
    head_dim: int,
    rotary_dim: int,
    rotary_pos_emb,
    rotary_interleaved: bool,
) -> torch.Tensor:
    if rotary_pos_emb is None or rotary_dim == 0:
        return tensor
    tensor_nope, tensor_pe = torch.split(
        tensor, [head_dim - rotary_dim, rotary_dim], dim=-1
    )
    positions = torch.arange(tensor.size(0), device=tensor.device, dtype=torch.float32)
    inv_freq = rotary_pos_emb.inv_freq[: rotary_dim // 2].to(tensor.device)
    interpolation_factor = getattr(rotary_pos_emb, "seq_len_interpolation_factor", None)
    if interpolation_factor is not None:
        positions = positions / interpolation_factor
    freqs = positions[:, None] * inv_freq[None, :]
    if rotary_interleaved:
        freqs = torch.stack((freqs, freqs), dim=-1).flatten(-2)
    else:
        freqs = torch.cat((freqs, freqs), dim=-1)
    cos = torch.cos(freqs).to(tensor_pe.dtype)[:, None, None, :]
    sin = torch.sin(freqs).to(tensor_pe.dtype)[:, None, None, :]
    tensor_pe = tensor_pe * cos + _rotate_half_oracle(tensor_pe, rotary_interleaved) * sin
    return torch.cat((tensor_nope, tensor_pe), dim=-1)


def _project_indexer(
    case: Case,
    args,
    rotary_pos_emb,
    input_norm_stats: Optional[torch.Tensor] = None,
):
    """Standard-DSA projection oracle independent of min-memory projection helpers."""
    indexer_input = _standard_indexer_input_oracle(
        case.hidden_states, args, norm_stats=input_norm_stats
    )
    q_index = F.linear(indexer_input, case.linear_q_weight).reshape(
        args.seq_len,
        args.batch_size,
        args.indexer_heads,
        args.indexer_head_dim,
    )
    k_index = F.linear(indexer_input, case.linear_k_weight)
    k_index = F.layer_norm(
        k_index,
        (args.indexer_head_dim,),
        case.k_norm_weight,
        case.k_norm_bias,
        args.layernorm_eps,
    ).reshape(args.seq_len, args.batch_size, 1, args.indexer_head_dim)
    q_index = _apply_rope_oracle(
        q_index,
        args.indexer_head_dim,
        args.indexer_rotary_dim,
        rotary_pos_emb,
        args.rotary_interleaved,
    )
    k_index = _apply_rope_oracle(
        k_index,
        args.indexer_head_dim,
        args.indexer_rotary_dim,
        rotary_pos_emb,
        args.rotary_interleaved,
    ).squeeze(2)
    if args.hadamard:
        # fast_hadamard_transform's BF16 reduction order is part of the reference DSA
        # mixed-precision semantics. A Python butterfly rounds at every stage and is not a
        # suitable elementwise oracle for this component.
        q_index = rotate_activation(q_index)
        k_index = rotate_activation(k_index)
    weights = F.linear(indexer_input, case.linear_weights_weight)
    weights = weights * (args.indexer_heads**-0.5) * (args.indexer_head_dim**-0.5)
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


def _reference_run(
    case: Case,
    args,
    rotary_pos_emb,
    topk_override: Optional[torch.Tensor] = None,
    input_norm_stats: Optional[torch.Tensor] = None,
):
    q_index, k_index, weights = _project_indexer(
        case, args, rotary_pos_emb, input_norm_stats=input_norm_stats
    )
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
    input_norm = _standard_input_norm(args, case.hidden_states.device, case.hidden_states.dtype)
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
        input_norm,
    )
    (output.float().sum() + loss.float()).backward()
    return {
        "output": output.detach(),
        "loss": loss.detach(),
        "grads": {name: tensor.grad.detach().clone() for name, tensor in _case_tensors(case)},
    }


def _min_memory_components(case: Case, args, rotary_pos_emb, use_triton: bool):
    # Match DSAMinMemoryGQAFn: the torch oracle deliberately uses a single full key block.
    key_block = args.key_block_size if use_triton else args.seq_len
    target_topk = min(args.topk, args.seq_len)
    q_indices = []
    all_weights = []
    all_topk_scores = []
    all_topk_indices = []
    all_selected_scores = []
    all_teacher_scores = []
    all_sparse_outputs = []
    with _triton_dispatch_enabled(use_triton), torch.no_grad():
        input_norm = _standard_input_norm(
            args, case.hidden_states.device, case.hidden_states.dtype
        )
        input_norm_stats = _indexer_input_norm_stats(case.hidden_states, input_norm)
        for q_start in range(0, args.seq_len, args.query_block_size):
            q_end = min(q_start + args.query_block_size, args.seq_len)
            topk_scores, topk_indices, q_index, weights = _topk_index_tile(
                case.hidden_states,
                q_start,
                q_end,
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
                input_norm,
                input_norm_stats,
            )
            if topk_indices.size(-1) < target_topk:
                pad = target_topk - topk_indices.size(-1)
                # q_end is in-bounds and future-causal for every row in this non-final tile.
                assert q_end < args.seq_len
                topk_indices = F.pad(topk_indices, (0, pad), value=q_end)
                topk_scores = F.pad(topk_scores, (0, pad), value=float("-inf"))
            selected_scores = _selected_index_scores_tile(
                case.hidden_states,
                q_start,
                q_end,
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
                input_norm,
                input_norm_stats,
            )
            teacher_scores = _teacher_scores_tile(
                case.query[q_start:q_end],
                case.key,
                topk_indices,
                args.head_dim**-0.5,
                q_start,
                _DummyPGCollection(),
            )
            sparse_output = _sparse_attention_tile(
                case.query[q_start:q_end],
                case.key,
                case.value,
                topk_indices,
                args.head_dim**-0.5,
                q_start,
            )
            q_indices.append(q_index)
            all_weights.append(weights)
            all_topk_scores.append(topk_scores)
            all_topk_indices.append(topk_indices)
            all_selected_scores.append(selected_scores)
            all_teacher_scores.append(teacher_scores)
            all_sparse_outputs.append(sparse_output)
    return {
        "q_index": torch.cat(q_indices, dim=0),
        "weights": torch.cat(all_weights, dim=0),
        "input_norm_stats": input_norm_stats,
        "topk_scores": torch.cat(all_topk_scores, dim=1),
        "topk_indices": torch.cat(all_topk_indices, dim=1),
        "selected_scores": torch.cat(all_selected_scores, dim=1),
        "teacher_scores": torch.cat(all_teacher_scores, dim=1),
        "sparse_output": torch.cat(all_sparse_outputs, dim=0),
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
    input_norm = _standard_input_norm(args, case.hidden_states.device, case.hidden_states.dtype)
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
        simplified_input_norm=input_norm,
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
    input_norm = _standard_input_norm(args, case.hidden_states.device, case.hidden_states.dtype)
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
        simplified_input_norm=input_norm,
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
        "index_scores": index_scores.detach(),
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
    input_norm = _standard_input_norm(args, case.hidden_states.device, case.hidden_states.dtype)
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
        simplified_input_norm=input_norm,
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
        simplified_input_norm=input_norm,
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
    failures += not _check_topk_support_with_score_error(
        "sparse_dense_loss_topk_support",
        components["topk_indices"],
        components["topk_scores"],
        reference["index_scores"],
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


def _rotate_half_oracle(x: torch.Tensor, interleaved: bool) -> torch.Tensor:
    """Independent RoPE rotation used by the simplified mathematical oracle."""
    if interleaved:
        pairs = x.unflatten(-1, (-1, 2))
        return torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _simplified_input_oracle(hidden_states: torch.Tensor, norm_spec) -> torch.Tensor:
    hidden_states = hidden_states.detach()
    if norm_spec is None:
        return hidden_states
    weight = norm_spec.weight
    if norm_spec.zero_centered_gamma:
        weight = weight + 1.0
    if norm_spec.normalization == "RMSNorm":
        hidden_float = hidden_states.float()
        normalized = hidden_float * torch.rsqrt(
            hidden_float.square().mean(dim=-1, keepdim=True) + norm_spec.eps
        )
        return (normalized * weight.float()).to(hidden_states.dtype)
    if norm_spec.normalization == "LayerNorm":
        return F.layer_norm(
            hidden_states,
            (hidden_states.size(-1),),
            weight,
            norm_spec.bias,
            norm_spec.eps,
        )
    raise AssertionError(f"Unsupported simplified oracle norm {norm_spec.normalization!r}")


def _simplified_q_index_oracle(case: Case, args, norm_spec) -> torch.Tensor:
    hidden = _simplified_input_oracle(case.hidden_states, norm_spec)
    q_index = F.linear(hidden, case.linear_q_weight).reshape(
        args.seq_len, args.batch_size, 1, args.indexer_head_dim
    )
    rotary_dim = args.indexer_rotary_dim
    if rotary_dim == 0:
        return q_index

    q_nope, q_pe = torch.split(
        q_index, [args.indexer_head_dim - rotary_dim, rotary_dim], dim=-1
    )
    positions = torch.arange(args.seq_len, device=q_index.device, dtype=torch.float32)
    inv_freq = 1.0 / (
        10000
        ** (
            torch.arange(0, rotary_dim, 2, device=q_index.device, dtype=torch.float32)
            / rotary_dim
        )
    )
    freqs = positions[:, None] * inv_freq[None, :]
    if args.rotary_interleaved:
        freqs = torch.stack((freqs, freqs), dim=-1).flatten(-2)
    else:
        freqs = torch.cat((freqs, freqs), dim=-1)
    cos = torch.cos(freqs).to(q_pe.dtype)[:, None, None, :]
    sin = torch.sin(freqs).to(q_pe.dtype)[:, None, None, :]
    q_pe = q_pe * cos + _rotate_half_oracle(q_pe, args.rotary_interleaved) * sin
    return torch.cat((q_nope, q_pe), dim=-1)


def _simplified_k_index_oracle(case: Case, args, norm_spec) -> torch.Tensor:
    if not args.simplified_learned_k:
        return case.key.detach()
    hidden = _simplified_input_oracle(case.hidden_states, norm_spec)
    k_index = F.linear(hidden, case.linear_k_weight).reshape(
        args.seq_len, args.batch_size, 1, args.indexer_head_dim
    )
    rotary = (
        _SimpleRotary(
            args.indexer_rotary_dim,
            case.hidden_states.device,
            args.rotary_interleaved,
        )
        if args.indexer_rotary_dim > 0
        else None
    )
    return _apply_rope_oracle(
        k_index,
        args.indexer_head_dim,
        args.indexer_rotary_dim,
        rotary,
        args.rotary_interleaved,
    )


def _simplified_scores_oracle(
    q_index: torch.Tensor, key: torch.Tensor, score_scale: float
) -> torch.Tensor:
    scores = torch.einsum(
        "qbd,kbd->bqk", q_index[:, :, 0].float(), key[:, :, 0].float()
    ) * score_scale
    return scores + _causal_mask(q_index.size(0), q_index.device).unsqueeze(0)


def _simplified_sparse_attention_oracle(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Explicit one-KV-group sparse attention, independent of DSA attention helpers."""
    sequence_length, batch_size, num_heads, head_dim = query.shape
    value_dim = value.size(-1)
    topk = topk_indices.size(-1)
    key_by_batch = key[:, :, 0].permute(1, 0, 2)
    value_by_batch = value[:, :, 0].permute(1, 0, 2)
    key_index = topk_indices[..., None].expand(batch_size, sequence_length, topk, head_dim)
    value_index = topk_indices[..., None].expand(
        batch_size, sequence_length, topk, value_dim
    )
    selected_key = torch.gather(
        key_by_batch[:, None].expand(batch_size, sequence_length, sequence_length, head_dim),
        2,
        key_index,
    )
    selected_value = torch.gather(
        value_by_batch[:, None].expand(
            batch_size, sequence_length, sequence_length, value_dim
        ),
        2,
        value_index,
    )
    query_bhqd = query.permute(1, 2, 0, 3)
    logits = torch.einsum("bhqd,bqkd->bhqk", query_bhqd.float(), selected_key.float())
    logits = logits * softmax_scale
    query_positions = torch.arange(sequence_length, device=query.device).view(
        1, 1, sequence_length, 1
    )
    logits = logits.masked_fill(topk_indices[:, None] > query_positions, float("-inf"))
    probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
    output = torch.einsum(
        "bhqk,bqkd->bhqd", probabilities.to(selected_value.dtype), selected_value
    )
    return output.permute(2, 0, 1, 3).reshape(
        sequence_length, batch_size, num_heads * value_dim
    )


def _simplified_teacher_oracle(
    query: torch.Tensor,
    key: torch.Tensor,
    topk_indices: Optional[torch.Tensor],
    softmax_scale: float,
    pg_collection,
) -> torch.Tensor:
    sequence_length, batch_size, _, head_dim = query.shape
    query_bhqd = query.permute(1, 2, 0, 3).float()
    key_bkd = key[:, :, 0].permute(1, 0, 2).float()
    if topk_indices is None:
        logits = torch.einsum("bhqd,bkd->bhqk", query_bhqd, key_bkd) * softmax_scale
        logits = logits + _causal_mask(sequence_length, query.device)[None, None]
    else:
        topk = topk_indices.size(-1)
        gather_index = topk_indices[..., None].expand(
            batch_size, sequence_length, topk, head_dim
        )
        selected_key = torch.gather(
            key_bkd[:, None].expand(batch_size, sequence_length, sequence_length, head_dim),
            2,
            gather_index,
        )
        logits = torch.einsum("bhqd,bqkd->bhqk", query_bhqd, selected_key) * softmax_scale
        query_positions = torch.arange(sequence_length, device=query.device).view(
            1, 1, sequence_length, 1
        )
        logits = logits.masked_fill(topk_indices[:, None] > query_positions, float("-inf"))
    teacher = torch.softmax(logits, dim=-1, dtype=torch.float32).sum(dim=1)
    if pg_collection.tp.size() > 1:
        torch.distributed.all_reduce(teacher, group=pg_collection.tp)
    return teacher / teacher.sum(dim=-1, keepdim=True)


def _kl_oracle(teacher: torch.Tensor, student_logits: torch.Tensor, loss_coeff: float):
    student = torch.softmax(student_logits, dim=-1, dtype=torch.float32)
    return (
        teacher
        * (torch.log(teacher + 1.0e-10) - torch.log(student + 1.0e-10))
    ).sum(dim=-1).mean() * loss_coeff


def _simplified_sparse_kl_oracle(
    scores: torch.Tensor,
    topk_indices: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    softmax_scale: float,
    loss_coeff: float,
    pg_collection,
) -> torch.Tensor:
    teacher = _simplified_teacher_oracle(
        query, key, topk_indices, softmax_scale, pg_collection
    )
    return _kl_oracle(teacher, scores.gather(-1, topk_indices), loss_coeff)


def _simplified_dense_kl_oracle(
    scores: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    softmax_scale: float,
    loss_coeff: float,
    pg_collection,
) -> torch.Tensor:
    teacher = _simplified_teacher_oracle(query, key, None, softmax_scale, pg_collection)
    return _kl_oracle(teacher, scores, loss_coeff)


def _simplified_reference_sparse_run(
    case: Case,
    args,
    pg_collection,
    simplified_input_norm,
    topk_override: Optional[torch.Tensor] = None,
):
    attention_scale = (
        args.attention_softmax_scale
        if args.attention_softmax_scale is not None
        else args.head_dim**-0.5
    )
    q_index = _simplified_q_index_oracle(case, args, simplified_input_norm)
    k_index = _simplified_k_index_oracle(case, args, simplified_input_norm)
    scores = _simplified_scores_oracle(
        q_index, k_index, args.indexer_head_dim**-0.5
    )
    natural_topk_indices = scores.topk(min(args.topk, args.seq_len), dim=-1).indices
    topk_indices = natural_topk_indices if topk_override is None else topk_override
    output = _simplified_sparse_attention_oracle(
        case.query,
        case.key,
        case.value,
        topk_indices,
        attention_scale,
    )
    loss = _simplified_sparse_kl_oracle(
        scores,
        topk_indices,
        case.query.detach(),
        case.key.detach(),
        attention_scale,
        args.loss_coeff,
        pg_collection,
    )
    grad_inputs = [case.query, case.key, case.value, case.linear_q_weight]
    grad_names = ["query", "key", "value", "linear_q_weight"]
    if args.simplified_learned_k:
        grad_inputs.append(case.linear_k_weight)
        grad_names.append("linear_k_weight")
    grads = torch.autograd.grad(output.float().sum() + loss, tuple(grad_inputs))
    return {
        "q_index": q_index.detach(),
        "scores": scores.detach(),
        "natural_topk_indices": natural_topk_indices.detach(),
        "topk_indices": topk_indices.detach(),
        "output": output.detach(),
        "loss": loss.detach(),
        "grads": {name: grad.detach() for name, grad in zip(grad_names, grads)},
    }


def _simplified_min_sparse_run(
    case: Case, args, pg_collection, use_triton: bool, simplified_input_norm
):
    attention_scale = (
        args.attention_softmax_scale
        if args.attention_softmax_scale is not None
        else args.head_dim**-0.5
    )
    indexer = _simplified_indexer_from_case(case, args, pg_collection)
    output, loss = dsa_min_memory_gqa(
        query=case.query,
        key=case.key,
        value=case.value,
        hidden_states=case.hidden_states,
        indexer=indexer,
        softmax_scale=attention_scale,
        loss_coeff=args.loss_coeff,
        use_indexer_rope=args.indexer_rotary_dim > 0,
        query_chunk_size=args.query_block_size,
        key_chunk_size=args.key_block_size,
        cache_routing=args.cache_routing,
        cache_indexer_k=args.cache_indexer_k,
        cache_selected_scores=args.cache_selected_scores,
        use_triton=use_triton,
        simplified_input_norm=simplified_input_norm,
    )
    grad_inputs = [case.query, case.key, case.value, case.linear_q_weight]
    grad_names = ["query", "key", "value", "linear_q_weight"]
    if args.simplified_learned_k:
        grad_inputs.append(case.linear_k_weight)
        grad_names.append("linear_k_weight")
    grads = torch.autograd.grad(output.float().sum() + loss, tuple(grad_inputs))
    return {
        "output": output.detach(),
        "loss": loss.detach(),
        "grads": {name: grad.detach() for name, grad in zip(grad_names, grads)},
    }


def _simplified_dense_runs(
    case: Case, args, pg_collection, use_triton: bool, simplified_input_norm
):
    attention_scale = (
        args.attention_softmax_scale
        if args.attention_softmax_scale is not None
        else args.head_dim**-0.5
    )
    q_index = _simplified_q_index_oracle(case, args, simplified_input_norm)
    k_index = _simplified_k_index_oracle(case, args, simplified_input_norm)
    scores = _simplified_scores_oracle(
        q_index, k_index, args.indexer_head_dim**-0.5
    )
    reference_loss = _simplified_dense_kl_oracle(
        scores,
        case.query.detach(),
        case.key.detach(),
        attention_scale,
        args.loss_coeff,
        pg_collection,
    )
    reference_params = [case.linear_q_weight]
    grad_names = ["linear_q_weight"]
    if args.simplified_learned_k:
        reference_params.append(case.linear_k_weight)
        grad_names.append("linear_k_weight")
    reference_grad_values = torch.autograd.grad(reference_loss, tuple(reference_params))
    reference_grads = {
        name: grad.detach() for name, grad in zip(grad_names, reference_grad_values)
    }

    min_case = _clone_case(case)
    indexer = _simplified_indexer_from_case(min_case, args, pg_collection)
    min_loss = dsa_dense_indexer_loss(
        query=min_case.query.detach(),
        key=min_case.key.detach(),
        hidden_states=min_case.hidden_states,
        indexer=indexer,
        softmax_scale=attention_scale,
        loss_coeff=args.loss_coeff,
        use_indexer_rope=args.indexer_rotary_dim > 0,
        query_chunk_size=args.query_block_size,
        key_chunk_size=args.key_block_size,
        use_triton=use_triton,
        simplified_input_norm=simplified_input_norm,
    )
    min_params = [min_case.linear_q_weight]
    if args.simplified_learned_k:
        min_params.append(min_case.linear_k_weight)
    min_grad_values = torch.autograd.grad(min_loss, tuple(min_params))
    min_grads = {name: grad.detach() for name, grad in zip(grad_names, min_grad_values)}
    return reference_loss.detach(), reference_grads, min_loss.detach(), min_grads


def _simplified_sparse_fwd_dense_loss_runs(
    case: Case,
    args,
    pg_collection,
    use_triton: bool,
    simplified_input_norm,
    topk_indices: torch.Tensor,
):
    attention_scale = (
        args.attention_softmax_scale
        if args.attention_softmax_scale is not None
        else args.head_dim**-0.5
    )
    reference_case = _clone_case(case)
    q_index = _simplified_q_index_oracle(reference_case, args, simplified_input_norm)
    k_index = _simplified_k_index_oracle(reference_case, args, simplified_input_norm)
    scores = _simplified_scores_oracle(
        q_index, k_index, args.indexer_head_dim**-0.5
    )
    reference_output = _simplified_sparse_attention_oracle(
        reference_case.query,
        reference_case.key,
        reference_case.value,
        topk_indices,
        attention_scale,
    )
    reference_loss = _simplified_dense_kl_oracle(
        scores,
        reference_case.query.detach(),
        reference_case.key.detach(),
        attention_scale,
        args.loss_coeff,
        pg_collection,
    )
    reference_inputs = [
        reference_case.query,
        reference_case.key,
        reference_case.value,
        reference_case.linear_q_weight,
    ]
    names = ["query", "key", "value", "linear_q_weight"]
    if args.simplified_learned_k:
        reference_inputs.append(reference_case.linear_k_weight)
        names.append("linear_k_weight")
    reference_grads = torch.autograd.grad(
        reference_output.float().sum() + reference_loss,
        tuple(reference_inputs),
    )

    min_case = _clone_case(case)
    indexer = _simplified_indexer_from_case(min_case, args, pg_collection)
    min_output, internal_sparse_loss = dsa_min_memory_gqa(
        query=min_case.query,
        key=min_case.key,
        value=min_case.value,
        hidden_states=min_case.hidden_states,
        indexer=indexer,
        softmax_scale=attention_scale,
        loss_coeff=0.0,
        use_indexer_rope=args.indexer_rotary_dim > 0,
        query_chunk_size=args.query_block_size,
        key_chunk_size=args.key_block_size,
        cache_routing=args.cache_routing,
        cache_indexer_k=args.cache_indexer_k,
        cache_selected_scores=False,
        use_triton=use_triton,
        simplified_input_norm=simplified_input_norm,
    )
    min_loss = dsa_dense_indexer_loss(
        query=min_case.query.detach(),
        key=min_case.key.detach(),
        hidden_states=min_case.hidden_states,
        indexer=indexer,
        softmax_scale=attention_scale,
        loss_coeff=args.loss_coeff,
        use_indexer_rope=args.indexer_rotary_dim > 0,
        query_chunk_size=args.query_block_size,
        key_chunk_size=args.key_block_size,
        use_triton=use_triton,
        simplified_input_norm=simplified_input_norm,
    )
    min_inputs = [
        min_case.query,
        min_case.key,
        min_case.value,
        min_case.linear_q_weight,
    ]
    if args.simplified_learned_k:
        min_inputs.append(min_case.linear_k_weight)
    min_grads = torch.autograd.grad(
        min_output.float().sum() + min_loss,
        tuple(min_inputs),
    )
    return {
        "reference_output": reference_output.detach(),
        "reference_loss": reference_loss.detach(),
        "reference_grads": {
            name: grad.detach() for name, grad in zip(names, reference_grads)
        },
        "min_output": min_output.detach(),
        "internal_sparse_loss": internal_sparse_loss.detach(),
        "min_loss": min_loss.detach(),
        "min_grads": {name: grad.detach() for name, grad in zip(names, min_grads)},
    }


def _run_simplified_parity(args, device: torch.device, dtype: torch.dtype) -> int:
    if args.num_query_groups != 1:
        raise SystemExit("Simplified DSA parity requires --num-query-groups 1.")
    if args.indexer_heads != 1:
        raise SystemExit("Simplified DSA parity requires --indexer-heads 1.")
    if not args.simplified_learned_k and args.indexer_head_dim != args.head_dim:
        raise SystemExit(
            "Main-K simplified DSA requires --indexer-head-dim equal to --head-dim."
        )
    if args.hadamard:
        raise SystemExit("Simplified DSA does not use Hadamard.")
    if args.cache_indexer_k and not args.simplified_learned_k:
        raise SystemExit("Only learned-K simplified DSA has a separate indexer-K cache.")

    use_triton = args.backend == "triton-min-memory"
    device, rank, world_size, pg_collection = _configure_distributed_for_dense_mode(
        device, args.distributed_backend
    )
    atol = args.atol if args.atol is not None else (5e-2 if dtype != torch.float32 else 5e-4)
    rtol = args.rtol if args.rtol is not None else (5e-2 if dtype != torch.float32 else 5e-4)
    print(
        f"Simplified DSA parity backend={args.backend} rank={rank}/{world_size} "
        f"device={device} dtype={dtype} local_Hq={args.num_query_heads} G=1 "
        f"S={args.seq_len} topk={args.topk} Dattn={args.head_dim} "
        f"Dindex={args.indexer_head_dim} learned_k={args.simplified_learned_k} "
        f"QBLOCK={args.query_block_size} KBLOCK={args.key_block_size} "
        f"rotary_dim={args.indexer_rotary_dim} "
        f"input_norm={args.simplified_input_norm} "
        f"zero_centered_gamma={args.zero_centered_gamma} "
        f"attention_scale={args.attention_softmax_scale}",
        flush=True,
    )

    base = _make_case(args, device, dtype)
    simplified_input_norm = None
    if args.simplified_input_norm != "none":
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed + 2719)
        norm_weight = torch.randn(
            args.hidden_size, device=device, dtype=dtype, generator=generator
        )
        norm_bias = (
            torch.randn(args.hidden_size, device=device, dtype=dtype, generator=generator)
            if args.simplified_input_norm == "layernorm"
            else None
        )
        simplified_input_norm = SimpleNamespace(
            normalization=(
                "LayerNorm" if args.simplified_input_norm == "layernorm" else "RMSNorm"
            ),
            weight=norm_weight,
            bias=norm_bias,
            eps=args.layernorm_eps,
            zero_centered_gamma=args.zero_centered_gamma,
        )
    if world_size > 1:
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed + 1009 + rank)
        base.query = torch.randn(
            base.query.shape, device=device, dtype=dtype, generator=generator
        ).requires_grad_(True)

    component_case = _clone_case(base)
    component_rotary = (
        _SimpleRotary(args.indexer_rotary_dim, device, args.rotary_interleaved)
        if args.indexer_rotary_dim > 0
        else None
    )
    component_topk = []
    component_topk_scores = []
    component_q_index = []
    with _triton_dispatch_enabled(use_triton), torch.no_grad():
        for q_start in range(0, args.seq_len, args.query_block_size):
            q_end = min(q_start + args.query_block_size, args.seq_len)
            topk_scores, topk_indices, q_index = _simplified_topk_index_tile(
                component_case.hidden_states,
                component_case.key,
                q_start,
                q_end,
                component_case.linear_q_weight,
                args.topk,
                args.indexer_head_dim,
                args.indexer_rotary_dim,
                component_rotary,
                args.rotary_interleaved,
                args.indexer_rotary_dim > 0,
                args.indexer_head_dim**-0.5,
                args.key_block_size if use_triton else args.seq_len,
                simplified_input_norm,
                linear_k_weight=(
                    component_case.linear_k_weight if args.simplified_learned_k else None
                ),
            )
            if topk_indices.size(-1) < min(args.topk, args.seq_len):
                assert q_end < args.seq_len
                topk_indices = F.pad(
                    topk_indices,
                    (0, min(args.topk, args.seq_len) - topk_indices.size(-1)),
                    value=q_end,
                )
                topk_scores = F.pad(
                    topk_scores,
                    (0, min(args.topk, args.seq_len) - topk_scores.size(-1)),
                    value=float("-inf"),
                )
            component_topk.append(topk_indices)
            component_topk_scores.append(topk_scores)
            component_q_index.append(q_index)
    component_topk = torch.cat(component_topk, dim=1)
    component_topk_scores = torch.cat(component_topk_scores, dim=1)
    component_q_index = torch.cat(component_q_index, dim=0)
    with _triton_dispatch_enabled(False):
        reference = _simplified_reference_sparse_run(
            _clone_case(base),
            args,
            pg_collection,
            simplified_input_norm,
            topk_override=component_topk,
        )
    min_memory = _simplified_min_sparse_run(
        _clone_case(base), args, pg_collection, use_triton, simplified_input_norm
    )
    forward_only_case = _clone_case(base)
    forward_only_indexer = _simplified_indexer_from_case(
        forward_only_case, args, pg_collection
    )
    attention_scale = (
        args.attention_softmax_scale
        if args.attention_softmax_scale is not None
        else args.head_dim**-0.5
    )
    with torch.no_grad():
        forward_only_output = dsa_min_memory_gqa_forward_only(
            query=forward_only_case.query,
            key=forward_only_case.key,
            value=forward_only_case.value,
            hidden_states=forward_only_case.hidden_states,
            indexer=forward_only_indexer,
            softmax_scale=attention_scale,
            use_indexer_rope=args.indexer_rotary_dim > 0,
            query_chunk_size=args.query_block_size,
            key_chunk_size=args.key_block_size,
            cache_indexer_k=args.cache_indexer_k,
            use_triton=use_triton,
            simplified_input_norm=simplified_input_norm,
        )
    ref_dense_loss, ref_dense_grads, min_dense_loss, min_dense_grads = (
        _simplified_dense_runs(
            _clone_case(base), args, pg_collection, use_triton, simplified_input_norm
        )
    )
    sparse_dense = _simplified_sparse_fwd_dense_loss_runs(
        _clone_case(base),
        args,
        pg_collection,
        use_triton,
        simplified_input_norm,
        component_topk,
    )

    failures = 0
    failures += not _check_tensor(
        "simplified_q_index",
        component_q_index,
        reference["q_index"],
        atol,
        rtol,
        args.fail_fast,
    )
    failures += not _check_topk_support_with_score_error(
        "simplified_topk_support",
        component_topk,
        component_topk_scores,
        reference["scores"],
        args.fail_fast,
    )
    valid_width = torch.arange(args.seq_len, device=device).add(1).clamp(max=args.topk)
    query_positions = torch.arange(args.seq_len, device=device).view(1, args.seq_len, 1)
    actual_valid_width = (component_topk <= query_positions).sum(dim=-1)
    expected_valid_width = valid_width.view(1, -1).expand_as(actual_valid_width)
    failures += not _check_tensor(
        "simplified_causal_support_width",
        actual_valid_width,
        expected_valid_width,
        0.0,
        0.0,
        args.fail_fast,
    )
    failures += not _check_tensor(
        "simplified_sparse_output",
        min_memory["output"],
        reference["output"],
        atol,
        rtol,
        args.fail_fast,
    )
    failures += not _check_tensor(
        "simplified_forward_only_output",
        forward_only_output,
        reference["output"],
        atol,
        rtol,
        args.fail_fast,
    )
    failures += not _check_tensor(
        "simplified_sparse_loss",
        min_memory["loss"],
        reference["loss"],
        atol,
        rtol,
        args.fail_fast,
    )
    for name, expected in reference["grads"].items():
        failures += not _check_tensor(
            f"simplified_sparse_grad_{name}",
            min_memory["grads"][name],
            expected,
            atol,
            rtol,
            args.fail_fast,
        )
    failures += not _check_tensor(
        "simplified_dense_loss",
        min_dense_loss,
        ref_dense_loss,
        atol,
        rtol,
        args.fail_fast,
    )
    for name, expected in ref_dense_grads.items():
        failures += not _check_tensor(
            f"simplified_dense_grad_{name}",
            min_dense_grads[name],
            expected,
            atol,
            rtol,
            args.fail_fast,
        )
    failures += not _check_tensor(
        "simplified_sparse_dense_output",
        sparse_dense["min_output"],
        sparse_dense["reference_output"],
        atol,
        rtol,
        args.fail_fast,
    )
    failures += not _check_tensor(
        "simplified_sparse_dense_internal_loss_zero",
        sparse_dense["internal_sparse_loss"],
        torch.zeros_like(sparse_dense["internal_sparse_loss"]),
        0.0,
        0.0,
        args.fail_fast,
    )
    failures += not _check_tensor(
        "simplified_sparse_dense_loss",
        sparse_dense["min_loss"],
        sparse_dense["reference_loss"],
        atol,
        rtol,
        args.fail_fast,
    )
    for name, expected in sparse_dense["reference_grads"].items():
        failures += not _check_tensor(
            f"simplified_sparse_dense_grad_{name}",
            sparse_dense["min_grads"][name],
            expected,
            atol,
            rtol,
            args.fail_fast,
        )

    if world_size > 1:
        for name, tensor in (
            ("simplified_tp_replicated_sparse_loss", min_memory["loss"]),
            (
                "simplified_tp_replicated_q_wgrad",
                min_memory["grads"]["linear_q_weight"],
            ),
            ("simplified_tp_replicated_dense_loss", min_dense_loss),
            (
                "simplified_tp_replicated_dense_q_wgrad",
                min_dense_grads["linear_q_weight"],
            ),
        ):
            gathered = [torch.empty_like(tensor) for _ in range(world_size)]
            torch.distributed.all_gather(gathered, tensor.contiguous())
            for peer_rank, peer_tensor in enumerate(gathered[1:], start=1):
                failures += not _check_tensor(
                    f"{name}_rank{peer_rank}",
                    peer_tensor,
                    gathered[0],
                    atol,
                    rtol,
                    args.fail_fast,
                )
        if args.simplified_learned_k:
            for name, tensor in (
                (
                    "simplified_tp_replicated_sparse_k_wgrad",
                    min_memory["grads"]["linear_k_weight"],
                ),
                (
                    "simplified_tp_replicated_dense_k_wgrad",
                    min_dense_grads["linear_k_weight"],
                ),
            ):
                gathered = [torch.empty_like(tensor) for _ in range(world_size)]
                torch.distributed.all_gather(gathered, tensor.contiguous())
                for peer_rank, peer_tensor in enumerate(gathered[1:], start=1):
                    failures += not _check_tensor(
                        f"{name}_rank{peer_rank}",
                        peer_tensor,
                        gathered[0],
                        atol,
                        rtol,
                        args.fail_fast,
                    )

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        fail_tensor = torch.tensor([failures], device=device, dtype=torch.int32)
        torch.distributed.all_reduce(fail_tensor, op=torch.distributed.ReduceOp.SUM)
        failures = int(fail_tensor.item())
    if failures:
        print(f"\nFAIL: {failures} simplified DSA parity checks failed.", flush=True)
        return 1
    print("\nPASS: all simplified DSA parity checks passed.", flush=True)
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


def _check_topk_support_with_score_error(
    name: str,
    actual_indices: torch.Tensor,
    actual_scores: torch.Tensor,
    reference_scores: torch.Tensor,
    fail_fast: bool,
) -> bool:
    """Validate top-k support while allowing only numerically unresolved cutoff ties."""
    if (
        actual_indices.shape != actual_scores.shape
        or actual_indices.shape[:2] != reference_scores.shape[:2]
    ):
        ok = False
        detail = (
            f"shape mismatch indices={tuple(actual_indices.shape)} "
            f"scores={tuple(actual_scores.shape)} reference={tuple(reference_scores.shape)}"
        )
    else:
        batch_size, query_len, topk = actual_indices.shape
        key_len = reference_scores.size(-1)
        query_positions = torch.arange(query_len, device=actual_indices.device).view(1, -1, 1)
        in_bounds = (actual_indices >= 0) & (actual_indices < key_len)
        causal = in_bounds & (actual_indices <= query_positions)
        expected_width = torch.arange(query_len, device=actual_indices.device).add(1).clamp(
            max=topk
        )
        expected_width = expected_width.view(1, -1).expand(batch_size, -1)
        width_ok = causal.sum(dim=-1) == expected_width

        sorted_support = actual_indices.masked_fill(~causal, key_len).sort(dim=-1).values
        duplicate = (sorted_support[..., 1:] == sorted_support[..., :-1]) & (
            sorted_support[..., 1:] < key_len
        )
        unique_ok = ~duplicate.any(dim=-1)

        gather_indices = actual_indices.clamp(min=0, max=max(key_len - 1, 0))
        reference_at_actual = reference_scores.gather(-1, gather_indices)
        finite = causal & torch.isfinite(actual_scores) & torch.isfinite(reference_at_actual)
        score_error = torch.where(
            finite,
            (actual_scores.float() - reference_at_actual.float()).abs(),
            torch.zeros((), device=actual_scores.device, dtype=torch.float32),
        )
        row_error = score_error.amax(dim=-1)
        allowance = row_error + 1.0e-5

        reference_top_values, reference_top_indices = reference_scores.topk(topk, dim=-1)
        threshold_index = (expected_width - 1).unsqueeze(-1)
        reference_threshold = reference_top_values.gather(-1, threshold_index).squeeze(-1)
        selected_min = reference_at_actual.masked_fill(~causal, float("inf")).amin(dim=-1)
        optimal_ok = selected_min >= reference_threshold - allowance

        actual_support = actual_indices.masked_fill(~causal, key_len).sort(dim=-1).values
        reference_causal = reference_top_indices <= query_positions
        reference_support = reference_top_indices.masked_fill(~reference_causal, key_len).sort(
            dim=-1
        ).values
        exact_support = (actual_support == reference_support).all(dim=-1)

        if topk < key_len:
            top_plus_one = reference_scores.topk(topk + 1, dim=-1).values
            boundary_margin = top_plus_one[..., topk - 1] - top_plus_one[..., topk]
            has_omitted_causal_key = expected_width == topk
            stable = has_omitted_causal_key & (boundary_margin > 2.0 * allowance)
        else:
            stable = torch.zeros_like(exact_support)
        stable_support_ok = ~stable | exact_support

        row_ok = width_ok & unique_ok & optimal_ok & stable_support_ok
        ok = bool(row_ok.all().item())
        near_tie_rows = (~exact_support & ~stable & width_ok & unique_ok & optimal_ok).sum()
        if ok:
            detail = (
                f"near_tie_rows={int(near_tie_rows.item())} "
                f"max_score_error={float(row_error.max().item()):.6e}"
            )
        else:
            batch_idx, query_idx = (int(v) for v in (~row_ok).nonzero()[0])
            actual_row = set(actual_support[batch_idx, query_idx].tolist()) - {key_len}
            reference_row = set(reference_support[batch_idx, query_idx].tolist()) - {key_len}
            detail = (
                f"first invalid row=({batch_idx}, {query_idx}) "
                f"missing={sorted(reference_row - actual_row)[:8]} "
                f"extra={sorted(actual_row - reference_row)[:8]} "
                f"selected_min={float(selected_min[batch_idx, query_idx].item()):.6e} "
                f"threshold={float(reference_threshold[batch_idx, query_idx].item()):.6e} "
                f"allowance={float(allowance[batch_idx, query_idx].item()):.6e}"
            )

    status = "OK " if ok else "BAD"
    print(f"{status} {name:<36} shape={tuple(actual_indices.shape)} {detail}", flush=True)
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


def _attention_aux_input_norm(args, device: torch.device, dtype: torch.dtype):
    if args.simplified_input_norm == "none":
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 2719)
    weight = torch.randn(
        args.hidden_size, device=device, dtype=dtype, generator=generator
    )
    bias = (
        torch.randn(args.hidden_size, device=device, dtype=dtype, generator=generator)
        if args.simplified_input_norm == "layernorm"
        else None
    )
    return SimpleNamespace(
        normalization=(
            "LayerNorm" if args.simplified_input_norm == "layernorm" else "RMSNorm"
        ),
        weight=weight,
        bias=bias,
        eps=args.layernorm_eps,
        zero_centered_gamma=args.zero_centered_gamma,
    )


def _attention_aux_support(
    case: Case,
    args,
    indexer,
    simplified_input_norm,
    use_triton: bool,
) -> torch.Tensor:
    query_chunk_size = min(args.query_block_size, args.seq_len)
    key_chunk_size = _routing_key_chunk_size(
        args.key_block_size, args.seq_len, use_triton
    )
    support = []
    with torch.no_grad(), _triton_dispatch_enabled(use_triton):
        input_norm_stats = (
            None
            if args.attention_aux_simplified
            else _indexer_input_norm_stats(case.hidden_states, simplified_input_norm)
        )
        for q_start in range(0, args.seq_len, query_chunk_size):
            q_end = min(q_start + query_chunk_size, args.seq_len)
            if args.attention_aux_simplified:
                _, indices, _ = _simplified_topk_index_tile(
                    case.hidden_states,
                    case.key,
                    q_start,
                    q_end,
                    case.linear_q_weight,
                    args.aux_topk,
                    args.indexer_head_dim,
                    args.indexer_rotary_dim,
                    indexer.rotary_pos_emb,
                    args.rotary_interleaved,
                    args.indexer_rotary_dim > 0,
                    indexer.softmax_scale,
                    key_chunk_size,
                    simplified_input_norm,
                    linear_k_weight=(
                        case.linear_k_weight if args.simplified_learned_k else None
                    ),
                )
            else:
                _, indices, _, _ = _topk_index_tile(
                    case.hidden_states,
                    q_start,
                    q_end,
                    case.linear_q_weight,
                    case.linear_k_weight,
                    case.k_norm_weight,
                    case.k_norm_bias,
                    True,
                    case.linear_weights_weight,
                    args.layernorm_eps,
                    args.indexer_heads,
                    args.indexer_head_dim,
                    args.aux_topk,
                    args.indexer_rotary_dim,
                    indexer.rotary_pos_emb,
                    args.rotary_interleaved,
                    args.indexer_rotary_dim > 0,
                    args.hadamard,
                    key_chunk_size,
                    simplified_input_norm,
                    input_norm_stats,
                )
            support.append(indices)
    return torch.cat(support, dim=1)


def _attention_aux_oracle(
    case: Case,
    support: torch.Tensor,
    args,
    world_size: int,
    attention_softmax_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    repeat_factor = args.num_query_heads // args.num_query_groups
    key_heads = case.key.repeat_interleave(repeat_factor, dim=2)
    value_heads = case.value.repeat_interleave(repeat_factor, dim=2)
    scores = torch.einsum(
        "qbhd,kbhd->bhqk", case.query.float(), key_heads.float()
    ) * attention_softmax_scale
    positions = torch.arange(args.seq_len, device=case.query.device)
    causal_invalid = positions.view(1, 1, 1, -1) > positions.view(1, 1, -1, 1)
    probabilities = torch.softmax(
        scores.masked_fill(causal_invalid, float("-inf")), dim=-1, dtype=torch.float32
    )
    gather_index = support[:, None].expand(-1, args.num_query_heads, -1, -1)
    captured_mass = torch.gather(probabilities, -1, gather_index).sum(dim=-1)
    dense_output = torch.einsum(
        "bhqk,kbhd->qbhd",
        probabilities.to(case.value.dtype).float(),
        value_heads.float(),
    ).to(case.value.dtype).float()
    with _triton_dispatch_enabled(False):
        sparse_output = _sparse_attention_tile(
            case.query,
            case.key,
            case.value,
            support,
            attention_softmax_scale,
            0,
        ).float()
    total_entries = (
        args.seq_len * args.batch_size * args.num_query_heads * world_size
    )
    mass_loss = (
        torch.relu(args.mass_target - captured_mass).square().sum()
        * args.mass_loss_coeff
        / total_entries
    )
    denominator = dense_output.detach().square().sum(dim=-1).clamp_min(1.0e-12)
    output_loss = (
        (
            (sparse_output - dense_output.detach()).square().sum(dim=-1)
            / denominator
        ).sum()
        * args.output_consistency_loss_coeff
        / total_entries
    )
    return mass_loss, output_loss, captured_mass.sum() / total_entries


def _run_attention_aux_parity(
    args, device: torch.device, dtype: torch.dtype
) -> int:
    if args.mass_loss_coeff < 0.0 or args.output_consistency_loss_coeff < 0.0:
        raise SystemExit("Attention auxiliary loss coefficients must be non-negative.")
    if args.mass_loss_coeff > 0.0 and not 0.0 < args.mass_target <= 1.0:
        raise SystemExit("--mass-target must be in (0, 1].")
    if args.mass_loss_coeff <= 0.0 and args.output_consistency_loss_coeff <= 0.0:
        raise SystemExit("attention-aux mode requires at least one positive loss coefficient.")
    if args.aux_topk is None:
        args.aux_topk = args.topk
    if args.aux_topk <= 0 or args.aux_topk > args.topk:
        raise SystemExit("--aux-topk must be in [1, --topk].")
    if args.query_block_size < args.aux_topk:
        raise SystemExit(
            "attention-aux parity currently requires --query-block-size >= --aux-topk."
        )
    if args.attention_aux_simplified:
        if args.num_query_groups != 1 or args.indexer_heads != 1:
            raise SystemExit(
                "Simplified attention-aux parity requires one query group and one indexer head."
            )
        if not args.simplified_learned_k and args.indexer_head_dim != args.head_dim:
            raise SystemExit(
                "Main-K simplified attention-aux parity requires matching head dimensions."
            )
        if args.hadamard:
            raise SystemExit("Simplified DSA does not use Hadamard.")
    elif args.simplified_learned_k:
        raise SystemExit("--simplified-learned-k requires --attention-aux-simplified.")

    use_triton = args.backend == "triton-min-memory"
    device, rank, world_size, pg_collection = _configure_distributed_for_dense_mode(
        device, args.distributed_backend
    )
    atol = args.atol if args.atol is not None else (5.0e-2 if dtype != torch.float32 else 5.0e-4)
    rtol = args.rtol if args.rtol is not None else (5.0e-2 if dtype != torch.float32 else 5.0e-4)
    print(
        f"DSA attention-aux parity backend={args.backend} rank={rank}/{world_size} "
        f"device={device} dtype={dtype} simplified={args.attention_aux_simplified} "
        f"learned_k={args.simplified_learned_k} S={args.seq_len} "
        f"Hq={args.num_query_heads} G={args.num_query_groups} "
        f"routing_topk={args.topk} aux_topk={args.aux_topk}",
        flush=True,
    )

    base = _make_case(args, device, dtype)
    actual_case = _clone_case(base)
    reference_case = _clone_case(base)
    simplified_input_norm = (
        _attention_aux_input_norm(args, device, dtype)
        if args.attention_aux_simplified
        else _standard_input_norm(args, device, dtype)
    )
    actual_indexer = (
        _simplified_indexer_from_case(actual_case, args, pg_collection)
        if args.attention_aux_simplified
        else _indexer_from_case(actual_case, args, pg_collection)
    )
    reference_indexer = (
        _simplified_indexer_from_case(reference_case, args, pg_collection)
        if args.attention_aux_simplified
        else _indexer_from_case(reference_case, args, pg_collection)
    )
    attention_softmax_scale = (
        args.attention_softmax_scale
        if args.attention_softmax_scale is not None
        else args.head_dim**-0.5
    )
    mass_loss, output_loss, captured_mass = dsa_main_attention_aux_loss(
        query=actual_case.query,
        key=actual_case.key,
        value=actual_case.value,
        hidden_states=actual_case.hidden_states,
        indexer=actual_indexer,
        attention_softmax_scale=attention_softmax_scale,
        use_indexer_rope=args.indexer_rotary_dim > 0,
        aux_topk=args.aux_topk,
        mass_loss_coeff=args.mass_loss_coeff,
        mass_target=args.mass_target,
        output_loss_coeff=args.output_consistency_loss_coeff,
        query_chunk_size=args.query_block_size,
        key_chunk_size=args.key_block_size,
        simplified_input_norm=simplified_input_norm,
        use_triton=use_triton,
    )
    actual_loss = mass_loss if args.mass_loss_coeff > 0.0 else output_loss
    if args.mass_loss_coeff > 0.0 and args.output_consistency_loss_coeff > 0.0:
        actual_loss = mass_loss + output_loss
    actual_loss.backward()

    support = _attention_aux_support(
        reference_case,
        args,
        reference_indexer,
        simplified_input_norm,
        use_triton,
    )
    reference_mass_loss, reference_output_loss, reference_captured_mass = (
        _attention_aux_oracle(
            reference_case,
            support,
            args,
            world_size,
            attention_softmax_scale,
        )
    )
    reference_loss = (
        reference_mass_loss
        if args.mass_loss_coeff > 0.0
        else reference_output_loss
    )
    if args.mass_loss_coeff > 0.0 and args.output_consistency_loss_coeff > 0.0:
        reference_loss = reference_mass_loss + reference_output_loss
    reference_loss.backward()

    failures = 0
    failures += not _check_tensor(
        "attention_aux_mass_loss",
        mass_loss,
        reference_mass_loss,
        atol,
        rtol,
        args.fail_fast,
    )
    failures += not _check_tensor(
        "attention_aux_output_loss",
        output_loss,
        reference_output_loss,
        atol,
        rtol,
        args.fail_fast,
    )
    failures += not _check_tensor(
        "attention_aux_captured_mass",
        captured_mass,
        reference_captured_mass,
        atol,
        rtol,
        args.fail_fast,
    )
    for name in ("query", "key", "value"):
        actual_grad = getattr(actual_case, name).grad
        reference_grad = getattr(reference_case, name).grad
        if actual_grad is None or reference_grad is None:
            ok = actual_grad is None and reference_grad is None
            print(
                f"{'OK ' if ok else 'BAD'} attention_aux_grad_{name:<25} "
                f"actual={actual_grad} reference={reference_grad}",
                flush=True,
            )
            failures += not ok
        else:
            failures += not _check_tensor(
                f"attention_aux_grad_{name}",
                actual_grad,
                reference_grad,
                atol,
                rtol,
                args.fail_fast,
            )
    for name in (
        "linear_q_weight",
        "linear_k_weight",
        "k_norm_weight",
        "k_norm_bias",
        "linear_weights_weight",
    ):
        if getattr(actual_case, name).grad is not None:
            print(f"BAD attention_aux_detached_{name}: unexpected gradient", flush=True)
            failures += 1

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        fail_tensor = torch.tensor([failures], device=device, dtype=torch.int32)
        torch.distributed.all_reduce(fail_tensor, op=torch.distributed.ReduceOp.SUM)
        failures = int(fail_tensor.item())
    if failures:
        print(f"\nFAIL: {failures} attention auxiliary parity checks failed.", flush=True)
        return 1
    print("\nPASS: all attention auxiliary parity checks passed.", flush=True)
    return 0


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "sparse",
            "dense-warmup",
            "dense-vs-full-topk",
            "sparse-fwd-dense-loss",
            "simplified",
            "attention-aux",
        ),
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
    parser.add_argument(
        "--aux-topk",
        type=int,
        default=None,
        help="Selected support size for attention-aux mode; defaults to --topk.",
    )
    parser.add_argument("--mass-loss-coeff", type=float, default=0.7)
    parser.add_argument("--mass-target", type=float, default=0.95)
    parser.add_argument("--output-consistency-loss-coeff", type=float, default=0.4)
    parser.add_argument(
        "--attention-aux-simplified",
        action="store_true",
        help="Exercise the simplified router in attention-aux mode.",
    )
    parser.add_argument("--indexer-rotary-dim", type=int, default=0)
    parser.add_argument("--rotary-interleaved", action="store_true")
    parser.add_argument(
        "--simplified-learned-k",
        action="store_true",
        help="Use a separate learned simplified-DSA K projection in simplified mode.",
    )
    parser.add_argument(
        "--simplified-input-norm",
        choices=("none", "rmsnorm", "layernorm"),
        default="none",
        help=(
            "Synthetic fused main-Q input norm used by simplified-mode parity checks. "
            "Use none to exercise --dsa-simplified-indexer-disable-main-input-norm math."
        ),
    )
    parser.add_argument(
        "--standard-input-norm",
        choices=("none", "rmsnorm", "layernorm"),
        default="none",
        help=(
            "Synthetic detached main-Q input norm used by standard-DSA parity checks. "
            "This exercises --dsa-standard-indexer-use-main-input-norm semantics."
        ),
    )
    parser.add_argument(
        "--zero-centered-gamma",
        action="store_true",
        help="Interpret the selected synthetic input-norm weight as zero-centered gamma.",
    )
    parser.add_argument("--hadamard", action="store_true")
    parser.add_argument("--loss-coeff", type=float, default=0.7)
    parser.add_argument(
        "--attention-softmax-scale",
        type=float,
        default=None,
        help=(
            "Optional main-attention/teacher softmax scale. Simplified routing remains fixed "
            "at head_dim**-0.5."
        ),
    )
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
        help="Distributed backend for TP parity checks. Defaults to nccl on CUDA.",
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
    if args.mode == "simplified":
        return _run_simplified_parity(args, device, dtype)
    if args.mode == "attention-aux":
        return _run_attention_aux_parity(args, device, dtype)

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
        f"hadamard={args.hadamard} rotary_dim={args.indexer_rotary_dim} "
        f"input_norm={args.standard_input_norm}",
        flush=True,
    )

    base = _make_case(args, device, dtype)
    min_case = _clone_case(base)
    comp_case = _clone_case(base)

    components = _min_memory_components(comp_case, args, rotary_pos_emb, use_triton)
    ref_case = _clone_case(base)
    # Top-k is discontinuous at BF16 rounding boundaries. Validate Triton RSTD
    # independently below, then hold it fixed while checking the remaining pipeline.
    with _triton_dispatch_enabled(False):
        reference = _reference_run(
            ref_case,
            args,
            rotary_pos_emb,
            topk_override=components["topk_indices"],
            input_norm_stats=components["input_norm_stats"],
        )
    min_memory = _min_memory_run(min_case, args, rotary_pos_emb, use_triton)

    failures = 0
    if components["input_norm_stats"] is not None:
        with torch.no_grad():
            hidden_float = comp_case.hidden_states.float()
            reference_rstd = torch.rsqrt(
                hidden_float.square().mean(dim=-1) + args.layernorm_eps
            )
    if components["input_norm_stats"] is not None:
        failures += not _check_tensor(
            "input_norm_rstd",
            components["input_norm_stats"],
            reference_rstd,
            2.0e-5,
            2.0e-5,
            args.fail_fast,
        )
    failures += not _check_tensor(
        "q_index", components["q_index"], reference["q_index"], atol, rtol, args.fail_fast
    )
    failures += not _check_tensor(
        "weights", components["weights"], reference["weights"], atol, rtol, args.fail_fast
    )
    failures += not _check_topk_support_with_score_error(
        "topk_support",
        components["topk_indices"],
        components["topk_scores"],
        reference["index_scores"],
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
