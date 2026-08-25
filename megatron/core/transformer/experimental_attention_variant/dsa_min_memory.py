# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Minimum-activation DSA-GQA training path.

This module keeps a tile-local PyTorch implementation as the correctness oracle and dispatches to
optional Triton kernels for CUDA-supported tiles.  By default, this file does not save full DSA
routing scores, top-k tensors, sparse masks, selected K/V, or attention probabilities across the
forward/backward boundary.  Explicit cache flags may trade bounded sequence-sized tensors for speed.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from megatron.core.models.common.embeddings.rope_utils import _rotate_half
from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import (
    YarnRotaryEmbedding,
    _yarn_find_correction_range,
    _yarn_get_concentration_factor,
    _yarn_linear_ramp_mask,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.experimental_attention_variant.dsa import rotate_activation
from megatron.core.transformer.experimental_attention_variant.dsa_min_memory_triton import (
    set_min_memory_triton_enabled,
    triton_indexer_loss_grad,
    triton_linear_wgrad,
    triton_scatter_selected_grad_to_sequence,
    triton_simplified_gathered_linear_wgrad,
    triton_simplified_index_scores_block,
    triton_simplified_input_norm_stats,
    triton_simplified_selected_index_scores,
    triton_simplified_selected_index_scores_backward,
    triton_simplified_selected_index_scores_backward_qk,
    triton_sparse_attention_backward_accumulate,
    triton_sparse_attention_backward_path,
    triton_sparse_attention_backward_supported,
    triton_sparse_attention_tile,
    triton_teacher_scores_tile,
    triton_topk_index_block,
)

# Optional cuDNN DSA kernels, used to A/B the indexer (scores + top-k) against
# the Triton min-memory kernels. Gated at runtime by `use_cudnn`.
try:
    from cudnn import DSA as _DSA
except ImportError:
    try:
        from cudnn.deepseek_sparse_attention import DSA as _DSA
    except ImportError:
        _DSA = None


_SIMPLIFIED_LEARNED_K_SUPPORT_CHUNK_SIZE = 64


def _module_weight(module) -> torch.Tensor:
    weight = getattr(module, "weight", None)
    if weight is None:
        raise RuntimeError(f"{module.__class__.__name__} does not expose a weight tensor.")
    return weight




def _grad_accumulator(tensor: torch.Tensor) -> torch.Tensor:
    dtype = torch.float32 if tensor.dtype in (torch.float16, torch.bfloat16) else tensor.dtype
    return torch.zeros(tensor.shape, device=tensor.device, dtype=dtype)


def _distributed_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


class _DSATimingProfiler:
    _MEDIAN_WINDOW = 5
    # keyed by label: list of completed (fwd, bwd) pairs, each entry is (totals, counts, order)
    _paired_history: Dict[str, List[Dict[str, Tuple[Dict[str, float], Dict[str, int], List[str]]]]] = {}
    # keyed by label: pending forward data waiting to be paired with a backward
    _pending_fwd: Dict[str, Tuple[Dict[str, float], Dict[str, int], List[str]]] = {}

    def __init__(
        self,
        enabled: bool,
        profile_rank: int,
        label: str,
        default_device: torch.device,
    ) -> None:
        rank = _distributed_rank()
        self.enabled = bool(enabled) and (profile_rank < 0 or rank == profile_rank)
        self.rank = rank
        self.label = label
        self.default_device = default_device
        self.records = []

    @contextmanager
    def record(self, name: str, device: Optional[torch.device] = None):
        if not self.enabled:
            yield
            return

        if device is None:
            device = self.default_device
        if device.type == "cuda" and torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            with torch.cuda.device(device):
                start_event.record()
            try:
                yield
            finally:
                with torch.cuda.device(device):
                    end_event.record()
                self.records.append((name, start_event, end_event, device))
            return

        start_time = time.perf_counter()
        try:
            yield
        finally:
            self.records.append((name, (time.perf_counter() - start_time) * 1000.0, None, None))

    def log(self, phase: str) -> None:
        if not self.enabled or not self.records:
            return

        synchronized_devices = set()
        totals = {}
        counts = {}
        order = []
        for name, start, end, device in self.records:
            if device is not None:
                device_key = (device.type, device.index)
                if device_key not in synchronized_devices:
                    torch.cuda.synchronize(device)
                    synchronized_devices.add(device_key)
                elapsed_ms = start.elapsed_time(end)
            else:
                elapsed_ms = start
            if name not in totals:
                totals[name] = 0.0
                counts[name] = 0
                order.append(name)
            totals[name] += elapsed_ms
            counts[name] += 1

        label = f" {self.label}" if self.label else ""
        parts = " ".join(
            f"{name}={totals[name]:.3f}ms(avg={totals[name]/counts[name]:.3f}ms)"
            for name in order
        )
        print(f"[rank{self.rank}] DSA min-memory {phase}{label}: {parts}", flush=True)

        _csv_exclude = {"selected_index_scores_fwd_score_fallback"}

        def _csv_values(p: str, t: Dict[str, float], c: Dict[str, int], o: List[str]) -> str:
            cols = [n for n in o if n not in _csv_exclude]
            return ",".join([p, self.label] + [f"{t[n]:.3f}" for n in cols] + [f"{t[n]/c[n]:.3f}" for n in cols])

        def _csv_header(o: List[str]) -> str:
            cols = [n for n in o if n not in _csv_exclude]
            return ",".join(["phase", "label"] + [f"{n}_total_ms" for n in cols] + [f"{n}_avg_ms" for n in cols])

        if phase == "forward":
            _DSATimingProfiler._pending_fwd[self.label] = (totals, counts, order)
        elif phase == "backward" and self.label in _DSATimingProfiler._pending_fwd:
            fwd_data = _DSATimingProfiler._pending_fwd.pop(self.label)
            pairs = _DSATimingProfiler._paired_history.setdefault(self.label, [])
            pairs.append({"forward": fwd_data, "backward": (totals, counts, order)})
            if len(pairs) >= _DSATimingProfiler._MEDIAN_WINDOW:
                sorted_pairs = sorted(pairs, key=lambda p: p["forward"][0].get("forward_total", 0.0))
                med = sorted_pairs[len(sorted_pairs) // 2]
                ft, fc, fo = med["forward"]
                bt, bc, bo = med["backward"]
                print(
                    f"[rank{self.rank}] CSV (median fwd of {_DSATimingProfiler._MEDIAN_WINDOW}):\n"
                    f"{_csv_header(fo)}\n"
                    f"{_csv_values('forward', ft, fc, fo)}\n"
                    f"{_csv_values('backward', bt, bc, bo)}",
                    flush=True,
                )
                _DSATimingProfiler._paired_history[self.label] = []


@contextmanager
def _profile_record(
    profile: Optional[_DSATimingProfiler], name: str, device: torch.device
):
    if profile is None:
        yield
        return
    with profile.record(name, device):
        yield


@contextmanager
def _triton_dispatch_enabled(enabled: bool):
    previous = set_min_memory_triton_enabled(enabled)
    try:
        yield
    finally:
        set_min_memory_triton_enabled(previous)


def _default_query_chunk_size(query_length: int) -> int:
    return min(query_length, 512)


def _default_key_chunk_size(key_length: int) -> int:
    return min(key_length, 1024)






def _chunk_size(config_value: Optional[int], default_value: int, maximum: int) -> int:
    if config_value is None or config_value <= 0:
        return min(default_value, maximum)
    return min(config_value, maximum)


def _routing_key_chunk_size(
    config_value: Optional[int], key_length: int, use_triton: bool
) -> int:
    if not use_triton:
        # The PyTorch backend is the numerical oracle. Streaming torch.topk over key chunks is
        # not tie-equivalent to a single full torch.topk. Exact zero ties are common in standard
        # DSA after ReLU, and simplified routing can also contain equal scores. Use one key block
        # so torch-min-memory preserves reference routing for both indexer formulations.
        return key_length
    return _chunk_size(config_value, _default_key_chunk_size(key_length), key_length)




def _linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.linear(x, weight, None)




def _rope_inv_freq_and_mscale(rotary_pos_emb, device: torch.device) -> Tuple[torch.Tensor, float]:
    if isinstance(rotary_pos_emb, YarnRotaryEmbedding):
        if rotary_pos_emb.inv_freq_extra.device.type == "cpu":
            rotary_pos_emb.inv_freq_extra = rotary_pos_emb.inv_freq_extra.to(device=device)
        if rotary_pos_emb.inv_freq_inter.device.type == "cpu":
            rotary_pos_emb.inv_freq_inter = rotary_pos_emb.inv_freq_inter.to(device=device)

        low, high = _yarn_find_correction_range(
            rotary_pos_emb.beta_fast,
            rotary_pos_emb.beta_slow,
            rotary_pos_emb.dim,
            rotary_pos_emb.rotary_base,
            rotary_pos_emb.original_max_position_embeddings,
            rotary_pos_emb.correction_range_round_to_int,
        )
        inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(
            low, high, rotary_pos_emb.dim // 2, device=rotary_pos_emb.inv_freq_extra.device
        ).to(dtype=torch.float32)
        inv_freq = (
            rotary_pos_emb.inv_freq_inter * (1 - inv_freq_mask)
            + rotary_pos_emb.inv_freq_extra * inv_freq_mask
        )
        mscale = _yarn_get_concentration_factor(
            rotary_pos_emb.scaling_factor, rotary_pos_emb.mscale, rotary_pos_emb.mscale_all_dim
        )
        return inv_freq, mscale

    if rotary_pos_emb.inv_freq.device.type == "cpu":
        rotary_pos_emb.inv_freq = rotary_pos_emb.inv_freq.to(device=device)
    return rotary_pos_emb.inv_freq, 1.0




def _apply_rope_at_positions(
    x: torch.Tensor,
    positions: torch.Tensor,
    index_head_dim: int,
    index_rotary_dim: int,
    rotary_pos_emb,
    rotary_interleaved: bool,
) -> torch.Tensor:
    if rotary_pos_emb is None or index_rotary_dim == 0:
        return x

    x_nope, x_pe = torch.split(x, [index_head_dim - index_rotary_dim, index_rotary_dim], dim=-1)
    inv_freq, mscale = _rope_inv_freq_and_mscale(rotary_pos_emb, x.device)
    inv_freq = inv_freq[: index_rotary_dim // 2]

    positions = positions.to(device=inv_freq.device, dtype=inv_freq.dtype)
    interpolation_factor = getattr(rotary_pos_emb, "seq_len_interpolation_factor", None)
    if interpolation_factor is not None and not isinstance(rotary_pos_emb, YarnRotaryEmbedding):
        positions = positions * (1 / interpolation_factor)
    freqs = positions.unsqueeze(-1) * inv_freq
    if not getattr(rotary_pos_emb, "rotary_interleaved", False):
        freqs = torch.cat((freqs, freqs), dim=-1)
    else:
        freqs = torch.stack((freqs, freqs), dim=-1).flatten(start_dim=-2)
    while freqs.dim() < x_pe.dim():
        freqs = freqs.unsqueeze(-2)

    cos = (torch.cos(freqs) * mscale).to(dtype=x_pe.dtype, device=x_pe.device)
    sin = (torch.sin(freqs) * mscale).to(dtype=x_pe.dtype, device=x_pe.device)
    x_pe = x_pe * cos + _rotate_half(x_pe, rotary_interleaved) * sin
    return torch.cat((x_nope, x_pe), dim=-1)


def _apply_rope_backward_at_positions(
    grad: torch.Tensor,
    positions: torch.Tensor,
    index_head_dim: int,
    index_rotary_dim: int,
    rotary_pos_emb,
    rotary_interleaved: bool,
) -> torch.Tensor:
    if rotary_pos_emb is None or index_rotary_dim == 0:
        return grad

    grad_nope, grad_pe = torch.split(
        grad, [index_head_dim - index_rotary_dim, index_rotary_dim], dim=-1
    )
    inv_freq, mscale = _rope_inv_freq_and_mscale(rotary_pos_emb, grad.device)
    inv_freq = inv_freq[: index_rotary_dim // 2]

    positions = positions.to(device=inv_freq.device, dtype=inv_freq.dtype)
    interpolation_factor = getattr(rotary_pos_emb, "seq_len_interpolation_factor", None)
    if interpolation_factor is not None and not isinstance(rotary_pos_emb, YarnRotaryEmbedding):
        positions = positions * (1 / interpolation_factor)
    freqs = positions.unsqueeze(-1) * inv_freq
    if not getattr(rotary_pos_emb, "rotary_interleaved", False):
        freqs = torch.cat((freqs, freqs), dim=-1)
    else:
        freqs = torch.stack((freqs, freqs), dim=-1).flatten(start_dim=-2)
    while freqs.dim() < grad_pe.dim():
        freqs = freqs.unsqueeze(-2)

    cos = (torch.cos(freqs) * mscale).to(dtype=grad_pe.dtype, device=grad_pe.device)
    sin = (torch.sin(freqs) * mscale).to(dtype=grad_pe.dtype, device=grad_pe.device)
    grad_pe = grad_pe * cos - _rotate_half(grad_pe, rotary_interleaved) * sin
    return torch.cat((grad_nope, grad_pe), dim=-1)


def _backward_indexer_transform(
    grad: torch.Tensor,
    positions: torch.Tensor,
    index_head_dim: int,
    index_rotary_dim: int,
    rotary_pos_emb,
    rotary_interleaved: bool,
    use_indexer_rope: bool,
    use_hadamard: bool,
) -> torch.Tensor:
    if use_hadamard:
        grad = rotate_activation(grad.to(dtype=torch.bfloat16)).to(dtype=torch.float32)
    if use_indexer_rope:
        grad = _apply_rope_backward_at_positions(
            grad,
            positions,
            index_head_dim,
            index_rotary_dim,
            rotary_pos_emb,
            rotary_interleaved,
        )
    return grad








def _gather_selected_indexer_k(
    full_k_index: torch.Tensor,
    topk_indices: torch.Tensor,
) -> torch.Tensor:
    batch_size = topk_indices.size(0)
    k_by_batch = full_k_index.permute(1, 0, 2)
    batch_index = torch.arange(batch_size, device=topk_indices.device).view(batch_size, 1, 1)
    return k_by_batch[batch_index, topk_indices]










def _mask_dense_causal_scores(
    scores: torch.Tensor,
    q_start: int,
    q_end: int,
    k_start: int,
    k_end: int,
) -> torch.Tensor:
    invalid = _causal_invalid_mask(q_start, q_end, k_start, k_end, scores.device)
    while invalid.dim() < scores.dim():
        invalid = invalid.unsqueeze(0)
    return scores.masked_fill(invalid, float("-inf"))


def _dense_teacher_logits_block(
    query_tile: torch.Tensor,
    key_block: torch.Tensor,
    softmax_scale: float,
    q_start: int,
    k_start: int,
) -> torch.Tensor:
    q_len = query_tile.size(0)
    k_len = key_block.size(0)
    num_query_heads = query_tile.size(2)
    num_query_groups = key_block.size(2)
    assert num_query_heads % num_query_groups == 0, (
        f"num_query_heads ({num_query_heads}) must be divisible by "
        f"num_query_groups ({num_query_groups})."
    )
    repeat_factor = num_query_heads // num_query_groups
    blocks = []
    for group_idx in range(num_query_groups):
        head_start = group_idx * repeat_factor
        head_end = head_start + repeat_factor
        q = query_tile[:, :, head_start:head_end, :].permute(1, 2, 0, 3)
        k = key_block[:, :, group_idx, :].permute(1, 0, 2)
        blocks.append(torch.einsum("brqd,bkd->brqk", q.float(), k.float()) * softmax_scale)
    scores = torch.cat(blocks, dim=1)
    return _mask_dense_causal_scores(scores, q_start, q_start + q_len, k_start, k_start + k_len)


def _update_running_softmax_stats(
    logits: torch.Tensor,
    running_max: Optional[torch.Tensor],
    running_sum: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    block_max = logits.max(dim=-1).values
    if running_max is None or running_sum is None:
        running_max = block_max
        running_sum = torch.exp(logits - running_max.unsqueeze(-1)).sum(dim=-1)
        return running_max, running_sum

    new_max = torch.maximum(running_max, block_max)
    running_sum = running_sum * torch.exp(running_max - new_max) + torch.exp(
        logits - new_max.unsqueeze(-1)
    ).sum(dim=-1)
    return new_max, running_sum




def _dense_teacher_mass_block(
    query_tile: torch.Tensor,
    key_block: torch.Tensor,
    teacher_max: torch.Tensor,
    teacher_sum: torch.Tensor,
    softmax_scale: float,
    q_start: int,
    k_start: int,
    pg_collection: ProcessGroupCollection,
) -> torch.Tensor:
    teacher_logits = _dense_teacher_logits_block(
        query_tile, key_block, softmax_scale, q_start, k_start
    )
    teacher = torch.exp(teacher_logits - teacher_max.unsqueeze(-1)) / teacher_sum.unsqueeze(-1)
    teacher = teacher.sum(dim=1)
    if pg_collection.tp.size() > 1:
        teacher = teacher.contiguous()
        torch.distributed.all_reduce(teacher, group=pg_collection.tp)
    return teacher


def _dense_teacher_norm(
    query_tile: torch.Tensor,
    key: torch.Tensor,
    teacher_max: torch.Tensor,
    teacher_sum: torch.Tensor,
    softmax_scale: float,
    q_start: int,
    key_chunk_size: int,
    pg_collection: ProcessGroupCollection,
) -> torch.Tensor:
    teacher_norm = query_tile.new_zeros(
        (query_tile.size(1), query_tile.size(0)), dtype=torch.float32
    )
    for k_start in range(0, key.size(0), key_chunk_size):
        k_end = min(k_start + key_chunk_size, key.size(0))
        teacher_mass = _dense_teacher_mass_block(
            query_tile,
            key[k_start:k_end],
            teacher_max,
            teacher_sum,
            softmax_scale,
            q_start,
            k_start,
            pg_collection,
        )
        teacher_norm = teacher_norm + teacher_mass.sum(dim=-1)
    return teacher_norm.clamp_min(1.0e-20)








def _accumulate_linear_weight_grad(
    grad_weight: Optional[torch.Tensor],
    grad_output: torch.Tensor,
    input_tensor: torch.Tensor,
) -> None:
    if grad_weight is None:
        return
    if triton_linear_wgrad(grad_output, input_tensor, grad_weight):
        return
    grad_2d = grad_output.reshape(-1, grad_output.size(-1)).to(dtype=torch.float32)
    input_2d = input_tensor.reshape(-1, input_tensor.size(-1)).to(dtype=torch.float32)
    grad_weight.add_(grad_2d.t().matmul(input_2d))




def _causal_invalid_mask(
    q_start: int,
    q_end: int,
    k_start: int,
    k_end: int,
    device: torch.device,
) -> torch.Tensor:
    query_positions = torch.arange(q_start, q_end, device=device, dtype=torch.long)
    key_positions = torch.arange(k_start, k_end, device=device, dtype=torch.long)
    return key_positions.view(1, k_end - k_start) > query_positions.view(q_end - q_start, 1)


def _selected_causal_invalid_mask(
    topk_indices: torch.Tensor,
    q_start: int,
) -> torch.Tensor:
    query_positions = torch.arange(
        q_start,
        q_start + topk_indices.size(1),
        device=topk_indices.device,
        dtype=topk_indices.dtype,
    )
    return topk_indices > query_positions.view(1, topk_indices.size(1), 1)


def _merge_topk(
    running_scores: Optional[torch.Tensor],
    running_indices: Optional[torch.Tensor],
    block_scores: torch.Tensor,
    block_indices: torch.Tensor,
    topk: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if running_scores is None or running_indices is None:
        return block_scores, block_indices
    merged_scores = torch.cat((running_scores, block_scores), dim=-1)
    merged_indices = torch.cat((running_indices, block_indices), dim=-1)
    keep = merged_scores.topk(min(topk, merged_scores.size(-1)), dim=-1).indices
    return torch.gather(merged_scores, -1, keep), torch.gather(merged_indices, -1, keep)


def _sort_topk_support_by_position(
    scores: torch.Tensor,
    indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    order = indices.argsort(dim=-1)
    return torch.gather(scores, -1, order), torch.gather(indices, -1, order)














def _gather_selected_kv(
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


def _sparse_attention_tile(
    query_tile: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    q_start: int,
) -> torch.Tensor:
    triton_output = triton_sparse_attention_tile(
        query_tile, key, value, topk_indices, softmax_scale, q_start
    )
    if triton_output is not None:
        return triton_output

    query_length, batch_size, num_query_heads, head_dim = query_tile.shape
    num_query_groups = key.size(2)
    repeat_factor = num_query_heads // num_query_groups
    value_head_dim = value.size(-1)
    output = value.new_empty(
        (query_length, batch_size, num_query_heads, value_head_dim), dtype=value.dtype
    )
    selected_invalid = _selected_causal_invalid_mask(topk_indices, q_start).unsqueeze(1)

    for group_idx in range(num_query_groups):
        head_start = group_idx * repeat_factor
        head_end = head_start + repeat_factor
        query_group = query_tile[:, :, head_start:head_end, :].permute(1, 2, 0, 3)
        selected_key = _gather_selected_kv(key, group_idx, topk_indices)
        selected_value = _gather_selected_kv(value, group_idx, topk_indices)
        scores = (
            torch.einsum("brqd,bqkd->brqk", query_group.float(), selected_key.float())
            * softmax_scale
        )
        scores = scores.masked_fill(selected_invalid, float("-inf"))
        probs = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32)
        group_output = torch.einsum(
            "brqk,bqkd->brqd", probs.to(selected_value.dtype), selected_value
        )
        output[:, :, head_start:head_end, :] = group_output.permute(2, 0, 1, 3)

    return output


def _teacher_scores_tile(
    query_tile: torch.Tensor,
    key: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    q_start: int,
    pg_collection: ProcessGroupCollection,
    profile: Optional[_DSATimingProfiler] = None,
    profile_suffix: str = "fwd",
) -> torch.Tensor:
    _, batch_size, num_query_heads, _ = query_tile.shape
    num_query_groups = key.size(2)
    repeat_factor = num_query_heads // num_query_groups
    with _profile_record(profile, f"teacher_scores_{profile_suffix}_compute", query_tile.device):
        teacher = triton_teacher_scores_tile(
            query_tile, key, topk_indices, softmax_scale, q_start
        )
        if teacher is None:
            teacher = query_tile.new_zeros(
                (batch_size, topk_indices.size(1), topk_indices.size(2)), dtype=torch.float32
            )
            selected_invalid = _selected_causal_invalid_mask(topk_indices, q_start).unsqueeze(1)

            for group_idx in range(num_query_groups):
                head_start = group_idx * repeat_factor
                head_end = head_start + repeat_factor
                query_group = query_tile[:, :, head_start:head_end, :].permute(1, 2, 0, 3)
                selected_key = _gather_selected_kv(key, group_idx, topk_indices)
                scores = (
                    torch.einsum("brqd,bqkd->brqk", query_group.float(), selected_key.float())
                    * softmax_scale
                )
                scores = scores.masked_fill(selected_invalid, float("-inf"))
                probs = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32)
                teacher = teacher + probs.sum(dim=1)

    if pg_collection.tp.size() > 1:
        with _profile_record(
            profile, f"teacher_scores_{profile_suffix}_all_reduce", query_tile.device
        ):
            teacher = teacher.contiguous()
            torch.distributed.all_reduce(teacher, group=pg_collection.tp)
    return teacher / teacher.sum(dim=-1, keepdim=True)


def _indexer_loss_tile(
    selected_index_scores: torch.Tensor,
    query_tile: torch.Tensor,
    key: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    loss_coeff: float,
    total_positions: int,
    q_start: int,
    pg_collection: ProcessGroupCollection,
    profile: Optional[_DSATimingProfiler] = None,
    profile_suffix: str = "fwd",
) -> torch.Tensor:
    teacher = _teacher_scores_tile(
        query_tile.detach(),
        key.detach(),
        topk_indices,
        softmax_scale,
        q_start,
        pg_collection,
        profile=profile,
        profile_suffix=profile_suffix,
    )
    with _profile_record(profile, f"indexer_kl_{profile_suffix}", query_tile.device):
        student = torch.nn.functional.softmax(selected_index_scores, dim=-1, dtype=torch.float32)
        kl = teacher * (torch.log(teacher + 1e-10) - torch.log(student + 1e-10))
        return kl.sum() * (loss_coeff / total_positions)














def _apply_indexer_input_norm_tile(
    hidden_tile: torch.Tensor,
    indexer_input_norm,
    norm_stats: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply the detached main-Q input norm using only query-tile scratch."""
    if indexer_input_norm is None:
        return hidden_tile
    weight = indexer_input_norm.weight
    if indexer_input_norm.zero_centered_gamma:
        weight = weight + 1.0
    if indexer_input_norm.normalization == "RMSNorm":
        hidden_float = hidden_tile.float()
        inv_rms = norm_stats
        if inv_rms is None:
            inv_rms = torch.rsqrt(
                hidden_float.square().mean(dim=-1, keepdim=True) + indexer_input_norm.eps
            )
        elif inv_rms.dim() == hidden_tile.dim() - 1:
            inv_rms = inv_rms.unsqueeze(-1)
        return (hidden_float * inv_rms * weight.float()).to(hidden_tile.dtype)
    if indexer_input_norm.normalization == "LayerNorm":
        return F.layer_norm(
            hidden_tile,
            (hidden_tile.size(-1),),
            weight,
            indexer_input_norm.bias,
            indexer_input_norm.eps,
        )
    raise NotImplementedError(
        "DSA cannot reproduce the fused main-Q input normalization "
        f"for normalization={indexer_input_norm.normalization!r}."
    )


_apply_simplified_input_norm_tile = _apply_indexer_input_norm_tile




def _project_simplified_q_index_tile(
    hidden_states: torch.Tensor,
    q_start: int,
    q_end: int,
    linear_q_weight: torch.Tensor,
    index_head_dim: int,
    index_rotary_dim: int,
    rotary_pos_emb,
    rotary_interleaved: bool,
    use_indexer_rope: bool,
    simplified_input_norm=None,
) -> torch.Tensor:
    hidden_tile = _apply_simplified_input_norm_tile(
        hidden_states[q_start:q_end], simplified_input_norm
    )
    q_index = _linear(hidden_tile, linear_q_weight).reshape(
        q_end - q_start, hidden_states.size(1), 1, index_head_dim
    )
    if use_indexer_rope:
        positions = torch.arange(q_start, q_end, device=q_index.device, dtype=torch.long)
        q_index = _apply_rope_at_positions(
            q_index,
            positions,
            index_head_dim,
            index_rotary_dim,
            rotary_pos_emb,
            rotary_interleaved,
        )
    return q_index


def _project_simplified_k_index_block(
    hidden_states: torch.Tensor,
    k_start: int,
    k_end: int,
    linear_k_weight: torch.Tensor,
    index_head_dim: int,
    index_rotary_dim: int,
    rotary_pos_emb,
    rotary_interleaved: bool,
    use_indexer_rope: bool,
    simplified_input_norm=None,
) -> torch.Tensor:
    hidden_block = _apply_simplified_input_norm_tile(
        hidden_states[k_start:k_end], simplified_input_norm
    )
    k_index = _linear(hidden_block, linear_k_weight).reshape(
        k_end - k_start, hidden_states.size(1), 1, index_head_dim
    )
    if use_indexer_rope:
        positions = torch.arange(k_start, k_end, device=k_index.device, dtype=torch.long)
        k_index = _apply_rope_at_positions(
            k_index,
            positions,
            index_head_dim,
            index_rotary_dim,
            rotary_pos_emb,
            rotary_interleaved,
        )
    return k_index


def _simplified_index_scores_block(
    q_index: torch.Tensor,
    key_block: torch.Tensor,
    score_scale: float,
    q_start: int,
    k_start: int,
) -> torch.Tensor:
    scores = triton_simplified_index_scores_block(
        q_index, key_block, score_scale, q_start, k_start
    )
    if scores is not None:
        return scores
    scores = torch.einsum(
        "qbd,kbd->bqk",
        q_index[:, :, 0, :].float(),
        key_block[:, :, 0, :].float(),
    ) * score_scale
    return _mask_dense_causal_scores(
        scores,
        q_start,
        q_start + q_index.size(0),
        k_start,
        k_start + key_block.size(0),
    )


def _gather_simplified_selected_key(
    key: torch.Tensor, topk_indices: torch.Tensor
) -> torch.Tensor:
    key_by_batch = key[:, :, 0, :].permute(1, 0, 2)
    batch_size, query_length, topk = topk_indices.shape
    gather_index = topk_indices[..., None].expand(
        batch_size, query_length, topk, key.size(-1)
    )
    return torch.gather(
        key_by_batch[:, None, :, :].expand(
            batch_size, query_length, key.size(0), key.size(-1)
        ),
        2,
        gather_index,
    )


def _simplified_selected_index_scores(
    q_index: torch.Tensor,
    key: torch.Tensor,
    topk_indices: torch.Tensor,
    score_scale: float,
    q_start: int,
) -> torch.Tensor:
    scores = triton_simplified_selected_index_scores(
        q_index, key, topk_indices, score_scale, q_start
    )
    if scores is not None:
        return scores
    selected_key = _gather_simplified_selected_key(key, topk_indices)
    scores = torch.einsum(
        "bqd,bqkd->bqk",
        q_index[:, :, 0, :].permute(1, 0, 2).float(),
        selected_key.float(),
    ) * score_scale
    invalid = _selected_causal_invalid_mask(topk_indices, q_start)
    return scores.masked_fill(invalid, float("-inf"))


def _simplified_selected_index_scores_backward(
    key: torch.Tensor,
    topk_indices: torch.Tensor,
    grad_scores: torch.Tensor,
    score_scale: float,
    q_start: int,
) -> torch.Tensor:
    grad_q = triton_simplified_selected_index_scores_backward(
        key, topk_indices, grad_scores, score_scale, q_start
    )
    if grad_q is not None:
        return grad_q
    invalid = _selected_causal_invalid_mask(topk_indices, q_start)
    grad_scores = grad_scores.float().masked_fill(invalid, 0.0)
    selected_key = _gather_simplified_selected_key(key, topk_indices)
    grad_q = torch.einsum("bqk,bqkd->bqd", grad_scores, selected_key.float())
    return (grad_q * score_scale).permute(1, 0, 2).unsqueeze(2).contiguous()


def _simplified_selected_index_scores_backward_qk(
    q_index: torch.Tensor,
    selected_k_index: torch.Tensor,
    topk_indices: torch.Tensor,
    grad_scores: torch.Tensor,
    score_scale: float,
    q_start: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    triton_grads = triton_simplified_selected_index_scores_backward_qk(
        q_index,
        selected_k_index,
        topk_indices,
        grad_scores,
        score_scale,
        q_start,
    )
    if triton_grads is not None:
        return triton_grads
    invalid = _selected_causal_invalid_mask(topk_indices, q_start)
    grad_scores = grad_scores.float().masked_fill(invalid, 0.0)
    q = q_index[:, :, 0, :].permute(1, 0, 2).float()
    selected_k = selected_k_index.float()
    grad_q = torch.einsum("bqk,bqkd->bqd", grad_scores, selected_k) * score_scale
    grad_selected_k = grad_scores.unsqueeze(-1) * q.unsqueeze(2) * score_scale
    return grad_q.permute(1, 0, 2).unsqueeze(2).contiguous(), grad_selected_k.contiguous()


def _accumulate_simplified_learned_k_sequence_grad(
    grad_selected_k: torch.Tensor,
    topk_indices: torch.Tensor,
    grad_k_linear_sequence: torch.Tensor,
    index_head_dim: int,
    index_rotary_dim: int,
    rotary_pos_emb,
    rotary_interleaved: bool,
    use_indexer_rope: bool,
) -> None:
    """Accumulate selected learned-K gradients into an FP32 sequence scratch."""
    grad_k_linear = _backward_indexer_transform(
        grad_selected_k,
        topk_indices,
        index_head_dim,
        index_rotary_dim,
        rotary_pos_emb,
        rotary_interleaved,
        use_indexer_rope,
        False,
    )
    scattered = triton_scatter_selected_grad_to_sequence(
        grad_k_linear, topk_indices, grad_k_linear_sequence.size(0)
    )
    if scattered is not None:
        grad_k_linear_sequence.add_(scattered)
        return

    for batch_idx in range(topk_indices.size(0)):
        grad_k_linear_sequence[:, batch_idx, :].index_add_(
            0,
            topk_indices[batch_idx].reshape(-1),
            grad_k_linear[batch_idx].reshape(-1, index_head_dim).float(),
        )


def _accumulate_simplified_learned_k_wgrad(
    grad_k_linear_sequence: torch.Tensor,
    hidden_states: torch.Tensor,
    grad_linear_k_weight: torch.Tensor,
    simplified_input_norm=None,
    norm_stats: Optional[torch.Tensor] = None,
    row_chunk_size: int = 8192,
    reuse_norm_stats_in_fallback: bool = False,
) -> None:
    """Accumulate learned-K WGRAD from an FP32 sequence-gradient scratch."""
    grad_k_linear = grad_k_linear_sequence.to(dtype=hidden_states.dtype)

    if simplified_input_norm is None:
        _accumulate_linear_weight_grad(
            grad_linear_k_weight, grad_k_linear, hidden_states
        )
        return
    if norm_stats is not None:
        sequence_length, batch_size, _ = grad_k_linear.shape
        grad_by_batch = grad_k_linear.permute(1, 0, 2).unsqueeze(2)
        sequence_indices = torch.arange(
            sequence_length, device=hidden_states.device, dtype=torch.long
        ).view(1, sequence_length, 1)
        sequence_indices = sequence_indices.expand(batch_size, -1, -1)
        if triton_simplified_gathered_linear_wgrad(
            grad_by_batch,
            hidden_states,
            sequence_indices,
            simplified_input_norm.weight,
            simplified_input_norm.bias,
            norm_stats,
            simplified_input_norm.normalization,
            simplified_input_norm.zero_centered_gamma,
            grad_linear_k_weight,
        ):
            return

    for row_start in range(0, hidden_states.size(0), row_chunk_size):
        row_end = min(row_start + row_chunk_size, hidden_states.size(0))
        input_chunk = _apply_simplified_input_norm_tile(
            hidden_states[row_start:row_end],
            simplified_input_norm,
            (
                norm_stats[row_start:row_end]
                if reuse_norm_stats_in_fallback and norm_stats is not None
                else None
            ),
        )
        _accumulate_linear_weight_grad(
            grad_linear_k_weight,
            grad_k_linear[row_start:row_end],
            input_chunk,
        )


def _simplified_topk_index_tile(
    hidden_states: torch.Tensor,
    key: torch.Tensor,
    q_start: int,
    q_end: int,
    linear_q_weight: torch.Tensor,
    index_topk: int,
    index_head_dim: int,
    index_rotary_dim: int,
    rotary_pos_emb,
    rotary_interleaved: bool,
    use_indexer_rope: bool,
    score_scale: float,
    key_chunk_size: int,
    simplified_input_norm=None,
    profile: Optional[_DSATimingProfiler] = None,
    profile_suffix: str = "fwd",
    linear_k_weight: Optional[torch.Tensor] = None,
    full_k_index: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with _profile_record(profile, f"routing_q_project_{profile_suffix}", hidden_states.device):
        q_index = _project_simplified_q_index_tile(
            hidden_states,
            q_start,
            q_end,
            linear_q_weight,
            index_head_dim,
            index_rotary_dim,
            rotary_pos_emb,
            rotary_interleaved,
            use_indexer_rope,
            simplified_input_norm,
        )
    causal_key_limit = min(q_end, key.size(0))
    topk = min(index_topk, causal_key_limit)
    running_scores = None
    running_indices = None
    unit_weights = q_index.new_ones((q_end - q_start, hidden_states.size(1), 1))
    for k_start in range(0, causal_key_limit, key_chunk_size):
        k_end = min(k_start + key_chunk_size, causal_key_limit)
        if linear_k_weight is None:
            key_block = key[k_start:k_end]
        elif full_k_index is not None:
            with _profile_record(
                profile, f"routing_k_cache_{profile_suffix}", hidden_states.device
            ):
                key_block = full_k_index[k_start:k_end]
        else:
            with _profile_record(
                profile, f"routing_k_project_{profile_suffix}", hidden_states.device
            ):
                key_block = _project_simplified_k_index_block(
                    hidden_states,
                    k_start,
                    k_end,
                    linear_k_weight,
                    index_head_dim,
                    index_rotary_dim,
                    rotary_pos_emb,
                    rotary_interleaved,
                    use_indexer_rope,
                    simplified_input_norm,
                )
        block_topk = min(topk, k_end - k_start)
        with _profile_record(
            profile, f"routing_block_score_topk_{profile_suffix}", hidden_states.device
        ):
            triton_topk = triton_topk_index_block(
                q_index,
                unit_weights,
                key_block[:, :, 0, :],
                block_topk,
                q_start,
                k_start,
                apply_relu=False,
                score_scale=score_scale,
            )
            if triton_topk is None:
                block_scores = _simplified_index_scores_block(
                    q_index, key_block, score_scale, q_start, k_start
                )
                block_scores, block_indices = block_scores.topk(block_topk, dim=-1)
                block_indices = block_indices + k_start
            else:
                block_scores, block_indices = triton_topk
        with _profile_record(profile, f"routing_merge_topk_{profile_suffix}", hidden_states.device):
            running_scores, running_indices = _merge_topk(
                running_scores, running_indices, block_scores, block_indices, topk
            )
    with _profile_record(profile, f"routing_final_sort_{profile_suffix}", hidden_states.device):
        running_scores, running_indices = _sort_topk_support_by_position(
            running_scores, running_indices
        )
    return running_scores, running_indices, q_index


def _simplified_sparse_forward_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    hidden_states: torch.Tensor,
    linear_q_weight: torch.Tensor,
    index_topk: int,
    index_head_dim: int,
    index_rotary_dim: int,
    rotary_pos_emb,
    use_indexer_rope: bool,
    attention_softmax_scale: float,
    indexer_score_scale: float,
    loss_coeff: float,
    query_chunk_size: int,
    key_chunk_size: int,
    pg_collection: ProcessGroupCollection,
    rotary_interleaved: bool,
    simplified_input_norm=None,
    profile: Optional[_DSATimingProfiler] = None,
    routing_topk_cache: Optional[list] = None,
    selected_scores_cache: Optional[list] = None,
    linear_k_weight: Optional[torch.Tensor] = None,
    full_k_index: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sq, batch_size, num_query_heads, _ = query.shape
    output = value.new_empty((sq, batch_size, num_query_heads, value.size(-1)))
    indexer_loss = query.new_zeros((), dtype=torch.float32)
    total_positions = batch_size * sq
    for q_start in range(0, sq, query_chunk_size):
        q_end = min(q_start + query_chunk_size, sq)
        with _profile_record(profile, "routing_topk_fwd", query.device):
            _, topk_indices, q_index = _simplified_topk_index_tile(
                hidden_states,
                key,
                q_start,
                q_end,
                linear_q_weight,
                index_topk,
                index_head_dim,
                index_rotary_dim,
                rotary_pos_emb,
                rotary_interleaved,
                use_indexer_rope,
                indexer_score_scale,
                key_chunk_size,
                simplified_input_norm,
                profile=profile,
                profile_suffix="fwd",
                linear_k_weight=linear_k_weight,
                full_k_index=full_k_index,
            )
        if routing_topk_cache is not None:
            routing_topk_cache.append(topk_indices)
        query_tile = query[q_start:q_end]
        with _profile_record(profile, "sparse_attention_fwd", query.device):
            output[q_start:q_end] = _sparse_attention_tile(
                query_tile, key, value, topk_indices, attention_softmax_scale, q_start
            )
        if loss_coeff > 0:
            with _profile_record(profile, "selected_index_scores_fwd", query.device):
                if linear_k_weight is None:
                    selected_scores = _simplified_selected_index_scores(
                        q_index, key, topk_indices, indexer_score_scale, q_start
                    )
                else:
                    if full_k_index is None:
                        selected_score_k_index = _project_simplified_k_index_block(
                            hidden_states,
                            0,
                            q_end,
                            linear_k_weight,
                            index_head_dim,
                            index_rotary_dim,
                            rotary_pos_emb,
                            rotary_interleaved,
                            use_indexer_rope,
                            simplified_input_norm,
                        )
                    else:
                        selected_score_k_index = full_k_index
                    selected_scores = query.new_empty(
                        topk_indices.shape, dtype=torch.float32
                    )
                    support_chunk_size = min(
                        _SIMPLIFIED_LEARNED_K_SUPPORT_CHUNK_SIZE,
                        topk_indices.size(-1),
                    )
                    for support_start in range(
                        0, topk_indices.size(-1), support_chunk_size
                    ):
                        support_end = min(
                            support_start + support_chunk_size, topk_indices.size(-1)
                        )
                        support_slice = slice(support_start, support_end)
                        selected_scores[:, :, support_slice] = (
                            _simplified_selected_index_scores(
                                q_index,
                                selected_score_k_index,
                                topk_indices[:, :, support_slice].contiguous(),
                                indexer_score_scale,
                                q_start,
                            )
                        )
            if selected_scores_cache is not None:
                selected_scores_cache.append(selected_scores)
            indexer_loss = indexer_loss + _indexer_loss_tile(
                selected_scores,
                query_tile,
                key,
                topk_indices,
                attention_softmax_scale,
                loss_coeff,
                total_positions,
                q_start,
                pg_collection,
                profile=profile,
                profile_suffix="fwd",
            )
    return output.reshape(sq, batch_size, num_query_heads * value.size(-1)), indexer_loss








class DSASimplifiedMinMemoryGQAFn(torch.autograd.Function):
    """Min-memory sparse attention with a one-head simplified Q/K router."""

    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        hidden_states: torch.Tensor,
        linear_q_weight: torch.Tensor,
        linear_k_weight: torch.Tensor,
        index_topk: int,
        index_head_dim: int,
        index_rotary_dim: int,
        rotary_pos_emb,
        use_indexer_rope: bool,
        attention_softmax_scale: float,
        indexer_score_scale: float,
        loss_coeff: float,
        query_chunk_size: int,
        key_chunk_size: int,
        pg_collection: ProcessGroupCollection,
        rotary_interleaved: bool,
        simplified_input_norm=None,
        profile_enabled: bool = False,
        profile_rank: int = 0,
        profile_label: str = "",
        cache_routing: bool = False,
        cache_selected_scores: bool = False,
        cache_indexer_k: bool = False,
        use_triton: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        profile = _DSATimingProfiler(profile_enabled, profile_rank, profile_label, query.device)
        key_chunk_size = _routing_key_chunk_size(key_chunk_size, key.size(0), use_triton)
        routing_topk_cache = [] if cache_routing else None
        selected_scores_cache = [] if cache_selected_scores else None
        use_learned_k = linear_k_weight.numel() > 0
        full_k_index = None
        with _triton_dispatch_enabled(use_triton):
            with profile.record("forward_total", query.device):
                with torch.no_grad():
                    if use_learned_k and cache_indexer_k:
                        with _profile_record(
                            profile, "indexer_k_cache_fwd_project", query.device
                        ):
                            full_k_index = _project_simplified_k_index_block(
                                hidden_states,
                                0,
                                hidden_states.size(0),
                                linear_k_weight,
                                index_head_dim,
                                index_rotary_dim,
                                rotary_pos_emb,
                                rotary_interleaved,
                                use_indexer_rope,
                                simplified_input_norm,
                            )
                    output, indexer_loss = _simplified_sparse_forward_impl(
                        query,
                        key,
                        value,
                        hidden_states,
                        linear_q_weight,
                        index_topk,
                        index_head_dim,
                        index_rotary_dim,
                        rotary_pos_emb,
                        use_indexer_rope,
                        attention_softmax_scale,
                        indexer_score_scale,
                        loss_coeff,
                        query_chunk_size,
                        key_chunk_size,
                        pg_collection,
                        rotary_interleaved,
                        simplified_input_norm,
                        profile=profile,
                        routing_topk_cache=routing_topk_cache,
                        selected_scores_cache=selected_scores_cache,
                        linear_k_weight=linear_k_weight if use_learned_k else None,
                        full_k_index=full_k_index,
                    )
        profile.log("forward")

        cached_k = full_k_index if full_k_index is not None else key.new_empty((0,))
        ctx.save_for_backward(
            query,
            key,
            value,
            hidden_states,
            linear_q_weight,
            linear_k_weight,
            cached_k,
        )
        ctx.use_learned_k = use_learned_k
        ctx.index_topk = index_topk
        ctx.index_head_dim = index_head_dim
        ctx.index_rotary_dim = index_rotary_dim
        ctx.rotary_pos_emb = rotary_pos_emb
        ctx.use_indexer_rope = use_indexer_rope
        ctx.attention_softmax_scale = attention_softmax_scale
        ctx.indexer_score_scale = indexer_score_scale
        ctx.loss_coeff = loss_coeff
        ctx.query_chunk_size = query_chunk_size
        ctx.key_chunk_size = key_chunk_size
        ctx.pg_collection = pg_collection
        ctx.rotary_interleaved = rotary_interleaved
        ctx.simplified_input_norm = simplified_input_norm
        ctx.profile_enabled = profile_enabled
        ctx.profile_rank = profile_rank
        ctx.profile_label = profile_label
        ctx.routing_topk_cache = (
            tuple(routing_topk_cache) if routing_topk_cache is not None else None
        )
        ctx.selected_scores_cache = (
            tuple(selected_scores_cache) if selected_scores_cache is not None else None
        )
        ctx.use_triton = use_triton
        return output, indexer_loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, grad_indexer_loss: torch.Tensor):
        (
            query,
            key,
            value,
            hidden_states,
            linear_q_weight,
            linear_k_weight,
            cached_k,
        ) = ctx.saved_tensors
        full_k_index = cached_k if cached_k.numel() > 0 else None
        sq, batch_size, num_query_heads, _ = query.shape
        grad_output = grad_output.reshape(
            sq, batch_size, num_query_heads, value.size(-1)
        )
        grad_query = _grad_accumulator(query) if ctx.needs_input_grad[0] else None
        grad_key = _grad_accumulator(key) if ctx.needs_input_grad[1] else None
        grad_value = _grad_accumulator(value) if ctx.needs_input_grad[2] else None
        grad_linear_q_weight = (
            _grad_accumulator(linear_q_weight) if ctx.needs_input_grad[4] else None
        )
        grad_linear_k_weight = (
            _grad_accumulator(linear_k_weight)
            if ctx.use_learned_k and ctx.needs_input_grad[5]
            else None
        )
        compute_loss_grad = (
            grad_indexer_loss is not None
            and ctx.loss_coeff > 0
            and (grad_linear_q_weight is not None or grad_linear_k_weight is not None)
        )
        grad_k_linear_sequence = (
            torch.zeros(
                (sq, batch_size, ctx.index_head_dim),
                device=query.device,
                dtype=torch.float32,
            )
            if compute_loss_grad and grad_linear_k_weight is not None
            else None
        )
        use_triton_attention_backward = (
            ctx.needs_input_grad[0] and ctx.needs_input_grad[1] and ctx.needs_input_grad[2]
        )
        grad_key_accum = None
        grad_value_accum = None
        profile = _DSATimingProfiler(
            ctx.profile_enabled, ctx.profile_rank, ctx.profile_label, query.device
        )

        with _triton_dispatch_enabled(ctx.use_triton), profile.record(
            "backward_total", query.device
        ):
            learned_k_norm_stats = None
            if (
                compute_loss_grad
                and ctx.use_learned_k
                and grad_linear_k_weight is not None
                and ctx.simplified_input_norm is not None
            ):
                with _profile_record(
                    profile, "indexer_loss_bwd_simplified_k_norm_stats", query.device
                ):
                    learned_k_norm_stats = triton_simplified_input_norm_stats(
                        hidden_states,
                        ctx.simplified_input_norm.eps,
                        ctx.simplified_input_norm.normalization,
                    )
            for chunk_idx, q_start in enumerate(range(0, sq, ctx.query_chunk_size)):
                q_end = min(q_start + ctx.query_chunk_size, sq)
                q_index = None
                if ctx.routing_topk_cache is not None:
                    with _profile_record(profile, "routing_topk_bwd_cached", query.device):
                        topk_indices = ctx.routing_topk_cache[chunk_idx]
                else:
                    with _profile_record(profile, "routing_topk_bwd", query.device):
                        with torch.no_grad():
                            _, topk_indices, q_index = _simplified_topk_index_tile(
                                hidden_states,
                                key,
                                q_start,
                                q_end,
                                linear_q_weight,
                                ctx.index_topk,
                                ctx.index_head_dim,
                                ctx.index_rotary_dim,
                                ctx.rotary_pos_emb,
                                ctx.rotary_interleaved,
                                ctx.use_indexer_rope,
                                ctx.indexer_score_scale,
                                ctx.key_chunk_size,
                                ctx.simplified_input_norm,
                                profile=profile,
                                profile_suffix="bwd",
                                linear_k_weight=(
                                    linear_k_weight if ctx.use_learned_k else None
                                ),
                                full_k_index=full_k_index,
                            )

                triton_attention_done = False
                if use_triton_attention_backward:
                    query_tile = query[q_start:q_end]
                    grad_output_tile = grad_output[q_start:q_end]
                    grad_query_tile = grad_query[q_start:q_end]
                    if grad_key_accum is None and triton_sparse_attention_backward_supported(
                        query_tile,
                        key,
                        value,
                        topk_indices,
                        grad_output_tile,
                        grad_query_tile,
                    ):
                        with _profile_record(
                            profile, "sparse_attention_bwd_scratch_alloc", query.device
                        ):
                            grad_key_accum = torch.zeros(
                                key.shape, device=key.device, dtype=torch.float32
                            )
                            grad_value_accum = torch.zeros(
                                value.shape, device=value.device, dtype=torch.float32
                            )
                    if grad_key_accum is not None and grad_value_accum is not None:
                        attention_path = triton_sparse_attention_backward_path(
                            query_tile, key, value, topk_indices
                        )
                        with _profile_record(profile, "sparse_attention_bwd_triton", query.device):
                            with _profile_record(
                                profile,
                                f"sparse_attention_bwd_triton_{attention_path}",
                                query.device,
                            ):
                                triton_attention_done = triton_sparse_attention_backward_accumulate(
                                    query_tile,
                                    key,
                                    value,
                                    topk_indices,
                                    grad_output_tile,
                                    grad_query_tile,
                                    grad_key_accum,
                                    grad_value_accum,
                                    ctx.attention_softmax_scale,
                                    q_start,
                                )

                if not triton_attention_done:
                    attention_inputs = []
                    query_tile = query[q_start:q_end].detach().requires_grad_(
                        ctx.needs_input_grad[0]
                    )
                    key_leaf = key.detach().requires_grad_(ctx.needs_input_grad[1])
                    value_leaf = value.detach().requires_grad_(ctx.needs_input_grad[2])
                    if ctx.needs_input_grad[0]:
                        attention_inputs.append(query_tile)
                    if ctx.needs_input_grad[1]:
                        attention_inputs.append(key_leaf)
                    if ctx.needs_input_grad[2]:
                        attention_inputs.append(value_leaf)
                    if attention_inputs:
                        with _profile_record(
                            profile, "sparse_attention_bwd_fallback", query.device
                        ):
                            with torch.enable_grad():
                                output_tile = _sparse_attention_tile(
                                    query_tile,
                                    key_leaf,
                                    value_leaf,
                                    topk_indices,
                                    ctx.attention_softmax_scale,
                                    q_start,
                                )
                            attention_grads = torch.autograd.grad(
                                output_tile,
                                attention_inputs,
                                grad_outputs=grad_output[q_start:q_end],
                                retain_graph=False,
                                allow_unused=True,
                            )
                            grad_iter = iter(attention_grads)
                            if ctx.needs_input_grad[0]:
                                grad = next(grad_iter)
                                if grad is not None:
                                    grad_query[q_start:q_end] = grad
                            if ctx.needs_input_grad[1]:
                                grad = next(grad_iter)
                                if grad is not None:
                                    grad_key.add_(grad)
                            if ctx.needs_input_grad[2]:
                                grad = next(grad_iter)
                                if grad is not None:
                                    grad_value.add_(grad)

                if compute_loss_grad:
                    with _profile_record(profile, "indexer_loss_bwd_prepare", query.device):
                        with torch.no_grad():
                            if q_index is None and (
                                ctx.selected_scores_cache is None or ctx.use_learned_k
                            ):
                                q_index = _project_simplified_q_index_tile(
                                    hidden_states,
                                    q_start,
                                    q_end,
                                    linear_q_weight,
                                    ctx.index_head_dim,
                                    ctx.index_rotary_dim,
                                    ctx.rotary_pos_emb,
                                    ctx.rotary_interleaved,
                                    ctx.use_indexer_rope,
                                    ctx.simplified_input_norm,
                                )
                            selected_score_k_index = None
                            if ctx.use_learned_k:
                                if full_k_index is not None:
                                    selected_score_k_index = full_k_index
                                else:
                                    selected_score_k_index = _project_simplified_k_index_block(
                                        hidden_states,
                                        0,
                                        q_end,
                                        linear_k_weight,
                                        ctx.index_head_dim,
                                        ctx.index_rotary_dim,
                                        ctx.rotary_pos_emb,
                                        ctx.rotary_interleaved,
                                        ctx.use_indexer_rope,
                                        ctx.simplified_input_norm,
                                    )
                                if ctx.selected_scores_cache is not None:
                                    selected_scores = ctx.selected_scores_cache[chunk_idx]
                                else:
                                    selected_scores = query.new_empty(
                                        topk_indices.shape, dtype=torch.float32
                                    )
                                    support_chunk_size = min(
                                        _SIMPLIFIED_LEARNED_K_SUPPORT_CHUNK_SIZE,
                                        topk_indices.size(-1),
                                    )
                                    for support_start in range(
                                        0, topk_indices.size(-1), support_chunk_size
                                    ):
                                        support_end = min(
                                            support_start + support_chunk_size,
                                            topk_indices.size(-1),
                                        )
                                        support_slice = slice(support_start, support_end)
                                        selected_scores[:, :, support_slice] = (
                                            _simplified_selected_index_scores(
                                                q_index,
                                                selected_score_k_index,
                                                topk_indices[:, :, support_slice].contiguous(),
                                                ctx.indexer_score_scale,
                                                q_start,
                                            )
                                        )
                            else:
                                selected_scores = (
                                    ctx.selected_scores_cache[chunk_idx]
                                    if ctx.selected_scores_cache is not None
                                    else _simplified_selected_index_scores(
                                        q_index,
                                        key,
                                        topk_indices,
                                        ctx.indexer_score_scale,
                                        q_start,
                                    )
                                )
                            teacher = _teacher_scores_tile(
                                query[q_start:q_end].detach(),
                                key.detach(),
                                topk_indices,
                                ctx.attention_softmax_scale,
                                q_start,
                                ctx.pg_collection,
                                profile=profile,
                                profile_suffix="bwd",
                            )
                            scale = grad_indexer_loss * (
                                ctx.loss_coeff / (batch_size * sq)
                            )
                            grad_scores = triton_indexer_loss_grad(
                                selected_scores, teacher, scale
                            )
                            if grad_scores is None:
                                student = torch.nn.functional.softmax(
                                    selected_scores, dim=-1, dtype=torch.float32
                                )
                                alpha = teacher * student / (student + 1.0e-10)
                                grad_scores = (
                                    student * alpha.sum(dim=-1, keepdim=True) - alpha
                                ) * scale
                    with _profile_record(
                        profile, "indexer_loss_bwd_simplified_q_wgrad", query.device
                    ):
                        if ctx.use_learned_k:
                            grad_q_index = torch.zeros_like(q_index, dtype=torch.float32)
                            # Bound the FP32 selected-K gradient scratch independently of
                            # top-k. This also keeps learned-K WGRAD memory stable as top-k
                            # is swept during adaptation experiments.
                            support_chunk_size = min(
                                _SIMPLIFIED_LEARNED_K_SUPPORT_CHUNK_SIZE,
                                topk_indices.size(-1),
                            )
                            for support_start in range(
                                0, topk_indices.size(-1), support_chunk_size
                            ):
                                support_end = min(
                                    support_start + support_chunk_size,
                                    topk_indices.size(-1),
                                )
                                support_slice = slice(support_start, support_end)
                                selected_k_chunk = _gather_selected_indexer_k(
                                    selected_score_k_index[:, :, 0, :],
                                    topk_indices[:, :, support_slice],
                                )
                                grad_q_chunk, grad_selected_k_chunk = (
                                    _simplified_selected_index_scores_backward_qk(
                                        q_index,
                                        selected_k_chunk,
                                        topk_indices[:, :, support_slice],
                                        grad_scores[:, :, support_slice],
                                        ctx.indexer_score_scale,
                                        q_start,
                                    )
                                )
                                grad_q_index.add_(grad_q_chunk)
                                if grad_k_linear_sequence is not None:
                                    with _profile_record(
                                        profile,
                                        "indexer_loss_bwd_simplified_k_scatter",
                                        query.device,
                                    ):
                                        _accumulate_simplified_learned_k_sequence_grad(
                                            grad_selected_k_chunk,
                                            topk_indices[:, :, support_slice],
                                            grad_k_linear_sequence,
                                            ctx.index_head_dim,
                                            ctx.index_rotary_dim,
                                            ctx.rotary_pos_emb,
                                            ctx.rotary_interleaved,
                                            ctx.use_indexer_rope,
                                        )
                        else:
                            grad_q_index = _simplified_selected_index_scores_backward(
                                key.detach(),
                                topk_indices,
                                grad_scores,
                                ctx.indexer_score_scale,
                                q_start,
                            )
                        positions = torch.arange(
                            q_start, q_end, device=query.device, dtype=torch.long
                        )
                        grad_q_linear = _backward_indexer_transform(
                            grad_q_index,
                            positions,
                            ctx.index_head_dim,
                            ctx.index_rotary_dim,
                            ctx.rotary_pos_emb,
                            ctx.rotary_interleaved,
                            ctx.use_indexer_rope,
                            False,
                        ).reshape(q_end - q_start, batch_size, ctx.index_head_dim)
                        # Match BF16/FP16 linear backward: round the activation-gradient operand,
                        # then accumulate the WGRAD reduction in FP32.
                        grad_q_linear = grad_q_linear.to(dtype=hidden_states.dtype)
                        q_input_tile = _apply_simplified_input_norm_tile(
                            hidden_states[q_start:q_end], ctx.simplified_input_norm
                        )
                        _accumulate_linear_weight_grad(
                            grad_linear_q_weight, grad_q_linear, q_input_tile
                        )

            if grad_k_linear_sequence is not None:
                with _profile_record(
                    profile, "indexer_loss_bwd_simplified_k_wgrad", query.device
                ):
                    _accumulate_simplified_learned_k_wgrad(
                        grad_k_linear_sequence,
                        hidden_states,
                        grad_linear_k_weight,
                        ctx.simplified_input_norm,
                        learned_k_norm_stats,
                    )

            with _profile_record(profile, "sparse_attention_bwd_finalize", query.device):
                if grad_key_accum is not None:
                    grad_key.add_(grad_key_accum)
                if grad_value_accum is not None:
                    grad_value.add_(grad_value_accum)
        profile.log("backward")
        return (
            grad_query,
            grad_key,
            grad_value,
            None,
            grad_linear_q_weight,
            grad_linear_k_weight,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _simplified_dense_softmax_stats(
    q_index: torch.Tensor,
    query_tile: torch.Tensor,
    key: torch.Tensor,
    hidden_states: torch.Tensor,
    linear_k_weight: Optional[torch.Tensor],
    index_head_dim: int,
    index_rotary_dim: int,
    rotary_pos_emb,
    rotary_interleaved: bool,
    use_indexer_rope: bool,
    simplified_input_norm,
    teacher_softmax_scale: float,
    student_score_scale: float,
    q_start: int,
    key_chunk_size: int,
    profile: Optional[_DSATimingProfiler],
    profile_suffix: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    teacher_max = None
    teacher_sum = None
    student_max = None
    student_sum = None
    for k_start in range(0, key.size(0), key_chunk_size):
        k_end = min(k_start + key_chunk_size, key.size(0))
        with _profile_record(
            profile, f"dense_indexer_kl_{profile_suffix}_teacher_stats", query_tile.device
        ):
            teacher_logits = _dense_teacher_logits_block(
                query_tile, key[k_start:k_end], teacher_softmax_scale, q_start, k_start
            )
            teacher_max, teacher_sum = _update_running_softmax_stats(
                teacher_logits, teacher_max, teacher_sum
            )
        with _profile_record(
            profile, f"dense_indexer_kl_{profile_suffix}_student_stats", query_tile.device
        ):
            student_key = (
                key[k_start:k_end]
                if linear_k_weight is None
                else _project_simplified_k_index_block(
                    hidden_states,
                    k_start,
                    k_end,
                    linear_k_weight,
                    index_head_dim,
                    index_rotary_dim,
                    rotary_pos_emb,
                    rotary_interleaved,
                    use_indexer_rope,
                    simplified_input_norm,
                )
            )
            student_logits = _simplified_index_scores_block(
                q_index, student_key, student_score_scale, q_start, k_start
            )
            student_max, student_sum = _update_running_softmax_stats(
                student_logits, student_max, student_sum
            )
    assert teacher_max is not None and teacher_sum is not None
    assert student_max is not None and student_sum is not None
    return teacher_max, teacher_sum, student_max, student_sum


def _simplified_dense_indexer_kl_loss_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    hidden_states: torch.Tensor,
    linear_q_weight: torch.Tensor,
    linear_k_weight: Optional[torch.Tensor],
    index_head_dim: int,
    index_rotary_dim: int,
    rotary_pos_emb,
    use_indexer_rope: bool,
    teacher_softmax_scale: float,
    student_score_scale: float,
    loss_coeff: float,
    query_chunk_size: int,
    key_chunk_size: int,
    pg_collection: ProcessGroupCollection,
    rotary_interleaved: bool,
    simplified_input_norm,
    profile: Optional[_DSATimingProfiler],
) -> torch.Tensor:
    total_kl = query.new_zeros((), dtype=torch.float32)
    total_positions = query.size(0) * query.size(1)
    for q_start in range(0, query.size(0), query_chunk_size):
        q_end = min(q_start + query_chunk_size, query.size(0))
        query_tile = query[q_start:q_end]
        with _profile_record(profile, "dense_indexer_kl_fwd_q_project", query.device):
            q_index = _project_simplified_q_index_tile(
                hidden_states,
                q_start,
                q_end,
                linear_q_weight,
                index_head_dim,
                index_rotary_dim,
                rotary_pos_emb,
                rotary_interleaved,
                use_indexer_rope,
                simplified_input_norm,
            )
        teacher_max, teacher_sum, student_max, student_sum = (
            _simplified_dense_softmax_stats(
                q_index,
                query_tile,
                key,
                hidden_states,
                linear_k_weight,
                index_head_dim,
                index_rotary_dim,
                rotary_pos_emb,
                rotary_interleaved,
                use_indexer_rope,
                simplified_input_norm,
                teacher_softmax_scale,
                student_score_scale,
                q_start,
                key_chunk_size,
                profile,
                "fwd",
            )
        )
        with _profile_record(profile, "dense_indexer_kl_fwd_loss", query.device):
            teacher_norm = _dense_teacher_norm(
                query_tile,
                key,
                teacher_max,
                teacher_sum,
                teacher_softmax_scale,
                q_start,
                key_chunk_size,
                pg_collection,
            )
            for k_start in range(0, key.size(0), key_chunk_size):
                k_end = min(k_start + key_chunk_size, key.size(0))
                teacher = _dense_teacher_mass_block(
                    query_tile,
                    key[k_start:k_end],
                    teacher_max,
                    teacher_sum,
                    teacher_softmax_scale,
                    q_start,
                    k_start,
                    pg_collection,
                )
                teacher = teacher / teacher_norm.unsqueeze(-1)
                student_key = (
                    key[k_start:k_end]
                    if linear_k_weight is None
                    else _project_simplified_k_index_block(
                        hidden_states,
                        k_start,
                        k_end,
                        linear_k_weight,
                        index_head_dim,
                        index_rotary_dim,
                        rotary_pos_emb,
                        rotary_interleaved,
                        use_indexer_rope,
                        simplified_input_norm,
                    )
                )
                student_logits = _simplified_index_scores_block(
                    q_index, student_key, student_score_scale, q_start, k_start
                )
                student = (
                    torch.exp(student_logits - student_max.unsqueeze(-1))
                    / student_sum.unsqueeze(-1)
                )
                total_kl = total_kl + (
                    teacher
                    * (torch.log(teacher + 1.0e-10) - torch.log(student + 1.0e-10))
                ).sum()
    return total_kl / total_positions * loss_coeff


class DSASimplifiedDenseIndexerLossFn(torch.autograd.Function):
    """Tiled dense KL for the one-head simplified Q/K indexer."""

    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        hidden_states: torch.Tensor,
        linear_q_weight: torch.Tensor,
        linear_k_weight: torch.Tensor,
        index_head_dim: int,
        index_rotary_dim: int,
        rotary_pos_emb,
        use_indexer_rope: bool,
        teacher_softmax_scale: float,
        student_score_scale: float,
        loss_coeff: float,
        query_chunk_size: int,
        key_chunk_size: int,
        pg_collection: ProcessGroupCollection,
        rotary_interleaved: bool,
        simplified_input_norm=None,
        profile_enabled: bool = False,
        profile_rank: int = 0,
        profile_label: str = "",
        use_triton: bool = True,
    ) -> torch.Tensor:
        profile = _DSATimingProfiler(profile_enabled, profile_rank, profile_label, query.device)
        with torch.no_grad(), _triton_dispatch_enabled(use_triton):
            with profile.record("dense_indexer_kl_fwd_total", query.device):
                loss = _simplified_dense_indexer_kl_loss_impl(
                    query,
                    key,
                    hidden_states,
                    linear_q_weight,
                    linear_k_weight if linear_k_weight.numel() > 0 else None,
                    index_head_dim,
                    index_rotary_dim,
                    rotary_pos_emb,
                    use_indexer_rope,
                    teacher_softmax_scale,
                    student_score_scale,
                    loss_coeff,
                    query_chunk_size,
                    key_chunk_size,
                    pg_collection,
                    rotary_interleaved,
                    simplified_input_norm,
                    profile,
                )
        profile.log("forward")
        ctx.save_for_backward(query, key, hidden_states, linear_q_weight, linear_k_weight)
        ctx.use_learned_k = linear_k_weight.numel() > 0
        ctx.index_head_dim = index_head_dim
        ctx.index_rotary_dim = index_rotary_dim
        ctx.rotary_pos_emb = rotary_pos_emb
        ctx.use_indexer_rope = use_indexer_rope
        ctx.teacher_softmax_scale = teacher_softmax_scale
        ctx.student_score_scale = student_score_scale
        ctx.loss_coeff = loss_coeff
        ctx.query_chunk_size = query_chunk_size
        ctx.key_chunk_size = key_chunk_size
        ctx.pg_collection = pg_collection
        ctx.rotary_interleaved = rotary_interleaved
        ctx.simplified_input_norm = simplified_input_norm
        ctx.profile_enabled = profile_enabled
        ctx.profile_rank = profile_rank
        ctx.profile_label = profile_label
        ctx.use_triton = use_triton
        return loss

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor):
        query, key, hidden_states, linear_q_weight, linear_k_weight = ctx.saved_tensors
        grad_linear_q_weight = (
            _grad_accumulator(linear_q_weight) if ctx.needs_input_grad[3] else None
        )
        grad_linear_k_weight = (
            _grad_accumulator(linear_k_weight)
            if ctx.use_learned_k and ctx.needs_input_grad[4]
            else None
        )
        if grad_loss is None or ctx.loss_coeff <= 0 or (
            grad_linear_q_weight is None and grad_linear_k_weight is None
        ):
            return (
                None,
                None,
                None,
                grad_linear_q_weight,
                grad_linear_k_weight,
            ) + (None,) * 16
        grad_k_linear_sequence = (
            torch.zeros(
                (query.size(0), query.size(1), ctx.index_head_dim),
                device=query.device,
                dtype=torch.float32,
            )
            if grad_linear_k_weight is not None
            else None
        )

        profile = _DSATimingProfiler(
            ctx.profile_enabled, ctx.profile_rank, ctx.profile_label, query.device
        )
        total_positions = query.size(0) * query.size(1)
        loss_scale = grad_loss * (ctx.loss_coeff / total_positions)
        with _triton_dispatch_enabled(ctx.use_triton), profile.record(
            "dense_indexer_kl_bwd_total", query.device
        ):
            learned_k_norm_stats = None
            if grad_k_linear_sequence is not None and ctx.simplified_input_norm is not None:
                learned_k_norm_stats = triton_simplified_input_norm_stats(
                    hidden_states,
                    ctx.simplified_input_norm.eps,
                    ctx.simplified_input_norm.normalization,
                )
            for q_start in range(0, query.size(0), ctx.query_chunk_size):
                q_end = min(q_start + ctx.query_chunk_size, query.size(0))
                query_tile = query[q_start:q_end]
                with torch.no_grad():
                    q_index = _project_simplified_q_index_tile(
                        hidden_states,
                        q_start,
                        q_end,
                        linear_q_weight,
                        ctx.index_head_dim,
                        ctx.index_rotary_dim,
                        ctx.rotary_pos_emb,
                        ctx.rotary_interleaved,
                        ctx.use_indexer_rope,
                        ctx.simplified_input_norm,
                    )
                    teacher_max, teacher_sum, student_max, student_sum = (
                        _simplified_dense_softmax_stats(
                            q_index,
                            query_tile,
                            key,
                            hidden_states,
                            linear_k_weight if ctx.use_learned_k else None,
                            ctx.index_head_dim,
                            ctx.index_rotary_dim,
                            ctx.rotary_pos_emb,
                            ctx.rotary_interleaved,
                            ctx.use_indexer_rope,
                            ctx.simplified_input_norm,
                            ctx.teacher_softmax_scale,
                            ctx.student_score_scale,
                            q_start,
                            ctx.key_chunk_size,
                            profile,
                            "bwd",
                        )
                    )
                    teacher_norm = _dense_teacher_norm(
                        query_tile,
                        key,
                        teacher_max,
                        teacher_sum,
                        ctx.teacher_softmax_scale,
                        q_start,
                        ctx.key_chunk_size,
                        ctx.pg_collection,
                    )
                    student_grad_norm = query_tile.new_zeros(
                        (query_tile.size(1), query_tile.size(0)), dtype=torch.float32
                    )
                    for k_start in range(0, key.size(0), ctx.key_chunk_size):
                        k_end = min(k_start + ctx.key_chunk_size, key.size(0))
                        teacher = _dense_teacher_mass_block(
                            query_tile,
                            key[k_start:k_end],
                            teacher_max,
                            teacher_sum,
                            ctx.teacher_softmax_scale,
                            q_start,
                            k_start,
                            ctx.pg_collection,
                        )
                        teacher = teacher / teacher_norm.unsqueeze(-1)
                        student_key = (
                            key[k_start:k_end]
                            if not ctx.use_learned_k
                            else _project_simplified_k_index_block(
                                hidden_states,
                                k_start,
                                k_end,
                                linear_k_weight,
                                ctx.index_head_dim,
                                ctx.index_rotary_dim,
                                ctx.rotary_pos_emb,
                                ctx.rotary_interleaved,
                                ctx.use_indexer_rope,
                                ctx.simplified_input_norm,
                            )
                        )
                        student_logits = _simplified_index_scores_block(
                            q_index,
                            student_key,
                            ctx.student_score_scale,
                            q_start,
                            k_start,
                        )
                        student = (
                            torch.exp(student_logits - student_max.unsqueeze(-1))
                            / student_sum.unsqueeze(-1)
                        )
                        student_grad_norm = student_grad_norm + (
                            teacher * student / (student + 1.0e-10)
                        ).sum(dim=-1)

                    grad_q_index = query_tile.new_zeros(
                        (q_end - q_start, query.size(1), 1, ctx.index_head_dim),
                        dtype=torch.float32,
                    )
                    for k_start in range(0, key.size(0), ctx.key_chunk_size):
                        k_end = min(k_start + ctx.key_chunk_size, key.size(0))
                        teacher = _dense_teacher_mass_block(
                            query_tile,
                            key[k_start:k_end],
                            teacher_max,
                            teacher_sum,
                            ctx.teacher_softmax_scale,
                            q_start,
                            k_start,
                            ctx.pg_collection,
                        )
                        teacher = teacher / teacher_norm.unsqueeze(-1)
                        student_key = (
                            key[k_start:k_end]
                            if not ctx.use_learned_k
                            else _project_simplified_k_index_block(
                                hidden_states,
                                k_start,
                                k_end,
                                linear_k_weight,
                                ctx.index_head_dim,
                                ctx.index_rotary_dim,
                                ctx.rotary_pos_emb,
                                ctx.rotary_interleaved,
                                ctx.use_indexer_rope,
                                ctx.simplified_input_norm,
                            )
                        )
                        student_logits = _simplified_index_scores_block(
                            q_index,
                            student_key,
                            ctx.student_score_scale,
                            q_start,
                            k_start,
                        )
                        student = (
                            torch.exp(student_logits - student_max.unsqueeze(-1))
                            / student_sum.unsqueeze(-1)
                        )
                        alpha = teacher * student / (student + 1.0e-10)
                        grad_scores = (
                            student * student_grad_norm.unsqueeze(-1) - alpha
                        ) * loss_scale
                        key_by_batch = student_key[:, :, 0, :].permute(1, 0, 2).float()
                        grad_q_block = torch.bmm(grad_scores, key_by_batch)
                        grad_q_index[:, :, 0, :].add_(
                            (grad_q_block * ctx.student_score_scale).permute(1, 0, 2)
                        )
                        if grad_k_linear_sequence is not None:
                            q_by_batch = q_index[:, :, 0, :].permute(1, 0, 2).float()
                            grad_k_index = torch.bmm(
                                grad_scores.transpose(1, 2), q_by_batch
                            ) * ctx.student_score_scale
                            grad_k_index = grad_k_index.permute(1, 0, 2).unsqueeze(2)
                            key_positions = torch.arange(
                                k_start, k_end, device=query.device, dtype=torch.long
                            )
                            grad_k_linear = _backward_indexer_transform(
                                grad_k_index,
                                key_positions,
                                ctx.index_head_dim,
                                ctx.index_rotary_dim,
                                ctx.rotary_pos_emb,
                                ctx.rotary_interleaved,
                                ctx.use_indexer_rope,
                                False,
                            ).reshape(k_end - k_start, query.size(1), ctx.index_head_dim)
                            grad_k_linear_sequence[k_start:k_end].add_(grad_k_linear)

                with _profile_record(
                    profile, "dense_indexer_kl_bwd_simplified_q_wgrad", query.device
                ):
                    positions = torch.arange(
                        q_start, q_end, device=query.device, dtype=torch.long
                    )
                    grad_q_linear = _backward_indexer_transform(
                        grad_q_index,
                        positions,
                        ctx.index_head_dim,
                        ctx.index_rotary_dim,
                        ctx.rotary_pos_emb,
                        ctx.rotary_interleaved,
                        ctx.use_indexer_rope,
                        False,
                    ).reshape(q_end - q_start, query.size(1), ctx.index_head_dim)
                    # Match BF16/FP16 linear backward while retaining FP32 WGRAD accumulation.
                    grad_q_linear = grad_q_linear.to(dtype=hidden_states.dtype)
                    q_input_tile = _apply_simplified_input_norm_tile(
                        hidden_states[q_start:q_end], ctx.simplified_input_norm
                    )
                    _accumulate_linear_weight_grad(
                        grad_linear_q_weight,
                        grad_q_linear,
                        q_input_tile,
                    )
            if grad_k_linear_sequence is not None:
                with _profile_record(
                    profile, "dense_indexer_kl_bwd_simplified_k_wgrad", query.device
                ):
                    _accumulate_simplified_learned_k_wgrad(
                        grad_k_linear_sequence,
                        hidden_states,
                        grad_linear_k_weight,
                        ctx.simplified_input_norm,
                        learned_k_norm_stats,
                    )
        profile.log("backward")
        return (
            None,
            None,
            None,
            grad_linear_q_weight,
            grad_linear_k_weight,
        ) + (None,) * 16


def dsa_min_memory_gqa_forward_only(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    hidden_states: torch.Tensor,
    indexer,
    softmax_scale: float,
    use_indexer_rope: bool,
    query_chunk_size: Optional[int],
    key_chunk_size: Optional[int],
    cache_indexer_k: bool = False,
    profile_enabled: bool = False,
    profile_rank: int = 0,
    profile_label: str = "",
    use_triton: bool = True,
    use_cudnn: bool = False,
    simplified_input_norm=None,
) -> torch.Tensor:
    """Run min-memory DSA-GQA for no-grad validation/eval forward passes."""
    if getattr(indexer.config, "dsa_indexer_mode", "standard") == "simplified":
        query_chunk_size = _chunk_size(
            query_chunk_size, _default_query_chunk_size(query.size(0)), query.size(0)
        )
        key_chunk_size = _routing_key_chunk_size(key_chunk_size, key.size(0), use_triton)
        profile = _DSATimingProfiler(profile_enabled, profile_rank, profile_label, query.device)
        use_learned_k = getattr(indexer.config, "dsa_simplified_use_learned_k", False)
        linear_k_weight = (
            _module_weight(indexer.linear_k)
            if use_learned_k
            else query.new_empty((0,))
        )
        full_k_index = None
        with torch.no_grad(), _triton_dispatch_enabled(use_triton):
            with profile.record("forward_total", query.device):
                if use_learned_k and cache_indexer_k:
                    with _profile_record(
                        profile, "indexer_k_cache_fwd_project", query.device
                    ):
                        full_k_index = _project_simplified_k_index_block(
                            hidden_states,
                            0,
                            hidden_states.size(0),
                            linear_k_weight,
                            indexer.index_head_dim,
                            indexer.index_rotary_dim,
                            indexer.rotary_pos_emb,
                            getattr(indexer.config, "rotary_interleaved", False),
                            use_indexer_rope,
                            simplified_input_norm,
                        )
                output, _ = _simplified_sparse_forward_impl(
                    query,
                    key,
                    value,
                    hidden_states,
                    _module_weight(indexer.linear_q),
                    indexer.index_topk,
                    indexer.index_head_dim,
                    indexer.index_rotary_dim,
                    indexer.rotary_pos_emb,
                    use_indexer_rope,
                    softmax_scale,
                    indexer.softmax_scale,
                    0.0,
                    query_chunk_size,
                    key_chunk_size,
                    indexer.pg_collection,
                    getattr(indexer.config, "rotary_interleaved", False),
                    simplified_input_norm,
                    profile=profile,
                    linear_k_weight=linear_k_weight if use_learned_k else None,
                    full_k_index=full_k_index,
                )
        profile.log("forward")
        return output


_MAIN_ATTENTION_AUX_MAX_QUERY_BLOCK_SIZE = 2048
_MAIN_ATTENTION_AUX_LOGITS_BUDGET_BYTES = 256 * 1024 * 1024










def _sparse_attention_backward_torch_fp32(
    query_tile: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected_indices: torch.Tensor,
    grad_output: torch.Tensor,
    softmax_scale: float,
    q_start: int,
    grad_query: Optional[torch.Tensor],
    grad_key: Optional[torch.Tensor],
    grad_value: Optional[torch.Tensor],
) -> None:
    """Accumulate sparse-attention gradients in FP32 for the Torch fallback."""
    if grad_query is None and grad_key is None and grad_value is None:
        return

    batch_size = query_tile.size(1)
    num_query_heads = query_tile.size(2)
    num_query_groups = key.size(2)
    assert num_query_heads % num_query_groups == 0
    repeat_factor = num_query_heads // num_query_groups
    selected_invalid = _selected_causal_invalid_mask(selected_indices, q_start)
    selected_valid = ~selected_invalid

    for group_idx in range(num_query_groups):
        head_start = group_idx * repeat_factor
        head_end = head_start + repeat_factor
        query_group = query_tile[:, :, head_start:head_end, :].permute(1, 2, 0, 3).float()
        selected_key = _gather_selected_kv(key, group_idx, selected_indices).float()
        selected_value = _gather_selected_kv(value, group_idx, selected_indices).float()

        scores = (
            torch.einsum("brqd,bqkd->brqk", query_group, selected_key) * softmax_scale
        )
        scores = scores.masked_fill(selected_invalid[:, None], float("-inf"))
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32)

        # The sparse output is model dtype. Round its incoming gradient before the value
        # and probability GEMMs, then keep every reduction and repeated-key scatter in FP32.
        grad_output_group = (
            grad_output[:, :, head_start:head_end, :]
            .permute(1, 2, 0, 3)
            .to(value.dtype)
            .float()
        )
        dprob = torch.einsum("brqd,bqkd->brqk", grad_output_group, selected_value)
        delta = (dprob * probs).sum(dim=-1, keepdim=True)
        dscores = probs * (dprob - delta)
        dscores = dscores * selected_valid[:, None]

        if grad_query is not None:
            grad_query_group = (
                torch.einsum("brqk,bqkd->brqd", dscores, selected_key) * softmax_scale
            )
            grad_query[:, :, head_start:head_end, :].add_(
                grad_query_group.permute(2, 0, 1, 3)
            )

        if grad_key is not None:
            grad_selected_key = (
                torch.einsum("brqk,brqd->bqkd", dscores, query_group) * softmax_scale
            )
            grad_selected_key = grad_selected_key * selected_valid[..., None]
            for batch_idx in range(batch_size):
                grad_key[:, batch_idx, group_idx, :].index_add_(
                    0,
                    selected_indices[batch_idx].reshape(-1),
                    grad_selected_key[batch_idx].reshape(-1, key.size(-1)),
                )

        if grad_value is not None:
            probs_for_value = probs.to(value.dtype).float()
            grad_selected_value = torch.einsum(
                "brqk,brqd->bqkd", probs_for_value, grad_output_group
            )
            grad_selected_value = grad_selected_value * selected_valid[..., None]
            for batch_idx in range(batch_size):
                grad_value[:, batch_idx, group_idx, :].index_add_(
                    0,
                    selected_indices[batch_idx].reshape(-1),
                    grad_selected_value[batch_idx].reshape(-1, value.size(-1)),
                )










def dsa_dense_indexer_loss(
    query: torch.Tensor,
    key: torch.Tensor,
    hidden_states: torch.Tensor,
    indexer,
    softmax_scale: float,
    loss_coeff: float,
    use_indexer_rope: bool,
    query_chunk_size: Optional[int],
    key_chunk_size: Optional[int],
    profile_enabled: bool = False,
    profile_rank: int = 0,
    profile_label: str = "",
    use_triton: bool = True,
    simplified_input_norm=None,
) -> torch.Tensor:
    """Run tiled dense DSA indexer KL for dense-attention warmup."""
    if getattr(indexer.config, "dsa_indexer_mode", "standard") == "simplified":
        use_learned_k = getattr(indexer.config, "dsa_simplified_use_learned_k", False)
        return DSASimplifiedDenseIndexerLossFn.apply(
            query,
            key,
            hidden_states,
            _module_weight(indexer.linear_q),
            (
                _module_weight(indexer.linear_k)
                if use_learned_k
                else query.new_empty((0,))
            ),
            indexer.index_head_dim,
            indexer.index_rotary_dim,
            indexer.rotary_pos_emb,
            use_indexer_rope,
            softmax_scale,
            indexer.softmax_scale,
            loss_coeff,
            _chunk_size(
                query_chunk_size, _default_query_chunk_size(query.size(0)), query.size(0)
            ),
            _chunk_size(key_chunk_size, _default_key_chunk_size(key.size(0)), key.size(0)),
            indexer.pg_collection,
            getattr(indexer.config, "rotary_interleaved", False),
            simplified_input_norm,
            profile_enabled,
            profile_rank,
            profile_label,
            use_triton,
        )


def dsa_min_memory_gqa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    hidden_states: torch.Tensor,
    indexer,
    softmax_scale: float,
    loss_coeff: float,
    use_indexer_rope: bool,
    query_chunk_size: Optional[int],
    key_chunk_size: Optional[int],
    cache_routing: bool = False,
    cache_indexer_k: bool = False,
    cache_selected_scores: bool = False,
    profile_enabled: bool = False,
    profile_rank: int = 0,
    profile_label: str = "",
    use_triton: bool = True,
    use_cudnn: bool = False,
    simplified_input_norm=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the minimum-activation DSA-GQA training backend."""
    if getattr(indexer.config, "dsa_indexer_mode", "standard") == "simplified":
        use_learned_k = getattr(indexer.config, "dsa_simplified_use_learned_k", False)
        if cache_indexer_k and not use_learned_k:
            raise ValueError(
                "Simplified DSA using main-attention K cannot cache a separate indexer K."
            )
        return DSASimplifiedMinMemoryGQAFn.apply(
            query,
            key,
            value,
            hidden_states,
            _module_weight(indexer.linear_q),
            (
                _module_weight(indexer.linear_k)
                if use_learned_k
                else query.new_empty((0,))
            ),
            indexer.index_topk,
            indexer.index_head_dim,
            indexer.index_rotary_dim,
            indexer.rotary_pos_emb,
            use_indexer_rope,
            softmax_scale,
            indexer.softmax_scale,
            loss_coeff,
            _chunk_size(
                query_chunk_size, _default_query_chunk_size(query.size(0)), query.size(0)
            ),
            _routing_key_chunk_size(key_chunk_size, key.size(0), use_triton),
            indexer.pg_collection,
            getattr(indexer.config, "rotary_interleaved", False),
            simplified_input_norm,
            profile_enabled,
            profile_rank,
            profile_label,
            cache_routing,
            cache_selected_scores,
            cache_indexer_k,
            use_triton,
        )
