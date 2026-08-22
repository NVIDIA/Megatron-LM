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
    triton_gathered_linear_wgrad,
    triton_indexer_loss_grad,
    triton_k_ln_backward_prepare,
    triton_k_ln_param_reduce,
    triton_linear_wgrad,
    triton_selected_k_linear,
    triton_selected_index_scores_from_hidden,
    triton_selected_index_scores,
    triton_selected_index_scores_backward,
    triton_selected_index_kl_loss,
    triton_sparse_attention_backward_accumulate,
    triton_sparse_attention_backward_path,
    triton_sparse_attention_backward_supported,
    triton_sparse_attention_tile,
    triton_teacher_scores_tile,
    triton_topk_index_block,
    triton_scatter_selected_grad_to_sequence,
)


def _module_weight(module) -> torch.Tensor:
    weight = getattr(module, "weight", None)
    if weight is None:
        raise RuntimeError(f"{module.__class__.__name__} does not expose a weight tensor.")
    return weight


def _module_bias(module, like: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    bias = getattr(module, "bias", None)
    if bias is None:
        return like.new_empty((0,)), False
    return bias, True


def _grad_accumulator(tensor: torch.Tensor) -> torch.Tensor:
    dtype = torch.float32 if tensor.dtype in (torch.float16, torch.bfloat16) else tensor.dtype
    return torch.zeros(tensor.shape, device=tensor.device, dtype=dtype)


def _distributed_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


class _DSATimingProfiler:
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
                order.append(name)
            totals[name] += elapsed_ms

        label = f" {self.label}" if self.label else ""
        parts = " ".join(f"{name}={totals[name]:.3f}ms" for name in order)
        print(f"[rank{self.rank}] DSA min-memory {phase}{label}: {parts}", flush=True)


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


def _default_topk_score_chunk_size(topk: int) -> int:
    return min(topk, 128)


def _native_wgrad_topk_score_chunk_size(topk: int) -> int:
    # Keeping the full support for a query tile lets backward reuse selected-K, fuse the
    # sparse-KL score gradient, and compute Q-side WGRAD once per query tile.
    if topk <= 512:
        return topk
    return _default_topk_score_chunk_size(topk)


def _chunk_size(config_value: Optional[int], default_value: int, maximum: int) -> int:
    if config_value is None or config_value <= 0:
        return min(default_value, maximum)
    return min(config_value, maximum)


def _routing_key_chunk_size(
    config_value: Optional[int], key_length: int, use_triton: bool
) -> int:
    if not use_triton:
        # The PyTorch backend is the numerical oracle. Streaming torch.topk over key chunks is
        # not tie-equivalent to a single full torch.topk, and exact zero ties are common after
        # the indexer ReLU. Use one key block so torch-min-memory preserves reference routing.
        return key_length
    return _chunk_size(config_value, _default_key_chunk_size(key_length), key_length)


def _default_rotary_interleaved(rotary_pos_emb) -> bool:
    return getattr(rotary_pos_emb, "rotary_interleaved", False)


def _linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.linear(x, weight, None)


def _layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    has_bias: bool,
    eps: float,
) -> torch.Tensor:
    return F.layer_norm(x, (x.size(-1),), weight, bias if has_bias else None, eps)


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


def _rope_triton_inputs(
    rotary_pos_emb,
    device: torch.device,
    index_rotary_dim: int,
) -> Tuple[torch.Tensor, float, float]:
    if rotary_pos_emb is None or index_rotary_dim == 0:
        return torch.empty((1,), device=device, dtype=torch.float32), 1.0, 1.0

    inv_freq, mscale = _rope_inv_freq_and_mscale(rotary_pos_emb, device)
    inv_freq = inv_freq[: index_rotary_dim // 2]
    interpolation_factor = getattr(rotary_pos_emb, "seq_len_interpolation_factor", None)
    if interpolation_factor is not None and not isinstance(rotary_pos_emb, YarnRotaryEmbedding):
        interpolation_scale = 1.0 / interpolation_factor
    else:
        interpolation_scale = 1.0
    return inv_freq, mscale, interpolation_scale


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


def _project_q_index_tile(
    hidden_states: torch.Tensor,
    q_start: int,
    q_end: int,
    linear_q_weight: torch.Tensor,
    linear_weights_weight: torch.Tensor,
    index_n_heads: int,
    index_head_dim: int,
    index_rotary_dim: int,
    rotary_pos_emb,
    rotary_interleaved: bool,
    use_indexer_rope: bool,
    use_hadamard: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    hidden_tile = hidden_states[q_start:q_end]
    q = _linear(hidden_tile, linear_q_weight)
    q = q.reshape(q_end - q_start, hidden_states.size(1), index_n_heads, index_head_dim)
    if use_indexer_rope:
        positions = torch.arange(q_start, q_end, device=q.device, dtype=torch.long)
        q = _apply_rope_at_positions(
            q, positions, index_head_dim, index_rotary_dim, rotary_pos_emb, rotary_interleaved
        )
    if use_hadamard:
        q = rotate_activation(q)

    weights = _linear(hidden_tile, linear_weights_weight)
    weights = weights * (index_n_heads**-0.5) * (index_head_dim**-0.5)
    return q, weights


def _project_k_index_block(
    hidden_states: torch.Tensor,
    k_start: int,
    k_end: int,
    linear_k_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    has_k_norm_bias: bool,
    k_norm_eps: float,
    index_head_dim: int,
    index_rotary_dim: int,
    rotary_pos_emb,
    rotary_interleaved: bool,
    use_indexer_rope: bool,
    use_hadamard: bool,
) -> torch.Tensor:
    k = _linear(hidden_states[k_start:k_end], linear_k_weight)
    k = _layer_norm(k, k_norm_weight, k_norm_bias, has_k_norm_bias, k_norm_eps)
    if use_indexer_rope:
        k = k.reshape(k_end - k_start, hidden_states.size(1), 1, index_head_dim)
        positions = torch.arange(k_start, k_end, device=k.device, dtype=torch.long)
        k = _apply_rope_at_positions(
            k, positions, index_head_dim, index_rotary_dim, rotary_pos_emb, rotary_interleaved
        )
        k = k.reshape(k_end - k_start, hidden_states.size(1), index_head_dim)
    if use_hadamard:
        k = rotate_activation(k)
    return k


def _gather_selected_hidden(
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
) -> torch.Tensor:
    batch_size = topk_indices.size(0)
    hidden_by_batch = hidden_states.permute(1, 0, 2)
    batch_index = torch.arange(batch_size, device=topk_indices.device).view(batch_size, 1, 1)
    return hidden_by_batch[batch_index, topk_indices]


def _gather_selected_indexer_k(
    full_k_index: torch.Tensor,
    topk_indices: torch.Tensor,
) -> torch.Tensor:
    batch_size = topk_indices.size(0)
    k_by_batch = full_k_index.permute(1, 0, 2)
    batch_index = torch.arange(batch_size, device=topk_indices.device).view(batch_size, 1, 1)
    return k_by_batch[batch_index, topk_indices]


def _project_selected_k_linear_for_wgrad(
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    linear_k_weight: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    selected_hidden = None
    k_linear = triton_selected_k_linear(hidden_states, topk_indices, linear_k_weight)
    if k_linear is None:
        selected_hidden = _gather_selected_hidden(hidden_states, topk_indices)
        k_linear = _linear(selected_hidden, linear_k_weight)
    return selected_hidden, k_linear


def _project_selected_k_index(
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    linear_k_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    has_k_norm_bias: bool,
    k_norm_eps: float,
    index_head_dim: int,
    index_rotary_dim: int,
    rotary_pos_emb,
    rotary_interleaved: bool,
    use_indexer_rope: bool,
    use_hadamard: bool,
) -> torch.Tensor:
    return _project_selected_k_index_for_wgrad(
        hidden_states,
        topk_indices,
        linear_k_weight,
        k_norm_weight,
        k_norm_bias,
        has_k_norm_bias,
        k_norm_eps,
        index_head_dim,
        index_rotary_dim,
        rotary_pos_emb,
        rotary_interleaved,
        use_indexer_rope,
        use_hadamard,
    )[2]


def _project_selected_k_index_for_wgrad(
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    linear_k_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    has_k_norm_bias: bool,
    k_norm_eps: float,
    index_head_dim: int,
    index_rotary_dim: int,
    rotary_pos_emb,
    rotary_interleaved: bool,
    use_indexer_rope: bool,
    use_hadamard: bool,
) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    selected_hidden, k_linear = _project_selected_k_linear_for_wgrad(
        hidden_states, topk_indices, linear_k_weight
    )
    k = _layer_norm(k_linear, k_norm_weight, k_norm_bias, has_k_norm_bias, k_norm_eps)
    if use_indexer_rope:
        k = _apply_rope_at_positions(
            k,
            topk_indices,
            index_head_dim,
            index_rotary_dim,
            rotary_pos_emb,
            rotary_interleaved,
        )
    if use_hadamard:
        k = rotate_activation(k)
    return selected_hidden, k_linear, k


def _index_scores_for_block(
    q_index: torch.Tensor,
    weights: torch.Tensor,
    k_index: torch.Tensor,
) -> torch.Tensor:
    scores = torch.einsum("qbhd,tbd->bqht", q_index.float(), k_index.float())
    scores = torch.relu(scores)
    scores = scores * weights.permute(1, 0, 2).unsqueeze(-1).float()
    return scores.sum(dim=2)


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


def _dense_indexer_softmax_stats(
    q_index: torch.Tensor,
    weights: torch.Tensor,
    query_tile: torch.Tensor,
    key: torch.Tensor,
    hidden_states: torch.Tensor,
    q_start: int,
    q_end: int,
    linear_k_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    has_k_norm_bias: bool,
    k_norm_eps: float,
    index_head_dim: int,
    index_rotary_dim: int,
    rotary_pos_emb,
    rotary_interleaved: bool,
    use_indexer_rope: bool,
    use_hadamard: bool,
    softmax_scale: float,
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
                query_tile, key[k_start:k_end], softmax_scale, q_start, k_start
            )
            teacher_max, teacher_sum = _update_running_softmax_stats(
                teacher_logits, teacher_max, teacher_sum
            )

        with _profile_record(
            profile, f"dense_indexer_kl_{profile_suffix}_student_stats", query_tile.device
        ):
            k_index = _project_k_index_block(
                hidden_states,
                k_start,
                k_end,
                linear_k_weight,
                k_norm_weight,
                k_norm_bias,
                has_k_norm_bias,
                k_norm_eps,
                index_head_dim,
                index_rotary_dim,
                rotary_pos_emb,
                rotary_interleaved,
                use_indexer_rope,
                use_hadamard,
            )
            student_logits = _index_scores_for_block(q_index, weights, k_index)
            student_logits = _mask_dense_causal_scores(
                student_logits, q_start, q_end, k_start, k_end
            )
            student_max, student_sum = _update_running_softmax_stats(
                student_logits, student_max, student_sum
            )

    assert teacher_max is not None and teacher_sum is not None
    assert student_max is not None and student_sum is not None
    return teacher_max, teacher_sum, student_max, student_sum


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


def _dense_indexer_kl_loss_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    hidden_states: torch.Tensor,
    linear_q_weight: torch.Tensor,
    linear_k_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    has_k_norm_bias: bool,
    linear_weights_weight: torch.Tensor,
    k_norm_eps: float,
    index_n_heads: int,
    index_head_dim: int,
    index_rotary_dim: int,
    rotary_pos_emb,
    use_indexer_rope: bool,
    use_hadamard: bool,
    softmax_scale: float,
    loss_coeff: float,
    query_chunk_size: int,
    key_chunk_size: int,
    pg_collection: ProcessGroupCollection,
    rotary_interleaved: bool,
    profile: Optional[_DSATimingProfiler],
) -> torch.Tensor:
    total_kl = query.new_zeros((), dtype=torch.float32)
    total_positions = query.size(0) * query.size(1)
    for q_start in range(0, query.size(0), query_chunk_size):
        q_end = min(q_start + query_chunk_size, query.size(0))
        query_tile = query[q_start:q_end]
        with _profile_record(profile, "dense_indexer_kl_fwd_q_project", query.device):
            q_index, weights = _project_q_index_tile(
                hidden_states,
                q_start,
                q_end,
                linear_q_weight,
                linear_weights_weight,
                index_n_heads,
                index_head_dim,
                index_rotary_dim,
                rotary_pos_emb,
                rotary_interleaved,
                use_indexer_rope,
                use_hadamard,
            )
        teacher_max, teacher_sum, student_max, student_sum = _dense_indexer_softmax_stats(
            q_index,
            weights,
            query_tile,
            key,
            hidden_states,
            q_start,
            q_end,
            linear_k_weight,
            k_norm_weight,
            k_norm_bias,
            has_k_norm_bias,
            k_norm_eps,
            index_head_dim,
            index_rotary_dim,
            rotary_pos_emb,
            rotary_interleaved,
            use_indexer_rope,
            use_hadamard,
            softmax_scale,
            key_chunk_size,
            profile,
            "fwd",
        )
        with _profile_record(profile, "dense_indexer_kl_fwd_loss", query.device):
            teacher_norm = _dense_teacher_norm(
                query_tile,
                key,
                teacher_max,
                teacher_sum,
                softmax_scale,
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
                    softmax_scale,
                    q_start,
                    k_start,
                    pg_collection,
                )
                teacher = teacher / teacher_norm.unsqueeze(-1)
                k_index = _project_k_index_block(
                    hidden_states,
                    k_start,
                    k_end,
                    linear_k_weight,
                    k_norm_weight,
                    k_norm_bias,
                    has_k_norm_bias,
                    k_norm_eps,
                    index_head_dim,
                    index_rotary_dim,
                    rotary_pos_emb,
                    rotary_interleaved,
                    use_indexer_rope,
                    use_hadamard,
                )
                student_logits = _index_scores_for_block(q_index, weights, k_index)
                student_logits = _mask_dense_causal_scores(
                    student_logits, q_start, q_end, k_start, k_end
                )
                student = (
                    torch.exp(student_logits - student_max.unsqueeze(-1))
                    / student_sum.unsqueeze(-1)
                )
                total_kl = total_kl + (
                    teacher * (torch.log(teacher + 1.0e-10) - torch.log(student + 1.0e-10))
                ).sum()
    return total_kl / total_positions * loss_coeff


def _index_scores_for_selected(
    q_index: torch.Tensor,
    weights: torch.Tensor,
    selected_k_index: torch.Tensor,
) -> torch.Tensor:
    q_index = q_index.permute(1, 0, 2, 3)
    weights = weights.permute(1, 0, 2)
    scores = torch.einsum("bqhd,bqkd->bqhk", q_index.float(), selected_k_index.float())
    scores = torch.relu(scores)
    scores = scores * weights.unsqueeze(-1).float()
    return scores.sum(dim=2)


def _selected_index_scores_backward_torch(
    q_index: torch.Tensor,
    weights: torch.Tensor,
    selected_k_index: torch.Tensor,
    topk_indices: torch.Tensor,
    grad_selected_scores: torch.Tensor,
    q_start: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = q_index.permute(1, 0, 2, 3).to(dtype=torch.float32)
    w = weights.permute(1, 0, 2).to(dtype=torch.float32)
    k = selected_k_index.to(dtype=torch.float32)
    grad_scores = grad_selected_scores.to(dtype=torch.float32)

    invalid = _selected_causal_invalid_mask(topk_indices, q_start)
    grad_scores = grad_scores.masked_fill(invalid, 0.0)

    dot = torch.einsum("bqhd,bqkd->bqhk", q, k)
    relu_mask = dot > 0
    relu_dot = torch.relu(dot)

    grad_weights = (grad_scores.unsqueeze(2) * relu_dot).sum(dim=-1)
    grad_dot = grad_scores.unsqueeze(2) * w.unsqueeze(-1) * relu_mask.to(dtype=torch.float32)
    grad_q_index = torch.einsum("bqhk,bqkd->bqhd", grad_dot, k)
    grad_selected_k = torch.einsum("bqhk,bqhd->bqkd", grad_dot, q)

    return (
        grad_q_index.permute(1, 0, 2, 3).contiguous(),
        grad_weights.permute(1, 0, 2).contiguous(),
        grad_selected_k.contiguous(),
    )


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


def _layer_norm_backward_manual(
    grad_output: torch.Tensor,
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_float = input_tensor.to(dtype=torch.float32)
    grad_float = grad_output.to(dtype=torch.float32)
    weight_float = weight.to(dtype=torch.float32)

    mean = input_float.mean(dim=-1, keepdim=True)
    centered = input_float - mean
    variance = centered.pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(variance + eps)
    normalized = centered * rstd

    grad_weight = (grad_float * normalized).sum(dim=tuple(range(grad_float.dim() - 1)))
    grad_bias = grad_float.sum(dim=tuple(range(grad_float.dim() - 1)))

    grad_normalized = grad_float * weight_float
    grad_input = (
        grad_normalized
        - grad_normalized.mean(dim=-1, keepdim=True)
        - normalized * (grad_normalized * normalized).mean(dim=-1, keepdim=True)
    ) * rstd
    return grad_input, grad_weight, grad_bias


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


def _topk_index_tile(
    hidden_states: torch.Tensor,
    q_start: int,
    q_end: int,
    linear_q_weight: torch.Tensor,
    linear_k_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    has_k_norm_bias: bool,
    linear_weights_weight: torch.Tensor,
    k_norm_eps: float,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    index_rotary_dim: int,
    rotary_pos_emb,
    rotary_interleaved: bool,
    use_indexer_rope: bool,
    use_hadamard: bool,
    key_chunk_size: int,
    profile: Optional[_DSATimingProfiler] = None,
    profile_suffix: str = "fwd",
    full_k_index: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with _profile_record(profile, f"routing_q_project_{profile_suffix}", hidden_states.device):
        q_index, weights = _project_q_index_tile(
            hidden_states,
            q_start,
            q_end,
            linear_q_weight,
            linear_weights_weight,
            index_n_heads,
            index_head_dim,
            index_rotary_dim,
            rotary_pos_emb,
            rotary_interleaved,
            use_indexer_rope,
            use_hadamard,
        )
    causal_key_limit = min(q_end, hidden_states.size(0))
    topk = min(index_topk, causal_key_limit)
    running_scores = None
    running_indices = None
    for k_start in range(0, causal_key_limit, key_chunk_size):
        k_end = min(k_start + key_chunk_size, causal_key_limit)
        if full_k_index is None:
            with _profile_record(
                profile, f"routing_k_project_{profile_suffix}", hidden_states.device
            ):
                k_index = _project_k_index_block(
                    hidden_states,
                    k_start,
                    k_end,
                    linear_k_weight,
                    k_norm_weight,
                    k_norm_bias,
                    has_k_norm_bias,
                    k_norm_eps,
                    index_head_dim,
                    index_rotary_dim,
                    rotary_pos_emb,
                    rotary_interleaved,
                    use_indexer_rope,
                    use_hadamard,
                )
        else:
            with _profile_record(
                profile, f"routing_k_cache_{profile_suffix}", hidden_states.device
            ):
                k_index = full_k_index[k_start:k_end]
        block_topk = min(topk, k_end - k_start)
        with _profile_record(
            profile, f"routing_block_score_topk_{profile_suffix}", hidden_states.device
        ):
            triton_topk = triton_topk_index_block(
                q_index, weights, k_index, block_topk, q_start, k_start
            )
            if triton_topk is None:
                block_scores = _index_scores_for_block(q_index, weights, k_index)
                invalid = _causal_invalid_mask(q_start, q_end, k_start, k_end, block_scores.device)
                block_scores = block_scores.masked_fill(invalid.unsqueeze(0), float("-inf"))
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
    return running_scores, running_indices, q_index, weights


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


def _indexer_loss_from_teacher_and_projected(
    q_index: torch.Tensor,
    weights: torch.Tensor,
    selected_k_index: torch.Tensor,
    topk_indices: torch.Tensor,
    teacher: torch.Tensor,
    loss_coeff: float,
    total_positions: int,
    q_start: int,
    profile: Optional[_DSATimingProfiler] = None,
) -> torch.Tensor:
    loss_scale = loss_coeff / total_positions
    with _profile_record(profile, "selected_index_scores_fwd_score", selected_k_index.device):
        fused_loss = triton_selected_index_kl_loss(
            q_index,
            weights,
            selected_k_index,
            topk_indices,
            teacher,
            loss_scale,
            q_start,
        )
        if fused_loss is not None:
            return fused_loss

    with _profile_record(profile, "selected_index_scores_fwd_score_fallback", selected_k_index.device):
        selected_index_scores = triton_selected_index_scores(
            q_index, weights, selected_k_index, topk_indices, q_start
        )
        if selected_index_scores is None:
            selected_index_scores = _index_scores_for_selected(q_index, weights, selected_k_index)
            invalid = _selected_causal_invalid_mask(topk_indices, q_start)
            selected_index_scores = selected_index_scores.masked_fill(invalid, float("-inf"))
    with _profile_record(profile, "indexer_kl_fwd", selected_k_index.device):
        student = torch.nn.functional.softmax(selected_index_scores, dim=-1, dtype=torch.float32)
        kl = teacher * (torch.log(teacher + 1e-10) - torch.log(student + 1e-10))
        return kl.sum() * loss_scale


def _selected_index_scores_tile(
    hidden_states: torch.Tensor,
    q_start: int,
    q_end: int,
    topk_indices: torch.Tensor,
    q_index: torch.Tensor,
    weights: torch.Tensor,
    linear_k_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    has_k_norm_bias: bool,
    k_norm_eps: float,
    index_head_dim: int,
    index_rotary_dim: int,
    rotary_pos_emb,
    rotary_interleaved: bool,
    use_indexer_rope: bool,
    use_hadamard: bool,
    profile: Optional[_DSATimingProfiler] = None,
    profile_prefix: Optional[str] = None,
) -> torch.Tensor:
    if profile_prefix == "selected_index_scores_fwd":
        project_name = "selected_index_scores_fwd_project_k"
    else:
        project_name = f"{profile_prefix}_project_selected_k" if profile_prefix else ""
    with _profile_record(profile, project_name, hidden_states.device):
        selected_k_index = _project_selected_k_index(
            hidden_states,
            topk_indices,
            linear_k_weight,
            k_norm_weight,
            k_norm_bias,
            has_k_norm_bias,
            k_norm_eps,
            index_head_dim,
            index_rotary_dim,
            rotary_pos_emb,
            rotary_interleaved,
            use_indexer_rope,
            use_hadamard,
        )
    return _selected_index_scores_from_projected(
        q_index,
        weights,
        selected_k_index,
        topk_indices,
        q_start,
        profile=profile,
        profile_prefix=profile_prefix,
    )


def _selected_index_scores_from_hidden_fused(
    hidden_states: torch.Tensor,
    q_start: int,
    topk_indices: torch.Tensor,
    q_index: torch.Tensor,
    weights: torch.Tensor,
    linear_k_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    has_k_norm_bias: bool,
    k_norm_eps: float,
    index_rotary_dim: int,
    rotary_pos_emb,
    rotary_interleaved: bool,
    use_indexer_rope: bool,
    use_hadamard: bool,
    return_k_linear: bool,
) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    inv_freq, mscale, interpolation_scale = _rope_triton_inputs(
        rotary_pos_emb if use_indexer_rope else None,
        hidden_states.device,
        index_rotary_dim,
    )
    return triton_selected_index_scores_from_hidden(
        hidden_states,
        topk_indices,
        linear_k_weight,
        k_norm_weight,
        k_norm_bias,
        q_index,
        weights,
        inv_freq,
        q_start,
        k_norm_eps,
        index_rotary_dim,
        rotary_interleaved,
        use_indexer_rope,
        use_hadamard,
        has_k_norm_bias,
        mscale,
        interpolation_scale,
        return_k_linear=return_k_linear,
    )


def _selected_index_scores_from_projected(
    q_index: torch.Tensor,
    weights: torch.Tensor,
    selected_k_index: torch.Tensor,
    topk_indices: torch.Tensor,
    q_start: int,
    profile: Optional[_DSATimingProfiler] = None,
    profile_prefix: Optional[str] = None,
) -> torch.Tensor:
    if profile_prefix == "selected_index_scores_fwd":
        score_name = "selected_index_scores_fwd_score"
        fallback_name = "selected_index_scores_fwd_score_fallback"
    else:
        score_name = f"{profile_prefix}_selected_score" if profile_prefix else ""
        fallback_name = f"{profile_prefix}_selected_score_fallback" if profile_prefix else ""
    with _profile_record(profile, score_name, selected_k_index.device):
        triton_scores = triton_selected_index_scores(
            q_index, weights, selected_k_index, topk_indices, q_start
        )
        if triton_scores is not None:
            return triton_scores
    with _profile_record(profile, fallback_name, selected_k_index.device):
        selected_scores = _index_scores_for_selected(q_index, weights, selected_k_index)
        invalid = _selected_causal_invalid_mask(topk_indices, q_start)
        return selected_scores.masked_fill(invalid, float("-inf"))


def _selected_index_scores_tile_chunked(
    hidden_states: torch.Tensor,
    q_start: int,
    q_end: int,
    topk_indices: torch.Tensor,
    q_index: torch.Tensor,
    weights: torch.Tensor,
    linear_k_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    has_k_norm_bias: bool,
    k_norm_eps: float,
    index_head_dim: int,
    index_rotary_dim: int,
    rotary_pos_emb,
    rotary_interleaved: bool,
    use_indexer_rope: bool,
    use_hadamard: bool,
    topk_score_chunk_size: int,
    profile: Optional[_DSATimingProfiler] = None,
    profile_prefix: Optional[str] = None,
) -> torch.Tensor:
    if topk_score_chunk_size <= 0 or topk_score_chunk_size >= topk_indices.size(-1):
        return _selected_index_scores_tile(
            hidden_states,
            q_start,
            q_end,
            topk_indices,
            q_index,
            weights,
            linear_k_weight,
            k_norm_weight,
            k_norm_bias,
            has_k_norm_bias,
            k_norm_eps,
            index_head_dim,
            index_rotary_dim,
            rotary_pos_emb,
            rotary_interleaved,
            use_indexer_rope,
            use_hadamard,
            profile=profile,
            profile_prefix=profile_prefix,
        )

    score_chunks = []
    for topk_start in range(0, topk_indices.size(-1), topk_score_chunk_size):
        topk_end = min(topk_start + topk_score_chunk_size, topk_indices.size(-1))
        score_chunks.append(
            _selected_index_scores_tile(
                hidden_states,
                q_start,
                q_end,
                topk_indices[..., topk_start:topk_end],
                q_index,
                weights,
                linear_k_weight,
                k_norm_weight,
                k_norm_bias,
                has_k_norm_bias,
                k_norm_eps,
                index_head_dim,
                index_rotary_dim,
                rotary_pos_emb,
                rotary_interleaved,
                use_indexer_rope,
                use_hadamard,
                profile=profile,
                profile_prefix=profile_prefix,
            )
        )
    return torch.cat(score_chunks, dim=-1)


def _native_indexer_loss_wgrad_chunk(
    hidden_states: torch.Tensor,
    q_start: int,
    q_end: int,
    topk_indices: torch.Tensor,
    q_index: torch.Tensor,
    weights: torch.Tensor,
    grad_selected_scores: Optional[torch.Tensor],
    linear_q_weight: torch.Tensor,
    linear_k_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    has_k_norm_bias: bool,
    linear_weights_weight: torch.Tensor,
    k_norm_eps: float,
    index_head_dim: int,
    index_rotary_dim: int,
    rotary_pos_emb,
    rotary_interleaved: bool,
    use_indexer_rope: bool,
    use_hadamard: bool,
    grad_linear_q_weight: Optional[torch.Tensor],
    grad_linear_k_weight: Optional[torch.Tensor],
    grad_k_norm_weight: Optional[torch.Tensor],
    grad_k_norm_bias: Optional[torch.Tensor],
    grad_linear_weights_weight: Optional[torch.Tensor],
    profile: Optional[_DSATimingProfiler],
    selected_hidden: Optional[torch.Tensor] = None,
    k_linear: Optional[torch.Tensor] = None,
    selected_k_index: Optional[torch.Tensor] = None,
    selected_scores: Optional[torch.Tensor] = None,
    teacher: Optional[torch.Tensor] = None,
    loss_scale: Optional[torch.Tensor] = None,
) -> bool:
    with _profile_record(profile, "indexer_loss_bwd_native_total", hidden_states.device):
        if k_linear is None:
            with _profile_record(
                profile, "indexer_loss_bwd_native_project_selected_k", hidden_states.device
            ):
                selected_hidden, k_linear, selected_k_index = _project_selected_k_index_for_wgrad(
                    hidden_states,
                    topk_indices,
                    linear_k_weight,
                    k_norm_weight,
                    k_norm_bias,
                    has_k_norm_bias,
                    k_norm_eps,
                    index_head_dim,
                    index_rotary_dim,
                    rotary_pos_emb,
                    rotary_interleaved,
                    use_indexer_rope,
                    use_hadamard,
                )

        if grad_selected_scores is None:
            return False
        else:
            with _profile_record(
                profile, "indexer_loss_bwd_selected_score_bwd", hidden_states.device
            ):
                if selected_k_index is None:
                    return False
                selected_score_grads = triton_selected_index_scores_backward(
                    q_index,
                    weights,
                    selected_k_index,
                    topk_indices,
                    grad_selected_scores,
                    q_start,
                )
                if selected_score_grads is None:
                    selected_score_grads = _selected_index_scores_backward_torch(
                        q_index,
                        weights,
                        selected_k_index,
                        topk_indices,
                        grad_selected_scores,
                        q_start,
                    )
                grad_q_index, grad_weights, grad_selected_k = selected_score_grads

        hidden_tile = hidden_states[q_start:q_end]
        with _profile_record(profile, "indexer_loss_bwd_native_q_wgrad", hidden_states.device):
            if grad_linear_q_weight is not None:
                query_positions = torch.arange(
                    q_start, q_end, device=hidden_states.device, dtype=torch.long
                )
                grad_q_linear = _backward_indexer_transform(
                    grad_q_index,
                    query_positions,
                    index_head_dim,
                    index_rotary_dim,
                    rotary_pos_emb,
                    rotary_interleaved,
                    use_indexer_rope,
                    use_hadamard,
                )
                grad_q_linear = grad_q_linear.reshape(
                    q_end - q_start, hidden_states.size(1), -1
                )
                _accumulate_linear_weight_grad(
                    grad_linear_q_weight, grad_q_linear, hidden_tile
                )

            if grad_linear_weights_weight is not None:
                weights_scale = (q_index.size(2) ** -0.5) * (index_head_dim ** -0.5)
                _accumulate_linear_weight_grad(
                    grad_linear_weights_weight, grad_weights * weights_scale, hidden_tile
                )

        with _profile_record(profile, "indexer_loss_bwd_native_k_ln_wgrad", hidden_states.device):
            if (
                grad_linear_k_weight is not None
                or grad_k_norm_weight is not None
                or (has_k_norm_bias and grad_k_norm_bias is not None)
            ):
                with _profile_record(
                    profile, "indexer_loss_bwd_inverse_indexer_transform", hidden_states.device
                ):
                    grad_k_norm = _backward_indexer_transform(
                        grad_selected_k,
                        topk_indices,
                        index_head_dim,
                        index_rotary_dim,
                        rotary_pos_emb,
                        rotary_interleaved,
                        use_indexer_rope,
                        use_hadamard,
                    )
                grad_k_linear_dtype = (
                    torch.float32 if hidden_states.dtype == torch.float32 else hidden_states.dtype
                )
                with _profile_record(
                    profile, "indexer_loss_bwd_ln_row_backward", hidden_states.device
                ):
                    prepared_ln = triton_k_ln_backward_prepare(
                        grad_k_norm,
                        k_linear,
                        k_norm_weight,
                        k_norm_eps,
                        grad_k_norm_weight,
                        grad_k_norm_bias if has_k_norm_bias else None,
                        grad_k_linear_dtype,
                    )
                if prepared_ln is not None:
                    grad_k_linear, partial_norm_weight, partial_norm_bias = prepared_ln
                    if grad_linear_k_weight is not None:
                        with _profile_record(
                            profile, "indexer_loss_bwd_linear_k_wgrad", hidden_states.device
                        ):
                            wgrad_done = False
                            with _profile_record(
                                profile,
                                "indexer_loss_bwd_linear_k_wgrad_compress",
                                hidden_states.device,
                            ):
                                compressed_grad_k_linear = (
                                    triton_scatter_selected_grad_to_sequence(
                                        grad_k_linear,
                                        topk_indices,
                                        hidden_states.size(0),
                                    )
                                )
                            if compressed_grad_k_linear is not None:
                                with _profile_record(
                                    profile,
                                    "indexer_loss_bwd_linear_k_wgrad_dense",
                                    hidden_states.device,
                                ):
                                    _accumulate_linear_weight_grad(
                                        grad_linear_k_weight,
                                        compressed_grad_k_linear,
                                        hidden_states,
                                    )
                                wgrad_done = True
                            if not wgrad_done:
                                wgrad_done = triton_gathered_linear_wgrad(
                                    grad_k_linear,
                                    hidden_states,
                                    topk_indices,
                                    grad_linear_k_weight,
                                )
                            if not wgrad_done:
                                prepared_ln = None
                    if prepared_ln is not None:
                        with _profile_record(
                            profile, "indexer_loss_bwd_ln_param_reduce", hidden_states.device
                        ):
                            triton_k_ln_param_reduce(
                                partial_norm_weight,
                                partial_norm_bias,
                                grad_k_norm_weight,
                                grad_k_norm_bias if has_k_norm_bias else None,
                            )
                        return True

                if selected_hidden is None:
                    selected_hidden = _gather_selected_hidden(hidden_states, topk_indices)
                with _profile_record(profile, "indexer_loss_bwd_ln_backward", hidden_states.device):
                    grad_k_linear, grad_norm_weight, grad_norm_bias = _layer_norm_backward_manual(
                        grad_k_norm,
                        k_linear,
                        k_norm_weight,
                        k_norm_eps,
                    )
                if grad_k_norm_weight is not None:
                    grad_k_norm_weight.add_(grad_norm_weight)
                if has_k_norm_bias and grad_k_norm_bias is not None:
                    grad_k_norm_bias.add_(grad_norm_bias)
                if grad_linear_k_weight is not None:
                    with _profile_record(
                        profile, "indexer_loss_bwd_linear_k_wgrad", hidden_states.device
                    ):
                        _accumulate_linear_weight_grad(
                            grad_linear_k_weight, grad_k_linear, selected_hidden
                        )

    return True


def _forward_min_memory_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    hidden_states: torch.Tensor,
    linear_q_weight: torch.Tensor,
    linear_k_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    has_k_norm_bias: bool,
    linear_weights_weight: torch.Tensor,
    k_norm_eps: float,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    index_rotary_dim: int,
    rotary_pos_emb,
    use_indexer_rope: bool,
    use_hadamard: bool,
    softmax_scale: float,
    loss_coeff: float,
    query_chunk_size: int,
    key_chunk_size: int,
    pg_collection: ProcessGroupCollection,
    rotary_interleaved: Optional[bool] = None,
    profile: Optional[_DSATimingProfiler] = None,
    routing_topk_cache: Optional[list] = None,
    selected_scores_cache: Optional[list] = None,
    full_k_index: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sq, batch_size, num_query_heads, _ = query.shape
    output = value.new_empty((sq, batch_size, num_query_heads, value.size(-1)))
    indexer_loss = query.new_zeros((), dtype=torch.float32)
    total_positions = batch_size * sq
    if rotary_interleaved is None:
        rotary_interleaved = _default_rotary_interleaved(rotary_pos_emb)

    for q_start in range(0, sq, query_chunk_size):
        q_end = min(q_start + query_chunk_size, sq)
        with _profile_record(profile, "routing_topk_fwd", query.device):
            _, topk_indices, q_index, weights = _topk_index_tile(
                hidden_states,
                q_start,
                q_end,
                linear_q_weight,
                linear_k_weight,
                k_norm_weight,
                k_norm_bias,
                has_k_norm_bias,
                linear_weights_weight,
                k_norm_eps,
                index_n_heads,
                index_head_dim,
                index_topk,
                index_rotary_dim,
                rotary_pos_emb,
                rotary_interleaved,
                use_indexer_rope,
                use_hadamard,
                key_chunk_size,
                profile=profile,
                profile_suffix="fwd",
                full_k_index=full_k_index,
            )
        if routing_topk_cache is not None:
            routing_topk_cache.append(topk_indices)
        query_tile = query[q_start:q_end]
        with _profile_record(profile, "sparse_attention_fwd", query.device):
            output[q_start:q_end] = _sparse_attention_tile(
                query_tile, key, value, topk_indices, softmax_scale, q_start
            )
        if loss_coeff > 0:
            selected_index_scores = None
            with _profile_record(profile, "selected_index_scores_fwd", query.device):
                selected_k_index = None
                if full_k_index is not None:
                    with _profile_record(
                        profile, "selected_index_scores_fwd_gather_k_cache", query.device
                    ):
                        selected_k_index = _gather_selected_indexer_k(
                            full_k_index, topk_indices
                        )
                    selected_index_scores = _selected_index_scores_from_projected(
                        q_index,
                        weights,
                        selected_k_index,
                        topk_indices,
                        q_start,
                        profile=profile,
                        profile_prefix="selected_index_scores_fwd",
                    )
                else:
                    with _profile_record(
                        profile, "selected_index_scores_fwd_project_score_fused", query.device
                    ):
                        fused_scores = _selected_index_scores_from_hidden_fused(
                            hidden_states,
                            q_start,
                            topk_indices,
                            q_index,
                            weights,
                            linear_k_weight,
                            k_norm_weight,
                            k_norm_bias,
                            has_k_norm_bias,
                            k_norm_eps,
                            index_rotary_dim,
                            rotary_pos_emb,
                            rotary_interleaved,
                            use_indexer_rope,
                            use_hadamard,
                            return_k_linear=False,
                        )
                        if fused_scores is not None:
                            selected_index_scores, _ = fused_scores
                    if selected_index_scores is None:
                        with _profile_record(
                            profile, "selected_index_scores_fwd_project_k", query.device
                        ):
                            selected_k_index = _project_selected_k_index(
                                hidden_states,
                                topk_indices,
                                linear_k_weight,
                                k_norm_weight,
                                k_norm_bias,
                                has_k_norm_bias,
                                k_norm_eps,
                                index_head_dim,
                                index_rotary_dim,
                                rotary_pos_emb,
                                rotary_interleaved,
                                use_indexer_rope,
                                use_hadamard,
                            )
            if selected_scores_cache is not None:
                if selected_index_scores is None:
                    selected_index_scores = _selected_index_scores_from_projected(
                        q_index,
                        weights,
                        selected_k_index,
                        topk_indices,
                        q_start,
                        profile=profile,
                        profile_prefix="selected_index_scores_fwd",
                    )
                selected_scores_cache.append(selected_index_scores)
            if selected_index_scores is not None:
                indexer_loss = indexer_loss + _indexer_loss_tile(
                    selected_index_scores,
                    query_tile,
                    key,
                    topk_indices,
                    softmax_scale,
                    loss_coeff,
                    total_positions,
                    q_start,
                    pg_collection,
                    profile=profile,
                    profile_suffix="fwd",
                )
            else:
                teacher = _teacher_scores_tile(
                    query_tile.detach(),
                    key.detach(),
                    topk_indices,
                    softmax_scale,
                    q_start,
                    pg_collection,
                    profile=profile,
                    profile_suffix="fwd",
                )
                indexer_loss = indexer_loss + _indexer_loss_from_teacher_and_projected(
                    q_index,
                    weights,
                    selected_k_index,
                    topk_indices,
                    teacher,
                    loss_coeff,
                    total_positions,
                    q_start,
                    profile=profile,
                )

    return output.reshape(sq, batch_size, num_query_heads * value.size(-1)), indexer_loss


class DSADenseIndexerLossFn(torch.autograd.Function):
    """Tiled dense DSA indexer KL used by dense-attention warmup."""

    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        hidden_states: torch.Tensor,
        linear_q_weight: torch.Tensor,
        linear_k_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        k_norm_bias: torch.Tensor,
        linear_weights_weight: torch.Tensor,
        has_k_norm_bias: bool,
        k_norm_eps: float,
        index_n_heads: int,
        index_head_dim: int,
        index_rotary_dim: int,
        rotary_pos_emb,
        use_indexer_rope: bool,
        use_hadamard: bool,
        softmax_scale: float,
        loss_coeff: float,
        query_chunk_size: int,
        key_chunk_size: int,
        pg_collection: ProcessGroupCollection,
        rotary_interleaved: bool,
        profile_enabled: bool = False,
        profile_rank: int = 0,
        profile_label: str = "",
        use_triton: bool = True,
    ) -> torch.Tensor:
        profile = _DSATimingProfiler(profile_enabled, profile_rank, profile_label, query.device)
        with torch.no_grad(), _triton_dispatch_enabled(use_triton):
            with profile.record("dense_indexer_kl_fwd_total", query.device):
                loss = _dense_indexer_kl_loss_impl(
                    query,
                    key,
                    hidden_states,
                    linear_q_weight,
                    linear_k_weight,
                    k_norm_weight,
                    k_norm_bias,
                    has_k_norm_bias,
                    linear_weights_weight,
                    k_norm_eps,
                    index_n_heads,
                    index_head_dim,
                    index_rotary_dim,
                    rotary_pos_emb,
                    use_indexer_rope,
                    use_hadamard,
                    softmax_scale,
                    loss_coeff,
                    query_chunk_size,
                    key_chunk_size,
                    pg_collection,
                    rotary_interleaved,
                    profile,
                )
        profile.log("forward")

        ctx.save_for_backward(
            query,
            key,
            hidden_states,
            linear_q_weight,
            linear_k_weight,
            k_norm_weight,
            k_norm_bias,
            linear_weights_weight,
        )
        ctx.has_k_norm_bias = has_k_norm_bias
        ctx.k_norm_eps = k_norm_eps
        ctx.index_n_heads = index_n_heads
        ctx.index_head_dim = index_head_dim
        ctx.index_rotary_dim = index_rotary_dim
        ctx.rotary_pos_emb = rotary_pos_emb
        ctx.use_indexer_rope = use_indexer_rope
        ctx.use_hadamard = use_hadamard
        ctx.softmax_scale = softmax_scale
        ctx.loss_coeff = loss_coeff
        ctx.query_chunk_size = query_chunk_size
        ctx.key_chunk_size = key_chunk_size
        ctx.pg_collection = pg_collection
        ctx.rotary_interleaved = rotary_interleaved
        ctx.profile_enabled = profile_enabled
        ctx.profile_rank = profile_rank
        ctx.profile_label = profile_label
        ctx.use_triton = use_triton
        return loss

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor):
        (
            query,
            key,
            hidden_states,
            linear_q_weight,
            linear_k_weight,
            k_norm_weight,
            k_norm_bias,
            linear_weights_weight,
        ) = ctx.saved_tensors

        grad_linear_q_weight = (
            _grad_accumulator(linear_q_weight) if ctx.needs_input_grad[3] else None
        )
        grad_linear_k_weight = (
            _grad_accumulator(linear_k_weight) if ctx.needs_input_grad[4] else None
        )
        grad_k_norm_weight = _grad_accumulator(k_norm_weight) if ctx.needs_input_grad[5] else None
        grad_k_norm_bias = _grad_accumulator(k_norm_bias) if ctx.needs_input_grad[6] else None
        grad_linear_weights_weight = (
            _grad_accumulator(linear_weights_weight) if ctx.needs_input_grad[7] else None
        )
        compute_grads = (
            grad_loss is not None
            and ctx.loss_coeff > 0
            and (
                ctx.needs_input_grad[3]
                or ctx.needs_input_grad[4]
                or ctx.needs_input_grad[5]
                or ctx.needs_input_grad[6]
                or ctx.needs_input_grad[7]
            )
        )
        if not compute_grads:
            return (
                None,
                None,
                None,
                grad_linear_q_weight,
                grad_linear_k_weight,
                grad_k_norm_weight,
                grad_k_norm_bias,
                grad_linear_weights_weight,
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

        profile = _DSATimingProfiler(
            ctx.profile_enabled, ctx.profile_rank, ctx.profile_label, query.device
        )
        total_positions = query.size(0) * query.size(1)
        scale = grad_loss * (ctx.loss_coeff / total_positions)

        with _triton_dispatch_enabled(ctx.use_triton), profile.record(
            "dense_indexer_kl_bwd_total", query.device
        ):
            for q_start in range(0, query.size(0), ctx.query_chunk_size):
                q_end = min(q_start + ctx.query_chunk_size, query.size(0))
                query_tile = query[q_start:q_end]
                with torch.no_grad():
                    q_index_stats, weights_stats = _project_q_index_tile(
                        hidden_states,
                        q_start,
                        q_end,
                        linear_q_weight,
                        linear_weights_weight,
                        ctx.index_n_heads,
                        ctx.index_head_dim,
                        ctx.index_rotary_dim,
                        ctx.rotary_pos_emb,
                        ctx.rotary_interleaved,
                        ctx.use_indexer_rope,
                        ctx.use_hadamard,
                    )
                    teacher_max, teacher_sum, student_max, student_sum = (
                        _dense_indexer_softmax_stats(
                            q_index_stats,
                            weights_stats,
                            query_tile,
                            key,
                            hidden_states,
                            q_start,
                            q_end,
                            linear_k_weight,
                            k_norm_weight,
                            k_norm_bias,
                            ctx.has_k_norm_bias,
                            ctx.k_norm_eps,
                            ctx.index_head_dim,
                            ctx.index_rotary_dim,
                            ctx.rotary_pos_emb,
                            ctx.rotary_interleaved,
                            ctx.use_indexer_rope,
                            ctx.use_hadamard,
                            ctx.softmax_scale,
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
                        ctx.softmax_scale,
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
                            ctx.softmax_scale,
                            q_start,
                            k_start,
                            ctx.pg_collection,
                        )
                        teacher = teacher / teacher_norm.unsqueeze(-1)
                        k_index_stats = _project_k_index_block(
                            hidden_states,
                            k_start,
                            k_end,
                            linear_k_weight,
                            k_norm_weight,
                            k_norm_bias,
                            ctx.has_k_norm_bias,
                            ctx.k_norm_eps,
                            ctx.index_head_dim,
                            ctx.index_rotary_dim,
                            ctx.rotary_pos_emb,
                            ctx.rotary_interleaved,
                            ctx.use_indexer_rope,
                            ctx.use_hadamard,
                        )
                        student_logits = _index_scores_for_block(
                            q_index_stats, weights_stats, k_index_stats
                        )
                        student_logits = _mask_dense_causal_scores(
                            student_logits, q_start, q_end, k_start, k_end
                        )
                        student = (
                            torch.exp(student_logits - student_max.unsqueeze(-1))
                            / student_sum.unsqueeze(-1)
                        )
                        student_grad_norm = student_grad_norm + (
                            teacher * student / (student + 1.0e-10)
                        ).sum(dim=-1)

                lq_weight = linear_q_weight.detach().requires_grad_(ctx.needs_input_grad[3])
                lk_weight = linear_k_weight.detach().requires_grad_(ctx.needs_input_grad[4])
                kn_weight = k_norm_weight.detach().requires_grad_(ctx.needs_input_grad[5])
                kn_bias = k_norm_bias.detach().requires_grad_(ctx.needs_input_grad[6])
                lw_weight = linear_weights_weight.detach().requires_grad_(ctx.needs_input_grad[7])
                loss_inputs = []
                if ctx.needs_input_grad[3]:
                    loss_inputs.append(lq_weight)
                if ctx.needs_input_grad[4]:
                    loss_inputs.append(lk_weight)
                if ctx.needs_input_grad[5]:
                    loss_inputs.append(kn_weight)
                if ctx.needs_input_grad[6]:
                    loss_inputs.append(kn_bias)
                if ctx.needs_input_grad[7]:
                    loss_inputs.append(lw_weight)

                with _profile_record(profile, "dense_indexer_kl_bwd_q_project", query.device):
                    with torch.enable_grad():
                        q_index, weights = _project_q_index_tile(
                            hidden_states,
                            q_start,
                            q_end,
                            lq_weight,
                            lw_weight,
                            ctx.index_n_heads,
                            ctx.index_head_dim,
                            ctx.index_rotary_dim,
                            ctx.rotary_pos_emb,
                            ctx.rotary_interleaved,
                            ctx.use_indexer_rope,
                            ctx.use_hadamard,
                        )

                for k_start in range(0, key.size(0), ctx.key_chunk_size):
                    k_end = min(k_start + ctx.key_chunk_size, key.size(0))
                    with torch.no_grad():
                        teacher = _dense_teacher_mass_block(
                            query_tile,
                            key[k_start:k_end],
                            teacher_max,
                            teacher_sum,
                            ctx.softmax_scale,
                            q_start,
                            k_start,
                            ctx.pg_collection,
                        )
                        teacher = teacher / teacher_norm.unsqueeze(-1)

                    with _profile_record(profile, "dense_indexer_kl_bwd_wgrad", query.device):
                        with torch.enable_grad():
                            k_index = _project_k_index_block(
                                hidden_states,
                                k_start,
                                k_end,
                                lk_weight,
                                kn_weight,
                                kn_bias,
                                ctx.has_k_norm_bias,
                                ctx.k_norm_eps,
                                ctx.index_head_dim,
                                ctx.index_rotary_dim,
                                ctx.rotary_pos_emb,
                                ctx.rotary_interleaved,
                                ctx.use_indexer_rope,
                                ctx.use_hadamard,
                            )
                            student_logits = _index_scores_for_block(q_index, weights, k_index)
                            student_logits = _mask_dense_causal_scores(
                                student_logits, q_start, q_end, k_start, k_end
                            )
                        student = (
                            torch.exp(student_logits.detach() - student_max.unsqueeze(-1))
                            / student_sum.unsqueeze(-1)
                        )
                        alpha = teacher * student / (student + 1.0e-10)
                        grad_scores = (student * student_grad_norm.unsqueeze(-1) - alpha) * scale
                        loss_grads = torch.autograd.grad(
                            student_logits,
                            loss_inputs,
                            grad_outputs=grad_scores,
                            retain_graph=k_end < key.size(0),
                            allow_unused=True,
                        )
                        grad_iter = iter(loss_grads)
                        if ctx.needs_input_grad[3]:
                            grad = next(grad_iter)
                            if grad is not None:
                                grad_linear_q_weight.add_(grad)
                        if ctx.needs_input_grad[4]:
                            grad = next(grad_iter)
                            if grad is not None:
                                grad_linear_k_weight.add_(grad)
                        if ctx.needs_input_grad[5]:
                            grad = next(grad_iter)
                            if grad is not None:
                                grad_k_norm_weight.add_(grad)
                        if ctx.needs_input_grad[6]:
                            grad = next(grad_iter)
                            if grad is not None:
                                grad_k_norm_bias.add_(grad)
                        if ctx.needs_input_grad[7]:
                            grad = next(grad_iter)
                            if grad is not None:
                                grad_linear_weights_weight.add_(grad)

        profile.log("backward")
        return (
            None,
            None,
            None,
            grad_linear_q_weight,
            grad_linear_k_weight,
            grad_k_norm_weight,
            grad_k_norm_bias,
            grad_linear_weights_weight,
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


class DSAMinMemoryGQAFn(torch.autograd.Function):
    """Recompute DSA-GQA routing, sparse attention, and sparse KL one query tile at a time."""

    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        hidden_states: torch.Tensor,
        linear_q_weight: torch.Tensor,
        linear_k_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        k_norm_bias: torch.Tensor,
        linear_weights_weight: torch.Tensor,
        has_k_norm_bias: bool,
        k_norm_eps: float,
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        index_rotary_dim: int,
        rotary_pos_emb,
        use_indexer_rope: bool,
        use_hadamard: bool,
        softmax_scale: float,
        loss_coeff: float,
        query_chunk_size: int,
        key_chunk_size: int,
        pg_collection: ProcessGroupCollection,
        rotary_interleaved: Optional[bool] = None,
        profile_enabled: bool = False,
        profile_rank: int = 0,
        profile_label: str = "",
        cache_routing: bool = False,
        cache_indexer_k: bool = False,
        cache_selected_scores: bool = False,
        use_triton: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        profile = _DSATimingProfiler(profile_enabled, profile_rank, profile_label, query.device)
        key_chunk_size = _routing_key_chunk_size(key_chunk_size, key.size(0), use_triton)
        routing_topk_cache = [] if cache_routing else None
        selected_scores_cache = [] if cache_selected_scores else None
        full_k_index = None
        with _triton_dispatch_enabled(use_triton):
            with profile.record("forward_total", query.device):
                with torch.no_grad():
                    if cache_indexer_k:
                        with _profile_record(profile, "indexer_k_cache_fwd_project", query.device):
                            full_k_index = _project_k_index_block(
                                hidden_states,
                                0,
                                hidden_states.size(0),
                                linear_k_weight,
                                k_norm_weight,
                                k_norm_bias,
                                has_k_norm_bias,
                                k_norm_eps,
                                index_head_dim,
                                index_rotary_dim,
                                rotary_pos_emb,
                                (
                                    _default_rotary_interleaved(rotary_pos_emb)
                                    if rotary_interleaved is None
                                    else rotary_interleaved
                                ),
                                use_indexer_rope,
                                use_hadamard,
                            )
                    output, indexer_loss = _forward_min_memory_impl(
                        query,
                        key,
                        value,
                        hidden_states,
                        linear_q_weight,
                        linear_k_weight,
                        k_norm_weight,
                        k_norm_bias,
                        has_k_norm_bias,
                        linear_weights_weight,
                        k_norm_eps,
                        index_n_heads,
                        index_head_dim,
                        index_topk,
                        index_rotary_dim,
                        rotary_pos_emb,
                        use_indexer_rope,
                        use_hadamard,
                        softmax_scale,
                        loss_coeff,
                        query_chunk_size,
                        key_chunk_size,
                        pg_collection,
                        rotary_interleaved=rotary_interleaved,
                        profile=profile,
                        routing_topk_cache=routing_topk_cache,
                        selected_scores_cache=selected_scores_cache,
                        full_k_index=full_k_index,
                    )
        profile.log("forward")

        ctx.save_for_backward(
            query,
            key,
            value,
            hidden_states,
            linear_q_weight,
            linear_k_weight,
            k_norm_weight,
            k_norm_bias,
            linear_weights_weight,
        )
        ctx.has_k_norm_bias = has_k_norm_bias
        ctx.k_norm_eps = k_norm_eps
        ctx.index_n_heads = index_n_heads
        ctx.index_head_dim = index_head_dim
        ctx.index_topk = index_topk
        ctx.index_rotary_dim = index_rotary_dim
        ctx.rotary_pos_emb = rotary_pos_emb
        ctx.rotary_interleaved = (
            _default_rotary_interleaved(rotary_pos_emb)
            if rotary_interleaved is None
            else rotary_interleaved
        )
        ctx.use_indexer_rope = use_indexer_rope
        ctx.use_hadamard = use_hadamard
        ctx.softmax_scale = softmax_scale
        ctx.loss_coeff = loss_coeff
        ctx.query_chunk_size = query_chunk_size
        ctx.key_chunk_size = key_chunk_size
        ctx.pg_collection = pg_collection
        ctx.profile_enabled = profile_enabled
        ctx.profile_rank = profile_rank
        ctx.profile_label = profile_label
        ctx.cache_routing = cache_routing
        ctx.routing_topk_cache = tuple(routing_topk_cache) if routing_topk_cache is not None else None
        ctx.cache_indexer_k = cache_indexer_k
        ctx.full_k_index_cache = full_k_index
        ctx.cache_selected_scores = cache_selected_scores
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
            k_norm_weight,
            k_norm_bias,
            linear_weights_weight,
        ) = ctx.saved_tensors
        sq, batch_size, num_query_heads, _ = query.shape
        value_head_dim = value.size(-1)

        grad_output = grad_output.reshape(sq, batch_size, num_query_heads, value_head_dim)
        grad_query = _grad_accumulator(query) if ctx.needs_input_grad[0] else None
        grad_key = _grad_accumulator(key) if ctx.needs_input_grad[1] else None
        grad_value = _grad_accumulator(value) if ctx.needs_input_grad[2] else None
        grad_linear_q_weight = (
            _grad_accumulator(linear_q_weight) if ctx.needs_input_grad[4] else None
        )
        grad_linear_k_weight = (
            _grad_accumulator(linear_k_weight) if ctx.needs_input_grad[5] else None
        )
        grad_k_norm_weight = _grad_accumulator(k_norm_weight) if ctx.needs_input_grad[6] else None
        grad_k_norm_bias = _grad_accumulator(k_norm_bias) if ctx.needs_input_grad[7] else None
        grad_linear_weights_weight = (
            _grad_accumulator(linear_weights_weight) if ctx.needs_input_grad[8] else None
        )

        total_positions = batch_size * sq
        compute_loss_grads = (
            grad_indexer_loss is not None
            and ctx.loss_coeff > 0
            and (
                ctx.needs_input_grad[4]
                or ctx.needs_input_grad[5]
                or ctx.needs_input_grad[6]
                or ctx.needs_input_grad[7]
                or ctx.needs_input_grad[8]
            )
        )
        use_triton_attention_backward = (
            ctx.needs_input_grad[0] and ctx.needs_input_grad[1] and ctx.needs_input_grad[2]
        )
        grad_key_accum = None
        grad_value_accum = None
        profile = _DSATimingProfiler(
            ctx.profile_enabled, ctx.profile_rank, ctx.profile_label, query.device
        )

        with _triton_dispatch_enabled(ctx.use_triton), profile.record("backward_total", query.device):
            cached_topk = ctx.routing_topk_cache
            cached_selected_scores = ctx.selected_scores_cache
            full_k_index = ctx.full_k_index_cache
            for chunk_idx, q_start in enumerate(range(0, sq, ctx.query_chunk_size)):
                q_end = min(q_start + ctx.query_chunk_size, sq)

                if cached_topk is not None:
                    with _profile_record(profile, "routing_topk_bwd_cached", query.device):
                        topk_indices = cached_topk[chunk_idx]
                else:
                    with _profile_record(profile, "routing_topk_bwd", query.device):
                        with torch.no_grad():
                            _, topk_indices, _, _ = _topk_index_tile(
                                hidden_states,
                                q_start,
                                q_end,
                                linear_q_weight,
                                linear_k_weight,
                                k_norm_weight,
                                k_norm_bias,
                                ctx.has_k_norm_bias,
                                linear_weights_weight,
                                ctx.k_norm_eps,
                                ctx.index_n_heads,
                                ctx.index_head_dim,
                                ctx.index_topk,
                                ctx.index_rotary_dim,
                                ctx.rotary_pos_emb,
                                ctx.rotary_interleaved,
                                ctx.use_indexer_rope,
                                ctx.use_hadamard,
                                ctx.key_chunk_size,
                                profile=profile,
                                profile_suffix="bwd",
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
                        with _profile_record(
                            profile, "sparse_attention_bwd_triton", query.device
                        ):
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
                                    ctx.softmax_scale,
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

                if not triton_attention_done and attention_inputs:
                    with _profile_record(profile, "sparse_attention_bwd_fallback", query.device):
                        with torch.enable_grad():
                            output_tile = _sparse_attention_tile(
                                query_tile,
                                key_leaf,
                                value_leaf,
                                topk_indices,
                                ctx.softmax_scale,
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

                if compute_loss_grads:
                    lq_weight = None
                    lk_weight = None
                    kn_weight = None
                    kn_bias = None
                    lw_weight = None
                    loss_inputs = None
                    use_native_wgrad = True
                    topk_score_chunk_size = _native_wgrad_topk_score_chunk_size(
                        topk_indices.size(-1)
                    )
                    selected_hidden_for_native = None
                    k_linear_for_native = None
                    selected_k_index_for_native = None
                    selected_scores = (
                        cached_selected_scores[chunk_idx]
                        if cached_selected_scores is not None
                        else None
                    )

                    with _profile_record(profile, "indexer_loss_bwd_prepare", query.device):
                        with torch.no_grad():
                            q_index, weights = _project_q_index_tile(
                                hidden_states.detach(),
                                q_start,
                                q_end,
                                linear_q_weight,
                                linear_weights_weight,
                                ctx.index_n_heads,
                                ctx.index_head_dim,
                                ctx.index_rotary_dim,
                                ctx.rotary_pos_emb,
                                ctx.rotary_interleaved,
                                ctx.use_indexer_rope,
                                ctx.use_hadamard,
                            )
                            if topk_indices.size(-1) <= 512:
                                with _profile_record(
                                    profile,
                                    "indexer_loss_bwd_prepare_project_selected_k",
                                    query.device,
                                ):
                                    if full_k_index is not None:
                                        selected_k_index_for_native = _gather_selected_indexer_k(
                                            full_k_index, topk_indices
                                        )
                                        selected_hidden_for_native, k_linear_for_native = (
                                            _project_selected_k_linear_for_wgrad(
                                                hidden_states.detach(),
                                                topk_indices,
                                                linear_k_weight,
                                            )
                                        )
                                    else:
                                        (
                                            selected_hidden_for_native,
                                            k_linear_for_native,
                                            selected_k_index_for_native,
                                        ) = _project_selected_k_index_for_wgrad(
                                            hidden_states.detach(),
                                            topk_indices,
                                            linear_k_weight,
                                            k_norm_weight,
                                            k_norm_bias,
                                            ctx.has_k_norm_bias,
                                            ctx.k_norm_eps,
                                            ctx.index_head_dim,
                                            ctx.index_rotary_dim,
                                            ctx.rotary_pos_emb,
                                            ctx.rotary_interleaved,
                                            ctx.use_indexer_rope,
                                            ctx.use_hadamard,
                                        )
                                if (
                                    selected_scores is None
                                    and selected_k_index_for_native is not None
                                ):
                                    selected_scores = _selected_index_scores_from_projected(
                                        q_index,
                                        weights,
                                        selected_k_index_for_native,
                                        topk_indices,
                                        q_start,
                                        profile=profile,
                                        profile_prefix="indexer_loss_bwd_prepare",
                                    )
                            else:
                                if selected_scores is None:
                                    selected_scores = _selected_index_scores_tile_chunked(
                                        hidden_states.detach(),
                                        q_start,
                                        q_end,
                                        topk_indices,
                                        q_index,
                                        weights,
                                        linear_k_weight,
                                        k_norm_weight,
                                        k_norm_bias,
                                        ctx.has_k_norm_bias,
                                        ctx.k_norm_eps,
                                        ctx.index_head_dim,
                                        ctx.index_rotary_dim,
                                        ctx.rotary_pos_emb,
                                        ctx.rotary_interleaved,
                                        ctx.use_indexer_rope,
                                        ctx.use_hadamard,
                                        _default_topk_score_chunk_size(topk_indices.size(-1)),
                                        profile=profile,
                                        profile_prefix="indexer_loss_bwd_prepare",
                                    )
                            teacher = _teacher_scores_tile(
                                query[q_start:q_end],
                                key,
                                topk_indices,
                                ctx.softmax_scale,
                                q_start,
                                ctx.pg_collection,
                                profile=profile,
                                profile_suffix="bwd",
                            )
                            scale = grad_indexer_loss * (ctx.loss_coeff / total_positions)
                            grad_selected_scores = None
                            with _profile_record(
                                profile, "indexer_loss_bwd_score_kl_grad", query.device
                            ):
                                grad_selected_scores = triton_indexer_loss_grad(
                                    selected_scores, teacher, scale
                                )
                                if grad_selected_scores is None:
                                    student = torch.nn.functional.softmax(
                                        selected_scores, dim=-1, dtype=torch.float32
                                    )
                                    teacher_over_student = teacher * student / (student + 1e-10)
                                    grad_selected_scores = (
                                        student * teacher_over_student.sum(dim=-1, keepdim=True)
                                    ) - teacher_over_student
                                    grad_selected_scores = grad_selected_scores * scale

                    q_index_for_grads = None
                    weights_for_grads = None
                    num_topk_chunks = (
                        topk_indices.size(-1) + topk_score_chunk_size - 1
                    ) // topk_score_chunk_size
                    for chunk_idx, topk_start in enumerate(
                        range(0, topk_indices.size(-1), topk_score_chunk_size)
                    ):
                        topk_end = min(topk_start + topk_score_chunk_size, topk_indices.size(-1))
                        chunk_topk_indices = topk_indices[..., topk_start:topk_end]
                        chunk_grad_scores = (
                            grad_selected_scores[..., topk_start:topk_end]
                            if grad_selected_scores is not None
                            else None
                        )
                        native_done = False
                        if use_native_wgrad:
                            native_done = _native_indexer_loss_wgrad_chunk(
                                hidden_states.detach(),
                                q_start,
                                q_end,
                                chunk_topk_indices,
                                q_index,
                                weights,
                                chunk_grad_scores,
                                linear_q_weight,
                                linear_k_weight,
                                k_norm_weight,
                                k_norm_bias,
                                ctx.has_k_norm_bias,
                                linear_weights_weight,
                                ctx.k_norm_eps,
                                ctx.index_head_dim,
                                ctx.index_rotary_dim,
                                ctx.rotary_pos_emb,
                                ctx.rotary_interleaved,
                                ctx.use_indexer_rope,
                                ctx.use_hadamard,
                                grad_linear_q_weight,
                                grad_linear_k_weight,
                                grad_k_norm_weight,
                                grad_k_norm_bias,
                                grad_linear_weights_weight,
                                profile,
                                (
                                    selected_hidden_for_native[..., topk_start:topk_end, :]
                                    if selected_hidden_for_native is not None
                                    else None
                                ),
                                (
                                    k_linear_for_native[..., topk_start:topk_end, :]
                                    if k_linear_for_native is not None
                                    else None
                                ),
                                (
                                    selected_k_index_for_native[..., topk_start:topk_end, :]
                                    if selected_k_index_for_native is not None
                                    else None
                                ),
                                (
                                    selected_scores[..., topk_start:topk_end]
                                    if grad_selected_scores is not None
                                    else selected_scores
                                ),
                                (
                                    teacher[..., topk_start:topk_end]
                                    if grad_selected_scores is not None
                                    else teacher
                                ),
                                scale,
                            )
                        if native_done:
                            continue

                        if grad_selected_scores is None:
                            with _profile_record(
                                profile, "indexer_loss_bwd_score_kl_grad", query.device
                            ):
                                grad_selected_scores = triton_indexer_loss_grad(
                                    selected_scores, teacher, scale
                                )
                                if grad_selected_scores is None:
                                    student = torch.nn.functional.softmax(
                                        selected_scores, dim=-1, dtype=torch.float32
                                    )
                                    teacher_over_student = teacher * student / (student + 1e-10)
                                    grad_selected_scores = student * teacher_over_student.sum(
                                        dim=-1, keepdim=True
                                    ) - teacher_over_student
                                    grad_selected_scores = grad_selected_scores * scale
                            chunk_grad_scores = grad_selected_scores[..., topk_start:topk_end]

                        if loss_inputs is None:
                            lq_weight = linear_q_weight.detach().requires_grad_(
                                ctx.needs_input_grad[4]
                            )
                            lk_weight = linear_k_weight.detach().requires_grad_(
                                ctx.needs_input_grad[5]
                            )
                            kn_weight = k_norm_weight.detach().requires_grad_(
                                ctx.needs_input_grad[6]
                            )
                            kn_bias = k_norm_bias.detach().requires_grad_(ctx.needs_input_grad[7])
                            lw_weight = linear_weights_weight.detach().requires_grad_(
                                ctx.needs_input_grad[8]
                            )
                            loss_inputs = []
                            if ctx.needs_input_grad[4]:
                                loss_inputs.append(lq_weight)
                            if ctx.needs_input_grad[5]:
                                loss_inputs.append(lk_weight)
                            if ctx.needs_input_grad[6]:
                                loss_inputs.append(kn_weight)
                            if ctx.needs_input_grad[7]:
                                loss_inputs.append(kn_bias)
                            if ctx.needs_input_grad[8]:
                                loss_inputs.append(lw_weight)

                        if q_index_for_grads is None or weights_for_grads is None:
                            with _profile_record(
                                profile, "indexer_loss_bwd_param_grad_q_project", query.device
                            ):
                                with torch.enable_grad():
                                    q_index_for_grads, weights_for_grads = _project_q_index_tile(
                                        hidden_states.detach(),
                                        q_start,
                                        q_end,
                                        lq_weight,
                                        lw_weight,
                                        ctx.index_n_heads,
                                        ctx.index_head_dim,
                                        ctx.index_rotary_dim,
                                        ctx.rotary_pos_emb,
                                        ctx.rotary_interleaved,
                                        ctx.use_indexer_rope,
                                        ctx.use_hadamard,
                                    )
                        with torch.enable_grad():
                            selected_scores = _selected_index_scores_tile(
                                hidden_states.detach(),
                                q_start,
                                q_end,
                                chunk_topk_indices,
                                q_index_for_grads,
                                weights_for_grads,
                                lk_weight,
                                kn_weight,
                                kn_bias,
                                ctx.has_k_norm_bias,
                                ctx.k_norm_eps,
                                ctx.index_head_dim,
                                ctx.index_rotary_dim,
                                ctx.rotary_pos_emb,
                                ctx.rotary_interleaved,
                                ctx.use_indexer_rope,
                                ctx.use_hadamard,
                                profile=profile,
                                profile_prefix="indexer_loss_bwd_param_grad",
                            )

                        with _profile_record(
                            profile, "indexer_loss_bwd_param_grad_autograd", query.device
                        ):
                            loss_grads = torch.autograd.grad(
                                selected_scores,
                                loss_inputs,
                                grad_outputs=chunk_grad_scores,
                                retain_graph=chunk_idx + 1 < num_topk_chunks,
                                allow_unused=True,
                            )
                            grad_iter = iter(loss_grads)
                            if ctx.needs_input_grad[4]:
                                grad = next(grad_iter)
                                if grad is not None:
                                    grad_linear_q_weight.add_(grad)
                            if ctx.needs_input_grad[5]:
                                grad = next(grad_iter)
                                if grad is not None:
                                    grad_linear_k_weight.add_(grad)
                            if ctx.needs_input_grad[6]:
                                grad = next(grad_iter)
                                if grad is not None:
                                    grad_k_norm_weight.add_(grad)
                            if ctx.needs_input_grad[7]:
                                grad = next(grad_iter)
                                if grad is not None:
                                    grad_k_norm_bias.add_(grad)
                            if ctx.needs_input_grad[8]:
                                grad = next(grad_iter)
                                if grad is not None:
                                    grad_linear_weights_weight.add_(grad)

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
            grad_k_norm_weight,
            grad_k_norm_bias,
            grad_linear_weights_weight,
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
            None,
            None,
        )


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
) -> torch.Tensor:
    """Run min-memory DSA-GQA for no-grad validation/eval forward passes."""
    k_norm_bias, has_k_norm_bias = _module_bias(indexer.k_norm, query)
    query_chunk_size = _chunk_size(
        query_chunk_size, _default_query_chunk_size(query.size(0)), query.size(0)
    )
    key_chunk_size = _routing_key_chunk_size(key_chunk_size, key.size(0), use_triton)
    rotary_interleaved = getattr(indexer.config, "rotary_interleaved", False)
    k_norm_eps = getattr(indexer.k_norm, "eps", indexer.config.layernorm_epsilon)
    linear_q_weight = _module_weight(indexer.linear_q)
    linear_k_weight = _module_weight(indexer.linear_k)
    k_norm_weight = _module_weight(indexer.k_norm)
    linear_weights_weight = _module_weight(indexer.linear_weights_proj)
    profile = _DSATimingProfiler(profile_enabled, profile_rank, profile_label, query.device)
    full_k_index = None

    with torch.no_grad(), _triton_dispatch_enabled(use_triton):
        with profile.record("forward_total", query.device):
            if cache_indexer_k:
                with _profile_record(profile, "indexer_k_cache_fwd_project", query.device):
                    full_k_index = _project_k_index_block(
                        hidden_states,
                        0,
                        hidden_states.size(0),
                        linear_k_weight,
                        k_norm_weight,
                        k_norm_bias,
                        has_k_norm_bias,
                        k_norm_eps,
                        indexer.index_head_dim,
                        indexer.index_rotary_dim,
                        indexer.rotary_pos_emb,
                        rotary_interleaved,
                        use_indexer_rope,
                        indexer.config.dsa_indexer_use_hadamard,
                    )
            output, _ = _forward_min_memory_impl(
                query,
                key,
                value,
                hidden_states,
                linear_q_weight,
                linear_k_weight,
                k_norm_weight,
                k_norm_bias,
                has_k_norm_bias,
                linear_weights_weight,
                k_norm_eps,
                indexer.index_n_heads,
                indexer.index_head_dim,
                indexer.index_topk,
                indexer.index_rotary_dim,
                indexer.rotary_pos_emb,
                use_indexer_rope,
                indexer.config.dsa_indexer_use_hadamard,
                softmax_scale,
                0.0,
                query_chunk_size,
                key_chunk_size,
                indexer.pg_collection,
                rotary_interleaved=rotary_interleaved,
                profile=profile,
                full_k_index=full_k_index,
            )
    profile.log("forward")
    return output


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
) -> torch.Tensor:
    """Run tiled dense DSA indexer KL for dense-attention warmup."""
    k_norm_bias, has_k_norm_bias = _module_bias(indexer.k_norm, query)
    return DSADenseIndexerLossFn.apply(
        query,
        key,
        hidden_states,
        _module_weight(indexer.linear_q),
        _module_weight(indexer.linear_k),
        _module_weight(indexer.k_norm),
        k_norm_bias,
        _module_weight(indexer.linear_weights_proj),
        has_k_norm_bias,
        getattr(indexer.k_norm, "eps", indexer.config.layernorm_epsilon),
        indexer.index_n_heads,
        indexer.index_head_dim,
        indexer.index_rotary_dim,
        indexer.rotary_pos_emb,
        use_indexer_rope,
        indexer.config.dsa_indexer_use_hadamard,
        softmax_scale,
        loss_coeff,
        _chunk_size(query_chunk_size, _default_query_chunk_size(query.size(0)), query.size(0)),
        _chunk_size(key_chunk_size, _default_key_chunk_size(key.size(0)), key.size(0)),
        indexer.pg_collection,
        getattr(indexer.config, "rotary_interleaved", False),
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the minimum-activation DSA-GQA training backend."""
    k_norm_bias, has_k_norm_bias = _module_bias(indexer.k_norm, query)
    return DSAMinMemoryGQAFn.apply(
        query,
        key,
        value,
        hidden_states,
        _module_weight(indexer.linear_q),
        _module_weight(indexer.linear_k),
        _module_weight(indexer.k_norm),
        k_norm_bias,
        _module_weight(indexer.linear_weights_proj),
        has_k_norm_bias,
        getattr(indexer.k_norm, "eps", indexer.config.layernorm_epsilon),
        indexer.index_n_heads,
        indexer.index_head_dim,
        indexer.index_topk,
        indexer.index_rotary_dim,
        indexer.rotary_pos_emb,
        use_indexer_rope,
        indexer.config.dsa_indexer_use_hadamard,
        softmax_scale,
        loss_coeff,
        _chunk_size(query_chunk_size, _default_query_chunk_size(query.size(0)), query.size(0)),
        _routing_key_chunk_size(key_chunk_size, key.size(0), use_triton),
        indexer.pg_collection,
        getattr(indexer.config, "rotary_interleaved", False),
        profile_enabled,
        profile_rank,
        profile_label,
        cache_routing,
        cache_indexer_k,
        cache_selected_scores,
        use_triton,
    )
