# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Uncompressed sparse attention ratio-one orchestration over unmodified Core dev 7b150acf.

Core CSA gates compression/indexing on ratio > 1; Uncompressed sparse attention ratio one instead
keeps every projected KV token. This module supplies that model-specific
orchestration, candidate state and selection, not attention kernels or CP
communication. Ratios zero/two continue to use native CSA.forward.
"""
import torch
from torch import nn
from torch.nn import functional as F

from megatron.core.tensor_parallel.mappings import async_gather_from_sequence_parallel_region
from megatron.core.transformer.experimental_attention_variant.csa import (
    _OutputInverseRope,
    batch_of_row,
    get_window_topk_idxs_thd,
    unfused_compressed_sparse_attn,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils
from megatron.core.transformer.experimental_attention_variant.csa_utils.fused_sparse_attention import (
    csa_sparse_attn,
    get_flash_mla_topk_alignment,
)
from megatron.core.transformer.experimental_attention_variant.dsa import fused_qk_topk_naive


def candidate_pool(scores, visible, *, blocks, block_size):
    """Uncompressed sparse attention block maxima and mandatory newest, possibly partial, block."""
    if blocks < 1 or block_size < 1:
        raise ValueError(
            "Uncompressed sparse attention candidate block count and width must be positive"
        )
    width = scores.shape[-1]
    if width == 0:
        return torch.zeros_like(scores, dtype=torch.bool)
    padded = F.pad(scores, (0, (-width) % block_size), value=-torch.inf)
    block_scores = padded.reshape(
        *scores.shape[:-1], (width + block_size - 1) // block_size, block_size
    ).amax(-1)
    newest = (visible - 1).div(block_size, rounding_mode="floor")
    block_ids = torch.arange(block_scores.shape[-1], device=scores.device)
    block_scores = torch.where(block_ids == newest, torch.inf, block_scores)
    values, indices = block_scores.topk(min(blocks, block_scores.shape[-1]), dim=-1)
    chosen = torch.zeros_like(block_scores, dtype=torch.bool).scatter(
        -1, indices, values > -torch.inf
    )
    return chosen.repeat_interleave(block_size, dim=-1)[..., :width]


def flash_positions(positions, alignment):
    """Pad index columns with -1 (no key) to FlashMLA's top-k alignment."""
    return F.pad(positions, (0, (-positions.shape[-1]) % alignment), value=-1).contiguous()


def select_indices(
    q,
    k,
    weights,
    prefix,
    global_start,
    *,
    topk,
    candidate_blocks,
    block_size,
    publish=False,
    candidates=None,
):
    """Per-document Core scorer plus Uncompressed sparse attention two-level candidate policy."""
    if topk < 0:
        raise ValueError("Uncompressed sparse attention index top-k must be nonnegative")
    lengths = prefix[1:] - prefix[:-1]
    capacity = min(topk, int(lengths.max()))
    selected = torch.full((q.shape[0], capacity), -1, dtype=torch.int32, device=q.device)
    local_positions = cp_utils.thd_cp_position_ids(prefix, global_start, q.shape[0])
    row_ids = torch.arange(q.shape[0], device=q.device) + global_start
    documents = torch.bucketize(row_ids, prefix[1:], right=True)
    pools = []
    for document, (start, end) in enumerate(zip(prefix[:-1].tolist(), prefix[1:].tolist())):
        rows = ((documents == document) & (row_ids < end)).nonzero().flatten()
        count = min(topk, end - start)
        visible = (local_positions[rows] + 1).unsqueeze(-1)
        cols = torch.arange(end - start, device=q.device)
        mask = torch.where(cols < visible, 0.0, -torch.inf)
        scores, _ = fused_qk_topk_naive(
            q[rows].unsqueeze(1),
            k[start:end].unsqueeze(1),
            weights[rows].unsqueeze(1),
            0,
            mask=mask.unsqueeze(0),
        )
        scores = scores.squeeze(0)
        pool = (
            candidate_pool(scores, visible, blocks=candidate_blocks, block_size=block_size)
            if publish
            else None
        )
        if candidates is not None:
            if candidates[document].shape != scores.shape:
                raise ValueError(
                    "Uncompressed sparse attention candidate reader changed packed query geometry"
                )
            scores = scores.masked_fill(~candidates[document], -torch.inf)
        # Position order is the release contract. A candidate-limited top-k
        chosen = scores.topk(count, dim=-1, sorted=False).indices.sort(-1).values
        valid = (chosen < visible) & torch.isfinite(scores.gather(-1, chosen))
        selected[rows, :count] = torch.where(valid, chosen + start, -1).int()
        pools.append(pool)
    return selected, tuple(pools) if publish else candidates


class RatioOneAttention(nn.Module):
    """Native sparse attention with Uncompressed sparse attention ratio-one source/candidate semantics."""

    def __init__(
        self,
        config,
        *,
        group,
        rotary_pos_emb,
        layer,
        index_owner,
        candidate_owner,
        state,
        candidate_blocks,
        block_size,
        use_flash=False,
    ):
        super().__init__()
        self.config, self.group, self.rotary_pos_emb = config, group, rotary_pos_emb
        self.pg_collection = type("Groups", (), {"cp": group})()
        self.layer, self.index_owner, self.candidate_owner = layer, index_owner, candidate_owner
        self.state, self.use_flash = state, use_flash
        self.candidate_blocks, self.block_size = candidate_blocks, block_size
        self.qk_pos_emb_head_dim = config.qk_pos_emb_head_dim
        self.v_head_dim = config.v_head_dim
        self.attn_sink = nn.Parameter(torch.zeros(config.num_attention_heads, dtype=torch.float32))

    def forward(self, query, window_kv, main_kv, index_q, index_k, weights, params):
        if params.qkv_format != "thd" or params.cp_partition_mode != "contiguous":
            raise ValueError(
                "Uncompressed sparse attention ratio one requires contiguous packed THD"
            )
        prefix = params.cu_seqlens_q_padded
        if prefix is None:
            prefix = params.cu_seqlens_q
        rows = query.shape[0]
        global_start = self.group.rank() * rows
        if self.group.size() == 1:
            window, main, keys = window_kv, main_kv, index_k.detach()
        else:
            window = async_gather_from_sequence_parallel_region(window_kv, group=self.group).wait()
            main = async_gather_from_sequence_parallel_region(main_kv, group=self.group).wait()
            keys = async_gather_from_sequence_parallel_region(
                index_k.detach(), group=self.group
            ).wait()
        geometry = (
            "ratio_one_selection",
            prefix.data_ptr(),
            global_start,
            rows,
            self.state.grad_mode,
        )
        pool_key = ("candidate_pool", self.candidate_owner, geometry)

        def produce():
            publish = self.layer == self.candidate_owner
            prior = None if publish else self.state.values.get(pool_key)
            if self.layer > self.candidate_owner and prior is None:
                raise RuntimeError("Uncompressed sparse attention candidate publisher has not run")
            indices, pool = select_indices(
                index_q.detach(),
                keys,
                weights.detach(),
                prefix,
                global_start,
                topk=self.config.dsa_indexer_topk,
                candidate_blocks=self.candidate_blocks,
                block_size=self.block_size,
                publish=publish,
                candidates=prior,
            )
            if publish:
                self.state.values[pool_key] = pool
            return indices

        indices = self.state.resolve(self.layer, self.index_owner, geometry, produce)
        total = window.shape[0]
        local_slice = slice(global_start, global_start + rows)
        window_indices = get_window_topk_idxs_thd(self.config.csa_window_size, prefix, total)[
            local_slice
        ]
        starts = prefix[batch_of_row(prefix, total_q=total)][local_slice].unsqueeze(-1)
        window_indices = torch.where(window_indices >= 0, window_indices + starts, -1)
        selected = torch.where(indices >= 0, indices + total, -1)
        positions = torch.cat((window_indices, selected), dim=-1).int()
        kv = torch.cat((window, main), dim=0)
        scale = self.config.v_head_dim**-0.5
        if self.use_flash:
            positions = flash_positions(positions, get_flash_mla_topk_alignment())
            result = csa_sparse_attn(
                query.contiguous(), kv.contiguous(), self.attn_sink, positions, scale, is_thd=True
            )
        else:
            result = unfused_compressed_sparse_attn(query, kv, self.attn_sink, positions, scale)
        inverse = _OutputInverseRope(
            self,
            query.dtype,
            int(params.max_seqlen_q),
            True,
            cu_seqlens_q=prefix,
            thd_cp_global_start=global_start if self.group.size() > 1 else None,
            thd_cp_rows=rows if self.group.size() > 1 else None,
        )
        return inverse.apply(result.reshape(rows, self.config.num_attention_heads, -1)).reshape(
            rows, 1, -1
        )
