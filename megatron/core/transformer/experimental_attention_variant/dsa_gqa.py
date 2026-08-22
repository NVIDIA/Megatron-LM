# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as torch_checkpoint
from megatron.core.extensions.transformer_engine import TELinear, TENorm
from megatron.core.models.common.embeddings import (
    RotaryEmbedding,
    YarnRotaryEmbedding,
    apply_rotary_pos_emb,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
    DSAMainAttentionAuxLossAutoScaler,
    DSAMainAttentionAuxLossLoggingHelper,
    fused_qk_topk_chunked,
    fused_qk_topk_naive,
    rotate_activation,
)
from megatron.core.transformer.experimental_attention_variant.dsa_diagnostics import (
    assert_tp_support_consistent,
    compute_dsa_attention_diagnostics,
)
from megatron.core.transformer.experimental_attention_variant.dsa_min_memory import (
    dsa_dense_indexer_loss,
    dsa_main_attention_aux_loss,
    dsa_min_memory_gqa,
    dsa_min_memory_gqa_forward_only,
)
from megatron.core.utils import is_using_quantization_scales


def _repeat_grouped_key_value(key: torch.Tensor, value: torch.Tensor, num_query_heads: int):
    """Expand grouped keys/values to per-query-head layout for reference attention math."""
    num_query_groups = key.size(2)
    assert num_query_heads % num_query_groups == 0, (
        f"num_query_heads ({num_query_heads}) must be divisible by num_query_groups "
        f"({num_query_groups})."
    )
    repeat_factor = num_query_heads // num_query_groups
    if repeat_factor == 1:
        return key, value
    key = key.repeat_interleave(repeat_factor, dim=2)
    value = value.repeat_interleave(repeat_factor, dim=2)
    return key, value


def _gather_block_cache_sequence(
    cache: torch.Tensor, block_table_row: torch.Tensor, sequence_length: int, block_size_tokens: int
) -> torch.Tensor:
    """Materialize a per-request sequence from a paged block cache."""
    if sequence_length == 0:
        return cache.new_empty((0,) + cache.shape[2:])
    positions = torch.arange(sequence_length, device=cache.device, dtype=torch.long)
    block_ids = block_table_row[(positions // block_size_tokens).to(block_table_row.device)].long()
    local_positions = positions % block_size_tokens
    return cache[block_ids, local_positions]


def _build_shifted_causal_mask(
    query_length: int, key_length: int, query_start_position: int, device: torch.device
) -> torch.Tensor:
    """Build a causal mask for a query chunk that starts at a non-zero KV offset."""
    if query_length == 0 or key_length == 0:
        return torch.empty((query_length, key_length), dtype=torch.float32, device=device)
    query_positions = torch.arange(
        query_start_position, query_start_position + query_length, device=device, dtype=torch.long
    )
    key_positions = torch.arange(key_length, device=device, dtype=torch.long)
    invalid = key_positions.view(1, key_length) > query_positions.view(query_length, 1)
    return torch.zeros((query_length, key_length), dtype=torch.float32, device=device).masked_fill(
        invalid, float("-inf")
    )


@dataclass(frozen=True)
class _DSAIndexerInputNormSpec:
    normalization: str
    weight: torch.Tensor
    bias: Optional[torch.Tensor]
    eps: float
    zero_centered_gamma: bool


def _indexer_input_norm_spec(
    linear_qkv, config: TransformerConfig
) -> Optional[_DSAIndexerInputNormSpec]:
    """Describe a norm fused into main QKV without registering another parameter copy."""
    norm_weight = getattr(linear_qkv, "layer_norm_weight", None)
    if norm_weight is None:
        return None
    norm_bias = getattr(linear_qkv, "layer_norm_bias", None)
    return _DSAIndexerInputNormSpec(
        normalization=config.normalization,
        weight=norm_weight.detach(),
        bias=None if norm_bias is None else norm_bias.detach(),
        eps=getattr(linear_qkv, "eps", config.layernorm_epsilon),
        zero_centered_gamma=config.layernorm_zero_centered_gamma,
    )


def _normalized_indexer_input(
    hidden_states: torch.Tensor,
    norm_spec: Optional[_DSAIndexerInputNormSpec],
) -> torch.Tensor:
    """Return the detached activation seen by the main Q projection.

    TE layer specs commonly fuse the attention input norm into ``linear_qkv``. An indexer that
    opts into the main-Q input norm must consume that normalized activation too; otherwise its
    projection weights are applied to a different distribution. With an unfused spec,
    ``hidden_states`` has already passed through the transformer's input norm.
    """
    hidden_states = hidden_states.detach()
    if norm_spec is None:
        return hidden_states

    norm_weight = norm_spec.weight
    if norm_spec.zero_centered_gamma:
        norm_weight = norm_weight + 1.0
    eps = norm_spec.eps
    normalized_shape = (hidden_states.size(-1),)
    with torch.no_grad():
        if norm_spec.normalization == "RMSNorm":
            hidden_float = hidden_states.float()
            inv_rms = torch.rsqrt(hidden_float.square().mean(dim=-1, keepdim=True) + eps)
            return (hidden_float * inv_rms * norm_weight.float()).to(hidden_states.dtype)
        if norm_spec.normalization == "LayerNorm":
            return F.layer_norm(
                hidden_states, normalized_shape, norm_weight, norm_spec.bias, eps
            )
    raise NotImplementedError(
        "DSA cannot reproduce the fused main-Q input normalization "
        f"for normalization={norm_spec.normalization!r}."
    )


# Preserve internal names imported by existing simplified-DSA tests and downstream code.
_SimplifiedIndexerInputNormSpec = _DSAIndexerInputNormSpec
_simplified_indexer_norm_spec = _indexer_input_norm_spec
_simplified_indexer_input = _normalized_indexer_input


def _simplified_indexer_uses_main_input_norm(config: TransformerConfig) -> bool:
    """Whether simplified DSA should reproduce a norm fused into main QKV."""
    return getattr(config, "dsa_indexer_mode", "standard") == "simplified" and not getattr(
        config, "dsa_simplified_indexer_disable_main_input_norm", False
    )


def _build_selected_causal_mask(
    topk_indices: torch.Tensor, query_start_position: int = 0
) -> torch.Tensor:
    """Build a causal mask only on the selected top-k support."""
    batch_size, query_length, _ = topk_indices.shape
    query_positions = torch.arange(
        query_start_position,
        query_start_position + query_length,
        device=topk_indices.device,
        dtype=topk_indices.dtype,
    )
    invalid = topk_indices > query_positions.view(1, query_length, 1)
    return torch.zeros(
        (batch_size, query_length, topk_indices.size(-1)),
        dtype=torch.float32,
        device=topk_indices.device,
    ).masked_fill(invalid, float("-inf"))


def compute_gqa_dsa_indexer_loss(
    index_scores: Optional[torch.Tensor],
    topk_indices: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    softmax_scale: float,
    loss_coeff: float,
    sparse_loss: bool,
    pg_collection: ProcessGroupCollection,
    sparse_loss_use_topk_only: bool = False,
    query_chunk_size: Optional[int] = None,
    selected_index_scores: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute DSA indexer KL loss for grouped-query attention."""
    sq, b, np, hn = query.size()
    sk, _, ng, _ = key.size()
    if index_scores is None and selected_index_scores is None:
        raise AssertionError("Either index_scores or selected_index_scores must be provided.")
    if index_scores is not None:
        assert sq == index_scores.size(1), "Query sequence length must match index_scores."
        assert sk == index_scores.size(2), "Key sequence length must match index_scores."
    if selected_index_scores is not None:
        assert sparse_loss and sparse_loss_use_topk_only, (
            "selected_index_scores is only supported for topk-only sparse loss."
        )
        assert sq == selected_index_scores.size(1), (
            "Query sequence length must match selected_index_scores."
        )
        assert topk_indices.size(-1) == selected_index_scores.size(-1), (
            "selected_index_scores and topk_indices must have matching top-k dimension."
        )

    if np != ng:
        assert np % ng == 0, f"num_query_heads ({np}) must be divisible by num_query_groups ({ng})."
        repeat_factor = np // ng
        key = key.repeat_interleave(repeat_factor, dim=2)

    if query_chunk_size is None or query_chunk_size <= 0:
        query_chunk_size = sq
    else:
        query_chunk_size = min(query_chunk_size, sq)

    loss_ref = index_scores if index_scores is not None else selected_index_scores

    if sparse_loss and sparse_loss_use_topk_only and query_chunk_size < sq:
        query = query.permute(1, 2, 0, 3)
        key = key.permute(1, 2, 0, 3)
        total_kl = loss_ref.new_zeros((), dtype=torch.float32)
        total_positions = 0
        topk = topk_indices.size(-1)

        for q_start in range(0, sq, query_chunk_size):
            q_end = min(q_start + query_chunk_size, sq)
            chunk_len = q_end - q_start
            query_chunk = query[:, :, q_start:q_end, :]
            topk_indices_chunk = topk_indices[:, q_start:q_end, :]
            gather_index = topk_indices_chunk[:, None, :, :, None].expand(
                b, np, chunk_len, topk, hn
            )
            selected_key = torch.gather(
                key[:, :, None, :, :].expand(b, np, chunk_len, sk, hn),
                3,
                gather_index,
            )
            selected_causal_mask = _build_selected_causal_mask(
                topk_indices_chunk, query_start_position=q_start
            ).unsqueeze(1)
            teacher_scores = (
                torch.einsum("bnsh,bnskh->bnsk", query_chunk.float(), selected_key.float())
                * softmax_scale
            )
            teacher_scores = teacher_scores + selected_causal_mask
            teacher_scores = torch.nn.functional.softmax(
                teacher_scores, dim=-1, dtype=torch.float32
            )
            teacher_scores = teacher_scores.sum(dim=1)
            if pg_collection.tp.size() > 1:
                torch.distributed.all_reduce(teacher_scores.contiguous(), group=pg_collection.tp)
            teacher_scores = teacher_scores / teacher_scores.sum(dim=-1, keepdim=True)

            if selected_index_scores is not None:
                student_logits = selected_index_scores[:, q_start:q_end, :]
            else:
                student_logits = index_scores[:, q_start:q_end, :].gather(-1, topk_indices_chunk)
            student_scores = torch.nn.functional.softmax(student_logits, dim=-1, dtype=torch.float32)
            kl_per_element = teacher_scores * (
                torch.log(teacher_scores + 1e-10) - torch.log(student_scores + 1e-10)
            )
            total_kl = total_kl + kl_per_element.sum()
            total_positions += b * chunk_len

        return total_kl / total_positions * loss_coeff
    elif sparse_loss and sparse_loss_use_topk_only:
        query = query.permute(1, 2, 0, 3)
        key = key.permute(1, 2, 0, 3)
        topk = topk_indices.size(-1)
        gather_index = topk_indices[:, None, :, :, None].expand(b, np, sq, topk, hn)
        selected_key = torch.gather(
            key[:, :, None, :, :].expand(b, np, sq, sk, hn),
            3,
            gather_index,
        )
        selected_causal_mask = _build_selected_causal_mask(topk_indices).unsqueeze(1)
        attention_scores = (
            torch.einsum("bnsh,bnskh->bnsk", query.float(), selected_key.float()) * softmax_scale
        )
        attention_scores = attention_scores + selected_causal_mask
        attention_scores = torch.nn.functional.softmax(
            attention_scores, dim=-1, dtype=torch.float32
        )
        if selected_index_scores is not None:
            index_scores = torch.nn.functional.softmax(
                selected_index_scores, dim=-1, dtype=torch.float32
            )
        else:
            index_scores = torch.nn.functional.softmax(
                index_scores.gather(-1, topk_indices), dim=-1, dtype=torch.float32
            )
        attention_scores = attention_scores.sum(dim=1)
    else:
        query = query.permute(1, 2, 0, 3).reshape(b * np, sq, hn)
        key = key.permute(1, 2, 3, 0).reshape(b * np, hn, sk)
        attention_scores = torch.bmm(query.float(), key.float()) * softmax_scale
        attention_scores = attention_scores.reshape(b, np, sq, sk)

        causal_mask = torch.triu(
            torch.full((sq, sk), float('-inf'), dtype=torch.float32, device=attention_scores.device),
            diagonal=1,
        )
        index_mask = torch.full(
            (b, sq, sk), float("-inf"), dtype=torch.float32, device=attention_scores.device
        ).scatter_(-1, topk_indices, 0)

        attention_scores = attention_scores + causal_mask.view(1, 1, sq, sk)
        if sparse_loss:
            attention_scores = attention_scores + index_mask.view(b, 1, sq, sk)
            index_scores = index_scores + index_mask

        attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32)
        index_scores = torch.nn.functional.softmax(index_scores, dim=-1, dtype=torch.float32)

        attention_scores = attention_scores.sum(dim=1)

    if pg_collection.tp.size() > 1:
        torch.distributed.all_reduce(attention_scores.contiguous(), group=pg_collection.tp)
    attention_scores = attention_scores / attention_scores.sum(dim=-1, keepdim=True)

    kl_per_element = attention_scores * (
        torch.log(attention_scores + 1e-10) - torch.log(index_scores + 1e-10)
    )
    return kl_per_element.sum(dim=-1).mean() * loss_coeff


def _dense_grouped_dsa_fn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    mask: Optional[torch.Tensor] = None,
):
    """Dense-mask reference grouped-query sparse attention."""
    sq, b, np, hn = query.size()
    skv = key.size(0)
    key, value = _repeat_grouped_key_value(key, value, np)
    hnv = value.size(3)

    query = query.permute(1, 2, 0, 3).reshape(b * np, sq, hn)
    key = key.permute(1, 2, 3, 0).reshape(b * np, hn, skv)
    attention_scores = torch.bmm(query.float(), key.float()) * softmax_scale
    attention_scores = attention_scores.reshape(b, np, sq, skv)

    index_mask = torch.full((b, sq, skv), float("-inf"), device=attention_scores.device)
    index_mask.scatter_(-1, topk_indices, 0)
    if mask is None:
        mask = torch.triu(
            torch.full((sq, skv), float("-inf"), dtype=torch.float32, device=index_mask.device),
            diagonal=1,
        )
    elif mask.dtype == torch.bool:
        mask = torch.zeros(
            mask.shape,
            dtype=torch.float32,
            device=mask.device,
        ).masked_fill(mask, float("-inf"))
    else:
        mask = mask.to(dtype=torch.float32, device=index_mask.device)
    if mask.dim() == 2:
        mask = mask.view(1, sq, skv)
    else:
        assert mask.shape == (b, sq, skv), "mask shape must be [sq, skv] or [b, sq, skv]"
    index_mask = index_mask + mask
    attention_scores = attention_scores + index_mask.unsqueeze(1)
    attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32)

    value = value.permute(1, 2, 0, 3).reshape(b * np, skv, hnv)
    attention_scores = attention_scores.reshape(b * np, sq, skv)
    output = torch.bmm(attention_scores.to(value.dtype), value)
    output = output.reshape(b, np, sq, hnv).permute(2, 0, 1, 3).contiguous()
    return output.reshape(sq, b, np * hnv)


def _sparse_grouped_dsa_fn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    mask: Optional[torch.Tensor] = None,
    query_chunk_size: Optional[int] = None,
):
    """Gather-based grouped-query sparse attention."""
    sq, b, np, hn = query.size()
    skv = key.size(0)
    num_query_groups = key.size(2)
    assert np % num_query_groups == 0, (
        f"num_query_heads ({np}) must be divisible by num_query_groups ({num_query_groups})."
    )
    repeat_factor = np // num_query_groups
    topk = topk_indices.size(-1)
    if query_chunk_size is None or query_chunk_size <= 0:
        query_chunk_size = sq
    else:
        query_chunk_size = min(query_chunk_size, sq)

    query = query.permute(1, 2, 0, 3).unflatten(1, (num_query_groups, repeat_factor))
    key = key.permute(1, 2, 0, 3)
    value = value.permute(1, 2, 0, 3)
    hnv = value.size(-1)

    output = value.new_empty((b, num_query_groups, repeat_factor, sq, hnv))

    for q_start in range(0, sq, query_chunk_size):
        q_end = min(q_start + query_chunk_size, sq)
        chunk_len = q_end - q_start
        topk_indices_chunk = topk_indices[:, q_start:q_end, :]
        if mask is None:
            selected_mask = _build_selected_causal_mask(
                topk_indices_chunk, query_start_position=q_start
            )
        elif mask.dim() == 2:
            selected_mask = mask[q_start:q_end].unsqueeze(0).expand(b, chunk_len, skv).gather(2, topk_indices_chunk)
        else:
            selected_mask = mask[:, q_start:q_end, :].gather(2, topk_indices_chunk)
        if selected_mask.dtype == torch.bool:
            selected_mask = torch.zeros(
                selected_mask.shape,
                dtype=torch.float32,
                device=selected_mask.device,
            ).masked_fill(selected_mask, float("-inf"))
        selected_mask = selected_mask.unsqueeze(1)

        key_gather_index = topk_indices_chunk[:, :, :, None].expand(b, chunk_len, topk, hn)
        value_gather_index = topk_indices_chunk[:, :, :, None].expand(b, chunk_len, topk, hnv)

        for group_idx in range(num_query_groups):
            key_group = key[:, group_idx, :, :]
            value_group = value[:, group_idx, :, :]
            gathered_key = torch.gather(
                key_group[:, None, :, :].expand(b, chunk_len, skv, hn),
                2,
                key_gather_index,
            )
            gathered_value = torch.gather(
                value_group[:, None, :, :].expand(b, chunk_len, skv, hnv),
                2,
                value_gather_index,
            )
            query_group = query[:, group_idx, :, q_start:q_end, :]
            attention_scores = (
                torch.einsum("brsh,bskh->brsk", query_group.float(), gathered_key.float())
                * softmax_scale
            )
            attention_scores = attention_scores + selected_mask
            attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32)
            output[:, group_idx, :, q_start:q_end, :] = torch.einsum(
                "brsk,bskd->brsd", attention_probs.to(gathered_value.dtype), gathered_value
            )

    output = output.view(b, np, sq, hnv).permute(2, 0, 1, 3).contiguous()
    return output.reshape(sq, b, np * hnv)


def unfused_grouped_dsa_fn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    mask: Optional[torch.Tensor] = None,
    query_chunk_size: Optional[int] = None,
    use_gather: bool = False,
):
    """Reference grouped-query sparse attention with optional gather-based backend."""
    if use_gather:
        return _sparse_grouped_dsa_fn(
            query,
            key,
            value,
            topk_indices,
            softmax_scale,
            mask=mask,
            query_chunk_size=query_chunk_size,
        )
    return _dense_grouped_dsa_fn(
        query,
        key,
        value,
        topk_indices,
        softmax_scale,
        mask=mask,
    )


@dataclass
class DSGQAIndexerSubmodules:
    linear_q: Union[ModuleSpec, type] = None
    linear_k: Union[ModuleSpec, type] = None
    k_norm: Union[ModuleSpec, type] = None
    linear_weights_proj: Union[ModuleSpec, type] = None


@dataclass
class SimplifiedDSGQAIndexerSubmodules:
    linear_q: Union[ModuleSpec, type] = None
    linear_k: Union[ModuleSpec, type] = None


@dataclass
class DSGQAAttentionSubmodules:
    indexer: Union[ModuleSpec, type] = None
    dense_core_attention: Union[ModuleSpec, type] = None


class DSGQAIndexer(MegatronModule):
    """Token-level DSA indexer for grouped-query attention."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: DSGQAIndexerSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        super().__init__(config=config)
        self.hidden_size = config.hidden_size
        self.index_n_heads = config.dsa_indexer_n_heads
        self.index_head_dim = config.dsa_indexer_head_dim
        self.index_topk = config.dsa_indexer_topk
        self.softmax_scale = self.index_head_dim**-0.5
        self.index_rotary_dim = int(self.index_head_dim * config.rotary_percent)
        self.index_rotary_dim -= self.index_rotary_dim % 2

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        self.pg_collection = pg_collection

        self.rotary_pos_emb = None
        if self.index_rotary_dim > 0:
            if config.rope_type == 'rope':
                self.rotary_pos_emb = RotaryEmbedding(
                    self.index_rotary_dim,
                    rotary_percent=1.0,
                    rotary_interleaved=config.rotary_interleaved,
                    seq_len_interpolation_factor=config.rotary_seq_len_interpolation_factor,
                    rotary_base=config.rotary_base,
                    rope_scaling=config.use_rope_scaling,
                    rope_scaling_factor=config.rope_scaling_factor,
                    use_cpu_initialization=config.use_cpu_initialization,
                    cp_group=self.pg_collection.cp,
                )
            elif config.rope_type == 'yarn':
                self.rotary_pos_emb = YarnRotaryEmbedding(
                    self.index_rotary_dim,
                    rotary_interleaved=config.rotary_interleaved,
                    seq_len_interpolation_factor=config.rotary_seq_len_interpolation_factor,
                    rotary_base=config.rotary_base,
                    scaling_factor=config.rotary_scaling_factor,
                    original_max_position_embeddings=config.original_max_position_embeddings,
                    beta_fast=config.beta_fast,
                    beta_slow=config.beta_slow,
                    mscale=config.mscale,
                    mscale_all_dim=config.mscale_all_dim,
                    use_cpu_initialization=config.use_cpu_initialization,
                    cp_group=self.pg_collection.cp,
                )

        self.linear_q = build_module(
            submodules.linear_q,
            self.hidden_size,
            self.index_n_heads * self.index_head_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        self.linear_k = build_module(
            submodules.linear_k,
            self.hidden_size,
            self.index_head_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        k_norm_config = copy.copy(config)
        k_norm_config.normalization = "LayerNorm"
        self.k_norm = build_module(
            submodules.k_norm,
            config=k_norm_config,
            hidden_size=self.index_head_dim,
            eps=config.layernorm_epsilon,
        )
        self.linear_weights_proj = build_module(
            submodules.linear_weights_proj,
            self.hidden_size,
            self.index_n_heads,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        if self.pg_collection.tp.size() > 1:
            for param in self.parameters():
                setattr(param, "average_gradients_across_tp_domain", True)

    def _apply_rope(self, x: torch.Tensor, use_rope: bool, packed_seq_params=None):
        if not use_rope or self.rotary_pos_emb is None or self.index_rotary_dim == 0:
            return x

        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            None, None, x, self.config, packed_seq_params
        )
        if self.config.rope_type == "rope":
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=False)
            mscale = 1.0
        else:
            rotary_pos_emb, mscale = self.rotary_pos_emb(rotary_seq_len, packed_seq=False)

        x_nope, x_pe = torch.split(
            x, [self.index_head_dim - self.index_rotary_dim, self.index_rotary_dim], dim=-1
        )
        x_pe = apply_rotary_pos_emb(
            x_pe,
            rotary_pos_emb,
            config=self.config,
            cu_seqlens=None,
            mscale=mscale,
            cp_group=self.pg_collection.cp,
        )
        return torch.cat([x_nope, x_pe], dim=-1)

    def _get_dynamic_rotary_pos_emb(self, inference_context) -> Tuple[torch.Tensor, float]:
        n = inference_context.active_token_count
        if n == 0:
            rotary_seq_len = 1
        else:
            rotary_seq_len = (
                int(inference_context.token_to_position_in_request[:n].max().item()) + 1
            )
        if self.config.rope_type == "rope":
            return self.rotary_pos_emb(rotary_seq_len, packed_seq=False), 1.0
        return self.rotary_pos_emb(rotary_seq_len, packed_seq=False)

    def _apply_rope_dynamic(self, q: torch.Tensor, k: torch.Tensor, inference_context):
        if self.rotary_pos_emb is None or self.index_rotary_dim == 0:
            return q, k

        rotary_pos_emb, mscale = self._get_dynamic_rotary_pos_emb(inference_context)
        q_nope, q_pe = torch.split(
            q, [self.index_head_dim - self.index_rotary_dim, self.index_rotary_dim], dim=-1
        )
        k_nope, k_pe = torch.split(
            k, [self.index_head_dim - self.index_rotary_dim, self.index_rotary_dim], dim=-1
        )

        active_token_count = inference_context.active_token_count
        q_pe = q_pe.clone()
        k_pe = k_pe.clone()
        if active_token_count > 0:
            q_positions = inference_context.token_to_pos_ids[:active_token_count]
            k_positions = inference_context.token_to_position_in_request[:active_token_count]
            q_pe[:active_token_count] = apply_rotary_pos_emb(
                q_pe[:active_token_count],
                rotary_pos_emb[q_positions],
                config=self.config,
                cu_seqlens=None,
                mscale=mscale,
                cp_group=self.pg_collection.cp,
            )
            k_pe[:active_token_count] = apply_rotary_pos_emb(
                k_pe[:active_token_count],
                rotary_pos_emb[k_positions],
                config=self.config,
                cu_seqlens=None,
                mscale=mscale,
                cp_group=self.pg_collection.cp,
            )
        q = torch.cat([q_nope, q_pe], dim=-1)
        k = torch.cat([k_nope, k_pe], dim=-1)
        if active_token_count < q.size(0):
            q[active_token_count:] = 0
            k[active_token_count:] = 0
        return q, k

    def forward_before_topk(
        self,
        hidden_states: torch.Tensor,
        use_rope: bool,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.config.sequence_parallel and self.pg_collection.tp.size() > 1:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, group=self.pg_collection.tp
            )

        seqlen, batch_size, _ = hidden_states.size()

        q, _ = self.linear_q(hidden_states)
        q = q.reshape(seqlen, batch_size, self.index_n_heads, self.index_head_dim)
        q = self._apply_rope(q, use_rope=use_rope, packed_seq_params=packed_seq_params)

        k, _ = self.linear_k(hidden_states)
        k = self.k_norm(k)
        k = k.reshape(seqlen, batch_size, 1, self.index_head_dim)
        k = self._apply_rope(k, use_rope=use_rope, packed_seq_params=packed_seq_params)
        k = k.reshape(seqlen, batch_size, self.index_head_dim)

        if self.config.dsa_indexer_use_hadamard:
            q = rotate_activation(q)
            k = rotate_activation(k)

        weights, _ = self.linear_weights_proj(hidden_states)
        weights = weights * (self.index_n_heads**-0.5) * self.softmax_scale
        return q, k, weights

    def forward_with_scores(
        self,
        hidden_states: torch.Tensor,
        use_rope: bool,
        mask: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert packed_seq_params is None, "Packed sequence is not supported for DSA-GQA."
        q, k, weights = self.forward_before_topk(hidden_states, use_rope, packed_seq_params)
        key_chunk_size = getattr(self.config, "dsa_indexer_topk_key_chunk_size", None)
        if key_chunk_size is not None and key_chunk_size > 0:
            return fused_qk_topk_chunked(q, k, weights, self.index_topk, mask, key_chunk_size)
        return fused_qk_topk_naive(q, k, weights, self.index_topk, mask)

    def forward_before_topk_dynamic(
        self,
        hidden_states: torch.Tensor,
        use_rope: bool,
        inference_context,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.config.sequence_parallel and self.pg_collection.tp.size() > 1:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, group=self.pg_collection.tp
            )

        seqlen, batch_size, _ = hidden_states.size()
        assert batch_size == 1, "Dynamic DSA-GQA expects batch=1 flattened token layout."

        q, _ = self.linear_q(hidden_states)
        q = q.reshape(seqlen, batch_size, self.index_n_heads, self.index_head_dim)

        k, _ = self.linear_k(hidden_states)
        k = self.k_norm(k)
        k = k.reshape(seqlen, batch_size, 1, self.index_head_dim)

        if use_rope:
            q, k = self._apply_rope_dynamic(q, k, inference_context)

        k = k.reshape(seqlen, batch_size, self.index_head_dim)

        if self.config.dsa_indexer_use_hadamard:
            q = rotate_activation(q)
            k = rotate_activation(k)

        weights, _ = self.linear_weights_proj(hidden_states)
        weights = weights * (self.index_n_heads**-0.5) * self.softmax_scale
        return q, k, weights


class SimplifiedDSGQAIndexer(MegatronModule):
    """One-head dot-product router over main-attention or separately learned K vectors."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SimplifiedDSGQAIndexerSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        super().__init__(config=config)
        # The public config enforces one global KV group. Attention internally rewrites
        # num_query_groups to the TP size when G < TP so its local dense-attention modules see
        # one group per partition; the actual K passed to this indexer remains one replicated
        # local group on every TP rank.
        self.hidden_size = config.hidden_size
        self.use_learned_k = getattr(config, "dsa_simplified_use_learned_k", False)
        self.index_n_heads = 1
        self.index_head_dim = config.dsa_indexer_head_dim
        self.index_topk = config.dsa_indexer_topk
        self.softmax_scale = self.index_head_dim**-0.5
        self.index_rotary_dim = int(self.index_head_dim * config.rotary_percent)
        self.index_rotary_dim -= self.index_rotary_dim % 2

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        self.pg_collection = pg_collection

        self.rotary_pos_emb = None
        if self.index_rotary_dim > 0:
            if config.rope_type == 'rope':
                self.rotary_pos_emb = RotaryEmbedding(
                    self.index_rotary_dim,
                    rotary_percent=1.0,
                    rotary_interleaved=config.rotary_interleaved,
                    seq_len_interpolation_factor=config.rotary_seq_len_interpolation_factor,
                    rotary_base=config.rotary_base,
                    rope_scaling=config.use_rope_scaling,
                    rope_scaling_factor=config.rope_scaling_factor,
                    use_cpu_initialization=config.use_cpu_initialization,
                    cp_group=self.pg_collection.cp,
                )
            elif config.rope_type == 'yarn':
                self.rotary_pos_emb = YarnRotaryEmbedding(
                    self.index_rotary_dim,
                    rotary_interleaved=config.rotary_interleaved,
                    seq_len_interpolation_factor=config.rotary_seq_len_interpolation_factor,
                    rotary_base=config.rotary_base,
                    scaling_factor=config.rotary_scaling_factor,
                    original_max_position_embeddings=config.original_max_position_embeddings,
                    beta_fast=config.beta_fast,
                    beta_slow=config.beta_slow,
                    mscale=config.mscale,
                    mscale_all_dim=config.mscale_all_dim,
                    use_cpu_initialization=config.use_cpu_initialization,
                    cp_group=self.pg_collection.cp,
                )

        self.linear_q = build_module(
            submodules.linear_q,
            self.hidden_size,
            self.index_head_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        self.linear_k = None
        if self.use_learned_k:
            self.linear_k = build_module(
                submodules.linear_k,
                self.hidden_size,
                self.index_head_dim,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                skip_weight_param_allocation=False,
                parallel_mode="duplicated",
            )
        if self.pg_collection.tp.size() > 1:
            for param in self.parameters():
                setattr(param, "average_gradients_across_tp_domain", True)

    def _apply_rope(self, q: torch.Tensor, use_rope: bool, packed_seq_params=None):
        if not use_rope or self.rotary_pos_emb is None or self.index_rotary_dim == 0:
            return q
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            None, None, q, self.config, packed_seq_params
        )
        if self.config.rope_type == "rope":
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=False)
            mscale = 1.0
        else:
            rotary_pos_emb, mscale = self.rotary_pos_emb(rotary_seq_len, packed_seq=False)
        q_nope, q_pe = torch.split(
            q, [self.index_head_dim - self.index_rotary_dim, self.index_rotary_dim], dim=-1
        )
        q_pe = apply_rotary_pos_emb(
            q_pe,
            rotary_pos_emb,
            config=self.config,
            cu_seqlens=None,
            mscale=mscale,
            cp_group=self.pg_collection.cp,
        )
        return torch.cat([q_nope, q_pe], dim=-1)

    def _apply_rope_dynamic(self, q: torch.Tensor, inference_context):
        if self.rotary_pos_emb is None or self.index_rotary_dim == 0:
            return q
        n = inference_context.active_token_count
        rotary_seq_len = (
            1
            if n == 0
            else int(inference_context.token_to_position_in_request[:n].max().item()) + 1
        )
        if self.config.rope_type == "rope":
            rotary_pos_emb, mscale = self.rotary_pos_emb(rotary_seq_len, packed_seq=False), 1.0
        else:
            rotary_pos_emb, mscale = self.rotary_pos_emb(rotary_seq_len, packed_seq=False)
        q_nope, q_pe = torch.split(
            q, [self.index_head_dim - self.index_rotary_dim, self.index_rotary_dim], dim=-1
        )
        q_pe = q_pe.clone()
        if n > 0:
            positions = inference_context.token_to_pos_ids[:n]
            q_pe[:n] = apply_rotary_pos_emb(
                q_pe[:n],
                rotary_pos_emb[positions],
                config=self.config,
                cu_seqlens=None,
                mscale=mscale,
                cp_group=self.pg_collection.cp,
            )
        q = torch.cat([q_nope, q_pe], dim=-1)
        if n < q.size(0):
            q[n:] = 0
        return q

    def _apply_rope_dynamic_qk(
        self, q: torch.Tensor, k: torch.Tensor, inference_context
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rotary_pos_emb is None or self.index_rotary_dim == 0:
            return q, k
        n = inference_context.active_token_count
        rotary_seq_len = (
            1
            if n == 0
            else int(inference_context.token_to_position_in_request[:n].max().item()) + 1
        )
        if self.config.rope_type == "rope":
            rotary_pos_emb, mscale = self.rotary_pos_emb(rotary_seq_len, packed_seq=False), 1.0
        else:
            rotary_pos_emb, mscale = self.rotary_pos_emb(rotary_seq_len, packed_seq=False)
        q_nope, q_pe = torch.split(
            q, [self.index_head_dim - self.index_rotary_dim, self.index_rotary_dim], dim=-1
        )
        k_nope, k_pe = torch.split(
            k, [self.index_head_dim - self.index_rotary_dim, self.index_rotary_dim], dim=-1
        )
        q_pe = q_pe.clone()
        k_pe = k_pe.clone()
        if n > 0:
            q_positions = inference_context.token_to_pos_ids[:n]
            k_positions = inference_context.token_to_position_in_request[:n]
            q_pe[:n] = apply_rotary_pos_emb(
                q_pe[:n],
                rotary_pos_emb[q_positions],
                config=self.config,
                cu_seqlens=None,
                mscale=mscale,
                cp_group=self.pg_collection.cp,
            )
            k_pe[:n] = apply_rotary_pos_emb(
                k_pe[:n],
                rotary_pos_emb[k_positions],
                config=self.config,
                cu_seqlens=None,
                mscale=mscale,
                cp_group=self.pg_collection.cp,
            )
        q = torch.cat([q_nope, q_pe], dim=-1)
        k = torch.cat([k_nope, k_pe], dim=-1)
        if n < q.size(0):
            q[n:] = 0
            k[n:] = 0
        return q, k

    def forward_q(
        self,
        hidden_states: torch.Tensor,
        use_rope: bool,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> torch.Tensor:
        if self.config.sequence_parallel and self.pg_collection.tp.size() > 1:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, group=self.pg_collection.tp
            )
        seqlen, batch_size, _ = hidden_states.shape
        q, _ = self.linear_q(hidden_states)
        q = q.reshape(seqlen, batch_size, 1, self.index_head_dim)
        return self._apply_rope(q, use_rope=use_rope, packed_seq_params=packed_seq_params)

    def forward_q_dynamic(self, hidden_states: torch.Tensor, use_rope: bool, inference_context):
        q = self.forward_q(hidden_states, use_rope=False)
        return self._apply_rope_dynamic(q, inference_context) if use_rope else q

    def forward_qk(
        self,
        hidden_states: torch.Tensor,
        use_rope: bool,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.use_learned_k or self.linear_k is None:
            raise RuntimeError("Simplified DSA learned-K projection is not enabled.")
        if self.config.sequence_parallel and self.pg_collection.tp.size() > 1:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, group=self.pg_collection.tp
            )
        seqlen, batch_size, _ = hidden_states.shape
        q, _ = self.linear_q(hidden_states)
        k, _ = self.linear_k(hidden_states)
        q = q.reshape(seqlen, batch_size, 1, self.index_head_dim)
        k = k.reshape(seqlen, batch_size, 1, self.index_head_dim)
        return (
            self._apply_rope(q, use_rope=use_rope, packed_seq_params=packed_seq_params),
            self._apply_rope(k, use_rope=use_rope, packed_seq_params=packed_seq_params),
        )

    def forward_qk_dynamic(self, hidden_states: torch.Tensor, use_rope: bool, inference_context):
        if not self.use_learned_k or self.linear_k is None:
            raise RuntimeError("Simplified DSA learned-K projection is not enabled.")
        if self.config.sequence_parallel and self.pg_collection.tp.size() > 1:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, group=self.pg_collection.tp
            )
        seqlen, batch_size, _ = hidden_states.shape
        q, _ = self.linear_q(hidden_states)
        k, _ = self.linear_k(hidden_states)
        q = q.reshape(seqlen, batch_size, 1, self.index_head_dim)
        k = k.reshape(seqlen, batch_size, 1, self.index_head_dim)
        return self._apply_rope_dynamic_qk(q, k, inference_context) if use_rope else (q, k)


class _DSAZeroParamDependency(torch.autograd.Function):
    """Attach zero indexer grads without reading overlap-gathered parameter storage."""

    @staticmethod
    def forward(ctx, output: torch.Tensor, *params: torch.Tensor) -> torch.Tensor:
        ctx.param_metadata = tuple((param.shape, param.dtype, param.device) for param in params)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        zero_grads = tuple(
            torch.zeros(shape, dtype=dtype, device=device)
            for shape, dtype, device in ctx.param_metadata
        )
        return (grad_output, *zero_grads)


def _simplified_index_scores(
    q_index: torch.Tensor,
    main_key: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Return token scores [B,Q,K] for the one-KV-group simplified router."""
    assert q_index.size(2) == 1 and main_key.size(2) == 1
    return torch.einsum(
        "qbd,kbd->bqk", q_index[:, :, 0, :].float(), main_key[:, :, 0, :].float()
    ) * softmax_scale


def _simplified_qk_topk_naive(
    q_index: torch.Tensor,
    main_key: torch.Tensor,
    topk: int,
    softmax_scale: float,
    mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    scores = _simplified_index_scores(q_index, main_key, softmax_scale)
    if mask is not None:
        scores = scores + mask
    return scores, scores.topk(min(topk, scores.size(-1)), dim=-1).indices


def _simplified_qk_topk_chunked(
    q_index: torch.Tensor,
    main_key: torch.Tensor,
    topk: int,
    softmax_scale: float,
    mask: Optional[torch.Tensor],
    key_chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    running_scores = None
    running_indices = None
    for k_start in range(0, main_key.size(0), key_chunk_size):
        k_end = min(k_start + key_chunk_size, main_key.size(0))
        block_scores = _simplified_index_scores(
            q_index, main_key[k_start:k_end], softmax_scale
        )
        if mask is not None:
            block_scores = block_scores + mask[..., k_start:k_end]
        block_topk = min(topk, block_scores.size(-1))
        block_scores, block_indices = block_scores.topk(block_topk, dim=-1)
        block_indices = block_indices + k_start
        if running_scores is None:
            running_scores, running_indices = block_scores, block_indices
            continue
        merged_scores = torch.cat((running_scores, block_scores), dim=-1)
        merged_indices = torch.cat((running_indices, block_indices), dim=-1)
        keep = merged_scores.topk(min(topk, merged_scores.size(-1)), dim=-1).indices
        running_scores = torch.gather(merged_scores, -1, keep)
        running_indices = torch.gather(merged_indices, -1, keep)
    return running_scores, running_indices


class DSGQACoreAttention(MegatronModule):
    """Token-level DSA core attention for grouped-query attention."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: DSGQAAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        k_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        cp_comm_type: str = "p2p",
        pg_collection: ProcessGroupCollection = None,
    ):
        super().__init__(config=config)
        self.layer_number = layer_number
        self.indexer = build_module(
            submodules.indexer, config=config, pg_collection=pg_collection
        )
        self.dense_core_attention = None
        if (
            getattr(config, "dsa_fwd_use_dense_attn", False)
            or getattr(config, "dsa_fwd_skip_dsa", False)
        ) and (
            submodules.dense_core_attention is not None
        ):
            self.dense_core_attention = build_module(
                submodules.dense_core_attention,
                config=config,
                layer_number=layer_number,
                attn_mask_type=attn_mask_type,
                attention_type=attention_type,
                softmax_scale=softmax_scale,
                cp_comm_type=cp_comm_type,
                pg_collection=pg_collection,
            )
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(
                k_channels if k_channels is not None else config.kv_channels
            )
        self.softmax_scale = softmax_scale

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        use_indexer_rope: bool = False,
        indexer_input_norm: Optional[_DSAIndexerInputNormSpec] = None,
        attn_mask_type: AttnMaskType = None,
        attention_bias: torch.Tensor = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        assert attention_bias is None, "attention_bias is not supported for DSA-GQA."
        assert packed_seq_params is None, "Packed sequence is not supported for DSA-GQA."
        if key.size(0) != hidden_states.size(0):
            if self.config.sequence_parallel:
                raise NotImplementedError(
                    "DSA-GQA does not currently support sequence parallelism."
                )
            raise NotImplementedError(
                "DSA-GQA currently supports full-sequence attention only. Decode-time "
                "indexer caching is not implemented yet."
            )

        sq, b, _, _ = query.size()
        skv = key.size(0)
        dsa_min_memory_backend = getattr(self.config, "dsa_min_memory_backend", "reference")
        if getattr(self.config, "dsa_fwd_skip_dsa", False) or dsa_min_memory_backend in (
            "triton-min-memory",
            "torch-min-memory",
        ):
            return self._forward_min_memory(
                query=query,
                key=key,
                value=value,
                attention_mask=attention_mask,
                hidden_states=hidden_states,
                use_indexer_rope=use_indexer_rope,
                indexer_input_norm=indexer_input_norm,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )

        sparse_attention_use_gather = getattr(self.config, "dsa_sparse_attention_use_gather", False)
        simplified_indexer = getattr(self.config, "dsa_indexer_mode", "standard") == "simplified"
        simplified_learned_k = simplified_indexer and getattr(
            self.config, "dsa_simplified_use_learned_k", False
        )

        hidden_states = hidden_states.detach()
        if _simplified_indexer_uses_main_input_norm(self.config) or getattr(
            self.config, "dsa_standard_indexer_use_main_input_norm", False
        ):
            hidden_states = _normalized_indexer_input(hidden_states, indexer_input_norm)

        if attn_mask_type is not None:
            assert attn_mask_type == AttnMaskType.causal, 'Only causal mask is supported for now'
            routing_mask = torch.triu(
                torch.full((sq, skv), float('-inf'), dtype=torch.float32, device=query.device),
                diagonal=1,
            )
            sparse_attention_mask = None if sparse_attention_use_gather else routing_mask
        else:
            assert attention_mask.shape == (b, 1, sq, skv), 'attention_mask shape mismatch'
            sparse_attention_mask = attention_mask.squeeze(1)
            routing_mask = torch.zeros_like(
                sparse_attention_mask, dtype=torch.float32
            ).masked_fill(sparse_attention_mask, float('-inf'))
            if not sparse_attention_use_gather:
                sparse_attention_mask = routing_mask

        train_main_only = getattr(self.config, "dsa_train_main_only", False)
        indexer_loss_coeff = (
            0.0
            if train_main_only
            else (getattr(self.config, 'dsa_indexer_loss_coeff', 0.0) or 0.0)
        )
        if self.training and torch.is_grad_enabled():
            sparse_indexer_loss = getattr(self.config, "dsa_indexer_use_sparse_loss", False)
            sparse_indexer_loss_use_topk_only = getattr(
                self.config, "dsa_indexer_sparse_loss_use_topk_only", False
            )
            recompute_indexer_loss = getattr(self.config, "dsa_indexer_loss_recompute", False)
            if simplified_indexer:
                if simplified_learned_k:
                    q_index, k_index = self.indexer.forward_qk(
                        hidden_states,
                        use_rope=use_indexer_rope,
                        packed_seq_params=packed_seq_params,
                    )
                else:
                    q_index = self.indexer.forward_q(
                        hidden_states,
                        use_rope=use_indexer_rope,
                        packed_seq_params=packed_seq_params,
                    )
                    k_index = key.detach()
                weights = None
            else:
                q_index, k_index, weights = self.indexer.forward_before_topk(
                    hidden_states, use_rope=use_indexer_rope, packed_seq_params=packed_seq_params
                )
            key_chunk_size = getattr(self.config, "dsa_indexer_topk_key_chunk_size", None)
            recompute_topk = getattr(self.config, "dsa_indexer_topk_recompute", False)
            use_chunked_topk = (
                key_chunk_size is not None
                and key_chunk_size > 0
                and (
                    indexer_loss_coeff <= 0
                    or (sparse_indexer_loss and sparse_indexer_loss_use_topk_only)
                )
            )
            if use_chunked_topk:
                def _compute_chunked_topk(
                    q_index_tensor: torch.Tensor,
                    k_index_tensor: torch.Tensor,
                    weights_tensor: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
                    if simplified_indexer:
                        return _simplified_qk_topk_chunked(
                            q_index_tensor,
                            k_index_tensor,
                            self.indexer.index_topk,
                            self.indexer.softmax_scale,
                            routing_mask,
                            key_chunk_size,
                        )
                    return fused_qk_topk_chunked(
                        q_index_tensor, k_index_tensor, weights_tensor,
                        self.indexer.index_topk, routing_mask, key_chunk_size,
                    )

                routing_inputs_require_grad = q_index.requires_grad or k_index.requires_grad or (
                    weights is not None and weights.requires_grad
                )
                if recompute_topk and routing_inputs_require_grad:
                    topk_scores, topk_indices = torch_checkpoint.checkpoint(
                        _compute_chunked_topk,
                        q_index,
                        k_index,
                        weights if weights is not None else q_index.new_empty((0,)),
                        use_reentrant=False,
                    )
                else:
                    topk_scores, topk_indices = _compute_chunked_topk(
                        q_index,
                        k_index,
                        weights if weights is not None else q_index.new_empty((0,)),
                    )
                index_scores = None
            else:
                if simplified_indexer:
                    index_scores, topk_indices = _simplified_qk_topk_naive(
                        q_index,
                        k_index,
                        self.indexer.index_topk,
                        self.indexer.softmax_scale,
                        routing_mask,
                    )
                else:
                    index_scores, topk_indices = fused_qk_topk_naive(
                        q_index, k_index, weights, self.indexer.index_topk, routing_mask
                    )
                topk_scores = None

            indexer_loss = None
            if indexer_loss_coeff > 0:
                query_detached = query.detach()
                key_detached = key.detach()

                if use_chunked_topk and sparse_indexer_loss and sparse_indexer_loss_use_topk_only:
                    def _compute_sparse_topk_only_indexer_loss(
                        selected_scores_tensor: torch.Tensor,
                    ) -> torch.Tensor:
                        return compute_gqa_dsa_indexer_loss(
                            None,
                            topk_indices,
                            query_detached,
                            key_detached,
                            self.softmax_scale,
                            indexer_loss_coeff,
                            sparse_indexer_loss,
                            self.indexer.pg_collection,
                            sparse_indexer_loss_use_topk_only,
                            getattr(self.config, "dsa_indexer_loss_query_chunk_size", None),
                            selected_index_scores=selected_scores_tensor,
                        )

                    if recompute_indexer_loss and topk_scores.requires_grad:
                        indexer_loss = torch_checkpoint.checkpoint(
                            _compute_sparse_topk_only_indexer_loss,
                            topk_scores,
                            use_reentrant=False,
                        )
                    else:
                        indexer_loss = _compute_sparse_topk_only_indexer_loss(topk_scores)
                else:
                    def _compute_indexer_loss(index_scores_tensor: torch.Tensor) -> torch.Tensor:
                        return compute_gqa_dsa_indexer_loss(
                            index_scores_tensor,
                            topk_indices,
                            query_detached,
                            key_detached,
                            self.softmax_scale,
                            indexer_loss_coeff,
                            sparse_indexer_loss,
                            self.indexer.pg_collection,
                            sparse_indexer_loss_use_topk_only,
                            getattr(self.config, "dsa_indexer_loss_query_chunk_size", None),
                        )

                    if recompute_indexer_loss and index_scores.requires_grad:
                        indexer_loss = torch_checkpoint.checkpoint(
                            _compute_indexer_loss,
                            index_scores,
                            use_reentrant=False,
                        )
                    else:
                        indexer_loss = _compute_indexer_loss(index_scores)
                DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                    loss=indexer_loss,
                    raw_loss=indexer_loss / indexer_loss_coeff,
                    layer_number=self.layer_number,
                    num_layers=self.config.num_layers,
                )

            recompute_sparse_attention = getattr(self.config, "dsa_sparse_attention_recompute", False)
            sparse_attention_use_gather = getattr(
                self.config, "dsa_sparse_attention_use_gather", False
            )
            sparse_attention_query_chunk_size = getattr(
                self.config, "dsa_sparse_attention_query_chunk_size", None
            )
            if recompute_sparse_attention and (
                query.requires_grad or key.requires_grad or value.requires_grad
            ):
                def _compute_sparse_attention(
                    query_tensor: torch.Tensor,
                    key_tensor: torch.Tensor,
                    value_tensor: torch.Tensor,
                ) -> torch.Tensor:
                    return unfused_grouped_dsa_fn(
                        query_tensor,
                        key_tensor,
                        value_tensor,
                        topk_indices,
                        self.softmax_scale,
                        mask=sparse_attention_mask,
                        query_chunk_size=sparse_attention_query_chunk_size,
                        use_gather=sparse_attention_use_gather,
                    )

                output = torch_checkpoint.checkpoint(
                    _compute_sparse_attention,
                    query,
                    key,
                    value,
                    use_reentrant=False,
                )
            else:
                output = unfused_grouped_dsa_fn(
                    query,
                    key,
                    value,
                    topk_indices,
                    self.softmax_scale,
                    mask=sparse_attention_mask,
                    query_chunk_size=sparse_attention_query_chunk_size,
                    use_gather=sparse_attention_use_gather,
                )
            if indexer_loss is not None:
                output = DSAIndexerLossAutoScaler.apply(output, indexer_loss)
            return output

        if simplified_indexer:
            if simplified_learned_k:
                q_index, k_index = self.indexer.forward_qk(
                    hidden_states,
                    use_rope=use_indexer_rope,
                    packed_seq_params=packed_seq_params,
                )
            else:
                q_index = self.indexer.forward_q(
                    hidden_states,
                    use_rope=use_indexer_rope,
                    packed_seq_params=packed_seq_params,
                )
                k_index = key.detach()
            _, topk_indices = _simplified_qk_topk_naive(
                q_index,
                k_index,
                self.indexer.index_topk,
                self.indexer.softmax_scale,
                routing_mask,
            )
        else:
            _, topk_indices = self.indexer.forward_with_scores(
                hidden_states,
                use_rope=use_indexer_rope,
                mask=routing_mask,
                packed_seq_params=packed_seq_params,
            )
        return unfused_grouped_dsa_fn(
            query,
            key,
            value,
            topk_indices,
            self.softmax_scale,
            mask=sparse_attention_mask,
            query_chunk_size=getattr(self.config, "dsa_sparse_attention_query_chunk_size", None),
            use_gather=sparse_attention_use_gather,
        )

    def _forward_min_memory(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        use_indexer_rope: bool = False,
        indexer_input_norm: Optional[_DSAIndexerInputNormSpec] = None,
        attn_mask_type: AttnMaskType = None,
        attention_bias: torch.Tensor = None,
        packed_seq_params: PackedSeqParams = None,
    ) -> torch.Tensor:
        """Minimum-activation DSA-GQA path for training and no-grad validation."""
        dsa_min_memory_backend = getattr(self.config, "dsa_min_memory_backend", "reference")
        skip_dsa = getattr(self.config, "dsa_fwd_skip_dsa", False)
        dense_warmup = getattr(self.config, "dsa_fwd_use_dense_attn", False)
        train_main_only = getattr(self.config, "dsa_train_main_only", False)
        sparse_indexer_loss = getattr(self.config, "dsa_indexer_use_sparse_loss", False)
        sparse_fwd_dense_loss = (
            not train_main_only
            and not skip_dsa
            and not dense_warmup
            and not sparse_indexer_loss
        )
        simplified_indexer = getattr(self.config, "dsa_indexer_mode", "standard") == "simplified"
        if simplified_indexer and not _simplified_indexer_uses_main_input_norm(self.config):
            indexer_input_norm = None
        assert attention_bias is None, "attention_bias is not supported for DSA-GQA."
        assert packed_seq_params is None, "Packed sequence is not supported for DSA-GQA."
        if attn_mask_type != AttnMaskType.causal:
            raise NotImplementedError(
                f"dsa_min_memory_backend='{dsa_min_memory_backend}' only supports causal fixed-length batches."
            )
        if query.size(0) != key.size(0) or key.size(0) != hidden_states.size(0):
            raise NotImplementedError(
                f"dsa_min_memory_backend='{dsa_min_memory_backend}' requires full-sequence self attention."
            )
        if skip_dsa:
            if self.dense_core_attention is None:
                raise RuntimeError("DSA skip mode requires an original dense core attention spec.")
            output = self.dense_core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
            zero_indexer_loss = output.new_zeros((), dtype=torch.float32)
            DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                loss=zero_indexer_loss,
                raw_loss=zero_indexer_loss,
                layer_number=self.layer_number,
                num_layers=self.config.num_layers,
            )
            if not torch.is_grad_enabled():
                return output
            if not self.training:
                raise NotImplementedError(
                    "dsa_fwd_skip_dsa supports training and no-grad validation only."
                )
            trainable_indexer_params = tuple(
                param
                for param in self.indexer.parameters()
                if param.requires_grad and param.numel() > 0
            )
            if not trainable_indexer_params:
                return output
            return _DSAZeroParamDependency.apply(output, *trainable_indexer_params)
        if getattr(self.config, "dsa_sparse_attention_use_gather", False):
            raise NotImplementedError(
                f"dsa_min_memory_backend='{dsa_min_memory_backend}' bypasses the reference gather backend; "
                "do not set dsa_sparse_attention_use_gather."
            )
        if dense_warmup and getattr(self.config, "dsa_indexer_use_sparse_loss", False):
            raise NotImplementedError(
                "dsa_fwd_use_dense_attn uses dense indexer loss; do not set "
                "dsa_indexer_use_sparse_loss."
            )
        if dense_warmup and (
            getattr(self.config, "dsa_kernel_cache_routing", False)
            or getattr(self.config, "dsa_kernel_cache_indexer_k", False)
            or getattr(self.config, "dsa_kernel_cache_selected_scores", False)
        ):
            raise NotImplementedError("dsa_fwd_use_dense_attn does not support DSA cache flags.")
        if sparse_fwd_dense_loss and getattr(
            self.config, "dsa_kernel_cache_selected_scores", False
        ):
            raise NotImplementedError(
                "Sparse-forward dense-loss mode has no selected-score sparse loss; do not set "
                "dsa_kernel_cache_selected_scores."
            )
        if train_main_only and getattr(
            self.config, "dsa_kernel_cache_selected_scores", False
        ):
            raise NotImplementedError(
                "dsa_train_main_only has no selected-score KL backward; do not set "
                "dsa_kernel_cache_selected_scores."
            )
        if train_main_only and (skip_dsa or dense_warmup):
            raise NotImplementedError(
                "dsa_train_main_only requires sparse DSA forward attention."
            )
        if not simplified_indexer and not getattr(self.config, "dsa_indexer_use_hadamard", False):
            raise NotImplementedError(
                f"dsa_min_memory_backend='{dsa_min_memory_backend}' requires dsa_indexer_use_hadamard."
            )
        if self.config.fp8 is not None or self.config.fp8_param or is_using_quantization_scales(
            self.config
        ):
            raise NotImplementedError(
                f"dsa_min_memory_backend='{dsa_min_memory_backend}' does not yet support quantized/FP8 "
                "indexer projections."
            )
        if not simplified_indexer and self.config.layernorm_zero_centered_gamma:
            raise NotImplementedError(
                f"dsa_min_memory_backend='{dsa_min_memory_backend}' does not yet support "
                "layernorm_zero_centered_gamma in the DSA indexer norm."
            )
        if dense_warmup:
            if self.dense_core_attention is None:
                raise RuntimeError("Dense DSA warmup requires an original dense core attention spec.")
            if not torch.is_grad_enabled():
                return self.dense_core_attention(
                    query,
                    key,
                    value,
                    attention_mask,
                    attn_mask_type=attn_mask_type,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                )
            if not self.training:
                raise NotImplementedError(
                    f"dsa_min_memory_backend='{dsa_min_memory_backend}' currently supports training only."
                )

            indexer_loss_coeff = getattr(self.config, "dsa_indexer_loss_coeff", 0.0) or 0.0
            if indexer_loss_coeff <= 0:
                raise NotImplementedError(
                    f"dsa_min_memory_backend='{dsa_min_memory_backend}' expects dsa_indexer_loss_coeff > 0 "
                    "for dense indexer warmup."
                )

            output = self.dense_core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
            indexer_loss = dsa_dense_indexer_loss(
                query=query.detach(),
                key=key.detach(),
                hidden_states=hidden_states.detach(),
                indexer=self.indexer,
                softmax_scale=self.softmax_scale,
                loss_coeff=indexer_loss_coeff,
                use_indexer_rope=use_indexer_rope,
                query_chunk_size=getattr(self.config, "dsa_kernel_query_block_size", None),
                key_chunk_size=getattr(self.config, "dsa_kernel_key_block_size", None),
                simplified_input_norm=indexer_input_norm,
                profile_enabled=getattr(self.config, "dsa_min_memory_profile", False),
                profile_rank=getattr(self.config, "dsa_min_memory_profile_rank", 0),
                profile_label=f"layer={self.layer_number}",
                use_triton=dsa_min_memory_backend == "triton-min-memory",
            )
            DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                loss=indexer_loss,
                raw_loss=indexer_loss / indexer_loss_coeff,
                layer_number=self.layer_number,
                num_layers=self.config.num_layers,
            )
            return DSAIndexerLossAutoScaler.apply(output, indexer_loss)

        del attention_mask
        if not torch.is_grad_enabled():
            return dsa_min_memory_gqa_forward_only(
                query=query,
                key=key,
                value=value,
                hidden_states=hidden_states.detach(),
                indexer=self.indexer,
                softmax_scale=self.softmax_scale,
                use_indexer_rope=use_indexer_rope,
                query_chunk_size=getattr(self.config, "dsa_kernel_query_block_size", None),
                key_chunk_size=getattr(self.config, "dsa_kernel_key_block_size", None),
                simplified_input_norm=indexer_input_norm,
                cache_indexer_k=getattr(self.config, "dsa_kernel_cache_indexer_k", False),
                profile_enabled=getattr(self.config, "dsa_min_memory_profile", False),
                profile_rank=getattr(self.config, "dsa_min_memory_profile_rank", 0),
                profile_label=f"layer={self.layer_number}",
                use_triton=dsa_min_memory_backend == "triton-min-memory",
                use_cudnn=getattr(self.config, "dsa_use_cudnn", False),
            )
        if not self.training:
            raise NotImplementedError(
                f"dsa_min_memory_backend='{dsa_min_memory_backend}' currently supports training only."
            )

        configured_indexer_loss_coeff = (
            getattr(self.config, "dsa_indexer_loss_coeff", 0.0) or 0.0
        )
        if not train_main_only and configured_indexer_loss_coeff <= 0:
            raise NotImplementedError(
                f"dsa_min_memory_backend='{dsa_min_memory_backend}' expects dsa_indexer_loss_coeff > 0 "
                "for indexer training."
            )
        indexer_loss_coeff = 0.0 if train_main_only else configured_indexer_loss_coeff

        sparse_loss_coeff = indexer_loss_coeff if sparse_indexer_loss else 0.0
        output, indexer_loss = dsa_min_memory_gqa(
            query=query,
            key=key,
            value=value,
            hidden_states=hidden_states.detach(),
            indexer=self.indexer,
            softmax_scale=self.softmax_scale,
            loss_coeff=sparse_loss_coeff,
            use_indexer_rope=use_indexer_rope,
            query_chunk_size=getattr(self.config, "dsa_kernel_query_block_size", None),
            key_chunk_size=getattr(self.config, "dsa_kernel_key_block_size", None),
            simplified_input_norm=indexer_input_norm,
            cache_routing=getattr(self.config, "dsa_kernel_cache_routing", False),
            cache_indexer_k=getattr(self.config, "dsa_kernel_cache_indexer_k", False),
            cache_selected_scores=getattr(
                self.config, "dsa_kernel_cache_selected_scores", False
            ),
            profile_enabled=getattr(self.config, "dsa_min_memory_profile", False),
            profile_rank=getattr(self.config, "dsa_min_memory_profile_rank", 0),
            profile_label=f"layer={self.layer_number}",
            use_triton=dsa_min_memory_backend == "triton-min-memory",
            use_cudnn=getattr(self.config, "dsa_use_cudnn", False),
        )
        if sparse_fwd_dense_loss:
            indexer_loss = dsa_dense_indexer_loss(
                query=query.detach(),
                key=key.detach(),
                hidden_states=hidden_states.detach(),
                indexer=self.indexer,
                softmax_scale=self.softmax_scale,
                loss_coeff=indexer_loss_coeff,
                use_indexer_rope=use_indexer_rope,
                query_chunk_size=getattr(self.config, "dsa_kernel_query_block_size", None),
                key_chunk_size=getattr(self.config, "dsa_kernel_key_block_size", None),
                simplified_input_norm=indexer_input_norm,
                profile_enabled=getattr(self.config, "dsa_min_memory_profile", False),
                profile_rank=getattr(self.config, "dsa_min_memory_profile_rank", 0),
                profile_label=f"layer={self.layer_number}",
                use_triton=dsa_min_memory_backend == "triton-min-memory",
            )
        if not train_main_only:
            DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                loss=indexer_loss,
                raw_loss=indexer_loss / indexer_loss_coeff,
                layer_number=self.layer_number,
                num_layers=self.config.num_layers,
            )
        mass_loss_coeff = getattr(self.config, "dsa_topk_mass_loss_coeff", 0.0)
        output_loss_coeff = getattr(
            self.config, "dsa_output_consistency_loss_coeff", 0.0
        )
        if mass_loss_coeff <= 0.0 and output_loss_coeff <= 0.0:
            if train_main_only:
                return output
            return DSAIndexerLossAutoScaler.apply(output, indexer_loss)

        if not train_main_only:
            output = DSAIndexerLossAutoScaler.apply(output, indexer_loss)
        mass_loss, output_loss, captured_mass = dsa_main_attention_aux_loss(
            query=query,
            key=key,
            value=value,
            hidden_states=hidden_states.detach(),
            indexer=self.indexer,
            attention_softmax_scale=self.softmax_scale,
            use_indexer_rope=use_indexer_rope,
            aux_topk=self.config.dsa_attention_aux_topk,
            mass_loss_coeff=mass_loss_coeff,
            mass_target=self.config.dsa_topk_mass_target,
            output_loss_coeff=output_loss_coeff,
            query_chunk_size=getattr(self.config, "dsa_kernel_query_block_size", None),
            key_chunk_size=getattr(self.config, "dsa_kernel_key_block_size", None),
            simplified_input_norm=indexer_input_norm,
            profile_enabled=getattr(self.config, "dsa_min_memory_profile", False),
            profile_rank=getattr(self.config, "dsa_min_memory_profile_rank", 0),
            profile_label=f"layer={self.layer_number}",
            use_triton=dsa_min_memory_backend == "triton-min-memory",
        )
        zero = captured_mass.new_zeros(())
        raw_mass_loss = mass_loss / mass_loss_coeff if mass_loss_coeff > 0.0 else zero
        raw_output_loss = (
            output_loss / output_loss_coeff if output_loss_coeff > 0.0 else zero
        )
        DSAMainAttentionAuxLossLoggingHelper.save_loss_to_tracker(
            captured_mass=captured_mass,
            mass_loss=mass_loss,
            raw_mass_loss=raw_mass_loss,
            output_loss=output_loss,
            raw_output_loss=raw_output_loss,
            layer_number=self.layer_number,
            num_layers=self.config.num_layers,
            tp_group=self.indexer.pg_collection.tp,
        )
        aux_loss = mass_loss if mass_loss_coeff > 0.0 else output_loss
        if mass_loss_coeff > 0.0 and output_loss_coeff > 0.0:
            aux_loss = mass_loss + output_loss
        return DSAMainAttentionAuxLossAutoScaler.apply(output, aux_loss)

    def forward_dynamic(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        hidden_states: torch.Tensor,
        inference_context,
        provider_layer_number: int,
        block_table: torch.Tensor,
        use_indexer_rope: bool = False,
        indexer_input_norm: Optional[_DSAIndexerInputNormSpec] = None,
    ) -> torch.Tensor:
        assert not self.training, "Dynamic DSA-GQA inference only supports eval mode."
        assert value_cache is not None, "Dynamic DSA-GQA requires value cache."
        if getattr(self.config, "dsa_fwd_skip_dsa", False):
            raise NotImplementedError("dsa_fwd_skip_dsa is not supported by dynamic inference.")

        simplified_indexer = getattr(self.config, "dsa_indexer_mode", "standard") == "simplified"
        simplified_learned_k = simplified_indexer and getattr(
            self.config, "dsa_simplified_use_learned_k", False
        )
        if _simplified_indexer_uses_main_input_norm(self.config) or getattr(
            self.config, "dsa_standard_indexer_use_main_input_norm", False
        ):
            hidden_states = _normalized_indexer_input(hidden_states, indexer_input_norm)
        if simplified_indexer:
            if simplified_learned_k:
                q_index, k_index_current = self.indexer.forward_qk_dynamic(
                    hidden_states,
                    use_rope=use_indexer_rope,
                    inference_context=inference_context,
                )
                inference_context.append_dsa_key_cache(provider_layer_number, k_index_current)
                dsa_key_cache, dsa_block_table = inference_context.dsa_key_cache(
                    provider_layer_number
                )
                block_table = dsa_block_table
            else:
                q_index = self.indexer.forward_q_dynamic(
                    hidden_states,
                    use_rope=use_indexer_rope,
                    inference_context=inference_context,
                )
                dsa_key_cache = None
            weights = None
        else:
            q_index, k_index_current, weights = self.indexer.forward_before_topk_dynamic(
                hidden_states,
                use_rope=use_indexer_rope,
                inference_context=inference_context,
            )
            inference_context.append_dsa_key_cache(provider_layer_number, k_index_current)
            dsa_key_cache, dsa_block_table = inference_context.dsa_key_cache(
                provider_layer_number
            )
            block_table = dsa_block_table

        query_lengths = inference_context.active_attn_metadata["mha_metadata"].state_data["query_lengths"]
        kv_lengths = inference_context.active_attn_metadata["mha_metadata"].state_data["kv_seq_lengths"]
        kv_offsets = inference_context.request_kv_length_offsets[
            inference_context.paused_request_count : inference_context.total_request_count
        ]

        sq, b, np, _ = query.size()
        value_head_dim = value_cache.size(-1)
        output = value_cache.new_zeros((sq, b, np * value_head_dim))

        q_cursor = 0
        block_size_tokens = inference_context.block_size_tokens
        num_requests = inference_context.padded_active_request_count
        diagnostics = getattr(inference_context, "dsa_diagnostics", None)
        diagnostics_enabled = diagnostics is not None and diagnostics.enabled
        active_request_ids = (
            inference_context.request_ids[
                inference_context.paused_request_count : inference_context.total_request_count
            ]
            if diagnostics_enabled
            else None
        )

        for request_idx in range(num_requests):
            query_length = int(query_lengths[request_idx].item())
            if query_length == 0:
                continue

            key_length = int(kv_lengths[request_idx].item())
            query_start = q_cursor
            query_end = q_cursor + query_length
            q_cursor = query_end

            if key_length == 0:
                continue

            block_table_row = block_table[request_idx]
            request_key = _gather_block_cache_sequence(
                key_cache, block_table_row, key_length, block_size_tokens
            ).unsqueeze(1)
            request_value = _gather_block_cache_sequence(
                value_cache, block_table_row, key_length, block_size_tokens
            ).unsqueeze(1)
            if simplified_indexer:
                if simplified_learned_k:
                    request_index_key = _gather_block_cache_sequence(
                        dsa_key_cache, block_table_row, key_length, block_size_tokens
                    ).unsqueeze(1).unsqueeze(2)
                else:
                    request_index_key = request_key
            else:
                request_index_key = _gather_block_cache_sequence(
                    dsa_key_cache, block_table_row, key_length, block_size_tokens
                ).unsqueeze(1)

            request_query = query[query_start:query_end]
            request_q_index = q_index[query_start:query_end]
            request_weights = None if weights is None else weights[query_start:query_end]
            request_offset = int(kv_offsets[request_idx].item()) if request_idx < kv_offsets.numel() else 0
            request_mask = _build_shifted_causal_mask(
                query_length, key_length, request_offset, request_query.device
            )
            request_id = None
            diagnostic_queries = []
            if diagnostics_enabled and diagnostics.layer_enabled(self.layer_number):
                assert active_request_ids is not None
                if request_idx >= active_request_ids.numel():
                    raise RuntimeError("DSA diagnostics encountered an unmapped padded request.")
                request_id = int(active_request_ids[request_idx].item())
                diagnostic_queries = diagnostics.selected_queries(
                    request_id=request_id,
                    layer_number=self.layer_number,
                    query_start_position=request_offset,
                    query_length=query_length,
                )

            key_chunk_size = getattr(self.config, "dsa_indexer_topk_key_chunk_size", None)
            if simplified_indexer and (key_chunk_size is None or key_chunk_size <= 0):
                key_chunk_size = getattr(self.config, "dsa_kernel_key_block_size", None)
            if simplified_indexer:
                query_chunk_size = getattr(self.config, "dsa_kernel_query_block_size", None)
                if query_chunk_size is None or query_chunk_size <= 0:
                    query_chunk_size = query_length
                topk_chunks = []
                for local_q_start in range(0, query_length, query_chunk_size):
                    local_q_end = min(local_q_start + query_chunk_size, query_length)
                    q_index_chunk = request_q_index[local_q_start:local_q_end]
                    mask_chunk = request_mask[local_q_start:local_q_end]
                    if key_chunk_size is not None and key_chunk_size > 0:
                        _, topk_chunk = _simplified_qk_topk_chunked(
                            q_index_chunk,
                            request_index_key,
                            self.indexer.index_topk,
                            self.indexer.softmax_scale,
                            mask_chunk,
                            key_chunk_size,
                        )
                    else:
                        _, topk_chunk = _simplified_qk_topk_naive(
                            q_index_chunk,
                            request_index_key,
                            self.indexer.index_topk,
                            self.indexer.softmax_scale,
                            mask_chunk,
                        )
                    topk_chunks.append(topk_chunk)
                topk_indices = torch.cat(topk_chunks, dim=1)
            elif key_chunk_size is not None and key_chunk_size > 0:
                _, topk_indices = fused_qk_topk_chunked(
                    request_q_index,
                    request_index_key,
                    request_weights,
                    self.indexer.index_topk,
                    request_mask,
                    key_chunk_size,
                )
            else:
                _, topk_indices = fused_qk_topk_naive(
                    request_q_index,
                    request_index_key,
                    request_weights,
                    self.indexer.index_topk,
                    request_mask,
                )

            diagnostic_topk_indices = None
            if diagnostic_queries:
                diagnostic_rows = torch.tensor(
                    [item["local_index"] for item in diagnostic_queries],
                    device=request_q_index.device,
                    dtype=torch.long,
                )
                diagnostic_topk = min(max(diagnostics.topk_values), key_length)
                if diagnostic_topk <= topk_indices.size(-1):
                    diagnostic_topk_indices = topk_indices.index_select(1, diagnostic_rows)
                else:
                    diagnostic_q_index = request_q_index.index_select(0, diagnostic_rows)
                    diagnostic_mask = request_mask.index_select(0, diagnostic_rows)
                    diagnostic_key_chunk_size = key_chunk_size
                    if diagnostic_key_chunk_size is None or diagnostic_key_chunk_size <= 0:
                        diagnostic_key_chunk_size = getattr(
                            self.config, "dsa_kernel_key_block_size", None
                        )
                    if diagnostic_key_chunk_size is None or diagnostic_key_chunk_size <= 0:
                        diagnostic_key_chunk_size = 2048
                    diagnostic_weights = (
                        None
                        if request_weights is None
                        else request_weights.index_select(0, diagnostic_rows)
                    )
                    if simplified_indexer:
                        _, diagnostic_topk_indices = _simplified_qk_topk_chunked(
                            diagnostic_q_index,
                            request_index_key,
                            diagnostic_topk,
                            self.indexer.softmax_scale,
                            diagnostic_mask,
                            diagnostic_key_chunk_size,
                        )
                    else:
                        _, diagnostic_topk_indices = fused_qk_topk_chunked(
                            diagnostic_q_index,
                            request_index_key,
                            diagnostic_weights,
                            diagnostic_topk,
                            diagnostic_mask,
                            diagnostic_key_chunk_size,
                        )
            sparse_attention_query_chunk_size = getattr(
                self.config, "dsa_sparse_attention_query_chunk_size", None
            )
            if simplified_indexer and (
                sparse_attention_query_chunk_size is None
                or sparse_attention_query_chunk_size <= 0
            ):
                sparse_attention_query_chunk_size = getattr(
                    self.config, "dsa_kernel_query_block_size", None
                )
            request_output = unfused_grouped_dsa_fn(
                request_query,
                request_key,
                request_value,
                topk_indices,
                self.softmax_scale,
                mask=request_mask,
                query_chunk_size=sparse_attention_query_chunk_size,
                use_gather=(
                    True
                    if simplified_indexer
                    else getattr(self.config, "dsa_sparse_attention_use_gather", False)
                ),
            )
            output[query_start:query_end] = request_output

            if diagnostic_queries:
                tp_group = self.indexer.pg_collection.tp
                assert_tp_support_consistent(
                    topk_indices.index_select(1, diagnostic_rows),
                    tp_group,
                    "model",
                )
                assert_tp_support_consistent(
                    diagnostic_topk_indices,
                    tp_group,
                    "expanded",
                )
                for diagnostic_idx, query_metadata in enumerate(diagnostic_queries):
                    local_idx = query_metadata["local_index"]
                    metrics = compute_dsa_attention_diagnostics(
                        query=request_query[local_idx : local_idx + 1],
                        key=request_key,
                        value=request_value,
                        indexer_support=diagnostic_topk_indices[0, diagnostic_idx],
                        model_support=topk_indices[0, local_idx],
                        softmax_scale=self.softmax_scale,
                        topk_values=diagnostics.topk_values,
                        query_position=query_metadata["position"],
                        tp_group=tp_group,
                        model_output=request_output[local_idx],
                        dump_support_indices=diagnostics.dump_support_indices,
                    )
                    diagnostics.record(
                        {
                            "request_id": request_id,
                            "layer": self.layer_number,
                            "phase": query_metadata["phase"],
                            "offset": query_metadata["offset"],
                            "query_position": query_metadata["position"],
                            "prompt_length": query_metadata["prompt_length"],
                            "context_length": key_length,
                            "model_topk": self.indexer.index_topk,
                            "diagnostic_score_precision": "fp32",
                            "indexer_mode": (
                                "simplified_learned_k"
                                if simplified_learned_k
                                else "simplified_main_k" if simplified_indexer else "standard"
                            ),
                            **metrics,
                        }
                    )

        if q_cursor != inference_context.active_token_count:
            raise RuntimeError(
                f"DSA-GQA dynamic inference consumed {q_cursor} query tokens but context has "
                f"{inference_context.active_token_count} active tokens."
            )

        if is_using_quantization_scales(self.config):
            output[inference_context.padding_slice] = 0.0

        return output


class DSGroupedSelfAttention(SelfAttention):
    """Self-attention that swaps in token-level DSA for grouped-query attention."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.padding,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        pp_layer_offset: Optional[int] = None,
        # Upstream's TransformerLayer now passes a module instance name top-down.
        name: str | None = None,
    ):
        if config.experimental_attention_variant == "dsa":
            submodules = copy.copy(submodules)
            dense_core_attention = submodules.core_attention
            if getattr(config, "dsa_indexer_mode", "standard") == "simplified":
                indexer_spec = ModuleSpec(
                    module=SimplifiedDSGQAIndexer,
                    submodules=SimplifiedDSGQAIndexerSubmodules(
                        linear_q=ModuleSpec(module=TELinear),
                        linear_k=ModuleSpec(module=TELinear),
                    ),
                )
            else:
                indexer_spec = ModuleSpec(
                    module=DSGQAIndexer,
                    submodules=DSGQAIndexerSubmodules(
                        linear_q=ModuleSpec(module=TELinear),
                        linear_k=ModuleSpec(module=TELinear),
                        k_norm=ModuleSpec(module=TENorm),
                        linear_weights_proj=ModuleSpec(module=TELinear),
                    ),
                )
            submodules.core_attention = ModuleSpec(
                module=DSGQACoreAttention,
                submodules=DSGQAAttentionSubmodules(
                    indexer=indexer_spec,
                    dense_core_attention=dense_core_attention,
                ),
            )
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            pp_layer_offset=pp_layer_offset,
            name=name,
        )

    def _use_indexer_rope(self, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, rotary_pos_cos_sin) -> bool:
        no_rope = (
            self.config.no_rope_freq[self.layer_number - 1] if self.config.no_rope_freq else False
        )
        if no_rope:
            return False

        position_embedding_type = getattr(self.config, "position_embedding_type", None)
        if position_embedding_type is not None:
            return position_embedding_type in ("rope", "yarn")
        return any(
            tensor is not None
            for tensor in (rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, rotary_pos_cos_sin)
        )

    def _get_core_attention_extra_kwargs(
        self,
        hidden_states: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        inference_context,
        rotary_pos_emb,
        rotary_pos_cos,
        rotary_pos_sin,
        rotary_pos_cos_sin,
        attn_mask_type: AttnMaskType,
        packed_seq_params: Optional[PackedSeqParams],
    ) -> dict:
        if self.config.experimental_attention_variant != "dsa":
            return {}
        indexer_input_norm = None
        simplified_indexer = getattr(self.config, "dsa_indexer_mode", "standard") == "simplified"
        normalized_simplified_indexer = _simplified_indexer_uses_main_input_norm(self.config)
        normalized_standard_indexer = (
            not simplified_indexer
            and getattr(self.config, "dsa_standard_indexer_use_main_input_norm", False)
        )
        if (
            (normalized_simplified_indexer or normalized_standard_indexer)
            and not getattr(self.config, "dsa_fwd_skip_dsa", False)
        ):
            indexer_input_norm = _indexer_input_norm_spec(self.linear_qkv, self.config)
        return {
            "hidden_states": hidden_states,
            "use_indexer_rope": self._use_indexer_rope(
                rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, rotary_pos_cos_sin
            ),
            "indexer_input_norm": indexer_input_norm,
        }

    def _dynamic_core_attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        inference_context,
        block_table: torch.Tensor,
        attn_mask_type: AttnMaskType,
        attention_bias: torch.Tensor,
        packed_seq_params: Optional[PackedSeqParams],
        hidden_states: torch.Tensor,
        use_indexer_rope: bool,
        indexer_input_norm: Optional[_DSAIndexerInputNormSpec] = None,
    ) -> torch.Tensor:
        if self.config.experimental_attention_variant != "dsa":
            return super()._dynamic_core_attention_forward(
                query,
                key,
                value,
                attention_mask,
                inference_context,
                block_table,
                attn_mask_type,
                attention_bias,
                packed_seq_params,
                hidden_states=hidden_states,
                use_indexer_rope=use_indexer_rope,
            )
        if packed_seq_params is not None:
            raise NotImplementedError("Packed sequence is not supported for DSA-GQA dynamic inference.")
        if inference_context.using_cuda_graph_this_step():
            raise NotImplementedError("DSA-GQA dynamic inference does not yet support CUDA graphs.")

        provider_layer_number = self.layer_number - self._get_pp_layer_offset_for_inference()
        return self.core_attention.forward_dynamic(
            query=query,
            key_cache=key,
            value_cache=value,
            hidden_states=hidden_states,
            inference_context=inference_context,
            provider_layer_number=provider_layer_number,
            block_table=block_table,
            use_indexer_rope=use_indexer_rope,
            indexer_input_norm=indexer_input_norm,
        )
