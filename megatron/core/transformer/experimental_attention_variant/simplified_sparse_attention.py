# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Simplified sparse attention with learned, detached-input BF16 indexer projections."""

import copy
import math
from functools import partial
from typing import Optional

import torch

from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.fp8_utils import get_fp8_disabled_context
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import accepts_parameter

from .simplified_sparse_attention_cudnn import (
    bind_cudnn_sparse_attention,
    run_dense_teacher,
    run_sparse_attention,
)


class SimplifiedSparseIndexer(MegatronModule):
    """One learned query/key head without RoPE, weights, or a scoring activation.

    Indexer projections stay BF16 even inside a quantized backbone. Their input
    is detached so the auxiliary objective cannot update the backbone or norm.
    """

    def __init__(self, config: TransformerConfig, pg_collection: ProcessGroupCollection) -> None:
        super().__init__(config=config)
        self.tp_group = pg_collection.tp
        indexer_config = copy.copy(config)
        indexer_config.fp8 = indexer_config.fp4 = None
        indexer_config.fp8_param = indexer_config.fp4_param = False
        indexer_config.quant_recipe = None
        indexer_config.params_dtype = torch.bfloat16
        with get_fp8_disabled_context(config, is_init=True):
            for name in ("linear_q", "linear_k"):
                projection = TELinear(
                    config.hidden_size,
                    128,
                    config=indexer_config,
                    init_method=config.init_method,
                    bias=False,
                    skip_bias_add=False,
                    skip_weight_param_allocation=False,
                    parallel_mode="duplicated",
                )
                self.add_module(name, projection)

    def forward(self, normalized_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project the already-normalized QKV input to [S,B,1,D] indexer tensors."""
        normalized_input = normalized_input.detach()
        with get_fp8_disabled_context(self.config):
            query, _ = self.linear_q(normalized_input)
            key, _ = self.linear_k(normalized_input)
        return query.unsqueeze(2), key.unsqueeze(2)


class SimplifiedSparseAttentionCore(MegatronModule):
    """Causal top-k sparse attention and its indexer objective.

    With ``ssa_train_indexer_only`` (phase 1), a frozen dense attention teacher replaces sparse
    attention and the indexer learns from it over every causal key.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        pg_collection: ProcessGroupCollection,
        softmax_scale: Optional[float] = None,
        cp_comm_type: Optional[str] = None,
    ) -> None:
        super().__init__(config=config)
        self.tp_group = pg_collection.tp
        if attention_type != "self" or attn_mask_type != AttnMaskType.causal:
            raise ValueError("Simplified sparse attention requires causal self-attention.")
        if (
            config.num_attention_heads not in (16, 32, 96)
            or config.num_query_groups != 1
            or config.kv_channels != 256
        ):
            raise ValueError(
                "Simplified sparse attention requires 16, 32 or 96 query heads, one K/V head, "
                "and head dimension 256."
            )
        if config.attention_dropout != 0 or config.window_size is not None:
            raise ValueError(
                "Simplified sparse attention does not support attention dropout or "
                "sliding-window attention."
            )
        if config.softmax_type != "vanilla":
            raise ValueError("Simplified sparse attention requires vanilla attention softmax.")
        if config.apply_query_key_layer_scaling:
            raise ValueError(
                "Simplified sparse attention does not support query/key layer scaling."
            )
        if config.qk_clip or config.log_max_attention_logit:
            raise ValueError(
                "Simplified sparse attention does not expose logits for QK clipping or logit "
                "logging."
            )
        if pg_collection.tp.size() != 1 or pg_collection.cp.size() != 1:
            raise NotImplementedError("Simplified sparse attention currently requires TP1 and CP1.")
        if softmax_scale is not None and not math.isclose(
            softmax_scale, 256**-0.5, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("Simplified sparse attention requires softmax scale 1/sqrt(256).")
        self.layer_number = layer_number
        self.kernels = bind_cudnn_sparse_attention(dense_teacher=config.ssa_train_indexer_only)
        self.indexer = SimplifiedSparseIndexer(config, pg_collection)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        attn_mask_type: Optional[AttnMaskType] = None,
        attention_bias: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        *,
        indexer_input: torch.Tensor,
    ) -> torch.Tensor:
        """Return attention output with the scaled indexer loss attached to backward."""
        if attention_mask is not None or attention_bias is not None:
            raise NotImplementedError(
                "Simplified sparse attention does not support custom masks or bias."
            )
        if attn_mask_type not in (None, AttnMaskType.causal):
            raise NotImplementedError("Simplified sparse attention requires causal attention.")
        if packed_seq_params is not None:
            raise NotImplementedError(
                "Simplified sparse attention currently requires unpadded SBHD."
            )

        train_indexer = self.training and torch.is_grad_enabled()
        loss_coeff = self.config.dsa_indexer_loss_coeff if train_indexer else 0.0
        with torch.set_grad_enabled(train_indexer):
            query_indexer, key_indexer = self.indexer(indexer_input)
        if self.config.ssa_train_indexer_only:
            output, indexer_loss = run_dense_teacher(
                self.kernels,
                query,
                key,
                value,
                query_indexer,
                key_indexer,
                loss_coeff=loss_coeff,
                calculate_per_token_loss=self.config.calculate_per_token_loss,
            )
        else:
            output, indexer_loss = run_sparse_attention(
                self.kernels,
                query,
                key,
                value,
                query_indexer,
                key_indexer,
                topk=self.config.dsa_indexer_topk,
                loss_coeff=loss_coeff,
                calculate_per_token_loss=self.config.calculate_per_token_loss,
            )
        if train_indexer:
            logged_loss = indexer_loss.detach()
            if self.config.calculate_per_token_loss:
                logged_loss = logged_loss / query.shape[0]
            DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                loss=logged_loss,
                layer_number=self.layer_number,
                num_layers=self.config.num_layers,
                raw_loss=logged_loss / loss_coeff if loss_coeff > 0 else logged_loss,
            )
            output = DSAIndexerLossAutoScaler.apply(output, indexer_loss)
        return output


class SimplifiedSparseAttention(SelfAttention):
    """Reuse ordinary QKV/output projections around the simplified sparse core."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.causal,
        cp_comm_type: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        pp_layer_offset: Optional[int] = None,
        is_mtp_layer: bool = False,
        name: Optional[str] = None,
    ) -> None:
        if config.experimental_attention_variant != "ssa":
            raise ValueError(
                "SimplifiedSparseAttention requires experimental_attention_variant='ssa'."
            )
        if not accepts_parameter(submodules.linear_qkv, "return_layernorm_output"):
            raise ValueError(
                "Simplified sparse attention reuses the normalized QKV input and requires a "
                "fused-norm linear_qkv such as TELayerNormColumnParallelLinear."
            )
        submodules = copy.copy(submodules)
        submodules.linear_qkv = partial(submodules.linear_qkv, return_layernorm_output=True)
        submodules.core_attention = SimplifiedSparseAttentionCore
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            pp_layer_offset=pp_layer_offset,
            is_mtp_layer=is_mtp_layer,
            name=name,
        )

    def forward_pre_attn_and_core_attn(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        key_value_states: Optional[torch.Tensor] = None,
        inference_context=None,
        *args,
        inference_params=None,
        **kwargs,
    ) -> torch.Tensor:
        """Reject decode caching before entering the inherited inference paths."""
        if inference_context is not None or inference_params is not None:
            raise NotImplementedError(
                "Simplified sparse attention does not support decode caching."
            )
        return super().forward_pre_attn_and_core_attn(
            hidden_states, attention_mask, key_value_states, None, *args, **kwargs
        )

    def _needs_qkv_layernorm_output(self) -> bool:
        return True

    def _get_core_attention_extra_kwargs(
        self, *, qkv_layernorm_output: Optional[torch.Tensor] = None, **kwargs
    ) -> dict[str, torch.Tensor]:
        if qkv_layernorm_output is None:
            raise RuntimeError("Simplified sparse attention requires the normalized QKV input.")
        return {"indexer_input": qkv_layernorm_output}
