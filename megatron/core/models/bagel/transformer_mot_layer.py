# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""Transformer MoT (Mixture of Transformers) Layer implementation.

This module lives in examples/bagel/model and implements transformer layers that
support separate processing for understanding (und) and generation (gen) tokens,
following the MoT architecture. It aligns with megatron.core.transformer.transformer_layer.
"""

import logging
from abc import ABC
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.cuda_graphs import is_graph_capturing
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.module import GraphableMegatronModule, MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    BaseTransformerLayer,
    get_transformer_layer_offset,
)
from megatron.core.utils import (
    deprecate_inference_params,
    get_pg_rank,
    get_pg_size,
    divide,
    log_single_rank,
    nvtx_range_pop,
    nvtx_range_push,
)

from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from .mot_streams import mot_overlap_region

# Import flex_attention for BlockMask support
try:
    from torch.nn.attention.flex_attention import flex_attention
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.functional import scaled_dot_product_attention
    torch._dynamo.config.cache_size_limit = 512
    torch._dynamo.config.accumulated_cache_size_limit = 4096
    flex_attention = torch.compile(flex_attention)
    HAVE_FLEX_ATTENTION = True
except ImportError:
    HAVE_FLEX_ATTENTION = False
    flex_attention = None

try:
    from megatron.core.extensions.transformer_engine import (
        TELinear,
        is_te_min_version,
        set_save_original_input,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False
    TELinear = None
    is_te_min_version = lambda version: False
    set_save_original_input = lambda module: None

logger = logging.getLogger(__name__)



# =============================================================================
# MoTTransformerLayerSubmodules - Configuration for MoT transformer layer
# =============================================================================


@dataclass
class MoTTransformerLayerSubmodules:
    """
    Configuration class for specifying the submodules of a MoT transformer layer.

    This includes separate layernorms and MLPs for understanding and generation tokens.
    """

    # Input layernorm (understanding)
    input_layernorm: Union[ModuleSpec, type] = IdentityOp
    # Input layernorm for generation tokens (MoT)
    input_layernorm_gen: Union[ModuleSpec, type] = IdentityOp

    # Self attention with MoT support
    self_attention: Union[ModuleSpec, type] = IdentityOp
    self_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    # Pre-MLP layernorm (understanding)
    pre_mlp_layernorm: Union[ModuleSpec, type] = IdentityOp
    # Pre-MLP layernorm for generation tokens (MoT)
    pre_mlp_layernorm_gen: Union[ModuleSpec, type] = IdentityOp

    # MLP (understanding)
    mlp: Union[ModuleSpec, type] = IdentityOp
    # MLP for generation tokens (MoT)
    mlp_gen: Union[ModuleSpec, type] = IdentityOp

    mlp_bda: Union[ModuleSpec, type] = IdentityFuncOp

    # Cross attention (optional, for encoder-decoder)
    pre_cross_attn_layernorm: Union[ModuleSpec, type] = IdentityOp
    cross_attention: Union[ModuleSpec, type] = IdentityOp
    cross_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    # Mapping for sharded tensor keys
    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


# =============================================================================
# MoTTransformerLayer - Transformer layer with MoT support
# =============================================================================


class MoTTransformerLayer(GraphableMegatronModule, BaseTransformerLayer):
    """
    A single transformer layer with MoT (Mixture of Transformers) support.

    This layer supports separate processing paths for understanding (und)
    and generation (gen) tokens, with:
    - Separate input layernorms for und/gen
    - Separate pre-MLP layernorms for und/gen
    - Separate MLPs for und/gen
    - Support for freezing und token gradients

    The layer takes input with size [s, b, h] and returns output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MoTTransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
    ):
        """
        Initialize MoTTransformerLayer.

        Args:
            config (TransformerConfig): Configuration for the transformer model.
            submodules (MoTTransformerLayerSubmodules): Submodule specifications.
            layer_number (int): Layer number (1-indexed).
            hidden_dropout (float, optional): Dropout rate for hidden states.
            pg_collection (ProcessGroupCollection, optional): Process group collection.
            vp_stage (int, optional): Virtual pipeline stage.
        """
        super().__init__(config=config, vp_stage=vp_stage)

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp

        self.submodules_config = submodules
        self.layer_number = layer_number + get_transformer_layer_offset(
            self.config, vp_stage, get_pg_rank(pg_collection.pp)
        )
        self.hidden_dropout = config.hidden_dropout if hidden_dropout is None else hidden_dropout

        # MoT specific: freeze understanding token gradients
        self.freeze_und = getattr(config, 'freeze_und', False)

        # MoT specific: overlap und/gen branches on side CUDA streams (training only)
        self.mot_stream_overlap = getattr(config, 'mot_stream_overlap', False)

        # =====================================================================
        # Build modules
        # =====================================================================

        # [Module 1: Input Layernorm] - Separate for und/gen
        self.input_layernorm = build_module(
            submodules.input_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.input_layernorm_gen = build_module(
            submodules.input_layernorm_gen,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        # Attention optional kwargs
        attention_optional_kwargs = {}
        if config.context_parallel_size > 1 and config.cp_comm_type is not None:
            if isinstance(config.cp_comm_type, list):
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type[self.layer_number]
            else:
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type
        attention_optional_kwargs["pg_collection"] = pg_collection

        # [Module 2: SelfAttention] with MoT support
        self.self_attention = build_module(
            submodules.self_attention,
            config=self.config,
            layer_number=self.layer_number,
            **attention_optional_kwargs,
        )

        # [Module 3: BiasDropoutFusion]
        self.self_attn_bda = build_module(submodules.self_attn_bda)

        # [Module 4: Pre-MLP Layernorm] - Separate for und/gen
        self.pre_mlp_layernorm = build_module(
            submodules.pre_mlp_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.pre_mlp_layernorm_gen = build_module(
            submodules.pre_mlp_layernorm_gen,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        # [Module 5: MLP] - Separate for und/gen
        additional_mlp_kwargs = self._get_mlp_kwargs(submodules.mlp, pg_collection)
        self.mlp = build_module(submodules.mlp, config=self.config, **additional_mlp_kwargs)
        if hasattr(self.mlp, 'set_layer_number'):
            self.mlp.set_layer_number(self.layer_number)

        additional_mlp_gen_kwargs = self._get_mlp_kwargs(submodules.mlp_gen, pg_collection)
        self.mlp_gen = build_module(submodules.mlp_gen, config=self.config, **additional_mlp_gen_kwargs)
        if hasattr(self.mlp_gen, 'set_layer_number'):
            self.mlp_gen.set_layer_number(self.layer_number)

        # [Module 6: MLP BiasDropoutFusion]
        self.mlp_bda = build_module(submodules.mlp_bda)

        # [Module 7-9: Cross attention] (optional)
        self.pre_cross_attn_layernorm = build_module(
            submodules.pre_cross_attn_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.cross_attention = build_module(
            submodules.cross_attention,
            config=self.config,
            layer_number=self.layer_number,
            **attention_optional_kwargs,
        )
        self.cross_attn_bda = build_module(submodules.cross_attn_bda, config=self.config)

        # Check if this is a MoE layer (either und or gen path may be MoE)
        try:
            from megatron.core.transformer.moe.moe_layer import MoELayer
            self.is_moe_layer = isinstance(self.mlp, MoELayer) or isinstance(self.mlp_gen, MoELayer)
        except ImportError:
            self.is_moe_layer = False

    def _get_mlp_kwargs(
        self, mlp_spec: Union[ModuleSpec, type], pg_collection: ProcessGroupCollection
    ) -> dict:
        """Get additional kwargs for MLP initialization."""
        additional_mlp_kwargs = {}

        if not isinstance(mlp_spec, ModuleSpec):
            return additional_mlp_kwargs

        try:
            from megatron.core.extensions.transformer_engine import TEFusedMLP
        except ImportError:
            TEFusedMLP = None

        try:
            from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP, TEGroupedMLP
            from megatron.core.transformer.moe.moe_layer import MoELayer
            moe_classes = (MoELayer, GroupedMLP, TEGroupedMLP, SequentialMLP)
        except ImportError:
            moe_classes = ()

        if mlp_spec.module in moe_classes:
            additional_mlp_kwargs["pg_collection"] = pg_collection
        elif mlp_spec.module == MLP:
            additional_mlp_kwargs["tp_group"] = pg_collection.tp
        elif TEFusedMLP is not None and mlp_spec.module == TEFusedMLP:
            additional_mlp_kwargs["tp_group"] = pg_collection.tp

        return additional_mlp_kwargs

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[Any] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        mode: str = "und",
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass through the MoT transformer layer.

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h].
            attention_mask (Tensor, optional): Attention mask.
            packed_und_token_indexes (Tensor, optional): Indexes of understanding tokens.
            packed_gen_token_indexes (Tensor, optional): Indexes of generation tokens.
            context (Tensor, optional): Context for cross-attention.
            context_mask (Tensor, optional): Mask for cross-attention.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            rotary_pos_cos (Tensor, optional): Rotary embedding cosine.
            rotary_pos_sin (Tensor, optional): Rotary embedding sine.
            rotary_pos_cos_sin (Tensor, optional): Combined rotary embedding.
            attention_bias (Tensor, optional): Attention bias.
            inference_context: Inference context for KV cache.
            packed_seq_params (PackedSeqParams, optional): Packed sequence parameters.
            sequence_len_offset (Tensor, optional): Sequence length offset.
            mode (str): Processing mode - "und" or "gen".

        Returns:
            Tuple[Tensor, Optional[Tensor]]: Output hidden states and context.
        """
        if self.training:
            return self._forward_train(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            return self._forward_inference(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                rotary_pos_cos_sin=rotary_pos_cos_sin,
                attention_bias=attention_bias,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                mode=mode,
            )

    def _forward_train(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor],
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Training forward pass with MoT — compact [Lund+Lgen, 1, h] interface.

        Reads Lund/Lgen from packed_seq_params; no scatter-gather or index arrays.
        """
        assert packed_seq_params is not None, "packed_seq_params required for MoT training"
        Lund = packed_seq_params.padded_und_seqlen

        # =====================================================================
        # Input layernorm with MoT (compact slicing)
        # =====================================================================
        residual = hidden_states
        hidden_states = self._apply_input_layernorm_mot(hidden_states, Lund)

        # =====================================================================
        # Self attention
        # =====================================================================
        nvtx_range_push(suffix="self_attention")
        attention_output_with_bias = self.self_attention(
            hidden_states,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
        )
        # print(f"finished self attention for rank {torch.distributed.get_rank()}")
        nvtx_range_pop(suffix="self_attention")

        # Detach und attention output so no gradient flows back through und weights
        if self.freeze_und and Lund > 0:
            if isinstance(attention_output_with_bias, tuple):
                attn_out, attn_bias = attention_output_with_bias
                attn_out = torch.cat([attn_out[:Lund].detach(), attn_out[Lund:]], dim=0)
                attention_output_with_bias = (attn_out, attn_bias)
            else:
                attention_output_with_bias = torch.cat([attention_output_with_bias[:Lund].detach(), attention_output_with_bias[Lund:]], dim=0)

        # Bias-dropout-add
        nvtx_range_push(suffix="self_attn_bda")
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )
        nvtx_range_pop(suffix="self_attn_bda")

        # =====================================================================
        # Cross attention (if applicable)
        # =====================================================================
        residual = hidden_states
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)

        attention_output_with_bias = self.cross_attention(
            pre_cross_attn_layernorm_output,
            attention_mask=context_mask,
            key_value_states=context,
            inference_context=None,
        )

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
            context = attention_output_with_bias["context"]

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # =====================================================================
        # MLP with MoT (compact slicing)
        # =====================================================================
        output = self._forward_mlp_mot(hidden_states, Lund)

        return output, context

    def _forward_inference(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor],
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[Any] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        mode: str = "und",
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Inference forward pass with MoT.

        Uses mode to determine which layernorms and MLPs to use.
        """
        # Choose modules based on mode
        if mode == "gen":
            input_layernorm = self.input_layernorm_gen if not isinstance(self.input_layernorm_gen, IdentityOp) else self.input_layernorm
            pre_mlp_layernorm = self.pre_mlp_layernorm_gen if not isinstance(self.pre_mlp_layernorm_gen, IdentityOp) else self.pre_mlp_layernorm
            mlp = self.mlp_gen if not isinstance(self.mlp_gen, IdentityOp) else self.mlp
        else:
            input_layernorm = self.input_layernorm
            pre_mlp_layernorm = self.pre_mlp_layernorm
            mlp = self.mlp

        # =====================================================================
        # Input layernorm
        # =====================================================================
        residual = hidden_states
        hidden_states = input_layernorm(hidden_states)

        # =====================================================================
        # Self attention
        # =====================================================================
        attention_output_with_bias = self.self_attention(
            hidden_states,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            mode=mode,
        )

        # Bias-dropout-add
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # =====================================================================
        # MLP
        # =====================================================================
        residual = hidden_states
        hidden_states = pre_mlp_layernorm(hidden_states)

        mlp_output_with_bias = mlp(hidden_states)

        with self.bias_dropout_add_exec_handler():
            output = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )

        return output, context

    def _apply_input_layernorm_mot(self, hidden_states: Tensor, Lund: int) -> Tensor:
        """Apply input layernorm with MoT — compact interface.

        Slices hidden_states[:Lund] for und and hidden_states[Lund:] for gen;
        applies separate layernorms and concatenates back.
        """
        local_und_h = hidden_states[:Lund]    # [Lund, 1, h]
        local_gen_h = hidden_states[Lund:]    # [Lgen, 1, h]
        ln_gen = (
            self.input_layernorm_gen
            if not isinstance(self.input_layernorm_gen, IdentityOp)
            else self.input_layernorm
        )

        und_normed = self.input_layernorm(local_und_h)
        gen_normed = ln_gen(local_gen_h)
        return torch.cat([und_normed, gen_normed], dim=0)

    def _forward_mlp_mot(self, hidden_states: Tensor, Lund: int) -> Tensor:
        """Forward through MLP with MoT — compact interface.

        Slices hidden_states[:Lund] for und and hidden_states[Lund:] for gen;
        applies separate pre-MLP layernorms and MLPs; adds residual per branch.
        Returns compact [Lund+Lgen, 1, h].
        """
        residual = hidden_states
        und_h = hidden_states[:Lund]    # [Lund, 1, h]
        gen_h = hidden_states[Lund:]    # [Lgen, 1, h]

        pre_mlp_ln_gen = (
            self.pre_mlp_layernorm_gen
            if not isinstance(self.pre_mlp_layernorm_gen, IdentityOp)
            else self.pre_mlp_layernorm
        )
        mlp_gen = self.mlp_gen if not isinstance(self.mlp_gen, IdentityOp) else self.mlp

        def _run_und():
            x = self.pre_mlp_layernorm(und_h)
            nvtx_range_push(suffix="mlp_und")
            y = self.mlp(x)
            nvtx_range_pop(suffix="mlp_und")
            if isinstance(y, tuple):
                y = y[0]
            if self.freeze_und:
                y = y.detach()
            return y

        def _run_gen():
            x = pre_mlp_ln_gen(gen_h)
            nvtx_range_push(suffix="mlp_gen")
            y = mlp_gen(x)
            nvtx_range_pop(suffix="mlp_gen")
            if isinstance(y, tuple):
                y = y[0]
            return y

        if self.mot_stream_overlap and Lund > 0 and gen_h.shape[0] > 0:
            with mot_overlap_region() as (und_s, gen_s):
                with torch.cuda.stream(und_s):
                    und_mlp_out = _run_und()
                with torch.cuda.stream(gen_s):
                    gen_mlp_out = _run_gen()
        else:
            und_mlp_out = _run_und()
            gen_mlp_out = _run_gen()

        output = torch.cat([und_mlp_out, gen_mlp_out], dim=0)
        output = output + residual

        return output

    def bias_dropout_add_exec_handler(self):
        """Context manager for bias-dropout-add execution."""
        return nullcontext()

    def sharded_state_dict(
        self,
        prefix: str = '',
        sharded_offsets: tuple = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """Generate a sharded state dictionary for distributed checkpointing."""
        sharded_state_dict = {}

        # Get state dict from all submodules
        for name, module in self.named_children():
            if hasattr(module, 'sharded_state_dict'):
                module_sharded_state_dict = module.sharded_state_dict(
                    prefix=f'{prefix}{name}.',
                    sharded_offsets=sharded_offsets,
                    metadata=metadata,
                )
                sharded_state_dict.update(module_sharded_state_dict)

        # Apply key mapping if specified
        if self.submodules_config.sharded_state_dict_keys_map:
            sharded_state_dict = apply_prefix_mapping(
                sharded_state_dict, self.submodules_config.sharded_state_dict_keys_map
            )

        return sharded_state_dict

