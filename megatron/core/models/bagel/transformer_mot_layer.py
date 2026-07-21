# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""Transformer MoT (Mixture of Transformers) Layer implementation.

This module lives in examples/bagel/model and implements transformer layers that
support separate processing for understanding (und) and generation (gen) tokens,
following the MoT architecture. It aligns with megatron.core.transformer.transformer_layer.
"""

import functools
import inspect
import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import Tensor

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.module import GraphableMegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    BaseTransformerLayer,
    MlpBuilder,
    get_transformer_layer_offset,
)
from megatron.core.utils import get_pg_rank, nvtx_range_pop, nvtx_range_push

from .alignment_audit import (
    audit_branch_tensor,
    audit_compact_tensor,
    audit_mlp_linears,
    layer_alignment_audit_enabled,
)
from .mot_streams import mot_overlap_region
from .mot_utils import attach_zero_grad_dependency

# Import flex_attention for BlockMask support
try:
    from torch.nn.attention.flex_attention import flex_attention

    torch._dynamo.config.cache_size_limit = 512
    torch._dynamo.config.accumulated_cache_size_limit = 4096
    flex_attention = torch.compile(flex_attention)
    HAVE_FLEX_ATTENTION = True
except ImportError:
    HAVE_FLEX_ATTENTION = False
    flex_attention = None

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
    mlp: Union[ModuleSpec, MlpBuilder, type] = IdentityOp
    # MLP for generation tokens (MoT)
    mlp_gen: Union[ModuleSpec, MlpBuilder, type] = IdentityOp

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
        self.mlp = self._build_mlp(submodules.mlp, pg_collection)
        if hasattr(self.mlp, 'set_layer_number'):
            self.mlp.set_layer_number(self.layer_number)

        self.mlp_gen = self._build_mlp(submodules.mlp_gen, pg_collection)
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

            effective_mlp_gen = (
                self.mlp_gen if not isinstance(self.mlp_gen, IdentityOp) else self.mlp
            )
            self._mlp_is_moe = isinstance(self.mlp, MoELayer)
            self._mlp_gen_is_moe = isinstance(effective_mlp_gen, MoELayer)
            self.is_moe_layer = self._mlp_is_moe or self._mlp_gen_is_moe
        except ImportError:
            self._mlp_is_moe = False
            self._mlp_gen_is_moe = False
            self.is_moe_layer = False

    def _build_mlp(
        self,
        mlp_spec: Union[ModuleSpec, MlpBuilder, type],
        pg_collection: ProcessGroupCollection,
    ):
        """Build an MLP through the TransformerLayer builder protocol."""
        try:
            from megatron.core.extensions.transformer_engine import TEFusedMLP
        except ImportError:
            TEFusedMLP = None

        try:
            from megatron.core.transformer.moe.moe_layer import MoELayer
        except ImportError:
            MoELayer = None

        builder = mlp_spec
        if isinstance(mlp_spec, ModuleSpec):
            module = mlp_spec.module
            if module is MLP or (TEFusedMLP is not None and module is TEFusedMLP):
                builder = functools.partial(
                    module.as_mlp_submodule,
                    submodules=mlp_spec.submodules,
                    **mlp_spec.params,
                )
            elif MoELayer is not None and module is MoELayer:
                builder_kwargs = dict(mlp_spec.params)
                if mlp_spec.submodules is not None:
                    builder_kwargs["submodules"] = mlp_spec.submodules
                builder = functools.partial(module, **builder_kwargs)
            else:
                return build_module(mlp_spec, config=self.config)

        kwargs = {
            "config": self.config,
            "pg_collection": pg_collection,
            "is_mtp_layer": False,
        }
        try:
            signature = inspect.signature(builder)
        except (TypeError, ValueError):
            signature = None

        if signature is not None and "layer_number" in signature.parameters:
            kwargs["layer_number"] = self.layer_number

        return builder(**kwargs)

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
        alignment_audit = layer_alignment_audit_enabled(self.layer_number)
        if alignment_audit:
            audit_compact_tensor(
                "layer_input",
                hidden_states,
                Lund,
                layer_number=self.layer_number,
            )

        # =====================================================================
        # Input layernorm with MoT (compact slicing)
        # =====================================================================
        residual = hidden_states
        hidden_states = self._apply_input_layernorm_mot(hidden_states, Lund)
        if alignment_audit:
            audit_compact_tensor(
                "input_norm",
                hidden_states,
                Lund,
                layer_number=self.layer_number,
            )

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
        if alignment_audit:
            projected_attention = (
                attention_output_with_bias[0]
                if isinstance(attention_output_with_bias, tuple)
                else attention_output_with_bias
            )
            audit_compact_tensor(
                "attention.o_proj",
                projected_attention,
                Lund,
                layer_number=self.layer_number,
            )

        # Detach und attention output so no gradient flows back through und weights
        if self.freeze_und and Lund > 0:
            if isinstance(attention_output_with_bias, tuple):
                attn_out, attn_bias = attention_output_with_bias
                attn_out = torch.cat([attn_out[:Lund].detach(), attn_out[Lund:]], dim=0)
                attention_output_with_bias = (attn_out, attn_bias)
            else:
                attention_output_with_bias = torch.cat(
                    [attention_output_with_bias[:Lund].detach(), attention_output_with_bias[Lund:]],
                    dim=0,
                )

        # Bias-dropout-add
        nvtx_range_push(suffix="self_attn_bda")
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )
        nvtx_range_pop(suffix="self_attn_bda")
        if alignment_audit:
            audit_compact_tensor(
                "attention.residual",
                hidden_states,
                Lund,
                layer_number=self.layer_number,
            )

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
        if alignment_audit:
            audit_compact_tensor(
                "layer_output",
                output,
                Lund,
                layer_number=self.layer_number,
            )

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
            input_layernorm = (
                self.input_layernorm_gen
                if not isinstance(self.input_layernorm_gen, IdentityOp)
                else self.input_layernorm
            )
            pre_mlp_layernorm = (
                self.pre_mlp_layernorm_gen
                if not isinstance(self.pre_mlp_layernorm_gen, IdentityOp)
                else self.pre_mlp_layernorm
            )
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
        local_und_h = hidden_states[:Lund]  # [Lund, 1, h]
        local_gen_h = hidden_states[Lund:]  # [Lgen, 1, h]
        ln_gen = (
            self.input_layernorm_gen
            if not isinstance(self.input_layernorm_gen, IdentityOp)
            else self.input_layernorm
        )

        if local_und_h.shape[0] > 0:
            und_normed = self.input_layernorm(local_und_h)
        else:
            und_normed = attach_zero_grad_dependency(local_und_h, self.input_layernorm)

        if local_gen_h.shape[0] > 0:
            gen_normed = ln_gen(local_gen_h)
        else:
            gen_normed = attach_zero_grad_dependency(local_gen_h, ln_gen)
        return torch.cat([und_normed, gen_normed], dim=0)

    def _forward_mlp_mot(self, hidden_states: Tensor, Lund: int) -> Tensor:
        """Forward through MLP with MoT — compact interface.

        Slices hidden_states[:Lund] for und and hidden_states[Lund:] for gen;
        applies separate pre-MLP layernorms and MLPs; adds residual per branch.
        Returns compact [Lund+Lgen, 1, h].
        """
        residual = hidden_states
        und_h = hidden_states[:Lund]  # [Lund, 1, h]
        gen_h = hidden_states[Lund:]  # [Lgen, 1, h]

        pre_mlp_ln_gen = (
            self.pre_mlp_layernorm_gen
            if not isinstance(self.pre_mlp_layernorm_gen, IdentityOp)
            else self.pre_mlp_layernorm
        )
        mlp_gen = self.mlp_gen if not isinstance(self.mlp_gen, IdentityOp) else self.mlp
        alignment_audit = layer_alignment_audit_enabled(self.layer_number)

        def _run_und():
            x = (
                self.pre_mlp_layernorm(und_h)
                if und_h.shape[0] > 0
                else attach_zero_grad_dependency(und_h, self.pre_mlp_layernorm)
            )
            if alignment_audit:
                audit_branch_tensor(
                    "mlp.pre_norm",
                    "und",
                    x,
                    layer_number=self.layer_number,
                )
            nvtx_range_push(suffix="mlp_und")
            with audit_mlp_linears(
                self.mlp,
                branch="und",
                layer_number=self.layer_number,
            ):
                y = self.mlp(x)
            nvtx_range_pop(suffix="mlp_und")
            if isinstance(y, tuple):
                y = y[0]
            if und_h.shape[0] == 0:
                y = attach_zero_grad_dependency(y, self.mlp)
            if self.freeze_und:
                y = y.detach()
            return y

        def _run_gen():
            x = (
                pre_mlp_ln_gen(gen_h)
                if gen_h.shape[0] > 0
                else attach_zero_grad_dependency(gen_h, pre_mlp_ln_gen)
            )
            if alignment_audit:
                audit_branch_tensor(
                    "mlp.pre_norm",
                    "gen",
                    x,
                    layer_number=self.layer_number,
                )
            nvtx_range_push(suffix="mlp_gen")
            with audit_mlp_linears(
                mlp_gen,
                branch="gen",
                layer_number=self.layer_number,
            ):
                y = mlp_gen(x)
            nvtx_range_pop(suffix="mlp_gen")
            if isinstance(y, tuple):
                y = y[0]
            if gen_h.shape[0] == 0:
                y = attach_zero_grad_dependency(y, mlp_gen)
            return y

        def _skip_und():
            output = attach_zero_grad_dependency(
                und_h, self.pre_mlp_layernorm, self.mlp
            )
            return output.detach() if self.freeze_und else output

        def _skip_gen():
            return attach_zero_grad_dependency(gen_h, pre_mlp_ln_gen, mlp_gen)

        if self.mot_stream_overlap and Lund > 0 and gen_h.shape[0] > 0:
            with mot_overlap_region() as (und_s, gen_s):
                with torch.cuda.stream(und_s):
                    und_mlp_out = _run_und()
                with torch.cuda.stream(gen_s):
                    gen_mlp_out = _run_gen()
        else:
            und_mlp_out = (
                _run_und()
                if und_h.shape[0] > 0 or getattr(self, '_mlp_is_moe', False)
                else _skip_und()
            )
            gen_mlp_out = (
                _run_gen()
                if gen_h.shape[0] > 0 or getattr(self, '_mlp_gen_is_moe', False)
                else _skip_gen()
            )

        output = torch.cat([und_mlp_out, gen_mlp_out], dim=0)
        output = output + residual

        return output

    def bias_dropout_add_exec_handler(self):
        """Context manager for bias-dropout-add execution."""
        return nullcontext()

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """Generate a sharded state dictionary for distributed checkpointing."""
        sharded_state_dict = {}

        # Get state dict from all submodules
        for name, module in self.named_children():
            if hasattr(module, 'sharded_state_dict'):
                module_sharded_state_dict = module.sharded_state_dict(
                    prefix=f'{prefix}{name}.', sharded_offsets=sharded_offsets, metadata=metadata
                )
                sharded_state_dict.update(module_sharded_state_dict)

        # Apply key mapping if specified
        if self.submodules_config.sharded_state_dict_keys_map:
            sharded_state_dict = apply_prefix_mapping(
                sharded_state_dict, self.submodules_config.sharded_state_dict_keys_map
            )

        return sharded_state_dict
