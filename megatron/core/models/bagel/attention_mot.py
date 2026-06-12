import torch
import torch.distributed as dist
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_pg_size, divide
from megatron.core.transformer.module import MegatronModule
from megatron.core import tensor_parallel
import logging
logger = logging.getLogger(__name__)

from torch import Tensor
from megatron.core.packed_seq_params import PackedSeqParams
from .mot_packed_seq_params import MoTPackedSeqParams
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb

@dataclass
class SelfAttentionMoTSubmodules:
    """
    Configuration class for specifying the submodules of a MoT self-attention.

    This includes separate QKV projections for understanding and generation tokens.
    """

    # Standard (understanding) projections
    linear_qkv: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None

    # MoT: Generation token projections
    linear_qkv_gen: Union[ModuleSpec, type] = None
    linear_proj_gen: Union[ModuleSpec, type] = None
    q_layernorm_gen: Union[ModuleSpec, type] = None
    k_layernorm_gen: Union[ModuleSpec, type] = None


class SelfAttentionMoT(MegatronModule):
    """
    Self-attention layer with MoT (Mixture of Transformers) support.

    This attention layer supports separate QKV projections for understanding (und)
    and generation (gen) tokens, following the MoT architecture.

    Key features:
    - Separate linear_qkv and linear_proj for und/gen tokens
    - Separate q_layernorm and k_layernorm for und/gen tokens
    - Support for freeze_und to detach und token gradients
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionMoTSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.padding,
        cp_comm_type: str = None,
        pg_collection: ProcessGroupCollection = None,
    ):
        """
        Initialize SelfAttentionMoT.

        Args:
            config (TransformerConfig): Configuration for the transformer model.
            submodules (SelfAttentionMoTSubmodules): Submodule specifications.
            layer_number (int): Layer number in the transformer.
            attn_mask_type (AttnMaskType): Type of attention mask.
            cp_comm_type (str): Context parallel communication type.
            pg_collection (ProcessGroupCollection): Process group collection.
        """
        super().__init__(config=config)

        self.config = config
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        self.attention_type = "self"

        # MoT specific: freeze understanding token gradients
        self.freeze_und = getattr(config, 'freeze_und', False)

        # Projection sizes
        self.query_projection_size = self.config.kv_channels * self.config.num_attention_heads
        self.kv_projection_size = self.config.kv_channels * self.config.num_query_groups

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp
        self.cp_group = getattr(pg_collection, 'cp', None)
        self.cp_size  = self.cp_group.size() if self.cp_group is not None else 1

        # Per attention head and per partition values
        tp_size = get_pg_size(self.pg_collection.tp)
        self.hidden_size_per_attention_head = divide(
            self.query_projection_size, self.config.num_attention_heads
        )
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, tp_size)
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, tp_size)

        assert self.num_attention_heads_per_partition % self.cp_size == 0, (
            f"num_attention_heads_per_partition ({self.num_attention_heads_per_partition}) "
            f"must be divisible by cp_size ({self.cp_size})"
        )

        # Key and value hidden sizes
        self.key_hidden_size = self.hidden_size_per_attention_head
        self.val_hidden_size = self.hidden_size_per_attention_head

        # Core attention
        self.core_attention = build_module(
            submodules.core_attention,
            config=self.config,
            layer_number=self.layer_number,
            attn_mask_type=self.attn_mask_type,
            attention_type=self.attention_type,
            cp_comm_type=cp_comm_type,
            softmax_scale=self.config.softmax_scale,
            pg_collection=self.pg_collection,
        )

        self.checkpoint_core_attention = (
            self.config.recompute_granularity == 'selective'
            and "core_attn" in self.config.recompute_modules
        )

        # QKV projection output dimension (full, pre-TP — passed to ColumnParallelLinear which
        # internally divides by tp_size).
        self.linear_qkv_out_dim = self.query_projection_size + 2 * self.kv_projection_size
        if self.config.attention_output_gate:
            self.linear_qkv_out_dim += self.config.kv_channels * self.config.num_attention_heads
        # Per-TP-rank output dimension actually returned by linear_qkv when gather_output=False.
        # All forward-path reshapes must use this; using the full dim breaks at TP>1.
        self.linear_qkv_out_dim_per_partition = divide(self.linear_qkv_out_dim, tp_size)

        # =====================================================================
        # Standard (understanding) projections
        # =====================================================================
        self.linear_qkv = build_module(
            submodules.linear_qkv,
            self.config.hidden_size,
            self.linear_qkv_out_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='qkv',
            tp_group=self.pg_collection.tp,
        )

        self.linear_proj = build_module(
            submodules.linear_proj,
            self.query_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name='proj',
            tp_group=self.pg_collection.tp,
        )

        # Q/K layer norms for understanding tokens
        if submodules.q_layernorm is not None:
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.q_layernorm = None

        if submodules.k_layernorm is not None:
            self.k_layernorm = build_module(
                submodules.k_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.k_layernorm = None

        # =====================================================================
        # MoT: Generation token projections
        # =====================================================================
        if submodules.linear_qkv_gen is not None:
            self.linear_qkv_gen = build_module(
                submodules.linear_qkv_gen,
                self.config.hidden_size,
                self.linear_qkv_out_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=self.config.add_bias_linear or self.config.add_qkv_bias,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='qkv_gen',
                tp_group=self.pg_collection.tp,
            )
        else:
            self.linear_qkv_gen = None

        if submodules.linear_proj_gen is not None:
            self.linear_proj_gen = build_module(
                submodules.linear_proj_gen,
                self.query_projection_size,
                self.config.hidden_size,
                config=self.config,
                init_method=self.config.output_layer_init_method,
                bias=self.config.add_bias_linear,
                input_is_parallel=True,
                skip_bias_add=True,
                is_expert=False,
                tp_comm_buffer_name='proj_gen',
                tp_group=self.pg_collection.tp,
            )
        else:
            self.linear_proj_gen = None

        # Q/K layer norms for generation tokens
        if submodules.q_layernorm_gen is not None:
            self.q_layernorm_gen = build_module(
                submodules.q_layernorm_gen,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.q_layernorm_gen = None

        if submodules.k_layernorm_gen is not None:
            self.k_layernorm_gen = build_module(
                submodules.k_layernorm_gen,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.k_layernorm_gen = None

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        inference_context: Optional[Any] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[MoTPackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        mode: str = "und",
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass with MoT (separate processing for und/gen tokens).

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h]. s = [und + gen]
            attention_mask (Tensor, optional): Attention mask.
            inference_context: Inference context for KV cache.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            rotary_pos_cos (Tensor, optional): Rotary embedding cosine.
            rotary_pos_sin (Tensor, optional): Rotary embedding sine.
            rotary_pos_cos_sin (Tensor, optional): Combined rotary embedding.
            attention_bias (Tensor, optional): Attention bias.
            packed_seq_params (MoTPackedSeqParams, optional): Packed sequence parameters. contains:
                - packed_und_token_indexes: global indexes of understanding tokens, needed by cp > 1.
                - packed_gen_token_indexes: global indexes of generation tokens, needed by cp > 1.
                - local_und_token_indexes: local indexes of understanding tokens. if cp = 1, global == local.
                - local_gen_token_indexes: local indexes of generation tokens. if cp = 1, global == local.
                - padded_und_seqlen: padded sequence length of understanding tokens in hidden_states.
                - padded_gen_seqlen: padded sequence length of generation tokens in hidden_states.
            sequence_len_offset (Tensor, optional): Sequence length offset.
            mode (str): Processing mode - "und" or "gen".

        Returns:
            Tuple[Tensor, Optional[Tensor]]: Output tensor and optional bias.
        """
        if self.training:
            return self._forward_train(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
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

    def _forward_train(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor],
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Training forward pass with MoT.

        Processes understanding and generation tokens through separate projections.
        """
        seq_len, batch_size, hidden_size = hidden_states.shape
        assert batch_size == 1, "batch_size must be 1 for MoT with packed sequence"
        assert packed_seq_params is not None, "packed_seq_params must be provided for MoT with packed sequence"
        Lund = getattr(packed_seq_params, 'padded_und_seqlen', 0)
        Lgen = getattr(packed_seq_params, 'padded_gen_seqlen', 0)


        # =====================================================================
        # QKV projection with MoT (separate for und/gen)
        # =====================================================================

        # Reshape for token-level indexing: [s, b, h] -> [s*b, h]
        hidden_states_flat = hidden_states.view(-1, hidden_size)

        # Process understanding tokens
        und_qkv = None
        und_hidden = hidden_states_flat[:Lund]
        und_qkv, _ = self.linear_qkv(und_hidden)

        if self.freeze_und:
            und_qkv = und_qkv.detach()


        # Process generation tokens
        gen_qkv = None
        gen_hidden = hidden_states_flat[Lund:]
        if self.linear_qkv_gen is not None:
            gen_qkv, _ = self.linear_qkv_gen(gen_hidden)
        else:
            raise ValueError("linear_qkv_gen is None")

        assert und_qkv is not None or gen_qkv is not None, "und_qkv and gen_qkv cannot be None at the same time"
        parts = [x for x in (und_qkv, gen_qkv) if x is not None]
        qkv_output_flat = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]

        # Reshape back
        qkv_output = qkv_output_flat.view(seq_len, batch_size, self.linear_qkv_out_dim_per_partition)

        # =====================================================================
        # Split QKV and apply layernorms
        # =====================================================================
        query, key, value = self._split_qkv(qkv_output)

        # Apply Q/K layernorms with MoT
        if self.q_layernorm is not None or self.k_layernorm is not None:
            query, key = self._apply_qk_layernorm_mot(
                query, key, Lund, Lgen
            )

        # =====================================================================
        # Apply rotary positional embeddings to query and key
        # =====================================================================
        if rotary_pos_emb is not None:

            if isinstance(rotary_pos_emb, tuple):
                q_pos_emb, k_pos_emb = rotary_pos_emb
            else:
                q_pos_emb = rotary_pos_emb
                k_pos_emb = rotary_pos_emb

            if q_pos_emb is not None:
                query = apply_rotary_pos_emb(
                    query,
                    q_pos_emb,
                    config=self.config,
                )
            if k_pos_emb is not None:
                key = apply_rotary_pos_emb(
                    key,
                    k_pos_emb,
                    config=self.config,
                )


        # # =====================================================================
        # # Core attention (with flex_attention support)
        # # =====================================================================
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query, key, value, attention_mask,
                attn_mask_type=self.attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(
                query, key, value, attention_mask,
                attn_mask_type=self.attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )

        # =====================================================================
        # Output projection with MoT
        # =====================================================================
        output = self._apply_output_projection_mot(
            core_attn_out, Lund, Lgen
        )

        return output

    def _forward_inference(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor],
        inference_context: Optional[Any] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        mode: str = "und",
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Inference forward pass with MoT.

        Uses mode to determine which projections to use.
        """
        # Choose projections based on mode
        if mode == "gen" and self.linear_qkv_gen is not None:
            linear_qkv = self.linear_qkv_gen
            linear_proj = self.linear_proj_gen if self.linear_proj_gen is not None else self.linear_proj
            q_layernorm = self.q_layernorm_gen if self.q_layernorm_gen is not None else self.q_layernorm
            k_layernorm = self.k_layernorm_gen if self.k_layernorm_gen is not None else self.k_layernorm
        else:
            linear_qkv = self.linear_qkv
            linear_proj = self.linear_proj
            q_layernorm = self.q_layernorm
            k_layernorm = self.k_layernorm

        # QKV projection
        qkv_output, _ = linear_qkv(hidden_states)

        # Split QKV
        query, key, value = self._split_qkv(qkv_output)

        # Apply Q/K layernorms
        if q_layernorm is not None:
            query = q_layernorm(query)
        if k_layernorm is not None:
            key = k_layernorm(key)

        # Apply rotary positional embeddings to query and key
        if rotary_pos_emb is not None:
            # rotary_pos_emb can be a tuple (q_pos_emb, k_pos_emb) or a single tensor
            if isinstance(rotary_pos_emb, tuple):
                q_pos_emb, k_pos_emb = rotary_pos_emb
            else:
                q_pos_emb = rotary_pos_emb
                k_pos_emb = rotary_pos_emb

            if q_pos_emb is not None:
                query = apply_rotary_pos_emb(
                    query,
                    q_pos_emb,
                    config=self.config,
                )
            if k_pos_emb is not None:
                key = apply_rotary_pos_emb(
                    key,
                    k_pos_emb,
                    config=self.config,
                )

        # Core attention
        core_attn_out = self.core_attention(
            query, key, value, attention_mask,
            attn_mask_type=self.attn_mask_type,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
        )

        # Output projection
        output, bias = linear_proj(core_attn_out)

        return output, bias

    def _split_qkv(self, qkv: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Split the packed QKV tensor into query, key, and value.

        Uses the same interleaved-by-query-groups layout as attention.py:
        For each query group: [query_heads, key_head, value_head]
        """
        # qkv shape: [s, b, ng * (np/ng + 2) * hn]
        seq_len, batch_size, _ = qkv.shape

        # Calculate dimensions
        num_query_heads_per_group = (
            self.num_attention_heads_per_partition // self.num_query_groups_per_partition
        )
        num_qkv_heads_per_group = num_query_heads_per_group + 2

        # Reshape to [sq, b, ng, (np/ng + 2) * hn] - organized by query groups
        new_tensor_shape = qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            num_qkv_heads_per_group * self.hidden_size_per_attention_head,
        )
        qkv = qkv.view(*new_tensor_shape)

        # Split along the last dimension (dim=3)
        # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
        split_arg_list = [
            num_query_heads_per_group * self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]
        (query, key, value) = torch.split(qkv, split_arg_list, dim=3)

        # Query [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(seq_len, batch_size, -1, self.hidden_size_per_attention_head)
        return query, key, value

    def _apply_qk_layernorm_mot(
        self,
        query: Tensor,
        key: Tensor,
        padded_und_seqlen: int,
        padded_gen_seqlen: int,
    ) -> Tuple[Tensor, Tensor]:
        """Apply Q/K layernorms with MoT (separate for und/gen tokens)."""
        seq_len, batch_size, num_heads, head_dim = query.shape

        # Flatten for token-level indexing
        query_flat = query.view(-1, num_heads, head_dim)
        key_flat = key.view(-1, key.shape[2], head_dim)

        query_out = query_flat.clone()
        key_out = key_flat.clone()

        # Apply layernorms to understanding tokens
        if self.q_layernorm is not None:
            query_out[:padded_und_seqlen] = self.q_layernorm(
                query_flat[:padded_und_seqlen]
            )
        if self.k_layernorm is not None:
            key_out[:padded_und_seqlen] = self.k_layernorm(
                key_flat[:padded_und_seqlen]
            )

        # Apply layernorms to generation tokens
        q_norm = self.q_layernorm_gen if self.q_layernorm_gen is not None else self.q_layernorm
        k_norm = self.k_layernorm_gen if self.k_layernorm_gen is not None else self.k_layernorm

        if q_norm is not None:
            query_out[padded_und_seqlen:] = q_norm(query_flat[padded_und_seqlen:])
        if k_norm is not None:
            key_out[padded_und_seqlen:] = k_norm(key_flat[padded_und_seqlen:])

        # Reshape back
        query = query_out.view(seq_len, batch_size, num_heads, head_dim)
        key = key_out.view(seq_len, batch_size, key.shape[2], head_dim)

        return query, key

    def _apply_output_projection_mot(
        self,
        attn_output: Tensor,
        padded_und_seqlen: int,
        padded_gen_seqlen: int,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Apply output projection with MoT (separate for und/gen tokens)."""
        seq_len, batch_size, hidden_size = attn_output.shape

        # Flatten for token-level indexing
        attn_output_flat = attn_output.view(-1, hidden_size)
        # output_flat = attn_output_flat.new_zeros(attn_output_flat.shape[0], self.config.hidden_size)

        # Process understanding tokens
        und_output = None
        und_output, _ = self.linear_proj(attn_output_flat[:padded_und_seqlen])
        if self.freeze_und:
            und_output = und_output.detach()

        # Process generation tokens
        gen_output = None
        linear_proj = self.linear_proj_gen if self.linear_proj_gen is not None else self.linear_proj
        gen_output, _ = linear_proj(attn_output_flat[padded_und_seqlen:])

        assert und_output is not None or gen_output is not None, "und_output and gen_output cannot be None at the same time"
        parts = [x for x in (und_output, gen_output) if x is not None]
        output_flat = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]

        # Reshape back
        output = output_flat.view(seq_len, batch_size, self.config.hidden_size)

        return output, None

    def _checkpointed_attention_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor],
        attn_mask_type=None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tensor:
        """Checkpoint the attention computation to save memory."""

        def custom_forward(*inputs):
            q, k, v, mask = inputs[:4]
            return self.core_attention(
                q, k, v, mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )

        return tensor_parallel.checkpoint(
            custom_forward, False, query, key, value, attention_mask
        )