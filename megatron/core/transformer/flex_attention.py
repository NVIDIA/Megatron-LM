# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# FlexAttention module for BlockMask-based attention

import math
from typing import Optional, Tuple

import torch
from torch import Tensor

# Import flex_attention for BlockMask support
try:
    from torch.nn.attention.flex_attention import flex_attention as torch_flex_attention
    torch._dynamo.config.cache_size_limit = 512
    torch._dynamo.config.accumulated_cache_size_limit = 4096
    compiled_flex_attention = torch.compile(torch_flex_attention)
    HAVE_FLEX_ATTENTION = True
except ImportError:
    HAVE_FLEX_ATTENTION = False
    compiled_flex_attention = None


from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import divide


class FlexAttention(MegatronModule):
    """
    FlexAttention module for BlockMask-based attention.

    This module uses PyTorch's flex_attention for efficient sparse attention
    with BlockMask. It handles the padding/unpadding logic required when
    the actual sequence length differs from the BlockMask size.

    We use the following notation:
     h: hidden size
     n: number of attention heads
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float = None,
        softmax_scale: float = None,
        cp_comm_type: str = None,
        pg_collection: ProcessGroupCollection = None,
    ):
        """
        Initialize FlexAttention.

        Args:
            config (TransformerConfig): Configuration for the transformer model.
            layer_number (int): Layer number in the transformer.
            attn_mask_type (AttnMaskType): Type of attention mask.
            attention_type (str): Type of attention (e.g., "self", "cross").
            attention_dropout (float, optional): Dropout rate for attention.
            softmax_scale (float, optional): Scale factor for softmax.
            cp_comm_type (str, optional): Context parallel communication type.
            pg_collection (ProcessGroupCollection, optional): Process group collection.
        """
        super().__init__(config=config)

        if not HAVE_FLEX_ATTENTION:
            raise ImportError(
                "FlexAttention requires PyTorch with flex_attention support. "
                "Please upgrade to PyTorch 2.5+ or use DotProductAttention instead."
            )

        self.config: TransformerConfig = config
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type

        projection_size = self.config.kv_channels * self.config.num_attention_heads

        # Per attention head and per partition values.
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp'])
        else:
            assert hasattr(
                pg_collection, 'tp'
            ), "FlexAttention pg_collection must have tp process group"
        self.pg_collection = pg_collection
        self.tp_group = self.pg_collection.tp

        world_size = pg_collection.tp.size()
        self.hidden_size_per_partition = divide(projection_size, world_size)
        self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)

        # Softmax scale
        if softmax_scale is None:
            self.softmax_scale = 1.0 / math.sqrt(self.hidden_size_per_attention_head)
        else:
            self.softmax_scale = softmax_scale

        if self.config.apply_query_key_layer_scaling:
            self.softmax_scale /= self.layer_number

        # Dropout (note: flex_attention doesn't directly support dropout,
        # so we apply it after attention if needed)
        self.attention_dropout = torch.nn.Dropout(
            self.config.attention_dropout if attention_dropout is None else attention_dropout
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType = None,
        attention_bias: Tensor = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tensor:
        """
        Forward pass for FlexAttention.

        Args:
            query (Tensor): Query tensor of shape [seq_len, batch_size, num_heads, head_dim].
            key (Tensor): Key tensor of shape [seq_len, batch_size, num_kv_heads, head_dim].
            value (Tensor): Value tensor of shape [seq_len, batch_size, num_kv_heads, head_dim].
            attention_mask (Tensor): BlockMask for attention.
            attn_mask_type (AttnMaskType, optional): Attention mask type (unused).
            attention_bias (Tensor, optional): Attention bias (not supported).
            packed_seq_params (PackedSeqParams, optional): Packed sequence parameters (not supported).

        Returns:
            Tensor: Output tensor of shape [seq_len, batch_size, hidden_size].
        """
        assert attention_bias is None, "Attention bias is not supported for FlexAttention."

        # query/key/value shape: [seq_len, batch_size, num_heads, head_dim]
        # flex_attention expects: [batch, num_heads, seq_len, head_dim]
        seq_len, batch_size, num_heads, head_dim = query.shape
        num_kv_heads = key.shape[2]

        # Get the padded sequence length from BlockMask
        # BlockMask shape: (batch, num_heads, padded_seq_len, padded_seq_len)
        padded_seq_len = attention_mask.shape[2]
        pad_size = padded_seq_len - seq_len

        # Transpose to [num_heads, seq_len, head_dim] (squeeze batch dim for padding)
        # Following qwen2_navit.py pattern: permute then pad
        query_flex = query.squeeze(1).permute(1, 0, 2)  # [num_heads, seq_len, head_dim]
        key_flex = key.squeeze(1).permute(1, 0, 2)      # [num_kv_heads, seq_len, head_dim]
        value_flex = value.squeeze(1).permute(1, 0, 2)  # [num_kv_heads, seq_len, head_dim]

        # Pad sequences to match BlockMask size (pad in seq_len dimension)
        if pad_size > 0:
            # Pad: [num_heads, seq_len, head_dim] -> [num_heads, padded_seq_len, head_dim]
            pad_tensor_q = query_flex.new_zeros((num_heads, pad_size, head_dim))
            query_flex = torch.cat([query_flex, pad_tensor_q], dim=1)

            pad_tensor_k = key_flex.new_zeros((num_kv_heads, pad_size, head_dim))
            key_flex = torch.cat([key_flex, pad_tensor_k], dim=1)

            pad_tensor_v = value_flex.new_zeros((num_kv_heads, pad_size, head_dim))
            value_flex = torch.cat([value_flex, pad_tensor_v], dim=1)

        # Call flex_attention with BlockMask
        # flex_attention expects: [batch, num_heads, seq_len, head_dim]
        context = compiled_flex_attention(
            query_flex.unsqueeze(0),   # [1, num_heads, padded_seq_len, head_dim]
            key_flex.unsqueeze(0),     # [1, num_kv_heads, padded_seq_len, head_dim]
            value_flex.unsqueeze(0),   # [1, num_kv_heads, padded_seq_len, head_dim]
            enable_gqa=True,
            block_mask=attention_mask,
        )

        # Remove padding from output: [1, num_heads, padded_seq_len, head_dim] -> [num_heads, seq_len, head_dim]
        end_index = context.shape[2] - pad_size
        context = context[0, :, :end_index, :]  # [num_heads, seq_len, head_dim]

        # Transpose back to [seq_len, batch_size, num_heads, head_dim]
        context = context.permute(1, 0, 2).unsqueeze(1)  # [seq_len, 1, num_heads, head_dim]

        # Apply dropout if needed
        if self.training and self.config.attention_dropout > 0:
            context = self.attention_dropout(context)

        # Reshape to [seq_len, batch_size, hidden_size]
        # Use reshape instead of view since permute may produce non-contiguous tensor
        context = context.reshape(seq_len, batch_size, self.hidden_size_per_partition)

        return context

    def sharded_state_dict(
        self,
        prefix: str = '',
        sharded_offsets: Tuple[Tuple[int, int, int]] = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """Sharded state dict (empty for FlexAttention as it has no learnable parameters)."""
        return {}
