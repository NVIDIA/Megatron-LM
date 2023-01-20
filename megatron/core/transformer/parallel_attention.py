# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.core_attention import CoreAttention
from megatron.core.utils import divide

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.enums import AttnType, AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig


class ParallelAttention(MegatronModule):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int = 1,
        attention_type=AttnType.self_attn,
        attn_mask_type=AttnMaskType.padding,
    ):
        super(ParallelAttention, self).__init__(config)

        self.config = config
        self.hidden_size = config.hidden_size
        self.kv_channels = config.kv_channels
        self.num_attention_heads = config.num_attention_heads
        self.init_method = config.init_method
        self.output_layer_init_method = config.output_layer_init_method
        self.params_dtype = config.params_dtype
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.async_tensor_model_parallel_allreduce = config.async_tensor_model_parallel_allreduce
        self.recompute_granularity = config.recompute_granularity
        self.use_cpu_initialization = config.use_cpu_initialization
        self.perform_initialization = config.perform_initialization
        self.gradient_accumulation_fusion = config.gradient_accumulation_fusion
        self.sequence_parallel_enabled = config.sequence_parallel_enabled

        projection_size = self.kv_channels * self.num_attention_heads

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = divide(projection_size, self.num_attention_heads)
        self.num_attention_heads_per_partition = divide(self.num_attention_heads, world_size)

        # Strided linear layer.
        if attention_type == AttnType.self_attn:
            self.query_key_value = tensor_parallel.ColumnParallelLinear(
                self.hidden_size,
                3 * projection_size,
                gather_output=False,
                init_method=self.init_method,
                async_tensor_model_parallel_allreduce=config.async_tensor_model_parallel_allreduce,
                params_dtype=self.params_dtype,
                use_cpu_initialization=self.use_cpu_initialization,
                perform_initialization=self.perform_initialization,
                gradient_accumulation_fusion=self.gradient_accumulation_fusion,
                sequence_parallel_enabled=self.sequence_parallel_enabled,
            )
        else:
            assert attention_type == AttnType.cross_attn
            self.query = tensor_parallel.ColumnParallelLinear(
                self.hidden_size,
                projection_size,
                gather_output=False,
                init_method=self.init_method,
                async_tensor_model_parallel_allreduce=config.async_tensor_model_parallel_allreduce,
                params_dtype=self.params_dtype,
                use_cpu_initialization=self.use_cpu_initialization,
                perform_initialization=self.perform_initialization,
                gradient_accumulation_fusion=self.gradient_accumulation_fusion,
                sequence_parallel_enabled=self.sequence_parallel_enabled,
            )

            self.key_value = tensor_parallel.ColumnParallelLinear(
                self.hidden_size,
                2 * projection_size,
                gather_output=False,
                init_method=self.init_method,
                async_tensor_model_parallel_allreduce=self.async_tensor_model_parallel_allreduce,
                params_dtype=self.params_dtype,
                use_cpu_initialization=self.use_cpu_initialization,
                perform_initialization=self.perform_initialization,
                gradient_accumulation_fusion=self.gradient_accumulation_fusion,
                sequence_parallel_enabled=self.sequence_parallel_enabled,
            )

        self.core_attention = CoreAttention(
            config=self.config, layer_number=self.layer_number, attn_mask_type=self.attn_mask_type
        )
        self.checkpoint_core_attention = self.recompute_granularity == 'selective'

        # Output.
        self.dense = tensor_parallel.RowParallelLinear(
            projection_size,
            self.hidden_size,
            input_is_parallel=True,
            init_method=self.output_layer_init_method,
            skip_bias_add=True,
            params_dtype=self.params_dtype,
            use_cpu_initialization=self.use_cpu_initialization,
            perform_initialization=self.perform_initialization,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
        )

    def _checkpointed_attention_forward(self, query_layer, key_layer, value_layer, attention_mask):
        """Forward method with selective activation checkpointing."""

        def custom_forward(*inputs):
            query_layer = inputs[0]
            key_layer = inputs[1]
            value_layer = inputs[2]
            attention_mask = inputs[3]
            output_ = self.core_attention(query_layer, key_layer, value_layer, attention_mask)
            return output_

        hidden_states = tensor_parallel.checkpoint(
            custom_forward, False, query_layer, key_layer, value_layer, attention_mask
        )

        return hidden_states

    def _allocate_memory(self, inference_max_sequence_len, batch_size):
        return torch.empty(
            inference_max_sequence_len,
            batch_size,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
            dtype=self.params_dtype,
            device=torch.cuda.current_device(),
        )

    def forward(self, hidden_states, attention_mask, encoder_output=None, inference_params=None):
        # hidden_states: [sq, b, h]

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        # @jcasper how should we do inference_params?
        # can do 1. args, 2. add inference params to TransformerConfig
        # 3. create another config object 4. something else?
        if inference_params:
            if self.layer_number not in inference_params.key_value_memory_dict:
                inf_max_seq_len = inference_params.max_sequence_len
                inf_max_batch_size = inference_params.max_batch_size
                inference_key_memory = self._allocate_memory(inf_max_seq_len, inf_max_batch_size)
                inference_value_memory = self._allocate_memory(inf_max_seq_len, inf_max_batch_size)
                inference_params.key_value_memory_dict[self.layer_number] = (
                    inference_key_memory,
                    inference_value_memory,
                )
            else:
                inference_key_memory, inference_value_memory = inference_params.key_value_memory_dict[
                    self.layer_number
                ]

        # =====================
        # Query, Key, and Value
        # =====================

        if self.attention_type == AttnType.self_attn:
            # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
            mixed_x_layer, _ = self.query_key_value(hidden_states)

            # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
            new_tensor_shape = mixed_x_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                3 * self.hidden_size_per_attention_head,
            )
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer, key_layer, value_layer) = tensor_parallel.split_tensor_along_last_dim(mixed_x_layer, 3)
        else:
            # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
            mixed_kv_layer, _ = self.key_value(encoder_output)

            # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
            new_tensor_shape = mixed_kv_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                2 * self.hidden_size_per_attention_head,
            )
            mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

            # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
            (key_layer, value_layer) = tensor_parallel.split_tensor_along_last_dim(mixed_kv_layer, 2)

            # Attention head [sq, b, h] --> [sq, b, hp]
            query_layer, _ = self.query(hidden_states)
            # [sq, b, hp] --> [sq, b, np, hn]
            new_tensor_shape = query_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )
            query_layer = query_layer.view(*new_tensor_shape)

        # ==================================
        # Adjust key and value for inference
        # ==================================

        if inference_params:
            batch_start = inference_params.batch_size_offset
            batch_end = batch_start + key_layer.size(1)
            assert batch_end <= inference_key_memory.size(1)
            sequence_start = inference_params.sequence_len_offset
            sequence_end = sequence_start + key_layer.size(0)
            assert sequence_end <= inference_key_memory.size(0)
            # Copy key and values.
            inference_key_memory[sequence_start:sequence_end, batch_start:batch_end, ...] = key_layer
            inference_value_memory[sequence_start:sequence_end, batch_start:batch_end, ...] = value_layer
            key_layer = inference_key_memory[:sequence_end, batch_start:batch_end, ...]
            value_layer = inference_value_memory[:sequence_end, batch_start:batch_end, ...]

        # ==================================
        # core attention computation
        # ==================================

        if self.checkpoint_core_attention:
            context_layer = self._checkpointed_attention_forward(query_layer, key_layer, value_layer, attention_mask)
        else:
            context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)

        return output, bias
