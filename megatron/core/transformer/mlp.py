# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import (
    ReplicaId,
    ShardedStateDict,
    ShardedTensorFactory,
)
from megatron.core.fusions.fused_bias_geglu import bias_geglu_impl
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class MLPSubmodules:
    linear_fc1: Union[ModuleSpec, type] = None
    linear_fc2: Union[ModuleSpec, type] = None


class MLP(MegatronModule):
    """
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.


    Returns an output and a bias to be added to the output.
    If config.add_bias_linear is False, the bias returned is None.

    We use the following notation:
     h: hidden size
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        is_expert: bool = False,
        input_size: int = None,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config

        self.input_size = input_size if input_size != None else self.config.hidden_size

        # If this is a gated linear unit we double the output width
        # see https://arxiv.org/pdf/2002.05202.pdf
        ffn_hidden_size = self.config.ffn_hidden_size
        if self.config.gated_linear_unit:
            ffn_hidden_size *= 2

        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            self.input_size,
            ffn_hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name='fc1',
        )

        self.activation_func = self.config.activation_func

        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            self.config.ffn_hidden_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name='fc2',
        )

    def forward(self, hidden_states):
        """Perform the forward pass through the MLP block."""
        # [s, b, 4 * h/p]
        intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)

        if self.config.bias_activation_fusion:
            if self.activation_func == F.gelu:
                if self.config.gated_linear_unit:
                    intermediate_parallel = bias_geglu_impl(intermediate_parallel, bias_parallel)
                else:
                    assert self.config.add_bias_linear is True
                    intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
            elif self.activation_func == F.silu and self.config.gated_linear_unit:
                intermediate_parallel = bias_swiglu_impl(
                    intermediate_parallel,
                    bias_parallel,
                    self.config.activation_func_fp8_input_store,
                )
            else:
                raise ValueError("Only support fusion of gelu and swiglu")
        else:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            if self.config.gated_linear_unit:

                def glu(x):
                    x = torch.chunk(x, 2, dim=-1)
                    return self.config.activation_func(x[0]) * x[1]

                intermediate_parallel = glu(intermediate_parallel)
            else:
                intermediate_parallel = self.activation_func(intermediate_parallel)

        # [s, b, h]
        output, output_bias = self.linear_fc2(intermediate_parallel)

        return output, output_bias

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        sharded_state_dict = {}
        for name, module in self._modules.items():
            sub_sd = module.sharded_state_dict(f'{prefix}{name}.', sharded_offsets, metadata)
            if self.config.gated_linear_unit and name == 'linear_fc1':
                assert f'{prefix}{name}.weight' in sub_sd, sub_sd.keys()
                for k, v in sub_sd.items():
                    if k in (f'{prefix}{name}.weight', f'{prefix}{name}.bias'):
                        sub_sd[k] = apply_swiglu_sharded_factory(v, sharded_offsets)
            sharded_state_dict.update(sub_sd)
        return sharded_state_dict


def apply_swiglu_sharded_factory(original_sh_ten, sharded_offsets):
    # We must split the tensor into 2 parts, each sharded separately.
    # This requires a ShardedTensorFactory which `chunk`s during saving
    # and `cat`s during loading

    swiglu_shard_axis = 0
    prepend_axis_num = len(sharded_offsets)
    original_shape = original_sh_ten.local_shape
    original_numel = int(np.prod(original_shape))
    local_axis_size = original_shape[swiglu_shard_axis]
    assert (
        original_sh_ten.global_offset[swiglu_shard_axis + prepend_axis_num] % local_axis_size == 0
    )
    rank_offset = (
        original_sh_ten.global_offset[swiglu_shard_axis + prepend_axis_num] // local_axis_size
    )
    axis_frag = original_sh_ten.axis_fragmentations[swiglu_shard_axis + prepend_axis_num]

    @torch.no_grad()
    def sh_ten_build_fn(
        key: str, t: torch.Tensor, replica_id: ReplicaId, flattened_range: Optional[slice]
    ):
        offset_w = (swiglu_shard_axis + prepend_axis_num, rank_offset, axis_frag * 2)
        offset_v = (swiglu_shard_axis + prepend_axis_num, rank_offset + axis_frag, axis_frag * 2)
        if flattened_range is None:
            tensor_w, tensor_v = torch.chunk(t, 2, dim=swiglu_shard_axis)
            return [
                ShardedTensor.from_rank_offsets(
                    key,
                    tensor_w,
                    *sharded_offsets,
                    offset_w,
                    replica_id=replica_id,
                    prepend_axis_num=prepend_axis_num,
                ),
                ShardedTensor.from_rank_offsets(
                    key,
                    tensor_v,
                    *sharded_offsets,
                    offset_v,
                    replica_id=replica_id,
                    prepend_axis_num=prepend_axis_num,
                ),
            ]
        else:
            # Here we need to map a slice `t` (`flattened_range` specifies slice start and stop)
            # of the *original* flattened tensor into slices `w` and `v` of chunked
            # and flattened tensor.
            # Example:
            # If original tensor has (16, 5) shape and flattened_range is `slice(8, 64)`,
            # then `t` has shape `(56,)` and we need to create 2 tensors:
            # w: first 32 elements of `t` with flattened_range slice(8, 40)
            # v: last 24 elements of `t` with flattened_range slice(0, 24)
            # Global offsets are the same as in the non-flattened case
            assert t.ndim == 1, (key, t.shape)
            non_flat_local_shape = (original_shape[0] // 2, *original_shape[1:])
            chunk_numel = original_numel // 2
            result = []
            if flattened_range.start < chunk_numel:
                # Non-empty `w` chunk
                tensor_w = t[: chunk_numel - flattened_range.start]
                flattened_range_w = slice(
                    flattened_range.start, min(chunk_numel, flattened_range.stop)
                )
                assert len(tensor_w) == flattened_range_w.stop - flattened_range_w.start
                result.append(
                    ShardedTensor.from_rank_offsets_flat(
                        key,
                        tensor_w,
                        non_flat_local_shape,
                        *sharded_offsets,
                        offset_w,
                        replica_id=replica_id,
                        prepend_axis_num=prepend_axis_num,
                        flattened_range=flattened_range_w,
                    )
                )
            if flattened_range.stop > chunk_numel:
                # Non-empty `v` chunk
                tensor_v = t[-(flattened_range.stop - chunk_numel) :]
                flattened_range_v = slice(
                    max(chunk_numel, flattened_range.start) - chunk_numel,
                    flattened_range.stop - chunk_numel,
                )
                assert len(tensor_v) == flattened_range_v.stop - flattened_range_v.start, (
                    len(tensor_v),
                    flattened_range_v,
                )

                result.append(
                    ShardedTensor.from_rank_offsets_flat(
                        key,
                        tensor_v,
                        non_flat_local_shape,
                        *sharded_offsets,
                        offset_v,
                        replica_id=replica_id,
                        prepend_axis_num=prepend_axis_num,
                        flattened_range=flattened_range_v,
                    )
                )
            assert sum(sh_ten.data.numel() for sh_ten in result) == t.numel(), (result, t.shape)
            return result

    def sh_ten_merge_fn(sub_state_dict):
        with torch.no_grad():
            return torch.cat(sub_state_dict)

    return ShardedTensorFactory(
        original_sh_ten.key,
        original_sh_ten.data,
        sh_ten_build_fn,
        sh_ten_merge_fn,
        original_sh_ten.replica_id,
    )
