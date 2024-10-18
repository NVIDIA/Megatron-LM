# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import itertools
from copy import deepcopy
from functools import partial
from math import ceil
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import (
    LocalNonpersistentObject,
    ReplicaId,
    ShardedStateDict,
    ShardedTensorFactory,
)
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.fusions.fused_bias_geglu import bias_geglu_impl
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl
from megatron.core.jit import jit_fuser
from megatron.core.tensor_parallel.layers import (
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu,
)
from megatron.core.tensor_parallel.utils import divide
from megatron.core.transformer.mlp import MLP, MLPSubmodules, apply_swiglu_sharded_factory
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe import grouped_gemm_util as gg
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import make_sharded_object_for_checkpoint


class GroupedMLP(MegatronModule):
    """An efficient implementation of the Experts layer using GroupedGEMM.

    Executes multiple experts in parallel to maximize computational efficiency.
    """

    def __init__(self, num_local_experts: int, config: TransformerConfig):
        super().__init__(config=config)
        self.config: TransformerConfig = config
        self.num_local_experts = num_local_experts
        gg.assert_grouped_gemm_is_available()
        assert (
            config.add_bias_linear == False
        ), "bias not supported in Grouped GEMM yet, please set '--disable-bias-linear' instead."

        self.expert_parallel = config.expert_model_parallel_size > 1
        if self.config.gated_linear_unit:
            if self.config.activation_func not in (F.silu, F.gelu):
                raise ValueError("Activation function must be silu or gelu when using GroupedMLP.")

            @jit_fuser
            def glu(x):
                x = torch.chunk(x, 2, dim=-1)
                return self.config.activation_func(x[0]) * x[1]

            self.activation_func = glu
        else:
            self.activation_func = self.config.activation_func

        # How many feature each rank holds for fc1 and fc2, respectively.
        self.moe_extended_tp = config.moe_extended_tp
        if config.moe_extended_tp:
            tp_size = parallel_state.get_tensor_and_expert_parallel_world_size()
        else:
            tp_size = parallel_state.get_tensor_model_parallel_world_size()

        fc1_output_size = self.config.ffn_hidden_size * self.num_local_experts
        if config.gated_linear_unit:
            # Project to 4h. If using swiglu double the output width,
            # see https://arxiv.org/pdf/2002.05202.pdf
            fc1_output_size *= 2
        fc1_output_size_per_partition = divide(fc1_output_size, tp_size)

        fc2_input_size = self.config.ffn_hidden_size * self.num_local_experts
        fc2_input_size_per_partition = divide(fc2_input_size, tp_size)

        # Note: The current kernel implementations of grouped_gemm
        # does not support transposition with CUTLASS grouped GEMM
        # (https://github.com/fanshiqing/grouped_gemm/blob/main/csrc/grouped_gemm.cu#L355-L358)
        # and as a result we avoid allocate the transpose of weights.
        # Initialize weight.
        if config.use_cpu_initialization:
            self.weight1 = Parameter(
                torch.empty(
                    self.config.hidden_size,
                    fc1_output_size_per_partition,
                    dtype=config.params_dtype,
                )
            )
            self.weight2 = Parameter(
                torch.empty(
                    fc2_input_size_per_partition, self.config.hidden_size, dtype=config.params_dtype
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_cpu(
                    self.weight1,
                    self.config.hidden_size,
                    fc1_output_size,
                    fc1_output_size_per_partition,
                    partition_dim=1,
                    init_method=config.init_method,
                    params_dtype=config.params_dtype,
                )
                _initialize_affine_weight_cpu(
                    self.weight2,
                    fc2_input_size,
                    self.config.hidden_size,
                    fc2_input_size_per_partition,
                    partition_dim=0,
                    init_method=config.output_layer_init_method,
                    params_dtype=config.params_dtype,
                )
        else:
            self.weight1 = Parameter(
                torch.empty(
                    self.config.hidden_size,
                    fc1_output_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            self.weight2 = Parameter(
                torch.empty(
                    fc2_input_size_per_partition,
                    self.config.hidden_size,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(
                    self.weight1,
                    config.init_method,
                    partition_dim=1,
                    expert_parallel=self.expert_parallel,
                )
                _initialize_affine_weight_gpu(
                    self.weight2,
                    config.output_layer_init_method,
                    partition_dim=0,
                    expert_parallel=self.expert_parallel,
                )
        setattr(self.weight1, 'allreduce', not self.expert_parallel)
        setattr(self.weight2, 'allreduce', not self.expert_parallel)

        def remove_extra_states_check(self, incompatible_keys):
            """
            Remove _extra_state from unexpected keys.
            These keys are for dist ckpt compatibility with SequentialMLP.
            """
            keys = deepcopy(incompatible_keys.unexpected_keys)
            for key in keys:
                if '_extra_state' in key:
                    incompatible_keys.unexpected_keys.remove(key)

        self.register_load_state_dict_post_hook(remove_extra_states_check)

    def forward(self, permuted_local_hidden_states: torch.Tensor, tokens_per_expert: torch.Tensor):
        """Forward step of the GroupedMLP."""
        if permuted_local_hidden_states.nelement() != 0:
            # Reshape the weights for the grouped GEMMs.
            w1 = self.weight1.view(self.num_local_experts, self.config.hidden_size, -1)
            w2 = self.weight2.view(self.num_local_experts, -1, self.config.hidden_size)

            fc1_output = gg.ops.gmm(
                permuted_local_hidden_states, w1, tokens_per_expert, trans_b=False
            )

            intermediate_parallel = self.activation_func(fc1_output)

            fc2_output = gg.ops.gmm(intermediate_parallel, w2, tokens_per_expert, trans_b=False)
        else:
            # No token is allocated for local experts.
            assert torch.count_nonzero(tokens_per_expert) == 0

            # Make sure params of experts still have gradients even given zero tokens.
            w1 = self.weight1.view(self.config.hidden_size, -1)
            w2 = self.weight2.view(-1, self.config.hidden_size)
            h = torch.matmul(permuted_local_hidden_states, w1)
            h = self.activation_func(h)
            h = torch.matmul(h, w2)

            fc2_output = h

        return fc2_output, None

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """
        Maps local expert to global experts.
        The sharded_state_dict for the weight parts are compatible with the SequentialMLP,
        whereas the optimizer states are not due to the limitation from weight transposing.
        That is, for finetuning scenario, the checkpoint is compatible with the SequentialMLP.
        """
        if self.moe_extended_tp:
            raise NotImplementedError(
                'Currently distributed checkpointing is not supported for moe_extended_tp'
            )

        sharded_state_dict = {}
        num_global_experts = (
            parallel_state.get_expert_model_parallel_world_size() * self.num_local_experts
        )
        local_expert_indices_offset = (
            parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
        )
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()

        prepend_axis_num = len(sharded_offsets)
        replica_id = (
            0,
            0,
            parallel_state.get_data_modulo_expert_parallel_rank(with_context_parallel=True),
        )

        local_ffn_dim_size = (
            self.weight2.numel() // self.num_local_experts // self.config.hidden_size
        )

        @torch.no_grad()
        def sh_ten_build_fn(
            key: str,
            t: torch.Tensor,
            replica_id: ReplicaId,
            flattened_range: Optional[slice],
            tp_axis: int,
            with_glu: bool,
        ):
            # TODO: write a generic implementation to cover both cases with and without GLU
            if tp_axis == 1:
                # weight1
                if with_glu:
                    last_dim_size = local_ffn_dim_size * 2
                else:
                    last_dim_size = local_ffn_dim_size
                real_shape = (self.num_local_experts, self.config.hidden_size, last_dim_size)
            elif tp_axis == 0:
                # weight2
                real_shape = (self.num_local_experts, local_ffn_dim_size, self.config.hidden_size)
                assert with_glu == False
            else:
                raise ValueError("tp_axis should be 0 or 1.")
            if flattened_range is None:
                # weights
                t = t.view(real_shape).transpose(-1, -2)
                # change tp_axis due to the transposing
                tp_axis = 1 - tp_axis
                if with_glu:
                    local_tensors = torch.chunk(t, 2, -2)
                    sub_states = [
                        ShardedTensor.from_rank_offsets(
                            key,
                            local_tensors[0].contiguous(),
                            *sharded_offsets,
                            (
                                prepend_axis_num,
                                parallel_state.get_expert_model_parallel_rank(),
                                parallel_state.get_expert_model_parallel_world_size(),
                            ),
                            (prepend_axis_num + 1, tp_rank, tp_size * 2),
                            replica_id=replica_id,
                            prepend_axis_num=prepend_axis_num,
                        ),
                        ShardedTensor.from_rank_offsets(
                            key,
                            local_tensors[1].contiguous(),
                            *sharded_offsets,
                            (
                                prepend_axis_num,
                                parallel_state.get_expert_model_parallel_rank(),
                                parallel_state.get_expert_model_parallel_world_size(),
                            ),
                            (prepend_axis_num + 1, tp_size + tp_rank, tp_size * 2),
                            replica_id=replica_id,
                            prepend_axis_num=prepend_axis_num,
                        ),
                    ]
                else:
                    sub_states = ShardedTensor.from_rank_offsets(
                        key,
                        t.contiguous(),
                        *sharded_offsets,
                        (
                            prepend_axis_num,
                            parallel_state.get_expert_model_parallel_rank(),
                            parallel_state.get_expert_model_parallel_world_size(),
                        ),
                        (prepend_axis_num + 1 + tp_axis, tp_rank, tp_size),
                        replica_id=replica_id,
                        prepend_axis_num=prepend_axis_num,
                    )
            else:
                # flattened optmizer states
                # the non-flattened weight shape is [local_expert_num, hidden_size, ffn_size]
                #
                # For the case without GLU, it is straightforward, we just need to split each
                # expert along the dim-0.
                #
                # For the case with GLU, we need to split the experts along dim-0 and split the
                # two tensors for GLU along dim-2.
                # To split along the non-first dim, we need to chunk the tensor into small pieces,
                # since they belong to different tenors and are interleaved in the flattened space.
                # Refer to the below sketch graph.
                # |................|           |........|........|
                # |............FFFF|           |........|....BBBB|
                # |FFFFFFFFFFFFFFFF|     ->    |AAAAAAAA|BBBBBBBB|
                # |FFFFFFFFFFFFFFFF|           |AAAAAAAA|BBBBBBBB|
                # |FF..............|           |AA......|........|
                # |................|           |........|........|
                #
                # But too many chunks have severe performance issues. We merge these chunks during
                # the save process along with some length information and recover them during the
                # load process.
                assert t.ndim == 1, (key, t.shape)
                if with_glu:
                    non_flat_local_shape = (1, self.config.hidden_size, local_ffn_dim_size)
                    chunk_numel = local_ffn_dim_size
                    sub_states = []
                    start_pos = 0
                    for local_expert_idx in range(self.num_local_experts):
                        first_glu_idx = -1
                        w_start_range = -1
                        v_start_range = -1
                        w_tensors = []
                        v_tensors = []
                        w_lens = []
                        v_lens = []
                        for input_dim_idx in range(self.config.hidden_size):
                            for glu_idx in range(2):
                                local_idx = (
                                    local_expert_idx * self.config.hidden_size * 2
                                    + input_dim_idx * 2
                                    + glu_idx
                                )
                                if (
                                    flattened_range.start < chunk_numel * (local_idx + 1)
                                    and flattened_range.stop > chunk_numel * local_idx
                                ):
                                    if first_glu_idx == -1:
                                        first_glu_idx = glu_idx
                                    end_pos = min(
                                        flattened_range.stop,
                                        chunk_numel * (local_idx + 1) - flattened_range.start,
                                    )
                                    local_tensor = t[start_pos:end_pos]
                                    local_flattened_range = slice(
                                        max(0, flattened_range.start - chunk_numel * local_idx),
                                        min(
                                            chunk_numel,
                                            flattened_range.stop - chunk_numel * local_idx,
                                        ),
                                    )
                                    assert (
                                        len(local_tensor)
                                        == local_flattened_range.stop - local_flattened_range.start
                                    )
                                    start_pos += len(local_tensor)
                                    expert_global_idx = (
                                        local_expert_indices_offset + local_expert_idx
                                    )
                                    if glu_idx == 0:
                                        w_tensors.append(local_tensor)
                                        w_lens.append(len(local_tensor))
                                        if w_start_range == -1:
                                            w_start_range = max(
                                                0, flattened_range.start - chunk_numel * local_idx
                                            )
                                    else:
                                        v_tensors.append(local_tensor)
                                        v_lens.append(len(local_tensor))
                                        if v_start_range == -1:
                                            v_start_range = max(
                                                0, flattened_range.start - chunk_numel * local_idx
                                            )
                        sub_states.append(
                            {
                                'w_tensors': ShardedTensor.from_rank_offsets_flat(
                                    key,
                                    (
                                        torch.cat(w_tensors, -1)
                                        if len(w_tensors) > 0
                                        else torch.Tensor()
                                    ),
                                    non_flat_local_shape,
                                    *sharded_offsets,
                                    (prepend_axis_num, expert_global_idx, num_global_experts),
                                    (prepend_axis_num + 1 + tp_axis, tp_rank, tp_size * 2),
                                    replica_id=replica_id,
                                    prepend_axis_num=prepend_axis_num,
                                    flattened_range=slice(
                                        w_start_range, w_start_range + sum(w_lens)
                                    ),
                                ),
                                'w_lens': LocalNonpersistentObject(w_lens),
                                'v_tensors': ShardedTensor.from_rank_offsets_flat(
                                    key,
                                    (
                                        torch.cat(v_tensors, -1)
                                        if len(v_tensors) > 0
                                        else torch.Tensor()
                                    ),
                                    non_flat_local_shape,
                                    *sharded_offsets,
                                    (prepend_axis_num, expert_global_idx, num_global_experts),
                                    (
                                        prepend_axis_num + 1 + tp_axis,
                                        tp_rank + tp_size,
                                        tp_size * 2,
                                    ),
                                    replica_id=replica_id,
                                    prepend_axis_num=prepend_axis_num,
                                    flattened_range=slice(
                                        v_start_range, v_start_range + sum(v_lens)
                                    ),
                                ),
                                'v_lens': LocalNonpersistentObject(v_lens),
                                'first_glu_idx': LocalNonpersistentObject(first_glu_idx),
                            }
                        )
                else:
                    non_flat_local_shape = (
                        real_shape[0] // self.num_local_experts,
                        *real_shape[1:],
                    )
                    chunk_numel = local_ffn_dim_size * self.config.hidden_size
                    sub_states = []
                    start_pos = 0
                    for local_expert_idx in range(self.num_local_experts):
                        if (
                            flattened_range.start < chunk_numel * (local_expert_idx + 1)
                            and flattened_range.stop > chunk_numel * local_expert_idx
                        ):
                            end_pos = min(
                                flattened_range.stop,
                                chunk_numel * (local_expert_idx + 1) - flattened_range.start,
                            )
                            local_tensor = t[start_pos:end_pos]
                            local_flattened_range = slice(
                                max(0, flattened_range.start - chunk_numel * local_expert_idx),
                                min(
                                    chunk_numel,
                                    flattened_range.stop - chunk_numel * local_expert_idx,
                                ),
                            )
                            assert (
                                len(local_tensor)
                                == local_flattened_range.stop - local_flattened_range.start
                            )
                            start_pos += len(local_tensor)
                            expert_global_idx = local_expert_indices_offset + local_expert_idx
                            sub_states.append(
                                ShardedTensor.from_rank_offsets_flat(
                                    key,
                                    local_tensor,
                                    non_flat_local_shape,
                                    *sharded_offsets,
                                    (prepend_axis_num, expert_global_idx, num_global_experts),
                                    (prepend_axis_num + 1 + tp_axis, tp_rank, tp_size),
                                    replica_id=replica_id,
                                    prepend_axis_num=prepend_axis_num,
                                    flattened_range=local_flattened_range,
                                )
                            )
            return sub_states

        @torch.no_grad()
        def sh_ten_merge_fn(sub_state_dict, tp_axis: int, with_glu: bool):
            if tp_axis == 1:
                # weight1
                weight_shape = (self.config.hidden_size, -1)
            elif tp_axis == 0:
                # weight2
                weight_shape = (-1, self.config.hidden_size)
                assert with_glu == False
            else:
                raise ValueError("tp_axis should be 0 or 1.")
            if isinstance(sub_state_dict, list) and isinstance(sub_state_dict[0], dict):
                # flattened tensor with glu
                res = []
                for local_expert_dict in sub_state_dict:
                    w_tensors = torch.split(
                        local_expert_dict['w_tensors'], local_expert_dict['w_lens']
                    )
                    v_tensors = torch.split(
                        local_expert_dict['v_tensors'], local_expert_dict['v_lens']
                    )
                    first_glu_idx = local_expert_dict['first_glu_idx']
                    if first_glu_idx == 0:
                        res += [
                            x for x in itertools.chain(*itertools.zip_longest(w_tensors, v_tensors))
                        ]
                    else:
                        res += [
                            x for x in itertools.chain(*itertools.zip_longest(v_tensors, w_tensors))
                        ]
                return torch.cat(res)
            elif isinstance(sub_state_dict, list) and sub_state_dict[0].ndim == 1:
                # flattened tensor without glu
                return torch.cat(sub_state_dict)
            else:
                if with_glu:
                    sub_state_dict = torch.cat(sub_state_dict, -2)
                return sub_state_dict.transpose(-1, -2).reshape(weight_shape)

        state_dict = self.state_dict(prefix='', keep_vars=True)
        for name, tensor in state_dict.items():
            if name == 'weight1':
                tp_axis = 1
                with_glu = self.config.gated_linear_unit
                wkey = f'{prefix}experts.linear_fc1.weight'
            else:
                tp_axis = 0
                with_glu = False
                wkey = f'{prefix}experts.linear_fc2.weight'
            sharded_state_dict[f'{prefix}{name}'] = ShardedTensorFactory(
                wkey,
                tensor,
                partial(sh_ten_build_fn, tp_axis=tp_axis, with_glu=with_glu),
                partial(sh_ten_merge_fn, tp_axis=tp_axis, with_glu=with_glu),
                replica_id,
            )

        replica_id = (
            0,
            parallel_state.get_tensor_model_parallel_rank(),
            parallel_state.get_data_modulo_expert_parallel_rank(with_context_parallel=True),
        )
        # Add fake _extra_state to be compatible with SequentialMLP
        for expert_local_idx in range(self.num_local_experts):
            expert_global_idx = local_expert_indices_offset + expert_local_idx
            expert_sharded_offsets = (
                *sharded_offsets,
                (len(sharded_offsets), expert_global_idx, num_global_experts),
            )
            for mod in ['linear_fc1', 'linear_fc2']:
                sharded_state_dict[f'{prefix}expert{expert_global_idx}.{mod}._extra_state'] = (
                    make_sharded_object_for_checkpoint(
                        None,
                        f'{prefix}experts.{mod}._extra_state',
                        expert_sharded_offsets,
                        replica_id,
                    )
                )

        return sharded_state_dict


class TEGroupedMLP(MegatronModule):
    """An efficient implementation of the Experts layer using TE's GroupedLinear.

    Executes multiple experts in parallel to maximize computational efficiency.
    """

    def __init__(self, num_local_experts, config: TransformerConfig, submodules: MLPSubmodules):
        super().__init__(config=config)
        self.moe_extended_tp = config.moe_extended_tp
        self.num_local_experts = num_local_experts
        self.input_size = self.config.hidden_size

        # Double the output width with gated linear unit, see https://arxiv.org/pdf/2002.05202.pdf
        ffn_hidden_size = self.config.ffn_hidden_size
        if self.config.gated_linear_unit:
            ffn_hidden_size *= 2

        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            self.num_local_experts,
            self.input_size,
            ffn_hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=True,
            tp_comm_buffer_name='fc1',
        )

        self.activation_func = self.config.activation_func

        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            self.num_local_experts,
            self.config.ffn_hidden_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=True,
            tp_comm_buffer_name='fc2',
        )

        def remove_extra_states_check(self, incompatible_keys):
            """
            Remove extra _extra_state from unexpected keys.
            These keys are for dist ckpt compatibility with SequentialMLP.
            """
            keys = deepcopy(incompatible_keys.unexpected_keys)
            for key in keys:
                if '_extra_state' in key:
                    incompatible_keys.unexpected_keys.remove(key)

        self.register_load_state_dict_post_hook(remove_extra_states_check)

    def forward(
        self, permuted_local_hidden_states: torch.Tensor, tokens_per_expert: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward of TEGroupedMLP

        Args:
            permuted_local_hidden_states (torch.Tensor): The permuted input hidden states of the
            local experts.
            tokens_per_expert (torch.Tensor): The number of tokens per expert.

        Return:
            output (torch.Tensor): The output of the local experts.
        """
        tokens_per_expert = tokens_per_expert.tolist()
        intermediate_parallel, bias_parallel = self.linear_fc1(
            permuted_local_hidden_states, tokens_per_expert
        )

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

        output, output_bias = self.linear_fc2(intermediate_parallel, tokens_per_expert)

        return output, output_bias

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """
        Maps local expert to global experts.
        The sharded state dict is interchangable with SequentialMLP's.
        """
        if self.moe_extended_tp:
            raise NotImplementedError(
                'Currently distributed checkpointing is not supported for moe_extended_tp'
            )
        sharded_state_dict = {}
        for name, module in self._modules.items():
            sub_sd = module.sharded_state_dict(f'{name}.', sharded_offsets, metadata)
            if name == 'linear_fc1' and self.config.gated_linear_unit:
                num_global_experts = (
                    parallel_state.get_expert_model_parallel_world_size() * self.num_local_experts
                )
                local_expert_indices_offset = (
                    parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
                )
                ep_axis = len(sharded_offsets)
                for i in range(self.num_local_experts):
                    new_sharded_offsets = (
                        *sharded_offsets,
                        (ep_axis, local_expert_indices_offset + i, num_global_experts),
                    )
                    for k in (f'{name}.weight{i}', f'{name}.bias{i}'):
                        if k in sub_sd:
                            sub_sd[k] = apply_swiglu_sharded_factory(sub_sd[k], new_sharded_offsets)
            # Add prefix here to match sequential's keys
            replace_prefix_for_sharding(sub_sd, f'{name}.', f'{prefix}experts.{name}.')
            sharded_state_dict.update({f"{prefix}{k}": v for k, v in sub_sd.items()})
        return sharded_state_dict


class SequentialMLP(MegatronModule):
    """An implementation of the Experts layer using a sequence of MLP layers.

    This class executes each expert sequentially.
    """

    def __init__(self, num_local_experts, config: TransformerConfig, submodules: MLPSubmodules):
        super().__init__(config=config)
        self.add_bias = config.add_bias_linear
        self.moe_extended_tp = config.moe_extended_tp
        self.num_local_experts = num_local_experts
        self.local_experts = torch.nn.ModuleList()
        for _ in range(self.num_local_experts):
            expert = MLP(self.config, submodules, is_expert=True)
            self.local_experts.append(expert)

    def _pad_tensor_for_fp8(self, hidden):
        """Padding tensor shape to multiples of 16."""
        actual_num_tokens = hidden.shape[0]
        divisor = 16
        padded_num_tokens = ceil(actual_num_tokens / divisor) * divisor - actual_num_tokens
        if padded_num_tokens > 0:
            pad_tensor = torch.zeros(
                padded_num_tokens, hidden.shape[1], dtype=hidden.dtype, device=hidden.device
            )
            hidden = torch.cat((hidden, pad_tensor), dim=0)
        return hidden

    def forward(self, permuted_local_hidden_states: torch.Tensor, tokens_per_expert: torch.Tensor):
        """Forward step of the SequentialMLP."""
        if self.num_local_experts == 1:
            if self.config.fp8:
                hidden = self._pad_tensor_for_fp8(permuted_local_hidden_states)
                output, output_bias = self.local_experts[0](hidden)
                output = output[: permuted_local_hidden_states.shape[0]]
            else:
                output, output_bias = self.local_experts[0](permuted_local_hidden_states)

            return output, output_bias
        else:
            tokens_per_expert = tokens_per_expert.tolist()
            tokens_list = torch.split(permuted_local_hidden_states, tokens_per_expert)

            output_local_list = []
            output_bias_list = []

            for expert, tokens in zip(self.local_experts, tokens_list):
                if self.config.fp8:
                    hidden = self._pad_tensor_for_fp8(tokens)
                    output, output_bias = expert(hidden)
                    output = output[: tokens.shape[0]]
                else:
                    output, output_bias = expert(tokens)
                output_local_list.append(output)
                if self.add_bias:
                    output_bias_list.append(output_bias.expand_as(output))

            output_local = torch.cat(output_local_list, dim=0)
            if self.add_bias:
                output_bias_local = torch.cat(output_bias_list, dim=0)
            else:
                output_bias_local = None

            return output_local, output_bias_local

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Maps local expert to global experts."""
        if self.moe_extended_tp:
            raise NotImplementedError(
                'Currently distributed checkpointing is not supported for moe_extended_tp'
            )

        sharded_state_dict = {}
        num_global_experts = (
            parallel_state.get_expert_model_parallel_world_size() * self.num_local_experts
        )
        local_expert_indices_offset = (
            parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
        )

        expert_sharded_prefix = f'{prefix}experts.'
        for expert_local_idx, expert in enumerate(self.local_experts):
            expert_global_idx = local_expert_indices_offset + expert_local_idx
            expert_state_dict_prefix = f'{prefix}local_experts.{expert_local_idx}.'
            expert_sharded_offsets = (
                *sharded_offsets,
                (len(sharded_offsets), expert_global_idx, num_global_experts),
            )

            expert_state_dict = expert.sharded_state_dict(
                expert_state_dict_prefix, expert_sharded_offsets, metadata
            )
            # Remove expert layers indexing from sharded keys
            replace_prefix_for_sharding(
                expert_state_dict, expert_state_dict_prefix, expert_sharded_prefix
            )
            # Adjust replica ids - replication along DP modulo EP
            for k, sh_ten in expert_state_dict.items():
                replica_id = sh_ten.replica_id
                assert (
                    len(replica_id) == 3
                ), f'Expected replica_id for {k} to be in (PP, TP, DP) format, got: {replica_id}'
                sh_ten.replica_id = (
                    *replica_id[:2],
                    parallel_state.get_data_modulo_expert_parallel_rank(with_context_parallel=True),
                )

            sharded_state_dict.update(expert_state_dict)
        return sharded_state_dict
