# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import logging
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from itertools import chain
from math import ceil
from typing import Optional, Protocol, Tuple

import torch
import torch.nn.functional as F

from megatron.core import tensor_parallel
from megatron.core.activations import squared_relu
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.fusions.fused_bias_geglu import quick_gelu, weighted_bias_quick_geglu_impl
from megatron.core.fusions.fused_bias_swiglu import weighted_bias_swiglu_impl
from megatron.core.fusions.fused_weighted_squared_relu import weighted_squared_relu_impl
from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.tensor_parallel.layers import (
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu,
    set_tensor_model_parallel_attributes,
)
from megatron.core.tensor_parallel.utils import divide
from megatron.core.transformer.mlp import (
    MLP,
    MLPSubmodules,
    TEActivationFunctionBuilder,
    apply_swiglu_sharded_factory,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import (
    ProcessGroupCollection,
    get_align_size_for_quantization,
    skip_routed_expert_padding,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    sharded_state_dict_default,
)
from megatron.core.typed_torch import apply_module, not_none

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import Fp8Padding, Fp8Unpadding
else:
    Fp8Padding, Fp8Unpadding = None, None

try:
    import flashinfer.fused_moe as fused_moe
    from flashinfer.fused_moe.core import ActivationType

    HAVE_FLASHINFER = True
except ImportError:
    HAVE_FLASHINFER = False

from megatron.core.inference.moe import ActivationType as McoreActivationType
from megatron.core.inference.moe import (
    InferenceGroupedGemmBackend,
    mcore_fused_moe,
    resolve_inference_grouped_gemm_backend,
)

logger = logging.getLogger(__name__)


class GroupedMLP(MegatronModule):
    """An efficient implementation of the Experts layer using GroupedGEMM.

    Executes multiple experts in parallel to maximize computational efficiency.
    """

    # TODO(M4): breaking api, switched from pass in tp_group to pass in pg_collection.
    def __init__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=config)
        self.config: TransformerConfig = config
        self.num_local_experts = num_local_experts
        gg.assert_grouped_gemm_is_available()
        assert (
            config.add_bias_linear == False
        ), "bias not supported in Grouped GEMM yet, please set '--disable-bias-linear' instead."
        assert (
            config.moe_latent_size is None
        ), "MoE latent projection not supported in GroupedMLP yet."

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
        self.activation_recompute = (
            self.config.recompute_granularity == 'selective'
            and "moe_act" in self.config.recompute_modules
        )
        if self.activation_recompute and (self.config.fp8 or self.config.fp4):
            raise ValueError(
                "moe_act recompute for fp8 or fp4 cannot work with the legacy GroupedMLP."
            )

        @jit_fuser
        def activation_func_with_probs(x, probs):
            dtype = x.dtype
            res = self.activation_func(x) * probs
            return res.to(dtype)

        self.activation_func_with_probs = activation_func_with_probs

        self.ep_group = pg_collection.ep
        # use pg_collection.expt_tp_group as tensor parallel group in this module.
        self.tp_group = pg_collection.expt_tp
        # use pg_collection.expt_dp_group as data parallel group in this module.
        self.dp_group = pg_collection.expt_dp
        # How many feature each rank holds for fc1 and fc2, respectively.
        tp_size = self.tp_group.size()
        tp_rank = self.tp_group.rank()

        fc1_output_size = self.config.moe_ffn_hidden_size * self.num_local_experts
        if config.gated_linear_unit:
            # Project to 4h. If using swiglu double the output width,
            # see https://arxiv.org/pdf/2002.05202.pdf
            fc1_output_size *= 2
        fc1_output_size_per_partition = divide(fc1_output_size, tp_size)

        fc2_input_size = self.config.moe_ffn_hidden_size * self.num_local_experts
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
                    rank=tp_rank,
                    world_size=tp_size,
                )
                _initialize_affine_weight_cpu(
                    self.weight2,
                    fc2_input_size,
                    self.config.hidden_size,
                    fc2_input_size_per_partition,
                    partition_dim=0,
                    init_method=config.output_layer_init_method,
                    params_dtype=config.params_dtype,
                    rank=tp_rank,
                    world_size=tp_size,
                )
            else:
                # Ensure TP attrs are set even when not initializing
                set_tensor_model_parallel_attributes(
                    tensor=self.weight1, is_parallel=True, dim=1, stride=1
                )
                set_tensor_model_parallel_attributes(
                    tensor=self.weight2, is_parallel=True, dim=0, stride=1
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
                    self.weight1, config.init_method, partition_dim=1, is_expert=True
                )
                _initialize_affine_weight_gpu(
                    self.weight2, config.output_layer_init_method, partition_dim=0, is_expert=True
                )
            else:
                # Ensure TP attrs are set even when not initializing
                set_tensor_model_parallel_attributes(
                    tensor=self.weight1, is_parallel=True, dim=1, stride=1
                )
                set_tensor_model_parallel_attributes(
                    tensor=self.weight2, is_parallel=True, dim=0, stride=1
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

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ):
        """Forward step of the GroupedMLP."""
        assert self.config.bf16, "Currently GroupedMLP for MoE only supports bf16."
        if self.activation_recompute:
            self.activation_checkpoint = tensor_parallel.CheckpointWithoutOutput()

        if self.config.moe_apply_probs_on_input:
            assert (
                self.config.moe_router_topk == 1
            ), "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = (
                permuted_probs.unsqueeze(-1) * permuted_local_hidden_states
            )
            permuted_local_hidden_states = permuted_local_hidden_states.to(original_dtype)
            # Probs already applied, so reset to 1.
            permuted_probs = torch.ones_like(permuted_probs)

        if permuted_local_hidden_states.nelement() != 0:
            # Reshape the weights for the grouped GEMMs.
            w1 = self.weight1.view(self.num_local_experts, self.config.hidden_size, -1)
            w2 = self.weight2.view(self.num_local_experts, -1, self.config.hidden_size)

            fc1_output = gg.ops.gmm(
                permuted_local_hidden_states, w1, tokens_per_expert, trans_b=False
            )
            if self.activation_recompute:
                intermediate_parallel = self.activation_checkpoint.checkpoint(
                    self.activation_func_with_probs, fc1_output, permuted_probs.unsqueeze(-1)
                )
                fc2_output = gg.ops.gmm(intermediate_parallel, w2, tokens_per_expert, trans_b=False)
                self.activation_checkpoint.discard_output_and_register_recompute(fc2_output)
            else:
                intermediate_parallel = self.activation_func_with_probs(
                    fc1_output, permuted_probs.unsqueeze(-1)
                )
                fc2_output = gg.ops.gmm(intermediate_parallel, w2, tokens_per_expert, trans_b=False)
        else:
            # No token is allocated for local experts.
            assert torch.count_nonzero(tokens_per_expert) == 0

            # Make sure params of experts still have gradients even given zero tokens.
            w1 = self.weight1.view(self.config.hidden_size, -1)
            w2 = self.weight2.view(-1, self.config.hidden_size)
            h = torch.matmul(permuted_local_hidden_states, w1)
            if self.activation_recompute:
                h = self.activation_checkpoint.checkpoint(
                    self.activation_func_with_probs, h, permuted_probs.unsqueeze(-1)
                )
                fc2_output = torch.matmul(h, w2)
                self.activation_checkpoint.discard_output_and_register_recompute(fc2_output)
            else:
                h = self.activation_func_with_probs(h, permuted_probs.unsqueeze(-1))
                fc2_output = torch.matmul(h, w2)

        return fc2_output, None

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """
        Maps local expert to global experts.
        The sharded_state_dict for the weight parts are compatible with the SequentialMLP,
        whereas the optimizer states are not due to the limitation from weight transposing.
        That is, for finetuning scenario, the checkpoint is compatible with the SequentialMLP.

        When `singleton_local_shards` metadata flag is True, experts are broken down into
        separate tensors and stored under separate global keys. Additionally, similarly to MLP,
        layers with GLU activations are broken down into separate `w` and `v` tensors.
        """
        singleton_local_shards = (metadata or {}).get('singleton_local_shards', False)
        sharded_state_dict = {}
        ep_size = self.ep_group.size()
        ep_rank = self.ep_group.rank()
        tp_size = self.tp_group.size()
        tp_rank = self.tp_group.rank()
        dp_rank = self.dp_group.rank()
        num_global_experts = ep_size * self.num_local_experts
        local_expert_indices_offset = ep_rank * self.num_local_experts

        prepend_axis_num = len(sharded_offsets)
        replica_id = (0, 0, dp_rank)

        local_ffn_dim_size = (
            self.weight2.numel() // self.num_local_experts // self.config.hidden_size
        )

        def _break_into_individual_experts(
            experts_ten: torch.Tensor,
            key: str,
            tp_offset: Tuple[int, int, int],
            replica_id: ReplicaId,
        ):
            """Breaks experts into individual tensors and stores them under separate global keys"""
            experts_state = []
            assert len(experts_ten) == self.num_local_experts, (
                experts_ten.shape,
                self.num_local_experts,
            )
            for local_expert_idx, expert_ten in enumerate(experts_ten):
                global_expert_idx = local_expert_indices_offset + local_expert_idx
                expert_key = key.replace(
                    f'{prefix}experts.', f'{prefix}experts.{global_expert_idx}.'
                )
                experts_state.append(
                    ShardedTensor.from_rank_offsets(
                        expert_key,
                        expert_ten.contiguous(),
                        *sharded_offsets,
                        tp_offset,
                        replica_id=replica_id,
                        prepend_axis_num=prepend_axis_num,
                    )
                )
            return experts_state

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
                    assert tp_axis == 0, tp_axis
                    if singleton_local_shards:
                        w_tensor, v_tensor = torch.chunk(t, 2, -2)
                        w_key = f'{key}_w'
                        v_key = f'{key}_v'
                        sub_states = {
                            'singleton_local_shards': LocalNonpersistentObject(True),
                            'data': {
                                'w': _break_into_individual_experts(
                                    w_tensor,
                                    w_key,
                                    (prepend_axis_num, tp_rank, tp_size),
                                    replica_id,
                                ),
                                'v': _break_into_individual_experts(
                                    v_tensor,
                                    v_key,
                                    (prepend_axis_num, tp_rank, tp_size),
                                    replica_id,
                                ),
                            },
                        }
                    else:
                        local_tensors = torch.chunk(t, 2, -2)
                        sub_states = [
                            ShardedTensor.from_rank_offsets(
                                key,
                                local_tensors[0].contiguous(),
                                *sharded_offsets,
                                (prepend_axis_num, ep_rank, ep_size),
                                (prepend_axis_num + 1, tp_rank, tp_size * 2),
                                replica_id=replica_id,
                                prepend_axis_num=prepend_axis_num,
                            ),
                            ShardedTensor.from_rank_offsets(
                                key,
                                local_tensors[1].contiguous(),
                                *sharded_offsets,
                                (prepend_axis_num, ep_rank, ep_size),
                                (prepend_axis_num + 1, tp_size + tp_rank, tp_size * 2),
                                replica_id=replica_id,
                                prepend_axis_num=prepend_axis_num,
                            ),
                        ]
                else:
                    if singleton_local_shards:
                        sub_states = {
                            'singleton_local_shards': LocalNonpersistentObject(True),
                            'data': _break_into_individual_experts(
                                t, key, (prepend_axis_num + tp_axis, tp_rank, tp_size), replica_id
                            ),
                        }
                    else:
                        sub_states = ShardedTensor.from_rank_offsets(
                            key,
                            t.contiguous(),
                            *sharded_offsets,
                            (prepend_axis_num, ep_rank, ep_size),
                            (prepend_axis_num + 1 + tp_axis, tp_rank, tp_size),
                            replica_id=replica_id,
                            prepend_axis_num=prepend_axis_num,
                        )
            return sub_states  # pylint: disable=possibly-used-before-assignment

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
            if isinstance(sub_state_dict, dict):
                assert sub_state_dict['singleton_local_shards']
                if with_glu:
                    assert isinstance(sub_state_dict['data'], dict)
                    sub_state_dict = torch.cat(
                        (
                            torch.stack(sub_state_dict['data']['w']),
                            torch.stack(sub_state_dict['data']['v']),
                        ),
                        dim=-2,
                    )
                else:
                    assert isinstance(sub_state_dict['data'], list)
                    sub_state_dict = torch.stack(sub_state_dict['data'])
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

            this_replica_id = list(copy.deepcopy(replica_id))

            sharded_state_dict[f'{prefix}{name}'] = ShardedTensorFactory(
                wkey,
                tensor,
                partial(sh_ten_build_fn, tp_axis=tp_axis, with_glu=with_glu),
                partial(sh_ten_merge_fn, tp_axis=tp_axis, with_glu=with_glu),
                tuple(this_replica_id),
            )

        replica_id = (0, tp_rank, dp_rank)
        # Add fake _extra_state to be compatible with SequentialMLP
        for expert_local_idx in range(self.num_local_experts):
            expert_global_idx = local_expert_indices_offset + expert_local_idx
            if singleton_local_shards:
                expert_sharded_offsets = sharded_offsets
            else:
                expert_sharded_offsets = (
                    *sharded_offsets,
                    (len(sharded_offsets), expert_global_idx, num_global_experts),
                )
            for mod in ['linear_fc1', 'linear_fc2']:
                if singleton_local_shards:
                    expert_key = f'{prefix}experts.{expert_global_idx}.{mod}._extra_state'
                else:
                    expert_key = f'{prefix}experts.{mod}._extra_state'
                sharded_state_dict[f'{prefix}expert{expert_global_idx}.{mod}._extra_state'] = (
                    make_sharded_object_for_checkpoint(
                        None, expert_key, expert_sharded_offsets, replica_id
                    )
                )

        return sharded_state_dict

    def backward_dw(self):
        """Performs backward pass for weight gradients in Experts.
        Empty implementation for compatibility with SequentialMLP and TEGroupedMLP.
        """
        pass


class GroupedLinearFc1Interface(Protocol):
    """Interface for linear_fc1 module in TEGroupedMLP."""

    def forward(
        self, permuted_local_hidden_states: torch.Tensor, tokens_per_expert: list[int], /
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward method for linear_fc1 module."""
        ...

    def backward_dw(self) -> None:
        """Backward method for linear_fc1 module."""
        ...


class GroupedLinearFc1Builder(Protocol):
    """Protocol describing how to build a linear_fc1 layer in TEGroupedMLP."""

    def __call__(
        self,
        num_local_experts: int,
        input_size: int,
        output_size: int,
        /,
        *,
        config: TransformerConfig,
        init_method: Callable[[torch.Tensor], None],
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: str | None,
        pg_collection: ProcessGroupCollection | None,
    ) -> GroupedLinearFc1Interface:
        """Builds a linear_fc1 layer for TEGroupedMLP."""
        ...


class GroupedLinearFc2Interface(Protocol):
    """Protocol for linear_fc2 module in TEGroupedMLP."""

    def forward(
        self, intermediate_parallel: torch.Tensor, tokens_per_expert: list[int], /
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward method for linear_fc2 module."""
        ...

    def backward_dw(self) -> None:
        """Backward method for linear_fc2 module."""
        ...


class GroupedLinearFc2Builder(Protocol):
    """Protocol describing how to build a linear_fc2 layer in TEGroupedMLP."""

    def __call__(
        self,
        num_local_experts: int,
        input_size: int,
        output_size: int,
        /,
        *,
        config: TransformerConfig,
        init_method: Callable[[torch.Tensor], None],
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: str | None,
        pg_collection: ProcessGroupCollection | None,
    ) -> GroupedLinearFc2Interface:
        """Builds a linear_fc2 layer for TEGroupedMLP."""
        ...


@dataclass
class GroupedMLPSubmodules:
    """
    The dataclass for ModuleSpecs of TEGroupedMLP submodules
    including  linear fc1, activation function, linear fc2.
    """

    linear_fc1: GroupedLinearFc1Builder

    linear_fc2: GroupedLinearFc2Builder

    activation_func: TEActivationFunctionBuilder | None = None
    """
    Builder for an activation function module; only used if config.use_te_activation_func is True.
    """


class TEGroupedMLP(MegatronModule):
    """An efficient implementation of the Experts layer using TE's GroupedLinear.

    Executes multiple experts in parallel to maximize computational efficiency.
    """

    # TODO(M4): breaking api, switched from pass in tp_group to pass in pg_collection.
    def __init__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        submodules: GroupedMLPSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=config)
        self.num_local_experts = num_local_experts
        self.input_size = self.config.hidden_size
        assert not (
            self.config.add_bias_linear and config.bias_dropout_fusion
        ), "bias_dropout_fusion is not supported in TEGroupedMLP when add_bias_linear=True"

        self.ep_group = pg_collection.ep
        self.tp_group = pg_collection.expt_tp

        # Double the output width with gated linear unit, see https://arxiv.org/pdf/2002.05202.pdf
        ffn_hidden_size = not_none(self.config.moe_ffn_hidden_size)
        if self.config.gated_linear_unit:
            ffn_hidden_size *= 2

        self.linear_fc1 = submodules.linear_fc1(
            self.num_local_experts,
            self.input_size if self.config.moe_latent_size is None else self.config.moe_latent_size,
            ffn_hidden_size,
            config=self.config,
            init_method=not_none(self.config.init_method),
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            is_expert=True,
            tp_comm_buffer_name='fc1',
            pg_collection=pg_collection,
        )

        if self.config.use_te_activation_func and not (submodules.activation_func is None):
            self.activation_func = apply_module(submodules.activation_func(config=self.config))
        else:
            self.activation_func = self.config.activation_func

        self.linear_fc2 = submodules.linear_fc2(
            self.num_local_experts,
            not_none(self.config.moe_ffn_hidden_size),
            (
                self.config.hidden_size
                if self.config.moe_latent_size is None
                else self.config.moe_latent_size
            ),
            config=self.config,
            init_method=not_none(self.config.output_layer_init_method),
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=True,
            tp_comm_buffer_name='fc2',
            pg_collection=pg_collection,
        )

        self.offload_expert_fc1 = (
            self.config.fine_grained_activation_offloading
            and "expert_fc1" in self.config.offload_modules
        )

        self.offload_moe_act = (
            self.config.fine_grained_activation_offloading
            and "moe_act" in self.config.offload_modules
        )

        self.activation_recompute = (
            self.config.recompute_granularity == 'selective'
            and "moe_act" in self.config.recompute_modules
        )
        if self.activation_recompute and (self.config.fp8 or self.config.fp4):
            from megatron.core.extensions.transformer_engine import set_save_original_input

            set_save_original_input(self.linear_fc2)

        # This is to avoid the CPU overhead of multiple d2h copies
        if self.offload_expert_fc1 and not self.config.fp8:
            from megatron.core.extensions.transformer_engine import set_save_original_input

            set_save_original_input(self.linear_fc1)

        # Fused implementation with Transformer Engine op fuser API
        if self.config.use_transformer_engine_op_fuser:
            assert (
                self._is_fused_impl_supported()
            ), "Fused GroupedMLP is not supported for this configuration."
        self._with_fused_impl: bool = self.config.use_transformer_engine_op_fuser
        self._fused_ops: Optional[Tuple[torch.nn.Module]] = None
        if (
            self.config.gated_linear_unit
            and self.config.moe_mlp_glu_interleave_size is not None
            and not self._with_fused_impl
        ):
            logger.warning(
                "`moe_mlp_glu_interleave_size=%s` is enabled, but fused MoE MLP implementation "
                "is not supported for this configuration. The non-fused path may incur extra "
                "tensor reordering/copy overhead each forward pass.",
                self.config.moe_mlp_glu_interleave_size,
            )

        if self.config.fp8 or self.config.fp4:
            assert HAVE_TE, "FP8 and FP4 requires TE."
            align_size = 256 if self._with_fused_impl else None
            self.quantization_padding = Fp8Padding(self.num_local_experts, align_size=align_size)
            self.quantization_unpadding = Fp8Unpadding(
                self.num_local_experts, align_size=align_size
            )

    @staticmethod
    def _apply_bias(intermediate_parallel, bias_parallel, tokens_per_expert, permuted_probs):
        if bias_parallel is None:
            return intermediate_parallel
        shape = intermediate_parallel.shape
        return (
            torch.cat(
                [
                    t + b * p
                    for t, b, p in zip(
                        torch.split(intermediate_parallel.view(-1, shape[-1]), tokens_per_expert),
                        bias_parallel,
                        torch.split(permuted_probs, tokens_per_expert),
                    )
                ]
            )
            .view(shape)
            .to(intermediate_parallel.dtype)
        )

    def _is_fused_impl_supported(self) -> bool:
        """Check if the TE op fuser supports implementing this module."""

        # Check Transformer Engine installation
        if not HAVE_TE:
            return False  # Transformer Engine is not available
        try:
            from transformer_engine.pytorch.ops import GroupedLinear, ScaledSwiGLU
        except ImportError:
            return False  # Transformer Engine version is too old

        # Check for unsupported features
        if self.tp_group.size() > 1:
            return False  # Tensor parallelism is not supported
        if self.offload_expert_fc1 or self.offload_moe_act:
            return False  # Fine-grained activation offloading is not supported
        if self.config.moe_apply_probs_on_input:
            return False  # Pre-multiplying probs is not supported

        # Check grouped linear modules
        if not isinstance(self.linear_fc1, te.pytorch.GroupedLinear):
            return False
        if not isinstance(self.linear_fc2, te.pytorch.GroupedLinear):
            return False
        if self.linear_fc1.need_backward_dw() or self.linear_fc2.need_backward_dw():
            return False  # Delayed weight gradient compuation is not supported

        # Check activation
        if self.activation_func != F.silu or not self.config.gated_linear_unit:
            return False  # Expected SwiGLU activation

        return True

    def _make_fused_ops(self) -> torch.nn.Module:
        """Construct fused module for FC1, activation, and FC2."""

        # Container for fusible ops
        ops = te.pytorch.ops.Sequential()

        # Check if there are 1 or "num_gemms" params in the GroupedLinear module.
        fc1_single_grouped_parameter = self.linear_fc1.single_grouped_parameter
        fc1_weight_dtype = (
            self.linear_fc1.weight.dtype
            if fc1_single_grouped_parameter
            else self.linear_fc1.weight0.dtype
        )
        fc2_single_grouped_parameter = self.linear_fc2.single_grouped_parameter
        fc2_weight_dtype = (
            self.linear_fc2.weight.dtype
            if fc2_single_grouped_parameter
            else self.linear_fc2.weight0.dtype
        )

        # TODO:ksivamani: Why meta device?
        op = te.pytorch.ops.GroupedLinear(
            self.linear_fc1.num_gemms,
            self.linear_fc1.in_features,
            self.linear_fc1.out_features,
            bias=self.linear_fc1.use_bias,
            device=torch.cuda.current_device(),
            dtype=fc1_weight_dtype,
            accumulate_into_main_grad=self.linear_fc1.fuse_wgrad_accumulation,
            single_grouped_parameter=fc1_single_grouped_parameter,
        )

        # Copy the weights from GroupedLinear module to GroupedLinear op.
        if fc1_single_grouped_parameter:
            setattr(op, "weight", getattr(self.linear_fc1, "weight"))

        for idx in range(self.linear_fc1.num_gemms):
            if not fc1_single_grouped_parameter:
                setattr(op, f"weight{idx}", getattr(self.linear_fc1, f"weight{idx}"))
            if self.linear_fc1.use_bias:
                setattr(op, f"bias{idx}", getattr(self.linear_fc1, f"bias{idx}"))
        ops.append(op)

        # Activation and post-multiply probs
        op = te.pytorch.ops.ScaledSwiGLU(
            glu_interleave_size=self.config.moe_mlp_glu_interleave_size
        )
        ops.append(op)

        # FC2
        has_bias = self.linear_fc2.use_bias
        op = te.pytorch.ops.GroupedLinear(
            self.linear_fc2.num_gemms,
            self.linear_fc2.in_features,
            self.linear_fc2.out_features,
            bias=self.linear_fc2.use_bias,
            device=torch.cuda.current_device(),
            dtype=fc2_weight_dtype,
            accumulate_into_main_grad=self.linear_fc2.fuse_wgrad_accumulation,
            single_grouped_parameter=fc2_single_grouped_parameter,
        )

        # Copy the weights from GroupedLinear module to GroupedLinear op.
        if fc2_single_grouped_parameter:
            setattr(op, "weight", getattr(self.linear_fc2, "weight"))

        for idx in range(self.linear_fc2.num_gemms):
            if not fc2_single_grouped_parameter:
                setattr(op, f"weight{idx}", getattr(self.linear_fc2, f"weight{idx}"))
            if self.linear_fc2.use_bias:
                setattr(op, f"bias{idx}", getattr(self.linear_fc2, f"bias{idx}"))
        ops.append(op)

        # Emulate submodule pre-forward hooks
        ops.register_forward_pre_hook(self._make_fused_impl_pre_forward_hook())

        return ops

    def _make_fused_impl_pre_forward_hook(self) -> Callable:
        """Make function that calls submodule pre-forward callback hooks.

        This is intended for compatibility with
        DistributedDataParallel hooks that trigger parameter
        all-gathers. It does not support general pre-forward hooks
        since they may manipulate intermediate tensors that are never
        instantiated by the fused implementation.

        """

        def forward_pre_hook(module, *_) -> None:
            for submodule in chain(self.linear_fc1.modules(), self.linear_fc2.modules()):
                for hook in submodule._forward_pre_hooks.values():
                    # Assume that hook does not interact with input
                    ret = hook(submodule, None)
                    if ret is not None:
                        raise RuntimeError(
                            f"Applying a fused implementation for {self.__class__.__name__}, "
                            f"but a {submodule.__class__.__name__} submodule "
                            "has a pre-forward hook that modifies the input tensor."
                        )

        return forward_pre_hook

    def _fused_forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass using Transformer Engine operation fuser API."""

        # Construct fused impl if needed
        # Note: We initialize during the first forward pass in case
        # the params are modified after the constructor.
        # Note: The fused impl is stored in a tuple to avoid
        # registering submodules.
        if self._fused_ops is None:
            self._fused_ops = (self._make_fused_ops(),)
        (ops,) = self._fused_ops

        # Apply padding if needed
        unpadded_tokens_per_expert = None
        if skip_routed_expert_padding(self.config):
            pass
        elif self.config.fp8 or self.config.fp4:
            tokens_per_expert = tokens_per_expert.tolist()
            unpadded_tokens_per_expert = tokens_per_expert
            permuted_local_hidden_states, tokens_per_expert = self.quantization_padding(
                permuted_local_hidden_states, tokens_per_expert
            )
            permuted_probs, _ = self.quantization_padding(
                permuted_probs.unsqueeze(-1), unpadded_tokens_per_expert
            )
            permuted_probs = permuted_probs.squeeze(-1)
            tokens_per_expert = torch.tensor(
                tokens_per_expert, dtype=torch.int, device=permuted_probs.device
            )

        # Call fused impl
        output = ops(
            permuted_local_hidden_states,
            tokens_per_expert,  # FC1
            permuted_probs,  # Scaled SwiGLU
            tokens_per_expert,  # FC2
        )

        # Remove padding if needed
        if unpadded_tokens_per_expert is not None:
            output = self.quantization_unpadding(output, unpadded_tokens_per_expert)

        return output

    def bias_act_func(self, intermediate_parallel, bias_parallel, permuted_probs):
        """
        Applies bias and activation function to the output of linear_fc1.
        """
        if self.config.use_te_activation_func:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            intermediate_parallel = self.activation_func(intermediate_parallel)
            if permuted_probs is not None:
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = intermediate_parallel * permuted_probs
                intermediate_parallel = intermediate_parallel.to(original_dtype)
        elif self.config.bias_activation_fusion:
            if self.activation_func == F.silu and self.config.gated_linear_unit:
                # dtype is handled inside the fused kernel
                intermediate_parallel = weighted_bias_swiglu_impl(
                    intermediate_parallel,
                    bias_parallel,
                    permuted_probs,
                    self.config.activation_func_fp8_input_store,
                )
            elif self.activation_func == quick_gelu and self.config.gated_linear_unit:
                intermediate_parallel = weighted_bias_quick_geglu_impl(
                    intermediate_parallel,
                    bias_parallel,
                    permuted_probs,
                    self.config.activation_func_fp8_input_store,
                    self.config.glu_linear_offset,
                    self.config.activation_func_clamp_value,
                )
            else:
                raise ValueError("Only support fusion of swiglu and quick_gelu in TEGroupedMLP.")
        elif self.activation_func == squared_relu and self.config.use_fused_weighted_squared_relu:
            assert bias_parallel is None, "Bias is not supported with fused weighted squared relu."
            intermediate_parallel = weighted_squared_relu_impl(
                intermediate_parallel, permuted_probs
            )
        else:
            if self.config.gated_linear_unit:

                def glu(x):
                    x_glu, x_linear = torch.chunk(x, 2, dim=-1)
                    if (val := self.config.activation_func_clamp_value) is not None:
                        x_glu = x_glu.clamp(min=None, max=val)
                        x_linear = x_linear.clamp(min=-val, max=val)
                    return self.config.activation_func(x_glu) * (
                        x_linear + self.config.glu_linear_offset
                    )

                intermediate_parallel = glu(intermediate_parallel)
            else:
                intermediate_parallel = self.activation_func(intermediate_parallel)
            original_dtype = intermediate_parallel.dtype
            intermediate_parallel = intermediate_parallel * permuted_probs
            intermediate_parallel = intermediate_parallel.to(original_dtype)
        return intermediate_parallel

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward of TEGroupedMLP

        Args:
            permuted_local_hidden_states (torch.Tensor): The permuted input hidden states of the
            local experts.
            tokens_per_expert (torch.Tensor): The number of tokens per expert.
            permuted_probs (torch.Tensor): The permuted probs of each token produced by the router.

        Return:
            output (torch.Tensor): The output of the local experts.
        """

        # Call fused impl if enabled
        if self._with_fused_impl:
            output = self._fused_forward(
                permuted_local_hidden_states, tokens_per_expert, permuted_probs
            )
            output_bias = None
            return output, output_bias

        # Apply padding if needed
        unpadded_tokens_per_expert = None
        tokens_per_expert: list[int] = tokens_per_expert.tolist()
        permuted_probs = permuted_probs.unsqueeze(-1)
        if skip_routed_expert_padding(self.config):
            pass
        elif self.config.fp8 or self.config.fp4:
            unpadded_tokens_per_expert = tokens_per_expert
            permuted_local_hidden_states, tokens_per_expert = self.quantization_padding(
                permuted_local_hidden_states, tokens_per_expert
            )
            permuted_probs, _ = self.quantization_padding(
                permuted_probs, unpadded_tokens_per_expert
            )

        if self.config.moe_apply_probs_on_input:
            assert (
                self.config.moe_router_topk == 1
            ), "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = permuted_probs * permuted_local_hidden_states
            permuted_local_hidden_states = permuted_local_hidden_states.to(original_dtype)
            # Probs already applied, so reset to 1.
            permuted_probs = torch.ones_like(permuted_probs)

        expert_fc1_manager = off_interface(
            self.offload_expert_fc1, permuted_local_hidden_states, "expert_fc1"
        )
        with expert_fc1_manager as permuted_local_hidden_states:
            fc1_output, bias_parallel = apply_module(self.linear_fc1)(
                permuted_local_hidden_states, tokens_per_expert
            )
        fc1_output = expert_fc1_manager.group_offload(
            fc1_output,
            forced_released_tensors=[permuted_local_hidden_states],
            delay_offload=self.config.delay_offload_until_cuda_graph,
        )

        def bias_act_func(intermediate_parallel, bias_parallel, permuted_probs):

            # Whether activation function is interleaved GLU
            with_glu_interleaving = (
                self.config.gated_linear_unit
                and self.config.moe_mlp_glu_interleave_size is not None
            )

            def remove_glu_interleaving(x: torch.Tensor) -> torch.Tensor:
                """Reorder tensor so gate and linear units are contiguous.

                Should only be applied if the activation function is
                an interleaved GLU.

                """
                shape = x.size()
                interleave_size = self.config.moe_mlp_glu_interleave_size
                x = x.reshape(-1, shape[-1] // (2 * interleave_size), 2, interleave_size)
                x = x.transpose(1, 2).contiguous()
                x = x.view(shape)
                return x

            if self.config.use_te_activation_func:
                if bias_parallel is not None:
                    intermediate_parallel = intermediate_parallel + bias_parallel
                if with_glu_interleaving:
                    intermediate_parallel = remove_glu_interleaving(intermediate_parallel)
                intermediate_parallel = self.activation_func(intermediate_parallel)
                if permuted_probs is not None:
                    original_dtype = intermediate_parallel.dtype
                    intermediate_parallel = intermediate_parallel * permuted_probs
                    intermediate_parallel = intermediate_parallel.to(original_dtype)
            elif self.config.bias_activation_fusion and not with_glu_interleaving:
                if self.activation_func == F.silu and self.config.gated_linear_unit:
                    # dtype is handled inside the fused kernel
                    intermediate_parallel = weighted_bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        permuted_probs,
                        self.config.activation_func_fp8_input_store,
                    )
                elif self.activation_func == quick_gelu and self.config.gated_linear_unit:
                    intermediate_parallel = weighted_bias_quick_geglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        permuted_probs,
                        self.config.activation_func_fp8_input_store,
                        self.config.glu_linear_offset,
                        self.config.activation_func_clamp_value,
                    )
                else:
                    raise ValueError(
                        "Only support fusion of swiglu and quick_gelu in TEGroupedMLP."
                    )
            elif (
                self.activation_func == squared_relu and self.config.use_fused_weighted_squared_relu
            ):
                assert bias_parallel is None
                intermediate_parallel = weighted_squared_relu_impl(
                    intermediate_parallel, permuted_probs
                )
            else:
                if self.config.gated_linear_unit:

                    def glu(x):
                        if with_glu_interleaving:
                            x = remove_glu_interleaving(x)
                        x_glu, x_linear = torch.chunk(x, 2, dim=-1)
                        if (val := self.config.activation_func_clamp_value) is not None:
                            x_glu = x_glu.clamp(min=None, max=val)
                            x_linear = x_linear.clamp(min=-val, max=val)
                        return self.config.activation_func(x_glu) * (
                            x_linear + self.config.glu_linear_offset
                        )

                    intermediate_parallel = glu(intermediate_parallel)
                else:
                    intermediate_parallel = self.activation_func(intermediate_parallel)
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = intermediate_parallel * permuted_probs
                intermediate_parallel = intermediate_parallel.to(original_dtype)
            return intermediate_parallel

        moe_act_manager = off_interface(self.offload_moe_act, fc1_output, "moe_act")
        if self.activation_recompute:
            self.activation_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            with moe_act_manager as fc1_output:
                bias_act_output = self.activation_checkpoint.checkpoint(
                    self.bias_act_func, fc1_output, bias_parallel, permuted_probs
                )
        else:
            with moe_act_manager as fc1_output:
                bias_act_output = bias_act_func(fc1_output, bias_parallel, permuted_probs)

        output, output_bias = apply_module(self.linear_fc2)(bias_act_output, tokens_per_expert)
        if self.activation_recompute:
            self.activation_checkpoint.discard_output_and_register_recompute(output)

        # Delay the offload of the moe act until after the linear_fc2 has been computed
        # to make sure the fc1_output is reloaded to GPU before recomputing moe_act.
        output = moe_act_manager.group_offload(
            output,
            forced_released_tensors=[fc1_output],
            delay_offload=self.config.delay_offload_until_cuda_graph,
        )
        output = self._apply_bias(output, output_bias, tokens_per_expert, permuted_probs)

        # upad and concat the output
        if unpadded_tokens_per_expert is not None:
            output = self.quantization_unpadding(output, unpadded_tokens_per_expert)

        output_bias = None

        return output, output_bias

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """
        Maps local expert to global experts.
        The sharded state dict is interchangable with SequentialMLP's.
        """
        # Guard for cases metadata is not provided
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        singleton_local_shards = (metadata or {}).get('singleton_local_shards', False)
        sharded_state_dict = {}
        for name, module in self._modules.items():
            sub_sd = sharded_state_dict_default(
                module, f'{name}.', sharded_offsets, metadata, tp_group=self.tp_group
            )
            if name == 'linear_fc1' and self.config.gated_linear_unit:
                num_global_experts = self.ep_group.size() * self.num_local_experts
                local_expert_indices_offset = self.ep_group.rank() * self.num_local_experts
                ep_axis = len(sharded_offsets)
                for i in range(self.num_local_experts):
                    if singleton_local_shards:
                        new_sharded_offsets = sharded_offsets
                    else:
                        new_sharded_offsets = (
                            *sharded_offsets,
                            (ep_axis, local_expert_indices_offset + i, num_global_experts),
                        )
                    for k in (f'{name}.weight{i}', f'{name}.bias{i}'):
                        if k in sub_sd:
                            sub_sd[k] = apply_swiglu_sharded_factory(
                                sub_sd[k], new_sharded_offsets, singleton_local_shards
                            )
            if singleton_local_shards:
                replace_prefix_for_sharding(sub_sd, '', f'{prefix}experts.')
            else:
                # Add prefix here to match sequential's keys
                replace_prefix_for_sharding(sub_sd, f'{name}.', f'{prefix}experts.{name}.')
            sharded_state_dict.update({f"{prefix}{k}": v for k, v in sub_sd.items()})
        return sharded_state_dict

    def backward_dw(self):
        """Performs backward pass for weight gradients in TEGroupedMLP.

        This method executes the backward pass for weight gradients by calling
        backward_dw() on the linear layers in reverse order (fc2 followed by fc1).
        If an error occurs during execution, it is caught and re-raised with a
        descriptive message.
        """
        self.linear_fc2.backward_dw()
        self.linear_fc1.backward_dw()


class InferenceGroupedMLP(TEGroupedMLP):
    """Inference-optimized GroupedMLP with GPU-resident offsets.

    Inherits from TEGroupedMLP to reuse weight initialization and checkpoint compatibility.
    Supports three forward paths:
    - Training: delegates to parent TEGroupedMLP
    - Inference + CUDA graphed: FlashInfer cutlass_fused_moe (fused permute + GEMM)
    - Inference + eager: torch.nn.functional.grouped_mm with GPU-resident cumsum offsets
    """

    def __init__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        submodules: GroupedMLPSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        # Initialize parent TEGroupedMLP (creates linear_fc1, linear_fc2)
        super().__init__(
            num_local_experts=num_local_experts,
            config=config,
            submodules=submodules,
            pg_collection=pg_collection,
        )

        # Concatenated weights are built lazily on first forward to ensure
        # checkpoint loading has already populated the per-expert parameters.
        self._concatenated_weights_built = False

        self.is_inference_cuda_graphed_iteration = False

        if HAVE_FLASHINFER:
            self._flashinfer_activation_type = self._resolve_flashinfer_activation_type()

        self._mcore_activation_type = self._resolve_mcore_activation_type()
        self.inference_grouped_gemm_backend = config.inference_grouped_gemm_backend

    def _resolve_flashinfer_activation_type(self):
        """Map megatron activation config to FlashInfer ActivationType."""
        assert (
            HAVE_FLASHINFER
        ), "flashinfer-python is required to resolve FlashInfer activation type."
        func = self.config.activation_func
        if func == F.silu:
            return ActivationType.Silu
        elif func == F.gelu:
            return ActivationType.Gelu
        elif func == F.relu:
            return ActivationType.Relu
        elif func == squared_relu:
            return ActivationType.Relu2
        raise ValueError(f"No FlashInfer ActivationType mapping for activation_func={func}")

    def _resolve_mcore_activation_type(self):
        """Map megatron activation config to mcore_fused_moe ActivationType."""
        func = self.config.activation_func
        if func == squared_relu:
            return McoreActivationType.SQUARED_RELU
        raise ValueError(f"No mcore_fused_moe ActivationType mapping for activation_func={func}")

    def set_inference_cuda_graphed_iteration(self):
        """Enable CUDA-graphed iteration mode."""
        self.is_inference_cuda_graphed_iteration = True

    def unset_inference_cuda_graphed_iteration(self):
        """Disable CUDA-graphed iteration mode."""
        self.is_inference_cuda_graphed_iteration = False

    def _build_concatenated_mxfp8_weights(self):
        """Build stacked MXFP8 weight tensors from per-expert MXFP8Tensor attributes.

        After quantize_model_to_mxfp8, each per-expert weight (weight0, weight1, ...)
        has been replaced with an MXFP8Tensor. This method stacks their data and
        scales into _fc1_weight / _fc2_weight for scaled_grouped_mm.

        Note: this creates a contiguous copy since per-expert MXFP8Tensor attributes
        are not contiguous across experts. This is a one-time cost at first forward.

        Unlike _build_concatenated_weights, this does not create nn.Parameter views
        back into the buffer — MXFP8 weights are not nn.Parameters (they are plain
        MXFP8Tensor attributes set by quantize_model_to_mxfp8). This path is only
        intended for non-colocated inference.
        """

        for linear_name, buf_name in [('linear_fc1', '_fc1_weight'), ('linear_fc2', '_fc2_weight')]:
            linear = getattr(self, linear_name)
            q_list, s_list = [], []
            for i in range(self.num_local_experts):
                w = getattr(linear, f'weight{i}')
                if isinstance(w, MXFP8Tensor):
                    mxfp8 = w
                elif hasattr(w, 'data') and isinstance(w.data, MXFP8Tensor):
                    mxfp8 = w.data
                else:
                    raise RuntimeError(
                        f"Expected MXFP8Tensor for {linear_name}.weight{i}, "
                        f"got {type(w).__name__}. Was quantize_model_to_mxfp8 called?"
                    )
                q_list.append(mxfp8.data)
                s_list.append(mxfp8.scale)

            setattr(
                self,
                buf_name,
                MXFP8Tensor(
                    data=torch.stack(q_list, dim=0).contiguous(),
                    scale=torch.stack(s_list, dim=0).contiguous(),
                ),
            )

    @torch.inference_mode(False)  # needed for non-colocated inference.
    def _build_concatenated_weights(self):
        """Create big contiguous weight tensors that share storage with TE's per-expert parameters.

        Creates _fc1_weight and _fc2_weight as contiguous tensors of shape
        [num_experts, out_features, in_features]. Instead of replacing TE's parameters
        (which breaks TE's internal bookkeeping), we redirect each parameter's .data
        to be a view into the contiguous buffer. The nn.Parameter objects themselves
        remain untouched in TE's module, preserving FP8 scaling state, etc.

        This allows:
        - TE's forward to work correctly (same Parameter objects, same internal state)
        - Training updates to flow through (param.data is a view into the big tensor)
        - torch.nn.functional.grouped_mm / FlashInfer to use the big tensor directly
        """
        # Get device/dtype from existing TE weights
        device = self.linear_fc1.weight0.device
        dtype = self.linear_fc1.weight0.dtype

        fc1_shape = self.linear_fc1.weight0.shape  # [out_features, in_features]
        fc2_shape = self.linear_fc2.weight0.shape

        # Create big contiguous tensors
        _fc1_weight = torch.empty(self.num_local_experts, *fc1_shape, device=device, dtype=dtype)
        _fc2_weight = torch.empty(self.num_local_experts, *fc2_shape, device=device, dtype=dtype)

        # Copy existing TE weights into big tensors, then point param.data to the views
        for i in range(self.num_local_experts):
            fc1_param = getattr(self.linear_fc1, f'weight{i}')
            fc2_param = getattr(self.linear_fc2, f'weight{i}')

            # Copy initialized data into contiguous buffer
            _fc1_weight[i].copy_(fc1_param.data)
            _fc2_weight[i].copy_(fc2_param.data)

            # Redirect param.data to view into contiguous buffer.
            # The nn.Parameter object stays the same — TE's internal state is preserved.
            fc1_param.data = _fc1_weight[i]
            fc2_param.data = _fc2_weight[i]

        # Register big tensors as non-persistent buffers (for .to() device movement, not saved)
        self.register_buffer('_fc1_weight', _fc1_weight, persistent=False)
        self.register_buffer('_fc2_weight', _fc2_weight, persistent=False)

    def _flashinfer_forward(self, hidden_states, routing_map, probs):
        """FlashInfer fused MoE kernel for CUDA-graphed inference iterations."""
        assert HAVE_FLASHINFER, "flashinfer-python is required for FlashInfer forward path."
        assert probs.dtype == torch.float32, "FlashInfer forward path requires fp32 probabilities."
        output = fused_moe.cutlass_fused_moe(
            hidden_states,
            routing_map.int(),
            probs,
            self._fc1_weight,
            self._fc2_weight,
            hidden_states.dtype,
            quant_scales=None,
            activation_type=self._flashinfer_activation_type,
            ep_size=self.ep_group.size(),
            ep_rank=self.ep_group.rank(),
        )[0]
        return output, None

    def _mcore_fused_moe_forward(
        self, hidden_states, probs, routing_map=None, tokens_per_expert=None, skip_permute=False
    ):
        """Torch grouped_mm fused MoE forward via mcore_fused_moe."""
        local_expert_start = self.ep_group.rank() * self.num_local_experts
        output = mcore_fused_moe(
            hidden_states,
            probs,
            self._fc1_weight,
            self._fc2_weight,
            activation_type=self._mcore_activation_type,
            num_local_experts=self.num_local_experts,
            local_expert_start=local_expert_start,
            routing_map=routing_map,
            tokens_per_expert=tokens_per_expert,
            skip_permute=skip_permute,
            disable_fused_quant_kernels=self.config.inference_moe_disable_fused_quant_kernels,
        )
        return output, None

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: Optional[torch.Tensor],
        permuted_probs: torch.Tensor,
        routing_map: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with three modes:

        - Training: delegates to parent TEGroupedMLP.
        - Inference + CUDA graphed: FlashInfer cutlass_fused_moe. tokens_per_expert
          is not used in this path; the FlashInfer kernel operates directly on
          routing_map.
        - Inference + eager: torch.nn.functional.grouped_mm with GPU-resident cumsum offsets.

        Args:
            permuted_local_hidden_states: [num_tokens, hidden_size] input hidden states.
            tokens_per_expert: [num_experts] number of tokens routed to each expert.
                None when using the CUDA-graphed FlashInfer path.
            permuted_probs: [num_tokens, topk] routing probabilities.
            routing_map: [num_tokens, topk] token-to-expert assignment indices.
                Required for the FlashInfer CUDA-graphed path, None otherwise.
        """

        if self.training:
            assert (
                not self.config.fp8_recipe == "mxfp8"
            ), "MXFP8 inference optimized is not compatible with training / colocated RL."
            return super().forward(permuted_local_hidden_states, tokens_per_expert, permuted_probs)

        # Lazily build concatenated weights on first forward (after checkpoint load)
        if not self._concatenated_weights_built:
            if self.config.fp8_recipe == "mxfp8":
                self._build_concatenated_mxfp8_weights()
            else:
                self._build_concatenated_weights()
            self._concatenated_weights_built = True

        resolved_backend = resolve_inference_grouped_gemm_backend(
            self.inference_grouped_gemm_backend,
            self.is_inference_cuda_graphed_iteration,
            is_mxfp8=self.config.fp8_recipe == "mxfp8",
        )

        if resolved_backend == InferenceGroupedGemmBackend.FLASHINFER:
            assert routing_map is not None, "routing_map is required for FlashInfer forward pass."
            assert (
                self.is_inference_cuda_graphed_iteration
            ), "FlashInfer forward path is only used in CUDA-graphed inference iterations."
            return self._flashinfer_forward(
                permuted_local_hidden_states, routing_map, permuted_probs
            )
        elif resolved_backend == InferenceGroupedGemmBackend.TORCH:
            return self._mcore_fused_moe_forward(
                permuted_local_hidden_states,
                permuted_probs,
                routing_map=routing_map,
                tokens_per_expert=tokens_per_expert,
                skip_permute=(not self.is_inference_cuda_graphed_iteration),
            )
        elif resolved_backend == InferenceGroupedGemmBackend.TE:
            return super().forward(permuted_local_hidden_states, tokens_per_expert, permuted_probs)


class SequentialMLP(MegatronModule):
    """An implementation of the Experts layer using a sequence of MLP layers.

    This class executes each expert sequentially.
    """

    # TODO(M4): breaking api, switched from pass in tp_group to pass in pg_collection.
    def __init__(
        self,
        num_local_experts,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):

        if config.moe_ffn_hidden_size == config.ffn_hidden_size:
            super().__init__(config=config)
        else:
            # Local SequentialMLP can still be used here by overriding the ffn_hidden_size
            # with a deepcopied config.
            sequential_mlp_config = deepcopy(config)
            sequential_mlp_config.ffn_hidden_size = config.moe_ffn_hidden_size
            super().__init__(config=sequential_mlp_config)

        self.num_local_experts = num_local_experts
        self.local_experts = torch.nn.ModuleList()
        self.ep_group = pg_collection.ep
        self.tp_group = pg_collection.expt_tp
        # use pg_collection.expt_dp_group as data parallel group in this module.
        # TODO (Hepteract): expt_dp wont be needed here once distributed checkpoint is refactored
        self.dp_group = pg_collection.expt_dp

        for _ in range(self.num_local_experts):
            expert = MLP(
                self.config,
                submodules,
                ffn_hidden_size=self.config.moe_ffn_hidden_size,
                is_expert=True,
                tp_group=pg_collection.expt_tp,
            )
            self.local_experts.append(expert)

    def _pad_tensor_for_quantization(self, hidden, probs):
        """Padding tensor shape to multiples of 16/32."""
        actual_num_tokens = hidden.shape[0]
        divisor = get_align_size_for_quantization(self.config)
        padded_num_tokens = ceil(actual_num_tokens / divisor) * divisor - actual_num_tokens
        if padded_num_tokens > 0:
            pad_tensor = torch.zeros(
                padded_num_tokens, hidden.shape[1], dtype=hidden.dtype, device=hidden.device
            )
            hidden = torch.cat((hidden, pad_tensor), dim=0)
            pad_probs = torch.zeros(padded_num_tokens, dtype=probs.dtype, device=probs.device)
            probs = torch.cat((probs, pad_probs), dim=0)
        return hidden, probs

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ):
        """Forward step of the SequentialMLP."""

        if self.config.moe_apply_probs_on_input:
            assert (
                self.config.moe_router_topk == 1
            ), "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = (
                permuted_probs.unsqueeze(-1) * permuted_local_hidden_states
            )
            permuted_local_hidden_states = permuted_local_hidden_states.to(original_dtype)
            # Probs already applied, so reset to 1.
            permuted_probs = torch.ones_like(permuted_probs)

        if self.num_local_experts == 1:
            if self.config.fp8 or self.config.fp4:
                hidden, probs = self._pad_tensor_for_quantization(
                    permuted_local_hidden_states, permuted_probs
                )
                output, output_bias = self.local_experts[0](hidden, probs)
                output = output[: permuted_local_hidden_states.shape[0]]
            else:
                output, output_bias = self.local_experts[0](
                    permuted_local_hidden_states, permuted_probs
                )

            return output, output_bias
        else:
            tokens_per_expert = tokens_per_expert.tolist()
            tokens_list = torch.split(permuted_local_hidden_states, tokens_per_expert)
            probs_list = torch.split(permuted_probs, tokens_per_expert)

            output_local_list = []

            for expert, tokens, probs in zip(self.local_experts, tokens_list, probs_list):
                if self.config.fp8 or self.config.fp4:
                    hidden, probs = self._pad_tensor_for_quantization(tokens, probs)
                    output, output_bias = expert(hidden, probs)
                    output = output[: tokens.shape[0]]
                else:
                    output, output_bias = expert(tokens, probs)
                output_local_list.append(output)

            output_local = torch.cat(output_local_list, dim=0)
            output_bias_local = None
            # Note: if bias is enabled on experts, it is already added to the output at this point
            return output_local, output_bias_local

    def backward_dw(self):
        """Backward pass for weight gradients in SequentialMLP."""
        for expert in self.local_experts:
            expert.backward_dw()

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Maps local expert to global experts."""
        # Guard for cases metadata is not provided
        metadata = ensure_metadata_has_dp_cp_group(metadata)

        sharded_state_dict = {}
        num_global_experts = self.ep_group.size() * self.num_local_experts
        local_expert_indices_offset = self.ep_group.rank() * self.num_local_experts

        singleton_local_shards = (metadata or {}).get('singleton_local_shards', False)

        for expert_local_idx, expert in enumerate(self.local_experts):
            expert_global_idx = local_expert_indices_offset + expert_local_idx
            expert_state_dict_prefix = f'{prefix}local_experts.{expert_local_idx}.'
            if singleton_local_shards:
                expert_sharded_prefix = f'{prefix}experts.{expert_global_idx}.'
                expert_sharded_offsets = sharded_offsets
            else:
                expert_sharded_prefix = f'{prefix}experts.'
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

                sh_ten.replica_id = (*replica_id[:2], self.dp_group.rank())

            sharded_state_dict.update(expert_state_dict)
        return sharded_state_dict
