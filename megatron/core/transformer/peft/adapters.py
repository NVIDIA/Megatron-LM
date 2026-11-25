# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from NVIDIA-NeMo/Megatron-Bridge (Apache-2.0).
# Source: https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/
# c031e61d82dafbe4655a2e2d7e742b88f7e81c3a/src/megatron/bridge/peft/utils.py

# LinearAdapter is adapted from lora_layers.py at the same source revision.
"""Low-rank delta modules for plain and tensor-parallel linear layers."""

import logging
import math
from dataclasses import replace
from typing import Callable, Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn

from megatron.core import ModelParallelConfig
from megatron.core.dist_checkpointing.mapping import (
    ShardedStateDict,
    ShardedTensor,
    ShardedTensorFactory,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.mappings import all_to_all_hp2sp as all2all_hp2sp
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.tensor_parallel.random import is_checkpointing
from megatron.core.transformer.mlp import apply_swiglu_sharded_factory

from .grad_sync import _broadcast_shared_expert_weights, mark_expert_parallel_replicated
from .linear_attributes import HAVE_TE, TELinear, is_grouped_expert_linear
from .process_groups import (
    _get_pg_collection,
    _get_process_group,
    _get_tensor_parallel_group,
    _process_group_rank,
    _process_group_size,
)
from .tensor_helpers import (
    init_method_const,
    init_method_kaiming_uniform,
    init_method_normal,
    init_method_uniform,
    pad_seq_to_mult,
    unpad_seq_to_mult,
)

logger = logging.getLogger(__name__)


class ParallelLinearAdapter(nn.Module):
    """Parallel Linear Adapter for Parameter-Efficient Fine-Tuning (PEFT) in distributed settings.

    This adapter implements a low-rank adaptation pattern using two linear layers with configurable
    parallelization strategies. It supports both tensor and sequence parallelism patterns used in
    large language model training.

    The adapter follows the pattern: input -> linear_in -> activation -> linear_out -> scaling
    where linear_in and linear_out are parallelized according to the base layer configuration.

    Args:
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        dim: Adapter bottleneck dimension (rank).
        base_linear_name: Name of the base linear layer being adapted.
        activation: Activation function name (default: 'swish').
        column_init_method: Initialization method for column parallel layer (default: 'xavier').
        row_init_method: Initialization method for row parallel layer (default: 'zero').
        input_is_parallel: Whether input is already parallelized (default: False).
        dropout: Dropout probability (default: 0.0).
        model_parallel_config: Configuration for model parallelism (default: None).
        alpha: Scaling factor for adapter output (default: None, uses dim).
        dropout_position: Where to apply dropout ('pre' or 'post', default: 'pre').
        a2a_experimental: Whether to use experimental all-to-all communication (default: False).
        is_expert: Whether this adapter is for expert layers in MoE (default: False).
        disable_sequence_parallel_comm: Whether to disable sequence parallel communication (default: True).
        base_linear_is_parallel: Whether the base linear layer uses parallelization (default: True).
        replicate_adapter: Whether both low-rank matrices should be duplicated across TP ranks.
        sequence_parallel_input_regather: Whether eligible LoRA-A projections retain the sequence-local input and
            re-gather it in backward using MCore's sequence-parallel linear path (default: False).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dim: int,
        base_linear_name: str,
        activation: str = "swish",
        column_init_method: str = "xavier",
        row_init_method: str = "zero",
        input_is_parallel: bool = False,
        dropout: float = 0.0,
        model_parallel_config: Optional[ModelParallelConfig] = None,
        alpha: Optional[float] = None,
        dropout_position: str = "pre",
        a2a_experimental: bool = False,
        is_expert: bool = False,
        disable_tensor_parallel_comm: bool = False,
        disable_sequence_parallel_comm: bool = True,
        base_linear_is_parallel: bool = True,
        replicate_adapter: bool = False,
        pg_collection: ProcessGroupCollection | None = None,
        sequence_parallel_input_regather: bool = False,
    ) -> None:
        """Initialize the ParallelLinearAdapter.

        Args:
            in_features: Input feature dimension.
            out_features: Output feature dimension.
            dim: Adapter bottleneck dimension.
            base_linear_name: Name of the base linear layer.
            activation: Activation function name.
            column_init_method: Initialization for column parallel layers.
            row_init_method: Initialization for row parallel layers.
            input_is_parallel: Whether input is already parallelized.
            dropout: Dropout probability.
            model_parallel_config: Model parallelism configuration.
            alpha: Scaling factor (uses dim if None).
            dropout_position: When to apply dropout.
            a2a_experimental: Use experimental all-to-all communication.
            is_expert: Whether for expert layers in MoE.
            disable_tensor_parallel_comm: Disable tensor parallel communication.
            disable_sequence_parallel_comm: Disable sequence parallel communication.
            replicate_adapter: Duplicate both low-rank matrices across TP ranks.
            sequence_parallel_input_regather: Re-gather eligible LoRA-A sequence-parallel inputs in backward.
        """
        super().__init__()
        self.base_linear_name = base_linear_name
        self.activation = self._get_activation_fn(activation)
        self.dim = dim
        self.alpha = alpha if alpha is not None else self.dim
        self.input_is_parallel = input_is_parallel
        self.dropout_position = dropout_position
        self.use_a2a = a2a_experimental
        self.is_expert = is_expert
        self.base_linear_is_parallel = base_linear_is_parallel
        self.replicate_adapter = replicate_adapter
        self.sequence_parallel_input_regather = sequence_parallel_input_regather
        self._sequence_parallel_input_regather_fallback_logged = False
        self.use_legacy_shared_expert_adapter_checkpoint = False

        # megatron_gpt_peft_models will provide this arg, but deprecated ones do not.
        # in case this arg is not provided, use the dummy default config.
        if model_parallel_config is None:
            model_parallel_config = ModelParallelConfig()
        self.pg_collection = _get_pg_collection(
            pg_collection,
            model_parallel_config,
            required_pgs=["ep", "expt_tp", "expt_dp"] if is_expert else ["tp"],
        )
        self.tp_group = _get_tensor_parallel_group(self.pg_collection, is_expert=is_expert)
        self.ep_group = _get_process_group(self.pg_collection, "ep")
        self.expert_dp_group = _get_process_group(self.pg_collection, "expt_dp")
        _sequence_parallel = model_parallel_config.sequence_parallel
        self.config = model_parallel_config

        # Ensure adapter parameters are initialized when creating adapter layers.
        # In some flows (e.g., after import), perform_initialization may be False to skip heavy init.
        model_parallel_config = replace(model_parallel_config, perform_initialization=True)

        if replicate_adapter:
            if is_expert:
                raise ValueError("Replicated adapters are only supported for non-expert linears")
            if not HAVE_TE:
                raise RuntimeError("Replicated adapters require Transformer Engine")
            self.linear_in = TELinear(
                in_features,
                dim,
                parallel_mode="duplicated",
                config=model_parallel_config,
                init_method=self._get_init_fn(column_init_method),
                bias=False,
                skip_bias_add=False,
                skip_weight_param_allocation=False,
                tp_group=None,
            )
            self.linear_out = TELinear(
                dim,
                out_features,
                parallel_mode="duplicated",
                config=model_parallel_config,
                init_method=self._get_init_fn(row_init_method),
                bias=False,
                skip_bias_add=False,
                skip_weight_param_allocation=False,
                tp_group=None,
            )
        else:
            model_parallel_config = replace(model_parallel_config, sequence_parallel=False)

        if not replicate_adapter and input_is_parallel:
            self.linear_in = RowParallelLinear(
                in_features,
                dim,
                config=model_parallel_config,
                input_is_parallel=True,
                skip_bias_add=True,
                bias=False,
                init_method=self._get_init_fn(column_init_method, fan_in=in_features, fan_out=dim),
                is_expert=is_expert,
                tp_group=self.tp_group,
            )
        elif not replicate_adapter:
            self.linear_in = ColumnParallelLinear(
                in_features,
                dim,
                config=model_parallel_config,
                bias=False,
                gather_output=True,
                init_method=self._get_init_fn(column_init_method, fan_in=in_features, fan_out=dim),
                disable_grad_reduce=_sequence_parallel,
                is_expert=is_expert,
                tp_group=self.tp_group,
            )

        # (@adithyare) we use this option to mirror the behavior
        # a column parallel layer with two low-rank column parallel layers
        # if the original column parallel layer uses gather_output=False,
        # then we will use the self.liner_out layer defined below.
        if not replicate_adapter:
            lin_out_gather_output = True if input_is_parallel else False
            if (
                self.use_a2a
                and input_is_parallel
                and _sequence_parallel
                or (disable_tensor_parallel_comm and not input_is_parallel)
            ):
                lin_out_gather_output = False

            if not base_linear_is_parallel:
                lin_out_gather_output = True

            self.linear_out = ColumnParallelLinear(
                dim,
                out_features,
                config=model_parallel_config,
                bias=False,
                gather_output=lin_out_gather_output,
                init_method=self._get_init_fn(row_init_method, fan_in=dim, fan_out=out_features),
                is_expert=is_expert,
                tp_group=self.tp_group,
            )

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

        # cast all parameters when using amp O2 training
        if model_parallel_config.bf16:
            self.bfloat16()
        elif model_parallel_config.fp16:
            self.half()

        if self._uses_grouped_expert_sharding():
            self._synchronize_shared_expert_parameters()
            self._register_shared_expert_grad_sync_hooks()

        self.disable_sequence_parallel_comm = disable_sequence_parallel_comm
        if not _sequence_parallel:
            self.disable_sequence_parallel_comm = True

        if not base_linear_is_parallel:
            self.disable_sequence_parallel_comm = True

    def _sequence_parallel_input_regather_eligibility(
        self, x: torch.Tensor
    ) -> tuple[bool, str | None]:
        """Return whether the targeted sequence-parallel linear path is safe."""
        if not self.sequence_parallel_input_regather:
            return False, None
        if not self.training or not torch.is_grad_enabled():
            return False, None
        if is_checkpointing():
            return False, None
        if not self.linear_in.weight.requires_grad:
            return False, "the LoRA-A weight is frozen"
        if self.input_is_parallel:
            return False, "the adapter is row-parallel"
        if self.disable_sequence_parallel_comm:
            return False, "sequence-parallel communication is disabled"
        if _process_group_size(self.tp_group, self.config.tensor_model_parallel_size or 1) <= 1:
            return False, "tensor parallel size is one"
        if self.is_expert:
            return False, "expert adapters use a separate path"
        if self.use_a2a:
            return False, "all-to-all adapters use a separate path"
        if not isinstance(self.activation, nn.Identity):
            return False, "the adapter activation is not identity"
        if self.config.cpu_offloading and self.config.cpu_offloading_activations:
            return False, "CPU activation offload is enabled"

        surface = self.base_linear_name.rsplit(".", maxsplit=1)[-1]
        if surface not in {"linear_qkv", "linear_fc1"}:
            return False, f"{surface} is not an eligible column-parallel LoRA surface"

        cuda_graph_impl = getattr(self.config, "cuda_graph_impl", "none")
        if cuda_graph_impl not in (None, "none"):
            return False, "CUDA graphs are enabled"

        recompute_granularity = getattr(self.config, "recompute_granularity", None)
        if recompute_granularity == "full":
            return False, "full-layer activation recompute already covers the adapter"
        recompute_modules = getattr(self.config, "recompute_modules", None) or []
        if (
            recompute_granularity == "selective"
            and surface == "linear_fc1"
            and "mlp" in recompute_modules
        ):
            return False, "selective MLP recompute already covers linear_fc1"
        return True, None

    def _log_sequence_parallel_input_regather_fallback(self, reason: str | None) -> None:
        """Log the first static fallback reason at debug level."""
        if reason is None or self._sequence_parallel_input_regather_fallback_logged:
            return
        logger.debug(
            "LoRA sequence-parallel input re-gather requested for %s but using the existing path: %s",
            self.base_linear_name,
            reason,
        )
        self._sequence_parallel_input_regather_fallback_logged = True

    def _get_activation_fn(self, activation: str) -> nn.Module:
        """Get activation function by name.

        Args:
            activation: Name of the activation function.

        Returns:
            PyTorch activation module.

        Note:
            Defaults to Identity if activation name is not recognized.
        """
        activation_map = {
            "identity": nn.Identity(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }
        return activation_map.get(activation, nn.Identity())

    def _get_init_fn(
        self, init_method: str, fan_in: int | None = None, fan_out: int | None = None
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Get initialization function by method name.

        When the adapter weight is sharded across tensor-parallel ranks, Megatron
        initializes each shard locally, so a fan-dependent init function (xavier,
        kaiming) computed on the shard would give a distribution that depends on
        the TP size. Passing the full (unsharded) ``fan_in``/``fan_out`` resolves
        those methods to a fixed-scale distribution computed from the full tensor
        dims, making initialization independent of the sharding. At TP=1 the
        resolved distributions are identical to the tensor-derived ones.

        Args:
            init_method: Name of the initialization method.
            fan_in: Full (unsharded) input dimension of the weight, if known.
            fan_out: Full (unsharded) output dimension of the weight, if known.

        Returns:
            Initialization function.

        Raises:
            NotImplementedError: If init_method is not supported.
        """
        if init_method == "xavier":
            if fan_in is not None and fan_out is not None:
                init_fn = init_method_normal(math.sqrt(2.0 / (fan_in + fan_out)))
            else:
                init_fn = nn.init.xavier_normal_
        elif init_method == "normal":
            init_fn = init_method_normal(0.2)
        elif init_method == "kaiming":
            if fan_in is not None:
                # kaiming_uniform_(a=sqrt(5)) on the full fan_in:
                # bound = sqrt(2/(1+a^2)) * sqrt(3/fan_in) = 1/sqrt(fan_in)
                init_fn = init_method_uniform(1.0 / math.sqrt(fan_in))
            else:
                init_fn = init_method_kaiming_uniform(math.sqrt(5))
        elif init_method == "zero":
            init_fn = init_method_const(0.0)
        else:
            raise NotImplementedError("out_init_method should be zero, normal, kaiming or xavier")
        return init_fn

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass of the parallel linear adapter.

        Performs the adaptation computation with proper handling of parallel communication
        patterns, dropout, and expert routing for MoE scenarios.

        Args:
            x: Input tensor.

        Returns:
            Adapted output tensor with scaling applied.
        """
        del args, kwargs

        if self.dropout_position == "pre":
            x = self.dropout(x)

        pad_len = 0
        if self.is_expert:
            x, pad_len = pad_seq_to_mult(x, self.config.expert_tensor_parallel_size)

        use_sequence_parallel_input_regather, fallback_reason = (
            self._sequence_parallel_input_regather_eligibility(x)
        )
        if not self.input_is_parallel and not self.replicate_adapter:
            # MCore's SP linear keeps the local input in ctx, launches the
            # backward all-gather asynchronously before dgrad, and waits only
            # before wgrad. The baseline path keeps communication external.
            self.linear_in.sequence_parallel = use_sequence_parallel_input_regather
        if use_sequence_parallel_input_regather:
            # ColumnParallelLinear returns output and bias; adapters do not use the bias.
            x, _ = self.linear_in(x)
        else:
            self._log_sequence_parallel_input_regather_fallback(fallback_reason)
            if (
                not self.disable_sequence_parallel_comm
                and not self.input_is_parallel
                and not self.is_expert
            ):
                # for attention_qkv and linear_fc1
                # layernorm before lora is impacted by sequence parallel,
                # hence seq dim need to be gathered right before lora linear layers
                # this function also handles the backward pass correctly
                x = gather_from_sequence_parallel_region(x, group=self.tp_group)

            if self.config.cpu_offloading and self.config.cpu_offloading_activations:
                x.activation_offloading = True
            # ColumnParallelLinear returns output and bias; adapters do not use the bias.
            x, _ = self.linear_in(x)

        x = self.activation(x)
        x = x * (self.alpha / self.dim)

        if self.config.cpu_offloading and self.config.cpu_offloading_activations:
            x.activation_offloading = True
        x, _ = self.linear_out(x)

        if (
            not self.disable_sequence_parallel_comm
            and self.input_is_parallel
            and not self.is_expert
        ):
            # for attention_dense and linear_fc2
            # layernorm after lora is impacted by sequence parallel,
            # hence seq dim need to be scattered right after lora linear layers
            # this function also handles the backward pass correctly
            if self.use_a2a:
                # all2all hidden_size / TP to seq_len / TP
                x = all2all_hp2sp(x, self.tp_group)
            else:
                x = scatter_to_sequence_parallel_region(x, group=self.tp_group)

        # Add dropout if available
        if self.dropout_position == "post":
            x = self.dropout(x)

        if pad_len > 0:
            # Remove MoE padding.
            x = unpad_seq_to_mult(x, pad_len)

        return x

    def local_experts_per_rank(self) -> int:
        """Return the number of global expert slots owned by this EP rank."""

        ep_size = _process_group_size(self.ep_group, self.config.expert_model_parallel_size or 1)
        num_global_experts = getattr(self.config, "num_moe_experts", None)
        if num_global_experts is None:
            return 1
        if int(num_global_experts) % int(ep_size) != 0:
            raise ValueError(
                f"num_moe_experts={num_global_experts} must be divisible by expert_model_parallel_size={ep_size}"
            )
        return int(num_global_experts) // int(ep_size)

    def _uses_grouped_expert_sharding(self) -> bool:
        """Return whether this shared adapter needs an explicit expert axis."""

        return self.is_expert and is_grouped_expert_linear(self.base_linear_name)

    def _synchronize_shared_expert_parameters(self) -> None:
        """Broadcast shared expert adapter parameters from EP group rank zero."""

        _broadcast_shared_expert_weights(self._shared_expert_weights(), self.ep_group)

    def _register_shared_expert_grad_sync_hooks(self) -> None:
        """Keep shared grouped-expert adapters synchronized across EP ranks.

        MCore's expert DDP only reduces over expert-DP; :func:`mark_expert_parallel_replicated` adds the
        cross-EP sum (once per step from the finalize step, with a per-layer fallback hook until then).
        """

        for weight in self._shared_expert_weights():
            mark_expert_parallel_replicated(weight, ep_group=self.ep_group)

    def _shared_expert_weights(self) -> list[torch.Tensor]:
        """Return the adapter weights shared by every expert on this EP rank."""

        return [
            weight
            for module in (self.linear_in, self.linear_out)
            if isinstance(weight := getattr(module, "weight", None), torch.Tensor)
        ]

    def _expert_axis_info(self, sharded_offsets: Tuple) -> Tuple[int, int, int]:
        """Return the global expert-axis sharding metadata for this rank."""

        ep_size = _process_group_size(self.ep_group, self.config.expert_model_parallel_size or 1)
        ep_rank = _process_group_rank(self.ep_group)
        local_experts = self.local_experts_per_rank()
        if local_experts <= 0:
            raise ValueError(f"local_experts_per_rank must be positive, got {local_experts}")

        num_global_experts = getattr(self.config, "num_moe_experts", None)
        if num_global_experts is None:
            num_global_experts = ep_size * local_experts
        num_global_experts = int(num_global_experts)
        first_expert_slot = ep_rank * local_experts
        if first_expert_slot >= num_global_experts:
            raise ValueError(
                f"Invalid expert adapter sharding for {self.base_linear_name}: "
                f"ep_rank={ep_rank}, local_experts_per_rank={local_experts}, "
                f"num_global_experts={num_global_experts}"
            )

        expert_axis = len(sharded_offsets)
        return expert_axis, first_expert_slot, num_global_experts

    def _keep_expert_extra_state(self) -> bool:
        """Keep one unsharded adapter extra-state entry."""

        tp_rank = _process_group_rank(self.tp_group)
        ep_rank = _process_group_rank(self.ep_group)
        return tp_rank == 0 and ep_rank == 0

    def _set_expert_replica_ids(self, *state_dicts: ShardedStateDict) -> None:
        """Mark expert adapter replicas across expert data-parallel ranks."""

        edp_rank = _process_group_rank(self.expert_dp_group)
        for state_dict in state_dicts:
            for value in state_dict.values():
                if not hasattr(value, "replica_id"):
                    continue
                replica_id = value.replica_id
                if isinstance(replica_id, int):
                    replica_id = (0, 0, replica_id)
                if len(replica_id) != 3:
                    raise ValueError(
                        f"Expected replica_id for {self.base_linear_name} to be in "
                        f"(PP, TP, DP) format, got: {replica_id}"
                    )
                dp_replica_id = (
                    0 if getattr(value, "is_data_parallel_fully_shard", False) else edp_rank
                )
                value.replica_id = (*replica_id[:2], dp_replica_id)

    def _apply_expert_axis_factory(
        self, sharded_tensor: ShardedTensor, sharded_offsets: Tuple, *, split_swiglu: bool = False
    ) -> ShardedTensorFactory:
        """Map one shared 2D adapter tensor to this rank's global expert slots."""

        expert_axis, first_expert_slot, num_global_experts = self._expert_axis_info(sharded_offsets)
        local_experts = self.local_experts_per_rank()
        base_prepend_axis_num = len(sharded_offsets)
        output_prepend_axis_num = base_prepend_axis_num + 1
        swiglu_shard_axis = 0

        preserved_rank_offsets = []
        for axis, local_axis_shape in enumerate(sharded_tensor.local_shape):
            base_global_axis_idx = axis + base_prepend_axis_num
            output_global_axis_idx = base_global_axis_idx + 1
            axis_fragments = sharded_tensor.axis_fragmentations[base_global_axis_idx]
            if axis_fragments <= 1:
                continue
            global_offset = sharded_tensor.global_offset[base_global_axis_idx]
            if global_offset % local_axis_shape != 0:
                raise ValueError(
                    f"Cannot preserve non-integral sharding for {sharded_tensor.key}: "
                    f"offset={global_offset}, local_axis_shape={local_axis_shape}"
                )
            preserved_rank_offsets.append(
                (output_global_axis_idx, global_offset // local_axis_shape, axis_fragments)
            )

        swiglu_axis_frag = None
        swiglu_rank_offset = None
        base_swiglu_global_axis = swiglu_shard_axis + base_prepend_axis_num
        output_swiglu_global_axis = swiglu_shard_axis + output_prepend_axis_num
        if split_swiglu:
            local_axis_size = sharded_tensor.local_shape[swiglu_shard_axis]
            if sharded_tensor.global_offset[base_swiglu_global_axis] % local_axis_size != 0:
                raise ValueError(
                    f"Cannot split SwiGLU tensor {sharded_tensor.key}: "
                    f"offset={sharded_tensor.global_offset[base_swiglu_global_axis]}, local_axis_size={local_axis_size}"
                )
            swiglu_rank_offset = (
                sharded_tensor.global_offset[base_swiglu_global_axis] // local_axis_size
            )
            swiglu_axis_frag = sharded_tensor.axis_fragmentations[base_swiglu_global_axis]
            preserved_rank_offsets = [
                rank_offset
                for rank_offset in preserved_rank_offsets
                if rank_offset[0] != output_swiglu_global_axis
            ]

        @torch.no_grad()
        def sh_ten_build_fn(key: str, tensor: torch.Tensor, replica_id, flattened_range):
            del flattened_range
            sharded_tensors = []
            for expert_index in range(local_experts):
                expert_offset = (expert_axis, first_expert_slot + expert_index, num_global_experts)
                if not split_swiglu:
                    sharded_tensors.append(
                        ShardedTensor.from_rank_offsets(
                            key,
                            tensor,
                            *sharded_offsets,
                            *preserved_rank_offsets,
                            expert_offset,
                            replica_id=replica_id,
                            prepend_axis_num=output_prepend_axis_num,
                        )
                    )
                    continue

                tensor_w, tensor_v = torch.chunk(tensor, 2, dim=swiglu_shard_axis)
                offset_w = (output_swiglu_global_axis, swiglu_rank_offset, swiglu_axis_frag * 2)
                offset_v = (
                    output_swiglu_global_axis,
                    swiglu_rank_offset + swiglu_axis_frag,
                    swiglu_axis_frag * 2,
                )
                for tensor_part, swiglu_offset in ((tensor_w, offset_w), (tensor_v, offset_v)):
                    sharded_tensors.append(
                        ShardedTensor.from_rank_offsets(
                            key,
                            tensor_part,
                            *sharded_offsets,
                            *preserved_rank_offsets,
                            expert_offset,
                            swiglu_offset,
                            replica_id=replica_id,
                            prepend_axis_num=output_prepend_axis_num,
                        )
                    )
            return sharded_tensors

        def sh_ten_merge_fn(sub_state_dict):
            if not isinstance(sub_state_dict, list):
                sub_state_dict = [sub_state_dict]
            if split_swiglu:
                if len(sub_state_dict) % 2 != 0:
                    raise ValueError(
                        f"Expected even number of SwiGLU shards for {sharded_tensor.key}"
                    )
                sub_state_dict = [
                    torch.cat(sub_state_dict[index : index + 2], dim=swiglu_shard_axis)
                    for index in range(0, len(sub_state_dict), 2)
                ]
            if len(sub_state_dict) == 1:
                return sub_state_dict[0]
            return torch.stack(sub_state_dict, dim=0).mean(dim=0)

        return ShardedTensorFactory(
            sharded_tensor.key,
            sharded_tensor.data,
            sh_ten_build_fn,
            sh_ten_merge_fn,
            sharded_tensor.replica_id,
            flattened_range=sharded_tensor.flattened_range,
        )

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: Tuple = (),
        metadata: Optional[Dict] = None,
        mamba_dim_info: Optional[Dict] = None,
    ) -> ShardedStateDict:
        """Create sharded state dictionary for distributed checkpointing.

        Special treatment is given to the linear_fc1 adapter since tensor parallelism is
        sharded separately for the two logical matrices (gate and up) in SwiGLU.

        Args:
            prefix: Prefix for parameter names.
            sharded_offsets: Offsets for sharded parameters.
            metadata: Additional metadata for sharding.

        Returns:
            Sharded state dictionary for distributed checkpointing.
        """
        sharded_state_dict = {}
        # Shared grouped-expert adapters have one 2D weight per EP rank, but the
        # checkpoint must expose the global expert axis so EP changes can reshard it.
        # Non-grouped expert adapters already sit under .local_experts.* and keep
        # their existing expert-DP replica metadata instead.
        use_expert_axis = (
            self._uses_grouped_expert_sharding()
            and not self.use_legacy_shared_expert_adapter_checkpoint
        )
        split_swiglu = (
            getattr(self, "split_output_for_checkpoint", True)
            and "linear_fc1" in self.base_linear_name
            and getattr(self.config, "gated_linear_unit", False)
        )
        linear_in_sd = self.linear_in.sharded_state_dict(
            f"{prefix}linear_in.", sharded_offsets, metadata
        )
        linear_out_sd = self.linear_out.sharded_state_dict(
            f"{prefix}linear_out.", sharded_offsets, metadata
        )

        if use_expert_axis:
            keep_extra_state = self._keep_expert_extra_state()
            if not keep_extra_state:
                for sd in (linear_in_sd, linear_out_sd):
                    for key in [k for k in sd if "_extra_state" in k]:
                        del sd[key]
            for key, value in list(linear_in_sd.items()):
                if isinstance(value, ShardedTensor):
                    linear_in_sd[key] = self._apply_expert_axis_factory(value, sharded_offsets)
            for key, value in list(linear_out_sd.items()):
                if isinstance(value, ShardedTensor):
                    linear_out_sd[key] = self._apply_expert_axis_factory(
                        value, sharded_offsets, split_swiglu=split_swiglu
                    )
        elif self.is_expert:
            if _process_group_rank(self.tp_group) > 0:
                for state_dict in (linear_in_sd, linear_out_sd):
                    for key in [k for k in state_dict if "_extra_state" in k]:
                        del state_dict[key]

        if split_swiglu and not use_expert_axis:
            for k, v in linear_out_sd.items():
                if k in (f"{prefix}linear_out.weight", f"{prefix}linear_out.bias"):
                    linear_out_sd[k] = apply_swiglu_sharded_factory(v, sharded_offsets)

        # Special handling for Mamba in_proj layer which needs to be split into 5 tensors
        if mamba_dim_info is not None:
            from megatron.core.ssm.mamba_mixer import _split_tensor_factory

            # Split linear_out.weight into 5 parts: z, x, B, C, dt
            # The in_proj output dimension is: d_inner * 2 + 2 * ngroups * d_state + nheads
            # After TP sharding: d_inner_local_tp * 2 + 2 * ngroups_local_tp * d_state + nheads_local_tp
            for k, v in linear_out_sd.items():
                if k == f"{prefix}linear_out.weight" and isinstance(v, ShardedTensor):
                    in_proj_dim_local = (
                        mamba_dim_info["d_inner_local_tp"] * 2
                        + 2 * mamba_dim_info["ngroups_local_tp"] * mamba_dim_info["d_state"]
                        + mamba_dim_info["nheads_local_tp"]
                    )
                    # Verify the dimension matches
                    if v.data.size(0) == in_proj_dim_local:
                        linear_out_sd[k] = _split_tensor_factory(
                            v,
                            [
                                mamba_dim_info["d_inner_local_tp"],  # z
                                mamba_dim_info["d_inner_local_tp"],  # x
                                mamba_dim_info["ngroups_local_tp"] * mamba_dim_info["d_state"],  # B
                                mamba_dim_info["ngroups_local_tp"] * mamba_dim_info["d_state"],  # C
                                mamba_dim_info["nheads_local_tp"],  # dt
                            ],
                            ["z", "x", "B", "C", "dt"],
                            0,  # split along dimension 0
                        )

        if self.is_expert:
            self._set_expert_replica_ids(linear_in_sd, linear_out_sd)

        sharded_state_dict.update(linear_in_sd)
        sharded_state_dict.update(linear_out_sd)
        return sharded_state_dict


class LinearAdapter(nn.Module):
    """Delta-only LoRA adapter for a plain ``nn.Linear``, mirroring :class:`ParallelLinearAdapter`'s role.

    This adapter holds *only* the low-rank LoRA delta (``linear_in`` -> ``linear_out``) and
    produces just the scaled adaptation term. It is intended to be wrapped together with the
    original linear by :class:`LoRALinear` (the ``to_wrap`` / ``adapter`` pattern), so that
    base weights and adapter weights live in distinct submodules and adapter state is
    checkpointed under the ``adapter.`` prefix.

    Args:
        orig_linear: The linear module to augment (only its shape/dtype/device are used; its
            weights are *not* copied).
        dim: LoRA's dimension (in_features -> dim -> out_features).
        alpha: LoRA's scaling alpha.
        dropout: Dropout probability (default: 0.0).
        dropout_position: Where to apply dropout relative to LoRA (choices: ['pre', 'post'], default='pre').
        lora_A_init_method: Initialization method for lora_A (choices: ['xavier', 'uniform']).
        lora_dtype: Adapter weight dtype. Defaults to the original linear's weight dtype.
    """

    def __init__(
        self,
        orig_linear: nn.Linear,
        dim: int = 8,
        alpha: int = 32,
        dropout: float = 0.0,
        dropout_position: Literal["pre", "post"] = "pre",
        lora_A_init_method: Literal["xavier", "uniform"] = "xavier",
        lora_dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initialize the LoRA delta weights from the original Linear's shape and dtype.

        Args:
            orig_linear: The original Linear module to adapt (weights are not copied).
            dim: LoRA rank dimension.
            alpha: LoRA scaling factor.
            dropout: Dropout probability.
            dropout_position: When to apply dropout ('pre' or 'post' LoRA computation).
            lora_A_init_method: Initialization method for LoRA matrix A.
            lora_dtype: Data type for LoRA weights.
        """
        super().__init__()
        assert isinstance(orig_linear, nn.Linear)
        self.in_features = orig_linear.in_features
        self.out_features = orig_linear.out_features
        self._init_adapter(
            dim=dim,
            alpha=alpha,
            dropout=dropout,
            dropout_position=dropout_position,
            lora_A_init_method=lora_A_init_method,
            lora_dtype=lora_dtype,
            device=orig_linear.weight.device,
            base_dtype=orig_linear.weight.dtype,
        )

    @torch.no_grad
    def _init_adapter(
        self,
        dim: int = 8,
        alpha: int = 32,
        dropout: float = 0.0,
        dropout_position: Literal["pre", "post"] = "pre",
        lora_A_init_method: Literal["xavier", "uniform"] = "xavier",
        lora_dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        base_dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initialize the LoRA delta weights.

        Args:
            dim: LoRA's dimension (in_features -> dim -> out_features).
            alpha: LoRA's scaling alpha.
            dropout: Dropout probability (default: 0.0).
            dropout_position: Where to apply dropout relative to LoRA (choices: ['pre', 'post'], default='pre').
            lora_A_init_method: Initialization method for lora_A (choices: ['xavier', 'uniform']).
            lora_dtype: Adapter weight dtype. Defaults to the base weight dtype.
            device: Device for the LoRA weights.
            base_dtype: Base weight dtype, used when ``lora_dtype`` is not provided.
        """
        self.dim = dim
        self.alpha = alpha
        self.scale = alpha / dim

        in_features = self.in_features
        out_features = self.out_features
        dtype = lora_dtype or base_dtype

        self.linear_in = nn.Linear(in_features, dim, bias=False, dtype=dtype, device=device)
        self.linear_out = nn.Linear(dim, out_features, bias=False, dtype=dtype, device=device)
        if lora_A_init_method == "xavier":
            torch.nn.init.xavier_uniform_(self.linear_in.weight.data)
        else:
            nn.init.kaiming_uniform_(self.linear_in.weight.data, a=math.sqrt(5))
        self.linear_out.weight.data.fill_(0)
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()
        assert dropout_position in ["pre", "post"], dropout_position
        self.dropout_position = dropout_position

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the scaled LoRA delta only (no base-weight term).

        Args:
            x: Input tensor.

        Returns:
            The scaled low-rank adaptation ``scale * linear_out(linear_in(x))`` with dropout
            applied per ``dropout_position``.
        """
        # pylint: disable=C0115,C0116
        if self.dropout_position == "pre":
            x = self.dropout(x)
        lora_res = self.linear_out(self.linear_in(x))
        lora_res = lora_res * self.scale
        if self.dropout_position == "post":
            lora_res = self.dropout(lora_res)
        return lora_res
