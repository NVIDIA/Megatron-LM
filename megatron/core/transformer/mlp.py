# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, Protocol, cast

import numpy as np
import torch
import torch.nn.functional as F

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import (
    ReplicaId,
    ShardedStateDict,
    ShardedTensorFactory,
)
from megatron.core.fusions.fused_bias_geglu import (
    bias_geglu_impl,
    quick_gelu,
    weighted_bias_quick_geglu_impl,
)
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl, weighted_bias_swiglu_impl
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import (
    cat_with_oom_fallback,
    ensure_metadata_has_dp_cp_group,
    sharded_state_dict_default,
)
from megatron.core.typed_torch import apply_module, not_none
from megatron.core.utils import (
    get_pg_rank,
    get_pg_size,
    get_tensor_model_parallel_group_if_none,
    nvtx_range_pop,
    nvtx_range_push,
)

try:
    import transformer_engine  # pylint: disable=unused-import

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


class LinearFc1Interface(Protocol):
    """Interface for linear_fc1 module in MLP."""

    def forward(self, hidden_states: torch.Tensor, /) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward method for linear_fc1 module."""
        ...

    def backward_dw(self) -> None:
        """Backward method for linear_fc1 module."""
        ...


class LinearFc1Builder(Protocol):
    """Protocol describing how to build a linear_fc1 layer in MLP."""

    def __call__(
        self,
        input_size: int,
        output_size: int,
        /,
        *,
        config: TransformerConfig,
        init_method: Callable[[torch.Tensor], None],
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: str | None,
        tp_group: torch.distributed.ProcessGroup | None,
        stride: int = 1,
        name: str | None = None,
    ) -> LinearFc1Interface:
        """Builds a linear_fc1 layer for MLP."""
        ...


class TEActivationFunctionInterface(Protocol):
    """Interface for activation_function module in MLP."""

    def forward(self, input_: torch.Tensor, /) -> torch.Tensor:
        """Forward method for activation_function module."""
        ...


class TEActivationFunctionBuilder(Protocol):
    """Protocol for activation_function module in MLP."""

    def __call__(self, *, config: TransformerConfig) -> TEActivationFunctionInterface:
        """Builds an activation function module for MLP."""
        ...


class LinearFc2Interface(Protocol):
    """Interface for linear_fc2 module in MLP."""

    def forward(self, hidden_states: torch.Tensor, /) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward method for linear_fc2 module."""
        ...

    def backward_dw(self) -> None:
        """Backward method for linear_fc2 module."""
        ...


class LinearFc2Builder(Protocol):
    """Protocol describing how to build a linear_fc2 layer in MLP."""

    def __call__(
        self,
        input_size: int,
        output_size: int,
        /,
        *,
        config: TransformerConfig,
        init_method: Callable[[torch.Tensor], None],
        bias: bool,
        input_is_parallel: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: str | None,
        tp_group: torch.distributed.ProcessGroup | None,
        name: str | None = None,
    ) -> LinearFc2Interface:
        """Builds a linear_fc2 layer for MLP."""
        ...


@dataclass
class MLPSubmodules:
    """
    The dataclass for ModuleSpecs of MLP submodules
    including  linear fc1, activation function, linear fc2.
    """

    linear_fc1: LinearFc1Builder

    linear_fc2: LinearFc2Builder

    activation_func: TEActivationFunctionBuilder | None = None
    """
    Builder for an activation function module; only used if config.use_te_activation_func is True.
    """


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
        input_size: Optional[int] = None,
        ffn_hidden_size: Optional[int] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        name: str | None = None,
    ):
        """
        Args:
            name (str | None): module instance name passed top-down from its paranet module
        """
        super().__init__(config=config)

        self.config: TransformerConfig = config

        self.input_size = input_size if input_size != None else self.config.hidden_size

        self.tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)
        if ffn_hidden_size is None:
            if is_expert:
                raise ValueError("MoE MLP requires `ffn_hidden_size`, but it was not provided.")
            warnings.warn(
                "MLP requires ffn_hidden_size, but it was not provided. Using \
                    config.ffn_hidden_size by default.",
                DeprecationWarning,
                stacklevel=2,
            )
            ffn_hidden_size = not_none(self.config.ffn_hidden_size)

        # If this is a gated linear unit we double the output width
        # see https://arxiv.org/pdf/2002.05202.pdf
        # For GLU/SwiGLU, use stride=2 because each TP rank stores interleaved [gate, up] portions.
        # This is critical for correct weight resharding across different TP sizes.
        if self.config.gated_linear_unit:
            ffn_hidden_size *= 2
            fc1_stride = 2
            if self.config.use_kitchen:
                # Kitchen Linear doesn't support stride != 1.
                # Weight resharding across TP sizes will have aforementioned problems.
                fc1_stride = 1
        else:
            fc1_stride = 1

        # Use moe_latent_size only for routed experts. 'is_expert' is false for
        # shared_experts.
        use_latent_size = (self.config.moe_latent_size is not None) and is_expert

        self.linear_fc1 = submodules.linear_fc1(
            self.input_size if not use_latent_size else not_none(self.config.moe_latent_size),
            ffn_hidden_size,
            config=self.config,
            init_method=not_none(self.config.init_method),
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name="fc1",
            tp_group=tp_group,
            stride=fc1_stride,
            name=(name + ".linear_fc1") if name is not None else None,
        )

        if self.config.use_te_activation_func and not (submodules.activation_func is None):
            self.activation_func = apply_module(submodules.activation_func(config=self.config))
        else:
            self.activation_func = self.config.activation_func

        self.linear_fc2 = submodules.linear_fc2(
            not_none(self.config.ffn_hidden_size),
            not_none(
                self.config.hidden_size if not use_latent_size else self.config.moe_latent_size
            ),
            config=self.config,
            init_method=not_none(self.config.output_layer_init_method),
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name="fc2",
            tp_group=tp_group,
            name=(name + ".linear_fc2") if name is not None else None,
        )

    def forward(
        self, hidden_states: torch.Tensor, per_token_scale: torch.Tensor | None = None, **kwargs
    ):
        """Perform the forward pass through the MLP block."""
        # [s, b, 4 * h/p]
        nvtx_range_push(suffix="linear_fc1")
        intermediate_parallel, bias_parallel = apply_module(self.linear_fc1)(hidden_states)
        nvtx_range_pop(suffix="linear_fc1")

        nvtx_range_push(suffix="activation")
        if self.config.use_te_activation_func:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            intermediate_parallel = self.activation_func(intermediate_parallel)
            if per_token_scale is not None:
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = intermediate_parallel * per_token_scale.unsqueeze(-1)
                intermediate_parallel = intermediate_parallel.to(original_dtype)
        elif self.config.bias_activation_fusion:
            if per_token_scale is not None:
                if self.activation_func == F.silu and self.config.gated_linear_unit:
                    # dtype is handled inside the fused kernel
                    intermediate_parallel = weighted_bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        per_token_scale.unsqueeze(-1),
                        self.config.activation_func_fp8_input_store,
                    )
                elif self.activation_func == quick_gelu and self.config.gated_linear_unit:
                    intermediate_parallel = weighted_bias_quick_geglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        per_token_scale.unsqueeze(-1),
                        self.config.activation_func_fp8_input_store,
                        self.config.glu_linear_offset,
                        self.config.activation_func_clamp_value,
                    )
                else:
                    raise ValueError(
                        "Only support fusion of swiglu and quick_gelu with per_token_scale in MLP."
                    )
            else:
                if self.activation_func == F.gelu:
                    if self.config.gated_linear_unit:
                        intermediate_parallel = bias_geglu_impl(
                            intermediate_parallel, bias_parallel
                        )
                    else:
                        assert self.config.add_bias_linear is True
                        intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
                elif self.activation_func == F.silu and self.config.gated_linear_unit:
                    intermediate_parallel = bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        self.config.activation_func_fp8_input_store,
                        self.config.cpu_offloading
                        and self.config.cpu_offloading_activations
                        and HAVE_TE,
                    )
                else:
                    raise ValueError("Only support fusion of gelu and swiglu")
        else:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
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

            if per_token_scale is not None:
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = intermediate_parallel * per_token_scale.unsqueeze(-1)
                intermediate_parallel = intermediate_parallel.to(original_dtype)
        nvtx_range_pop(suffix="activation")

        # [s, b, h]
        nvtx_range_push(suffix="linear_fc2")

        output, output_bias = apply_module(self.linear_fc2)(
            cast(torch.Tensor, intermediate_parallel)
        )
        nvtx_range_pop(suffix="linear_fc2")

        if per_token_scale is not None and output_bias is not None:
            # if this MLP is an expert, and bias is required, we add the bias to output directly
            # without doing bda later.
            output += output_bias.unsqueeze(0) * per_token_scale.unsqueeze(-1)
            output_bias = None

        return output, output_bias

    # pylint: disable=missing-function-docstring
    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """Return the sharded state dictionary of the module."""
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        sharded_state_dict = {}
        singleton_local_shards = (metadata or {}).get('singleton_local_shards', False)
        for name, module in self._modules.items():
            sub_sd = sharded_state_dict_default(
                module, f"{prefix}{name}.", sharded_offsets, metadata
            )
            if self.config.gated_linear_unit and name == "linear_fc1":
                for k, v in sub_sd.items():
                    if k in (f"{prefix}{name}.weight", f"{prefix}{name}.bias"):
                        sub_sd[k] = apply_swiglu_sharded_factory(
                            v,
                            sharded_offsets,
                            singleton_local_shards,
                            tp_group=self.tp_group,
                            dp_group=metadata['dp_cp_group'],
                        )
            sharded_state_dict.update(sub_sd)
        return sharded_state_dict

    def backward_dw(self):
        self.linear_fc2.backward_dw()
        self.linear_fc1.backward_dw()

    @classmethod
    def as_mlp_submodule(
        cls,
        submodules: MLPSubmodules,
        config: TransformerConfig,
        pg_collection: ProcessGroupCollection,
        is_mtp_layer: bool,
        is_expert: bool = False,
        input_size: int | None = None,
        ffn_hidden_size: int | None = None,
        name: str | None = None,
    ) -> MLP:
        """Helper function to build an MLP as a TransformerLayer's mlp submodule."""
        del is_mtp_layer
        assert hasattr(
            pg_collection, 'tp'
        ), 'TP process group is required for MLP in TransformerLayer'
        return cls(
            config=config,
            submodules=submodules,
            tp_group=pg_collection.tp,
            is_expert=is_expert,
            input_size=input_size,
            ffn_hidden_size=ffn_hidden_size,
            name=name,
        )


# pylint: disable=missing-function-docstring
def apply_swiglu_sharded_factory(
    original_sh_ten,
    sharded_offsets,
    singleton_local_shards: bool = False,
    tp_group: torch.distributed.ProcessGroup | None = None,
    dp_group: torch.distributed.ProcessGroup | None = None,
):
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
    axis_frag = original_sh_ten.axis_fragmentations[swiglu_shard_axis + prepend_axis_num]

    # Only FSDP2 supports torch_dist ShardedTensor. (Add other DP sharding algos here if needed.)
    is_torch_fsdp2_param = getattr(original_sh_ten, "is_torch_fsdp2_param", False)
    if is_torch_fsdp2_param:
        assert dp_group is not None
        dp_size = get_pg_size(dp_group)
        is_dp_sharded = dp_size > 1
    else:
        is_dp_sharded = False

    @torch.no_grad()
    def sh_ten_build_fn(
        key: str, t: torch.Tensor, replica_id: ReplicaId, flattened_range: Optional[slice]
    ):
        rank_offset = (
            original_sh_ten.global_offset[swiglu_shard_axis + prepend_axis_num] // local_axis_size
        )
        if singleton_local_shards:
            offset_w = (swiglu_shard_axis + prepend_axis_num, rank_offset, axis_frag)
            offset_v = (swiglu_shard_axis + prepend_axis_num, rank_offset, axis_frag)
            w_key = f'{key}_w'
            v_key = f'{key}_v'
        else:
            offset_w = (swiglu_shard_axis + prepend_axis_num, rank_offset, axis_frag * 2)
            offset_v = (
                swiglu_shard_axis + prepend_axis_num,
                rank_offset + axis_frag,
                axis_frag * 2,
            )
            w_key = key
            v_key = key

        tensor_w, tensor_v = torch.chunk(t, 2, dim=swiglu_shard_axis)
        return [
            ShardedTensor.from_rank_offsets(
                w_key,
                tensor_w,
                *sharded_offsets,
                offset_w,
                replica_id=replica_id,
                prepend_axis_num=prepend_axis_num,
            ),
            ShardedTensor.from_rank_offsets(
                v_key,
                tensor_v,
                *sharded_offsets,
                offset_v,
                replica_id=replica_id,
                prepend_axis_num=prepend_axis_num,
            ),
        ]

    @torch.no_grad()
    def dp_sh_ten_build_fn(
        key: str, t: torch.Tensor, replica_id: ReplicaId, flattened_range: Optional[slice]
    ):
        assert not singleton_local_shards, (
            "FSDP does not support singleton ShardedTensor for SwiGLU fused FC1. "
            "Set singleton_local_shards=False, which is the default in MCore."
        )
        # FSDP shards the TP-local [W; V] SwiGLU FC1 tensor over DP along dim 0.
        # TP sharding produces TP-sharded pairs of W/V, followed by DP sharding!
        assert tp_group is not None and dp_group is not None
        tp_size = get_pg_size(tp_group)
        global_axis = swiglu_shard_axis + prepend_axis_num
        tp_rank = get_pg_rank(tp_group)
        dp_rank = get_pg_rank(dp_group)
        # Size of a TP shard for W + V.
        tp_local_axis_size = original_sh_ten.global_shape[global_axis] // tp_size
        assert tp_local_axis_size % 2 == 0  # W and V should be symmetrically shaped.
        # Size of a TP shard for W or V. "Half" size TP-shard.
        half_axis_size = tp_local_axis_size // 2
        # Check that the TP-local W or V is cleanly divisible by DP.
        assert half_axis_size % local_axis_size == 0, (
            "SwiGLU FC1 FSDP ShardedTensor requires each DP shard of "
            "linear_fc1 to be completely inside either the W or V half."
        )
        # Number of DP shards per W or V TP-shard, and make sure
        # that DP sharding spans both W and V.
        shards_per_half = half_axis_size // local_axis_size
        assert dp_size == 2 * shards_per_half

        # Compute if DP rank maps to W or V in [W; V].
        swiglu_half_idx, half_dp_shard_idx = divmod(dp_rank, shards_per_half)
        # If W, then 0. If V, then 1.
        assert swiglu_half_idx in (0, 1)
        # Map [ W; V ] to this rank's shard [ {W_tpx; V_tpx}_dpy ].
        shard_rank_offset = (
            # W or V half of the [W; V] global data.
            swiglu_half_idx * tp_size * shards_per_half
            # TP Shard Index
            + tp_rank * shards_per_half
            # TP-DP Shard Index
            + half_dp_shard_idx
        )

        return [
            ShardedTensor.from_rank_offsets(
                key,
                t,
                *sharded_offsets,
                (global_axis, shard_rank_offset, axis_frag),
                replica_id=replica_id,
                prepend_axis_num=prepend_axis_num,
            )
        ]

    # Construct a ShardedTensorFactory.
    sh_ten_factory_build_function = dp_sh_ten_build_fn if is_dp_sharded else sh_ten_build_fn
    return ShardedTensorFactory(
        original_sh_ten.key,
        original_sh_ten.data,
        sh_ten_factory_build_function,
        cat_with_oom_fallback,
        original_sh_ten.replica_id,
        flattened_range=original_sh_ten.flattened_range,
    )
