# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import warnings
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Callable, Dict, Optional, Set, Tuple

import nvidia_kitchen
import torch
from nvidia_kitchen.config import QLinearParams, get_qlinear_params_from_qat_params

from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.models.backends import BackendSpecProvider
from megatron.core.parallel_state import (
    get_expert_data_parallel_rank,
    get_expert_model_parallel_rank,
    get_expert_model_parallel_world_size,
)
from megatron.core.quantization.quant_config import MatchContext, QuantizationConfig
from megatron.core.tensor_parallel.random import (
    get_cuda_rng_tracker,
    get_data_parallel_rng_tracker_name,
    get_expert_parallel_rng_tracker_name,
)
from megatron.core.tensor_parallel.utils import divide
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP, TEGroupedMLP
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from megatron.core.utils import get_tensor_model_parallel_group_if_none

# Parsing constant
_KITCHEN_CONFIG_TYPE_KEY = "kitchen_config_type"


class KitchenConfigType(Enum):
    """Configuration object types in config dictionary"""

    QLINEAR_PARAMS = "QLinearParams"
    # Could be extended with attention params e.g. QAttentionParams


@dataclass
class QLinearParamsConfigSchema:
    """Dataclass to parse values from config dict of 'QLinearParams' type"""

    kitchen_config_type: KitchenConfigType
    recipe_idx: int

    @classmethod
    def parse_config_dict(cls, config_dict: Dict[Any, Any]) -> 'QLinearParamsConfigSchema':
        """
        Parse config dictionary and return a schema instance.


        Expected config format: {"kitchen_config_type": "QLinearParams", "recipe_idx": <int>}
        """
        expected_keys = cls.get_expected_keys()
        actual_keys = set(config_dict.keys())

        # Check for missing keys
        missing = expected_keys - actual_keys
        if missing:
            raise KeyError(f"Missing required keys: {missing}")

        # Check for unexpected keys
        unexpected = actual_keys - expected_keys
        if unexpected:
            raise KeyError(f"Unexpected keys in config: {unexpected}")

        try:
            config_type = KitchenConfigType(config_dict[_KITCHEN_CONFIG_TYPE_KEY])
        except ValueError:
            raise ValueError(f"Unsupported config type '{config_dict['kitchen_config_type']}'.")

        if config_type != KitchenConfigType.QLINEAR_PARAMS:
            raise ValueError(f"Parsing config dict of incorrect type '{config_type}'")

        # Create instance with converted enum
        return cls(kitchen_config_type=config_type, recipe_idx=config_dict["recipe_idx"])

    @classmethod
    def get_expected_keys(cls) -> Set[str]:
        """Get expected keys from the dataclass fields."""
        return {field.name for field in fields(cls)}

    def __post_init__(self):
        # config type check
        if not isinstance(self.kitchen_config_type, KitchenConfigType):
            raise TypeError(
                "kitchen_config_type must be KitchenConfigType, "
                f"got {type(self.kitchen_config_type)}"
            )

        if self.kitchen_config_type != KitchenConfigType.QLINEAR_PARAMS:
            raise TypeError(
                f"kitchen_config_type must be QLinearParams got {self.kitchen_config_type}"
            )
        # recipe_idx check
        if not isinstance(self.recipe_idx, int) or self.recipe_idx <= 0:
            raise ValueError(f"recipe_idx must be a positive integer, got {self.recipe_idx}")

    def to_kitchen_qlinear(self) -> QLinearParams:
        """Converts to kitchen library's QLinearParams object."""
        return get_qlinear_params_from_qat_params(self.recipe_idx)


@dataclass
class KitchenQuantizationParams:
    """Quantization parameters used for kitchen extensions"""

    qlinear_params: Optional[QLinearParams]
    # Could be extended with attention params,
    # sparsity, etc.
    # match_input is what selected the config.
    match_input: MatchContext
    params_config_key: str

    @staticmethod
    def parse_from_config(quant_config: QuantizationConfig) -> "KitchenQuantizationParams":
        """Parses quantization config for a layer or throw an error."""
        assert (
            quant_config is not None
        ), "Kitchen extension expects a quantization config for linear layers."
        config = quant_config.config
        try:
            config_type = KitchenConfigType(config[_KITCHEN_CONFIG_TYPE_KEY])
        except KeyError:
            raise ValueError(
                f"Kitchen config dictionary must have '{_KITCHEN_CONFIG_TYPE_KEY}' key."
            )
        except ValueError:
            raise ValueError(f"Unsupported config type '{config['kitchen_config_type']}'.")

        if config_type == KitchenConfigType.QLINEAR_PARAMS:
            return KitchenQuantizationParams(
                qlinear_params=QLinearParamsConfigSchema.parse_config_dict(
                    config
                ).to_kitchen_qlinear(),
                match_input=quant_config.match_input,
                params_config_key=quant_config.config_key,
            )
        else:
            raise NotImplementedError(f"Unhandled configuration type {config_type}")


def _get_extra_kitchen_kwargs(config: TransformerConfig):
    extra_kitchen_kwargs = {"params_dtype": config.params_dtype}

    if config.use_cpu_initialization:
        raise ValueError("Kitchen backend does not support use_cpu_initialization.")
    elif config.init_model_with_meta_device:
        extra_kitchen_kwargs["device"] = "meta"
    else:
        extra_kitchen_kwargs["device"] = torch.cuda.current_device()
    return extra_kitchen_kwargs


class KitchenLinear(nvidia_kitchen.Linear):
    """
    Wrapper for Kitchen's `Linear` layer.

    Note that if Megatron's parallel_state has not been initialized
    yet, the tp_group passed to Kitchen will be None and must be set later
    via set_tensor_parallel_group().

    parallel_mode currently supports 3 different values:
        - "column": Split the weight matrix along output dimension (for KitchenColumnParallelLinear)
        - "row": Split the weight matrix along input dimension (for KitchenRowParallelLinear)
        - "duplicated": No tensor parallelism and weight is duplicated across TP ranks
        - Note: For expert linear layers, we will disable communication logic here
                as TP communication is handled in token_dispatcher.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        parallel_mode: Optional[str],
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        skip_bias_add: bool,
        skip_weight_param_allocation: bool,
        tp_comm_buffer_name: Optional[str] = None,
        layer_number: Optional[int] = None,
        is_expert: bool = False,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        self.config = config

        # Kitchen returns a zero length Tensor when bias=False and
        # return_bias=True, but we prefer None.  So in that case we
        # tell TE to not return the bias, and return None
        # ourselves. This way our forward always returns two values
        # and we don't have to deal with the zero length Tensor.
        self.kitchen_return_bias = skip_bias_add and bias
        self.is_first_microbatch = True
        self.disable_parameter_transpose_cache = self.config.disable_parameter_transpose_cache
        if skip_weight_param_allocation:
            raise ValueError('Kitchen linear layers do not support skip_weight_param_allocation')

        # Save params for finish_init
        self.stashed_input_size = input_size
        self.stashed_output_size = output_size
        self.stashed_parallel_mode = parallel_mode
        self.stashed_init_method = init_method
        self.stashed_bias = bias
        self.stashed_tp_comm_buffer_name = tp_comm_buffer_name
        self.stashed_layer_number = layer_number
        self.stashed_is_expert = is_expert
        self.stashed_tp_group = tp_group

        self.init_finished = False

    def finish_init(self, quantization_config: QuantizationConfig):
        """Required post-init of quantization configuration."""
        extra_kwargs = _get_extra_kitchen_kwargs(self.config)

        # Restore args from stash
        input_size = self.stashed_input_size
        output_size = self.stashed_output_size
        parallel_mode = self.stashed_parallel_mode
        init_method = self.stashed_init_method
        bias = self.stashed_bias
        tp_comm_buffer_name = self.stashed_tp_comm_buffer_name
        layer_number = self.stashed_layer_number
        is_expert = self.stashed_is_expert
        tp_group = self.stashed_tp_group

        self.kitchen_quant_params = KitchenQuantizationParams.parse_from_config(quantization_config)
        assert self.kitchen_quant_params.qlinear_params is not None
        extra_kwargs["qlinear_params"] = self.kitchen_quant_params.qlinear_params

        if tp_comm_buffer_name:
            self.config.tp_comm_overlap = False
            warnings.warn(
                f"The user buffer name {tp_comm_buffer_name} is not supported in "
                "Kitchen. Disabling TP communication overlap for this layer."
            )
            extra_kwargs["ub_name"] = tp_comm_buffer_name

        extra_kwargs["layer_number"] = layer_number

        if parallel_mode == "duplicated":
            assert tp_group is None, "duplicated linear should not have tp_group set"
            tp_size = 1
        else:
            assert tp_group is not None, "Parallel linear should always have tp_group set"
            tp_size = tp_group.size()

        self.expert_parallel = self.config.expert_model_parallel_size > 1
        if is_expert:
            rng_tracker_name = get_expert_parallel_rng_tracker_name()
        else:
            if parallel_mode == "duplicated":
                rng_tracker_name = get_data_parallel_rng_tracker_name()
            else:
                rng_tracker_name = None
        extra_kwargs["rng_tracker_name"] = rng_tracker_name

        kitchen_parallel_mode = parallel_mode
        if parallel_mode == "duplicated":
            # Handle non-parallel case
            tp_group = None
            tp_size = 1
            explicit_expert_comm = False
            kitchen_parallel_mode = None
        else:
            # Disable communications in kitchen when using TP or EP by megatron
            explicit_expert_comm = is_expert and (tp_size > 1 or self.expert_parallel)

            if explicit_expert_comm:
                if parallel_mode == "column":
                    output_size = divide(output_size, tp_size)
                elif parallel_mode == "row":
                    input_size = divide(input_size, tp_size)
                kitchen_parallel_mode = None
                tp_size = 1
                tp_group = None

        super().__init__(
            in_features=input_size,
            out_features=output_size,
            sequence_parallel=self.config.sequence_parallel,
            fuse_wgrad_accumulation=self.config.gradient_accumulation_fusion,
            # Pass None if not initialized for backward compatibility with the ckpt converter.
            tp_group=tp_group if torch.distributed.is_initialized() else None,
            tp_size=tp_size,
            get_rng_state_tracker=(
                get_cuda_rng_tracker if get_cuda_rng_tracker().is_initialized() else None
            ),
            init_method=(init_method if self.config.perform_initialization else (lambda w: None)),
            bias=bias,
            return_bias=self.kitchen_return_bias,
            parallel_mode=kitchen_parallel_mode,
            **extra_kwargs,
        )

        for param in self.parameters():
            if is_expert:
                # Reduce the gradient on the expert_data_parallel group for expert linear layers
                setattr(param, 'allreduce', not self.expert_parallel)
            else:
                # Reduce the gradient on DP group
                setattr(param, 'allreduce', True)
                if parallel_mode == "duplicated":
                    # Reduce the gradient further on the TP group since the weight is
                    # duplicated across TP ranks
                    setattr(param, 'sequence_parallel', self.config.sequence_parallel)

        del self.stashed_input_size
        del self.stashed_output_size
        del self.stashed_parallel_mode
        del self.stashed_init_method
        del self.stashed_bias
        del self.stashed_tp_comm_buffer_name
        del self.stashed_layer_number
        del self.stashed_is_expert
        del self.stashed_tp_group
        self.init_finished = True

    def forward(self, x):
        """Forward."""
        assert self.init_finished
        _is_first_microbatch = (
            None if self.disable_parameter_transpose_cache else self.is_first_microbatch
        )
        out = super().forward(x, is_first_microbatch=_is_first_microbatch)
        self.is_first_microbatch = False

        # Kitchen only returns a tuple when return_bias is True, otherwise
        # it returns a single Tensor, we always want to return two
        # values regardless of the arguments.
        if self.kitchen_return_bias:
            return out
        return out, None

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Replicate cross TP/DP."""

        # Provide the dist-ckpt support when KitchenLinear is directly used
        # It can only happen with duplicated parallel mode
        assert (
            self.parallel_mode is None
        ), "KitchenLinear sharded_state_dict can only be used with duplicated parallel mode"
        state_dict = self.state_dict(prefix='', keep_vars=True)
        return make_sharded_tensors_for_checkpoint(state_dict, prefix, None, sharded_offsets)


class KitchenColumnParallelLinear(KitchenLinear):
    """
    Wrapper for the Kitchen's `Linear` layer but specialized similar
    to megatron's `ColumnParallelLinear` layer.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        skip_weight_param_allocation: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
        layer_number: Optional[int] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        if gather_output:
            raise ValueError('Kitchen linear layers do not support gather_output = True')
        tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)
        world_size = tp_group.size()
        rank = tp_group.rank()

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            parallel_mode="column",
            config=config,
            init_method=(init_method if config.perform_initialization else (lambda w: None)),
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            skip_weight_param_allocation=skip_weight_param_allocation,
            tp_comm_buffer_name=tp_comm_buffer_name,
            layer_number=layer_number,
            tp_group=tp_group,
        )

        if config.use_cpu_initialization:
            raise ValueError("Kitchen extension doesn't support use_cpu_initialization.")

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Sharding along axis 0, bias sharded"""
        state_dict = self.state_dict(prefix='', keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {'weight': 0, 'bias': 0}, sharded_offsets
        )

    def __repr__(self):
        return (
            f"{type(self).__name__}(in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.use_bias}, TP={self.tp_size})"
        )


class KitchenRowParallelLinear(KitchenLinear):
    """
    Wrapper for Kitchen's `Linear` layer but specialized similar
    to megatron's `RowParallelLinear` layer.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        input_is_parallel: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: Optional[str] = None,
        layer_number: Optional[int] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        if not input_is_parallel:
            raise ValueError("Kitchen linear layers do not support input_is_parallel = False")
        tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            parallel_mode="row",
            config=config,
            init_method=(init_method if config.perform_initialization else (lambda w: None)),
            bias=bias,
            skip_bias_add=skip_bias_add,
            skip_weight_param_allocation=False,
            # We don't currently use this for row parallel layers # pylint: disable=line-too-long
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
            layer_number=layer_number,
            tp_group=tp_group,
        )
        if config.use_cpu_initialization:
            raise ValueError("Kitchen extension does not support use_cpu_initialization.")

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Sharding along axis 1, bias not sharded"""
        state_dict = self.state_dict(prefix='', keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {'weight': 1}, sharded_offsets
        )

    def __repr__(self):
        return (
            f"{type(self).__name__}(in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.use_bias}, TP={self.tp_size})"
        )


class KitchenGroupedLinear(nvidia_kitchen.GroupedLinear):
    """
    Wrapper for Kitchen's `GroupedLinear` layer.

    Note that if Megatron's parallel_state has not been initialized
    yet, the tp_group passed to TE will be None and must be set later
    via set_tensor_parallel_group().
    """

    def __init__(
        self,
        num_gemms: int,
        input_size: int,
        output_size: int,
        *,
        parallel_mode: Optional[str],
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
        layer_number: Optional[int] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):

        self.config = config

        # Kitchen returns a zero length Tensor when bias=False and
        # return_bias=True, but we prefer None.  So in that case we
        # tell TE to not return the bias, and return None
        # ourselves. This way our forward always returns two values
        # and we don't have to deal with the zero length Tensor.
        self.kitchen_return_bias = skip_bias_add and bias
        self.is_first_microbatch = True
        self.disable_parameter_transpose_cache = self.config.disable_parameter_transpose_cache

        # Stash parameters for finish_init
        self.stashed_num_gemms = num_gemms
        self.stashed_input_size = input_size
        self.stashed_output_size = output_size
        self.stashed_parallel_mode = parallel_mode
        self.stashed_init_method = init_method
        self.stashed_bias = bias
        self.stashed_is_expert = is_expert
        self.stashed_tp_comm_buffer_name = tp_comm_buffer_name
        self.stashed_layer_number = layer_number
        self.stashed_tp_group = tp_group
        self.init_finished = False

    def finish_init(self, quantization_config: QuantizationConfig) -> None:
        """Required post-init of quantization configuration."""
        # Restore parameters from stash
        num_gemms = self.stashed_num_gemms
        input_size = self.stashed_input_size
        output_size = self.stashed_output_size
        parallel_mode = self.stashed_parallel_mode
        init_method = self.stashed_init_method
        bias = self.stashed_bias
        is_expert = self.stashed_is_expert
        tp_comm_buffer_name = self.stashed_tp_comm_buffer_name
        layer_number = self.stashed_layer_number
        tp_group = self.stashed_tp_group

        extra_kwargs = _get_extra_kitchen_kwargs(self.config)
        extra_kwargs["ub_name"] = tp_comm_buffer_name
        extra_kwargs["layer_number"] = layer_number

        self.kitchen_quant_params = KitchenQuantizationParams.parse_from_config(quantization_config)
        assert self.kitchen_quant_params.qlinear_params is not None
        extra_kwargs["qlinear_params"] = self.kitchen_quant_params.qlinear_params

        self.expert_parallel = self.config.expert_model_parallel_size > 1
        if is_expert:
            extra_kwargs["rng_tracker_name"] = get_expert_parallel_rng_tracker_name()

        # The comms between TP and EP group is explicitly handled by MoE token dispatcher.
        # So we disable comms by making Kitchen agnostic of model parallel.
        tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)
        tp_size = tp_group.size()

        self.explicit_expert_comm = is_expert and (tp_size > 1 or self.expert_parallel)

        if self.explicit_expert_comm:
            if parallel_mode == "column":
                output_size = divide(output_size, tp_size)
            elif parallel_mode == "row":
                input_size = divide(input_size, tp_size)
            parallel_mode = None
            tp_size = 1
            tp_group = None

        super().__init__(
            num_gemms=num_gemms,
            in_features=input_size,
            out_features=output_size,
            sequence_parallel=self.config.sequence_parallel,
            fuse_wgrad_accumulation=self.config.gradient_accumulation_fusion,
            tp_group=tp_group if torch.distributed.is_initialized() else None,
            tp_size=tp_size,
            get_rng_state_tracker=(
                get_cuda_rng_tracker if get_cuda_rng_tracker().is_initialized() else None
            ),
            init_method=(init_method if self.config.perform_initialization else (lambda w: None)),
            bias=bias,
            return_bias=self.kitchen_return_bias,
            parallel_mode=parallel_mode,
            **extra_kwargs,
        )

        for param in self.parameters():
            setattr(param, 'allreduce', not (is_expert and self.expert_parallel))

        def merge_extra_states(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        ):
            """
            Merge multiple "_extra_state" into one.
            """
            self.init_fp8_metadata(num_gemms=self.num_gemms)
            fp8_checkpoint = self.fp8_meta["fp8_checkpoint"] or self.fp8 or self.fp8_calibration

            try:
                state_list = [
                    state_dict.pop(f"{prefix}_extra_state{i}") for i in range(1, self.num_gemms)
                ]
            except KeyError:
                # "_extra_state{i}" only exists for dist-ckpt. Return for torch native ckpt.
                return

            if not fp8_checkpoint:
                return
            state_list = [state_dict.pop(f"{prefix}_extra_state")] + state_list
            state_list = [self._decode_extra_state(state) for state in state_list]
            extra_fp8_variables = state_list[0]['extra_fp8_variables']
            extra_fp8_variables['num_gemms'] = self.num_gemms
            extra_state = {"extra_fp8_variables": extra_fp8_variables}
            state_dict[f"{prefix}_extra_state"] = self._encode_extra_state(extra_state)

        self._register_load_state_dict_pre_hook(merge_extra_states, with_module=True)
        del self.stashed_num_gemms
        del self.stashed_input_size
        del self.stashed_output_size
        del self.stashed_parallel_mode
        del self.stashed_init_method
        del self.stashed_bias
        del self.stashed_is_expert
        del self.stashed_tp_comm_buffer_name
        del self.stashed_layer_number
        del self.stashed_tp_group
        self.init_finished = True

    def forward(self, x, m_splits):
        """Forward."""
        assert self.init_finished
        _is_first_microbatch = (
            None if self.disable_parameter_transpose_cache else self.is_first_microbatch
        )
        out = super().forward(x, m_splits, is_first_microbatch=_is_first_microbatch)
        self.is_first_microbatch = False

        # Kitchen only returns a tuple when return_bias is True, otherwise
        # it returns a single Tensor, we always want to return two
        # values regardless of the arguments.
        if self.kitchen_return_bias:
            return out
        return out, None

    def _encode_extra_state(self, state):
        torch.cuda.synchronize()
        state_serialized = bytearray(pickle.dumps(state))
        state_serialized = torch.frombuffer(state_serialized, dtype=torch.uint8)
        return state_serialized

    def _decode_extra_state(self, state):
        if isinstance(state, torch.Tensor):
            return pickle.loads(state.detach().cpu().numpy().tobytes())
        elif isinstance(state, io.BytesIO):
            state.seek(0)
            return torch.load(state, map_location="cuda")
        else:
            raise RuntimeError("Unsupported checkpoint format.")

    def _split_extra_state(self, state):
        fp8_checkpoint = self.fp8_meta["fp8_checkpoint"]
        # Kitchen is compatible with TE checkpoint format, but never
        # uses fp8_checkpoints.
        assert not fp8_checkpoint
        return [state] * self.num_gemms

    def _sharded_state_dict_grouped(
        self, tp_axis_map, prefix='', sharded_offsets=(), metadata=None
    ):
        """
        prefix should be module_name to make keys identical to sequetial ones.
        """
        assert self.init_finished
        sharded_state_dict = {}
        full_state_dict = self.state_dict(prefix='', keep_vars=True)
        num_global_experts = get_expert_model_parallel_world_size() * self.num_gemms
        local_expert_indices_offset = get_expert_model_parallel_rank() * self.num_gemms
        ep_axis = len(sharded_offsets)
        extra_states = self._split_extra_state(full_state_dict['_extra_state'])
        for gemm_idx in range(self.num_gemms):
            state_dict = {
                f'{gemm_idx}.weight': full_state_dict[f'weight{gemm_idx}'],
                f'{gemm_idx}._extra_state': extra_states[gemm_idx],
            }
            if self.use_bias:
                state_dict[f'{gemm_idx}.bias'] = full_state_dict[f'bias{gemm_idx}']
            sub_sd = make_sharded_tensors_for_checkpoint(
                state_dict,
                '',
                tp_axis_map,
                (
                    *sharded_offsets,
                    (ep_axis, local_expert_indices_offset + gemm_idx, num_global_experts),
                ),
            )
            # Remove expert layers indexing from sharded keys
            replace_prefix_for_sharding(sub_sd, f'{gemm_idx}.', prefix)
            sharded_state_dict.update(
                {
                    f'{prefix}weight{gemm_idx}': sub_sd[f'{gemm_idx}.weight'],
                    f'{prefix}_extra_state{"" if gemm_idx == 0 else gemm_idx}': sub_sd[
                        f'{gemm_idx}._extra_state'
                    ],
                }
            )
            if self.use_bias:
                sharded_state_dict[f'{prefix}bias{gemm_idx}'] = sub_sd[f'{gemm_idx}.bias']
        # Adjust replica ids - replication along DP modulo EP
        for k, sh_ten in sharded_state_dict.items():
            replica_id = sh_ten.replica_id
            assert (
                len(replica_id) == 3
            ), f'Expected replica_id for {k} to be in (PP, TP, DP) format, got: {replica_id}'
            if getattr(sh_ten, "is_data_parallel_fully_shard", False):
                edp_replica_id = 0
            else:
                edp_replica_id = get_expert_data_parallel_rank()
            sh_ten.replica_id = (*replica_id[:2], edp_replica_id)
        return sharded_state_dict


class KitchenColumnParallelGroupedLinear(KitchenGroupedLinear):
    """
    Wrapper for Kitchen's `GroupedLinear` layer but specialized
    to column-parallel style.
    """

    def __init__(
        self,
        num_gemms: int,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: Optional[str] = None,
        layer_number: Optional[int] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(
            num_gemms=num_gemms,
            input_size=input_size,
            output_size=output_size,
            parallel_mode="column",
            config=config,
            init_method=(init_method if config.perform_initialization else (lambda w: None)),
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
            layer_number=layer_number,
            tp_group=tp_group,
        )

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """
        For each gemm, sharding along axis 0, bias sharded.
        Assume sharded_offsets[-1] is the expert parallel offset.
        """
        tp_axis_map = {}
        for gemm_idx in range(self.num_gemms):
            tp_axis_map.update({f'{gemm_idx}.weight': 0, f'{gemm_idx}.bias': 0})
        return super()._sharded_state_dict_grouped(tp_axis_map, prefix, sharded_offsets, metadata)


class KitchenRowParallelGroupedLinear(KitchenGroupedLinear):
    """
    Wrapper for Kitchen's `GroupedLinear` layer but specialized
    to row-parallel style.
    """

    def __init__(
        self,
        num_gemms: int,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: Optional[str] = None,
        layer_number: Optional[int] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(
            num_gemms=num_gemms,
            input_size=input_size,
            output_size=output_size,
            parallel_mode="row",
            config=config,
            init_method=(init_method if config.perform_initialization else (lambda w: None)),
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
            layer_number=layer_number,
            tp_group=tp_group,
        )

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """
        For each gemm, sharding along axis 1, bias not sharded.
        Assume sharded_offsets[-1] is the expert parallel offset.
        """
        tp_axis_map = {f'{gemm_idx}.weight': 1 for gemm_idx in range(self.num_gemms)}
        return super()._sharded_state_dict_grouped(tp_axis_map, prefix, sharded_offsets, metadata)


class KitchenLayerNormColumnParallelLinear(nvidia_kitchen.LayerNormLinear):
    """
    Wrapper for Kitchen's `LayerNormLinear` layer that combines
    layernorm and linear layers
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        init_method: Callable,
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        skip_weight_param_allocation: bool = False,
        layer_number: Optional[int] = None,
        tp_comm_buffer_name: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        self.config = config

        if gather_output:
            raise ValueError('Kitchen linear layers do not support gather_output = True')

        if is_expert:
            raise ValueError('Kitchen linear layers do not yet support MoE')

        if skip_weight_param_allocation:
            raise ValueError('Kitchen linear layers do not support skip_weight_param_allocation')

        tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)
        # Kitchen returns a zero length Tensor when bias=False and
        # return_bias=True, but we prefer None.  So in that case we
        # tell Kitchen to not return the bias, and return None
        # ourselves. This way our forward always returns two values
        # and we don't have to deal with the zero length Tensor.
        self.kitchen_return_bias = skip_bias_add and bias
        self.is_first_microbatch = True
        self.disable_parameter_transpose_cache = self.config.disable_parameter_transpose_cache
        self.tp_size = tp_group.size()
        self.tp_rank = tp_group.rank()

        if self.config.tp_comm_overlap:
            raise ValueError("Kitchen LayerNormLinear does not support tp_comm_overlap")

        if self.config.symmetric_ar_type is not None:
            raise ValueError("Kitchen LayerNormLinear does not support symmetric all-reduce")

        if config.use_cpu_initialization:
            raise ValueError("Kitchen extension does not support use_cpu_initialization")

        # Stash parameters for finish_init.
        self.stashed_input_size = input_size
        self.stashed_output_size = output_size
        self.stashed_init_method = init_method
        self.stashed_gather_output = gather_output
        self.stashed_bias = bias
        self.stashed_skip_bias_add = skip_bias_add
        self.stashed_is_expert = is_expert
        self.stashed_skip_weight_param_allocation = skip_weight_param_allocation
        self.stashed_layer_number = layer_number
        self.stashed_tp_comm_buffer_name = tp_comm_buffer_name
        self.stashed_tp_group = tp_group
        self.init_finished = False

    def finish_init(self, quantization_config: QuantizationConfig) -> None:
        """Required post-init of quantization configuration."""
        # Restore parameters from stash
        input_size = self.stashed_input_size
        output_size = self.stashed_output_size
        init_method = self.stashed_init_method
        gather_output = self.stashed_gather_output
        bias = self.stashed_bias
        skip_bias_add = self.stashed_skip_bias_add
        is_expert = self.stashed_is_expert
        skip_weight_param_allocation = self.stashed_skip_weight_param_allocation
        layer_number = self.stashed_layer_number
        tp_comm_buffer_name = self.stashed_tp_comm_buffer_name
        tp_group = self.stashed_tp_group

        extra_kwargs = _get_extra_kitchen_kwargs(self.config)
        extra_kwargs["normalization"] = self.config.normalization
        self.kitchen_quant_params = KitchenQuantizationParams.parse_from_config(quantization_config)
        assert self.kitchen_quant_params.qlinear_params is not None
        extra_kwargs["qlinear_params"] = self.kitchen_quant_params.qlinear_params
        extra_kwargs["ub_name"] = tp_comm_buffer_name

        super().__init__(
            in_features=input_size,
            out_features=output_size,
            eps=self.config.layernorm_epsilon,
            sequence_parallel=self.config.sequence_parallel,
            fuse_wgrad_accumulation=self.config.gradient_accumulation_fusion,
            tp_group=tp_group if torch.distributed.is_initialized() else None,
            tp_size=self.config.tensor_model_parallel_size,
            get_rng_state_tracker=(
                get_cuda_rng_tracker if get_cuda_rng_tracker().is_initialized() else None
            ),
            init_method=(init_method if self.config.perform_initialization else (lambda w: None)),
            bias=bias,
            return_bias=self.kitchen_return_bias,
            parallel_mode="column",
            return_layernorm_output=False,
            zero_centered_gamma=self.config.layernorm_zero_centered_gamma,
            layer_number=layer_number,
            **extra_kwargs,
        )
        del self.stashed_input_size
        del self.stashed_output_size
        del self.stashed_init_method
        del self.stashed_gather_output
        del self.stashed_bias
        del self.stashed_skip_bias_add
        del self.stashed_is_expert
        del self.stashed_skip_weight_param_allocation
        del self.stashed_layer_number
        del self.stashed_tp_comm_buffer_name
        del self.stashed_tp_group
        self.init_finished = True

    def forward(self, x):
        """Forward."""
        assert self.init_finished
        _is_first_microbatch = (
            None if self.disable_parameter_transpose_cache else self.is_first_microbatch
        )
        out = super().forward(x, is_first_microbatch=_is_first_microbatch)
        self.is_first_microbatch = False

        # Kitchen only returns a tuple when return_bias is True, otherwise
        # it returns a single Tensor, we always want to return two
        # values regardless of the arguments.
        if self.kitchen_return_bias:
            return out
        return out, None

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Sharding along axis 0, bias sharded"""
        assert self.init_finished
        state_dict = self.state_dict(prefix='', keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {'weight': 0, 'bias': 0}, sharded_offsets
        )

    def __repr__(self):
        return (
            f"{type(self).__name__}(in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.use_bias}, TP={self.tp_size})"
        )


class KitchenSpecProvider(BackendSpecProvider):
    """A protocol for providing the submodules used in Spec building."""

    def __init__(self, fallback: BackendSpecProvider):
        self.fallback = fallback

    def column_parallel_linear(self) -> type:
        """Which column parallel linear module kitchen backend uses"""
        return KitchenColumnParallelLinear

    def row_parallel_linear(self) -> type:
        """Which row parallel linear module kitchen backend uses"""
        return KitchenRowParallelLinear

    def fuse_layernorm_and_linear(self) -> bool:
        """Does kitchen backend support a single module for layernorm and linear"""
        # NOTE(kwyss): This is coupled with get_mlp_module_spec_for_backend and
        # the initialization of TransformerLayerSubmodules such as in
        # get_gpt_layer_local_spec or get_gpt_layer_with_transformer_engine_spec
        # where an explicit norm may be provided. Kitchen extension chooses to
        # match the topology of the fallback with this code.
        # Arguably, we should pass the info down to get_mlp_module_spec_for_backend
        # explicitly about whether to include a norm.
        return self.fallback.fuse_layernorm_and_linear()

    def column_parallel_layer_norm_linear(self) -> Optional[type]:
        """Which module for sequential layernorm and linear"""
        return KitchenLayerNormColumnParallelLinear

    def layer_norm(self, rms_norm: bool = False, for_qk: bool = False) -> type:
        """Which module to use for layer norm"""
        return self.fallback.layer_norm(rms_norm=rms_norm, for_qk=for_qk)

    def core_attention(self) -> type:
        """Which module to use for attention"""
        return self.fallback.core_attention()

    def grouped_mlp_modules(
        self, moe_use_grouped_gemm: bool, moe_use_legacy_grouped_gemm: bool
    ) -> Tuple[type, Optional[MLPSubmodules]]:
        """Which module and submodules to use for grouped mlp"""
        if moe_use_grouped_gemm and not moe_use_legacy_grouped_gemm:
            # NOTE: TEGroupedMLP is a bit of a misnomer.
            # It doesn't strictly require TE except for the GroupedLinear,
            # which Kitchen also provides an implementation of.
            return TEGroupedMLP, MLPSubmodules(
                linear_fc1=KitchenColumnParallelGroupedLinear,
                linear_fc2=KitchenRowParallelGroupedLinear,
            )
        elif moe_use_grouped_gemm:
            warnings.warn(
                'The legacy GroupedMLP will be deprecated in Megatron-Core v0.12.0. '
                'Please update the TransformerEngine to version>=1.7.0 and use TEGroupedMLP.'
            )
            return GroupedMLP, None
        else:
            return SequentialMLP, MLPSubmodules(
                linear_fc1=KitchenColumnParallelLinear, linear_fc2=KitchenRowParallelLinear
            )
