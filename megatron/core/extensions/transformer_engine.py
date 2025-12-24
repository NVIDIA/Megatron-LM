# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import dataclasses
import enum
import inspect
import io
import os
import pickle
import warnings
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from packaging.version import Version as PkgVersion
from torch import Tensor
from torch.nn.parameter import Parameter

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.enums import Fp4Recipe, Fp8Recipe
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_amax_reduction_group,
    get_context_parallel_group,
    get_hierarchical_context_parallel_groups,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_world_size,
    model_parallel_is_initialized,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.quantization.quant_config import QuantizationConfig
from megatron.core.tensor_parallel.layers import (
    _initialize_affine_weight_cpu,
    set_tensor_model_parallel_attributes,
)
from megatron.core.tensor_parallel.random import (
    get_cuda_rng_tracker,
    get_data_parallel_rng_tracker_name,
    get_expert_parallel_rng_tracker_name,
)
from megatron.core.tensor_parallel.utils import divide
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    is_layer_window_attention,
    make_sharded_tensors_for_checkpoint,
)
from megatron.core.utils import (
    get_pg_rank,
    get_pg_size,
    get_te_version,
    get_tensor_model_parallel_group_if_none,
    is_te_min_version,
    is_torch_min_version,
)

try:
    import transformer_engine as te
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager, fp8_autocast

    HAVE_TE = True
except ImportError:
    from unittest.mock import MagicMock

    te = MagicMock()
    HAVE_TE = False

_TE_CONFIG_TYPE_KEY = "transformer_engine_config_type"


class TransformerEngineConfigType(enum.Enum):
    """Configuration object types in config dictionary"""

    TEQuantizationParams = "TEQuantizationParams"


@dataclasses.dataclass
class TEQuantizationRecipe:
    """Class to capture options for opening an autocast context in forward"""

    fp8_quantization_recipe: Optional[Fp8Recipe] = None
    """
    An FP8 quantization override if the module should use FP8.
    If no FP8 or FP4 quantization is configured, the recipe is execution
    in high-precision (BF16).
    """
    fp4_quantization_recipe: Optional[Fp4Recipe] = None
    """
    An FP4 quantization override if the module should use FP4.
    If no FP8 or FP4 quantization is configured, the recipe is execution
    in high-precision (BF16).
    """
    custom_recipe_factory: Optional[str] = None
    """The path to a custom recipe factory if a custom Fp4 or Fp8 recipe is configured"""
    fp8_format: str = "e4m3"
    """A format to select from an FP8Recipe"""
    override_quantized_autocast: bool = True
    """
    If the quantization autocast context for a targeted module is enabled,
    whether to override it and change (or disable) the quantization recipe.
    """
    override_nonquantized_autocast: bool = False
    """
    If the quantization autocast context for a targeted module is not enabled,
    whether to override it and enable a quantization recipe.
    """
    tp_only_amax_red: bool = False
    """
    If an amax reduction is applicable, such as in per-tensor quantization recipe,
    whether to reduce only along TP groups.
    """

    @classmethod
    def parse_from_config(cls, quant_config: Dict[Any, Any]) -> "TEQuantizationRecipe":
        """
        Parse config from quantization dictionary.
        """
        kwargs = {}
        class_keys = cls.get_config_keys()
        for field in class_keys:
            if field in quant_config:
                kwargs[field] = quant_config[field]
        for field in quant_config:
            if field not in class_keys:
                raise ValueError(f"Field '{field}' not valid for this configuration.")
        instance = TEQuantizationRecipe(**kwargs)
        if instance.fp8_quantization_recipe == Fp8Recipe.delayed:
            raise ValueError("Delayed scaling not in scope of te per-module quantization config.")
        if (
            instance.fp8_quantization_recipe is not None
            and instance.fp4_quantization_recipe is not None
        ):
            raise ValueError("fp8 and fp4 quantization settings are mutually exclusive.")
        if (
            instance.fp8_quantization_recipe == Fp8Recipe.custom
            or instance.fp4_quantization_recipe == Fp4Recipe.custom
        ):
            if instance.custom_recipe_factory is None:
                raise ValueError("custom fp8 or fp4 recipe requires custom_recipe_factory")
        return instance

    @classmethod
    def get_config_keys(cls) -> Set[str]:
        """Get expected keys from the dataclass fields."""
        return {field.name for field in dataclasses.fields(cls)}


@dataclasses.dataclass
class TEQuantizationParams:
    """Class to capture precision options for training and evaluation."""

    training_recipe: TEQuantizationRecipe
    """Precision override for when self.training is True"""
    evaluation_recipe: Optional[TEQuantizationRecipe]
    """
    Precision override for when self.training is False.
    If None, training_recipe is used.
    """

    @staticmethod
    def parse_from_config(quant_config: QuantizationConfig) -> "TEQuantizationParams":
        """Parses quantization config for a layer or throw an error."""
        config = quant_config.config
        try:
            config_type = TransformerEngineConfigType(config[_TE_CONFIG_TYPE_KEY])
        except KeyError:
            raise ValueError(
                f"TransformerEngine config dictionary must have '{_TE_CONFIG_TYPE_KEY}' key."
            )
        except ValueError:
            raise ValueError(f"Unsupported config type '{config[_TE_CONFIG_TYPE_KEY]}'.")

        if config_type == TransformerEngineConfigType.TEQuantizationParams:
            if 'training_recipe' not in config.keys():
                raise ValueError(
                    "TransformerEngine config dictionary must have 'training_recipe' key"
                )
            training_recipe = TEQuantizationRecipe.parse_from_config(config['training_recipe'])
            if 'evaluation_recipe' not in config.keys():
                evaluation_recipe = None
                assert len(config.keys()) == 2
            else:
                evaluation_recipe = TEQuantizationRecipe.parse_from_config(
                    config['evaluation_recipe']
                )
                assert len(config.keys()) == 3
            return TEQuantizationParams(
                training_recipe=training_recipe, evaluation_recipe=evaluation_recipe
            )
        else:
            raise NotImplementedError(f"Unhandled configuration type {config_type}")


def _get_fp8_autocast_for_quant_recipe(qrecipe: TEQuantizationRecipe):
    if FP8GlobalStateManager.is_fp8_enabled():
        if not qrecipe.override_quantized_autocast:
            return nullcontext()
    else:
        if not qrecipe.override_nonquantized_autocast:
            return nullcontext()

    if qrecipe.fp8_quantization_recipe is None and qrecipe.fp4_quantization_recipe is None:
        # Force BF16 for this layer and override autocast
        return fp8_autocast(enabled=False)
    else:
        amax_group = None
        if model_parallel_is_initialized():
            amax_group = get_amax_reduction_group(
                with_context_parallel=True, tp_only_amax_red=qrecipe.tp_only_amax_red
            )
        if (
            qrecipe.fp8_quantization_recipe == Fp8Recipe.custom
            or qrecipe.fp4_quantization_recipe == Fp4Recipe.custom
        ):
            from megatron.core.fp8_utils import _get_custom_recipe

            assert qrecipe.custom_recipe_factory is not None
            quant_recipe = _get_custom_recipe(qrecipe.custom_recipe_factory)
        elif qrecipe.fp8_quantization_recipe is not None:
            if qrecipe.fp8_format == "e4m3":
                fp8_format = te.common.recipe.Format.E4M3
            elif qrecipe.fp8_format == "hybrid":
                fp8_format = te.common.recipe.Format.HYBRID
            else:
                raise ValueError(f"Unhandled fp8_format {qrecipe.fp8_format}")

            if qrecipe.fp8_quantization_recipe == Fp8Recipe.tensorwise:
                quant_recipe = te.common.recipe.Float8CurrentScaling(fp8_format=fp8_format)
            elif qrecipe.fp8_quantization_recipe == Fp8Recipe.blockwise:
                quant_recipe = te.common.recipe.Float8BlockScaling(fp8_format=fp8_format)
            elif qrecipe.fp8_quantization_recipe == Fp8Recipe.mxfp8:
                quant_recipe = te.common.recipe.MXFP8BlockScaling(fp8_format=fp8_format)
            else:
                raise ValueError(f"Unhandled fp8 recipe: {qrecipe.fp8_quantization_recipe}")
        else:
            # Fp4 configured.
            if qrecipe.fp4_quantization_recipe == Fp4Recipe.nvfp4:
                quant_recipe = te.common.recipe.NVFP4BlockScaling()
            else:
                raise ValueError(f"Unhandled fp4 recipe: {qrecipe.fp8_quantization_recipe}")

        return fp8_autocast(enabled=True, fp8_recipe=quant_recipe, fp8_group=amax_group)


def _get_fp8_autocast_for_quant_params(qparams: TEQuantizationParams | None, training: bool):
    if qparams is None:
        return nullcontext()
    elif not training and qparams.evaluation_recipe is not None:
        return _get_fp8_autocast_for_quant_recipe(qparams.evaluation_recipe)
    else:
        return _get_fp8_autocast_for_quant_recipe(qparams.training_recipe)


def _get_should_context_be_quantized_recipe(
    qrecipe: TEQuantizationRecipe, is_original_context_quantized: bool
):
    if is_original_context_quantized:
        if not qrecipe.override_quantized_autocast:
            return is_original_context_quantized
    else:
        if not qrecipe.override_nonquantized_autocast:
            return is_original_context_quantized
    if qrecipe.fp8_quantization_recipe is None and qrecipe.fp4_quantization_recipe is None:
        # Force BF16 for this layer and override autocast
        return False
    else:
        return True


def _get_should_context_be_quantized_params(
    qparams: TEQuantizationParams | None, training: bool, is_context_quantized: bool
):
    if qparams is None:
        return is_context_quantized
    elif not training and qparams.evaluation_recipe is not None:
        return _get_should_context_be_quantized_recipe(
            qparams.evaluation_recipe, is_context_quantized
        )
    else:
        return _get_should_context_be_quantized_recipe(
            qparams.training_recipe, is_context_quantized
        )


def _get_extra_te_kwargs(config: TransformerConfig):
    extra_transformer_engine_kwargs = {"params_dtype": config.params_dtype}

    if is_te_min_version("0.12.0"):
        if config.use_cpu_initialization:
            extra_transformer_engine_kwargs["device"] = "cpu"
        elif config.init_model_with_meta_device:
            extra_transformer_engine_kwargs["device"] = "meta"
        else:
            extra_transformer_engine_kwargs["device"] = torch.cuda.current_device()
    return extra_transformer_engine_kwargs


def condition_init_method(config, init_method):
    """Condition TE init_method on config.perform_initialization."""
    return init_method if config.perform_initialization else (lambda w: None)


def split_te_layernorm_column_parallel_linear(
    fused_layer,
    config,
    init_method: Optional[callable] = None,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
):
    """
    Split a TELayerNormColumnParallelLinear into separate TENorm and TEColumnParallelLinear layers.

    Args:
        fused_layer: The fused TELayerNormColumnParallelLinear layer to split
        config: TransformerConfig to use for creating the new layers
        init_method: Initialization method for the linear layer (optional)
        tp_group: Tensor parallel group (optional)

    Returns:
        A tuple of (TENorm, TEColumnParallelLinear) with weights copied from the fused layer
    """

    # Extract dimensions from the fused layer
    in_features = fused_layer.in_features
    out_features = fused_layer.out_features * fused_layer.tp_size

    # Create the norm layer
    norm_layer = TENorm(config=config, hidden_size=in_features, eps=fused_layer.eps)

    with torch.no_grad():
        # Copy layer norm weight
        norm_layer.weight.copy_(fused_layer.layer_norm_weight)

        # Copy layer norm bias if it exists
        if hasattr(norm_layer, 'bias') and hasattr(fused_layer, 'layer_norm_bias'):
            if fused_layer.layer_norm_bias is not None:
                norm_layer.bias.copy_(fused_layer.layer_norm_bias)

    # Create the column parallel linear layer
    linear_layer = TEColumnParallelLinear(
        input_size=in_features,
        output_size=out_features,
        config=config,
        init_method=init_method or (lambda x: None),  # Dummy init since we'll copy weights
        gather_output=False,
        bias=fused_layer.use_bias,
        skip_bias_add=fused_layer.te_return_bias,
        is_expert=False,
        tp_comm_buffer_name=fused_layer.ub_name,
        tp_group=tp_group or fused_layer.tp_group,
    )

    with torch.no_grad():
        # Copy weight
        linear_layer.weight.copy_(fused_layer.weight)

        # Copy bias if it exists
        if fused_layer.use_bias and hasattr(fused_layer, 'bias'):
            linear_layer.bias.copy_(fused_layer.bias)

    # TODO(Peter): Do we need this
    # Copy FP8 metadata if applicable
    if hasattr(fused_layer, 'fp8_meta') and fused_layer.fp8_meta is not None:
        if hasattr(linear_layer, 'fp8_meta'):
            # Copy FP8 scaling factors and other metadata
            for key in fused_layer.fp8_meta:
                if key in linear_layer.fp8_meta:
                    if isinstance(fused_layer.fp8_meta[key], dict):
                        for subkey in fused_layer.fp8_meta[key]:
                            if subkey in linear_layer.fp8_meta[key]:
                                linear_layer.fp8_meta[key][subkey] = fused_layer.fp8_meta[key][
                                    subkey
                                ]
                    else:
                        linear_layer.fp8_meta[key] = fused_layer.fp8_meta[key]

    # Set the same configuration flags
    linear_layer.sequence_parallel = fused_layer.sequence_parallel
    linear_layer.is_first_microbatch = fused_layer.is_first_microbatch
    linear_layer.disable_parameter_transpose_cache = fused_layer.disable_parameter_transpose_cache

    return norm_layer, linear_layer


if HAVE_TE and is_te_min_version("1.13.0"):

    class TEActivationOp:
        """
        A conditional wrapper to initialize an instance of Transformer-Engine's activation
        function operators (e.g. Silu, SwiGLU, etc)
        """

        def __new__(cls, config: TransformerConfig):

            layer_type = None
            if config.gated_linear_unit:
                if config.activation_func == F.silu:
                    layer_type = te.pytorch.ops.SwiGLU
                elif config.activation_func == F.gelu:
                    layer_type = te.pytorch.ops.GEGLU
                elif config.activation_func == F.silu:
                    layer_type = te.pytorch.ops.ReGLU
            else:
                if config.activation_func == F.gelu:
                    layer_type = te.pytorch.ops.GELU
                elif config.activation_func == F.silu:
                    layer_type = te.pytorch.ops.ReLU
            if layer_type is None:
                raise Exception(
                    'Only SwiGLU, GEGLU, ReGLU, GELU, ReLU are supported by '
                    'transformer engine. Please set use_te_activation_func=False'
                )
            activation_func_kwargs = {}
            if config.activation_func_fp8_input_store:
                activation_func_kwargs["cache_quantized_input"] = True
            layer = layer_type(**activation_func_kwargs)
            return layer

else:
    TEActivationOp = None


class TENorm:
    """A conditional wrapper to initialize an instance of
    Transformer-Engine's `LayerNorm` or `RMSNorm` based on input."""

    # TODO should we ditch normalization config and just use spec to choose LayerNorm vs RMSNorm?
    def __new__(cls, config: TransformerConfig, hidden_size: int, eps: float = 1e-5):
        if not HAVE_TE:
            raise ImportError(
                "Transformer Engine is not installed. "
                "Please install it with `pip install transformer-engine`."
            )

        if config.normalization == "LayerNorm":
            instance = te.pytorch.LayerNorm(
                hidden_size=hidden_size,
                eps=eps,
                sequence_parallel=config.sequence_parallel,
                zero_centered_gamma=config.layernorm_zero_centered_gamma,
                **_get_extra_te_kwargs(config),
            )
        elif config.normalization == "RMSNorm":
            assert hasattr(
                te.pytorch, "RMSNorm"
            ), "Transformer-Engine >= v0.11 required to use this feature"
            instance = te.pytorch.RMSNorm(
                hidden_size=hidden_size,
                eps=eps,
                sequence_parallel=config.sequence_parallel,
                zero_centered_gamma=config.layernorm_zero_centered_gamma,
                **_get_extra_te_kwargs(config),
            )
        else:
            raise Exception("Only LayerNorm and RMSNorm are curently supported")

        return instance


class TELinear(te.pytorch.Linear):
    """Wrapper for the Transformer-Engine's `Linear` layer.

    Note that if Megatron's parallel_state has not been initialized
    yet, the tp_group passed to TE will be None and must be set later
    via set_tensor_parallel_group().

    parallel_mode currently supports 3 different values:
        - "column": Split the weight matrix along output dimension (used in TEColumnParallelLinear)
        - "row": Split the weight matrix along input dimension (used in TERowParallelLinear)
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
        is_expert: bool = False,
        symmetric_ar_type: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        if not HAVE_TE:
            raise ImportError(
                "Transformer Engine is not installed. "
                "Please install it with `pip install transformer-engine`."
            )

        self.config = config

        # TE returns a zero length Tensor when bias=False and
        # return_bias=True, but we prefer None.  So in that case we
        # tell TE to not return the bias, and return None
        # ourselves. This way our forward always returns two values
        # and we don't have to deal with the zero length Tensor.
        self.te_return_bias = skip_bias_add and bias
        self.is_first_microbatch = True
        self.disable_parameter_transpose_cache = self.config.disable_parameter_transpose_cache
        self.symmetric_ar_type = symmetric_ar_type
        if skip_weight_param_allocation:
            raise ValueError(
                "Transformer Engine linear layers do not support skip_weight_param_allocation"
            )

        extra_kwargs = _get_extra_te_kwargs(config)

        if self.config.delay_wgrad_compute:
            if is_te_min_version("2.3.0"):
                extra_kwargs["delay_wgrad_compute"] = self.config.delay_wgrad_compute
            else:
                raise RuntimeError("Only TE with version >=2.3.0 supports delay_wgrad_compute now.")

        if (
            self.config.tp_comm_overlap
            and tp_comm_buffer_name
            and tp_comm_buffer_name not in ["qkv", "proj", "fc1", "fc2"]
        ):
            self.config.tp_comm_overlap = False
            warnings.warn(
                f"The user buffer name {tp_comm_buffer_name} is not supported in"
                "Transformer Engine. Disabling TP communication overlap "
                "for this layer."
            )

        if is_te_min_version("0.8.0"):
            if self.config.tp_comm_overlap and parallel_mode != "duplicated":
                if is_te_min_version("1.5.0"):
                    # Use old overlap flags if they were supplied instead
                    extra_kwargs["ub_overlap_ag"] = (
                        self.config.tp_comm_overlap_ag
                        if hasattr(self.config, "tp_comm_overlap_ag")
                        else self.config.tp_comm_split_ag or self.config.tp_comm_atomic_ag
                    )
                    extra_kwargs["ub_overlap_rs"] = (
                        self.config.tp_comm_overlap_rs
                        if hasattr(self.config, "tp_comm_overlap_rs")
                        else self.config.tp_comm_split_rs or self.config.tp_comm_atomic_rs
                    )
                    # Disable ub overlap for experts.
                    if is_expert:
                        extra_kwargs["ub_overlap_ag"] = False
                        extra_kwargs["ub_overlap_rs"] = False
                else:
                    extra_kwargs["ub_split_ag"] = self.config.tp_comm_split_ag
                    extra_kwargs["ub_atomic_gemm_ag"] = self.config.tp_comm_atomic_ag
                    extra_kwargs["ub_split_rs"] = self.config.tp_comm_split_rs
                    extra_kwargs["ub_atomic_gemm_rs"] = self.config.tp_comm_atomic_rs
                    # Disable ub overlap for experts.
                    if is_expert:
                        extra_kwargs["ub_split_ag"] = False
                        extra_kwargs["ub_atomic_gemm_ag"] = False
                        extra_kwargs["ub_split_rs"] = False
                        extra_kwargs["ub_atomic_gemm_rs"] = False
                if is_te_min_version("1.0.0", check_equality=False):
                    assert (
                        tp_comm_buffer_name is not None
                    ), "Buffer name should be set to configure communication overlap settings"
                    extra_kwargs["ub_name"] = tp_comm_buffer_name

        if symmetric_ar_type is not None:
            assert is_torch_min_version("2.7.0a0"), "Must have at least torch version 2.7 or higher"
            assert is_te_min_version("2.3.0") or get_te_version() == PkgVersion(
                "2.3.0.dev0+39c0e70"
            ), "Must have at least TE version 2.3 or higher to use symmetric memory all reduce"
            extra_kwargs["symmetric_ar_type"] = symmetric_ar_type
        if parallel_mode == "duplicated":
            assert tp_group is None, "duplicated linear should not have tp_group set"
            tp_size = 1
        else:
            tp_size = get_pg_size(tp_group)

        self.expert_parallel = self.config.expert_model_parallel_size > 1
        if is_expert:
            rng_tracker_name = get_expert_parallel_rng_tracker_name()
        else:
            if parallel_mode == "duplicated":
                rng_tracker_name = get_data_parallel_rng_tracker_name()
            else:
                rng_tracker_name = None
        if is_te_min_version("1.7.0"):
            extra_kwargs["rng_tracker_name"] = rng_tracker_name

        te_parallel_mode = parallel_mode
        tp_group_for_te = tp_group
        if parallel_mode == "duplicated":
            # Handle non-parallel case
            tp_group_for_te = None
            tp_size = 1
            explicit_expert_comm = False
            te_parallel_mode = None
        else:
            # Disable communications in TE when using TP or EP by
            explicit_expert_comm = is_expert and (tp_size > 1 or self.expert_parallel)

            if explicit_expert_comm:
                if parallel_mode == "column":
                    output_size = divide(output_size, tp_size)
                elif parallel_mode == "row":
                    input_size = divide(input_size, tp_size)
                te_parallel_mode = None
                tp_size = 1
                tp_group_for_te = None

        super().__init__(
            in_features=input_size,
            out_features=output_size,
            sequence_parallel=self.config.sequence_parallel,
            fuse_wgrad_accumulation=self.config.gradient_accumulation_fusion,
            # Pass None if not initialized for backward compatibility with the ckpt converter.
            tp_group=tp_group_for_te if torch.distributed.is_initialized() else None,
            tp_size=tp_size,
            get_rng_state_tracker=(
                get_cuda_rng_tracker if get_cuda_rng_tracker().is_initialized() else None
            ),
            init_method=condition_init_method(config, init_method),
            bias=bias,
            return_bias=self.te_return_bias,
            parallel_mode=te_parallel_mode,
            **extra_kwargs,
        )
        self.te_quant_params: Optional[TEQuantizationParams] = None

        for param in self.parameters():
            if is_expert:
                # Reduce the gradient on the expert_data_parallel group for expert linear layers
                setattr(param, "allreduce", not self.expert_parallel)
            else:
                # Reduce the gradient on DP group
                setattr(param, "allreduce", True)
                if parallel_mode == "duplicated":
                    # Reduce the gradient further on the TP group since the weight is
                    # duplicated across TP ranks
                    setattr(param, "sequence_parallel", self.config.sequence_parallel)

        tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)
        self._tp_group = tp_group

    def finish_init(self, quantization_config: QuantizationConfig):
        """Post-init of quantization override"""
        if quantization_config is None:
            self.te_quant_params = None
        else:
            self.te_quant_params = TEQuantizationParams.parse_from_config(quantization_config)

    def will_execute_quantized(self, is_context_quantized: bool) -> bool:
        """Returns whether the module is configured to execute quantized."""
        return _get_should_context_be_quantized_params(
            self.te_quant_params, self.training, is_context_quantized
        )

    def forward(self, x):
        """Forward."""
        _is_first_microbatch = (
            None if self.disable_parameter_transpose_cache else self.is_first_microbatch
        )
        quant_context = _get_fp8_autocast_for_quant_params(self.te_quant_params, self.training)

        with quant_context:
            out = super().forward(x, is_first_microbatch=_is_first_microbatch)
        self.is_first_microbatch = False

        # TE only returns a tuple when return_bias is True, otherwise
        # it returns a single Tensor, we always want to return two
        # values regardless of the arguments.
        if self.te_return_bias:
            return out
        return out, None

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Replicate cross TP/DP."""

        # Provide the dist-ckpt support when TELinear is directly used
        # It can only happen with duplicated parallel mode
        assert (
            self.parallel_mode is None
        ), "TELinear sharded_state_dict can only be used with duplicated parallel mode"
        state_dict = self.state_dict(prefix="", keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict,
            prefix,
            None,
            sharded_offsets,
            tp_group=self._tp_group,
            dp_cp_group=metadata["dp_cp_group"],
        )

    def backward_dw(self):
        """Compute weight gradients during the backward pass if delay_wgrad_compute is enabled."""
        if self.config.delay_wgrad_compute:
            super().backward_dw()


class TELayerNormColumnParallelLinear(te.pytorch.LayerNormLinear):
    """Wrapper for the Transformer-Engine's `LayerNormLinear` layer
    that combines layernorm and linear layers."""

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
        tp_comm_buffer_name: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        if not HAVE_TE:
            raise ImportError(
                "Transformer Engine is not installed. "
                "Please install it with `pip install transformer-engine`."
            )

        self.config = config

        if gather_output:
            raise ValueError("Transformer Engine linear layers do not support gather_output = True")

        if is_expert:
            raise ValueError("Transformer Engine linear layers do not yet support MoE")

        if skip_weight_param_allocation:
            raise ValueError(
                "Transformer Engine linear layers do not support skip_weight_param_allocation"
            )

        # TODO: For backward compatibility, remove in v0.15.
        tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)
        self._tp_group = tp_group

        # TE returns a zero length Tensor when bias=False and
        # return_bias=True, but we prefer None.  So in that case we
        # tell TE to not return the bias, and return None
        # ourselves. This way our forward always returns two values
        # and we don't have to deal with the zero length Tensor.
        self.te_return_bias = skip_bias_add and bias
        self.is_first_microbatch = True
        self.disable_parameter_transpose_cache = self.config.disable_parameter_transpose_cache
        extra_kwargs = _get_extra_te_kwargs(config)
        self.tp_size = get_pg_size(tp_group)
        self.tp_rank = get_pg_rank(tp_group)

        if self.config.delay_wgrad_compute:
            if is_te_min_version("2.3.0"):
                extra_kwargs["delay_wgrad_compute"] = self.config.delay_wgrad_compute
            else:
                raise RuntimeError("Only TE with version >=2.3.0 supports delay_wgrad_compute now.")

        # Only Transformer-Engine version >= 0.11.0 supports `RMSNorm`
        if is_te_min_version("0.11.0"):
            extra_kwargs["normalization"] = self.config.normalization
        elif self.config.normalization != "LayerNorm":
            te_version = get_te_version()
            raise ValueError(
                f"Transformer Engine v{te_version} does not support {self.config.normalization}."
            )

        if is_te_min_version("0.8.0"):
            if self.config.tp_comm_overlap:
                extra_kwargs["ub_bulk_wgrad"] = self.config.tp_comm_bulk_wgrad
                extra_kwargs["ub_bulk_dgrad"] = self.config.tp_comm_bulk_dgrad
                if is_te_min_version("1.5.0", check_equality=False):
                    # Use old overlap flags if they were supplied instead
                    extra_kwargs["ub_overlap_ag"] = (
                        self.config.tp_comm_overlap_ag
                        if hasattr(self.config, "tp_comm_overlap_ag")
                        else self.config.tp_comm_split_ag or self.config.tp_comm_atomic_ag
                    )
                    if is_te_min_version("1.6.0.dev0", check_equality=False):
                        extra_kwargs["ub_overlap_rs_dgrad"] = (
                            self.config.tp_comm_overlap_rs_dgrad
                            if hasattr(self.config, "tp_comm_overlap_rs_dgrad")
                            else False
                        )
                    if tp_comm_buffer_name == "qkv" and self.config.tp_comm_overlap_disable_qkv:
                        extra_kwargs["ub_overlap_ag"] = False
                        extra_kwargs["ub_overlap_rs_dgrad"] = False

                    if tp_comm_buffer_name == "fc1" and self.config.tp_comm_overlap_disable_fc1:
                        extra_kwargs["ub_overlap_ag"] = False
                        extra_kwargs["ub_overlap_rs_dgrad"] = False
                else:
                    extra_kwargs["ub_atomic_gemm_ag"] = self.config.tp_comm_atomic_ag
                    extra_kwargs["ub_split_ag"] = self.config.tp_comm_split_ag
                if is_te_min_version("1.0.0", check_equality=False):
                    assert (
                        tp_comm_buffer_name is not None
                    ), "Buffer name should be set to configure communication overlap settings"
                    extra_kwargs["ub_name"] = tp_comm_buffer_name

        if self.config.symmetric_ar_type is not None:
            assert is_torch_min_version("2.7.0a0"), "Must have at least torch version 2.7 or higher"
            assert is_te_min_version("2.3.0") or get_te_version() == PkgVersion(
                "2.3.0.dev0+39c0e70"
            ), "Must have at least TE version 2.3 or higher to use symmetric memory all reduce"
            extra_kwargs["symmetric_ar_type"] = self.config.symmetric_ar_type

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
            init_method=(
                condition_init_method(config, init_method)
                if not config.use_cpu_initialization
                else lambda w: None
            ),
            bias=bias,
            return_bias=self.te_return_bias,
            parallel_mode="column",
            return_layernorm_output=False,
            zero_centered_gamma=self.config.layernorm_zero_centered_gamma,
            **extra_kwargs,
        )
        self.te_quant_params: Optional[TEQuantizationParams] = None

        if config.use_cpu_initialization:
            output_size_per_partition = divide(output_size, self.tp_size)
            _ = _initialize_affine_weight_cpu(
                self.weight,
                output_size,
                input_size,
                output_size_per_partition,
                0,
                init_method=condition_init_method(config, init_method),
                stride=1,
                return_master_weight=False,
                rank=self.tp_rank,
                world_size=self.tp_size,
                skip_set_tensor_parallel_attributes=True,
            )
            if bias:
                self.bias = Parameter(
                    torch.empty(output_size_per_partition, dtype=config.params_dtype)
                )
                set_tensor_model_parallel_attributes(self.bias, True, 0, 1)
                with torch.no_grad():
                    self.bias.zero_()
                setattr(self.bias, "allreduce", True)

    def finish_init(self, quantization_config: QuantizationConfig):
        """Post-init of quantization override"""
        if quantization_config is None:
            self.te_quant_params = None
        else:
            self.te_quant_params = TEQuantizationParams.parse_from_config(quantization_config)

    def will_execute_quantized(self, is_context_quantized: bool) -> bool:
        """Returns whether the module is configured to execute quantized."""
        return _get_should_context_be_quantized_params(
            self.te_quant_params, self.training, is_context_quantized
        )

    def forward(self, x):
        """Forward."""
        _is_first_microbatch = (
            None if self.disable_parameter_transpose_cache else self.is_first_microbatch
        )
        quant_context = _get_fp8_autocast_for_quant_params(self.te_quant_params, self.training)

        with quant_context:
            out = super().forward(x, is_first_microbatch=_is_first_microbatch)

        self.is_first_microbatch = False

        # TE only returns a tuple when return_bias is True, otherwise
        # it returns a single Tensor, we always want to return two
        # values regardless of the arguments.
        if self.te_return_bias:
            return out
        return out, None

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Sharding along axis 0, bias sharded"""
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        state_dict = self.state_dict(prefix="", keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict,
            prefix,
            {"weight": 0, "bias": 0},
            sharded_offsets,
            tp_group=self._tp_group,
            dp_cp_group=metadata["dp_cp_group"],
        )

    def __repr__(self):
        return (
            f"{type(self).__name__}(in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.use_bias}, TP={self.tp_size})"
        )

    def backward_dw(self):
        """Compute weight gradients during the backward pass if delay_wgrad_compute is enabled."""
        if self.config.delay_wgrad_compute:
            super().backward_dw()


class TEColumnParallelLinear(TELinear):
    """Wrapper for the Transformer-Engine's `Linear` layer
    but specialized similar to megatron's `ColumnParallelLinear` layer."""

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
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        if not HAVE_TE:
            raise ImportError(
                "Transformer Engine is not installed. "
                "Please install it with `pip install transformer-engine`."
            )

        if gather_output:
            raise ValueError("Transformer Engine linear layers do not support gather_output = True")
        tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)
        self._tp_group = tp_group
        world_size = get_pg_size(tp_group)
        rank = get_pg_rank(tp_group)

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            parallel_mode="column",
            config=config,
            init_method=(
                condition_init_method(config, init_method)
                if not config.use_cpu_initialization
                else lambda w: None
            ),
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            skip_weight_param_allocation=skip_weight_param_allocation,
            tp_comm_buffer_name=tp_comm_buffer_name,
            symmetric_ar_type=config.symmetric_ar_type,
            tp_group=tp_group,
        )

        if config.use_cpu_initialization:
            output_size_per_partition = divide(output_size, world_size)
            _ = _initialize_affine_weight_cpu(
                self.weight,
                output_size,
                input_size,
                output_size_per_partition,
                0,
                init_method=condition_init_method(config, init_method),
                stride=1,
                return_master_weight=False,
                rank=rank,
                world_size=world_size,
                skip_set_tensor_parallel_attributes=True,
            )
            if bias:
                self.bias = Parameter(
                    torch.empty(output_size_per_partition, dtype=config.params_dtype)
                )
                set_tensor_model_parallel_attributes(self.bias, True, 0, 1)
                with torch.no_grad():
                    self.bias.zero_()
                setattr(self.bias, "allreduce", True)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Sharding along axis 0, bias sharded"""
        state_dict = self.state_dict(prefix="", keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict,
            prefix,
            {"weight": 0, "bias": 0},
            sharded_offsets,
            tp_group=self._tp_group,
            dp_cp_group=metadata["dp_cp_group"],
        )

    def __repr__(self):
        return (
            f"{type(self).__name__}(in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.use_bias}, TP={self.tp_size})"
        )

    def backward_dw(self):
        """Compute weight gradients during the backward pass if delay_wgrad_compute is enabled."""
        if self.config.delay_wgrad_compute:
            super().backward_dw()


class TERowParallelLinear(TELinear):
    """Wrapper for the Transformer-Engine's `Linear` layer
    but specialized similar to megatron's `RowParallelLinear` layer."""

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
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        if not HAVE_TE:
            raise ImportError(
                "Transformer Engine is not installed. "
                "Please install it with `pip install transformer-engine`."
            )

        if not input_is_parallel:
            raise ValueError(
                "Transformer Engine linear layers do not support input_is_parallel = False"
            )
        tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)
        self._tp_group = tp_group

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            parallel_mode="row",
            config=config,
            init_method=(
                condition_init_method(config, init_method)
                if not config.use_cpu_initialization
                else lambda w: None
            ),
            bias=bias,
            skip_bias_add=skip_bias_add,
            skip_weight_param_allocation=False,
            # We don't currently use this for row parallel layers # pylint: disable=line-too-long
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
            symmetric_ar_type=config.symmetric_ar_type,
            tp_group=tp_group,
        )
        if config.use_cpu_initialization:
            world_size = get_pg_size(tp_group)
            rank = get_pg_rank(tp_group)
            input_size_per_partition = divide(input_size, world_size)
            self.master_weight = _initialize_affine_weight_cpu(
                self.weight,
                output_size,
                input_size,
                input_size_per_partition,
                1,
                init_method=condition_init_method(config, init_method),
                stride=1,
                return_master_weight=False,
                params_dtype=config.params_dtype,
                rank=rank,
                world_size=world_size,
                skip_set_tensor_parallel_attributes=True,
            )
            if bias:
                self.bias = Parameter(torch.empty(output_size, dtype=config.params_dtype))
                # Always initialize bias to zero.
                with torch.no_grad():
                    self.bias.zero_()
                setattr(self.bias, "allreduce", True)
                setattr(self.bias, "sequence_parallel", config.sequence_parallel)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Sharding along axis 1, bias not sharded"""
        state_dict = self.state_dict(prefix="", keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict,
            prefix,
            {"weight": 1},
            sharded_offsets,
            tp_group=self._tp_group,
            dp_cp_group=metadata["dp_cp_group"],
        )

    def __repr__(self):
        return (
            f"{type(self).__name__}(in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.use_bias}, TP={self.tp_size})"
        )

    def backward_dw(self):
        """Compute weight gradients during the backward pass if delay_wgrad_compute is enabled."""
        if self.config.delay_wgrad_compute:
            super().backward_dw()


class TEDotProductAttention(te.pytorch.DotProductAttention):
    """Wrapper for the Transformer-Engine's `DotProductAttention` layer
    that also has "flash attention" enabled.

    Note that if Megatron's parallel_state has not been initialized yet, the
    tp_group and cp_group passed to TE will be None and must be set later
    via set_tensor_parallel_group() and set_context_parallel_group().
    """

    cp_stream: torch.cuda.Stream = None

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        k_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        num_splits: Optional[int] = None,
        cp_comm_type: str = "p2p",
        pg_collection: ProcessGroupCollection = None,
    ):
        if not HAVE_TE:
            raise ImportError(
                "Transformer Engine is not installed. "
                "Please install it with `pip install transformer-engine`."
            )

        self.config = config
        self.te_forward_mask_type = False
        self.qkv_format: str = "sbhd"
        # Default to 1 split when batch-invariant mode is enabled, unless explicitly overridden
        self.num_splits: Optional[int] = (
            1 if (num_splits is None and self.config.batch_invariant_mode) else num_splits
        )

        if self.config.apply_query_key_layer_scaling != bool(
            int(os.getenv("NVTE_APPLY_QK_LAYER_SCALING", "0"))
        ):
            raise ValueError(
                f"apply_query_key_layer_scaling is {self.config.apply_query_key_layer_scaling} "
                f"but environment variable NVTE_APPLY_QK_LAYER_SCALING is "
                f"{os.getenv('NVTE_APPLY_QK_LAYER_SCALING')}. Transformer Engine does not support "
                f"setting query key layer scaling via argument, so these two must match."
            )

        extra_kwargs: dict[str, Any] = {}
        if is_te_min_version("0.11.0"):
            extra_kwargs["num_gqa_groups"] = self.config.num_query_groups
        elif self.config.num_query_groups != self.config.num_attention_heads:
            raise ValueError(
                f"Transformer Engine v{get_te_version()} does not support Grouped Query Attention, "
                f"use a newer version of Transformer Engine. "
                f"(num_query_groups ({self.config.num_query_groups}) != "
                f"num_attention_heads ({self.config.num_attention_heads}))"
            )

        if pg_collection is None:
            pg_collection = ProcessGroupCollection(
                tp=get_tensor_model_parallel_group(check_initialized=False),
                cp=get_context_parallel_group(check_initialized=False),
                hcp=get_hierarchical_context_parallel_groups(check_initialized=False),
            )
        else:
            assert hasattr(
                pg_collection, "tp"
            ), "TEDotProductAttention pg_collection must have tp pg"
            assert hasattr(
                pg_collection, "cp"
            ), "TEDotProductAttention pg_collection must have cp pg"
            if cp_comm_type == "a2a+p2p":
                assert hasattr(
                    pg_collection, "hcp"
                ), "TEDotProductAttention pg_collection must have hierarchical cp pg"
        self._tp_group = pg_collection.tp

        if is_te_min_version("0.10.0"):
            extra_kwargs["attention_type"] = attention_type
            # older version don't need attention_type

        if is_te_min_version("0.12.0", check_equality=False):
            self.te_forward_mask_type = True

        # This check is important as CP config can be disabled while having a valid CP group
        # Example - Disabling CP for encoder while a valid CP group exists for decoder
        if self.config.context_parallel_size > 1:
            assert is_te_min_version(
                "1.0.0"
            ), "Only Transformer-Engine version >= 1.0.0 supports context parallelism!"
            if getattr(TEDotProductAttention, "cp_stream") is None:
                TEDotProductAttention.cp_stream = torch.cuda.Stream()
            extra_kwargs["cp_group"] = pg_collection.cp
            extra_kwargs["cp_global_ranks"] = torch.distributed.get_process_group_ranks(
                pg_collection.cp
            )
            extra_kwargs["cp_stream"] = TEDotProductAttention.cp_stream
            if is_te_min_version("1.10.0"):
                if cp_comm_type is None:
                    extra_kwargs["cp_comm_type"] = "p2p"
                elif cp_comm_type == "a2a+p2p":
                    assert is_te_min_version("1.12.0"), (
                        f"Transformer-Engine v{get_te_version()} must be >= 1.12.0 to support"
                        "hierarchical cp commucation."
                    )
                    extra_kwargs["cp_comm_type"] = "a2a+p2p"
                    extra_kwargs["cp_group"] = get_hierarchical_context_parallel_groups(
                        check_initialized=False
                    )
                else:
                    extra_kwargs["cp_comm_type"] = cp_comm_type

        if self.config.deterministic_mode:
            if int(os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")) != 0:
                raise RuntimeError(
                    "deterministic_mode is on and we are using DotProductAttention from "
                    "Transformer Engine, but NVTE_ALLOW_NONDETERMINISTIC_ALGO is not 0. "
                    f"Currently set to: {os.getenv('NVTE_ALLOW_NONDETERMINISTIC_ALGO', 'not set')}."
                )

        if is_layer_window_attention(
            config.window_size, config.window_attn_skip_freq, layer_number
        ):
            # Check version
            assert is_te_min_version("1.2.0"), (
                f"Transformer-Engine v{get_te_version()} must be >= 1.2.0 to support"
                "sliding window attention."
            )
            extra_kwargs["window_size"] = config.window_size

        if is_te_min_version("1.10.0"):
            # TE 1.10.0 introduces the ability to set the different k and v channels
            kv_channels = (
                (k_channels, v_channels)
                if k_channels is not None and v_channels is not None
                else self.config.kv_channels
            )
            extra_kwargs["softmax_scale"] = softmax_scale
        else:
            kv_channels = self.config.kv_channels

        if self.config.softmax_type != "vanilla":
            assert is_te_min_version("2.8.0"), (
                f"Transformer-Engine v{get_te_version()} must be >= 2.8.0 to support"
                "`softmax_type`."
            )
            extra_kwargs["softmax_type"] = self.config.softmax_type

        self.kept_packed_seq_params = set(
            field.name for field in dataclasses.fields(PackedSeqParams)
        )

        if get_te_version() < PkgVersion("1.3.0"):
            # TE 1.3.0 introduces precomputing max_seqlen to remove unnecessary kernels and D2H
            # copies (#555)
            # These two arguments did not exist prior to 1.3.0
            self.kept_packed_seq_params.discard("max_seqlen_q")
            self.kept_packed_seq_params.discard("max_seqlen_kv")

        if get_te_version() < PkgVersion("1.10.0"):
            # TE 1.8.0 introduces cu_seqlens_padded which is the cu_seqlens with paddings counted
            # in each individual sequence in THD format dataset
            # These two arguments did not exist prior to 1.8.0. Full support added in 1.10.0 (#1012)
            self.kept_packed_seq_params.discard("cu_seqlens_q_padded")
            self.kept_packed_seq_params.discard("cu_seqlens_kv_padded")

        if config.qk_clip or config.log_max_attention_logit:
            # qk-clip is only supported in TE 2.9.0 and later
            assert is_te_min_version("2.9.0"), "qk-clip is only supported in TE 2.9.0 and later"

            # TE 2.9.0 introduces return_max_logit for qk-clip getting the max attention logits
            extra_kwargs["return_max_logit"] = True
            self.current_max_attn_logits = None

        super().__init__(
            num_attention_heads=self.config.num_attention_heads,
            kv_channels=kv_channels,
            attention_dropout=(
                self.config.attention_dropout if attention_dropout is None else attention_dropout
            ),
            attn_mask_type=attn_mask_type.name,
            sequence_parallel=self.config.sequence_parallel,
            tp_size=self.config.tensor_model_parallel_size,
            get_rng_state_tracker=(
                get_cuda_rng_tracker if get_cuda_rng_tracker().is_initialized() else None
            ),
            tp_group=pg_collection.tp,
            layer_number=layer_number,
            **extra_kwargs,
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType,
        attention_bias: Tensor = None,
        packed_seq_params: PackedSeqParams = None,
        num_splits: Optional[int] = None,
    ):
        """Forward."""
        if packed_seq_params is not None:
            # If Dynamic CP group is provided, update TE DPA CP group
            if packed_seq_params.cp_group is not None:
                self.cp_group = packed_seq_params.cp_group
                super().set_context_parallel_group(
                    self.cp_group,
                    torch.distributed.get_process_group_ranks(self.cp_group),
                    TEDotProductAttention.cp_stream,
                    self.cp_comm_type,
                )
            # If cp_group is None but local_cp_size is provided,
            # Indicates to turn off CP dynamically
            elif packed_seq_params.local_cp_size is not None:
                assert (
                    packed_seq_params.local_cp_size == 1
                ), "local_cp_size must be == 1 if provided without cp_group"
                super().set_context_parallel_group(None, None, None, self.cp_comm_type)
            self.kept_packed_seq_params.discard("cp_group")
            self.kept_packed_seq_params.discard("local_cp_size")

        # Default to constructor-provided num_splits unless explicitly overridden
        if num_splits is None:
            num_splits = self.num_splits
        if num_splits is not None:
            assert is_te_min_version("2.10.0"), (
                f"Transformer-Engine v{get_te_version()} must be >= 2.10.0 to support" "num_splits."
            )

        packed_seq_kwargs = (
            {key: getattr(packed_seq_params, key) for key in self.kept_packed_seq_params}
            if packed_seq_params is not None
            else {}
        )
        qkv_format = packed_seq_kwargs.get('qkv_format', self.qkv_format)

        attention_bias_kwargs = {}
        if attention_bias is not None:
            assert is_te_min_version("1.2.0"), (
                f"Transformer-Engine v{get_te_version()} must be >= 1.2.0 to support"
                "`attention_bias`."
            )
            attention_bias_kwargs = dict(
                core_attention_bias_type="post_scale_bias", core_attention_bias=attention_bias
            )

        if attn_mask_type == AttnMaskType.no_mask and self.config.window_size is not None:
            if (qkv_format == "bshd" and query.size(1) == 1) or (
                qkv_format == "sbhd" and query.size(0) == 1
            ):
                #  need to change mask type for SWA inference decode stage.
                attn_mask_type = AttnMaskType.causal_bottom_right
        if self.te_forward_mask_type:
            if qkv_format == "thd" and is_te_min_version("1.7.0"):
                # thd format uses flash attention with cuDNN kernel which requires is_padding=True,
                # so the only acceptable mask types are `padding_causal` and `padding`. These do not
                # necessarily indicate there are padded tokens in the sequence.
                if attn_mask_type == AttnMaskType.causal:
                    attn_mask_type = AttnMaskType.padding_causal
                elif attn_mask_type == AttnMaskType.no_mask:
                    attn_mask_type = AttnMaskType.padding
            _fa_kwargs = dict(
                attn_mask_type=attn_mask_type.name, **attention_bias_kwargs, **packed_seq_kwargs
            )
            if num_splits is not None:
                _fa_kwargs["num_splits"] = num_splits

            core_attn_out = super().forward(query, key, value, attention_mask, **_fa_kwargs)

            if self.config.qk_clip or self.config.log_max_attention_logit:
                # qk-clip is only supported in TE 2.9.0 and later
                assert is_te_min_version("2.9.0"), "qk-clip is only supported in TE 2.9.0 and later"

                # Update Q K outside of TE Attention API
                core_attn_out, batch_max_attention_logits = core_attn_out

                # Update QK_Clip balancing eta
                if self.current_max_attn_logits is None:
                    self.current_max_attn_logits = batch_max_attention_logits
                else:
                    self.current_max_attn_logits = torch.max(
                        self.current_max_attn_logits, batch_max_attention_logits
                    )

        else:
            _fa_kwargs = dict(**attention_bias_kwargs, **packed_seq_kwargs)
            if num_splits is not None:
                _fa_kwargs["num_splits"] = num_splits
            core_attn_out = super().forward(query, key, value, attention_mask, **_fa_kwargs)

        return core_attn_out

    def sharded_state_dict(
        self,
        prefix: str = '',
        sharded_offsets: Tuple[Tuple[int, int, int]] = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """Sharded state dict for the learnable softmax offset parameter"""
        if self.config.softmax_type == "learnable":
            state_dict = self.state_dict(prefix="", keep_vars=True)
        else:
            state_dict = {}
        return make_sharded_tensors_for_checkpoint(
            state_dict,
            prefix,
            {'softmax_offset': 0},
            sharded_offsets,
            tp_group=self._tp_group,
            dp_cp_group=metadata["dp_cp_group"],
        )


if HAVE_TE and is_te_min_version("1.9.0.dev0"):

    class TEGroupedLinear(te.pytorch.GroupedLinear):
        """
        Wrapper for the Transformer-Engine's `GroupedLinear` layer.

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
            pg_collection: Optional[ProcessGroupCollection] = None,
        ):
            self.config = config

            # TE returns a zero length Tensor when bias=False and
            # return_bias=True, but we prefer None.  So in that case we
            # tell TE to not return the bias, and return None
            # ourselves. This way our forward always returns two values
            # and we don't have to deal with the zero length Tensor.
            self.te_return_bias = skip_bias_add and bias
            self.is_first_microbatch = True
            self.disable_parameter_transpose_cache = self.config.disable_parameter_transpose_cache

            extra_kwargs = _get_extra_te_kwargs(config)

            if self.config.delay_wgrad_compute:
                if is_te_min_version("2.3.0"):
                    extra_kwargs["delay_wgrad_compute"] = self.config.delay_wgrad_compute
                else:
                    raise RuntimeError(
                        "Only TE with version >=2.3.0 supports delay_wgrad_compute now."
                    )

            extra_kwargs["ub_name"] = tp_comm_buffer_name

            self.expert_parallel = self.config.expert_model_parallel_size > 1
            if is_expert:
                extra_kwargs["rng_tracker_name"] = get_expert_parallel_rng_tracker_name()

            # The comms between TP and EP group is explicitly handled by MoE token dispatcher.
            # So we disable comms by making TE agnostic of model parallel.
            if pg_collection is None:
                pg_collection = ProcessGroupCollection.use_mpu_process_groups()
            self._pg_collection = pg_collection
            assert is_expert, "TEGroupedLinear only supports expert parallelism"
            tp_group = pg_collection.expt_tp
            self._tp_group = tp_group
            tp_size = get_pg_size(tp_group)
            tp_group_for_te = tp_group

            self.explicit_expert_comm = is_expert and (tp_size > 1 or self.expert_parallel)

            if self.explicit_expert_comm:
                if parallel_mode == "column":
                    output_size = divide(output_size, tp_size)
                elif parallel_mode == "row":
                    input_size = divide(input_size, tp_size)
                parallel_mode = None
                tp_size = 1
                tp_group_for_te = None

            super().__init__(
                num_gemms=num_gemms,
                in_features=input_size,
                out_features=output_size,
                sequence_parallel=self.config.sequence_parallel,
                fuse_wgrad_accumulation=self.config.gradient_accumulation_fusion,
                tp_group=tp_group_for_te if torch.distributed.is_initialized() else None,
                tp_size=tp_size,
                get_rng_state_tracker=(
                    get_cuda_rng_tracker if get_cuda_rng_tracker().is_initialized() else None
                ),
                init_method=condition_init_method(config, init_method),
                bias=bias,
                return_bias=self.te_return_bias,
                parallel_mode=parallel_mode,
                **extra_kwargs,
            )
            self.te_quant_params: Optional[TEQuantizationParams] = None
            for param in self.parameters():
                setattr(param, "allreduce", not (is_expert and self.expert_parallel))

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
                # When resume training, loading ckpt is out of fp8_autocast context.
                # So we need to manually detect from the state_dict.
                fp8_checkpoint = any("_extra_state" in str(key) for key in state_dict.keys())

                if not fp8_checkpoint:
                    return

                try:
                    state_list = [
                        state_dict.pop(f"{prefix}_extra_state{i}") for i in range(1, self.num_gemms)
                    ]
                except KeyError:
                    # "_extra_state{i}" only exists for dist-ckpt. Return for torch native ckpt.
                    return

                # Early return conditions:
                # 1. Empty state_dict
                # 2. Empty state_list
                # 3. _extra_state is None
                # 4. _extra_state does not contain any information
                if (
                    not state_dict
                    or not state_list
                    or state_dict.get(f"{prefix}_extra_state") is None
                    or self._decode_extra_state(state_dict[f"{prefix}_extra_state"]) is None
                ):
                    return

                state_list = [state_dict.pop(f"{prefix}_extra_state")] + state_list
                state_list = [self._decode_extra_state(state) for state in state_list]
                extra_fp8_variables = state_list[0]["extra_fp8_variables"]
                extra_fp8_variables["num_gemms"] = self.num_gemms
                extra_state = {"extra_fp8_variables": extra_fp8_variables}
                # TE 2.0 adds recipe in extra_state
                if is_te_min_version("2.0.0"):
                    self.fp8_meta["recipe"] = state_list[0]["recipe"]
                    extra_state["recipe"] = self.fp8_meta["recipe"]
                # Only delayed scaling has global fp8 meta tensors. We're not using
                # self.fp8_meta["recipe"].delayed() because it's available in TE 2.0 and later.
                if isinstance(self.fp8_meta["recipe"], te.common.recipe.DelayedScaling):
                    extra_state.update(
                        {
                            "scale_fwd": torch.cat(
                                [state["scale_fwd"].view(-1, 1) for state in state_list], dim=1
                            ).view(-1),
                            "amax_history_fwd": torch.cat(
                                [state["amax_history_fwd"].view(-1, 1) for state in state_list],
                                dim=1,
                            ).view(self.fp8_meta["recipe"].amax_history_len, -1),
                            "scale_bwd": torch.cat(
                                [state["scale_bwd"].view(-1, 1) for state in state_list], dim=1
                            ).view(-1),
                            "amax_history_bwd": torch.cat(
                                [state["amax_history_bwd"].view(-1, 1) for state in state_list],
                                dim=1,
                            ).view(self.fp8_meta["recipe"].amax_history_len, -1),
                        }
                    )
                    # TE 2.0 removes scale_inv_fwd and scale_inv_bwd
                    if not is_te_min_version("2.0.0"):
                        extra_state.update(
                            {
                                "scale_inv_fwd": torch.cat(
                                    [state["scale_inv_fwd"].view(-1, 1) for state in state_list],
                                    dim=1,
                                ).view(-1),
                                "scale_inv_bwd": torch.cat(
                                    [state["scale_inv_bwd"].view(-1, 1) for state in state_list],
                                    dim=1,
                                ).view(-1),
                            }
                        )
                state_dict[f"{prefix}_extra_state"] = self._encode_extra_state(extra_state)

            self._register_load_state_dict_pre_hook(merge_extra_states, with_module=True)

        def finish_init(self, quantization_config: QuantizationConfig):
            """Post-init of quantization override"""
            if quantization_config is None:
                self.te_quant_params = None
            else:
                self.te_quant_params = TEQuantizationParams.parse_from_config(quantization_config)

        def will_execute_quantized(self, is_context_quantized: bool) -> bool:
            """Returns whether the module is configured to execute quantized."""
            return _get_should_context_be_quantized_params(
                self.te_quant_params, self.training, is_context_quantized
            )

        def forward(self, x, m_splits):
            """Forward."""
            _is_first_microbatch = (
                None if self.disable_parameter_transpose_cache else self.is_first_microbatch
            )
            quant_context = _get_fp8_autocast_for_quant_params(self.te_quant_params, self.training)

            with quant_context:
                out = super().forward(x, m_splits, is_first_microbatch=_is_first_microbatch)
            self.is_first_microbatch = False

            # TE only returns a tuple when return_bias is True, otherwise
            # it returns a single Tensor, we always want to return two
            # values regardless of the arguments.
            if self.te_return_bias:
                return out
            return out, None

        def _encode_extra_state(self, state):
            # TE 2.0 changed the format of extra_state to be a byte tensor
            if is_te_min_version("2.0.0"):
                torch.cuda.synchronize()
                state_serialized = bytearray(pickle.dumps(state))
                state_serialized = torch.frombuffer(state_serialized, dtype=torch.uint8)
            else:
                state_serialized = io.BytesIO()
                torch.save(state, state_serialized)
            return state_serialized

        def _decode_extra_state(self, state):
            if isinstance(state, torch.Tensor):
                # No FP8 is indicated by an empty tensor we don't need to unpickle.
                if state.numel() == 0:
                    return
                return pickle.loads(state.detach().cpu().numpy().tobytes())
            elif isinstance(state, io.BytesIO):
                state.seek(0)
                return torch.load(state, map_location="cuda")
            else:
                raise RuntimeError("Unsupported checkpoint format.")

        def _split_extra_state(self, state):
            fp8_checkpoint = self.fp8_meta["fp8_checkpoint"] or self.fp8 or self.fp8_calibration

            if not fp8_checkpoint:
                return [state] * self.num_gemms

            state = self._decode_extra_state(state)
            extra_states = []
            extra_fp8_variables = state["extra_fp8_variables"]
            extra_fp8_variables["num_gemms"] = 1
            for gemm_idx in range(self.num_gemms):
                tmp_state = {"extra_fp8_variables": extra_fp8_variables}
                # TE 2.0 adds recipe in extra_state
                if is_te_min_version("2.0.0"):
                    tmp_state["recipe"] = state["recipe"]
                # Only delayed scaling has global fp8 meta tensors. We're not using
                # self.fp8_meta["recipe"].delayed() because it's available in TE 2.0 and later.
                if isinstance(self.fp8_meta["recipe"], te.common.recipe.DelayedScaling):
                    tmp_state.update(
                        {
                            "scale_fwd": state["scale_fwd"].view(3, -1)[:, gemm_idx],
                            "amax_history_fwd": state["amax_history_fwd"].view(
                                self.fp8_meta["recipe"].amax_history_len, 3, -1
                            )[:, :, gemm_idx],
                            "scale_bwd": state["scale_bwd"].view(2, -1)[:, gemm_idx],
                            "amax_history_bwd": state["amax_history_bwd"].view(
                                self.fp8_meta["recipe"].amax_history_len, 2, -1
                            )[:, :, gemm_idx],
                        }
                    )
                    # TE 2.0 removes scale_inv_fwd and scale_inv_bwd
                    if not is_te_min_version("2.0.0"):
                        tmp_state.update(
                            {
                                "scale_inv_fwd": state["scale_inv_fwd"].view(3, -1)[:, gemm_idx],
                                "scale_inv_bwd": state["scale_inv_bwd"].view(2, -1)[:, gemm_idx],
                            }
                        )
                extra_states.append(self._encode_extra_state(tmp_state))
            return extra_states

        def _sharded_state_dict_grouped(
            self, tp_axis_map, prefix="", sharded_offsets=(), metadata=None
        ):
            """
            prefix should be module_name to make keys identical to sequetial ones.
            """
            singleton_local_shards = (metadata or {}).get('singleton_local_shards', False)
            sharded_state_dict = {}
            full_state_dict = self.state_dict(prefix="", keep_vars=True)
            num_global_experts = get_pg_size(self._pg_collection.ep) * self.num_gemms
            local_expert_indices_offset = get_pg_rank(self._pg_collection.ep) * self.num_gemms
            ep_axis = len(sharded_offsets)
            extra_states = self._split_extra_state(full_state_dict["_extra_state"])
            for gemm_idx in range(self.num_gemms):
                global_expert_idx = local_expert_indices_offset + gemm_idx
                state_dict = {
                    f"{gemm_idx}.weight": full_state_dict[f"weight{gemm_idx}"],
                    f"{gemm_idx}._extra_state": extra_states[gemm_idx],
                }
                if self.use_bias:
                    state_dict[f"{gemm_idx}.bias"] = full_state_dict[f"bias{gemm_idx}"]
                if singleton_local_shards:
                    expert_prefix = f"{global_expert_idx}.{prefix}"
                    new_sharded_offsets = sharded_offsets
                else:
                    expert_prefix = prefix
                    new_sharded_offsets = (
                        *sharded_offsets,
                        (ep_axis, global_expert_idx, num_global_experts),
                    )
                sub_sd = make_sharded_tensors_for_checkpoint(
                    state_dict,
                    '',
                    tp_axis_map,
                    new_sharded_offsets,
                    tp_group=self._tp_group,
                    dp_cp_group=metadata["dp_cp_group"],
                )
                # Remove expert layers indexing from sharded keys
                replace_prefix_for_sharding(sub_sd, f"{gemm_idx}.", expert_prefix)
                sharded_state_dict.update(
                    {
                        f"{prefix}weight{gemm_idx}": sub_sd[f"{gemm_idx}.weight"],
                        f"{prefix}_extra_state{'' if gemm_idx == 0 else gemm_idx}": sub_sd[
                            f"{gemm_idx}._extra_state"
                        ],
                    }
                )
                if self.use_bias:
                    sharded_state_dict[f"{prefix}bias{gemm_idx}"] = sub_sd[f"{gemm_idx}.bias"]
            # Adjust replica ids - replication along DP modulo EP
            for k, sh_ten in sharded_state_dict.items():
                replica_id = sh_ten.replica_id
                assert (
                    len(replica_id) == 3
                ), f"Expected replica_id for {k} to be in (PP, TP, DP) format, got: {replica_id}"
                if getattr(sh_ten, "is_data_parallel_fully_shard", False):
                    edp_replica_id = 0
                else:
                    edp_replica_id = get_pg_rank(self._pg_collection.expt_dp)
                sh_ten.replica_id = (*replica_id[:2], edp_replica_id)
            return sharded_state_dict

        def backward_dw(self):
            """
            Compute weight gradients during the backward pass
            if delay_wgrad_compute is enabled.
            """
            if self.config.delay_wgrad_compute:
                super().backward_dw()

    class TEColumnParallelGroupedLinear(TEGroupedLinear):
        """
        Wrapper for the Transformer-Engine's `GroupedLinear` layer but specialized
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
            pg_collection: Optional[ProcessGroupCollection] = None,
        ):
            super().__init__(
                num_gemms=num_gemms,
                input_size=input_size,
                output_size=output_size,
                parallel_mode="column",
                config=config,
                init_method=condition_init_method(config, init_method),
                bias=bias,
                skip_bias_add=skip_bias_add,
                is_expert=is_expert,
                tp_comm_buffer_name=tp_comm_buffer_name,
                pg_collection=pg_collection,
            )

        def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
            """
            For each gemm, sharding along axis 0, bias sharded.
            Assume sharded_offsets[-1] is the expert parallel offset.
            """
            tp_axis_map = {}
            for gemm_idx in range(self.num_gemms):
                tp_axis_map.update({f"{gemm_idx}.weight": 0, f"{gemm_idx}.bias": 0})
            return super()._sharded_state_dict_grouped(
                tp_axis_map, prefix, sharded_offsets, metadata
            )

    class TERowParallelGroupedLinear(TEGroupedLinear):
        """
        Wrapper for the Transformer-Engine's `GroupedLinear` layer but specialized
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
            pg_collection: Optional[ProcessGroupCollection] = None,
        ):
            super().__init__(
                num_gemms=num_gemms,
                input_size=input_size,
                output_size=output_size,
                parallel_mode="row",
                config=config,
                init_method=condition_init_method(config, init_method),
                bias=bias,
                skip_bias_add=skip_bias_add,
                is_expert=is_expert,
                tp_comm_buffer_name=tp_comm_buffer_name,
                pg_collection=pg_collection,
            )

        def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
            """
            For each gemm, sharding along axis 1, bias not sharded.
            Assume sharded_offsets[-1] is the expert parallel offset.
            """
            tp_axis_map = {f"{gemm_idx}.weight": 1 for gemm_idx in range(self.num_gemms)}
            return super()._sharded_state_dict_grouped(
                tp_axis_map, prefix, sharded_offsets, metadata
            )

else:
    TEGroupedLinear = None  # type: ignore[assignment, misc]
    TEColumnParallelGroupedLinear = None  # type: ignore[assignment, misc]
    TERowParallelGroupedLinear = None  # type: ignore[assignment, misc]


if HAVE_TE and is_te_min_version("1.13.0"):

    class TEFusedMLP(MLP):
        """MLP wrapper using Transformer Engine's operation-based API."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # Fused implementation
            self._fused_impl: Optional[Tuple[te.pytorch.ops.Sequential]] = None

        def _make_fused_impl(self) -> te.pytorch.ops.Sequential:
            """Construct fused module matching MLP."""

            # Container for fusible ops
            fused_impl = te.pytorch.ops.Sequential()

            # Tensor parallelism configuration
            tp_world_size = get_tensor_model_parallel_world_size()
            tp_group = None
            if tp_world_size > 1:
                tp_group = get_tensor_model_parallel_group()

            # RNG state
            rng_state_tracker_function = None
            if get_cuda_rng_tracker().is_initialized():
                rng_state_tracker_function = get_cuda_rng_tracker

            # Check submodule types
            if not isinstance(self.linear_fc1, te.pytorch.LayerNormLinear):
                raise ValueError(
                    f"{self.__class__.__name__} expects FC1 to be "
                    "Transformer Engine LayerNormLinear, but found "
                    f"{self.linear_fc1.__class__.__name__}."
                )
            if not isinstance(self.linear_fc2, te.pytorch.Linear):
                raise ValueError(
                    f"{self.__class__.__name__} expects FC1 to be "
                    "Transformer Engine Linear, but found "
                    f"{self.linear_fc2.__class__.__name__}."
                )

            # Norm op
            norm_type = self.linear_fc1.normalization
            norm_shape = self.linear_fc1.weight.size(1)
            kwargs = {
                "eps": self.linear_fc1.eps,
                "device": "meta",
                "dtype": self.linear_fc1.layer_norm_weight.dtype,
                "zero_centered_gamma": self.linear_fc1.zero_centered_gamma,
            }
            op = None
            if norm_type == "LayerNorm":
                op = te.pytorch.ops.LayerNorm(norm_shape, **kwargs)
                op.weight = self.linear_fc1.layer_norm_weight
                op.bias = self.linear_fc1.layer_norm_bias
            elif norm_type == "RMSNorm":
                op = te.pytorch.ops.RMSNorm(norm_shape, **kwargs)
                op.weight = self.linear_fc1.layer_norm_weight
            else:
                raise ValueError(f"Unsupported normalization ({norm_type})")
            fused_impl.append(op)

            # FC1 linear op
            weight = self.linear_fc1.weight
            userbuffers_options = None
            if self.linear_fc1.config.tp_comm_overlap and self.linear_fc1.ub_name is not None:
                userbuffers_options = {"comm_name": self.linear_fc1.ub_name}
            op = te.pytorch.ops.BasicLinear(
                weight.size(1),
                weight.size(0) * tp_world_size,
                device="meta",
                dtype=weight.dtype,
                tensor_parallel_mode="column" if tp_world_size > 1 else None,
                tensor_parallel_group=tp_group,
                sequence_parallel=self.linear_fc1.sequence_parallel,
                rng_state_tracker_function=rng_state_tracker_function,
                accumulate_into_main_grad=self.linear_fc1.fuse_wgrad_accumulation,
                userbuffers_options=userbuffers_options,
            )
            op.weight = weight
            fused_impl.append(op)

            # FC1 bias op
            bias = self.linear_fc1.bias
            if isinstance(bias, torch.Tensor) and bias.numel() == 0:
                bias = None
            if bias is not None:
                op = te.pytorch.ops.Bias(bias.numel(), device="meta", dtype=bias.dtype)
                op.bias = bias
                fused_impl.append(op)

            # Activation op
            op = self._make_activation_op(
                self.activation_func,
                self.config.gated_linear_unit,
                self.config.activation_func_fp8_input_store,
            )
            fused_impl.append(op)

            # FC2 linear op
            weight = self.linear_fc2.weight
            userbuffers_options = None
            if self.linear_fc2.config.tp_comm_overlap and self.linear_fc2.ub_name is not None:
                userbuffers_options = {"comm_name": self.linear_fc2.ub_name}
            op = te.pytorch.ops.BasicLinear(
                weight.size(1),
                weight.size(0),
                device="meta",
                dtype=weight.dtype,
                rng_state_tracker_function=rng_state_tracker_function,
                accumulate_into_main_grad=self.linear_fc2.fuse_wgrad_accumulation,
                userbuffers_options=userbuffers_options,
            )
            op.weight = weight
            fused_impl.append(op)
            if tp_world_size > 1:
                if self.linear_fc2.sequence_parallel:
                    fused_impl.append(te.pytorch.ops.ReduceScatter(tp_group))
                else:
                    fused_impl.append(te.pytorch.ops.AllReduce(tp_group))

            # FC2 bias op
            if not self.linear_fc2.te_return_bias:
                bias = self.linear_fc2.bias
                if isinstance(bias, torch.Tensor) and bias.numel() == 0:
                    bias = None
                if bias is not None:
                    op = te.pytorch.ops.Bias(bias.numel(), device="meta", dtype=bias.dtype)
                    op.bias = bias
                    fused_impl.append(op)

            # Emulate submodule forward hooks if needed
            self._register_hooks_on_fused_impl(fused_impl)

            return fused_impl

        def _make_activation_op(
            self, activation_func: Callable, gated_linear_unit: bool, cache_quantized_input: bool
        ) -> te.pytorch.ops.FusibleOperation:
            """Construct activation op."""

            # Get op type
            op_type = None
            if (activation_func, gated_linear_unit) == (F.gelu, False):
                op_type = te.pytorch.ops.GELU
            elif (activation_func, gated_linear_unit) == (F.gelu, True):
                op_type = te.pytorch.ops.GEGLU
            elif (activation_func, gated_linear_unit) == (F.silu, False):
                if not is_te_min_version("2.8.0"):
                    raise NotImplementedError("SiLU activation requires Transformer Engine 2.8+")
                op_type = te.pytorch.ops.SiLU
            elif (activation_func, gated_linear_unit) == (F.silu, True):
                op_type = te.pytorch.ops.SwiGLU
            elif (activation_func, gated_linear_unit) == (F.relu, False):
                op_type = te.pytorch.ops.ReLU
            elif (activation_func, gated_linear_unit) == (F.relu, True):
                op_type = te.pytorch.ops.ReGLU

            # Could not find corresponding activation op
            if op_type is None:
                raise NotImplementedError(
                    "Transformer Engine operation-based API does not support "
                    f"activation_func={activation_func}, "
                    f"gated_linear_unit={gated_linear_unit}"
                )

            # Construct op
            kwargs = {}
            if is_te_min_version("2.3"):
                kwargs["cache_quantized_input"] = cache_quantized_input
            return op_type(**kwargs)

        def _register_hooks_on_fused_impl(self, fused_impl: torch.nn.Module) -> None:
            """Attempt to emulate submodule callback hooks.

            This is not always possible because Transformer Engine's
            op fuser does not expose intermediate tensors. Depending
            on what kernel fusions the op fuser chooses, the
            intermediate tensors may not even exist. Hooks that modify
            tensors will result in incorrect behavior.

            """

            # Get submodule hooks
            forward_pre_hooks = []
            forward_post_hooks = []
            backward_pre_hooks = []
            backward_post_hooks = []
            for submodule in self.modules():
                for hook in submodule._forward_pre_hooks.values():
                    forward_pre_hooks.append((submodule, hook))
                for hook in submodule._forward_hooks.values():
                    forward_post_hooks.append((submodule, hook))
                for hook in submodule._backward_pre_hooks.values():
                    backward_pre_hooks.append((submodule, hook))
                for hook in submodule._backward_hooks.values():
                    backward_post_hooks.append((submodule, hook))

            # Pre-forward hooks
            # Note: DDP pre-forward hooks are safe since they do not
            # interact with input tensor.
            if forward_pre_hooks:
                from megatron.core.distributed import distributed_data_parallel

                if any(
                    inspect.getmodule(hook) != distributed_data_parallel
                    for _, hook in forward_pre_hooks
                ):
                    warnings.warn(
                        "TEFusedMLP module has a submodule with a pre-forward hook. "
                        "TEFusedMLP module does not expose intermediate tensors, "
                        "so the hook may have incorrect behavior if it attempts to "
                        "access the input tensor."
                    )

                def forward_pre_hook(module, *_) -> None:
                    for submodule, hook in forward_pre_hooks:
                        # Assume that hook does not interact with input
                        ret = hook(submodule, None)
                        if ret is not None:
                            raise RuntimeError(
                                "TEFusedMLP module does not expose intermediate tensors, but "
                                "submodule has pre-forward hook that modifies input tensor."
                            )

                fused_impl.register_forward_pre_hook(forward_pre_hook)

            # Post-forward hooks
            if forward_post_hooks:
                warnings.warn(
                    "TEFusedMLP module has a submodule with a post-forward hook. "
                    "TEFusedMLP module does not expose intermediate tensors, "
                    "so the hook may have incorrect behavior if it attempts to "
                    "access the input or output tensors."
                )

                def forward_post_hook(module, *_) -> None:
                    for submodule, hook in forward_post_hooks:
                        # Assume that hook does not interact with input or output
                        ret = hook(submodule, None, None)
                        if ret is not None:
                            raise RuntimeError(
                                "TEFusedMLP module does not expose intermediate tensors, but "
                                "submodule has post-forward hook that modifies output tensor."
                            )

                fused_impl.register_forward_hook(forward_post_hook)

            # Backward hooks
            if backward_pre_hooks:
                raise RuntimeError(
                    "TEFusedMLP module does not support submodules with pre-backward hooks"
                )
            if backward_post_hooks:
                raise RuntimeError(
                    "TEFusedMLP module does not support submodules with post-backward hooks"
                )

        def forward(self, hidden_states: torch.Tensor) -> Tuple[Tensor, Optional[Tensor]]:
            """Forward."""

            # Construct fused impl if needed
            # Note: We initialize during the first forward pass in
            # case the params are modified after the constructor.
            # Note: The fused impl is stored in a tuple to avoid
            # registering as a submodule.
            if self._fused_impl is None:
                self._fused_impl = (self._make_fused_impl(),)

            # Apply fused impl
            out = self._fused_impl[0](hidden_states)

            # Return bias tensor if requested
            bias = None
            if self.linear_fc2.te_return_bias:
                bias = self.linear_fc2.bias
                if isinstance(bias, torch.Tensor) and bias.numel() == 0:
                    bias = None

            return out, bias

else:
    TEFusedMLP = None  # type: ignore[assignment, misc]


class TEDelayedScaling(te.common.recipe.DelayedScaling):
    """
    Wrapper for the Transformer-Engine's `DelayedScaling` layer.
    """

    def __init__(
        self,
        config: ModelParallelConfig,
        fp8_format: int,
        override_linear_precision: tuple = (False, False, False),
    ):
        if not HAVE_TE:
            raise ImportError(
                "Transformer Engine is not installed. "
                "Please install it with `pip install transformer-engine`."
            )

        extra_kwargs = _get_extra_te_kwargs(config)
        if is_te_min_version("1.6.0.dev0"):
            extra_kwargs["fp8_dpa"] = config.fp8_dot_product_attention
            extra_kwargs["fp8_mha"] = config.fp8_multi_head_attention
        if get_te_version() < PkgVersion("1.8.0"):
            extra_kwargs["interval"] = config.fp8_interval
        elif config.fp8_interval != 1:
            warnings.warn("fp8_interval is deprecated and ignored from Transformer-Engine v1.8.0.")

        super().__init__(
            margin=config.fp8_margin,
            fp8_format=fp8_format,
            amax_compute_algo=config.fp8_amax_compute_algo,
            amax_history_len=config.fp8_amax_history_len,
            override_linear_precision=override_linear_precision,
            **extra_kwargs,
        )


class TECudaRNGStatesTracker(te.pytorch.distributed.CudaRNGStatesTracker):
    """Wraps TransformerEngine's CudaRNGStatesTracker so that it is
    interchangeable with Megatron's RNG tracker"""

    def __init__(self, is_inference_rng_tracker=False):
        if not HAVE_TE:
            raise ImportError(
                "Transformer Engine is not installed. "
                "Please install it with `pip install transformer-engine`."
            )

        super().__init__()
        self.reset()
        self.is_inference_rng_tracker = is_inference_rng_tracker

    def is_initialized(self):
        """Checks if the internal RNG state has been set with set_states()."""
        return self._is_initialized

    def reset(self):
        """Reset the internal RNG state."""
        super().reset()
        self._is_initialized = False

    def set_states(self, states):
        """Set the internal RNG state."""
        super().set_states(states)
        self._is_initialized = True

    def add(self, name, seed):
        """Track the rng state."""
        super().add(name, seed)
        self._is_initialized = True


def te_checkpoint(
    forward_func, distribute_saved_activations, get_rng_state_tracker, tp_group, *args, **kwargs
):
    """Checkpointing with Transformer-Engine."""
    if not HAVE_TE:
        raise ImportError(
            "Transformer Engine is not installed. "
            "Please install it with `pip install transformer-engine`."
        )

    from transformer_engine.pytorch.distributed import checkpoint

    if is_te_min_version("1.5.0"):
        return checkpoint(
            forward_func,
            *args,
            distribute_saved_activations=distribute_saved_activations,
            get_rng_state_tracker=get_rng_state_tracker,
            tp_group=tp_group,
            **kwargs,
        )
    else:
        return checkpoint(
            forward_func, distribute_saved_activations, get_rng_state_tracker, tp_group, *args
        )


try:
    from transformer_engine.pytorch.attention import _SplitAlongDim

    SplitAlongDim = _SplitAlongDim.apply

except ImportError:
    SplitAlongDim = None

try:
    from transformer_engine.pytorch.cpu_offload import (
        get_cpu_offload_context as _get_cpu_offload_context,
    )

    def get_cpu_offload_context(
        enabled,
        num_layers,
        model_layers,
        activation_offloading,
        weight_offloading,
        double_buffering,
    ):
        """Get CPU offload context and sync function."""
        if is_te_min_version("2.5.0"):
            # Enables the additional double buffering switch for activations during LLM training
            context, sync_func = _get_cpu_offload_context(
                enabled,
                num_layers,
                model_layers,
                activation_offloading,
                weight_offloading,
                double_buffering,
            )
        elif is_te_min_version("1.10.0.dev0"):
            context, sync_func = _get_cpu_offload_context(
                enabled, num_layers, model_layers, activation_offloading, weight_offloading
            )
        else:
            context, sync_func = _get_cpu_offload_context(
                enabled, num_layers, activation_offloading, weight_offloading
            )

        return context, sync_func

except ImportError:
    get_cpu_offload_context = None  # type: ignore[assignment, misc]

try:
    if HAVE_TE and is_te_min_version("2.3.0"):
        from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb
    else:
        from transformer_engine.pytorch.attention import apply_rotary_pos_emb

    def fused_apply_rotary_pos_emb(
        t: torch.Tensor,
        freqs: torch.Tensor,
        transpose_output_memory: bool = False,
        interleaved: bool = False,
    ) -> torch.Tensor:
        """Apply rotary positional embedding to input tensor T in `sbhd` format."""
        if transpose_output_memory:
            warnings.warn(
                "transpose_output_memory is not supported by TE's fused RoPE and will be ignored."
            )
        if is_te_min_version("2.3.0"):
            return apply_rotary_pos_emb(
                t, freqs, tensor_format="sbhd", interleaved=interleaved, fused=True
            )
        else:
            if interleaved:
                raise ValueError("Only TE >= 2.3.0 supports interleaved fused RoPE.")

            return apply_rotary_pos_emb(t, freqs, tensor_format="sbhd", fused=True)

    def fused_apply_rotary_pos_emb_thd(
        t: torch.Tensor,
        cu_seqlens: torch.Tensor,
        freqs: torch.Tensor,
        cp_size: int = 1,
        cp_rank: int = 0,
    ) -> torch.Tensor:
        """
        Apply rotary positional embedding to input tensor T in `thd` format with CP support.
        """
        if is_te_min_version("1.12.0", check_equality=True):
            return apply_rotary_pos_emb(
                t,
                freqs,
                tensor_format="thd",
                fused=True,
                cu_seqlens=cu_seqlens,
                cp_size=cp_size,
                cp_rank=cp_rank,
            )
        else:
            assert cp_size == 1, "Only TE >= 1.12 supports RoPE fusion for THD format with CP."
            return apply_rotary_pos_emb(
                t, freqs, tensor_format="thd", fused=True, cu_seqlens=cu_seqlens
            )

except ImportError:
    pass

try:
    from transformer_engine.pytorch import Fp8Padding, Fp8Unpadding  # pylint: disable=unused-import

except ImportError:
    Fp8Padding = None
    Fp8Unpadding = None

try:
    from transformer_engine.pytorch.permutation import (
        moe_permute,
        moe_permute_with_probs,
        moe_sort_chunks_by_index,
        moe_sort_chunks_by_index_with_probs,
        moe_unpermute,
    )

    fused_permute = moe_permute
    fused_permute_with_probs = moe_permute_with_probs
    fused_sort_chunks_by_index = moe_sort_chunks_by_index
    fused_sort_chunks_by_index_with_probs = moe_sort_chunks_by_index_with_probs
    fused_unpermute = moe_unpermute

except ImportError:
    fused_permute = None
    fused_permute_with_probs = None
    fused_sort_chunks_by_index = None
    fused_sort_chunks_by_index_with_probs = None
    fused_unpermute = None

try:
    from transformer_engine.pytorch.cross_entropy import parallel_cross_entropy

    _TE_SUPPORTS_CG_CAPTURABLE = is_te_min_version("2.7.0")
    current_te_version = get_te_version()

    def te_parallel_cross_entropy(
        logits: torch.Tensor,
        labels: torch.Tensor,
        tp_group: torch.distributed.ProcessGroup,
        is_cg_capturable: bool = False,
    ):
        """Wrapper function for TE's Cross Entropy Loss kernel"""
        if _TE_SUPPORTS_CG_CAPTURABLE:
            # According to TE CrossEntropyFunction, ignore_idx defaults to -100
            return parallel_cross_entropy(
                logits, labels, 0.0, False, tp_group, -100, is_cg_capturable
            )
        else:
            return parallel_cross_entropy(logits, labels, 0.0, False, tp_group)

except ImportError:
    te_parallel_cross_entropy = None  # type: ignore[assignment, misc]

try:
    from transformer_engine.pytorch.cpp_extensions import general_gemm
    from transformer_engine.pytorch.module.base import get_workspace

    def te_general_gemm(
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: Optional[torch.dtype] = None,
        layout: str = "TN",
        out: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        grad: bool = False,
    ) -> List[torch.Tensor]:
        """
        Wrapper for TE's general_gemm function.
        It supports fp32, bf16, fp16, and fp8 GEMMs with TN, NN, and NT layouts.
        The output dtype can be specified by `out_dtype`.
        Note: not all combinations of these settings are supported. If not supported,
        cublaslt will throw an error.
        """
        return general_gemm(
            A,
            B,
            workspace=get_workspace(),
            out_dtype=out_dtype,
            quantization_params=None,
            gelu=None,
            gelu_in=None,
            accumulate=False,
            layout=layout,
            out=out,
            bias=bias,
            use_split_accumulator=False,
            grad=grad,
            ub=None,
            ub_type=None,
            extra_output=None,
            bulk_overlap=False,
        )

except ImportError:
    te_general_gemm = None  # type: ignore[assignment, misc]


if HAVE_TE and is_te_min_version("2.7.0.dev"):
    from transformer_engine.pytorch.router import (  # pylint: disable=unused-import
        fused_compute_score_for_moe_aux_loss,
        fused_moe_aux_loss,
        fused_topk_with_score_function,
    )

else:
    fused_topk_with_score_function = None
    fused_compute_score_for_moe_aux_loss = None
    fused_moe_aux_loss = None


def set_save_original_input(module):
    """
    Set the module to save the original input tensors.

    Some transformer-engine modules would save the quantized tensors by default in fp8 training.
    This method is used to set these modules to save the original input tensors directly.

    This can save the memory usage in some FP8 training scenarios, such as the attn linear_proj and
    the shared experts.
    The output-discarding recompute method also relies on this.
    """
    if hasattr(module, 'save_original_input'):
        module.save_original_input = True
    else:
        raise ValueError(
            "set_save_original_input is only needed on transformer-engine modules that save "
            "quantized tensors by default. It needs transformer-engine>=2.6.0dev0."
        )


try:
    # pylint: disable=unused-import
    from transformer_engine.pytorch import cpu_offload_v1 as cpu_offload
except ImportError:
    try:
        from transformer_engine.pytorch import cpu_offload
    except ImportError:
        cpu_offload = None
try:
    # pylint: disable=unused-import
    from transformer_engine.pytorch.float8_tensor import Float8Tensor
except ImportError:
    Float8Tensor = None
