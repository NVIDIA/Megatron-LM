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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/models/model_provider.py
# Source license: Apache-2.0.

import abc
import dataclasses
import os
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Callable, Generic, Mapping, Self, TypedDict, TypeVar, Union

from megatron.bridge.models.common.unimodal import _ddp_wrap, _print_num_params

try:
    from typing import Unpack
except ImportError:
    try:
        from typing_extensions import Unpack
    except ImportError:
        from unittest.mock import MagicMock

        Unpack = MagicMock()

import torch

from megatron.bridge.models.config import from_hf_pretrained, save_hf_pretrained
from megatron.bridge.utils.common_utils import get_local_rank_preinit
from megatron.bridge.utils.instantiate_utils import InstantiationMode
from megatron.core import parallel_state, tensor_parallel
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.quantization.utils import get_quant_config_or_none
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import Float16Module, MegatronModule
from megatron.core.utils import get_model_config
from megatron.training.models.base import ModelConfig

try:
    from megatron.core.fp8_utils import correct_amax_history_if_needed
except ImportError:
    correct_amax_history_if_needed = None


ModelT = TypeVar("ModelT", bound=MegatronModule)


def _finalize_model_quantization(model: torch.nn.Module) -> None:
    """Finalize module quantization against paths in the assembled model."""
    for name, module in model.named_modules():
        if hasattr(module, "finish_init"):
            quant_config = get_quant_config_or_none(name, module.config.quant_recipe)
            module.finish_init(quant_config)


def _apply_mixed_precision_wrapper(
    model: list[MegatronModule],
    model_config: Any,
    mixed_precision_wrapper: Callable[[Any, MegatronModule], MegatronModule],
) -> list[MegatronModule]:
    """Wrap a model while preserving parameters explicitly marked for FP32."""
    keep_in_fp32: list[tuple[torch.nn.Module, str, torch.Tensor]] = []
    for model_module in model:
        for submodule in model_module.modules():
            # Preserve the existing MCore expert-bias contract.
            if hasattr(submodule, "_maintain_float32_expert_bias"):
                expert_bias = getattr(submodule, "expert_bias", None)
                if expert_bias is not None:
                    keep_in_fp32.append((submodule, "expert_bias", expert_bias.data.clone()))

            # Model-specific modules can mark direct parameters that must not be
            # truncated by Float16Module's recursive half()/bfloat16() cast.
            parameter_names = vars(submodule).get("_keep_in_float32_parameter_names", ())
            if not isinstance(parameter_names, (list, tuple)):
                raise TypeError("_keep_in_float32_parameter_names must be a list or tuple")
            for parameter_name in parameter_names:
                parameter = getattr(submodule, parameter_name, None)
                if not isinstance(parameter, torch.nn.Parameter):
                    raise TypeError(
                        f"{type(submodule).__name__}.{parameter_name} must be a Parameter to remain in FP32"
                    )
                keep_in_fp32.append((submodule, parameter_name, parameter.data.clone()))

    wrapped_model = [mixed_precision_wrapper(model_config, model_module) for model_module in model]

    for submodule, parameter_name, fp32_data in keep_in_fp32:
        getattr(submodule, parameter_name).data = fp32_data

    return wrapped_model


def _wrap_keeping_float32_tensors(
    mixed_precision_wrapper: Callable[[Any, MegatronModule], MegatronModule],
    config: Any,
    module: MegatronModule,
) -> MegatronModule:
    """Wrap one model stage through ``_apply_mixed_precision_wrapper``."""
    return _apply_mixed_precision_wrapper([module], config, mixed_precision_wrapper)[0]


def build_distributed_models(
    model_config: ModelConfig, pg_collection: ProcessGroupCollection, **kwargs: Any
) -> list[MegatronModule]:
    """Build a builder-backed config through its ``ModelBuilder``, keeping FP32-marked tensors exact.

    Megatron Core's builders apply ``mixed_precision_wrapper`` directly, so ``Float16Module`` rounds
    router ``expert_bias`` values loaded before the wrap (Hugging Face weights) and casts
    parameters listed in ``_keep_in_float32_parameter_names`` to the model dtype. The wrapper is
    applied through ``_apply_mixed_precision_wrapper`` instead, which restores both after the cast,
    as the provider path does.

    Args:
        model_config: Builder-backed model config.
        pg_collection: Model communication process groups.
        **kwargs: Arguments for ``ModelBuilder.build_distributed_models``. Its
            ``mixed_precision_wrapper`` is ``Float16Module`` when omitted; ``None`` skips wrapping.

    Returns:
        Model stages created by the configured builder.
    """
    mixed_precision_wrapper = kwargs.get("mixed_precision_wrapper", Float16Module)
    if mixed_precision_wrapper is not None:
        kwargs["mixed_precision_wrapper"] = partial(
            _wrap_keeping_float32_tensors, mixed_precision_wrapper
        )
    builder = model_config.get_builder_cls()(model_config)
    return builder.build_distributed_models(pg_collection=pg_collection, **kwargs)


class ModelProviderMixin(abc.ABC, Generic[ModelT]):
    """A mixin that implements the ModelProvider pattern for Megatron Bridge.

    The ModelProvider pattern solves ecosystem fragmentation by providing a standardized
    way to instantiate models. This mixin provides a consistent `provide_distributed_model()` method
    that handles the complexity of distributed training setup, along with HuggingFace-inspired
    `.from_hf_pretrained()` and `.save_hf_pretrained()` for interoperability.

    For advanced customization, multiple hooks can be registered via `register_pre_wrap_hook`
    and `register_post_wrap_hook`. These hooks allow modifying the model before and after
    it's wrapped for distributed training (e.g., freezing layers, logging). The composed
    hooks can be accessed via the `pre_wrap_hook` and `post_wrap_hook` properties.

    Subclasses must implement the `provide` method to define the model architecture.
    """

    CONFIG_NAME = "mhub_model.json"
    DEFAULT_CONFIG_FORMAT = "json"

    @abc.abstractmethod
    def provide(
        self,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> ModelT:
        """Abstract method to provide the model instance.

        Subclasses must implement this method to return the specific Megatron model
        (e.g., `GPTModel`) with its configuration. This method is called by `get_model`
        to obtain the base model before it is wrapped for distributed training.

        Args:
            pre_process (bool, optional): Whether to include the embedding layer (used with pipeline parallelism).
            post_process (bool, optional): Whether to include the output layer (used with pipeline parallelism).
            vp_stage (int, optional): The virtual pipeline stage of the model.

        Returns:
            ModelT: The Megatron model instance.
        """
        pass

    @abc.abstractmethod
    def finalize(self) -> None:
        """Finalize provider state after configuration overrides are applied."""
        pass

    def apply_overrides_and_finalize(
        self, dtype: torch.dtype | None = None, overrides: Mapping[str, object] | None = None
    ) -> Self:
        """Apply dtype and attribute overrides, then finalize this provider.

        Args:
            dtype: Optional parameter dtype. Also sets ``fp16`` and ``bf16``.
            overrides: Provider attributes to set before finalization.

        Returns:
            This provider.
        """
        if dtype is not None:
            self.params_dtype = dtype
            self.fp16 = dtype == torch.float16
            self.bf16 = dtype == torch.bfloat16

        for name, value in (overrides or {}).items():
            if not hasattr(self, name):
                raise AttributeError(f"{type(self).__name__} has no attribute {name!r}.")
            setattr(self, name, value)

        # DeepSeek-V4 hash-routed MoE requires an explicit pipeline_model_parallel_layout
        # when PP > 1 (the hash layers must co-locate with the embedding on the first
        # stage). Recipes set this explicitly; auto-set it here so provider paths that
        # bypass recipes (e.g. mbridge/verl) also work for PP > 1 without the caller
        # supplying a layout. Only applies to DeepSeek-V4 and never overrides a user layout.
        if (
            getattr(self, "experimental_attention_variant", None) == "dsv4_hybrid"
            and (getattr(self, "pipeline_model_parallel_size", 1) or 1) > 1
            and getattr(self, "pipeline_model_parallel_layout", None) is None
        ):
            from megatron.bridge.models.deepseek.deepseek_v4_bridge import (
                set_deepseek_v4_pipeline_model_parallel_layout,
            )

            set_deepseek_v4_pipeline_model_parallel_layout(self)

        self.finalize()
        return self

    def provide_distributed_model(
        self,
        ddp_config: DistributedDataParallelConfig | None = None,
        model_type=ModelType.encoder_or_decoder,
        overlap_param_gather_with_optimizer_step: bool = False,
        fp16: bool | None = None,
        bf16: bool | None = None,
        use_megatron_fsdp: bool = False,
        use_torch_fsdp2: bool = False,
        wrap_with_ddp: bool = True,
        data_parallel_random_init: bool = False,
        use_cpu_initialization: bool | None = None,
        init_model_with_meta_device: bool | None = None,
        pre_wrap_hook: (
            Union[
                Callable[[list[MegatronModule]], list[MegatronModule]],
                list[Callable[[list[MegatronModule]], list[MegatronModule]]],
            ]
            | None
        ) = None,
        post_wrap_hook: Callable[[list[MegatronModule]], list[MegatronModule]] | None = None,
        mixed_precision_wrapper: (
            Callable[[Any, MegatronModule], MegatronModule] | None
        ) = Float16Module,
        pg_collection: ProcessGroupCollection | None = None,
        *,
        use_layer_wise_distributed_optimizer: bool = False,
        use_layer_wise_param_layout: bool = False,
    ) -> list[ModelT]:
        """Instantiate and wrap the model for distributed training.

        This method retrieves the model from `provide` and sets up the distributed
        environment, including data-parallel and model-parallel configurations.
        It's the primary entry point for creating a model that's ready for use
        in the Megatron ecosystem.

        Args:
            use_layer_wise_distributed_optimizer: Build DDP layouts for layer-wise optimizer state.
            use_layer_wise_param_layout: Use shard-aligned DDP buffers for LayerWise optimizers.
                False retains compact DDP buffers and variable-size parameter synchronization.
            ddp_config: Configuration for distributed data parallel.
            model_type: Type of model (encoder, decoder, or both).
            overlap_param_gather_with_optimizer_step: Whether to overlap param gathering.
            fp16: Override FP16 setting.
            bf16: Override BF16 setting.
            use_megatron_fsdp: Use Megatron's Fully Sharded Data Parallel
            use_torch_fsdp2: Use PyTorch FSDP2 instead of custom DDP.
            wrap_with_ddp: Whether to wrap model with DDP.
            data_parallel_random_init: Initialize parameters randomly across data parallel ranks.
            use_cpu_initialization: Override CPU initialization. None preserves the provider setting.
            init_model_with_meta_device: Initialize model on meta device.
            pre_wrap_hook: A single callable or list of callables to modify the model before it's wrapped.
                If provided, this will override all hooks registered via `register_pre_wrap_hook`.
                If a list is provided, hooks will be executed in order.
            post_wrap_hook: A single callable to modify the model after it's wrapped. If provided,
                this will override all hooks registered via `register_post_wrap_hook`.
            mixed_precision_wrapper: A module wrapper (e.g., `Float16Module`) applied when fp16/bf16
                is enabled. If None, no mixed precision wrapper is applied.
            pg_collection: Optional pre-initialized ProcessGroupCollection. If provided, skips
                model parallel initialization and uses the provided collection directly.
                This is used when `use_decentralized_pg=True` in the distributed config.

        Returns:
            A list containing the wrapped model instance.
        """
        warnings.warn(
            "ModelProviderMixin-based model configuration is deprecated. Migrate to ModelConfig + ModelBuilder.",
            DeprecationWarning,
            stacklevel=2,
        )

        if wrap_with_ddp and not ddp_config:
            raise ValueError("ddp_config is required when wrap_with_ddp is True")

        if not torch.distributed.is_initialized():
            os.environ["RANK"] = os.environ.get("RANK", "0")
            os.environ["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", "1")
            os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
            os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")
            torch.cuda.set_device(get_local_rank_preinit())
            torch.distributed.init_process_group("nccl")

        # If pg_collection is provided (e.g., from use_decentralized_pg=True),
        # use it directly. Otherwise, initialize model parallel state and get pg_collection from MPU.
        if pg_collection is None:
            if not parallel_state.is_initialized():
                print("Model parallel not initialized, initializing...")
                self.initialize_model_parallel(seed=0)
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        # Providers (GPT, Mamba, Gemma, etc.) expect pg_collection on self for PP/TP role checks.
        setattr(self, "_pg_collection", pg_collection)

        # Convert list of hooks to a single composed callable
        if isinstance(pre_wrap_hook, list):

            def composed_pre_wrap_hook(model: list[MegatronModule]) -> list[MegatronModule]:
                for hook in pre_wrap_hook:
                    model = hook(model)
                return model

            final_pre_wrap_hook = composed_pre_wrap_hook
        else:
            final_pre_wrap_hook = pre_wrap_hook or self.pre_wrap_hook
        final_post_wrap_hook = post_wrap_hook or self.post_wrap_hook

        model = get_model(
            self,
            ddp_config=ddp_config,
            model_type=model_type,
            overlap_param_gather_with_optimizer_step=overlap_param_gather_with_optimizer_step,
            fp16=fp16,
            bf16=bf16,
            use_megatron_fsdp=use_megatron_fsdp,
            use_torch_fsdp2=use_torch_fsdp2,
            wrap_with_ddp=wrap_with_ddp,
            data_parallel_random_init=data_parallel_random_init,
            use_cpu_initialization=use_cpu_initialization,
            init_model_with_meta_device=init_model_with_meta_device,
            pre_wrap_hook=final_pre_wrap_hook,
            mixed_precision_wrapper=mixed_precision_wrapper,
            pg_collection=pg_collection,
            use_layer_wise_distributed_optimizer=use_layer_wise_distributed_optimizer,
            use_layer_wise_param_layout=use_layer_wise_param_layout,
        )

        if final_post_wrap_hook:
            _model = final_post_wrap_hook(model)
            if _model is not None:
                model = _model

        return model

    def initialize_model_parallel(
        self, seed: int | None = None, seed_kwargs: dict | None = None, **model_parallel_kwargs
    ) -> None:
        """Initializes model parallelism and sets the random seed.

        This is a convenience method that sets up tensor, pipeline, and other
        forms of model parallelism based on the attributes of the provider instance.

        Args:
            seed: The random seed for model parallel RNG.
            seed_kwargs: Additional arguments for `model_parallel_cuda_manual_seed`.
            **model_parallel_kwargs: Additional arguments for `parallel_state.initialize_model_parallel`.
        """
        # Resolve public weight-shard counts before reading the derived GTP axes
        # used to construct process groups and configure GTP kernels.
        self.finalize()
        if not torch.distributed.is_initialized():
            torch.cuda.set_device(get_local_rank_preinit())
            torch.distributed.init_process_group("nccl")

        from megatron.bridge.models.conversion.runtime import configure_gtp_remat

        configure_gtp_remat(self)
        model_parallel_kwargs.setdefault(
            "gtp_remat_size", getattr(self, "gtp_weight_remat_size", 1)
        )
        model_parallel_kwargs.setdefault(
            "expert_gtp_remat_size", getattr(self, "expert_gtp_weight_remat_size", 1)
        )
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=getattr(self, "tensor_model_parallel_size", 1),
            pipeline_model_parallel_size=getattr(self, "pipeline_model_parallel_size", 1),
            virtual_pipeline_model_parallel_size=getattr(
                self, "virtual_pipeline_model_parallel_size", None
            ),
            context_parallel_size=getattr(self, "context_parallel_size", 1) or 1,
            expert_model_parallel_size=getattr(self, "expert_model_parallel_size", 1) or 1,
            expert_tensor_parallel_size=getattr(self, "expert_tensor_parallel_size", None),
            **model_parallel_kwargs,
        )
        if seed is not None:
            model_parallel_cuda_manual_seed(seed, **(seed_kwargs or {}))

    @property
    def meta_model(self) -> list[ModelT]:
        """Returns the model instantiated on the meta device for inspection.

        This is useful for examining the model architecture without allocating
        GPU memory.
        """
        return self(wrap_with_ddp=False, init_model_with_meta_device=True)

    @property
    def pre_wrap_hook(self) -> Callable[[list[MegatronModule]], list[MegatronModule]] | None:
        """A composed callable of all registered pre-wrap hooks.

        This read-only property returns a single function that executes all registered
        pre-wrap hooks in order. The hook is applied before the model is passed to the DDP
        wrapper and can be used for tasks like freezing layers or altering model structure.

        Use `register_pre_wrap_hook` to add a hook to the execution chain.

        Returns:
            A callable that executes all registered pre-wrap hooks in order, or None if no
            hooks are registered.
        """
        if not hasattr(self, "_pre_wrap_hooks") or not self._pre_wrap_hooks:
            return None

        def composed_hook(model: list[MegatronModule]) -> list[MegatronModule]:
            for hook in self._pre_wrap_hooks:
                model = hook(model)
            return model

        return composed_hook

    def register_pre_wrap_hook(
        self, hook: Callable[[list[MegatronModule]], list[MegatronModule]], prepend: bool = False
    ) -> None:
        """Registers a hook to be executed before the model is wrapped.

        The hook should be a callable that accepts a list of `MegatronModule` instances
        and returns a (potentially modified) list of `MegatronModule` instances.

        Args:
            hook: The hook to register.
            prepend: If True, the hook is inserted at the beginning of the execution
                chain. Otherwise, it is appended to the end.
        """
        if not hasattr(self, "_pre_wrap_hooks"):
            self._pre_wrap_hooks = []
        if prepend:
            self._pre_wrap_hooks.insert(0, hook)
        else:
            self._pre_wrap_hooks.append(hook)

    @property
    def post_wrap_hook(self) -> Callable[[list[MegatronModule]], list[MegatronModule]] | None:
        """A composed callable of all registered post-wrap hooks.

        This read-only property returns a single function that executes all registered
        post-wrap hooks in order. The hook is applied after the model has been wrapped by
        DDP and is useful for tasks like logging or attaching custom attributes.

        Use `register_post_wrap_hook` to add a hook to the execution chain.

        Returns:
            A callable that executes all registered post-wrap hooks in order, or None if no
            hooks are registered.
        """
        if not hasattr(self, "_post_wrap_hooks") or not self._post_wrap_hooks:
            return None

        def composed_hook(model: list[MegatronModule]) -> list[MegatronModule]:
            for hook in self._post_wrap_hooks:
                model = hook(model)
            return model

        return composed_hook

    def register_post_wrap_hook(
        self, hook: Callable[[list[MegatronModule]], list[MegatronModule]], prepend: bool = False
    ) -> None:
        """Registers a hook to be executed after the model is wrapped.

        The hook should be a callable that accepts a list of `MegatronModule` instances
        and returns a (potentially modified) list of `MegatronModule` instances.

        Args:
            hook: The hook to register.
            prepend: If True, the hook is inserted at the beginning of the execution
                chain. Otherwise, it is appended to the end.
        """
        if not hasattr(self, "_post_wrap_hooks"):
            self._post_wrap_hooks = []
        if prepend:
            self._post_wrap_hooks.insert(0, hook)
        else:
            self._post_wrap_hooks.append(hook)

    @classmethod
    def from_hf_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        trust_remote_code: bool = False,
        mode: InstantiationMode | None = None,
        config_name: str | None = None,
        **kwargs,
    ):
        """Load a pretrained model configuration from a directory or HuggingFace Hub.

        This method provides a HuggingFace-inspired interface for loading model
        configurations, enabling interoperability.

        Args:
            pretrained_model_name_or_path: The path to a local directory or a
                HuggingFace model identifier.
            trust_remote_code: Whether to trust remote code when loading.
            mode: The instantiation mode (e.g., `LENIENT`).
            config_name: The name of the configuration file (without extension).
            **kwargs: Additional keyword arguments for `from_hf_pretrained`.

        Returns:
            An instance of the model provider with the loaded configuration.
        """
        if config_name is None:
            config_name = cls.CONFIG_NAME.rsplit(".", 1)[0]
        if mode is None:
            mode = InstantiationMode.LENIENT
        return from_hf_pretrained(
            cls,
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            mode=mode,
            config_name=config_name,
            **kwargs,
        )

    def save_hf_pretrained(
        self,
        save_directory: str | Path,
        config_format: str | None = None,
        config_name: str | None = None,
        **kwargs,
    ):
        """Save the model configuration to a directory.

        This method provides a HuggingFace-inspired interface for saving model
        configurations, enabling interoperability.

        Args:
            save_directory: The directory where the configuration will be saved.
            config_format: The format for the configuration file (e.g., `json` or `yaml`).
            config_name: The name of the configuration file (without extension).
            **kwargs: Additional keyword arguments for `save_hf_pretrained`.
        """
        if config_name is None:
            config_name = self.CONFIG_NAME.rsplit(".", 1)[0]
        if config_format is None:
            config_format = self.DEFAULT_CONFIG_FORMAT
        return save_hf_pretrained(
            self, save_directory, config_format=config_format, config_name=config_name, **kwargs
        )


class GetModelKwargs(TypedDict, total=False):
    """Keyword arguments for the `provide_distributed_model` method.

    Attributes:
        ddp_config: Configuration for distributed data parallel.
        model_type: Type of model (encoder, decoder, or both).
        overlap_param_gather_with_optimizer_step: Whether to overlap param gathering.
        fp16: Override FP16 setting.
        bf16: Override BF16 setting.
        use_megatron_fsdp: Use Megatron's Fully Sharded Data Parallel
        use_torch_fsdp2: Use PyTorch FSDP2 instead of custom DDP.
        wrap_with_ddp: Whether to wrap model with DDP.
        data_parallel_random_init: Initialize parameters randomly across data parallel ranks.
        use_cpu_initialization: Override CPU initialization. None preserves the provider setting.
        init_model_with_meta_device: Initialize model on meta device.
        pre_wrap_hook: A single callable or list of callables that overrides all registered pre-wrap hooks.
        post_wrap_hook: A single callable that overrides all registered post-wrap hooks.
        mixed_precision_wrapper: Module wrapper to apply for fp16/bf16. None to skip.
        use_layer_wise_distributed_optimizer: Enable LayerWise optimizer model wrapping.
        use_layer_wise_param_layout: Opt into shard-aligned LayerWise buffers instead of compact buffers.
    """

    ddp_config: DistributedDataParallelConfig | None
    model_type: ModelType
    overlap_param_gather_with_optimizer_step: bool
    fp16: bool | None
    bf16: bool | None
    use_megatron_fsdp: bool
    use_torch_fsdp2: bool
    wrap_with_ddp: bool
    data_parallel_random_init: bool
    use_cpu_initialization: bool | None
    init_model_with_meta_device: bool | None
    pre_wrap_hook: (
        Union[
            Callable[[list[MegatronModule]], list[MegatronModule]],
            list[Callable[[list[MegatronModule]], list[MegatronModule]]],
        ]
        | None
    )
    post_wrap_hook: Callable[[list[MegatronModule]], list[MegatronModule]] | None
    mixed_precision_wrapper: Callable[[Any, MegatronModule], MegatronModule] | None
    use_layer_wise_distributed_optimizer: bool
    use_layer_wise_param_layout: bool


class ModelParallelKwargs(TypedDict, total=False):
    """Model-parallel and checkpoint inference override kwargs.

    Attributes map to `TransformerConfig`/provider fields that control parallelism.
    Only provided values are applied as overrides.
    """

    tensor_model_parallel_size: int
    tensor_parallel_num_weight_shards: int
    pipeline_model_parallel_size: int
    num_layers_in_first_pipeline_stage: int | None
    num_layers_in_last_pipeline_stage: int | None
    context_parallel_size: int
    expert_model_parallel_size: int
    expert_tensor_parallel_size: int
    expert_tensor_parallel_num_weight_shards: int
    sequence_parallel: bool
    virtual_pipeline_model_parallel_size: int | None
    hierarchical_context_parallel_sizes: list[int] | None
    pipeline_model_parallel_layout: list[list[str]] | None
    pipeline_dtype: torch.dtype
    moe_expert_capacity_factor: float | None
    moe_expert_rank_capacity_factor: float | None
    moe_paged_stash: bool
    moe_ncclep_zero_copy: bool
    moe_pad_expert_input_to_capacity: bool
    moe_router_force_load_balancing: bool
    moe_router_force_biased: float | None
    moe_hybridep_pad_uneven_dispatch_inputs: bool


def get_model(
    model_provider: ModelProviderMixin,
    ddp_config: DistributedDataParallelConfig,
    model_type=ModelType.encoder_or_decoder,
    overlap_param_gather_with_optimizer_step: bool = False,
    fp16: bool | None = None,
    bf16: bool | None = None,
    use_megatron_fsdp: bool = False,
    use_torch_fsdp2: bool = False,
    wrap_with_ddp: bool = True,
    data_parallel_random_init: bool = False,
    use_cpu_initialization: bool | None = None,
    init_model_with_meta_device: bool | None = None,
    pre_wrap_hook: (
        Union[
            Callable[[list[MegatronModule]], list[MegatronModule]],
            list[Callable[[list[MegatronModule]], list[MegatronModule]]],
        ]
        | None
    ) = None,
    mixed_precision_wrapper: Callable[[Any, MegatronModule], MegatronModule] | None = Float16Module,
    *,
    pg_collection: ProcessGroupCollection,
    use_layer_wise_distributed_optimizer: bool = False,
    use_layer_wise_param_layout: bool = False,
) -> list[MegatronModule]:
    """Create and configure a model for distributed training.

    This function handles the complete model creation pipeline including:
    - Model instantiation with proper pipeline parallel configuration
    - GPU memory allocation
    - Mixed precision (FP16/BF16) wrapping
    - Float8 tensor correction
    - Distributed Data Parallel (DDP) wrapping

    Args:
        model_provider: ModelProviderMixin instance that creates the model.
            Uses the provide() method with optional pre_process(bool), post_process(bool),
            vp_stage(int) arguments for pipeline parallelism
        ddp_config: Configuration for distributed data parallel training
        model_type: Type of model (encoder, decoder)
        overlap_param_gather_with_optimizer_step: Whether to overlap parameter
            gathering with optimizer step for performance optimization
        fp16: Enable FP16 mixed precision training. If None, uses model config
        bf16: Enable BF16 mixed precision training. If None, uses model config
        use_megatron_fsdp: Use Megatron's Fully Sharded Data Parallel
        use_torch_fsdp2: Use PyTorch's Fully Sharded Data Parallel v2
        use_layer_wise_distributed_optimizer: Build shard-aligned DDP layouts for layer-wise optimizers.
        use_layer_wise_param_layout: Use shard-aligned DDP buffers for LayerWise optimizers.
            False retains compact DDP buffers and variable-size parameter synchronization.
        wrap_with_ddp: Whether to wrap the model with DDP
        data_parallel_random_init: Whether to use random initialization for
            data parallel ranks (vs broadcasting from rank 0)
        use_cpu_initialization: Override CPU initialization. None preserves the provider setting
        init_model_with_meta_device: Whether to initialize the model on the meta device
        pre_wrap_hook: A callable or list of callables that takes a list of `MegatronModule`
            and returns a modified list, or `None` to clear the hook. If a list is provided,
            hooks will be executed in order.
        mixed_precision_wrapper: Wrapper class/function applied when fp16/bf16 is enabled. Defaults
            to Megatron-Core's `Float16Module`. If None, the wrapper is not applied.

    Returns:
        list[MegatronModule]: List of model modules. Contains multiple modules
            when using virtual pipeline parallelism, otherwise a single module
    """
    if fp16 is True and bf16 is True:
        raise ValueError("Only one of fp16 and bf16 can be enabled.")

    if fp16 is not None:
        model_provider.fp16 = fp16
    if bf16 is not None:
        model_provider.bf16 = bf16

    # Enabling one precision mode overrides the provider's existing mode even
    # when the caller omits the corresponding explicit False.
    if fp16 is True and bf16 is None:
        model_provider.bf16 = False
    elif bf16 is True and fp16 is None:
        model_provider.fp16 = False

    if fp16 is not None or bf16 is not None:
        if model_provider.fp16:
            selected_dtype = torch.float16
        elif model_provider.bf16:
            selected_dtype = torch.bfloat16
        else:
            selected_dtype = torch.float32
        for field_name in ("params_dtype", "pipeline_dtype", "autocast_dtype"):
            if hasattr(model_provider, field_name):
                setattr(model_provider, field_name, selected_dtype)

    if use_cpu_initialization is not None:
        model_provider.use_cpu_initialization = use_cpu_initialization
    if init_model_with_meta_device:
        model_provider.init_model_with_meta_device = True
        with torch.device("meta"):
            model = _create_model(model_provider, model_type, pg_collection=pg_collection)
    else:
        model = _create_model(model_provider, model_type, pg_collection=pg_collection)

    if pre_wrap_hook:
        if isinstance(pre_wrap_hook, list):
            # Execute hooks in order
            for hook in pre_wrap_hook:
                if not callable(hook):
                    raise RuntimeError("All elements in pre_wrap_hook list must be callable")
                _model = hook(model)
                if _model is not None:
                    model = _model
        else:
            if not callable(pre_wrap_hook):
                raise RuntimeError("pre_wrap_hook must be a callable or a list of callables")
            _model = pre_wrap_hook(model)
            if _model is not None:
                model = _model

    for model_module in model:
        _finalize_model_quantization(model_module)

    # Set tensor model parallel attributes if not set
    # Refresh after pre_wrap_hook in case it augmented the model.
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    _print_num_params(model, pg_collection=pg_collection)

    model_config = get_model_config(model[0])

    # GPU allocation.
    # For FSDP2, we don't allocate GPU memory here. We allocate GPU memory
    # in the fully_shard function of FSDP2 instead.
    if (
        not use_torch_fsdp2
        and not model_config.use_cpu_initialization
        and not model_config.init_model_with_meta_device
    ):
        for model_module in model:
            model_module.cuda(torch.cuda.current_device())

    if (model_config.fp16 or model_config.bf16) and mixed_precision_wrapper is not None:
        model = _apply_mixed_precision_wrapper(model, model_config, mixed_precision_wrapper)

    if correct_amax_history_if_needed is not None:
        correct_amax_history_if_needed(model)

    if wrap_with_ddp:
        effective_ddp_config = ddp_config
        if use_layer_wise_distributed_optimizer and not use_megatron_fsdp and not use_torch_fsdp2:
            # Direct callers can bypass ConfigContainer normalization. Keep the
            # ordinary DistOpt flag aligned with the requested LayerWise layout
            # without mutating their configuration object.
            effective_ddp_config = dataclasses.replace(
                ddp_config, use_distributed_optimizer=use_layer_wise_param_layout
            )
        model = _ddp_wrap(
            model,
            data_parallel_random_init,
            effective_ddp_config,
            overlap_param_gather_with_optimizer_step,
            use_megatron_fsdp=use_megatron_fsdp,
            use_torch_fsdp2=use_torch_fsdp2,
            use_layer_wise_distributed_optimizer=use_layer_wise_distributed_optimizer,
            use_layer_wise_param_layout=use_layer_wise_param_layout,
            pg_collection=pg_collection,
        )

    return model


def _create_model(
    model_provider: ModelProviderMixin, model_type: ModelType, pg_collection: ProcessGroupCollection
) -> list[MegatronModule]:
    """Create model instances with appropriate pipeline parallel configuration.

    Handles virtual pipeline parallelism (VPP) by creating multiple model
    instances when needed. Sets pre_process and post_process flags based on
    pipeline parallel rank.

    Args:
        model_provider: ModelProviderMixin instance that creates the model
        model_type: ModelType enum indicating encoder, decoder, or both

    Returns:
        list: List of model instances. Multiple instances for VPP, otherwise single
    """
    vp_size = getattr(model_provider, "virtual_pipeline_model_parallel_size", None)
    pp_group = pg_collection.pp
    if (pp_group.size() > 1) and (vp_size is not None):
        model = []
        for i in range(vp_size):
            pre_process = is_vp_first_stage(vp_stage=i, vp_size=vp_size) and is_pp_first_stage(
                pp_group
            )
            post_process = is_vp_last_stage(vp_stage=i, vp_size=vp_size) and is_pp_last_stage(
                pp_group
            )
            this_model = model_provider.provide(
                pre_process=pre_process, post_process=post_process, vp_stage=i
            )
            this_model.model_type = model_type
            this_model.vp_stage = i
            model.append(this_model)
    else:
        pre_process = is_pp_first_stage(pp_group)
        post_process = is_pp_last_stage(pp_group)
        model = model_provider.provide(pre_process=pre_process, post_process=post_process)
        model.model_type = model_type

    if not isinstance(model, list):
        model = [model]

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    return model
