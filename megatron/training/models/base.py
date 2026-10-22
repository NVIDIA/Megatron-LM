# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import abc
import functools
import importlib
import inspect
from dataclasses import dataclass, field, is_dataclass
from dataclasses import fields as dataclass_fields
from typing import Any, Callable, ClassVar, Generic, Protocol, TypeVar, runtime_checkable

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.module import Float16Module

# Hooks and callbacks that each run attaches to the live config. Not to be serialized.
_RUNTIME_FIELDS = frozenset(
    {
        "pre_wrap_hooks",
        "post_wrap_hooks",
        "timers",
        "finalize_model_grads_func",
        "grad_scale_func",
        "moe_grad_scale_func",
        "mtp_grad_scale_func",
        "no_sync_func",
        "grad_sync_func",
        "param_sync_func",
    }
)


def _encode_callable(value: Callable) -> dict[str, Any] | None:
    """Serialize a callable so that it can be deserialized with `instantiate()`."""

    if isinstance(value, functools.partial):
        inner = _encode_callable(value.func)
        if inner is None:
            return None
        encoded: dict[str, Any] = {
            "_target_": inner["_target_"],
            "_partial_": True,
            "_args_": [_model_value_to_dict(arg) for arg in value.args],
        }
        encoded.update({key: _model_value_to_dict(arg) for key, arg in value.keywords.items()})
        return encoded

    module = inspect.getmodule(value)
    qualname = getattr(value, "__qualname__", None)
    if module is None or qualname is None:
        return None
    # Encode only paths that resolve back to `value`; a bound method's path drops its instance.
    resolved = module
    for part in qualname.split("."):
        resolved = getattr(resolved, part, None)
    if resolved != value:
        return None
    return {"_target_": f"{module.__name__}.{qualname}", "_call_": False}


def _model_value_to_dict(value: Any) -> Any:
    """Preserve nested model-spec inputs, including dataclasses in partial kwargs."""
    if is_dataclass(value) and not isinstance(value, type):
        return _model_dataclass_to_dict(value)
    if callable(value):
        encoded = _encode_callable(value)
        return value if encoded is None else encoded
    if isinstance(value, dict):
        return {key: _model_value_to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_model_value_to_dict(item) for item in value]
    if hasattr(value, "as_dict"):
        return value.as_dict()
    return value


def _model_dataclass_to_dict(config) -> dict[str, Any]:
    """Encode model dataclasses consistently for checkpoint export and spec lookup."""
    result = {"_target_": f"{config.__class__.__module__}.{config.__class__.__qualname__}"}
    for f in dataclass_fields(config):
        if f.name.startswith("_") or f.name in _RUNTIME_FIELDS:
            continue
        value = getattr(config, f.name)
        if callable(value) and (isinstance(value, type) or not is_dataclass(value)):
            encoded = _encode_callable(value)
            if encoded is not None:
                result[f.name] = encoded
            continue
        result[f.name] = _model_value_to_dict(value)
    return result


@runtime_checkable
class Serializable(Protocol):
    """Protocol for serializable configurations."""

    def as_dict(self) -> dict[str, Any]:
        """Serialize to dictionary with _target_ for class identification."""
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Serializable":
        """Deserialize from dictionary using _target_ to identify class."""
        ...


@dataclass
class ModelConfig:
    """Base class for model configurations.

    Each model type (GPT, T5, Hybrid, etc.) defines a concrete subclass with its
    own model-specific parameters. This class is a pure data container - all model
    construction logic lives in the corresponding ``ModelBuilder`` subclass or
    an application-supplied model-provider callback.

    Builder-backed subclasses define:
        - ``builder``: a ``ClassVar[str]`` with the full import path to the
          associated ``ModelBuilder`` (e.g.
          ``'megatron.bridge.models.hybrid.HybridModelBuilder'``).

    Subclasses may also embed nested configs (e.g. ``TransformerConfig``) and
    proxy attribute access to them via ``__getattr__``/``__setattr__`` overrides.

    Serialization:
        Use ``as_dict()`` to serialize to a plain dict (includes a ``_target_`` key
        for class resolution and a ``_builder_`` key for builder resolution).
        Use ``from_dict()`` to reconstruct an instance from such a dict.

    Builder resolution:
        Call ``get_builder_cls()`` to dynamically import and return the builder
        class identified by the ``builder`` ClassVar.
    """

    # === Builder Metadata (Serializable) ===
    builder: ClassVar[str | None] = None
    """Class variable with full path to builder class (e.g.,
    'megatron.bridge.builders.GPTModelBuilder').
    """

    # === ModelOpt ===
    restore_modelopt_state: bool = False
    """Restore ModelOpt quantization/sparsity state."""

    extra_checkpoint_metadata: dict[str, Any] | None = None
    """Store for additional model-related metadata to save in the checkpoint. This 
    may be useful when extending Megatron-Core models, e.g. converting to/from HuggingFace.
    Must be serializable."""

    # === pre-wrap and post-wrap hooks ===
    pre_wrap_hooks: list[Callable[[list[MegatronModule]], list[MegatronModule]]] = field(default_factory=list)
    """List of functions that are executed before the model is wrapped with DDP/FSDP.
    Should take the model as the only argument and return a new model as the only return value.
    """

    post_wrap_hooks: list[Callable[[list[MegatronModule]], list[MegatronModule]]] = field(default_factory=list)
    """List of functions that are executed after model initialization is complete.
    Should take the model as the only argument and return a new model as the only return value.
    """

    def finalize(self) -> None:
        """Resolve nested transformer settings before deriving dependent run configs."""
        transformer = getattr(self, "transformer", None)
        finalize = getattr(transformer, "finalize", None)
        if callable(finalize):
            finalize()

    def get_builder_cls(self) -> type:
        """Get the appropriate builder type for this config.
        Dynamically imports the builder from the string path.
        """
        if self.builder is None:
            raise ValueError("This model config requires a model-provider callback.")
        module_path, class_name = self.builder.rsplit(".", 1)
        module = importlib.import_module(module_path)
        builder_cls = getattr(module, class_name)
        return builder_cls

    def as_dict(self) -> dict[str, Any]:
        """Serialize config to dictionary for saving.

        Includes:
        - _target_: Full class path for deserialization
        - _builder_: Full builder class path (serialized from ClassVar)
        - All dataclass fields, including nested dataclasses
        """

        result = _model_dataclass_to_dict(self)
        if self.builder is not None:
            result["_builder_"] = self.builder  # Serialize the builder path
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        """Deserialize config from dictionary.

        Uses _target_ to determine the correct class to instantiate.
        The builder is restored from _builder_ or from the class's ClassVar.

        Args:
            data: Dictionary with _target_ and config fields

        Returns:
            Instance of the appropriate ModelConfig subclass
        """

        def _from_dict(subdata):
            target = subdata.get("_target_")
            if target is None:
                raise ValueError("Cannot deserialize: missing '_target_' field")

            # Import the class from the target path
            module_path, class_name = target.rsplit(".", 1)
            module = importlib.import_module(module_path)
            config_cls = getattr(module, class_name)

            # Filter to valid fields for this class
            valid_fields = {f.name for f in dataclass_fields(config_cls)}
            filtered_data = {k: v for k, v in subdata.items() if k in valid_fields and not k.startswith("_")}

            return config_cls(**{key: _from_value(value) for key, value in filtered_data.items()})

        def _from_value(value):
            if isinstance(value, dict):
                if "_target_" in value:
                    if (
                        value.get("_call_", True) is False
                        or value.get("_partial_", False)
                        or "_args_" in value
                    ):
                        from megatron.training.config.instantiate_utils import instantiate

                        return instantiate(value)
                    return _from_dict(value)
                return {key: _from_value(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return type(value)(_from_value(item) for item in value)
            return value

        result = _from_dict(data)
        if "_builder_" in data:
            result.builder = data["_builder_"]

        return result


ModelT = TypeVar("ModelT", bound=MegatronModule)
BuildConfigT = TypeVar("BuildConfigT", bound=ModelConfig)


class ModelBuilder(abc.ABC, Generic[ModelT, BuildConfigT]):
    """Abstract base class for model builders.

    A builder takes a ``ModelConfig`` and produces distributed model instances -
    either a single pipeline stage via ``build_model()``, or a list of stages
    wrapped for distributed training via ``build_distributed_models()``.

    Each builder subclass should:
    1. Implement ``build_model()`` for the specific model type
    2. Implement ``build_distributed_models()`` to handle virtual pipeline parallelism,
       DDP/FSDP wrapping, and pre/post-wrap hook execution
    3. Be linked to its corresponding ``ModelConfig`` via the ``builder`` ClassVar

    Builders are factory objects, therefore any state saved in __init__ should not be modified
    and only used to build the model.

    Type Parameters:
        ModelT: The type of model this builder produces (e.g., MCoreGPTModel)
        BuildConfigT: The type of build config this builder accepts (e.g., GPTModelBuildConfig)
    """

    def __init__(self, model_config: ModelConfig):
        self._model_config = model_config

    @abc.abstractmethod
    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> ModelT:
        """Build a model from the provided configurations.

        Args:
            pg_collection: Process groups for distributed training
            pre_process: Include embedding layer
            post_process: Include output layer
            vp_stage: Virtual pipeline stage

        Returns:
            The constructed model
        """
        ...

    @abc.abstractmethod
    def build_distributed_models(
        self,
        pg_collection: ProcessGroupCollection,
        ddp_config: DistributedDataParallelConfig | None = None,
        overlap_param_gather_with_optimizer_step: bool = False,
        use_megatron_fsdp: bool = False,
        use_torch_fsdp2: bool = False,
        wrap_with_ddp: bool = True,
        data_parallel_random_init: bool = False,
        mixed_precision_wrapper: Callable[[Any, MegatronModule], MegatronModule] | None = Float16Module,
        model_type: ModelType = ModelType.encoder_or_decoder,
        use_layer_wise_distributed_optimizer: bool = False,
        use_layer_wise_param_layout: bool = True,
    ) -> list[ModelT]:
        """Build model stages and wrap for distributed training.

        Args:
            pg_collection: Model communication process groups.
            ddp_config: DistributedDataParallel configuration
            overlap_param_gather_with_optimizer_step: Whether to overlap parameter gather with optimizer step
            use_megatron_fsdp: Whether to use Megatron FSDP
            use_torch_fsdp2: Whether to use Torch FSDP 2.0
            wrap_with_ddp: Set to False to skip DDP wrapper
            data_parallel_random_init: Whether to use data parallel random initialization
            mixed_precision_wrapper: Mixed precision wrapper, e.g. ``Float16Module``
            model_type: Deprecated flag, only used for backwards compatibility.
            use_layer_wise_distributed_optimizer: Whether the layerwise wiring runs.
            use_layer_wise_param_layout: When ``use_layer_wise_distributed_optimizer=True``,
                controls whether to compute and supply a shard-aligned param layout to DDP.

        Returns:
            List of model stages. If the model does not support virtual pipeline parallelism,
                this function should still return a single-item list.
        """
        ...


def compose_hooks(
    hooks: list[Callable[[list[MegatronModule]], list[MegatronModule]]],
) -> Callable[[list[MegatronModule]], list[MegatronModule]]:
    """Utility to compose pre/post-wrap hooks into a single function, preserving order.

    If `hooks` is empty, the returned function is an identity operation.

    Args:
        hooks: the list of hooks.

    Returns:
        A single function that executes all functions in `hooks`.
    """

    def composed_hook(model: list[MegatronModule]) -> list[MegatronModule]:
        for hook in hooks:
            model = hook(model)
        return model

    return composed_hook
