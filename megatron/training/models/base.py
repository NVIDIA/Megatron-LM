# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import abc
import importlib
from dataclasses import dataclass, field, is_dataclass
from dataclasses import fields as dataclass_fields
from typing import Any, Callable, ClassVar, Generic, Protocol, TypeVar, runtime_checkable

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.module import Float16Module


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

    Each model type (GPT, T5, Mamba, etc.) defines a concrete subclass with its
    own model-specific parameters. This class is a pure data container - all model
    construction logic lives in the corresponding ``ModelBuilder`` subclass.

    Subclasses must define:
        - ``builder``: a ``ClassVar[str]`` with the full import path to the
          associated ``ModelBuilder`` (e.g.
          ``'megatron.bridge.models.mamba.MambaModelBuilder'``).

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
    builder: ClassVar[str]
    """Class variable with full path to builder class (e.g.,
    'megatron.bridge.builders.GPTModelBuilder').
    """

    # === ModelOpt ===
    restore_modelopt_state: bool = False
    """Restore ModelOpt quantization/sparsity state."""

    # === HuggingFace Metadata ===
    hf_model_id: str | None = None
    """HuggingFace model identifier."""

    generation_config: Any | None = None
    """Generation configuration."""

    # === pre-wrap and post-wrap hooks ===
    pre_wrap_hooks: list[Callable[[list[MegatronModule]], list[MegatronModule]]] = field(default_factory=list)
    """List of functions that are executed before the model is wrapped with DDP/FSDP.
    Should take the model as the only argument and return a new model as the only return value.
    """

    post_wrap_hooks: list[Callable[[list[MegatronModule]], list[MegatronModule]]] = field(default_factory=list)
    """List of functions that are executed after model initialization is complete.
    Should take the model as the only argument and return a new model as the only return value.
    """

    def get_builder_cls(self) -> type:
        """Get the appropriate builder type for this config.
        Dynamically imports the builder from the string path.
        """
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

        def _as_dict(config):
            result = {
                "_target_": f"{config.__class__.__module__}.{config.__class__.__qualname__}",
            }
            for f in dataclass_fields(config):
                value = getattr(config, f.name)
                # Skip non-serializable fields
                if callable(value) or f.name.startswith("_") or f.name in ["pre_wrap_hooks", "post_wrap_hooks"]:
                    continue

                if is_dataclass(value):
                    result[f.name] = _as_dict(value)  # recurse on nested dataclasses
                else:
                    result[f.name] = value

            return result

        result = _as_dict(self)
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

            # recurse on serialized nested dataclasses
            subconfigs = {}
            for k, v in filtered_data.items():
                if isinstance(v, dict) and "_target_" in v:
                    subconfigs[k] = _from_dict(v)
            filtered_data.update(subconfigs)

            return config_cls(**filtered_data)

        result = _from_dict(data)
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
