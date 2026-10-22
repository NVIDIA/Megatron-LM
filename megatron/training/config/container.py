# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import copy
import os
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from typing import Any, Type, TypeVar

try:
    import yaml

    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False

from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.msc_utils import maybe_msc
from megatron.core.optimizer import OptimizerConfig
from megatron.core.tensor_parallel.gtp_parallel_layout import GTPParallelLayout
from megatron.training.config.common_config import DistributedInitConfig, ProfilingConfig, RNGConfig
from megatron.training.config.dataset_config import (
    LAYOUT_MIRRORS,
    MIRROR_FIELDS,
    DataConfig,
    mirror_is_unset,
)
from megatron.training.config.inference_config import InferenceSetupConfig
from megatron.training.config.instantiate_utils import InstantiationMode, instantiate
from megatron.training.config.resilience_config import (
    RerunStateMachineConfig,
    StragglerDetectionConfig,
)
from megatron.training.config.rl_config import RLConfig
from megatron.training.config.training_config import (
    CheckpointConfig,
    LoggerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
    ValidationConfig,
)
from megatron.training.config.utils import sanitize_dataclass_config
from megatron.training.config.yaml_utils import safe_yaml_representers
from megatron.training.models import GPTModelConfig, HybridModelConfig, Serializable

T = TypeVar("T", bound="ConfigContainerBase")

@dataclass(kw_only=True)
class ConfigContainerBase:
    """
    Configuration container base class for Megatron configurations.

    Provides validation, runtime finalization, and YAML/Dict serialization.
    """

    _finalized: bool = field(default=False, init=False, repr=False, compare=False)

    def validate(self) -> None:
        """Check sub-configs without mutation; callers may validate repeatedly."""
        for config_field in dataclass_fields(self):
            config = getattr(self, config_field.name)
            validate = getattr(config, "validate", None)
            if callable(validate):
                validate()

    def finalize(self) -> None:
        """Validate and apply runtime derivations once, before runtime initialization.

        Called by ``initialize_megatron``, not by configuration-editing code.
        Re-entering initialization (e.g. an in-process restart) does not apply the
        derivations again. After finalization, treat layout and RNG policy as fixed;
        create a new container for a differently configured run. The lifecycle flag
        is private and is not saved in YAML/checkpoint configuration metadata.
        """
        if self._finalized:
            return
        self.validate()
        self._finalize()
        self._finalized = True

    def _finalize(self) -> None:
        """Apply shared runtime derivations after all validation has succeeded."""
        rng = getattr(self, "rng", None)
        if rng is not None:
            rng.finalize_model_config(getattr(self, "model", None))

    @classmethod
    def _upgrade_config_dict(cls, config_dict: dict[str, Any], legacy_args: Any) -> dict[str, Any]:
        """Hook for containers that complete configurations saved by older versions."""
        return config_dict

    @classmethod
    def from_dict(
        cls: Type[T],
        config_dict: dict[str, Any],
        mode: InstantiationMode = InstantiationMode.STRICT,
        *,
        legacy_args: Any = None,
    ) -> T:
        """
        Create a config container from a dictionary.

        Args:
            config_dict: Dictionary containing configuration
            mode: Serialization mode (strict or lenient)
            legacy_args: Optional validated argparse namespace used to complete sections that
                the dictionary predates (see ``PretrainConfigContainer``)

        Returns:
            A new instance of this class initialized with the dictionary values
        """
        # Make a copy to avoid modifying the input
        config_dict = copy.deepcopy(config_dict)

        assert "_target_" in config_dict

        # Apply backward compatibility: remove init=False fields that may have been
        # serialized by older versions (these are computed in __post_init__)
        config_dict = sanitize_dataclass_config(config_dict)
        config_dict = cls._upgrade_config_dict(config_dict, legacy_args)

        # Check for extra keys in strict mode
        expected_fields = {f.name for f in dataclass_fields(cls) if not f.name.startswith("_")}
        expected_fields.add("_target_")  # Add _target_ as a valid field
        extra_keys = set(config_dict.keys()) - expected_fields

        if extra_keys:
            if mode == InstantiationMode.STRICT:
                raise ValueError(f"Dictionary contains extra keys not in {cls.__qualname__}: {extra_keys}")
            else:
                # In lenient mode, remove extra keys
                for key in extra_keys:
                    config_dict.pop(key)

        # Use instantiate to create the object
        instance = instantiate(config_dict, mode=mode)

        return instance

    @classmethod
    def from_yaml(
        cls: Type[T],
        yaml_path: str,
        mode: InstantiationMode = InstantiationMode.LENIENT,
        *,
        legacy_args: Any = None,
    ) -> T:
        """
        Create a config container from a YAML file.

        Args:
            yaml_path: Path to the YAML file
            mode: Serialization mode (strict or lenient)
            legacy_args: Optional validated argparse namespace used to complete sections that
                the file predates (see ``PretrainConfigContainer``)

        Returns:
            A new instance of this class initialized with the YAML file values
        """
        if not HAVE_YAML:
            raise ImportError(
                "PyYAML is required to load a config from YAML. "
                "Install via `pip install pyyaml`."
            )

        from omegaconf import OmegaConf

        yaml_path_exists = maybe_msc.os.path.exists(yaml_path)

        if not yaml_path_exists:
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")

        with maybe_msc.open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Convert to OmegaConf first for better compatibility with instantiate
        conf = OmegaConf.create(config_dict)

        config_dict = OmegaConf.to_container(conf, resolve=True)
        if legacy_args is None:
            return cls.from_dict(config_dict, mode=mode)
        return cls.from_dict(config_dict, mode=mode, legacy_args=legacy_args)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the config container to a dictionary.

        Also converts any nested dataclasses (both ConfigContainer and regular dataclasses)
        to dictionaries recursively.

        Returns:
            Dictionary representation of this config
        """
        result = {}
        result["_target_"] = f"{self.__class__.__module__}.{self.__class__.__qualname__}"

        for f in dataclass_fields(self):
            if f.name.startswith("_"):
                continue

            value = getattr(self, f.name)
            result[f.name] = self._convert_value_to_dict(value)

        return result

    @classmethod
    def _convert_value_to_dict(cls, value: Any) -> Any:
        """
        Recursively convert a value to a dictionary representation.

        Handles:
        - ConfigContainer instances (using to_dict)
        - Serializable instances (using as_dict)
        - Classes which implement a to_cfg_dict method
        - Regular dataclasses (converting each non-private field)
        - Lists and tuples (converting each element)
        - Dictionaries (converting each value)
        - Other types (kept as-is)

        Args:
            value: The value to convert

        Returns:
            The converted value
        """
        if isinstance(value, ConfigContainerBase):
            return value.to_dict()
        elif isinstance(value, Serializable):
            return value.as_dict()
        elif hasattr(value, "to_cfg_dict"):
            # Allow non-Container classes to implement own custom method
            return value.to_cfg_dict()
        elif is_dataclass(value) and not isinstance(value, type):
            # Handle regular dataclasses
            result = {}

            # Add _target_ field for instantiation
            result["_target_"] = f"{value.__class__.__module__}.{value.__class__.__qualname__}"

            # Convert each field, handling nested dataclasses properly
            for field in dataclass_fields(value):
                if field.name.startswith("_"):
                    continue

                field_value = getattr(value, field.name)
                result[field.name] = cls._convert_value_to_dict(field_value)

            return result
        elif isinstance(value, (list, tuple)):
            return [cls._convert_value_to_dict(item) for item in value]
        elif isinstance(value, dict):
            return {k: cls._convert_value_to_dict(v) for k, v in value.items()}
        else:
            return value

    def to_yaml(self, yaml_path: str) -> None:
        """
        Save the config container to a YAML file.

        Args:
            yaml_path: Path where to save the YAML file.
        """
        if not HAVE_YAML:
            raise ImportError(
                "PyYAML is required to save a config to YAML. "
                "Install via `pip install pyyaml`."
            )

        config_dict = self.to_dict()

        with safe_yaml_representers():
            with maybe_msc.open(yaml_path, "w") as f:
                yaml.safe_dump(config_dict, f, default_flow_style=False)

    def print_yaml(self) -> None:
        """
        Print the config container to the console in YAML format.
        """
        if not HAVE_YAML:
            raise ImportError(
                "PyYAML is required to print a config as YAML. "
                "Install via `pip install pyyaml`."
            )

        config_dict = self.to_dict()
        with safe_yaml_representers():
            print(yaml.safe_dump(config_dict, default_flow_style=False))

    def __deepcopy__(self, memo):
        """Support for deep copying."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        for f in dataclass_fields(self):
            setattr(result, f.name, copy.deepcopy(getattr(self, f.name), memo))

        return result

def _current_world_size() -> int:
    """The world size the run is launched with: the initialized process group, else ``WORLD_SIZE``
    (the source of ``args.world_size``); ``finalize()`` runs before distributed initialization."""
    import torch

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return int(os.getenv("WORLD_SIZE", "1"))


@dataclass(kw_only=True)
class PretrainConfigContainer(ConfigContainerBase):
    """Top-level container holding all configuration objects."""

    train: TrainingConfig
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    model: HybridModelConfig | GPTModelConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    data: DataConfig = field(default_factory=DataConfig)
    """The dataset config, the loader settings and the dataset class. The GPT entry points build
    their datasets from it (``gpt_data_config_from_args``); every other entry point carries the
    loader settings only (``data.dataset is None``). See ``megatron.training.config.dataset_config``."""
    ddp: DistributedDataParallelConfig = field(default_factory=DistributedDataParallelConfig)
    dist: DistributedInitConfig = field(default_factory=DistributedInitConfig)
    rng: RNGConfig = field(default_factory=RNGConfig)
    logger: LoggerConfig
    checkpoint: CheckpointConfig
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    rl: RLConfig = field(default_factory=RLConfig)

    rerun_state_machine: RerunStateMachineConfig = field(default_factory=RerunStateMachineConfig)
    straggler: StragglerDetectionConfig | None = None

    def validate(self) -> None:
        """Validate each section, then the RL section against the rest of the run."""
        self.rl.validate_ddp_config(self.ddp)
        super().validate()
        transformer = getattr(self.model, "transformer", None)
        cuda_graph_impl = None if transformer is None else transformer.cuda_graph_impl
        self.rl.validate_run(
            cuda_graph_impl=cuda_graph_impl,
            micro_batch_size=self.train.micro_batch_size,
            global_batch_size=self.train.global_batch_size,
            skip_train=self.validation.skip_train,
            load_optim=self.checkpoint.load_optim,
            save_interval=self.checkpoint.save_interval,
            exit_interval=self.train.exit_interval,
        )
        self._validate_data_section()

    def get_data_parallel_size(self, world_size: int) -> int | None:
        """The data-parallel size of this run's model layout, or ``None`` without a model config.

        The same quotient as ``validate_args``: ``world_size // (tp * pp * cp * weight-remat)`` with
        the *configured* context-parallel size. The model config stores the resolved one (configured
        CP times the GTP sequence shards), so the shards are divided out first; ``GTPParallelLayout``
        checks the divisions.
        """
        transformer = getattr(self.model, "transformer", None)
        if transformer is None:
            return None
        shards = getattr(transformer, "gtp_remat_num_sequence_shards", 1) or 1
        resolved_cp = transformer.context_parallel_size
        if resolved_cp % shards:
            raise ValueError(
                f"context_parallel_size ({resolved_cp}) must be a multiple of the GTP sequence "
                f"shards ({shards})."
            )
        layout = GTPParallelLayout(
            world_size=world_size,
            tp=transformer.tensor_model_parallel_size,
            pp=transformer.pipeline_model_parallel_size,
            cp=resolved_cp // shards,
            weight_size=getattr(transformer, "gtp_weight_remat_size", 1) or 1,
            num_sequence_shards=shards,
        )
        return layout.dp

    def _written_mirrors(self) -> tuple[str, ...]:
        """The dataset mirrors this container writes at ``finalize()``: all of them, less the layout
        fields when there is no model config to derive them from."""
        if getattr(self.model, "transformer", None) is None:
            return tuple(name for name in MIRROR_FIELDS if name not in LAYOUT_MIRRORS)
        return MIRROR_FIELDS

    def _data_owner_values(self, world_size: int | None = None) -> dict[str, Any]:
        """The values the other sections own for the mirrors this container writes.

        ``data_parallel_size`` needs the world size as well as the model layout and is left out
        without one; the other layout fields need the model layout; the seed and the validation
        flags are always available.
        """
        if self.data.dataset is None:
            return {}
        mirrors = self._written_mirrors()
        values: dict[str, Any] = {}
        if "random_seed" in mirrors:
            values["random_seed"] = self.rng.seed
        if "full_validation" in mirrors:
            values["full_validation"] = self.validation.full_validation
        if "multiple_validation_sets" in mirrors:
            values["multiple_validation_sets"] = self.validation.multiple_validation_sets
        transformer = getattr(self.model, "transformer", None)
        if "context_parallel_size" in mirrors:
            values["context_parallel_size"] = transformer.context_parallel_size
        if "sequence_parallel_size" in mirrors:
            values["sequence_parallel_size"] = transformer.tensor_model_parallel_size * int(
                bool(transformer.sequence_parallel)
            )
        if "hybrid_context_parallel" in mirrors:
            values["hybrid_context_parallel"] = bool(
                getattr(transformer, "hybrid_context_parallel", False)
            )
        if "data_parallel_size" in mirrors and world_size is not None:
            values["data_parallel_size"] = self.get_data_parallel_size(world_size)
        return values

    @staticmethod
    def _check_mirror(dataset: Any, name: str, owned: Any) -> None:
        """An explicit mirror value must agree with the section that owns it; a mirror at Core's
        default carries no explicit value and is written over."""
        if mirror_is_unset(dataset, name):
            return
        explicit = getattr(dataset, name)
        if explicit != owned:
            raise ValueError(
                f"data.dataset.{name}={explicit!r} disagrees with the section that owns it "
                f"({owned!r}); the container writes this field at finalize(), leave it unset."
            )

    def _validate_data_section(self) -> None:
        """Cross-section checks for ``data.dataset``: the model capacity covers the sample length,
        an explicit mirror agrees with the section that owns it, and a mirror this container does
        not write is left unset (it is not saved with the section, so a reloaded run could not
        reproduce it). ``data_parallel_size`` needs the world size and is checked at ``finalize()``.
        """
        dataset = self.data.dataset
        if dataset is None:
            return
        capacity = getattr(self.model, "seq_length", None)
        if isinstance(capacity, int) and isinstance(dataset.sequence_length, int):
            if capacity < dataset.sequence_length:
                raise ValueError(
                    f"model.seq_length ({capacity}, the position-embedding capacity, "
                    "--max-position-embeddings) must be greater than or equal to "
                    f"data.dataset.sequence_length ({dataset.sequence_length}, --seq-length)."
                )
        for name, owned in self._data_owner_values().items():
            self._check_mirror(dataset, name, owned)
        written = self._written_mirrors()
        for name in MIRROR_FIELDS:
            if name in written or mirror_is_unset(dataset, name):
                continue
            raise ValueError(
                f"data.dataset.{name}={getattr(dataset, name)!r} is not written by this container "
                "(there is no model config to derive the layout from) and is not saved with the "
                "section: leave it unset."
            )

    def _finalize(self) -> None:
        dataset = self.data.dataset
        owned: dict[str, Any] = {}
        if dataset is not None:
            # With the world size known, data_parallel_size gets the agreement check as well,
            # before anything is written.
            owned = self._data_owner_values(world_size=_current_world_size())
            for name, value in owned.items():
                self._check_mirror(dataset, name, value)
        super()._finalize()
        for name, value in owned.items():
            setattr(dataset, name, value)
        self.data.finalize()
        if self.rl.perform_rl_step:
            self.rl.finalize_ddp_config(self.ddp)

    @classmethod
    def _upgrade_config_dict(cls, config_dict: dict[str, Any], legacy_args: Any) -> dict[str, Any]:
        """Complete a run configuration saved before the ``data`` section existed.

        The section takes its defaults when missing; with ``legacy_args`` the loader settings come
        from the legacy arguments. Dataset sections are supplied by the GPT entry points, as for a
        fresh run, never inferred from the arguments here.
        """
        if "data" in config_dict or legacy_args is None:
            return config_dict
        from megatron.training.argument_utils import dataloader_config_from_args

        config_dict["data"] = cls._convert_value_to_dict(
            DataConfig(dataloader=dataloader_config_from_args(legacy_args))
        )
        return config_dict


@dataclass(kw_only=True)
class InferenceConfigContainer(ConfigContainerBase):
    """Top-level container for inference entry points.

    This is the inference counterpart to :class:`PretrainConfigContainer`. It holds only the
    configs that inference actually needs and is intentionally shaped differently from the
    training container: there is no optimizer, LR schedule, train/validation loop, DDP, rerun
    state machine, or straggler detection.

    Explicitly NOT included (relative to ``PretrainConfigContainer``): ``TrainingConfig``,
    ``OptimizerConfig``, ``SchedulerConfig``, ``ValidationConfig``,
    ``DistributedDataParallelConfig``, ``RerunStateMachineConfig``, ``StragglerDetectionConfig``.
    """

    model: HybridModelConfig | GPTModelConfig
    """Which model to load for inference."""

    checkpoint: CheckpointConfig
    """Checkpoint configuration used to load model weights."""

    inference: InferenceSetupConfig
    """Declarative inference settings (the serializable, args-shaped layer). Use
    ``InferenceSetupConfig.to_inference_config(model, ...)`` to build the runtime
    ``megatron.core.inference.config.InferenceConfig`` consumed by the engine."""

    dist: DistributedInitConfig = field(default_factory=DistributedInitConfig)
    rng: RNGConfig = field(default_factory=RNGConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
