# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
import copy
import os
from dataclasses import dataclass, field, fields as dataclass_fields
from typing import Any, Type, TypeVar
import yaml
from omegaconf import OmegaConf
from megatron.training.config.common_config import RNGConfig, DistributedInitConfig, ProfilingConfig
from megatron.training.config.training_config import TrainingConfig, ValidationConfig, SchedulerConfig, LoggerConfig, CheckpointConfig
from megatron.core.optimizer import OptimizerConfig
from megatron.core.msc_utils import MultiStorageClientFeature
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.training.config.resilience_config import RerunStateMachineConfig, StragglerDetectionConfig
from megatron.training.config.utils import sanitize_dataclass_config
from megatron.training.config.instantiate_utils import InstantiationMode, instantiate

T = TypeVar("T", bound="ConfigContainerBase")

@dataclass(kw_only=True)
class ConfigContainerBase:
    """
    Configuration container base class for Megatron configurations.

    Provides YAML/Dict serialization and deserialization.
    """
    
    @classmethod
    def from_dict(
        cls: Type[T],
        config_dict: dict[str, Any],
        mode: InstantiationMode = InstantiationMode.STRICT,
    ) -> T:
        """
        Create a config container from a dictionary.

        Args:
            config_dict: Dictionary containing configuration
            mode: Serialization mode (strict or lenient)

        Returns:
            A new instance of this class initialized with the dictionary values
        """
        # Make a copy to avoid modifying the input
        config_dict = copy.deepcopy(config_dict)

        assert "_target_" in config_dict

        # Apply backward compatibility: remove init=False fields that may have been
        # serialized by older versions (these are computed in __post_init__)
        config_dict = sanitize_dataclass_config(config_dict)

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
    def from_yaml(cls: Type[T], yaml_path: str, mode: InstantiationMode = InstantiationMode.LENIENT) -> T:
        """
        Create a config container from a YAML file.

        Args:
            yaml_path: Path to the YAML file
            mode: Serialization mode (strict or lenient)

        Returns:
            A new instance of this class initialized with the YAML file values
        """
        if MultiStorageClientFeature.is_enabled():
            msc = MultiStorageClientFeature.import_package()
            yaml_path_exists = msc.os.path.exists(yaml_path)
        else:
            yaml_path_exists = os.path.exists(yaml_path)

        if not yaml_path_exists:
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")

        if MultiStorageClientFeature.is_enabled():
            msc = MultiStorageClientFeature.import_package()
            with msc.open(yaml_path, "r") as f:
                config_dict = yaml.safe_load(f)
        else:
            with open(yaml_path, "r") as f:
                config_dict = yaml.safe_load(f)

        # Convert to OmegaConf first for better compatibility with instantiate
        conf = OmegaConf.create(config_dict)

        return cls.from_dict(OmegaConf.to_container(conf, resolve=True), mode=mode)

    def __deepcopy__(self, memo):
        """Support for deep copying."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        for f in dataclass_fields(self):
            setattr(result, f.name, copy.deepcopy(getattr(self, f.name), memo))

        return result

@dataclass(kw_only=True)
class PretrainConfigContainer(ConfigContainerBase):
    """Top-level container holding all configuration objects."""

    train: TrainingConfig
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    # model: GPTModelConfig | MambaModelConfig  # TODO (@maanug): add support
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    # dataset: GPTDatasetConfig # TODO (@maanug): add support
    # tokenizer: TokenizerConfig # TODO (@maanug): add support
    ddp: DistributedDataParallelConfig = field(default_factory=DistributedDataParallelConfig)
    dist: DistributedInitConfig = field(default_factory=DistributedInitConfig)
    rng: RNGConfig = field(default_factory=RNGConfig)
    logger: LoggerConfig
    checkpoint: CheckpointConfig
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)

    rerun_state_machine: RerunStateMachineConfig = field(default_factory=RerunStateMachineConfig)
    straggler: StragglerDetectionConfig | None = None
