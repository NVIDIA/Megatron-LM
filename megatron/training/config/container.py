# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
import copy
import os
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
import torch
from typing import Any, Type, TypeVar

import yaml

from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.msc_utils import MultiStorageClientFeature
from megatron.core.optimizer import OptimizerConfig
from megatron.training.config.common_config import DistributedInitConfig, ProfilingConfig, RNGConfig
from megatron.training.config.instantiate_utils import InstantiationMode, instantiate
from megatron.training.config.resilience_config import (
    RerunStateMachineConfig,
    StragglerDetectionConfig,
)
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
from megatron.training.models import Serializable, HybridModelConfig
from megatron.core._rank_utils import safe_get_world_size
from megatron.training.utils import print_rank_0
from megatron.core.transformer.enums import AttnBackend

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
        from omegaconf import OmegaConf

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
        config_dict = self.to_dict()

        with safe_yaml_representers():
            if MultiStorageClientFeature.is_enabled():
                msc = MultiStorageClientFeature.import_package()
                with msc.open(yaml_path, "w") as f:
                    yaml.safe_dump(config_dict, f, default_flow_style=False)
            else:
                with open(yaml_path, "w") as f:
                    yaml.safe_dump(config_dict, f, default_flow_style=False)

    def print_yaml(self) -> None:
        """
        Print the config container to the console in YAML format.
        """
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

@dataclass(kw_only=True)
class PretrainConfigContainer(ConfigContainerBase):
    """Top-level container holding all configuration objects."""

    train: TrainingConfig
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    model: HybridModelConfig  # TODO (@maanug): add support for GPTModelConfig 
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    # dataset: GPTDatasetConfig # TODO (@maanug): add support
    ddp: DistributedDataParallelConfig = field(default_factory=DistributedDataParallelConfig)
    dist: DistributedInitConfig = field(default_factory=DistributedInitConfig)
    rng: RNGConfig = field(default_factory=RNGConfig)
    logger: LoggerConfig
    checkpoint: CheckpointConfig
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)

    rerun_state_machine: RerunStateMachineConfig = field(default_factory=RerunStateMachineConfig)
    straggler: StragglerDetectionConfig | None = None

    def get_data_parallel_size(self, world_size: int) -> int:
        """Calculate the data parallel size based on the model configuration."""
        model_cfg = self.model
        if hasattr(model_cfg, "dist_train") and getattr(model_cfg.dist_train, "use_dist_train", False) is True:
            # use language world size to calculate data parallel size for dist train
            world_size = model_cfg.dist_train.language_world_size
        total_model_size = (
            model_cfg.tensor_model_parallel_size
            * model_cfg.pipeline_model_parallel_size
            * model_cfg.context_parallel_size
        )
        assert world_size % total_model_size == 0, f"""
        world size ({world_size}) is not divisible by total_model_size ({model_cfg.tensor_model_parallel_size=} * {model_cfg.pipeline_model_parallel_size=} * {model_cfg.context_parallel_size=})
        """
        return world_size // total_model_size

    def set_data_parallel_size(self) -> None:
        """Calculate and set data_parallel_size for this config.

        This method calculates the data parallel size needed by setup methods, without
        triggering full validation or finalization of Megatron Core configs.
        """
        # Calculate data parallel size (needed for comm overlap setup)
        world_size = safe_get_world_size()
        self.data_parallel_size = self.get_data_parallel_size(world_size)

    def validate(self) -> None:
        """Performs validation checks on the combined configuration.

        Calculates dependent values like data_parallel_size and scheduler steps.
        Ensures compatibility across sub-configs.
        """

        if hasattr(self.ddp, "finalize"):
            self.ddp.finalize()
        if hasattr(self.optimizer, "finalize"):
            self.optimizer.finalize()
        if hasattr(self.model, "finalize"):
            self.model.finalize()

        self.train.finalize()
        self.scheduler.finalize()
        self.checkpoint.finalize()
        if self.profiling is not None:
            self.profiling.finalize()

        # Sync config. If TE RNG tracker is set in either ways, set them in both places.
        if self.rng.te_rng_tracker or self.model.use_te_rng_tracker:
            self.model.use_te_rng_tracker = self.rng.te_rng_tracker = True

        # Re-run post-inits of sub-configs
        for f in dataclass_fields(self):
            sub_cfg = getattr(self, f.name)
            if hasattr(sub_cfg, "__post_init__") and not hasattr(sub_cfg, "finalize"):
                sub_cfg.__post_init__()

        # Distributed - ensure data_parallel_size is calculated (might already be set by set_data_parallel_size)
        if not hasattr(self, "data_parallel_size") or self.data_parallel_size is None:
            self.set_data_parallel_size()

        # Resolve eval batch size defaults from training config
        if self.validation.eval_global_batch_size is None:
            assert self.train.global_batch_size is not None, (
                "train.global_batch_size must be set when eval_global_batch_size is not explicitly configured"
            )
            self.validation.eval_global_batch_size = self.train.global_batch_size
        if self.validation.eval_micro_batch_size is None:
            assert self.train.micro_batch_size is not None, (
                "train.micro_batch_size must be set when eval_micro_batch_size is not explicitly configured"
            )
            self.validation.eval_micro_batch_size = self.train.micro_batch_size

        # Eval batch size divisibility check
        eval_dp_product = self.validation.eval_micro_batch_size * self.data_parallel_size
        assert self.validation.eval_global_batch_size % eval_dp_product == 0, (
            f"eval_global_batch_size ({self.validation.eval_global_batch_size}) must be divisible by "
            f"eval_micro_batch_size * data_parallel_size ({self.validation.eval_micro_batch_size} * "
            f"{self.data_parallel_size} = {eval_dp_product})"
        )

        # Megatron-FSDP and Torch FSDP2 are mutually-exclusive.
        if self.dist.use_megatron_fsdp and self.dist.use_torch_fsdp2:
            raise ValueError("use_megatron_fsdp and use_torch_fsdp2 are mutually exclusive.")
        # Validate Megatron-FSDP configuration.
        if self.dist.use_megatron_fsdp or self.ddp.use_megatron_fsdp:
            self._validate_and_apply_megatron_fsdp_configs()

        # Deterministic mode validations and settings
        self._validate_and_apply_deterministic_mode()

    def _validate_and_apply_megatron_fsdp_configs(self) -> None:
        """
        Validate Megatron-FSDP configuration when Megatron-FSDP is used.
        """
        # Set configs needed for Megatron-FSDP.
        self.dist.use_megatron_fsdp = True
        self.ddp.use_megatron_fsdp = True

        # Megatron-FSDP always uses a distributed optimizer.
        if not self.ddp.use_distributed_optimizer or not self.optimizer.use_distributed_optimizer:
            print_rank_0("use_distributed_optimizer=True is required for Megatron-FSDP. Activating...")
        self.ddp.use_distributed_optimizer = True
        self.optimizer.use_distributed_optimizer = True

        if self.optimizer.use_precision_aware_optimizer:
            print_rank_0("Megatron-FSDP installs gradients in `param.decoupled_grad` when using FusedAdam.")
            # Megatron-FSDP uses a decoupled gradient for FusedAdam.
            # Aligned with FusedAdam(use_decoupled_grad=True) and
            # clip_grad_norm(use_decoupled_grad=True)!
            self.ddp.megatron_fsdp_use_decoupled_grad = True

        if self.ddp.average_in_collective and not self.ddp.disable_symmetric_registration:
            print_rank_0("average_in_collective not supported with NCCL symmetric registration. Deactivating...")
            self.ddp.average_in_collective = False

        # reuse_grad_buf_for_mxfp8_param_ag is not implemented for Megatron-FSDP
        if self.ddp.reuse_grad_buf_for_mxfp8_param_ag or self.optimizer.reuse_grad_buf_for_mxfp8_param_ag:
            print_rank_0("reuse_grad_buf_for_mxfp8_param_ag not implemented for Megatron FSDP. Deactivating...")
            self.ddp.reuse_grad_buf_for_mxfp8_param_ag = False
            self.optimizer.reuse_grad_buf_for_mxfp8_param_ag = False

        # Assertions / Guards
        if self.checkpoint.save is not None or self.checkpoint.load is not None:
            # only check if saving or loading
            assert self.checkpoint.ckpt_format == "fsdp_dtensor", (
                "Megatron-FSDP requires the fsdp_dtensor checkpointing format!"
            )
        assert os.getenv("CUDA_DEVICE_MAX_CONNECTIONS") != "1", (
            "FSDP requires CUDA_DEVICE_MAX_CONNECTIONS > 1 or unset."
        )
        if self.ddp.nccl_ub:
            # Without manual registration, UBR is really slow.
            self.ddp.fsdp_manual_registration = True
        else:
            # Only compatible with NCCL UBR.
            assert not self.ddp.fsdp_manual_registration, "DDP.fsdp_manual_registration requires DDP.nccl_ub!"
        if self.ddp.data_parallel_sharding_strategy == "optim_grads_params":
            assert self.train.check_weight_hash_across_dp_replicas_interval is None, (
                "TrainingConfig.check_weight_hash_across_dp_replicas_interval is not "
                "supported with the Megatron-FSDP optim_grads_params sharding strategy"
            )
        assert not self.dist.use_tp_pp_dp_mapping, "use_tp_pp_dp_mapping is not supported with Megatron FSDP"

    def _validate_and_apply_deterministic_mode(self) -> None:
        """Apply and validate deterministic mode requirements.

        This enforces restrictions and settings that must hold when
        the model is configured to run in deterministic mode.
        """
        if not getattr(self.model, "deterministic_mode", False):
            return

        # Disallow flash attention when running deterministically
        if getattr(self.model, "attention_backend", None) == AttnBackend.flash:
            raise AssertionError("Flash attention can not be used in deterministic mode.")

        # Disallow cross-entropy loss fusion as it is not deterministic
        assert not getattr(self.model, "cross_entropy_loss_fusion", False), (
            "Cross Entropy Fusion is currently not deterministic."
        )

        all_reduce_choices = ("Tree", "Ring", "CollnetDirect", "CollnetChain", "^NVLS")
        assert os.getenv("NCCL_ALGO", -1) != -1 and os.getenv("NCCL_ALGO") in all_reduce_choices, (
            f"NCCL_ALGO must be one of {all_reduce_choices}."
        )

        # Enable deterministic algorithms in torch
        torch.use_deterministic_algorithms(True)
