# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
import copy
import os
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
import warnings
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
from megatron.training.utils import print_rank_0, warn_rank_0
from megatron.core.transformer.enums import AttnBackend, CudaGraphScope
from megatron.core.transformer import TransformerConfig

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

        # Run validations
        self._validate_and_sync_distributed_optimizer_settings()
        self._validate_mixed_precision_consistency()
        self._validate_fine_grained_activation_offloading()

        # CUDA graph scope validation: check_for_nan_in_loss must be disabled with full_iteration graph
        if self.model.cuda_graph_impl == "local" and CudaGraphScope.full_iteration in self.model.cuda_graph_scope:
            assert not self.rerun_state_machine.check_for_nan_in_loss, (
                "check_for_nan_in_loss must be disabled when using full_iteration CUDA graph. "
                "Set rerun_state_machine.check_for_nan_in_loss=False."
            )
        if self.model.cuda_graph_impl == "none":
            self.model.cuda_graph_scope = []

        # ModelOpt/Quantization checks
        if getattr(self.model, "restore_modelopt_state", False):
            assert not self.model.gradient_accumulation_fusion, (
                "Gradient accumulation fusion is not supported with ModelOpt/Quantized models. "
                "Please set model.gradient_accumulation_fusion=False"
            )

        self.model.use_cpu_initialization = self.model.use_cpu_initialization or self.dist.lazy_mpu_init

        # Make sure all functionality that requires Gloo process groups is disabled.
        if not self.dist.use_gloo_process_groups:
            if self.optimizer.use_distributed_optimizer:
                # If using distributed optimizer, must use distributed checkpointing.
                # Legacy checkpointing uses Gloo process groups to collect full distributed
                # optimizer state in the CPU memory of DP rank 0.
                assert self.checkpoint.ckpt_format == "torch_dist"

        # Cross-validation between training and scheduler configs
        self._validate_training_scheduler_compatibility()

        # Calculate scheduler steps for both iteration-based and sample-based training
        self._calculate_scheduler_steps()

        if self.model.context_parallel_size > 1:
            assert self.model.seq_length % (self.model.context_parallel_size * 2) == 0, (
                "Sequence length must be divisible by 2 * context parallel size if context parallel is used."
            )

        self._validate_cp_comm_type()

        # Validate DeepEP or HybridEP is supported for the current GPU architecture
        if isinstance(self.model,  HybridModelConfig):
            validate_flex_dispatcher_backend(self.model.transformer)

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

    def _validate_and_sync_distributed_optimizer_settings(self) -> None:
        """Validate and synchronize distributed optimizer settings between DDP and optimizer configs.

        This function ensures that distributed optimizer settings are consistent across
        DDP and optimizer configurations. If either setting is enabled, both will be
        enabled to maintain consistency.
        """
        ddp_setting = self.ddp.use_distributed_optimizer
        optimizer_setting = self.optimizer.use_distributed_optimizer

        if ddp_setting or optimizer_setting:
            if ddp_setting != optimizer_setting:
                warn_rank_0(
                    f"Distributed optimizer settings were not in sync: "
                    f"ddp.use_distributed_optimizer={ddp_setting}, "
                    f"optimizer.use_distributed_optimizer={optimizer_setting}. "
                    f"Automatically enabling distributed optimizer for both settings."
                )
            self.ddp.use_distributed_optimizer = True
            self.optimizer.use_distributed_optimizer = True

    def _validate_mixed_precision_consistency(self) -> None:
        """Validate that mixed precision settings are consistent between model and optimizer configs.

        Raises:
            AssertionError: If precision settings are inconsistent in a way that would
                indicate ambiguous behavior.
        """
        model_cfg = self.model
        optimizer_cfg = self.optimizer

        # Mutually exclusive: cannot have both bf16 and fp16 enabled
        assert not (model_cfg.bf16 and model_cfg.fp16), (
            "Model config cannot have both bf16=True and fp16=True. Please set only one precision mode."
        )
        assert not (optimizer_cfg.bf16 and optimizer_cfg.fp16), (
            "Optimizer config cannot have both bf16=True and fp16=True. Please set only one precision mode."
        )

        # Validate across model and optimizer configs
        if optimizer_cfg.use_precision_aware_optimizer:
            # For bf16 training: optimizer.bf16 must match model.bf16
            if model_cfg.bf16:
                assert optimizer_cfg.bf16, (
                    "optimizer.bf16=True must be set when model.bf16=True and use_precision_aware_optimizer=True."
                )
            # For fp16 training: optimizer.fp16 must match model.fp16
            if model_cfg.fp16:
                assert optimizer_cfg.fp16, (
                    "optimizer.fp16=True must be set when model.fp16=True and use_precision_aware_optimizer=True."
                )
            # For fp32 training (neither bf16 nor fp16 on model)
            if not model_cfg.bf16 and not model_cfg.fp16:
                assert not optimizer_cfg.bf16 and not optimizer_cfg.fp16, (
                    "optimizer.bf16 and optimizer.fp16 must both be False when "
                    "model is using fp32 precision (model.bf16=False, model.fp16=False) and "
                    "use_precision_aware_optimizer=True."
                )

    def _validate_fine_grained_activation_offloading(self) -> None:
        """Validate fine-grained activation offloading configuration.

        This function ensures that fine-grained activation offloading is only enabled
        with compatible configurations (transformer_engine implementation) and that
        necessary environment variables are set for newer TE versions.

        Args:
            config: The configuration container to validate.

        Raises:
            ValueError: If fine-grained activation offloading is enabled with incompatible settings.
        """
        from megatron.core.utils import is_te_min_version

        model_cfg = self.model

        if not model_cfg.fine_grained_activation_offloading:
            return

        # Fine-grained activation offloading requires transformer_engine implementation
        if model_cfg.transformer_impl != "transformer_engine":
            raise ValueError(
                "Fine-grained activation offloading is only supported with transformer_engine implementation. "
                f"Current transformer_impl: {model_cfg.transformer_impl}"
            )

        # For TE >= 2.10.0, NVTE_CPU_OFFLOAD_V1 must be set to avoid offloading weights
        if is_te_min_version("2.10.0"):
            if os.getenv("NVTE_CPU_OFFLOAD_V1", "0") != "1":
                raise ValueError(
                    "For fine-grained activation offloading with TE >= 2.10.0, "
                    "NVTE_CPU_OFFLOAD_V1 environment variable should be set to 1 to avoid offloading weights."
                )

    def _validate_training_scheduler_compatibility(self) -> None:
        """Cross-validation between training and scheduler configs."""
        has_train_samples = self.train.train_samples is not None

        if has_train_samples:
            # Sample-based training validation
            assert self.scheduler.lr_decay_iters is None, (
                "Use lr_decay_samples for sample-based training, not lr_decay_iters"
            )
            assert self.scheduler.lr_warmup_iters == 0, (
                "Use lr_warmup_samples for sample-based training, not lr_warmup_iters"
            )
            assert not (self.scheduler.lr_warmup_fraction is not None and self.scheduler.lr_warmup_samples != 0), (
                "Can only specify one of lr_warmup_fraction or lr_warmup_samples"
            )
        else:
            # Iteration-based training validation
            assert self.scheduler.lr_decay_samples is None, (
                "Use lr_decay_iters for iteration-based training, not lr_decay_samples"
            )
            assert self.scheduler.lr_warmup_samples == 0, (
                "Use lr_warmup_iters for iteration-based training, not lr_warmup_samples"
            )
            assert not (self.scheduler.lr_warmup_fraction is not None and self.scheduler.lr_warmup_iters != 0), (
                "Can only specify one of lr_warmup_fraction or lr_warmup_iters"
            )

    def _calculate_scheduler_steps(self) -> None:
        """Calculate scheduler steps for both iteration-based and sample-based training."""
        is_sample_based = self.train.train_samples is not None

        if is_sample_based:
            if self.scheduler.lr_decay_samples is None:
                self.scheduler.lr_decay_samples = self.train.train_samples
            self.scheduler.lr_decay_steps = self.scheduler.lr_decay_samples
            self.scheduler.wd_incr_steps = self.train.train_samples

            if self.scheduler.lr_wsd_decay_samples is not None:
                self.scheduler.wsd_decay_steps = self.scheduler.lr_wsd_decay_samples

            # Warmup calculation for sample-based training
            if self.scheduler.lr_warmup_fraction is not None:
                self.scheduler.lr_warmup_steps = self.scheduler.lr_warmup_fraction * self.scheduler.lr_decay_steps
            else:
                self.scheduler.lr_warmup_steps = self.scheduler.lr_warmup_samples
        else:
            # Iteration-based training
            if self.scheduler.lr_decay_iters is None:
                self.scheduler.lr_decay_iters = self.train.train_iters
            if self.scheduler.lr_wsd_decay_iters is None and self.scheduler.lr_decay_style == "WSD":
                self.scheduler.lr_wsd_decay_iters = self.scheduler.lr_decay_iters
            self.scheduler.lr_decay_steps = self.scheduler.lr_decay_iters * self.train.global_batch_size
            self.scheduler.wd_incr_steps = self.train.train_iters * self.train.global_batch_size

            if self.scheduler.lr_wsd_decay_iters is not None:
                self.scheduler.wsd_decay_steps = self.scheduler.lr_wsd_decay_iters * self.train.global_batch_size

            if self.scheduler.lr_warmup_fraction is not None:
                self.scheduler.lr_warmup_steps = self.scheduler.lr_warmup_fraction * self.scheduler.lr_decay_steps
            else:
                self.scheduler.lr_warmup_steps = self.scheduler.lr_warmup_iters * self.train.global_batch_size

        # Enforce the Megatron Core invariant: lr_warmup_steps must be < lr_decay_steps.
        # This can be violated when train_iters is small (e.g. smoke runs) while
        # lr_warmup_iters is tuned for a full-length training run.
        if self.scheduler.lr_decay_steps <= 0:
            raise ValueError(
                f"lr_decay_steps must be > 0, got {self.scheduler.lr_decay_steps}. "
                "Please increase train_iters/train_samples or lr_decay_iters/lr_decay_samples."
            )
        if self.scheduler.lr_warmup_steps >= self.scheduler.lr_decay_steps:
            capped = self.scheduler.lr_decay_steps - 1
            warnings.warn(
                f"lr_warmup_steps ({self.scheduler.lr_warmup_steps}) >= lr_decay_steps "
                f"({self.scheduler.lr_decay_steps}); capping lr_warmup_steps to {capped}. "
                "Reduce lr_warmup_iters (or lr_warmup_samples) for short training runs.",
                UserWarning,
                stacklevel=2,
            )
            self.scheduler.lr_warmup_steps = capped

    def _validate_cp_comm_type(self) -> None:
        """Validate cp_comm_type and hierarchical_context_parallel_sizes consistency."""
        cp_comm_type = getattr(self.model, "cp_comm_type", None)
        hcp_sizes = getattr(self.model, "hierarchical_context_parallel_sizes", None)
        cp_size = getattr(self.model, "context_parallel_size", 1)

        if cp_size > 1 and cp_comm_type is not None:
            if isinstance(cp_comm_type, list):
                assert len(cp_comm_type) == self.model.num_layers, (
                    f"Length of cp_comm_type ({len(cp_comm_type)}) must equal num_layers ({self.model.num_layers})."
                )
            else:
                assert isinstance(cp_comm_type, str), (
                    f"cp_comm_type must be a str or list of str, got {type(cp_comm_type)}."
                )

        cp_comm_types = cp_comm_type if isinstance(cp_comm_type, list) else [cp_comm_type or "p2p"]
        if any("a2a+p2p" in ct for ct in cp_comm_types):
            assert hcp_sizes is not None, (
                "hierarchical_context_parallel_sizes must be set when cp_comm_type "
                "contains 'a2a+p2p'. Without it, CP communication is silently disabled "
                "and each rank attends only to its local chunk, producing artificially "
                "high throughput but broken training. Example: for cp=16 across 4 nodes "
                "of 8 GPUs, set hierarchical_context_parallel_sizes=[8, 2]."
            )

        if hcp_sizes is not None:
            from math import prod

            assert prod(hcp_sizes) == cp_size, (
                f"Product of hierarchical_context_parallel_sizes {hcp_sizes} "
                f"(={prod(hcp_sizes)}) must equal context_parallel_size (={cp_size})."
            )


def validate_flex_dispatcher_backend(model_config: TransformerConfig) -> None:
    """Validate DeepEP or HybridEP is supported for the current GPU architecture."""
    if model_config.moe_token_dispatcher_type == "flex":
        device_properties = torch.cuda.get_device_properties(0)
        if model_config.moe_flex_dispatcher_backend == "deepep":
            if not (
                device_properties.major in (8, 9) or device_properties.name.startswith(("NVIDIA B200", "NVIDIA B300"))
            ):
                raise ValueError(
                    f"DeepEP is supported for Ampere, Hopper, and Blackwell (B200/B300) GPUs. "
                    f"Current GPU: {device_properties.name}"
                )

        if model_config.moe_flex_dispatcher_backend == "hybridep":
            if not device_properties.major in [8, 9, 10]:
                raise ValueError(
                    "HybridEP is supported for GB200, GB300 with NVL72 and for Ampere, Hopper, B200 and B300 GPUs"
                )
