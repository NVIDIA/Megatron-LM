# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import copy
import logging
import os
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from dataclasses import replace
from typing import Any, Type, TypeVar

try:
    import yaml

    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False

from megatron.core._rank_utils import safe_get_world_size
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.msc_utils import maybe_msc
from megatron.core.num_microbatches_calculator import (
    get_current_global_batch_size,
    update_num_microbatches,
)
from megatron.core.optimizer import OptimizerConfig
from megatron.core.tensor_parallel.gtp_parallel_layout import GTPParallelLayout
from megatron.training.config.common_config import DistributedInitConfig, ProfilingConfig, RNGConfig
from megatron.training.config.dataset_config import DataConfig
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
from megatron.training.config.utils import sanitize_dataclass_config, serialize_dataclass
from megatron.training.config.yaml_utils import safe_yaml_representers
from megatron.training.models import (
    BertModelConfig,
    LLaVAModelConfig,
    MimoBuildConfig,
    T5ModelConfig,
)
from megatron.training.models.base import ModelConfig

T = TypeVar("T", bound="ConfigContainerBase")
logger = logging.getLogger(__name__)

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

        return cls.from_dict(OmegaConf.to_container(conf, resolve=True), mode=mode)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the config container to a dictionary.

        Also converts any nested dataclasses (both ConfigContainer and regular dataclasses)
        to dictionaries recursively.

        Returns:
            Dictionary representation of this config
        """
        return serialize_dataclass(self)

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

@dataclass(kw_only=True)
class PretrainConfigContainer(ConfigContainerBase):
    """Training configuration; requires a model with a non-null transformer config."""

    train: TrainingConfig
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    model: ModelConfig | BertModelConfig | T5ModelConfig | LLaVAModelConfig | MimoBuildConfig
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

    def get_model_world_size(self) -> int:
        """Return ranks in the model grid, which may be a subset of the launch."""
        launch_world_size = safe_get_world_size()
        world_size = (
            self.model.mimo_llm_world_size if isinstance(self.model, MimoBuildConfig) else None
        )
        if world_size is None:
            world_size = launch_world_size
        if not 0 < world_size <= launch_world_size:
            raise ValueError("Training model world size must fit within the global launch")
        return world_size

    def get_parallel_layout(self, model_world_size: int) -> GTPParallelLayout:
        """Build the model's parallel layout without discovering the launch environment.

        ``model_world_size`` is the number of ranks assigned to this model, not
        necessarily the global launch size. Runtime callers resolve it through
        ``get_model_world_size()``; callers inspecting a hypothetical layout may
        supply it directly. Transformer CP already includes GTP sequence shards,
        whereas the layout's CP excludes them.
        """
        transformer = self.model.transformer
        tp = transformer.tensor_model_parallel_size
        pp = transformer.pipeline_model_parallel_size
        cp = transformer.context_parallel_size
        weight_size = transformer.gtp_weight_remat_size
        if min(tp, pp, cp, weight_size) < 1:
            raise ValueError("Parallel sizes must be positive")
        sequence_shards = transformer.gtp_remat_num_sequence_shards
        if cp % sequence_shards:
            raise ValueError("Runtime context-parallel size must contain the GTP sequence shards")
        return GTPParallelLayout(
            world_size=model_world_size,
            tp=tp,
            pp=pp,
            cp=cp // sequence_shards,
            weight_size=weight_size,
            num_sequence_shards=sequence_shards,
        )

    def get_data_parallel_size(self) -> int:
        """Return replica DP, excluding batch partitions within GTP weight groups."""
        return self.get_parallel_layout(self.get_model_world_size()).dp

    def get_batch_parallel_size(self) -> int:
        """Return independent batch partitions, including GTP batch shards but not CP."""
        return self.get_parallel_layout(self.get_model_world_size()).batch_parallel_size

    def _optimizer_with_model_precision(self) -> OptimizerConfig:
        """Return an optimizer copy with authoritative model precision, without mutation."""
        precision = {
            name: getattr(self.model.transformer, name)
            for name in ("fp16", "bf16", "params_dtype", "fp8_recipe")
        }
        return replace(self.optimizer, **precision)

    def _resolve_batch_sizes(
        self, batch_parallel_size: int | None = None
    ) -> tuple[int | None, int | None, int | None]:
        """Check effective train/eval batch sizes without mutating configuration."""
        micro_batch_size = self.train.micro_batch_size
        if batch_parallel_size is not None:
            assert batch_parallel_size > 0
            assert micro_batch_size is not None, "Training requires a micro batch size"
        if micro_batch_size is not None:
            assert micro_batch_size > 0, "Training micro batch size must be positive"
        global_batch_size = self.train.global_batch_size
        if global_batch_size is None and batch_parallel_size is not None:
            global_batch_size = micro_batch_size * batch_parallel_size
        if global_batch_size is not None:
            assert global_batch_size > 0, "Training global batch size must be positive"
        batch_multiple = micro_batch_size
        if batch_multiple is not None and batch_parallel_size is not None:
            batch_multiple *= batch_parallel_size
        # Reduction mode and schedules validate the effective/current batch in the
        # calculator; a schedule may resume after entries for a different topology.
        if (
            not self.train.decrease_batch_size_if_needed
            and self.train.step_batch_size_schedule is None
            and global_batch_size is not None
            and batch_multiple is not None
        ):
            assert global_batch_size % batch_multiple == 0, (
                "Training global batch size must be divisible by its microbatch size "
                "times batch parallel size"
            )
        eval_global_batch_size = self.validation.eval_global_batch_size
        if eval_global_batch_size is None:
            eval_global_batch_size = global_batch_size
        eval_micro_batch_size = self.validation.eval_micro_batch_size
        if eval_micro_batch_size is None:
            eval_micro_batch_size = micro_batch_size
        if eval_micro_batch_size is not None:
            assert eval_micro_batch_size > 0, "Evaluation micro batch size must be positive"
        if eval_global_batch_size is not None:
            assert eval_global_batch_size > 0, "Evaluation global batch size must be positive"
        eval_batch_multiple = eval_micro_batch_size
        if eval_batch_multiple is not None and batch_parallel_size is not None:
            eval_batch_multiple *= batch_parallel_size
        if eval_global_batch_size is not None and eval_batch_multiple is not None:
            assert eval_global_batch_size % eval_batch_multiple == 0, (
                "Evaluation global batch size must be divisible by its microbatch size "
                "times batch parallel size"
            )
        return global_batch_size, eval_global_batch_size, eval_micro_batch_size

    def _resolve_scheduler_step_fields(self):
        """Calculate user-unspecified scheduler configuration arguments from configurable fields."""

        # Without a training horizon there is nothing to resolve; building the optimizer param
        # scheduler (get_optimizer_param_scheduler) is what requires one.
        if self.train.train_iters is None and self.train.train_samples is None:
            return
        # Iteration-based training.
        if self.train.train_iters:
            if self.scheduler.lr_decay_iters is None:
                self.scheduler.lr_decay_iters = self.train.train_iters
            self.scheduler.lr_decay_steps = self.scheduler.lr_decay_iters * self.train.global_batch_size
            self.scheduler.wd_incr_steps = self.train.train_iters * self.train.global_batch_size
            self.scheduler.wsd_decay_steps = None
            if self.scheduler.lr_wsd_decay_iters is not None:
                self.scheduler.wsd_decay_steps = self.scheduler.lr_wsd_decay_iters * self.train.global_batch_size
            if self.scheduler.lr_warmup_fraction is not None:
                self.scheduler.lr_warmup_steps = self.scheduler.lr_warmup_fraction * self.scheduler.lr_decay_steps
            else:
                self.scheduler.lr_warmup_steps = self.scheduler.lr_warmup_iters * self.train.global_batch_size
        # Sample-based training.
        elif self.train.train_samples:
            # We need to set training iters for later use. Technically
            # we need to adjust the training samples too (due to last
            # batch being incomplete) but we leave it as is for now.
            update_train_iters(self.train)
            if self.scheduler.lr_decay_samples is None:
                self.scheduler.lr_decay_samples = self.train.train_samples
            self.scheduler.lr_decay_steps = self.scheduler.lr_decay_samples
            self.scheduler.wd_incr_steps = self.train.train_samples
            self.scheduler.wsd_decay_steps = self.scheduler.lr_wsd_decay_samples
            if self.scheduler.lr_warmup_fraction is not None:
                self.scheduler.lr_warmup_steps = self.scheduler.lr_warmup_fraction * self.scheduler.lr_decay_steps
            else:
                self.scheduler.lr_warmup_steps = self.scheduler.lr_warmup_samples


    def validate(self) -> None:
        """Validate each section and the cross-config GTP, batch-size, and RL policies."""
        if self.model is None:
            raise ValueError("Training requires a model config")
        if getattr(self.model, "transformer", None) is None:
            raise ValueError("Training requires model.transformer to be configured")
        self.rl.validate_ddp_config(self.ddp)
        optimizer = self._optimizer_with_model_precision()
        # Validate effective precision without modifying either config or finalizing early.
        ConfigContainerBase.validate(replace(self, optimizer=optimizer))
        # Config-only topology checks also run when validating a recipe off-cluster.
        # Launch world-size checks remain in the runtime parallel-size helpers.
        transformer = self.model.transformer
        tp = transformer.tensor_model_parallel_size
        pp = transformer.pipeline_model_parallel_size
        cp = transformer.context_parallel_size
        weight_size = transformer.gtp_weight_remat_size
        if min(tp, pp, cp, weight_size) < 1:
            raise ValueError("Parallel sizes must be positive")
        sequence_shards = transformer.gtp_remat_num_sequence_shards
        if cp % sequence_shards:
            raise ValueError("Runtime context-parallel size must contain the GTP sequence shards")
        if isinstance(self.model, MimoBuildConfig) and self.model.mimo_llm_world_size is not None:
            world_size = self.model.mimo_llm_world_size
            if world_size <= 0:
                raise ValueError("MIMO language-model world size must be positive")
            if world_size % (tp * pp * (cp // sequence_shards) * weight_size):
                raise ValueError(
                    "MIMO language-model world size must be divisible by its model parallel size"
                )
        if sequence_shards > 1:
            if self.checkpoint.ckpt_format != "torch_dist":
                raise ValueError("GTP sequence sharding requires torch_dist checkpoints")
            if self.dist.use_torch_fsdp2 or self.ddp.use_megatron_fsdp:
                raise ValueError("GTP sequence sharding uses Megatron DDP, not FSDP")
            if self.dist.use_tp_pp_dp_mapping:
                raise ValueError("GTP sequence sharding does not support TP-PP-DP mapping")
        global_batch_size, _, _ = self._resolve_batch_sizes()
        self._validate_rl_config(global_batch_size)
        optimizer.validate_run(
            ddp_config=self.ddp,
            use_torch_fsdp2=self.dist.use_torch_fsdp2,
            ckpt_format=self.checkpoint.ckpt_format,
        )
        optimizer.validate_model(
            moe_single_grouped_weight=transformer.moe_single_grouped_weight,
            virtual_pipeline_model_parallel_size=transformer.virtual_pipeline_model_parallel_size,
        )
        self.scheduler.validate_run(
            train_iters=self.train.train_iters,
            train_samples=self.train.train_samples,
            skip_train=self.validation.skip_train,
            weight_decay=self.optimizer.weight_decay,
        )
        self._validate_data_section()

    def _validate_rl_config(self, global_batch_size: int | None) -> None:
        """Check RL policy against the supplied or resolved training batch size."""
        transformer = getattr(self.model, "transformer", None)
        cuda_graph_impl = None if transformer is None else transformer.cuda_graph_impl
        self.rl.validate_run(
            cuda_graph_impl=cuda_graph_impl,
            micro_batch_size=self.train.micro_batch_size,
            global_batch_size=global_batch_size,
            train_iters=self.train.train_iters,
            train_samples=self.train.train_samples,
            skip_train=self.validation.skip_train,
            load_optim=self.checkpoint.load_optim,
            save_interval=self.checkpoint.save_interval,
            exit_interval=self.train.exit_interval,
            # The rollout deployment (inference shards, model CP) is not a pretrain section;
            # validate_args checks that rule from the launch arguments.
            separate_inference_model=None,
        )

    def _validate_data_section(self) -> None:
        """Cross-section checks for ``data.dataset``, one per field.

        The model capacity covers the sample length. A value given on the dataset config for one
        of the six fields other sections own (``CONTAINER_WRITTEN_FIELDS``) must agree with that
        section: the container writes these fields at ``finalize()``, the arguments and the saved
        file leave them at Megatron Core's defaults, so a differing explicit value is a mistake.
        ``data_parallel_size`` needs the world size as well and is checked at ``finalize()``.
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
        validation = self.validation
        if (
            dataset.full_validation is not None
            and dataset.full_validation != validation.full_validation
        ):
            raise ValueError(
                f"data.dataset.full_validation={dataset.full_validation} disagrees with "
                f"validation.full_validation={validation.full_validation}; the container writes "
                "this field at finalize(), leave it unset."
            )
        if (
            dataset.multiple_validation_sets is not None
            and dataset.multiple_validation_sets != validation.multiple_validation_sets
        ):
            raise ValueError(
                f"data.dataset.multiple_validation_sets={dataset.multiple_validation_sets} "
                "disagrees with "
                f"validation.multiple_validation_sets={validation.multiple_validation_sets}; the "
                "container writes this field at finalize(), leave it unset."
            )
        transformer = self.model.transformer
        if (
            dataset.context_parallel_size is not None
            and dataset.context_parallel_size != transformer.context_parallel_size
        ):
            raise ValueError(
                f"data.dataset.context_parallel_size={dataset.context_parallel_size} disagrees "
                "with model.transformer.context_parallel_size="
                f"{transformer.context_parallel_size}; the container writes this field at "
                "finalize(), leave it unset."
            )
        sequence_parallel_size = transformer.tensor_model_parallel_size * int(
            bool(transformer.sequence_parallel)
        )
        if dataset.sequence_parallel_size not in (0, sequence_parallel_size):
            raise ValueError(
                f"data.dataset.sequence_parallel_size={dataset.sequence_parallel_size} disagrees "
                "with the model layout (tensor_model_parallel_size * sequence_parallel = "
                f"{sequence_parallel_size}); the container writes this field at finalize(), leave "
                "it unset."
            )
        dynamic_context_parallel = bool(getattr(transformer, "dynamic_context_parallel", False))
        # The deprecated hybrid_context_parallel alias counts as the canonical flag here (Core
        # warns about it in validate_without_tokenizer()); finalize() folds it and clears it.
        if (
            dataset.dynamic_context_parallel or dataset.hybrid_context_parallel
        ) and not dynamic_context_parallel:
            raise ValueError(
                "data.dataset.dynamic_context_parallel=True disagrees with "
                "model.transformer.dynamic_context_parallel=False; the container writes this field "
                "at finalize(), leave it unset."
            )

    def _finalize(self) -> None:
        dataset = self.data.dataset
        transformer = self.model.transformer
        data_parallel_size = None
        if dataset is not None:
            # The world size is known here: data_parallel_size gets the agreement check the other
            # fields got at validation, before anything is written.
            data_parallel_size = self.get_data_parallel_size()
            if dataset.data_parallel_size not in (1, data_parallel_size):
                raise ValueError(
                    f"data.dataset.data_parallel_size={dataset.data_parallel_size} disagrees with "
                    "the model layout and the world size "
                    f"(data_parallel_size={data_parallel_size}); the container writes this field "
                    "at finalize(), leave it unset."
                )
        batch_sizes = self._resolve_batch_sizes(self.get_batch_parallel_size())
        self._validate_rl_config(batch_sizes[0])
        # Resolve RNG and model settings before deriving optimizer precision.
        super()._finalize()
        finalize_model = getattr(self.model, "finalize", None)
        if callable(finalize_model):
            finalize_model()
        optimizer = self._optimizer_with_model_precision()
        changes = {
            name: (getattr(self.optimizer, name), getattr(optimizer, name))
            for name in ("fp16", "bf16", "params_dtype", "fp8_recipe")
            if getattr(self.optimizer, name) != getattr(optimizer, name)
        }
        if changes:
            logger.warning("Overriding optimizer precision from model config (old, new): %s", changes)
        (
            self.train.global_batch_size,
            self.validation.eval_global_batch_size,
            self.validation.eval_micro_batch_size,
        ) = batch_sizes
        self.optimizer = optimizer
        self.optimizer.finalize_optimizer_config()
        if self.optimizer.use_layer_wise_distributed_optimizer:
            self.ddp.use_distributed_optimizer = False
        self.scheduler.finalize_scheduler_config(self.optimizer.weight_decay)
        self._resolve_scheduler_step_fields()
        if dataset is not None:
            # The seven fields other sections own (CONTAINER_WRITTEN_FIELDS): the validation flags
            # from the validation config, the layout from the required model config (the
            # deprecated hybrid_context_parallel alias is folded into dynamic_context_parallel).
            dataset.full_validation = self.validation.full_validation
            dataset.multiple_validation_sets = self.validation.multiple_validation_sets
            dataset.data_parallel_size = data_parallel_size
            dataset.context_parallel_size = transformer.context_parallel_size
            dataset.sequence_parallel_size = transformer.tensor_model_parallel_size * int(
                bool(transformer.sequence_parallel)
            )
            dataset.dynamic_context_parallel = bool(
                getattr(transformer, "dynamic_context_parallel", False)
            )
            dataset.hybrid_context_parallel = False
        self.data.finalize()
        if self.rl.perform_rl_step:
            self.rl.finalize_ddp_config(self.ddp)


def update_train_iters(train: TrainingConfig) -> None:
    """Resolve the sample-based training horizon on the supplied training config."""
    # Imported lazily: megatron.training.utils imports megatron.training.config.
    from megatron.training.utils import print_rank_0

    # For iteration-based training, we don't need to do anything
    if train.train_iters:
        return

    if train.step_batch_size_schedule is not None:
        # Sample based training with step batch size schedule.
        iterations = 0
        consumed_samples = 0
        while consumed_samples < train.train_samples:
            update_num_microbatches(consumed_samples, consistency_check=False)
            consumed_samples += get_current_global_batch_size()
            iterations += 1
        # Reset
        update_num_microbatches(0, consistency_check=False)
        train.train_iters = iterations
    else:
        # Constant batch size with sample-based training.
        train.train_iters = train.train_samples // train.global_batch_size

    print_rank_0(f'setting training iterations to {train.train_iters}')


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

    model: ModelConfig | BertModelConfig | T5ModelConfig | LLaVAModelConfig | MimoBuildConfig | None
    """Model to load, or None for args-only entrypoints without model inputs."""

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
