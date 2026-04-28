# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
from dataclasses import dataclass, field

from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from megatron.training.config.common_config import DistributedInitConfig, ProfilingConfig, RNGConfig
from megatron.training.config.resilience_config import (
    RerunStateMachineConfig,
    StragglerDetectionConfig,
)
from megatron.training.config.training_config import (
    CheckpointConfig,
    LoggerConfig,
    SchedulerConfig,
    TrainingConfig,
    ValidationConfig,
)


@dataclass(kw_only=True)
class PretrainConfigContainer:
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
