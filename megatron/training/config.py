# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
from dataclasses import dataclass, field
import signal
from typing import Literal

@dataclass(kw_only=True)
class TrainingConfig:
    """Configuration settings related to the training loop."""

    micro_batch_size: int | None = None
    """Batch size per model instance (local batch size). Global batch size is local batch size times
    data parallel size times number of micro batches."""

    global_batch_size: int | None = None
    """Training batch size. If set, it should be a multiple of micro-batch-size times
    data-parallel-size. If this value is None, then use micro-batch-size * data-parallel-size
    as the global batch size. This choice will result in 1 for number of micro-batches."""

    rampup_batch_size: list[int] | None = field(default=None, metadata={"argparse_meta": {"nargs": 3}})
    """Batch size ramp up with the following values: <start batch size>, <batch size increment>,
    <ramp-up samples>
    For example:
        rampup-batch-size = [16, 8, 300000]
        global-batch-size 1024
    will start with global batch size 16 and over (1024 - 16) / 8 = 126 intervals will increase
    the batch size linearly to 1024. In each interval we will use approximately
    300000 / 126 = 2380 samples.
    """

    decrease_batch_size_if_needed: bool = False
    """If set, decrease batch size if microbatch_size * dp_size does not 
    divide batch_size. Old batch_size will be restored if training is re-started 
    with dp_size that divides batch_size // microbatch_size."""

    empty_unused_memory_level: Literal[0, 1, 2] = 0
    """Call torch.cuda.empty_cache() each iteration (training and eval), to reduce fragmentation.
    0=off, 1=moderate, 2=aggressive.
    """

    check_weight_hash_across_dp_replicas_interval: int | None = None
    """Interval to check weight hashes are same across DP replicas. If not specified, weight hashes not checked."""

    train_sync_interval: int | None = None
    """Training CPU-GPU synchronization interval, to ensure that CPU is not running too far ahead of GPU."""

    train_iters: int | None = None
    """Total number of iterations to train over all training runs.
    Note that either train_iters or train_samples should be provided.
    """

    train_samples: int | None = None
    """Total number of samples to train over all training runs.
    Note that either train_iters or train_samples should be provided."""

    exit_interval: int | None = None
    """Exit the program after the iteration is divisible by this value."""

    exit_duration_in_mins: int | None = None
    """Exit the program after this many minutes."""

    exit_signal_handler: bool = False
    """Dynamically save the checkpoint and shutdown the training if SIGTERM is received"""

    exit_signal: signal.Signals = signal.SIGTERM
    """Signal for the signal handler to detect."""

    exit_signal_handler_for_dataloader: bool = False
    """Use signal handler for dataloader workers"""

    manual_gc: bool = False
    """Disable the threshold-based default garbage collector and trigger the garbage collection
    manually. Manual garbage collection helps to align the timing of the collection across ranks
    which mitigates the impact of CPU-associated jitters. When the manual gc is enabled, garbage
    collection is performed only at the start and the end of the validation routine by default."""

    manual_gc_interval: int = 0
    """Training step interval to trigger manual garbage collection. Values > 0 will trigger garbage
    collections between training steps.
    """

    manual_gc_eval: bool = True
    """When using manual garbage collection, this controls garbage collection at the start and the
    end of each evaluation run.
    """

    iterations_to_skip: list[int] = field(default_factory=list)
    """List of iterations to skip during training, empty by default."""


@dataclass(kw_only=True)
class ValidationConfig:
    """Configuration settings related to validation during or after model training."""

    eval_iters: int | None = 100
    """Number of iterations to run for evaluation. Used for both validation and test. If not set,
    evaluation will not run."""

    eval_interval: int | None = None
    """Interval between running evaluation on validation set. If not set, evaluation will not run
    during training.
    """

    skip_train: bool = False
    """If set, bypass the training loop, perform evaluation for validation/test, and exit."""

    test_mode: bool = False
    """Run all real-time test alongside the experiment."""

    full_validation: bool = False
    """If set, each time validation occurs it uses the full validation dataset(s). This currently only works for GPT datasets!"""

    multiple_validation_sets: bool = False
    """If set, multiple datasets listed in the validation split are evaluated independently with a
       separate loss for each dataset in the list. This argument requires that no weights are 
       included in the list.
    """


@dataclass
class StragglerDetectionConfig:
    """Configuration settings for detecting and logging GPU stragglers."""

    log_straggler: bool = False
    """If set, tracks and logs straggler per GPU."""

    straggler_ctrlr_port: int = 65535
    """Port number to toggle StragglerDetector on/off at runtime"""

    straggler_minmax_count: int = 1
    """Number of ranks to report with high/low estimated throughput"""

    disable_straggler_on_startup: bool = False
    """If set, StragglerDetector is disabled on startup."""

