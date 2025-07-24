# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
from dataclasses import dataclass
import signal
from typing import Optional, Literal

@dataclass(kw_only=True)
class TrainingConfig:
    """Configuration settings related to the training loop and validation."""

    # ---------------- Training config. ----------------

    micro_batch_size: Optional[int] = None
    """Batch size per model instance (local batch size). Global batch size is local batch size times
    data parallel size times number of micro batches."""

    global_batch_size: Optional[int] = None
    """Training batch size. If set, it should be a multiple of micro-batch-size times
    data-parallel-size. If this value is None, then use micro-batch-size * data-parallel-size
    as the global batch size. This choice will result in 1 for number of micro-batches."""

    empty_unused_memory_level: Literal[0, 1, 2] = 0
    """Call torch.cuda.empty_cache() each iteration (training and eval), to reduce fragmentation.
    0=off, 1=moderate, 2=aggressive.
    """

    check_weight_hash_across_dp_replicas_interval: Optional[int] = None
    """Interval to check weight hashes are same across DP replicas. If not specified, weight hashes not checked."""

    train_sync_interval: Optional[int] = None
    """Training CPU-GPU synchronization interval, to ensure that CPU is not running too far ahead of GPU."""

    train_iters: Optional[int] = None
    """Total number of iterations to train over all training runs."""

    exit_interval: Optional[int] = None
    """Exit the program after the iteration is divisible by this value."""

    exit_duration_in_mins: Optional[int] = None
    """Exit the program after this many minutes."""

    exit_signal_handler: bool = False
    """Dynamically save the checkpoint and shutdown the training if SIGTERM is received"""

    exit_signal: int = signal.SIGTERM
    """Signal for the signal handler to detect."""

    exit_signal_handler_for_dataloader: bool = False
    """Use signal handler for dataloader workers"""

    manual_gc: bool = False
    """Disable the threshold-based default garbage collector and trigger the garbage collection
    manually. Manual garbage collection helps to align the timing of the collection across ranks
    which mitigates the impact of CPU-associated jitters. When the manual gc is enabled, garbage
    collection is performed only at the start and the end of the validation routine by default."""

    manual_gc_interval: int = 0
    """Training step interval to trigger manual garbage collection.
    When the value is set to 0, garbage collection is not triggered between training steps.
    """

    manual_gc_eval: bool = True
    """When using manual garbage collection,
    disable garbage collection at the start and the end of each evaluation run.
    """

    # ---------------- Validation config. ----------------

    eval_iters: int = 100
    """Number of iterations to run for evaluation validation/test for."""

    eval_interval: Optional[int] = 1000
    """Interval between running evaluation on validation set."""

    skip_train: bool = False
    """If set, bypass the training loop, optionally do evaluation for validation/test, and exit."""
