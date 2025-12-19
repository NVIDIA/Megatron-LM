# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
from dataclasses import dataclass, field
import signal
from typing import Literal
import os

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


@dataclass(kw_only=True)
class DistributedInitConfig:
    """Configuration settings for distributed training initialization."""

    # ---------------- Distributed config. ----------------

    distributed_backend: Literal["nccl", "gloo"] = "nccl"
    """Which backend to use for distributed training."""

    distributed_timeout_minutes: int = 10
    """Timeout minutes for torch.distributed."""

    align_grad_reduce: bool = True
    """If not set, all PP stages will launch gradient reduces simultaneously.
    Otherwise, each PP stage will independently launch as needed.
    """

    local_rank: int = field(default_factory=lambda: int(os.getenv("LOCAL_RANK", "0")))
    """local rank passed from distributed launcher."""

    lazy_mpu_init: bool = False
    """If set to True, initialize_megatron() skips DDP initialization and returns function to complete it instead.
    Also turns on --use-cpu-initialization flag. This is for external DDP manager."""

    use_megatron_fsdp: bool = False
    """Use Megatron's Fully Sharded Data Parallel. Cannot be used together with use_torch_fsdp2."""

    use_torch_fsdp2: bool = False
    """Use the torch FSDP2 implementation. FSDP2 is not currently working with Pipeline Parallel.
    It is still not in a stable release stage, and may therefore contain bugs or other
    potential issues."""

    nccl_communicator_config_path: str | None = None
    """Path to the yaml file with NCCL communicator configurations. The number of min/max thread
    groups and thread group cluster size of each communicator can be configured by setting
    `min_ctas`, `max_ctas`, and `cga_cluster_size`."""

    use_tp_pp_dp_mapping: bool = False
    """If set, distributed ranks initialize order is changed from tp-cp-ep-dp-pp to tp-cp-ep-pp-dp"""

    enable_gloo_process_groups: bool = True
    """If set, create Gloo process groups for communications."""

    use_sharp: bool = False
    """Set the use of SHARP for the collective communications of data-parallel process groups.
    When `True`, run barrier within each data-parallel process group,
    which specifies the SHARP application target groups.
    """

    sharp_enabled_group: Literal["dp", "dp_replica"] | None = None
    """IB SHARP can be enabled from only one communication group.
    By default, it is enabled from dp group if not specified and use_sharp=True.
    Available options: [dp, dp_replica]
    """

    high_priority_stream_groups: list[str] | None = None
    """Specify which communicator groups should use high priority streams during creation.
    Assigning high priority to communication streams ensures that communication kernels
    are scheduled with higher priority, minimizing the exposed communication when it is
    overlapped with other computation kernels.
    """

    distributed_timeout_seconds_after_init: int | None = None
    """Timeout in seconds for process groups after initialization. This timeout is applied to all process groups after initialization and the first iteration completes."""
