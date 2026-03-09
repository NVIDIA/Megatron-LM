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


@dataclass(kw_only=True)
class SchedulerConfig:
    """Configuration settings for the learning rate scheduler and weight decay."""

    # ---------------- Learning rate config. ----------------
    lr_decay_style: Literal["constant", "linear", "cosine", "inverse-square-root", "WSD"] = "linear"
    """Learning rate decay function."""

    lr_wsd_decay_style: Literal["exponential", "linear", "cosine", "minus_sqrt"] = "exponential"
    """Decay style for the annealing phase of WSD"""

    lr_decay_iters: int | None = None
    """number of iterations to decay learning rate over, If None defaults to train iters"""

    lr_decay_samples: int | None = None
    """number of samples to decay learning rate over, If None defaults to train samples"""

    lr_wsd_decay_iters: int | None = None
    """number of iterations for the annealing phase in the wsd schedule"""

    lr_wsd_decay_samples: int | None = None
    """number of samples for the annealing phase in the wsd schedule"""

    lr_warmup_fraction: float | None = None
    """fraction of lr-warmup-(iters/samples) to use for warmup (as a float)"""

    lr_warmup_iters: int = 0
    """number of iterations to linearly warmup learning rate over."""

    lr_warmup_samples: int = 0
    """number of samples to linearly warmup learning rate over."""

    lr_warmup_init: float = 0.0
    """Initial value for learning rate warmup. The scheduler starts warmup from this value."""

    lr_decay_steps: int | None = field(init=False, default=None)
    """number of samples to decay learning rate over. Calculated at runtime from 
    lr_decay_iters or lr_decay_samples.
    """

    lr_warmup_steps: int | None = field(init=False, default=None)
    """number of samples to warmup learning rate over. Calculated at runtime from
    lr_warmup_fraction, lr_warmup_iters, or lr_warmup_samples.
    """
    
    override_opt_param_scheduler: bool = field(default=False, metadata={"argparse_meta": {"arg_names": ["--override-opt_param-scheduler", "--override-opt-param-scheduler"]}})
    """Reset the values of the scheduler (learning rate, warmup iterations, minimum learning rate,
    maximum number of iterations, and decay style) from input arguments and ignore values from
    checkpoints. Note that all the above values will be reset."""

    use_checkpoint_opt_param_scheduler: bool = field(default=False, metadata={"argparse_meta": {"arg_names": ["--use-checkpoint-opt_param-scheduler", "--use-checkpoint-opt-param-scheduler"]}})
    """Use checkpoint to set the values of the scheduler (learning rate, warmup iterations,
    minimum learning rate, maximum number of iterations, and decay style) from checkpoint
    and ignore input arguments."""

    # ---------------- Regularization config. ----------------

    start_weight_decay: float | None = None
    """Initial weight decay coefficient for L2 regularization."""

    end_weight_decay: float | None = None
    """End of run weight decay coefficient for L2 regularization."""

    weight_decay_incr_style: Literal["constant", "linear", "cosine"] = "constant"
    """Weight decay increment function."""

    no_weight_decay_cond_type: Literal["qwen3_next"] | None = None
    """Type of no weight decay condition. Choices:
    None (default): param no weight decay if and only if it is 1D; or it is bias;
    or it is embedding and embedding_init_method_std is not None.
    "qwen3_next": In addition to the default rules, apply weight decay to qk layernorm as a special case."""

    wd_incr_steps: int | None = field(init=False, default=None)
    """Number of samples to increment weight decay over. Calculated at runtime."""

    wsd_decay_steps: int | None = field(init=False, default=None)
    """Number of samples to decay WSD weight decay. Calculated at runtime."""


@dataclass(kw_only=True)
class LoggerConfig:
    """Configuration settings for logging, including TensorBoard and WandB."""

    log_interval: int = 100
    """Report loss and timing interval."""

    log_params_norm: bool = False
    """If set, calculate and log parameters norm."""

    log_throughput: bool = False
    """If set, calculate and log throughput per GPU."""

    log_throughput_to_tensorboard: bool = False
    """Enable throughput logging to tensorboard."""

    throughput_window_size: int = 100
    """Number of batches to use for a rolling average of throughput."""

    log_progress: bool = False
    """If set, log progress (in terms of number of processed tokens and number of floating-point operations)
    to progress.txt file in checkpoint directory.
    """

    timing_log_level: Literal[0, 1, 2] = 0
    """Granularity level to measure and report timing.
    0: report only iteration time and make sure timing does not introduce extra overhead.
    1: report timing for operations that are executed very limited times (basically once) during each iteration
        (such as gradient all-reduce)
    2: report timing for operations that migh be executed numerous times during each iteration.
    Note that setting the level to 1 or 2 might cause increase in iteration time.
    """

    timing_log_option: Literal["max", "minmax", "all"] = "minmax"
    """Options for logging timing:
    max: report the max timing across all ranks
    minmax: report min and max timings across all ranks
    all: report timings of all ranks.
    """

    tensorboard_dir: str | None = None
    """Write TensorBoard logs to this directory."""

    tensorboard_log_interval: int = 1
    """Report to tensorboard interval."""

    tensorboard_queue_size: int = 1000
    """Size of the tensorboard queue for pending events and summaries
    before one of the 'add' calls forces a flush to disk.
    """

    log_timers_to_tensorboard: bool = False
    """If set, write timers to tensorboard."""

    log_loss_scale_to_tensorboard: bool = True
    """Disable loss-scale logging to tensorboard."""

    log_validation_ppl_to_tensorboard: bool = False
    """If set, write validation perplexity to tensorboard."""

    log_memory_to_tensorboard: bool = False
    """Enable memory logging to tensorboard."""

    memory_keys: dict[str, str] | None = None
    """Names of memory statistics to log from `torch.cuda.memory_stats()`"""

    log_memory_interval: int | None = None
    """Report memory interval."""

    log_device_memory_used: bool = False
    """Log device memory used (as reported by nvidia-smi)."""

    log_l2_norm_grad_to_tensorboard: bool = False
    """Enable gradients logging to tensorboard."""

    log_num_zeros_in_grad: bool = False
    """If set, calculate and log the number of zeros in gradient."""

    log_max_attention_logit: bool = False
    """Enable max attention logit logging to tensorboard."""

    log_runtime_to_tensorboard: bool = False
    """Enable runtime metrics logging to tensorboard."""

    runtime_time_unit: str = "hours"
    """Time unit to use for time logging. """

    barrier_with_L1_time: bool = field(default=True, metadata={"argparse_meta": {"arg_names": ["--no-barrier-with-level-1-timing"]}})
    """If not disabled, use barrier with level 1 time measurements. Note that this is up to the user to
    make sure calling barrier with their timers will not result in hangs. This can happen if for
    example the user adds a level 1 timer that is not called by all ranks.
    """

    log_world_size_to_tensorboard: bool = False
    """Enable world size logging to tensorboard."""

    wandb_project: str | None = None
    """The wandb project name. Ignore wandb by default."""

    wandb_exp_name: str | None = None
    """The wandb experiment name."""

    wandb_save_dir: str | None = None
    """Path to save the wandb results locally."""

    wandb_entity: str | None = None
    """The wandb entity name. It is useful when there are multiple sub-projects in a project."""

    logging_level: int | None = None
    """Set default logging level"""

    filter_warnings: bool = True
    """Filter out warning messages"""

    modules_to_filter: list[str] | None = None
    """List of modules to filter out from the logs"""

    set_level_for_all_loggers: bool = False
    """Set the logging level for all loggers. If False, only level for NeMo loggers will be set."""

    log_energy: bool = False
    """If set, log energy consumption (in Joules)."""

    save_config_filepath: str | None = None
    """If set, save the task configuration (ConfigContainer) to this file."""


@dataclass(kw_only=True)
class CheckpointConfig:
    """Configuration settings for model checkpointing (saving and loading)."""

    save: str | None = None
    """Output directory to save checkpoints to."""

    save_interval: int | None = field(default=None, metadata={"argparse_meta": {"arg_names": ["--save-interval", "--persistent-save-interval"]}})
    """Number of iterations between persistent checkpoint saves."""

    save_wgrads_interval: int | None = None
    """Number of iterations between wgrad (main_grad) saves."""

    save_dgrads_interval: int | None = None
    """Number of iterations between dgrad saves."""

    save_retain_interval: int | None = None
    """Number of iterations between retained checkpoints
    (other checkpoints except the last checkpoint are automatically deleted).
    """

    most_recent_k: int | None = -1
    """Number of latest checkpoint to be saved."""

    save_optim: bool = True
    """Do not save current optimizer."""

    save_rng: bool = True
    """Do not save current rng state."""

    load: str | None = None
    """Directory containing a model checkpoint."""

    load_optim: bool = True
    """Do not load optimizer when loading checkpoint."""

    load_main_params_from_ckpt: bool = False
    """Load main parameters from checkpoint. When loading a model from a checkpoint without loading
    the optimizer, the model parameters are updated but for fp16 optimizer with main parameters,
    the main parameters need to also be updated.
    """

    load_rng: bool = True
    """Do not load rng state when loading checkpoint."""

    non_persistent_save_interval: int | None = None
    """Number of iterations between non-persistent saves."""

    non_persistent_ckpt_type: Literal["global", "local", "in_memory"] | None = None
    """Type of non-persistent model checkpoints.
    "global" - Saved as a standard checkpoint (e.g., on Lustre) with old checkpoints being removed.
    "local" - [TBD] Each rank saves a portion of the checkpoint locally (e.g., on SSD/ramdisk).
    "in_memory" - [TBD] A special kind of local checkpoint that avoids serialization.
    None - No non-persistent checkpointing (default option)."""

    non_persistent_global_ckpt_dir: str | None = None
    """Directory containing global non-persistent model checkpoints."""

    non_persistent_local_ckpt_dir: str | None = None
    """Directory containing local non-persistent model checkpoints."""

    non_persistent_local_ckpt_algo: Literal["fully_parallel", "atomic"] = "fully_parallel"
    """Algorithm for local non-persistent checkpointing."""

    finetune: bool = False
    """Load model for finetuning. Do not load optimizer or rng state from checkpoint and set iteration to 0.
    Assumed when loading a release checkpoint."""

    pretrained_checkpoint: str | None = None
    """Directory containing a pretrained model checkpoint for finetuning."""

    ckpt_step: int | None = None
    """Checkpoint step to load model from."""

    use_checkpoint_args: bool = False
    """Override model-related command-line arguments with arguments from checkpoint"""

    use_mp_args_from_checkpoint_args: bool = False
    """Copy model parallelism command-line arguments from checkpoint"""

    use_tokenizer_model_from_checkpoint_args: bool = True
    """If set, do not use tokenizer model path from checkpoint"""

    exit_on_missing_checkpoint: bool = False
    """If 'load' is set, but checkpoint is not found (e.g., path typo), then exit instead of random initialization."""

    ckpt_format: Literal["torch", "torch_dist", "torch_dcp", "fsdp_dtensor"] = "torch_dist"
    """ Checkpoint format to use. torch is the format used by torch.save/load.
    torch_dist is a megatron built-in distributed checkpointing format.
    torch_dcp is the torch.distributed.checkpoint format.
    fsdp_dtensor is a torch DCP native, Megatron FSDP training-specific checkpoint format.
    """

    auto_detect_ckpt_format: bool = False
    """Determine if the checkpoint format is in legacy or distributed format. If False,
    expects distributed checkpoint iff args.ckpt_format != "torch". Might slow down 
    loading a bit (double rank0 ckpt load).
    """

    ckpt_convert_format: Literal["torch", "torch_dist"] | None = None
    """Checkpoint format for conversion."""

    ckpt_convert_save: str | None = None
    """Save directory for converted checkpoint."""

    ckpt_convert_update_legacy_dist_opt_format: bool = False
    """When loading a checkpoint, update the legacy format for the distributed optimizer,
    which previously used a merged param/grad buffer and a different bucket mapping.
    The legacy format was deprecated on Feb 13, 2024.
    """

    ckpt_fully_parallel_save: bool = True
    """Disable applying full save parallelization across DP for distributed checkpoints.
    Depending on ckpt format might decrease the number of files in the checkpoint.
    Makes DistributedOptimizer checkpoint non-reshardable."""

    async_save: bool = False
    """Apply async checkpointing save. Currently works only with `torch_dist` distributed checkpoint format."""

    use_persistent_ckpt_worker: bool = False
    """Use a persistent background worker for async checkpoint saves. When enabled, creates a dedicated
    worker thread/process for handling async saves. When disabled, uses temporal workers that are
    created and destroyed for each save operation."""

    ckpt_fully_parallel_load: bool = False
    """Apply full load parallelization across DP for distributed checkpoints."""

    ckpt_fully_parallel_load_exchange_algo: Literal["broadcast", "gather_rounds", "gather_object"] = "broadcast"
    """Algorithm for fully parallel load of distributed checkpoints.
    "broadcast"(default): Broadcast the checkpoint from rank 0 to all other ranks.
    "gather_rounds": Gather the checkpoint from all ranks in rounds.
    "gather_object": Gather the checkpoint from all ranks in a single operation.
    """

    ckpt_fully_parallel_save_process_group: Literal["dp", "ep_dp"] = "dp"
    """Process group for fully parallel save of distributed checkpoints.
    "dp"(default): Data parallel process group.
    "ep_dp": Expert data parallel process group.
    """

    ckpt_fully_parallel_load_process_group: Literal["dp", "ep_dp"] = "dp"
    """Process group for fully parallel load of distributed checkpoints.
    "dp"(default): Data parallel process group.
    "ep_dp": Expert data parallel process group.
    """

    ckpt_assume_constant_structure: bool = False
    """Assume the checkpoint structure is constant across saves to enable optimizations."""

    strict_fsdp_dtensor_load: bool = True
    """Whether to enforce strict loading for FSDP DTensor checkpoints. When False, allows partial loading."""

    dist_ckpt_strictness: Literal[
        "assume_ok_unexpected",
        "log_unexpected",
        "log_all",
        "raise_unexpected",
        "raise_all",
        "return_unexpected",
        "return_all",
        "ignore_all",
    ] = "assume_ok_unexpected"
    """Determine handling of key mismatch during checkpoint load. Check StrictHandling docs for flags meaning.
    NOTE: This flag controls only distributed checkpoint load from storage, not loading state dict into the model."""

    dist_ckpt_save_pre_mcore_014: bool = False
    """Revert checkpointing simplifications introduced in Megatron-Core v0.14.
    This option affects only checkpoint saving format and will be removed soon
    (checkpoint load format is determined based on checkpoint metadata)."""

    dist_ckpt_optim_fully_reshardable: bool = False
    """Make optimizer distributed checkpoint fully reshardable (TP/PP/EP/DP) as opposed to plain DP reshardability."""

    distrib_optim_fully_reshardable_mem_efficient: bool = False
    """During distributed optimizer checkpoint save and load tries to use as little memory as possible
    by using Gloo (instead of NCCL) and only one rank for saving. Turn on only if experiencing host or device memory
    issues. Has affect only with `dist_ckpt_optim_fully_reshardable` flag."""

    save_tokenizer_assets: bool = True
    """Save tokenizer files to checkpoint directory. When enabled, saves all tokenizer artifacts
    (vocab files, special tokens, tokenizer config) to make checkpoints self-contained and portable.
    Set to False for performance-sensitive scenarios where tokenizer files are not needed."""

    replication: bool = False
    """If set, replication of local checkpoints is enabled. Needs to be enabled on all ranks."""

    replication_jump: int | None = None
    """Specifies `J`, the spacing between ranks storing replicas of a given rank's data. Replicas
    for rank `n` may be on ranks `n+J`, `n+2J`, ..., or `n-J`, `n-2J`, etc. This flag has an
    effect only if --replication is used. and must be consistent across all ranks."""

    replication_factor: int = 2
    """Number of machines storing the replica of a given rank's data."""
