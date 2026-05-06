# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
import time
import logging
from functools import partial
from typing import NamedTuple, Callable, Any
from megatron.core._rank_utils import safe_get_rank
from megatron.core.config import set_experimental_flag
from megatron.core.jit import disable_jit_fuser
from megatron.core.utils import get_model_config
from megatron.training.checkpointing import load_checkpoint
from megatron.training.config import PretrainConfigContainer, SchedulerConfig
from megatron.training.initialize import initialize_megatron, set_jit_fusion_options
from megatron.training.utils import print_rank_0
from megatron.training.utils.log_utils import append_to_progress_log, barrier_and_log
from megatron.training.utils.train_utils import start_memory_history_recording
import torch
from megatron.core.transformer import MegatronModule, TransformerConfig
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig, finalize_model_grads
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel as megatron_FSDP
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.rerun_state_machine import RerunDataIterator
from megatron.core.optimizer import (
    MegatronOptimizer,
    OptimizerConfig,
    get_megatron_optimizer,
    get_mup_config_overrides,
)
from megatron.core.optimizer.muon import get_megatron_muon_optimizer

from megatron.training.state import GlobalState

try:
    from megatron.core.distributed import TorchFullyShardedDataParallel as torch_FSDP

    HAVE_FSDP2 = True
except ImportError:
    HAVE_FSDP2 = False


class SetupOutput(NamedTuple):
    """Represents the output of the main setup function.

    Contains all the initialized components necessary for training or evaluation.

    Attributes:
        state: The global state object holding configuration and runtime information.
        model: The initialized Megatron model.
        optimizer: The initialized optimizer.
        scheduler: The initialized learning rate scheduler.
        train_data_iterator: The data iterator for the training dataset, if applicable.
        valid_data_iterator: The data iterator for the validation dataset, if applicable.
        test_data_iterator: The data iterator for the testing dataset, if applicable.
        checkpoint_manager: The checkpoint manager for save/load operations.
        pg_collection: The process group collection initialized for this run.
    """

    state: GlobalState
    model: MegatronModule
    optimizer: MegatronOptimizer
    scheduler: OptimizerParamScheduler
    train_data_iterator: RerunDataIterator | list[RerunDataIterator] | None
    valid_data_iterator: RerunDataIterator | list[RerunDataIterator] | None
    test_data_iterator: RerunDataIterator | list[RerunDataIterator] | None
    # checkpoint_manager: CheckpointManager # TODO (@maanug): migrate
    pg_collection: ProcessGroupCollection


def setup(
    state: GlobalState,
    train_valid_test_datasets_provider: Callable[..., tuple[Any | None, Any | None, Any | None]],
    get_embedding_ranks: Callable[[list[int], int | None], list[int]] | None = None,
    get_position_embedding_ranks: Callable[[list[int], int | None], list[int]] | None = None,
    restart_store: torch.distributed.Store | None = None,
    checkpointing_context: dict[str, Any] = {},
    model_provider_func=None, # TODO (@maanug): temporary until all scripts can use ModelConfig+Builder
    # callback_manager: CallbackManager | None = None,  # TODO (@maanug): migrate
) -> SetupOutput:
    """Initialize the training/evaluation environment using an existing GlobalState.

    Performs all runtime setup using the provided `state` and its attached config (`state.cfg`).
    This includes:
      - enabling Megatron-Core experimental features
      - initializing async checkpoint workers (if enabled)
      - logging setup
      - torch.distributed and model-parallel initialization (via initialize_megatron)
      - tokenizer/model/optimizer/scheduler construction
      - optional checkpoint load
      - dataloader setup

    Args:
        state: The GlobalState instance to populate and use throughout setup.
        train_valid_test_datasets_provider: Callable returning the train/valid/test datasets or iterators.
        get_embedding_ranks: Optional function to determine embedding layer ranks for model-parallel init.
        get_position_embedding_ranks: Optional function to determine positional embedding ranks.
        restart_store: Optional torch.distributed Store used when in-process restart is enabled.
        callback_manager: Optional CallbackManager whose on_data_init_start hook is fired
            after the model/optimizer/checkpoint are ready but before any dataset files are
            opened. Use this for JIT warmup with mock data and MLPerf init_stop/run_start
            logging to ensure no real dataset I/O occurs before run_start is recorded.

    Returns:
        SetupOutput containing the populated state, model, optimizer, scheduler, dataloaders, and ckpt context.
    """
    cfg = state.cfg
    maybe_log_and_save_config(cfg)

    # Conditionally enable experimental features for Megatron Core
    set_experimental_flag(cfg.dist.enable_megatron_core_experimental)

    # Disable the JIT fuser if requested
    if cfg.dist.disable_jit_fuser:
        print_rank_0("Disabling JIT fuser.")
        disable_jit_fuser()

    # Initialize async checkpoint worker if enabled (idempotent if already initialized)
    state.initialize_async_checkpoint_worker()

    # TODO (@maanug): merge bridge and mlm initialize_megatron() impls
    initialize_megatron(
        get_embedding_ranks=get_embedding_ranks,
        get_position_embedding_ranks=get_position_embedding_ranks,
        store=restart_store,
    )
    # TODO (@maanug): temporary until initialize.py is refactored to build pgcollection as bridge does
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    # Set CPU affinity for optimal host-device transfers when fine-grained activation offloading is enabled
    if cfg.model.fine_grained_activation_offloading:
        from megatron.core.pipeline_parallel.utils import set_ideal_affinity_for_current_gpu

        set_ideal_affinity_for_current_gpu()

    timers = state.timers

    if cfg.logger.log_progress:
        append_to_progress_log(cfg.checkpoint.save, "Starting job")

    # TODO (@maanug):
    # if cfg.ft and cfg.ft.enable_ft_package:
    #     fault_tolerance.setup(cfg, state)
    #     fault_tolerance.maybe_setup_simulated_fault(cfg.ft)

    # Set pytorch JIT layer fusion options and warmup JIT functions.
    # TODO (@maanug): merge bridge and mlm impls
    set_jit_fusion_options()

    start_time_tensor = torch.tensor([state.start_time], dtype=torch.double, device="cuda")
    torch.distributed.all_reduce(start_time_tensor, op=torch.distributed.ReduceOp.MIN)
    state.start_time = start_time_tensor.item()

    print_rank_0("time to initialize megatron (seconds): {:.3f}".format(time.time() - state.start_time))
    barrier_and_log("after megatron is initialized")

    # Tokenizer
    timers("tokenizer-setup", log_level=0).start(barrier=True)
    tokenizer = state.tokenizer
    # Handle model vocab_size configuration with proper validation
    cfg.model.vocab_size, cfg.model.should_pad_vocab = _validate_and_set_vocab_size(
        model_vocab_size=cfg.model.vocab_size,
        tokenizer_vocab_size=tokenizer.vocab_size,
    )

    # cfg.dataset.tokenizer = tokenizer TODO (@maanug)
    timers("tokenizer-setup").stop()
    barrier_and_log("after tokenizer is built")

    # Model, optimizer, and learning rate.
    timers("model-and-optimizer-setup", log_level=0).start(barrier=True)

    # TODO (@maanug): PEFT?
    # TODO (@maanug): check and load modelopt state

    # Enable CUDA allocator history tracing before any model tensors are allocated,
    # so snapshots dumped later in training contain a full timeline + stack context.
    start_memory_history_recording(cfg.profiling)

    model = _build_distributed_model(cfg, pg_collection, model_provider_func)

    cfg.model.timers = timers
    cfg.optimizer.timers = timers
    optimizer, scheduler = setup_optimizer(
        optimizer_config=cfg.optimizer,
        scheduler_config=cfg.scheduler,
        model=model,
        use_gloo_process_groups=cfg.dist.use_gloo_process_groups,
        # TODO (@maanug): figure out pgcollection in this PR
        # Only pass pg_collection when use_decentralized_pg is True.
        # When False, mcore's optimizer will use parallel_state directly which supports Gloo.
        # pg_collection=pg_collection if cfg.dist.use_decentralized_pg else None,
    )
    timers("model-and-optimizer-setup").stop()
    barrier_and_log("after model, optimizer, and learning rate scheduler are built")

    # TODO (@maanug): check for local checkpoints
    # TODO (@maanug): load PEFT base checkpoint?

    should_load_checkpoint = (
        (cfg.checkpoint.load is not None and checkpoint_exists(cfg.checkpoint.load))
        or (
            cfg.checkpoint.pretrained_checkpoint is not None
            and checkpoint_exists(cfg.checkpoint.pretrained_checkpoint)
        )
    )

    if should_load_checkpoint:
        from megatron.training.global_vars import get_args

        timers("load-checkpoint", log_level=0).start(barrier=True)
        args = get_args()
        args.iteration, args.num_floating_point_operations_so_far = load_checkpoint(
            model,
            optimizer,
            scheduler,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=HAVE_FSDP2
            and cfg.dist.use_torch_fsdp2
            and cfg.checkpoint.ckpt_format == "torch_dist",
        )
        state.train_state.step = args.iteration
        state.train_state.floating_point_operations_so_far = args.num_floating_point_operations_so_far
        timers("load-checkpoint").stop(barrier=True)
        timers.log(["load-checkpoint"])

    _update_model_config_funcs(
        model,
        cfg.model.transformer,
        cfg.ddp,
        optimizer,
        align_grad_reduce=cfg.dist.align_grad_reduce,
        pg_collection=pg_collection,
    )

    # TODO (@maanug): data iterator setup

    print_rank_0("done with setup ...")
    timers.log(["model-and-optimizer-setup", "train/valid/test-data-iterators-setup"], barrier=True)

    return SetupOutput(
        state,
        model,
        optimizer,
        scheduler,
        # train_data_iterator,
        None,
        # valid_data_iterator,
        None,
        # test_data_iterator,
        None,
        # checkpoint_manager,
        pg_collection,
    )


def _build_distributed_model(cfg: PretrainConfigContainer, pg_collection: ProcessGroupCollection, model_provider_func=None) -> list[MegatronModule]:
    """Build distributed model from ModelConfig."""
    from megatron.training.models import ModelConfig

    model_config = cfg.model
    if isinstance(model_config, ModelConfig) and hasattr(model_config, "get_builder_cls"):
        builder_cls = model_config.get_builder_cls()
        builder = builder_cls(model_config)
        return builder.build_distributed_models(
            pg_collection=pg_collection,
            ddp_config=cfg.ddp,
            overlap_param_gather_with_optimizer_step=cfg.optimizer.overlap_param_gather_with_optimizer_step,
            use_megatron_fsdp=cfg.dist.use_megatron_fsdp,
            use_torch_fsdp2=cfg.dist.use_torch_fsdp2,
            data_parallel_random_init=cfg.rng.data_parallel_random_init,
        )
    else:
        from megatron.training.training import get_model
        from megatron.training.global_vars import get_args

        args = get_args()
        has_normal_optimizer = not args.skip_train
        has_rl_optimizer = args.perform_rl_step and not args.no_load_optim
        skip_optimizer = not (has_normal_optimizer or has_rl_optimizer)
        wrap_with_ddp = not skip_optimizer
        return get_model(model_provider_func, wrap_with_ddp=wrap_with_ddp)


def _update_model_config_funcs(
    model: MegatronModule,
    model_config: TransformerConfig,
    ddp_config: DistributedDataParallelConfig,
    optimizer: MegatronOptimizer | None,
    *,
    align_grad_reduce: bool = True,
    pg_collection: ProcessGroupCollection | None = None,
) -> None:
    """Update model config sync funcs based on initialized model."""
    if isinstance(model[0], (DistributedDataParallel, megatron_FSDP)) and ddp_config.overlap_grad_reduce:
        assert model_config.no_sync_func is None, (
            "When overlap_grad_reduce is True, config.no_sync_func must be None; "
            "a custom no_sync_func is not supported when overlapping grad-reduce"
        )
        model_config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
        if len(model) == 1:
            model_config.no_sync_func = model_config.no_sync_func[0]
        if align_grad_reduce:
            model_config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in model]
            if len(model) == 1:
                model_config.grad_sync_func = model_config.grad_sync_func[0]
    if ddp_config.overlap_param_gather and ddp_config.align_param_gather:
        model_config.param_sync_func = [model_chunk.start_param_sync for model_chunk in model]
        if len(model) == 1:
            model_config.param_sync_func = model_config.param_sync_func[0]
    if optimizer is not None:
        model_config.finalize_model_grads_func = partial(finalize_model_grads, pg_collection=pg_collection)
        model_config.grad_scale_func = optimizer.scale_loss


def _validate_and_set_vocab_size(model_vocab_size: int | None, tokenizer_vocab_size: int) -> tuple[int, bool]:
    """Validate and determine the correct vocab size for the model.

    Args:
        model_vocab_size: Vocab size set in model config (can be None)
        tokenizer_vocab_size: Unpadded tokenizer vocab size

    Returns:
        tuple[int, bool]: The validated unpadded vocab size and padding flag
            - vocab_size: The validated unpadded vocab size to use for the model
            - should_pad_vocab: True if vocab should be padded, False otherwise

    Raises:
        ValueError: If model vocab size is invalid
    """
    if model_vocab_size is None:
        # If model vocab size is not set, use the tokenizer's vocab size
        # Enable padding since this came from tokenizer
        return tokenizer_vocab_size, True
    elif model_vocab_size < tokenizer_vocab_size:
        # Vocab size smaller than tokenizer
        raise ValueError(
            f"Model vocab_size ({model_vocab_size}) cannot be smaller than tokenizer's vocab_size "
            f"({tokenizer_vocab_size})."
        )
    else:
        # Model vocab size is explicitly set and is >= tokenizer vocab size
        # Disable padding since this was explicitly set
        if model_vocab_size > tokenizer_vocab_size:
            logging.info(
                f"Using preset vocab_size: {model_vocab_size} over the tokenizer vocab_size: {tokenizer_vocab_size}, dummy tokens:"
                f" {model_vocab_size - tokenizer_vocab_size}."
            )
        return model_vocab_size, False


def setup_optimizer(
    optimizer_config: OptimizerConfig,
    scheduler_config: SchedulerConfig,
    model: MegatronModule | list[MegatronModule],
    use_gloo_process_groups: bool = False,
    pg_collection: ProcessGroupCollection | None = None,
) -> tuple[MegatronOptimizer, OptimizerParamScheduler]:
    """Set up the optimizer and scheduler.

    Args:
        optimizer_config: Configuration for the optimizer
        scheduler_config: Configuration for the scheduler
        model: The model to optimize
        use_gloo_process_groups: Whether to use Gloo process groups
        pg_collection: Optional process group collection for distributed training

    Returns:
        tuple containing the optimizer and scheduler
    """
    # TODO (@maanug): migrate bridge optimizer config overrides system
    config_overrides = None

    # Apply μP optimizer scaling if enabled on the model config.
    # Guard on the callable itself (None when mcore main lacks the symbol) so
    # unit tests can patch the module attribute without hitting AttributeError.
    model_chunks = model if isinstance(model, list) else [model]
    model_config = get_model_config(model_chunks[0])
    if get_mup_config_overrides is not None and getattr(model_config, "use_mup", False):
        mup_overrides = get_mup_config_overrides(
            config=optimizer_config,
            mup_width_mult=model_config.mup_width_mult,
            optimizer_type=optimizer_config.optimizer,
        )
        if mup_overrides:
            config_overrides = {**(config_overrides or {}), **mup_overrides}
            logging.info(
                f"μP enabled (width_mult={model_config.mup_width_mult:.4g}): "
                f"applied {len(mup_overrides)} optimizer param-group override(s)."
            )

    if "muon" not in optimizer_config.optimizer and "soap" not in optimizer_config.optimizer:
        optimizer = get_megatron_optimizer(
            config=optimizer_config,
            model_chunks=model,
            config_overrides=config_overrides,
            use_gloo_process_groups=use_gloo_process_groups,
            pg_collection=pg_collection,
        )
    else:
        optimizer = get_megatron_muon_optimizer(
            config=optimizer_config,
            model_chunks=model,
            config_overrides=config_overrides,
            use_gloo_process_groups=use_gloo_process_groups,
            layer_wise_distributed_optimizer="dist" in optimizer_config.optimizer,
            pg_collection=pg_collection,
        )

    scheduler = _get_scheduler(optimizer_config, scheduler_config, optimizer)

    return optimizer, scheduler


def _get_scheduler(
    optimizer_config: OptimizerConfig, scheduler_config: SchedulerConfig, optimizer: MegatronOptimizer
) -> OptimizerParamScheduler:
    """Get the optimizer parameter scheduler.

    Args:
        optimizer_config: Configuration for the optimizer
        scheduler_config: Configuration for the scheduler
        optimizer: The optimizer to schedule

    Returns:
        The optimizer parameter scheduler
    """
    scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=scheduler_config.lr_warmup_init,
        max_lr=optimizer_config.lr,
        min_lr=optimizer_config.min_lr,
        lr_warmup_steps=scheduler_config.lr_warmup_steps,
        lr_decay_steps=scheduler_config.lr_decay_steps,
        lr_decay_style=scheduler_config.lr_decay_style,
        start_wd=scheduler_config.start_weight_decay,
        end_wd=scheduler_config.end_weight_decay,
        wd_incr_steps=scheduler_config.wd_incr_steps,
        wd_incr_style=scheduler_config.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=scheduler_config.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=scheduler_config.override_opt_param_scheduler,
        wsd_decay_steps=scheduler_config.wsd_decay_steps,
        lr_wsd_decay_style=scheduler_config.lr_wsd_decay_style,
    )

    return scheduler


def maybe_log_and_save_config(cfg: PretrainConfigContainer) -> None:
    """Save configuration to disk and log non-default values on rank 0."""

    if safe_get_rank() != 0:
        return

    if cfg.logger.save_config_filepath is not None:
        try:
            cfg.to_yaml(cfg.logger.save_config_filepath)
        except Exception as e:
            print_rank_0(f"Error saving config to file {cfg.logger.save_config_filepath}: {e}")
