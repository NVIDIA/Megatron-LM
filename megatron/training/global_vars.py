# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Megatron global variables."""

import os
import re
import signal
import sys
from argparse import Namespace
from datetime import timedelta

import torch

from megatron.core import Timers
from megatron.core._rank_utils import safe_get_world_size
from megatron.core.config import set_experimental_flag
from megatron.core.energy_monitor import EnergyMonitor
from megatron.core.jit import disable_jit_fuser
from megatron.core.num_microbatches_calculator import (
    init_num_microbatches_calculator,
    unset_num_microbatches_calculator,
)
from megatron.training.dist_signal_handler import DistributedSignalHandler
from megatron.training.state import TrainState

_GLOBAL_ARGS = None
_GLOBAL_RUN_CONFIG = None
_GLOBAL_TRAIN_STATE = None
_GLOBAL_TOKENIZER = None
_GLOBAL_TENSORBOARD_WRITER = None
_GLOBAL_WANDB_WRITER = None
_GLOBAL_ONE_LOGGER = None
_GLOBAL_ADLR_AUTORESUME = None
_GLOBAL_TIMERS = None
_GLOBAL_ENERGY_MONITOR = None
_GLOBAL_SIGNAL_HANDLER = None
_GLOBAL_TELEMETRY_HANDLE = None

def get_args():
    """Return arguments."""
    _ensure_var_is_initialized(_GLOBAL_ARGS, 'args')
    return _GLOBAL_ARGS


def get_run_config():
    """Return the full pretrain config container."""
    _ensure_var_is_initialized(_GLOBAL_RUN_CONFIG, 'run config')
    return _GLOBAL_RUN_CONFIG


def is_run_config_initialized() -> bool:
    """Whether argument adapters have a run-owned config available yet."""
    return _GLOBAL_RUN_CONFIG is not None


def get_train_state():
    """Return the mutable state for the current training run.

    Returns ``None`` if ``set_global_variables`` was never called (e.g. tests that
    build args via ``parse_args`` and invoke ``save_checkpoint`` directly without
    going through full Megatron initialization). Checkpointing skips saving the
    train state in that case; training code always has one.
    """
    return _GLOBAL_TRAIN_STATE


def get_tokenizer():
    """Return tokenizer."""
    _ensure_var_is_initialized(_GLOBAL_TOKENIZER, 'tokenizer')
    return _GLOBAL_TOKENIZER


def get_tensorboard_writer():
    """Return tensorboard writer. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_TENSORBOARD_WRITER


def get_wandb_writer():
    """Return tensorboard writer. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_WANDB_WRITER


def get_one_logger():
    """Return one logger. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_ONE_LOGGER

def get_adlr_autoresume():
    """ADLR autoresume object. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_ADLR_AUTORESUME


def get_timers():
    """Return timers."""
    _ensure_var_is_initialized(_GLOBAL_TIMERS, 'timers')
    return _GLOBAL_TIMERS

def get_energy_monitor():
    """Return energy monitor."""
    _ensure_var_is_initialized(_GLOBAL_ENERGY_MONITOR, 'energy monitor')
    return _GLOBAL_ENERGY_MONITOR

def get_signal_handler():
    _ensure_var_is_initialized(_GLOBAL_SIGNAL_HANDLER, 'signal handler')
    return _GLOBAL_SIGNAL_HANDLER


def get_telemetry():
    """Return the telemetry handle. It can be None so no need to check if initialized."""
    return _GLOBAL_TELEMETRY_HANDLE


def _shutdown_telemetry():
    """Shut down telemetry and release its published child Resource carrier."""
    global _GLOBAL_TELEMETRY_HANDLE
    telemetry_handle = _GLOBAL_TELEMETRY_HANDLE
    _GLOBAL_TELEMETRY_HANDLE = None
    if telemetry_handle is not None:
        try:
            telemetry_handle.shutdown()
        except Exception:
            pass


def _set_signal_handler(exit_signal):

    global _GLOBAL_SIGNAL_HANDLER
    _ensure_var_is_not_initialized(_GLOBAL_SIGNAL_HANDLER, 'signal handler')
    _GLOBAL_SIGNAL_HANDLER = DistributedSignalHandler(exit_signal).__enter__()


def _graceful_shutdown(signum, frame):
    """
    Signal handler for user-initiated termination (SIGINT / SIGTERM).

    This handler attempts a best-effort graceful shutdown:
      - Logs a single termination message from rank 0
      - Synchronizes all ranks (barrier)
      - Flushes and shuts down telemetry
      - Destroys the distributed process group
      - Exits the process cleanly
    """
    from megatron.training.utils import print_rank_0
    print_rank_0("\nTermination requested. Performing orderly shutdown.")

    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # synchronize all ranks before exiting
            try:
                # avoid deadlock if ranks don't all reach here
                torch.distributed.barrier()
            except Exception:
                pass

            torch.distributed.destroy_process_group()
    except Exception:
        pass

    # Without this, any spans/metrics still sitting in the BatchSpanProcessor's
    # queue (which only flushes on a timer or on shutdown(), never automatically
    # on process exit) are silently dropped -- including the just-closed
    # top-level job span on this exit path. sys.exit() below unwinds the call
    # stack via SystemExit, so any span opened with `with managed_span(...)`/
    # `@trace_fn` still gets its __exit__ run and end_time set correctly; this
    # call is what makes sure that data actually reaches the exporter.
    try:
        _otel_handle = get_telemetry()
        if _otel_handle is not None:
            _otel_handle.shutdown()
    except Exception:
        pass

    sys.exit(0)


def set_global_variables(args, cfg_container, build_tokenizer=True):
    """Register caller-provided args/config and construct their runtime services."""

    from megatron.training.config import PretrainConfigContainer

    assert args is not None
    assert cfg_container is not None

    _ensure_var_is_not_initialized(_GLOBAL_ARGS, 'args')
    training = isinstance(cfg_container, PretrainConfigContainer)
    cfg_container.validate()
    set_run_config(cfg_container)
    set_args(args)
    # Every container is finalized here: the checkpoint derivations apply to inference containers
    # too, and the checkpoint converters bootstrap through this function without reaching
    # initialize_megatron. Finalizing constructs no runtime service.
    cfg_container.finalize()
    initialize_runtime_services(
        args,
        build_tokenizer=build_tokenizer,
        training=training,
    )
    if not training:
        # Legacy checkpoint converters preserve progress without training services.
        _set_train_state()


def initialize_runtime_services(
    args: Namespace, *, build_tokenizer: bool = True, training: bool = False
) -> None:
    """Build services from finalized config; normally called by initialize_megatron."""
    from megatron.training.argument_utils import resolve_tokenizer_vocab_size

    cfg = get_run_config()
    if build_tokenizer:
        _build_tokenizer(cfg, args)
    # Tokenizer-free bootstrap may discover the model family and rebuild the
    # tokenizer later. Bind known sizes now; otherwise defer until that rebuild.
    if build_tokenizer or cfg.tokenizer.padded_vocab_size is not None:
        resolve_tokenizer_vocab_size(cfg)
    _bind_dataset_tokenizer(cfg)
    _set_wandb_writer(args)
    _set_telemetry(args, include_training=training)

    if args.enable_experimental:
        set_experimental_flag(True)

    if cfg.dist.disable_jit_fuser:
        disable_jit_fuser()

    if training:
        initialize_training_runtime_services(args)


def initialize_training_runtime_services(args: Namespace) -> None:
    """Initialize training-only services after the shared runtime services."""
    cfg = get_run_config()
    if cfg.train.step_batch_size_schedule is not None:
        # Imported here, as elsewhere in this module: megatron.training.utils imports back
        # into megatron.training, which imports this module.
        from megatron.training.utils import print_rank_0
        print_rank_0(f'> using step batch size schedule: {cfg.train.step_batch_size_schedule}')

    init_num_microbatches_calculator(
        rank=args.rank,
        global_batch_size=cfg.train.global_batch_size,
        micro_batch_size=cfg.train.micro_batch_size,
        # Independent sequences, including when CP and weight sharding share ranks.
        data_parallel_size=cfg.get_batch_parallel_size(),
        decrease_batch_size_if_needed=cfg.train.decrease_batch_size_if_needed,
        step_batch_size_schedule=cfg.train.step_batch_size_schedule,
        seq_length=args.seq_length,
    )
    _set_tensorboard_writer(args)
    _set_one_logger(args)
    _set_adlr_autoresume()
    _set_timers(args)
    _set_energy_monitor(args)
    _set_train_state()

    if cfg.train.exit_signal_handler:
        _set_signal_handler(cfg.train.exit_signal)

    if cfg.train.exit_signal_handler_for_training:
        signal.signal(signal.SIGINT, _graceful_shutdown)
        signal.signal(signal.SIGTERM, _graceful_shutdown)


def unset_global_variables():
    """Unset global vars.

    Useful for multiple runs. See `tests/unit_tests/ckpt_converter/test_ckpt_converter.py` for an example.
    """

    global _GLOBAL_ARGS
    global _GLOBAL_RUN_CONFIG
    global _GLOBAL_TRAIN_STATE
    global _GLOBAL_NUM_MICROBATCHES_CALCULATOR
    global _GLOBAL_TOKENIZER
    global _GLOBAL_TENSORBOARD_WRITER
    global _GLOBAL_WANDB_WRITER
    global _GLOBAL_ONE_LOGGER
    global _GLOBAL_ADLR_AUTORESUME
    global _GLOBAL_TIMERS
    global _GLOBAL_ENERGY_MONITOR
    global _GLOBAL_SIGNAL_HANDLER
    global _GLOBAL_TELEMETRY_HANDLE

    _GLOBAL_ARGS = None
    _GLOBAL_RUN_CONFIG = None
    _GLOBAL_TRAIN_STATE = None
    _GLOBAL_NUM_MICROBATCHES_CALCULATOR = None
    _GLOBAL_TOKENIZER = None
    _GLOBAL_TENSORBOARD_WRITER = None
    _GLOBAL_WANDB_WRITER = None
    _GLOBAL_ONE_LOGGER = None
    _GLOBAL_ADLR_AUTORESUME = None
    _GLOBAL_TIMERS = None
    _GLOBAL_ENERGY_MONITOR = None
    _GLOBAL_SIGNAL_HANDLER = None
    _shutdown_telemetry()

    unset_num_microbatches_calculator()


def set_args(args):
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args


def set_run_config(cfg_container):
    global _GLOBAL_RUN_CONFIG
    _ensure_var_is_not_initialized(_GLOBAL_RUN_CONFIG, 'run config')
    _GLOBAL_RUN_CONFIG = cfg_container


def _set_train_state():
    """Create the train state for the current training run."""
    global _GLOBAL_TRAIN_STATE
    _ensure_var_is_not_initialized(_GLOBAL_TRAIN_STATE, 'train state')
    _GLOBAL_TRAIN_STATE = TrainState()


def _build_tokenizer(cfg, args):
    """Initialize tokenizer."""
    from megatron.training.vocab_utils import build_tokenizer_and_pad_vocab_size

    global _GLOBAL_TOKENIZER
    _ensure_var_is_not_initialized(_GLOBAL_TOKENIZER, 'tokenizer')

    if cfg.model is not None:
        make_vocab_size_divisible_by = cfg.model.make_vocab_size_divisible_by
        tp_size = cfg.model.transformer.tensor_model_parallel_size
    else:
        make_vocab_size_divisible_by = args.make_vocab_size_divisible_by
        tp_size = args.tensor_model_parallel_size

    _GLOBAL_TOKENIZER = build_tokenizer_and_pad_vocab_size(cfg.tokenizer, make_vocab_size_divisible_by, tp_size)
    # Resolve the declared model field once, before any args-to-config conversion.
    # The tokenizer includes added tokens; padded_vocab_size also includes TP padding.
    if (
        getattr(args, 'moe_num_hash_layers', 0) > 0
        and getattr(args, 'hash_moe_vocab_size', None) is None
    ):
        args.hash_moe_vocab_size = _GLOBAL_TOKENIZER.vocab_size
        if getattr(args, 'yaml_cfg', None) is not None:
            args.language_model.hash_moe_vocab_size = args.hash_moe_vocab_size
    return _GLOBAL_TOKENIZER


def rebuild_tokenizer(cfg, args):
    global _GLOBAL_TOKENIZER
    _GLOBAL_TOKENIZER = None
    return _build_tokenizer(cfg, args)


def _bind_dataset_tokenizer(cfg) -> None:
    """Bind the run's tokenizer to ``cfg.data.dataset`` (Megatron Core's ``validate()`` runs then).

    Only GPT entry points carry a dataset there, and nothing happens without a tokenizer
    (``build_tokenizer=False``: the provider then reports the unbound config). The container must
    be finalized first, as ``initialize_megatron`` and ``set_global_variables`` do (the fields other
    sections own are written then). ``DataConfig.bind_tokenizer`` decides the rest: a config bound
    earlier to this tokenizer is left as is, one bound to another tokenizer is an error.
    """
    data = getattr(cfg, "data", None)
    if getattr(data, "dataset", None) is None or _GLOBAL_TOKENIZER is None:
        return
    if not getattr(cfg, "_finalized", False):
        raise RuntimeError(
            "cfg.data.dataset cannot bind the tokenizer before the container is finalized: call "
            "cfg.finalize() before initialize_runtime_services(), as initialize_megatron() and "
            "set_global_variables() do."
        )
    data.bind_tokenizer(_GLOBAL_TOKENIZER)


def _set_tensorboard_writer(args):
    """Set tensorboard writer."""
    cfg = get_run_config()
    global _GLOBAL_TENSORBOARD_WRITER
    _ensure_var_is_not_initialized(_GLOBAL_TENSORBOARD_WRITER,
                                   'tensorboard writer')

    if cfg.logger.tensorboard_dir and args.rank == (safe_get_world_size() - 1):
        try:
            from torch.utils.tensorboard import SummaryWriter
            print('> setting tensorboard ...')
            _GLOBAL_TENSORBOARD_WRITER = SummaryWriter(
                log_dir=cfg.logger.tensorboard_dir,
                max_queue=cfg.logger.tensorboard_queue_size)
        except ModuleNotFoundError:
            print('WARNING: TensorBoard writing requested but is not '
                  'available (are you using PyTorch 1.1.0 or later?), '
                  'no TensorBoard logs will be written.', flush=True)


def _wandb_run_id(exp_name: str) -> str:
    """A wandb run id derived from the experiment name.

    The id is part of the run URL, so wandb rejects the characters that would be
    ambiguous there. Experiment names are often paths, so substitute rather than fail;
    the unmodified name is still what gets displayed.
    """
    return re.sub(r"[/\\#?%:]", "-", exp_name)


def _set_wandb_writer(args):
    cfg = get_run_config()
    global _GLOBAL_WANDB_WRITER
    _ensure_var_is_not_initialized(_GLOBAL_WANDB_WRITER,
                                   'wandb writer')
    if cfg.logger.wandb_project and args.rank == (safe_get_world_size() - 1):
        if cfg.logger.wandb_exp_name == '':
            raise ValueError("Please specify the wandb experiment name!")

        import wandb
        if cfg.logger.wandb_save_dir:
            save_dir = cfg.logger.wandb_save_dir
        else:
            # Defaults to the save dir.
            save_dir = os.path.join(cfg.checkpoint.save, 'wandb')
        # Private attributes are runtime state rather than configuration and can be large; keep
        # them out of the wandb config. Copy so the kitchen contents added below stay off args.
        wandb_config = {k: v for k, v in vars(args).items() if not k.startswith('_')}
        if 'kitchen_config_file' in wandb_config and wandb_config['kitchen_config_file'] is not None:
            # Log the contents of the config for discovery of what the quantization
            # settings were.
            with open(wandb_config['kitchen_config_file'], "r") as f:
                wandb_config['kitchen_config_file_contents'] = f.read()
        # wandb.init() creates a new run unless it is given an id to resume, and this
        # runs on every process start. A long job restarts many times -- a requeue, a
        # node failure, an in-job restarter -- and each restart would otherwise open a
        # fresh run, scattering one training run across many and leaving every chart
        # covering only the stretch between two failures. Deriving a stable id from the
        # experiment name and resuming it keeps them a single timeline. An unnamed run has
        # nothing stable to derive an id from, so wandb assigns one as before.
        run_id = cfg.logger.wandb_run_id or (
            _wandb_run_id(cfg.logger.wandb_exp_name) if cfg.logger.wandb_exp_name else None
        )
        wandb_kwargs = {
            'dir': save_dir,
            "id": run_id,
            'name': cfg.logger.wandb_exp_name,
            'project': cfg.logger.wandb_project,
            'config': wandb_config,
            "resume": "allow"}
        if cfg.logger.wandb_entity:
            wandb_kwargs['entity'] = cfg.logger.wandb_entity
        os.makedirs(wandb_kwargs['dir'], exist_ok=True)
        wandb.init(**wandb_kwargs)
        _GLOBAL_WANDB_WRITER = wandb


def _set_one_logger(args):
    cfg = get_run_config()
    global _GLOBAL_ONE_LOGGER
    _ensure_var_is_not_initialized(_GLOBAL_ONE_LOGGER, 'one logger')

    if cfg.logger.enable_one_logger and args.rank == (safe_get_world_size() - 1):
        if cfg.logger.one_logger_async or cfg.logger.wandb_project:
            one_logger_async = True
        else:
            one_logger_async = False
        try:
            from one_logger import OneLogger
            config = {
               'project': cfg.logger.one_logger_project,
               'name': cfg.logger.one_logger_run_name,
               'async': one_logger_async,
            }
            one_logger = OneLogger(config=config)
            _GLOBAL_ONE_LOGGER = one_logger
        except Exception:
            print('WARNING: one_logger package is required to enable e2e metrics '
                  'tracking. please go to '
                  'https://confluence.nvidia.com/display/MLWFO/Package+Repositories'
                  ' for details to install it')

def _set_adlr_autoresume():
    """Initialize ADLR autoresume."""
    global _GLOBAL_ADLR_AUTORESUME
    _ensure_var_is_not_initialized(_GLOBAL_ADLR_AUTORESUME, 'adlr autoresume')

    cfg = get_run_config()
    if cfg.fault_tolerance.adlr_autoresume:
        from megatron.training.utils import print_rank_0
        print_rank_0('enabling autoresume ...')
        sys.path.append(os.environ.get('SUBMIT_SCRIPTS', '.'))
        try:
            from userlib.auto_resume import AutoResume
        except ImportError:
            print_rank_0('ADLR autoresume is not available, exiting ...')
            sys.exit()

        _GLOBAL_ADLR_AUTORESUME = AutoResume


def _set_timers(args):
    """Initialize timers."""
    cfg = get_run_config()
    global _GLOBAL_TIMERS
    _ensure_var_is_not_initialized(_GLOBAL_TIMERS, 'timers')
    _GLOBAL_TIMERS = Timers(cfg.logger.timing_log_level, cfg.logger.timing_log_option)

def _set_energy_monitor(args):
    """Initialize energy monitor."""
    global _GLOBAL_ENERGY_MONITOR
    _ensure_var_is_not_initialized(_GLOBAL_ENERGY_MONITOR, 'energy monitor')
    _GLOBAL_ENERGY_MONITOR = EnergyMonitor()


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, '{} is not initialized.'.format(name)


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, '{} is already initialized.'.format(name)

def _set_telemetry(args, *, include_training: bool = True):
    """Initialise OTel telemetry handle following the wandb/tensorboard pattern."""
    global _GLOBAL_TELEMETRY_HANDLE
    from megatron.core.telemetry import telemetry as _otel

    cfg = get_run_config()
    telemetry_args = Namespace(**vars(args))
    telemetry_args.local_rank = cfg.dist.local_rank
    telemetry_args.otel_enabled = cfg.logger.otel_enabled
    telemetry_args.otel_service_name = cfg.logger.otel_service_name
    if include_training:
        for name in ('global_batch_size', 'micro_batch_size', 'train_iters', 'train_samples'):
            setattr(telemetry_args, name, getattr(cfg.train, name))
        for name in ('tensor_model_parallel_size', 'pipeline_model_parallel_size', 'recompute_granularity'):
            setattr(telemetry_args, name, getattr(cfg.model.transformer, name))
        telemetry_args.data_parallel_size = cfg.get_data_parallel_size()
        telemetry_args.optimizer = cfg.optimizer.optimizer
    _GLOBAL_TELEMETRY_HANDLE = _otel.setup(telemetry_args, include_training=include_training)

def destroy_global_vars():
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = None

    global _GLOBAL_RUN_CONFIG
    _GLOBAL_RUN_CONFIG = None

    global _GLOBAL_TRAIN_STATE
    _GLOBAL_TRAIN_STATE = None

    global _GLOBAL_TOKENIZER
    _GLOBAL_TOKENIZER = None

    global _GLOBAL_TENSORBOARD_WRITER
    _GLOBAL_TENSORBOARD_WRITER = None

    global _GLOBAL_WANDB_WRITER
    _GLOBAL_WANDB_WRITER = None

    global _GLOBAL_ONE_LOGGER
    _GLOBAL_ONE_LOGGER = None

    global _GLOBAL_ADLR_AUTORESUME
    _GLOBAL_ADLR_AUTORESUME = None

    global _GLOBAL_TIMERS
    _GLOBAL_TIMERS = None

    global _GLOBAL_ENERGY_MONITOR
    _GLOBAL_ENERGY_MONITOR = None

    global _GLOBAL_SIGNAL_HANDLER
    _GLOBAL_SIGNAL_HANDLER = None

    _shutdown_telemetry()
