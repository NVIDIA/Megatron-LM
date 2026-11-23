# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Pluggable checkpoint manager for the training loop.

The training loop saves, resumes and finalizes asynchronous checkpoints through a
:class:`CheckpointManager`. :class:`DefaultCheckpointManager` delegates to the functional
``save_checkpoint`` / ``load_checkpoint`` / ``maybe_finalize_async_save`` code, so behavior is
unchanged unless ``checkpoint.custom_manager_class`` names another implementation.

A custom manager is a class that is constructed with the run's ``CheckpointConfig`` and implements
``save(ctx)``, ``load(ctx)`` and ``finalize_async_saves(blocking, terminate)``::

    class MyCheckpointManager:
        def __init__(self, checkpoint_config): ...
        def save(self, ctx: CheckpointSaveContext) -> None: ...
        def load(self, ctx: CheckpointLoadContext) -> tuple[int, int]: ...
        def finalize_async_saves(self, blocking: bool = False, terminate: bool = False) -> None: ...

Select it with ``--custom-manager-class pkg.module.MyCheckpointManager``. The path must pass the
``_target_`` allowlist (``megatron.training.config.target_allowlist``), the same gate that protects
configs loaded from YAML; register the package prefix with ``target_allowlist.add_prefix`` first.

Scope: the manager serves the training loop's periodic save, the primary resume load and the
asynchronous-save finalization. One-off saves (upcycling, format conversion) and the auxiliary or
inference loads (RL reference weights, the distillation teacher, ``megatron/inference``) still call
the functional API directly.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch

    from megatron.training.config import CheckpointConfig
    from megatron.training.config.container import ConfigContainerBase
    from megatron.training.state import TrainState


@dataclass
class CheckpointSaveContext:
    """Everything one checkpoint save needs.

    Attributes:
        iteration: Training iteration being saved.
        model: List of model chunks.
        optimizer: The optimizer (may be ``None``).
        opt_param_scheduler: The optimizer parameter scheduler (may be ``None``).
        num_floating_point_operations_so_far: Cumulative FLOPs up to this point.
        checkpointing_context: The run's persistent checkpointing context (caches strategies and
            the local checkpoint manager across saves).
        non_persistent_ckpt: Whether this is a non-persistent (temporary) checkpoint.
        train_data_iterator: Training data iterator, whose state is saved when it supports it.
        preprocess_common_state_dict_fn: Optional hook applied to the common state dict.
        tp_group, pp_group, cp_group, dp_group, dp_cp_group, dp_gtp_remat_group, expt_dp_group:
            Process groups of this rank's module (``None`` falls back to the mpu groups).
        rng_state_key_prefix: Per-grid RNG key namespace of a multi-grid model.
        on_saved: Called once the checkpoint is durable (after finalization for async saves).
        train_state: The run's full ``TrainState``. Custom managers may use more than the default
            save path needs (``None`` when no run state is initialized, e.g. direct test calls).
        config: The run's full config container (``None`` when not available).
    """

    iteration: int
    model: list[Any]
    optimizer: Any
    opt_param_scheduler: Any
    num_floating_point_operations_so_far: int
    checkpointing_context: Optional[dict[str, Any]] = None
    non_persistent_ckpt: bool = False
    train_data_iterator: Any = None
    preprocess_common_state_dict_fn: Optional[Callable] = None
    tp_group: Optional["torch.distributed.ProcessGroup"] = None
    pp_group: Optional["torch.distributed.ProcessGroup"] = None
    cp_group: Optional["torch.distributed.ProcessGroup"] = None
    dp_group: Optional["torch.distributed.ProcessGroup"] = None
    dp_cp_group: Optional["torch.distributed.ProcessGroup"] = None
    dp_gtp_remat_group: Optional["torch.distributed.ProcessGroup"] = None
    expt_dp_group: Optional["torch.distributed.ProcessGroup"] = None
    rng_state_key_prefix: str = ""
    on_saved: Optional[Callable[[], None]] = None
    train_state: Optional["TrainState"] = None
    config: Optional["ConfigContainerBase"] = None


@dataclass
class CheckpointLoadContext:
    """Everything one checkpoint load needs.

    Attributes:
        model: List of (wrapped) model chunks to load into.
        optimizer: The optimizer to load into (may be ``None``).
        opt_param_scheduler: The optimizer parameter scheduler (may be ``None``).
        checkpointing_context: The run's persistent checkpointing context.
        strict: Whether to enforce strict key matching (see ``torch.nn.Module.load_state_dict``).
        restore_training_state: Restore optimizer, scheduler, rerun and data progress (training
            resume must opt in).
        record_finetune: Record whether finetune semantics applied on the train state; the primary
            training load passes ``True``.
        tp_group, pp_group, dp_cp_group, dp_group, expt_dp_group, checkpoint_group: Process groups
            of this rank's module (``None`` falls back to the mpu groups).
        rng_state_key_prefix: Per-grid RNG key namespace of a multi-grid model.
        model_sharded_state_dict_modifier: Optional in-place edit of each model sharded state dict
            before loading (torch_dist only).
        train_state: The run's full ``TrainState`` (``None`` when no run state is initialized).
        config: The run's full config container (``None`` when not available).
    """

    model: list[Any]
    optimizer: Any
    opt_param_scheduler: Any
    checkpointing_context: Optional[dict[str, Any]] = None
    strict: bool = True
    restore_training_state: bool = False
    record_finetune: bool = False
    tp_group: Optional["torch.distributed.ProcessGroup"] = None
    pp_group: Optional["torch.distributed.ProcessGroup"] = None
    dp_cp_group: Optional["torch.distributed.ProcessGroup"] = None
    dp_group: Optional["torch.distributed.ProcessGroup"] = None
    expt_dp_group: Optional["torch.distributed.ProcessGroup"] = None
    checkpoint_group: Optional["torch.distributed.ProcessGroup"] = None
    rng_state_key_prefix: str = ""
    model_sharded_state_dict_modifier: Optional[Callable[[dict], None]] = None
    train_state: Optional["TrainState"] = None
    config: Optional["ConfigContainerBase"] = None


@runtime_checkable
class CheckpointManager(Protocol):
    """Interface the training loop uses for checkpoint save, resume and async finalization.

    Implementations are constructed with the run's ``CheckpointConfig``.
    """

    def __init__(self, checkpoint_config: "CheckpointConfig") -> None:
        """Construct the manager; ``checkpoint_config`` is the only constructor argument."""
        ...

    def save(self, ctx: CheckpointSaveContext) -> None:
        """Save a checkpoint."""
        ...

    def load(self, ctx: CheckpointLoadContext) -> tuple[int, int]:
        """Load a checkpoint.

        Returns:
            ``(iteration, num_floating_point_operations_so_far)``; ``(0, 0)`` when nothing was
            loaded.
        """
        ...

    def finalize_async_saves(self, blocking: bool = False, terminate: bool = False) -> None:
        """Finalize pending asynchronous saves.

        Args:
            blocking: Wait for all pending saves to complete.
            terminate: Close the asynchronous queue after finalizing.
        """
        ...


class DefaultCheckpointManager:
    """Delegates to the functional checkpoint code (``save_checkpoint`` and friends)."""

    # ``save_checkpoint`` already starts and ends the fault-tolerance "checkpointing" section and
    # counts the save. The trainer does this for any manager that does not (see
    # ``manager_reports_checkpointing``), so each save is counted exactly once.
    reports_checkpointing = True

    def __init__(self, checkpoint_config: "CheckpointConfig") -> None:
        self.checkpoint_config = checkpoint_config

    def save(self, ctx: CheckpointSaveContext) -> None:
        """Save a checkpoint with ``save_checkpoint``."""
        from megatron.training.checkpointing import save_checkpoint

        save_checkpoint(
            ctx.iteration,
            ctx.model,
            ctx.optimizer,
            ctx.opt_param_scheduler,
            ctx.num_floating_point_operations_so_far,
            ctx.checkpointing_context,
            non_persistent_ckpt=ctx.non_persistent_ckpt,
            train_data_iterator=ctx.train_data_iterator,
            preprocess_common_state_dict_fn=ctx.preprocess_common_state_dict_fn,
            tp_group=ctx.tp_group,
            pp_group=ctx.pp_group,
            dp_cp_group=ctx.dp_cp_group,
            dp_group=ctx.dp_group,
            dp_gtp_remat_group=ctx.dp_gtp_remat_group,
            expt_dp_group=ctx.expt_dp_group,
            rng_state_key_prefix=ctx.rng_state_key_prefix,
            cp_group=ctx.cp_group,
            on_saved=ctx.on_saved,
        )

    def load(self, ctx: CheckpointLoadContext) -> tuple[int, int]:
        """Load a checkpoint with ``load_checkpoint``."""
        from megatron.training.checkpointing import load_checkpoint

        return load_checkpoint(
            ctx.model,
            ctx.optimizer,
            ctx.opt_param_scheduler,
            strict=ctx.strict,
            checkpointing_context=ctx.checkpointing_context,
            skip_load_to_model_and_opt=self._loads_in_place(ctx),
            tp_group=ctx.tp_group,
            pp_group=ctx.pp_group,
            dp_cp_group=ctx.dp_cp_group,
            dp_group=ctx.dp_group,
            expt_dp_group=ctx.expt_dp_group,
            checkpoint_group=ctx.checkpoint_group,
            rng_state_key_prefix=ctx.rng_state_key_prefix,
            model_sharded_state_dict_modifier=ctx.model_sharded_state_dict_modifier,
            restore_training_state=ctx.restore_training_state,
            record_finetune=ctx.record_finetune,
        )

    def _loads_in_place(self, ctx: CheckpointLoadContext) -> bool:
        """Whether the native loader restores tensors in place, so ``load_state_dict`` is skipped.

        This is the torch FSDP2 + ``torch_dist`` resume. It is specific to the native loader, so it
        stays here instead of being part of the context a custom manager implements.
        """
        if ctx.config is None or self.checkpoint_config.ckpt_format != "torch_dist":
            return False
        try:
            from megatron.core.distributed import TorchFullyShardedDataParallel  # noqa: F401
        except ImportError:
            return False
        from megatron.core.distributed import TorchFullyShardedDataParallelConfig

        return isinstance(ctx.config.ddp, TorchFullyShardedDataParallelConfig)

    def finalize_async_saves(self, blocking: bool = False, terminate: bool = False) -> None:
        """Finalize pending asynchronous saves with ``maybe_finalize_async_save``."""
        from megatron.training.async_utils import maybe_finalize_async_save

        maybe_finalize_async_save(blocking=blocking, terminate=terminate)


def create_checkpoint_manager(checkpoint_config: "CheckpointConfig") -> CheckpointManager:
    """Create the checkpoint manager the config selects.

    Returns a :class:`DefaultCheckpointManager` unless ``checkpoint_config.custom_manager_class``
    is set, in which case that class is imported and constructed with ``checkpoint_config``.

    Raises:
        ValueError: The path is malformed or not permitted by the ``_target_`` allowlist.
        ImportError: The module cannot be imported.
        AttributeError: The module has no such class.
        TypeError: The class does not implement the :class:`CheckpointManager` interface.
    """
    target = getattr(checkpoint_config, "custom_manager_class", None)
    if target is None:
        return DefaultCheckpointManager(checkpoint_config)

    from megatron.training.config.instantiate_utils import (
        InstantiationException,
        _reject_unsafe_target_name,
        target_allowlist,
    )

    if "." not in target:
        raise ValueError(
            f"Invalid checkpoint.custom_manager_class '{target}'. "
            "Expected a fully qualified class name like 'mypackage.module.ClassName'."
        )
    if not target_allowlist.is_allowed(target):
        raise ValueError(
            f"checkpoint.custom_manager_class '{target}' is not permitted by the _target_ "
            "allowlist. Register its package with "
            "megatron.training.config.target_allowlist.add_prefix('<package>.') first."
        )
    # Same unsafe-name denylist as the checkpoint reader, applied before anything is imported.
    try:
        _reject_unsafe_target_name(target=target, full_key="checkpoint.custom_manager_class")
    except InstantiationException as e:
        raise ValueError(f"Invalid checkpoint.custom_manager_class '{target}': {e}") from e

    module_path, class_name = target.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Could not import module '{module_path}' for custom checkpoint manager: {e}"
        ) from e
    try:
        manager_class = getattr(module, class_name)
    except AttributeError as e:
        raise AttributeError(f"Module '{module_path}' has no class '{class_name}': {e}") from e

    # Validate the class before calling it: the allowlist admits whole packages, so the target may
    # be any callable in them (e.g. a function with side effects), and calling it is the risk.
    if not inspect.isclass(manager_class):
        raise TypeError(
            f"Custom checkpoint manager '{target}' must be a class, got {type(manager_class).__name__}."
        )
    missing = [
        name
        for name in ("save", "load", "finalize_async_saves")
        if not callable(getattr(manager_class, name, None))
    ]
    if missing:
        raise TypeError(
            f"Custom checkpoint manager '{target}' must implement save(ctx), load(ctx) and "
            f"finalize_async_saves(blocking, terminate); missing: {', '.join(missing)}."
        )
    try:
        inspect.signature(manager_class).bind(checkpoint_config)
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"Custom checkpoint manager '{target}' must be constructible with a single "
            "CheckpointConfig argument."
        ) from e
    try:
        inspect.signature(manager_class.save).bind(None, None)
        inspect.signature(manager_class.load).bind(None, None)
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"Custom checkpoint manager '{target}': save and load must each accept one context "
            "argument."
        ) from e
    return manager_class(checkpoint_config)


_checkpoint_manager: Optional[CheckpointManager] = None
_checkpoint_manager_config: Any = None


def get_checkpoint_manager() -> CheckpointManager:
    """Return the run's checkpoint manager, creating it from the run config on first use.

    The manager is rebuilt if the run's checkpoint config object changed, so a second run in the
    same process never inherits the previous run's custom manager.
    """
    global _checkpoint_manager, _checkpoint_manager_config
    from megatron.training.global_vars import get_run_config

    checkpoint_config = get_run_config().checkpoint
    if _checkpoint_manager is None or _checkpoint_manager_config is not checkpoint_config:
        _checkpoint_manager = create_checkpoint_manager(checkpoint_config)
        _checkpoint_manager_config = checkpoint_config
    return _checkpoint_manager


def manager_reports_checkpointing(manager: CheckpointManager) -> bool:
    """Whether ``manager.save`` already reports the save to the fault-tolerance monitor.

    Custom managers default to ``False``, so the trainer brackets their ``save`` with the
    checkpointing section and counts it. Set ``reports_checkpointing = True`` on a manager whose
    ``save`` calls ``ft_integration.on_checkpointing_start/end`` itself (as the native one does).
    """
    return bool(getattr(manager, "reports_checkpointing", False))


def set_checkpoint_manager(manager: Optional[CheckpointManager]) -> None:
    """Install ``manager`` for the current run; ``None`` re-creates it from the config on next use."""
    global _checkpoint_manager, _checkpoint_manager_config
    _checkpoint_manager = manager
    if manager is not None:
        from megatron.training.global_vars import get_run_config

        _checkpoint_manager_config = get_run_config().checkpoint
    else:
        # Drop the stale config too; get_checkpoint_manager rebuilds from the run config.
        _checkpoint_manager_config = None
