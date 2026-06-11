# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Perfetto markers for the internals of the PyTorch DCP *save* path.

The MCore save path for the ``torch_dcp`` / ``fsdp_dtensor`` formats does not go
through ``torch.distributed.checkpoint.save`` directly when async save is on:
it calls the nvrx/mcore ``save_state_dict_async_plan`` helper, which performs the
*synchronous* (training-thread-blocking) part of the save -- planning plus the
GPU->CPU/shm staging (``prepare_write_data``) -- and hands the actual file write
to a background worker. The sync ``torch.distributed.checkpoint.save`` path goes
through ``state_dict_saver._save_state_dict`` instead. Both paths drive the same
sequence of *planner* and *storage-writer* methods:

    1. planner.set_up_planner()               # init state dict, flatten maps
    2. storage_writer.set_up_storage_writer()  # writer init
    3. planner.create_local_plan()            # which WriteItems this rank writes
    4. storage_writer.prepare_local_plan()
    5. planner.create_global_plan()           # merge/validate plans (coordinator)
    6. storage_writer.prepare_global_plan()
    7. planner.finish_plan()
    8. storage_writer.prepare_write_data()     # *** staging: D2H copy to CPU/shm
       storage_writer.write_data()             #     (sync path / background write)

Steps 1-7 are the planning collective (driven by ``_DistWrapper.reduce_scatter``,
already instrumented in ``torch_dcp_load_trace``). Step 8's ``prepare_write_data``
is where async save stages the whole sharded model+optimizer into pinned/shm
memory before the worker writes it -- usually the dominant blocking cost.

Rather than patch the (async, nvrx-owned) ``save_state_dict_async_plan`` driver,
we wrap the planner/storage-writer methods *at the class level* so the markers
fire regardless of which driver (sync ``_save_state_dict`` or async
``save_state_dict_async_plan``) calls them. The cross-rank collectives are
covered by reusing ``_patch_dist_wrapper_collectives`` from the load module
(``_DistWrapper`` is shared by save and load).

Everything is gated on ``CKPT_PERFETTO_TRACE=1`` (same switch as
``perfetto_trace``); when tracing is off ``apply_torch_dcp_save_trace_patch`` is
a no-op. Best-effort: any planner/writer class that can't be imported on this
PyT / nvidia-resiliency-ext build is simply skipped.
"""

import functools
import os

from megatron.core.perfetto_trace import trace_region

# Reuse the shared marker prefix and the _DistWrapper collective instrumentation
# from the load module so save markers group with load markers (``dcp.*``) and
# the gather/scatter/reduce_scatter collectives are covered exactly once.
from megatron.core.dist_checkpointing.strategies.torch_dcp_load_trace import (
    _PREFIX,
    _patch_dist_wrapper_collectives,
)

# SavePlanner methods, in call order (names match _save_state_dict /
# save_state_dict_async_plan).
_SAVE_PLANNER_METHODS = (
    "set_up_planner",
    "create_local_plan",
    "create_global_plan",
    "finish_plan",
)
# StorageWriter methods. ``prepare_write_data`` is nvrx-only (the staging step)
# and is skipped gracefully on writers that don't define it. ``write_data`` /
# ``write_preloaded_data`` cover the actual write (the latter runs in the async
# worker, so it only shows up on that process's trace, if any).
_SAVE_WRITER_METHODS = (
    "set_up_storage_writer",
    "prepare_local_plan",
    "prepare_global_plan",
    "prepare_write_data",
    "write_data",
    "write_preloaded_data",
    "finish",
)
# Marks a class method as already wrapped (kept on the wrapper fn).
_WRAPPED_FLAG = "_perfetto_dcp_save_wrapped"

_patched = False


def _tracing_enabled() -> bool:
    return os.environ.get("CKPT_PERFETTO_TRACE", "0") == "1"


def _wrap_class_method(cls, name: str) -> None:
    """Wrap ``cls.name`` with a ``trace_region`` (idempotent, best-effort).

    Patches at the class level so every instance is covered regardless of which
    driver calls it. Base classes are patched before subclasses by the caller so
    that an *inherited* (non-overridden) method is wrapped once: the subclass
    lookup then resolves to the already-wrapped base function (flag set) and is
    skipped, while a subclass *override* (no flag) is wrapped on the subclass.
    """
    if cls is None:
        return
    try:
        orig = getattr(cls, name)
    except AttributeError:
        return
    if orig is None or getattr(orig, _WRAPPED_FLAG, False):
        return

    label = _PREFIX + name

    @functools.wraps(orig)
    def _traced(self, *args, **kwargs):
        with trace_region(label):
            return orig(self, *args, **kwargs)

    setattr(_traced, _WRAPPED_FLAG, True)
    try:
        setattr(cls, name, _traced)
    except (AttributeError, TypeError):
        return


def _planner_classes() -> list:
    """Return the SavePlanner classes to instrument (base first)."""
    classes = []
    try:
        from torch.distributed.checkpoint.default_planner import DefaultSavePlanner

        classes.append(DefaultSavePlanner)
    except Exception:
        pass
    try:
        # Subclass used by the torch_dist path; harmless to patch for fsdp_dtensor.
        from megatron.core.dist_checkpointing.strategies.torch import MCoreSavePlanner

        classes.append(MCoreSavePlanner)
    except Exception:
        pass
    return classes


def _writer_classes() -> list:
    """Return the StorageWriter classes to instrument (base first)."""
    classes = []
    try:
        from torch.distributed.checkpoint import FileSystemWriter

        classes.append(FileSystemWriter)
    except Exception:
        pass
    # nvrx async writer (the one used with --async-strategy nvrx).
    try:
        from nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async import (
            FileSystemWriterAsync as NvrxFileSystemWriterAsync,
        )

        classes.append(NvrxFileSystemWriterAsync)
    except Exception:
        pass
    # mcore async writer (the --async-strategy mcore fallback).
    try:
        from megatron.core.dist_checkpointing.strategies.filesystem_async import (
            FileSystemWriterAsync as McoreFileSystemWriterAsync,
        )

        classes.append(McoreFileSystemWriterAsync)
    except Exception:
        pass
    return classes


def apply_torch_dcp_save_trace_patch() -> None:
    """Patch the DCP save planner/writer/collectives to emit Perfetto regions.

    Idempotent and gated on ``CKPT_PERFETTO_TRACE=1``. Safe to call on every
    save; it wraps each target class method exactly once and only when tracing
    is enabled.
    """
    global _patched
    if _patched or not _tracing_enabled():
        return

    # Cross-rank collective planning (reduce_scatter:plan / gather_object /
    # scatter_object). Shared with the load path's _DistWrapper instrumentation.
    _patch_dist_wrapper_collectives()

    # Per-phase planner + storage-writer methods (base classes first so
    # inherited methods are wrapped once -- see _wrap_class_method).
    for cls in _planner_classes():
        for name in _SAVE_PLANNER_METHODS:
            _wrap_class_method(cls, name)
    for cls in _writer_classes():
        for name in _SAVE_WRITER_METHODS:
            _wrap_class_method(cls, name)

    _patched = True
