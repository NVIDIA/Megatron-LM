# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Perfetto markers for the internals of ``torch.distributed.checkpoint.load``.

The MCore ``TorchDistLoadShardedStrategy.load`` already wraps the top-level
``checkpoint.load(...)`` call in a ``trace_region("checkpoint.load")``. That
single marker hides *where* the time inside DCP actually goes. This module
breaks that black box open.

What ``checkpoint.load`` does (PyT pinned commit)
-------------------------------------------------
``checkpoint.load`` (``state_dict_loader.load``) does some light state-dict
massaging and then delegates to the module-level ``_load_state_dict``, which
drives the load through a fixed sequence of *storage-reader* and *planner*
method calls (closures ``local_step`` / ``global_step`` / ``read_data``):

    1. storage_reader.read_metadata()         # parse / unpickle .metadata
    2. planner.set_up_planner()               # init state dict, flatten maps
    3. storage_reader.set_up_storage_reader() # stash storage_data (offsets)
    4. planner.create_local_plan()            # which ReadItems this rank needs
    5. storage_reader.prepare_local_plan()    # reader fixups (no-op for FS)
    6. planner.create_global_plan()           # merge/validate plans
    7. storage_reader.prepare_global_plan()   # reader fixups (no-op for FS)
    8. planner.finish_plan()                  # finalize the plan
    9. storage_reader.read_data()             # *** the bulk: open .distcp,
                                              #     torch.load, narrow, copy_ ***
       all_reads.wait()

Steps 1-8 are cheap (metadata + planning). Step 9 (``read_data``) is where
nearly all of the wall-clock goes: it groups the ``ReadItem``s by file, opens
each ``.distcp`` shard, ``torch.load``s every saved tensor, narrows it to the
requested slice and ``copy_``s it into the destination tensor.

These steps live inside nested closures of ``_load_state_dict``, so they can't
be wrapped one-by-one from the outside. Instead we instrument the *instance
methods* of the actual ``storage_reader`` and ``planner`` objects that
``_load_state_dict`` drives -- so every step shows up as its own Perfetto
region, in call order, nested under the existing ``checkpoint.load`` marker.

Robustness
----------
* Instance-level wrapping (not class patching) means we transparently cover
  every reader subclass MCore uses -- ``FileSystemReader``,
  ``CachedMetadataFileSystemReader`` (mcore/nvrx), the MSC reader -- and the
  ``MCoreLoadPlanner`` -- without caring about ``super()`` overrides.
* Everything is gated on ``CKPT_PERFETTO_TRACE=1`` (same switch as
  ``perfetto_trace``); when tracing is off, ``apply_torch_dcp_load_trace_patch``
  is a no-op and ``_load_state_dict`` is left untouched.
* The optional per-file breakdown of ``read_data`` (env
  ``CKPT_PERFETTO_TRACE_DCP_FILES=1``) mirrors the pinned
  ``FileSystemReader.read_data`` source and falls back to a single plain
  marker if the reader doesn't expose the expected internals.
"""

import functools
import os

from megatron.core.perfetto_trace import trace_region

# Phase -> the object whose method implements it. Names match the calls made by
# torch.distributed.checkpoint.state_dict_loader._load_state_dict.
_READER_METHODS = (
    "read_metadata",
    "set_up_storage_reader",
    "prepare_local_plan",
    "prepare_global_plan",
    "read_data",
)
_PLANNER_METHODS = (
    "set_up_planner",
    "create_local_plan",
    "create_global_plan",
    "finish_plan",
)

# Marker prefix so the DCP-internal regions are easy to spot/group in the trace.
_PREFIX = "dcp."

# Per-instance bookkeeping: remember which methods we've already wrapped so the
# patch stays idempotent even if the same reader/planner object is reused.
_WRAPPED_ATTR = "_perfetto_dcp_wrapped"

_patched = False


def _tracing_enabled() -> bool:
    return os.environ.get("CKPT_PERFETTO_TRACE", "0") == "1"


def _read_data_file_breakdown_enabled() -> bool:
    return os.environ.get("CKPT_PERFETTO_TRACE_DCP_FILES", "0") == "1"


def _wrap_instance_method(obj, name: str) -> None:
    """Shadow ``obj.name`` with a ``trace_region``-wrapped version (idempotent).

    We set a plain function as an *instance attribute*, which shadows the class
    method on attribute lookup. The closure captures the already-bound original
    method, so ``self`` is handled correctly and we never touch the class.
    """
    if obj is None:
        return
    bound = getattr(obj, name, None)
    if bound is None:
        return

    try:
        wrapped = obj.__dict__.get(_WRAPPED_ATTR)
    except AttributeError:
        # __slots__ object without __dict__: can't shadow instance attrs.
        return
    if wrapped is None:
        wrapped = set()
        try:
            setattr(obj, _WRAPPED_ATTR, wrapped)
        except (AttributeError, TypeError):
            return
    if name in wrapped:
        return

    label = _PREFIX + name

    @functools.wraps(bound)
    def _traced(*args, **kwargs):
        with trace_region(label):
            return bound(*args, **kwargs)

    try:
        setattr(obj, name, _traced)
    except (AttributeError, TypeError):
        return
    wrapped.add(name)


def _install_read_data_file_breakdown(reader) -> bool:
    """Replace ``reader.read_data`` with a per-``.distcp``-file traced version.

    Faithful reimplementation of the pinned ``FileSystemReader.read_data`` with
    a ``dcp.read_data`` outer region and a ``dcp.read_file:<name>`` region per
    shard file. Returns True if installed, False if the reader doesn't look like
    a FileSystem-style reader (caller then falls back to a single marker).
    """
    # Required internals from torch's FileSystemReader.
    needed = ("storage_data", "fs", "path", "_slice_file", "transforms")
    if not all(hasattr(reader, attr) for attr in needed):
        return False

    try:
        wrapped = reader.__dict__.get(_WRAPPED_ATTR)
    except AttributeError:
        return False
    if wrapped is None:
        wrapped = set()
        try:
            setattr(reader, _WRAPPED_ATTR, wrapped)
        except (AttributeError, TypeError):
            return False
    if "read_data" in wrapped:
        return True

    import io

    import torch
    from torch.distributed._shard._utils import narrow_tensor_by_index
    from torch.distributed.checkpoint.planner import LoadItemType
    from torch.futures import Future

    def read_data(plan, planner):
        with trace_region(_PREFIX + "read_data"):
            # group requests by file
            per_file = {}
            for read_item in plan.items:
                item_md = reader.storage_data[read_item.storage_index]
                per_file.setdefault(item_md.relative_path, []).append(read_item)

            for relative_path, reqs in per_file.items():
                with trace_region(_PREFIX + "read_file:" + str(relative_path)):
                    new_path = reader.fs.concat_path(reader.path, relative_path)
                    with reader.fs.create_stream(new_path, "rb") as stream:
                        for req in reqs:
                            item_md = reader.storage_data[req.storage_index]
                            file_slice = reader._slice_file(stream, item_md)
                            transform_from = reader.transforms.transform_load_stream(
                                req, item_md.transform_descriptors or (), file_slice
                            )

                            if req.type == LoadItemType.BYTE_IO:
                                read_bytes = io.BytesIO(transform_from.read(-1))
                                read_bytes.seek(0)
                                planner.load_bytes(req, read_bytes)
                            else:
                                if transform_from.seekable():
                                    seekable = transform_from
                                else:
                                    seekable = io.BytesIO(transform_from.read(-1))
                                    seekable.seek(0)

                                tensor = torch.load(
                                    seekable, map_location="cpu", weights_only=True
                                )
                                tensor = narrow_tensor_by_index(
                                    tensor, req.storage_offsets, req.lengths
                                )
                                target_tensor = planner.resolve_tensor(req).detach()

                                if target_tensor.size() != tensor.size():
                                    raise AssertionError(
                                        f"req {req.storage_index} mismatch sizes "
                                        f"{target_tensor.size()} vs {tensor.size()}"
                                    )
                                target_tensor.copy_(tensor)
                                planner.commit_tensor(req, target_tensor)

            fut: Future = Future()
            fut.set_result(None)
            return fut

    try:
        setattr(reader, "read_data", read_data)
    except (AttributeError, TypeError):
        return False
    wrapped.add("read_data")
    return True


def _instrument(storage_reader, planner) -> None:
    """Wrap the reader/planner methods that ``_load_state_dict`` drives."""
    file_breakdown = _read_data_file_breakdown_enabled()
    installed_deep_read = False
    if file_breakdown and storage_reader is not None:
        installed_deep_read = _install_read_data_file_breakdown(storage_reader)

    for name in _READER_METHODS:
        if name == "read_data" and installed_deep_read:
            continue  # already replaced with the per-file traced version
        _wrap_instance_method(storage_reader, name)

    for name in _PLANNER_METHODS:
        _wrap_instance_method(planner, name)


def apply_torch_dcp_load_trace_patch() -> None:
    """Patch ``_load_state_dict`` to emit per-phase Perfetto regions.

    Idempotent and gated on ``CKPT_PERFETTO_TRACE=1``. Safe to call on every
    load; it patches the ``_load_state_dict`` module function exactly once and
    only when tracing is enabled.
    """
    global _patched
    if _patched or not _tracing_enabled():
        return

    import torch.distributed.checkpoint.state_dict_loader as _sdl

    orig_load_state_dict = _sdl._load_state_dict

    @functools.wraps(orig_load_state_dict)
    def _traced_load_state_dict(
        state_dict,
        storage_reader,
        process_group=None,
        coordinator_rank=0,
        no_dist=False,
        planner=None,
    ):
        # Instrument the concrete objects this call will drive. ``planner`` may
        # be None here (the default planner is built inside _load_state_dict);
        # MCore always passes its MCoreLoadPlanner, so the planner phases are
        # covered in practice.
        _instrument(storage_reader, planner)
        return orig_load_state_dict(
            state_dict,
            storage_reader,
            process_group,
            coordinator_rank,
            no_dist,
            planner,
        )

    _sdl._load_state_dict = _traced_load_state_dict
    _patched = True
