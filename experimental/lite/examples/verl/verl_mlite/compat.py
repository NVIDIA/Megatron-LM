# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Small compatibility patches for dependency-version gaps in examples."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import os
import queue
import sys
import threading
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from functools import wraps
from pathlib import Path
from typing import Any

_BUCKETED_SENDER_MODULE = "verl.workers.rollout.vllm_rollout.bucketed_weight_transfer"


class _SyncBucketProducer:
    """Pack one sender bucket at a time into a caller-owned staging slot."""

    def __init__(self, weights, bucket_size: int):
        self._weights = iter(weights)
        self._bucket_size = bucket_size
        self._pending = None
        self._exhausted = False

    def next_bucket(self, staging):
        import torch

        if staging.device.type == "cuda":
            torch.cuda.set_device(staging.device)
        if self._exhausted:
            return "eof", None, None, 0, None, True

        offset = 0
        bucket_meta = {}
        while True:
            try:
                if self._pending is None:
                    name, weight = next(self._weights)
                else:
                    name, weight = self._pending
                    self._pending = None
            except StopIteration:
                self._exhausted = True
                if not bucket_meta:
                    return "eof", None, None, 0, None, True
                break

            if offset + weight.nbytes > self._bucket_size and bucket_meta:
                self._pending = (name, weight)
                break
            if weight.nbytes > self._bucket_size:
                return "direct", name, weight, 0, None, False

            bucket_meta[name] = {
                "name": name,
                "shape": weight.shape,
                "dtype": weight.dtype,
                "offset": offset,
                "handle": None,
            }
            staging[offset : offset + weight.nbytes].copy_(
                weight.view(-1).view(torch.uint8), non_blocking=True
            )
            offset += weight.nbytes
            if offset == self._bucket_size:
                break

        ready = None
        if staging.device.type == "cuda":
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(staging.device))
        return "bucket", bucket_meta, None, offset, ready, self._exhausted


def _install_bucketed_sender_prefetch(sender_cls: type) -> bool:
    """Overlap synchronous weight production with the receiver's bucket ACK."""
    if getattr(sender_cls, "_mlite_weight_prefetch_patch", False):
        return False

    original_async_send_weights = sender_cls.async_send_weights

    async def prefetched_async_send_weights(self, weights):
        import torch

        if not isinstance(weights, Iterable) or hasattr(weights, "__aiter__") or self.use_shm:
            return await original_async_send_weights(self, weights)

        executor = None
        stop = threading.Event()
        free_slots = None
        ready_results = None
        held_slot = None
        try:
            self._init_socket()
            self._init_buffer()
            if self.buffer.device.type != "cuda" and not getattr(
                self, "_mlite_prefetch_allow_cpu", False
            ):
                raise RuntimeError("MLite sender prefetch requires a CUDA IPC buffer")

            staging_slots = [torch.empty_like(self.buffer) for _ in range(2)]
            producer = _SyncBucketProducer(weights, self.bucket_size)
            free_slots = queue.Queue(maxsize=2)
            ready_results = queue.Queue(maxsize=2)
            for slot_index in range(2):
                free_slots.put_nowait(slot_index)
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlite-weight-prefetch")

            def put_ready(result):
                while not stop.is_set():
                    try:
                        ready_results.put(result, timeout=0.05)
                        return True
                    except queue.Full:
                        continue
                return False

            def produce():
                slot_index = None
                try:
                    while not stop.is_set():
                        try:
                            slot_index = free_slots.get(timeout=0.05)
                        except queue.Empty:
                            continue
                        result = producer.next_bucket(staging_slots[slot_index])
                        if not put_ready((*result, slot_index)):
                            return
                        slot_index = None
                        kind, *_, is_last = result
                        if kind == "eof" or is_last:
                            return
                except BaseException as exc:
                    put_ready(("error", exc, None, 0, None, True, slot_index))

            context = copy_context()
            worker_future = executor.submit(context.run, produce)
            while True:
                try:
                    result = ready_results.get(timeout=0.1)
                except queue.Empty:
                    if worker_future.done():
                        worker_future.result()
                        raise RuntimeError("MLite weight prefetch stopped without a terminal result")
                    continue

                kind, metadata_or_name, direct_weight, used_bytes, ready, is_last, held_slot = (
                    result
                )
                if kind == "error":
                    raise metadata_or_name
                if kind == "eof":
                    free_slots.put_nowait(held_slot)
                    held_slot = None
                    self.socket.send_pyobj({"bucket_meta": {}, "is_last": True})
                    self.socket.recv()
                    break
                if kind == "direct":
                    free_slots.put_nowait(held_slot)
                    held_slot = None
                    self._direct_send_large_weight(metadata_or_name, direct_weight)
                    continue

                if ready is not None:
                    ready.synchronize()
                staging = staging_slots[held_slot]
                self.buffer[:used_bytes].copy_(staging[:used_bytes], non_blocking=True)
                if self.buffer.device.type == "cuda":
                    torch.cuda.synchronize(self.buffer.device)
                free_slots.put_nowait(held_slot)
                held_slot = None

                self.socket.send_pyobj(
                    {"bucket_meta": metadata_or_name, "is_last": is_last}
                )
                self.socket.recv()
                if is_last:
                    break
        finally:
            stop.set()
            if held_slot is not None and free_slots is not None:
                try:
                    free_slots.put_nowait(held_slot)
                except queue.Full:
                    pass
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            self._cleanup()

    sender_cls.async_send_weights = prefetched_async_send_weights
    sender_cls._mlite_weight_prefetch_patch = True
    return True


def _weight_sync_probe_enabled() -> bool:
    return os.getenv("MLITE_WEIGHT_SYNC_PROBE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _instrument_bucketed_weight_sender(sender_cls: type) -> bool:
    """Patch veRL's sender only while the opt-in sync probe is enabled."""
    if getattr(sender_cls, "_mlite_weight_sync_probe_patch", False):
        return False

    import torch
    import torch.distributed as dist
    from torch.utils._python_dispatch import TorchDispatchMode

    from megatron.lite.primitive.ckpt.weight_sync_probe import (
        get_weight_sync_probe,
        weight_sync_probe_session,
    )

    probe = get_weight_sync_probe()
    original_init_socket = sender_cls._init_socket
    original_async_send_weights = sender_cls.async_send_weights

    class _ProfiledSocket:
        def __init__(self, socket):
            self._socket = socket

        def __getattr__(self, name):
            return getattr(self._socket, name)

        def send_pyobj(self, *args, **kwargs):
            with probe.measure("handshake"):
                return self._socket.send_pyobj(*args, **kwargs)

        def recv(self, *args, **kwargs):
            with probe.measure("handshake"):
                return self._socket.recv(*args, **kwargs)

    class _H2DCopyMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func is torch.ops.aten.copy_.default and len(args) >= 2:
                dst, src = args[:2]
                if (
                    isinstance(dst, torch.Tensor)
                    and isinstance(src, torch.Tensor)
                    and dst.device.type == "cuda"
                    and src.device.type == "cpu"
                ):
                    with probe.measure("h2d", nbytes=src.nbytes, device=dst.device):
                        return func(*args, **kwargs)
            return func(*args, **kwargs)

    def profiled_init_socket(self, *args, **kwargs):
        result = original_init_socket(self, *args, **kwargs)
        self.socket = _ProfiledSocket(self.socket)
        return result

    async def profiled_async_send_weights(self, weights):
        backend = os.getenv("MLITE_WEIGHT_SYNC_PROBE_BACKEND", "unknown")
        original_all_gather_into_tensor = dist.all_gather_into_tensor

        def profiled_all_gather_into_tensor(output, tensor, *args, **kwargs):
            with probe.measure("mbridge_gather", nbytes=output.nbytes, device=tensor.device):
                return original_all_gather_into_tensor(output, tensor, *args, **kwargs)

        with weight_sync_probe_session(backend), _H2DCopyMode():
            dist.all_gather_into_tensor = profiled_all_gather_into_tensor
            try:
                return await original_async_send_weights(self, weights)
            finally:
                dist.all_gather_into_tensor = original_all_gather_into_tensor

    sender_cls._init_socket = profiled_init_socket
    sender_cls.async_send_weights = profiled_async_send_weights
    sender_cls._mlite_weight_sync_probe_patch = True
    return True


class _SenderPatchLoader(importlib.abc.Loader):
    def __init__(self, loader: importlib.abc.Loader):
        self._loader = loader

    def create_module(self, spec):
        create_module = getattr(self._loader, "create_module", None)
        return create_module(spec) if create_module is not None else None

    def exec_module(self, module) -> None:
        self._loader.exec_module(module)
        _install_bucketed_sender_prefetch(module.BucketedWeightSender)
        if _weight_sync_probe_enabled():
            _instrument_bucketed_weight_sender(module.BucketedWeightSender)


class _SenderPatchFinder(importlib.abc.MetaPathFinder):
    _mlite_weight_sync_probe_finder = True

    def __init__(self):
        self._mlite_weight_sync_probe_requested = False

    def find_spec(self, fullname, path, target=None):
        if fullname != _BUCKETED_SENDER_MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = _SenderPatchLoader(spec.loader)
        return spec


def _patch_bucketed_weight_transfer() -> bool:
    module = sys.modules.get(_BUCKETED_SENDER_MODULE)
    if module is not None:
        return _install_bucketed_sender_prefetch(module.BucketedWeightSender)
    if any(getattr(finder, "_mlite_weight_sync_probe_finder", False) for finder in sys.meta_path):
        return False
    sys.meta_path.insert(0, _SenderPatchFinder())
    return True


def _patch_bucketed_weight_sender() -> bool:
    """Install production prefetch plus optional probe instrumentation."""
    changed = _patch_bucketed_weight_transfer()
    if not _weight_sync_probe_enabled():
        return changed

    module = sys.modules.get(_BUCKETED_SENDER_MODULE)
    if module is not None:
        changed = _instrument_bucketed_weight_sender(module.BucketedWeightSender) or changed
    else:
        finder = next(
            finder
            for finder in sys.meta_path
            if getattr(finder, "_mlite_weight_sync_probe_finder", False)
        )
        if not finder._mlite_weight_sync_probe_requested:
            finder._mlite_weight_sync_probe_requested = True
            changed = True
    return changed


def _patch_transformers_rope_ignore_keys() -> None:
    try:
        import transformers.modeling_rope_utils as rope_utils
    except Exception:
        return

    for cls in vars(rope_utils).values():
        if not isinstance(cls, type):
            continue
        if getattr(cls, "_verl_mlite_rope_ignore_keys_patch", False):
            continue
        descriptor = vars(cls).get("_check_received_keys")
        if descriptor is None:
            continue

        is_staticmethod = isinstance(descriptor, staticmethod)
        is_classmethod = isinstance(descriptor, classmethod)
        original = descriptor.__func__ if is_staticmethod or is_classmethod else descriptor

        def build_wrapper(check_received_keys: Any) -> Any:
            @wraps(check_received_keys)
            def patched(*args: Any, **kwargs: Any) -> Any:
                ignore_keys = kwargs.get("ignore_keys")
                if isinstance(ignore_keys, list):
                    kwargs["ignore_keys"] = set(ignore_keys)
                elif ignore_keys is not None and not isinstance(ignore_keys, set):
                    if isinstance(ignore_keys, Iterable) and not isinstance(
                        ignore_keys, (str, bytes)
                    ):
                        kwargs["ignore_keys"] = set(ignore_keys)
                return check_received_keys(*args, **kwargs)

            return patched

        patched = build_wrapper(original)
        if is_staticmethod:
            cls._check_received_keys = staticmethod(patched)
        elif is_classmethod:
            cls._check_received_keys = classmethod(patched)
        else:
            cls._check_received_keys = patched
        cls._verl_mlite_rope_ignore_keys_patch = True


def apply_runtime_patches() -> None:
    _patch_transformers_rope_ignore_keys()
    _patch_bucketed_weight_sender()


def _load_verl_file(relative_path: str, module_name: str):
    spec = importlib.util.find_spec("verl")
    if spec is None or spec.submodule_search_locations is None:
        raise ModuleNotFoundError("No module named 'verl'")

    path = Path(next(iter(spec.submodule_search_locations))) / relative_path
    file_spec = importlib.util.spec_from_file_location(module_name, path)
    if file_spec is None or file_spec.loader is None:
        raise ImportError(f"Unable to load VERL module from {path}")

    module = importlib.util.module_from_spec(file_spec)
    sys.modules[module_name] = module
    file_spec.loader.exec_module(module)
    return module


def load_verl_engine_api():
    # Prefer the canonical package import so the MLite engine registers into the
    # SAME EngineRegistry that verl's trainers resolve against. Loading base.py as
    # a standalone module (below) creates a *duplicate* registry, which silently
    # drops the mlite backend ("Unknown backend: mlite"). The file-load path is
    # only a fallback for environments where verl isn't importable as a package.
    try:
        from verl.workers.engine.base import BaseEngine, BaseEngineCtx, EngineRegistry
        from verl.workers.engine.utils import postprocess_batch_func, prepare_micro_batches
    except (ModuleNotFoundError, ImportError):
        base = _load_verl_file("workers/engine/base.py", "_verl_mlite_verl_engine_base")
        utils = _load_verl_file("workers/engine/utils.py", "_verl_mlite_verl_engine_utils")
        BaseEngine = base.BaseEngine
        BaseEngineCtx = base.BaseEngineCtx
        EngineRegistry = base.EngineRegistry
        postprocess_batch_func = utils.postprocess_batch_func
        prepare_micro_batches = utils.prepare_micro_batches

    return BaseEngine, BaseEngineCtx, EngineRegistry, postprocess_batch_func, prepare_micro_batches
