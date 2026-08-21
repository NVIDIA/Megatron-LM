# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Compatibility for VERL weight transfer used by the MLite engine."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from collections.abc import Iterable

_BUCKETED_SENDER_MODULE = "verl.workers.rollout.vllm_rollout.bucketed_weight_transfer"


class _SyncBucketProducer:
    """Pack one sender bucket at a time into a caller-owned staging slot."""

    def __init__(self, weights, bucket_size: int):
        from verl_mlite.rollout.layer_cluster import resync_layer_cluster_key

        self._weights = iter(weights)
        self._bucket_size = bucket_size
        self._pending = None
        self._exhausted = False
        self._layer_cluster_key = resync_layer_cluster_key

    def next_bucket(self, staging):
        import torch

        if staging.device.type == "cuda":
            torch.cuda.set_device(staging.device)
        if self._exhausted:
            return "eof", None, None, 0, None, True

        offset = 0
        bucket_meta = {}
        bucket_layer_key = None
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

            layer_key = self._layer_cluster_key(name)
            if (
                bucket_meta
                and bucket_layer_key is not None
                and layer_key != bucket_layer_key
            ):
                self._pending = (name, weight)
                break

            # Fix-A (resync IPC byte-alignment): pad every tensor's start
            # offset up to an 8-byte boundary. The receiver reconstructs each
            # tensor as ``buffer[offset:offset+size].view(dtype)``, and
            # ``Tensor.view(dtype)`` requires the byte ``storage_offset`` to be
            # divisible by ``dtype.itemsize``. A pure BF16/FP32 stream keeps
            # every offset even, so it never trips (why the proxy stays green).
            # But DS4 real-weight resync ships block-FP8 tensors (itemsize 1);
            # an odd-numel FP8 tensor leaves ``offset`` odd and the *next*
            # BF16/FP32 tensor in the same bucket crashes on the view. Aligning
            # to 8 bytes covers every dtype we transport (fp8/bf16/fp16/fp32)
            # and is byte-lossless — the receiver reads the padded offset we
            # record in ``bucket_meta`` unchanged. See
            # tests/unit/verl/test_resync_bucket_byte_alignment.py.
            offset = (offset + 7) & ~7
            if offset + weight.nbytes > self._bucket_size and bucket_meta:
                self._pending = (name, weight)
                break
            if weight.nbytes > self._bucket_size:
                return "direct", name, weight, 0, None, False

            if not bucket_meta:
                bucket_layer_key = layer_key
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
            # Padding can push a full bucket a few bytes past exact equality, so
            # break on ``>=`` rather than ``==``.
            if offset >= self._bucket_size:
                break

        ready = None
        if staging.device.type == "cuda":
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(staging.device))
        return "bucket", bucket_meta, None, offset, ready, self._exhausted


def _install_bucketed_sender_prefetch(sender_cls: type) -> bool:
    """Install the MLite-aware bucket packer without moving model work off-thread.

    ``weights`` is not an ordinary host iterator: advancing it can enter FSDP
    parameter materialization and PP/EP collectives.  Those operations must run
    on the actor thread which owns the rank's CUDA device and model lifecycle.
    Only the produced tensors may be copied into the IPC staging buffer here.
    """
    if getattr(sender_cls, "_mlite_weight_prefetch_patch", False):
        return False

    original_async_send_weights = sender_cls.async_send_weights

    async def prefetched_async_send_weights(self, weights):
        import torch

        if not isinstance(weights, Iterable) or hasattr(weights, "__aiter__") or self.use_shm:
            return await original_async_send_weights(self, weights)

        try:
            self._init_socket()
            self._init_buffer()
            if self.buffer.device.type != "cuda" and not getattr(
                self, "_mlite_prefetch_allow_cpu", False
            ):
                raise RuntimeError("MLite sender prefetch requires a CUDA IPC buffer")

            staging = torch.empty_like(self.buffer)
            producer = _SyncBucketProducer(weights, self.bucket_size)
            while True:
                kind, metadata_or_name, direct_weight, used_bytes, ready, is_last = (
                    producer.next_bucket(staging)
                )
                if kind == "eof":
                    self.socket.send_pyobj({"bucket_meta": {}, "is_last": True})
                    self.socket.recv()
                    break
                if kind == "direct":
                    self._direct_send_large_weight(metadata_or_name, direct_weight)
                    continue

                if ready is not None:
                    ready.synchronize()
                self.buffer[:used_bytes].copy_(staging[:used_bytes], non_blocking=True)
                if self.buffer.device.type == "cuda":
                    torch.cuda.synchronize(self.buffer.device)

                self.socket.send_pyobj(
                    {"bucket_meta": metadata_or_name, "is_last": is_last}
                )
                self.socket.recv()
                if is_last:
                    break
        finally:
            self._cleanup()

    sender_cls.async_send_weights = prefetched_async_send_weights
    sender_cls._mlite_weight_prefetch_patch = True
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


class _SenderPatchFinder(importlib.abc.MetaPathFinder):
    _mlite_weight_sender_finder = True

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
    if any(getattr(finder, "_mlite_weight_sender_finder", False) for finder in sys.meta_path):
        return False
    sys.meta_path.insert(0, _SenderPatchFinder())
    return True


def _patch_bucketed_weight_sender() -> bool:
    """Install bounded producer prefetch for VERL's bucketed sender."""
    return _patch_bucketed_weight_transfer()


def apply_runtime_patches() -> None:
    _patch_bucketed_weight_sender()


def load_verl_engine_api():
    from verl.workers.engine.base import BaseEngine, BaseEngineCtx, EngineRegistry
    from verl.workers.engine.utils import postprocess_batch_func, prepare_micro_batches

    return (
        BaseEngine,
        BaseEngineCtx,
        EngineRegistry,
        postprocess_batch_func,
        prepare_micro_batches,
    )
