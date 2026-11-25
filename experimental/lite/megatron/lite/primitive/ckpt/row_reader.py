# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Bounded native-payload prefetch and direct copies into caller-owned host row shards."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from math import prod

import torch
from safetensors import safe_open

from megatron.lite.primitive.ckpt.hf_weights import SafeTensorReader


class RowSliceReader(SafeTensorReader):
    """Workers own mmap handles and copy raw bytes only; CUDA stays on the caller."""

    def __init__(self, path, *, device="cpu", workers=4, prefetch_bytes=512 << 20):
        super().__init__(path, device=device)
        if workers < 1 or prefetch_bytes < 1:
            raise ValueError("Prefetch workers and byte budget must be positive")
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="hf-row-read")
        self._budget, self._pending_bytes = prefetch_bytes, 0
        self._queue, self._pending, self._rows = deque(), {}, {}
        self._closed = False
        self._descriptions = {}

    def _metadata(self, name):
        if name in self._descriptions:
            return self._descriptions[name]
        path = self.path / (self.index[name] if self.index else "model.safetensors")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            view = handle.get_slice(name)
            shape = tuple(view.get_shape())
            dtype = view[:0].dtype if shape else handle.get_tensor(name).dtype
        self._descriptions[name] = (path, shape, dtype)
        return path, shape, dtype

    @staticmethod
    def _slice(path, name, start, stop):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            return handle.get_slice(name)[start:stop]

    @staticmethod
    def _copy(path, name, start, stop, target):
        # No shared handles, decoding, casts, pinned allocation or CUDA in workers.
        with torch.no_grad(), safe_open(str(path), framework="pt", device="cpu") as handle:
            value = handle.get_slice(name)[start:stop] if target.ndim else handle.get_tensor(name)
            target.copy_(value)
        return target

    def prefetch_row_copies(self, requests, *, chunk_bytes=256 << 20):
        """Schedule (name, start, stop, target) into existing CPU buffers, no full copy."""
        if chunk_bytes < 1 or self._closed:
            raise ValueError("Cannot schedule rows with an invalid chunk size or closed reader")
        for name, start, stop, target in requests:
            path, shape, dtype = self._metadata(name)
            if (
                name in self._rows
                or target.device.type != "cpu"
                or not shape
                or not 0 <= start <= stop <= shape[0]
                or target.shape[0] < stop - start
                or tuple(target.shape[1:]) != shape[1:]
                or target.dtype != dtype
            ):
                raise ValueError(f"Invalid native row destination for {name}")
            target[stop - start :].zero_()  # only padding; never clear an entire table first
            rows = max(1, chunk_bytes // max(1, prod(shape[1:]) * target.element_size()))
            tasks = [
                self._pool.submit(
                    self._copy,
                    path,
                    name,
                    r,
                    min(r + rows, stop),
                    target[r - start : min(r + rows, stop) - start],
                )
                for r in range(start, stop, rows)
            ]
            self._rows[name] = (start, stop, target, tasks)

    def get_row_tensors(self, requests):
        """Join prefetched rows or read native slices with independent handles."""
        result, tasks = {}, {}
        for name, start, stop in requests:
            if name in self._rows:
                r0, r1, target, futures = self._rows.pop(name)
                if (start, stop) != (r0, r1):
                    raise ValueError(f"Prefetched row range changed for {name}")
                for future in futures:
                    future.result()
                result[name] = target
            else:
                path, shape, dtype = self._metadata(name)
                tasks[name] = self._pool.submit(self._slice, path, name, start, stop)
        result.update((name, future.result()) for name, future in tasks.items())
        return result

    def prefetch_tensors(self, names):
        """Queue logical weights and their actual quantization sidecars in use order."""
        keys = []
        for name in names:
            if not self.has_tensor(name) and self.has_tensor(name + "_packed"):
                keys.extend(name + suffix for suffix in ("_packed", "_scale", "_shape"))
            else:
                keys.append(name)
                scale = (name[:-7] if name.endswith(".weight") else name) + ".scale"
                keys.extend(k for k in (name + "_scale_inv", scale) if self.has_tensor(k))
        self._queue.extend(dict.fromkeys(keys))
        self._fill()

    def _fill(self):
        while self._queue:
            name = self._queue[0]
            path, shape, dtype = self._metadata(name)
            size = prod(shape) * dtype.itemsize
            if self._pending and self._pending_bytes + size > self._budget:
                break
            self._queue.popleft()
            # Allocate pinned storage only on the main thread, after CUDA init.
            target = torch.empty(shape, dtype=dtype, pin_memory=self.device.type == "cuda")
            future = self._pool.submit(self._copy, path, name, 0, shape[0] if shape else 0, target)
            self._pending[name] = (future, size)
            self._pending_bytes += size

    def _get_raw_tensor(self, name, device):
        item = self._pending.pop(name, None)
        if item is None:
            return super()._get_raw_tensor(name, device)
        future, size = item
        value = future.result()
        self._pending_bytes -= size
        result = value if device.type == "cpu" else value.to(device, non_blocking=True)
        self._fill()
        return result

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            for _, _, _, futures in self._rows.values():
                for future in futures:
                    future.result()
            for future, _ in self._pending.values():
                future.result()
        finally:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._rows.clear()
            self._pending.clear()
            self._queue.clear()
            super().close()
