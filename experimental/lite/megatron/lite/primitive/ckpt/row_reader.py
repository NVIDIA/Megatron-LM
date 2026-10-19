# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Concurrent native safetensors row slices without shared mmap or decode caches."""
from concurrent.futures import ThreadPoolExecutor

from safetensors import safe_open

from megatron.lite.primitive.ckpt.hf_weights import SafeTensorReader


class RowSliceReader(SafeTensorReader):
    def get_row_tensors(self, requests):
        """Read native row ranges concurrently, with one mmap context per request.

        Each request is (name, start, stop). Only those rows are materialized;
        workers never share the reader's handles, decode cache or CUDA context.
        """

        def read(request):
            name, start, stop = request
            path = self.path / (self.index[name] if self.index else "model.safetensors")
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                return handle.get_slice(name)[start:stop]

        requests = tuple(requests)
        with ThreadPoolExecutor(max_workers=min(4, max(1, len(requests)))) as pool:
            return dict(zip((r[0] for r in requests), pool.map(read, requests)))
