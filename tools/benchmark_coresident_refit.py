# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Portions adapted from NVIDIA-NeMo/RL under Apache-2.0.
# https://github.com/NVIDIA-NeMo/RL/tree/34dd6b3b4f8f108554ec0c9312d83ba7e639da9d

"""Compare co-resident refit with per-tensor copies on a CUDA GPU.

Run with: uv run python tools/benchmark_coresident_refit.py
Reports end-to-end copy wall time; verifies values and persistent addresses.
"""

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable

import torch

from megatron.core.resharding.coresident import copy_refit


def _reference(src: dict[str, torch.Tensor], dst: dict[str, torch.Tensor]) -> dict[str, float]:
    copied = shared = 0
    with torch.inference_mode():
        for name, tensor in dst.items():
            source = src[name]
            if source.shape != tensor.shape:
                raise ValueError(f"shape mismatch for {name}")
            nbytes = source.numel() * source.element_size()
            if tensor is source:
                shared += nbytes
                continue
            tensor.copy_(source)
            copied += nbytes
    torch.cuda.current_stream().synchronize()
    return {"bytes": float(copied), "bytes_shared": float(shared)}


def _measure(fn: Callable[[], object], repeats: int) -> float:
    for _ in range(3):
        fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(1000 * (time.perf_counter() - start))
    return statistics.median(times)


def _write_json(record: dict) -> None:
    sys.stdout.write(json.dumps(record) + "\n")
    sys.stdout.flush()


def main() -> None:
    """Benchmark refit and report one JSON record per placement and tensor size."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensors", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--mixed-dtypes", action="store_true", help="Alternate BF16 parameters and FP32 buffers"
    )
    args = parser.parse_args()
    if args.tensors < 1 or args.repeats < 1:
        parser.error("--tensors and --repeats must be positive")
    torch.cuda.set_device(0)
    _write_json({"gpu": torch.cuda.get_device_name(), "torch": torch.__version__})
    for elements in (4096, 262144, 4194304):
        for shared in (False, True):
            src = {
                str(i): torch.full(
                    (elements,),
                    (i + 1) / 128,
                    device="cuda",
                    dtype=torch.float32 if args.mixed_dtypes and i % 2 else torch.bfloat16,
                )
                for i in range(args.tensors)
            }
            dst = {
                name: tensor if shared and int(name) % 4 >= 2 else torch.empty_like(tensor)
                for name, tensor in src.items()
            }
            pointers = {name: tensor.data_ptr() for name, tensor in dst.items()}
            torch.cuda.synchronize()
            baseline_ms = _measure(
                lambda source=src, target=dst: _reference(source, target), args.repeats
            )
            for name, tensor in dst.items():
                if tensor.data_ptr() != src[name].data_ptr():
                    tensor.zero_()
            result = copy_refit(src, dst.items())
            assert all(torch.equal(tensor, src[name]) for name, tensor in dst.items())
            assert pointers == {name: tensor.data_ptr() for name, tensor in dst.items()}
            batched_ms = _measure(
                lambda source=src, target=dst: copy_refit(source, target.items()), args.repeats
            )
            _write_json(
                {
                    "elements": elements,
                    "tensors": args.tensors,
                    "half_shared": shared,
                    "mixed_dtypes": args.mixed_dtypes,
                    "baseline_ms": baseline_ms,
                    "batched_ms": batched_ms,
                    "speedup": baseline_ms / batched_ms,
                    **result,
                }
            )
            del src, dst


if __name__ == "__main__":
    main()
