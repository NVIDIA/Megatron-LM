# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Diagnostic collective fingerprints for the first real training step."""

import functools
import importlib
import inspect
import json
import os
import runpy
import sys
from collections import Counter
from pathlib import Path

import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.dbuffer import DBuffer
from megatron.training import training

os.environ["NCCL_DEBUG"] = "INFO"
records = []
gemm_records = []
gemm_counts = Counter()
active = False
compute_active = False
steps = 0


def signature(tensor):
    if tensor.numel() == 0:
        return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "digest": None}
    raw = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
    rows = 16 if raw.numel() and raw.numel() % 16 == 0 else 1
    digest = torch.hash_tensor(raw.reshape(rows, -1), dim=1)
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "digest": digest}


def tensor_signatures(value):
    if isinstance(value, torch.Tensor):
        return signature(value)
    if isinstance(value, (tuple, list)):
        return [tensor_signatures(item) for item in value]
    if value is None:
        return None
    return {"unsupported_type": type(value).__name__}


def resolve(value):
    if isinstance(value, torch.Tensor):
        return value.cpu().tolist()
    if isinstance(value, dict):
        return {key: resolve(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [resolve(item) for item in value]
    return value


gemm_module = importlib.import_module("transformer_engine.pytorch.cpp_extensions.gemm")
for gemm_name in ("general_gemm", "general_grouped_gemm"):
    original_gemm = getattr(gemm_module, gemm_name)
    gemm_signature = inspect.signature(original_gemm)

    def make_wrapper(original, call_signature, name):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            if not compute_active:
                return original(*args, **kwargs)
            bound = call_signature.bind(*args, **kwargs)
            bound.apply_defaults()
            values = bound.arguments
            gemm_counts[name] += 1
            entry = {
                "seq": len(gemm_records),
                "id": f"{name}:{gemm_counts[name]}",
                "name": name,
                "metadata": {
                    key: str(values.get(key))
                    for key in [
                        "layout",
                        "grad",
                        "accumulate",
                        "out_dtype",
                        "m_splits",
                        "use_bias",
                        "use_split_accumulator",
                        "single_output",
                    ]
                },
                "inputs": {
                    "A": tensor_signatures(values.get("A")),
                    "B": tensor_signatures(values.get("B")),
                },
            }
            if values.get("accumulate"):
                entry["inputs"]["prior_output"] = tensor_signatures(values.get("out"))
            result = original(*args, **kwargs)
            entry["outputs"] = tensor_signatures(result[0])
            gemm_records.append(entry)
            return result

        return wrapped

    replacement = make_wrapper(original_gemm, gemm_signature, gemm_name)
    for module in list(sys.modules.values()):
        if getattr(module, "__name__", "").startswith(("transformer_engine", "megatron")):
            if getattr(module, gemm_name, None) is original_gemm:
                setattr(module, gemm_name, replacement)


def wrap_collective(name):
    original = getattr(DBuffer, name)

    def wrapped(self, mesh_axis, *args, **kwargs):
        global compute_active
        if not active:
            return original(self, mesh_axis, *args, **kwargs)
        compute_active = False
        entry = {
            "seq": len(records),
            "name": name,
            "mesh": self.mesh.mesh.tolist(),
            "axis": mesh_axis,
            "placements": str(self.placements),
            "symmetric_memory": self.is_symmetric_memory,
            "input": signature(self.local_buffer),
        }
        result = original(self, mesh_axis, *args, **kwargs)
        entry["output"] = signature(result.local_buffer)
        records.append(entry)
        return result

    setattr(DBuffer, name, wrapped)


for collective in ("reduce_scatter", "allreduce"):
    wrap_collective(collective)

original_train_step = training.train_step


def train_step(*args, **kwargs):
    global active, compute_active, steps
    steps += 1
    active = steps == 1
    compute_active = active
    result = original_train_step(*args, **kwargs)
    active = False
    if steps == 1:
        torch.cuda.synchronize()
        out = Path(os.environ["COLLECTIVE_TRACE_DIR"])
        out.mkdir(parents=True, exist_ok=True)
        rank = torch.distributed.get_rank()
        (out / f"rank{rank}.json").write_text(json.dumps(resolve(records), indent=2))
        (out / f"gemm-rank{rank}.json").write_text(json.dumps(resolve(gemm_records), indent=2))
    return result


training.train_step = train_step
runpy.run_path("pretrain_gpt.py", run_name="__main__")
