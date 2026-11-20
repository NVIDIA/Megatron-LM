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

import json
import os
import runpy
from pathlib import Path

import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.dbuffer import DBuffer
from megatron.training import training

os.environ["NCCL_DEBUG"] = "INFO"
records = []
active = False
steps = 0


def signature(tensor):
    raw = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
    rows = 16 if raw.numel() and raw.numel() % 16 == 0 else 1
    digest = torch.hash_tensor(raw.reshape(rows, -1), dim=1)
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "digest": digest}


def wrap_collective(name):
    original = getattr(DBuffer, name)

    def wrapped(self, mesh_axis, *args, **kwargs):
        if not active:
            return original(self, mesh_axis, *args, **kwargs)
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
    global active, steps
    steps += 1
    active = steps == 1
    result = original_train_step(*args, **kwargs)
    active = False
    if steps == 1:
        torch.cuda.synchronize()
        for record in records:
            for side in ("input", "output"):
                record[side]["digest"] = record[side]["digest"].cpu().tolist()
        out = Path(os.environ["COLLECTIVE_TRACE_DIR"])
        out.mkdir(parents=True, exist_ok=True)
        rank = torch.distributed.get_rank()
        (out / f"rank{rank}.json").write_text(json.dumps(records, indent=2))
    return result


training.train_step = train_step
runpy.run_path("pretrain_gpt.py", run_name="__main__")
