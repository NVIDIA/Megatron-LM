# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""CPU tests for per-module meta tensor materialization."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parents[2]))
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.fsdp_module import FSDPModule


class _MetaChild(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(3, device="meta", dtype=torch.bfloat16))
        self.register_buffer("running_mean", torch.empty(1, device="meta", dtype=torch.float32))
        self.reset_calls = 0
        self.weight_id_after_reset = None
        self.buffer_id_after_reset = None

    def reset_parameters(self):
        self.reset_calls += 1
        with torch.no_grad():
            self.weight.fill_(2)
            self.running_mean.fill_(4)
        self.weight_id_after_reset = id(self.weight)
        self.buffer_id_after_reset = id(self.running_mean)


class _MetaBufferOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "indices", torch.empty(3, device="meta", dtype=torch.int64), persistent=False
        )
        self.reset_calls = 0
        self.buffer_id_after_reset = None

    def reset_parameters(self):
        self.reset_calls += 1
        self.indices.copy_(torch.tensor([5, 6, 7], device=self.indices.device))
        self.buffer_id_after_reset = id(self.indices)


class _MetaParent(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(2, device="meta", dtype=torch.float64))
        self.register_buffer("scale", torch.empty(2, device="meta", dtype=torch.int32))
        self.child = _MetaChild()
        self.buffer_only = _MetaBufferOnly()
        self.reset_calls = 0
        self.weight_id_after_reset = None
        self.buffer_id_after_reset = None

    def reset_parameters(self):
        # Parent reset hooks must see already-materialized descendants.
        assert not self.child.weight.is_meta
        self.reset_calls += 1
        with torch.no_grad():
            self.weight.fill_(3)
            self.scale.copy_(torch.tensor([8, 9], device=self.scale.device))
        self.weight_id_after_reset = id(self.weight)
        self.buffer_id_after_reset = id(self.scale)


def test_materialize_meta_module_is_non_recursive_and_skips_buffer_only_modules(
    monkeypatch: pytest.MonkeyPatch,
):
    """Children reset before parents while deliberately lazy buffers stay meta."""
    model = _MetaParent()
    lazy_buffer_id = id(model.buffer_only.indices)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))

    FSDPModule._materialize_meta_module(model, ignored_modules=set())

    assert [model.reset_calls, model.child.reset_calls, model.buffer_only.reset_calls] == [1, 1, 0]
    assert all(not tensor.is_meta for tensor in model.parameters())
    assert all(tensor.device.type == "cpu" for tensor in model.parameters())
    assert not model.scale.is_meta and model.scale.device.type == "cpu"
    assert not model.child.running_mean.is_meta
    assert model.child.running_mean.device.type == "cpu"
    # Match v1: buffer-only modules may keep nonpersistent state lazy for forward.
    assert model.buffer_only.indices.is_meta
    assert id(model.buffer_only.indices) == lazy_buffer_id

    assert model.weight.dtype == torch.float64
    assert model.scale.dtype == torch.int32
    assert model.child.weight.dtype == torch.bfloat16
    assert model.child.running_mean.dtype == torch.float32
    assert model.buffer_only.indices.dtype == torch.int64

    assert id(model.weight) == model.weight_id_after_reset
    assert id(model.scale) == model.buffer_id_after_reset
    assert id(model.child.weight) == model.child.weight_id_after_reset
    assert id(model.child.running_mean) == model.child.buffer_id_after_reset
    assert model.buffer_only.buffer_id_after_reset is None

    torch.testing.assert_close(model.weight, torch.full((2,), 3.0, dtype=torch.float64))
    assert torch.equal(model.scale, torch.tensor([8, 9], dtype=torch.int32))
    torch.testing.assert_close(model.child.weight, torch.full((3,), 2.0, dtype=torch.bfloat16))
    torch.testing.assert_close(model.child.running_mean, torch.tensor([4.0]))
