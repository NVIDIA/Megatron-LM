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

"""Unit tests for Megatron-FSDP v2 CUDA graph capture hooks."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.cuda_graph_runner import (
    CudaGraphRunner,
    _make_bwd_post_hook,
    _prepare_compiled_modules_for_capture,
    _restore_compiled_modules_after_capture_failure,
)


def test_capture_backward_post_hook_clears_only_unsharded_parameter_grads():
    """Warmup grads must not survive outside the CUDA graph private pool."""
    full_param = torch.nn.Parameter(torch.ones(4))
    full_param.grad = torch.full_like(full_param, 2)
    full_param.main_grad = torch.full_like(full_param, 3)

    dist_param = torch.nn.Parameter(torch.ones(2))
    dist_param.grad = torch.full_like(dist_param, 4)

    reshard_calls = []
    module = SimpleNamespace(
        _fsdp_param_groups=[
            SimpleNamespace(params=[full_param], dist_params=[dist_param])
        ],
        reshard=lambda: reshard_calls.append(True),
    )

    _make_bwd_post_hook(module)(module, (), ())

    assert full_param.grad is None
    assert torch.equal(full_param.main_grad, torch.full_like(full_param, 3))
    assert torch.equal(dist_param.grad, torch.full_like(dist_param, 4))
    assert reshard_calls == [True]


def test_module_compile_is_converted_to_compiled_forward_for_capture():
    module = torch.nn.Linear(2, 2)
    original_forward = module.forward
    compiled_call_impl = object()
    compiled_forward = object()
    module._compiled_call_impl = compiled_call_impl

    with patch.object(torch, "compile", return_value=compiled_forward) as compile_mock:
        saved = _prepare_compiled_modules_for_capture([module])

    compile_mock.assert_called_once_with(
        original_forward,
        dynamic=False,
        options={"triton.cudagraphs": False},
    )
    assert module._compiled_call_impl is None
    assert module.forward is compiled_forward

    _restore_compiled_modules_after_capture_failure(saved)
    assert module._compiled_call_impl is compiled_call_impl
    assert module.forward == original_forward


def test_module_compile_is_normalized_when_first_forward_is_recorded():
    module = torch.nn.Linear(2, 2)
    module._compiled_call_impl = object()

    def compiled_forward(input):
        return input

    runner = CudaGraphRunner(graph_pool=None)
    sample = torch.ones(1, 2)

    with patch.object(torch, "compile", return_value=compiled_forward):
        runner.record_module(module, (sample,), {})

    assert module._compiled_call_impl is None
    assert module.forward is compiled_forward
    assert len(runner._compiled_module_state) == 1

