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

"""Unit tests for Megatron-FSDP v2 CUDA graph capture hooks."""

from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.cuda_graph_runner import (
    CudaGraphRunner,
    _make_bwd_post_hook,
    _make_bwd_pre_hook,
    _prepare_compiled_modules_for_capture,
    _restore_compiled_modules_after_capture_failure,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.fsdp_module import FSDPModule
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.hooks import _pre_backward_setup
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.te_graph_runtime.graph import (
    _get_compatible_main_grad_buffer,
    _get_static_grad_buffers,
    _refresh_module_parameter_surface,
    _static_grad_context_wrapper,
    make_graphed_callables,
)
from megatron.core.tensor_parallel.layers import linear_with_grad_accumulation_and_async_allreduce


def test_capture_backward_post_hook_clears_only_unsharded_parameter_grads():
    """Warmup grads must not survive outside the CUDA graph private pool."""
    full_param = torch.nn.Parameter(torch.ones(4))
    full_param.grad = torch.full_like(full_param, 2)
    full_param.main_grad = torch.full_like(full_param, 3)

    dist_param = torch.nn.Parameter(torch.ones(2))
    dist_param.grad = torch.full_like(dist_param, 4)

    reshard_calls = []
    module = SimpleNamespace(
        _fsdp_param_groups=[SimpleNamespace(params=[full_param], dist_params=[dist_param])],
        reshard=lambda: reshard_calls.append(True),
    )

    _make_bwd_post_hook(module)(module, (), ())

    assert full_param.grad is None
    assert torch.equal(full_param.main_grad, torch.full_like(full_param, 3))
    assert torch.equal(dist_param.grad, torch.full_like(dist_param, 4))
    assert reshard_calls == [True]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_reduce_grad_skips_aliased_main_grad_copy():
    """Skip copying a parameter gradient that already aliases main grad."""
    param = torch.nn.Parameter(torch.ones(4, device="cuda"))
    main_grad = torch.zeros_like(param)
    param.grad = main_grad
    param.get_main_grad = lambda: main_grad

    reduce_calls = []
    release_calls = []
    dist_param = SimpleNamespace(dtype=param.dtype, grad=None)
    param_group = SimpleNamespace(
        requires_grad=True,
        sharding_strategy="optim_grads_params",
        enable_full_iteration_cuda_graph=False,
        main_grad_buffer=SimpleNamespace(inner_sharded=True),
        _full_grad_buffer_has_accumulated_grad=False,
        params=(param,),
        dist_params=(dist_param,),
        dist_grads=(None,),
        mp_policy=SimpleNamespace(use_decoupled_grad=False),
        _init_dist_grads=lambda: None,
        reduce_grad=lambda **_: reduce_calls.append(True),
        release_grad_buffer=lambda: release_calls.append(True),
    )
    module = SimpleNamespace(
        _fsdp_root_context=SimpleNamespace(rs_stream=None, is_last_backward=True),
        _fsdp_state=SimpleNamespace(enable_cuda_graph=True),
        _named_param_groups=[(("weight",), param_group)],
        _wait_for_previous_async_reduce_grad=lambda: None,
    )

    with patch.object(torch, "_foreach_copy_") as copy_mock:
        FSDPModule.reduce_grad(module)

    copy_mock.assert_not_called()
    assert param.grad is None
    assert reduce_calls == [True]
    assert release_calls == [True]


def test_module_compile_is_converted_to_compiled_forward_for_capture():
    module = torch.nn.Linear(2, 2)
    original_forward = module.forward
    compiled_call_impl = object()
    compiled_forward = object()
    module._compiled_call_impl = compiled_call_impl

    with patch.object(torch, "compile", return_value=compiled_forward) as compile_mock:
        saved = _prepare_compiled_modules_for_capture([module])

    compile_mock.assert_called_once_with(
        original_forward, dynamic=False, options={"triton.cudagraphs": False}
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


def test_parameter_surface_refresh_uses_current_registered_parameters():
    """Use parameters installed by a capture-time replacement hook."""
    module = torch.nn.Linear(2, 2)
    replacement_weight = torch.nn.Parameter(torch.full_like(module.weight, 2))
    replacement_bias = torch.nn.Parameter(torch.full_like(module.bias, 3))
    module.weight = replacement_weight
    module.bias = replacement_bias
    user_inputs = (torch.ones(1, 2),)

    module_params, input_surface = _refresh_module_parameter_surface(
        module, user_inputs, parameter_indices=(0,)
    )

    assert len(module_params) == 1
    assert module_params[0] is replacement_weight
    assert input_surface[0] is user_inputs[0]
    assert input_surface[1] is replacement_weight


def test_static_grad_context_uses_main_grad_and_restores_leaf_state():
    """Bind a main-grad buffer during capture and restore the original leaf state."""
    param = torch.nn.Parameter(torch.ones(4))
    original_grad = torch.full_like(param, 5)
    main_grad = torch.full_like(param, 7)
    param.grad = original_grad
    param.get_main_grad = lambda: main_grad

    grad_buffers = _get_static_grad_buffers((param,))
    with _static_grad_context_wrapper((param,), grad_buffers):
        assert param.grad is main_grad
        assert torch.count_nonzero(main_grad) == 0

    assert param.grad is original_grad


def test_static_grad_buffer_rejects_incompatible_gradient_contract():
    """Keep incompatible main grads on the normal autograd-owned path."""
    param = torch.nn.Parameter(torch.ones(2, 2, dtype=torch.bfloat16))

    param.get_main_grad = lambda: torch.zeros(2, 2, dtype=torch.float32)
    assert _get_compatible_main_grad_buffer(param) is None

    param.get_main_grad = lambda: torch.zeros(4, dtype=torch.bfloat16)
    assert _get_compatible_main_grad_buffer(param) is None

    transposed_main_grad = torch.zeros(2, 2, dtype=torch.bfloat16).t()
    param.get_main_grad = lambda: transposed_main_grad
    assert _get_compatible_main_grad_buffer(param) is None


def test_static_grad_buffer_skips_mfsdp_accumulation_strategy():
    """Keep accumulated M-FSDP gradients on the autograd-owned path."""
    param = torch.nn.Parameter(torch.ones(4))
    main_grad = torch.zeros_like(param)
    getter_calls = []
    param.__fsdp_param__ = True
    param.overwrite_main_grad = False
    param.get_main_grad = lambda: getter_calls.append(True) or main_grad

    assert _get_compatible_main_grad_buffer(param) is None
    assert getter_calls == []


def test_static_grad_buffer_skips_recorded_te_fused_wgrad():
    """Keep TE dummy wgrad separate from its fused main-grad destination."""
    param = torch.nn.Parameter(torch.ones(4))
    main_grad = torch.zeros_like(param)
    getter_calls = []
    param._mfsdp_recorded_te_wgrad = True
    param.get_main_grad = lambda: getter_calls.append(True) or main_grad

    assert _get_compatible_main_grad_buffer(param) is None
    assert getter_calls == []


def test_static_grad_buffer_checks_mfsdp_dtype_before_fetch():
    """Reject mixed-dtype M-FSDP main grad before materializing its buffer."""
    param = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    main_grad = torch.zeros(4, dtype=torch.float32)
    getter_calls = []
    param.__fsdp_param__ = True
    param.overwrite_main_grad = True
    param._gbuf = SimpleNamespace(dtype=torch.float32)
    param.get_main_grad = lambda: getter_calls.append(True) or main_grad

    assert _get_compatible_main_grad_buffer(param) is None
    assert getter_calls == []


def test_capture_backward_pre_hook_prefetches_only_te_fused_wgrad():
    """Materialize capture buffers only for recorded TE fused-wgrad groups."""
    fused_param = torch.nn.Parameter(torch.ones(4))
    fused_param._mfsdp_recorded_te_wgrad = True
    regular_param = torch.nn.Parameter(torch.ones(4))
    fused_init_calls = []
    fused_fetch_calls = []
    regular_init_calls = []
    regular_fetch_calls = []
    fused_group = SimpleNamespace(
        params=(fused_param,),
        main_grad_buffer=SimpleNamespace(fetch_buffer=lambda: fused_fetch_calls.append(True)),
        _init_dist_grads=lambda: fused_init_calls.append(True),
    )
    regular_group = SimpleNamespace(
        params=(regular_param,),
        main_grad_buffer=SimpleNamespace(fetch_buffer=lambda: regular_fetch_calls.append(True)),
        _init_dist_grads=lambda: regular_init_calls.append(True),
    )
    unshard_calls = []
    module = SimpleNamespace(
        _fsdp_param_groups=(fused_group, regular_group),
        unshard=lambda **kwargs: unshard_calls.append(kwargs),
    )

    _make_bwd_pre_hook(module)(module, ())

    assert unshard_calls == [{"bwd_pass": True}]
    assert fused_init_calls == [True]
    assert fused_fetch_calls == [True]
    assert regular_init_calls == []
    assert regular_fetch_calls == []


def test_trace_prefetches_static_main_grad_before_backward():
    """Trace the main-grad lifetime used by CUDA graph replay."""
    param = torch.nn.Parameter(torch.ones(4))
    init_calls = []
    fetch_calls = []
    unshard_calls = []
    param_group = SimpleNamespace(
        params=(param,),
        requires_grad=True,
        sharding_strategy="optim_grads_params",
        main_grad_buffer=SimpleNamespace(
            dtype=param.dtype, fetch_buffer=lambda: fetch_calls.append(True)
        ),
        _init_dist_grads=lambda: init_calls.append(True),
    )
    module = SimpleNamespace(
        _fsdp_root_context=SimpleNamespace(cuda_graph_active=False, enable_unshard_prefetch=False),
        _fsdp_state=SimpleNamespace(
            _is_root=False, enable_cuda_graph=True, enable_full_iteration_cuda_graph=False
        ),
        _fsdp_param_groups=(param_group,),
        unshard=lambda **kwargs: unshard_calls.append(kwargs),
    )

    _pre_backward_setup(module)

    assert unshard_calls == [{"async_op": False, "bwd_pass": True}]
    assert param.overwrite_main_grad
    assert init_calls == [True]
    assert fetch_calls == [True]


def test_static_grad_context_accumulates_into_main_grad_buffer():
    """Accumulate an autograd result directly into a compatible main-grad buffer."""
    param = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    main_grad = torch.zeros_like(param)
    param.get_main_grad = lambda: main_grad

    grad_buffers = _get_static_grad_buffers((param,))
    with _static_grad_context_wrapper((param,), grad_buffers):
        param.square().sum().backward()
        assert param.grad is main_grad
        torch.testing.assert_close(main_grad, torch.tensor([2.0, 4.0]))

    assert param.grad is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_cuda_graph_replay_preserves_mfsdp_microbatch_accumulation():
    """Accumulate two M-FSDP microbatches without static main-grad binding."""
    module = torch.nn.Linear(4, 3, bias=False, device="cuda")
    main_grad = torch.zeros_like(module.weight)
    module.weight.__fsdp_param__ = True
    module.weight.overwrite_main_grad = False
    module.weight.get_main_grad = lambda: main_grad
    sample = torch.ones(2, 4, device="cuda")

    graphed = make_graphed_callables(
        module, (), sample_kwargs={"input": sample}, num_warmup_iters=1
    )

    for value in (2.0, 3.0):
        graphed(input=torch.full_like(sample, value)).sum().backward()
    torch.cuda.synchronize()

    torch.testing.assert_close(module.weight.grad, torch.full_like(module.weight, 10.0))
    torch.testing.assert_close(main_grad, torch.zeros_like(main_grad))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_cuda_graph_replay_keeps_mixed_dtype_main_grad_lazy():
    """Return BF16 grads without fetching an incompatible FP32 main-grad buffer."""
    module = torch.nn.Linear(4, 3, bias=False, device="cuda", dtype=torch.bfloat16)
    main_grad = torch.zeros_like(module.weight, dtype=torch.float32)
    getter_calls = []
    module.weight.__fsdp_param__ = True
    module.weight.overwrite_main_grad = True
    module.weight._gbuf = SimpleNamespace(dtype=torch.float32)
    module.weight.get_main_grad = lambda: getter_calls.append(True) or main_grad
    sample = torch.ones(2, 4, device="cuda", dtype=torch.bfloat16)

    graphed = make_graphed_callables(
        module, (), sample_kwargs={"input": sample}, num_warmup_iters=1
    )
    graphed(input=torch.full_like(sample, 2.0)).sum().backward()
    torch.cuda.synchronize()

    assert getter_calls == []
    assert module.weight.grad.dtype == torch.bfloat16
    torch.testing.assert_close(module.weight.grad, torch.full_like(module.weight, 4.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_cuda_graph_links_adjacent_static_surfaces():
    """Reuse a producer output as its consumer's static input."""
    first = torch.nn.Linear(4, 4, bias=False, device="cuda")
    second = torch.nn.Linear(4, 4, bias=False, device="cuda")
    with torch.no_grad():
        first.weight.copy_(torch.eye(4, device="cuda") * 2)
        second.weight.copy_(torch.eye(4, device="cuda") * 3)

    second_forward = second.forward

    def record_input(input_tensor):
        """Record the consumer input address during capture.

        :param input_tensor: Consumer input.
        :type input_tensor: torch.Tensor
        :return: Consumer output.
        :rtype: torch.Tensor
        """
        second.capture_input_ptr = input_tensor.data_ptr()
        return second_forward(input_tensor)

    second.forward = record_input
    samples = (
        torch.ones(2, 4, device="cuda", requires_grad=True),
        torch.ones(2, 4, device="cuda", requires_grad=True),
    )
    first_graph, second_graph = make_graphed_callables(
        (first, second),
        ((), ()),
        sample_kwargs=({"input": samples[0]}, {"input_tensor": samples[1]}),
        num_warmup_iters=1,
        _input_output_aliases=({}, {0: (0, 0)}),
    )

    runtime_input = torch.full_like(samples[0], 5, requires_grad=True)
    first_output = first_graph(input=runtime_input)
    assert first_output.data_ptr() == second.capture_input_ptr
    second_graph(input_tensor=first_output).sum().backward()
    torch.cuda.synchronize()

    torch.testing.assert_close(runtime_input.grad, torch.full_like(runtime_input, 6))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
@pytest.mark.parametrize(
    ("overwrite_main_grad", "values", "expected"),
    [
        pytest.param(True, (2.0,), 4.0, id="overwrite"),
        # pytest.param(False, (2.0, 3.0), 10.0, id="accumulate"),
    ],
)
def test_cuda_graph_replay_te_fused_wgrad_main_grad(overwrite_main_grad, values, expected):
    """Write TE fused wgrad with the M-FSDP microbatch policy.
    :param overwrite_main_grad: Whether each microbatch replaces main grad.
    :type overwrite_main_grad: bool
    :param values: Runtime input values for consecutive microbatches.
    :type values: Tuple[float, ...]
    :param expected: Expected final value in every main-grad element.
    :type expected: float
    """
    module = torch.nn.Linear(4, 3, bias=False, device="cuda", dtype=torch.bfloat16)
    main_grad = torch.zeros_like(module.weight, dtype=torch.float32)
    module.weight.__fsdp_param__ = True
    module.weight.grad_added_to_main_grad = False
    module.weight.overwrite_main_grad = overwrite_main_grad
    module.weight._mfsdp_recorded_te_wgrad = True
    module.weight.get_main_grad = lambda: main_grad
    module.forward = partial(
        linear_with_grad_accumulation_and_async_allreduce,
        weight=module.weight,
        bias=None,
        gradient_accumulation_fusion=True,
        allreduce_dgrad=False,
        sequence_parallel=False,
        grad_output_buffer=None,
        wgrad_deferral_limit=0,
        tp_group=None,
    )
    sample = torch.ones(2, 4, device="cuda", dtype=torch.bfloat16)
    graphed = make_graphed_callables(
        module, (), sample_kwargs={"input": sample}, num_warmup_iters=1
    )

    main_grad.zero_()
    module.weight.grad = None
    for value in values:
        graphed(input=torch.full_like(sample, value)).sum().backward()
    torch.cuda.synchronize()

    torch.testing.assert_close(main_grad, torch.full_like(main_grad, expected))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_cuda_graph_replay_restores_leaf_grad_and_reuses_main_grad(dtype):
    """Replay once per input without accumulating the main grad twice.

    :param dtype: Parameter and main-gradient dtype under test.
    :type dtype: torch.dtype
    """
    module = torch.nn.Linear(4, 3, bias=False, device="cuda", dtype=dtype)
    main_grad = torch.zeros_like(module.weight)
    module.weight.get_main_grad = lambda: main_grad
    sample = torch.ones(2, 4, device="cuda", dtype=dtype)

    graphed = make_graphed_callables(
        module, (), sample_kwargs={"input": sample}, num_warmup_iters=1
    )

    assert module.weight.grad is None
    for value in (2.0, 3.0):
        runtime_input = torch.full_like(sample, value)
        graphed(input=runtime_input).sum().backward()
        torch.cuda.synchronize()

        expected = torch.full_like(main_grad, 2.0 * value)
        torch.testing.assert_close(main_grad, expected)
        assert module.weight.grad is not None
        assert module.weight.grad.data_ptr() == main_grad.data_ptr()
        module.weight.grad = None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_cuda_graph_capture_refreshes_parameter_after_backward_pre_hook():
    """Capture the parameter installed by a backward pre-hook."""
    module = torch.nn.Linear(4, 3, bias=False, device="cuda")
    sharded_weight = module.weight
    compute_weight = torch.nn.Parameter(torch.full_like(sharded_weight, 2))
    sample = torch.ones(2, 4, device="cuda", requires_grad=True)

    def install_compute_weight(_module, *_args):
        """Install the unsharded compute parameter."""
        module.weight = compute_weight

    def install_sharded_weight(_module, *_args):
        """Restore the optimizer-facing parameter."""
        module.weight = sharded_weight

    capture_hooks = {
        "forward_pre_hooks": {0: install_compute_weight},
        "forward_pre_hooks_with_kwargs": {0: True},
        "forward_hooks": {0: install_sharded_weight},
        "forward_hooks_with_kwargs": {0: True},
        "backward_pre_hooks": {0: install_compute_weight},
        "backward_hooks": {0: install_sharded_weight},
    }
    graphed = make_graphed_callables(
        module,
        (),
        sample_kwargs={"input": sample},
        num_warmup_iters=1,
        capture_time_hooks=[capture_hooks],
    )

    assert module.weight is sharded_weight
    runtime_input = torch.full_like(sample, 3, requires_grad=True)
    graphed(input=runtime_input).sum().backward()
    torch.cuda.synchronize()

    torch.testing.assert_close(compute_weight.grad, torch.full_like(compute_weight, 6))
    assert sharded_weight.grad is None
