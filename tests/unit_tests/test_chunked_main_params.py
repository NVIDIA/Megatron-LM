# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Regression tests for the master-plus-state staging window.

Load the transfer manager in isolation so planning and scheduling tests also run
on CPU-only development hosts without importing optional training extensions.
"""

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch


@pytest.fixture
def module():
    path = Path(__file__).resolve().parents[2] / 'megatron/core/optimizer/cpu_offloading/chunked_optimizer_state_offload.py'
    spec = importlib.util.spec_from_file_location('_chunked_main_params_test', path)
    loaded = importlib.util.module_from_spec(spec)
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and node.module == 'megatron.core.utils'
    )]
    loaded.log_single_rank = lambda *args: None
    previous = sys.modules.get(spec.name)
    sys.modules[spec.name] = loaded
    try:
        exec(compile(tree, str(path), 'exec'), loaded.__dict__)
        yield loaded
    finally:
        if previous is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous


def make_manager(module, params, limit, fraction=1.0, callback=lambda params: None):
    optimizer = torch.optim.SGD(params, lr=.01, momentum=.9)
    with mock.patch.object(torch.cuda, 'Stream'):
        manager = module.ChunkedOptimizerStateOffloader(
            optimizer, params, limit, fraction, (torch.float32,),
            chunk_master_params=True, copy_updated_params=callback,
        )
    return manager


@pytest.mark.parametrize('chunked', [False, True])
def test_planner_counts_master_bytes(module, chunked):
    params = [torch.nn.Parameter(torch.ones(8, 8)) for _ in range(4)]
    manager = make_manager(module, params, 512)
    manager.chunk_master_params = chunked
    chunks = manager._build_chunks(params)
    assert [len(chunk.params) for chunk in chunks] == ([1] * 4 if chunked else [2, 2])


@pytest.mark.parametrize('fraction', [0.5, 1.0])
def test_step_copies_each_chunk_before_offload(module, fraction):
    params = [torch.nn.Parameter(torch.ones(8, 8)) for _ in range(4)]
    copied = {}
    trace = []

    def copy(subset):
        for p in subset:
            copied[p] = p.detach().clone()
            trace.append(('copy', id(p)))

    manager = make_manager(module, params, 512, fraction, copy)
    for p in params:
        grad = torch.ones_like(p)
        proxy = grad.detach()
        proxy.grad = grad
        manager.grad_proxies[p] = proxy
        if not manager.is_param_offloaded(p):
            p.grad = proxy
    manager._schedule_master_h2d = mock.Mock(side_effect=AssertionError('full restore'))
    manager._wait_master_h2d = lambda: None
    manager._prefetch_state = lambda chunk: module._StatePrefetch(None)
    manager._wait_state_prefetch = lambda token: None
    manager._has_unregistered_cuda_state = lambda params: False
    manager._schedule_state_d2h = lambda params: None

    def offload(subset=None):
        if subset is not None:
            for p in subset:
                assert trace[-1] == ('copy', id(p))
                trace.append(('offload', id(p)))
                p.grad = None

    manager._schedule_master_d2h = offload
    for _ in range(3):
        manager.step()
    for p in params:
        torch.testing.assert_close(copied[p], torch.full_like(p, .9439))
    assert all(p.grad is None for p in manager.selected_params)
    assert sum(len(group['params']) for group in manager.optimizer.param_groups) == 4


def test_prefetch_does_not_restore_full_master_window(module):
    params = [torch.nn.Parameter(torch.ones(8, 8)) for _ in range(4)]
    manager = make_manager(module, params, 512)
    manager._schedule_master_h2d = mock.Mock(side_effect=AssertionError('full restore'))
    manager._schedule_master_d2h = mock.Mock()
    manager._prefetch_state = mock.Mock(return_value=module._StatePrefetch(None))
    manager.prefetch_master_for_step()
    manager.prefetch_for_step()
    manager.prefetch_for_step()
    manager._prefetch_state.assert_called_once_with(manager.chunks[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required for DMA and residency')
@pytest.mark.parametrize('fraction', [.5, 1.0])
@pytest.mark.parametrize('num_buffers', [2, 3])
def test_cuda_repeated_steps_and_master_residency(module, fraction, num_buffers):
    params = [torch.nn.Parameter(torch.randn(64, 64, device='cuda')) for _ in range(6)]
    reference = [torch.nn.Parameter(p.detach().clone()) for p in params]
    ref_opt = torch.optim.SGD(reference, lr=.01, momentum=.9)
    models = {p: p.detach().to(torch.bfloat16) for p in params}

    def copy(subset):
        for p in subset:
            models[p].copy_(p)

    manager = make_manager(module, params, 2 * 64 * 64 * 4, fraction, copy)
    manager.use_staging_pool(module._StagingSlotPool(num_buffers))
    manager.offload_for_forward()
    original_step_subset = manager._step_subset

    def bounded_step(subset, *args):
        # Each target fits one complete master+momentum bundle. A current and a
        # prefetched bundle may be CUDA-bound; the remaining selected masters must not be.
        assert sum(p.is_cuda for p in manager.selected_params) <= 2
        return original_step_subset(subset, *args)

    manager._step_subset = bounded_step
    for _ in range(4):
        for p, ref in zip(params, reference):
            grad = torch.randn_like(ref)
            ref.grad = grad.clone()
            proxy = grad.detach()
            proxy.grad = grad
            manager.grad_proxies[p] = proxy
            if not manager.is_param_offloaded(p):
                p.grad = grad
        manager.prefetch_master_for_step()
        manager.step()
        ref_opt.step()
        manager.synchronize_for_checkpoint()
        for p, ref in zip(params, reference):
            torch.testing.assert_close(p.to('cuda'), ref)
            torch.testing.assert_close(models[p], ref.to(torch.bfloat16))
        assert all(p.device.type == 'cpu' for p in manager.selected_params)
        assert not manager.staging_pool.in_use


def test_gradient_proxies_preserve_gradients_without_master_binding(module):
    path = Path(__file__).resolve().parents[2] / 'megatron/core/optimizer/optimizer.py'
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == 'Float16OptimizerWithFloat16Params')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == '_copy_model_grads_to_main_grads')
    namespace = {
        'torch': torch,
        'tensor_parallel': SimpleNamespace(
            copy_tensor_model_parallel_attributes=lambda *args: None,
            copy_gtp_attributes=lambda *args: None,
        ),
        'copy_optimizer_param_metadata': lambda *args: None,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    masters = [torch.nn.Parameter(torch.ones(8, 8)) for _ in range(2)]
    models = [torch.nn.Parameter(torch.ones(8, 8, dtype=torch.bfloat16)) for _ in masters]
    models[0].main_grad = torch.ones(8, 8)
    manager = make_manager(module, masters, 512)
    wrapper = SimpleNamespace(
        _optimizer_state_offloader=manager, float16_groups=[models],
        fp32_from_float16_groups=[masters], fp32_from_fp32_groups=[],
    )
    namespace[method.name](wrapper)
    proxy = manager.grad_proxies[masters[0]]
    torch.testing.assert_close(proxy.grad, models[0].main_grad)
    assert proxy is not proxy.grad
    assert all(p.grad is None for p in masters)
    assert manager.grad_proxies[masters[1]].grad is None
    proxy.grad.mul_(.5)
    torch.testing.assert_close(models[0].main_grad, torch.full((8, 8), .5))


def test_checkpoint_initialization_restores_and_releases_each_chunk(module):
    params = [torch.nn.Parameter(torch.ones(8, 8)) for _ in range(4)]
    manager = make_manager(module, params, 512)
    trace = []
    manager._schedule_master_h2d = mock.Mock(side_effect=AssertionError('full restore'))
    manager._wait_master_h2d = lambda: None
    manager._schedule_master_d2h = lambda subset=None: trace.append(('release', subset))
    manager._prefetch_state = lambda chunk: trace.append(('restore', chunk.params))
    manager._wait_state_prefetch = lambda token: None
    manager._schedule_state_d2h = lambda params: None
    manager.adopt_cpu_optimizer_state = lambda params=None: None

    def init(opt, config):
        trace.append(('init', tuple(p for group in opt.param_groups for p in group['params'])))

    manager.initialize_state_for_loading(init, None)
    for chunk in manager.chunks:
        index = next(i for i, (op, subset) in enumerate(trace)
                     if op == 'restore' and subset is chunk.params)
        assert trace[index + 1][0] == 'init'
        assert trace[index + 2] == ('release', chunk.params)
    assert len(manager.optimizer.param_groups[0]['params']) == 4


@pytest.mark.parametrize('fraction', [0.0, .25, .5, 1.0])
def test_cpu_init_selection_matches_runtime(module, fraction):
    models = [torch.nn.Parameter(torch.ones(shape, dtype=torch.bfloat16))
              for shape in [(8, 8), (3, 8), (8, 4), (2, 2)]]
    raw = torch.optim.SGD([{'params': models[:2]}, {'params': models[2:]}], lr=.1)
    chosen = module.ChunkedOptimizerStateOffloader.cpu_initial_master_param_ids(raw, fraction)
    masters = [torch.nn.Parameter(p.float()) for p in models]
    opt = torch.optim.SGD([{'params': masters[:2]}, {'params': masters[2:]}], lr=.1)
    manager = module.ChunkedOptimizerStateOffloader(
        opt, masters, 512, fraction, (torch.float32,),
        d2h_stream=object(), h2d_stream=object(),
    )
    selected = {id(p) for p in manager.selected_params}
    assert [id(p) in chosen for p in models] == [id(p) in selected for p in masters]


def test_cpu_master_retains_target_cuda_device(module):
    master = torch.nn.Parameter(torch.ones(4, 4))
    master._offload_original_device = torch.device('cuda', 2)
    manager = module.ChunkedOptimizerStateOffloader(
        torch.optim.SGD([master], lr=.1), [master], 512, 1., (torch.float32,),
        d2h_stream=object(), h2d_stream=object(),
    )
    assert manager._param_devices[master] == torch.device('cuda', 2)
    assert not manager._master_weights_resident


def _cpu_init_branch():
    path = Path(__file__).resolve().parents[2] / 'megatron/core/optimizer/optimizer.py'
    tree = ast.parse(path.read_text())
    branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                  and 'cpu_main_param_ids is not None' in ast.unparse(node.test))
    return compile(ast.Module(body=branch.body, type_ignores=[]), str(path), 'exec')


def test_cpu_init_does_not_clone_or_convert_on_gpu():
    source = torch.nn.Parameter(torch.arange(32, dtype=torch.bfloat16).reshape(4, 8))
    namespace = dict(torch=torch, param=source, high_precision_init_val=None)
    empty_like = torch.empty_like

    def allocate(tensor, **kwargs):
        assert kwargs.pop('pin_memory')
        assert kwargs['device'] == 'cpu' and kwargs['dtype'] == torch.float32
        return empty_like(tensor, **kwargs)

    with mock.patch.object(torch, 'empty_like', side_effect=allocate):
        with mock.patch.object(torch.Tensor, 'clone', side_effect=AssertionError('CUDA clone path')):
            exec(_cpu_init_branch(), namespace)
    result = namespace['main_param']
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, source.float())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required for pinned CPU initialization')
def test_cuda_master_initialization_has_no_full_fp32_gpu_copy():
    source = torch.nn.Parameter(torch.randn(2048, 2048, device='cuda', dtype=torch.bfloat16))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    namespace = dict(torch=torch, param=source, high_precision_init_val=None)
    exec(_cpu_init_branch(), namespace)
    torch.cuda.synchronize()
    result = namespace['main_param']
    assert result.device.type == 'cpu' and result.is_pinned()
    assert torch.cuda.max_memory_allocated() - before < source.numel() * 4
    torch.testing.assert_close(result, source.detach().cpu().float(), rtol=0, atol=0)


@pytest.mark.parametrize('count', [2, 3])
def test_staging_pool_size(module, count):
    pool = module._StagingSlotPool(count)
    assert len(pool.slots) == count
    with pytest.raises(ValueError):
        module._StagingSlotPool(4)


def test_three_buffers_wait_only_on_slot_and_cpu_sources(module):
    params = [torch.nn.Parameter(torch.ones(4, 4)) for _ in range(2)]
    manager = make_manager(module, params, 256)
    pool = module._StagingSlotPool(3)
    manager.use_staging_pool(pool)
    slot = pool.slots[0]
    slot.allocation_ready = object()
    slot.d2h_done = object()
    own_source, unrelated_source = object(), object()
    manager._cpu_source_ready[params[0]] = own_source
    manager._cpu_source_ready[params[1]] = unrelated_source
    manager._h2d_stream = mock.Mock()
    manager._order_chunk_h2d(module.OptimizerStateChunk((params[0],)), slot)
    waited = [call.args[0] for call in manager._h2d_stream.wait_event.call_args_list]
    assert waited == [slot.allocation_ready, slot.d2h_done, own_source]
    assert unrelated_source not in waited
    manager._h2d_stream.wait_stream.assert_not_called()


def test_two_buffers_keep_original_stream_dependencies(module):
    params = [torch.nn.Parameter(torch.ones(4, 4))]
    manager = make_manager(module, params, 256)
    manager._order_h2d_after_source_streams = mock.Mock()
    manager._order_chunk_h2d(manager.chunks[0], manager.staging_pool.slots[0])
    manager._order_h2d_after_source_streams.assert_called_once()


def test_d2h_completion_protects_shared_slot_and_cpu_buffer(module):
    params = [torch.nn.Parameter(torch.ones(4, 4)) for _ in range(2)]
    manager = make_manager(module, params, 256)
    pool = module._StagingSlotPool(3)
    manager.use_staging_pool(pool)
    manager._param_staging_slots[params[0]] = pool.slots[0]
    first, second = mock.Mock(), mock.Mock()
    with mock.patch.object(torch.cuda, 'Event', side_effect=[first, second]):
        manager._record_d2h_completion([params[0]])
        assert manager._cpu_source_ready[params[0]] is first
        manager._record_d2h_completion([params[0]])
    assert pool.slots[0].d2h_done is second
    assert manager._cpu_source_ready[params[0]] is second
    assert params[1] not in manager._cpu_source_ready
    first.record.assert_called_once_with(manager._d2h_stream)
    second.record.assert_called_once_with(manager._d2h_stream)


def test_pool_release_drops_slot_events(module):
    pool = module._StagingSlotPool(3)
    slot = pool.slots[0]
    slot.buffers['dummy'] = object()
    slot.allocation_ready = object()
    slot.d2h_done = object()
    with pool.hold():
        pool.release()
        assert slot.d2h_done is not None
    assert not pool.in_use
    assert slot.d2h_done is None and slot.allocation_ready is None
