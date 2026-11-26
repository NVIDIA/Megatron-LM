# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""DSA's deferred CP latent K/V gather equals the synchronous gather and reorder bit for bit.

``_DeferredCpKvGather`` launches the CP all-gathers of the latent key (and value) after the
indexer and waits only where the gathered tensors are first read. Its forward results and its
reduce-scatter backward must equal ``gather_from_sequence_parallel_region`` followed by the row
reorder, including when another collective on the same CP group runs between launch and finish.

Run with: torchrun --nproc-per-node 4 -m pytest -q <this file>
"""

import os

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.experimental_attention_variant.dsa import (
    _DeferredCpKvGather,
    _permute_rows,
)
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

_ROWS, _WIDTH = 24, 72


def _inputs(cp_rank, with_value):
    generator = torch.Generator(device="cuda").manual_seed(101 + cp_rank)
    key = torch.randn(_ROWS, 1, 1, _WIDTH, generator=generator, device="cuda", dtype=torch.bfloat16)
    value = (
        torch.randn(_ROWS, 1, 1, _WIDTH, generator=generator, device="cuda", dtype=torch.bfloat16)
        if with_value
        else None
    )
    return key, value


def _reorder(cp_size):
    # The same permutation on every rank, as the DSA layout builders produce.
    perm = torch.randperm(_ROWS * cp_size, generator=torch.Generator().manual_seed(7)).cuda()
    return perm, torch.argsort(perm)


def _leaf(tensor):
    return None if tensor is None else tensor.detach().clone().requires_grad_(True)


@pytest.mark.parametrize("with_value", [False, True])
@pytest.mark.parametrize("cp_size", [2, 4])
def test_deferred_cp_kv_gather_matches_synchronous(cp_size, with_value):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world < cp_size:
        pytest.skip(f"needs {cp_size} ranks")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=cp_size)
    try:
        cp_group = parallel_state.get_context_parallel_group()
        cp_rank = cp_group.rank()
        key, value = _inputs(cp_rank, with_value)
        perm, inv_perm = _reorder(cp_size)
        generator = torch.Generator(device="cuda").manual_seed(313 + cp_rank)
        grad_key = torch.randn(
            _ROWS * cp_size, 1, 1, _WIDTH, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        grad_value = torch.randn_like(grad_key) if with_value else None

        # Reference: the synchronous gather, then the reorder.
        key_ref, value_ref = _leaf(key), _leaf(value)
        out_key_ref = _permute_rows(
            gather_from_sequence_parallel_region(key_ref, group=cp_group), perm, inv_perm
        )
        outputs, grads = [out_key_ref], [grad_key]
        if with_value:
            out_value_ref = _permute_rows(
                gather_from_sequence_parallel_region(value_ref, group=cp_group), perm, inv_perm
            )
            outputs.append(out_value_ref)
            grads.append(grad_value)
        torch.autograd.backward(outputs, grads)

        # Deferred: launch, run another collective on the CP group, read the shape twin, finish.
        key_def, value_def = _leaf(key), _leaf(value)
        deferred = _DeferredCpKvGather(
            key_def, value_def, cp_group=cp_group, reorder_idx=perm, reorder_inv_idx=inv_perm
        )
        deferred.launch()
        deferred.launch()  # a second launch is a no-op
        between = torch.full((4,), float(cp_rank), device="cuda")
        torch.distributed.all_reduce(between, group=cp_group)
        twin = deferred.key_shape_twin(key_def)
        assert twin.shape == out_key_ref.shape and twin.dtype == out_key_ref.dtype
        out_key, out_value = deferred.finish(key_def, value_def)
        assert torch.equal(between, torch.full_like(between, sum(range(cp_size))))
        outputs, grads = [out_key], [grad_key]
        if with_value:
            outputs.append(out_value)
            grads.append(grad_value)
        else:
            assert out_value is None
        torch.autograd.backward(outputs, grads)

        assert torch.equal(out_key, out_key_ref)
        assert torch.equal(key_def.grad, key_ref.grad)
        if with_value:
            assert torch.equal(out_value, out_value_ref)
            assert torch.equal(value_def.grad, value_ref.grad)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("cp_size", [2, 4])
def test_deferred_cp_kv_gather_value_only(cp_size):
    """A key that is already global stays untouched; only the value is gathered."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world < cp_size:
        pytest.skip(f"needs {cp_size} ranks")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=cp_size)
    try:
        cp_group = parallel_state.get_context_parallel_group()
        key, value = _inputs(cp_group.rank(), with_value=True)
        global_key = key.repeat(cp_size, 1, 1, 1)
        perm, inv_perm = _reorder(cp_size)
        deferred = _DeferredCpKvGather(
            None, value, cp_group=cp_group, reorder_idx=perm, reorder_inv_idx=inv_perm
        )
        deferred.launch()
        assert deferred.key_shape_twin(global_key) is global_key
        out_key, out_value = deferred.finish(global_key, value)
        assert out_key is global_key
        expected = gather_from_sequence_parallel_region(value, group=cp_group).index_select(0, perm)
        assert torch.equal(out_value, expected)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("cp_size", [2, 4])
def test_deferred_cp_kv_gather_inside_cuda_graph(cp_size):
    """Captured in a CUDA graph (as full-iteration graphs do), the deferred gather and its
    reduce-scatter backward replay bit for bit like the synchronous gather, with another CP
    collective between launch and finish."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world < cp_size:
        pytest.skip(f"needs {cp_size} ranks")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=cp_size)
    try:
        cp_group = parallel_state.get_context_parallel_group()
        cp_rank = cp_group.rank()
        perm, inv_perm = _reorder(cp_size)
        static_key = torch.zeros(_ROWS, 1, 1, _WIDTH, device="cuda", dtype=torch.bfloat16)
        static_grad = torch.zeros(
            _ROWS * cp_size, 1, 1, _WIDTH, device="cuda", dtype=torch.bfloat16
        )
        static_between = torch.zeros(4, device="cuda")

        def step():
            key = static_key.detach().requires_grad_(True)
            deferred = _DeferredCpKvGather(
                key, None, cp_group=cp_group, reorder_idx=perm, reorder_inv_idx=inv_perm
            )
            deferred.launch()
            between = static_between.clone()
            torch.distributed.all_reduce(between, group=cp_group)
            out_key, _ = deferred.finish(key, None)
            (grad_key,) = torch.autograd.grad(out_key, key, static_grad)
            return out_key, grad_key, between

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):  # warm up NCCL and the allocator outside the capture
                step()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            static_out, static_grad_key, static_between_out = step()

        for replay in range(3):
            generator = torch.Generator(device="cuda").manual_seed(1000 * replay + cp_rank)
            key = torch.randn(
                _ROWS, 1, 1, _WIDTH, generator=generator, device="cuda", dtype=torch.bfloat16
            )
            grad = torch.randn(
                _ROWS * cp_size,
                1,
                1,
                _WIDTH,
                generator=generator,
                device="cuda",
                dtype=torch.bfloat16,
            )
            static_key.copy_(key)
            static_grad.copy_(grad)
            static_between.fill_(float(cp_rank + replay))
            graph.replay()
            torch.cuda.synchronize()

            key_ref = _leaf(key)
            out_ref = _permute_rows(
                gather_from_sequence_parallel_region(key_ref, group=cp_group), perm, inv_perm
            )
            out_ref.backward(grad)
            assert torch.equal(static_out, out_ref), replay
            assert torch.equal(static_grad_key, key_ref.grad), replay
            expected = float(sum(range(cp_size)) + cp_size * replay)
            assert torch.equal(static_between_out, torch.full_like(static_between_out, expected))
    finally:
        Utils.destroy_model_parallel()
