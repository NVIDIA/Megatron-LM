# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import gc
import inspect
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core import config
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    init_num_microbatches_calculator,
)
from megatron.core.tensor_parallel.random import (
    _mark_checkpoint_without_output_tensor,
    is_checkpoint_without_output_tensor,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.moe.fused_a2a import is_nccl_ep_bootstrapped, reset_hybrid_ep_buffer
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.moe_utils import get_align_size_for_quantization
from megatron.core.transformer.moe.ops.paged_stash import GLOBAL_BLOCK_SIZE, GLOBAL_MAX_BLOCKS
from megatron.core.transformer.moe.paged_stash import (
    PagedStashBuffer,
    PagedStashManager,
    PagedStashRunner,
    PagedTensor,
    _stash_buffer_dtype,
    check_paged_stash_host_spill,
    check_paged_stash_overflow,
    configure_runtime_schedule,
    paged_stash_init_chunk_handler,
    paged_stash_prepare_for_cuda_graph_capture,
    paged_stash_reset,
    paged_stash_te_graph_capture,
)
from megatron.core.transformer.moe.token_dispatcher import nccl_ep_release_context
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import (
    Utils,
    is_nccl_ep_available,
    is_nccl_ep_fp8_dispatch_available,
    is_nccl_ep_zero_copy_available,
)

# These tests configure mxfp8 + the TE op fuser, so they only run on Blackwell (sm100+). Mark the
# whole module for the GB200 CI bucket (selection there is marker-driven; see
# tests/unit_tests/find_test_cases.py and recipes/gb200/unit-tests.yaml).
pytestmark = pytest.mark.launch_on_gb200


def _make_schedule_manager(recorded_schedule, vp_size=1):
    manager = PagedStashManager.__new__(PagedStashManager)
    manager.enabled = True
    manager.status = 'captured'
    manager.vp_size = vp_size
    manager._pp_schedule = recorded_schedule
    manager.current_layer = [99] * vp_size
    manager.current_microbatch = [99] * vp_size
    manager.current_vp_stage = 0
    manager.current_schedule_index = len(manager._pp_schedule)
    manager._te_graph_capture = False
    manager.stash_buffers = {}
    manager.overflow = torch.zeros(1, dtype=torch.int64)
    manager.host_spill = torch.zeros(1, dtype=torch.int64)
    manager.paged_tensors_to_stash = []
    manager.paged_tensors_stash_in_progress = []
    return manager


def test_te_graph_capture_uses_capture_order_then_restores_runtime_schedule():
    runtime_schedule = [
        1_001_000,
        1_002_000,
        1_001_001,
        1_002_001,
        -1_002_000,
        -1_001_000,
        -1_002_001,
        -1_001_001,
    ]
    manager = _make_schedule_manager(runtime_schedule)
    runtime_current_layer = manager.current_layer
    runtime_current_microbatch = manager.current_microbatch
    runtime_current_vp_stage = manager.current_vp_stage

    runtime_state = manager.start_te_graph_capture([1, 1, 1, -1, -1, -1])

    assert manager._pp_schedule != runtime_schedule
    assert len(manager._pp_schedule) == 12
    assert manager.current_schedule_index == 0
    assert manager.current_layer is None
    assert manager.current_microbatch is None
    assert manager.current_vp_stage is None
    manager.prepare_te_graph_capture_forward()
    assert manager.current_layer == [1]
    assert manager.current_microbatch == [0]
    assert manager.current_vp_stage == 0
    assert manager.current_layer is not runtime_current_layer
    assert manager.current_microbatch is not runtime_current_microbatch

    # TE repeats the complete order for warmup and capture. The first forward naturally
    # wraps a fully consumed schedule without any per-callable cursor hook.
    manager.current_schedule_index = len(manager._pp_schedule)
    manager.prepare_te_graph_capture_forward()
    assert manager.current_schedule_index == 0

    manager.finish_te_graph_capture(runtime_state)
    assert not manager._te_graph_capture
    assert manager._pp_schedule is runtime_schedule
    assert manager.current_schedule_index == len(runtime_schedule)
    assert manager.current_layer is runtime_current_layer
    assert manager.current_microbatch is runtime_current_microbatch
    assert manager.current_vp_stage == runtime_current_vp_stage


def test_te_graph_capture_after_eval_restores_disabled_state_on_failure(monkeypatch):
    runtime_schedule = [1_001_000, -1_001_000]
    manager = _make_schedule_manager(runtime_schedule)
    manager.enabled = False
    runtime_current_layer = manager.current_layer
    runtime_current_microbatch = manager.current_microbatch
    monkeypatch.setattr(PagedStashManager, "STASH_MGR", manager)

    with pytest.raises(RuntimeError, match="capture failed"):
        with paged_stash_te_graph_capture(True, order=[1, -1]):
            assert manager.enabled
            assert manager._te_graph_capture
            raise RuntimeError("capture failed")

    assert not manager.enabled
    assert not manager._te_graph_capture
    assert manager._pp_schedule is runtime_schedule
    assert manager.current_layer is runtime_current_layer
    assert manager.current_microbatch is runtime_current_microbatch


def test_te_graph_capture_reallocates_buffers_released_by_warmup_fallback(monkeypatch):
    manager = _make_schedule_manager([1_001_000, -1_001_000])
    manager.stash_buffers = None
    reset_calls = []

    class FakeStashBuffer:
        def reset(self):
            reset_calls.append("reset")

    allocation_args = []

    def fake_allocate_stash_buffers(
        moe_paged_stash_buffer_size_factor_cuda, moe_paged_stash_buffer_size_factor_cpu
    ):
        allocation_args.append(
            (moe_paged_stash_buffer_size_factor_cuda, moe_paged_stash_buffer_size_factor_cpu)
        )
        manager.stash_buffers = {"dtype": {128: FakeStashBuffer()}}
        manager.overflow.fill_(1)
        manager.host_spill.fill_(1)

    monkeypatch.setattr(manager, "allocate_stash_buffers", fake_allocate_stash_buffers)
    config = SimpleNamespace(
        moe_paged_stash_buffer_size_factor_cuda=1.25, moe_paged_stash_buffer_size_factor_cpu=0.5
    )

    runtime_state = manager.start_te_graph_capture([1, -1], config=config)

    assert allocation_args == [(1.25, 0.5)]
    assert reset_calls == ["reset"]
    assert manager.overflow.item() == 0
    assert manager.host_spill.item() == 0
    manager.finish_te_graph_capture(runtime_state)


@pytest.mark.parametrize("pipeline_model_parallel_size", [1, 2])
def test_runtime_schedule_is_selected_at_every_pipeline_size(pipeline_model_parallel_size):
    """Chunk-wise graphs key the stash by the live (vp, layer, microbatch) coordinates, at PP=1
    as on any pipeline stage; forward-only passes and the other graph implementations keep the
    recorded pipeline order. (That a runtime-keyed layer really stashes at PP=1 is checked on the
    device by ``TestChunkWiseGraphPagedStash``.)"""
    manager = SimpleNamespace()
    config = SimpleNamespace(
        cuda_graph_impl="chunk_wise",
        moe_paged_stash=True,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
    )
    configure_runtime_schedule(manager, True, config)
    assert manager.runtime_schedule
    configure_runtime_schedule(manager, False, config)
    assert not manager.runtime_schedule
    configure_runtime_schedule(
        manager, True, SimpleNamespace(cuda_graph_impl="transformer_engine", moe_paged_stash=True)
    )
    assert not manager.runtime_schedule


def test_chunk_capture_after_evaluation_restarts_the_runtime_coordinates(monkeypatch):
    """A forward-only evaluation right before the chunk capture leaves the manager disabled and
    its microbatch counter advanced by every evaluation forward (a disabled reset returns before
    it restarts the coordinates). The capture runs before the next training reset: it re-enables
    the manager in runtime-keyed mode and restarts the coordinates, whose key encoding holds
    three digits per field. An unfinished discovery is rejected; a manager that no training
    schedule drove is left alone."""
    config = SimpleNamespace(cuda_graph_impl="chunk_wise", moe_paged_stash=True)

    manager = _make_schedule_manager([1_001_000, -1_001_000])
    monkeypatch.setattr(PagedStashManager, "STASH_MGR", manager)
    manager.enabled = False
    manager.runtime_schedule = False
    manager.current_layer = [3]
    manager.current_microbatch = [1500]
    paged_stash_prepare_for_cuda_graph_capture(config)
    assert manager.enabled and manager.runtime_schedule
    # The first capture-time forward opens microbatch 1, layer 1.
    paged_stash_init_chunk_handler(1, 0)
    assert (manager.current_layer, manager.current_microbatch) == ([1], [1])
    assert manager.get_schedule_layer(1, 1, manager.current_microbatch[0]) == 1_001_001

    manager = _make_schedule_manager([1_001_000])
    monkeypatch.setattr(PagedStashManager, "STASH_MGR", manager)
    manager.status = 'capture'
    with pytest.raises(AssertionError, match="finish schedule and capacity discovery"):
        paged_stash_prepare_for_cuda_graph_capture(config)

    manager = _make_schedule_manager([])
    monkeypatch.setattr(PagedStashManager, "STASH_MGR", manager)
    manager.status = 'begin'
    manager.enabled = False
    manager.runtime_schedule = False
    paged_stash_prepare_for_cuda_graph_capture(config)
    assert not manager.enabled and not manager.runtime_schedule
    assert (manager.current_layer, manager.current_microbatch) == ([99], [99])


def test_paged_stash_schedule_supports_distinct_vp_layer_templates():
    manager = _make_schedule_manager(
        [
            1_001_000,
            1_002_000,
            2_001_000,
            -2_001_000,
            1_001_001,
            1_002_001,
            -1_002_000,
            -1_001_000,
            2_001_001,
            -1_002_001,
            -1_001_001,
            -2_001_001,
        ],
        vp_size=2,
    )

    rebuilt = manager._build_te_graph_capture_schedule([1, 2, -2, 1, -1, 2, -1, -2])

    assert rebuilt == manager._pp_schedule


def test_paged_stash_schedule_rejects_invalid_recording_or_order():
    manager = _make_schedule_manager([1_001_000, -1_002_000])
    with pytest.raises(RuntimeError, match="backward layer order"):
        manager._build_te_graph_capture_schedule([1, -1])

    manager = _make_schedule_manager([1_001_000, -1_001_000])
    with pytest.raises(RuntimeError, match="unbalanced forward/backward"):
        manager._build_te_graph_capture_schedule([1, 1, -1])
    with pytest.raises(RuntimeError, match="chunk-level integer PP order"):
        manager._build_te_graph_capture_schedule([1.0, -1.0])
    with pytest.raises(RuntimeError, match="invalid VP stage"):
        manager._build_te_graph_capture_schedule([2, -2])


def test_paged_stash_schedule_skips_valid_vp_stage_without_paged_layers():
    manager = _make_schedule_manager([1_001_000, -1_001_000], vp_size=2)

    assert manager._build_te_graph_capture_schedule([1, 2, -2, -1]) == manager._pp_schedule


def test_te_graph_capture_joins_auxiliary_streams_per_layer(monkeypatch):
    manager = PagedStashManager.__new__(PagedStashManager)
    manager._te_graph_capture = True
    manager._unpack_stream_status = 'reloading'
    manager._unpack_stream = object()

    calls = []
    manager.wait_for_stash_to_complete = lambda: calls.append("stash")

    class FakeCurrentStream:
        def wait_stream(self, stream):
            calls.append(("reload", stream))

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: FakeCurrentStream())

    manager.finish_te_graph_capture_group_io()

    assert calls == ["stash", ("reload", manager.unpack_stream)]
    assert manager._unpack_stream_status == 'idle'

    manager._te_graph_capture = False
    manager._unpack_stream_status = 'reloading'
    calls.clear()
    manager.finish_te_graph_capture_group_io()
    assert calls == []
    assert manager._unpack_stream_status == 'reloading'


@pytest.mark.parametrize(
    "cuda_graph_modules", [[CudaGraphModule.moe], []], ids=["explicit-moe", "full-layer"]
)
def test_te_whole_moe_graph_overflow_fails_instead_of_dynamic_fallback(cuda_graph_modules):
    runner = PagedStashRunner.__new__(PagedStashRunner)
    runner.config = SimpleNamespace(
        cuda_graph_impl="transformer_engine", cuda_graph_modules=cuda_graph_modules
    )
    runner._te_graph_capture_finished = False

    # Warmup runs eagerly, so overflow can still use the dynamic fallback.
    runner._raise_if_te_whole_moe_graph_overflow(
        stash_overflow_ranks=1, overbudget_ranks=0, training=True
    )

    runner.mark_te_graph_captured(num_microbatches=2)

    # Evaluation also runs eagerly even after training graphs have been captured.
    runner._raise_if_te_whole_moe_graph_overflow(
        stash_overflow_ranks=1, overbudget_ranks=0, training=False
    )

    with pytest.raises(RuntimeError, match="Dynamic fallback is not supported"):
        runner._raise_if_te_whole_moe_graph_overflow(
            stash_overflow_ranks=1, overbudget_ranks=0, training=True
        )
    with pytest.raises(RuntimeError, match="expert-rank token budget overflow on 2 rank"):
        runner._raise_if_te_whole_moe_graph_overflow(
            stash_overflow_ranks=0, overbudget_ranks=2, training=True
        )

    runner._raise_if_te_whole_moe_graph_overflow(
        stash_overflow_ranks=0, overbudget_ranks=0, training=True
    )


@pytest.mark.parametrize(
    "cuda_graph_modules", [[CudaGraphModule.moe], []], ids=["explicit-moe", "full-layer"]
)
def test_te_whole_moe_paged_stash_requires_fixed_runtime_microbatch_count(cuda_graph_modules):
    runner = PagedStashRunner.__new__(PagedStashRunner)
    runner.config = SimpleNamespace(
        cuda_graph_impl="transformer_engine",
        cuda_graph_modules=cuda_graph_modules,
        moe_paged_stash=True,
    )
    runner._te_graph_runtime_num_microbatches = None
    runner._te_graph_capture_finished = False

    runner._validate_te_whole_moe_graph_runtime(training=True, num_microbatches=2)
    assert runner._te_graph_runtime_num_microbatches is None

    runner.mark_te_graph_captured(num_microbatches=2)
    runner._validate_te_whole_moe_graph_runtime(training=False, num_microbatches=4)
    assert runner._te_graph_runtime_num_microbatches == 2

    with pytest.raises(RuntimeError, match="expected 2, got 3"):
        runner._validate_te_whole_moe_graph_runtime(training=True, num_microbatches=3)
    runner._validate_te_whole_moe_graph_runtime(training=True, num_microbatches=2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dense_pipeline_stage_allocates_cuda_overflow_flags_without_local_paged_layers():
    manager = PagedStashManager.__new__(PagedStashManager)
    manager.status = 'captured'
    manager.stash_buffers = None
    manager.device = None
    manager.overflow = None
    manager.host_spill = None
    manager.max_avg_tokens_across_vp_stages = None
    manager.max_tokens_across_vp_stages = None
    manager.paged_tensors_to_stash = []
    manager.paged_tensors_stash_in_progress = []

    manager.prepare_stash_buffers()

    assert manager.stash_buffers == {}
    assert manager.overflow.is_cuda
    assert manager.host_spill.is_cuda


@pytest.fixture(autouse=True)
def reset_paged_stash_manager():
    """The manager's state machine never returns to 'begin' on its own."""
    yield
    PagedStashManager.STASH_MGR = None


@pytest.mark.parametrize("eager", [False, True])
def test_nccl_ep_dispatch_ignores_stale_zero_copy_buffers(monkeypatch, eager):
    """A prior zero-copy dispatcher must not lend its receive buffer to ordinary dispatch."""
    from megatron.core.transformer.moe import token_dispatcher

    manager = token_dispatcher._NCCLEPManager.__new__(token_dispatcher._NCCLEPManager)
    manager.router_topk = 2
    manager._max_tokens_per_rank = 64
    manager._recv_capacity = None if eager else 32
    manager.hidden_dim = 8
    manager.num_local_experts = 2
    manager.alignment = 0
    manager.dispatch_fwd_quant_recipe = None
    manager.combine_bwd_quant_recipe = None
    manager._zc_quant = False
    manager.eager = eager
    manager.token_indices = torch.zeros(4, 2, dtype=torch.int64)
    manager.token_probs = torch.full((4, 2), 0.5)
    manager.over_budget = torch.tensor(False)
    manager.required_recv = torch.tensor(0, dtype=torch.int64)
    monkeypatch.setattr(manager, "_ensure_bootstrap", lambda: None)

    # Deliberately use a different capacity, as when switching models or dispatch modes.
    stale_weights = torch.ones(128)
    monkeypatch.setattr(token_dispatcher._NCCLEPManager, "_zc_recv_topk_weights_buf", stale_weights)

    class Buffer:
        total_recv_tokens = torch.tensor(4, dtype=torch.int64)

    monkeypatch.setattr(token_dispatcher, "new_nccl_ep_buffer", lambda **kwargs: Buffer())

    def dispatch(buffer, tokens, indices, weights, *, recv_tokens, recv_topk_weights):
        if eager:
            assert recv_tokens is None and recv_topk_weights is None
        rows = 4 if eager else manager._recv_capacity
        output = torch.zeros(rows, manager.hidden_dim)
        probs = torch.zeros(rows) if recv_topk_weights is None else recv_topk_weights
        # TE enforces one receive weight for each received-token slot.
        assert probs.numel() == output.shape[0]
        return output, torch.tensor([2, 2]), probs

    monkeypatch.setattr(token_dispatcher, "nccl_ep_dispatch", dispatch)
    output = manager.dispatch(torch.zeros(4, manager.hidden_dim))

    assert manager.dispatched_probs.shape == (output.shape[0],)
    assert token_dispatcher._NCCLEPManager._zc_recv_topk_weights_buf is stale_weights
    assert torch.equal(stale_weights, torch.ones(128))


def _global_tokens_per_expert_from_local_routing_map(routing_map: torch.Tensor) -> torch.Tensor:
    """Per-expert token counts from a local routing map, summed across the default process group.

    ``routing_map`` is shaped [num_local_token_rows, num_experts] (as in
    ``_HybridEPManager``). Tests here assume world size equals expert-parallel size (all GPUs
    are EP ranks); ``all_reduce`` on the world group aggregates disjoint local maps.
    """
    counts = routing_map.sum(dim=0).to(torch.int64)
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
    return counts


def _tokens_per_expert_from_routing_map(routing_map: torch.Tensor, layer: MoELayer) -> torch.Tensor:
    """Per-local-expert assignment counts from the routing map (columns for this EP rank)."""
    counts = _global_tokens_per_expert_from_local_routing_map(routing_map)
    idx = torch.as_tensor(layer.local_expert_indices, device=counts.device, dtype=torch.long)
    return counts[idx].to(torch.int64).clone()


def _pad_token_counts_to_align_size(
    tokens_per_expert: torch.Tensor, pad_multiple: int
) -> torch.Tensor:
    """Round each count up to a multiple of ``pad_multiple`` (``n + (-n % m)`` like budget)."""
    t = tokens_per_expert.to(torch.int64)
    return t + (-t % pad_multiple)


class TestStashBufferDtype:
    def test_native_dtypes_are_kept(self):
        for dtype in (torch.bfloat16, torch.float32, torch.int64, torch.bool):
            assert _stash_buffer_dtype(dtype) is dtype

    def test_one_byte_dtypes_are_byte_copied(self):
        dtypes = [torch.float8_e4m3fn, torch.float8_e8m0fnu]
        if hasattr(torch, "float4_e2m1fn_x2"):
            dtypes.append(torch.float4_e2m1fn_x2)
        for dtype in dtypes:
            assert _stash_buffer_dtype(dtype) is torch.uint8

    def test_multi_byte_dtypes_are_rejected(self):
        with pytest.raises(ValueError, match="complex64"):
            _stash_buffer_dtype(torch.complex64)


@pytest.mark.parametrize("capability", [(10, 7), (10, 0), (10, 3), (9, 0), (12, 0)])
@pytest.mark.parametrize("with_config", [False, True])
def test_paged_stash_copy_device_sizes(monkeypatch, capability, with_config):
    manager = SimpleNamespace(iteration=0, status="begin")
    monkeypatch.setattr(PagedStashManager, "get_instance", lambda: manager)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    stash_config = SimpleNamespace(moe_paged_stash_page_size=64) if with_config else None
    paged_stash_reset(config=stash_config)
    expected = (4096, 8192) if capability == (10, 7) else (1024, 2048)
    assert (manager.copy_block_size, manager.copy_max_blocks) == expected


def test_disabled_paged_stash_does_not_query_device(monkeypatch):
    manager = SimpleNamespace(iteration=0)
    monkeypatch.setattr(PagedStashManager, "get_instance", lambda: manager)

    def unexpected_device_query():
        pytest.fail("Disabled paged stashing must not query the CUDA device")

    monkeypatch.setattr(torch.cuda, "get_device_capability", unexpected_device_query)
    paged_stash_reset(enabled=False)
    assert not manager.enabled


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestPagedTensorRoundTrip:
    """Stash -> reload through the copy/pop kernels under different launch configurations."""

    @pytest.mark.parametrize(
        "block_size,max_blocks", [(GLOBAL_BLOCK_SIZE, GLOBAL_MAX_BLOCKS), (4096, 8192)]
    )
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.bfloat16])
    # 64: one unmasked block per row; 1000 and 7168: masked tail of the last block.
    @pytest.mark.parametrize("hidden_size", [64, 1000, 7168])
    # With more rows than programs, each program loops over several rows.
    @pytest.mark.parametrize("more_rows_than_programs", [False, True])
    def test_round_trip_is_bit_exact(
        self, block_size, max_blocks, dtype, hidden_size, more_rows_than_programs
    ):
        device = torch.device("cuda")
        page_size = 64
        num_tokens = 2 * max_blocks + 100 if more_rows_than_programs else 300
        max_num_tokens = 2 * num_tokens
        if dtype == torch.uint8:
            src = torch.randint(0, 256, (max_num_tokens, hidden_size), dtype=dtype, device=device)
        else:
            src = torch.randn(max_num_tokens, hidden_size, device=device).to(dtype)
        overflow = torch.zeros(1, dtype=torch.int64, device=device)
        host_spill = torch.zeros(1, dtype=torch.int64, device=device)
        stash = PagedStashBuffer(
            max_num_tokens, hidden_size, page_size, device, overflow, host_spill, dtype
        )
        # Pages come back in arbitrary order after earlier stash/reload rounds.
        stash.free_list_cuda.copy_(torch.randperm(stash.num_cuda_pages, device=device))
        paged = PagedTensor(
            src.clone(),
            num_tokens_tensor=torch.tensor([num_tokens], dtype=torch.int64, device=device),
            max_num_tokens=max_num_tokens,
            hidden_size=hidden_size,
            page_size=page_size,
        )

        paged.offload_to_stash(stash, block_size=block_size, max_blocks=max_blocks)
        paged._tensor = torch.zeros_like(src)
        paged.reload_from_stash(stash, block_size=block_size, max_blocks=max_blocks)
        torch.cuda.synchronize()

        assert overflow.item() == 0
        assert torch.equal(paged._tensor[:num_tokens], src[:num_tokens])

    @pytest.mark.parametrize("hidden_size", [64, 1000])
    def test_reload_zero_fills_padded_rows_for_static_shape_consumers(self, hidden_size):
        """``zero_padded_tokens`` zero-fills the rows in ``[num_tokens, max_num_tokens)`` of the
        reloaded buffer (a chunk CUDA graph consumer reads all ``max_num_tokens`` static rows,
        which are otherwise uninitialised memory) and keeps the valid rows bit-exact; without
        the flag the pop kernel leaves those rows untouched."""
        device = torch.device("cuda")
        page_size, num_tokens, max_num_tokens = 64, 300, 600
        src = torch.randn(max_num_tokens, hidden_size, device=device).to(torch.bfloat16)
        garbage = torch.full_like(src, 7.0)

        def round_trip(**reload_kwargs):
            overflow = torch.zeros(1, dtype=torch.int64, device=device)
            host_spill = torch.zeros(1, dtype=torch.int64, device=device)
            stash = PagedStashBuffer(
                max_num_tokens, hidden_size, page_size, device, overflow, host_spill, src.dtype
            )
            paged = PagedTensor(
                src.clone(),
                num_tokens_tensor=torch.tensor([num_tokens], dtype=torch.int64, device=device),
                max_num_tokens=max_num_tokens,
                hidden_size=hidden_size,
                page_size=page_size,
            )
            paged.offload_to_stash(stash)
            paged._tensor = garbage.clone()
            paged.reload_from_stash(stash, **reload_kwargs)
            torch.cuda.synchronize()
            assert overflow.item() == 0
            assert torch.equal(paged._tensor[:num_tokens], src[:num_tokens])
            return paged._tensor[num_tokens:]

        assert torch.count_nonzero(round_trip(zero_padded_tokens=True)) == 0
        assert torch.equal(round_trip(), garbage[num_tokens:])

    def test_reload_kernel_constants_ignore_the_token_budget_without_zero_fill(self, monkeypatch):
        """The pop kernel's compile-time constants select its specialization. The row count is
        one of them and only the zero-fill reads it: without ``zero_padded_tokens`` a reload
        passes the same constants for every token budget (one compiled kernel for variable
        budgets), with it the row count of the buffer it fills."""
        from megatron.core.transformer.moe import paged_stash as paged_stash_module

        device = torch.device("cuda")
        page_size, hidden_size = 64, 64
        pop_kernel = paged_stash_module.paged_stash_pop_kernel
        launches = []

        class RecordingKernel:
            def __getitem__(self, grid):
                launch = pop_kernel[grid]

                def record(*args, **kwargs):
                    launches.append({key: kwargs[key] for key in kwargs if key.isupper()})
                    return launch(*args, **kwargs)

                return record

        monkeypatch.setattr(paged_stash_module, "paged_stash_pop_kernel", RecordingKernel())

        def reload_constants(max_num_tokens, **reload_kwargs):
            src = torch.randn(max_num_tokens, hidden_size, device=device).to(torch.bfloat16)
            overflow = torch.zeros(1, dtype=torch.int64, device=device)
            host_spill = torch.zeros(1, dtype=torch.int64, device=device)
            stash = PagedStashBuffer(
                max_num_tokens, hidden_size, page_size, device, overflow, host_spill, src.dtype
            )
            paged = PagedTensor(
                src.clone(),
                num_tokens_tensor=torch.tensor([200], dtype=torch.int64, device=device),
                max_num_tokens=max_num_tokens,
                hidden_size=hidden_size,
                page_size=page_size,
            )
            paged.offload_to_stash(stash)
            paged._tensor = torch.zeros_like(src)
            paged.reload_from_stash(stash, **reload_kwargs)
            torch.cuda.synchronize()
            assert torch.equal(paged._tensor[:200], src[:200])
            return launches.pop()

        constants = reload_constants(600)
        assert constants["ZERO_PADDED_TOKENS"] is False
        assert reload_constants(1200) == constants
        assert reload_constants(600, zero_padded_tokens=True)["MAX_NUM_TOKENS"] == 600
        assert reload_constants(1200, zero_padded_tokens=True)["MAX_NUM_TOKENS"] == 1200


class MoEModelTestContainer:
    def __init__(
        self,
        tp_size,
        ep_size,
        pp_size,
        cp_size=1,
        moe_tp_size=None,
        data_parallel_random_init=False,
        num_moe_experts=8,
        num_layers=1,
        moe_router_topk=2,
        moe_router_load_balancing_type="aux_loss",
        moe_token_dispatcher_type="alltoall",
        moe_expert_capacity_factor=None,
        moe_pad_expert_input_to_capacity=False,
        moe_aux_loss_coeff=0.1,
        test_dtype=torch.float32,
        **kwargs,
    ):
        self.num_local_experts = num_moe_experts // ep_size
        self.num_layers = num_layers
        self.test_dtype = test_dtype
        if moe_tp_size is None:
            moe_tp_size = tp_size
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
            expert_model_parallel_size=ep_size,
            context_parallel_size=cp_size,
            expert_tensor_parallel_size=moe_tp_size,
        )
        _set_random_seed(seed_=123, data_parallel_random_init=data_parallel_random_init)
        self.config = TransformerConfig(
            params_dtype=test_dtype,
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=ep_size,
            pipeline_model_parallel_size=pp_size,
            context_parallel_size=cp_size,
            expert_tensor_parallel_size=moe_tp_size,
            fp8='e4m3',
            fp8_recipe='mxfp8',
            fp8_wgrad=True,
            fp8_amax_compute_algo='most_recent',
            fp8_amax_history_len=1,
            fp8_interval=1,
            fp8_margin=0,
            moe_router_topk=moe_router_topk,
            num_moe_experts=num_moe_experts,
            moe_router_load_balancing_type=moe_router_load_balancing_type,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            moe_expert_capacity_factor=moe_expert_capacity_factor,
            moe_pad_expert_input_to_capacity=moe_pad_expert_input_to_capacity,
            moe_aux_loss_coeff=moe_aux_loss_coeff,
            num_layers=num_layers,
            moe_router_dtype="fp32",
            hidden_size=kwargs.get("hidden_size", 16),
            num_attention_heads=kwargs.get("num_attention_heads", 8),
            use_cpu_initialization=kwargs.get("use_cpu_initialization", True),
            sequence_parallel=tp_size > 1,
            add_bias_linear=kwargs.get("add_bias_linear", False),
            moe_permute_fusion=kwargs.get("moe_permute_fusion", False),
            moe_flex_dispatcher_backend=kwargs.get("moe_flex_dispatcher_backend", None),
            moe_ncclep_zero_copy=kwargs.get("moe_ncclep_zero_copy", False),
            moe_dispatch_fwd_dtype=kwargs.get("moe_dispatch_fwd_dtype", 'bf16'),
            moe_combine_bwd_dtype=kwargs.get("moe_combine_bwd_dtype", 'bf16'),
            moe_grouped_gemm=kwargs.get("moe_grouped_gemm", False),
            moe_use_grouped_tensor=kwargs.get("moe_use_grouped_tensor", False),
            moe_paged_stash=kwargs.get("moe_paged_stash", False),
            moe_expert_rank_capacity_factor=kwargs.get("moe_expert_rank_capacity_factor", None),
            moe_router_padding_for_fp8=kwargs.get("moe_router_padding_for_fp8", True),
            use_transformer_engine_op_fuser=kwargs.get("use_transformer_engine_op_fuser", False),
            moe_mlp_glu_interleave_size=kwargs.get("moe_mlp_glu_interleave_size", None),
            moe_router_padding_for_quantization=kwargs.get(
                "moe_router_padding_for_quantization", False
            ),
            gated_linear_unit=kwargs.get("gated_linear_unit", False),
            activation_func=kwargs.get("activation_func", F.gelu),
            bias_activation_fusion=kwargs.get("bias_activation_fusion", False),
            activation_func_fp8_input_store=kwargs.get("activation_func_fp8_input_store", False),
            recompute_granularity=kwargs.get("recompute_granularity", None),
            recompute_modules=kwargs.get("recompute_modules", None),
            moe_router_force_biased=kwargs.get("moe_router_force_biased", None),
            # Shrinking the CUDA factor and zeroing the CPU one (no host-spill fallback) is how
            # a test forces a paged-stash overflow: the pool is provisioned once at the
            # capture->captured transition, so a small factor overflows on the first real step.
            moe_paged_stash_buffer_size_factor_cuda=kwargs.get(
                "moe_paged_stash_buffer_size_factor_cuda", 0.5
            ),
            moe_paged_stash_buffer_size_factor_cpu=kwargs.get(
                "moe_paged_stash_buffer_size_factor_cpu", 1.5
            ),
        )
        self.moe_layers = [self._create_moe_layer(layer_number=i) for i in range(num_layers)]
        self.moe_layer = self.moe_layers[0]

    def _create_moe_layer(self, layer_number=0):
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=self.config.num_moe_experts, moe_grouped_gemm=True
        )
        quantization_context = get_fp8_context(self.config, layer_number, is_init=True)
        with quantization_context:
            moe_layer = (
                MoELayer(self.config, get_submodules(transformer_layer_spec.submodules.mlp))
                .cuda()
                .to(dtype=self.test_dtype)
            )
            moe_layer.set_layer_number(layer_number)
            return moe_layer

    def zero_grad(self):
        for layer in self.moe_layers:
            layer.zero_grad()

    def __del__(self):
        torch.distributed.barrier()
        torch.cuda.synchronize()
        Utils.destroy_model_parallel()

    def destroy(self):
        Utils.destroy_model_parallel()


def _forward_backward_all_layers(container: MoEModelTestContainer, hidden_states: torch.Tensor):
    """Forward/backward all MoE layers; returns output, input grad, last layer routing state."""
    initial_hidden_states = hidden_states.cuda().requires_grad_(True)
    hidden_states = initial_hidden_states
    quantization_context = get_fp8_context(container.config)
    with quantization_context:
        for layer in container.moe_layers:
            hidden_states, _ = layer(hidden_states)
        output = hidden_states
    last_layer = container.moe_layers[-1]
    comm = getattr(last_layer.token_dispatcher, "_comm_manager", None)
    routing_map = getattr(comm, "routing_map", None)
    tokens_per_expert = (
        comm.get_number_of_tokens_per_expert()
        if comm is not None and hasattr(comm, "get_number_of_tokens_per_expert")
        else None
    )
    output.backward(torch.ones_like(output))
    return (output.detach(), initial_hidden_states.grad, routing_map, tokens_per_expert)


def is_hybrid_ep_available():
    from megatron.core.transformer.moe.fused_a2a import HAVE_HYBRIDEP

    return HAVE_HYBRIDEP


def _te_grouped_mlp_op_fuser_environment_supported() -> bool:
    """Cheap gate matching the start of ``TEGroupedMLP._is_fused_impl_supported`` (experts.py)."""
    if not HAVE_TE:
        return False
    try:
        from transformer_engine.pytorch.ops import GroupedLinear, ScaledSwiGLU  # noqa: F401
    except ImportError:
        return False
    return is_te_min_version("2.14.0")


def _te_grouped_tensor_environment_supported() -> bool:
    """Return whether TE GroupedLinear exposes the device-initiated grouped-tensor API."""
    if not HAVE_TE:
        return False
    try:
        from transformer_engine.pytorch import GroupedLinear
    except ImportError:
        return False
    return "use_grouped_tensor" in inspect.signature(GroupedLinear.__init__).parameters


_TE_GROUPED_MLP_OP_FUSER_SKIP_REASON = (
    "TEGroupedMLP op fuser (tests use use_transformer_engine_op_fuser=True) requires TE>=2.14 "
    "with GroupedLinear/ScaledSwiGLU ops"
)


def _is_mxfp8_supported() -> bool:
    """MXFP8 quantization in TE requires compute capability >= 10.0 (Blackwell)."""
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] >= 10


_MXFP8_SKIP_REASON = (
    "MXFP8 (tests configure fp8_recipe='mxfp8') requires compute capability >= 10.0 (Blackwell)"
)


def test_checkpoint_without_output_view_bypasses_paged_stash_save_hook():
    tensor = torch.randn(2, 4, 4)
    _mark_checkpoint_without_output_tensor(tensor)
    # Match TE GroupedLinear's reshape + split path: both operations create new tensor objects
    # that alias the CheckpointWithoutOutput-owned storage.
    saved_view = tensor.reshape(8, 4).split(2)[0]

    # The ownership check happens before the manager needs CUDA streams or capture state.
    assert is_checkpoint_without_output_tensor(saved_view)
    manager = object.__new__(PagedStashManager)
    assert manager.on_save_for_backward(saved_view) is saved_view


@pytest.mark.skipif(not _is_mxfp8_supported(), reason=_MXFP8_SKIP_REASON)
@pytest.mark.skipif(
    not _te_grouped_tensor_environment_supported(),
    reason="Installed TE GroupedLinear does not expose use_grouped_tensor",
)
@pytest.mark.skipif(not is_hybrid_ep_available(), reason="Hybrid EP are not available")
class TestPagedStashingGroupedTensor:
    """Paged stashing with device-initiated GroupedLinear and no TE operation fuser."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    def test_forward_backward_without_op_fuser(self):
        config.ENABLE_EXPERIMENTAL = True

        container = MoEModelTestContainer(
            tp_size=1,
            ep_size=4,
            pp_size=1,
            num_moe_experts=8,
            num_layers=4,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="flex",
            moe_permute_fusion=True,
            hidden_size=1024,
            moe_flex_dispatcher_backend="hybridep",
            test_dtype=torch.bfloat16,
            moe_grouped_gemm=True,
            moe_use_grouped_tensor=True,
            moe_paged_stash=True,
            moe_expert_rank_capacity_factor=1.5,
            moe_paged_stash_buffer_size_factor_cuda=2.0,
            moe_paged_stash_buffer_size_factor_cpu=0.0,
            use_transformer_engine_op_fuser=False,
            moe_router_padding_for_quantization=True,
            gated_linear_unit=True,
            activation_func=F.silu,
            bias_activation_fusion=True,
            recompute_granularity="selective",
            recompute_modules=["moe_act"],
        )

        assert container.config.use_transformer_engine_op_fuser is False
        assert container.config.moe_use_grouped_tensor is True
        assert container.config.recompute_modules == ["moe_act"]

        hidden_states = torch.randn((1024, 1, container.config.hidden_size), dtype=torch.bfloat16)

        # Capture the activation layout and token maxima.
        paged_stash_reset(True, config=container.config)
        paged_stash_init_chunk_handler(1, 0)
        output_ref, hidden_states_grad_ref, _, _ = _forward_backward_all_layers(
            container, hidden_states
        )

        stash_manager = PagedStashManager.get_instance()
        assert (
            stash_manager.max_tokens_across_vp_stages
        ), "No dynamic GroupedLinear/activation tensors were captured for paged stashing"
        assert any(
            dtype == torch.bfloat16
            for dtype, _hidden_size in stash_manager.max_tokens_across_vp_stages
        ), "The fused activation's BF16 saved tensors were not captured for paged stashing"
        container.zero_grad()

        # Allocate the stash buffers from the capture and exercise the real stash/reload path.
        paged_stash_reset(True, config=container.config)
        paged_stash_init_chunk_handler(1, 0)
        output, hidden_states_grad, _, _ = _forward_backward_all_layers(container, hidden_states)

        overflow = check_paged_stash_overflow()
        assert overflow.any().item() == 0
        torch.testing.assert_close(output, output_ref, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(hidden_states_grad, hidden_states_grad_ref, atol=1e-4, rtol=1e-4)


# Stash-free step vs. step with stash (eager or replayed), on float32 copies of the BF16 loss and
# gradients. The stash only copies bytes, and on GB200 the three steps agree bit for bit; the
# tolerance leaves room for kernels that do not reproduce bitwise and stays two orders below the
# gradient error of a reload from the wrong pages (about 2e-2).
_STASH_RTOL = 1e-4
_STASH_ATOL = 1e-4


def _cuda_page_use(stash_manager):
    """(pages allocated since the last reset, pages currently held) over the CUDA page buffers.

    Each buffer's free list is a ring with monotonic head (allocation) and tail (release)
    counters that ``paged_stash_reset`` sets back to (0, num_cuda_pages); the modulo is applied
    only when indexing the ring. The stash kernels advance them on the device, so they also
    account for stash / reload work that a replayed CUDA graph issues without any Python call.
    """
    allocated = 0
    held = 0
    for dtype_buffers in stash_manager.stash_buffers.values():
        for stash_buffer in dtype_buffers.values():
            head = stash_buffer.free_list_head[0].item()
            tail = stash_buffer.free_list_tail[0].item()
            allocated += head
            held += stash_buffer.num_cuda_pages - (tail - head)
    return allocated, held


@pytest.mark.skipif(not _is_mxfp8_supported(), reason=_MXFP8_SKIP_REASON)
@pytest.mark.skipif(
    not _te_grouped_tensor_environment_supported(),
    reason="TE GroupedLinear does not expose the grouped-tensor API",
)
@pytest.mark.skipif(not is_hybrid_ep_available(), reason="Hybrid EP are not available")
@pytest.mark.skipif(
    not is_te_min_version("2.19.0"), reason="chunk-wise paged stash requires TE >= 2.19"
)
class TestChunkWiseGraphPagedStash:
    """Chunk-wise TE CUDA graphs must stash MoE activations through the page buffers.

    ``configure_runtime_schedule`` keys the stash by the live (vp, layer, microbatch) coordinates
    at every pipeline size. The tests drive the real path: two eager warm-up steps (schedule and
    capacity discovery, cuda_graph_warmup_steps=2), chunk-graph capture, then one replayed step
    whose page traffic is read back from the device free lists -- a replayed graph leaves no
    Python trace, so the buffers themselves are the evidence that pages were used.
    """

    seq_length = 512
    vocab_size = 128

    def teardown_method(self, method):
        destroy_num_microbatches_calculator()
        reset_hybrid_ep_buffer()
        Utils.destroy_model_parallel()

    def _build_model(self, **config_overrides):
        from megatron.core.models.gpt.gpt_model import GPTModel

        # The paged-stash MoE recipe of TestPagedStashingGroupedTensor (GroupedTensor path, no
        # op fuser) under chunk-wise graphs: the HybridEP flex dispatcher with a rank capacity
        # gives the MoE its static shapes. CPU factor 0: no host spill, every page is a CUDA page.
        config_kwargs = dict(
            num_layers=2,
            hidden_size=256,
            num_attention_heads=4,
            ffn_hidden_size=512,
            moe_ffn_hidden_size=512,
            num_moe_experts=4,
            moe_router_topk=2,
            moe_router_load_balancing_type="none",
            moe_aux_loss_coeff=0.0,
            moe_router_dtype="fp32",
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="hybridep",
            moe_permute_fusion=True,
            moe_grouped_gemm=True,
            moe_use_grouped_tensor=True,
            use_transformer_engine_op_fuser=False,
            moe_expert_rank_capacity_factor=2.0,
            moe_paged_stash=True,
            moe_paged_stash_buffer_size_factor_cuda=2.0,
            moe_paged_stash_buffer_size_factor_cpu=0.0,
            moe_router_padding_for_quantization=True,
            gated_linear_unit=True,
            activation_func=F.silu,
            bias_activation_fusion=True,
            add_bias_linear=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            fp8='e4m3',
            fp8_recipe='mxfp8',
            fp8_wgrad=True,
            fp8_amax_compute_algo='most_recent',
            fp8_amax_history_len=1,
            fp8_interval=1,
            fp8_margin=0,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            cuda_graph_impl="chunk_wise",
            # The batch carries no attention mask (causal, like --no-create-attention-mask-in-
            # dataloader); the chunk capture must not sample one the replay cannot refresh.
            create_attention_mask_in_dataloader=False,
            cuda_graph_modules=[],
            cuda_graph_warmup_steps=2,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            use_cpu_initialization=True,
        )
        config_kwargs.update(config_overrides)
        transformer_config = TransformerConfig(**config_kwargs)
        assert transformer_config.pipeline_model_parallel_size == 1
        layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=transformer_config.num_moe_experts, moe_grouped_gemm=True
        )
        model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=layer_spec,
            vocab_size=self.vocab_size,
            max_sequence_length=self.seq_length,
            position_embedding_type="rope",
        ).cuda()
        model.train()
        # The helper's post-capture reset calls the DDP wrapper's zero_grad_buffer(); this test
        # drives the bare model and zeroes gradients itself.
        model.zero_grad_buffer = lambda: None
        return model

    def _get_batch(self):
        data = torch.arange(self.seq_length, dtype=torch.int64, device="cuda")
        return {
            "input_ids": (data * 7 % self.vocab_size).unsqueeze(0),
            "labels": ((data + 1) * 7 % self.vocab_size).unsqueeze(0),
            "position_ids": data.unsqueeze(0),
            "loss_mask": torch.ones((1, self.seq_length), dtype=torch.float32, device="cuda"),
        }

    @staticmethod
    def _forward(model, batch):
        """One training forward, prepared the way the pipeline schedules prepare it."""
        # schedules.py resets the stash at the start of every training iteration.
        paged_stash_reset(enabled=True, config=model.config)
        for param in model.parameters():
            # Caller-owned leaf-grad buffers: with a leaf grad of None, AccumulateGrad may adopt
            # TE's recyclable static grad buffer instead of copying out of it.
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            else:
                param.grad.zero_()
        per_token_loss = model(
            input_ids=batch["input_ids"],
            position_ids=batch["position_ids"],
            attention_mask=None,
            labels=batch["labels"],
            loss_mask=batch["loss_mask"],
        )
        loss = (per_token_loss.float() * batch["loss_mask"]).sum() / batch["loss_mask"].sum()
        assert torch.isfinite(loss), "non-finite loss"
        return loss

    @staticmethod
    def _evaluate(model, batch):
        """One forward-only pass, prepared the way the pipeline schedules prepare evaluation."""
        paged_stash_reset(enabled=False, config=model.config)
        model.eval()
        with torch.no_grad():
            model(
                input_ids=batch["input_ids"],
                position_ids=batch["position_ids"],
                attention_mask=None,
                labels=batch["labels"],
                loss_mask=batch["loss_mask"],
            )
        model.train()

    @classmethod
    def _split_tokens_between_two_expert_ranks(cls, model, batch):
        """Top-1 router weights for a one-layer model whose experts live on two ranks: the logits
        of the first rank's experts are a projection of the router input that is orthogonal to
        the batch mean (scaled by 1, 1/2, 1/4, ...), those of the second rank's experts its
        negative. The projection has zero mean over the batch, so about half of its tokens go to
        each rank (to its first expert), whereas a constant input (all tokens alike) takes a
        single route. Returns the fraction of the batch's tokens that the first rank receives."""
        (layer,) = model.decoder.layers
        router_inputs = []
        hook = layer.mlp.router.register_forward_hook(
            lambda module, args, output: router_inputs.append(args[0].detach().float())
        )
        try:
            cls._evaluate(model, batch)
        finally:
            hook.remove()
        # The untrained router may have sent that pass down one route.
        layer.mlp.token_dispatcher.reset_over_budget()
        tokens = router_inputs[0].reshape(-1, router_inputs[0].shape[-1])
        mean = tokens.mean(dim=0)
        direction = torch.randn(mean.shape, generator=torch.Generator().manual_seed(0)).to(mean)
        direction -= (direction @ mean) / (mean @ mean) * mean
        weight = layer.mlp.router.weight
        scales = 0.5 ** torch.arange(weight.shape[0] // 2).to(mean)
        with torch.no_grad():
            weight.copy_((torch.cat([scales, -scales])[:, None] * direction).to(weight))
        return ((tokens @ direction) > 0).float().mean().item()

    @staticmethod
    def _step_result(model, loss):
        """Loss and every parameter gradient of the step that just ran its backward."""
        torch.cuda.synchronize()
        grads = {
            name: param.grad.detach().float().clone() for name, param in model.named_parameters()
        }
        return loss.detach().float().clone(), grads

    @staticmethod
    def _assert_same_step(result, reference, what):
        """The stash moves bytes: a step that stashes, eagerly or from a replayed graph, computes
        the loss and the gradients of the step that kept its activations in place."""
        loss, grads = result
        reference_loss, reference_grads = reference
        torch.testing.assert_close(
            loss,
            reference_loss,
            rtol=_STASH_RTOL,
            atol=_STASH_ATOL,
            msg=lambda msg: f"{what}: loss\n{msg}",
        )
        assert grads.keys() == reference_grads.keys()
        for name, reference_grad in reference_grads.items():
            torch.testing.assert_close(
                grads[name],
                reference_grad,
                rtol=_STASH_RTOL,
                atol=_STASH_ATOL,
                msg=lambda msg, name=name: f"{what}: gradient of {name}\n{msg}",
            )

    @contextmanager
    def _chunk_graph_run(self, **model_parallel_sizes):
        """Model-parallel state, the calculator and the capture stream of one chunk-graph run;
        yields the list the test adds its graph helper to, for the teardown."""
        # A MoEModelTestContainer that still awaits collection destroys the model-parallel state
        # from its finalizer: collect it before this run initializes its own.
        gc.collect()
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1, **model_parallel_sizes
        )
        # The capture sizes its TE warm-up from the run's micro-batch count.
        init_num_microbatches_calculator(
            rank=0,
            rampup_batch_size=None,
            global_batch_size=Utils.world_size,
            micro_batch_size=1,
            data_parallel_size=Utils.world_size,
        )
        # Graph-safe RNG tracker, as the training loop enables for CUDA graphs; the CPU generator
        # initializes the weights (use_cpu_initialization).
        model_parallel_cuda_manual_seed(123, te_rng_tracker=True, force_reset_rng=True)
        torch.manual_seed(123)

        helpers = []
        # Warm up on the stream the capture uses: AccumulateGrad nodes created by the warm-up
        # backwards keep a reference to their stream, and a default-stream reference would pull
        # a non-capturing stream into the graph (see _run_iters_with_cuda_graph in the offload
        # tests and the training loop's side stream).
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        prev_stream = torch.cuda.current_stream()
        torch.cuda.set_stream(side_stream)
        try:
            yield helpers
        finally:
            torch.cuda.synchronize()
            torch.cuda.set_stream(prev_stream)
            torch.cuda.synchronize()
            for helper in helpers:
                if helper.graphs_created():
                    helper.delete_cuda_graphs()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    def test_pp1_replay_uses_stash_pages(self, monkeypatch):
        """PP=1, one rank's own experts, three layers.

        The first layer is checkpointed (full recompute, method "block"): its forward saves
        nothing, and its recompute runs inside the backward, which consumes the recomputed
        activations on the spot, so that layer moves no page; the other two stash in the forward
        and reload in the backward. A forward-only evaluation runs between the warm-up and the
        capture (the training loop evaluates at the end of an iteration and captures at the
        start of the next): it leaves the manager disabled, and the capture must still record
        the stash. The eager step with stash and the replayed step reproduce the loss and every
        gradient of the step that stashed nothing."""
        from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

        config.ENABLE_EXPERIMENTAL = True
        with self._chunk_graph_run(expert_model_parallel_size=1) as helpers:
            model = self._build_model(
                num_layers=3,
                recompute_granularity="full",
                recompute_method="block",
                recompute_num_layers=1,
            )
            batch = self._get_batch()
            stash_manager = PagedStashManager.get_instance()

            def assert_page_use(step, after_forward):
                torch.cuda.synchronize()
                allocated, held = _cuda_page_use(stash_manager)
                if after_forward:
                    assert allocated > 0, f"{step}: the PP=1 chunk_wise forward stashed nothing"
                    assert held > 0, f"{step}: the forward holds no stash pages before backward"
                else:
                    assert held == 0, f"{step}: the backward left {held} stash pages unreleased"
                return allocated

            # Warm-up step 1: records the stash schedule and the token maxima; nothing is
            # stashed yet, which makes it the reference for the steps that do.
            loss = self._forward(model, batch)
            loss.backward()
            assert stash_manager.status == 'capture'
            assert (
                stash_manager.max_tokens_across_vp_stages
            ), "no GroupedLinear activations were captured for paged stashing"
            reference = self._step_result(model, loss)

            # Warm-up step 2: the page buffers exist and the eager run already stashes at PP=1.
            loss = self._forward(model, batch)
            assert stash_manager.status == 'captured'
            assert stash_manager.runtime_schedule
            assert stash_manager.stash_buffers
            allocated = assert_page_use("eager", after_forward=True)
            loss.backward()
            assert (
                assert_page_use("eager", after_forward=False) == allocated
            ), "the eager backward (the first layer's recompute) must not stash"
            self._assert_same_step(
                self._step_result(model, loss), reference, "eager step with stash"
            )

            # Evaluation, then the capture after cuda_graph_warmup_steps=2, like the training loop.
            self._evaluate(model, batch)
            assert not stash_manager.enabled
            helper = TECudaGraphHelper(
                model=[model], config=model.config, seq_length=self.seq_length, micro_batch_size=1
            )
            helpers.append(helper)
            helper.create_cudagraphs()
            assert helper.graphs_created()
            assert len(model.decoder.cuda_graphs) > 0

            replays = []
            original_replay = torch.cuda.CUDAGraph.replay

            def counting_replay(graph, *args, **kwargs):
                replays.append(graph)
                return original_replay(graph, *args, **kwargs)

            monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", counting_replay)

            # Replayed step: the forward graph must stash into the pages, the backward graph must
            # reload and release them.
            loss = self._forward(model, batch)
            torch.cuda.synchronize()
            forward_replays = len(replays)
            assert forward_replays > 0, "the decoder forward did not replay a CUDA graph"
            allocated = assert_page_use("replay", after_forward=True)
            loss.backward()
            torch.cuda.synchronize()
            assert len(replays) > forward_replays, "the decoder backward did not replay a graph"
            assert (
                assert_page_use("replay", after_forward=False) == allocated
            ), "the backward graph (the first layer's recompute) must not stash"
            assert check_paged_stash_overflow().any().item() == 0
            assert check_paged_stash_host_spill().any().item() == 0
            self._assert_same_step(self._step_result(model, loss), reference, "replayed step")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    def test_pp1_with_every_moe_layer_recomputed_allocates_no_page_buffers(self):
        """PP=1, two layers, each a recompute segment of its own (full recompute, "uniform").

        Every MoE forward that saves activations then runs inside the backward as the last
        layer of its segment, which that backward consumes on the spot, so nothing is worth a
        stash. The flag must then cost nothing: the recomputed activations do not size the
        page buffers, none is allocated (the manager says so once), and the eager step, the
        capture and the replayed step reproduce the step that ran before the stash was armed."""
        from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

        config.ENABLE_EXPERIMENTAL = True
        with self._chunk_graph_run(expert_model_parallel_size=1) as helpers:
            model = self._build_model(
                num_layers=2,
                recompute_granularity="full",
                recompute_method="uniform",
                recompute_num_layers=1,
            )
            batch = self._get_batch()
            stash_manager = PagedStashManager.get_instance()

            # Warm-up step 1: schedule and capacity discovery. The only activations saved for
            # a backward are those of the recompute forwards, which are never stashed.
            loss = self._forward(model, batch)
            loss.backward()
            assert stash_manager.status == 'capture'
            assert stash_manager.runtime_schedule
            assert not stash_manager.max_tokens_across_vp_stages, (
                "activations of a backward-time recompute sized the page buffers: "
                f"{stash_manager.max_tokens_across_vp_stages}"
            )
            reference = self._step_result(model, loss)

            # Warm-up step 2: the stash is armed and has nothing to hold.
            loss = self._forward(model, batch)
            assert stash_manager.status == 'captured'
            assert stash_manager.stash_buffers == {}, "page buffers were allocated for nothing"
            assert stash_manager._warned_nothing_to_stash
            loss.backward()
            self._assert_same_step(self._step_result(model, loss), reference, "eager step")

            self._evaluate(model, batch)
            helper = TECudaGraphHelper(
                model=[model], config=model.config, seq_length=self.seq_length, micro_batch_size=1
            )
            helpers.append(helper)
            helper.create_cudagraphs()
            assert helper.graphs_created()
            assert len(model.decoder.cuda_graphs) > 0

            loss = self._forward(model, batch)
            loss.backward()
            torch.cuda.synchronize()
            assert stash_manager.stash_buffers == {}
            assert check_paged_stash_overflow().any().item() == 0
            assert check_paged_stash_host_spill().any().item() == 0
            self._assert_same_step(self._step_result(model, loss), reference, "replayed step")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    def test_pp1_recompute_segment_stashes_all_but_its_last_layer(self, monkeypatch):
        """PP=1, two layers in one recompute segment (full recompute, "uniform", two layers).

        The forward saves nothing; the recompute runs both layers inside the backward, which
        then walks them in reverse. The first layer's activations wait for the second layer's
        backward, so they are stashed (as the recorded-order schedule of the eager mode does);
        the second layer's are consumed at once and stay in place. The page buffers are sized
        for exactly that one layer, and the eager step with stash and the replayed step
        reproduce the loss and every gradient of the step that stashed nothing."""
        from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

        config.ENABLE_EXPERIMENTAL = True
        with self._chunk_graph_run(expert_model_parallel_size=1) as helpers:
            model = self._build_model(
                num_layers=2,
                recompute_granularity="full",
                recompute_method="uniform",
                recompute_num_layers=2,
            )
            batch = self._get_batch()
            stash_manager = PagedStashManager.get_instance()

            def page_use():
                torch.cuda.synchronize()
                return _cuda_page_use(stash_manager)

            # Warm-up step 1: schedule and capacity discovery, nothing is stashed. Record the
            # tensors each recompute commit holds and those that are counted for the buffers.
            commits, counted = [], []
            with monkeypatch.context() as patch:
                defer = stash_manager.defer_recompute_commit
                count = stash_manager._count_stash_tokens

                def recording_defer(ctx, schedule_layer):
                    defer(ctx, schedule_layer)
                    commits.append(len(stash_manager._deferred_recompute_commit[2]))

                def recording_count(*record):
                    counted.append(record)
                    count(*record)

                patch.setattr(stash_manager, "defer_recompute_commit", recording_defer)
                patch.setattr(stash_manager, "_count_stash_tokens", recording_count)
                loss = self._forward(model, batch)
                loss.backward()
            assert stash_manager.status == 'capture'
            assert len(commits) == 2 and commits[0] == commits[1] > 0, commits
            # The buffers are sized for the first layer of the segment only.
            assert len(counted) == commits[0], (len(counted), commits)
            assert stash_manager.max_tokens_across_vp_stages
            reference = self._step_result(model, loss)

            # Warm-up step 2: the eager forward stashes nothing (it saves nothing), the eager
            # backward stashes the first layer of the segment and releases its pages again.
            loss = self._forward(model, batch)
            assert stash_manager.status == 'captured'
            assert stash_manager.stash_buffers
            assert page_use() == (0, 0), "the no-grad forward of a checkpoint stashed"
            loss.backward()
            allocated, held = page_use()
            assert allocated > 0, "the recompute segment's first layer was not stashed"
            assert held == 0, f"the backward left {held} stash pages unreleased"
            self._assert_same_step(
                self._step_result(model, loss), reference, "eager step with stash"
            )

            self._evaluate(model, batch)
            helper = TECudaGraphHelper(
                model=[model], config=model.config, seq_length=self.seq_length, micro_batch_size=1
            )
            helpers.append(helper)
            helper.create_cudagraphs()
            assert helper.graphs_created()

            # Replayed step: the stash and the reload of the first layer are part of the
            # backward graph.
            loss = self._forward(model, batch)
            assert page_use() == (0, 0)
            loss.backward()
            assert page_use() == (allocated, 0), "the backward graph moved other pages than eager"
            assert check_paged_stash_overflow().any().item() == 0
            assert check_paged_stash_host_spill().any().item() == 0
            self._assert_same_step(self._step_result(model, loss), reference, "replayed step")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    def test_capture_on_constant_samples_leaves_no_over_budget_flag(self):
        """Transformer Engine warms every block up on constant sample inputs before it records
        the graphs. All tokens then take one route, so the rank that hosts the chosen expert
        receives the tokens of both ranks of its expert-parallel group: twice the tokens of a
        rank, against a capacity of 1.5 times that the batch itself respects (two expert ranks,
        top-1). The dispatcher's over-budget flag accumulates until it is reset;
        the capture must leave it clear, or the first replayed step fails as if its own, fitting
        batch had overflowed the static buffers."""
        from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

        if Utils.world_size % 2 != 0:
            pytest.skip("expert_model_parallel_size=2 needs an even number of ranks")
        config.ENABLE_EXPERIMENTAL = True
        with self._chunk_graph_run(expert_model_parallel_size=2) as helpers:
            model = self._build_model(
                num_layers=1,
                expert_model_parallel_size=2,
                num_moe_experts=8,
                moe_router_topk=1,
                moe_router_pre_softmax=True,
                moe_expert_rank_capacity_factor=1.5,
            )
            batch = self._get_batch()
            # Every rank holds the same batch and weights, so the two expert ranks receive this
            # fraction of the group's tokens and the rest: within the capacity up to 3/4.
            first_rank_share = self._split_tokens_between_two_expert_ranks(model, batch)
            assert 0.3 < first_rank_share < 0.7, first_rank_share

            schedule_runs = []

            def forward_backward_func(*, model, data_iterator, **schedule_kwargs):
                schedule_runs.append(schedule_kwargs)
                loss = self._forward(model[0], next(data_iterator[0]))
                loss.backward()
                return loss

            # The training loop's overflow handling around the schedule.
            runner = PagedStashRunner(model.config, False, [model], None, forward_backward_func)
            assert len(runner.moe_layers) == model.config.num_layers

            def run_step():
                return runner(
                    model=[model],
                    data_iterator=iter([batch]),
                    num_microbatches=1,
                    seq_length=self.seq_length,
                    forward_only=False,
                )

            # Two eager warm-up steps: the real batch fits the rank capacity on every rank, so
            # the runner never repeats a step without the capacity.
            for step in (1, 2):
                run_step()
                assert len(schedule_runs) == step, "the test batch must fit the rank capacity"
            assert PagedStashManager.get_instance().status == 'captured'

            helper = TECudaGraphHelper(
                model=[model], config=model.config, seq_length=self.seq_length, micro_batch_size=1
            )
            helpers.append(helper)
            helper.create_cudagraphs()
            assert helper.graphs_created()
            runner.mark_te_graph_captured(num_microbatches=1)
            _, over_budget_ranks, _ = runner.check_moe_overflow()
            assert over_budget_ranks == 0, (
                f"{over_budget_ranks} rank(s) still flag the over-budget dispatch of the capture "
                "warm-ups"
            )

            # The replayed step, whose batch fits: no static-buffer overflow error.
            loss = run_step()
            assert len(schedule_runs) == 3 and torch.isfinite(loss)
            assert runner.check_moe_overflow()[:2] == (0, 0)


@pytest.mark.skipif(not _is_mxfp8_supported(), reason=_MXFP8_SKIP_REASON)
@pytest.mark.skipif(
    not _te_grouped_mlp_op_fuser_environment_supported(),
    reason=_TE_GROUPED_MLP_OP_FUSER_SKIP_REASON,
)
@pytest.mark.skipif(not is_hybrid_ep_available(), reason="Hybrid EP are not available")
class TestPagedStashing:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        reset_hybrid_ep_buffer()
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.flaky_in_dev
    def test_forward_backward_4_layers(self):
        """Test paged stashing with 4 MoE layers: ref run vs paged run match."""
        if not is_hybrid_ep_available():
            pytest.skip("Hybrid EP is not available")

        config.ENABLE_EXPERIMENTAL = True

        container = MoEModelTestContainer(
            tp_size=1,
            ep_size=4,
            pp_size=1,
            num_moe_experts=8,
            num_layers=4,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="flex",
            moe_permute_fusion=True,
            hidden_size=1024,
            moe_flex_dispatcher_backend="hybridep",
            test_dtype=torch.bfloat16,
            moe_grouped_gemm=True,
            moe_paged_stash=True,
            moe_expert_rank_capacity_factor=1.5,
            use_transformer_engine_op_fuser=True,
            moe_mlp_glu_interleave_size=32,
            moe_router_padding_for_quantization=True,
            gated_linear_unit=True,
            activation_func=F.silu,
        )

        seq_length = 1024
        batch_size = 1
        hidden_size = container.config.hidden_size
        hidden_states = torch.randn((seq_length, batch_size, hidden_size), dtype=torch.bfloat16)

        # First iteration: capture schedule, capacity, etc.
        paged_stash_reset(True, config=container.config)
        paged_stash_init_chunk_handler(1, 0)
        output_ref, hidden_states_grad_ref, routing_map_ref, tokens_per_expert_ref = (
            _forward_backward_all_layers(container, hidden_states)
        )

        container.zero_grad()

        # Second iteration: run with paged stash.
        paged_stash_reset(True, config=container.config)
        paged_stash_init_chunk_handler(1, 0)
        output, hidden_states_grad, routing_map, tokens_per_expert = _forward_backward_all_layers(
            container, hidden_states
        )

        overflow = check_paged_stash_overflow()
        assert overflow.any().item() == 0

        assert torch.allclose(
            output, output_ref, atol=1e-4, rtol=1e-4
        ), f"output != output_ref: max diff = {(output - output_ref).abs().max().item()}"
        assert torch.allclose(hidden_states_grad, hidden_states_grad_ref, atol=1e-4, rtol=1e-4), (
            f"hidden_states_grad != ref: max diff = "
            f"{(hidden_states_grad - hidden_states_grad_ref).abs().max().item()}"
        )
        if routing_map is not None and tokens_per_expert is not None:
            num_tokens_per_ep_rank = tokens_per_expert.sum().item()
            assert (
                num_tokens_per_ep_rank > 0
            ), f"num_tokens_per_ep_rank={num_tokens_per_ep_rank} (expected > 0)"
            assert routing_map_ref is not None and tokens_per_expert_ref is not None
            tpe_f = tokens_per_expert.float()
            ref_f = tokens_per_expert_ref.float()
            assert torch.allclose(
                tpe_f, ref_f, atol=1e-4, rtol=1e-4
            ), f"tokens_per_expert != ref: max diff = {(tpe_f - ref_f).abs().max().item()}"


@pytest.mark.skipif(not _is_mxfp8_supported(), reason=_MXFP8_SKIP_REASON)
@pytest.mark.skipif(
    not _te_grouped_mlp_op_fuser_environment_supported(),
    reason=_TE_GROUPED_MLP_OP_FUSER_SKIP_REASON,
)
@pytest.mark.skipif(not is_hybrid_ep_available(), reason="Hybrid EP are not available")
class TestPagedStashingOverBudget:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        reset_hybrid_ep_buffer()
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.flaky_in_dev
    def test_overload_factor_and_over_budget(self):
        """Budget matches HybridEP setup_metadata; over_budget matches map-derived load."""
        if not is_hybrid_ep_available():
            pytest.skip("Hybrid EP is not available")

        config.ENABLE_EXPERIMENTAL = True

        container = MoEModelTestContainer(
            tp_size=1,
            ep_size=4,
            pp_size=1,
            num_moe_experts=8,
            num_layers=4,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="flex",
            moe_permute_fusion=True,
            hidden_size=1024,
            moe_flex_dispatcher_backend="hybridep",
            test_dtype=torch.bfloat16,
            moe_grouped_gemm=True,
            moe_paged_stash=True,
            moe_expert_rank_capacity_factor=1.5,
            use_transformer_engine_op_fuser=True,
            moe_mlp_glu_interleave_size=32,
            moe_router_padding_for_quantization=True,
            gated_linear_unit=True,
            activation_func=F.silu,
            moe_router_force_biased=1,
        )

        seq_length = 1024
        batch_size = 1
        topk = container.config.moe_router_topk
        capacity_factor = container.config.moe_expert_rank_capacity_factor
        hidden_states = torch.randn(
            (seq_length, batch_size, container.config.hidden_size), dtype=torch.bfloat16
        )

        num_tokens = seq_length * batch_size * topk
        pad_multiple = get_align_size_for_quantization(container.config)
        budget = int(num_tokens * capacity_factor)
        budget += -budget % pad_multiple

        # This just captures; needed for setup.
        paged_stash_reset(True, config=container.config)
        paged_stash_init_chunk_handler(1, 0)
        _forward_backward_all_layers(container, hidden_states)
        container.zero_grad()
        for layer in container.moe_layers:
            layer.token_dispatcher.reset_over_budget()

        # This does the stashing needed for the test.
        paged_stash_reset(True, config=container.config)
        paged_stash_init_chunk_handler(1, 0)
        _forward_backward_all_layers(container, hidden_states)

        overflow = check_paged_stash_overflow()
        num_layers = len(container.moe_layers)
        stash_cuda = container.config.moe_paged_stash_buffer_size_factor_cuda
        stash_cpu = container.config.moe_paged_stash_buffer_size_factor_cpu
        stash_buffer_size = num_tokens * num_layers * (stash_cuda + stash_cpu)

        total_tokens = 0
        for layer_idx, layer in enumerate(container.moe_layers):
            comm = getattr(layer.token_dispatcher, "_comm_manager", None)
            routing_map = getattr(comm, "routing_map", None) if comm is not None else None
            over_budget_tensor = (
                layer.token_dispatcher.check_over_budget()
                if hasattr(layer.token_dispatcher, "check_over_budget")
                else None
            )
            over_budget = over_budget_tensor.item() if over_budget_tensor is not None else False

            assert routing_map is not None, f"layer {layer_idx}: routing_map is None"
            assert routing_map.dim() == 2, f"layer {layer_idx}: expected 2D routing_map"
            assert routing_map.shape[1] == container.config.num_moe_experts, (
                f"layer {layer_idx}: routing_map has {routing_map.shape[1]} experts, "
                f"expected {container.config.num_moe_experts}"
            )
            tokens_per_expert_from_map = _tokens_per_expert_from_routing_map(routing_map, layer)
            tokens_per_expert_from_map_padded = _pad_token_counts_to_align_size(
                tokens_per_expert_from_map, pad_multiple
            )
            tokens_per_ep_rank_from_map = tokens_per_expert_from_map_padded.sum().item()
            total_tokens += tokens_per_ep_rank_from_map

            # Padded map-derived tokens strictly over budget iff dispatcher reports over_budget
            if tokens_per_ep_rank_from_map > budget:
                assert over_budget, (
                    f"layer {layer_idx}: tokens_per_ep_rank_from_map "
                    f"({tokens_per_ep_rank_from_map}) > budget ({budget}), "
                    f"but over_budget flag was not set"
                )
            else:
                assert not over_budget, (
                    f"layer {layer_idx}: tokens_per_ep_rank_from_map "
                    f"({tokens_per_ep_rank_from_map}) <= budget ({budget}), "
                    f"but over_budget flag was set"
                )

        overflow_set = overflow.any().item()
        stash_exceeded = total_tokens > stash_buffer_size
        assert overflow_set == stash_exceeded, (
            f"overflow {overflow_set} should match total_tokens > stash_buffer_size "
            f"({total_tokens} > {stash_buffer_size})"
        )


@pytest.mark.skipif(not _is_mxfp8_supported(), reason=_MXFP8_SKIP_REASON)
@pytest.mark.skipif(
    not _te_grouped_mlp_op_fuser_environment_supported(),
    reason=_TE_GROUPED_MLP_OP_FUSER_SKIP_REASON,
)
@pytest.mark.skipif(not is_nccl_ep_available(), reason="NCCL EP is not available")
class TestNcclEpPagedStashing:
    """Paged stashing with the NCCL EP flex backend in its static-shape path.

    ncclep's CUDA-graph / paged-stash path requires moe_expert_rank_capacity_factor, which feeds
    the experts the full fixed-size recv buffer and is only valid with fp8/fp4 + the CuTe DSL
    grouped GEMM (the container always configures mxfp8; NVTE_CUTEDSL_FUSED_GROUPED_MLP=1 must be
    set in the environment). This mirrors TestPagedStashing: run the paged-stash path twice and
    assert the two
    passes agree (a determinism guard for the static ncclep path), plus no paged-stash overflow.
    """

    def setup_method(self, method):
        self.container = None

    def teardown_method(self, method):
        try:
            self.container = None
            gc.collect()
            nccl_ep_release_context()
            assert not is_nccl_ep_bootstrapped(), "NCCL EP context survived release"
        finally:
            Utils.destroy_model_parallel()

    def _run_step(self, hidden_states):
        paged_stash_reset(True, config=self.container.config)
        paged_stash_init_chunk_handler(1, 0)
        out, _, _, _ = _forward_backward_all_layers(self.container, hidden_states)
        self.container.zero_grad()
        torch.cuda.synchronize()
        return out

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.parametrize("wire_dtype", ["bf16", "mxfp8"])
    def test_over_budget(self, wire_dtype):
        """Budget matches _NCCLEPManager._ensure_bootstrap; over_budget matches map-derived load.

        Mirrors TestPagedStashingOverBudget for HybridEP, plus the peak capacity NCCL EP
        reports -- HybridEP has no equivalent because it recovers by going dropless and so
        never needs to know how much was required. The capacity factor is deliberately below
        1.0: each rank receives num_tokens*topk on average, so 1.0 sits exactly at the mean
        and anything under it overflows.

        wire_dtype="mxfp8" runs the same overflow accounting with MXFP8 dispatch-fwd /
        combine-bwd wire payloads (the opaque carrier); the budget arithmetic under test is
        payload-dtype independent, so the assertions are unchanged.
        """
        if not is_nccl_ep_available():
            pytest.skip("NCCL EP is not available")
        if wire_dtype == "mxfp8" and not is_nccl_ep_fp8_dispatch_available():
            pytest.skip("NCCL EP MXFP8 wire needs EpBuffer quant-recipe support and MXFP8 hardware")

        config.ENABLE_EXPERIMENTAL = True

        self.container = MoEModelTestContainer(
            tp_size=1,
            ep_size=4,
            pp_size=1,
            num_moe_experts=8,
            num_layers=4,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="flex",
            moe_permute_fusion=True,
            hidden_size=1024,
            moe_flex_dispatcher_backend="ncclep",
            test_dtype=torch.bfloat16,
            moe_grouped_gemm=True,
            moe_use_legacy_grouped_gemm=False,
            moe_paged_stash=True,
            moe_expert_rank_capacity_factor=0.6,
            use_transformer_engine_op_fuser=True,
            moe_mlp_glu_interleave_size=32,
            moe_router_padding_for_quantization=True,
            gated_linear_unit=True,
            activation_func=F.silu,
            moe_dispatch_fwd_dtype=wire_dtype,
            moe_combine_bwd_dtype=wire_dtype,
        )

        seq_length = 1024
        batch_size = 1
        topk = self.container.config.moe_router_topk
        capacity_factor = self.container.config.moe_expert_rank_capacity_factor
        hidden_states = torch.randn(
            (seq_length, batch_size, self.container.config.hidden_size), dtype=torch.bfloat16
        )

        num_tokens = seq_length * batch_size * topk
        pad_multiple = get_align_size_for_quantization(self.container.config)
        budget = int(num_tokens * capacity_factor)
        budget += -budget % pad_multiple

        paged_stash_reset(True, config=self.container.config)
        paged_stash_init_chunk_handler(1, 0)
        _forward_backward_all_layers(self.container, hidden_states)

        # NCCL EP's manager keeps token_probs/token_indices rather than the routing map, and a
        # rank's received load depends on every rank's routing, so the HybridEP map-derived
        # cross-check does not transfer. Check the device-side accounting against the budget
        # instead: required_recv is filled by ep_prepare before any dropping, and over_budget is
        # the same comparison made on device, so the two must agree with config arithmetic.
        budget_report = [
            (
                layer.token_dispatcher._comm_manager._recv_capacity,
                bool(layer.token_dispatcher.check_over_budget().item()),
                int(layer.token_dispatcher.check_required_capacity().item()),
            )
            for layer in self.container.moe_layers
        ]
        any_over_budget = False
        for layer_idx, (recv_capacity, over_budget, required) in enumerate(budget_report):
            assert recv_capacity == budget, (
                f"layer {layer_idx}: dispatcher budget ({recv_capacity}) != expected "
                f"({budget}) for capacity factor {capacity_factor}"
            )
            assert required > 0, f"layer {layer_idx}: required capacity was never recorded"
            assert over_budget == (required > budget), (
                f"layer {layer_idx}: over_budget={over_budget} disagrees with required "
                f"({required}) vs budget ({budget})"
            )
            any_over_budget |= over_budget

        assert any_over_budget, (
            f"no layer exceeded budget {budget} at capacity factor {capacity_factor}; "
            "the test is not exercising overflow"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.parametrize("zero_copy", [False, True])
    def test_over_budget_recovery(self, zero_copy):
        """Degrade an over-budget step to a dropless replay, then restore the grown static budget."""
        if not is_nccl_ep_available():
            pytest.skip("NCCL EP is not available")
        if zero_copy and not is_nccl_ep_zero_copy_available():
            pytest.skip("NCCL EP zero-copy TE API is not available")

        config.ENABLE_EXPERIMENTAL = True

        self.container = MoEModelTestContainer(
            tp_size=1,
            ep_size=4,
            pp_size=1,
            num_moe_experts=8,
            num_layers=4,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="flex",
            moe_permute_fusion=True,
            hidden_size=1024,
            moe_flex_dispatcher_backend="ncclep",
            test_dtype=torch.bfloat16,
            moe_grouped_gemm=True,
            moe_use_legacy_grouped_gemm=False,
            moe_paged_stash=True,
            moe_expert_rank_capacity_factor=0.6,
            moe_ncclep_zero_copy=zero_copy,
            use_transformer_engine_op_fuser=True,
            moe_mlp_glu_interleave_size=32,
            moe_router_padding_for_quantization=True,
            gated_linear_unit=True,
            activation_func=F.silu,
        )

        seq_length = 1024
        batch_size = 1
        hidden_states = torch.randn(
            (seq_length, batch_size, self.container.config.hidden_size), dtype=torch.bfloat16
        )

        # 1. Undersized budget: the step drops tokens and reports it.
        out_dropped = self._run_step(hidden_states)
        over_1 = [l.token_dispatcher.check_over_budget().item() for l in self.container.moe_layers]
        req_1 = [
            l.token_dispatcher.check_required_capacity().item() for l in self.container.moe_layers
        ]

        required_t = torch.tensor([max(req_1)], dtype=torch.int64, device="cuda")
        torch.distributed.all_reduce(required_t, op=torch.distributed.ReduceOp.MAX)
        required = int(required_t.item())
        budget_before = self.container.moe_layers[0].token_dispatcher._comm_manager._recv_capacity

        # 2. prepare_for_rerun: clear the capacity factor (-> eager, which has no budget to
        #    exceed), record the peak to grow to, release the EP context, replay dropless.
        for layer in self.container.moe_layers:
            layer.token_dispatcher.reset_over_budget()
            layer.token_dispatcher._comm_manager.moe_expert_rank_capacity_factor = None
            layer.token_dispatcher.grow_ep_recv_capacity(required)
            layer.token_dispatcher.invalidate_ep_bootstrap()
        nccl_ep_release_context()

        out_replay = self._run_step(hidden_states)
        eager_2 = [l.token_dispatcher._comm_manager.eager for l in self.container.moe_layers]
        zc_2 = [l.token_dispatcher._comm_manager.zero_copy for l in self.container.moe_layers]

        dropped_finite = bool(torch.isfinite(out_dropped).all())
        replay_finite = bool(torch.isfinite(out_replay).all())
        # atol=0 so the comparison is purely relative: these activations are ~1e-15, and
        # allclose's default atol=1e-8 would call any result "close", including a completely
        # wrong one.
        dropped_differs = not torch.allclose(out_dropped, out_replay, rtol=1e-2, atol=0)

        # 3. Success branch: restore the capacity factor -> static returns at the grown budget.
        for layer in self.container.moe_layers:
            layer.token_dispatcher.reset_over_budget()
            layer.token_dispatcher._comm_manager.moe_expert_rank_capacity_factor = (
                self.container.config.moe_expert_rank_capacity_factor
            )
            layer.token_dispatcher.invalidate_ep_bootstrap()
        nccl_ep_release_context()

        out_restored = self._run_step(hidden_states)
        eager_3 = [l.token_dispatcher._comm_manager.eager for l in self.container.moe_layers]
        zc_3 = [l.token_dispatcher._comm_manager.zero_copy for l in self.container.moe_layers]
        caps_3 = [
            l.token_dispatcher._comm_manager._recv_capacity for l in self.container.moe_layers
        ]
        over_3 = [l.token_dispatcher.check_over_budget().item() for l in self.container.moe_layers]

        assert required > budget_before, (
            f"nothing exceeded budget {budget_before} at capacity factor "
            f"{self.container.config.moe_expert_rank_capacity_factor}; not exercising overflow"
        )
        assert all(eager_2), f"replay did not degrade to eager: {eager_2}"
        assert not any(zc_2), f"replay must drop zero-copy while eager: {zc_2}"
        assert replay_finite, "eager replay produced non-finite values"
        assert (
            dropped_finite
        ), "the dropped step produced non-finite values; dropped tokens must contribute 0"
        assert not any(eager_3), f"restore did not return to static: {eager_3}"
        assert all(
            z == zero_copy for z in zc_3
        ), f"restore did not return zero_copy to {zero_copy}: {zc_3}"
        assert all(
            c >= required > budget_before for c in caps_3
        ), f"budget did not grow: {budget_before} -> {caps_3}, observed peak {required}"
        assert not any(over_3), f"still over budget after growing to {required}: {over_3}"
        # Only ranks that actually dropped can differ: overflow is per-rank, so a rank whose
        # experts stayed under budget legitimately reproduces the same output.
        if any(over_1):
            assert dropped_differs, (
                "this rank was over budget, so the dropped step must differ from the dropless "
                "replay"
            )
        # The correctness check: two different execution modes must agree
        torch.testing.assert_close(out_restored, out_replay, rtol=1e-2, atol=0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.parametrize("zero_copy", [False, True])
    @pytest.mark.parametrize("wire_dtype", ["bf16", "mxfp8"])
    def test_forward_backward_4_layers(self, zero_copy, wire_dtype):
        """Test paged stashing with 4 MoE layers on ncclep static shape: two passes match.

        zero_copy=True additionally exercises the ncclEP symm-mem zero-copy IO under paged stash.
        wire_dtype="mxfp8" additionally sends the dispatch-fwd / combine-bwd payloads as the
        MXFP8 carrier; both passes use the same wire dtype, so quantization is common-mode and
        the two-pass determinism tolerance is unchanged."""
        if not is_nccl_ep_available():
            pytest.skip("NCCL EP is not available")
        if zero_copy and not is_nccl_ep_zero_copy_available():
            pytest.skip("NCCL EP zero-copy TE API is not available")
        if wire_dtype == "mxfp8" and not is_nccl_ep_fp8_dispatch_available():
            pytest.skip("NCCL EP MXFP8 wire needs EpBuffer quant-recipe support and MXFP8 hardware")

        config.ENABLE_EXPERIMENTAL = True

        self.container = MoEModelTestContainer(
            tp_size=1,
            ep_size=4,
            pp_size=1,
            num_moe_experts=8,
            num_layers=4,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="flex",
            moe_permute_fusion=True,
            hidden_size=1024,
            moe_flex_dispatcher_backend="ncclep",
            test_dtype=torch.bfloat16,
            moe_grouped_gemm=True,
            moe_use_legacy_grouped_gemm=False,
            moe_paged_stash=True,
            moe_expert_rank_capacity_factor=1.5,
            use_transformer_engine_op_fuser=True,
            moe_mlp_glu_interleave_size=32,
            moe_router_padding_for_quantization=True,
            gated_linear_unit=True,
            activation_func=F.silu,
            moe_ncclep_zero_copy=zero_copy,
            moe_dispatch_fwd_dtype=wire_dtype,
            moe_combine_bwd_dtype=wire_dtype,
        )

        seq_length = 1024
        batch_size = 1
        hidden_states = torch.randn(
            (seq_length, batch_size, self.container.config.hidden_size), dtype=torch.bfloat16
        )

        # First iteration: capture schedule, capacity, etc.
        paged_stash_reset(True, config=self.container.config)
        paged_stash_init_chunk_handler(1, 0)
        output_ref, hidden_states_grad_ref, routing_map_ref, tokens_per_expert_ref = (
            _forward_backward_all_layers(self.container, hidden_states)
        )

        self.container.zero_grad()

        # Second iteration: run with paged stash.
        paged_stash_reset(True, config=self.container.config)
        paged_stash_init_chunk_handler(1, 0)
        output, hidden_states_grad, routing_map, tokens_per_expert = _forward_backward_all_layers(
            self.container, hidden_states
        )

        overflow = check_paged_stash_overflow()
        assert overflow.any().item() == 0

        assert torch.allclose(
            output, output_ref, atol=1e-4, rtol=1e-4
        ), f"output != output_ref: max diff = {(output - output_ref).abs().max().item()}"
        assert torch.allclose(hidden_states_grad, hidden_states_grad_ref, atol=1e-4, rtol=1e-4), (
            f"hidden_states_grad != ref: max diff = "
            f"{(hidden_states_grad - hidden_states_grad_ref).abs().max().item()}"
        )
        if routing_map is not None and tokens_per_expert is not None:
            num_tokens_per_ep_rank = tokens_per_expert.sum().item()
            assert (
                num_tokens_per_ep_rank > 0
            ), f"num_tokens_per_ep_rank={num_tokens_per_ep_rank} (expected > 0)"
            assert routing_map_ref is not None and tokens_per_expert_ref is not None
            tpe_f = tokens_per_expert.float()
            ref_f = tokens_per_expert_ref.float()
            assert torch.allclose(
                tpe_f, ref_f, atol=1e-4, rtol=1e-4
            ), f"tokens_per_expert != ref: max diff = {(tpe_f - ref_f).abs().max().item()}"
