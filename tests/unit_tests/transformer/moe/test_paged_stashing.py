# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core import config
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.moe_utils import get_align_size_for_quantization
from megatron.core.transformer.moe.paged_stash import (
    PagedStashBuffer,
    PagedStashManager,
    PagedStashRunner,
    PagedTensor,
    PipelinePostScheduleFunction,
    check_paged_stash_overflow,
    paged_stash_init_chunk_handler,
    paged_stash_reset,
)
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils

# These tests configure mxfp8 + the TE op fuser, so they only run on Blackwell (sm100+). Mark the
# whole module for the GB200 CI bucket (selection there is marker-driven; see
# tests/unit_tests/find_test_cases.py and recipes/gb200/unit-tests.yaml).
pytestmark = pytest.mark.launch_on_gb200


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
            moe_ncclep_static_shape=kwargs.get("moe_ncclep_static_shape", False),
            moe_grouped_gemm=kwargs.get("moe_grouped_gemm", False),
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
            moe_router_force_biased=kwargs.get("moe_router_force_biased", None),
            moe_paged_stash_buffer_size_factor_cuda=0.5,
            moe_paged_stash_buffer_size_factor_cpu=1.5,
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


def is_nccl_ep_available():
    from megatron.core.transformer.moe.fused_a2a import HAVE_TE_EP

    return HAVE_TE_EP


def _te_grouped_mlp_op_fuser_environment_supported() -> bool:
    """Cheap gate matching the start of ``TEGroupedMLP._is_fused_impl_supported`` (experts.py)."""
    if not HAVE_TE:
        return False
    try:
        from transformer_engine.pytorch.ops import GroupedLinear, ScaledSwiGLU  # noqa: F401
    except ImportError:
        return False
    return is_te_min_version("2.14.0")


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


@pytest.mark.parametrize(
    "enabled,config_overrides,expected_runtime,expected_stash",
    [
        (False, {}, False, False),
        (True, {"cuda_graph_impl": "none"}, False, False),
        (True, {"cuda_graph_granularity": "layer"}, False, False),
        (True, {"cuda_graph_dynamic_microbatches": False}, False, False),
        (True, {"moe_paged_stash": False}, False, False),
        (True, {"pipeline_model_parallel_size": 1}, True, False),
        (True, {}, True, True),
    ],
)
def test_chunk_graph_runtime_schedule_is_strictly_gated(
    monkeypatch, enabled, config_overrides, expected_runtime, expected_stash
):
    """Legacy eager/layer/static schedules must retain their existing paged-stash behavior."""
    manager = PagedStashManager.__new__(PagedStashManager)
    manager.iteration = 0
    manager.status = 'begin'
    monkeypatch.setattr(PagedStashManager, 'get_instance', lambda: manager)

    config_values = {
        'cuda_graph_impl': 'local',
        'cuda_graph_granularity': 'chunk',
        'cuda_graph_dynamic_microbatches': True,
        'moe_paged_stash': True,
        'pipeline_model_parallel_size': 2,
        'moe_paged_stash_page_size': 64,
    }
    config_values.update(config_overrides)
    paged_stash_reset(enabled=enabled, config=SimpleNamespace(**config_values))

    assert manager.chunk_graph_runtime_schedule is expected_runtime
    assert manager.chunk_graph_runtime_stash_activations is expected_stash


def test_chunk_graph_missing_slots_extend_capacity_profile(monkeypatch):
    manager = PagedStashManager.__new__(PagedStashManager)
    manager.iteration = 1
    manager.status = 'capture'
    manager.vp_size = 2
    manager.current_layer = [7, 8]
    manager.current_microbatch = [3, 4]
    manager._pp_schedule = [1001001]
    manager.defer_chunk_graph_profile_completion = True
    monkeypatch.setattr(PagedStashManager, 'get_instance', lambda: manager)

    config = SimpleNamespace(
        cuda_graph_impl='local',
        cuda_graph_granularity='chunk',
        cuda_graph_dynamic_microbatches=True,
        moe_paged_stash=True,
        pipeline_model_parallel_size=2,
        moe_paged_stash_page_size=64,
    )
    paged_stash_reset(enabled=True, config=config)

    assert manager.status == 'capture'
    assert not manager.defer_chunk_graph_profile_completion
    assert manager._pp_schedule == []
    assert manager.current_layer == [1, 1]
    assert manager.current_microbatch == [0, 0]


@pytest.mark.parametrize("stash_activations", [False, True])
def test_chunk_graph_runtime_schedule_uses_matching_activation(monkeypatch, stash_activations):
    """PP1 discards immediately; PP>1 stashes and reloads the matching activation."""
    manager = PagedStashManager.__new__(PagedStashManager)
    manager.status = 'captured'
    manager.chunk_graph_runtime_schedule = True
    manager.chunk_graph_runtime_stash_activations = stash_activations
    manager.vp_size = 1
    manager.current_vp_stage = 0
    manager.current_layer = [1]
    manager.current_microbatch = [7]
    manager.current_schedule_index = 0
    manager._pp_schedule = [999999999]
    manager._unpack_stream_status = 'idle'

    events = []
    manager.stash_paged_tensors = lambda key: events.append(('stash', key))
    manager.remove_paged_tensor_from_stash = lambda: events.append(('discard', None))
    manager.reload_paged_tensors = lambda key: events.append(('reload', key))
    manager.wait_for_stash_to_complete = lambda: events.append(('wait', None))
    monkeypatch.setattr(
        torch.cuda, 'current_stream', lambda: SimpleNamespace(wait_stream=lambda _: None)
    )

    ctx = SimpleNamespace()
    tensor = torch.ones(1, requires_grad=True)
    assert PipelinePostScheduleFunction.forward(ctx, tensor, manager) is tensor
    assert manager.current_schedule_index == 1

    PipelinePostScheduleFunction.backward(ctx, torch.ones_like(tensor))

    schedule_layer = manager.get_schedule_layer(1, 1, 7)
    expected = (
        [('stash', schedule_layer), ('reload', schedule_layer), ('wait', None)]
        if stash_activations
        else [('discard', None), ('wait', None)]
    )
    assert events == expected
    assert manager.current_schedule_index == 2


def test_paged_stash_retry_preserves_batch_structure():
    """Retry input must not inherit in-place container changes from the first attempt."""
    runner = PagedStashRunner.__new__(PagedStashRunner)
    original_tokens = torch.arange(8)
    source = iter([{'tokens': original_tokens, 'metadata': {'shape': [8]}}])

    retry_iterators, attempt_iterators = runner.data_read(
        source, model=[object()], training=True, num_microbatches=1
    )
    first_attempt = next(attempt_iterators[0])
    first_attempt['tokens'] = first_attempt['tokens'].view(1, -1)
    first_attempt['metadata']['shape'][0] = 1

    _, replay_iterators = runner.data_read(
        retry_iterators, model=[object()], training=True, num_microbatches=1
    )
    replay = next(replay_iterators[0])

    assert replay['tokens'].shape == (8,)
    assert replay['tokens'].data_ptr() == original_tokens.data_ptr()
    assert replay['metadata']['shape'] == [8]


def test_local_chunk_graph_retry_runs_eager_and_restores_config():
    """A dropless fallback must not replay the static-shape chunk graph."""
    config = SimpleNamespace(
        cuda_graph_impl="local", cuda_graph_granularity="chunk", moe_paged_stash=True
    )
    runner = PagedStashRunner.__new__(PagedStashRunner)
    runner.config = config
    runner._configs_to_sync_moe_paged_stash = [config]
    runner.moe_layers = []
    runner.model = []

    seen_graph_impls = []
    runner.forward_backward_func = lambda **_kwargs: seen_graph_impls.append(config.cuda_graph_impl)
    runner.data_read = lambda data_iterator, _model, _training, _num_microbatches: (
        data_iterator,
        [],
    )
    overflow_results = iter(((1, 0, 0), (0, 0, 0)))
    runner.check_moe_overflow = lambda: next(overflow_results)
    runner.prepare_for_rerun = lambda is_training: None

    runner(model=[], data_iterator=None, num_microbatches=1, seq_length=1, forward_only=False)

    assert seen_graph_impls == ['local', 'none']
    assert config.cuda_graph_impl == 'local'
    assert config.moe_paged_stash


@pytest.mark.parametrize("graph_created,expected_release_calls", [(False, 1), (True, 0)])
def test_local_chunk_graph_retry_preserves_captured_stash_buffers(
    monkeypatch, graph_created, expected_release_calls
):
    """Captured local graphs retain stash-buffer addresses across an eager retry."""
    from megatron.core.transformer.cuda_graphs import _CudagraphGlobalRecord
    from megatron.core.transformer.moe import moe_logging

    config = SimpleNamespace(
        cuda_graph_impl="local", cuda_graph_granularity="chunk", moe_paged_stash=True
    )
    release_calls = []
    stash_manager = SimpleNamespace(
        overflow=None, host_spill=None, release_stash_buffers=lambda: release_calls.append(True)
    )
    runner = PagedStashRunner.__new__(PagedStashRunner)
    runner.config = config
    runner._configs_to_sync_moe_paged_stash = [config]
    runner.moe_layers = []
    runner.model = []
    runner.optimizer = None
    runner.copy_main_params = False
    runner.forward_backward_func = lambda **_kwargs: None
    runner.stash_manager = stash_manager
    metric_clear_calls = []
    overload_clear_calls = []
    monkeypatch.setattr(
        moe_logging,
        "get_moe_metrics_tracker",
        lambda: SimpleNamespace(clear=lambda: metric_clear_calls.append(True)),
    )
    monkeypatch.setattr(
        moe_logging,
        "get_moe_overload_factor_tracker",
        lambda: SimpleNamespace(clear=lambda: overload_clear_calls.append(True)),
    )
    monkeypatch.setattr(_CudagraphGlobalRecord, "cudagraph_created", graph_created)

    runner.prepare_for_rerun(is_training=True)

    assert len(release_calls) == expected_release_calls
    assert metric_clear_calls == [True]
    assert overload_clear_calls == [True]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.internal
def test_paged_stash_cuda_graph_replay_accepts_variable_token_counts():
    """A captured stash/pop pair must consume the replay-time device token count."""
    device = torch.device("cuda")
    max_num_tokens = 24
    hidden_size = 16
    page_size = 8
    overflow = torch.zeros(1, dtype=torch.int64, device=device)
    host_spill = torch.zeros(1, dtype=torch.int64, device=device)
    stash_buffer = PagedStashBuffer(
        num_tokens=page_size,
        hidden_size=hidden_size,
        page_size=page_size,
        device=device,
        overflow=overflow,
        host_spill=host_spill,
        dtype=torch.float32,
        num_tokens_host=max_num_tokens,
    )
    source = torch.empty((max_num_tokens, hidden_size), dtype=torch.float32, device=device)
    restored = torch.empty_like(source)
    num_tokens = torch.ones(1, dtype=torch.int64, device=device)
    paged_tensor = PagedTensor(
        source,
        num_tokens_tensor=num_tokens,
        original_shape=source.shape,
        max_num_tokens=max_num_tokens,
        hidden_size=hidden_size,
        page_size=page_size,
    )

    # Compile the Triton kernels before CUDA graph capture.
    paged_tensor.offload_to_stash(stash_buffer)
    paged_tensor._tensor = restored
    paged_tensor.reload_from_stash(stash_buffer, zero_padded_tokens=True)
    torch.cuda.synchronize()
    stash_buffer.reset()
    overflow.zero_()
    host_spill.zero_()

    paged_tensor._tensor = source
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        paged_tensor.offload_to_stash(stash_buffer)
        paged_tensor._tensor = restored
        paged_tensor.reload_from_stash(stash_buffer, zero_padded_tokens=True)

    for replay_index, replay_num_tokens in enumerate((3, 17, 8, 23)):
        expected = torch.arange(
            max_num_tokens * hidden_size, dtype=torch.float32, device=device
        ).view(max_num_tokens, hidden_size)
        expected.add_(replay_index * expected.numel())
        source.copy_(expected)
        restored.fill_(-1)
        num_tokens.fill_(replay_num_tokens)
        overflow.zero_()
        host_spill.zero_()

        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(
            restored[:replay_num_tokens], expected[:replay_num_tokens], rtol=0, atol=0
        )
        torch.testing.assert_close(
            restored[replay_num_tokens:],
            torch.zeros_like(restored[replay_num_tokens:]),
            rtol=0,
            atol=0,
        )
        assert not overflow.item()
        assert host_spill.item() == int(replay_num_tokens > page_size)
        torch.testing.assert_close(
            stash_buffer.free_list_tail - stash_buffer.free_list_head,
            stash_buffer.free_list_capacity,
            rtol=0,
            atol=0,
        )

    # The default reload mode is the legacy path: only the valid prefix is restored and the
    # caller-owned tail remains untouched.
    legacy_num_tokens = 3
    expected = torch.arange(max_num_tokens * hidden_size, dtype=torch.float32, device=device).view(
        max_num_tokens, hidden_size
    )
    source.copy_(expected)
    restored.fill_(-1)
    num_tokens.fill_(legacy_num_tokens)
    overflow.zero_()
    host_spill.zero_()
    paged_tensor._tensor = source
    paged_tensor.offload_to_stash(stash_buffer)
    paged_tensor._tensor = restored
    paged_tensor.reload_from_stash(stash_buffer)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        restored[:legacy_num_tokens], expected[:legacy_num_tokens], rtol=0, atol=0
    )
    torch.testing.assert_close(
        restored[legacy_num_tokens:],
        torch.full_like(restored[legacy_num_tokens:], -1),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        stash_buffer.free_list_tail - stash_buffer.free_list_head,
        stash_buffer.free_list_capacity,
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.internal
def test_paged_stash_cuda_graph_replay_reports_empty_pool_overflow():
    """A captured zero-capacity stash must report overflow without corrupting its free list."""
    device = torch.device("cuda")
    num_tokens_value = 8
    hidden_size = 16
    page_size = 8
    overflow = torch.zeros(1, dtype=torch.int64, device=device)
    host_spill = torch.zeros(1, dtype=torch.int64, device=device)
    stash_buffer = PagedStashBuffer(
        num_tokens=0,
        hidden_size=hidden_size,
        page_size=page_size,
        device=device,
        overflow=overflow,
        host_spill=host_spill,
        dtype=torch.float32,
        num_tokens_host=0,
    )
    source = torch.randn((num_tokens_value, hidden_size), dtype=torch.float32, device=device)
    num_tokens = torch.tensor([num_tokens_value], dtype=torch.int64, device=device)
    paged_tensor = PagedTensor(
        source,
        num_tokens_tensor=num_tokens,
        original_shape=source.shape,
        max_num_tokens=num_tokens_value,
        hidden_size=hidden_size,
        page_size=page_size,
    )

    # Compile before capture, then reset the device-visible state used by replay.
    paged_tensor.offload_to_stash(stash_buffer)
    torch.cuda.synchronize()
    assert overflow.item() == 1
    overflow.zero_()
    host_spill.zero_()
    stash_buffer.reset()

    paged_tensor._tensor = source
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        paged_tensor.offload_to_stash(stash_buffer)
    torch.cuda.synchronize()

    overflow.zero_()
    host_spill.zero_()
    graph.replay()
    torch.cuda.synchronize()

    assert overflow.item() == 1
    assert host_spill.item() == 0
    torch.testing.assert_close(
        stash_buffer.free_list_head, torch.zeros_like(stash_buffer.free_list_head), rtol=0, atol=0
    )
    torch.testing.assert_close(
        stash_buffer.free_list_tail, torch.zeros_like(stash_buffer.free_list_tail), rtol=0, atol=0
    )


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

    ncclep's CUDA-graph / paged-stash path requires moe_ncclep_static_shape=True, which feeds the
    experts the full fixed-size recv buffer and is only valid with fp8/fp4 + the CuTe DSL grouped
    GEMM (the container always configures mxfp8; NVTE_CUTEDSL_FUSED_GROUPED_MLP=1 must be set in the
    environment). This mirrors TestPagedStashing: run the paged-stash path twice and assert the two
    passes agree (a determinism guard for the static ncclep path), plus no paged-stash overflow.
    """

    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    def test_forward_backward_4_layers(self):
        """Test paged stashing with 4 MoE layers on ncclep static shape: two passes match."""
        if not is_nccl_ep_available():
            pytest.skip("NCCL EP is not available")

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
            moe_flex_dispatcher_backend="ncclep",
            moe_ncclep_static_shape=True,
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
