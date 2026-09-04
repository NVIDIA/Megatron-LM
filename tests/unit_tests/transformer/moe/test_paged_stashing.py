# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import inspect
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core import config
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.moe_utils import get_align_size_for_quantization
from megatron.core.transformer.moe.paged_stash import (
    PagedStashManager,
    PagedStashRunner,
    check_paged_stash_overflow,
    mark_paged_stash_recompute_managed,
    paged_stash_init_chunk_handler,
    paged_stash_reset,
    paged_stash_te_graph_capture,
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


def test_recompute_managed_tensor_bypasses_paged_stash_save_hook():
    tensor = torch.randn(8, 4)
    tensor.grouped_tensor_scale_inv = False
    mark_paged_stash_recompute_managed(tensor)

    # The ownership check happens before the manager needs CUDA streams or capture state.
    manager = object.__new__(PagedStashManager)
    assert manager.on_save_for_backward(tensor) is tensor


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
    # NCCL EP static-shape paged stashing aborts in dev CI with a pybind11 GIL dec_ref failure.
    @pytest.mark.flaky_in_dev
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
