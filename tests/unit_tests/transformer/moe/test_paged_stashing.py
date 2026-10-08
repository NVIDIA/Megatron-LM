# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import gc
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core import config
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.transformer.moe.fused_a2a import is_nccl_ep_bootstrapped, reset_hybrid_ep_buffer
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.moe_utils import get_align_size_for_quantization
from megatron.core.transformer.moe.ops.paged_stash import GLOBAL_BLOCK_SIZE, GLOBAL_MAX_BLOCKS
from megatron.core.transformer.moe.paged_stash import (
    PagedStashBuffer,
    PagedStashManager,
    PagedTensor,
    _stash_buffer_dtype,
    check_paged_stash_overflow,
    paged_stash_init_chunk_handler,
    paged_stash_reset,
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


@pytest.fixture(autouse=True)
def reset_paged_stash_manager():
    """The manager's state machine never returns to 'begin' on its own."""
    yield
    PagedStashManager.STASH_MGR = None


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
            moe_ncclep_zero_copy=kwargs.get("moe_ncclep_zero_copy", False),
            moe_dispatch_fwd_dtype=kwargs.get("moe_dispatch_fwd_dtype", 'bf16'),
            moe_combine_bwd_dtype=kwargs.get("moe_combine_bwd_dtype", 'bf16'),
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
            moe_use_legacy_grouped_gemm=False,
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
