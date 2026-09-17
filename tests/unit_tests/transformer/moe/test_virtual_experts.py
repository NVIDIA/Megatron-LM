# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Focused virtual-expert regressions: planning, routing, GTP transport and training parity.

Run on one node with four Blackwell GPUs::

    python -m torch.distributed.run --standalone --nproc-per-node=4 -m pytest -q \
        tests/unit_tests/transformer/moe/test_virtual_experts.py

Eight-GPU H100 nodes run the non-MXFP8 cases; MXFP8 compute skips on pre-Blackwell
hardware. The existing GB200 CI selection uses ``launch_on_gb200``. All inputs are
synthetic; no dataset or custom launcher is required.
"""

import gc
import os
from dataclasses import replace
from functools import partialmethod

import pytest
import torch
import torch.distributed as dist
from torch.nn import functional as F
from transformer_engine.pytorch import cpu_offload as te_cpu_offload
from transformer_engine.pytorch.attention.dot_product_attention import (
    backends as te_attention_backends,
)
from transformer_engine.pytorch.quantization import FP8GlobalStateManager

from megatron.core.activations import squared_relu
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.fp8_utils import get_fp8_context, is_mxfp8tensor
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as offload,
)
from megatron.core.pipeline_parallel.fine_grained_activation_offload import PipelineOffloadManager
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.quantization.quant_config import RecipeConfig
from megatron.core.tensor_parallel import generalized_tensor_parallelism as gtp
from megatron.core.tensor_parallel import gtp_cuda_graphs
from megatron.core.tensor_parallel.generalized_tensor_parallelism import GTPShardedParam
from megatron.core.tensor_parallel.gtp_api import HAVE_GTP
from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.moe import fused_a2a, moe_utils
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.moe_logging import destroy_moe_metrics_tracker
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.virtual_expert_load_balancer import (
    VirtualExpertLoadBalancer,
    plan_virtual_expert_routes,
)
from megatron.core.transformer.moe.virtual_expert_triton import VirtualExpertPlannerWorkspace
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.internal

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
requires_hybridep = pytest.mark.skipif(
    not torch.cuda.is_available() or not fused_a2a.HAVE_HYBRIDEP,
    reason="requires CUDA and HybridEP",
)


@pytest.fixture(scope="session", autouse=True)
def ensure_test_data():
    """These synthetic regressions need no downloaded unit-test datasets."""


@pytest.fixture(scope="module", autouse=True)
def supported_world():
    """Support four-GPU Blackwell and eight-GPU H100 nodes."""
    if int(os.environ.get("WORLD_SIZE", "1")) not in (4, 8):
        pytest.skip("virtual-expert regressions require a four- or eight-rank torchrun launch")


@pytest.fixture(scope="module")
def hybridep_kernel_cache(tmp_path_factory):
    """Reuse compiled transport kernels across model instances within this test module."""
    return tmp_path_factory.mktemp("virtual-expert-hybridep")


@pytest.fixture(autouse=True)
def virtual_expert_environment(monkeypatch, hybridep_kernel_cache):
    """Keep the recipe's kernel/offload settings local to these tests."""
    monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")
    monkeypatch.setenv("NVTE_USE_CUTLASS_GROUPED_GEMM", "0")
    monkeypatch.setenv("NVTE_CPU_OFFLOAD_V1", "1")
    monkeypatch.setenv("HYBRID_EP_CACHE_DIR", str(hybridep_kernel_cache))
    # TE may already be imported by another test/conftest and caches this flag at
    # import time. Match a V1 launch without reloading TE or affecting other tests.
    for module in (te_cpu_offload, te_attention_backends):
        if hasattr(module, "NVTE_CPU_OFFLOAD_V1"):
            monkeypatch.setattr(module, "NVTE_CPU_OFFLOAD_V1", True)


@pytest.fixture
def mxfp8_grouped_mlp(monkeypatch, virtual_expert_environment):
    """Enable TE's real fused kernels even when pytest imported TE before setting its flags."""
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("MXFP8 requires Blackwell")
    from transformer_engine.pytorch.ops.fused import grouped_mlp
    from transformer_engine.pytorch.ops.fuser import (
        OperationFuser,
        register_forward_backward_fusion,
    )

    # TE caches capability checks and registers these fusions at import time.
    # Refresh the checks under this test's settings; preserve the original registry.
    checks = (
        grouped_mlp._GroupedMLP_CuTeGEMMBase.is_supported,
        grouped_mlp.GroupedMLP_CuTeGEMMUnary.is_supported,
        grouped_mlp._grouped_gemm_dsrelu_backward_supported,
    )
    for check in checks:
        check.cache_clear()
    monkeypatch.setattr(
        OperationFuser,
        "forward_backward_fusion_functions",
        list(OperationFuser.forward_backward_fusion_functions),
    )
    try:
        assert (
            grouped_mlp.GroupedMLP_CuTeGEMMUnary.is_supported()
        ), "MXFP8 parity requires TE's fused SReLU kernels and a compatible cuDNN frontend"
        for cls, fuse in (
            (grouped_mlp.GroupedMLP_CuTeGEMMGLU, grouped_mlp.fuse_ops),
            (grouped_mlp.GroupedMLP_CuTeGEMMUnary, grouped_mlp.fuse_srelu_ops),
        ):
            if cls.is_supported() and fuse not in OperationFuser.forward_backward_fusion_functions:
                register_forward_backward_fusion(fuse, prepend=True)
        yield
    finally:
        for check in checks:
            check.cache_clear()


def _full_weight(p):
    """All-gather the shard over its GTP group: the unpadded, unsharded weight."""
    shards = [torch.empty_like(p.data) for _ in range(p.group.size())]
    dist.all_gather(shards, p.data, group=p.group)
    full = torch.cat(shards, dim=0)
    return full[: full.shape[0] - p.pad_length] if p.pad_length else full


def _assert_clean(p):
    """No in-flight gather and no stale hand-off marker after a consume."""
    assert p._prefetch_handle is None
    assert not getattr(p, "_already_ag_drained", False)


@requires_cuda
@pytest.mark.skipif(not HAVE_GTP, reason="GTP requires TransformerEngine >= 2.19")
class TestVirtualExpertGTP:
    """Weight peeks and reused gradient buffers used by virtual-expert transport."""

    @pytest.fixture(scope="class")
    def four_rank_gtp_group(self):
        """Keep the exact four-shard oracle on both four- and eight-GPU nodes."""
        Utils.initialize_model_parallel()
        groups = [
            dist.new_group(list(range(start, start + 4)))
            for start in range(0, dist.get_world_size(), 4)
        ]
        group = groups[dist.get_rank() // 4]
        try:
            yield group
        finally:
            torch.cuda.synchronize()
            dist.barrier()
            dist.destroy_process_group(group)
            Utils.destroy_model_parallel()

    @pytest.fixture(autouse=True)
    def reset_gtp_state(self):
        """Release GTP/FP8 state after each transport case."""
        yield
        FP8GlobalStateManager.reset()
        gtp.reset_gtp_state()

    @pytest.mark.parametrize("fwd", [True, False], ids=["forward", "backward"])
    @pytest.mark.parametrize("readiness", ["missing", "in_flight", "ready"])
    def test_peek_returns_ready_current_data(
        self, monkeypatch, four_rank_gtp_group, fwd, readiness
    ):
        """Every readiness path exposes current bytes before consume, with exactly one gather.

        Two same-key cache buffers prevent a key-only lookup from identifying the actual AG output.
        The first consume can replace its ticket; subsequent peeks must follow the actual gather.
        """
        check_states = True
        cache = gtp.GTPWeightCache()
        monkeypatch.setattr(gtp, "_GTP_CACHE", cache)
        monkeypatch.setattr(gtp, "_GTP_GROUPED_BUF_PARITY_COUNTER", {})
        monkeypatch.setattr(gtp.GTP_CONFIG, "check_param_states", check_states)
        monkeypatch.setattr(gtp.GTP_CONFIG, "weight_prefetch", True)
        p = GTPShardedParam(torch.zeros(8, 16, dtype=torch.bfloat16, device="cuda"))
        p.group = four_rank_gtp_group
        p.chain_id = "GTP_remat_grouped_fc1_ungraphed"
        held = [cache.reserve(p, p.dtype, fwd=fwd) for _ in range(2)]
        buffers = [cache.get(ticket) for ticket in held]
        for ticket in held:
            cache.release(ticket)
        assert buffers[0].data_ptr() != buffers[1].data_ptr()

        calls = []
        gather = p._all_gather_weight

        def counted_gather(*args, **kwargs):
            calls.append(kwargs["fwd"])
            return gather(*args, **kwargs)

        monkeypatch.setattr(p, "_all_gather_weight", counted_gather)
        peek = p.peek_group_for_forward if fwd else p.peek_group_for_backward
        consume = p.materialize_group_for_forward if fwd else p.materialize_group_for_backward
        pointers = []
        for step in range(3):
            p.data.fill_(16 * step + dist.get_rank())
            expected = _full_weight(p)
            if readiness != "missing":
                _, p._prefetch_handle = p._all_gather_weight(async_op=True, fwd=fwd)
                assert p._prefetch_handle is not None
                if check_states:
                    assert p.state == gtp.GTPWeightState.ASYNC_WAIT
            if readiness == "ready":
                # An external drain (or another peek) has finished the gather's host-side work.
                p._wait_param_gather()
                p._already_ag_drained = True
            initialized, previous, following = p.prefetch_initialized, p.prev_w, p.next_w
            peeked = peek()
            # Read immediately: a later consume must not supply the missing completion wait.
            torch.testing.assert_close(peeked, expected, rtol=0, atol=0)
            assert p._prefetch_handle is None and p._already_ag_drained
            if check_states:
                assert p.state == gtp.GTPWeightState.DATA_READY
            assert (p.prefetch_initialized, p.prev_w, p.next_w) == (
                initialized,
                previous,
                following,
            )
            # A second peek on a different stream still observes completion and issues no gather.
            with torch.cuda.stream(torch.cuda.Stream()):
                again = peek()
                torch.testing.assert_close(again, expected, rtol=0, atol=0)
            assert again.data_ptr() == peeked.data_ptr()
            gathered = consume()
            assert gathered.data_ptr() == peeked.data_ptr()
            torch.testing.assert_close(gathered, expected, rtol=0, atol=0)
            assert len(calls) == step + 1
            _assert_clean(p)
            pointers.append(peeked.data_ptr())
        assert (
            pointers[1] == pointers[2]
        ), "the established deterministic schedule must reuse storage"

    @pytest.mark.parametrize("eager_first", [True, False], ids=["eager-first", "graph-first"])
    def test_eager_wgrad_preserves_graph_rings(self, monkeypatch, four_rank_gtp_group, eager_first):
        """Eager allocation cannot suppress or rebuild graph rings, even after a reset."""
        monkeypatch.setattr(gtp, "_EAGER_WGRAD_RINGS", {})
        monkeypatch.setattr(gtp_cuda_graphs, "_GRAPH_WGRAD_RINGS", {})
        monkeypatch.setattr(gtp, "_FULL_ITERATION", False)
        monkeypatch.setattr(gtp.GTP_CONFIG, "async_reduction", True)
        monkeypatch.setattr(gtp.GTP_CONFIG, "graph_wgrad_ring_size", 2)
        weights = [gtp.GTPShardedParam(torch.zeros(3, 4, device="cuda")) for _ in range(5)]
        for weight in weights:
            weight.group = four_rank_gtp_group
            weight.chain_id = gtp.GTPChain.GRAPHED.value
            weight.pad_length = 2
            weight.main_grad = torch.zeros_like(weight)
        graphed, eager = weights[:4], weights[-1]
        eager.chain_id = gtp.GTPChain.UNGRAPHED.value
        for previous, current in zip(graphed, graphed[1:]):
            previous.next_w, current.prev_w = current, previous
        monkeypatch.setattr(gtp, "_GTP_PARAMS", weights)

        old_slots = []
        for _ in range(2):
            if eager_first:
                eager_view = eager.get_wgrad_tensor(persistent=True)
            gtp.initialize_graph_wgrad_rings()
            if not eager_first:
                eager_view = eager.get_wgrad_tensor(persistent=True)
            assert len(gtp_cuda_graphs._GRAPH_WGRAD_RINGS) == 1
            key, slots = next(iter(gtp_cuda_graphs._GRAPH_WGRAD_RINGS.items()))
            assert len(slots) == 2
            assert [slot.index for slot in slots] == [0, 1]
            assert all(slot.key == key for slot in slots)
            assert graphed[1]._gtp_graph_wgrad_ring_slot is slots[0]
            assert graphed[2]._gtp_graph_wgrad_ring_slot is slots[1]
            assert graphed[3]._gtp_graph_wgrad_ring_slot is slots[0]
            assert len(gtp._EAGER_WGRAD_RINGS) == 1
            assert eager_view.shape == (10, 4)
            eager_view.fill_(7)
            for slot in slots:
                slot.tensor.fill_(3)
            torch.testing.assert_close(eager_view, torch.full_like(eager_view, 7))
            assert len({slot.tensor.data_ptr() for slot in slots} | {eager_view.data_ptr()}) == 3
            assert all(slot is not old for slot in slots for old in old_slots)
            gtp.initialize_graph_wgrad_rings()
            assert next(iter(gtp_cuda_graphs._GRAPH_WGRAD_RINGS.values())) is slots
            assert eager.get_wgrad_tensor().data_ptr() == eager_view.data_ptr()
            old_slots = [*slots, eager._gtp_eager_wgrad_ring_slot]
            torch.cuda.synchronize()
            gtp.reset_gtp_state()
            assert not gtp_cuda_graphs._GRAPH_WGRAD_RINGS
            assert not gtp._EAGER_WGRAD_RINGS
            for weight in weights:
                assert not hasattr(weight, "_gtp_graph_wgrad_ring_slot")
                assert not hasattr(weight, "_gtp_eager_wgrad_ring_slot")

    @pytest.mark.parametrize("async_reduction", [False, True], ids=["sync", "async"])
    def test_persistent_wgrad_reuse_preserves_every_reduction(
        self, monkeypatch, four_rank_gtp_group, async_reduction
    ):
        """Full gradients, padding and exactly-once completion survive shared/repeated writers."""
        group = four_rank_gtp_group
        rank = dist.get_rank(group)
        monkeypatch.setattr(gtp, "_EAGER_WGRAD_RINGS", {})
        monkeypatch.setattr(gtp, "_GTP_GROUPED_BUF_PARITY_COUNTER", {})
        monkeypatch.setattr(gtp.GTP_CONFIG, "async_reduction", async_reduction)
        monkeypatch.setattr(gtp.GTP_CONFIG, "reduce_scatter_with_fp32_accumulation", True)
        monkeypatch.setattr(gtp.GTP_CONFIG, "calculate_per_token_loss", False)
        # Both FC roles deliberately have the same shape; each has two independent experts.
        layers = []
        expected, completions = {}, {}
        for layer in range(3):
            roles = []
            for role in ("fc1", "fc2"):
                weights = [
                    gtp.GTPShardedParam(torch.zeros(33, 16, dtype=torch.bfloat16, device="cuda"))
                    for _ in range(2)
                ]
                for expert, weight in enumerate(weights):
                    weight.group, weight.pad_length, weight.expert_idx = group, 2, expert
                    weight.chain_id = f"GTP_remat_grouped_{role}_ungraphed"
                    weight.is_routed_expert = True
                    weight._debug_name = f"layers.{layer}.{role}.weight{expert}"
                    weight.main_grad = torch.full_like(weight, 0.5)
                    weight._double_buffer_parity()  # Normally assigned by the forward gathers.
                    expected[id(weight)] = weight.main_grad.clone()
                    completions[id(weight)] = 0

                    def completed(weight=weight):
                        completions[id(weight)] += 1

                    weight.register_grad_accum_hook(None, completed)
                weights[0].weight_list = weights
                roles.append(weights)
            layers.append(roles)
        for previous, current in zip(layers, layers[1:]):
            for prev_weights, weights in zip(previous, current):
                prev_weights[0].next_w, weights[0].prev_w = weights[0], prev_weights[0]

        pointers = {}
        try:
            # Repeating the tail before the cascade drains its first RS exercises early reuse.
            for step in range(3):
                for layer_index in (2, 2, 1, 0):
                    for role_index in (1, 0):
                        weights = layers[layer_index][role_index]
                        if async_reduction:
                            with torch.cuda.stream(gtp.get_rs_stream(weights[0].chain_id, group)):
                                torch.cuda._sleep(1_000_000)
                        grads = []
                        for weight in weights:
                            grad = weight.get_wgrad_tensor(persistent=True)
                            prior = pointers.setdefault(id(weight), grad.data_ptr())
                            assert grad.data_ptr() == prior
                            if step == 1:
                                # Foreign gradients must copy into the same padded ring storage.
                                grad = torch.empty_like(grad)
                            base = (
                                torch.arange(130 * 16, device="cuda").view(130, 16) % 11 - 5
                            ).float() / 4 + (step + layer_index + weight.expert_idx) / 8
                            grad.copy_(base + rank / 4)
                            mean = torch.nn.functional.pad(base + 3 / 8, (0, 0, 0, 2))
                            # Mean 250.25 + main_grad 0.5 rounds to 251 once, or 250.5 if the
                            # RS output is first rounded to BF16. Padding must still stay zero.
                            grad[:, 0] = 251 if rank == 3 else 250
                            mean[:130, 0] = 250.25
                            expected[id(weight)].add_(mean.chunk(4)[rank])
                            grads.append(grad)
                        weights[0].finalize_group_grads(grads)
                torch.cuda.synchronize()
                for layer in layers:
                    for weights in layer:
                        for weight in weights:
                            torch.testing.assert_close(
                                weight.main_grad, expected[id(weight)], rtol=0, atol=0
                            )
                            assert not torch.count_nonzero(
                                weight._gtp_eager_wgrad_ring_slot.tensor[130:]
                            )
                            calls_per_step = 2 if layer is layers[-1] else 1
                            assert completions[id(weight)] == (step + 1) * calls_per_step
                # Two buffers per role/expert, independent of the three-layer model depth.
                assert len(set(pointers.values())) == 8
                assert len(gtp._EAGER_WGRAD_RINGS) == 8
        finally:
            torch.cuda.synchronize()
            for layer in layers:
                for weights in layer:
                    weights[0]._wait_reduce_scatter(finalize_grad=True)


def _assert_numerical_parity(actual, expected, tolerance, name, peak_tolerance=None):
    """Bound aggregate and peak error against each parameter's own signal scale."""
    actual, expected = (actual.float(), expected.float())
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), name
    error = actual - expected
    error_norm, expected_norm = error.norm().item(), expected.norm().item()
    relative_l2 = (
        error_norm / expected_norm if expected_norm else (0 if error_norm == 0 else float('inf'))
    )
    assert (
        error_norm <= tolerance * expected_norm
    ), f'{name}: relative L2 error {relative_l2:.6%} exceeds {tolerance:.6%}'
    torch.testing.assert_close(
        actual,
        expected,
        rtol=0,
        atol=(tolerance if peak_tolerance is None else peak_tolerance)
        * expected.abs().max().item(),
        msg=lambda msg: f'{name}: {msg}',
    )


def _assert_bitwise_parity(actual, expected, name):
    """Require identical finite values and storage bits, including signed zero."""
    assert actual.dtype == expected.dtype, name
    assert actual.shape == expected.shape, name
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), name
    torch.testing.assert_close(
        actual.contiguous().view(torch.uint8),
        expected.contiguous().view(torch.uint8),
        rtol=0,
        atol=0,
        msg=lambda msg: f'{name}: {msg}',
    )


@requires_hybridep
@pytest.mark.launch_on_gb200
@pytest.mark.usefixtures("mxfp8_grouped_mlp")
@pytest.mark.parametrize('egtp_size', [1, 2], ids=['ep2-mixed-mtp', 'ep2-egtp2-mxfp8'])
def test_virtual_expert_hybrid_training_parity(monkeypatch, egtp_size):
    """Match HybridEP losses, gradients and updates with repeated MTP, GTP/EGTP and offload."""
    monkeypatch.setenv('NVTE_GROUPED_LINEAR_SINGLE_PARAM', '0')
    # Compile once per shape/process, including that first compilation in the time budget.
    # Model teardown releases transport buffers; later instances reload the real kernels.
    monkeypatch.setattr(
        fused_a2a.HybridEPBuffer,
        '__init__',
        partialmethod(fused_a2a.HybridEPBuffer.__init__, load_cached_kernels=True),
    )
    # Same EP layout in both variants shares the expensive HybridEP kernel compilation.
    Utils.initialize_model_parallel(
        expert_model_parallel_size=2, gtp_remat_size=4, expert_gtp_remat_size=egtp_size
    )
    monkeypatch.setattr(gtp.GTP_CONFIG, 'async_reduction', True)
    monkeypatch.setattr(gtp.GTP_CONFIG, 'weight_prefetch', True)
    monkeypatch.setattr(gtp.GTP_CONFIG, 'calculate_per_token_loss', True)
    monkeypatch.setattr(gtp.GTP_CONFIG, 'reduce_scatter_with_fp32_accumulation', True)
    pg = ProcessGroupCollection.use_mpu_process_groups()
    recipe = RecipeConfig.from_config_dict(
        {
            'configs': {
                'bf16': {
                    'transformer_engine_config_type': 'TEQuantizationParams',
                    'training_recipe': {},
                }
            },
            'matchers': {
                'mtp': {
                    'type': 'glob',
                    'enabled': True,
                    'pattern': '*mtp.layers.*',
                    'config': 'bf16',
                }
            },
        }
    )

    def train(virtual):
        gtp.reset_gtp_state()
        gtp._GTP_PARAMS.clear()
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        model_parallel_cuda_manual_seed(1234)
        torch.manual_seed(1234)
        config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            ffn_hidden_size=256,
            moe_ffn_hidden_size=256,
            num_moe_experts=4,
            expert_model_parallel_size=2,
            tensor_parallel_num_weight_shards=4,
            expert_tensor_parallel_num_weight_shards=egtp_size,
            gtp_remat_opt_in_modules=['moe_latent_proj'],
            moe_router_topk=2,
            moe_latent_size=128,
            moe_shared_expert_intermediate_size=256,
            moe_router_score_function='sigmoid',
            moe_router_topk_scaling_factor=3.16,
            moe_router_load_balancing_type='quantile_balancing',
            moe_aux_loss_coeff=0,
            moe_router_dtype='fp32',
            moe_router_fusion=False,
            moe_token_dispatcher_type='flex',
            moe_flex_dispatcher_backend='hybridep',
            moe_virtual_expert_load_balance=virtual,
            moe_grouped_gemm=True,
            use_transformer_engine_op_fuser=True,
            moe_use_grouped_tensor=True,
            use_fused_weighted_squared_relu=True,
            activation_func=squared_relu,
            gated_linear_unit=False,
            gradient_accumulation_fusion=True,
            add_bias_linear=False,
            normalization='RMSNorm',
            bf16=True,
            params_dtype=torch.bfloat16,
            hidden_dropout=0,
            attention_dropout=0,
            fp8='e4m3',
            fp8_recipe='mxfp8',
            fp8_param=True,
            moe_router_padding_for_quantization=True,
            # Fused FP8 EGTP backward requires native FP8 weights when re-gathering.
            # The op fuser does not honor the BF16 override's execution policy yet;
            # keep that MTP parameter-storage override in the unsharded-expert case.
            quant_recipe=recipe if egtp_size == 1 else None,
            mtp_num_layers=2,
            mtp_use_repeated_layer=True,
            mtp_hsm=True,
            mtp_loss_scaling_factor=0.1,
            calculate_per_token_loss=True,
            disable_parameter_transpose_cache=True,
            fine_grained_activation_offloading=virtual,
            offload_modules=['fused_group_mlp'] if virtual else [],
            min_offloaded_tensor_size=0,
        )
        module = model = optimizer = None
        history, plans = ([], [])
        try:
            with get_fp8_context(config, is_init=True):
                module = HybridModel(
                    config,
                    hybrid_stack_spec,
                    vocab_size=128,
                    max_sequence_length=32,
                    hybrid_layer_pattern='EE/E/E',
                    pg_collection=pg,
                ).cuda()
            model = DDP(
                config,
                DistributedDataParallelConfig(
                    grad_reduce_in_fp32=False,
                    overlap_grad_reduce=True,
                    use_distributed_optimizer=True,
                    fp8_param_gather=True,
                    overlap_param_gather=True,
                    reuse_grad_buf_for_mxfp8_param_ag=True,
                    reduce_scatter_with_fp32_accumulation=True,
                    check_for_nan_in_grad=False,
                    average_in_collective=False,
                ),
                module,
                pg_collection=pg,
            )
            gtp.tag_gtp_params_with_names(model)
            gtp.classify_gtp_remat_chains(model, cuda_graph_modules=[], cuda_graph_impl='none')
            optimizer = get_megatron_optimizer(
                OptimizerConfig(
                    optimizer='adam',
                    lr=1e-05,
                    bf16=True,
                    use_distributed_optimizer=True,
                    fp8_recipe='mxfp8',
                    reuse_grad_buf_for_mxfp8_param_ag=True,
                    overlap_param_gather=True,
                    clip_grad=0,
                    weight_decay=0,
                ),
                [model],
                use_gloo_process_groups=False,
                pg_collection=pg,
            )
            assert any((getattr(p, 'gtp_remat_size', 1) == 4 for p in model.parameters()))
            experts = [
                (name, layer)
                for name, layer in module.named_modules()
                if isinstance(layer, MoELayer)
            ]
            assert len(experts) == 3
            for name, layer in experts:
                assert is_mxfp8tensor(layer.experts.linear_fc1.weight0) == (
                    egtp_size > 1 or 'mtp' not in name
                )
                for linear in (layer.experts.linear_fc1, layer.experts.linear_fc2):
                    for weight in linear.parameters():
                        assert getattr(weight, 'is_gtp_weight_remat', False) == (egtp_size > 1)
                        assert (
                            weight.numel() * egtp_size == linear.in_features * linear.out_features
                        )
                        if egtp_size > 1:
                            assert weight.group is pg.expt_gtp_remat
                            assert weight.group.size() == egtp_size
                if virtual:
                    manager = layer.token_dispatcher._comm_manager
                    dispatch = manager.plan_dispatch

                    def record(*args, manager=manager, dispatch=dispatch):
                        dispatch(*args)
                        plans.append(manager._plan)

                    monkeypatch.setattr(manager, 'plan_dispatch', record)
            keys = set(module.state_dict())
            for step in range(2):
                optimizer.zero_grad()
                model.zero_grad_buffer()
                for child in optimizer.chained_optimizers:
                    child._copy_main_params_to_param_buffer()
                model.set_is_first_microbatch()
                rng = torch.Generator(device='cuda').manual_seed(5678 + step)
                tokens = torch.randint(0, 128, (2, 16), device='cuda', generator=rng)
                labels = torch.randint(0, 128, (2, 16), device='cuda', generator=rng)
                positions = torch.arange(16, device='cuda').expand(2, -1)
                with get_fp8_context(config):
                    loss = model(
                        tokens,
                        positions,
                        attention_mask=None,
                        labels=labels,
                        loss_mask=torch.ones_like(tokens, dtype=torch.float32),
                    )
                loss.sum().backward()
                gtp.wait_for_gtp_grad_reduction_on_current_stream()
                model.finish_grad_sync()
                torch.cuda.synchronize()
                history.append(
                    {
                        'loss': loss.detach().float().cpu(),
                        **{
                            name: parameter.main_grad.detach().float().cpu().clone()
                            for name, parameter in module.named_parameters()
                        },
                    }
                )
                assert all((layer.router.weight.main_grad.norm() > 0 for _, layer in experts))
                if virtual:
                    assert len(plans) == 4
                    assert (
                        plans[-1].experts_to_copy.data_ptr() != plans[-2].experts_to_copy.data_ptr()
                    )
                    active = (
                        torch.stack([(plan.experts_to_copy >= 0).any() for plan in plans])
                        .any()
                        .int()
                    )
                    torch.distributed.all_reduce(active)
                    assert active.item(), 'parity must move at least one virtual expert'
                    plans.clear()
                before = [p.detach().clone() for p in optimizer.get_parameters()]
                assert optimizer.step()[0], 'optimizer skipped an update'
                assert any(
                    (not torch.equal(p, old) for p, old in zip(optimizer.get_parameters(), before))
                )
                for index, (p, old) in enumerate(zip(optimizer.get_parameters(), before)):
                    history[-1][f'optimizer update {index}'] = (p.detach() - old).float().cpu()
                if config.fine_grained_activation_offloading:
                    offload.reset(process_group=pg.tp_dp_cp)
                    assert PipelineOffloadManager.get_instance().offload_summary_total_bytes > 0
            return (keys, history)
        finally:
            torch.cuda.synchronize()
            del optimizer, model, module
            VirtualExpertLoadBalancer.finalize()
            FP8GlobalStateManager.reset()
            gtp.reset_gtp_state()
            gtp._GTP_PARAMS.clear()
            offload.reset_instance()
            destroy_moe_metrics_tracker()
            gc.collect()
            fused_a2a.reset_hybrid_ep_buffer()

    try:
        reference_keys, reference = train(False)
        actual_keys, actual = train(True)
        assert actual_keys == reference_keys
        for step, (values, expected) in enumerate(zip(actual, reference)):
            assert values.keys() == expected.keys()
            for name in values:
                # MXFP8 can round tiny gradients to zero or change their sign. Adam's
                # first step amplifies this: eight zero/nonzero changes among 16,384
                # elements produced 2.19% update L2 with only 0.26% gradient L2.
                if name == 'loss':
                    tolerance, peak_tolerance = 1e-4, 1e-4
                elif name.startswith('optimizer update'):
                    tolerance, peak_tolerance = 0.025, 2.1
                else:
                    tolerance, peak_tolerance = 0.01, 0.02
                _assert_numerical_parity(
                    values[name],
                    expected[name],
                    tolerance,
                    f'step {step}: {name}',
                    peak_tolerance=peak_tolerance,
                )
    finally:
        Utils.destroy_model_parallel()


@requires_hybridep
def test_bf16_virtual_expert_routing_parity(monkeypatch):
    """Moving routes preserves BF16 outputs/dgrads; only wgrad summation may round differently."""
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.transformer.spec_utils import get_submodules

    monkeypatch.setattr(
        fused_a2a.HybridEPBuffer,
        '__init__',
        partialmethod(fused_a2a.HybridEPBuffer.__init__, load_cached_kernels=True),
    )
    monkeypatch.setenv('NVTE_GROUPED_LINEAR_SINGLE_PARAM', '0')
    Utils.initialize_model_parallel(expert_model_parallel_size=2)
    pg = ProcessGroupCollection.use_mpu_process_groups()
    spec = get_submodules(
        get_gpt_layer_with_transformer_engine_spec(
            num_experts=4, moe_grouped_gemm=True
        ).submodules.mlp
    )

    def run(virtual):
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        model_parallel_cuda_manual_seed(1234)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            ffn_hidden_size=256,
            moe_ffn_hidden_size=256,
            num_moe_experts=4,
            expert_model_parallel_size=2,
            moe_router_topk=2,
            moe_router_score_function='sigmoid',
            moe_router_topk_scaling_factor=3.16,
            moe_router_dtype='fp32',
            moe_router_load_balancing_type='none',
            moe_aux_loss_coeff=0,
            moe_token_dispatcher_type='flex',
            moe_flex_dispatcher_backend='hybridep',
            moe_virtual_expert_load_balance=virtual,
            bf16=True,
            params_dtype=torch.bfloat16,
            add_bias_linear=False,
            activation_func=squared_relu,
            gated_linear_unit=False,
            gradient_accumulation_fusion=True,
            moe_grouped_gemm=True,
            use_transformer_engine_op_fuser=True,
            use_fused_weighted_squared_relu=True,
            moe_use_grouped_tensor=True,
        )
        layer = None
        try:
            layer = MoELayer(config, spec, pg_collection=pg).cuda()
            assert config.fp8 is None
            for parameter in layer.parameters():
                parameter.main_grad = torch.zeros_like(parameter, dtype=torch.float32)
                parameter.grad_added_to_main_grad = False
            for parameter in layer.experts.parameters():
                assert not is_mxfp8tensor(parameter)
                assert parameter.dtype == torch.bfloat16
            # Force traffic to opposite EP owners on successive uses of the same layer.
            # Distinct per-rank inputs expose dropped or duplicated contributions.
            with torch.no_grad():
                layer.router.weight.zero_()
                layer.router.weight[:, 0].copy_(torch.tensor([1, 0.5, -0.5, -1], device='cuda'))
                layer.router.weight[2, 1] = 2
            routes, inputs, outputs, upstreams, plans = [], [], [], [], []

            def capture_routes(module, args, output):
                probs, indices = output
                probs.retain_grad()
                routes.append((probs, indices.clone()))

            layer.router.register_forward_hook(capture_routes)
            if virtual:
                manager = layer.token_dispatcher._comm_manager
                dispatch = manager.plan_dispatch

                def record(*args):
                    dispatch(*args)
                    plans.append(manager._plan)

                monkeypatch.setattr(manager, 'plan_dispatch', record)
            weights = {
                name: p.detach().float().cpu().clone() for name, p in layer.named_parameters()
            }
            for use in range(2):
                rng = torch.Generator(device='cuda').manual_seed(8765 + 10 * use + pg.ep.rank())
                x = torch.randn(32, 1, 128, device='cuda', dtype=torch.bfloat16, generator=rng)
                x[..., 0] = 1 if use == 0 else -1
                x[..., 1] = 0
                x[-8:, :, 1] = 1 if use == 0 else -1
                x.requires_grad_()
                y, bias = layer(x)
                assert bias is None
                inputs.append(x)
                outputs.append(y)
                upstreams.append(torch.randn(y.shape, device='cuda', dtype=y.dtype, generator=rng))
            if virtual:
                assert len(plans) == 2
                assert plans[0].experts_to_copy.data_ptr() != plans[1].experts_to_copy.data_ptr()
                for use, plan in enumerate(plans):
                    # The hot expert must run on both its owner and a virtual copy, so the
                    # comparison exercises split/reduced wgrads, not just whole-expert moves.
                    hot_routes = routes[use][1] == (0 if use == 0 else 3)
                    destinations = (plan.virtual_experts[hot_routes] // 4).long()
                    counts = torch.bincount(destinations, minlength=2)
                    torch.distributed.all_reduce(counts, group=pg.ep)
                    assert (counts > 0).all(), 'hot expert must retain native and virtual work'
            # Both forwards precede backward, as with a repeated MTP layer. Weights stay fixed;
            # optimizer drift must not weaken exact forward/input-gradient expectations.
            # Backpropagate in reverse use order, as the repeated MTP dependency does.
            for y, upstream in reversed(list(zip(outputs, upstreams))):
                y.backward(upstream)
            torch.cuda.synchronize()
            values = {}
            for use, (x, y, (probs, indices)) in enumerate(zip(inputs, outputs, routes)):
                expected_routes = (
                    torch.tensor([0, 1] if use == 0 else [2, 3], device='cuda')
                    .expand(32, -1)
                    .clone()
                )
                expected_routes[-8:] = torch.tensor([0, 2] if use == 0 else [1, 3], device='cuda')
                torch.testing.assert_close(indices.sort(dim=-1).values.long(), expected_routes)
                values[f'use {use}: routes'] = indices.cpu()
                values[f'use {use}: probabilities'] = probs.detach().cpu()
                values[f'use {use}: output'] = y.detach().cpu()
                values[f'use {use}: input gradient'] = x.grad.cpu()
                values[f'use {use}: probability gradient'] = probs.grad.cpu()
            for name, parameter in layer.named_parameters():
                grad = parameter.main_grad if name.startswith('experts.') else parameter.grad
                values[f'wgrad {name}'] = grad.detach().cpu().clone()
            return weights, values
        finally:
            torch.cuda.synchronize()
            del layer
            VirtualExpertLoadBalancer.finalize()
            destroy_moe_metrics_tracker()
            gc.collect()
            fused_a2a.reset_hybrid_ep_buffer()

    try:
        reference_weights, reference = run(False)
        actual_weights, actual = run(True)
        assert actual_weights.keys() == reference_weights.keys()
        for name in reference_weights:
            _assert_bitwise_parity(actual_weights[name], reference_weights[name], f'initial {name}')
        assert actual.keys() == reference.keys()
        for name, expected in reference.items():
            if name.startswith('wgrad experts.'):
                assert expected.norm() > 0, name
                _assert_numerical_parity(actual[name], expected, 1e-5, f'BF16 {name}')
            else:
                _assert_bitwise_parity(actual[name], expected, f'BF16 {name}')
    finally:
        Utils.destroy_model_parallel()


@requires_hybridep
@pytest.mark.launch_on_gb200
@pytest.mark.usefixtures("mxfp8_grouped_mlp")
def test_mxfp8_expert_recompute_parity():
    """Selective activation recomputation preserves outputs and every expert/input gradient."""
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.transformer.spec_utils import get_submodules

    Utils.initialize_model_parallel(expert_model_parallel_size=4)
    spec = get_submodules(
        get_gpt_layer_with_transformer_engine_spec(
            num_experts=8, moe_grouped_gemm=True
        ).submodules.mlp
    )

    def run(recompute):
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        model_parallel_cuda_manual_seed(1234)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            ffn_hidden_size=256,
            moe_ffn_hidden_size=256,
            num_moe_experts=8,
            expert_model_parallel_size=4,
            bf16=True,
            params_dtype=torch.bfloat16,
            add_bias_linear=False,
            activation_func=squared_relu,
            gated_linear_unit=False,
            gradient_accumulation_fusion=True,
            moe_grouped_gemm=True,
            use_transformer_engine_op_fuser=True,
            use_fused_weighted_squared_relu=True,
            moe_use_grouped_tensor=True,
            fp8="e4m3",
            fp8_recipe="mxfp8",
            fp8_param=True,
            moe_router_padding_for_quantization=True,
            moe_token_dispatcher_type="alltoall",
            recompute_granularity="selective" if recompute else None,
            recompute_modules=["moe_act"] if recompute else [],
        )
        with get_fp8_context(config, is_init=True):
            experts = MoELayer(config, spec).cuda().experts
        for parameter in experts.parameters():
            parameter.main_grad = torch.zeros(
                parameter.shape, dtype=torch.float32, device=parameter.device
            )
            parameter.grad_added_to_main_grad = False
        rng = torch.Generator(device="cuda").manual_seed(8765)
        x = torch.randn(
            512, 128, device="cuda", dtype=torch.bfloat16, generator=rng, requires_grad=True
        )
        probs = torch.rand(512, device="cuda", generator=rng, requires_grad=True)
        upstream = torch.randn(x.shape, device="cuda", dtype=x.dtype, generator=rng)
        counts = torch.tensor([256, 256], device="cuda", dtype=torch.int64)
        with get_fp8_context(config):
            y, _ = experts(x, counts, probs)
        y.backward(upstream)
        return {
            "output": y.detach(),
            "input gradient": x.grad,
            "probability gradient": probs.grad,
            **{name: p.main_grad for name, p in experts.named_parameters()},
        }

    try:
        reference = run(False)
        actual = run(True)
        assert actual.keys() == reference.keys()
        for name, expected in reference.items():
            assert expected.norm() > 0, f"{name}: empty reference"
            # Requantizing the recomputed activation changes FC2 wgrads slightly
            # (observed relative L2 below 2%); other gradients do not use that tensor.
            if name.startswith("linear_fc2."):
                _assert_numerical_parity(
                    actual[name], expected, 0.025, f"recomputed {name}", peak_tolerance=0.06
                )
            else:
                _assert_bitwise_parity(actual[name], expected, f"recomputed {name}")
    finally:
        FP8GlobalStateManager.reset()
        Utils.destroy_model_parallel()


def _check_equal(actual, expected, label, errors):
    """Keep comparisons collective-safe while checking every element, including canaries."""
    try:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    except AssertionError as exc:
        errors.append(f"{label}: {exc}")


def _report(errors, group):
    """Fail on every rank if any rank saw a mismatch."""
    gathered = [None for _ in range(dist.get_world_size(group))]
    dist.all_gather_object(gathered, errors, group=group)
    combined = [error for rank_errors in gathered for error in rank_errors]
    assert not combined, "\n".join(combined)


def _reference_plan(routes, num_experts):
    """CPU greedy placement and stable route assignment, using only semantic input routes."""
    ep_size, num_tokens, topk = routes.shape
    local_experts, capacity = num_experts // ep_size, num_tokens * topk
    counts = torch.stack([torch.bincount(row.flatten(), minlength=num_experts) for row in routes])
    totals = counts.sum(0).tolist()
    loads = counts.sum(0).reshape(ep_size, local_experts).sum(1).tolist()
    quotas = [[0] * ep_size for _ in range(ep_size)]
    while max(loads) > capacity:
        sender = max(range(ep_size), key=lambda r: (loads[r], -r))
        receiver = min(range(ep_size), key=lambda r: (loads[r], r))
        # A receiver takes its entire deficit from one sender, even beyond that sender's
        # excess. This bounds the number of virtual slots by the sender's native experts.
        moved = capacity - loads[receiver]
        quotas[sender][receiver] += moved
        loads[sender] -= moved
        loads[receiver] = capacity
    allocation = [[0] * ep_size for _ in range(num_experts)]
    for expert, count in enumerate(totals):
        allocation[expert][expert // local_experts] = count
    for sender, pending in enumerate(quotas):
        experts = range(sender * local_experts, (sender + 1) * local_experts)
        while any(pending):
            destination = max(range(ep_size), key=lambda r: (pending[r], -r))
            expert = max(experts, key=lambda e: (allocation[e][sender], -e))
            moved = min(pending[destination], allocation[expert][sender])
            assert moved > 0
            allocation[expert][sender] -= moved
            allocation[expert][destination] += moved
            pending[destination] -= moved
    copies = []
    for destination in range(ep_size):
        remote = sorted(
            (
                e
                for e in range(num_experts)
                if e // local_experts != destination and allocation[e][destination]
            ),
            key=lambda e: (allocation[e][destination], e),
            reverse=True,
        )
        assert len(remote) <= local_experts
        copies.append(remote + [-1] * (local_experts - len(remote)))
    # Stable global order is source rank, token, top-k lane. Partition each expert's
    # positions directly by the CPU allocation, without reading kernel boundaries or slots.
    order = routes.flatten().argsort(stable=True)
    mapped = torch.empty(routes.numel(), dtype=torch.int16)
    cursor = 0
    for expert, destinations in enumerate(allocation):
        for destination, count in enumerate(destinations):
            if count:
                local = (
                    expert % local_experts
                    if destination == expert // local_experts
                    else (local_experts + copies[destination].index(expert))
                )
                mapped[order[cursor : cursor + count]] = destination * (2 * local_experts) + local
                cursor += count
    assert cursor == routes.numel()
    return (
        counts.int(),
        torch.tensor(allocation, dtype=torch.int32),
        torch.tensor(copies, dtype=torch.int32),
        mapped.view_as(routes),
    )


def _routes_for_skew(ep_size, num_experts, num_tokens, topk, skew):
    """Generate distinct top-k choices with exact balance, ties or controlled load skew."""
    if skew == "ties":
        return torch.arange(9).repeat_interleave(4).reshape(4, 9, 1)
    rows = torch.arange(num_tokens)[:, None] + torch.arange(topk)
    local_experts = num_experts // ep_size
    if skew == "rank_ties":
        return (rows % (num_experts // 2)).repeat(ep_size, 1, 1)
    if skew in ("balanced", "local", "remote"):
        return torch.stack(
            [
                (
                    (rows + rank * local_experts) % num_experts
                    if skew == "balanced"
                    else rows % local_experts
                    + ((rank + (skew == "remote")) % ep_size) * local_experts
                )
                for rank in range(ep_size)
            ]
        )
    if skew == "hot_expert" and topk == 1:
        return torch.zeros((ep_size, num_tokens, topk), dtype=torch.int64)
    weights = torch.ones(num_experts)
    if skew == "hot_expert":
        weights[0] = 20
    elif skew == "hot_rank":
        weights[:local_experts] = 12
    elif skew == "concentrated":
        weights[topk:] = 0
    else:
        assert skew == "random"
    return torch.multinomial(
        weights.expand(ep_size * num_tokens, -1),
        topk,
        replacement=False,
        generator=torch.Generator().manual_seed(1234),
    ).reshape(ep_size, num_tokens, topk)


def test_virtual_expert_planner_reference_tie_breaks():
    """Hand-computed over-migration exercises rank/expert ties and reversed slot ties."""
    # Two equally overloaded senders and two equally empty receivers pair in rank order.
    _, _, copies, mapped = _reference_plan(_routes_for_skew(4, 8, 4, 1, "rank_ties"), 8)
    assert copies.tolist() == [[-1, -1], [-1, -1], [0, -1], [2, -1]]
    assert mapped.flatten().tolist() == [10, 1, 14, 5] * 4
    routes = _routes_for_skew(4, 12, 9, 1, "ties")
    _, allocation, copies, mapped = _reference_plan(routes, 12)
    assert allocation.tolist() == [
        [0, 0, 0, 4],
        [0, 0, 0, 4],
        [3, 0, 0, 1],
        [4, 0, 0, 0],
        [2, 2, 0, 0],
        [0, 4, 0, 0],
        [0, 3, 1, 0],
        [0, 0, 4, 0],
        [0, 0, 4, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    assert copies.tolist() == [[3, 4, -1], [6, -1, -1], [-1, -1, -1], [1, 0, 2]]
    expected = [22] * 4 + [21] * 4 + [2] * 3 + [23] + [3] * 4 + [4] * 2 + [7] * 2
    expected += [8] * 4 + [9] * 3 + [12] + [13] * 4 + [14] * 4
    assert mapped.flatten().tolist() == expected


def _check_planner_reference(ep_size, num_experts, num_tokens, topk, skews):
    Utils.initialize_distributed()
    world_size = dist.get_world_size()
    groups = (
        [
            dist.new_group(list(range(start, min(start + ep_size, world_size))))
            for start in range(0, world_size, ep_size)
        ]
        if ep_size != world_size
        else []
    )
    group = groups[dist.get_rank() // ep_size] if groups else dist.group.WORLD
    # A trailing subgroup also exercises fewer ranks, including the EP1 utility case.
    ep_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    device = torch.device("cuda", torch.cuda.current_device())
    local_experts = num_experts // ep_size
    workspace = VirtualExpertPlannerWorkspace(num_experts=num_experts, device=device, group=group)
    errors = []
    try:
        for iteration, skew in enumerate(skews.split()):
            routes = _routes_for_skew(ep_size, num_experts, num_tokens, topk, skew)
            counts, allocation, copies, mapped = _reference_plan(routes, num_experts)
            dtype = torch.int32 if iteration % 2 else torch.int64
            own = routes[rank].to(device=device, dtype=dtype)
            if iteration % 3:
                # A nonzero offset and poisoned gaps catch a wrapper that forgets contiguous().
                storage = torch.full(
                    (num_tokens, 2 * topk + 1), num_experts + 7, device=device, dtype=dtype
                )
                storage[:, 1::2] = own
                own = storage[:, 1::2]
                assert not own.is_contiguous()
            plan = plan_virtual_expert_routes(own, workspace)
            actual = plan.virtual_experts.cpu()
            actual_copies = plan.experts_to_copy.cpu()
            for label, value, expected in (
                ("histogram", workspace.gathered_counts.cpu(), counts),
                ("allocation", workspace.field("allocation").cpu(), allocation),
                ("copies", actual_copies, copies),
                ("routes", actual, mapped[rank]),
            ):
                _check_equal(value, expected, f"{skew} {label}", errors)
            # Independently decode the public outputs and count real routes, rather than
            # accepting a self-consistent allocation/slot table with the wrong expert owners.
            try:
                assert actual.shape == (num_tokens, topk) and actual.dtype == torch.int16
                assert ((actual >= 0) & (actual < 2 * num_experts)).all()
                destination, local = actual.long() // (2 * local_experts), actual.long() % (
                    2 * local_experts
                )
                semantic = destination * local_experts + local
                remote = local >= local_experts
                semantic[remote] = actual_copies[
                    destination[remote], local[remote] - local_experts
                ].long()
                torch.testing.assert_close(semantic, routes[rank], rtol=0, atol=0)
                if skew in ("balanced", "local", "remote"):
                    assert (actual_copies == -1).all()
                if skew in ("local", "remote"):
                    assert (
                        (destination == rank) if skew == "local" else (destination != rank)
                    ).all()
                observed = torch.bincount(
                    (routes[rank] * ep_size + destination).flatten(),
                    minlength=num_experts * ep_size,
                ).to(device=device, dtype=torch.int32)
            except (AssertionError, IndexError) as exc:
                errors.append(f"{skew} semantic routes: {exc}")
                observed = torch.zeros(num_experts * ep_size, device=device, dtype=torch.int32)
            dist.all_reduce(observed, group=group)
            _check_equal(
                observed.cpu().reshape(num_experts, ep_size),
                allocation,
                f"{skew} route counts",
                errors,
            )
            _check_equal(
                observed.reshape(num_experts, ep_size).sum(0).cpu(),
                torch.full((ep_size,), num_tokens * topk, dtype=torch.int64),
                f"{skew} balanced load",
                errors,
            )
            # The real dispatcher provides this cross-rank ordering between planner launches.
            dist.barrier(group=group, device_ids=[device.index])
        _report(errors, dist.group.WORLD)
    finally:
        torch.cuda.synchronize(device)
        workspace.destroy()
        dist.barrier(device_ids=[device.index])
        if groups:
            dist.destroy_process_group(group)


@requires_cuda
@pytest.mark.parametrize(
    "ep_size,num_experts,num_tokens,topk,skews",
    [
        (4, 8, 17, 3, "balanced concentrated balanced"),
        (4, 12, 9, 1, "ties local remote"),
        (4, 512, 33, 10, "random concentrated balanced"),
    ],
    ids=["changing-plan-and-strided-input", "ties-and-empty-experts", "nt4-experts-and-topk"],
)
def test_virtual_expert_planner_matches_reference(ep_size, num_experts, num_tokens, topk, skews):
    """Preserve expert identity and exactly balance routes, including reused workspace."""
    _check_planner_reference(ep_size, num_experts, num_tokens, topk, skews)


def _virtual_expert_hybridep_config(**overrides):
    """Build a minimal virtual-expert HybridEP config, then apply one override."""
    kwargs = dict(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=2,
        expert_model_parallel_size=2,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="hybridep",
        moe_virtual_expert_load_balance=True,
        moe_grouped_gemm=True,
        moe_router_dtype="fp32",
        use_transformer_engine_op_fuser=True,
        gradient_accumulation_fusion=True,
        add_bias_linear=False,
        activation_func=F.silu,
        gated_linear_unit=True,
        bf16=True,
        params_dtype=torch.bfloat16,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def test_virtual_expert_hybridep_defaults_a_dropless_rank_capacity():
    """The backend is dropless by construction and allows the whole-layer moe graph."""
    config = _virtual_expert_hybridep_config(cuda_graph_impl="local", cuda_graph_modules=["moe"])

    assert config.moe_expert_rank_capacity_factor == 1.0
    assert config.moe_single_grouped_weight is False


def test_virtual_expert_hybridep_accepts_native_mxfp8_with_router_padding():
    """Native MXFP8 parameters are the only quantized storage the push understands."""
    config = _virtual_expert_hybridep_config(
        fp8="e4m3", fp8_recipe="mxfp8", fp8_param=True, moe_router_padding_for_quantization=True
    )

    assert (config.fp8, config.fp8_recipe, config.fp8_param) == ("e4m3", "mxfp8", True)
    assert config.moe_router_padding_for_quantization


@pytest.mark.parametrize(
    ("fp8", "fp8_recipe", "fp8_param"),
    [("e4m3", "mxfp8", False), ("e4m3", "tensorwise", True), ("hybrid", "mxfp8", True)],
)
def test_virtual_expert_hybridep_rejects_unsupported_fp8_parameter_storage(
    fp8, fp8_recipe, fp8_param
):
    with pytest.raises(ValueError, match="MXFP8 E4M3 with native FP8 parameters"):
        _virtual_expert_hybridep_config(fp8=fp8, fp8_recipe=fp8_recipe, fp8_param=fp8_param)


@pytest.mark.parametrize("scope", ["moe_router", "moe_preprocess"])
def test_virtual_expert_hybridep_rejects_partial_moe_cuda_graph_scopes(scope):
    """Only the whole-layer moe scope preserves the planner's per-forward metadata."""
    with pytest.raises(AssertionError, match="moe CUDA graph scope only"):
        _virtual_expert_hybridep_config(cuda_graph_impl="local", cuda_graph_modules=[scope])


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"moe_token_dispatcher_type": "alltoall"}, "moe_token_dispatcher_type='flex'"),
        ({"expert_model_parallel_size": 1}, "2<=expert_model_parallel_size<=64"),
        ({"num_moe_experts": 3}, "num_moe_experts divisible"),
        ({"moe_router_topk": 3}, "1<=moe_router_topk"),
        ({"moe_expert_capacity_factor": 1.0}, "moe_expert_capacity_factor=None"),
        (
            {"recompute_granularity": "selective", "recompute_modules": ["moe"]},
            "no MoE layer recompute",
        ),
    ],
)
def test_virtual_expert_rejects_unsupported_layout(overrides, match):
    """Reject combinations that would drop routes or invalidate runtime storage."""
    with pytest.raises(ValueError, match=match):
        _virtual_expert_hybridep_config(**overrides)


@requires_cuda
@pytest.mark.parametrize(
    "quantile,expert_bias,compact_supported",
    [(True, False, True), (True, False, False), (False, False, True)],
    ids=["nt4-compact", "nt4-dense-fallback", "fused-seq-aux"],
)
def test_nt4_compact_router(monkeypatch, quantile, expert_bias, compact_supported):
    """Recipe-sized expert IDs and scores agree with a dense unfused router and CPU math."""
    monkeypatch.setattr(moe_utils, "hybrid_ep_dense_topk_routing", lambda *_: compact_supported)
    Utils.initialize_model_parallel(1, 1)
    _set_random_seed(seed_=123, data_parallel_random_init=False)
    config = _virtual_expert_hybridep_config(
        moe_virtual_expert_load_balance=False,
        num_moe_experts=512,
        moe_router_topk=10,
        moe_router_score_function="sigmoid",
        moe_router_dtype="fp32",
        moe_router_topk_scaling_factor=3.16,
        moe_router_load_balancing_type="quantile_balancing" if quantile else "seq_aux_loss",
        moe_aux_loss_coeff=0 if quantile else 1e-4,
        moe_router_enable_expert_bias=expert_bias,
        moe_router_fusion=not quantile and not expert_bias,
    )
    torch.manual_seed(321)
    logits_cpu = torch.randn(256, 512)
    if quantile or expert_bias:
        logits_cpu[0] = -100  # Bias resolves ties between selected zero-probability routes.
    bias = torch.randn(512) * 0.3 if quantile or expert_bias else torch.zeros(512)
    scores = logits_cpu.sigmoid()
    selection = logits_cpu - bias if quantile else scores + bias
    expected_ids = selection.argsort(dim=1, descending=True)[:, :10].sort(dim=1).values
    selected = scores.gather(1, expected_ids)
    expected_probs = 3.16 * selected / (selected.sum(dim=1, keepdim=True) + 1e-20)
    dy = torch.randn(256, 512, device="cuda")
    calls = []
    fused = moe_utils.fused_topk_with_score_function

    def record_fused(**kwargs):
        calls.append((virtual, kwargs["logits"].shape, kwargs.get("topk_indices")))
        if not virtual:
            assert "topk_indices" not in kwargs, "ordinary routing must work with older TE APIs"
        return fused(**kwargs)

    monkeypatch.setattr(moe_utils, "fused_topk_with_score_function", record_fused)
    monkeypatch.setattr(moe_utils.MoEAuxLossAutoScaler, "main_loss_backward_scale", None)
    moe_utils.MoEAuxLossAutoScaler.set_loss_scale(torch.tensor(0.37, device="cuda"))
    pg = ProcessGroupCollection.use_mpu_process_groups()
    try:
        reference = TopKRouter(
            replace(config, moe_router_fusion=False, moe_token_dispatcher_type="alltoall"), pg
        ).cuda()
        reference.set_layer_number(1)
        if quantile or expert_bias:
            (reference.qb_beta if quantile else reference.expert_bias).copy_(bias)
        ref_logits = logits_cpu.cuda().requires_grad_()
        ref_probs, ref_map = reference.routing(ref_logits.view(256, 1, 512))
        (ref_probs * dy).sum().backward()
        torch.testing.assert_close(
            ref_map.cpu(),
            torch.zeros_like(logits_cpu, dtype=torch.bool).scatter(1, expected_ids, True),
        )
        for virtual in ((False, True) if compact_supported and not expert_bias else (False,)):
            router = TopKRouter(replace(config, moe_virtual_expert_load_balance=virtual), pg).cuda()
            router.set_layer_number(1)
            if quantile or expert_bias:
                (router.qb_beta if quantile else router.expert_bias).copy_(bias)
            logits = logits_cpu.cuda().requires_grad_()
            probs, ids = router.routing(logits.view(256, 1, 512))
            compact = virtual or (quantile and compact_supported)
            assert (ids.dtype != torch.bool) == compact
            if not compact:
                assert ids.shape == probs.shape == (256, 512)
                ids = ids.to(torch.int8).topk(10, dim=1).indices
                probs = probs.gather(1, ids)
            assert probs.shape == ids.shape == (256, 10) and ids.dtype == torch.int64
            sorted_ids, order = ids.sort(dim=1)
            torch.testing.assert_close(sorted_ids.cpu(), expected_ids, rtol=0, atol=0)
            torch.testing.assert_close(
                probs.gather(1, order).cpu(), expected_probs, rtol=1e-5, atol=1e-7
            )
            assert ids.max() > 64
            if quantile or expert_bias:
                assert not probs[0].any()
            (probs * dy.gather(1, ids)).sum().backward()
            torch.testing.assert_close(logits.grad, ref_logits.grad, rtol=2e-4, atol=1e-6)
            assert logits.grad[1:].norm() > 0
            if quantile:
                alpha = selection.sort(dim=1, descending=True).values[:, 10:11]
                expected_beta = (logits_cpu - alpha).sort(dim=0, descending=True).values[5]
                torch.testing.assert_close(router.qb_beta_accum.cpu(), expected_beta)
                assert router.qb_beta_count.item() == 1
                # No duplicate accumulation during no-grad recomputation or eval.
                with torch.no_grad():
                    router.routing(logits.view(256, 1, 512))
                router.eval()
                router.routing(logits.view(256, 1, 512))
                torch.testing.assert_close(router.qb_beta_accum.cpu(), expected_beta)
                assert router.qb_beta_count.item() == 1
                assert "qb_beta" in router.state_dict()
                assert "qb_beta_accum" not in router.state_dict()
                router.config.moe_router_fusion = True
                with pytest.raises(AssertionError, match="does not support moe_router_fusion"):
                    router.routing(logits.view(256, 1, 512))
            elif expert_bias:
                torch.testing.assert_close(
                    router.local_tokens_per_expert, ref_map.sum(dim=0).float(), rtol=0, atol=0
                )
        for virtual, shape, index_buffer in calls:
            if not virtual:
                assert shape == (256, 512) and index_buffer is None
            else:
                assert shape == (256, 512) and index_buffer.shape == (256, 10)
        assert bool(calls) == config.moe_router_fusion
    finally:
        destroy_moe_metrics_tracker()
        Utils.destroy_model_parallel()
