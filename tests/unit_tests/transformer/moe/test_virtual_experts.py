# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Focused virtual-expert regressions: routing, peer transport and training parity.

Run on one node with four Blackwell GPUs::

    python -m torch.distributed.run --standalone --nproc-per-node=4 -m pytest -q \
        tests/unit_tests/transformer/moe/test_virtual_experts.py

Eight-GPU H100 nodes run the non-MXFP8 cases; MXFP8 compute skips on pre-Blackwell
hardware. The existing GB200 CI selection uses ``launch_on_gb200``. All inputs are
synthetic; no dataset or custom launcher is required.
"""

import gc
import os
from contextlib import nullcontext
from dataclasses import replace
from functools import partialmethod
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
from torch.nn import functional as F
from transformer_engine.pytorch import cpu_offload as te_cpu_offload
from transformer_engine.pytorch.attention.dot_product_attention import (
    backends as te_attention_backends,
)
from transformer_engine.pytorch.quantization import FP8GlobalStateManager
from triton.runtime import _allocation

from megatron.core import parallel_state as ps
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
from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.moe import fused_a2a, moe_utils
from megatron.core.transformer.moe import router as router_module
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.moe_logging import destroy_moe_metrics_tracker
from megatron.core.transformer.moe.ops import virtual_expert_triton as vt
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.token_dispatcher import _VirtualExpertHybridEPManager
from megatron.core.transformer.moe.virtual_expert_load_balancer import VirtualExpertLoadBalancer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.determinism.utils import RacingStreams
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
def grouped_mlp_fusions(monkeypatch, virtual_expert_environment):
    """Refresh TE fusion support after setting this test's environment."""
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
        for cls, name, legacy_name in (
            (grouped_mlp.GroupedMLP_CuTeGEMMGLU, "fuse_glu_ops", "fuse_ops"),
            (grouped_mlp.GroupedMLP_CuTeGEMMUnary, "fuse_unary_activation_ops", "fuse_srelu_ops"),
        ):
            fuse = getattr(grouped_mlp, name, None)
            if fuse is None:
                fuse = getattr(grouped_mlp, legacy_name)
            if cls.is_supported() and fuse not in OperationFuser.forward_backward_fusion_functions:
                register_forward_backward_fusion(fuse, prepend=True)
        yield grouped_mlp
    finally:
        for check in checks:
            check.cache_clear()


@pytest.fixture
def mxfp8_grouped_mlp(grouped_mlp_fusions):
    """Require real Blackwell fused kernels for MXFP8 numerical comparisons."""
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("MXFP8 requires Blackwell")
    assert (
        grouped_mlp_fusions.GroupedMLP_CuTeGEMMUnary.is_supported()
    ), "MXFP8 parity requires TE's fused SReLU kernels and a compatible cuDNN frontend"


@pytest.fixture
def caller_allocator():
    def allocator(*args):
        raise AssertionError("MoonEP used the caller's allocator")

    token = _allocation._allocator.set(allocator)
    try:
        yield allocator
    finally:
        _allocation._allocator.reset(token)


@requires_cuda
@pytest.mark.parametrize("mxfp8", [False, True], ids=["bf16-fp32-grad", "mxfp8-bf16-grad"])
def test_virtual_expert_transport_allocator_replay(monkeypatch, caller_allocator, mxfp8):
    """Real transport preserves the caller allocator after success, failure and graph capture."""
    Utils.initialize_distributed()
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())
    dist.all_reduce(torch.zeros(1, device=device))
    symm_mem.set_backend("NCCL")
    local, member = 2, 32768
    weight_dtype = torch.uint8 if mxfp8 else torch.bfloat16
    grad_dtype = torch.bfloat16 if mxfp8 else torch.float32
    scale_member = member // 32 if mxfp8 else 0
    w = SimpleNamespace(
        rank=rank,
        world_size=world,
        num_local_experts=local,
        member_numels=(member, member),
        num_sms=32,
        weight_arena=symm_mem.empty(
            2 * local * (member + scale_member), dtype=weight_dtype, device=device
        ),
        grad_arena=symm_mem.empty(4 * local * member, dtype=grad_dtype, device=device),
        weight_grid_barrier=torch.zeros(1, device=device, dtype=torch.int32),
        grad_grid_barrier=torch.zeros(1, device=device, dtype=torch.int32),
    )
    w.weight_handle = symm_mem.rendezvous(w.weight_arena, dist.group.WORLD)
    w.grad_handle = symm_mem.rendezvous(w.grad_arena, dist.group.WORLD)
    weights = torch.full((2, local, member), rank + 1, device=device, dtype=weight_dtype)
    scales = torch.full((2, local, scale_member), rank + 11, device=device, dtype=torch.uint8)
    native = w.grad_arena[2 * local * member :].view(2, local, member)

    def tables(tensor):
        return tuple(
            torch.tensor(
                [tensor[k, j].data_ptr() for j in range(local)], device=device, dtype=torch.int64
            )
            for k in range(2)
        )

    sources, grads = tables(weights), tables(native)
    scale_sources = tables(scales) if mxfp8 else None
    plan = torch.tensor(
        [[(d - 1) % world * local + j for j in range(local)] for d in range(world)],
        device=device,
        dtype=torch.int32,
    )
    allocations = []
    fail_allocation = False
    original = triton.set_allocator

    def track_allocator(allocator):
        def scratch(size, alignment, stream):
            assert _allocation._allocator.get() is scratch
            if fail_allocation:
                raise RuntimeError("descriptor scratch allocation failed")
            allocations.append((size, alignment))
            return allocator(size, alignment, stream)

        original(scratch)

    monkeypatch.setattr(triton, "set_allocator", track_allocator)

    def push():
        vt.launch_virtual_expert_weight_prefetch(
            w, sources=sources, scale_sources=scale_sources, experts_to_copy=plan
        )

    def reduce():
        vt.launch_virtual_expert_grad_reduce(w, native_grads=grads, experts_to_copy=plan)

    graphs = []
    try:
        for graph in (False, True):
            graphs.clear()
            for call in (push, reduce):
                # Fail before kernel submission on every rank; the caller's allocator must survive.
                fail_allocation = True
                with pytest.raises(RuntimeError, match="descriptor scratch allocation failed"):
                    call()
                assert _allocation._allocator.get() is caller_allocator
                fail_allocation = False
                w.grad_arena.fill_(rank + 1)
                count = len(allocations)
                call()
                assert len(allocations) > count
                assert _allocation._allocator.get() is caller_allocator
                dist.barrier()
                if graph:
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        call()
                    assert _allocation._allocator.get() is caller_allocator
                    graphs.append(g)
                else:
                    graphs.append(None)
            for step in range(3):
                weights.fill_(rank + 1 + step)
                scales.fill_(rank + 11 + step)
                w.weight_arena.zero_()
                w.grad_arena.fill_(rank + 1 + step)
                dist.barrier()
                count = len(allocations)
                for call, g in zip((push, reduce), graphs):
                    call() if g is None else g.replay()
                    assert _allocation._allocator.get() is caller_allocator
                torch.cuda.synchronize()
                if graph:
                    assert (
                        len(allocations) == count
                    ), "replay must reuse captured descriptor scratch"
                previous, following = (rank - 1) % world, (rank + 1) % world
                for section in w.weight_arena.view(2, local * (member + scale_member)):
                    torch.testing.assert_close(
                        section[: local * member],
                        torch.full_like(section[: local * member], previous + 1 + step),
                        rtol=0,
                        atol=0,
                    )
                    if mxfp8:
                        torch.testing.assert_close(
                            section[local * member :],
                            torch.full_like(section[local * member :], previous + 11 + step),
                            rtol=0,
                            atol=0,
                        )
                torch.testing.assert_close(
                    native, torch.full_like(native, rank + following + 2 + 2 * step), rtol=0, atol=0
                )
                dist.barrier()
    finally:
        torch.cuda.synchronize()
        graphs.clear()
        w.weight_handle = w.grad_handle = None


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
    # Compare dispatch paths without nondeterministic TE dactivation/dprob accumulation.
    monkeypatch.setenv('NVTE_ALLOW_NONDETERMINISTIC_ALGO', '0')
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
                # Match MoonEP's shared-expert launch order in both A/B variants so BF16
                # input-gradient accumulation compares the same execution schedule.
                layer.run_shared_experts_after_router = True
                # Tiny MXFP8 inputs round every nonempty expert to one block. Concentrate
                # choices on one owner so this test still exercises real replica transfers.
                layer.router.qb_beta.copy_(
                    torch.tensor([-100, -100, 100, 100], device='cuda', dtype=torch.float32)
                )
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

                    expected_alignment = moe_utils.get_align_size_for_quantization(layer.config)
                    assert layer.experts._with_fused_impl == (egtp_size > 1 or 'mtp' not in name)

                    def record(
                        *args,
                        manager=manager,
                        dispatch=dispatch,
                        expected_alignment=expected_alignment,
                    ):
                        assert manager._alignment == expected_alignment
                        assert manager.rank_capacity == manager._compute_rank_capacity(
                            args[0].shape[0]
                        )
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
                if virtual:
                    managers = [layer.token_dispatcher._comm_manager for _, layer in experts]
                    assert len({id(manager.planner) for manager in managers}) == 1
                    for (_, layer), manager in zip(experts, managers):
                        assert manager.pad_multiple == manager._alignment
                        assert (manager.tokens_per_expert % manager._alignment == 0).all()
                        assert manager.tokens_per_expert.sum() <= manager.rank_capacity
                        assert (
                            layer.experts._fused_ops is not None
                        ) == layer.experts._with_fused_impl
                        if layer.experts._with_fused_impl:
                            (sequence,) = layer.experts._fused_ops
                            assert any(
                                type(op).__name__ == 'GroupedMLP_CuTeGEMMUnary'
                                for group in sequence._module_groups
                                for op, _ in group._forward_ops
                            ), 'MXFP8 must execute the fused CuTeDSL MLP'
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
                # Storing split native/virtual wgrad partials in BF16 can erase tiny
                # cancellation residuals, which Adam amplifies in individual updates.
                # With matched launch order, observed maxima are 0.26% gradient L2
                # (0.68% peak) and 1.20% update L2 (94.46% peak).
                if name == 'loss':
                    tolerance, peak_tolerance = 1e-5, 1e-5
                elif name.startswith('optimizer update'):
                    tolerance, peak_tolerance = 0.015, 1.1
                else:
                    tolerance, peak_tolerance = 0.005, 0.01
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
@pytest.mark.parametrize("use_op_fuser", [True, False], ids=["op-fuser", "grouped-tensor"])
def test_bf16_virtual_expert_routing_parity(monkeypatch, use_op_fuser):
    """Moving routes preserves outputs/input/probability gradients; wgrad sums may round."""
    from megatron.core.fusions import fused_weighted_squared_relu as srelu
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.transformer.spec_utils import get_submodules

    # Compare routing with the same probability-gradient reduction strategy in both runs,
    # even when virtual experts change the number of rows processed on a rank.
    monkeypatch.setattr(
        srelu,
        "weighted_squared_relu_back",
        torch.compile(
            srelu.weighted_squared_relu_back._torchdynamo_orig_callable,
            options={"batch_invariant": True},
        ),
    )
    monkeypatch.setattr(
        fused_a2a.HybridEPBuffer,
        '__init__',
        partialmethod(fused_a2a.HybridEPBuffer.__init__, load_cached_kernels=True),
    )
    monkeypatch.setenv('NVTE_GROUPED_LINEAR_SINGLE_PARAM', '0')
    Utils.initialize_model_parallel(expert_model_parallel_size=2)
    pg = ProcessGroupCollection.use_mpu_process_groups()
    group = ps.get_model_parallel_group()
    finalize = VirtualExpertLoadBalancer.finalize
    resources_alive_at_finalize = []

    def observe_finalize():
        resources_alive_at_finalize.append(
            ps.get_model_parallel_group() is group
            and bool(VirtualExpertLoadBalancer.storages)
            and VirtualExpertLoadBalancer.planner is not None
        )
        finalize()

    monkeypatch.setattr(VirtualExpertLoadBalancer, 'finalize', observe_finalize)
    spec = get_submodules(
        get_gpt_layer_with_transformer_engine_spec(
            num_experts=4, moe_grouped_gemm=True
        ).submodules.mlp
    )

    def run(virtual):
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        model_parallel_cuda_manual_seed(1234)
        config = _virtual_expert_hybridep_config(
            ffn_hidden_size=256,
            moe_ffn_hidden_size=256,
            num_moe_experts=4,
            moe_router_topk=2,
            moe_router_score_function='sigmoid',
            moe_router_topk_scaling_factor=3.16,
            moe_router_load_balancing_type='none',
            moe_aux_loss_coeff=0,
            moe_virtual_expert_load_balance=virtual,
            activation_func=squared_relu,
            gated_linear_unit=False,
            use_transformer_engine_op_fuser=use_op_fuser,
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
            if virtual:
                for linear in (layer.experts.linear_fc1, layer.experts.linear_fc2):
                    assert linear.num_gemms == 4
                    assert set(dict(linear.named_parameters(recurse=False))) == {
                        "weight0",
                        "weight1",
                    }
                    assert not any(
                        key.startswith(("weight2", "weight3")) for key in linear.state_dict()
                    )
            # Force traffic to opposite EP owners on successive uses of the same layer.
            # Distinct per-rank inputs expose dropped or duplicated contributions.
            with torch.no_grad():
                layer.router.weight.zero_()
                layer.router.weight[:, 0].copy_(torch.tensor([1, 0.5, -0.5, -1], device='cuda'))
                layer.router.weight[2, 1] = 2
            routes, inputs, outputs, upstreams, plans, planned_routes = [], [], [], [], [], []

            def capture_routes(module, args, output):
                probs, indices = output
                probs.retain_grad()
                if indices.dtype == torch.bool:
                    indices = indices.to(torch.int8).topk(config.moe_router_topk, dim=-1).indices
                else:
                    indices = indices.long()
                routes.append((probs, indices.clone()))

            layer.router.register_forward_hook(capture_routes)
            if virtual:
                manager = layer.token_dispatcher._comm_manager
                dispatch = manager.plan_dispatch

                def record(*args):
                    dispatch(*args)
                    plans.append(manager._plan)
                    planned_routes.append(args[0].clone())

                monkeypatch.setattr(manager, 'plan_dispatch', record)
            weights = {
                name: p.detach().float().cpu().clone() for name, p in layer.named_parameters()
            }
            # Use multiple alignment blocks to retain native and virtual work for the hot expert.
            num_tokens = 512
            for use in range(2):
                rng = torch.Generator(device='cuda').manual_seed(8765 + 10 * use + pg.ep.rank())
                x = torch.randn(
                    num_tokens, 1, 128, device='cuda', dtype=torch.bfloat16, generator=rng
                )
                x[..., 0] = 1 if use == 0 else -1
                x[..., 1] = 0
                x[-num_tokens // 4 :, :, 1] = 1 if use == 0 else -1
                x.requires_grad_()
                y, bias = layer(x)
                assert bias is None
                if virtual:
                    assert manager._alignment == moe_utils.get_align_size_for_quantization(config)
                    assert manager.pad_multiple == manager._alignment
                    assert manager.tokens_per_expert.sum() <= manager.rank_capacity
                    assert (manager.tokens_per_expert % manager._alignment == 0).all()
                if virtual and not use_op_fuser:
                    for fc_layer, linear in enumerate(
                        (layer.experts.linear_fc1, layer.experts.linear_fc2)
                    ):
                        runtime_weights = manager.runtime_weights(fc_layer)
                        assert not hasattr(linear, "_runtime_weights")
                        assert all(
                            getattr(linear, f"weight{index}") is weight
                            for index, weight in enumerate(runtime_weights)
                        )
                        registered = {id(param) for param in linear.parameters(recurse=False)}
                        assert all(id(weight) not in registered for weight in runtime_weights)
                        assert linear._parameters["weight2"] is None
                        assert linear._parameters["weight3"] is None
                        assert not any(
                            key.startswith(("weight2", "weight3")) for key in linear.state_dict()
                        )
                inputs.append(x)
                outputs.append(y)
                upstreams.append(torch.randn(y.shape, device='cuda', dtype=y.dtype, generator=rng))
            if virtual:
                assert len(plans) == 2
                assert plans[0].experts_to_copy.data_ptr() != plans[1].experts_to_copy.data_ptr()
                for use, plan in enumerate(plans):
                    # The hot expert must run on both its owner and a virtual copy, so the
                    # comparison exercises split/reduced wgrads, not just whole-expert moves.
                    # Runtime destinations use the adapter's route order, which can differ
                    # from the router hook's top-k tie order for boolean maps.
                    hot_routes = planned_routes[use] == (0 if use == 0 else 3)
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
                # Compare in expert order; MoonEP returns compact scores directly.
                indices, order = indices.sort(dim=-1)
                gather_indices = order if virtual else indices
                route_probs = probs.gather(1, gather_indices)
                route_grads = probs.grad.gather(1, gather_indices)
                expected_routes = (
                    torch.tensor([0, 1] if use == 0 else [2, 3], device='cuda')
                    .expand(num_tokens, -1)
                    .clone()
                )
                expected_routes[-num_tokens // 4 :] = torch.tensor(
                    [0, 2] if use == 0 else [1, 3], device='cuda'
                )
                torch.testing.assert_close(indices.sort(dim=-1).values.long(), expected_routes)
                values[f'use {use}: routes'] = indices.cpu()
                values[f'use {use}: probabilities'] = route_probs.detach().cpu()
                values[f'use {use}: output'] = y.detach().cpu()
                values[f'use {use}: input gradient'] = x.grad.cpu()
                values[f'use {use}: probability gradient'] = route_grads.cpu()
            for name, parameter in layer.named_parameters():
                grad = parameter.main_grad if name.startswith('experts.') else parameter.grad
                values[f'wgrad {name}'] = grad.detach().cpu().clone()
            return weights, values
        finally:
            torch.cuda.synchronize()
            del layer
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
    assert resources_alive_at_finalize == [True]
    assert not VirtualExpertLoadBalancer.storages
    assert VirtualExpertLoadBalancer.planner is None
    finalize()  # Cleanup stays safe after the model-parallel groups are gone.


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


@requires_hybridep
@pytest.mark.parametrize(
    "precision", ["bf16", pytest.param("mxfp8", marks=pytest.mark.launch_on_gb200)]
)
@pytest.mark.parametrize(
    "recompute,offload_scope,use_op_fuser",
    [
        ("moe_act", None, False),
        ("moe", None, True),
        ("full", None, True),
        (None, "expert_fc1", False),
        (None, "moe_act", False),
        (None, "fused_group_mlp", True),
    ],
    ids=[
        "moe-act-unfused",
        "moe-layer",
        "full-layer",
        "offload-fc1",
        "offload-moe-act",
        "offload-fused-mlp",
    ],
)
def test_virtual_expert_recompute_offload_scopes(
    monkeypatch, grouped_mlp_fusions, precision, recompute, offload_scope, use_op_fuser
):
    """Replay MoonEP transfers and shared-expert input gradients across recompute/offload."""
    if precision == "mxfp8" and torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("MXFP8 requires Blackwell")
    if offload_scope == "expert_fc1":
        pytest.skip("FC1 offload needs a separate HybridEP/TE capacity-buffer fix.")

    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.tensor_parallel import checkpoint
    from megatron.core.transformer.spec_utils import get_submodules

    monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "0")
    monkeypatch.setattr(
        fused_a2a.HybridEPBuffer,
        "__init__",
        partialmethod(fused_a2a.HybridEPBuffer.__init__, load_cached_kernels=True),
    )
    Utils.initialize_model_parallel(expert_model_parallel_size=2)
    pg = ProcessGroupCollection.use_mpu_process_groups()
    spec = get_submodules(
        get_gpt_layer_with_transformer_engine_spec(
            num_experts=4, moe_grouped_gemm=True
        ).submodules.mlp
    )

    def run(enabled):
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        model_parallel_cuda_manual_seed(1234)
        torch.manual_seed(1234)
        overrides = dict(
            num_moe_experts=4,
            ffn_hidden_size=256,
            moe_ffn_hidden_size=256,
            moe_router_topk=2,
            moe_shared_expert_intermediate_size=256,
            moe_router_score_function="sigmoid",
            moe_router_topk_scaling_factor=3.16,
            moe_router_load_balancing_type="none",
            moe_aux_loss_coeff=0,
            gated_linear_unit=False,
            activation_func=squared_relu,
            use_fused_weighted_squared_relu=True,
            use_transformer_engine_op_fuser=use_op_fuser,
            moe_use_grouped_tensor=True,
            min_offloaded_tensor_size=0,
        )
        if precision == "mxfp8":
            overrides.update(
                fp8="e4m3",
                fp8_recipe="mxfp8",
                fp8_param=True,
                moe_router_padding_for_quantization=True,
            )
        if enabled and recompute:
            overrides.update(
                recompute_granularity="full" if recompute == "full" else "selective",
                recompute_modules=[] if recompute == "full" else [recompute],
            )
            if recompute == "full":
                overrides.update(recompute_method="uniform", recompute_num_layers=1)
        if enabled and offload_scope:
            overrides.update(
                fine_grained_activation_offloading=True, offload_modules=[offload_scope]
            )
        config = _virtual_expert_hybridep_config(**overrides)
        layer = None
        try:
            with get_fp8_context(config, is_init=True):
                layer = MoELayer(config, spec, pg_collection=pg).cuda()
            weights = {
                name: parameter.detach().float().cpu().clone()
                for name, parameter in layer.named_parameters()
            }
            for parameter in layer.parameters():
                parameter.main_grad = torch.zeros_like(parameter, dtype=torch.float32)
                parameter.grad_added_to_main_grad = False
            with torch.no_grad():
                layer.router.weight.zero_()
                layer.router.weight[:, 0].copy_(torch.tensor([1, 0.5, -0.5, -1], device="cuda"))
                layer.router.weight[2, 1] = 2
            if enabled and offload_scope:
                offload.init_chunk_handler(
                    pp_rank=pg.pp.rank(),
                    vp_size=None,
                    vp_stage=None,
                    min_offloaded_tensor_size=0,
                    delta_offload_bytes_across_pp_ranks=0,
                    activation_offload_fraction=1.0,
                )
            inputs, outputs, upstreams = [], [], []
            for use in range(2):
                generator = torch.Generator(device="cuda").manual_seed(
                    8765 + 10 * use + pg.ep.rank()
                )
                x = torch.randn(
                    32, 1, 128, device="cuda", dtype=torch.bfloat16, generator=generator
                )
                x[..., 0] = 1 if use == 0 else -1
                x.requires_grad_()
                if enabled and recompute == "full":

                    def replay(input_tensor):
                        with get_fp8_context(config):
                            return layer(input_tensor)[0]

                    y = checkpoint(replay, False, x)
                else:
                    with get_fp8_context(config):
                        y, bias = layer(x)
                    assert bias is None
                inputs.append(x)
                outputs.append(y)
                upstreams.append(
                    torch.randn(y.shape, device="cuda", dtype=y.dtype, generator=generator)
                )
            for y, upstream in reversed(list(zip(outputs, upstreams))):
                y.backward(upstream)
            torch.cuda.synchronize()
            values = {
                **{f"output {use}": y.detach().cpu() for use, y in enumerate(outputs)},
                **{f"input gradient {use}": x.grad.cpu() for use, x in enumerate(inputs)},
                **{
                    f"expert gradient {name}": parameter.main_grad.cpu().clone()
                    for name, parameter in layer.experts.named_parameters()
                },
            }
            if enabled and offload_scope:
                offload.reset(process_group=pg.tp_dp_cp)
                assert (
                    PipelineOffloadManager.get_instance().offload_summary_bytes[offload_scope] > 0
                )
            return weights, values
        finally:
            torch.cuda.synchronize()
            del layer
            VirtualExpertLoadBalancer.finalize()
            FP8GlobalStateManager.reset()
            offload.reset_instance()
            destroy_moe_metrics_tracker()
            gc.collect()
            fused_a2a.reset_hybrid_ep_buffer()

    try:
        reference_weights, reference = run(False)
        actual_weights, actual = run(True)
        assert actual_weights.keys() == reference_weights.keys()
        for name, expected in reference_weights.items():
            _assert_bitwise_parity(actual_weights[name], expected, f"initial {name}")
        assert actual.keys() == reference.keys()
        for name, expected in reference.items():
            _assert_bitwise_parity(actual[name], expected, name)
    finally:
        Utils.destroy_model_parallel()


@requires_hybridep
def test_virtual_expert_manager_rejects_mismatched_topk():
    """A supplied transport top-k must satisfy the planner's fixed-layout contract."""
    Utils.initialize_model_parallel(expert_model_parallel_size=2)
    try:
        config = _virtual_expert_hybridep_config(moe_router_topk=1, moe_router_pre_softmax=True)
        with pytest.raises(ValueError, match="topk=2"):
            _VirtualExpertHybridEPManager(
                ps.get_expert_model_parallel_group(), 1, 2, config, router_topk=2
            )
    finally:
        Utils.destroy_model_parallel()


def test_virtual_expert_rejects_unsupported_layout():
    """Reject incompatible routing/storage layouts and partial MoE graph capture."""
    for overrides, match in (
        ({"moe_token_dispatcher_type": "alltoall"}, "moe_token_dispatcher_type='flex'"),
        ({"moe_hybridep_routing_map_mode": "bool"}, "moe_hybridep_routing_map_mode='indices'"),
        ({"expert_model_parallel_size": 1}, "2<=expert_model_parallel_size<=64"),
        ({"num_moe_experts": 3}, "num_moe_experts divisible"),
        ({"moe_router_topk": 3}, "1<=moe_router_topk"),
        ({"moe_expert_capacity_factor": 1.0}, "moe_expert_capacity_factor=None"),
    ):
        with pytest.raises(ValueError, match=match):
            _virtual_expert_hybridep_config(**overrides)
    for scope in ("moe_router", "moe_preprocess"):
        with pytest.raises(AssertionError, match="moe CUDA graph scope only"):
            _virtual_expert_hybridep_config(cuda_graph_impl="local", cuda_graph_modules=[scope])


@requires_cuda
@pytest.mark.parametrize(
    "mode,compact_supported",
    [
        ("quantile", False),
        ("unfused", False),
        ("fused", True),
        ("fused", False),
        ("hash", False),
        ("hash-fused", True),
    ],
    ids=[
        "quantile-dense",
        "unfused-dense",
        "fused-indices",
        "fused-fallback",
        "hash-dense",
        "hash-fusion-enabled",
    ],
)
def test_virtual_expert_upstream_router_formats(monkeypatch, mode, compact_supported):
    """Recipe-sized expert IDs and scores agree with a dense unfused router and CPU math."""
    monkeypatch.setattr(router_module, "HAVE_HYBRIDEP_DENSE_ROUTING", True)
    monkeypatch.setattr(
        router_module, "fused_topk_with_score_function_supports_topk_indices", compact_supported
    )
    quantile = mode == "quantile"
    hash_routing = mode in ("hash", "hash-fused")
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
        moe_router_fusion=mode in ("fused", "hash-fused"),
        moe_num_hash_layers=int(hash_routing),
        hash_moe_vocab_size=512 if hash_routing else None,
    )
    torch.manual_seed(321)
    logits_cpu = torch.randn(256, 512)
    if quantile or hash_routing:
        logits_cpu[0] = -100  # Selected routes must survive score underflow to zero.
    bias = torch.randn(512) * 0.3 if quantile else torch.zeros(512)
    scores = logits_cpu.sigmoid()
    selection = logits_cpu - bias if quantile else scores + bias
    expected_ids = selection.argsort(dim=1, descending=True)[:, :10].sort(dim=1).values
    routing_kwargs = {}
    if hash_routing:
        input_ids = torch.arange(256).reshape(2, 128)
        expected_ids = ((input_ids.t().reshape(-1, 1) + torch.arange(10)) % 512).sort(dim=1).values
        routing_kwargs["input_ids"] = input_ids.cuda()
    selected = scores.gather(1, expected_ids)
    expected_probs = 3.16 * selected / (selected.sum(dim=1, keepdim=True) + 1e-20)
    dy = torch.randn(256, 512, device="cuda")
    calls = []
    fused = moe_utils.fused_topk_with_score_function

    def record_fused(**kwargs):
        calls.append((virtual, kwargs["logits"].shape, kwargs.get("topk_indices")))
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
        if quantile:
            reference.qb_beta.copy_(bias)
        ref_logits = logits_cpu.cuda().requires_grad_()
        ref_probs, ref_map = reference.routing(ref_logits.view(128, 2, 512), **routing_kwargs)
        (ref_probs * dy).sum().backward()
        torch.testing.assert_close(
            ref_map.cpu(),
            torch.zeros_like(logits_cpu, dtype=torch.bool).scatter(1, expected_ids, True),
        )
        for virtual in (False, True):
            router = TopKRouter(replace(config, moe_virtual_expert_load_balance=virtual), pg).cuda()
            router.set_layer_number(1)
            if quantile:
                router.qb_beta.copy_(bias)
            logits = logits_cpu.cuda().requires_grad_()
            probs, ids = router.routing(logits.view(128, 2, 512), **routing_kwargs)
            if virtual:
                assert probs.shape == ids.shape == (256, 10)
                assert ids.dtype == torch.int64
                # Exercise the actual dispatcher adapter without allocating transport
                # storage; the CPU oracle below checks its IDs, weights and gradients.
                manager = object.__new__(_VirtualExpertHybridEPManager)
                manager.router_topk = config.moe_router_topk
                planned = []
                monkeypatch.setattr(manager, "plan_dispatch", planned.append)
                manager.setup_metadata(ids, probs)
                (ids,) = planned
                probs = manager.token_probs
            else:
                assert probs.shape == (256, 512)
                if ids.dtype == torch.bool:
                    ids = ids.to(torch.int8).topk(10, dim=1).indices
                else:
                    # Upstream fused HybridEP routing returns int16 ids with full-width probs.
                    assert ids.shape == (256, 10) and ids.dtype == torch.int16
                    ids = ids.long()
                probs = probs.gather(1, ids)
            assert probs.shape == ids.shape == (256, 10) and ids.dtype == torch.int64
            sorted_ids, order = ids.sort(dim=1)
            torch.testing.assert_close(sorted_ids.cpu(), expected_ids, rtol=0, atol=0)
            torch.testing.assert_close(
                probs.gather(1, order).cpu(), expected_probs, rtol=1e-5, atol=1e-7
            )
            assert ids.max() > 64
            if quantile or hash_routing:
                assert not probs[0].any()
            (probs * dy.gather(1, ids)).sum().backward()
            torch.testing.assert_close(logits.grad, ref_logits.grad, rtol=2e-4, atol=1e-6)
            assert logits.grad[1:].norm() > 0
            if virtual:
                # Exercise the actual router plus metadata path, including probability
                # gradients. Quantile beta accumulation is checked separately below.
                def route(logits):
                    scores, indices = router.routing(logits, **routing_kwargs)
                    manager.setup_metadata(indices, scores)
                    return manager.token_probs, indices

                router.eval()
                # Importing the kernel-suite harness enables process-wide determinism.
                # Keep this replay local so collecting this module cannot change other tests.
                replay_reference = None
                for replay in range(3):
                    with RacingStreams() if replay else nullcontext():
                        replay_logits = logits.detach().view(128, 2, 512).clone().requires_grad_()
                        replay_scores, replay_ids = route(replay_logits)
                        (replay_grad,) = torch.autograd.grad(
                            replay_scores, replay_logits, grad_outputs=dy.gather(1, ids)
                        )
                        observed = tuple(
                            value.detach().clone()
                            for value in (replay_scores, replay_ids, replay_grad)
                        )
                    torch.cuda.synchronize()
                    if replay_reference is None:
                        replay_reference = observed
                    else:
                        for name, actual, expected in zip(
                            ("scores", "indices", "gradients"), observed, replay_reference
                        ):
                            _assert_bitwise_parity(actual, expected, f"router[{mode}] {name}")
                with torch.no_grad():
                    static_logits = logits.detach().view(128, 2, 512).clone()
                    # Warm up before capture; replay must read new scores, not cached IDs.
                    route(static_logits)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured_scores, captured_ids = route(static_logits)
                    changed_logits = static_logits.flip(0).contiguous()
                    expected_scores, expected_indices = route(changed_logits)
                    static_logits.copy_(changed_logits)
                    graph.replay()
                    torch.testing.assert_close(captured_ids, expected_indices, rtol=0, atol=0)
                    torch.testing.assert_close(captured_scores, expected_scores, rtol=0, atol=0)
                router.train()
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
        for virtual, shape, index_buffer in calls:
            assert shape == (256, 512)
            if compact_supported:
                assert index_buffer.shape == (256, 10)
                assert index_buffer.dtype == (torch.int64 if virtual else torch.int16)
            else:
                assert index_buffer is None
                assert not virtual  # MoonEP uses the unfused dense helper in this case.
        assert bool(calls) == (mode == "fused")
    finally:
        destroy_moe_metrics_tracker()
        Utils.destroy_model_parallel()
