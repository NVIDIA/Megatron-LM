# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Small HybridModel parity: GTP/EGTP, MXFP8, repeated MTP, and real CPU offload."""

import gc
from functools import partialmethod

import pytest
import torch
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
from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.moe import fused_a2a
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.moe_logging import destroy_moe_metrics_tracker
from megatron.core.transformer.moe.virtual_expert_load_balancer import VirtualExpertLoadBalancer
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

pytestmark = [
    pytest.mark.internal,
    pytest.mark.skipif(
        not torch.cuda.is_available() or not fused_a2a.HAVE_HYBRIDEP,
        reason='requires CUDA and HybridEP',
    ),
]


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


@pytest.mark.launch_on_gb200
@pytest.mark.parametrize('egtp_size', [1, 2], ids=['ep2-mixed-mtp', 'ep2-egtp2-mxfp8'])
def test_virtual_expert_hybrid_training_parity(monkeypatch, egtp_size):
    """Match HybridEP losses, gradients and updates with repeated MTP, GTP/EGTP and offload."""
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip('MXFP8 requires Blackwell')
    monkeypatch.setenv('NVTE_CUTEDSL_FUSED_GROUPED_MLP', '1')
    monkeypatch.setenv('NVTE_CPU_OFFLOAD_V1', '1')
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
                # Measured maxima across four ranks: gradients <0.4%, Adam updates <1.9%.
                # Adam can amplify isolated sign changes in near-zero gradients.
                if name == 'loss':
                    tolerance, peak_tolerance = 1e-4, 1e-4
                elif name.startswith('optimizer update'):
                    tolerance, peak_tolerance = 0.02, 2.1
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


@pytest.mark.launch_on_gb200
def test_mxfp8_expert_recompute_parity():
    """Selective activation recomputation preserves outputs and every expert/input gradient."""
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.transformer.spec_utils import get_submodules

    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("MXFP8 requires Blackwell")
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
