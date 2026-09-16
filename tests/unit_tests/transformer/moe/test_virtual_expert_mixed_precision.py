# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Production precision overrides with MXFP8 main experts and a repeated BF16 MTP expert."""

import gc

import pytest
import torch
from transformer_engine.pytorch.quantization import FP8GlobalStateManager

from megatron.core.activations import squared_relu
from megatron.core.fp8_utils import get_fp8_context, is_mxfp8tensor
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.quantization.quant_config import RecipeConfig
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe import fused_a2a
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.virtual_expert_load_balancer import VirtualExpertLoadBalancer
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.moe.test_virtual_expert_hybridep import (
    MXFP8_COMPONENTS,
    _assert_numerical_parity,
    _set_main_grad,
)

pytestmark = pytest.mark.launch_on_gb200


@pytest.mark.parametrize("grad_dtype", [torch.float32, torch.bfloat16])
def test_virtual_expert_mixed_precision_repeated_layer_matches_hybridep(monkeypatch, grad_dtype):
    """Keep MTP BF16, share only compatible slots, and match outputs and all training gradients."""
    monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")
    Utils.initialize_model_parallel(expert_model_parallel_size=4)
    model_parallel_cuda_manual_seed(1234)
    recipe = RecipeConfig.from_config_dict(
        {
            "configs": {
                "bf16": {
                    "transformer_engine_config_type": "TEQuantizationParams",
                    "training_recipe": {},
                }
            },
            "matchers": {
                "mtp": {
                    "type": "glob",
                    "enabled": True,
                    "pattern": "*mtp.layers.*",
                    "config": "bf16",
                }
            },
        }
    )
    common = dict(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        ffn_hidden_size=256,
        moe_ffn_hidden_size=256,
        num_moe_experts=8,
        moe_router_topk=2,
        expert_model_parallel_size=4,
        bf16=True,
        params_dtype=torch.bfloat16,
        add_bias_linear=False,
        gated_linear_unit=False,
        activation_func=squared_relu,
        gradient_accumulation_fusion=True,
        moe_use_grouped_tensor=True,
        use_fused_weighted_squared_relu=True,
        activation_func_tanh_clamp_scale=16,
        moe_grouped_gemm=True,
        use_transformer_engine_op_fuser=True,
        moe_router_dtype="fp32",
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="hybridep",
        moe_router_load_balancing_type="none",
        fp8="e4m3",
        fp8_recipe="mxfp8",
        fp8_param=True,
        moe_router_padding_for_quantization=True,
        quant_recipe=recipe,
        recompute_granularity="selective",
        recompute_modules=["moe_act"],
    )
    if grad_dtype == torch.bfloat16:
        common.update(
            moe_router_load_balancing_type="quantile_balancing",
            moe_router_score_function="sigmoid",
            moe_router_quantile_balancing_estimation_scope="global_batch",
            moe_router_fusion=True,
            moe_aux_loss_coeff=0.0,
        )
    spec = get_submodules(
        get_gpt_layer_with_transformer_engine_spec(
            num_experts=8, moe_grouped_gemm=True
        ).submodules.mlp
    )

    def build(enabled):
        config = TransformerConfig(**common, moe_virtual_expert_load_balance=enabled)
        with get_fp8_context(config, is_init=True):
            layers = torch.nn.ModuleList(
                MoELayer(config, spec, name=name).cuda()
                for name in ("decoder.layers.0.mlp", "mtp.layers.0.mlp")
            )
        for parameter in layers.parameters():
            _set_main_grad(parameter, grad_dtype)
        assert is_mxfp8tensor(layers[0].experts.linear_fc1.weight0)
        assert not is_mxfp8tensor(layers[1].experts.linear_fc1.weight0)
        assert not layers[1].experts._with_fused_impl
        return config, layers

    try:
        ref_config, reference = build(False)
        ve_config, virtual = build(True)
        virtual.load_state_dict(reference.state_dict())
        with torch.no_grad():
            for source, target in zip(reference.parameters(), virtual.parameters()):
                if is_mxfp8tensor(source):
                    for component in MXFP8_COMPONENTS:
                        getattr(target, component).copy_(getattr(source, component))
        managers = [layer.token_dispatcher._comm_manager for layer in virtual]
        bf16_calls = []
        make_ops = virtual[1].experts._make_fused_ops

        def check_bf16(module, inputs):
            assert not FP8GlobalStateManager.is_fp8_enabled()
            bf16_calls.append(module)

        def checked_ops():
            ops = make_ops()
            ops[0].register_forward_pre_hook(check_bf16)
            ops[-1].register_forward_pre_hook(check_bf16)
            return ops

        monkeypatch.setattr(virtual[1].experts, "_make_fused_ops", checked_ops)
        plans = []
        for manager in managers:
            original = manager.plan_dispatch

            def record(*args, manager=manager, original=original):
                original(*args)
                plans.append(manager._plan)

            monkeypatch.setattr(manager, "plan_dispatch", record)

        generator = torch.Generator(device="cuda").manual_seed(2345 + Utils.rank)
        inputs = torch.randn(32, 1, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
        upstream = torch.randn(
            inputs.shape, device="cuda", dtype=torch.bfloat16, generator=generator
        )

        def run(config, layers):
            x = inputs.detach().clone().requires_grad_()
            for parameter in layers.parameters():
                parameter.main_grad.zero_()
                parameter.grad = None
            with get_fp8_context(config):
                y = layers[0](x)[0]
                # A residual retains enough signal for both repeated MTP uses after squared ReLU.
                y = y + x
                y = layers[1](y)[0] + y
                y = layers[1](y)[0] + y
            y.backward(upstream)
            return [y.detach().clone(), x.grad.clone()] + [
                p.main_grad.clone() for p in layers.parameters()
            ]

        # Keep each arm's HybridEP buffer alive across passes. Their native/runtime expert
        # counts differ, so switch buffers once instead of recompiling transport every pass.
        reference_values = [run(ref_config, reference) for _ in range(3)]
        torch.cuda.synchronize()
        torch.distributed.barrier()
        fused_a2a.reset_hybrid_ep_buffer()
        for step, expected in enumerate(reference_values):
            actual = run(ve_config, virtual)
            assert len(VirtualExpertLoadBalancer.storages) == 2
            assert managers[0].virtual_experts.storage is not managers[1].virtual_experts.storage
            assert all(
                not hasattr(parameter, "grad_added_to_main_grad") and parameter.grad is None
                for manager in managers
                for fc in manager.virtual_experts.runtime_weights
                for parameter in fc
            )
            for manager in managers:
                owner = manager.virtual_experts
                assert not any(owner.config.gtp)
                for sources, runtime in zip(owner.parameters, owner.runtime_weights):
                    assert all(
                        native.main_grad.data_ptr() == source.main_grad.data_ptr()
                        and not native.overwrite_main_grad
                        for source, native in zip(sources, runtime)
                    )
            assert all(
                manager._plan is None and not manager.over_budget.item() for manager in managers
            )
            assert any((plan.experts_to_copy >= 0).any().item() for plan in plans)
            for index, (value, ref) in enumerate(zip(actual, expected)):
                if ref.count_nonzero() == 0:
                    torch.testing.assert_close(value, ref, rtol=0, atol=0)
                else:
                    _assert_numerical_parity(value, ref, 0.06, f"step {step}, tensor {index}")
        assert len(bf16_calls) == 12
        assert reference.state_dict().keys() == virtual.state_dict().keys()
    finally:
        VirtualExpertLoadBalancer.finalize()
        gc.collect()
        fused_a2a.reset_hybrid_ep_buffer()
        Utils.destroy_model_parallel()
