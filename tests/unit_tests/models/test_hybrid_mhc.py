# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from contextlib import nullcontext

import pytest
import torch

from megatron.core.models.hybrid.hybrid_block import (
    HybridStack,
    HybridStackSubmodules,
    HyperConnectionHybridLayer,
)
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import (
    MegatronModule,
    convert_module_to_dtype_except_fp32_marked,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from tests.unit_tests.test_utilities import Utils


class _DummyHybridLayer(MegatronModule):
    """Same-shape residual layer used to isolate HybridStack mHC plumbing."""

    def __init__(self, config: TransformerConfig, layer_number: int, **_kwargs):
        super().__init__(config=config)
        self.layer_number = layer_number
        self.proj = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.seen_hidden_shapes = []

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        inference_context=None,
        packed_seq_params=None,
        **_kwargs,
    ):
        self.seen_hidden_shapes.append(tuple(hidden_states.shape))
        return hidden_states + 0.125 * self.proj(hidden_states)


def _get_pg_collection() -> ProcessGroupCollection:
    return ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'pp', 'cp'])


def _get_dummy_submodules() -> HybridStackSubmodules:
    layer_spec = ModuleSpec(module=_DummyHybridLayer)
    return HybridStackSubmodules(
        mamba_layer=layer_spec,
        gdn_layer=layer_spec,
        attention_layer=layer_spec,
        dsa_layer=layer_spec,
        mlp_layer=layer_spec,
        moe_layer=layer_spec,
    )


def _get_config(num_layers: int, **kwargs) -> TransformerConfig:
    return TransformerConfig(
        num_layers=num_layers,
        hidden_size=32,
        num_attention_heads=4,
        use_cpu_initialization=True,
        enable_hyper_connections=True,
        num_residual_streams=2,
        hidden_dropout=0.0,
        mhc_sinkhorn_iterations=3,
        **kwargs,
    )


def _get_stack(
    config: TransformerConfig,
    num_local_layers: int,
    *,
    pre_process: bool = True,
    post_process: bool = True,
    pp_layer_offset: int = 0,
) -> HybridStack:
    return HybridStack(
        config=config,
        submodules=_get_dummy_submodules(),
        pre_process=pre_process,
        post_process=post_process,
        post_layer_norm=False,
        layer_type_list=[Symbols.MAMBA] * num_local_layers,
        pp_layer_offset=pp_layer_offset,
        pg_collection=_get_pg_collection(),
    )


@pytest.mark.internal
class TestHybridStackMHC:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("learned_output_contract", [False, True])
    def test_constructor_and_sharded_state(self, learned_output_contract):
        config = _get_config(num_layers=3, mhc_learned_output_contract=learned_output_contract)
        stack = _get_stack(config, num_local_layers=3)

        assert all(isinstance(layer, HyperConnectionHybridLayer) for layer in stack.layers)
        if learned_output_contract:
            assert stack.hc_head_fn.shape == (
                config.num_residual_streams,
                config.hidden_size * config.num_residual_streams,
            )
        state = stack.sharded_state_dict(prefix="decoder.", metadata={})
        for name in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
            assert hasattr(stack, name) is learned_output_contract
            assert (f"decoder.{name}" in state) is learned_output_contract

    @pytest.mark.parametrize("learned_output_contract", [False, True])
    def test_output_contract(self, learned_output_contract):
        config = _get_config(num_layers=1, mhc_learned_output_contract=learned_output_contract)
        stack = _get_stack(config, num_local_layers=1).cuda()
        residual_outputs = []

        def capture_residual_streams(_module, _inputs, output):
            output[0].retain_grad()
            residual_outputs.append(output[0])

        if learned_output_contract:
            with torch.no_grad():
                stack.hc_head_fn.zero_()
                stack.hc_head_base.copy_(torch.logit(torch.tensor([0.25, 0.75], device="cuda")))
        hidden = torch.randn(8, 2, config.hidden_size, device="cuda", requires_grad=True)
        with stack.layers[-1].register_forward_hook(capture_residual_streams):
            actual = stack(hidden, attention_mask=None)

        streams = residual_outputs[0].unflatten(-1, (2, config.hidden_size))
        weights = (
            [0.25 + config.layernorm_epsilon, 0.75 + config.layernorm_epsilon]
            if learned_output_contract
            else [0.5, 0.5]
        )
        expected = weights[0] * streams[..., 0, :] + weights[1] * streams[..., 1, :]
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        stream_grads = residual_outputs[0].grad.unflatten(-1, (2, config.hidden_size))
        for index, weight in enumerate(weights):
            torch.testing.assert_close(
                stream_grads[..., index, :], torch.full_like(streams[..., index, :], weight)
            )

    def test_wrapped_residual_layer_does_not_double_count_input(self):
        config = _get_config(num_layers=1)
        wrapper = _get_stack(config, num_local_layers=1).cuda().layers[0]
        hidden = torch.randn(8, 2, config.hidden_size * config.num_residual_streams, device="cuda")
        aggregated, h_res, h_post, residual = wrapper.hyper_connection(hidden)
        branch = 0.125 * wrapper.inner_layer.proj(aggregated)
        expected = wrapper.hyper_connection.fused_h_res_h_post_bda(
            h_res, residual, h_post, (branch, None), dropout_prob=0.0, training=True, fused=False
        )

        actual, _ = wrapper(hidden, attention_mask=None)

        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize(
        "recompute_kwargs",
        [
            {},
            {
                "recompute_granularity": "selective",
                "recompute_modules": ["core_attn", "mhc"],
                "mhc_recompute_layer_num": 2,
            },
            {
                "recompute_granularity": "full",
                "recompute_method": "uniform",
                "recompute_num_layers": 1,
            },
        ],
        ids=["none", "selective_mhc", "full"],
    )
    @pytest.mark.parametrize("learned_output_contract", [False, True])
    def test_forward_backward(self, recompute_kwargs, learned_output_contract):
        config = _get_config(
            num_layers=3, mhc_learned_output_contract=learned_output_contract, **recompute_kwargs
        )
        stack = _get_stack(config, num_local_layers=3).cuda()
        hidden_states = torch.randn(8, 2, config.hidden_size, device="cuda", requires_grad=True)

        output = stack(hidden_states, attention_mask=None)

        assert output.shape == hidden_states.shape
        assert torch.isfinite(output).all()
        output.float().sum().backward()
        assert hidden_states.grad is not None
        for layer in stack.layers:
            assert layer.inner_layer.proj.weight.grad is not None
            assert layer.hyper_connection.mapping_proj.weight.grad is not None
            assert all(
                shape == (8, 2, config.hidden_size)
                for shape in layer.inner_layer.seen_hidden_shapes
            )
        for name in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
            if learned_output_contract:
                assert getattr(stack, name).grad is not None
            else:
                assert not hasattr(stack, name)

    @pytest.mark.parametrize("recompute_method", ["uniform", "block"])
    @pytest.mark.parametrize(
        "precision, autocast_forward",
        [
            ("fp32", False),
            ("bf16_fp32_mixing", False),
            ("bf16_fused", False),
            ("bf16_fp32_mixing", True),
        ],
    )
    @pytest.mark.parametrize("learned_output_contract", [False, True])
    def test_full_recompute_matches_forward_backward(
        self, recompute_method, precision, learned_output_contract, autocast_forward
    ):
        dtype = torch.float32 if precision == "fp32" else torch.bfloat16
        config_kwargs = dict(
            num_layers=3,
            mhc_learned_output_contract=learned_output_contract,
            bf16=dtype == torch.bfloat16,
            params_dtype=dtype,
            use_fused_mhc=precision == "bf16_fused",
            mhc_norm_eps_inside_sqrt=precision == "bf16_fp32_mixing",
            mhc_keep_mappings_in_fp32=precision == "bf16_fp32_mixing",
        )
        reference = _get_stack(_get_config(**config_kwargs), num_local_layers=3).cuda()
        config = _get_config(
            **config_kwargs,
            recompute_granularity="full",
            recompute_method=recompute_method,
            recompute_num_layers=2,
        )
        checkpointed = _get_stack(config, num_local_layers=3).cuda()
        for stack in (reference, checkpointed):
            convert_module_to_dtype_except_fp32_marked(stack, dtype)
        checkpointed.load_state_dict(reference.state_dict())
        hidden = torch.randn(8, 2, config.hidden_size, device="cuda", dtype=dtype)
        reference_input = hidden.clone().requires_grad_()
        checkpointed_input = hidden.clone().requires_grad_()

        forward_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if autocast_forward
            else nullcontext()
        )
        with forward_context:
            expected = reference(reference_input, attention_mask=None)
            actual = checkpointed(checkpointed_input, attention_mask=None)
        torch.testing.assert_close(actual, expected)
        # Backward runs after the forward autocast scope has ended; checkpoint replay must
        # restore the forward autocast state internally.
        expected.float().square().mean().backward()
        actual.float().square().mean().backward()
        grad_close_kwargs = {"rtol": 1e-2, "atol": 1e-3} if autocast_forward else {}
        torch.testing.assert_close(
            checkpointed_input.grad, reference_input.grad, **grad_close_kwargs
        )
        for name, param in checkpointed.named_parameters():
            torch.testing.assert_close(
                param.grad, reference.get_parameter(name).grad, msg=name, **grad_close_kwargs
            )

    @pytest.mark.parametrize("recompute_method", ["uniform", "block"])
    def test_full_recompute_real_moe_matches_forward_backward(self, recompute_method):
        config_kwargs = dict(
            num_layers=2,
            num_moe_experts=2,
            moe_ffn_hidden_size=64,
            moe_grouped_gemm=True,
            moe_token_dispatcher_type="alltoall",
            add_bias_linear=False,
        )
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=['tp', 'pp', 'cp', 'ep', 'expt_tp', 'tp_ep', 'expt_dp']
        )

        def build_stack(config):
            return HybridStack(
                config=config,
                submodules=hybrid_stack_spec.submodules,
                post_layer_norm=False,
                layer_type_list=[Symbols.MOE] * 2,
                pg_collection=pg_collection,
            ).cuda()

        reference = build_stack(_get_config(**config_kwargs))
        checkpointed = build_stack(
            _get_config(
                **config_kwargs,
                recompute_granularity="full",
                recompute_method=recompute_method,
                recompute_num_layers=1,
            )
        )
        checkpointed.load_state_dict(reference.state_dict())
        hidden = torch.randn(8, 2, 32, device="cuda")
        reference_input = hidden.clone().requires_grad_()
        checkpointed_input = hidden.clone().requires_grad_()

        expected = reference(reference_input, attention_mask=None)
        actual = checkpointed(checkpointed_input, attention_mask=None)
        torch.testing.assert_close(actual, expected)
        expected.square().mean().backward()
        actual.square().mean().backward()
        torch.testing.assert_close(checkpointed_input.grad, reference_input.grad)
        for name, param in checkpointed.named_parameters():
            torch.testing.assert_close(param.grad, reference.get_parameter(name).grad, msg=name)
        for stack in (reference, checkpointed):
            for layer in stack.layers:
                assert layer.inner_layer.mlp.token_dispatcher.probs is None


class _ResidualOnlyLayer(MegatronModule):
    """A same-shape residual branch independent of attention and expert kernels."""

    def __init__(self, config, layer_number, **kwargs):
        super().__init__(config=config)
        self.layer_number = layer_number
        self.proj = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states, **kwargs):
        return hidden_states + 0.125 * self.proj(hidden_states)


def _config(**kwargs):
    return TransformerConfig(
        num_layers=3,
        hidden_size=32,
        num_attention_heads=4,
        num_residual_streams=4,
        enable_hyper_connections=True,
        hidden_dropout=0.0,
        mhc_sinkhorn_iterations=3,
        use_cpu_initialization=True,
        **kwargs,
    )


def _stack(config):
    return HybridStack(
        config=config,
        submodules=HybridStackSubmodules(mamba_layer=ModuleSpec(module=_ResidualOnlyLayer)),
        post_layer_norm=False,
        layer_type_list=[Symbols.MAMBA] * config.num_layers,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=["tp", "pp", "cp", "dp_cp"]
        ),
    ).cuda()


@pytest.mark.internal
class TestHybridMHCContraction:
    def setup_method(self):
        # These cases share mHC's compiled helpers and can exhaust Dynamo's recompilation
        # limit. Recompute parity requires the reference and checkpointed sides to compile.
        torch._dynamo.reset()
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self):
        Utils.destroy_model_parallel()
        torch._dynamo.reset()

    @pytest.mark.parametrize("learned_output_contract", [False, True])
    def test_output_contraction_and_head_parameter_inventory(self, learned_output_contract):
        config = _config(mhc_learned_output_contract=learned_output_contract)
        stack = _stack(config)
        sharded_state = stack.sharded_state_dict(prefix="decoder.", metadata={})
        for name in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
            assert hasattr(stack, name) is learned_output_contract
            assert (name in stack.state_dict()) is learned_output_contract
            assert (f"decoder.{name}" in sharded_state) is learned_output_contract

        streams_seen = []

        def capture_streams(module, inputs, output):
            streams_seen.append(output[0])
            output[0].retain_grad()

        weights = [0.125, 0.25, 0.5, 0.75] if learned_output_contract else [0.25] * 4
        if learned_output_contract:
            with torch.no_grad():
                stack.hc_head_fn.zero_()
                stack.hc_head_base.copy_(torch.logit(torch.tensor(weights, device="cuda")))
            weights = [weight + config.layernorm_epsilon for weight in weights]
        inputs = torch.randn(8, 2, config.hidden_size, device="cuda", requires_grad=True)
        with stack.layers[-1].register_forward_hook(capture_streams):
            actual = stack(inputs, attention_mask=None)
        streams = streams_seen[0].unflatten(-1, (4, config.hidden_size))
        expected = sum(weight * streams[..., index, :] for index, weight in enumerate(weights))
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        stream_grads = streams_seen[0].grad.unflatten(-1, (4, config.hidden_size))
        for index, weight in enumerate(weights):
            torch.testing.assert_close(
                stream_grads[..., index, :], torch.full_like(stream_grads[..., index, :], weight)
            )

    @pytest.mark.parametrize("learned_output_contract", [False, True])
    @pytest.mark.parametrize("recompute_method", ["uniform", "block"])
    @pytest.mark.parametrize(
        "fp32_mixing, autocast_forward", [(False, False), (True, False), (True, True)]
    )
    @pytest.mark.parametrize("distribute_saved_activations", [False, True])
    def test_full_recompute_matches_forward_and_all_gradients(
        self,
        learned_output_contract,
        recompute_method,
        fp32_mixing,
        autocast_forward,
        distribute_saved_activations,
    ):
        dtype = torch.bfloat16 if fp32_mixing else torch.float32
        config_kwargs = dict(
            mhc_learned_output_contract=learned_output_contract,
            mhc_norm_eps_inside_sqrt=fp32_mixing,
            mhc_keep_mappings_in_fp32=fp32_mixing,
            bf16=fp32_mixing,
            params_dtype=dtype,
        )
        baseline = _stack(_config(**config_kwargs))
        checkpointed = _stack(
            _config(
                **config_kwargs,
                recompute_granularity="full",
                recompute_method=recompute_method,
                recompute_num_layers=2,
                distribute_saved_activations=distribute_saved_activations,
            )
        )
        for stack in (baseline, checkpointed):
            convert_module_to_dtype_except_fp32_marked(stack, dtype)
        checkpointed.load_state_dict(baseline.state_dict(), strict=True)
        hidden = torch.randn(8, 2, 32, device="cuda", dtype=dtype)
        reference_input, actual_input = (hidden.clone().requires_grad_() for _ in range(2))
        forward_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if autocast_forward
            else nullcontext()
        )
        with forward_context:
            expected = baseline(reference_input, attention_mask=None)
            actual = checkpointed(actual_input, attention_mask=None)
        torch.testing.assert_close(actual, expected)
        # Backward runs after the forward autocast scope has ended; checkpoint replay must
        # restore the forward autocast state internally.
        grad = torch.randn_like(actual)
        expected.backward(grad)
        actual.backward(grad)
        grad_close_kwargs = {"rtol": 1e-2, "atol": 1e-3} if autocast_forward else {}
        torch.testing.assert_close(actual_input.grad, reference_input.grad, **grad_close_kwargs)
        for name, parameter in checkpointed.named_parameters():
            torch.testing.assert_close(
                parameter.grad, baseline.get_parameter(name).grad, msg=name, **grad_close_kwargs
            )
