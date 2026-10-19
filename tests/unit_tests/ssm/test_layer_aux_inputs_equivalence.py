# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Bitwise forward/backward checks of legacy and canonical output/bias entry points."""

import copy
from dataclasses import replace
from functools import partial

import pytest
import torch

from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
from megatron.core.models.hybrid.hybrid_layer_specs import (
    gated_delta_product_stack_spec,
    hybrid_stack_spec,
    wide_residual_hybrid_stack_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_product import HAVE_EINOPS, HAVE_FLA, HAVE_MAMBA_SSM
from megatron.core.tensor_parallel.random import (
    CheckpointWithoutOutputManager,
    get_cuda_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.layer_interface import LayerAuxInputs, LayerAuxOutputs
from megatron.core.transformer.moe.fused_a2a import HAVE_DEEP_EP
from megatron.core.transformer.residual_recompute import ResidualStreamRecomputeContext
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.transformer.wide_residual_config import WideResidualConfig
from tests.unit_tests.determinism.utils import assert_bit_exact
from tests.unit_tests.test_utilities import Utils


def make_config(**config_kwargs):
    defaults = dict(
        hidden_size=256,
        num_layers=1,
        num_attention_heads=16,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        hidden_dropout=0.1,
        attention_dropout=0.0,
        add_bias_linear=False,
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_head_dim=64,
        qk_pos_emb_head_dim=32,
        v_head_dim=64,
        rope_type='rope',
        rotary_base=10000,
        rotary_percent=1.0,
        dsa_indexer_n_heads=8,
        dsa_indexer_head_dim=64,
        dsa_indexer_topk=32,
        num_moe_experts=2,
        moe_router_topk=2,
        moe_shared_expert_intermediate_size=256,
        moe_token_dispatcher_type="alltoall",
        activation_func=torch.nn.functional.silu,
    )
    defaults.update(config_kwargs)
    return MLATransformerConfig(**defaults)


def make_stack(symbol, *, stack_spec=hybrid_stack_spec, **config_kwargs):
    model_parallel_cuda_manual_seed(123)
    config_kwargs.setdefault("num_layers", len(symbol))
    config = make_config(**config_kwargs)
    return (
        HybridStack(
            config,
            stack_spec.submodules,
            layer_config_list=validate_segment_layers(symbol, config),
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
            post_layer_norm=False,
            post_process=False,
        )
        .cuda()
        .train()
    )


def apply_output_bias(output):
    assert isinstance(output, tuple) and len(output) == 2
    hidden_states, aux_outputs = output
    bias = aux_outputs.bias if isinstance(aux_outputs, LayerAuxOutputs) else aux_outputs
    assert isinstance(hidden_states, torch.Tensor)
    assert bias is None or isinstance(bias, torch.Tensor)
    return hidden_states if bias is None else hidden_states + bias


def assert_forward_backward(module, x, legacy_call, inputs_call):
    """Replay on the same weights and RNG state; compare every populated gradient."""
    cuda_state = torch.cuda.get_rng_state()
    tp_state = copy.deepcopy(get_cuda_rng_tracker().get_states())

    def run(call):
        module.zero_grad(set_to_none=True)
        x.grad = None
        output = call(x)
        saved_output = output.detach().clone()
        output.float().square().sum().backward()
        grads = {
            name: p.grad.detach().clone()
            for name, p in module.named_parameters()
            if p.grad is not None
        }
        grads["input"] = x.grad.detach().clone()
        return saved_output, grads

    expected, expected_grads = run(legacy_call)
    torch.cuda.set_rng_state(cuda_state)
    get_cuda_rng_tracker().set_states(tp_state)
    actual, actual_grads = run(inputs_call)
    assert_bit_exact(expected, expected_grads, actual, actual_grads)


class TestLayerAuxInputsEquivalence:
    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(
        not (HAVE_EINOPS and HAVE_FLA and HAVE_MAMBA_SSM), reason="Requires GDP dependencies"
    )
    @pytest.mark.parametrize("test_core", [False, True])
    def test_gdp_core_and_skeleton(self, test_core):
        check_equivalence(
            "M",
            test_core,
            stack_spec=gated_delta_product_stack_spec,
            mamba_num_heads=4,
            mamba_head_dim=64,
            mamba_num_groups=2,
            mamba_state_dim=128,
        )

    @pytest.mark.parametrize("symbol", ["*", "-", "E", "M", "G", "+", "D"])
    @pytest.mark.parametrize("test_core", [False, True])
    def test_real_cores_and_skeletons(self, symbol, test_core):
        check_equivalence(symbol, test_core)

    @pytest.mark.parametrize("test_core", [False, True])
    def test_gdn2_core_and_skeleton(self, test_core):
        check_equivalence("G", test_core, experimental_attention_variant="gdn2")

    @pytest.mark.parametrize(
        "dispatcher,overlap",
        [
            ("alltoall", False),
            ("alltoall", True),
            ("allgather", False),
            ("flex", False),
            ("flex", True),
        ],
    )
    @pytest.mark.parametrize("pre_norm", [False, True])
    def test_moe_output_pair_and_pre_norm(self, dispatcher, overlap, pre_norm):
        if dispatcher == "flex":
            if not HAVE_DEEP_EP:
                pytest.skip("DeepEP not installed")
            if Utils.world_size < 2 or Utils.world_size % 2:
                pytest.skip("DeepEP equivalence requires an even world size >= 2")
            Utils.destroy_model_parallel()
            Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=2)
        ep_size = 2 if dispatcher == "flex" else 1
        spec = hybrid_stack_spec
        if pre_norm:
            moe_layer = spec.submodules.moe_layer
            mlp = moe_layer.submodules.mlp
            mlp = partial(mlp, submodules=replace(mlp.keywords["submodules"], pre_norm=TENorm))
            moe_layer = replace(moe_layer, submodules=replace(moe_layer.submodules, mlp=mlp))
            spec = replace(spec, submodules=replace(spec.submodules, moe_layer=moe_layer))
        stack = make_stack(
            "E",
            stack_spec=spec,
            moe_token_dispatcher_type=dispatcher,
            moe_shared_expert_overlap=overlap,
            moe_flex_dispatcher_backend="deepep" if dispatcher == "flex" else None,
            expert_model_parallel_size=ep_size,
        )
        core = stack.layers[0].mlp
        x = torch.randn(64, 2, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)

        def adapted(value):
            output, aux_outputs = core(value, LayerAuxInputs())
            assert aux_outputs.bias is None
            return output

        assert_forward_backward(core, x, lambda value: core(core.pre_norm(value))[0], adapted)

    @pytest.mark.parametrize("symbol", ["*", "-", "M"])
    def test_residual_stream_recompute(self, symbol):
        stack = make_stack(
            symbol,
            stack_spec=wide_residual_hybrid_stack_spec,
            wide_residual=WideResidualConfig(num_streams=2),
            recompute_granularity="selective",
            recompute_modules=["residual_stream"],
        )
        layer = stack.layers[0]
        x = torch.randn(32, 2, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)

        def run(value, shared):
            context = ResidualStreamRecomputeContext(CheckpointWithoutOutputManager(), True)
            if shared:
                output, aux_outputs = layer(
                    value, LayerAuxInputs(residual_stream_recompute_context=context)
                )
                assert aux_outputs.bias is None
            else:
                output = layer(value, residual_stream_recompute_context=context)
                if symbol != "M":
                    output = output[0]
            context.finalize(output)
            return output

        assert_forward_backward(
            layer, x, lambda value: run(value, False), lambda value: run(value, True)
        )

    @pytest.mark.parametrize("symbol", ["*", "-", "M"])
    def test_mhc_recompute(self, symbol):
        stack = make_stack(
            symbol,
            enable_mhc_connections=True,
            mhc_num_residual_streams=2,
            recompute_granularity="selective",
            recompute_modules=["mhc"],
            mhc_recompute_layer_num=1,
        )
        layer = stack.layers[0]
        x = torch.randn(32, 2, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)

        def run(value, shared):
            manager = CheckpointWithoutOutputManager()
            manager.is_last_layer_in_recompute_block = True
            if shared:
                output, aux_outputs = layer(value, LayerAuxInputs(mhc_recompute_manager=manager))
                assert aux_outputs.bias is None
            else:
                output = layer.forward.__wrapped__(
                    layer, value, LayerAuxInputs(mhc_recompute_manager=manager)
                )[0]
            manager.discard_all_outputs_and_register_unified_recompute(output)
            return output

        assert_forward_backward(
            layer, x, lambda value: run(value, False), lambda value: run(value, True)
        )

    @pytest.mark.parametrize("symbol", ["*", "M"])
    def test_static_inference_prefill_decode(self, symbol):
        layer = make_stack(symbol, hidden_dropout=0.0).layers[0].eval()
        contexts = [StaticInferenceContext(2, 33), StaticInferenceContext(2, 33)]
        with torch.inference_mode():
            for offset, length in [(0, 32), (32, 1)]:
                x = torch.randn(length, 2, 256, device="cuda", dtype=torch.bfloat16)
                outputs = []
                for shared, context in zip([False, True], contexts):
                    context.sequence_len_offset = offset
                    if offset:
                        context.enable_decode_mode()
                    sequence_offset = torch.full((2,), offset, device="cuda", dtype=torch.int32)
                    if shared:
                        output, aux_outputs = layer(
                            x,
                            LayerAuxInputs(
                                inference_context=context, sequence_len_offset=sequence_offset
                            ),
                        )
                        assert aux_outputs.bias is None
                    else:
                        kwargs = {"sequence_len_offset": sequence_offset} if symbol == "*" else {}
                        output = layer(x, inference_context=context, **kwargs)
                        if symbol != "M":
                            output = output[0]
                    outputs.append(output.detach().clone())
                assert_bit_exact(outputs[0], {}, outputs[1], {})
                left, right = contexts
                assert left.key_value_memory_dict.keys() == right.key_value_memory_dict.keys()
                for key in left.key_value_memory_dict:
                    for a, b in zip(
                        left.key_value_memory_dict[key], right.key_value_memory_dict[key]
                    ):
                        # Attention caches allocate the full maximum length; compare only written KV.
                        if symbol == "*":
                            a, b = a[: offset + length], b[: offset + length]
                        assert_bit_exact(a, {}, b, {})


def check_equivalence(symbol, test_core, *, stack_spec=hybrid_stack_spec, **config_kwargs):
    stack = make_stack(symbol, stack_spec=stack_spec, **config_kwargs)
    layer = stack.layers[0]
    module = layer
    if test_core:
        if symbol == "M":
            module = layer.mixer
        elif symbol in ("-", "E"):
            module = layer.mlp
        else:
            module = layer.self_attention
    x = torch.randn(64, 2, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    mask = torch.ones(2, 1, 64, 64, device="cuda", dtype=torch.bool).triu(1)

    def legacy(value):
        output = (
            module(value)
            if symbol == "M" or (test_core and symbol in ("-", "E"))
            else module(value, attention_mask=mask)
        )
        if test_core:
            result, bias = output
            return result if bias is None else result + bias
        return output if symbol == "M" else output[0]

    def adapted(value):
        output = module(value, LayerAuxInputs(attention_mask=mask))
        if not test_core:
            assert output[1].bias is None
        return apply_output_bias(output)

    assert_forward_backward(module, x, legacy, adapted)


class TestLayerAuxInputsCudaGraphs:
    def setup_method(self):
        from megatron.core.tensor_parallel.random import initialize_rng_tracker

        Utils.initialize_model_parallel(1, 1)
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)

    def teardown_method(self):
        from megatron.core.tensor_parallel.random import initialize_rng_tracker
        from megatron.core.transformer.cuda_graphs import delete_cuda_graphs

        delete_cuda_graphs()
        initialize_rng_tracker(force_reset=True)
        Utils.destroy_model_parallel()

    def test_hybrid_local_capture_replay_matches_eager(self):
        from megatron.core.transformer.cuda_graphs import create_cudagraphs

        eager = make_stack("*M", hidden_dropout=0.0)
        graphed = make_stack(
            "*M",
            hidden_dropout=0.0,
            cuda_graph_impl="local",
            cuda_graph_modules=[],
            cuda_graph_warmup_steps=1,
        )
        graphed.load_state_dict(eager.state_dict())
        for param in graphed.parameters():
            param.main_grad = torch.zeros_like(param)
        x = torch.randn(32, 2, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        expected = eager(x, attention_mask=None)
        expected_value = expected.detach().clone()
        expected.float().sum().backward()
        expected_grad = x.grad.detach().clone()
        x.grad = None
        recorded = graphed(x, attention_mask=None)
        assert_bit_exact(expected_value, {}, recorded, {})
        recorded.float().sum().backward()
        create_cudagraphs()
        for _ in range(2):
            x.grad = None
            replayed = graphed(x, attention_mask=None)
            # Backward replay can reuse forward graph buffers; snapshot the value first.
            replayed_value = replayed.detach().clone()
            replayed.float().sum().backward()
            assert_bit_exact(
                expected_value, {"input": expected_grad}, replayed_value, {"input": x.grad}
            )
        assert all(layer.cudagraph_manager.cudagraph_runners for layer in graphed.layers)


@pytest.mark.internal
class TestLayerAuxInputsNVLS:
    def setup_method(self):
        if Utils.world_size < 2 or Utils.world_size & (Utils.world_size - 1):
            pytest.skip("NVLS communication requires a power-of-two world size >= 2")
        if torch.cuda.get_device_capability()[0] < 9:
            pytest.skip("NVLS requires Hopper or newer GPUs with NVLink")
        Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=Utils.world_size)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self):
        from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
        from megatron.core.transformer.moe.token_dispatcher_inference import (
            NVLSAllGatherVDispatcher,
        )

        torch.cuda.synchronize()
        NVLSAllGatherVDispatcher._delete_buffers()
        SymmetricMemoryManager.destroy()
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("overlap", [False, True])
    def test_nvls_core_prefill_decode(self, overlap):
        """Inference kernels have no backward; compare the real NVLS core at both phases."""
        from megatron.core.inference.utils import InferenceMode
        from megatron.core.models.gpt.moe_module_specs import get_inference_optimized_moe_spec
        from megatron.core.transformer.moe.token_dispatcher_inference import (
            NVLSAllGatherVDispatcher,
        )
        from tests.unit_tests.inference.test_moe_dispatching_and_routing import _make_base_config

        config = _make_base_config(
            expert_model_parallel_size=Utils.world_size,
            inference_grouped_gemm_backend="torch",
            inference_moe_token_dispatcher_type="nvls",
            moe_shared_expert_overlap=overlap,
            cuda_graph_impl="none",
        )
        groups = ProcessGroupCollection.use_mpu_process_groups()
        NVLSAllGatherVDispatcher.allocate_buffers(
            per_rank_worst_case_token_count=128,
            topk=config.moe_router_topk,
            hidden_size=config.hidden_size,
            ep_group=groups.ep,
        )
        core = get_inference_optimized_moe_spec()(config=config, pg_collection=groups).cuda().eval()
        context = StaticInferenceContext(max_batch_size=2, max_sequence_length=33)
        with torch.inference_mode(), InferenceMode.active():
            for offset, length in [(0, 32), (32, 1)]:
                context.sequence_len_offset = offset
                if offset:
                    context.enable_decode_mode()
                x = torch.randn(length, 2, config.hidden_size, device="cuda", dtype=torch.bfloat16)
                legacy, bias = core(x)
                assert bias is None
                legacy = legacy.detach().clone()
                output, aux_outputs = core(x, LayerAuxInputs(inference_context=context))
                assert aux_outputs.bias is None
                assert_bit_exact(legacy, {}, output, {})
                assert core.token_dispatcher is core._inference_token_dispatcher
                assert core.token_dispatcher._shared_expert_output is None
