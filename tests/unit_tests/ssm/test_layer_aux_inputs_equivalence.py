# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real forward/backward equivalence of explicit and packed layer invocations."""

import copy
from contextlib import nullcontext
from unittest.mock import patch

import pytest
import torch

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TEFusedMLP,
    TEFusedMLPWithGroupedLinear,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
from megatron.core.models.hybrid.hybrid_layer_specs import (
    gated_delta_product_stack_spec,
    hybrid_stack_spec,
    wide_residual_hybrid_stack_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_product import HAVE_EINOPS, HAVE_FLA, HAVE_MAMBA_SSM
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import CrossAttention, CrossAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.dsa import hadamard_transform
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.layer_interface import (
    LayerAuxInputs,
    LayerAuxOutputs,
    LayerExecutionContext,
    LayerInterface,
)
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionBlock
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.transformer.transformer_layer import HyperConnectionTransformerLayer
from megatron.core.transformer.wide_residual_config import WideResidualConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.determinism.utils import (
    assert_bit_exact,
    capture_rng_state,
    restore_rng_state,
)
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


def make_stack(
    symbol,
    *,
    stack_spec=hybrid_stack_spec,
    post_process=False,
    hash_moe_layer_threshold=None,
    **config_kwargs,
):
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
            post_process=post_process,
            hash_moe_layer_threshold=hash_moe_layer_threshold,
        )
        .cuda()
        .train()
    )


def make_layer(kind, pattern):
    """Construct production layers and independent explicit-call arguments."""
    groups = ProcessGroupCollection.use_mpu_process_groups()
    spec, kwargs = hybrid_stack_spec, {}
    width = 256
    if kind == "GatedDeltaProductMixer":
        if not (HAVE_EINOPS and HAVE_FLA and HAVE_MAMBA_SSM):
            pytest.skip("Requires GDP dependencies")
        spec = gated_delta_product_stack_spec
        kwargs = dict(mamba_num_heads=4, mamba_head_dim=64, mamba_num_groups=2, mamba_state_dim=128)
    elif kind == "GatedDeltaNet2":
        kwargs["experimental_attention_variant"] = "gdn2"
    elif kind == "FusedMLASelfAttention":
        kwargs["mla_down_proj_fusion"] = True
    elif kind == "DSv4HybridSelfAttention":
        kwargs.update(
            experimental_attention_variant="dsv4_hybrid",
            csa_compress_ratios=[0],
            dsa_kernel_backend="none",
            output_projection_lora_rank=64,
        )
    elif kind == "AbsorbedMLASelfAttention" and hadamard_transform is None:
        pytest.skip("Requires fast_hadamard_transform")
    elif kind.startswith("WideResidual"):
        spec = wide_residual_hybrid_stack_spec
        kwargs["wide_residual"] = WideResidualConfig(num_streams=2)
        width *= 2
    elif kind in ("HyperConnectionHybridLayer", "HyperConnectionTransformerLayer", "HybridStack"):
        kwargs.update(enable_mhc_connections=True, mhc_num_residual_streams=2, use_fused_mhc=False)
        if kind == "HybridStack":
            kwargs["mtp_num_layers"] = 1  # Exercise its named multistream output, too.
        else:
            width *= 2
    elif kind == "ShortcutMoEBlock":
        kwargs.update(moe_shortcut_connection=True, moe_shortcut_post_norm=True)
    elif kind in ("MultiTokenPredictionLayer", "MultiTokenPredictionBlock"):
        kwargs["mtp_num_layers"] = 1
    elif kind == "MoELayer":
        kwargs.update(moe_num_hash_layers=1, hash_moe_vocab_size=64, num_moe_experts=4)
    elif kind in ("MLP", "CrossAttention"):
        kwargs["add_bias_linear"] = True

    if kind == "HyperConnectionTransformerLayer":
        layer_spec = get_gpt_layer_with_transformer_engine_spec()
        layer_spec.module = HyperConnectionTransformerLayer
        layer_spec.submodules.self_attention_hyper_connection = HyperConnectionModule
        layer_spec.submodules.mlp_hyper_connection = HyperConnectionModule
        module = build_module(layer_spec, config=make_config(**kwargs), pg_collection=groups)
    elif kind == "CrossAttention":
        module = CrossAttention(
            make_config(**kwargs),
            CrossAttentionSubmodules(
                linear_q=TEColumnParallelLinear,
                linear_kv=TEColumnParallelLinear,
                core_attention=TEDotProductAttention,
                linear_proj=TERowParallelLinear,
            ),
            layer_number=1,
            attn_mask_type=AttnMaskType.padding,
            pg_collection=groups,
        )
    elif kind in ("TEFusedMLP", "TEFusedMLPWithGroupedLinear"):
        cls = TEFusedMLP if kind == "TEFusedMLP" else TEFusedMLPWithGroupedLinear
        if cls is None:
            pytest.skip("Requires TE fused ops")
        if kind == "TEFusedMLPWithGroupedLinear" and (
            not is_te_min_version("2.14.0") or torch.cuda.get_device_capability()[0] < 10
        ):
            pytest.skip("Requires TE >= 2.14 and Blackwell MXFP8 support")
        module = cls(
            make_config(gated_linear_unit=True),
            MLPSubmodules(
                linear_fc1=TELayerNormColumnParallelLinear, linear_fc2=TERowParallelLinear
            ),
            pg_collection=groups,
        )
    elif kind in ("MultiTokenPredictionLayer", "MultiTokenPredictionBlock"):
        module = MultiTokenPredictionBlock(
            config=make_config(**kwargs),
            spec=hybrid_stack_spec.submodules.mtp_block_spec,
            pg_collection=groups,
            mtp_layer_pattern=pattern,
            mtp_num_depths=1,
            hybrid_submodules=hybrid_stack_spec.submodules,
        )
        if kind == "MultiTokenPredictionLayer":
            module = module.layers[0]
    else:
        stack = make_stack(
            pattern,
            stack_spec=spec,
            post_process=kind == "HybridStack",
            hash_moe_layer_threshold=1 if kind == "MoELayer" else None,
            **kwargs,
        )
        module = stack if kind == "HybridStack" else stack.layers[0]
        if kind in ("MLP", "MoELayer"):
            module = module.mlp
        elif kind in ("MambaMixer", "GatedDeltaProductMixer"):
            module = module.mixer
        elif kind in (
            "SelfAttention",
            "MLASelfAttention",
            "FusedMLASelfAttention",
            "AbsorbedMLASelfAttention",
            "DSv4HybridSelfAttention",
            "GatedDeltaNet",
            "GatedDeltaNet2",
        ):
            module = module.self_attention
    module = module.cuda().train()
    assert type(module).__name__ == kind
    assert isinstance(module, LayerInterface)
    if kind == "MoELayer":
        assert module.router.is_hash_layer
    x = torch.randn(64, 2, width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    mask = torch.ones(2, 1, 64, 64, device="cuda", dtype=torch.bool).triu(1)
    inputs = LayerAuxInputs(attention_mask=mask)

    if kind == "CrossAttention":
        context_mask = torch.zeros_like(mask)
        context_mask[..., :16] = True
        inputs = inputs.update(
            context=torch.randn_like(x, requires_grad=True), context_mask=context_mask
        )
        explicit = dict(attention_mask=inputs.context_mask, key_value_states=inputs.context)
    elif kind == "SelfAttention":
        rope = RotaryEmbedding(module.config.kv_channels, rotary_percent=1.0, cp_group=groups.cp)(
            64
        )
        inputs = inputs.update(rotary_pos_emb=rope)
        explicit = dict(attention_mask=mask, rotary_pos_emb=rope)
    elif kind in (
        "MLASelfAttention",
        "FusedMLASelfAttention",
        "AbsorbedMLASelfAttention",
        "DSv4HybridSelfAttention",
    ):
        inputs = inputs.update(position_ids=torch.arange(64, device="cuda").expand(2, -1))
        explicit = dict(attention_mask=mask, position_ids=inputs.position_ids)
    elif kind in ("GatedDeltaNet", "GatedDeltaNet2"):
        explicit = dict(attention_mask=mask)
    elif kind == "MoELayer":
        padding = torch.zeros(2, 64, dtype=torch.bool, device="cuda")
        padding[:, -1] = True
        inputs = inputs.update(
            padding_mask=padding, input_ids=torch.arange(64, device="cuda").expand(2, -1)
        )
        explicit = dict(padding_mask=padding, input_ids=inputs.input_ids)
    elif kind in (
        "MLP",
        "TEFusedMLP",
        "TEFusedMLPWithGroupedLinear",
        "MambaMixer",
        "GatedDeltaProductMixer",
    ):
        explicit = {}
    else:
        if kind == "ShortcutMoEBlock":
            inputs = inputs.update(
                execution_context=LayerExecutionContext(lambda *_: nullcontext())
            )
        if kind in ("MultiTokenPredictionLayer", "MultiTokenPredictionBlock"):
            inputs = inputs.update(decoder_input=torch.randn_like(x, requires_grad=True))
            if kind == "MultiTokenPredictionLayer":
                inputs = inputs.update(cp_layout_plan=object())
        explicit = dict(aux_inputs=inputs)
        if kind == "HyperConnectionHybridLayer":
            explicit["inner_inputs"] = inputs
        if kind in (
            "TransformerLayer",
            "HyperConnectionTransformerLayer",
            "WideResidualTransformerLayer",
            "MLPLayer",
            "MoETransformerLayer",
        ):
            # Keep the independent reference on TransformerLayer's legacy core-call path.
            explicit = dict(attention_mask=mask)
    return module, x, inputs, explicit


def assert_forward_backward(module, x, inputs, explicit):
    """Compare real computations on the same weights and restored RNG state."""
    rng_state = copy.deepcopy(capture_rng_state())
    input_tensors = {
        name: value
        for name, value in vars(inputs).items()
        if isinstance(value, torch.Tensor) and value.requires_grad
    }
    expected_input_grad = {
        "CrossAttention": "context",
        "MultiTokenPredictionLayer": "decoder_input",
        "MultiTokenPredictionBlock": "decoder_input",
    }.get(type(module).__name__)

    kind = type(module).__name__
    core = {
        "TransformerLayer": lambda: module.self_attention,
        "MLPLayer": lambda: module.mlp,
        "MultiTokenPredictionLayer": lambda: module.mtp_model_layer,
    }.get(kind, lambda: None)()
    binder_context = (
        patch.object(core, "bind_layer_inputs", wraps=core.bind_layer_inputs)
        if core is not None
        else nullcontext(None)
    )
    with binder_context as binder:

        def run(packed=False, *, aux_keyword=False):
            module.zero_grad(set_to_none=True)
            for value in (x, *input_tensors.values()):
                value.grad = None
            if aux_keyword:
                result = module(hidden_states=x, aux_inputs=inputs)
            elif not packed and "aux_inputs" in explicit:
                result = module.forward(hidden_states=x, **explicit)
            else:
                result = module(x, inputs) if packed else module(hidden_states=x, **explicit)
            if packed or aux_keyword:
                assert isinstance(result, tuple) and len(result) == 2
                hidden, aux = result
                assert isinstance(aux, LayerAuxOutputs)
            elif isinstance(result, torch.Tensor):
                hidden, aux = result, LayerAuxOutputs()
            else:
                hidden, aux = result
                if not isinstance(aux, LayerAuxOutputs):
                    aux = (
                        LayerAuxOutputs(mhc_multistream=aux)
                        if isinstance(module, HybridStack)
                        else LayerAuxOutputs(bias=aux)
                    )
            outputs = dict(hidden_states=hidden, bias=aux.bias, mhc_multistream=aux.mhc_multistream)
            saved = {
                name: value.detach().clone() for name, value in outputs.items() if value is not None
            }
            loss = (hidden if aux.bias is None else hidden + aux.bias).float().square().sum()
            if aux.mhc_multistream is not None:
                loss = loss + aux.mhc_multistream.float().square().sum()
            loss.backward()
            if expected_input_grad is not None:
                assert input_tensors[expected_input_grad].grad is not None
            grads = {
                name: p.grad.detach().clone()
                for name, p in module.named_parameters()
                if p.grad is not None
            }
            grads["input.hidden_states"] = x.grad.detach().clone()
            grads.update(
                {
                    "input." + name: value.grad.detach().clone()
                    for name, value in input_tensors.items()
                    if value.grad is not None
                }
            )
            return saved, grads

        # Fused TE wrappers construct execution views lazily; initialize them before saving RNG.
        if kind.startswith("TEFusedMLP"):
            module(hidden_states=x, **explicit)
            rng_state = copy.deepcopy(capture_rng_state())
        expected, expected_grads = run()
        if binder is not None:
            assert binder.call_count == (1 if kind == "MultiTokenPredictionLayer" else 0)
            binder.reset_mock()
        restore_rng_state(rng_state)
        actual, actual_grads = run(packed=True)
        if binder is not None:
            assert binder.call_count == 1
            assert binder.call_args.args[1] is inputs
            if kind == "MultiTokenPredictionLayer":
                assert binder.call_args.args[1].cp_layout_plan is inputs.cp_layout_plan
        assert expected.keys() == actual.keys()
        for name in expected:
            assert_bit_exact(expected[name], expected_grads, actual[name], actual_grads)
        if kind == "TransformerLayer":
            binder.reset_mock()
            restore_rng_state(rng_state)
            keyword_actual, keyword_grads = run(aux_keyword=True)
            assert binder.call_count == 1
            assert binder.call_args.args[1] is inputs
            assert expected.keys() == keyword_actual.keys()
            for name in expected:
                assert_bit_exact(
                    expected[name], expected_grads, keyword_actual[name], keyword_grads
                )


class TestLayerAuxInputsEquivalence:
    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "kind,pattern",
        [
            ("SelfAttention", "*"),
            ("CrossAttention", None),
            ("MLASelfAttention", "+"),
            ("FusedMLASelfAttention", "+"),
            ("AbsorbedMLASelfAttention", "D"),
            ("DSv4HybridSelfAttention", "W"),
            ("MLP", "-"),
            ("TEFusedMLP", None),
            ("TEFusedMLPWithGroupedLinear", None),
            ("MoELayer", "E"),
            ("MambaMixer", "M"),
            ("GatedDeltaNet", "G"),
            ("GatedDeltaNet2", "G"),
            ("GatedDeltaProductMixer", "M"),
            ("TransformerLayer", "*"),
            ("MLPLayer", "-"),
            ("MoETransformerLayer", "E"),
            ("HyperConnectionTransformerLayer", None),
            ("WideResidualTransformerLayer", "*"),
            ("MambaLayer", "M"),
            ("WideResidualMambaLayer", "M"),
            ("HybridStack", "M*-"),
            ("HyperConnectionHybridLayer", "*"),
            ("ShortcutMoEBlock", "*E"),
            ("MultiTokenPredictionLayer", "-"),
            ("MultiTokenPredictionBlock", "-"),
        ],
        ids=lambda value: value,
    )
    def test_real_layer_forward_backward_matches_explicit_call(self, kind, pattern):
        module, x, inputs, explicit = make_layer(kind, pattern)
        assert_forward_backward(module, x, inputs, explicit)
