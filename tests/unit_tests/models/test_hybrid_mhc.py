# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.models.hybrid.hybrid_block import (
    HybridStack,
    HybridStackSubmodules,
    HyperConnectionHybridLayer,
)
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer
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


class _StubTransformerLayer(TransformerLayer):
    """Minimal TransformerLayer that exercises only the mHC wrapper guard."""

    def __init__(self, config: TransformerConfig):
        torch.nn.Module.__init__(self)
        self.config = config
        self.layer_number = 1

    def _forward_attention(self, *args, **kwargs):
        hidden_states = kwargs.get("hidden_states", args[0] if args else None)
        return hidden_states, None

    def _forward_mlp(self, hidden_states, *_args, **_kwargs):
        return hidden_states


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


def _get_dummy_stack_spec() -> ModuleSpec:
    return ModuleSpec(
        module=HybridStack, params={"post_layer_norm": False}, submodules=_get_dummy_submodules()
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


def test_mhc_mtp_requires_hybrid_contract():
    config = _get_config(num_layers=1, mtp_num_layers=1)

    with pytest.raises(ValueError, match="requires the HybridModel MTP contract"):
        MultiTokenPredictionLayer(
            config=config,
            submodules=object(),
            layer_number=1,
            pg_collection=None,
        )


@pytest.mark.internal
class TestHybridStackMHC:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor_and_sharded_state(self):
        config = _get_config(num_layers=3)
        stack = _get_stack(config, num_local_layers=3)

        assert all(isinstance(layer, HyperConnectionHybridLayer) for layer in stack.layers)
        assert stack.hc_head_fn.shape == (
            config.num_residual_streams,
            config.hidden_size * config.num_residual_streams,
        )
        state = stack.sharded_state_dict(prefix="decoder.", metadata={})
        for name in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
            assert f"decoder.{name}" in state

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
        ids=["none", "selective_mhc", "full_uniform"],
    )
    def test_forward_backward(self, recompute_kwargs):
        config = _get_config(num_layers=3, **recompute_kwargs)
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
            assert getattr(stack, name).grad is not None

    def test_full_recompute_forwards_input_ids(self, monkeypatch):
        config = _get_config(
            num_layers=2,
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
        )
        stack = _get_stack(config, num_local_layers=2).cuda()
        hidden_states = torch.randn(8, 2, config.hidden_size, device="cuda", requires_grad=True)
        input_ids = torch.arange(8, device="cuda").repeat(2, 1)
        seen_input_ids = []
        original = HyperConnectionHybridLayer._call_inner_transformer_layer_without_local_bda

        def record_input_ids(
            layer,
            hidden_states,
            attention_mask,
            inference_context,
            rotary_pos_emb,
            sequence_len_offset,
            packed_seq_params,
            padding_mask,
            input_ids=None,
        ):
            seen_input_ids.append(input_ids)
            return original(
                layer,
                hidden_states,
                attention_mask,
                inference_context,
                rotary_pos_emb,
                sequence_len_offset,
                packed_seq_params,
                padding_mask,
                input_ids,
            )

        monkeypatch.setattr(
            HyperConnectionHybridLayer,
            "_call_inner_transformer_layer_without_local_bda",
            record_input_ids,
        )

        output = stack(hidden_states, attention_mask=None, input_ids=input_ids)
        output.float().sum().backward()

        assert len(seen_input_ids) == 4
        assert all(
            captured is not None and torch.equal(captured, input_ids) for captured in seen_input_ids
        )

    def test_fused_bf16_forward_backward(self):
        config = _get_config(
            num_layers=2, bf16=True, params_dtype=torch.bfloat16, use_fused_mhc=True
        )
        stack = _get_stack(config, num_local_layers=2).cuda().bfloat16()
        hidden_states = torch.randn(
            8, 2, config.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )

        output = stack(hidden_states, attention_mask=None)

        assert output.shape == hidden_states.shape
        assert output.dtype == torch.bfloat16
        assert torch.isfinite(output).all()
        output.float().sum().backward()
        assert hidden_states.grad is not None
        assert all(
            layer.hyper_connection.mapping_proj.weight.grad is not None for layer in stack.layers
        )

    def test_pipeline_boundary_shapes(self):
        config = _get_config(num_layers=2)
        first_stage = _get_stack(
            config, num_local_layers=1, pre_process=True, post_process=False
        ).cuda()
        last_stage = _get_stack(
            config, num_local_layers=1, pre_process=False, post_process=True, pp_layer_offset=1
        ).cuda()
        hidden_states = torch.randn(8, 2, config.hidden_size, device="cuda")

        pipeline_hidden = first_stage(hidden_states, attention_mask=None)
        assert pipeline_hidden.shape == (8, 2, config.hidden_size * config.num_residual_streams)

        last_stage.set_input_tensor(pipeline_hidden.detach())
        output = last_stage(hidden_states, attention_mask=None)
        assert output.shape == hidden_states.shape

    def test_real_attention_mlp_forward_backward(self):
        config = _get_config(num_layers=2)
        stack = HybridStack(
            config=config,
            submodules=hybrid_stack_spec.submodules,
            post_layer_norm=False,
            layer_type_list=[Symbols.ATTENTION, Symbols.MLP],
            pp_layer_offset=0,
            pg_collection=_get_pg_collection(),
        ).cuda()
        hidden_states = torch.randn(8, 2, config.hidden_size, device="cuda", requires_grad=True)

        output = stack(hidden_states, attention_mask=None)

        assert output.shape == hidden_states.shape
        assert all(isinstance(layer, HyperConnectionHybridLayer) for layer in stack.layers)
        assert all(isinstance(layer.inner_layer, TransformerLayer) for layer in stack.layers)
        output.float().sum().backward()
        assert hidden_states.grad is not None
        assert all(
            layer.hyper_connection.mapping_proj.weight.grad is not None for layer in stack.layers
        )

    def test_real_moe_raw_branch_forward_backward(self):
        config = _get_config(
            num_layers=1,
            num_moe_experts=2,
            moe_ffn_hidden_size=64,
            moe_grouped_gemm=True,
            add_bias_linear=False,
        )
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=['tp', 'pp', 'cp', 'ep', 'expt_tp', 'tp_ep', 'expt_dp']
        )
        stack = HybridStack(
            config=config,
            submodules=hybrid_stack_spec.submodules,
            post_layer_norm=False,
            layer_type_list=[Symbols.MOE],
            pp_layer_offset=0,
            pg_collection=pg_collection,
        ).cuda()
        hidden_states = torch.randn(8, 2, config.hidden_size, device="cuda", requires_grad=True)

        output = stack(hidden_states, attention_mask=None)

        wrapped_layer = stack.layers[0]
        assert isinstance(wrapped_layer, HyperConnectionHybridLayer)
        assert isinstance(wrapped_layer.inner_layer.mlp, MoELayer)
        assert output.shape == hidden_states.shape
        output.float().sum().backward()
        assert hidden_states.grad is not None
        assert wrapped_layer.hyper_connection.mapping_proj.weight.grad is not None
        assert any(param.grad is not None for param in wrapped_layer.inner_layer.mlp.parameters())

    def test_hybrid_model_forward_backward(self):
        config = _get_config(num_layers=3)
        model = HybridModel(
            config=config,
            hybrid_stack_spec=_get_dummy_stack_spec(),
            vocab_size=64,
            max_sequence_length=8,
            hybrid_layer_pattern="M*-",
            parallel_output=False,
        ).cuda()
        input_ids = torch.arange(8, dtype=torch.int64, device="cuda").repeat((2, 1))
        position_ids = torch.arange(8, dtype=torch.int64, device="cuda").repeat((2, 1))

        logits = model(input_ids=input_ids, position_ids=position_ids, attention_mask=None)

        assert logits.shape == (2, 8, model.vocab_size)
        assert torch.isfinite(logits).all()
        logits.float().mean().backward()
        assert all(layer.inner_layer.proj.weight.grad is not None for layer in model.decoder.layers)
        assert all(
            layer.hyper_connection.mapping_proj.weight.grad is not None
            for layer in model.decoder.layers
        )

    def test_hybrid_model_mtp_forward_backward(self):
        config = _get_config(num_layers=1, mtp_num_layers=1, mtp_loss_scaling_factor=0.1)
        model = HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=64,
            max_sequence_length=8,
            hybrid_layer_pattern="-/-",
            parallel_output=True,
        ).cuda()
        input_ids = torch.arange(8, dtype=torch.int64, device="cuda").repeat((2, 1))
        position_ids = torch.arange(8, dtype=torch.int64, device="cuda").repeat((2, 1))

        logits = model(input_ids=input_ids, position_ids=position_ids, attention_mask=None)

        assert logits.shape == (2, 8, model.vocab_size)
        assert torch.isfinite(logits).all()
        assert not any("mtp_model_layer.hc_head_" in name for name, _ in model.named_parameters())
        logits.float().mean().backward()
        mtp_params = [param for name, param in model.named_parameters() if name.startswith("mtp.")]
        assert mtp_params
        assert all(param.grad is not None for param in mtp_params)

    def test_recompute_plan(self):
        config = _get_config(
            num_layers=3,
            recompute_granularity="selective",
            recompute_modules=["core_attn", "mhc"],
            mhc_recompute_layer_num=2,
        )
        stack = _get_stack(config, num_local_layers=3)

        managers, block_ends = stack._build_mhc_recompute_layer_plan(True)

        assert block_ends == [False, True, True]
        assert managers[0] is managers[1]
        assert managers[1] is not managers[2]

    def test_boundary_bda_skips_recompute_manager(self, monkeypatch):
        config = _get_config(num_layers=1)
        layer = HyperConnectionHybridLayer(
            config=config, layer=_DummyHybridLayer(config, layer_number=1)
        )
        hidden_states = torch.randn(
            4, 2, config.hidden_size * config.num_residual_streams, requires_grad=True
        )
        manager = type("_FakeManager", (), {})()
        manager.is_last_layer_in_recompute_block = True
        seen_managers = []

        def fake_hyper_connection_forward(
            hidden_states, mhc_recompute_manager=None, return_residual=False
        ):
            assert mhc_recompute_manager is manager
            assert return_residual
            sequence_length, batch_size, _ = hidden_states.shape
            n = config.num_residual_streams
            hidden_size = config.hidden_size
            aggregated = hidden_states.view(sequence_length, batch_size, n, hidden_size).mean(dim=2)
            h_res = torch.empty(sequence_length, batch_size, n, n)
            h_post = torch.empty(sequence_length, batch_size, n)
            return aggregated, h_res, h_post, hidden_states

        def fake_bda(
            h_res, residual, h_post, output_with_bias, dropout_prob, training, fused, manager=None
        ):
            seen_managers.append(manager)
            return residual

        monkeypatch.setattr(layer.hyper_connection, "forward", fake_hyper_connection_forward)
        monkeypatch.setattr(layer.hyper_connection, "fused_h_res_h_post_bda", fake_bda)

        output, _ = layer(hidden_states, attention_mask=None, mhc_recompute_manager=manager)
        assert output is hidden_states
        assert seen_managers == [None]

        manager.is_last_layer_in_recompute_block = False
        layer(hidden_states, attention_mask=None, mhc_recompute_manager=manager)
        assert seen_managers[-1] is manager

    def test_transformer_layer_wrapper_escape_hatch(self):
        config = _get_config(num_layers=1)
        layer = _StubTransformerLayer(config)
        hidden_states = torch.randn(4, 2, config.hidden_size)

        with pytest.raises(RuntimeError, match="must not be called directly"):
            layer.forward(hidden_states=hidden_states, attention_mask=None)

        output, context = layer.forward(
            hidden_states=hidden_states, attention_mask=None, _called_from_hybrid_mhc_wrapper=True
        )
        assert output is hidden_states
        assert context is None
