# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_logging import (
    destroy_moe_metrics_tracker,
    get_moe_metrics_tracker,
)
from megatron.core.transformer.moe.router import Router
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize("mtp_num_layers", [1, 2])
@pytest.mark.parametrize("mtp_pattern", ["*E", "*EE"])
def test_hybrid_mtp_moe_losses_use_prediction_depth(
    mtp_num_layers: int, mtp_pattern: str, monkeypatch
) -> None:
    """Real Hybrid forward/backward logs every MTP router in its prediction-depth slot."""
    Utils.initialize_model_parallel(1, 1)
    destroy_moe_metrics_tracker()
    monkeypatch.setattr(MTPLossLoggingHelper, "tracker", {})
    try:
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=4,
            hidden_size=32,
            num_attention_heads=4,
            ffn_hidden_size=64,
            moe_ffn_hidden_size=32,
            num_moe_experts=4,
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            moe_z_loss_coeff=0.001,
            mtp_num_layers=mtp_num_layers,
            mtp_loss_scaling_factor=0.3,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            use_cpu_initialization=True,
        )
        model = HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            hybrid_layer_pattern="*-*E" + f"/{mtp_pattern}" * mtp_num_layers,
            vocab_size=64,
            max_sequence_length=8,
            position_embedding_type="none",
        ).cuda()
        expected_inner_numbers = [i + 1 for i, symbol in enumerate(mtp_pattern) if symbol == "E"]
        for depth, layer in enumerate(model.mtp.layers, 1):
            routers = [
                module for module in layer.mtp_model_layer.modules() if isinstance(module, Router)
            ]
            assert [router.layer_number for router in routers] == expected_inner_numbers
            assert all(router.mtp_layer_number == depth for router in routers)
            assert all(
                router.get_loss_logging_layer_number() == config.num_layers + depth
                for router in routers
            )
        main_routers = [module for module in model.decoder.modules() if isinstance(module, Router)]
        assert [router.layer_number for router in main_routers] == [4]
        assert main_routers[0].mtp_layer_number is None

        tracker = get_moe_metrics_tracker()
        original_record = tracker.record
        records = []

        def record(name, value, layer_number, num_layers, **kwargs):
            records.append((name, layer_number, value.detach().clone()))
            original_record(name, value, layer_number, num_layers, **kwargs)

        monkeypatch.setattr(tracker, "record", record)
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device="cuda")
        positions = torch.arange(8, device="cuda").unsqueeze(0)
        attention_mask = torch.triu(torch.ones(8, 8, device="cuda", dtype=torch.bool), 1)[
            None, None
        ]
        loss = model(
            tokens,
            positions,
            attention_mask,
            labels=tokens.roll(-1, dims=1),
            loss_mask=torch.ones_like(tokens, dtype=torch.float32),
        )
        loss.mean().backward()
        assert torch.isfinite(loss).all()
        for metric in ("load_balancing_loss", "z_loss"):
            expected_numbers = [4] + [
                config.num_layers + depth
                for depth in range(1, mtp_num_layers + 1)
                for _ in expected_inner_numbers
            ]
            actual = [(index, value) for name, index, value in records if name == metric]
            assert [index for index, _ in actual] == expected_numbers
            expected = torch.zeros(config.num_layers + mtp_num_layers, device="cuda")
            for index, value in actual:
                expected[index - 1] += value
            torch.testing.assert_close(tracker.metrics[metric].values, expected)
        for module in model.modules():
            if isinstance(module, Router):
                assert module.weight.grad is not None
                assert torch.isfinite(module.weight.grad).all()
    finally:
        destroy_moe_metrics_tracker()
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("model_kind", ["gpt", "hybrid"])
@pytest.mark.parametrize("enable_router_losses", [False, True])
def test_mtp1_moe_forward_backward_with_router_loss_controls(
    model_kind: str, enable_router_losses: bool, monkeypatch
) -> None:
    """Compare two attention/MoE blocks plus MTP1 across both native model paths."""
    Utils.initialize_model_parallel(1, 1)
    destroy_moe_metrics_tracker()
    monkeypatch.setattr(MTPLossLoggingHelper, "tracker", {})
    try:
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=2 if model_kind == "gpt" else 4,
            hidden_size=32,
            num_attention_heads=4,
            kv_channels=8,
            ffn_hidden_size=64,
            moe_ffn_hidden_size=32,
            num_moe_experts=4,
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01 if enable_router_losses else 0.0,
            moe_z_loss_coeff=0.001 if enable_router_losses else None,
            mtp_num_layers=1,
            mtp_loss_scaling_factor=0.3,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            use_cpu_initialization=True,
        )
        common = dict(
            config=config, vocab_size=64, max_sequence_length=8, position_embedding_type="none"
        )
        if model_kind == "hybrid":
            model = HybridModel(
                **common, hybrid_stack_spec=hybrid_stack_spec, hybrid_layer_pattern="*E*E/*E"
            ).cuda()
        else:
            layer_spec = get_gpt_layer_with_transformer_engine_spec(num_experts=4)
            model = GPTModel(
                **common,
                transformer_layer_spec=layer_spec,
                mtp_block_spec=get_gpt_mtp_block_spec(config, layer_spec, True),
            ).cuda()
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device="cuda")
        positions = torch.arange(8, device="cuda").unsqueeze(0)
        attention_mask = torch.triu(torch.ones(8, 8, device="cuda", dtype=torch.bool), 1)[
            None, None
        ]
        # Run the unmodified model forward/backward before inspecting any metric
        # metadata so the upstream failure is attributable to the real router path.
        loss = model(
            tokens,
            positions,
            attention_mask,
            labels=tokens.roll(-1, dims=1),
            loss_mask=torch.ones_like(tokens, dtype=torch.float32),
        )
        loss.mean().backward()
        assert loss.shape == tokens.shape
        assert torch.isfinite(loss).all()
        routers = [module for module in model.modules() if isinstance(module, Router)]
        assert len(routers) == 3
        for router in routers:
            assert router.weight.grad is not None
            assert torch.isfinite(router.weight.grad).all()
        metrics = get_moe_metrics_tracker().metrics
        if enable_router_losses:
            expected_positions = [0, 1, 2] if model_kind == "gpt" else [1, 3, 4]
            for name in ("load_balancing_loss", "z_loss"):
                values = metrics[name].values
                assert values.shape == (config.num_layers + 1,)
                assert values.nonzero().flatten().tolist() == expected_positions
        else:
            assert "load_balancing_loss" not in metrics
            assert "z_loss" not in metrics
    finally:
        destroy_moe_metrics_tracker()
        Utils.destroy_model_parallel()
