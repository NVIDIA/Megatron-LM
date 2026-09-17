# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Hybrid integration regressions independent of the MoE routing algorithm."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols as LayerSymbols
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.hybrid.layers.hybrid_hyper_connection import HyperConnectionHybridLayer
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.moe import router as router_module
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionLayer,
    MultiTokenPredictionLayerSubmodules,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer


class SizeOneGroup:
    @staticmethod
    def size():
        return 1

    @staticmethod
    def rank():
        return 0


def test_hybrid_stack_marks_mtp_moe_and_propagates_mtp_depth(monkeypatch):
    import megatron.core.models.hybrid.hybrid_block as hybrid_block_module

    captured_build_kwargs = {}

    class _MtpMoEStub(torch.nn.Module):
        def __init__(self, layer_number, is_mtp_layer):
            super().__init__()
            self.layer_number = layer_number
            self.router = SimpleNamespace(is_mtp_layer=is_mtp_layer)

    def fake_build_module(_spec, **kwargs):
        captured_build_kwargs.update(kwargs)
        return _MtpMoEStub(layer_number=kwargs["layer_number"], is_mtp_layer=kwargs["is_mtp_layer"])

    monkeypatch.setattr(hybrid_block_module, "build_module", fake_build_module)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        num_moe_experts=2,
        use_cpu_initialization=True,
    )

    stack = HybridStack(
        config=config,
        submodules=HybridStackSubmodules(moe_layer=object()),
        layer_type_list=[LayerSymbols.MOE],
        post_process=False,
        pg_collection=SimpleNamespace(pp=object(), tp=object(), cp=SizeOneGroup(), tp_cp=object()),
        is_mtp_layer=True,
        mtp_layer_number=2,
    )

    assert captured_build_kwargs["is_mtp_layer"] is True
    assert stack.is_mtp_layer is True
    assert stack.mtp_layer_number == 2
    assert stack.layers[0].router.is_mtp_layer is True
    assert stack.layers[0].router.mtp_layer_number == 2


def test_mtp_layer_passes_its_depth_to_nested_hybrid_stack(monkeypatch):
    import megatron.core.models.hybrid.hybrid_block as hybrid_block_module
    import megatron.core.models.hybrid.hybrid_layer_allocation as allocation_module
    import megatron.core.transformer.multi_token_prediction as mtp_module

    captured_stack_kwargs = {}

    class _IdentityNorm(torch.nn.Module):
        def __init__(self, **_kwargs):
            super().__init__()

        def forward(self, hidden_states):
            return hidden_states

    class _RecordingHybridStack(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            captured_stack_kwargs.update(kwargs)
            self.layers = torch.nn.ModuleList([torch.nn.Identity()])

    monkeypatch.setattr(hybrid_block_module, "HybridStack", _RecordingHybridStack)
    monkeypatch.setattr(
        allocation_module, "validate_segment_layers", lambda _pattern, _config: [LayerSymbols.MOE]
    )
    monkeypatch.setattr(mtp_module, "build_module", lambda *_args, **_kwargs: torch.nn.Identity())

    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        num_moe_experts=2,
        use_cpu_initialization=True,
        mtp_num_layers=2,
    )
    submodules = MultiTokenPredictionLayerSubmodules(
        enorm=_IdentityNorm,
        hnorm=_IdentityNorm,
        layer_norm=_IdentityNorm,
        eh_proj=object(),
        mtp_model_layer=None,
    )

    layer = MultiTokenPredictionLayer(
        config=config,
        submodules=submodules,
        layer_number=2,
        pg_collection=SimpleNamespace(cp=None, tp=None, pp=SizeOneGroup()),
        mtp_layer_pattern="E",
        hybrid_submodules=HybridStackSubmodules(),
    )

    assert layer.layer_number == 2
    assert captured_stack_kwargs["is_mtp_layer"] is True
    assert captured_stack_kwargs["mtp_layer_number"] == 2


def test_hybrid_mtp_aux_metric_uses_enclosing_depth_slot():
    """An internal `/WE` MoE logs to its MTP depth, not its Hybrid sublayer number."""
    router = TopKRouter.__new__(TopKRouter)
    torch.nn.Module.__init__(router)
    router.config = SimpleNamespace(mtp_num_layers=1, mtp_use_repeated_layer=False, num_layers=86)
    router.is_mtp_layer = True
    router.layer_number = 2
    router.mtp_layer_number = 1
    router.calculate_per_token_loss = False

    activation = torch.ones(2)
    tracker = mock.MagicMock()
    with mock.patch.object(router_module, "get_moe_metrics_tracker", return_value=tracker):
        router.attach_and_log_load_balancing_loss(
            activation,
            aux_loss_coeff=0.1,
            aux_loss=torch.tensor(0.5),
            aux_loss_name="seq_load_balancing_loss",
            reduce_group=mock.sentinel.reduce_group,
        )

    record_args = tracker.record.call_args.args
    assert record_args[2] == 87
    assert record_args[3] == 87


def test_hybrid_mtp_z_loss_metric_uses_enclosing_depth_slot():
    """Z-loss uses the MTP depth instead of an internal `/WE` sublayer number."""
    router = TopKRouter.__new__(TopKRouter)
    torch.nn.Module.__init__(router)
    router.config = SimpleNamespace(
        moe_z_loss_coeff=0.1, mtp_num_layers=1, mtp_use_repeated_layer=False, num_layers=86
    )
    router.is_mtp_layer = True
    router.layer_number = 2
    router.mtp_layer_number = 1
    router.calculate_per_token_loss = False
    router.tp_cp_group = SizeOneGroup()
    router.tp_dp_cp_group = mock.sentinel.tp_dp_cp_group

    tracker = mock.MagicMock()
    with mock.patch.object(router_module, "get_moe_metrics_tracker", return_value=tracker):
        router.apply_z_loss(torch.zeros(2, 2, requires_grad=True))

    record_args = tracker.record.call_args.args
    assert record_args[2] == 87
    assert record_args[3] == 87


@pytest.mark.parametrize("pre_process", [False, True])
def test_hybrid_sp_padding_uses_local_pipeline_activations(monkeypatch, pre_process):
    hidden = torch.randn(2, 2, 8)
    decoder = mock.Mock(input_tensor=hidden)
    model = SimpleNamespace(
        config=TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=2,
            tensor_model_parallel_size=2,
            sequence_parallel=True,
        ),
        decoder=decoder,
        position_embedding_type="none",
        pre_process=pre_process,
        post_process=False,
        share_embeddings_and_output_weights=False,
        mtp_process=False,
        pg_collection=SimpleNamespace(tp=object()),
    )
    mask = torch.tensor([[False, True, False, True], [True, False, True, False]])
    scatter = mock.Mock(side_effect=lambda tensor, group: tensor[:2].contiguous())
    monkeypatch.setattr(
        "megatron.core.models.hybrid.hybrid_model.tensor_parallel.scatter_to_sequence_parallel_region",
        scatter,
    )
    HybridModel.forward(
        model,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        decoder_input=hidden if pre_process else None,
        padding_mask=mask,
    )
    scatter.assert_called_once()
    torch.testing.assert_close(decoder.call_args.kwargs["padding_mask"], mask[:, :2])


@pytest.mark.parametrize("branch", ["attention", "mlp"])
def test_hybrid_mhc_norm_uses_wrapper_checkpoint_manager(monkeypatch, branch):
    # Exercise the actual raw-branch and norm helpers, isolating only the checkpoint engine.
    layer = TransformerLayer.__new__(TransformerLayer)
    torch.nn.Module.__init__(layer)
    layer.config = SimpleNamespace(
        fp32_residual_connection=False,
        inference_fuse_tp_communication=False,
        bias_dropout_fusion=False,
    )
    layer.self_attention = IdentityOp()
    layer.cross_attention = IdentityOp()
    layer.mlp = IdentityOp()
    layer.hidden_dropout = 0.0
    layer.is_moe_layer = False
    layer.recompute_input_layernorm = layer.recompute_pre_mlp_layernorm = False
    layer.mhc_checkpoint_input_layernorm = layer.mhc_checkpoint_pre_mlp_layernorm = True
    layer._input_layernorm_returns_residual = layer._pre_mlp_layernorm_returns_residual = False
    layer.offload_attn_norm = layer.offload_mlp_norm = False
    layer.off_interface = lambda _enabled, hidden, _name: nullcontext(hidden)
    layer.input_layernorm = layer.pre_mlp_layernorm = torch.nn.LayerNorm(8)
    layer._group_offload_output_with_bias = lambda output, *_args, **_kwargs: output
    layer._run_mlp = lambda hidden, *_args, **_kwargs: (hidden * 2, None)

    class Attention(torch.nn.Module):
        def forward(self, hidden, **kwargs):
            return hidden * 2, None

    if branch == "attention":
        layer.self_attention = Attention()
    else:
        layer.mlp = torch.nn.Linear(8, 8)
    checkpoint = mock.Mock()
    checkpoint.checkpoint.side_effect = lambda norm, hidden: norm(hidden)
    factory = mock.Mock(return_value=checkpoint)
    monkeypatch.setattr(
        "megatron.core.transformer.transformer_layer.tensor_parallel.CheckpointWithoutOutput",
        factory,
    )
    manager = object()
    hidden = torch.randn(4, 2, 8, requires_grad=True)
    output, _, _, _ = HyperConnectionHybridLayer._call_inner_transformer_layer_without_local_bda(
        SimpleNamespace(inner_layer=layer),
        hidden,
        None,
        None,
        None,
        None,
        None,
        None,
        mhc_recompute_manager=manager,
    )
    factory.assert_called_once_with(ckpt_manager=manager, retain_input_tensors=False)
    checkpoint.discard_output_and_register_recompute.assert_called_once_with(output[0])
    torch.testing.assert_close(output[0], layer.input_layernorm(hidden) * 2)
    output[0].square().sum().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
