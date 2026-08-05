# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Focused tests for direct hybrid MTP dispatch."""

from contextlib import nullcontext
from inspect import signature
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import torch
from torch import nn

from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionBlock,
    MultiTokenPredictionLayer,
)


def test_direct_mtp_specs_use_the_existing_hybrid_stack_forward_path():
    """Direct MTP descriptors must not fall through to the GPT tuple-return path."""

    layer = MultiTokenPredictionLayer.__new__(MultiTokenPredictionLayer)
    torch.nn.Module.__init__(layer)
    layer.config = SimpleNamespace(sequence_parallel=False, fp8=None, fp4=None)
    layer.mtp_layer_pattern = None
    layer.mtp_layer_specs = (object(), object())
    layer._concat_embeddings = Mock(side_effect=lambda hidden_states, _: hidden_states)
    layer._postprocess = Mock(side_effect=lambda hidden_states: hidden_states)
    layer.mtp_model_layer = Mock(
        side_effect=lambda **kwargs: kwargs["hidden_states"]
        + torch.ones_like(kwargs["hidden_states"])
    )

    hidden_states = torch.zeros(2, 1, 4)
    rotary_pos_emb = {4: torch.ones(2, 1, 1, 4)}
    output = layer._proj_and_transformer_layer(
        hidden_states=hidden_states,
        decoder_input=torch.zeros_like(hidden_states),
        attention_mask=None,
        rotary_pos_emb=rotary_pos_emb,
    )

    assert torch.equal(output, torch.ones_like(hidden_states))
    assert layer.mtp_model_layer.call_args.kwargs["rotary_pos_emb"] is rotary_pos_emb
    assert "context" not in layer.mtp_model_layer.call_args.kwargs


def test_direct_hybrid_mtp_keeps_native_checkpoint_prefix():
    layer = MultiTokenPredictionLayer.__new__(MultiTokenPredictionLayer)
    torch.nn.Module.__init__(layer)
    layer.mtp_layer_pattern = None
    layer.mtp_layer_specs = (object(),)
    sharded_weight = ShardedTensor.from_rank_offsets("mtp.mtp_model_layer.weight", torch.ones(1))

    with patch.object(
        MegatronModule,
        "sharded_state_dict",
        return_value={"mtp.mtp_model_layer.weight": sharded_weight},
    ):
        layer.sharded_state_dict(prefix="mtp.")

    assert sharded_weight.key == "mtp.mtp_model_layer.weight"


def test_mtp_layer_maps_prediction_depth_to_global_metric_offset():
    layer = MultiTokenPredictionLayer.__new__(MultiTokenPredictionLayer)
    torch.nn.Module.__init__(layer)
    layer.config = SimpleNamespace(num_layers=4)
    layer.mtp_metric_layer_count = 2
    layer.mtp_model_layer = Mock()

    layer.set_moe_metric_depth(depth_index=1, num_layers=8)

    layer.mtp_model_layer.set_moe_metric_layer_offset.assert_called_once_with(6, 8)


def test_repeated_direct_mtp_builds_one_existing_layer():
    block = MultiTokenPredictionBlock.__new__(MultiTokenPredictionBlock)
    torch.nn.Module.__init__(block)
    block.config = SimpleNamespace(mtp_num_layers=2)
    block.submodules = SimpleNamespace(layer_specs=[object()])
    block.mtp_layer_pattern = None
    block.mtp_layer_specs = (object(), object())
    block.mtp_num_depths = 2
    block.hybrid_submodules = object()
    block.moe_metric_num_layers = 8
    block.mtp_use_repeated_layer = True
    block.vp_stage = None
    block.name = None

    built_layer = nn.Identity()
    with (
        patch(
            "megatron.core.transformer.multi_token_prediction.get_fp8_context",
            return_value=nullcontext(),
        ),
        patch(
            "megatron.core.transformer.multi_token_prediction.build_module",
            return_value=built_layer,
        ) as build_module,
    ):
        block._build_layers(pg_collection=object())

    assert list(block.layers) == [built_layer]
    assert build_module.call_args.kwargs["mtp_layer_specs"] is block.mtp_layer_specs


class _RepeatedMtpLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.metric_depths = []

    def set_moe_metric_depth(self, depth_index, num_layers):
        self.metric_depths.append((depth_index, num_layers))

    def forward(self, **kwargs):
        return (
            kwargs["hidden_states"] + 1,
            kwargs["input_ids"],
            kwargs["position_ids"],
            kwargs["padding_mask"],
        )


def test_repeated_mtp_retargets_moe_metric_slots_for_every_depth():
    block = MultiTokenPredictionBlock.__new__(MultiTokenPredictionBlock)
    torch.nn.Module.__init__(block)
    block.config = SimpleNamespace(mtp_num_layers=2, mtp_detach_heads=False)
    block.vp_stage = None
    block.mtp_use_repeated_layer = True
    block.mtp_layer_specs = (object(),)
    block.moe_metric_num_layers = 8
    repeated_layer = _RepeatedMtpLayer()
    block.layers = nn.ModuleList([repeated_layer])

    with patch(
        "megatron.core.transformer.multi_token_prediction.get_mtp_layer_offset", return_value=0
    ):
        output = block(
            input_ids=torch.zeros(1, 2, dtype=torch.long),
            position_ids=torch.zeros(1, 2, dtype=torch.long),
            hidden_states=torch.zeros(2, 1, 4),
            attention_mask=None,
        )

    assert repeated_layer.metric_depths == [(0, 8), (1, 8)]
    assert output.shape == (6, 1, 4)


def test_repeated_legacy_mtp_does_not_retarget_moe_metric_slots():
    block = MultiTokenPredictionBlock.__new__(MultiTokenPredictionBlock)
    torch.nn.Module.__init__(block)
    block.config = SimpleNamespace(mtp_num_layers=2, mtp_detach_heads=False)
    block.vp_stage = None
    block.mtp_use_repeated_layer = True
    repeated_layer = _RepeatedMtpLayer()
    block.layers = nn.ModuleList([repeated_layer])

    with patch(
        "megatron.core.transformer.multi_token_prediction.get_mtp_layer_offset", return_value=0
    ):
        block(
            input_ids=torch.zeros(1, 2, dtype=torch.long),
            position_ids=torch.zeros(1, 2, dtype=torch.long),
            hidden_states=torch.zeros(2, 1, 4),
            attention_mask=None,
        )

    assert repeated_layer.metric_depths == []
    assert not hasattr(block, "mtp_layer_specs")
    assert not hasattr(block, "moe_metric_num_layers")


def test_legacy_hybrid_mtp_builder_does_not_receive_direct_only_kwargs():
    block = MultiTokenPredictionBlock.__new__(MultiTokenPredictionBlock)
    torch.nn.Module.__init__(block)
    block.config = SimpleNamespace(mtp_num_layers=2)
    block.submodules = SimpleNamespace(layer_specs=[object()])
    block.mtp_layer_pattern = "ME"
    block.mtp_num_depths = 2
    block.hybrid_submodules = object()
    block.mtp_use_repeated_layer = True
    block.vp_stage = None
    block.name = "mtp"

    with (
        patch(
            "megatron.core.transformer.multi_token_prediction.get_fp8_context",
            return_value=nullcontext(),
        ),
        patch(
            "megatron.core.transformer.multi_token_prediction.build_module",
            return_value=nn.Identity(),
        ) as build_module,
    ):
        block._build_layers(pg_collection=object())

    kwargs = build_module.call_args.kwargs
    assert kwargs["mtp_layer_pattern"] == "ME"
    assert "mtp_layer_specs" not in kwargs
    assert "moe_metric_num_layers" not in kwargs
    assert not hasattr(block, "mtp_layer_specs")
    assert not hasattr(block, "moe_metric_num_layers")


def test_direct_mtp_parameters_do_not_shift_legacy_positional_signatures():
    layer_parameters = signature(MultiTokenPredictionLayer.__init__).parameters
    block_parameters = signature(MultiTokenPredictionBlock.__init__).parameters

    assert list(layer_parameters).index("mtp_layer_specs") > list(layer_parameters).index("name")
    assert list(layer_parameters).index("moe_metric_num_layers") > list(layer_parameters).index(
        "name"
    )
    assert list(block_parameters).index("mtp_layer_specs") > list(block_parameters).index("name")
    assert list(block_parameters).index("moe_metric_num_layers") > list(block_parameters).index(
        "name"
    )


def test_repeated_mtp_restores_metric_depth_inside_recompute_closure():
    layer = MultiTokenPredictionLayer.__new__(MultiTokenPredictionLayer)
    torch.nn.Module.__init__(layer)
    layer.config = SimpleNamespace(
        num_layers=4,
        fp8=False,
        fp4=False,
        distribute_saved_activations=False,
        recompute_method="uniform",
        recompute_num_layers=1,
    )
    layer.mtp_metric_layer_count = 2
    layer.mtp_model_layer = Mock()
    layer._proj_and_transformer_layer = Mock(side_effect=lambda **kwargs: kwargs["hidden_states"])
    closures = []

    def checkpoint(function, distribute_saved_activations, *args):
        closures.append((function, args))
        return function(*args)

    hidden_states = torch.zeros(2, 1, 4, requires_grad=True)
    decoder_input = torch.zeros_like(hidden_states)
    with patch(
        "megatron.core.transformer.multi_token_prediction.tensor_parallel.checkpoint",
        side_effect=checkpoint,
    ):
        for depth_index in range(2):
            layer.set_moe_metric_depth(depth_index, num_layers=8)
            layer._checkpointed_forward(hidden_states, decoder_input)

    # Both checkpoint closures recompute after the outer loop has left the
    # shared router targeted at depth 1. Each closure must restore its own slot.
    layer.mtp_model_layer.reset_mock()
    for function, args in closures:
        function(*args)

    assert layer.mtp_model_layer.set_moe_metric_layer_offset.call_args_list == [
        call(4, 8),
        call(6, 8),
    ]


def test_selective_moe_recompute_restores_direct_mtp_metric_slot():
    layer = MoELayer.__new__(MoELayer)
    torch.nn.Module.__init__(layer)
    layer.training = True
    layer.config = SimpleNamespace(sequence_parallel=True, fp8=False, fp4=False)
    layer.attn_tp_group = SimpleNamespace(size=lambda: 1)
    layer.moe_layer_recompute = True
    layer.fwd_execution_map = {"route", "expert_compute", "postprocess"}
    layer.router = Mock(metric_layer_number=5, metric_num_layers=8)
    layer.shared_experts_compute = Mock(return_value=None)
    layer.route = Mock(return_value=(torch.ones(1), torch.ones(1)))
    layer.preprocess = Mock(side_effect=lambda hidden, probs, _routing: (hidden, probs))
    layer.dispatch = Mock(side_effect=lambda hidden, probs: (hidden, probs))
    layer.routed_experts_compute = Mock(side_effect=lambda hidden, _probs: (hidden, None))
    layer.combine = Mock(side_effect=lambda output: output)
    layer.postprocess = Mock(side_effect=lambda output, _shared: output)
    closures = []

    def checkpoint(function, _distribute_saved_activations, *args):
        closures.append((function, args))
        return function(*args)

    hidden_states = torch.zeros(2, 1, 4, requires_grad=True)
    with patch(
        "megatron.core.transformer.moe.moe_layer.tensor_parallel.checkpoint", side_effect=checkpoint
    ):
        layer(hidden_states)

    layer.router.metric_layer_number = 9
    layer.router.set_metric_layer_number.reset_mock()
    function, args = closures[0]
    function(*args)

    layer.router.set_metric_layer_number.assert_called_once_with(5, 8)
