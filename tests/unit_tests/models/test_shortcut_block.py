# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
import transformer_engine as te

from megatron.core.fusions import wide_residual_shortcut as kernels
from megatron.core.fusions.wide_residual_shortcut import (
    PendingShortcutAdd,
    api,
    forward_kernel,
    mxfp8_handoff,
    replay_kernel,
)
from megatron.core.fusions.wide_residual_shortcut.residual_replay import residual_replay_scope
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.hybrid import hybrid_layer_specs as specs
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols as LayerSymbols
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.hybrid.shortcut_block import (
    ShortcutMoEBlock,
    group_layers_into_shortcut_blocks,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_layer import MambaLayerSubmodules
from megatron.core.ssm.wide_residual_mamba_layer import WideResidualMambaLayer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import TwoStageAttentionLayer
from megatron.core.transformer.residual_recompute import build_residual_stream_recompute_plan
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.transformer.wide_residual import StreamwiseSigmoidWideResidualConnection
from megatron.core.transformer.wide_residual_config import WideResidualConfig
from tests.unit_tests.determinism.kernels.harness import bytes_equal
from tests.unit_tests.test_utilities import Utils

# Shortcut-owned norms are Transformer Engine norms, which need a GPU to build; the pairing
# validation tests use only fake layers and run anywhere.
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="Transformer Engine norms")


def _shortcut_config(*, parallel: bool = False, wide_residual: WideResidualConfig | None = None):
    return TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        normalization="RMSNorm",
        # Sequence parallelism is what marks the norm parameters, and TransformerConfig only
        # allows it alongside tensor parallelism.
        tensor_model_parallel_size=2,
        sequence_parallel=True,
        num_moe_experts=1,
        moe_shortcut_connection=True,
        moe_shortcut_post_norm=True,
        moe_shortcut_parallel=parallel,
        add_bias_linear=False,
        wide_residual=wide_residual,
    )


class _FakeCompute(torch.nn.Module, TwoStageAttentionLayer):
    def __init__(self, config, *, supports_two_stage: bool = True):
        super().__init__()
        self.config = config
        self.layer_number = 1
        self.is_first_layer = True
        self.is_last_layer = False
        self.supports_two_stage = supports_two_stage

    def supports_two_stage_attention(self) -> bool:
        return self.supports_two_stage

    def forward_pre_attn_and_core_attn(self, hidden_states, **kwargs):
        return (hidden_states,)

    def forward_post_core_attn(self, hidden_states, **kwargs):
        return hidden_states


class _FakeMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tp_group = None


class _FakeNorm:
    def __new__(cls, *, config, hidden_size, eps):
        norm_cls = {"LayerNorm": torch.nn.LayerNorm, "RMSNorm": torch.nn.RMSNorm}[
            config.normalization
        ]
        norm = norm_cls(hidden_size, eps=eps)
        for parameter in norm.parameters():
            setattr(parameter, "sequence_parallel", config.sequence_parallel)
        return norm


class _FakeMoE(torch.nn.Module):
    is_moe_layer = True
    _maybe_unflatten_for_moe = TransformerLayer._maybe_unflatten_for_moe
    _maybe_reflatten_from_moe = TransformerLayer._maybe_reflatten_from_moe

    def __init__(self, config, *, layer_number: int = 2):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.is_first_layer = layer_number == 1
        self.is_last_layer = False
        self.pre_mlp_layernorm = _FakeNorm(
            config=config, hidden_size=config.hidden_size, eps=config.layernorm_epsilon
        )
        self.mlp = _FakeMLP()
        if config.wide_residual is not None:
            self.residual_read_mlp = StreamwiseSigmoidWideResidualConnection(
                config, self.layer_number, "test", operation="read"
            )
            self.residual_write_mlp = StreamwiseSigmoidWideResidualConnection(
                config, layer_number, "mlp", operation="write"
            )

    def _pre_mlp_layernorm_and_residual(self, hidden_states):
        """Stand-in for the layer protocol: norm output, residual, and an empty payload."""
        return hidden_states, hidden_states, ()

    def _get_mlp_residual_connection(self, operation):
        if self.config.wide_residual is None:
            return None
        match operation:
            case "read":
                return self.residual_read_mlp
            case "write":
                return self.residual_write_mlp
            case _:
                raise ValueError(f"Unsupported residual connection operation: {operation}.")


@GPU
@pytest.mark.parametrize(
    ("compute_symbol", "parallel"),
    [
        pytest.param(LayerSymbols.MAMBA, False, id="mamba-serial"),
        pytest.param(LayerSymbols.GDN, True, id="gdn-overlap"),
        pytest.param(LayerSymbols.ATTENTION, False, id="attention-serial"),
        pytest.param(LayerSymbols.DS_ATTENTION, False, id="dsa-serial"),
        pytest.param(LayerSymbols.MLA, True, id="mla-overlap"),
    ],
)
def test_group_layers_into_shortcut_blocks(compute_symbol, parallel):
    """Supported pairs are registered together while an initial MoE keeps its normal path."""
    config = _shortcut_config(parallel=parallel)
    leading_moe = _FakeMoE(config, layer_number=1)
    compute = _FakeCompute(config)
    paired_moe = _FakeMoE(config, layer_number=3)
    trailing_layer = torch.nn.Identity()

    grouped = group_layers_into_shortcut_blocks(
        torch.nn.ModuleList([leading_moe, compute, paired_moe, trailing_layer]),
        [LayerSymbols.MOE, compute_symbol, LayerSymbols.MOE, LayerSymbols.MLP],
        config,
    )

    assert len(grouped) == 3
    assert grouped[0] is leading_moe
    assert grouped[2] is trailing_layer
    shortcut = grouped[1]
    assert isinstance(shortcut, ShortcutMoEBlock)
    assert shortcut.compute_layer is compute
    assert shortcut.moe_layer is paired_moe
    assert shortcut.attn_layer_idx == compute.layer_number - 1
    assert shortcut.moe_layer_idx == paired_moe.layer_number - 1
    assert shortcut.attn_local_idx == 1
    assert shortcut.moe_local_idx == 2
    assert shortcut.overlap_mode is parallel
    assert shortcut.shortcut_pre_mlp_layernorm is not paired_moe.pre_mlp_layernorm
    assert isinstance(shortcut.shortcut_post_norm, te.pytorch.RMSNorm)
    for norm in (shortcut.shortcut_pre_mlp_layernorm, shortcut.shortcut_post_norm):
        assert all(parameter.sequence_parallel for parameter in norm.parameters())

    state_keys = set(grouped.state_dict())
    assert "1.shortcut_pre_mlp_layernorm.weight" in state_keys
    assert "1.shortcut_post_norm.weight" in state_keys


@GPU
def test_wide_shortcut_owns_independent_registered_read():
    """The routed shortcut reads wide X_l through its own ordinary-width controller."""

    config = _shortcut_config(wide_residual=WideResidualConfig(num_streams=3))
    compute = _FakeCompute(config)
    compute.residual_stream_hidden_size = config.wide_residual.num_streams * config.hidden_size
    moe = _FakeMoE(config)
    block = ShortcutMoEBlock(compute, moe, overlap_a2a=False).cuda()

    state_keys = set(block.state_dict())
    assert "shortcut_residual_read.read_map.logit" in state_keys
    assert "moe_layer.residual_read_mlp.read_map.logit" in state_keys
    assert (
        block.shortcut_residual_read.read_map.logit
        is not block.moe_layer.residual_read_mlp.read_map.logit
    )

    wide_hidden = torch.randn(
        5, 3 * config.hidden_size, device=torch.cuda.current_device(), requires_grad=True
    )
    shortcut_hidden = block._read_shortcut_hidden(wide_hidden, recompute_context=None)

    assert shortcut_hidden.shape == (5, config.hidden_size)
    shortcut_hidden.square().mean().backward()
    gradient = block.shortcut_residual_read.read_map.logit.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient[: config.wide_residual.num_streams]) > 0
    assert wide_hidden.grad is not None


@GPU
def test_wide_shortcut_mtp_pair_stays_ordinary_width():
    """MTP auxiliary stacks do not inherit the decoder's wide residual stream."""

    config = _shortcut_config(wide_residual=WideResidualConfig(num_streams=3))
    compute = _FakeCompute(config)
    compute.is_mtp_layer = True
    moe = _FakeMoE(config)
    moe.is_mtp_layer = True

    block = ShortcutMoEBlock(compute, moe, overlap_a2a=False)
    ordinary_hidden = torch.randn(3, config.hidden_size)

    assert block.is_mtp_layer
    assert block.shortcut_residual_read is None
    assert block._read_shortcut_hidden(ordinary_hidden, recompute_context=None) is ordinary_hidden
    assert not any("shortcut_residual_read" in key for key in block.state_dict())


def test_shortcut_pair_rejects_mixed_mtp_ownership():
    config = _shortcut_config()
    compute = _FakeCompute(config)
    compute.is_mtp_layer = True

    with pytest.raises(ValueError, match="agree.*MTP"):
        ShortcutMoEBlock(compute, _FakeMoE(config), overlap_a2a=False)


@GPU
@pytest.mark.parametrize("residual_replay", [False, True], ids=["eager", "replay"])
def test_shortcut_owns_cp_layout_transitions(monkeypatch, residual_replay):
    config = _shortcut_config()
    attn_layer = _FakeCompute(config)
    moe_layer = _FakeMoE(config)
    block = ShortcutMoEBlock(
        attn_layer, moe_layer, overlap_a2a=False, attn_local_idx=4, moe_local_idx=5
    )
    calls = []
    observed = {}
    attn_context, moe_context = (
        build_residual_stream_recompute_plan(2, 2) if residual_replay else (None, None)
    )
    expected_layer_kwargs = (
        {"residual_stream_recompute_context": attn_context} if residual_replay else {}
    )
    input_padding_mask = torch.tensor([[False, True]])
    moe_padding_mask = torch.tensor([[True, False]])
    masks_by_layout = {"zigzag": moe_padding_mask}

    class RecordingCPLayoutState:
        def prepare_layer(self, layer_idx, hidden_states):
            calls.append(("prepare", layer_idx))
            return hidden_states + layer_idx, f"packed-{layer_idx}"

        def finalize_layer(self, layer_idx, hidden_states):
            calls.append(("finalize", layer_idx))
            return hidden_states + layer_idx

        def get_layer_padding_mask(self, layer_idx, padding_mask, padding_mask_by_layout):
            assert layer_idx == 5
            assert padding_mask is input_padding_mask
            assert padding_mask_by_layout is masks_by_layout
            return moe_padding_mask

    def attn_forward(hidden_states, packed_seq_params=None, **kwargs):
        observed["attn_packed"] = packed_seq_params
        observed["cp_metadata"] = kwargs.get("packed_sequence_cp_metadata")
        assert {
            key: value
            for key, value in kwargs.items()
            if key == "residual_stream_recompute_context"
        } == expected_layer_kwargs
        return (hidden_states,)

    def attn_post(hidden_states, **kwargs):
        assert kwargs == expected_layer_kwargs
        return hidden_states

    def route(
        shortcut_hidden, padding_mask=None, packed_seq_params=None, recompute_context=None, **_
    ):
        assert recompute_context is attn_context
        observed["route_packed"] = packed_seq_params
        assert padding_mask is moe_padding_mask
        return shortcut_hidden, shortcut_hidden

    def shared(
        hidden_states, padding_mask=None, packed_seq_params=None, recompute_context=None, **_
    ):
        assert recompute_context is moe_context
        observed["shared_packed"] = packed_seq_params
        assert padding_mask is moe_padding_mask
        return torch.zeros_like(hidden_states), None, hidden_states, ()

    def postprocess(
        residual,
        combined_output,
        shared_expert_output,
        packed_seq_params=None,
        moe_unflatten_mbs=None,
        mlp_state=(),
        recompute_context=None,
        **_,
    ):
        assert recompute_context is moe_context
        observed["postprocess_packed"] = packed_seq_params
        return residual

    @contextmanager
    def quant_context_factory(*_args):
        yield

    monkeypatch.setattr(attn_layer, "forward_pre_attn_and_core_attn", attn_forward)
    monkeypatch.setattr(attn_layer, "forward_post_core_attn", attn_post)
    monkeypatch.setattr(block, "_moe_router_preprocess", route)
    monkeypatch.setattr(block, "_launch_dispatch", lambda hidden, probs, **_: (hidden, probs))
    monkeypatch.setattr(
        moe_layer.mlp, "routed_experts_compute", lambda hidden, probs: (hidden, None), raising=False
    )
    monkeypatch.setattr(block, "_launch_combine", lambda output, **_: output)
    monkeypatch.setattr(block, "_moe_shared_experts", shared)
    monkeypatch.setattr(block, "_postprocess", postprocess)

    cp_metadata = object()
    hidden_states = torch.zeros(2, 1, config.hidden_size)
    output = block(
        hidden_states=hidden_states,
        attention_mask=None,
        inference_context=None,
        rotary_pos_emb=None,
        sequence_len_offset=None,
        packed_seq_params="packed-input",
        padding_mask=input_padding_mask,
        quant_context_factory=quant_context_factory,
        cp_layout_state=RecordingCPLayoutState(),
        packed_sequence_cp_metadata=cp_metadata,
        attn_recompute_context=attn_context,
        moe_recompute_context=moe_context,
        padding_mask_by_layout=masks_by_layout,
    )

    assert calls == [("prepare", 4), ("prepare", 5), ("prepare", 5), ("finalize", 5)]
    assert observed == {
        "attn_packed": "packed-4",
        "cp_metadata": cp_metadata,
        "route_packed": "packed-5",
        "shared_packed": "packed-5",
        "postprocess_packed": "packed-5",
    }
    torch.testing.assert_close(output, hidden_states + 14)


@GPU
def test_inference_runs_paired_layer_atomically(monkeypatch):
    """Inference routes on the pair input and runs the paired layer's ordinary forward.

    The two-stage training path never updates KV-cache or recurrent state, so inference must
    not use it; it must also select the MoE layer's inference token dispatcher first.
    """
    config = _shortcut_config()
    compute_layer = _FakeCompute(config)
    moe_layer = _FakeMoE(config)
    block = ShortcutMoEBlock(compute_layer, moe_layer, overlap_a2a=False)
    inference_context = object()
    observed = {}

    def compute_forward(hidden_states, **kwargs):
        observed["compute_context"] = kwargs["inference_context"]
        return hidden_states + 1, None

    def two_stage(*_args, **_kwargs):
        raise AssertionError("inference must not use the two-stage training path")

    def route(
        shortcut_hidden, padding_mask=None, packed_seq_params=None, recompute_context=None, **_
    ):
        observed["route_input"] = shortcut_hidden
        return shortcut_hidden, shortcut_hidden

    def shared(
        hidden_states, padding_mask=None, packed_seq_params=None, recompute_context=None, **_
    ):
        observed["shared_input"] = hidden_states
        return torch.zeros_like(hidden_states), None, hidden_states, ()

    def postprocess(residual, combined_output, shared_expert_output, **_kwargs):
        return residual + combined_output

    @contextmanager
    def quant_context_factory(*_args):
        yield

    monkeypatch.setattr(compute_layer, "forward", compute_forward)
    monkeypatch.setattr(compute_layer, "forward_pre_attn_and_core_attn", two_stage)
    monkeypatch.setattr(
        moe_layer.mlp,
        "select_token_dispatcher",
        lambda: observed.setdefault("dispatcher_selected", True),
        raising=False,
    )
    monkeypatch.setattr(block, "_moe_router_preprocess", route)
    monkeypatch.setattr(block, "_launch_dispatch", lambda hidden, probs, **_: (hidden, probs))
    monkeypatch.setattr(
        moe_layer.mlp, "routed_experts_compute", lambda hidden, probs: (hidden, None), raising=False
    )
    monkeypatch.setattr(block, "_launch_combine", lambda output, **_: output)
    monkeypatch.setattr(block, "_moe_shared_experts", shared)
    monkeypatch.setattr(block, "_postprocess", postprocess)

    hidden_states = torch.zeros(2, 1, config.hidden_size)
    InferenceMode.set_active()
    try:
        output = block(
            hidden_states=hidden_states,
            attention_mask=None,
            inference_context=inference_context,
            rotary_pos_emb=None,
            sequence_len_offset=None,
            packed_seq_params=None,
            padding_mask=None,
            quant_context_factory=quant_context_factory,
        )
    finally:
        InferenceMode.unset_active()

    assert observed["dispatcher_selected"]
    assert observed["compute_context"] is inference_context
    torch.testing.assert_close(observed["route_input"], hidden_states)
    torch.testing.assert_close(observed["shared_input"], hidden_states + 1)
    torch.testing.assert_close(output, hidden_states + 1 + hidden_states)


def test_parallel_stream_is_initialized_once(monkeypatch):
    streams = []

    def make_stream(*, priority):
        stream = SimpleNamespace(priority=priority)
        streams.append(stream)
        return stream

    monkeypatch.setattr(ShortcutMoEBlock, "_parallel_stream", None)
    monkeypatch.setattr(torch.cuda, "Stream", make_stream)

    first = ShortcutMoEBlock._get_a2a_overlap_stream()
    second = ShortcutMoEBlock._get_a2a_overlap_stream()

    assert first is second
    assert streams == [first]
    assert first.priority == -1


@pytest.mark.parametrize(
    ("compute_symbol", "compute", "error"),
    [
        pytest.param(
            LayerSymbols.MLP,
            lambda config: _FakeCompute(config),
            "must be preceded",
            id="unsupported-predecessor",
        ),
        pytest.param(
            LayerSymbols.MAMBA,
            lambda config: _FakeCompute(config, supports_two_stage=False),
            "does not support two-stage attention",
            id="atomic-forward-only",
        ),
    ],
)
def test_group_layers_rejects_invalid_shortcut_pair(compute_symbol, compute, error):
    config = _shortcut_config()
    with pytest.raises(ValueError, match=error):
        group_layers_into_shortcut_blocks(
            torch.nn.ModuleList([compute(config), _FakeMoE(config)]),
            [compute_symbol, LayerSymbols.MOE],
            config,
        )


def test_group_layers_rejects_pair_split_across_pipeline_stages():
    """A stage that starts with an MoE has lost its shortcut partner to the previous stage."""
    config = _shortcut_config()
    with pytest.raises(ValueError, match="cannot cross pipeline-stage boundaries"):
        group_layers_into_shortcut_blocks(
            torch.nn.ModuleList([_FakeMoE(config, layer_number=5), _FakeCompute(config)]),
            [LayerSymbols.MOE, LayerSymbols.MAMBA],
            config,
            pp_layer_offset=4,
        )


@GPU
@pytest.mark.parametrize("tokens_per_sample", [None, 2])
def test_shared_experts_propagates_the_layers_residual_and_mlp_state(tokens_per_sample):
    """The layer owns the unpack and the payload; the block must carry both through."""
    config = _shortcut_config()
    moe_layer = _FakeMoE(config)
    block = ShortcutMoEBlock(_FakeCompute(config), moe_layer, overlap_a2a=False)

    hidden_states = torch.arange(4 * config.hidden_size).float().reshape(4, 1, config.hidden_size)
    normalized = hidden_states * 2
    # Distinct objects, so reusing hidden_states or dropping the payload is detectable.
    layer_residual = hidden_states.clone()
    layer_mlp_state = ("h_res", "h_post")
    moe_layer._pre_mlp_layernorm_and_residual = lambda states: (
        normalized,
        layer_residual,
        layer_mlp_state,
    )
    moe_layer.mlp.shared_experts_compute = lambda states: states

    packed = SimpleNamespace(tokens_per_sample=tokens_per_sample)
    shared_expert_output, mbs, residual, mlp_state = block._moe_shared_experts(
        hidden_states, packed_seq_params=packed
    )
    expected = normalized
    if tokens_per_sample is not None:
        expected = normalized.view(2, 2, config.hidden_size).transpose(0, 1).contiguous()
    assert mbs == (None if tokens_per_sample is None else 2)

    assert torch.equal(shared_expert_output, expected)
    assert residual is layer_residual
    assert mlp_state is layer_mlp_state


@GPU
def test_eager_overlap_matches_serial_output_and_gradients(monkeypatch):
    """Schedules preserve numerics while each physical layer uses its own quant context."""
    config = TransformerConfig(num_layers=2, hidden_size=4, num_attention_heads=1)
    active_quant_layers = []

    def assert_quant_layer(layer_number):
        assert active_quant_layers[-1] == layer_number

    class FakeCompute(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.layer_number = 1
            self.input_scale = torch.nn.Parameter(torch.tensor(2.0))
            self.output_scale = torch.nn.Parameter(torch.tensor(3.0))

        def forward_pre_attn_and_core_attn(self, hidden_states, **kwargs):
            assert_quant_layer(0)
            return hidden_states * self.input_scale, hidden_states

        def forward_post_core_attn(self, projected, residual, **kwargs):
            assert_quant_layer(0)
            return projected * self.output_scale

    class FakeMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tp_group = None
            self.dispatch_scale = torch.nn.Parameter(torch.tensor(5.0))
            self.expert_scale = torch.nn.Parameter(torch.tensor(7.0))
            self.combine_scale = torch.nn.Parameter(torch.tensor(11.0))

        def dispatch(self, route_input, route_probs):
            assert_quant_layer(1)
            return route_input * self.dispatch_scale, route_probs

        def routed_experts_compute(self, dispatched_input, dispatched_probs):
            assert_quant_layer(1)
            return dispatched_input + dispatched_probs * self.expert_scale, None

        def combine(self, routed_output):
            assert_quant_layer(1)
            return routed_output * self.combine_scale

        def postprocess(self, combined_output, shared_expert_output):
            assert_quant_layer(1)
            return combined_output + shared_expert_output

    class FakeMoE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.layer_number = 2
            self.route_scale = torch.nn.Parameter(torch.tensor(13.0))
            self.prob_scale = torch.nn.Parameter(torch.tensor(17.0))
            self.shared_scale = torch.nn.Parameter(torch.tensor(19.0))
            self.mlp = FakeMLP()

        def shortcut_route_preprocess(
            self,
            shortcut_hidden,
            padding_mask=None,
            packed_seq_params=None,
            recompute_context=None,
            **_,
        ):
            assert recompute_context is None
            assert_quant_layer(1)
            return shortcut_hidden * self.route_scale, shortcut_hidden * self.prob_scale

        def shortcut_shared_experts(
            self,
            hidden_states,
            padding_mask=None,
            packed_seq_params=None,
            recompute_context=None,
            **_,
        ):
            assert recompute_context is None
            assert_quant_layer(1)
            return hidden_states * self.shared_scale, None, hidden_states, ()

        @staticmethod
        def _maybe_reflatten_from_moe(output, packed_seq_params, mbs):
            return output

        def _apply_mlp_bda_step(self, output_with_bias, residual, mlp_state=()):
            assert_quant_layer(1)
            return output_with_bias[0] + residual

    @contextmanager
    def quant_context_factory(config, layer_number):
        active_quant_layers.append(layer_number)
        try:
            yield
        finally:
            active_quant_layers.pop()

    def run(overlap):
        block = ShortcutMoEBlock(FakeCompute(), FakeMoE(), overlap_a2a=overlap)
        monkeypatch.setattr(
            block, "_moe_router_preprocess", block.moe_layer.shortcut_route_preprocess
        )
        monkeypatch.setattr(block, "_moe_shared_experts", block.moe_layer.shortcut_shared_experts)
        if overlap:
            monkeypatch.setattr(
                block,
                "_launch_dispatch",
                lambda route_input, route_probs, async_op=False: block.moe_layer.mlp.dispatch(
                    route_input, route_probs
                ),
            )
            monkeypatch.setattr(
                block,
                "_wait_dispatch",
                lambda dispatched_input, dispatched_probs: (dispatched_input, dispatched_probs),
            )
            monkeypatch.setattr(
                block,
                "_launch_combine",
                lambda output, async_op=False: block.moe_layer.mlp.combine(output),
            )
            monkeypatch.setattr(block, "_wait_combine", lambda combined_output: combined_output)

        hidden_states = torch.arange(1.0, 5.0, requires_grad=True)
        output = block(
            hidden_states=hidden_states,
            attention_mask=None,
            inference_context=None,
            rotary_pos_emb=None,
            sequence_len_offset=None,
            packed_seq_params=None,
            padding_mask=None,
            quant_context_factory=quant_context_factory,
        )
        output.sum().backward()
        gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in block.named_parameters()
            if parameter.grad is not None
        }
        gradients["hidden_states"] = hidden_states.grad.detach().clone()
        return output, gradients

    monkeypatch.setattr(
        torch.cuda, "Event", lambda *args, **kwargs: SimpleNamespace(record=lambda stream: None)
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: object())
    serial_output, serial_gradients = run(overlap=False)
    overlap_output, overlap_gradients = run(overlap=True)

    torch.testing.assert_close(overlap_output, serial_output)
    assert overlap_gradients.keys() == serial_gradients.keys()
    for name in overlap_gradients:
        torch.testing.assert_close(overlap_gradients[name], serial_gradients[name])


@GPU
def test_shortcut_norm_recompute_and_offload(monkeypatch):
    """The shortcut path wires pre-norm recompute and post-norm offload together."""
    from megatron.core.models.hybrid import shortcut_block as shortcut_block_module

    calls = {}

    class RecordingCheckpoint:
        def checkpoint(self, run_function, hidden_states):
            calls["checkpoint_input"] = hidden_states
            calls["checkpoint_output"] = run_function(hidden_states)
            return calls["checkpoint_output"]

        def discard_output_and_register_recompute(self, hook_tensor):
            calls["recompute_hook"] = hook_tensor

    class RecordingOffload:
        def __init__(self, enabled, tensor, name):
            calls["offload_enabled"] = enabled
            calls["offload_input"] = tensor
            calls["offload_name"] = name
            self.tensor = tensor

        def __enter__(self):
            calls["offload_entered"] = True
            return self.tensor

        def __exit__(self, *_args):
            calls["offload_exited"] = True

        def group_offload(self, tensor, forced_released_tensors=None):
            calls["offloaded_tensor"] = tensor
            calls["forced_released_tensors"] = forced_released_tensors
            return tensor

    monkeypatch.setattr(shortcut_block_module, "CheckpointWithoutOutput", RecordingCheckpoint)
    monkeypatch.setattr(
        shortcut_block_module, "_get_offloading_interface", lambda: RecordingOffload
    )

    config = _shortcut_config()
    config.recompute_granularity = "selective"
    config.recompute_modules = ["shortcut_pre_mlp_layernorm"]
    config.fine_grained_activation_offloading = True
    config.offload_modules = ["shortcut_post_norm"]

    moe_layer = _FakeMoE(config)
    mlp = moe_layer.mlp
    mlp.route = lambda hidden_states, padding_mask=None, packed_seq_params=None: (
        torch.ones_like(hidden_states),
        torch.ones_like(hidden_states, dtype=torch.bool),
    )
    mlp.preprocess = (
        lambda hidden_states, probs, routing_map, padding_mask=None, packed_seq_params=None: (
            hidden_states,
            probs,
        )
    )
    mlp.dispatch = lambda hidden_states, probs: (hidden_states, probs)
    mlp.routed_experts_compute = lambda hidden_states, probs: (hidden_states + probs, None)

    def combine(routed_output):
        calls["combined_output"] = routed_output * 2
        return calls["combined_output"]

    mlp.combine = combine
    mlp.shared_experts_compute = torch.zeros_like
    mlp.postprocess = lambda combined_output, shared_expert_output: combined_output
    moe_layer._apply_mlp_bda_step = (
        lambda output_with_bias, residual, mlp_state=(): output_with_bias[0] + residual
    )

    block = ShortcutMoEBlock(_FakeCompute(config), moe_layer, overlap_a2a=False).cuda()
    hidden_states = torch.randn(
        2, 1, config.hidden_size, device=torch.cuda.current_device(), requires_grad=True
    )

    @contextmanager
    def quant_context_factory(*_args):
        yield

    output = block(
        hidden_states=hidden_states,
        attention_mask=None,
        inference_context=None,
        rotary_pos_emb=None,
        sequence_len_offset=None,
        packed_seq_params=None,
        padding_mask=None,
        quant_context_factory=quant_context_factory,
    )
    output.sum().backward()

    assert calls["checkpoint_input"] is hidden_states
    assert calls["recompute_hook"] is calls["combined_output"]
    assert block.shortcut_pre_mlp_layernorm_checkpoint is None
    assert calls["offload_enabled"]
    assert calls["offload_name"] == "shortcut_post_norm"
    assert calls["offload_input"] is calls["combined_output"]
    assert calls["offload_entered"] and calls["offload_exited"]
    assert calls["forced_released_tensors"][0] is calls["offload_input"]
    assert hidden_states.grad is not None


# ---------------------------------------------------------------- fused wide-residual shortcut blocks
#
# ShortcutMoEBlock with moe_shortcut_wide_residual_norm_fusion runs the one-reader kernel
# (wide_residual_write_read_rmsnorm_quant) between its compute write and shared-expert read, and
# the two-reader kernel (shortcut_add_wide_residual_write_two_reader_rmsnorm_quant) for the
# shortcut add plus the next pair's reads. Kernel arithmetic is checked against independent FP64
# references in tests/unit_tests/fusions/test_wide_residual_shortcut.py; these tests check the
# block schedule bitwise against a serial oracle built from the same kernels.


class _BF16Norm(torch.nn.RMSNorm):
    def __init__(self, config, hidden_size, eps):
        super().__init__(hidden_size, eps=eps, device="cuda", dtype=torch.bfloat16)


class _TanhMixer(torch.nn.Module, TwoStageAttentionLayer):
    def __init__(self, config, d_model, **kwargs):
        super().__init__()
        linear = lambda i, o: torch.nn.Linear(i, o, bias=False, device="cuda", dtype=torch.bfloat16)
        self.input, self.output = linear(d_model, 32), linear(32, d_model)

    def forward(self, x, **kwargs):
        return self.forward_post_core_attn(self.forward_pre_attn_and_core_attn(x))

    def supports_two_stage_attention(self):
        return True

    def forward_pre_attn_and_core_attn(self, x, **kwargs):
        return torch.tanh(self.input(x))

    def forward_post_core_attn(self, x):
        return self.output(x), None


class _SharedExperts(torch.nn.Module):
    use_shared_expert = True
    shared_expert_overlap = False
    tp_group = None

    def __init__(self, hidden):
        super().__init__()
        linear = lambda o: torch.nn.Linear(
            hidden, o, bias=False, device="cuda", dtype=torch.bfloat16
        )
        self.router, self.shared, self.routed = linear(1), linear(hidden), linear(hidden)
        self.combine_calls = 0

    def route(self, x, padding_mask=None, input_ids=None, packed_seq_params=None):
        return self.router(x).sigmoid(), None

    def preprocess(self, x, probs, routing, padding_mask=None, packed_seq_params=None):
        return x, probs

    def dispatch(self, x, probs):
        return x, probs

    def routed_experts_compute(self, x, probs):
        return self.routed(x) * probs, None

    def combine(self, x):
        return x

    def shared_experts_compute(self, x):
        return self.shared(x)

    def postprocess(self, x, shared):
        self.combine_calls += 1
        return x if shared is None else x + shared


class _WideMoE(torch.nn.Module):
    supports_wide_residual_connections = True
    is_mtp_layer = False
    is_last_layer = False

    def __init__(self, config, layer_number, pg=None, pg_collection=None, **kwargs):
        pg = pg_collection if pg_collection is not None else pg
        super().__init__()
        self.config, self.layer_number = config, layer_number
        connection = lambda operation: StreamwiseSigmoidWideResidualConnection(
            config, layer_number, "mlp", pg, operation=operation
        )
        self.residual_read_mlp, self.residual_write_mlp = connection("read"), connection("write")
        self.pre_mlp_layernorm = _BF16Norm(config, config.hidden_size, 1e-5)
        self.mlp = _SharedExperts(config.hidden_size)

    def _get_mlp_residual_connection(self, operation):
        return self.residual_read_mlp if operation == "read" else self.residual_write_mlp

    def _pre_mlp_layernorm_and_residual(self, x, residual_stream_recompute_context=None):
        x, state = self.residual_read_mlp(
            x,
            fp32_residual_connection=self.config.fp32_residual_connection,
            branch_input_dtype=self.config.params_dtype,
        )
        return self.pre_mlp_layernorm(x), state[0], state

    def _maybe_unflatten_for_moe(self, x, padding_mask=None, **kwargs):
        return x, padding_mask, None, None

    def _maybe_reflatten_from_moe(self, x, *args):
        return x

    def _apply_mlp_bda_step(self, output, residual, state, residual_stream_recompute_context=None):
        return self.residual_write_mlp(
            output, state=state, dropout_probability=0.0, training=self.training
        )


def _fused_config(**overrides):
    """BF16 params, FP32 residual, RMSNorm and the fused wide-residual shortcut."""
    options = dict(
        normalization="RMSNorm",
        layernorm_epsilon=1e-5,
        params_dtype=torch.bfloat16,
        bf16=True,
        fp32_residual_connection=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        add_bias_linear=False,
        moe_shortcut_connection=True,
        moe_shortcut_wide_residual_norm_fusion=True,
    )
    return TransformerConfig(**(options | overrides))


def _fused_blocks(streams=3, enabled=True, retention=True, pairs=2, hidden=128, fp32_residual=True):
    # Bind before the fake module tests create the process-wide overlap stream.
    torch.cuda.set_device(Utils.local_rank % torch.cuda.device_count())
    config = _fused_config(
        fp32_residual_connection=fp32_residual,
        num_layers=2 * pairs,
        hidden_size=hidden,
        num_attention_heads=4,
        num_moe_experts=1,
        moe_shared_expert_intermediate_size=128,
        moe_shortcut_wide_residual_norm_fusion=enabled,
        wide_residual=WideResidualConfig(num_streams=streams, learned_retention=retention),
    )
    pg = ProcessGroupCollection(tp=None)
    physical = torch.nn.ModuleList()
    for index in range(1, 2 * pairs, 2):
        compute = WideResidualMambaLayer(
            config,
            MambaLayerSubmodules(mixer=_TanhMixer, norm=_BF16Norm),
            layer_number=index,
            pg_collection=pg,
        )
        physical.extend([compute, _WideMoE(config, index + 1, pg)])
    return group_layers_into_shortcut_blocks(
        physical, [LayerSymbols.GDN, LayerSymbols.MOE] * pairs, config
    ).cuda()


def _run_block(block, x, defer=False):
    """Run one block; a PendingShortcutAdd input starts with the two-reader boundary and
    ``defer`` returns one instead of finishing the shortcut add."""
    block.defers_residual_write = defer
    return block(x, *(None,) * 6, lambda *a: nullcontext())


def _run_blocks(blocks, x, manager=None):
    """Drive adjacent fused blocks like HybridStack: defer every MoE write but the last."""
    for i, block in enumerate(blocks):
        if manager is not None:
            manager.advance(2)
        x = _run_block(block, x, defer=i + 1 < len(blocks))
        if i + 1 < len(blocks):
            assert isinstance(x, PendingShortcutAdd)
    return x


def _explicit_pairs(blocks, residual, manager=None, prepared=None):
    """Serial scheduling oracle using the same one-/two-reader arithmetic contract.

    Independent of ShortcutMoEBlock.forward and its overlap/handoff logic. ``prepared`` holds
    the first block's normalized (compute, shortcut) inputs when a two-reader boundary already
    produced them.
    """
    for i, block in enumerate(blocks):
        if manager is not None:
            manager.advance(2)
        compute, moe = block.compute_layer, block.moe_layer
        if prepared is None:
            shortcut = block._read_shortcut_hidden(residual, recompute_context=None)
            route, probs = block._moe_router_preprocess(shortcut)
            compute_input, _ = compute.residual_read(
                residual, fp32_residual_connection=True, branch_input_dtype=torch.bfloat16
            )
            compute_input = compute.norm(compute_input.to(torch.bfloat16))
        else:
            compute_input, route = prepared
            probs, _ = moe.mlp.route(route)
        routed, _ = moe.mlp.routed_experts_compute(route, probs)
        update, _ = compute.mixer.forward_post_core_attn(
            compute.mixer.forward_pre_attn_and_core_attn(compute_input)
        )
        producer, consumer_read = compute.residual_write, moe.residual_read_mlp
        consumer_write = moe.residual_write_mlp
        residual, shared_input = kernels.wide_residual_write_read_rmsnorm_quant(
            residual,
            update,
            producer.write_map(return_logits=True),
            producer.write_logits()[1],
            consumer_read.read_map(return_logits=True),
            moe.pre_mlp_layernorm.weight,
            num_streams=block.config.wide_residual.num_streams,
        )
        shared = moe.mlp.shared_experts_compute(shared_input)
        if i + 1 < len(blocks):
            nxt = blocks[i + 1]
            residual, compute_input, shortcut = (
                kernels.shortcut_add_wide_residual_write_two_reader_rmsnorm_quant(
                    PendingShortcutAdd(
                        residual,
                        routed,
                        shared,
                        consumer_write.write_map(return_logits=True),
                        consumer_write.write_logits()[1],
                    ),
                    (
                        nxt.compute_layer.residual_read.read_map(return_logits=True),
                        nxt.shortcut_residual_read.read_map(return_logits=True),
                    ),
                    (nxt.compute_layer.norm.weight, nxt.shortcut_pre_mlp_layernorm.weight),
                    num_streams=block.config.wide_residual.num_streams,
                )
            )
            prepared = compute_input, shortcut
        else:
            residual = block._postprocess(residual, routed, shared, mlp_state=(residual,))
    return residual


@GPU
def test_fused_block_flags_and_validation():
    blocks = _fused_blocks(pairs=3)
    assert all(block.using_shortcut_moe_fused_residual_kernel for block in blocks)
    assert all(block.compute_layer.using_shortcut_moe_fused_residual_kernel for block in blocks)
    unfused = _fused_blocks(pairs=1, enabled=False)[0]
    assert not unfused.using_shortcut_moe_fused_residual_kernel
    assert not unfused.compute_layer.using_shortcut_moe_fused_residual_kernel
    with pytest.raises(ValueError, match="require the fused residual kernels"):
        _run_block(unfused, torch.randn(32, 1, 384, device="cuda"), defer=True)
    block = _fused_blocks(pairs=1)[0].eval()
    with pytest.raises(ValueError, match="requires InferenceMode"):
        block(
            torch.randn(64, 1, 384, device="cuda"),
            None,
            object(),
            *(None,) * 4,
            lambda *a: torch.no_grad(),
        )


@GPU
@pytest.mark.parametrize("checkpointed,gtp", [(False, False), (True, False), (True, True)])
def test_direct_mxfp8_handoff_always_declares_deferred_consumer(monkeypatch, checkpointed, gtp):
    blocks = _fused_blocks()
    mixer = blocks[1].compute_layer.mixer
    mixer.config = blocks[1].config
    mixer.in_proj = mixer.input
    mixer.recompute_in_proj = checkpointed
    mixer.in_proj.weight.is_gtp_weight_remat = gtp
    monkeypatch.setattr(mxfp8_handoff, "eligible", lambda linear, *_: linear is mixer.in_proj)
    residual = torch.randn(1024, 1, 384, device="cuda", requires_grad=True)

    def boundary():
        # The two-reader boundary runs at the start of the consuming block and emits its
        # compute input.
        pending = _run_block(blocks[0], residual, defer=True)
        return blocks[1]._shortcut_add(pending, lambda *a: nullcontext())[1]

    if checkpointed and not gtp:
        with pytest.raises(ValueError, match="requires GTP weight finalization"):
            boundary()
        # There is no weight-gradient restriction during inference.
        with torch.no_grad():
            compute = boundary()
        assert compute._rowwise_data is not None
    else:
        compute = boundary()
        reader = mxfp8_handoff.reader_for(compute)
        assert reader is not None and reader.consumer() is mixer.in_proj


@GPU
@pytest.mark.parametrize(
    "streams,retention,overlap,interval,tokens,hidden,dtype",
    [
        (s, t, o, i, 1024, 128, torch.float32)
        for s in range(2, 7)
        for t in (False, True)
        for o in (False, True)
        for i in (None, 16)
    ]
    # Production shapes with the overlapped schedule.
    + [
        (3, True, True, None, m, h, torch.float32)
        for m, h in ((16384, 3072), (8192, 4096), (8192, 10240))
    ]
    # A BF16 residual stream (no fp32_residual_connection) with BF16 connection logits.
    + [
        (s, True, o, i, m, h, torch.bfloat16)
        for s, o, i, m, h in (
            (3, False, None, 1024, 128),
            (3, True, 16, 1024, 128),
            (5, True, None, 1024, 128),
            (3, True, None, 8192, 4096),
        )
    ],
)
def test_fused_schedule_matches_serial_oracle(
    streams, retention, overlap, interval, tokens, hidden, dtype
):
    """Block outputs, input gradients and every parameter gradient match bit for bit."""
    torch.manual_seed(1234)
    blocks = _fused_blocks(
        streams,
        retention=retention,
        pairs=9 if interval else 2,
        hidden=hidden,
        fp32_residual=dtype == torch.float32,
    )
    if dtype == torch.bfloat16:
        blocks = blocks.bfloat16()  # BF16 training casts every parameter, logits included
    blocks[0].config.residual_stream_recompute_num_layers = interval
    explicit = copy.deepcopy(blocks)
    if overlap:
        for block in blocks:
            block.overlap_mode = True
            block.route_ready_event = torch.cuda.Event()
    r = torch.randn(tokens, 1, streams * hidden, device="cuda", dtype=dtype, requires_grad=True)
    other = r.detach().clone().requires_grad_(True)
    with residual_replay_scope(blocks[0].config, True) as manager:
        actual = _run_blocks(blocks, r, manager)
    if interval is not None:
        assert manager is not None
        assert any(not slot.resident() for slot in manager.slots.values())
    with residual_replay_scope(explicit[0].config, True) as manager:
        expected = _explicit_pairs(explicit, other, manager)
    assert isinstance(actual, torch.Tensor) and actual.dtype == dtype
    assert torch.isfinite(actual).all()
    assert bytes_equal(actual, expected), "block output"
    incoming = torch.randn_like(actual)
    actual.backward(incoming)
    expected.backward(incoming)
    assert torch.isfinite(r.grad).all()
    assert bytes_equal(r.grad, other.grad), "input gradient"
    for (name, a), (other_name, b) in zip(blocks.named_parameters(), explicit.named_parameters()):
        assert name == other_name
        if a.grad is None or b.grad is None:
            assert a.grad is None and b.grad is None
        else:
            assert torch.isfinite(a.grad).all(), name
            assert bytes_equal(a.grad, b.grad), name


def _lone_then_blocks(
    streams=3, retention=True, pairs=2, hidden=128, enabled=True, lone_last=False
):
    """A lone wide-residual Mamba layer followed by fused blocks (M -> [* E] [* E] ...).

    ``lone_last`` moves the lone layer after the blocks instead.
    """
    torch.cuda.set_device(Utils.local_rank % torch.cuda.device_count())
    config = _fused_config(
        num_layers=1 + 2 * pairs,
        hidden_size=hidden,
        num_attention_heads=4,
        num_moe_experts=1,
        moe_shared_expert_intermediate_size=128,
        moe_shortcut_wide_residual_norm_fusion=enabled,
        wide_residual=WideResidualConfig(num_streams=streams, learned_retention=retention),
    )
    pg = ProcessGroupCollection(tp=None)
    submodules = MambaLayerSubmodules(mixer=_TanhMixer, norm=_BF16Norm)
    physical, symbols = torch.nn.ModuleList(), []
    for index in range(1, 2 * pairs + 2):
        lone = index == (2 * pairs + 1 if lone_last else 1)
        pair_index = index - (0 if lone_last else 1)
        if lone or pair_index % 2:
            physical.append(
                WideResidualMambaLayer(config, submodules, layer_number=index, pg_collection=pg)
            )
            symbols.append(LayerSymbols.MAMBA if lone else LayerSymbols.GDN)
        else:
            physical.append(_WideMoE(config, index, pg))
            symbols.append(LayerSymbols.MOE)
    return group_layers_into_shortcut_blocks(physical, symbols, config).cuda()


def _run_stack(layers, x, manager=None):
    """Drive a lone layer and fused blocks like HybridStack's eager loop."""
    for i, layer in enumerate(layers):
        block = isinstance(layer, ShortcutMoEBlock)
        if manager is not None:
            manager.advance(2 if block else 1)
        if block:
            x = _run_block(layer, x, defer=i + 1 < len(layers))
        else:
            x = layer(x)
            if layer.defers_residual_write:
                assert isinstance(x, PendingShortcutAdd) and x.shared is None
    return x


def _two_reader_from_lone(lone, block, residual, pending):
    """The lone layer's write plus ``block``'s compute and shortcut reads, from the kernel API."""
    write = lone.residual_write
    return kernels.shortcut_add_wide_residual_write_two_reader_rmsnorm_quant(
        PendingShortcutAdd(
            residual,
            pending.routed,
            None,
            write.write_map(return_logits=True),
            write.write_logits()[1],
        ),
        (
            block.compute_layer.residual_read.read_map(return_logits=True),
            block.shortcut_residual_read.read_map(return_logits=True),
        ),
        (block.compute_layer.norm.weight, block.shortcut_pre_mlp_layernorm.weight),
        num_streams=block.config.wide_residual.num_streams,
    )


@GPU
@pytest.mark.parametrize(
    "streams,retention,interval",
    [(2, False, None), (3, True, None), (3, True, 16), (6, True, None)],
)
def test_lone_layer_write_fuses_into_next_block(streams, retention, interval):
    """M -> [* E]: the lone layer's write and the block's two reads are one two-reader kernel.

    Outputs and all gradients match a serial oracle bit for bit, and the deferred write equals
    the lone layer's own (unfused) residual write.
    """
    torch.manual_seed(4321)
    layers = _lone_then_blocks(streams, retention, pairs=9 if interval else 2)
    lone, blocks = layers[0], layers[1:]
    assert lone.defers_residual_write
    assert not any(b.compute_layer.defers_residual_write for b in blocks)
    blocks[0].config.residual_stream_recompute_num_layers = interval
    explicit = copy.deepcopy(layers)
    hidden = blocks[0].config.hidden_size
    r = torch.randn(1024, 1, streams * hidden, device="cuda", requires_grad=True)
    other = r.detach().clone().requires_grad_(True)
    with residual_replay_scope(blocks[0].config, True) as manager:
        actual = _run_stack(layers, r, manager)

    class FirstPairAdvanced:
        """Forward the oracle's advances except the first pair's, already taken here."""

        def __init__(self, inner):
            self.inner, self.skip = inner, True

        def advance(self, count):
            if self.skip:
                self.skip = False
            else:
                self.inner.advance(count)

    with residual_replay_scope(explicit[1].config, True) as manager:
        if manager is not None:
            manager.advance(1)
        pending = explicit[0](other)
        if manager is not None:
            manager.advance(2)
        residual, *prepared = _two_reader_from_lone(explicit[0], explicit[1], other, pending)
        expected = _explicit_pairs(
            explicit[1:],
            residual,
            FirstPairAdvanced(manager) if manager is not None else None,
            tuple(prepared),
        )
    assert bytes_equal(actual, expected), "stack output"
    incoming = torch.randn_like(actual)
    actual.backward(incoming)
    expected.backward(incoming)
    assert torch.isfinite(r.grad).all()
    assert bytes_equal(r.grad, other.grad), "input gradient"
    for (name, a), (other_name, b) in zip(layers.named_parameters(), explicit.named_parameters()):
        assert name == other_name
        if a.grad is None or b.grad is None:
            assert a.grad is None and b.grad is None, name
        else:
            assert torch.isfinite(a.grad).all(), name
            assert bytes_equal(a.grad, b.grad), name
    # The deferred write reproduces the lone layer's own residual write.
    with torch.no_grad():
        current = _two_reader_from_lone(lone, blocks[0], r, lone(r))[0]
        lone.defers_residual_write = False
        try:
            written = lone(r)
        finally:
            lone.defers_residual_write = True
    torch.testing.assert_close(current, written, rtol=2e-6, atol=2e-6)


@GPU
def test_lone_write_deferral_scope():
    """Only a lone wide-residual Mamba layer directly before a fused block defers its write."""
    layers = _lone_then_blocks(pairs=1)
    assert isinstance(layers[1], ShortcutMoEBlock)
    assert layers[0].defers_residual_write
    unfused = _lone_then_blocks(pairs=1, enabled=False)
    assert not unfused[1].using_shortcut_moe_fused_residual_kernel
    assert not unfused[0].defers_residual_write
    trailing = _lone_then_blocks(pairs=1, lone_last=True)
    assert isinstance(trailing[0], ShortcutMoEBlock)
    assert not trailing[1].defers_residual_write


def taped_forward(forward, *args, **kwargs):
    """Launch ``forward`` with a backward tape under the inference schedule, one CTA per row."""
    choose = forward_kernel.choose_forward_config

    def same_schedule(m, h, streams, readers=1, **options):
        cfg = choose(m, h, streams, readers, **(options | {"inference": True}))
        # Split inference readers reduce independently; retain normal tape ownership.
        return cfg._replace(PARTS=1, SPLIT=False, PAIRED_REDUCE=cfg.PAIRED_REDUCE and not cfg.SPLIT)

    with patch.object(forward_kernel, "choose_forward_config", side_effect=same_schedule):
        return forward(*args, **(kwargs | {"inference": False}))


def forbid_backward_state(monkeypatch):
    """Fail if inference enters autograd or allocates backward state; returns one entry per
    non-warmup forward launch."""

    def forbidden(*args, **kwargs):
        raise AssertionError("inference entered autograd or allocated backward state")

    targets = (api._Boundary, "apply"), (api, "current_manager"), (forward_kernel, "SavedWrite")
    for owner, name in targets:
        monkeypatch.setattr(owner, name, forbidden)
    launch, observed = forward_kernel._forward.run, []

    def inspect(*args, **kwargs):
        if not kwargs.get("warmup"):
            assert kwargs["saved_coefficients"] is None and args[0].summed_update is None
            assert all(stat is None for stat in kwargs["read_tensors"].rstd)
            observed.append(True)
        return launch(*args, **kwargs)

    monkeypatch.setattr(forward_kernel._forward, "run", inspect)
    return observed


def _inference_and_taped(monkeypatch, execute):
    """Run ``execute`` on the no-tape inference kernel, then on the fused training kernel with
    the same schedule and a backward tape (the bitwise oracle); also return reader counts."""
    launch, calls = forward_kernel.forward, []

    def observe(*args, **kwargs):
        result = launch(*args, **kwargs)
        assert kwargs["inference"] is True and result[1] is None
        calls.append(len(args[1]))
        return result

    with monkeypatch.context() as patcher:
        patcher.setattr(forward_kernel, "forward", observe)
        actual = execute()
    with monkeypatch.context() as patcher:
        patcher.setattr(forward_kernel, "forward", lambda *a, **k: taped_forward(launch, *a, **k))
        expected = execute()
    return actual, expected, calls


@GPU
@pytest.mark.parametrize("hidden", [128, 3072, 4096, 10240])
@pytest.mark.parametrize("mode", [torch.no_grad, torch.inference_mode])
def test_eager_no_grad_block_uses_inference_kernel(monkeypatch, hidden, mode):
    blocks = _fused_blocks(hidden=hidden).eval()
    residual = torch.randn(1024, 1, 3 * hidden, device="cuda", requires_grad=True)
    with mode():
        actual, expected, calls = _inference_and_taped(
            monkeypatch, lambda: _run_blocks(blocks, residual)
        )
        assert torch.equal(actual, expected)
        assert actual.grad_fn is None and not actual.requires_grad
    assert calls == [1, 2, 1]


@GPU
@pytest.mark.parametrize("tokens", [1, 3, 31, 32, 64, 96, 1024])
@pytest.mark.parametrize("with_context", [False, True])
def test_fused_block_actual_engine_mode(monkeypatch, tokens, with_context):
    block = _fused_blocks(pairs=1)[0].eval()
    block.overlap_mode = True
    residual = torch.randn(tokens, 1, 384, device="cuda", requires_grad=True)
    context = object() if with_context else None
    observed = forbid_backward_state(monkeypatch)
    mixer = block.compute_layer.mixer
    calls = []

    def atomic(x, inference_context=None, **kwargs):
        assert inference_context is context
        assert not torch.is_grad_enabled()
        calls.append(True)
        return mixer.output(torch.tanh(mixer.input(x))), None

    def forbidden(*args, **kwargs):
        raise AssertionError("inference entered the training split or overlap stream")

    monkeypatch.setattr(mixer, "forward", atomic)
    monkeypatch.setattr(mixer, "forward_pre_attn_and_core_attn", forbidden)
    monkeypatch.setattr(mixer, "forward_post_core_attn", forbidden)
    monkeypatch.setattr(block, "_get_a2a_overlap_stream", forbidden)
    with InferenceMode.active():
        result = block(residual, None, context, *(None,) * 4, lambda *args: torch.no_grad())
    assert calls == [True] and observed == [True]
    assert not result.requires_grad and result.grad_fn is None
    assert torch.is_grad_enabled()


@GPU
@pytest.mark.parametrize("tokens", [1, 3, 31, 32, 64, 96])
def test_fused_block_short_eager_no_grad(tokens):
    block = _fused_blocks(pairs=1)[0].eval()
    with torch.inference_mode():
        result = _run_block(block, torch.randn(tokens, 1, 384, device="cuda"))
    assert not result.requires_grad


# ---------------------------------------------------------------- HybridModel inference
#
# The no-tape contract is the fused forward with the same schedule and backward tape. The
# ordinary unfused path rounds reads before RMSNorm and is not a bitwise oracle.


@pytest.fixture
def model_parallel():
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    yield
    Utils.destroy_model_parallel()


def _hybrid_model(mixer, optimized, hidden, dynamic, pattern=None, fp32_residual=True):
    pattern = pattern or ("GEGE" if mixer == "gdn" else "MEME") + ("*" if dynamic else "")
    config = _fused_config(
        fp32_residual_connection=fp32_residual,
        num_layers=len(pattern),
        hidden_size=hidden,
        num_attention_heads=hidden // 64,
        mamba_num_heads=4,
        mamba_head_dim=64,
        mamba_num_groups=2,
        mamba_state_dim=32,
        linear_key_head_dim=32,
        linear_value_head_dim=64,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        ffn_hidden_size=256,
        num_moe_experts=2,
        moe_router_topk=2,
        moe_router_pre_softmax=True,
        moe_router_dtype="fp32",
        inference_grouped_gemm_backend="torch",
        moe_shared_expert_intermediate_size=256,
        moe_grouped_gemm=True,
        gated_linear_unit=True,
        activation_func=F.silu,
        moe_shortcut_parallel=True,
        wide_residual=WideResidualConfig(num_streams=3, learned_retention=True),
        transformer_impl="inference_optimized" if optimized else "transformer_engine",
    )
    if mixer == "gdp":
        spec = specs.wide_residual_gated_delta_product_inference_stack_spec
        spec = spec if optimized else specs.wide_residual_gated_delta_product_stack_spec
    else:
        spec = specs.wide_residual_hybrid_inference_stack_spec
        spec = spec if optimized else specs.wide_residual_hybrid_stack_spec
    spec = specs.shortcut_fusion_stack_spec(spec)
    model = HybridModel(
        config, spec, vocab_size=128, max_sequence_length=128, hybrid_layer_pattern=pattern
    )
    model = model.cuda().eval()
    assert sum(
        isinstance(b, ShortcutMoEBlock) and b.using_shortcut_moe_fused_residual_kernel
        for b in model.decoder.layers
    ) == pattern.count("E")
    return model


def _static_prefill_decode(model, keys=(1, 3)):
    ids = torch.randint(0, 128, (2, 99), device="cuda")
    positions = torch.arange(99, device="cuda").expand(2, -1)

    def execute():
        context = StaticInferenceContext(2, 128)
        outputs, states = [], []
        for start, end in [(0, 96), (96, 97), (97, 98), (98, 99)]:
            context.sequence_len_offset = start
            if start:
                context.enable_decode_mode()
            result = model(
                ids[:, start:end],
                positions[:, start:end],
                None,
                inference_context=context,
                runtime_gather_output=True,
            )
            assert not result.requires_grad
            outputs.append(result.clone())
            memory = context.key_value_memory_dict
            states.append({k: tuple(v.clone() for v in vs) for k, vs in memory.items()})
        assert set(states[0]) == set(keys)
        return outputs, states

    return execute


def _dynamic_prefill_decode(model):
    from megatron.core.inference.config import InferenceConfig, MambaInferenceStateConfig
    from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
    from megatron.core.inference.inference_request import DynamicInferenceRequest
    from megatron.core.inference.sampling_params import SamplingParams

    prompts = [torch.randint(0, 128, (length,), device="cuda") for length in (64, 96)]
    generated = torch.tensor([17, 23, 29])

    def request(index, prompt, tokens):
        sampling = SamplingParams(num_tokens_to_generate=tokens)
        return DynamicInferenceRequest(
            request_id=index, prompt_tokens=prompt, sampling_params=sampling
        )

    def execute():
        context = DynamicInferenceContext(
            model_config=model.config,
            inference_config=InferenceConfig(
                max_sequence_length=128,
                buffer_size_gb=0.1,
                block_size_tokens=256,
                max_requests=4,
                max_tokens=256,
                materialize_only_last_token_logits=True,
                mamba_inference_state_config=MambaInferenceStateConfig.from_model(model),
            ),
        )
        for index, prompt in enumerate(prompts):
            context.add_request(request(index, prompt, 4))
        outputs, states = [], []
        for step in range(4):
            context.initialize_attention_state()
            if step == 1:
                assert context.padded_batch_dimensions.decode_req_count > 0
                assert context.padded_batch_dimensions.prefill_req_count > 0
            ids, positions = context.current_input_and_position_ids()
            result = model(
                ids, positions, None, inference_context=context, runtime_gather_output=True
            )
            assert not result.requires_grad
            outputs.append(result.clone())
            active = context.get_active_request_count()
            slots = context.mamba_metadata.request_to_mamba_state_idx[:active].to(
                device="cuda", dtype=torch.long
            )
            states.append(
                {
                    key: tuple(v.index_select(0, slots) for v in context.mamba_states_cache(key))
                    for key in (1, 3)
                }
            )
            if step < 3:
                context.update_requests(torch.ones(active, dtype=torch.bool), generated[:active])
            if step == 0:
                context.add_request(request(2, prompts[0][:32], 3))
        return outputs, states

    return execute


@GPU
@pytest.mark.usefixtures("model_parallel")
@pytest.mark.parametrize(
    "dynamic,mixer,optimized,hidden",
    [(False, mixer, False, hidden) for mixer in ("mamba", "gdp") for hidden in (256, 3072)]
    + [(True, mixer, opt, 3072) for mixer in ("mamba", "gdp", "gdn") for opt in (False, True)],
)
def test_hybrid_inference_prefill_decode(monkeypatch, dynamic, mixer, optimized, hidden):
    """Real HybridModel prefill/decode (static or dynamic batching, routed experts, recurrent
    caches) uses only the no-tape kernel and matches the taped forward bit for bit."""
    model = _hybrid_model(mixer, optimized, hidden, dynamic)

    def forbidden(*args, **kwargs):
        raise AssertionError("inference used the training-only mixer split")

    for block in model.decoder.layers[:2]:
        compute_mixer = block._compute_parts()[3]
        monkeypatch.setattr(compute_mixer, "forward_pre_attn_and_core_attn", forbidden)
        monkeypatch.setattr(compute_mixer, "forward_post_core_attn", forbidden)
    execute = (_dynamic_prefill_decode if dynamic else _static_prefill_decode)(model)
    with torch.inference_mode(), InferenceMode.active():
        (outputs, states), (taped_outputs, taped_states), calls = _inference_and_taped(
            monkeypatch, execute
        )
    assert calls == [1, 2, 1] * 4
    # Recurrent caches advance on every step.
    for before, after in zip(states, states[1:]):
        assert all(not torch.equal(before[k][1], after[k][1]) for k in before)
    assert all(torch.equal(a, b) for a, b in zip(outputs, taped_outputs, strict=True))
    for a, b in zip(states, taped_states, strict=True):
        for key in a:
            assert all(torch.equal(av, bv) for av, bv in zip(a[key], b[key]))


@GPU
@pytest.mark.usefixtures("model_parallel")
@pytest.mark.parametrize("hidden,fp32_residual", [(256, True), (3072, True), (3072, False)])
def test_hybrid_lone_layer_write_fuses_into_block(monkeypatch, hidden, fp32_residual):
    """HybridModel M -> [M E] [M E]: the lone Mamba layer's write runs in the first block's
    two-reader kernel (a single update read twice) during prefill and decode, bit for bit with
    the taped forward."""
    # Without fp32_residual_connection the stream is BF16 (FP32 connection logits).
    model = _hybrid_model("mamba", False, hidden, False, "MMEME", fp32_residual)
    lone, first = model.decoder.layers[0], model.decoder.layers[1]
    assert lone.defers_residual_write and isinstance(first, ShortcutMoEBlock)
    execute = _static_prefill_decode(model, keys=(1, 2, 4))
    launch, dtypes = forward_kernel.forward, set()

    def record(residual, *args, **kwargs):
        dtypes.add(residual.dtype)
        return launch(residual, *args, **kwargs)

    monkeypatch.setattr(forward_kernel, "forward", record)
    with torch.inference_mode(), InferenceMode.active():
        (outputs, states), (taped_outputs, taped_states), calls = _inference_and_taped(
            monkeypatch, execute
        )
    assert dtypes == {torch.float32 if fp32_residual else torch.bfloat16}
    # Per step: block 1's two-reader write of the lone update, its one-reader, then block 2's
    # shortcut add (two updates) and one-reader.
    assert calls == [1, 1, 2, 1] * 4
    assert all(torch.equal(a, b) for a, b in zip(outputs, taped_outputs, strict=True))
    for a, b in zip(states, taped_states, strict=True):
        for key in a:
            assert all(torch.equal(av, bv) for av, bv in zip(a[key], b[key]))


def _unchain(model):
    """Turn off the lone-layer seams (each writes, then reads and norms unfused); writes into a
    fused block stay deferred."""
    layers = model.decoder.layers
    for layer, successor in zip(layers, layers[1:]):
        if not isinstance(successor, ShortcutMoEBlock):
            layer.defers_residual_write = False


# Nano-like stacks: M = lone Mamba, - = MLP-only layer, ME = fused shortcut block.
LONE_SEAM_STACKS = [
    # M->-, -->M, M->[ME], [ME]->[ME], [ME]->M, M->-, -->M, M->-
    ("M-MMEMEM-M-", "MTMBBMTMT", (1, 3, 4, 6, 8, 10), [1, 1, 1, 1, 2, 1, 2, 1, 1, 1]),
    # M->-, -->[ME], [ME]->M, M->[ME], [ME]->M, M->-, -->M, M->-
    ("M-MEMMEM-M-", "MTBMBMTMT", (1, 3, 5, 6, 8, 10), [1, 1, 1, 2, 1, 1, 2, 1, 1, 1]),
]


def _kinds(layers):
    return "".join(
        (
            "B"
            if isinstance(l, ShortcutMoEBlock)
            else "M" if isinstance(l, WideResidualMambaLayer) else "T"
        )
        for l in layers
    )


@GPU
@pytest.mark.usefixtures("model_parallel")
@pytest.mark.parametrize("fp32_residual", [True, False])
@pytest.mark.parametrize("pattern,kinds,keys,updates_per_launch", LONE_SEAM_STACKS)
def test_hybrid_lone_seams_fuse(
    monkeypatch, fp32_residual, pattern, kinds, keys, updates_per_launch
):
    """Every residual seam of a HybridModel runs one fused kernel.

    lone -> lone, lone -> block, block -> block and block -> lone seams apply the predecessor's
    deferred write with the successor's read(s) and RMSNorm(s) (a block's write is a shortcut
    add of two updates), bit for bit with the taped forward and within BF16 noise of the model
    with the lone seams unfused. ``updates_per_launch`` lists each launch's update count.
    """
    model = _hybrid_model("mamba", False, 3072, False, pattern, fp32_residual)
    layers = model.decoder.layers
    assert _kinds(layers) == kinds
    # Every layer but the last hands its write to the next layer's kernel.
    assert [l.defers_residual_write for l in layers] == [True] * (len(layers) - 1) + [False]
    execute = _static_prefill_decode(model, keys=keys)
    with torch.inference_mode(), InferenceMode.active():
        (outputs, states), (taped_outputs, taped_states), calls = _inference_and_taped(
            monkeypatch, execute
        )
    assert calls == updates_per_launch * 4
    assert all(torch.equal(a, b) for a, b in zip(outputs, taped_outputs, strict=True))
    for a, b in zip(states, taped_states, strict=True):
        for key in a:
            assert all(torch.equal(av, bv) for av, bv in zip(a[key], b[key]))
    _unchain(model)
    with torch.inference_mode(), InferenceMode.active():
        unchained, _ = execute()
    for a, b in zip(outputs, unchained, strict=True):
        # The fused reader folds the unrounded residual and norms it in FP32; the unchained read
        # rounds to BF16 first. Over eleven layers this is a few BF16 ulps of the largest logit
        # (|logits| <= ~3.5), the same order as the shortcut-block fusion vs no fusion.
        torch.testing.assert_close(a.float(), b.float(), rtol=0, atol=0.1)


@GPU
@pytest.mark.usefixtures("model_parallel")
@pytest.mark.parametrize("fp32_residual", [True, False])
def test_hybrid_lone_seams_fuse_training(fp32_residual):
    """Training through the chained lone seams (forward and backward, 1024 tokens) matches the
    unchained model: outputs and every parameter gradient within BF16 noise."""
    model = _hybrid_model("mamba", False, 3072, False, "M-MMEMEM-M-", fp32_residual).train()
    ids = torch.randint(0, 128, (1, 1024), device="cuda")
    positions = torch.arange(1024, device="cuda")[None]

    def step():
        model.zero_grad(set_to_none=True)
        logits = model(ids, positions, None)
        logits.float().square().mean().backward()
        grads = {
            n: p.grad.detach().float().clone()
            for n, p in model.named_parameters()
            if p.grad is not None
        }
        return logits.detach().float(), grads

    layers = model.decoder.layers
    assert all(l.defers_residual_write for l in layers[:-1])
    chained, chained_grads = step()
    _unchain(model)
    # Only writes into a fused block stay deferred.
    assert [l.defers_residual_write for l in layers[:-1]] == [
        isinstance(successor, ShortcutMoEBlock) for successor in layers[1:]
    ]
    unchained, unchained_grads = step()
    torch.testing.assert_close(chained, unchained, rtol=0, atol=0.1)
    assert chained_grads.keys() == unchained_grads.keys() and len(chained_grads) > 50
    # Controller logit gradients nearly cancel at init (the shortcut-block fusion alone moves
    # them several-fold); the kernel tests check them against FP64. For every other parameter
    # the error is bounded by the model's own sensitivity: a half-ulp BF16 perturbation of the
    # embedding output moves these gradients by a median ~4% and the Mamba dt_bias/A_log
    # reductions by up to ~15%.
    errors = {}
    for name, a in chained_grads.items():
        b = unchained_grads[name]
        assert torch.isfinite(a).all(), name
        if not name.endswith(("map.logit", "retention_logit")):
            errors[name] = ((a - b).norm() / b.norm().clamp_min(1e-20)).item()
    assert sorted(errors.values())[len(errors) // 2] < 0.05, errors
    worst = max(errors, key=errors.get)
    assert errors[worst] < 0.3, (worst, errors[worst])


@GPU
@pytest.mark.usefixtures("model_parallel")
@pytest.mark.parametrize("interval", [2, 4])
@pytest.mark.parametrize("pattern", [stack[0] for stack in LONE_SEAM_STACKS])
def test_hybrid_lone_seams_checkpoint_replay(monkeypatch, pattern, interval):
    """Checkpoint replay intervals rebuild the chained seams' residuals bit for bit: outputs and
    every gradient equal training without replay."""
    model = _hybrid_model("mamba", False, 3072, False, pattern, False).train()
    assert all(l.defers_residual_write for l in model.decoder.layers[:-1])
    replays = []
    launch = replay_kernel.replay_interval
    monkeypatch.setattr(
        replay_kernel, "replay_interval", lambda *a, **k: replays.append(1) or launch(*a, **k)
    )
    ids = torch.randint(0, 128, (1, 1024), device="cuda")
    positions = torch.arange(1024, device="cuda")[None]

    def step(num_layers):
        model.config.residual_stream_recompute_num_layers = num_layers
        model.zero_grad(set_to_none=True)
        logits = model(ids, positions, None)
        logits.float().square().mean().backward()
        return logits.detach(), {n: p.grad.clone() for n, p in model.named_parameters()}

    expected, expected_grads = step(None)
    assert not replays
    actual, actual_grads = step(interval)
    assert replays, "checkpoint replay did not run"
    assert torch.equal(actual, expected)
    for name, grad in actual_grads.items():
        assert bytes_equal(grad, expected_grads[name]), name
