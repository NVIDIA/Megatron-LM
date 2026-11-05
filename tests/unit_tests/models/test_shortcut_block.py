# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import transformer_engine as te

from megatron.core.fusions.wide_residual_shortcut import (
    PendingShortcutAdd,
    forward_kernel,
    kernel_utils,
    mxfp8_handoff,
    replay_kernel,
)
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.hybrid import hybrid_layer_specs as specs
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols as LayerSymbols
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.hybrid.shortcut_block import (
    ShortcutMoEBlock,
    group_layers_into_shortcut_blocks,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import wide_residual
from megatron.core.transformer.module import TwoStageAttentionLayer
from megatron.core.transformer.residual_recompute import build_residual_stream_recompute_plan
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.transformer.wide_residual import (
    StreamwiseSigmoidWideResidualConnection,
    apply_pending_write,
    is_fusable_rmsnorm,
)
from megatron.core.transformer.wide_residual_config import WideResidualConfig
from tests.unit_tests.determinism.kernels.harness import bytes_equal
from tests.unit_tests.test_utilities import Utils

# The shortcut-owned norms are Transformer Engine norms, which need a full TransformerConfig to
# build and a GPU to run.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="shortcut norms are Transformer Engine norms"
)


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

    def route(shortcut_hidden, padding_mask=None, packed_seq_params=None, recompute_context=None):
        assert recompute_context is attn_context
        observed["route_packed"] = packed_seq_params
        assert padding_mask is moe_padding_mask
        return shortcut_hidden, shortcut_hidden

    def shared(hidden_states, padding_mask=None, packed_seq_params=None, recompute_context=None):
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

    def route(shortcut_hidden, padding_mask=None, packed_seq_params=None, recompute_context=None):
        observed["route_input"] = shortcut_hidden
        return shortcut_hidden, shortcut_hidden

    def shared(hidden_states, padding_mask=None, packed_seq_params=None, recompute_context=None):
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
            self, shortcut_hidden, padding_mask=None, packed_seq_params=None, recompute_context=None
        ):
            assert recompute_context is None
            assert_quant_layer(1)
            return shortcut_hidden * self.route_scale, shortcut_hidden * self.prob_scale

        def shortcut_shared_experts(
            self, hidden_states, padding_mask=None, packed_seq_params=None, recompute_context=None
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


# ------------------------------------------- fused residual seams (moe_shortcut_wide_residual_norm_fusion)

REQUIRES_FUSED_BACKWARD = pytest.mark.skipif(
    not kernel_utils.qualified_device(torch.cuda.get_device_properties(0)),
    reason="the fused shortcut backward runs only on a qualified GB300",
)


def _fused_config(**overrides):
    options = dict(
        normalization="RMSNorm",
        params_dtype=torch.bfloat16,
        bf16=True,
        fp32_residual_connection=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        add_bias_linear=False,
        moe_shortcut_connection=True,
        moe_shortcut_wide_residual_norm_fusion=True,
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=2,
        moe_shared_expert_intermediate_size=128,
        wide_residual=WideResidualConfig(num_streams=3, learned_retention=True),
    )
    return TransformerConfig(**(options | overrides))


@pytest.fixture
def model_parallel():
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    yield
    Utils.destroy_model_parallel()


def _hybrid_model(pattern, mixer="mamba", hidden=3072, optimized=False, **overrides):
    """A small fused HybridModel: M Mamba (GDP with mixer="gdp"), G GDN, * attention, E MoE,
    - MLP-only."""
    config = _fused_config(
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
        moe_router_topk=2,
        moe_router_pre_softmax=True,
        moe_router_dtype="fp32",
        inference_grouped_gemm_backend="torch",
        moe_shared_expert_intermediate_size=256,
        moe_grouped_gemm=True,
        gated_linear_unit=True,
        activation_func=F.silu,
        moe_shortcut_parallel=True,
        transformer_impl="inference_optimized" if optimized else "transformer_engine",
        **overrides,
    )
    family = "gated_delta_product" if mixer == "gdp" else "hybrid"
    spec = getattr(specs, f"wide_residual_{family}{'_inference' * optimized}_stack_spec")
    if config.moe_shortcut_wide_residual_norm_fusion:
        spec = specs.shortcut_fusion_stack_spec(spec)
    model = HybridModel(
        config, spec, vocab_size=128, max_sequence_length=128, hybrid_layer_pattern=pattern
    )
    blocks = [b for b in model.decoder.layers if isinstance(b, ShortcutMoEBlock)]
    fused = config.moe_shortcut_wide_residual_norm_fusion
    assert len(blocks) == pattern.count("E") and all(
        b.fuses_residual_seams == fused for b in blocks
    )
    return model.cuda()


def _unfuse(model):
    """Run every seam unfused (ordinary reads, norms and writes) with the same weights."""
    for module in model.modules():
        if hasattr(module, "defers_residual_write"):
            module.defers_residual_write = False
        if isinstance(module, ShortcutMoEBlock):
            module.fuses_residual_seams = False


def _static_prefill_decode(model):
    ids = torch.randint(0, 128, (2, 99), device="cuda")
    positions = torch.arange(99, device="cuda").expand(2, -1)
    context = StaticInferenceContext(2, 128)
    outputs = []
    for start, end in [(0, 96), (96, 97), (97, 98), (98, 99)]:
        context.sequence_len_offset = start
        if start:
            context.enable_decode_mode()
        outputs.append(
            model(
                ids[:, start:end],
                positions[:, start:end],
                None,
                inference_context=context,
                runtime_gather_output=True,
            ).clone()
        )
    return outputs


def _dynamic_prefill_decode(model):
    from megatron.core.inference.config import InferenceConfig, MambaInferenceStateConfig
    from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
    from megatron.core.inference.inference_request import DynamicInferenceRequest
    from megatron.core.inference.sampling_params import SamplingParams

    torch.manual_seed(0)
    prompts = [torch.randint(0, 128, (length,), device="cuda") for length in (64, 96)]
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

    def request(index, prompt, tokens):
        sampling = SamplingParams(num_tokens_to_generate=tokens)
        request = DynamicInferenceRequest(
            request_id=index, prompt_tokens=prompt, sampling_params=sampling
        )
        context.add_request(request)

    for index, prompt in enumerate(prompts):
        request(index, prompt, 4)
    outputs = []
    for step in range(4):
        context.initialize_attention_state()
        ids, positions = context.current_input_and_position_ids()
        outputs.append(
            model(ids, positions, None, inference_context=context, runtime_gather_output=True)
        )
        active = context.get_active_request_count()
        if step < 3:
            tokens = torch.tensor([17, 23, 29])[:active]
            context.update_requests(torch.ones(active, dtype=torch.bool), tokens)
        if step == 0:  # decode and prefill requests in one step
            request(2, prompts[0][:32], 3)
    return outputs


# (pattern, mixer, launches per step): a fused block's in-block seam writes one update; a seam
# into a block applies the previous write with both block inputs (two updates after a block).
INFERENCE_CASES = [
    ("MEME", "mamba", [1, 2, 1]),
    # A first MLP-only layer reads unfused and defers its write into the first block.
    ("-MEME", "mamba", [1, 1, 2, 1]),
    ("MEME", "gdp", [1, 2, 1]),
    ("MMEME", "mamba", [1, 1, 2, 1]),
    ("M-MMEMEM-M-", "mamba", [1, 1, 1, 1, 2, 1, 2, 1, 1, 1]),
    ("M-MEMMEM-M-", "mamba", [1, 1, 1, 2, 1, 1, 2, 1, 1, 1]),
    # A lone attention layer keeps its seams: M -> * and * -> [ME] stay unfused.
    ("M*MEME", "mamba", [1, 2, 1]),
    # An attention compute layer inside a block; the lone Mamba write enters its two-reader seam.
    ("M*EME", "mamba", [1, 1, 2, 1]),
]


@pytest.mark.usefixtures("model_parallel")
@pytest.mark.parametrize(
    "pattern,mixer,launches,dynamic,optimized,hidden,fp32_residual",
    [(*case, False, False, 3072, True) for case in INFERENCE_CASES]
    + [("MEME", "mamba", [1, 2, 1], False, False, 256, True)]
    + [("M-MMEMEM-M-", "mamba", INFERENCE_CASES[4][2], False, False, 3072, False)]
    + [
        (p + "*", mixer, [1, 2, 1], True, optimized, 3072, True)
        for p, mixer in (("MEME", "mamba"), ("MEME", "gdp"), ("GEGE", "gdn"))
        for optimized in (False, True)
    ],
)
def test_fused_seams_inference_matches_unfused(
    monkeypatch, pattern, mixer, launches, dynamic, optimized, hidden, fp32_residual
):
    """Prefill/decode (static or dynamic batching) runs one fused kernel per linked seam and
    matches the same model with every seam unfused within BF16 noise."""
    model = _hybrid_model(
        pattern, mixer, hidden, optimized, fp32_residual_connection=fp32_residual
    ).eval()
    execute = _dynamic_prefill_decode if dynamic else _static_prefill_decode
    launch, calls = forward_kernel.forward, []  # each fused launch's update count
    monkeypatch.setattr(
        forward_kernel,
        "forward",
        lambda r, u, *a, **k: calls.append(len(u)) or launch(r, u, *a, **k),
    )
    with torch.inference_mode(), InferenceMode.active():
        torch.manual_seed(1)
        fused = execute(model)
        assert calls == launches * 4
        _unfuse(model)
        torch.manual_seed(1)
        unfused = execute(model)
    assert len(calls) == len(launches) * 4  # the unfused model launches no fused kernel
    for a, b in zip(fused, unfused, strict=True):
        # The fused reader norms the unrounded FP32 mix; the unfused read rounds to BF16 first.
        torch.testing.assert_close(a.float(), b.float(), rtol=0, atol=0.1)


@pytest.mark.usefixtures("model_parallel")
@pytest.mark.parametrize(
    "pattern,mixer,fp32_residual,interval",
    [
        ("M-MMEMEM-M-", "mamba", True, 2),
        ("M-MEMMEM-M-", "mamba", False, 4),
        ("MEME", "gdp", True, 2),
        ("GE*E", "gdn", True, 3),
    ],
)
@REQUIRES_FUSED_BACKWARD
def test_fused_seams_training_matches_unfused(monkeypatch, pattern, mixer, fp32_residual, interval):
    """Training forward/backward through every fused seam matches the unfused model: outputs and
    parameter gradients within BF16 noise. The kernels apply each norm's epsilon and run its
    forward pre-hooks (DDP's overlapped parameter all-gather waits there), and checkpoint replay
    every ``interval`` layers reproduces outputs and gradients bit for bit."""
    model = _hybrid_model(
        pattern, mixer, fp32_residual_connection=fp32_residual, layernorm_epsilon=1e-6
    ).train()
    launch, epsilons = forward_kernel.forward, set()
    monkeypatch.setattr(
        forward_kernel, "forward", lambda *a, **k: epsilons.add(k["epsilon"]) or launch(*a, **k)
    )
    replays, replay = [], replay_kernel.replay_interval
    monkeypatch.setattr(
        replay_kernel, "replay_interval", lambda *a, **k: replays.append(1) or replay(*a, **k)
    )
    norms = {n: m for n, m in model.decoder.named_modules() if is_fusable_rmsnorm(m)}
    hooked = set()
    for name, norm in norms.items():
        norm.register_forward_pre_hook(lambda m, args, name=name: hooked.add(name))
    ids = torch.randint(0, 128, (1, 1024), device="cuda")
    positions = torch.arange(1024, device="cuda")[None]

    def step(replay_interval=None):
        model.config.residual_stream_recompute_num_layers = replay_interval
        model.zero_grad(set_to_none=True)
        logits = model(ids, positions, None)
        logits.float().square().mean().backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        return logits.detach(), grads

    fused = step()
    assert epsilons == {1e-6} and hooked == set(norms) and not replays
    replayed = step(interval)
    assert replays and torch.equal(replayed[0], fused[0])
    assert replayed[1].keys() == fused[1].keys()
    assert all(bytes_equal(g, fused[1][n]) for n, g in replayed[1].items())
    _unfuse(model)
    unfused = step()
    hook = model.embedding.register_forward_hook(lambda module, args, out: out * (1 + 2**-7))
    perturbed = step()
    hook.remove()

    def deviation(run):
        """Max logit difference and median/max relative gradient error against unfused."""
        (logits, grads), (reference, reference_grads) = run, unfused
        assert grads.keys() == reference_grads.keys()
        assert all(torch.isfinite(g).all() for g in grads.values())
        errors = torch.stack(
            [
                (g.float() - reference_grads[name].float()).norm()
                / reference_grads[name].float().norm().clamp_min(1e-20)
                for name, g in grads.items()
                # Controller-logit gradients nearly cancel at init (kernel tests check them).
                if not name.endswith(("map.logit", "retention_logit"))
            ]
        )
        return (logits.float() - reference.float()).abs().max(), errors.median(), errors.max()

    # The fused seams move outputs and gradients no more than a one-ulp BF16 perturbation of the
    # embedding output does (the fused reader norms the unrounded FP32 mix).
    for actual, noise in zip(deviation(fused), deviation(perturbed)):
        assert actual <= 2 * noise, (actual, noise)


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"recompute_granularity": "selective", "recompute_modules": [m]}, f"{m}'] recompute")
        for m in ("layernorm", "shortcut_pre_mlp_layernorm", "residual_stream")
    ]
    + [
        ({"moe_shortcut_post_norm": True}, "moe_shortcut_post_norm"),
        ({"add_bias_linear": True}, "add_bias_linear"),
        ({"wide_residual": WideResidualConfig(num_streams=7)}, "num_streams"),
        ({"residual_stream_recompute_num_layers": 2, "cuda_graph_impl": "local"}, "CUDA graphs"),
    ],
)
def test_fusion_rejects_unsupported_config(overrides, message):
    with pytest.raises(ValueError, match=message):
        _fused_config(**overrides)


@pytest.mark.usefixtures("model_parallel")
def test_fusion_spec_keeps_sharded_checkpoint_keys():
    """The split input norms keep the unsplit stack's sharded checkpoint keys."""

    def keys(fused):
        model = _hybrid_model("M-G*EMEM-", hidden=256, moe_shortcut_wide_residual_norm_fusion=fused)
        return {
            value.key: tuple(value.local_shape)
            for value in model.sharded_state_dict().values()
            if hasattr(value, "local_shape")
        }

    assert keys(True) == keys(False)
    fused = specs.shortcut_fusion_stack_spec(specs.wide_residual_hybrid_stack_spec)
    assert fused is specs.wide_residual_shortcut_fusion_stack_spec
    assert specs.shortcut_fusion_stack_spec(fused) is fused  # fusion and custom specs unchanged


@pytest.mark.parametrize(
    "recompute,offload,gtp,eligible,expected",
    [
        (False, False, False, True, "linear"),
        (False, False, False, False, None),
        (True, True, False, True, None),
        (True, False, True, True, "linear"),
        (True, False, False, True, "GTP weight finalization"),
    ],
)
def test_direct_mxfp8_consumer_selection(monkeypatch, recompute, offload, gtp, eligible, expected):
    """Reader zero goes to its projection as MXFP8 only when eligible; a checkpointed (GDP)
    input projection stays BF16 under fine-grained offloading and needs GTP finalization."""
    seen = []
    monkeypatch.setattr(mxfp8_handoff, "eligible", lambda *args: eligible)
    monkeypatch.setattr(
        wide_residual, "_apply_boundary", lambda *a, **k: seen.append(k["mxfp8_consumer"])
    )
    linear = SimpleNamespace(weight=SimpleNamespace(is_gtp_weight_remat=gtp))
    branch = SimpleNamespace(
        recompute_in_proj=recompute,
        config=SimpleNamespace(fine_grained_activation_offloading=offload),
    )
    read = SimpleNamespace(num_streams=3, read_map=lambda return_logits: None)
    pending = PendingShortcutAdd(None, None, None, None, None)
    call = lambda: apply_pending_write(pending, (read,), (torch.nn.RMSNorm(8),), (branch, linear))
    if expected not in ("linear", None):
        with pytest.raises(ValueError, match=expected):
            call()
    else:
        call()
        assert seen == [linear if expected else None]
