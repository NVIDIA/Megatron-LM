# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compose the n-gram memory into hybrid stack specs and run it through the hybrid layers.

Everything here runs on CPU. The composition tests copy the hybrid stack specs, which hold
Transformer Engine modules; the integration tests replace the expensive layers with stand-ins
around the real TransformerLayer, HybridStack, HybridModel and hyper-connection wrapper paths.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core import tensor_parallel
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.models.engram import apply_engram_to_hybrid_stack_spec
from megatron.core.models.engram.config import EngramConfig
from megatron.core.models.engram.engram import Engram
from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols, validate_segment_layers
from megatron.core.models.hybrid.hybrid_layer_specs import (
    hybrid_stack_spec,
    wide_residual_hybrid_stack_spec,
)
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.hybrid.layers.hybrid_hyper_connection import HyperConnectionHybridLayer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.wide_residual_config import WideResidualConfig

from ._test_utils import packed_seq_params as tagged_packed_seq_params
from ._test_utils import write_tokenizer_map

# The hybrid stack specs hold Transformer Engine modules, which the composition copies.
_requires_te = pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine is not available")


def _engram_config(tmp_path, layer_ids=(2,), variant="deepseek"):
    artifact = write_tokenizer_map(tmp_path / "map.json", vocab_size=16, layer_ids=layer_ids)
    return EngramConfig(
        global_vocab_sizes=(17, 19),
        layer_ids=layer_ids,
        max_ngram_order=3,
        num_hash_heads=2,
        memory_dim=8,
        kernel_size=4,
        hash_seed=0,
        boundary_token_id=0,
        tokenizer_map_path=str(artifact),
        variant=variant,
    )


def _apply(tmp_path, pattern, layer_ids=(2,), enable_mhc_connections=False):
    transformer_config = SimpleNamespace(enable_mhc_connections=enable_mhc_connections)
    return apply_engram_to_hybrid_stack_spec(
        hybrid_stack_spec, _engram_config(tmp_path, layer_ids), pattern, transformer_config
    )


def _transformer_config(**overrides):
    # The local backend agrees with the NVTE attention env vars that conftest sets for every test.
    values = dict(
        num_layers=4,
        hidden_size=8,
        num_attention_heads=2,
        use_cpu_initialization=True,
        attention_backend=AttnBackend.local,
    )
    values.update(overrides)
    return TransformerConfig(**values)


@_requires_te
def test_engram_is_attached_to_the_selected_layer_type_only(tmp_path):
    # "M*M-": layer 2 is attention, layer 4 is MLP.
    spec = _apply(tmp_path, "M*M-", layer_ids=(2,))
    assert spec.submodules.attention_layer.submodules.engram.module is Engram
    # Mamba layers have no composition point at all, and untouched types keep theirs unset.
    assert not hasattr(spec.submodules.mamba_layer.submodules, "engram")
    assert (
        spec.submodules.mlp_layer.submodules.engram
        is not spec.submodules.attention_layer.submodules.engram
    )

    spec = _apply(tmp_path, "M*M-", layer_ids=(2, 4))
    assert spec.submodules.attention_layer.submodules.engram.module is Engram
    assert spec.submodules.mlp_layer.submodules.engram.module is Engram


@_requires_te
def test_shared_stack_spec_singleton_is_not_mutated(tmp_path):
    before = hybrid_stack_spec.submodules.attention_layer.submodules.engram
    _apply(tmp_path, "M*M*", layer_ids=(2,))
    # Attaching in place would carry a stale EngramConfig into every later model in the process.
    assert hybrid_stack_spec.submodules.attention_layer.submodules.engram is before


@_requires_te
def test_layer_id_on_a_type_that_cannot_carry_the_memory_is_rejected(tmp_path):
    # Mamba exposes no composition point.
    with pytest.raises(ValueError, match="assigns to a 'M' layer"):
        _apply(tmp_path, "M*M*", layer_ids=(1,))


@_requires_te
@pytest.mark.parametrize(
    "symbol, field_names",
    [
        (Symbols.ATTENTION, ("attention_layer",)),
        (Symbols.GDN, ("gdn_layer", "gdn2_layer")),
        (Symbols.KDA, ("kda_layer",)),
        (Symbols.DS_ATTENTION, ("dsa_layer",)),
        (Symbols.MLA, ("mla_layer", "mla_fused_down_proj_layer")),
        (Symbols.CSA, ("csa_layer", "csa_qk_layernorm_layer")),
        # The stack spec has no dedicated H or W layer spec, so HybridStack builds them
        # from the C fields.
        (Symbols.HCA, ("csa_layer", "csa_qk_layernorm_layer")),
        (Symbols.WINDOW, ("csa_layer", "csa_qk_layernorm_layer")),
        (Symbols.MLP, ("mlp_layer",)),
        (Symbols.MOE, ("moe_layer",)),
    ],
)
def test_every_field_the_stack_builds_a_symbol_from_carries_the_memory(
    tmp_path, symbol, field_names
):
    # HybridStack picks among these fields from the layer config (gdn2, MLA down-projection
    # fusion, CSA qk_layernorm), so the memory has to follow whichever one it builds.
    spec = _apply(tmp_path, Symbols.MAMBA + symbol, layer_ids=(2,))
    for field_name in field_names:
        assert getattr(spec.submodules, field_name).submodules.engram.module is Engram


@_requires_te
def test_pipeline_separators_and_mtp_do_not_shift_layer_numbering(tmp_path):
    # Pipe symbols are stage boundaries, not layers: layer 3 is still Mamba.
    with pytest.raises(ValueError, match="assigns to a 'M' layer"):
        _apply(tmp_path, "M*|M*", layer_ids=(3,))
    _apply(tmp_path, "M*|M*", layer_ids=(4,))

    with pytest.raises(ValueError, match="describes only 4 layers"):
        _apply(tmp_path, "M*M*", layer_ids=(5,))
    with pytest.raises(ValueError, match="requires --hybrid-layer-pattern"):
        _apply(tmp_path, None, layer_ids=(2,))


@_requires_te
def test_multi_token_prediction_is_accepted(tmp_path):
    # The nested MTP stack is built from these same submodules; HybridStack passes
    # is_mtp_layer for every layer type, and TransformerLayer skips the memory on MTP layers,
    # so the decoder layer IDs keep their meaning and no second memory is built.
    spec = _apply(tmp_path, "M*M*/M-", layer_ids=(2,))
    assert spec.submodules.attention_layer.submodules.engram.module is Engram
    assert spec.submodules.mlp_layer.submodules.engram is IdentityOp


@_requires_te
def test_hyper_connections_are_accepted(tmp_path):
    # HyperConnectionHybridLayer adds the memory to the n-stream tensor before its read gate and
    # tells the wrapped TransformerLayer to skip its own injection, so composition is the same
    # with hyper connections on (test_mhc_wrapper_adds_the_memory_to_the_n_stream_once).
    spec = _apply(tmp_path, "M*M*", layer_ids=(2,), enable_mhc_connections=True)
    assert spec.submodules.attention_layer.submodules.engram.module is Engram
    assert not hasattr(spec.submodules.mamba_layer.submodules, "engram")


@_requires_te
def test_wide_residual_is_rejected_at_startup(tmp_path):
    # The wide-residual stack spec exposes the same composition point, so attaching succeeds;
    # the startup validation that EngramConfig.from_args runs is what rejects the model.
    engram_config = _engram_config(tmp_path, layer_ids=(2,))
    config = _transformer_config(wide_residual=WideResidualConfig(num_streams=2))
    spec = apply_engram_to_hybrid_stack_spec(
        wide_residual_hybrid_stack_spec, engram_config, "M*M*", config
    )
    assert spec.submodules.attention_layer.submodules.engram.module is Engram
    with pytest.raises(ValueError, match="wide_residual"):
        engram_config.validate_startup(config, 16)


def test_shortcut_moe_is_rejected_at_startup(tmp_path):
    engram_config = _engram_config(tmp_path, layer_ids=(1,))
    config = _transformer_config(
        num_layers=2, num_moe_experts=2, moe_shortcut_connection=True, add_bias_linear=False
    )
    with pytest.raises(ValueError, match="moe_shortcut_connection"):
        engram_config.validate_startup(config, 16)


class _MemoryStub(torch.nn.Module):
    """Engram stand-in: a constant residual that records the token IDs it receives."""

    def __init__(self, value=1.0):
        super().__init__()
        self.value = value
        self.calls = []
        self.add_requests = []

    def forward(self, hidden_states, input_ids, packed_seq_params=None, add_to_hidden_states=False):
        self.calls.append(input_ids)
        self.add_requests.append(add_to_hidden_states)
        residual = torch.full_like(hidden_states, self.value)
        return hidden_states + residual if add_to_hidden_states else residual


class _SingleRankGroup:
    @staticmethod
    def size():
        return 1

    @staticmethod
    def rank():
        return 0


class _RecordingLayer(TransformerLayer):
    """TransformerLayer stand-in that records its call and may own an Engram memory."""

    def __init__(self, layer_number, engram=None, config=None):
        torch.nn.Module.__init__(self)
        self.layer_number = layer_number
        self.config = config if config is not None else SimpleNamespace(cuda_graph_impl="none")
        self.engram = engram
        self.calls = []

    def supports_two_stage_attention(self):
        return True

    def forward(self, hidden_states, **kwargs):
        self.calls.append(kwargs)
        return hidden_states, None


def _make_stack(layers, pattern, **config_overrides):
    """Use production stack/config initialization; replace only the expensive layers."""
    config = _transformer_config(num_layers=len(layers), **config_overrides)
    group = _SingleRankGroup()
    with mock.patch("megatron.core.models.hybrid.hybrid_block.build_module", side_effect=layers):
        return HybridStack(
            config=config,
            # build_module is replaced, so the specs only have to be configured.
            submodules=HybridStackSubmodules(
                attention_layer=IdentityOp, mlp_layer=IdentityOp, moe_layer=IdentityOp
            ),
            layer_config_list=validate_segment_layers(pattern, config),
            post_process=False,
            post_layer_norm=False,
            pg_collection=SimpleNamespace(pp=group, tp=group, cp=group, tp_cp=group),
        )


def _transformer_layer(submodules=None, layer_number=1, is_mtp_layer=False, **config_overrides):
    """A real TransformerLayer whose sublayers, other than the memory, are identities."""
    return TransformerLayer(
        _transformer_config(**config_overrides),
        submodules if submodules is not None else TransformerLayerSubmodules(),
        layer_number=layer_number,
        add_layer_offset=False,
        pg_collection=ProcessGroupCollection(),
        is_mtp_layer=is_mtp_layer,
    )


def test_transformer_layer_builds_the_memory_for_configured_decoder_layers_only(tmp_path):
    engram_spec = ModuleSpec(
        module=Engram, params={"engram_config": _engram_config(tmp_path, layer_ids=(2,))}
    )
    submodules = TransformerLayerSubmodules(engram=engram_spec)

    assert isinstance(_transformer_layer(submodules, layer_number=2).engram, Engram)
    assert _transformer_layer(submodules, layer_number=1).engram is None
    # MTP layers number their layers locally, so a decoder layer ID never selects one.
    assert _transformer_layer(submodules, layer_number=2, is_mtp_layer=True).engram is None
    assert _transformer_layer(layer_number=2).engram is None


def test_transformer_layer_adds_the_memory_before_the_input_layernorm(monkeypatch):
    layer = _transformer_layer()
    layer.engram = _MemoryStub(value=0.5)
    hidden_states = torch.randn(4, 2, 8)
    input_ids = torch.arange(8).view(2, 4)
    normed_inputs = []

    class _StopAfterLayerNorm(Exception):
        pass

    def capture_layernorm_input(hidden_states, **_kwargs):
        normed_inputs.append(hidden_states)
        raise _StopAfterLayerNorm

    monkeypatch.setattr(layer, "_run_input_layernorm", capture_layernorm_input)
    with pytest.raises(_StopAfterLayerNorm):
        layer._forward_attention(hidden_states, input_ids=input_ids)
    torch.testing.assert_close(normed_inputs[0], hidden_states + 0.5)
    assert layer.engram.calls == [input_ids]
    # The layer has the module add its memory, so the module chooses the precision of the add.
    assert layer.engram.add_requests == [True]

    # An outer hyper-connection wrapper that already applied the memory skips it here.
    with pytest.raises(_StopAfterLayerNorm):
        layer._forward_attention(hidden_states, input_ids=input_ids, skip_engram=True)
    assert normed_inputs[1] is hidden_states
    assert len(layer.engram.calls) == 1


@pytest.mark.parametrize("variant", ["deepseek", "deepseek_v41"])
def test_transformer_layer_returns_what_the_module_adds(tmp_path, variant):
    """The layer returns the module's hidden states, not its own sum.

    BF16 shows the difference for the deepseek_v41 fusion, which adds in FP32 and rounds once.
    """
    engram_spec = ModuleSpec(
        module=Engram,
        params={"engram_config": _engram_config(tmp_path, layer_ids=(2,), variant=variant)},
    )
    layer = _transformer_layer(
        TransformerLayerSubmodules(engram=engram_spec), layer_number=2, params_dtype=torch.bfloat16
    )
    generator = torch.Generator().manual_seed(5)
    input_ids = torch.randint(0, 16, (4, 64), generator=generator)
    hidden_states = torch.randn(64, 4, 8, generator=generator).to(torch.bfloat16)

    added = layer._maybe_apply_engram(hidden_states, input_ids)
    assert torch.equal(added, layer.engram(hidden_states, input_ids, add_to_hidden_states=True))
    rounded_first = hidden_states + layer.engram(hidden_states, input_ids)
    assert torch.count_nonzero(rounded_first - hidden_states) > 0
    assert torch.equal(added, rounded_first) == (variant == "deepseek")


def test_two_stage_attention_path_refuses_an_engram_layer(monkeypatch):
    layer = _transformer_layer()
    layer.engram = _MemoryStub()
    monkeypatch.setattr(layer, "supports_two_stage_attention", lambda: True)
    with pytest.raises(AssertionError, match="two-stage attention path"):
        layer.forward_pre_attn_and_core_attn(torch.randn(4, 2, 8))


@pytest.mark.parametrize("full_recompute", [False, True])
def test_hybrid_stack_passes_token_ids_to_engram_layers_only(monkeypatch, full_recompute):
    monkeypatch.setattr(
        tensor_parallel,
        "checkpoint",
        lambda function, _distribute_saved_activations, *args: function(*args),
    )
    engram_layer = _RecordingLayer(layer_number=1, engram=_MemoryStub())
    plain_layer = _RecordingLayer(layer_number=2)
    recompute = (
        dict(recompute_granularity="full", recompute_method="uniform", recompute_num_layers=2)
        if full_recompute
        else {}
    )
    stack = _make_stack([engram_layer, plain_layer], Symbols.ATTENTION * 2, **recompute)
    input_ids = torch.arange(8).view(2, 4)

    HybridStack.forward(
        stack, hidden_states=torch.randn(4, 2, 8), attention_mask=None, input_ids=input_ids
    )

    assert engram_layer.calls[0]["input_ids"] is input_ids
    # Layers without a memory or hash router keep their exact kwarg set.
    assert "input_ids" not in plain_layer.calls[0]


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_moe_unflatten_takes_the_local_slice_of_unscattered_engram_ids(monkeypatch, tp_rank):
    """An Engram-owning MoE layer under sequence parallelism gets the token IDs unscattered (the
    memory hashes the whole row); the per-sample unflatten must take this TP rank's slice."""
    tp_size, tokens_per_sample, micro_batch_size = 2, 4, 4
    full_ids = torch.arange(micro_batch_size * tokens_per_sample).view(1, -1)
    local_tokens = full_ids.shape[1] // tp_size
    hidden_states = torch.randn(local_tokens, 1, 3)
    layer = SimpleNamespace(
        is_moe_layer=True,
        config=SimpleNamespace(sequence_parallel=True),
        pg_collection=SimpleNamespace(tp=object()),
    )
    packed_seq_params = SimpleNamespace(tokens_per_sample=tokens_per_sample)
    local_ids = full_ids[:, tp_rank * local_tokens : (tp_rank + 1) * local_tokens]

    def scatter(tensor, group):  # the TP split along the first dimension, for this rank
        return tensor.chunk(tp_size, dim=0)[tp_rank]

    monkeypatch.setattr(tensor_parallel, "scatter_to_sequence_parallel_region", scatter)
    _, _, ids, mbs = TransformerLayer._maybe_unflatten_for_moe(
        layer, hidden_states, None, full_ids, packed_seq_params
    )
    assert mbs == local_tokens // tokens_per_sample
    torch.testing.assert_close(ids, local_ids.view(mbs, tokens_per_sample))

    # IDs that are already local (hash routing scatters them) are only reshaped.
    def unexpected(tensor, group):
        raise AssertionError("local token IDs must not be scattered again")

    monkeypatch.setattr(tensor_parallel, "scatter_to_sequence_parallel_region", unexpected)
    _, _, ids, _ = TransformerLayer._maybe_unflatten_for_moe(
        layer, hidden_states, None, local_ids, packed_seq_params
    )
    torch.testing.assert_close(ids, local_ids.view(mbs, tokens_per_sample))


def test_hybrid_stack_rejects_engram_inside_shortcut_pairs():
    shortcut = dict(num_moe_experts=2, moe_shortcut_connection=True, add_bias_linear=False)
    config = _transformer_config(num_layers=3, **shortcut)

    def build_stack(engram_layer_number):
        layers = [
            _RecordingLayer(
                layer_number,
                engram=_MemoryStub() if layer_number == engram_layer_number else None,
                config=config,
            )
            for layer_number in (1, 2, 3)
        ]
        # The shortcut block reads its TP group from the paired MoE layer's MLP.
        layers[1].mlp = SimpleNamespace(tp_group=None)
        # The pair-owned shortcut norms are Transformer Engine norms; they are irrelevant here.
        with mock.patch(
            "megatron.core.models.hybrid.shortcut_block.build_module",
            side_effect=lambda *_args, **_kwargs: torch.nn.Identity(),
        ):
            return _make_stack(
                layers, Symbols.ATTENTION + Symbols.MOE + Symbols.ATTENTION, **shortcut
            )

    # Both halves of the pair bypass TransformerLayer._forward_attention.
    for layer_number in (1, 2):
        with pytest.raises(ValueError, match=r"shortcut MoE pair \(layers 1 and 2\)"):
            build_stack(layer_number)
    # The trailing attention layer is not paired and runs the ordinary layer path.
    build_stack(3)


class _PassthroughHyperConnection(torch.nn.Module):
    """Identity mHC that records the n-stream tensor its read gate receives."""

    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, hidden_states, **_kwargs):
        self.inputs.append(hidden_states)
        return hidden_states, hidden_states, hidden_states, hidden_states

    def fused_h_res_h_post_bda(self, _h_res, residual, _h_post, layer_output_with_bias, **_kwargs):
        return residual + layer_output_with_bias[0]


class _RecordingMlpLayer(_RecordingLayer):
    """An MLP-only layer, which the wrapper runs through its raw-branch fast path."""

    def __init__(self, engram):
        super().__init__(layer_number=1, engram=engram)
        self.config = SimpleNamespace(bias_dropout_fusion=False, cuda_graph_impl="none")
        self.self_attention = IdentityOp()
        self.cross_attention = IdentityOp()
        self.mlp = torch.nn.Identity()
        self.mlp_norm_manager = None
        self.hidden_dropout = 0.0
        self.recompute_pre_mlp_layernorm = False
        self.mhc_checkpoint_pre_mlp_layernorm = False

    def _forward_mlp_output_with_bias(self, hidden_states, input_ids=None, **kwargs):
        self.calls.append(dict(hidden_states=hidden_states, input_ids=input_ids, **kwargs))
        return (torch.zeros_like(hidden_states), None), hidden_states


def _wrap_with_passthrough_mhc(inner_layer):
    wrapper = HyperConnectionHybridLayer.__new__(HyperConnectionHybridLayer)
    torch.nn.Module.__init__(wrapper)
    wrapper.config = SimpleNamespace(
        fp32_residual_connection=False, params_dtype=None, cuda_graph_impl="none"
    )
    wrapper.inner_layer = inner_layer
    wrapper.layer_number = inner_layer.layer_number
    wrapper.hyper_connection = _PassthroughHyperConnection()
    return wrapper


@pytest.mark.parametrize("fast_path", [False, True])
def test_mhc_wrapper_adds_the_memory_to_the_n_stream_once(fast_path):
    memory = _MemoryStub(value=0.25)
    if fast_path:
        inner_layer = _RecordingMlpLayer(engram=memory)
    else:
        # An inner layer with neither attention nor MLP takes the full-layer fallback path.
        inner_layer = _RecordingLayer(layer_number=1, engram=memory)
        inner_layer.self_attention = IdentityOp()
        inner_layer.cross_attention = IdentityOp()
        inner_layer.mlp = IdentityOp()
    wrapper = _wrap_with_passthrough_mhc(inner_layer)
    hidden_states = torch.randn(4, 2, 8)
    input_ids = torch.arange(8).view(2, 4)

    wrapper(hidden_states, attention_mask=None, input_ids=input_ids)

    # The memory joins the residual streams before the read gate, exactly once.
    torch.testing.assert_close(wrapper.hyper_connection.inputs[0], hidden_states + 0.25)
    assert memory.calls == [input_ids]
    assert memory.add_requests == [True]
    if not fast_path:
        assert inner_layer.calls[0]["skip_engram"] is True
        assert inner_layer.calls[0]["input_ids"] is input_ids


def test_mhc_wrapper_without_memory_keeps_the_inner_kwargs():
    inner_layer = _RecordingLayer(layer_number=1)
    inner_layer.self_attention = IdentityOp()
    inner_layer.cross_attention = IdentityOp()
    inner_layer.mlp = IdentityOp()
    wrapper = _wrap_with_passthrough_mhc(inner_layer)

    wrapper(torch.randn(4, 2, 8), attention_mask=None)

    assert "skip_engram" not in inner_layer.calls[0]


class _RecordingDecoder:
    def __init__(self):
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return kwargs["hidden_states"]


def _hybrid_model_stub(*, pre_process=True, sequence_parallel=False):
    return SimpleNamespace(
        config=SimpleNamespace(
            fine_grained_activation_offloading=False,
            moe_paged_stash=False,
            moe_num_hash_layers=0,
            sequence_parallel=sequence_parallel,
            freeze_base_model_for_mtp=False,
            engram_enabled=True,
        ),
        hybrid_layer_config_list=None,
        _has_custom_layers=False,
        decoder=_RecordingDecoder(),
        position_embedding_type="none",
        pre_process=pre_process,
        post_process=False,
        share_embeddings_and_output_weights=False,
        mtp_process=False,
        vocab_size=128,
        pg_collection=SimpleNamespace(tp=object()),
    )


@pytest.mark.parametrize("pre_process", [True, False])
def test_hybrid_model_passes_unscattered_ids_under_sequence_parallelism(monkeypatch, pre_process):
    def unexpected_scatter(tensor, group):
        pytest.fail("Engram token IDs must reach the decoder unscattered")

    monkeypatch.setattr(
        "megatron.core.models.hybrid.hybrid_model.tensor_parallel."
        "scatter_to_sequence_parallel_region",
        unexpected_scatter,
    )
    model = _hybrid_model_stub(pre_process=pre_process, sequence_parallel=True)
    input_ids = torch.arange(8).view(2, 4)
    # Sequence-parallel decoder activations hold half of the 4 positions on this rank.
    hidden_states = torch.randn(2, 2, 8)
    model.decoder.input_tensor = hidden_states

    HybridModel.forward(
        model,
        input_ids=input_ids,
        position_ids=torch.arange(4).repeat(2, 1),
        attention_mask=None,
        decoder_input=hidden_states if pre_process else None,
    )

    assert model.decoder.kwargs["input_ids"] is input_ids


def test_hybrid_model_rejects_unsupported_engram_inputs_on_every_stage():
    hidden_states = torch.randn(4, 2, 8)
    forward_kwargs = dict(
        position_ids=torch.arange(4).repeat(2, 1), attention_mask=None, decoder_input=hidden_states
    )
    with pytest.raises(ValueError, match="requires input token IDs on every pipeline stage"):
        HybridModel.forward(_hybrid_model_stub(), input_ids=None, **forward_kwargs)
    with pytest.raises(ValueError, match="does not support inference"):
        HybridModel.forward(
            _hybrid_model_stub(),
            input_ids=torch.arange(8).view(2, 4),
            inference_context=SimpleNamespace(),
            **forward_kwargs,
        )


def _hybrid_model(monkeypatch, config, **kwargs):
    """Construct a single-rank HybridModel that must be rejected before any submodule is built."""

    def fail_build(*_args, **_kwargs):
        pytest.fail("HybridModel submodule construction must not begin")

    monkeypatch.setattr("megatron.core.models.hybrid.hybrid_model.build_module", fail_build)
    # Per-stage pipeline logging needs a default process group, which these CPU tests lack.
    monkeypatch.setattr(
        "megatron.core.models.hybrid.hybrid_layer_allocation.log_on_each_pipeline_stage",
        lambda *_args, **_kwargs: None,
    )
    return HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=16,
        max_sequence_length=8,
        pg_collection=SimpleNamespace(tp=None, pp=None, cp=None, dp_cp=None, embd=None),
        **kwargs,
    )


def test_hybrid_model_rejects_engram_with_a_layer_config_list(monkeypatch):
    config = _transformer_config(num_layers=2, engram_enabled=True)
    with pytest.raises(ValueError, match="hybrid_layer_config_list"):
        _hybrid_model(
            monkeypatch,
            config,
            hybrid_layer_config_list=validate_segment_layers(Symbols.ATTENTION * 2, config),
        )


def test_hybrid_model_rejects_engram_with_hash_routing_under_sequence_parallelism(monkeypatch):
    config = _transformer_config(
        num_layers=2,
        tensor_model_parallel_size=2,
        sequence_parallel=True,
        num_moe_experts=2,
        moe_num_hash_layers=1,
        hash_moe_vocab_size=32,
        add_bias_linear=False,
        engram_enabled=True,
    )
    with pytest.raises(ValueError, match="hash-routed MoE layers"):
        _hybrid_model(monkeypatch, config, hybrid_layer_pattern=Symbols.ATTENTION + Symbols.MOE)


def test_hybrid_model_rejects_engram_with_attention_residuals(monkeypatch):
    config = _transformer_config(num_layers=2, engram_enabled=True)
    # Set after construction: this is the model's own check, for a config that did not go
    # through the Engram startup validation.
    config.enable_attention_residuals = True
    with pytest.raises(ValueError, match="enable_attention_residuals"):
        _hybrid_model(monkeypatch, config, hybrid_layer_pattern=Symbols.ATTENTION * 2)
