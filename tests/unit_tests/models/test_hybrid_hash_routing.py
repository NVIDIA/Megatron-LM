# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core import recompute as recompute_module
from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols as LayerSymbols
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.transformer.moe import router as router_module
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionLayer,
    MultiTokenPredictionLayerSubmodules,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer


class RecordingTransformerLayer(TransformerLayer):
    """Minimal TransformerLayer stand-in that records its call signature."""

    def __init__(self, layer_number=1, cuda_graph_impl="none"):
        torch.nn.Module.__init__(self)
        self.layer_number = layer_number
        self.config = SimpleNamespace(cuda_graph_impl=cuda_graph_impl)
        self.calls = []
        self.used_cuda_graph = False
        if cuda_graph_impl == "local":
            self.cudagraph_manager = self._run_local_cuda_graph
        if cuda_graph_impl == "transformer_engine":
            self.cuda_graphs = [object()]

    def forward(self, hidden_states, **kwargs):
        self.calls.append(kwargs)
        return hidden_states, None

    def _te_cuda_graph_replay(self, *args, **kwargs):
        self.used_cuda_graph = True
        return self.forward(*args, **kwargs)

    def _run_local_cuda_graph(self, module, args, kwargs):
        assert module is self
        self.used_cuda_graph = True
        return self.forward(*args, **kwargs)


class RecordingNonTransformerLayer(torch.nn.Module):
    """Mamba-like layer whose signature deliberately excludes input_ids."""

    def __init__(self, layer_number=2):
        super().__init__()
        self.layer_number = layer_number
        self.num_calls = 0

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        inference_context=None,
        rotary_pos_emb=None,
        *,
        packed_seq_params=None,
    ):
        self.num_calls += 1
        return hidden_states


class RecordingDecoder:
    def __init__(self):
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return kwargs['hidden_states']


def make_stack(layers, **config_overrides):
    config = {
        'cuda_graph_impl': "none",
        'flash_decode': False,
        'fp8': False,
        'fp8_recipe': None,
        'fp4': None,
        'enable_hyper_connections': False,
        'recompute_granularity': None,
        'recompute_method': None,
        'recompute_num_layers': None,
        'distribute_saved_activations': False,
    }
    config.update(config_overrides)
    stack = SimpleNamespace(
        config=SimpleNamespace(**config),
        pre_process=True,
        post_process=False,
        post_layer_norm=False,
        input_tensor=None,
        layers=layers,
        num_layers_per_pipeline_rank=len(layers),
        training=True,
        _mhc_block_end_plan=None,
    )
    stack._build_mhc_recompute_layer_plan = lambda _enabled: (
        [None] * len(layers),
        [False] * len(layers),
    )
    stack._finalize_mhc_recompute_layer = lambda **_kwargs: None
    return stack


def run_stack(stack, input_ids):
    hidden_states = torch.randn(4, 2, 8, requires_grad=True)
    output = HybridStack.forward(
        stack, hidden_states=hidden_states, attention_mask=None, input_ids=input_ids
    )
    assert output.shape == hidden_states.shape
    return output


@pytest.mark.parametrize("recompute_granularity", [None, "selective"])
def test_hybrid_stack_forwards_input_ids_only_to_transformer_layers(recompute_granularity):
    transformer_layer = RecordingTransformerLayer()
    non_transformer_layer = RecordingNonTransformerLayer()
    stack = make_stack(
        [transformer_layer, non_transformer_layer], recompute_granularity=recompute_granularity
    )
    input_ids = torch.arange(8).reshape(2, 4)

    run_stack(stack, input_ids)

    assert transformer_layer.calls[0]['input_ids'] is input_ids
    assert non_transformer_layer.num_calls == 1


def test_hybrid_stack_omits_input_ids_keyword_when_not_provided():
    transformer_layer = RecordingTransformerLayer()
    stack = make_stack([transformer_layer])

    run_stack(stack, input_ids=None)

    assert 'input_ids' not in transformer_layer.calls[0]


def test_hybrid_stack_full_recompute_preserves_ids_and_non_transformer_signature(monkeypatch):
    monkeypatch.setattr(
        recompute_module.tensor_parallel,
        "checkpoint",
        lambda function, _distribute_saved_activations, *args: function(*args),
    )
    transformer_layer = RecordingTransformerLayer()
    non_transformer_layer = RecordingNonTransformerLayer()
    stack = make_stack(
        [transformer_layer, non_transformer_layer],
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=2,
    )
    input_ids = torch.arange(8).reshape(2, 4)

    run_stack(stack, input_ids)

    assert transformer_layer.calls[0]['input_ids'] is input_ids
    assert non_transformer_layer.num_calls == 1


@pytest.mark.parametrize("cuda_graph_impl", ["local", "transformer_engine"])
def test_hybrid_stack_preserves_hash_ids_in_cuda_graph_signature(cuda_graph_impl):
    transformer_layer = RecordingTransformerLayer(cuda_graph_impl=cuda_graph_impl)
    stack = make_stack([transformer_layer])
    input_ids = torch.arange(8).reshape(2, 4)

    run_stack(stack, input_ids)

    assert transformer_layer.used_cuda_graph
    assert transformer_layer.calls[0]['input_ids'] is input_ids


@pytest.mark.parametrize("moe_n_hash_layers,expects_input_ids", [(0, False), (1, True)])
def test_hybrid_model_passes_ids_to_decoder_only_for_hash_routing(
    moe_n_hash_layers, expects_input_ids
):
    decoder = RecordingDecoder()
    config = SimpleNamespace(
        fine_grained_activation_offloading=False,
        moe_paged_stash=False,
        moe_n_hash_layers=moe_n_hash_layers,
        actual_vocab_size=128,
    )
    model = SimpleNamespace(
        config=config,
        decoder=decoder,
        position_embedding_type='none',
        pre_process=True,
        post_process=False,
        share_embeddings_and_output_weights=False,
        mtp_process=False,
        vocab_size=128,
    )
    input_ids = torch.arange(8).reshape(2, 4)
    hidden_states = torch.randn(4, 2, 8)

    output = HybridModel.forward(
        model,
        input_ids=input_ids,
        position_ids=torch.arange(4).repeat(2, 1),
        attention_mask=None,
        decoder_input=hidden_states,
    )

    assert output is hidden_states
    assert config.actual_vocab_size == 128
    assert ('input_ids' in decoder.kwargs) is expects_input_ids
    if expects_input_ids:
        assert decoder.kwargs['input_ids'] is input_ids


def test_hybrid_hash_moe_pp_does_not_require_explicit_pipeline_layout():
    config = TransformerConfig(
        num_layers=4,
        hidden_size=64,
        num_attention_heads=4,
        use_cpu_initialization=True,
        pipeline_model_parallel_size=2,
        num_moe_experts=4,
        moe_n_hash_layers=3,
        actual_vocab_size=128,
        is_hybrid_model=True,
    )

    assert config.pipeline_model_parallel_layout is None

    with pytest.raises(
        AssertionError,
        match="pipeline_model_parallel_layout must be set",
    ):
        TransformerConfig(
            num_layers=4,
            hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=True,
            pipeline_model_parallel_size=2,
            num_moe_experts=4,
            moe_n_hash_layers=3,
            actual_vocab_size=128,
            is_hybrid_model=False,
        )


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
        return _MtpMoEStub(
            layer_number=kwargs["layer_number"],
            is_mtp_layer=kwargs["is_mtp_layer"],
        )

    monkeypatch.setattr(hybrid_block_module, "build_module", fake_build_module)
    config = SimpleNamespace(
        fp8=False,
        fp4=None,
        enable_hyper_connections=False,
        cuda_graph_impl="none",
    )

    stack = HybridStack(
        config=config,
        submodules=HybridStackSubmodules(moe_layer=object()),
        layer_type_list=[LayerSymbols.MOE],
        post_process=False,
        pg_collection=SimpleNamespace(pp=object(), tp=object()),
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

    monkeypatch.setattr(
        hybrid_block_module, "HybridStack", _RecordingHybridStack
    )
    monkeypatch.setattr(
        allocation_module,
        "validate_segment_layers",
        lambda _pattern: [LayerSymbols.MOE],
    )
    monkeypatch.setattr(
        mtp_module,
        "build_module",
        lambda *_args, **_kwargs: torch.nn.Identity(),
    )

    config = SimpleNamespace(
        enable_hyper_connections=False,
        sequence_parallel=False,
        pipeline_model_parallel_size=1,
        pipeline_model_parallel_layout=None,
        hidden_size=8,
        layernorm_epsilon=1e-5,
        init_method=lambda tensor: tensor,
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
        pg_collection=SimpleNamespace(cp=None, tp=None),
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
    router.config = SimpleNamespace(
        mtp_num_layers=1,
        mtp_use_repeated_layer=False,
        num_layers=86,
    )
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
