# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core import recompute as recompute_module
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_model import HybridModel
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
        'recompute_granularity': None,
        'recompute_method': None,
        'recompute_num_layers': None,
        'distribute_saved_activations': False,
    }
    config.update(config_overrides)
    return SimpleNamespace(
        config=SimpleNamespace(**config),
        pre_process=True,
        post_process=False,
        post_layer_norm=False,
        input_tensor=None,
        layers=layers,
        num_layers_per_pipeline_rank=len(layers),
        training=True,
    )


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
