# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core import recompute as recompute_module
from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols as LayerSymbols
from megatron.core.models.hybrid.hybrid_model import (
    HybridModel,
    _get_hash_moe_layer_threshold,
    _validate_hash_moe_pipeline_placement,
)
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


class RecordingTransformerLayer(TransformerLayer):
    """Minimal TransformerLayer stand-in that records its call signature."""

    def __init__(self, layer_number=1, cuda_graph_impl="none", is_hash_layer=True):
        torch.nn.Module.__init__(self)
        self.layer_number = layer_number
        self.config = SimpleNamespace(cuda_graph_impl=cuda_graph_impl)
        self.mlp = SimpleNamespace(router=SimpleNamespace(is_hash_layer=is_hash_layer))
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


class RecordingHashMlpTransformerLayer(TransformerLayer):
    """Hybrid MOE-slot stand-in that exercises the wrapper's raw MLP fast path."""

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.layer_number = 1
        self.config = SimpleNamespace(
            bias_dropout_fusion=False, cuda_graph_impl="none", inference_fuse_tp_communication=False
        )
        self.self_attention = IdentityOp()
        self.cross_attention = IdentityOp()
        self.mlp = torch.nn.Identity()
        self.mlp.router = SimpleNamespace(is_hash_layer=True)
        self.mlp_norm_manager = None
        self.hidden_dropout = 0.0
        self.recompute_pre_mlp_layernorm = False
        self.mhc_checkpoint_pre_mlp_layernorm = False
        self.calls = []

    def _forward_mlp_output_with_bias(
        self,
        hidden_states,
        inference_context=None,
        padding_mask=None,
        packed_seq_params=None,
        input_ids=None,
        mhc_recompute_manager=None,
    ):
        self.calls.append(
            {
                'inference_context': inference_context,
                'padding_mask': padding_mask,
                'packed_seq_params': packed_seq_params,
                'input_ids': input_ids,
                'mhc_recompute_manager': mhc_recompute_manager,
            }
        )
        return (torch.zeros_like(hidden_states), None), hidden_states


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


class RecordingMoE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, hidden_states, padding_mask=None, input_ids=None):
        self.calls.append((padding_mask, input_ids))
        return hidden_states, None


class SizeOneGroup:
    @staticmethod
    def size():
        return 1


class PassthroughHyperConnection(torch.nn.Module):
    """Minimal mHC transform that lets the canonical wrapper execute on CPU."""

    def forward(self, hidden_states, **_kwargs):
        return hidden_states, hidden_states, hidden_states, hidden_states

    def fused_h_res_h_post_bda(self, _h_res, residual, _h_post, layer_output_with_bias, **_kwargs):
        return residual + layer_output_with_bias[0]


def wrap_with_passthrough_mhc(inner_layer):
    wrapper = HyperConnectionHybridLayer.__new__(HyperConnectionHybridLayer)
    torch.nn.Module.__init__(wrapper)
    wrapper.config = SimpleNamespace(fp32_residual_connection=False, params_dtype=None)
    wrapper.inner_layer = inner_layer
    wrapper.layer_number = inner_layer.layer_number
    wrapper.hyper_connection = PassthroughHyperConnection()
    return wrapper


def make_stack(layers, **config_overrides):
    config = {
        'cuda_graph_impl': "none",
        'flash_decode': False,
        'fp8': False,
        'fp8_recipe': None,
        'fp4': None,
        'enable_mhc_connections': False,
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
        _cp_layout_manager=None,
        _mhc_block_end_plan=None,
    )
    stack._build_mhc_recompute_layer_plan = lambda _enabled: (
        [None] * len(layers),
        [False] * len(layers),
    )
    stack._finalize_mhc_recompute_layer = lambda **_kwargs: None
    stack._uses_hash_routing = HybridStack._uses_hash_routing
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
    learned_transformer_layer = RecordingTransformerLayer(layer_number=2, is_hash_layer=False)
    non_transformer_layer = RecordingNonTransformerLayer()
    stack = make_stack(
        [transformer_layer, learned_transformer_layer, non_transformer_layer],
        recompute_granularity=recompute_granularity,
    )
    input_ids = torch.arange(8).reshape(2, 4)

    run_stack(stack, input_ids)

    assert transformer_layer.calls[0]['input_ids'] is input_ids
    assert 'input_ids' not in learned_transformer_layer.calls[0]
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
    learned_transformer_layer = RecordingTransformerLayer(layer_number=2, is_hash_layer=False)
    non_transformer_layer = RecordingNonTransformerLayer()
    stack = make_stack(
        [transformer_layer, learned_transformer_layer, non_transformer_layer],
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=3,
    )
    input_ids = torch.arange(8).reshape(2, 4)

    run_stack(stack, input_ids)

    assert transformer_layer.calls[0]['input_ids'] is input_ids
    assert 'input_ids' not in learned_transformer_layer.calls[0]
    assert non_transformer_layer.num_calls == 1


def test_hybrid_mhc_wrapper_passes_ids_and_recompute_manager_to_mlp_branch():
    inner_layer = RecordingHashMlpTransformerLayer()
    wrapped_layer = wrap_with_passthrough_mhc(inner_layer)
    manager = SimpleNamespace(is_last_layer_in_recompute_block=False)
    hidden_states = torch.randn(4, 2, 8, requires_grad=True)
    input_ids = torch.arange(8).reshape(2, 4)

    output, _ = wrapped_layer(
        hidden_states, attention_mask=None, mhc_recompute_manager=manager, input_ids=input_ids
    )

    assert output.shape == hidden_states.shape
    assert inner_layer.calls[0]['input_ids'] is input_ids
    assert inner_layer.calls[0]['mhc_recompute_manager'] is manager


def test_hybrid_mhc_wrapper_discards_managed_pre_mlp_norm_checkpoint():
    inner_layer = RecordingHashMlpTransformerLayer()
    inner_layer.mhc_checkpoint_pre_mlp_layernorm = True
    inner_layer.pre_mlp_norm_checkpoint = mock.MagicMock()
    wrapped_layer = wrap_with_passthrough_mhc(inner_layer)
    manager = SimpleNamespace(is_last_layer_in_recompute_block=False)
    hidden_states = torch.randn(4, 2, 8, requires_grad=True)

    wrapped_layer(
        hidden_states,
        attention_mask=None,
        mhc_recompute_manager=manager,
        input_ids=torch.arange(8).reshape(2, 4),
    )

    inner_layer.pre_mlp_norm_checkpoint.discard_output_and_register_recompute.assert_called_once()


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
        sequence_parallel=False,
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
    expected_input_ids = input_ids if expects_input_ids else None
    assert decoder.kwargs['input_ids'] is expected_input_ids


def test_hybrid_model_sequence_shards_hash_ids_with_decoder_input(monkeypatch):
    decoder = RecordingDecoder()
    tp_group = object()
    scattered = []

    def fake_scatter(tensor, group):
        assert group is tp_group
        scattered.append(tensor)
        return tensor[: tensor.shape[0] // 2].contiguous()

    monkeypatch.setattr(
        "megatron.core.models.hybrid.hybrid_model.tensor_parallel."
        "scatter_to_sequence_parallel_region",
        fake_scatter,
    )
    model = SimpleNamespace(
        config=SimpleNamespace(
            fine_grained_activation_offloading=False,
            moe_paged_stash=False,
            moe_n_hash_layers=1,
            actual_vocab_size=128,
            sequence_parallel=True,
        ),
        decoder=decoder,
        position_embedding_type="none",
        pre_process=True,
        post_process=False,
        share_embeddings_and_output_weights=False,
        mtp_process=False,
        vocab_size=128,
        pg_collection=SimpleNamespace(tp=tp_group),
    )
    input_ids = torch.arange(8).reshape(2, 4)
    padding_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    hidden_states = torch.randn(2, 2, 8)

    HybridModel.forward(
        model,
        input_ids=input_ids,
        position_ids=torch.arange(4).repeat(2, 1),
        attention_mask=None,
        decoder_input=hidden_states,
        padding_mask=padding_mask,
    )

    assert len(scattered) == 2
    assert torch.equal(decoder.kwargs["input_ids"], input_ids[:, :2])
    assert torch.equal(decoder.kwargs["padding_mask"], padding_mask[:, :2])


def test_chunked_hash_moe_keeps_ids_and_padding_aligned():
    moe = RecordingMoE()
    layer = SimpleNamespace(
        config=SimpleNamespace(
            mlp_chunks_for_prefill=1,
            mlp_chunks_for_training=2,
            transformer_impl="local",
            inference_fuse_tp_communication=False,
        ),
        training=True,
        mlp=moe,
        is_moe_layer=True,
        recompute_mlp=False,
    )
    hidden_states = torch.randn(4, 2, 8)
    input_ids = torch.arange(8).reshape(2, 4)
    padding_mask = torch.tensor(
        [[False, False, True, True], [False, True, False, True]], dtype=torch.bool
    )

    output, bias = TransformerLayer._run_mlp(
        layer,
        hidden_states,
        hidden_states,
        padding_mask,
        inference_context=None,
        input_ids=input_ids,
    )

    assert bias is None
    assert torch.equal(output, hidden_states)
    assert len(moe.calls) == 2
    for idx, (seen_padding, seen_ids) in enumerate(moe.calls):
        assert torch.equal(seen_padding, padding_mask[:, idx * 2 : (idx + 1) * 2])
        assert torch.equal(seen_ids, input_ids[:, idx * 2 : (idx + 1) * 2])


def test_packed_hash_moe_unflattens_ids_in_hidden_state_order():
    layer = SimpleNamespace(is_moe_layer=True)
    hidden_states = torch.arange(8).reshape(8, 1, 1)
    input_ids = torch.arange(8).reshape(1, 8)

    hidden_states, _, input_ids, mbs = TransformerLayer._maybe_unflatten_for_moe(
        layer,
        hidden_states,
        padding_mask=None,
        input_ids=input_ids,
        packed_seq_params=SimpleNamespace(tokens_per_sample=4),
    )

    assert mbs == 2
    assert input_ids.shape == (2, 4)
    assert torch.equal(hidden_states.reshape(-1), input_ids.T.reshape(-1))


def test_hybrid_hash_moe_pp_does_not_require_explicit_pipeline_layout():
    config = TransformerConfig(
        num_layers=4,
        hidden_size=64,
        num_attention_heads=4,
        use_cpu_initialization=True,
        pipeline_model_parallel_size=2,
        pipeline_dtype=torch.float32,
        num_moe_experts=4,
        moe_n_hash_layers=3,
        actual_vocab_size=128,
        is_hybrid_model=True,
    )

    assert config.pipeline_model_parallel_layout is None

    with pytest.raises(AssertionError, match="pipeline_model_parallel_layout must be set"):
        TransformerConfig(
            num_layers=4,
            hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=True,
            pipeline_model_parallel_size=2,
            pipeline_dtype=torch.float32,
            num_moe_experts=4,
            moe_n_hash_layers=3,
            actual_vocab_size=128,
            is_hybrid_model=False,
        )


def test_hash_moe_threshold_counts_only_moe_positions():
    assert _get_hash_moe_layer_threshold("-E-E-E-E", 3) == 6


def test_hash_moe_threshold_rejects_count_larger_than_pattern():
    with pytest.raises(ValueError, match="exceeds the 2 MoE layers"):
        _get_hash_moe_layer_threshold("-E-E", 3)


def test_hash_moe_pipeline_placement_allows_later_learned_moe_stage():
    _validate_hash_moe_pipeline_placement(
        [LayerSymbols.MOE], layer_offset=6, hash_moe_layer_threshold=4, pre_process=False
    )

    with pytest.raises(ValueError, match="non-embedding stage contains hash MoE"):
        _validate_hash_moe_pipeline_placement(
            [LayerSymbols.MOE], layer_offset=4, hash_moe_layer_threshold=6, pre_process=False
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
        return _MtpMoEStub(layer_number=kwargs["layer_number"], is_mtp_layer=kwargs["is_mtp_layer"])

    monkeypatch.setattr(hybrid_block_module, "build_module", fake_build_module)
    config = SimpleNamespace(
        fp8=False, fp4=None, enable_mhc_connections=False, cuda_graph_impl="none"
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
        allocation_module, "validate_segment_layers", lambda _pattern: [LayerSymbols.MOE]
    )
    monkeypatch.setattr(mtp_module, "build_module", lambda *_args, **_kwargs: torch.nn.Identity())

    config = SimpleNamespace(
        enable_mhc_connections=False,
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
        hash_moe_layer_threshold=6,
    )

    assert layer.layer_number == 2
    assert captured_stack_kwargs["is_mtp_layer"] is True
    assert captured_stack_kwargs["mtp_layer_number"] == 2
    assert captured_stack_kwargs["hash_moe_layer_threshold"] == 6


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


@pytest.mark.parametrize("score_function", ["sigmoid", "sqrtsoftplus"])
def test_hash_top1_preserves_gate_weight_and_gradient(score_function):
    """Non-softmax top-1 hash routing must not normalize its only weight to one."""
    router = TopKRouter.__new__(TopKRouter)
    torch.nn.Module.__init__(router)
    router.score_function = score_function
    router.topk = 1
    router.num_experts = 2
    router.config = SimpleNamespace(
        moe_router_force_load_balancing=False,
        moe_router_force_biased=None,
        moe_router_topk_scaling_factor=None,
    )
    router.tid2eid = torch.tensor([[0], [1]], dtype=torch.int32)

    logits = torch.tensor([[0.25, -0.5], [0.75, 0.5]], requires_grad=True)
    input_ids = torch.tensor([[0, 1]])
    routing_probs, _ = router._hash_routing(logits, input_ids)

    if score_function == "sigmoid":
        scores = torch.sigmoid(logits)
    else:
        scores = torch.nn.functional.softplus(logits).sqrt()
    expected = torch.zeros_like(logits).scatter(1, router.tid2eid.long(), scores.diag()[:, None])
    torch.testing.assert_close(routing_probs, expected)

    routing_probs.sum().backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) == 2
