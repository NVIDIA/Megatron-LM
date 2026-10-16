# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Index-sharing and repeated-MTP consumers inherit the producer's certificate that every
top-k row selects a key, so the sparse-attention backend can skip its empty-row masking."""

import dataclasses
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from megatron.core import tensor_parallel
from megatron.core.extensions.transformer_engine import TELinear, TENorm
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant import dsa as dsa_module
from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels, dsa_kernels
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
)
from megatron.core.transformer.forward_sharing import (
    ForwardSharingState,
    forward_sharing_lifetime,
    get_forward_sharing_state,
    is_mtp_repeated_sharing_source,
    mtp_repeated_sharing_lifetime,
    preserve_forward_sharing_for_checkpoint,
)
from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer
from megatron.core.transformer.spec_utils import ModuleSpec
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_dsa_checkpoint_topk_reuse import (
    _ATTENTION_MASK_ARG,
    _assert_similar,
    _make_config,
    _make_inputs,
    _skip_if_cudnn_dsa_unavailable,
)
from tests.unit_tests.transformer.experimental_attention_variant.test_mtp_dsa_cross_depth_sharing import (
    _make_config as make_small_dsa_config,
)

_SEQ_LEN = 4
# Small-config schedule (dsa_indexer_topk_freq=4, dsa_indexer_skip_topk_offset=3): layers 1-3
# select their own top-k and layer 4 reuses layer 3's.
_PRODUCER_LAYER = 3
_SKIP_LAYER = 4


@pytest.fixture
def cpu_dsa_layers(monkeypatch):
    """Build DSA layers on CPU without Transformer Engine indexer modules."""
    monkeypatch.setattr(dsa_module, "build_module", lambda *_args, **_kwargs: nn.Identity())


@pytest.fixture
def certificates(monkeypatch):
    """Record the certificate and top-k tensor each sparse-attention call receives."""
    seen = []
    original = dsa_module._run_sparse_attention

    def recording_sparse_attention(**kwargs):
        seen.append(
            SimpleNamespace(
                certificate=kwargs["all_topk_rows_nonempty"],
                topk=kwargs["topk_indices"],
                grad_enabled=torch.is_grad_enabled(),
            )
        )
        return original(**kwargs)

    monkeypatch.setattr(dsa_module, "_run_sparse_attention", recording_sparse_attention)
    return seen


@pytest.fixture
def cpu_checkpoint_rng(monkeypatch):
    """Keep real checkpoint/autograd, replacing only CUDA RNG access for CPU tensors."""
    monkeypatch.setattr(
        tensor_parallel.random, "_get_all_rng_states", lambda: (torch.get_rng_state(),)
    )
    monkeypatch.setattr(tensor_parallel.random, "_set_all_rng_states", torch.set_rng_state)


def _fake_forward_before_topk(x, _qr, _packed_seq_params):
    generator = torch.Generator().manual_seed(7)
    rows, batch = x.shape[:2]
    q = torch.randn(rows, batch, 2, 8, generator=generator)
    k = torch.randn(rows, batch, 8, generator=generator)
    weights = torch.rand(rows, batch, 2, generator=generator) + 0.5
    return q, k, weights


def _make_cpu_layer(config, layer_number: int, *, is_mtp_layer: bool = False):
    attention = dsa_module.DSAttention(
        config=config,
        submodules=dsa_module.DSAttentionSubmodules(indexer=None),
        layer_number=layer_number,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        pg_collection=SimpleNamespace(tp=None, cp=None),
        is_mtp_layer=is_mtp_layer,
    )
    if attention.indexer is not None:
        attention.indexer.forward_before_topk = _fake_forward_before_topk
    return attention


def _cpu_inputs(config, scale=None):
    generator = torch.Generator().manual_seed(11)
    heads = config.num_attention_heads
    absorbed_dim = config.kv_lora_rank + config.qk_pos_emb_head_dim
    inputs = {
        "query": torch.randn(_SEQ_LEN, 1, heads, absorbed_dim, generator=generator),
        "key": torch.randn(_SEQ_LEN, 1, 1, absorbed_dim, generator=generator),
        "x": torch.randn(_SEQ_LEN, 1, config.hidden_size, generator=generator),
        "qr": torch.randn(_SEQ_LEN, 1, config.q_lora_rank, generator=generator),
        "up_v_weight": torch.randn(
            heads, config.v_head_dim, config.kv_lora_rank, generator=generator
        ),
    }
    if scale is not None:
        inputs["query"] = inputs["query"] * scale
        inputs["key"] = inputs["key"] * scale
    return inputs


def _mask_with_empty_first_row():
    """One carrier for every call: causal layouts ignore it, explicit layouts empty row 0."""
    mask = torch.ones((1, 1, _SEQ_LEN, _SEQ_LEN), dtype=torch.bool).triu(1)
    mask[:, :, 0, :] = True
    return mask


def _cpu_forward(attention, inputs, attention_mask, attn_mask_type):
    return attention(
        query=inputs["query"],
        key=inputs["key"],
        value=None,
        attention_mask=attention_mask,
        x=inputs["x"],
        qr=inputs["qr"],
        up_v_weight=inputs["up_v_weight"],
        attn_mask_type=attn_mask_type,
    )


def _all_rows_nonempty(topk: torch.Tensor) -> bool:
    return bool((topk >= 0).any(dim=-1).all())


@pytest.mark.parametrize(
    "payload_type", [dsa_module._DSAIndexSharingPayload, dsa_module._DSAMTPRepeatedSharingPayload]
)
def test_payload_snapshot_copies_every_per_layer_dict(payload_type):
    """Checkpoint snapshots must carry the certificate, or replays lose it silently."""
    state = ForwardSharingState()
    payload = state.get_or_create(payload_type)
    for position, item in enumerate(dataclasses.fields(payload_type)):
        getattr(payload, item.name)[_PRODUCER_LAYER] = position
    payload.topk_rows_nonempty_by_layer[_PRODUCER_LAYER] = True

    saved = state.snapshot().get(payload_type)
    for item in dataclasses.fields(payload_type):
        live, copied = getattr(payload, item.name), getattr(saved, item.name)
        assert copied is not live, item.name
        assert copied == live, item.name

    payload.topk_rows_nonempty_by_layer[_PRODUCER_LAYER] = False
    assert saved.topk_rows_nonempty_by_layer == {_PRODUCER_LAYER: True}


def test_mtp_source_reentry_drops_certificate_with_indices(cpu_dsa_layers):
    config = make_small_dsa_config(num_layers=2)
    layer = _make_cpu_layer(config, 1, is_mtp_layer=True)
    number = layer.layer_number
    state = get_forward_sharing_state(config=config)
    with forward_sharing_lifetime(config=config):
        with mtp_repeated_sharing_lifetime(state, number) as flags:
            flags[number] = True
            _, payload = layer._prepare_mtp_sharing_payload(None, None)
            payload.topk_by_layer[number] = torch.zeros((1, _SEQ_LEN, 4), dtype=torch.int64)
            payload.topk_rows_nonempty_by_layer[number] = True
            saved = state.snapshot().get(dsa_module._DSAMTPRepeatedSharingPayload)

            _, fresh = layer._prepare_mtp_sharing_payload(None, None)
            assert fresh is payload
            assert number not in fresh.topk_by_layer
            assert number not in fresh.topk_rows_nonempty_by_layer
            assert saved.topk_rows_nonempty_by_layer == {number: True}
    assert state._entries == {}


@pytest.mark.parametrize(
    "producer_mask_type", [AttnMaskType.causal, None], ids=["causal", "explicit-empty-row"]
)
def test_skip_layer_inherits_producer_certificate(cpu_dsa_layers, certificates, producer_mask_type):
    config = make_small_dsa_config(num_layers=_SKIP_LAYER)
    producer = _make_cpu_layer(config, _PRODUCER_LAYER)
    skip = _make_cpu_layer(config, _SKIP_LAYER)
    assert not producer.skip_topk
    assert skip.skip_topk and skip.source_layer == _PRODUCER_LAYER
    mask = _mask_with_empty_first_row()
    inputs = _cpu_inputs(config)
    certified = producer_mask_type is not None

    with forward_sharing_lifetime(None, mask, config) as state:
        _cpu_forward(producer, inputs, mask, producer_mask_type)
        # The skip layer's own layout is provable either way; only the producer differs.
        _cpu_forward(skip, inputs, mask, AttnMaskType.causal)
        payload = state.get(dsa_module._DSAIndexSharingPayload)
        assert payload.topk_rows_nonempty_by_layer[_PRODUCER_LAYER] is certified
        shared_topk = payload.topk_by_layer[_PRODUCER_LAYER]
        # The certificate describes the shared tensor itself.
        assert _all_rows_nonempty(shared_topk) is certified

    # Before certificates were shared, the skip layer always received False.
    assert [call.certificate for call in certificates] == [certified, certified]
    assert all(call.topk is shared_topk for call in certificates)


@pytest.mark.parametrize("always_select_tail", [True, False], ids=["tail", "no-tail"])
def test_kpool_producer_certifies_rows_only_with_tail_selection(
    cpu_dsa_layers, certificates, always_select_tail
):
    """Without its tail, KPool leaves the rows before the first complete pool empty."""
    config = make_small_dsa_config(num_layers=_SKIP_LAYER)
    producer = _make_cpu_layer(config, _PRODUCER_LAYER)
    skip = _make_cpu_layer(config, _SKIP_LAYER)
    pool_size = 2
    indexer = producer.indexer
    indexer.index_kpool = pool_size
    indexer.index_kpool_always_select_tail = always_select_tail
    indexer.index_kpool_use_quantization = False
    indexer.index_kpool_compress_ape = torch.zeros(pool_size, 8)

    def kpool_forward_before_topk(x, qr, packed_seq_params):
        q, k, weights = _fake_forward_before_topk(x, qr, packed_seq_params)
        indexer._kpool_gate_score = torch.zeros_like(k)
        return q, k, weights

    indexer.forward_before_topk = kpool_forward_before_topk
    mask = torch.ones((1, 1, _SEQ_LEN, _SEQ_LEN), dtype=torch.bool).triu(1)
    inputs = _cpu_inputs(config)

    with forward_sharing_lifetime(None, mask, config) as state:
        _cpu_forward(producer, inputs, mask, AttnMaskType.causal)
        _cpu_forward(skip, inputs, mask, AttnMaskType.causal)
        payload = state.get(dsa_module._DSAIndexSharingPayload)
        shared_topk = payload.topk_by_layer[_PRODUCER_LAYER]
        assert _all_rows_nonempty(shared_topk) is always_select_tail
        assert payload.topk_rows_nonempty_by_layer[_PRODUCER_LAYER] is always_select_tail

    # The producer's plain causal layout alone would prove every row non-empty.
    assert [call.certificate for call in certificates] == [always_select_tail] * 2
    assert all(call.topk is shared_topk for call in certificates)


def test_republished_false_certificate_replaces_stale_true(cpu_dsa_layers, certificates):
    config = make_small_dsa_config(num_layers=_SKIP_LAYER)
    producer = _make_cpu_layer(config, _PRODUCER_LAYER)
    skip = _make_cpu_layer(config, _SKIP_LAYER)
    mask = _mask_with_empty_first_row()
    inputs = _cpu_inputs(config)

    with forward_sharing_lifetime(None, mask, config):
        payload = skip._get_dsa_index_sharing_payload(None, mask)
        _cpu_forward(producer, inputs, mask, AttnMaskType.causal)
        _cpu_forward(skip, inputs, mask, AttnMaskType.causal)
        assert payload.topk_rows_nonempty_by_layer[_PRODUCER_LAYER] is True

        _cpu_forward(producer, inputs, mask, None)
        # Written explicitly, not merely absent.
        assert payload.topk_rows_nonempty_by_layer[_PRODUCER_LAYER] is False
        assert not _all_rows_nonempty(payload.topk_by_layer[_PRODUCER_LAYER])
        _cpu_forward(skip, inputs, mask, AttnMaskType.causal)

    assert [call.certificate for call in certificates] == [True, True, False, False]


def test_consumer_layout_still_gates_inherited_certificate(cpu_dsa_layers, certificates):
    config = make_small_dsa_config(num_layers=_SKIP_LAYER)
    producer = _make_cpu_layer(config, _PRODUCER_LAYER)
    skip = _make_cpu_layer(config, _SKIP_LAYER)
    # Plain causal: a provable layout for the producer, an explicit mask for the consumer.
    mask = torch.ones((1, 1, _SEQ_LEN, _SEQ_LEN), dtype=torch.bool).triu(1)
    inputs = _cpu_inputs(config)

    with forward_sharing_lifetime(None, mask, config):
        _cpu_forward(producer, inputs, mask, AttnMaskType.causal)
        _cpu_forward(skip, inputs, mask, None)

    assert [call.certificate for call in certificates] == [True, False]


def test_missing_certificate_means_false(cpu_dsa_layers, certificates):
    config = make_small_dsa_config(num_layers=_SKIP_LAYER)
    skip = _make_cpu_layer(config, _SKIP_LAYER)
    mask = _mask_with_empty_first_row()
    causal_topk = torch.tensor(
        [[[0, -1, -1, -1], [0, 1, -1, -1], [0, 1, 2, -1], [0, 1, 2, 3]]], dtype=torch.int64
    )

    with forward_sharing_lifetime(None, mask, config):
        payload = skip._get_dsa_index_sharing_payload(None, mask)
        payload.topk_by_layer[_PRODUCER_LAYER] = causal_topk
        _cpu_forward(skip, _cpu_inputs(config), mask, AttnMaskType.causal)

    assert [call.certificate for call in certificates] == [False]


@pytest.mark.parametrize("certified", [True, False])
@pytest.mark.parametrize(
    "source_computes_topk", [True, False], ids=["computing-source", "index-share-source"]
)
def test_repeated_mtp_consumers_inherit_source_certificate(
    cpu_dsa_layers, certificates, source_computes_topk, certified
):
    # The repeated physical MTP layer is global layer N+1: 3 computes, 4 reuses layer 3.
    config = make_small_dsa_config(num_layers=2 if source_computes_topk else 3)
    mtp_layer = _make_cpu_layer(config, 1, is_mtp_layer=True)
    number = mtp_layer.layer_number
    assert mtp_layer.skip_topk is not source_computes_topk
    mask = _mask_with_empty_first_row()
    inputs = _cpu_inputs(config)
    producer_mask_type = AttnMaskType.causal if certified else None
    expected_calls = config.mtp_num_layers

    with forward_sharing_lifetime(None, mask, config) as state:
        if not source_computes_topk:
            decoder_producer = _make_cpu_layer(config, _PRODUCER_LAYER)
            _cpu_forward(decoder_producer, inputs, mask, producer_mask_type)
            expected_calls += 1
        with mtp_repeated_sharing_lifetime(state, number) as flags:
            for depth in range(config.mtp_num_layers):
                flags[number] = depth == 0
                # Only the top-k producer's layout differs; every consumer's is provable.
                computes = depth == 0 and source_computes_topk
                mask_type = producer_mask_type if computes else AttnMaskType.causal
                _cpu_forward(mtp_layer, inputs, mask, mask_type)
            payload = state.get(dsa_module._DSAMTPRepeatedSharingPayload)
            assert payload.topk_rows_nonempty_by_layer[number] is certified
            shared_topk = payload.topk_by_layer[number]
            assert _all_rows_nonempty(shared_topk) is certified

    assert [call.certificate for call in certificates] == [certified] * expected_calls
    assert all(call.topk is shared_topk for call in certificates[-config.mtp_num_layers :])


@pytest.mark.parametrize("certified", [True, False])
def test_repeated_mtp_certificate_crosses_full_recompute_checkpoint(
    cpu_dsa_layers, certificates, cpu_checkpoint_rng, certified
):
    """MTP indices cross the checkpoint as explicit tensors; the certificate stays with them."""
    config = make_small_dsa_config(
        num_layers=2,
        mtp_num_layers=2,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
    )
    core = _make_cpu_layer(config, 1, is_mtp_layer=True)
    assert not core.skip_topk
    producer_mask_type = AttnMaskType.causal if certified else None
    mask = _mask_with_empty_first_row()
    depth_inputs = [torch.randn(_SEQ_LEN, 1, 3, requires_grad=True) for _ in range(2)]
    decoder_inputs = [torch.randn_like(value) for value in depth_inputs]

    def projected_forward(hidden_states, decoder_input, attention_mask=None, **_kwargs):
        state = get_forward_sharing_state(None, attention_mask, config)
        is_source = is_mtp_repeated_sharing_source(state, core.layer_number)
        scale = (hidden_states + decoder_input).mean()
        mask_type = producer_mask_type if is_source else AttnMaskType.causal
        return _cpu_forward(core, _cpu_inputs(config, scale), attention_mask, mask_type)

    layer = SimpleNamespace(
        config=config,
        mtp_layer_pattern=None,
        mtp_model_layer=SimpleNamespace(self_attention=SimpleNamespace(core_attention=core)),
        _proj_and_transformer_layer=projected_forward,
    )
    outputs = []
    with forward_sharing_lifetime(None, mask, config) as state:
        with mtp_repeated_sharing_lifetime(state, core.layer_number) as flags:
            for depth in range(2):
                flags[core.layer_number] = depth == 0
                outputs.append(
                    MultiTokenPredictionLayer._checkpointed_forward(
                        layer, depth_inputs[depth], decoder_inputs[depth], attention_mask=mask
                    )
                )
    assert state._entries == {}
    sum(value.float().sum() for value in outputs).backward()

    first_calls = [call for call in certificates if not call.grad_enabled]
    replays = [call for call in certificates if call.grad_enabled]
    assert len(first_calls) == 2 and len(replays) == 2
    assert all(depth_input.grad is not None for depth_input in depth_inputs)
    assert [call.certificate for call in certificates] == [certified] * 4


def test_skip_layer_replay_reads_certificate_from_checkpoint_snapshot(
    cpu_dsa_layers, certificates, cpu_checkpoint_rng
):
    """Full recompute builds the skip layer's backward from its replay of the snapshot."""
    config = make_small_dsa_config(num_layers=_SKIP_LAYER)
    producer = _make_cpu_layer(config, _PRODUCER_LAYER)
    skip = _make_cpu_layer(config, _SKIP_LAYER)
    mask = _mask_with_empty_first_row()
    inputs = _cpu_inputs(config)
    leaves = {
        name: inputs[name].clone().requires_grad_(True) for name in ("query", "key", "up_v_weight")
    }

    def checkpointed(attention):
        def layer(query, key, x, qr, up_v_weight, attention_mask):
            layer_inputs = dict(query=query, key=key, x=x, qr=qr, up_v_weight=up_v_weight)
            return _cpu_forward(attention, layer_inputs, attention_mask, AttnMaskType.causal)

        wrapped = preserve_forward_sharing_for_checkpoint(
            layer, None, config, attention_mask_arg=_ATTENTION_MASK_ARG
        )
        return tensor_parallel.checkpoint(
            wrapped,
            False,
            leaves["query"],
            leaves["key"],
            inputs["x"],
            inputs["qr"],
            leaves["up_v_weight"],
            mask,
        )

    with forward_sharing_lifetime(None, mask, config):
        checkpointed(producer)
        output = checkpointed(skip)
    output.sum().backward()

    # First calls of the producer and the skip layer, then the skip layer's replay only:
    # the producer's output does not reach the loss.
    assert [(call.certificate, call.grad_enabled) for call in certificates] == [
        (True, False),
        (True, False),
        (True, True),
    ]
    assert leaves["query"].grad is not None


@pytest.mark.parametrize(
    "producer_mask_type", [AttnMaskType.causal, None], ids=["causal", "explicit-empty-row"]
)
def test_frozen_producer_replay_reuses_its_first_call_certificate(
    cpu_dsa_layers, certificates, cpu_checkpoint_rng, monkeypatch, producer_mask_type
):
    """dsa_indexer_reuse_checkpoint_topk with a frozen indexer: the producer's replay takes its
    first call's indices from the snapshot, and the certificate published with them."""
    # Enable the fused dispatch; without a kernel backend it falls back to PyTorch.
    monkeypatch.setattr(dsa_kernels, "use_fused_dsa_kernels", lambda _config: True)
    config = make_small_dsa_config(num_layers=_SKIP_LAYER)
    assert not config.dsa_indexer_loss_coeff
    config.dsa_indexer_reuse_checkpoint_topk = True
    producer = _make_cpu_layer(config, _PRODUCER_LAYER)
    indexer_calls = []

    def counting_forward_before_topk(x, qr, packed_seq_params):
        indexer_calls.append(torch.is_grad_enabled())
        return _fake_forward_before_topk(x, qr, packed_seq_params)

    producer.indexer.forward_before_topk = counting_forward_before_topk
    mask = _mask_with_empty_first_row()
    inputs = _cpu_inputs(config)
    leaves = {
        name: inputs[name].clone().requires_grad_(True) for name in ("query", "key", "up_v_weight")
    }

    def layer(query, key, x, qr, up_v_weight, attention_mask):
        layer_inputs = dict(query=query, key=key, x=x, qr=qr, up_v_weight=up_v_weight)
        return _cpu_forward(producer, layer_inputs, attention_mask, producer_mask_type)

    with forward_sharing_lifetime(None, mask, config) as state:
        wrapped = preserve_forward_sharing_for_checkpoint(
            layer, None, config, attention_mask_arg=_ATTENTION_MASK_ARG
        )
        output = tensor_parallel.checkpoint(
            wrapped,
            False,
            leaves["query"],
            leaves["key"],
            inputs["x"],
            inputs["qr"],
            leaves["up_v_weight"],
            mask,
        )
        payload = state.get(dsa_module._DSAIndexSharingPayload)
        published = payload.topk_rows_nonempty_by_layer[_PRODUCER_LAYER]
    output.sum().backward()

    certified = producer_mask_type is not None
    assert published is certified
    # Only the first call runs the indexer.
    assert indexer_calls == [False]
    assert [(call.certificate, call.grad_enabled) for call in certificates] == [
        (certified, False),
        (certified, True),
    ]
    assert certificates[1].topk is certificates[0].topk
    assert leaves["query"].grad is not None


# Index sharing typically runs 1-3 consumers per producer; two exercise the consumer-to-consumer
# order.
_CONSUMERS = 2


def _make_gpu_config(*, tp: int, cp: int, deterministic: bool, consumer: str, recompute: str):
    config = _make_config(tp=tp, cp=cp, deterministic=deterministic)
    # Set after validation, like deterministic_mode in _make_config.
    if consumer == "skip-layers":
        # Layer 1 selects top-k and layers 2 and 3 reuse it.
        config.num_layers = 1 + _CONSUMERS
        config.dsa_indexer_topk_freq = 1 + _CONSUMERS
    else:
        # The repeated MTP layer (global layer 3) selects top-k at depth 0 and every later
        # depth reuses it.
        config.mtp_num_layers = 1 + _CONSUMERS
        config.mtp_use_repeated_layer = True
        config.mtp_repeated_layer_shared_components = ["sparse_attention_index"]
    config.dsa_indexer_reuse_checkpoint_topk = recompute == "full-reuse-topk"
    return config


def _make_gpu_calls(config, pg_collection: ProcessGroupCollection, consumer: str):
    """Return the attention module of every call in forward order, the producer first."""
    indexer_spec = ModuleSpec(
        module=dsa_module.DSAIndexer,
        submodules=dsa_module.DSAIndexerSubmodules(
            linear_wq_b=ModuleSpec(module=TELinear),
            linear_wk=ModuleSpec(module=TELinear),
            k_norm=ModuleSpec(module=TENorm),
            linear_weights_proj=ModuleSpec(module=TELinear),
        ),
    )

    def make(layer_number: int, is_mtp_layer: bool = False):
        attention = dsa_module.DSAttention(
            config=config,
            submodules=dsa_module.DSAttentionSubmodules(indexer=indexer_spec),
            layer_number=layer_number,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            cp_comm_type="allgather",
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
        )
        return attention.cuda().train()

    if consumer == "skip-layers":
        calls = [make(layer_number) for layer_number in range(1, 2 + _CONSUMERS)]
        assert all(call.skip_topk and call.source_layer == 1 for call in calls[1:])
    else:
        # One physical layer, called once per MTP depth.
        calls = [make(1, is_mtp_layer=True)] * (1 + _CONSUMERS)
        assert calls[0].mtp_sparse_attention_index_share
    assert calls[0].index_share and calls[0].indexer is not None
    return calls


@pytest.fixture
def cudnn_certificates(monkeypatch):
    """Record the certificate the cuDNN sparse-attention forward and backward receive."""
    seen = {"forward": [], "backward": []}
    forward = dsa_cudnn_kernels.run_fused_absorbed_sparse_attention
    backward = dsa_cudnn_kernels._run_sparse_attention_backward

    def recording_forward(*args, **kwargs):
        result = forward(*args, **kwargs)
        assert result is not None, "cuDNN sparse attention declined the layer"
        seen["forward"].append(kwargs.get("all_topk_rows_nonempty", False))
        return result

    def recording_backward(*args, **kwargs):
        seen["backward"].append(kwargs.get("all_topk_rows_nonempty", False))
        return backward(*args, **kwargs)

    monkeypatch.setattr(dsa_cudnn_kernels, "run_fused_absorbed_sparse_attention", recording_forward)
    monkeypatch.setattr(dsa_cudnn_kernels, "_run_sparse_attention_backward", recording_backward)
    return seen


def _skip_if_deterministic_sparse_backward_unavailable(config, tp: int) -> None:
    """Deterministic mode declines cuDNN sparse attention without a deterministic backward."""
    query = torch.empty(0, 0, config.num_attention_heads // tp, 0, device="cuda")
    if not dsa_cudnn_kernels._supports_deterministic_sparse_attention_backward(query):
        pytest.skip("installed cuDNN DSA has no deterministic sparse-attention backward")


def _shared_topk_step(calls, inputs, *, consumer: str, recompute: str, share_certificate: bool):
    """Train the producer and its consumers for one step; return outputs, loss and grads."""
    producer = calls[0]
    config = producer.config
    packed_seq_params = inputs["packed_seq_params"]
    DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
    producer.zero_grad(set_to_none=True)
    leaves = [
        {
            name: inputs[name].detach().clone().requires_grad_(True)
            for name in ("query", "key", "up_v_weight")
        }
        for _ in calls
    ]

    def call(attention):
        def layer(query, key, x, qr, up_v_weight, attention_mask):
            return attention(
                query=query,
                key=key,
                value=None,
                attention_mask=attention_mask,
                x=x,
                qr=qr,
                attn_mask_type=AttnMaskType.causal,
                packed_seq_params=packed_seq_params,
                up_v_weight=up_v_weight,
            )

        if recompute == "none":
            return layer
        wrapped = preserve_forward_sharing_for_checkpoint(
            layer, packed_seq_params, config, attention_mask_arg=_ATTENTION_MASK_ARG
        )
        return lambda *args: tensor_parallel.checkpoint(wrapped, False, *args)

    mtp = consumer == "mtp-depths"
    payload_type = (
        dsa_module._DSAMTPRepeatedSharingPayload if mtp else dsa_module._DSAIndexSharingPayload
    )
    outputs = []
    with forward_sharing_lifetime(packed_seq_params, None, config) as state:
        depth_scope = (
            mtp_repeated_sharing_lifetime(state, producer.layer_number) if mtp else nullcontext()
        )
        with depth_scope as is_source_by_layer:
            for position, (attention, layer_leaves) in enumerate(zip(calls, leaves)):
                if mtp:
                    is_source_by_layer[attention.layer_number] = position == 0
                if position == 1:
                    published = state.get(payload_type).topk_rows_nonempty_by_layer
                    assert published[producer.layer_number] is True
                    if not share_certificate:
                        # Exactly the consumers' behavior before certificates were shared.
                        del published[producer.layer_number]
                outputs.append(
                    call(attention)(
                        layer_leaves["query"],
                        layer_leaves["key"],
                        inputs["x"],
                        inputs["qr"],
                        layer_leaves["up_v_weight"],
                        None,
                    )
                )
    generator = torch.Generator(device="cuda")
    generator.manual_seed(7)
    grad_outputs = [
        torch.randn(output.shape, device="cuda", dtype=output.dtype, generator=generator)
        for output in outputs
    ]
    # As in training, the consumers' backward runs before the producer's and on its tensors.
    torch.autograd.backward(outputs, grad_outputs)

    loss = DSAIndexerLossLoggingHelper.tracker["values"][producer.layer_number - 1]
    grads = {
        f"call{position}.{name}": leaf.grad.detach().clone()
        for position, layer_leaves in enumerate(leaves)
        for name, leaf in layer_leaves.items()
    }
    for name, parameter in producer.indexer.named_parameters():
        assert parameter.grad is not None, f"indexer {name} has no gradient"
        grads[f"indexer.{name}"] = parameter.grad.detach().clone()
    return [output.detach() for output in outputs], loss.detach().clone(), grads


# The TP2+CP2 case shares the real-kernel cuDNN/flash_mla setup of the TP/SP parity tests,
# which are disabled in dev (flaky_in_dev) and LTS (flaky) CI because it fails with a CUDA
# error there. Re-enable together with them once the kernel/build root cause is resolved.
@pytest.mark.parametrize(
    ("tp", "cp"),
    [
        pytest.param(1, 1, id="tp1cp1"),
        pytest.param(2, 2, id="tp2cp2", marks=[pytest.mark.flaky, pytest.mark.flaky_in_dev]),
    ],
)
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize(
    ("consumer", "recompute"),
    [
        ("skip-layers", "none"),
        ("skip-layers", "full"),
        # The producer's replay reuses the very top-k tensors the consumers' backward used.
        ("skip-layers", "full-reuse-topk"),
        ("mtp-depths", "none"),
    ],
)
def test_shared_certificate_is_bitwise_identical_on_cudnn(
    tp, cp, deterministic, consumer, recompute, cudnn_certificates, monkeypatch
):
    _skip_if_cudnn_dsa_unavailable(tp * cp)
    Utils.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)
    try:
        model_parallel_cuda_manual_seed(1234)
        torch.manual_seed(1234)
        DSAIndexerLossAutoScaler.set_loss_scale(torch.ones((), device="cuda"))
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        config = _make_gpu_config(
            tp=tp, cp=cp, deterministic=deterministic, consumer=consumer, recompute=recompute
        )
        if deterministic:
            _skip_if_deterministic_sparse_backward_unavailable(config, tp)
        calls = _make_gpu_calls(config, pg_collection, consumer)
        inputs = _make_inputs(
            config, tp_rank=pg_collection.tp.rank(), cp_rank=pg_collection.cp.rank(), cp=cp
        )
        reused_topk_losses = []
        from_topk = dsa_cudnn_kernels.run_sparse_indexer_loss_from_topk

        def counting_from_topk(*args, **kwargs):
            reused_topk_losses.append(None)
            return from_topk(*args, **kwargs)

        monkeypatch.setattr(
            dsa_cudnn_kernels, "run_sparse_indexer_loss_from_topk", counting_from_topk
        )

        results = {}
        for share_certificate in (False, True):
            reused_topk_losses.clear()
            for recorded in cudnn_certificates.values():
                recorded.clear()
            results[share_certificate] = _shared_topk_step(
                calls,
                inputs,
                consumer=consumer,
                recompute=recompute,
                share_certificate=share_certificate,
            )
            assert len(reused_topk_losses) == int(recompute == "full-reuse-topk")
            # Before certificates were shared, every consumer call received False.
            consumer_flags = [share_certificate] * _CONSUMERS
            forward_flags = [True] + consumer_flags
            if recompute != "none":
                # Backward replays every checkpoint, the producer's last.
                forward_flags += consumer_flags + [True]
            assert cudnn_certificates == {
                "forward": forward_flags,
                "backward": consumer_flags + [True],
            }

        baseline_outputs, baseline_loss, baseline_grads = results[False]
        outputs, loss, grads = results[True]
        # The skipped masks only touched rows that do not exist, so forward is bitwise equal.
        for output, baseline_output in zip(outputs, baseline_outputs):
            torch.testing.assert_close(output, baseline_output, rtol=0, atol=0)
        if deterministic:
            # Every grad, including the producer's after its consumers' backward ran.
            torch.testing.assert_close(loss, baseline_loss, rtol=0, atol=0)
            for name, grad in grads.items():
                torch.testing.assert_close(grad, baseline_grads[name], rtol=0, atol=0, msg=name)
        else:
            # Atomic accumulation in the sparse backward kernels differs run to run.
            torch.testing.assert_close(loss, baseline_loss, rtol=1e-4, atol=1e-7)
            for name, grad in grads.items():
                _assert_similar(grad, baseline_grads[name], label=name, eps=1e-4)
    finally:
        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
        Utils.destroy_model_parallel()
