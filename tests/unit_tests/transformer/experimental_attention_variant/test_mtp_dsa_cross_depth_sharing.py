# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Coverage for component sharing across repeated MTP prediction depths."""

import inspect
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import megatron.core.parallel_state as parallel_state
from megatron.core import tensor_parallel
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_dsa_module_spec_for_backend,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend, AttnMaskType, CudaGraphModule
from megatron.core.transformer.experimental_attention_variant import dsa as dsa_module
from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
    AbsorbedMLASelfAttention,
)
from megatron.core.transformer.forward_sharing import (
    _MTPRepeatedSharingPayload,
    forward_sharing_lifetime,
    get_forward_sharing_state,
    is_mtp_repeated_sharing_source,
    mtp_repeated_sharing_lifetime,
)
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionBlock,
    MultiTokenPredictionLayer,
)
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.transformer.transformer_layer import (
    HyperConnectionTransformerLayer,
    TransformerLayer,
)
from megatron.core.utils import init_method_normal, scaled_init_method_normal
from tests.unit_tests.test_utilities import Utils

LATENT_KV = "latent_kv"
SPARSE_ATTENTION_INDEX = "sparse_attention_index"
BOTH_SHARED_COMPONENTS = [LATENT_KV, SPARSE_ATTENTION_INDEX]


def _make_config(**overrides) -> MLATransformerConfig:
    kwargs = dict(
        num_layers=2,
        mtp_num_layers=3,
        mtp_use_repeated_layer=True,
        mtp_repeated_layer_shared_components=[SPARSE_ATTENTION_INDEX],
        hidden_size=64,
        num_attention_heads=4,
        num_query_groups=4,
        kv_channels=8,
        multi_latent_attention=True,
        experimental_attention_variant="dsa",
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_head_dim=8,
        qk_pos_emb_head_dim=4,
        v_head_dim=8,
        dsa_indexer_n_heads=4,
        dsa_indexer_head_dim=8,
        dsa_indexer_topk=4,
        dsa_indexer_topk_freq=4,
        dsa_indexer_skip_topk_offset=3,
        dsa_indexer_loss_coeff=0.0,
        dsa_indexer_rotate_activation=False,
        dsa_indexer_scoring_relu=False,
        dsa_kernel_backend="none",
        attention_backend=AttnBackend.unfused,
        add_bias_linear=False,
        qk_layernorm=True,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        apply_rope_fusion=False,
        rope_type="rope",
        rotary_base=10000,
        gradient_accumulation_fusion=False,
        init_method=init_method_normal(0.02),
        output_layer_init_method=scaled_init_method_normal(0.02, 2, multiplier=2.0),
        use_cpu_initialization=False,
        perform_initialization=True,
    )
    kwargs.update(overrides)
    return MLATransformerConfig(**kwargs)


@pytest.mark.parametrize(
    "shared_components",
    [
        pytest.param(None, id="none"),
        pytest.param([], id="empty"),
        pytest.param([LATENT_KV], id="latent-kv-only"),
        pytest.param([SPARSE_ATTENTION_INDEX], id="sparse-attention-index-only"),
        pytest.param(BOTH_SHARED_COMPONENTS, id="both"),
        pytest.param(list(reversed(BOTH_SHARED_COMPONENTS)), id="both-reversed"),
    ],
)
def test_cross_depth_shared_components_accept_supported_values(shared_components):
    config = _make_config(mtp_repeated_layer_shared_components=shared_components)

    assert config.mtp_repeated_layer_shared_components == shared_components


@pytest.mark.parametrize(
    ("shared_components", "message"),
    [
        (SPARSE_ATTENTION_INDEX, "must be a list or None"),
        ([1], "entries must be strings"),
        (["unknown"], "Unsupported mtp_repeated_layer_shared_components"),
        ([SPARSE_ATTENTION_INDEX, SPARSE_ATTENTION_INDEX], "must not contain duplicate components"),
    ],
)
def test_cross_depth_shared_components_reject_invalid_lists(shared_components, message):
    with pytest.raises(ValueError, match=message):
        _make_config(mtp_repeated_layer_shared_components=shared_components)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"experimental_attention_variant": None}, "requires experimental_attention_variant"),
        ({"mtp_use_repeated_layer": False}, "requires mtp_use_repeated_layer"),
        ({"mtp_num_layers": 1}, "requires mtp_num_layers > 1"),
        (
            {"recompute_granularity": "full", "recompute_method": "uniform"},
            "does not support full activation recomputation",
        ),
        ({"recompute_granularity": "selective"}, "does not support core_attn recompute"),
        (
            {"cuda_graph_impl": "transformer_engine", "cuda_graph_modules": ["attn"]},
            "does not support CUDA graph capture that includes attention",
        ),
        (
            {"cuda_graph_impl": "local", "cuda_graph_modules": []},
            "does not support CUDA graph capture that includes attention",
        ),
        (
            {"cuda_graph_impl": "full_iteration", "cuda_graph_modules": []},
            "does not support CUDA graph capture that includes attention",
        ),
    ],
)
def test_cross_depth_sharing_rejects_incompatible_config(overrides, message):
    with pytest.raises(ValueError, match=message):
        _make_config(**overrides)


@pytest.mark.parametrize(
    "modules",
    [
        [],
        ["mlp"],
        ["moe"],
        ["moe_act"],
        ["shared_experts"],
        ["layernorm"],
        ["mhc"],
        ["mla_up_proj"],
        ["layernorm", "mlp", "mla_up_proj"],
        ["layernorm", "moe", "shared_experts"],
    ],
)
def test_index_sharing_accepts_selective_recompute_outside_dsa(modules):
    config = _make_config(
        recompute_granularity="selective",
        recompute_modules=modules,
        num_moe_experts=4,
        moe_grouped_gemm=True,
        moe_shared_expert_intermediate_size=64,
        enable_hyper_connections="mhc" in modules,
    )
    assert config.recompute_modules == modules


@pytest.mark.parametrize("modules", [None, ["core_attn"], ["mlp", "core_attn"]])
def test_index_sharing_rejects_selective_recompute_that_reenters_dsa(modules):
    with pytest.raises(ValueError, match="does not support core_attn recompute"):
        _make_config(recompute_granularity="selective", recompute_modules=modules)


@pytest.mark.parametrize("method", ["uniform", "block"])
def test_index_sharing_rejects_full_recompute(method):
    with pytest.raises(ValueError, match="does not support full activation recomputation"):
        _make_config(recompute_granularity="full", recompute_method=method, recompute_num_layers=1)


@pytest.mark.parametrize(
    "shared_components", [[SPARSE_ATTENTION_INDEX], [LATENT_KV], BOTH_SHARED_COMPONENTS]
)
def test_mlp_recompute_after_repeat_cleanup_preserves_gradients(monkeypatch, shared_components):
    """Exercise MCore's real checkpoint backward with CPU RNG and a DSA-owned payload."""
    monkeypatch.setattr(dsa_module, "build_module", lambda *_args, **_kwargs: nn.Identity())
    # The test has no CUDA operations. Keep the checkpoint/autograd implementation intact
    # while replacing only its device-specific RNG snapshot with the CPU equivalent.
    monkeypatch.setattr(
        tensor_parallel.random, "_get_all_rng_states", lambda: (torch.get_rng_state(),)
    )
    monkeypatch.setattr(tensor_parallel.random, "_set_all_rng_states", torch.set_rng_state)
    initial = torch.randn(4, 1, 8)
    weight = torch.randn(8, 8)

    def run(recompute):
        config = _make_config(
            mtp_repeated_layer_shared_components=shared_components,
            recompute_granularity="selective" if recompute else None,
            recompute_modules=["mlp"],
        )
        attention = dsa_module.DSAttention(
            config=config,
            submodules=dsa_module.DSAttentionSubmodules(indexer=None),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            pg_collection=SimpleNamespace(),
            is_mtp_layer=True,
        )
        state = get_forward_sharing_state(config=config)
        inputs = initial.clone().requires_grad_(True)
        mlp_weight = weight.clone().requires_grad_(True)
        calls = []

        def mlp(value):
            active_call = state.get(_MTPRepeatedSharingPayload)
            calls.append(
                active_call.is_source_by_layer[config.num_layers + 1]
                if active_call is not None
                else None
            )
            return torch.nn.functional.gelu(value @ mlp_weight)

        outputs = []
        hidden = inputs
        with (
            forward_sharing_lifetime(config=config),
            mtp_repeated_sharing_lifetime(state, config.num_layers + 1) as sharing_call,
        ):
            for depth in range(config.mtp_num_layers):
                sharing_call[config.num_layers + 1] = depth == 0
                _, payload = attention._prepare_mtp_sharing_payload(None, None)
                if depth == 0:
                    payload.topk_by_layer[config.num_layers + 1] = torch.zeros(1, dtype=torch.int64)
                    if LATENT_KV in shared_components:
                        payload.latent_kv_by_layer[config.num_layers + 1] = inputs.square()
                if LATENT_KV in shared_components:
                    hidden = hidden + payload.latent_kv_by_layer[config.num_layers + 1]
                hidden = (
                    tensor_parallel.checkpoint(mlp, False, hidden) if recompute else mlp(hidden)
                )
                outputs.append(hidden)
        assert state.get(_MTPRepeatedSharingPayload) is None
        sum(output.square().mean() for output in outputs).backward()
        assert calls[: config.mtp_num_layers] == [True] + [False] * (config.mtp_num_layers - 1)
        if recompute:
            assert calls[config.mtp_num_layers :] == [None] * config.mtp_num_layers
        else:
            assert len(calls) == config.mtp_num_layers
        return torch.stack(outputs).detach(), inputs.grad, mlp_weight.grad

    for actual, expected in zip(run(True), run(False)):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("shared_components", [[LATENT_KV], BOTH_SHARED_COMPONENTS])
@pytest.mark.parametrize(
    "modules",
    [
        [],
        ["mlp"],
        ["moe"],
        ["moe_act"],
        ["shared_experts"],
        ["layernorm"],
        ["mhc"],
        ["layernorm", "moe", "shared_experts"],
    ],
)
def test_latent_kv_sharing_accepts_selective_recompute_outside_shared_producer(
    shared_components, modules
):
    config = _make_config(
        mtp_repeated_layer_shared_components=shared_components,
        recompute_granularity="selective",
        recompute_modules=modules,
        num_moe_experts=4,
        moe_grouped_gemm=True,
        moe_shared_expert_intermediate_size=64,
        enable_hyper_connections="mhc" in modules,
    )
    assert config.recompute_modules == modules


@pytest.mark.parametrize("shared_components", [[LATENT_KV], BOTH_SHARED_COMPONENTS])
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"recompute_granularity": "full", "recompute_method": "uniform"},
            "does not support full activation recomputation",
        ),
        (
            {"recompute_granularity": "full", "recompute_method": "block"},
            "does not support full activation recomputation",
        ),
        ({"recompute_granularity": "selective"}, "does not support core_attn recompute"),
        (
            {"recompute_granularity": "selective", "recompute_modules": ["mlp", "core_attn"]},
            "does not support core_attn recompute",
        ),
        (
            {"recompute_granularity": "selective", "recompute_modules": ["mla_up_proj"]},
            "does not support mla_up_proj recompute",
        ),
        (
            {
                "recompute_granularity": "selective",
                "recompute_modules": ["layernorm", "mla_up_proj"],
            },
            "does not support mla_up_proj recompute",
        ),
    ],
)
def test_latent_kv_sharing_rejects_conflicting_recompute(shared_components, overrides, message):
    with pytest.raises(ValueError, match=message):
        _make_config(mtp_repeated_layer_shared_components=shared_components, **overrides)


def test_latent_kv_runtime_rejects_discarding_shared_source_storage():
    attention = SimpleNamespace(
        training=True,
        cache_mla_latents=False,
        core_attention=SimpleNamespace(mtp_cross_depth_share=True, mtp_latent_kv_share=True),
        checkpoint_core_attention=False,
        recompute_up_proj=True,
    )
    with pytest.raises(RuntimeError, match="source KV storage would be discarded"):
        AbsorbedMLASelfAttention.forward(
            attention, hidden_states=torch.zeros(1, 1, 4), attention_mask=None
        )


def test_cross_depth_sharing_accepts_supported_cuda_graph_scopes():
    moe_config = _make_config(
        cuda_graph_impl="transformer_engine",
        cuda_graph_modules=["moe_router", "moe_preprocess"],
        num_moe_experts=4,
    )
    assert moe_config.cuda_graph_modules == [
        CudaGraphModule.moe_router,
        CudaGraphModule.moe_preprocess,
    ]


def test_dsa_state_rejects_latent_kv_consumer_without_source(monkeypatch):
    monkeypatch.setattr(dsa_module, "build_module", lambda *_args, **_kwargs: nn.Identity())
    attention = dsa_module.DSAttention(
        config=_make_config(mtp_repeated_layer_shared_components=[LATENT_KV]),
        submodules=dsa_module.DSAttentionSubmodules(indexer=None),
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        pg_collection=SimpleNamespace(),
        is_mtp_layer=True,
    )

    forward_state = get_forward_sharing_state(config=attention.config)
    with mtp_repeated_sharing_lifetime(forward_state, attention.layer_number) as sharing_call:
        sharing_call[attention.layer_number] = False
        with pytest.raises(RuntimeError, match="requires published source"):
            attention._prepare_mtp_sharing_payload(None, None)


@pytest.mark.parametrize("fail_second_layer", [False, True])
def test_mtp_payload_requires_a_source_and_isolates_layers(monkeypatch, fail_second_layer):
    monkeypatch.setattr(dsa_module, "build_module", lambda *_args, **_kwargs: nn.Identity())
    config = _make_config()
    layers = [
        dsa_module.DSAttention(
            config=config,
            submodules=dsa_module.DSAttentionSubmodules(indexer=None),
            layer_number=number,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            pg_collection=SimpleNamespace(),
            is_mtp_layer=True,
        )
        for number in (1, 2)
    ]
    state = get_forward_sharing_state(config=config)
    with pytest.raises(RuntimeError, match="active layer"):
        layers[0]._prepare_mtp_sharing_payload(None, None)
    first_number, second_number = (layer.layer_number for layer in layers)
    with forward_sharing_lifetime(config=config):
        ordinary = state.get_or_create(dsa_module._DSAIndexSharingPayload)
        ordinary.topk_by_layer[1] = torch.zeros(1)
        with mtp_repeated_sharing_lifetime(state, first_number) as flags:
            assert flags == {first_number: None}
            with pytest.raises(RuntimeError, match="is_source flag"):
                layers[0]._prepare_mtp_sharing_payload(None, None)
            flags[first_number] = False
            with pytest.raises(RuntimeError, match="requires published source"):
                layers[0]._prepare_mtp_sharing_payload(None, None)
            flags[first_number] = True
            is_source, first = layers[0]._prepare_mtp_sharing_payload(None, None)
            assert is_source
            first.topk_by_layer[first_number] = torch.ones(1)
            first.topk_length_by_layer[first_number] = torch.ones(1, dtype=torch.int64)
            first.latent_kv_by_layer[first_number] = torch.ones(1)
            with pytest.raises(RuntimeError, match="already active"):
                with mtp_repeated_sharing_lifetime(state, first_number):
                    pass
            try:
                with mtp_repeated_sharing_lifetime(state, second_number) as second_flags:
                    assert second_flags is flags
                    assert set(flags) == {first_number, second_number}
                    flags[second_number] = True
                    _, second = layers[1]._prepare_mtp_sharing_payload(None, None)
                    assert second is first
                    with pytest.raises(RuntimeError, match="requires published source"):
                        layers[1]._require_mtp_sharing_payload(None, None)
                    second_topk = torch.zeros(1)
                    second.topk_by_layer[second_number] = second_topk
                    second.topk_length_by_layer[second_number] = torch.zeros(1, dtype=torch.int64)
                    second.latent_kv_by_layer[second_number] = torch.zeros(1)
                    flags[first_number] = False
                    assert layers[0]._prepare_mtp_sharing_payload(None, None) == (False, first)
                    assert flags[second_number]
                    if fail_second_layer:
                        raise RuntimeError("second layer failed")
            except RuntimeError as error:
                assert fail_second_layer and str(error) == "second layer failed"
            assert second_number not in flags
            assert second.topk_by_layer[second_number] is second_topk
            assert layers[0]._prepare_mtp_sharing_payload(None, None) == (False, first)
            with pytest.raises(RuntimeError, match="active layer"):
                layers[1]._prepare_mtp_sharing_payload(None, None)
        assert state.get(dsa_module._DSAIndexSharingPayload) is ordinary
        assert state.get(_MTPRepeatedSharingPayload) is None
        tensors = state.get(dsa_module._DSAMTPRepeatedSharingPayload)
        assert tensors is first
        assert set(tensors.topk_by_layer) == {first_number, second_number}
        with mtp_repeated_sharing_lifetime(state, first_number) as flags:
            flags[first_number] = True
            _, fresh = layers[0]._prepare_mtp_sharing_payload(None, None)
            assert fresh is first
            assert first_number not in fresh.topk_by_layer
            assert first_number not in fresh.topk_length_by_layer
            assert first_number not in fresh.latent_kv_by_layer
            assert tensors.topk_by_layer[second_number] is second_topk
            assert second_number in tensors.topk_length_by_layer
            assert second_number in tensors.latent_kv_by_layer
            flags[first_number] = False
            with pytest.raises(RuntimeError, match="requires published source"):
                layers[0]._prepare_mtp_sharing_payload(None, None)
    assert state._entries == {}


def test_mtp_snapshot_copies_dictionaries_and_cleanup_releases_live_tensors():
    config = SimpleNamespace()
    state = get_forward_sharing_state(config=config)
    indices = torch.zeros(2, dtype=torch.int64)
    lengths = torch.ones(2, dtype=torch.int64)
    source = torch.ones(2, requires_grad=True)
    latent_kv = source.square()
    with forward_sharing_lifetime(config=config):
        with mtp_repeated_sharing_lifetime(state, 7) as flags:
            flags[7] = True
            payload = state.get_or_create(dsa_module._DSAMTPRepeatedSharingPayload)
            payload.topk_by_layer[7] = indices
            payload.topk_length_by_layer[7] = lengths
            payload.latent_kv_by_layer[7] = latent_kv
            snapshot = state.snapshot()
            saved = snapshot.get(dsa_module._DSAMTPRepeatedSharingPayload)
            assert saved is not payload
            assert saved.topk_by_layer is not payload.topk_by_layer
            assert saved.topk_length_by_layer is not payload.topk_length_by_layer
            assert saved.latent_kv_by_layer is not payload.latent_kv_by_layer
            payload.topk_by_layer.clear()
            payload.topk_length_by_layer.clear()
            payload.latent_kv_by_layer.clear()
            flags[7] = False
            assert snapshot.get(_MTPRepeatedSharingPayload).is_source_by_layer[7]
        assert state.get(_MTPRepeatedSharingPayload) is None
        assert state.get(dsa_module._DSAMTPRepeatedSharingPayload) is payload
        with mtp_repeated_sharing_lifetime(state, 7) as fresh_flags:
            assert fresh_flags == {7: None}
    assert state._entries == {}
    assert saved.topk_by_layer[7] is indices
    assert saved.topk_length_by_layer[7] is lengths
    assert saved.latent_kv_by_layer[7] is latent_kv
    saved.latent_kv_by_layer[7].sum().backward()
    torch.testing.assert_close(source.grad, 2 * source.detach())


def test_dsa_state_computes_index_once_then_reuses_and_cleans_it(monkeypatch):
    monkeypatch.setattr(dsa_module, "build_module", lambda *_args, **_kwargs: nn.Identity())
    config = _make_config(dsa_indexer_topk_freq=1, dsa_indexer_skip_topk_offset=0)
    attention = dsa_module.DSAttention(
        config=config,
        submodules=dsa_module.DSAttentionSubmodules(indexer=None),
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        pg_collection=SimpleNamespace(tp=None, cp=None),
        is_mtp_layer=True,
    )
    sequence_length = 4
    indexer_calls = []
    topk_calls = []
    sparse_topk_pointers = []
    source_topk = torch.tensor(
        [[[0, -1, -1, -1], [0, 1, -1, -1], [0, 1, 2, -1], [0, 1, 2, 3]]], dtype=torch.int64
    )

    def fake_forward_before_topk(_x, _qr, _packed_seq_params):
        indexer_calls.append(True)
        return (
            torch.randn(sequence_length, 1, 2, 4),
            torch.randn(sequence_length, 1, 4),
            torch.ones(sequence_length, 1, 2),
        )

    def fake_topk(*_args, **_kwargs):
        topk_calls.append(True)
        return torch.empty(0), source_topk

    def fake_sparse_attention(**kwargs):
        sparse_topk_pointers.append(kwargs["topk_indices"].data_ptr())
        return torch.randn(sequence_length, 1, config.hidden_size)

    attention.indexer.forward_before_topk = fake_forward_before_topk
    monkeypatch.setattr(dsa_module, "fused_qk_topk_naive", fake_topk)
    monkeypatch.setattr(dsa_module, "_run_sparse_attention", fake_sparse_attention)

    def run_depth():
        return attention(
            query=torch.randn(
                sequence_length,
                1,
                config.num_attention_heads,
                config.kv_lora_rank + config.qk_pos_emb_head_dim,
            ),
            key=torch.randn(
                sequence_length, 1, 1, config.kv_lora_rank + config.qk_pos_emb_head_dim
            ),
            value=None,
            attention_mask=None,
            x=torch.randn(sequence_length, 1, config.hidden_size),
            qr=torch.randn(sequence_length, 1, config.q_lora_rank),
            up_v_weight=torch.randn(
                config.num_attention_heads, config.v_head_dim, config.kv_lora_rank
            ),
            attn_mask_type=AttnMaskType.causal,
        )

    forward_state = get_forward_sharing_state(config=config)
    with (
        forward_sharing_lifetime(config=config),
        mtp_repeated_sharing_lifetime(forward_state, config.num_layers + 1) as sharing_call,
    ):
        outputs = []
        for depth in range(config.mtp_num_layers):
            sharing_call[config.num_layers + 1] = depth == 0
            outputs.append(run_depth())

        shared_payload = forward_state.get(dsa_module._DSAMTPRepeatedSharingPayload)
        assert shared_payload is not None
    assert forward_state._entries == {}

    assert all(output.shape == outputs[0].shape for output in outputs)
    assert len(indexer_calls) == 1
    assert len(topk_calls) == 1
    assert sparse_topk_pointers == [source_topk.data_ptr()] * config.mtp_num_layers
    index_payload = attention._get_dsa_index_sharing_payload(None, None)
    assert attention.layer_number not in index_payload.topk_by_layer


def test_dsa_state_keeps_latent_kv_attached_to_the_source_graph(monkeypatch):
    monkeypatch.setattr(dsa_module, "build_module", lambda *_args, **_kwargs: nn.Identity())
    attention = dsa_module.DSAttention(
        config=_make_config(mtp_num_layers=2, mtp_repeated_layer_shared_components=[LATENT_KV]),
        submodules=dsa_module.DSAttentionSubmodules(indexer=None),
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        pg_collection=SimpleNamespace(),
        is_mtp_layer=True,
    )
    source_input = torch.randn(4, 1, 8, requires_grad=True)
    latent_kv = source_input.square()
    forward_state = get_forward_sharing_state(config=attention.config)

    with forward_sharing_lifetime(config=attention.config):
        # An interrupted source repeat must not leak its graph into the next repeat.
        with mtp_repeated_sharing_lifetime(forward_state, attention.layer_number) as sharing_call:
            sharing_call[attention.layer_number] = True
            attention._prepare_mtp_sharing_payload(None, None)
            shared_payload = forward_state.get(dsa_module._DSAMTPRepeatedSharingPayload)
            shared_payload.latent_kv_by_layer[attention.layer_number] = torch.randn_like(latent_kv)

        with mtp_repeated_sharing_lifetime(forward_state, attention.layer_number) as sharing_call:
            sharing_call[attention.layer_number] = True
            attention._prepare_mtp_sharing_payload(None, None)
            shared_payload = forward_state.get(dsa_module._DSAMTPRepeatedSharingPayload)
            assert attention.layer_number not in shared_payload.latent_kv_by_layer
            shared_payload.latent_kv_by_layer[attention.layer_number] = latent_kv

            sharing_call[attention.layer_number] = False
            _, payload = attention._prepare_mtp_sharing_payload(None, None)
            reused_latent_kv = payload.latent_kv_by_layer[attention.layer_number]

        assert forward_state.get(_MTPRepeatedSharingPayload) is None
    assert forward_state._entries == {}

    reused_latent_kv.sum().backward()

    assert reused_latent_kv.data_ptr() == latent_kv.data_ptr()
    assert reused_latent_kv.grad_fn is not None
    assert source_input.grad is not None


def test_latent_kv_sharing_publishes_before_combined_fused_early_return(monkeypatch):
    class FakeIndexer(nn.Module):
        def forward_before_topk(self, x, qr, packed_seq_params):
            del qr, packed_seq_params
            return (
                torch.randn(x.size(0), x.size(1), 2, 4),
                torch.randn(x.size(0), x.size(1), 4),
                torch.ones(x.size(0), x.size(1), 2),
            )

    monkeypatch.setattr(dsa_module, "build_module", lambda *_args, **_kwargs: FakeIndexer())
    monkeypatch.setattr(dsa_module.dsa_kernels, "use_fused_dsa_kernels", lambda _config: True)
    fused_key_pointers = []

    def fake_fused_attention(**kwargs):
        fused_key_pointers.append(kwargs["key"].data_ptr())
        query = kwargs["query"]
        return query.new_zeros((query.size(0), query.size(1), config.hidden_size)), None

    monkeypatch.setattr(dsa_module.dsa_kernels, "run_fused_dsa_attention", fake_fused_attention)
    config = _make_config(
        mtp_num_layers=2,
        mtp_repeated_layer_shared_components=[LATENT_KV],
        dsa_indexer_topk_freq=1,
        dsa_indexer_skip_topk_offset=0,
    )
    attention = dsa_module.DSAttention(
        config=config,
        submodules=dsa_module.DSAttentionSubmodules(indexer=None),
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        pg_collection=SimpleNamespace(tp=None, cp=None),
        is_mtp_layer=True,
    )
    sequence_length = 4
    source_key = torch.randn(
        sequence_length, 1, 1, config.kv_lora_rank + config.qk_pos_emb_head_dim
    )

    def run_depth(key):
        return attention(
            query=torch.randn(
                sequence_length,
                1,
                config.num_attention_heads,
                config.kv_lora_rank + config.qk_pos_emb_head_dim,
            ),
            key=key,
            value=None,
            attention_mask=None,
            x=torch.randn(sequence_length, 1, config.hidden_size),
            qr=torch.randn(sequence_length, 1, config.q_lora_rank),
            up_v_weight=torch.randn(
                config.num_attention_heads, config.v_head_dim, config.kv_lora_rank
            ),
            attn_mask_type=AttnMaskType.causal,
        )

    forward_state = get_forward_sharing_state(config=config)
    with mtp_repeated_sharing_lifetime(forward_state, config.num_layers + 1) as sharing_call:
        sharing_call[config.num_layers + 1] = True
        run_depth(source_key)
        sharing_call[config.num_layers + 1] = False
        run_depth(None)

    assert fused_key_pointers == [source_key.data_ptr(), source_key.data_ptr()]


@pytest.mark.parametrize(
    ("shared_components", "is_mtp_layer", "fused_enabled", "expected_log_count"),
    [
        pytest.param(BOTH_SHARED_COMPONENTS, True, True, 1, id="both-fused"),
        pytest.param([SPARSE_ATTENTION_INDEX], True, True, 1, id="index-only-fused"),
        pytest.param([LATENT_KV], True, True, 0, id="latent-kv-only-fused"),
        pytest.param(None, True, True, 0, id="sharing-off-fused"),
        pytest.param(BOTH_SHARED_COMPONENTS, True, False, 0, id="both-unfused"),
        pytest.param(BOTH_SHARED_COMPONENTS, False, True, 0, id="non-mtp-fused"),
    ],
)
def test_cross_depth_sharing_logs_when_combined_fused_dsa_is_disabled(
    monkeypatch, shared_components, is_mtp_layer, fused_enabled, expected_log_count
):
    logged_messages = []
    monkeypatch.setattr(dsa_module, "_warned_mtp_index_sharing_fused_bypass", False)
    monkeypatch.setattr(
        dsa_module.dsa_kernels, "use_fused_dsa_kernels", lambda _config: fused_enabled
    )
    monkeypatch.setattr(
        dsa_module,
        "log_single_rank",
        lambda _logger, _level, message: logged_messages.append(message),
    )
    monkeypatch.setattr(dsa_module, "build_module", lambda *_args, **_kwargs: nn.Identity())
    config = _make_config(
        mtp_repeated_layer_shared_components=shared_components,
        dsa_indexer_topk_freq=1,
        dsa_indexer_skip_topk_offset=0,
    )

    for _ in range(2):
        dsa_module.DSAttention(
            config=config,
            submodules=dsa_module.DSAttentionSubmodules(indexer=None),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            pg_collection=SimpleNamespace(),
            is_mtp_layer=is_mtp_layer,
        )

    fallback_messages = [
        message
        for message in logged_messages
        if "combined fused DSA kernel does not expose" in message
    ]
    assert len(fallback_messages) == expected_log_count


@pytest.mark.parametrize(
    "shared_components",
    [
        pytest.param([LATENT_KV], id="latent-kv-only"),
        pytest.param([SPARSE_ATTENTION_INDEX], id="sparse-attention-index-only"),
        pytest.param(BOTH_SHARED_COMPONENTS, id="both"),
    ],
)
def test_repeated_mtp_sharing_accepts_ordinary_index_share_skip_layer(
    monkeypatch, shared_components
):
    monkeypatch.setattr(dsa_module, "build_module", lambda *_args, **_kwargs: nn.Identity())
    config = _make_config(num_layers=3, mtp_repeated_layer_shared_components=shared_components)

    attention = dsa_module.DSAttention(
        config=config,
        submodules=dsa_module.DSAttentionSubmodules(indexer=None),
        # The repeated physical MTP layer is global layer N+1=4, a skip layer.
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        pg_collection=SimpleNamespace(),
        is_mtp_layer=True,
    )

    assert attention.skip_topk
    assert attention.source_layer == 3
    assert attention.indexer is None


@pytest.mark.parametrize("source_has_length", [False, True])
def test_dsa_state_reuses_ordinary_skip_index_across_mtp_depths(monkeypatch, source_has_length):
    """Repeated state may consume, but must not alias or remove, an ordinary source slot."""
    tracked_losses = []
    sparse_topk_pointers = []
    sparse_length_pointers = []
    monkeypatch.setattr(
        dsa_module.DSAIndexerLossLoggingHelper,
        "save_loss_to_tracker",
        staticmethod(lambda **kwargs: tracked_losses.append(kwargs["loss"])),
    )
    original_sparse_attention = dsa_module._run_sparse_attention

    def observed_sparse_attention(*args, **kwargs):
        sparse_topk_pointers.append(kwargs["topk_indices"].data_ptr())
        sparse_length_pointers.append(
            None if kwargs["topk_length"] is None else kwargs["topk_length"].data_ptr()
        )
        return original_sparse_attention(*args, **kwargs)

    monkeypatch.setattr(dsa_module, "_run_sparse_attention", observed_sparse_attention)
    config = _make_config(num_layers=3)
    attention = dsa_module.DSAttention(
        config=config,
        submodules=dsa_module.DSAttentionSubmodules(indexer=None),
        # The repeated physical MTP layer is global layer N+1=4; its source is layer 3.
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        pg_collection=SimpleNamespace(tp=None, cp=None),
        is_mtp_layer=True,
    )
    sequence_length = 4
    source_topk = torch.tensor(
        [[[0, -1, -1, -1], [0, 1, -1, -1], [0, 1, 2, -1], [0, 1, 2, 3]]], dtype=torch.int64
    )
    index_payload = attention._get_dsa_index_sharing_payload(None, None)
    index_payload.topk_by_layer[attention.source_layer] = source_topk
    source_length = torch.arange(1, sequence_length + 1, dtype=torch.int32).view(1, -1)
    if source_has_length:
        index_payload.topk_length_by_layer[attention.source_layer] = source_length

    def run_depth():
        query = torch.randn(
            sequence_length,
            1,
            config.num_attention_heads,
            config.kv_lora_rank + config.qk_pos_emb_head_dim,
        )
        key = torch.randn(sequence_length, 1, 1, config.kv_lora_rank + config.qk_pos_emb_head_dim)
        return attention(
            query=query,
            key=key,
            value=None,
            attention_mask=None,
            x=torch.randn(sequence_length, 1, config.hidden_size),
            qr=torch.randn(sequence_length, 1, config.q_lora_rank),
            up_v_weight=torch.randn(
                config.num_attention_heads, config.v_head_dim, config.kv_lora_rank
            ),
            attn_mask_type=AttnMaskType.causal,
        )

    forward_state = get_forward_sharing_state(config=config)
    with mtp_repeated_sharing_lifetime(forward_state, config.num_layers + 1) as sharing_call:
        outputs = []
        for depth in range(config.mtp_num_layers):
            sharing_call[config.num_layers + 1] = depth == 0
            outputs.append(run_depth())

        shared_payload = forward_state.get(dsa_module._DSAMTPRepeatedSharingPayload)
        assert shared_payload is not None
    assert forward_state.get(_MTPRepeatedSharingPayload) is None

    assert all(output.shape == outputs[0].shape for output in outputs)
    assert attention.indexer is None
    assert tracked_losses == []
    assert sparse_topk_pointers == [source_topk.data_ptr()] * config.mtp_num_layers
    expected_length_pointer = source_length.data_ptr() if source_has_length else None
    assert sparse_length_pointers == [expected_length_pointer] * config.mtp_num_layers
    assert index_payload.topk_by_layer[attention.source_layer].data_ptr() == source_topk.data_ptr()
    assert attention.layer_number not in index_payload.topk_by_layer
    if source_has_length:
        assert (
            index_payload.topk_length_by_layer[attention.source_layer].data_ptr()
            == source_length.data_ptr()
        )
    else:
        assert attention.source_layer not in index_payload.topk_length_by_layer
    assert attention.layer_number not in index_payload.topk_length_by_layer


@pytest.mark.parametrize(
    "shared_components",
    [
        pytest.param([LATENT_KV], id="latent-kv-only"),
        pytest.param([SPARSE_ATTENTION_INDEX], id="sparse-attention-index-only"),
        pytest.param(BOTH_SHARED_COMPONENTS, id="both"),
    ],
)
def test_mtp_block_and_layer_publish_one_repeat(shared_components):
    config = _make_config(mtp_repeated_layer_shared_components=shared_components)
    observed_repeats = []

    class FakeRepeatedLayer:
        def __init__(self):
            self.config = config

        def __call__(
            self,
            input_ids,
            position_ids,
            hidden_states,
            attention_mask,
            padding_mask=None,
            **kwargs,
        ):
            forward_state = get_forward_sharing_state(
                kwargs.get("packed_seq_params"), attention_mask, self.config
            )
            observed_repeats.append(
                is_mtp_repeated_sharing_source(forward_state, config.num_layers + 1)
            )
            return hidden_states + 1, input_ids, position_ids, padding_mask

    block = SimpleNamespace(
        config=config, vp_stage=None, mtp_use_repeated_layer=True, layers=[FakeRepeatedLayer()]
    )
    hidden_states = torch.randn(4, 1, 3)

    output = MultiTokenPredictionBlock.forward(
        block,
        input_ids=torch.arange(4).view(1, 4),
        position_ids=torch.arange(4).view(1, 4),
        hidden_states=hidden_states,
        attention_mask=None,
    )

    assert output.shape == (16, 1, 3)
    assert observed_repeats == [True, False, False]
    assert get_forward_sharing_state(config=config)._entries == {}


@pytest.mark.parametrize("nested", [False, True])
def test_mtp_block_cleans_dsa_payload_when_a_later_depth_fails(monkeypatch, nested):
    """The config fallback must not retain a source graph after stack unwind."""
    monkeypatch.setattr(dsa_module, "build_module", lambda *_args, **_kwargs: nn.Identity())
    attention = dsa_module.DSAttention(
        config=_make_config(
            num_layers=3, mtp_repeated_layer_shared_components=BOTH_SHARED_COMPONENTS
        ),
        submodules=dsa_module.DSAttentionSubmodules(indexer=None),
        # The repeated physical MTP layer is global layer N+1=4; its source is layer 3.
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        pg_collection=SimpleNamespace(),
        is_mtp_layer=True,
    )
    forward_state = get_forward_sharing_state(config=attention.config)
    ordinary_source_topk = torch.zeros((1, 4, 4), dtype=torch.int64)
    source_input = torch.randn(4, 1, 8, requires_grad=True)

    class FailingRepeatedLayer:
        def __init__(self):
            self.config = attention.config

        def __call__(self, input_ids, position_ids, hidden_states, padding_mask=None, **_kwargs):
            if not is_mtp_repeated_sharing_source(forward_state, attention.layer_number):
                raise RuntimeError("injected depth failure")
            _, shared_payload = attention._prepare_mtp_sharing_payload(None, None)
            shared_payload.latent_kv_by_layer[attention.layer_number] = source_input.square()
            shared_payload.topk_by_layer[attention.layer_number] = ordinary_source_topk
            return hidden_states + 1, input_ids, position_ids, padding_mask

    block = SimpleNamespace(
        config=attention.config,
        vp_stage=None,
        mtp_use_repeated_layer=True,
        layers=[FailingRepeatedLayer()],
    )

    outer_lifetime = forward_sharing_lifetime(config=attention.config) if nested else nullcontext()
    with outer_lifetime:
        if nested:
            index_payload = attention._get_dsa_index_sharing_payload(None, None)
            index_payload.topk_by_layer[attention.source_layer] = ordinary_source_topk
        with pytest.raises(RuntimeError, match="injected depth failure"):
            MultiTokenPredictionBlock.forward(
                block,
                input_ids=torch.arange(4).view(1, 4),
                position_ids=torch.arange(4).view(1, 4),
                hidden_states=torch.randn(4, 1, 3),
                attention_mask=None,
            )
        assert forward_state.get(_MTPRepeatedSharingPayload) is None
        if nested:
            assert forward_state.get(dsa_module._DSAIndexSharingPayload) is index_payload
            assert index_payload.topk_by_layer[attention.source_layer] is ordinary_source_topk
            assert forward_state.get(dsa_module._DSAMTPRepeatedSharingPayload) is not None
    assert forward_state._entries == {}


@pytest.mark.parametrize(
    "callable_object",
    [
        MultiTokenPredictionLayer._proj_and_transformer_layer,
        TransformerLayer._forward_attention,
        HyperConnectionTransformerLayer._forward_attention,
        AbsorbedMLASelfAttention.forward,
        dsa_module.DSAttention.forward,
    ],
)
def test_attention_call_signatures_do_not_expose_mtp_prediction_depth(callable_object):
    assert "mtp_prediction_depth" not in inspect.signature(callable_object).parameters


def test_mtp_layer_preserves_inner_signature_without_prediction_depth():
    calls = []

    class StrictLegacyTransformerLayer:
        def __call__(
            self,
            hidden_states,
            attention_mask=None,
            context=None,
            context_mask=None,
            rotary_pos_emb=None,
            rotary_pos_cos=None,
            rotary_pos_sin=None,
            attention_bias=None,
            packed_seq_params=None,
            sequence_len_offset=None,
            padding_mask=None,
            input_ids=None,
            *,
            inference_params=None,
        ):
            calls.append(True)
            return hidden_states, None

    layer = SimpleNamespace(
        config=SimpleNamespace(fp8=None, sequence_parallel=False),
        sequence_parallel=False,
        mtp_layer_pattern=None,
        mtp_model_layer=StrictLegacyTransformerLayer(),
        mhc_enabled=True,
        _concat_embeddings=lambda hidden_states, _decoder_input: hidden_states,
    )
    hidden_states = torch.randn(4, 1, 3)

    output = MultiTokenPredictionLayer._proj_and_transformer_layer(
        layer, hidden_states=hidden_states, decoder_input=torch.randn_like(hidden_states)
    )

    assert output is hidden_states
    assert calls == [True]


@pytest.mark.parametrize(
    "shared_components", [pytest.param(None, id="none"), pytest.param([], id="empty")]
)
def test_mtp_block_preserves_legacy_layer_signature_when_sharing_is_disabled(shared_components):
    observed_depths = []

    class LegacyRepeatedLayer:
        def __call__(
            self,
            *,
            input_ids,
            position_ids,
            hidden_states,
            attention_mask,
            padding_mask,
            inference_params,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            packed_seq_params,
            sequence_roll_context,
            roll_depth,
            sequence_len_offset,
            embedding,
        ):
            observed_depths.append(roll_depth)
            return hidden_states + 1, input_ids, position_ids, padding_mask

    block = SimpleNamespace(
        config=SimpleNamespace(
            pipeline_model_parallel_size=1,
            mtp_num_layers=2,
            mtp_detach_heads=False,
            mtp_repeated_layer_shared_components=shared_components,
        ),
        vp_stage=None,
        mtp_use_repeated_layer=True,
        layers=[LegacyRepeatedLayer()],
    )

    output = MultiTokenPredictionBlock.forward(
        block,
        input_ids=torch.arange(4).view(1, 4),
        position_ids=torch.arange(4).view(1, 4),
        hidden_states=torch.randn(4, 1, 3),
        attention_mask=None,
    )

    assert output.shape == (12, 1, 3)
    assert observed_depths == [0, 1]


def test_cross_depth_sharing_rejects_hybrid_mtp_pattern():
    config = _make_config()

    with pytest.raises(ValueError, match="GPT MTP path only"):
        MultiTokenPredictionLayer(
            config=config,
            submodules=SimpleNamespace(mtp_model_layer=None),
            pg_collection=SimpleNamespace(cp=None, tp=None),
            mtp_layer_pattern="M",
        )


def _rope_positions(total_tokens: int, segment_lengths: list[int] | None, device) -> torch.Tensor:
    if segment_lengths is None:
        return torch.arange(total_tokens, device=device)
    return torch.cat([torch.arange(length, device=device) for length in segment_lengths])


def _apply_rope(
    x: torch.Tensor, positions: torch.Tensor, base: float, interleaved: bool
) -> torch.Tensor:
    dtype = x.dtype
    dim = x.size(-1)
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=x.device) / dim))
    freqs = torch.outer(positions.float(), freqs)
    freqs = torch.polar(torch.ones_like(freqs), freqs).view(x.size(0), 1, 1, -1)
    if interleaved:
        x_pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    else:
        x_pairs = x.float().reshape(*x.shape[:-1], 2, -1).transpose(-1, -2).contiguous()
    x_complex = torch.view_as_complex(x_pairs)
    output = torch.view_as_real(x_complex * freqs).flatten(-2)
    if not interleaved:
        output = torch.cat([output[..., 0::2], output[..., 1::2]], dim=-1)
    return output.to(dtype=dtype)


def _causal_segment_mask(
    total_tokens: int, segment_lengths: list[int] | None, device
) -> torch.Tensor:
    valid = torch.zeros((total_tokens, total_tokens), dtype=torch.bool, device=device)
    start = 0
    for length in segment_lengths or [total_tokens]:
        valid[start : start + length, start : start + length] = torch.tril(
            torch.ones((length, length), dtype=torch.bool, device=device)
        )
        start += length
    return valid


class _NativeSharedAbsorbedDSA(nn.Module):
    """PyTorch-native DSA reference with explicit cross-depth tensor reuse."""

    _PARAMETER_MAP = {
        "q_down_weight": "linear_q_down_proj.weight",
        "q_norm_weight": "q_layernorm.weight",
        "q_up_weight": "linear_q_up_proj.weight",
        "kv_down_weight": "linear_kv_down_proj.weight",
        "kv_norm_weight": "kv_layernorm.weight",
        "kv_up_weight": "linear_kv_up_proj.weight",
        "output_weight": "linear_proj.weight",
        "index_q_weight": "core_attention.indexer.linear_wq_b.weight",
        "index_k_weight": "core_attention.indexer.linear_wk.weight",
        "index_k_norm_weight": "core_attention.indexer.k_norm.weight",
        "index_k_norm_bias": "core_attention.indexer.k_norm.bias",
        "index_weight_proj": "core_attention.indexer.linear_weights_proj.weight",
    }

    def __init__(self, real_attention, config: MLATransformerConfig):
        super().__init__()
        self.config = config
        real_parameters = dict(real_attention.named_parameters())
        for native_name, real_name in self._PARAMETER_MAP.items():
            setattr(self, native_name, nn.Parameter(real_parameters[real_name].detach().clone()))

    def _project_query(self, hidden_states: torch.Tensor, positions: torch.Tensor):
        qr = F.linear(hidden_states, self.q_down_weight)
        qr = F.rms_norm(
            qr, (self.config.q_lora_rank,), self.q_norm_weight, self.config.layernorm_epsilon
        )
        query = F.linear(qr, self.q_up_weight).view(
            hidden_states.size(0),
            hidden_states.size(1),
            self.config.num_attention_heads,
            self.config.qk_head_dim + self.config.qk_pos_emb_head_dim,
        )
        query_nope, query_rope = torch.split(
            query, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1
        )
        query_rope = _apply_rope(query_rope, positions, self.config.rotary_base, interleaved=True)

        kv_up = self.kv_up_weight.view(
            self.config.num_attention_heads,
            self.config.qk_head_dim + self.config.v_head_dim,
            self.config.kv_lora_rank,
        )
        k_up = kv_up[:, : self.config.qk_head_dim]
        query_absorbed = torch.einsum("sbhd,hdc->sbhc", query_nope, k_up)
        return torch.cat([query_absorbed, query_rope], dim=-1), qr, kv_up

    def _project_key(self, hidden_states: torch.Tensor, positions: torch.Tensor):
        kv_combined = F.linear(hidden_states, self.kv_down_weight)
        kv_latent, key_rope = torch.split(
            kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
        )
        kv_latent = F.rms_norm(
            kv_latent,
            (self.config.kv_lora_rank,),
            self.kv_norm_weight,
            self.config.layernorm_epsilon,
        )
        key_rope = _apply_rope(
            key_rope.unsqueeze(2), positions, self.config.rotary_base, interleaved=True
        )
        return torch.cat([kv_latent.unsqueeze(2), key_rope], dim=-1)

    def _compute_topk(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        detached_hidden = hidden_states.detach()
        qr = qr.detach()
        query = F.linear(qr, self.index_q_weight).view(
            hidden_states.size(0),
            hidden_states.size(1),
            self.config.dsa_indexer_n_heads,
            self.config.dsa_indexer_head_dim,
        )
        query_rope, query_nope = torch.split(
            query,
            [
                self.config.qk_pos_emb_head_dim,
                self.config.dsa_indexer_head_dim - self.config.qk_pos_emb_head_dim,
            ],
            dim=-1,
        )
        query_rope = _apply_rope(query_rope, positions, self.config.rotary_base, interleaved=False)
        query = torch.cat([query_rope, query_nope], dim=-1)

        key = F.linear(detached_hidden, self.index_k_weight)
        key = F.layer_norm(
            key,
            (self.config.dsa_indexer_head_dim,),
            self.index_k_norm_weight,
            self.index_k_norm_bias,
            self.config.dsa_indexer_k_norm_epsilon or self.config.layernorm_epsilon,
        )
        key = key.unsqueeze(2)
        key_rope, key_nope = torch.split(
            key,
            [
                self.config.qk_pos_emb_head_dim,
                self.config.dsa_indexer_head_dim - self.config.qk_pos_emb_head_dim,
            ],
            dim=-1,
        )
        key_rope = _apply_rope(key_rope, positions, self.config.rotary_base, interleaved=False)
        key = torch.cat([key_rope, key_nope], dim=-1).squeeze(2)

        weights = F.linear(detached_hidden, self.index_weight_proj)
        weights = weights * (self.config.dsa_indexer_n_heads**-0.5)
        weights = weights * (self.config.dsa_indexer_head_dim**-0.5)
        scores = torch.einsum("sbhd,tbd->bsht", query.float(), key.float())
        scores = (scores * weights.transpose(0, 1).unsqueeze(-1)).sum(dim=2)
        scores = scores.masked_fill(~valid.unsqueeze(0), float("-inf"))
        topk_scores, topk = scores.topk(min(self.config.dsa_indexer_topk, key.size(0)), dim=-1)
        return topk.masked_fill(topk_scores == float("-inf"), -1)

    def forward_depth(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        valid: torch.Tensor,
        shared_key: torch.Tensor | None,
        shared_topk: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query, qr, kv_up = self._project_query(hidden_states, positions)
        key = self._project_key(hidden_states, positions) if shared_key is None else shared_key
        topk = (
            self._compute_topk(hidden_states, qr, positions, valid)
            if shared_topk is None
            else shared_topk
        )

        scores = torch.einsum("sbhd,tbnd->bhst", query.float(), key.float())
        scores = scores * (self.config.qk_head_dim + self.config.qk_pos_emb_head_dim) ** -0.5
        selected = torch.zeros_like(valid, dtype=torch.int32).unsqueeze(0)
        selected.scatter_add_(-1, topk.clamp_min(0), (topk >= 0).to(dtype=selected.dtype))
        sparse_valid = valid.unsqueeze(0) & (selected > 0)
        scores = scores.masked_fill(~sparse_valid.unsqueeze(1), float("-inf"))
        probabilities = torch.softmax(scores, dim=-1).to(dtype=key.dtype)
        latent_value = key[..., : self.config.kv_lora_rank]
        latent_output = torch.einsum("bhst,tbnd->sbhd", probabilities, latent_value)
        v_up = kv_up[:, self.config.qk_head_dim :]
        output = torch.einsum("sbhc,hdc->sbhd", latent_output, v_up)
        output = F.linear(output.reshape(*hidden_states.shape[:-1], -1), self.output_weight)
        return output, key, topk


def _assert_similarity(left: torch.Tensor, right: torch.Tensor, tolerance: float = 2e-2):
    assert torch.isfinite(left).all()
    assert torch.isfinite(right).all()
    cosine = F.cosine_similarity(
        left.flatten().double().unsqueeze(0), right.flatten().double().unsqueeze(0)
    ).item()
    left = left.double()
    right = right.double()
    denominator = (left.square() + right.square()).sum()
    similarity = 1.0 if denominator == 0 else (2.0 * (left * right).sum() / denominator).item()
    assert cosine > 1 - tolerance
    assert similarity > 1 - tolerance


@pytest.mark.parametrize(
    (
        "shared_components",
        "shares_latent_kv",
        "shares_sparse_attention_index",
        "segment_lengths",
        "apply_rope_fusion",
        "topk_frequency",
    ),
    [
        pytest.param([LATENT_KV], True, False, None, False, 4, id="latent-kv-only"),
        pytest.param(
            [SPARSE_ATTENTION_INDEX],
            False,
            True,
            None,
            False,
            1,
            id="sparse-index-only-frequency-one",
        ),
        pytest.param(BOTH_SHARED_COMPONENTS, True, True, None, False, 4, id="both-unpacked"),
        pytest.param(BOTH_SHARED_COMPONENTS, True, True, [5, 4], False, 4, id="both-packed"),
        pytest.param(BOTH_SHARED_COMPONENTS, True, True, None, True, 4, id="both-fused-rope"),
    ],
)
@pytest.mark.launch_on_gb200
def test_repeated_mtp_dsa_eager_components_match_native_reference(
    monkeypatch,
    shared_components,
    shares_latent_kv,
    shares_sparse_attention_index,
    segment_lengths,
    apply_rope_fusion,
    topk_frequency,
):
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
    try:
        model_parallel_cuda_manual_seed(1234)
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        config = _make_config(
            mtp_repeated_layer_shared_components=shared_components,
            apply_rope_fusion=apply_rope_fusion,
            dsa_indexer_topk_freq=topk_frequency,
        )
        attention_spec = get_dsa_module_spec_for_backend(config, backend=TESpecProvider())
        real_attention = (
            build_module(attention_spec, config=config, layer_number=1, is_mtp_layer=True)
            .bfloat16()
            .cuda()
        )
        native_attention = _NativeSharedAbsorbedDSA(real_attention, config).bfloat16().cuda()

        total_tokens = sum(segment_lengths) if segment_lengths is not None else 9
        positions = _rope_positions(total_tokens, segment_lengths, device="cuda")
        valid = _causal_segment_mask(total_tokens, segment_lengths, device="cuda")
        if segment_lengths is None:
            attention_mask = torch.zeros(
                (1, 1, total_tokens, total_tokens), dtype=torch.float32, device="cuda"
            ).masked_fill(~valid.view(1, 1, total_tokens, total_tokens), float("-inf"))
            packed_seq_params = None
        else:
            cu_seqlens = torch.tensor(
                [0, segment_lengths[0], total_tokens], dtype=torch.int32, device="cuda"
            )
            attention_mask = None
            packed_seq_params = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                max_seqlen_q=max(segment_lengths),
                max_seqlen_kv=max(segment_lengths),
            )

        call_counts = {"q": 0, "kv": 0, "indexer": 0, "sparse": 0}
        hooks = [
            real_attention.linear_q_down_proj.register_forward_hook(
                lambda *_args: call_counts.__setitem__("q", call_counts["q"] + 1)
            ),
            real_attention.linear_kv_down_proj.register_forward_hook(
                lambda *_args: call_counts.__setitem__("kv", call_counts["kv"] + 1)
            ),
            real_attention.core_attention.indexer.linear_wq_b.register_forward_hook(
                lambda *_args: call_counts.__setitem__("indexer", call_counts["indexer"] + 1)
            ),
        ]
        original_sparse_attention = dsa_module._run_sparse_attention

        def counted_sparse_attention(*args, **kwargs):
            call_counts["sparse"] += 1
            return original_sparse_attention(*args, **kwargs)

        monkeypatch.setattr(dsa_module, "_run_sparse_attention", counted_sparse_attention)

        real_inputs = [
            torch.randn(
                total_tokens,
                1,
                config.hidden_size,
                dtype=torch.bfloat16,
                device="cuda",
                requires_grad=True,
            )
            for _ in range(config.mtp_num_layers)
        ]
        native_inputs = [value.detach().clone().requires_grad_(True) for value in real_inputs]
        output_grads = [torch.randn_like(value) for value in real_inputs]

        real_outputs = []
        native_outputs = []
        shared_latent_kv = None
        shared_sparse_index = None
        native_key = native_topk = None
        forward_state = get_forward_sharing_state(
            packed_seq_params=packed_seq_params, attention_mask=attention_mask, config=config
        )
        with mtp_repeated_sharing_lifetime(forward_state, config.num_layers + 1) as sharing_call:
            for depth in range(config.mtp_num_layers):
                sharing_call[config.num_layers + 1] = depth == 0
                real_output, _ = real_attention(
                    real_inputs[depth],
                    attention_mask=attention_mask,
                    packed_seq_params=packed_seq_params,
                )
                if depth == 0:
                    shared_payload = forward_state.get(dsa_module._DSAMTPRepeatedSharingPayload)
                    if shares_latent_kv:
                        shared_latent_kv = shared_payload.latent_kv_by_layer[config.num_layers + 1]
                    if shares_sparse_attention_index:
                        shared_sparse_index = shared_payload.topk_by_layer[config.num_layers + 1]
                real_outputs.append(real_output)

                native_output, current_key, current_topk = native_attention.forward_depth(
                    native_inputs[depth], positions, valid, native_key, native_topk
                )
                if depth == 0:
                    if shares_latent_kv:
                        native_key = current_key
                    if shares_sparse_attention_index:
                        native_topk = current_topk
                native_outputs.append(native_output)

        assert forward_state.get(_MTPRepeatedSharingPayload) is None

        assert (shared_latent_kv is not None) is shares_latent_kv
        assert (shared_sparse_index is not None) is shares_sparse_attention_index
        if shared_latent_kv is not None:
            assert shared_latent_kv.grad_fn is not None
        for output, reference in zip(real_outputs, native_outputs):
            _assert_similarity(output, reference)

        torch.autograd.backward(real_outputs, output_grads)
        torch.autograd.backward(native_outputs, output_grads)
        for real_input, native_input in zip(real_inputs, native_inputs):
            _assert_similarity(real_input.grad, native_input.grad)

        real_parameters = dict(real_attention.named_parameters())
        for native_name, real_name in native_attention._PARAMETER_MAP.items():
            native_parameter = getattr(native_attention, native_name)
            real_parameter = real_parameters[real_name]
            if native_parameter.grad is None or real_parameter.grad is None:
                assert native_parameter.grad is None and real_parameter.grad is None
                continue
            _assert_similarity(real_parameter.grad, native_parameter.grad)

        expected_depths = config.mtp_num_layers
        assert call_counts == {
            "q": expected_depths,
            "kv": 1 if shares_latent_kv else expected_depths,
            "indexer": 1 if shares_sparse_attention_index else expected_depths,
            "sparse": expected_depths,
        }
        for hook in hooks:
            hook.remove()
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.launch_on_gb200
def test_tp2_cp2_reuses_canonical_cp_global_latent_kv(monkeypatch):
    """A consumer must not gather or reorder the already-canonical source key again."""
    if Utils.world_size < 4:
        pytest.skip("TP2+CP2 MTP latent-KV sharing requires at least four distributed ranks")
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, context_parallel_size=2)
    try:
        model_parallel_cuda_manual_seed(3917)
        model_parallel_rank = (
            parallel_state.get_context_parallel_rank()
            * parallel_state.get_tensor_model_parallel_world_size()
            + parallel_state.get_tensor_model_parallel_rank()
        )
        torch.manual_seed(3917 + model_parallel_rank)
        torch.cuda.manual_seed(3917 + model_parallel_rank)

        config = _make_config(
            mtp_num_layers=2,
            tensor_model_parallel_size=2,
            context_parallel_size=2,
            sequence_parallel=True,
            cp_comm_type="all_gather",
            mtp_repeated_layer_shared_components=[LATENT_KV],
        )
        attention_spec = get_dsa_module_spec_for_backend(config, backend=TESpecProvider())
        attention = (
            build_module(
                attention_spec,
                config=config,
                layer_number=1,
                cp_comm_type=config.cp_comm_type,
                is_mtp_layer=True,
            )
            .bfloat16()
            .cuda()
        )
        core_attention = attention.core_attention
        gather_counts = {"tp": 0, "cp": 0}
        sparse_key_pointers = []
        original_sparse_attention = dsa_module._run_sparse_attention

        def observed_sparse_attention(*args, **kwargs):
            sparse_key_pointers.append(kwargs["key"].data_ptr())
            return original_sparse_attention(*args, **kwargs)

        monkeypatch.setattr(dsa_module, "_run_sparse_attention", observed_sparse_attention)
        original_gather = dsa_module.gather_from_sequence_parallel_region

        def observed_gather(tensor, group=None, **kwargs):
            if tensor.size(-1) == config.kv_lora_rank + config.qk_pos_emb_head_dim:
                if group is parallel_state.get_tensor_model_parallel_group():
                    gather_counts["tp"] += 1
                elif group is parallel_state.get_context_parallel_group():
                    gather_counts["cp"] += 1
            return original_gather(tensor, group=group, **kwargs)

        monkeypatch.setattr(dsa_module, "gather_from_sequence_parallel_region", observed_gather)

        local_tokens = 8
        cp_local_tokens = local_tokens * config.tensor_model_parallel_size
        global_tokens = cp_local_tokens * config.context_parallel_size
        local_heads = config.num_attention_heads // config.tensor_model_parallel_size
        absorbed_dim = config.kv_lora_rank + config.qk_pos_emb_head_dim
        producer_query = torch.randn(
            local_tokens, 1, local_heads, absorbed_dim, dtype=torch.bfloat16, device="cuda"
        )
        producer_key = torch.randn(
            local_tokens,
            1,
            1,
            absorbed_dim,
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        )
        producer_x = torch.randn(
            local_tokens, 1, config.hidden_size, dtype=torch.bfloat16, device="cuda"
        )
        producer_qr = torch.randn(
            local_tokens, 1, config.q_lora_rank, dtype=torch.bfloat16, device="cuda"
        )
        consumer_query = torch.randn_like(producer_query, requires_grad=True)
        consumer_x = torch.randn_like(producer_x)
        consumer_qr = torch.randn_like(producer_qr)
        up_v_weight = attention._get_v_up_weight()

        forward_state = get_forward_sharing_state(config=config)
        with mtp_repeated_sharing_lifetime(forward_state, config.num_layers + 1) as sharing_call:
            sharing_call[config.num_layers + 1] = True
            core_attention(
                query=producer_query,
                key=producer_key,
                value=None,
                attention_mask=None,
                x=producer_x,
                qr=producer_qr,
                attn_mask_type=AttnMaskType.causal,
                up_v_weight=up_v_weight,
            )
            shared_latent_kv = forward_state.get(
                dsa_module._DSAMTPRepeatedSharingPayload
            ).latent_kv_by_layer[config.num_layers + 1]
            sharing_call[config.num_layers + 1] = False
            consumer_output = core_attention(
                query=consumer_query,
                key=None,
                value=None,
                attention_mask=None,
                x=consumer_x,
                qr=consumer_qr,
                attn_mask_type=AttnMaskType.causal,
                up_v_weight=up_v_weight,
            )
        assert forward_state.get(_MTPRepeatedSharingPayload) is None
        consumer_output.float().square().mean().backward()

        assert shared_latent_kv.size(0) == global_tokens
        assert producer_key.grad is not None and producer_key.grad.float().norm() > 0
        assert consumer_query.grad is not None and consumer_query.grad.float().norm() > 0
        assert gather_counts == {"tp": 1, "cp": 1}
        assert sparse_key_pointers == [shared_latent_kv.data_ptr()] * 2
    finally:
        Utils.destroy_model_parallel()
