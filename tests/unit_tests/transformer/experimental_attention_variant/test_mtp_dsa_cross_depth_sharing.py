# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Coverage for sparse-attention index sharing across repeated MTP invocations."""

import inspect
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from megatron.core import tensor_parallel
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
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.transformer.transformer_layer import (
    HyperConnectionTransformerLayer,
    TransformerLayer,
)
from megatron.core.utils import init_method_normal, scaled_init_method_normal

SPARSE_ATTENTION_INDEX = "sparse_attention_index"


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
        pytest.param([SPARSE_ATTENTION_INDEX], id="sparse-attention-index-only"),
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


def test_mlp_recompute_after_repeat_cleanup_preserves_gradients(monkeypatch):
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
            recompute_granularity="selective" if recompute else None, recompute_modules=["mlp"]
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
            assert tensors.topk_by_layer[second_number] is second_topk
            assert second_number in tensors.topk_length_by_layer
            flags[first_number] = False
            with pytest.raises(RuntimeError, match="requires published source"):
                layers[0]._prepare_mtp_sharing_payload(None, None)
    assert state._entries == {}


def test_mtp_snapshot_copies_dictionaries_and_cleanup_releases_live_tensors():
    config = SimpleNamespace()
    state = get_forward_sharing_state(config=config)
    indices = torch.zeros(2, dtype=torch.int64)
    lengths = torch.ones(2, dtype=torch.int64)
    with forward_sharing_lifetime(config=config):
        with mtp_repeated_sharing_lifetime(state, 7) as flags:
            flags[7] = True
            payload = state.get_or_create(dsa_module._DSAMTPRepeatedSharingPayload)
            payload.topk_by_layer[7] = indices
            payload.topk_length_by_layer[7] = lengths
            snapshot = state.snapshot()
            saved = snapshot.get(dsa_module._DSAMTPRepeatedSharingPayload)
            assert saved is not payload
            assert saved.topk_by_layer is not payload.topk_by_layer
            assert saved.topk_length_by_layer is not payload.topk_length_by_layer
            payload.topk_by_layer.clear()
            payload.topk_length_by_layer.clear()
            flags[7] = False
            assert snapshot.get(_MTPRepeatedSharingPayload).is_source_by_layer[7]
        assert state.get(_MTPRepeatedSharingPayload) is None
        assert state.get(dsa_module._DSAMTPRepeatedSharingPayload) is payload
        with mtp_repeated_sharing_lifetime(state, 7) as fresh_flags:
            assert fresh_flags == {7: None}
    assert state._entries == {}
    assert saved.topk_by_layer[7] is indices
    assert saved.topk_length_by_layer[7] is lengths


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


@pytest.mark.parametrize(
    ("shared_components", "is_mtp_layer", "fused_enabled", "expected_log_count"),
    [
        pytest.param([SPARSE_ATTENTION_INDEX], True, True, 1, id="index-only-fused"),
        pytest.param(None, True, True, 0, id="sharing-off-fused"),
        pytest.param([SPARSE_ATTENTION_INDEX], True, False, 0, id="index-only-unfused"),
        pytest.param([SPARSE_ATTENTION_INDEX], False, True, 0, id="non-mtp-fused"),
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


def test_repeated_mtp_sharing_accepts_ordinary_index_share_skip_layer(monkeypatch):
    monkeypatch.setattr(dsa_module, "build_module", lambda *_args, **_kwargs: nn.Identity())
    config = _make_config(
        num_layers=3, mtp_repeated_layer_shared_components=[SPARSE_ATTENTION_INDEX]
    )

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


def test_mtp_block_and_layer_publish_one_repeat():
    config = _make_config(mtp_repeated_layer_shared_components=[SPARSE_ATTENTION_INDEX])
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
            num_layers=3, mtp_repeated_layer_shared_components=[SPARSE_ATTENTION_INDEX]
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

    class FailingRepeatedLayer:
        def __init__(self):
            self.config = attention.config

        def __call__(self, input_ids, position_ids, hidden_states, padding_mask=None, **_kwargs):
            if not is_mtp_repeated_sharing_source(forward_state, attention.layer_number):
                raise RuntimeError("injected depth failure")
            _, shared_payload = attention._prepare_mtp_sharing_payload(None, None)
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
