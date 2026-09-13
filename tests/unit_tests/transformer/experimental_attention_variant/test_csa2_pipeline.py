# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CSA2 chunk planning, state transport contracts and local split autograd parity.

CPU stack tests use native projection/FFN adapters around the production TransformerLayer,
HybridStack, single-pass mHC, CSA2 compressor/indexer/attention and auxiliary objective.
They exercise the chunk interface without claiming distributed schedule or kernel coverage.
"""

from dataclasses import replace

import pytest
import torch
from torch import nn

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.pipeline_payload import backward_pipeline_payload
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.csa import (
    CompressedSparseAttentionSubmodules,
    CompressorSubmodules,
    get_window_topk_idxs_thd,
)
from megatron.core.transformer.experimental_attention_variant.csa2 import (
    CompressedSparseAttention2,
    CSA2Compressor,
    CSA2Indexer,
    CSA2IndexerSubmodules,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_hybrid_adapter import (
    CSA2HybridAdapter,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_pipeline import (
    CSA2PipelinePayload,
    build_csa2_pipeline_plan,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from tests.unit_tests.transformer.experimental_attention_variant.test_csa2 import (
    _CPUFrequencyTable,
    _groups,
    _Linear,
    _packed,
    _record_losses,
    _RMSNorm,
)
from tests.unit_tests.transformer.experimental_attention_variant.test_dsv41 import _make_config


def _config(dtype=torch.float32, coefficient=0.3, enable_hyper_connections=True):
    return _make_config(
        params_dtype=dtype,
        num_layers=12,
        csa_compress_ratios=[0, 0, 2, 0, 2, 0, 1, 0, 1, 0, 1, 0],
        csa2_kv_source_layers=[2, 6],
        csa2_index_source_layers=[2, 6, 8],
        csa2_candidate_source_layer=6,
        csa2_candidate_topk_blocks=1,
        dsa_indexer_topk=2,
        dsa_indexer_loss_coeff=coefficient,
        dsa_kernel_backend="none",
        use_fused_mhc=False,
        bias_dropout_fusion=False,
        enable_hyper_connections=enable_hyper_connections,
    )


def _pattern(cuts, pattern="DE" * 6):
    points = (0, *cuts, len(pattern))
    return "|".join(pattern[start:end] for start, end in zip(points, points[1:]))


@pytest.mark.parametrize(
    "block, fields, owners",
    [
        (10, ("global_kv", "global_indices"), (16, 16, None)),
        (20, (), (None, None, None)),
        (
            22,
            ("global_kv", "indexer_k", "global_indices", "candidate_indices", "candidate_lengths"),
            (40, 40, 40),
        ),
        (24, ("global_kv", "indexer_k", "candidate_indices", "candidate_lengths"), (40, None, 40)),
        (
            30,
            ("global_kv", "indexer_k", "global_indices", "candidate_indices", "candidate_lengths"),
            (40, 56, 40),
        ),
        (38, ("global_kv", "global_indices"), (40, 72, None)),
    ],
)
def test_reference_layout_live_fields(block, fields, owners):
    ratios = [0, 0] + [2] * 18 + [1] * 20
    config = _make_config(
        num_layers=80,
        csa_compress_ratios=[r for ratio in ratios for r in (ratio, 0)],
        csa2_kv_source_layers=[4, 16, 28, 40],
        csa2_index_source_layers=[4, 16, 28, 40, 48, 56, 64, 72],
        csa2_candidate_source_layer=40,
    )
    chunks = build_csa2_pipeline_plan(config, _pattern((block * 2,), "DE" * 40), pp_size=2)
    boundary = chunks[0].outgoing
    assert boundary is chunks[1].incoming
    assert boundary.field_names == ("hidden_states", "pre_mix", *fields)
    assert (
        boundary.kv_source_layer,
        boundary.index_source_layer,
        boundary.candidate_source_layer,
    ) == owners


def test_vpp_placement_and_ffn_relay():
    chunks = build_csa2_pipeline_plan(_config(), "DEDEDE|D|E|DEDE", pp_size=2)
    assert [(c.pp_rank, c.vp_stage, c.layer_offset) for c in chunks] == [
        (0, 0, 0),
        (1, 0, 6),
        (0, 1, 7),
        (1, 1, 8),
    ]
    # The FFN-only chunk must relay K/candidates to the next Reindex; no top-k consumer remains.
    assert (
        chunks[2].incoming.field_names
        == chunks[2].outgoing.field_names
        == (
            "hidden_states",
            "pre_mix",
            "global_kv",
            "indexer_k",
            "candidate_indices",
            "candidate_lengths",
        )
    )
    assert chunks[2].incoming.last_attention_layer == chunks[2].outgoing.last_attention_layer == 6


@pytest.mark.parametrize("use_fused_mhc", [False, True])
@pytest.mark.parametrize("max_seqlen", [0, 1, 7])
def test_prepared_specs_need_no_device_values(use_fused_mhc, max_seqlen):
    config = replace(_config(torch.bfloat16), use_fused_mhc=use_fused_mhc)
    plan = build_csa2_pipeline_plan(config, _pattern((3, 4, 6, 7, 8, 10)), qkv_format="thd")
    # Meta tensors deliberately have no readable contents. Any item/tolist/cpu
    # in shape planning would fail here, even on hosts without a GPU.
    prefixes = torch.empty(4, dtype=torch.int64, device="meta")
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=prefixes,
        cu_seqlens_kv=prefixes,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
    )
    for chunk in plan[:-1]:
        descriptor = chunk.outgoing.payload_spec(config, 16, 1, params)
        specs = {spec.name: spec for spec in descriptor.tensor_specs}
        assert specs["cu_seqlens"].shape == specs["cu_seqlens_padded"].shape == (4,)
        assert specs["cu_seqlens"].dtype == specs["cu_seqlens_padded"].dtype == torch.int64
        assert specs["pre_mix"].dtype == (torch.bfloat16 if use_fused_mhc else torch.float32)
        assert descriptor.metadata == (chunk.outgoing.layer_offset, max_seqlen)
        if chunk.outgoing.compress_ratio == 2 and max_seqlen < 2:
            assert specs["global_kv"].shape[0] == 0
    params.max_seqlen_q = torch.empty((), device="meta")
    with pytest.raises(ValueError, match="host integer"):
        plan[0].outgoing.payload_spec(config, 16, 1, params)


@pytest.mark.parametrize(
    "pattern, kwargs, message",
    [
        ("DE|", {}, "nonempty"),
        ("DEDEDEDEDEDE/DE", {}, "without MTP"),
        ("DE|DE", {}, "exactly num_layers"),
        ("DEDEDE|DEDEDE", {"pp_size": 3}, "divisible"),
        ("DEDEDE|DEDEDE", {"qkv_format": "bshd"}, "qkv_format"),
        ("DEEEDE|DEDEDE", {}, "D Hybrid symbol"),
    ],
)
def test_invalid_plan(pattern, kwargs, message):
    with pytest.raises(ValueError, match=message):
        build_csa2_pipeline_plan(_config(), pattern, **kwargs)


class _Attention(nn.Module):
    """Native QKV/output projections around actual CSA2, avoiding TE/GPU setup."""

    def __init__(self, config, layer_number, pg_collection, **kwargs):
        super().__init__()
        self.config = config
        modules = CompressedSparseAttentionSubmodules(
            compressor=ModuleSpec(
                CSA2Compressor, submodules=CompressorSubmodules(_Linear, _Linear, _RMSNorm)
            ),
            indexer=ModuleSpec(
                CSA2Indexer, submodules=CSA2IndexerSubmodules(_Linear, _Linear, _RMSNorm, _Linear)
            ),
        )
        self.core = CompressedSparseAttention2(
            config,
            modules,
            layer_number,
            AttnMaskType.causal,
            "self",
            pg_collection,
            rotary_pos_emb=_CPUFrequencyTable(config.qk_pos_emb_head_dim),
        )

        def linear(dim_in, dim_out):
            return nn.Linear(dim_in, dim_out, bias=False, dtype=config.params_dtype)

        self.q = linear(config.hidden_size, config.num_attention_heads * config.v_head_dim)
        self.kv = linear(config.hidden_size, config.v_head_dim)
        self.qr = linear(config.hidden_size, config.q_lora_rank)
        self.out = linear(config.num_attention_heads * config.v_head_dim, config.hidden_size)

    def forward(self, x, *, packed_seq_params=None, csa2_state=None, **kwargs):
        q = self.q(x).unflatten(-1, (self.config.num_attention_heads, self.config.v_head_dim))
        kv = self.kv(x).unsqueeze(2)
        if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
            q, kv = q.squeeze(1), kv.squeeze(1)
        output = self.core(
            q,
            kv,
            kv,
            None,
            x=x,
            qr=self.qr(x),
            csa2_state=csa2_state,
            packed_seq_params=packed_seq_params,
        )
        return self.out(output.reshape(*x.shape[:2], -1)), None


class _FFN(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False, dtype=config.params_dtype
        )

    def forward(self, x, **kwargs):
        return self.proj(x).tanh(), None


def _stack(config, chunk=None, attention=_Attention):
    groups = _groups()
    groups.pp = groups.tp
    stack = HybridStack(
        config,
        HybridStackSubmodules(
            forward_adapter=CSA2HybridAdapter,
            dsa_layer=ModuleSpec(
                TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=_RMSNorm,
                    self_attention=attention,
                    self_attn_bda=get_bias_dropout_add,
                ),
            ),
            moe_layer=ModuleSpec(
                TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    pre_mlp_layernorm=_RMSNorm, mlp=_FFN, mlp_bda=get_bias_dropout_add
                ),
            ),
        ),
        layer_type_list=list("DE" * 6 if chunk is None else chunk.layer_pattern),
        pp_layer_offset=0 if chunk is None else chunk.layer_offset,
        pre_process=chunk is None or chunk.incoming is None,
        post_process=chunk is None or chunk.outgoing is None,
        post_layer_norm=False,
        pg_collection=groups,
    )
    if chunk is not None:
        stack.forward_adapter.configure_pipeline(chunk)
    return stack


def _split_stacks(full, plan):
    stacks = [_stack(full.config, chunk) for chunk in plan]
    for stack, chunk in zip(stacks, plan):
        for local, layer in enumerate(stack.layers):
            layer.load_state_dict(full.layers[chunk.layer_offset + local].state_dict())
    return stacks


def _run_chunks(stacks, plan, x, params=None):
    payload = None
    outputs = []
    for stack, chunk in zip(stacks, plan):
        stack.set_input_tensor(payload)
        output = stack(
            x if chunk.incoming is None else None,
            None,
            packed_seq_params=params if chunk.incoming is None else None,
        )
        assert stack.input_tensor is None
        if chunk.outgoing is not None:
            assert isinstance(output, CSA2PipelinePayload)
            outputs.append(output)
            payload = output
    return output, outputs


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layout", ["sbhd", "thd", "short"])
@pytest.mark.parametrize("coefficient", [0, 0.3])
@pytest.mark.parametrize("cuts", [(6,), (4, 8), (4, 7, 8, 10), (1, 5, 7, 9)])
@pytest.mark.parametrize("enable_hyper_connections", [False, True], ids=["residual", "mhc"])
def test_split_hybrid_forward_backward(
    monkeypatch, dtype, layout, coefficient, cuts, enable_hyper_connections
):
    """Include Full reset, Reindex/Reuse, FFN relay, aux off/on and two live microbatches."""
    torch.manual_seed(391)
    records = _record_losses(monkeypatch)
    config = _config(dtype, coefficient, enable_hyper_connections)
    full = _stack(config)
    plan = build_csa2_pipeline_plan(
        config, _pattern(cuts), qkv_format="thd" if layout == "thd" else "sbhd"
    )
    stacks = _split_stacks(full, plan)
    params = None
    shape = (1, 1) if layout == "short" else (9, 2)
    if layout == "thd":
        params, _, valid = _packed([1, 4, 0, 2], [2, 5, 0, 4], tail=3)
        shape = (valid.numel(), 1)
    pairs = []
    for microbatch in range(2):
        x = torch.randn(*shape, config.hidden_size, dtype=dtype, requires_grad=True)
        reference_x = x.detach().clone().requires_grad_()
        expected = full(reference_x, None, packed_seq_params=params)
        actual, payloads = _run_chunks(stacks, plan, x, params)
        for payload in payloads:
            descriptor = payload.boundary.payload_spec(config, *shape, params)
            assert descriptor.tensor_specs == payload.tensor_specs
            assert descriptor.metadata == payload.metadata
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        pairs.append((actual, expected, x, reference_x, payloads))
    # A new forward must neither mutate nor reuse an outstanding microbatch's state.
    for first, second in zip(pairs[0][-1], pairs[1][-1]):
        assert all(a is not b for a, b in zip(first.tensors, second.tensors))
        assert "indexer_k" not in first.boundary.field_names or next(
            s for s in first.tensor_specs if s.name == "indexer_k"
        ).requires_grad == bool(coefficient)
    for actual, expected, x, reference_x, _ in reversed(pairs):
        probe = torch.randn_like(actual) * 0.1
        (actual * probe).sum().backward()
        (expected * probe).sum().backward()
        torch.testing.assert_close(x.grad, reference_x.grad, atol=0, rtol=0)
    for stack, chunk in zip(stacks, plan):
        for local, layer in enumerate(stack.layers):
            reference = dict(full.layers[chunk.layer_offset + local].named_parameters())
            for name, parameter in layer.named_parameters():
                torch.testing.assert_close(
                    parameter.grad, reference[name].grad, atol=0, rtol=0, msg=name
                )
    assert len(records) == (
        12 if coefficient else 0
    )  # Three supervised layers, two runs, two microbatches.
    assert not any(
        "csa2_state" in key or "pipeline_payload" in key
        for stack in stacks
        for key in stack.state_dict()
    )


@pytest.mark.parametrize(
    "cache_device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="Fused THD indices require CUDA"
            ),
        ),
    ],
)
def test_payload_replay_snapshot_and_thd_caches(monkeypatch, cache_device):
    _record_losses(monkeypatch)
    if cache_device == "cuda":
        pytest.importorskip("cutlass.cute")
    config = _config()
    plan = build_csa2_pipeline_plan(config, _pattern((7, 9)), qkv_format="thd")
    stacks = _split_stacks(_stack(config), plan)
    params, _, valid = _packed([3, 0, 4], [5, 0, 6], tail=2)
    x = torch.randn(valid.numel(), 1, config.hidden_size, requires_grad=True)
    _, payloads = _run_chunks(stacks, plan, x, params)
    payload = payloads[0]
    hidden, state, mhc, restored_params = payload.restore(use_fused_kernels=True)
    _, other, other_mhc, _ = payload.restore(use_fused_kernels=True)
    assert state is not other and mhc is not other_mhc
    assert state.global_kv is other.global_kv and mhc.pre_mix is other_mhc.pre_mix
    assert (
        state.global_kv.requires_grad
        and state.indexer_k.requires_grad
        and mhc.pre_mix.requires_grad
    )
    assert state.fused_indices is state.fused_window_indices is None
    assert state.global_kv_flat.shape == (state.compressed_layout.capacity, config.v_head_dim)
    torch.testing.assert_close(state.global_kv_flat, state.global_kv[:, 0], atol=0, rtol=0)
    torch.testing.assert_close(state.indexer_k_flat, state.indexer_k[:, 0], atol=0, rtol=0)
    # Working-state mutation and later input-buffer reuse cannot alter the exported fields.
    state.global_kv = None
    state.last_layer = 99
    mhc.pre_mix = None
    params.cu_seqlens_q.zero_()
    params.cu_seqlens_q_padded.zero_()
    again = payload.restore()[1]
    assert again.last_layer == 6 and again.global_kv is other.global_kv
    torch.testing.assert_close(again.thd_layout.valid_tokens, valid)
    # Reuse needs only KV/top-k. Its fused addresses are derived at the receiver.
    received = replace(payloads[1], tensors=tuple(t.to(cache_device) for t in payloads[1].tensors))
    _, reuse, _, _ = received.restore(use_fused_kernels=True)
    assert reuse.indexer_k is None and reuse.candidates is None
    assert reuse.global_indices is not None and reuse.global_kv_flat is not None
    core = stacks[-1].layers[1].inner_layer.self_attention.core
    query = torch.empty(
        hidden.shape[0], config.num_attention_heads, config.v_head_dim, device=cache_device
    )
    window = None
    if cache_device == "cpu":
        # Native lowering accepts an explicit window; CUDA uses the fused builder.
        layout = reuse.thd_layout
        window = get_window_topk_idxs_thd(
            config.csa_window_size, layout.cu_seqlens_padded, total_q=hidden.shape[0]
        )
        starts = layout.cu_seqlens_padded[layout.sequence_ids.clamp_min(0)]
        window = torch.where(window >= 0, window + starts[:, None], -1)
        visible = (
            (window >= 0)
            & layout.valid_tokens[:, None]
            & layout.valid_tokens[window.clamp_min(0).long()]
        )
        window = window.masked_fill(~visible, -1)
    actual = core._fused_indices(query, window, reuse.global_indices, reuse)
    assert actual is reuse.fused_indices
    assert core._fused_indices(query, window, reuse.global_indices, reuse) is actual
    for row in range(hidden.shape[0]):
        expected = []
        if valid[row]:
            sequence = reuse.thd_layout.sequence_ids[row].item()
            start = reuse.thd_layout.cu_seqlens_padded[sequence].item()
            expected.extend(range(max(start, row - config.csa_window_size + 1), row + 1))
            expected.extend(
                i + hidden.shape[0] for i in reuse.global_indices[row].tolist() if i >= 0
            )
        length = reuse.fused_topk_length[row].item()
        assert sorted(actual[row, :length].tolist()) == sorted(expected)


def test_payload_rejects_missing_fields_wrong_sources_and_metadata(monkeypatch):
    _record_losses(monkeypatch)
    config = _config()
    plan = build_csa2_pipeline_plan(config, _pattern((7, 9)))
    stacks = _split_stacks(_stack(config), plan)
    x = torch.randn(5, 2, config.hidden_size, requires_grad=True)
    _, payloads = _run_chunks(stacks, plan, x)
    payload = payloads[0]
    with pytest.raises(ValueError, match="required tensor fields"):
        replace(payload, tensors=payload.tensors[:-1]).restore()
    with pytest.raises(ValueError, match="pre_mix"):
        replace(
            payload, tensors=(payload.tensors[0], payload.tensors[1][..., :1], *payload.tensors[2:])
        ).restore()
    hidden, state, mhc, _ = payload.restore()
    state.kv_source_layer = 2
    with pytest.raises(ValueError, match="kv_source_layer=6"):
        payload.boundary.export_payload(hidden, state, mhc)
    with pytest.raises(ValueError, match="incoming boundary"):
        stacks[-1].set_input_tensor(payload)
    with pytest.raises(ValueError, match="adapter.configure_pipeline"):
        _stack(config).set_input_tensor(payloads[-1])


def test_chunk_configuration_and_input_contract(monkeypatch):
    _record_losses(monkeypatch)
    config = _config()
    plan = build_csa2_pipeline_plan(config, _pattern((7,)))
    full = _stack(config)
    with pytest.raises(ValueError, match="does not match this HybridStack segment"):
        full.forward_adapter.configure_pipeline(plan[0])
    first, receiver = _split_stacks(full, plan)
    thd_plan = build_csa2_pipeline_plan(config, _pattern((7,)), qkv_format="thd")
    with pytest.raises(ValueError, match="already configured"):
        receiver.forward_adapter.configure_pipeline(thd_plan[1])
    x = torch.randn(5, 2, config.hidden_size, requires_grad=True)
    payload = first(x, None)
    with pytest.raises(ValueError, match="first CSA2 pipeline chunk"):
        first.set_input_tensor(payload)
    with pytest.raises(ValueError, match="receive a payload"):
        receiver.set_input_tensor(x)
    with pytest.raises(ValueError, match="requires a payload from set_input_tensor"):
        receiver(None, None)


@pytest.mark.parametrize("as_list", [False, True])
def test_model_set_input_tensor_consumes_one_payload_per_forward(monkeypatch, as_list):
    _record_losses(monkeypatch)
    config = _config()
    plan = build_csa2_pipeline_plan(config, _pattern((7,)))
    full = _stack(config)
    first, receiver = _split_stacks(full, plan)
    # Exercise the existing model setter without the CUDA embedding/head constructor.
    model = HybridModel.__new__(HybridModel)
    nn.Module.__init__(model)
    model.decoder = receiver
    x = torch.randn(5, 2, config.hidden_size, requires_grad=True)
    payload = first(x, None)
    model.set_input_tensor([payload] if as_list else payload)
    output = receiver(None, None)
    assert receiver.input_tensor is None
    with pytest.raises(ValueError, match="requires a payload from set_input_tensor"):
        receiver(None, None)
    # Explicitly resubmitting the snapshot is valid and restores a fresh working state.
    model.set_input_tensor([payload] if as_list else payload)
    replay = receiver(None, None)
    torch.testing.assert_close(output, replay, rtol=0, atol=0)
    torch.testing.assert_close(output, full(x, None), rtol=0, atol=0)
    (output.sum() + replay.sum()).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_failed_forward_consumes_payload_and_allows_new_microbatch(monkeypatch):
    _record_losses(monkeypatch)
    config = _config()
    plan = build_csa2_pipeline_plan(config, _pattern((7,)), qkv_format="thd")
    first, receiver = _split_stacks(_stack(config), plan)
    params, _, valid = _packed([3, 4], [5, 6], tail=2)
    different, _, _ = _packed([2, 4], [5, 6], tail=2)
    x = torch.randn(valid.numel(), 1, config.hidden_size, requires_grad=True)
    payload = first(x, None, packed_seq_params=params)
    receiver.set_input_tensor(payload)
    with pytest.raises(ValueError, match="different THD layout"):
        receiver(None, None, packed_seq_params=different)
    assert receiver.input_tensor is None
    receiver.set_input_tensor(payload)
    output = receiver(None, None, packed_seq_params=params)
    assert receiver.input_tensor is None
    output.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_unconfigured_stack_preserves_tensor_input_contract(monkeypatch):
    _record_losses(monkeypatch)
    config = _config()
    stack = _stack(config)
    stack.pre_process = False
    hidden = torch.randn(5, 2, config.hidden_size * config.num_residual_streams, requires_grad=True)
    stack.set_input_tensor(hidden)
    output = stack(None, None)
    assert isinstance(output, torch.Tensor) and stack.input_tensor is hidden
    output.sum().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()


@pytest.mark.parametrize("enable_hyper_connections", [False, True], ids=["residual", "mhc"])
def test_generic_stack_without_adapter_preserves_tensor_input_and_gradients(
    enable_hyper_connections,
):
    config = TransformerConfig(
        num_layers=2,
        hidden_size=8,
        num_attention_heads=2,
        use_cpu_initialization=True,
        enable_hyper_connections=enable_hyper_connections,
        mhc_single_pass=enable_hyper_connections,
        num_residual_streams=4,
        use_fused_mhc=False,
        bias_dropout_fusion=False,
        hidden_dropout=0,
        attention_dropout=0,
    )
    groups = _groups()
    groups.pp = groups.tp
    stack = HybridStack(
        config,
        HybridStackSubmodules(
            mlp_layer=ModuleSpec(
                TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    pre_mlp_layernorm=_RMSNorm, mlp=_FFN, mlp_bda=get_bias_dropout_add
                ),
            )
        ),
        layer_type_list=["-", "-"],
        pre_process=False,
        post_layer_norm=False,
        pg_collection=groups,
    )
    assert stack.forward_adapter is None
    assert hybrid_stack_spec.submodules.forward_adapter is None
    width = config.hidden_size * (config.num_residual_streams if enable_hyper_connections else 1)
    x = torch.randn(5, 2, width, requires_grad=True)
    stack.set_input_tensor(x)
    output = stack(None, None)
    replay = stack(None, None)
    torch.testing.assert_close(output, replay, rtol=0, atol=0)
    assert output.shape == (5, 2, config.hidden_size) and stack.input_tensor is x
    (output.sum() + replay.sum()).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(parameter.grad is not None for parameter in stack.parameters())


@pytest.mark.parametrize("mix_dtype", [torch.float32, torch.bfloat16])
def test_payload_preserves_residual_and_mixing_dtypes(monkeypatch, mix_dtype):
    _record_losses(monkeypatch)
    config = _config(torch.bfloat16)
    plan = build_csa2_pipeline_plan(config, _pattern((7,)))
    stacks = _split_stacks(_stack(config), plan)
    x = torch.randn(5, 2, config.hidden_size, dtype=torch.bfloat16, requires_grad=True)
    _, (payload,) = _run_chunks(stacks, plan, x)
    hidden, mix, *shared = payload.tensors
    payload = replace(payload, tensors=(hidden.float(), mix.to(mix_dtype), *shared))
    restored_hidden, state, mhc, _ = payload.restore()
    specs = {spec.name: spec for spec in payload.tensor_specs}
    assert restored_hidden.dtype == specs["hidden_states"].dtype == torch.float32
    assert mhc.pre_mix.dtype == specs["pre_mix"].dtype == mix_dtype
    assert state.dtype == specs["global_kv"].dtype == torch.bfloat16
    # The first received attention query checks compute dtype, not residual dtype.
    state.validate_forward(8, torch.empty(5, 2, 4, 16, dtype=torch.bfloat16))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layout", ["sbhd", "thd"])
@pytest.mark.parametrize("coefficient", [0, 0.3])
@pytest.mark.parametrize("cuts", [(4,), (7, 8, 10)])
def test_received_leaf_joint_backward_matches_unsplit(
    monkeypatch, dtype, layout, coefficient, cuts
):
    """Sever cross-chunk autograd edges and explicitly return every shared gradient."""
    _record_losses(monkeypatch)
    torch.manual_seed(433)
    config = _config(dtype, coefficient)
    full = _stack(config)
    plan = build_csa2_pipeline_plan(config, _pattern(cuts), qkv_format=layout)
    stacks = _split_stacks(full, plan)
    params = None
    shape = (7, 2)
    if layout == "thd":
        params, _, valid = _packed([1, 4, 0, 2], [2, 6, 0, 3], tail=1)
        shape = (valid.numel(), 1)
    x = torch.randn(*shape, config.hidden_size, dtype=dtype, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    expected = full(reference_x, None, packed_seq_params=params)
    incoming, outputs = [], []
    payload = None
    for i, stack in enumerate(stacks):
        incoming.append(payload)
        stack.set_input_tensor(payload)
        output = stack(x if i == 0 else None, None, packed_seq_params=params if i == 0 else None)
        outputs.append(output)
        if i + 1 < len(stacks):
            tensors = tuple(
                t.detach().clone().requires_grad_(spec.requires_grad)
                for t, spec in zip(output.tensors, output.tensor_specs)
            )
            payload = stacks[i + 1].forward_adapter.make_pipeline_payload(tensors, output.metadata)
            assert all(t.is_leaf for t in payload.tensors)
    probe = torch.randn_like(expected) * 0.1
    (expected * probe).sum().backward()
    gradient = backward_pipeline_payload(incoming[-1], (outputs[-1] * probe).sum(), None)
    for i in reversed(range(len(stacks) - 1)):
        gradient = backward_pipeline_payload(incoming[i], outputs[i], gradient)
    tolerance = dict(atol=2e-6, rtol=5e-5) if dtype == torch.float32 else dict(atol=2e-3, rtol=5e-2)
    torch.testing.assert_close(outputs[-1], expected, atol=0, rtol=0)
    torch.testing.assert_close(x.grad, reference_x.grad, **tolerance)
    for stack, chunk in zip(stacks, plan):
        for local, layer in enumerate(stack.layers):
            reference = dict(full.layers[chunk.layer_offset + local].named_parameters())
            for name, parameter in layer.named_parameters():
                ref = reference[name]
                assert (parameter.grad is None) == (ref.grad is None), name
                if ref.grad is not None:
                    torch.testing.assert_close(parameter.grad, ref.grad, **tolerance, msg=name)


def test_distributed_adapter_binds_both_layouts_and_rejects_other_boundaries():
    from types import SimpleNamespace

    config = _config()
    pattern = _pattern((7, 8, 10))
    adapter = CSA2HybridAdapter(
        config,
        layer_type_list=["E"],
        pp_layer_offset=7,
        pre_process=False,
        post_process=False,
        is_mtp_layer=False,
    )
    factory = adapter.configure_distributed_pipeline(
        pattern, SimpleNamespace(size=lambda: 4, rank=lambda: 1)
    )
    assert set(adapter._pipeline_chunks) == {"sbhd", "thd"}
    assert factory is not None
    with pytest.raises(ValueError, match="incoming pipeline boundary"):
        factory((), (-1, -1))


def test_v41_ordinary_pp_config_and_nonhybrid_guard():
    from megatron.core.transformer.transformer_block import TransformerBlock

    config = replace(_config(), pipeline_model_parallel_size=2, pipeline_dtype=torch.float32)
    assert config.mhc_single_pass
    with pytest.raises(ValueError, match="requires HybridModel"):
        TransformerBlock(config, None)
    with pytest.raises(ValueError, match="overlap"):
        replace(config, overlap_p2p_comm=True)
    vpp_config = replace(config, virtual_pipeline_model_parallel_size=2)
    with pytest.raises(ValueError, match="batch_p2p_comm=False"):
        replace(vpp_config, overlap_p2p_comm=True)
    vpp_config = replace(vpp_config, overlap_p2p_comm=True, batch_p2p_comm=False)
    assert vpp_config.overlap_p2p_comm
    with pytest.raises(ValueError, match="ring_exchange"):
        replace(vpp_config, use_ring_exchange_p2p=True)
    with pytest.raises(ValueError, match="requires HybridModel"):
        TransformerBlock(vpp_config, None)
    vpp_config = replace(vpp_config, overlap_p2p_comm_warmup_flush=True)
    assert vpp_config.overlap_p2p_comm_warmup_flush
    with pytest.raises(ValueError, match="overlap"):
        replace(vpp_config, overlap_p2p_comm=False)
    with pytest.raises(ValueError, match="overlap"):
        replace(vpp_config, batch_p2p_comm=True)


@pytest.mark.parametrize("layout", ["sbhd", "thd"])
def test_distributed_adapter_binds_each_virtual_chunk(layout):
    from types import SimpleNamespace

    config = replace(
        _config(),
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
        pipeline_dtype=torch.float32,
    )
    pattern = _pattern((7, 8, 10))
    plan = build_csa2_pipeline_plan(config, pattern, pp_size=2, qkv_format=layout)
    for chunk in plan:
        adapter = CSA2HybridAdapter(
            config,
            layer_type_list=list(chunk.layer_pattern),
            pp_layer_offset=chunk.layer_offset,
            pre_process=chunk.incoming is None,
            post_process=chunk.outgoing is None,
            is_mtp_layer=False,
        )
        group = SimpleNamespace(size=lambda: 2, rank=lambda: chunk.pp_rank)
        with pytest.raises(ValueError, match="vp_stage"):
            adapter.configure_distributed_pipeline(pattern, group)
        with pytest.raises(ValueError, match="segment count"):
            adapter.configure_distributed_pipeline(_pattern((6,)), group, chunk.vp_stage)
        adapter.configure_distributed_pipeline(pattern, group, chunk.vp_stage)
        assert adapter._pipeline_chunks[layout] == chunk


@pytest.mark.parametrize("version,pp_size", [("v4.1", 2), ("v4.1", 4), ("v4", 2)])
@pytest.mark.parametrize("overlap,warmup_flush", [(False, False), (True, False), (True, True)])
def test_vpp_cli_derives_chunks_and_preserves_legacy_pp2_guard(
    version, pp_size, overlap, warmup_flush
):
    from argparse import ArgumentParser

    from megatron.training.arguments import add_megatron_arguments, validate_args

    pattern = "|".join(["DE"] * (pp_size * 2))
    args = add_megatron_arguments(ArgumentParser()).parse_args(
        [
            "--dsv4-version",
            version,
            "--experimental-attention-variant",
            "dsv4_hybrid",
            "--hybrid-layer-pattern",
            pattern,
            "--pipeline-model-parallel-size",
            str(pp_size),
            *([] if overlap else ["--no-overlap-p2p-communication"]),
            *(["--overlap-p2p-communication-warmup-flush"] if warmup_flush else []),
            "--hidden-size",
            "32",
            "--num-attention-heads",
            "4",
            "--micro-batch-size",
            "1",
            "--global-batch-size",
            "8",
            "--seq-length",
            "16",
            "--max-position-embeddings",
            "16",
            "--train-iters",
            "2",
            "--lr",
            "0.001",
        ]
    )
    args.rank, args.world_size = 0, pp_size
    if version == "v4" and not overlap:
        with pytest.raises(AssertionError, match="greater than 2"):
            validate_args(args)
    else:
        args = validate_args(args)
        assert args.virtual_pipeline_model_parallel_size == 2
        assert args.overlap_p2p_comm == overlap
        assert args.overlap_p2p_comm_warmup_flush == warmup_flush
        assert args.num_layers == pp_size * 4
