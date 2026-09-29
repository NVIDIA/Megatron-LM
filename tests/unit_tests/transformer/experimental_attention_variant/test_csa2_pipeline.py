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
from megatron.core.models.hybrid.hybrid_model import get_hybrid_state_components
from megatron.core.models.hybrid.hybrid_stack_adapter import HybridStatePayload
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.csa import (
    CompressedSparseAttentionSubmodules,
    CompressorSubmodules,
)
from megatron.core.transformer.experimental_attention_variant.csa2 import (
    CompressedSparseAttention2,
    CSA2Compressor,
    CSA2Indexer,
    CSA2IndexerSubmodules,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_pipeline import (
    build_csa2_pipeline_plan,
)
from megatron.core.transformer.spec_utils import ModuleSpec
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
    assert boundary.field_names == fields
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
        == ("global_kv", "indexer_k", "candidate_indices", "candidate_lengths")
    )
    assert chunks[2].incoming.last_attention_layer == chunks[2].outgoing.last_attention_layer == 6


@pytest.mark.parametrize("use_fused_mhc", [False, True])
@pytest.mark.parametrize("max_seqlen", [0, 1, 7])
@pytest.mark.parametrize("single_pass", [False, True])
def test_prepared_specs_need_no_device_values(use_fused_mhc, max_seqlen, single_pass):
    config = replace(
        _config(torch.bfloat16), use_fused_mhc=use_fused_mhc, mhc_single_pass=single_pass
    )
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
        descriptor = _stack(config, chunk).forward_adapter.pipeline_payload_spec(16, 1, params)[1]
        specs = {spec.name: spec for spec in descriptor.tensor_specs}
        assert specs["cu_seqlens"].shape == specs["cu_seqlens_padded"].shape == (4,)
        assert specs["cu_seqlens"].dtype == specs["cu_seqlens_padded"].dtype == torch.int64
        assert specs["hidden_states"].shape == (
            16,
            1,
            config.hidden_size * config.num_residual_streams,
        )
        assert ("pre_mix" in specs) == single_pass
        if single_pass:
            assert specs["pre_mix"].dtype == torch.float32
        assert descriptor.metadata == (chunk.outgoing.layer_offset, max_seqlen)
        if chunk.outgoing.compress_ratio == 2 and max_seqlen < 2:
            assert specs["global_kv"].shape[0] == 0
    params.max_seqlen_q = torch.empty((), device="meta")
    with pytest.raises(ValueError, match="host integer"):
        plan[0].outgoing.tensor_fields(config, 16, 1, params)


@pytest.mark.parametrize(
    "pattern, kwargs, message",
    [
        ("DE|", {}, "Invalid Hybrid state pipeline placement"),
        ("DEDEDEDEDEDE/DE", {}, "Invalid Hybrid state pipeline placement"),
        ("DE|DE", {}, "Invalid Hybrid state pipeline placement"),
        ("DEDEDE|DEDEDE", {"pp_size": 3}, "Invalid Hybrid state pipeline placement"),
        ("DEDEDE|DEDEDE", {"qkv_format": "bshd"}, "Invalid Hybrid state pipeline placement"),
        ("MEMEMEMEMEME", {}, "D/W/E/- sublayers"),
        ("DEEEDE|DEDEDE", {}, "D Hybrid symbol"),
    ],
)
def test_invalid_plan(pattern, kwargs, message):
    with pytest.raises(ValueError, match=message):
        build_csa2_pipeline_plan(_config(), pattern, **kwargs)


class _Attention(nn.Module):
    """Native QKV/output projections around actual CSA2, avoiding TE/GPU setup."""

    uses_attention_mask = False

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


def _stack(config, chunk=None, attention=_Attention, *, state_components=()):
    groups = _groups()
    groups.pp = groups.tp
    stack = HybridStack(
        config,
        HybridStackSubmodules(
            state_components=get_hybrid_state_components(config, state_components),
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
        boundary = chunk.incoming or chunk.outgoing
        if getattr(boundary, "placement", ()):
            stack.forward_adapter.bind_placement("DE" * 6, boundary.placement)
            stack.forward_adapter.configure_cuda_graphs(stack.layers)
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
            assert isinstance(output, HybridStatePayload)
            outputs.append(output)
            payload = output
    return output, outputs


def test_payload_rejects_missing_fields_wrong_sources_and_metadata(monkeypatch):
    _record_losses(monkeypatch)
    config = _config()
    plan = build_csa2_pipeline_plan(config, _pattern((7, 9)))
    stacks = _split_stacks(_stack(config), plan)
    x = torch.randn(5, 2, config.hidden_size, requires_grad=True)
    _, payloads = _run_chunks(stacks, plan, x)
    payload = payloads[0]
    with pytest.raises(ValueError, match="tensor count"):
        replace(payload, tensors=payload.tensors[:-1]).restore()
    with pytest.raises(ValueError, match="pre_mix"):
        replace(
            payload, tensors=(payload.tensors[0], payload.tensors[1][..., :1], *payload.tensors[2:])
        ).restore()
    hidden, context, _ = payload.restore()
    state, mhc = context.cross_layer_state, context.mhc_state
    state.kv_source_layer = 2
    with pytest.raises(ValueError, match="source version"):
        stacks[0].forward_adapter.finalize_forward(hidden, None, context)
    with pytest.raises(ValueError, match="incoming boundary"):
        stacks[-1].set_input_tensor(payload)
    with pytest.raises(ValueError, match="adapter.configure_pipeline"):
        _stack(config).set_input_tensor(payloads[-1])


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
    with pytest.raises(ValueError, match="prefixes do not match"):
        receiver(None, None, packed_seq_params=different)
    assert receiver.input_tensor is None
    receiver.set_input_tensor(payload)
    output = receiver(None, None, packed_seq_params=params)
    assert receiver.input_tensor is None
    output.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("mode", ["full", "graph", "pipeline"])
def test_csa2_nonhybrid_capabilities_do_not_depend_on_mhc(mode):
    from megatron.core.transformer.transformer_block import TransformerBlock

    config = _config(enable_hyper_connections=False)
    assert not config.mhc_single_pass
    if mode == "full":
        config.recompute_granularity = "full"
        config.recompute_method, config.recompute_num_layers = "uniform", 1
    elif mode == "graph":
        config.cuda_graph_impl = "transformer_engine"
    else:
        config.pipeline_model_parallel_size = 2
    with pytest.raises(ValueError, match="CSA2.*requires HybridModel"):
        TransformerBlock(config, None)


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
