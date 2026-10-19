# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MXFP8 dense indexer scorer for ``experimental_attention_variant='dsa'`` (CPU, mocked).

The cuDNN frontend, Transformer Engine quantizer and device checks are replaced by CPU
fakes; these tests pin config validation, argument plumbing, scale-storage sizing, the
explicit errors on paths that cannot honour MXFP8, and the absence of host syncs. They also
pin the indexer-loss score rule: with MXFP8 compute the scorer decides only the selection,
and the selected scores that feed the loss are recomputed in FP32 from the BF16 inputs
(``dsa_cudnn_fe_fakes.Fe131DSA.sparse_indexer_score_recompute_wrapper``), on the ranked and
the every-column exit, while BF16 compute keeps the scorer's own values and indices-only
calls never recompute. The numerics on a real GB300 are covered by
``test_dsa_cudnn_mxfp8_scorer_gpu.py``.
"""

import math
import random
from types import SimpleNamespace

import pytest
import torch

from megatron.core.quantization import indexer_quantization
from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.experimental_attention_variant import dsa as dsa_module
from megatron.core.transformer.experimental_attention_variant import (
    dsa_cudnn_kernels,
    dsa_kernels,
    dsa_layout,
    dsa_masking,
)
from megatron.core.transformer.forward_sharing import forward_sharing_lifetime
from tests.unit_tests.transformer.experimental_attention_variant import dsa_cudnn_fe_fakes as fakes

MXFP8 = dsa_cudnn_kernels.IndexerScorerOptions(precision="mxfp8")
BF16 = dsa_cudnn_kernels.IndexerScorerOptions()
HEADS, HEAD_DIM = 32, 128


@pytest.fixture
def mxfp8_runtime(monkeypatch):
    """Fake FE 1.31 plus a CPU MXFP8 quantizer; returns the recorded quantizer calls."""
    fakes.Fe131DSA.reset()
    quantize_calls = []
    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", fakes.Fe131DSA)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_frontend_version", lambda: (1, 31))
    monkeypatch.setattr(dsa_cudnn_kernels, "_mxfp8_scorer_device_problem", lambda device: None)
    monkeypatch.setattr(indexer_quantization, "HAVE_TE_MXFP8", True)
    monkeypatch.setattr(indexer_quantization, "HAVE_TRITON", True)
    monkeypatch.setattr(
        indexer_quantization,
        "quantize_indexer_mxfp8",
        lambda x, **kwargs: fakes.fake_quantize_indexer_mxfp8(x, calls=quantize_calls, **kwargs),
    )
    return quantize_calls


def _inputs(sq, sk, heads=HEADS, dim=HEAD_DIM, seed=0, w_dtype=torch.bfloat16, w_scale=1.0):
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn((1, sq, heads, dim), generator=generator).to(torch.bfloat16)
    k = torch.randn((1, sk, dim), generator=generator).to(torch.bfloat16)
    w = (torch.rand((1, sq, heads), generator=generator) * w_scale).to(w_dtype)
    return q, k, w


def _ternary_inputs(sq, sk, seed=0):
    """Q/K in {-1, 0, 1}, exact in E4M3, so the MXFP8 and BF16 scorers compute equal scores."""
    generator = torch.Generator().manual_seed(seed)
    q = torch.randint(-1, 2, (1, sq, HEADS, HEAD_DIM), generator=generator).to(torch.bfloat16)
    k = torch.randint(-1, 2, (1, sk, HEAD_DIM), generator=generator).to(torch.bfloat16)
    w = torch.rand((1, sq, HEADS), generator=generator).to(torch.bfloat16)
    return q, k, w


def _score(q, k, w, layout, cp_size, cp_rank, topk, options, return_topk_scores=True):
    starts, ends, cu, max_seqlen = layout
    return dsa_cudnn_kernels._indexer_topk_bshd(
        q,
        k,
        w,
        topk,
        varlen_starts=starts,
        varlen_ends=ends,
        return_scores=False,
        return_topk_scores=return_topk_scores,
        use_local_indexer_varlen=True,
        packed_cu_seqlens_q=cu,
        packed_cu_seqlens_k=cu,
        packed_max_seqlen_q=max_seqlen,
        packed_max_seqlen_k=max_seqlen,
        packed_cp_size=cp_size,
        local_packed_cp_rank=cp_rank,
        scorer_options=options,
    )


# ---------------------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------------------


def _make_mxfp8_config(monkeypatch, **overrides):
    from megatron.core import utils as core_utils
    from tests.unit_tests.transformer.experimental_attention_variant.test_mtp_dsa_cross_depth_sharing import (
        _make_config as make_small_dsa_config,
    )

    # Config validation only; neither a CUDA device nor the kernel dependencies are needed.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(core_utils, "_missing_cudnn_dsa_kernel_dependencies", lambda: [])
    fields = dict(
        dsa_kernel_backend="cudnn",
        attention_backend=AttnBackend.auto,
        dsa_indexer_n_heads=HEADS,
        dsa_indexer_head_dim=HEAD_DIM,
        dsa_indexer_loss_coeff=0.01,
        dsa_indexer_use_sparse_loss=True,
        dsa_indexer_precision="mxfp8",
    )
    fields.update(overrides)
    return make_small_dsa_config(**fields)


@pytest.mark.parametrize("heads", [32, 64])
def test_mxfp8_dsa_config_is_accepted(monkeypatch, heads):
    config = _make_mxfp8_config(monkeypatch, dsa_indexer_n_heads=heads)
    options = dsa_cudnn_kernels._scorer_options_from_config(config)
    assert options.precision == "mxfp8"
    assert options.skip_masked_score_fill


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"dsa_kernel_backend": "none"}, "dsa_kernel_backend='cudnn'"),
        ({"attention_backend": AttnBackend.unfused}, "requires the fused DSA kernels"),
        ({"dsa_indexer_head_dim": 64}, "dsa_indexer_head_dim=128"),
        ({"dsa_indexer_n_heads": 16}, "n_heads of 32 or 64"),
        ({"dsa_indexer_use_sparse_loss": False}, "sparse indexer loss"),
        (
            {
                "dsa_indexer_weights_proj_output_dtype": "fp32",
                "dsa_indexer_weights_proj_use_quantization": False,
            },
            "BF16 indexer weights",
        ),
        ({"dsa_indexer_precision": "fp8"}, "must be 'bf16' or 'mxfp8'"),
        (
            {"dsa_indexer_kpool": 4, "dsa_indexer_topk": 4, "dsa_indexer_loss_coeff": 0.0},
            "KPool selection .* bypasses it",
        ),
        ({"dsa_indexer_kpool": 4, "dsa_indexer_topk": 4}, "KPool selection .* bypasses it"),
    ],
)
def test_mxfp8_dsa_config_validation(monkeypatch, overrides, message):
    with pytest.raises(ValueError, match=message):
        _make_mxfp8_config(monkeypatch, **overrides)


def test_bf16_dsa_config_is_unchanged(monkeypatch):
    config = _make_mxfp8_config(
        monkeypatch, dsa_indexer_precision="bf16", dsa_indexer_n_heads=4, dsa_indexer_head_dim=8
    )
    assert dsa_cudnn_kernels._scorer_options_from_config(config).precision == "bf16"


# ---------------------------------------------------------------------------------------
# No silent BF16 fallback in DSAttention
# ---------------------------------------------------------------------------------------

_FALLBACK_SEQ_LEN = 4
_FUSED_DSA_HOOKS = (
    "run_fused_dsa_attention",
    "run_fused_qk_topk",
    "run_fused_qk_topk_with_loss",
    "run_fused_qk_topk_with_scores",
)


def _fake_forward_before_topk(x, _qr, _packed_seq_params):
    generator = torch.Generator().manual_seed(7)
    rows, batch = x.shape[:2]
    q = torch.randn(rows, batch, 2, 8, generator=generator)
    k = torch.randn(rows, batch, 8, generator=generator)
    weights = torch.rand(rows, batch, 2, generator=generator) + 0.5
    return q, k, weights


def _cpu_dsa_forward(config, *, training):
    """Run one computing DSA layer on CPU with every cuDNN indexer hook declining."""
    attention = dsa_module.DSAttention(
        config=config,
        submodules=dsa_module.DSAttentionSubmodules(indexer=None),
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        pg_collection=SimpleNamespace(tp=None, cp=None),
    )
    assert not attention.skip_topk
    attention.indexer.forward_before_topk = _fake_forward_before_topk
    attention.train(training)
    generator = torch.Generator().manual_seed(11)
    heads = config.num_attention_heads
    absorbed_dim = config.kv_lora_rank + config.qk_pos_emb_head_dim
    mask = torch.ones((1, 1, _FALLBACK_SEQ_LEN, _FALLBACK_SEQ_LEN), dtype=torch.bool).triu(1)
    with forward_sharing_lifetime(None, mask, config):
        return attention(
            query=torch.randn(_FALLBACK_SEQ_LEN, 1, heads, absorbed_dim, generator=generator),
            key=torch.randn(_FALLBACK_SEQ_LEN, 1, 1, absorbed_dim, generator=generator),
            value=None,
            attention_mask=mask,
            x=torch.randn(_FALLBACK_SEQ_LEN, 1, config.hidden_size, generator=generator),
            qr=torch.randn(_FALLBACK_SEQ_LEN, 1, config.q_lora_rank, generator=generator),
            up_v_weight=torch.randn(
                heads, config.v_head_dim, config.kv_lora_rank, generator=generator
            ),
            attn_mask_type=AttnMaskType.causal,
        )


@pytest.fixture
def declining_fused_hooks(monkeypatch):
    """Build DSA layers on CPU and make every fused cuDNN hook decline its layout."""
    monkeypatch.setattr(dsa_module, "build_module", lambda *_args, **_kwargs: torch.nn.Identity())
    calls = []
    for name in _FUSED_DSA_HOOKS:
        monkeypatch.setattr(
            dsa_kernels, name, lambda *_args, _name=name, **_kwargs: calls.append(_name)
        )
    return calls


@pytest.mark.parametrize(
    "training, hook",
    [(True, "run_fused_qk_topk_with_loss"), (False, "run_fused_qk_topk")],
    ids=["topk-with-loss", "topk-only"],
)
def test_mxfp8_raises_when_a_fused_hook_declines(
    monkeypatch, declining_fused_hooks, training, hook
):
    config = _make_mxfp8_config(monkeypatch)
    assert dsa_kernels.use_fused_dsa_kernels(config)
    with pytest.raises(RuntimeError, match="cuDNN indexer hook declined"):
        _cpu_dsa_forward(config, training=training)
    # The hook was offered the layout and declined; nothing reached the BF16 reference.
    assert hook in declining_fused_hooks


@pytest.mark.parametrize("training", [True, False], ids=["topk-with-loss", "topk-only"])
def test_mxfp8_raises_when_fused_kernels_are_disabled_after_validation(
    monkeypatch, declining_fused_hooks, training
):
    # Config validation rejects this combination; a config mutated afterwards must still
    # not reach the BF16 reference scorer.
    config = _make_mxfp8_config(monkeypatch)
    config.attention_backend = AttnBackend.unfused
    with pytest.raises(RuntimeError, match="fused DSA kernels are disabled"):
        _cpu_dsa_forward(config, training=training)
    assert declining_fused_hooks == []


def test_bf16_still_falls_back_when_a_fused_hook_declines(monkeypatch, declining_fused_hooks):
    config = _make_mxfp8_config(
        monkeypatch, dsa_indexer_precision="bf16", dsa_indexer_n_heads=4, dsa_indexer_head_dim=8
    )
    output = _cpu_dsa_forward(config, training=False)
    assert torch.isfinite(output).all()
    assert "run_fused_qk_topk" in declining_fused_hooks


def test_mxfp8_rejects_the_bf16_standalone_indexer_scorer(monkeypatch):
    config = _make_mxfp8_config(monkeypatch)
    indexer = SimpleNamespace(config=config)
    with pytest.raises(RuntimeError, match="always scores in BF16"):
        dsa_module.DSAIndexer.forward_with_scores(indexer, x=None, qr=None)


def _kpool_config(monkeypatch, precision):
    return _make_mxfp8_config(
        monkeypatch,
        dsa_indexer_precision=precision,
        dsa_indexer_n_heads=4,
        dsa_indexer_head_dim=8,
        dsa_indexer_kpool=4,
        dsa_indexer_topk=4,
        dsa_indexer_loss_coeff=0.0,
        use_cpu_initialization=True,
    )


def _cpu_kpool_dsa_forward(monkeypatch, config, *, training):
    """Run one computing KPool DSA layer on CPU; return the KPool selector and hook calls."""
    real_indexer_cls = dsa_module.DSAIndexer

    def build(spec, *args, **kwargs):
        if spec is real_indexer_cls:
            kwargs.setdefault("submodules", dsa_module.DSAIndexerSubmodules())
            return real_indexer_cls(*args, **kwargs)
        return torch.nn.Identity()

    monkeypatch.setattr(dsa_module, "build_module", build)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    hook_calls, kpool_calls = [], []
    for name in _FUSED_DSA_HOOKS:
        monkeypatch.setattr(
            dsa_kernels, name, lambda *_args, _name=name, **_kwargs: hook_calls.append(_name)
        )

    def kpool_topk(q, k, weights, topk, kpool, gate, ape, **_kwargs):
        kpool_calls.append(kpool)
        rows, batch = q.shape[:2]
        indices = torch.arange(topk).clamp(max=rows - 1)
        return None, indices.expand(batch, rows, topk).contiguous()

    monkeypatch.setattr(dsa_module, "fused_qk_topk_kpool", kpool_topk)
    attention = dsa_module.DSAttention(
        config=config,
        submodules=dsa_module.DSAttentionSubmodules(indexer=real_indexer_cls),
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        pg_collection=SimpleNamespace(tp=None, cp=None),
    )
    assert attention.indexer.index_kpool == 4

    def forward_before_topk(x, _qr, _packed_seq_params):
        generator = torch.Generator().manual_seed(7)
        rows, batch = x.shape[:2]
        q = torch.randn(rows, batch, 4, 8, generator=generator)
        k = torch.randn(rows, batch, 8, generator=generator)
        weights = torch.rand(rows, batch, 4, generator=generator)
        attention.indexer._kpool_gate_score = torch.randn(rows, batch, 1, generator=generator)
        return q, k, weights

    attention.indexer.forward_before_topk = forward_before_topk
    attention.train(training)
    generator = torch.Generator().manual_seed(11)
    heads = config.num_attention_heads
    absorbed_dim = config.kv_lora_rank + config.qk_pos_emb_head_dim
    seq = 8
    mask = torch.ones((1, 1, seq, seq), dtype=torch.bool).triu(1)
    with forward_sharing_lifetime(None, mask, config):
        attention(
            query=torch.randn(seq, 1, heads, absorbed_dim, generator=generator),
            key=torch.randn(seq, 1, 1, absorbed_dim, generator=generator),
            value=None,
            attention_mask=mask,
            x=torch.randn(seq, 1, config.hidden_size, generator=generator),
            qr=torch.randn(seq, 1, config.q_lora_rank, generator=generator),
            up_v_weight=torch.randn(
                heads, config.v_head_dim, config.kv_lora_rank, generator=generator
            ),
            attn_mask_type=AttnMaskType.causal,
        )
    return kpool_calls, hook_calls


@pytest.mark.parametrize("training", [True, False])
def test_bf16_kpool_layer_is_unchanged(monkeypatch, training):
    config = _kpool_config(monkeypatch, "bf16")
    kpool_calls, hook_calls = _cpu_kpool_dsa_forward(monkeypatch, config, training=training)
    # KPool selects; no cuDNN indexer hook (the only MXFP8-capable scorer) is offered the layout.
    assert kpool_calls == [4]
    assert hook_calls == []


@pytest.mark.parametrize("training", [True, False])
def test_mxfp8_raises_in_the_kpool_branch_after_validation(monkeypatch, training):
    # Config validation rejects MXFP8 with KPool; a config mutated afterwards must still not
    # reach the KPool selector, which never runs the cuDNN MXFP8 scorer.
    config = _kpool_config(monkeypatch, "bf16")
    config.dsa_indexer_precision = "mxfp8"
    with pytest.raises(RuntimeError, match="KPool indexer top-k .* the KPool selector"):
        _cpu_kpool_dsa_forward(monkeypatch, config, training=training)


# ---------------------------------------------------------------------------------------
# Scale storage sizing
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("num_heads", [1, 32, 64])
def test_static_scale_storage_covers_every_boundary_layout(num_heads):
    rng = random.Random(num_heads)
    for _ in range(200):
        total = rng.randint(1, 4096)
        num_segments = rng.randint(1, 12)
        cuts = sorted(rng.randint(0, total) for _ in range(num_segments - 1))
        cu = torch.tensor([0, *cuts, total], dtype=torch.int32)
        padded = indexer_quantization.make_indexer_mxfp8_scale_cu_seqlens(cu, num_heads)
        shape = dsa_cudnn_kernels._packed_thd_mxfp8_scale_shape(
            total, num_segments, num_heads, HEAD_DIM
        )
        assert shape[0] == 1 and shape[2] == 4 and shape[1] % 128 == 0
        assert int(padded[-1]) * num_heads <= shape[1]


@pytest.mark.parametrize(
    "seq_len, cp_size, max_num_seqs", [(4096, 1, 2), (131072, 32, 2), (1048576, 64, 2)]
)
def test_static_scale_storage_for_glm52_slot_shapes(seq_len, cp_size, max_num_seqs):
    """The GLM-5.2 4K/128K/1M slots: storage stays within a small bound of the payload."""
    local_rows = seq_len // cp_size
    segments = max_num_seqs * (2 if cp_size > 1 else 1)
    key_rows = seq_len + (seq_len // (2 * cp_size) if cp_size > 1 else 0)
    q_shape = dsa_cudnn_kernels._packed_thd_mxfp8_scale_shape(local_rows, segments, HEADS, HEAD_DIM)
    k_shape = dsa_cudnn_kernels._packed_thd_mxfp8_scale_shape(key_rows, segments, 1, HEAD_DIM)
    assert q_shape[1] <= (local_rows + segments * 3) * HEADS
    assert k_shape[1] <= key_rows + segments * 127


# ---------------------------------------------------------------------------------------
# Plumbing through the packed THD scorer
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seq_lens, cp_size, cp_rank, max_num_seqs",
    [((256, 128, 128), 1, 0, None), ((256, 128), 1, 0, 4), ((256, 512), 2, 1, 3)],
)
def test_mxfp8_scorer_receives_quantized_inputs_and_padded_scales(
    mxfp8_runtime, seq_lens, cp_size, cp_rank, max_num_seqs
):
    layout = fakes.packed_zigzag_layout(seq_lens, cp_size, cp_rank, max_num_seqs)
    sq = layout[0].numel()
    sk = sq * cp_size
    q, k, w = _inputs(sq, sk, seed=sq)
    _score(q, k, w, layout, cp_size, cp_rank, 16, MXFP8)

    (call,) = fakes.Fe131DSA.calls
    assert call["precision"] == "mxfp8" and call["sf_vec_size"] == 32
    assert call["q"].dtype == torch.float8_e4m3fn and call["k"].dtype == torch.float8_e4m3fn
    assert call["w"].dtype == torch.bfloat16
    torch.testing.assert_close(call["w"], w[0], rtol=0, atol=0)
    torch.testing.assert_close(call["q"].float(), q[0].to(torch.float8_e4m3fn).float())

    if cp_size == 1:
        score_cu_q = score_cu_k = layout[2]
        score_k_rows = sk
    else:
        cp_layout = dsa_layout.build_packed_cp_indexer_layout(
            layout[2], layout[2], cp_size=cp_size, cp_rank=cp_rank, key_size=sk
        )
        score_cu_q = cp_layout.segment_cu_q.to(torch.int32)
        score_cu_k = cp_layout.segment_cu_k.to(torch.int32)
        score_k_rows = int(cp_layout.source_indices.numel())
        torch.testing.assert_close(
            call["k"].float()[:, 0, :],
            k[0].index_select(0, cp_layout.source_indices).to(torch.float8_e4m3fn).float(),
        )
    torch.testing.assert_close(call["cu_seqlens_q"], score_cu_q, rtol=0, atol=0)
    torch.testing.assert_close(
        call["cu_seqlens_q_scale_padded"],
        indexer_quantization.make_indexer_mxfp8_scale_cu_seqlens(score_cu_q, HEADS),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        call["cu_seqlens_k_scale_padded"],
        indexer_quantization.make_indexer_mxfp8_scale_cu_seqlens(score_cu_k, 1),
        rtol=0,
        atol=0,
    )
    segments = score_cu_q.numel() - 1
    assert tuple(call["q_scale"].shape) == dsa_cudnn_kernels._packed_thd_mxfp8_scale_shape(
        sq, segments, HEADS, HEAD_DIM
    )
    assert tuple(call["k_scale"].shape) == dsa_cudnn_kernels._packed_thd_mxfp8_scale_shape(
        score_k_rows, segments, 1, HEAD_DIM
    )

    q_call, k_call = mxfp8_runtime
    assert q_call["shape"] == (sq, HEADS, HEAD_DIM)
    assert k_call["shape"] == (score_k_rows, HEAD_DIM)
    assert q_call["out_scale_shape"] == tuple(call["q_scale"].shape)
    assert k_call["out_scale_shape"] == tuple(call["k_scale"].shape)


@pytest.mark.parametrize("cp_size, cp_rank", [(1, 0), (2, 0), (2, 1)])
def test_mxfp8_selects_with_the_mxfp8_scorer_and_scores_the_loss_from_bf16(
    mxfp8_runtime, cp_size, cp_rank
):
    """The MXFP8 scorer decides the selection; the KL predict is recomputed from BF16."""
    layout = fakes.packed_zigzag_layout((512, 256), cp_size, cp_rank)
    sq = layout[0].numel()
    q, k, w = _inputs(sq, sq * cp_size, seed=7 + cp_rank)
    indices, lengths, scores = _score(q, k, w, layout, cp_size, cp_rank, 32, MXFP8)

    # Selection: the BF16 scorer on E4M3-rounded inputs selects the same keys.
    q8 = q.to(torch.float8_e4m3fn).to(torch.bfloat16)
    k8 = k.to(torch.float8_e4m3fn).to(torch.bfloat16)
    ref_indices, ref_lengths, mxfp8_values = _score(q8, k8, w, layout, cp_size, cp_rank, 32, BF16)
    torch.testing.assert_close(indices, ref_indices, rtol=0, atol=0)
    torch.testing.assert_close(lengths, ref_lengths, rtol=0, atol=0)

    # Loss scores: one FP32 recompute from the caller's BF16 Q/K/W (original K, global key
    # ids, one 128-key tile, no topk_length) over the selected keys.
    (call,) = fakes.Fe131DSA.recompute_calls
    assert torch.equal(call["q"], q) and torch.equal(call["k"], k) and torch.equal(call["w"], w)
    assert call["topk_indices"].shape[-1] == 128 and call["topk_length"] is None
    assert call["qhead_per_kv_head"] == HEADS and call["topk_indices_global"] is False
    assert scores.dtype == torch.float32
    want, valid = _bf16_log_probs(q, k, w, indices, lengths)
    got = dsa_masking.masked_log_softmax(scores, valid)
    torch.testing.assert_close(got[valid], want[valid], rtol=0, atol=1e-5)
    # ... and not from the MXFP8 scorer's own values.
    own = dsa_masking.masked_log_softmax(mxfp8_values, valid)
    assert not torch.allclose(got[valid], own[valid], rtol=0, atol=1e-3)


def test_mxfp8_and_skipped_fill_compose(monkeypatch, mxfp8_runtime):
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", fakes.Fe131CleanLogitsDSA)
    fakes.Fe131CleanLogitsDSA.reset()
    layout = fakes.packed_zigzag_layout((256, 256), 2, 0, 3)
    sq = layout[0].numel()
    q, k, w = _inputs(sq, sq * 2, seed=11)
    options = dsa_cudnn_kernels.IndexerScorerOptions(skip_masked_score_fill=True, precision="mxfp8")
    skipped = _score(q, k, w, layout, 2, 0, 24, options)
    (call,) = fakes.Fe131CleanLogitsDSA.calls
    assert call["precision"] == "mxfp8" and call["clean_logits"] is False

    filled = _score(q, k, w, layout, 2, 0, 24, MXFP8)
    for actual, expected in zip(skipped, filled):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_mxfp8_scorer_does_not_synchronize_with_the_host(mxfp8_runtime, monkeypatch):
    """Scoring, selection and the loss-score recompute, ranked and every-column."""
    layout = fakes.packed_zigzag_layout((256, 512), 2, 1, 3)
    sq = layout[0].numel()
    q, k, w = _inputs(sq, sq * 2)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("host synchronization in the MXFP8 scorer path")

    for name in ("item", "tolist", "__bool__", "__int__", "nonzero"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    monkeypatch.setattr(torch, "nonzero", forbidden)
    for topk in (16, 512):
        for return_topk_scores in (False, True):
            _score(q, k, w, layout, 2, 1, topk, MXFP8, return_topk_scores)
    assert len(fakes.Fe131DSA.recompute_calls) == 2


# ---------------------------------------------------------------------------------------
# Selected scores for the indexer loss
# ---------------------------------------------------------------------------------------


_LOG_FLT_MIN = math.log(torch.finfo(torch.float32).tiny)


def _dense_logits(q, k, w):
    """The BF16 path's FP32 indexer logits ``sum_h relu(q_h . k) w_h``, ``(sq, sk)``."""
    return fakes._dense_scores(q[0], k[0], w[0], 1.0)


def _gathered(dense, indices, lengths):
    """``dense`` at each slot's selected key, and the valid slots (``slot < length``)."""
    valid = torch.arange(indices.size(-1)).view(1, 1, -1) < lengths.unsqueeze(-1)
    return torch.gather(dense, 1, indices[0].clamp(min=0).long()).unsqueeze(0), valid


def _bf16_log_probs(q, k, w, indices, lengths):
    """FP32 log-softmax of the BF16 path's logits over each row's selected keys, floored at
    ``log(FLT_MIN)`` as the recomputed payload is (keys whose logit is more than 87.3 below
    the row's log-sum-exp over the selection; these inputs have such keys)."""
    gathered, valid = _gathered(_dense_logits(q, k, w), indices, lengths)
    return dsa_masking.masked_log_softmax(gathered, valid).clamp_min(_LOG_FLT_MIN), valid


@pytest.mark.parametrize("width", ["ranked", "every-column"])
@pytest.mark.parametrize("loss", ["on", "frozen"])
@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
def test_loss_scores_follow_the_scorer_precision(mxfp8_runtime, precision, loss, width):
    """Which path supplies the indexer-loss scores, per compute precision, loss mode and width.

    ``every-column`` asks for at least as many keys as the widest row of the CP rank has, so
    the packed THD scorer returns every column without ranking (its early exit); ``ranked``
    runs the top-k. Only MXFP8 with the loss on recomputes; BF16 returns the scorer's own FP32
    values; an indices-only (frozen-indexer or selection-only) call keeps no scores.
    """
    layout = fakes.packed_zigzag_layout((256, 128), 2, 1)
    sq = layout[0].numel()
    q, k, w = _inputs(sq, sq * 2, seed=17)
    # The CP2 rank-1 host width bound: max(2 * half, L - half) with half = L // 4.
    topk = 16 if width == "ranked" else max(2 * (layout[3] // 4), layout[3] - layout[3] // 4)
    options = MXFP8 if precision == "mxfp8" else BF16
    indices, lengths, scores = _score(q, k, w, layout, 2, 1, topk, options, loss == "on")

    recomputes = fakes.Fe131DSA.recompute_calls
    scorer_calls = fakes.Fe131DSA.calls
    assert len(scorer_calls) == 1 and scorer_calls[0]["precision"] == precision
    row_lengths = torch.minimum(layout[1] - layout[0], torch.tensor(topk))
    assert torch.equal(lengths[0].long(), row_lengths)
    if loss == "frozen":
        assert scores is None and recomputes == []
        return
    want, valid = _bf16_log_probs(q, k, w, indices, lengths)
    if precision == "bf16":
        assert recomputes == []
        dense, _ = _gathered(_dense_logits(q, k, w), indices, lengths)
        torch.testing.assert_close(scores[valid], dense[valid], rtol=1e-6, atol=1e-5)
    else:
        (call,) = recomputes
        assert call["topk_indices"].shape[-1] == (256 if topk > 128 else 128)
        got = dsa_masking.masked_log_softmax(scores, valid)
        torch.testing.assert_close(got[valid], want[valid], rtol=0, atol=1e-5)
        assert bool((scores[valid] <= 1e-6).all())


def test_the_loss_score_rule_leaves_bf16_and_indices_only_calls_untouched(mxfp8_runtime):
    """BF16 values pass through as the very same tensors. An indices-only call keeps the very
    same indices and no scores, whatever values the kernel returned, and never recomputes; a
    loss-on call without values raises."""
    rule = dsa_cudnn_kernels._recomputes_indexer_loss_scores
    assert [precision for precision in ("bf16", "mxfp8") if rule(precision)] == ["mxfp8"]
    q, k, w = _inputs(1, 8)
    indices = torch.tensor([[[3, 1, -1]]], dtype=torch.int32)
    values = torch.tensor([[[2.0, float("nan"), float("-inf")]]])
    common = dict(
        q_bshd=q,
        k_bshd=k.unsqueeze(2),
        w_bsh=w,
        starts=torch.zeros(1, dtype=torch.int64),
        ends=torch.tensor([2]),  # key 3 lies outside the row: kept as is for BF16
        topk_k=3,
    )
    select = dsa_cudnn_kernels._selected_scores_for_indexer_loss
    out = select(indices, values, return_topk_scores=True, precision="bf16", **common)
    assert out[0] is indices and out[1] is values
    for precision in ("bf16", "mxfp8"):
        for kernel_values in (None, values):
            out = select(
                indices, kernel_values, return_topk_scores=False, precision=precision, **common
            )
            assert out[0] is indices and out[1] is None
        with pytest.raises(RuntimeError, match="returned no selected scores"):
            select(indices, None, return_topk_scores=True, precision=precision, **common)
    assert fakes.Fe131DSA.recompute_calls == []


@pytest.mark.parametrize("width", ["ranked", "every-column"])
def test_indices_only_calls_never_recompute_whatever_the_kernel_returns(
    mxfp8_runtime, monkeypatch, width
):
    """The loss is on when the caller asks for scores, not when a kernel happens to return
    values: a selection kernel that always returns its values leaves an MXFP8 indices-only
    call with the same indices, no scores and no recompute."""
    layout = fakes.packed_zigzag_layout((256, 128), 2, 1)
    sq = layout[0].numel()
    q, k, w = _inputs(sq, sq * 2, seed=19)
    topk = 16 if width == "ranked" else max(2 * (layout[3] // 4), layout[3] - layout[3] // 4)
    expected = _score(q, k, w, layout, 2, 1, topk, MXFP8, return_topk_scores=False)
    real = dsa_cudnn_kernels._indexer_topk_packed_thd_cudnn
    returned = []

    def always_values(**kwargs):
        result = real(**dict(kwargs, return_topk_scores=True))
        returned.append(result[1])
        return result

    monkeypatch.setattr(dsa_cudnn_kernels, "_indexer_topk_packed_thd_cudnn", always_values)
    indices, lengths, scores = _score(q, k, w, layout, 2, 1, topk, MXFP8, return_topk_scores=False)
    assert returned[0] is not None and scores is None
    assert torch.equal(indices, expected[0]) and torch.equal(lengths, expected[1])
    assert fakes.Fe131DSA.recompute_calls == []


@pytest.mark.parametrize("kind", ["ternary", "gauss"])
def test_fe_mxfp8_loss_inputs_equal_the_bf16_path_given_the_same_selection(
    mxfp8_runtime, monkeypatch, kind
):
    """Given one selection, the MXFP8 payload and the BF16 path's give the same KL loss and
    the same predict for the indexer backward. Ternary inputs are exact in E4M3, so both real
    paths select the same keys; for Gaussian inputs the BF16 path's payload (its FP32 scores
    of the selected keys) is taken at the MXFP8 selection. These inputs keep every selected
    key's probability above FLT_MIN (asserted: its logit within 87.3 of the row's
    log-sum-exp over the selection); below it the payload floors at log(FLT_MIN) and only the
    KL value differs (``test_recompute_underflow_stays_finite``)."""
    layout = fakes.packed_zigzag_layout((256, 128), 1, 0)
    sq = layout[0].numel()
    if kind == "ternary":
        q, k, w = _ternary_inputs(sq, sq, seed=43)
    else:
        q, k, w = _inputs(sq, sq, seed=43, w_scale=0.05)
    indices, lengths, payload = _score(q, k, w, layout, 1, 0, 24, MXFP8)
    if kind == "ternary":
        bf16_indices, bf16_lengths, bf16_payload = _score(q, k, w, layout, 1, 0, 24, BF16)
        assert torch.equal(bf16_indices, indices) and torch.equal(bf16_lengths, lengths)
    else:
        bf16_payload, valid = _gathered(_dense_logits(q, k, w), indices, lengths)
        bf16_payload = bf16_payload.masked_fill(~valid, torch.finfo(torch.float32).min)
    assert not torch.allclose(payload, bf16_payload)
    selected = torch.arange(24).view(1, 1, -1) < lengths.unsqueeze(-1)
    assert bool((payload[selected] > _LOG_FLT_MIN).all())
    seen = []
    target_generator = torch.Generator().manual_seed(3)

    def fake_target(q_attn, k_attn, lse, topk_indices, topk_length, softmax_scale, **kwargs):
        return torch.rand(topk_indices.shape, generator=target_generator) + 0.1

    def fake_backward(q, w, k, attn_score, index_score, topk_indices, **kwargs):
        seen.append(index_score.clone())
        return torch.zeros_like(q), torch.zeros_like(w), torch.zeros_like(k)

    monkeypatch.setattr(dsa_cudnn_kernels, "_compute_attn_target", fake_target)
    monkeypatch.setattr(dsa_cudnn_kernels, "_run_sparse_indexer_backward", fake_backward)
    common = dict(
        q_idx_bshd=q,
        k_idx_bsd=k,
        w_bsh=w,
        topk_indices_cmp=indices,
        topk_length_cmp=lengths,
        query=torch.randn((sq, 1, 64, 576)),
        kv_full=torch.randn((sq, 1, 576)),
        lse=torch.zeros((sq, 64)),
        softmax_scale=1.0,
        loss_coeff=0.1,
        query_valid_rows=None,
        calculate_per_token_loss=False,
        tp_group=None,
    )
    losses = []
    for scores in (payload, bf16_payload):
        target_generator.manual_seed(3)
        losses.append(
            dsa_cudnn_kernels._compute_sparse_indexer_loss_and_grads(
                indexer_score_payload=scores, **common
            )[0]
        )
    torch.testing.assert_close(losses[0], losses[1], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(seen[0], seen[1], rtol=1e-5, atol=1e-7)


def test_recompute_pads_to_one_shape_per_top_k_and_masks_invalid_slots(mxfp8_runtime):
    q, k, w = _inputs(8, 64, seed=41)
    indices = torch.tensor([[[3, -1, 5, 1], [-1, -1, -1, -1]] + [[0, 1, 2, 3]] * 6])
    for topk_k in (4, 200):
        payload = dsa_cudnn_kernels._recompute_selected_indexer_log_probs(
            q, k.unsqueeze(2), w, indices.int(), topk_k=topk_k
        )
        call = fakes.Fe131DSA.recompute_calls[-1]
        assert call["topk_indices"].shape[-1] == (128 if topk_k == 4 else 256)
        assert bool((call["topk_indices"][..., 4:] == -1).all())
        assert payload.shape == (1, 8, 4) and payload.dtype == torch.float32
        assert bool(torch.isinf(payload[0, 0, 1])) and bool(torch.isinf(payload[0, 1]).all())
        assert bool(torch.isfinite(payload[0, 0, [0, 2, 3]]).all())


def test_recompute_underflow_stays_finite(mxfp8_runtime, monkeypatch):
    def huge_gap(*_args, **_kwargs):
        predict = torch.zeros((1, 1, 128))
        predict[0, 0, 0] = 1.0
        return {"predict": predict}

    monkeypatch.setattr(fakes.Fe131DSA, "sparse_indexer_score_recompute_wrapper", huge_gap)
    q, k, w = _inputs(1, 8)
    indices = torch.tensor([[[0, 1, 2, -1]]], dtype=torch.int32)
    payload = dsa_cudnn_kernels._recompute_selected_indexer_log_probs(
        q, k.unsqueeze(2), w, indices, topk_k=4
    )
    floor = torch.log(torch.tensor(torch.finfo(torch.float32).tiny))
    assert fakes._TOLIST(payload[0, 0, :3]) == [0.0, float(floor), float(floor)]
    assert payload[0, 0, 3] == float("-inf")


def test_missing_recompute_entry_point_raises_instead_of_changing_the_loss(
    mxfp8_runtime, monkeypatch
):
    monkeypatch.delattr(fakes.Fe131DSA, "sparse_indexer_score_recompute_wrapper")
    layout = fakes.packed_zigzag_layout((256, 128), 1, 0)
    q, k, w = _inputs(384, 384)
    with pytest.raises(RuntimeError, match="lacks sparse_indexer_score_recompute_wrapper"):
        _score(q, k, w, layout, 1, 0, 16, MXFP8)
    assert _score(q, k, w, layout, 1, 0, 16, MXFP8, False)[2] is None
    assert _score(q, k, w, layout, 1, 0, 16, BF16)[2] is not None


def test_selections_the_caller_drops_never_reach_the_recompute(mxfp8_runtime, monkeypatch):
    """A slot whose scorer value is not finite (NaN, a -inf prefill column), or that lies
    outside its row's ``[starts, ends)`` (below or past it) or past the keys, is dropped
    before the recompute gathers its K row, and the row ends with one key fewer, as without
    the recompute. Each case is the only bound its slot breaks: the key below its row is a
    valid key inside the previous segment, and the key past the keys lies inside a row whose
    ``ends`` exceed them (bounds are not clamped to the key count)."""
    starts, ends, cu, max_seqlen = fakes.packed_zigzag_layout((256, 128), 1, 0)
    q, k, w = _inputs(384, 384, seed=73)
    sk = k.size(1)
    ends = ends.clone()
    ends[383] = sk + 8
    layout = (starts, ends, cu, max_seqlen)

    def selection(**_kwargs):
        indices = (starts.view(1, -1, 1) + torch.arange(16).view(1, 1, -1)).to(torch.int32)
        values = torch.zeros(indices.shape)
        indices[0, 300, 0] = int(fakes._TOLIST(ends)[300])  # one past the row
        indices[0, 383, 1] = sk + 5  # past every key, inside the row's [starts, ends)
        indices[0, 310, 4] = int(fakes._TOLIST(starts)[310]) - 1  # one below the row
        values[0, 200, 2] = float("nan")
        values[0, 201, 3] = float("-inf")
        return indices, values

    monkeypatch.setattr(dsa_cudnn_kernels, "_indexer_topk_packed_thd_cudnn", selection)
    _indices, lengths, scores = _score(q, k, w, layout, 1, 0, 16, MXFP8)
    (call,) = fakes.Fe131DSA.recompute_calls
    sent = call["topk_indices"][0, :, :16].long()
    dropped = [int(sent[r, c]) for r, c in ((300, 0), (383, 1), (310, 4), (200, 2), (201, 3))]
    assert dropped == [-1, -1, -1, -1, -1]
    in_row = (sent >= starts.view(-1, 1)) & (sent < ends.view(-1, 1)) & (sent < sk)
    assert bool((in_row | (sent == -1)).all())
    expected = torch.minimum(ends - starts, torch.tensor(16))
    expected[[300, 383, 310, 200, 201]] -= 1
    assert torch.equal(lengths[0].long(), expected)
    valid = torch.arange(16).view(1, 1, -1) < lengths.unsqueeze(-1)
    assert bool(torch.isfinite(scores[valid]).all())


def test_prefilled_uncovered_rows_stay_dropped_with_the_loss_on(mxfp8_runtime):
    """With the -inf prefill, rows past every segment select only -inf columns; the
    recompute keeps them dropped, so every row's length equals the BF16 path's."""
    starts, ends, cu, max_seqlen = fakes.packed_zigzag_layout((128, 64), 1, 0)
    total = int(fakes._TOLIST(cu)[-1])
    starts = torch.cat((starts, torch.full((4,), total, dtype=torch.int64)))
    ends = torch.cat((ends, total + torch.arange(4) + 1))
    layout = (starts, ends, cu, max_seqlen)
    q, k, w = _inputs(starts.numel(), starts.numel(), seed=59)
    _indices, lengths, _scores = _score(q, k, w, layout, 1, 0, 16, MXFP8)
    assert fakes._TOLIST(lengths[0, -4:]) == [0, 0, 0, 0]
    _bf16_indices, bf16_lengths, _bf16_scores = _score(q, k, w, layout, 1, 0, 16, BF16)
    assert torch.equal(lengths, bf16_lengths)


def test_checkpoint_score_stash_holds_the_recompute_and_the_replay_accepts_it(
    mxfp8_runtime, monkeypatch
):
    """``dsa_indexer_reuse_checkpoint_topk``: the first call keeps the recomputed payload, and
    the replay's loss consumes exactly that tensor."""
    starts, ends, cu, max_seqlen = fakes.packed_zigzag_layout((256, 128), 1, 0)
    sq = starts.numel()
    q, k, w = _inputs(sq, sq, seed=47)
    config = SimpleNamespace(
        dsa_indexer_precision="mxfp8", dsa_indexer_skip_masked_score_fill=False, kv_lora_rank=512
    )
    packed = SimpleNamespace(
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=None,
        cu_seqlens_kv_padded=None,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
    )
    query = torch.randn((sq, 1, 64, 576))
    key = torch.randn((sq, 1, 1, 576))
    result = dsa_cudnn_kernels.run_fused_qk_topk_with_scores(
        q.permute(1, 0, 2, 3),
        k.permute(1, 0, 2),
        w.permute(1, 0, 2),
        24,
        starts,
        ends,
        64,
        use_local_indexer_varlen=True,
        packed_seq_params=packed,
        query=query,
        key=key,
        config=config,
    )
    assert result is not None
    indices, lengths, scores = result
    assert scores.dtype == torch.float32 and len(fakes.Fe131DSA.recompute_calls) == 1
    want, valid = _bf16_log_probs(q, k, w, indices, lengths)
    got = dsa_masking.masked_log_softmax(scores, valid)
    torch.testing.assert_close(got[valid], want[valid], rtol=0, atol=1e-5)
    applied = {}
    monkeypatch.setattr(
        dsa_cudnn_kernels,
        "SparseIndexerLossFromTopKFunc",
        SimpleNamespace(apply=lambda *args: applied.setdefault("args", args)),
    )
    dsa_cudnn_kernels.run_sparse_indexer_loss_from_topk(
        q.permute(1, 0, 2, 3),
        k.permute(1, 0, 2),
        w.permute(1, 0, 2),
        indices,
        lengths,
        scores,
        query,
        key,
        softmax_scale=1.0,
        loss_coeff=0.1,
        pg_collection=SimpleNamespace(tp=None),
        config=config,
    )
    assert any(arg is scores for arg in applied["args"])
    assert len(fakes.Fe131DSA.recompute_calls) == 1


# ---------------------------------------------------------------------------------------
# Paths that cannot honour MXFP8 and missing runtime support
# ---------------------------------------------------------------------------------------


def test_mxfp8_rejects_plain_causal_scoring(mxfp8_runtime):
    q, k, w = _inputs(64, 64)
    with pytest.raises(RuntimeError, match="plain_causal"):
        dsa_cudnn_kernels._indexer_topk_bshd(
            q, k, w, 8, return_scores=False, return_topk_scores=True, scorer_options=MXFP8
        )


def _single_sequence_pack(q, k, w, layout, cp_size, cp_rank, options, with_metadata=True):
    starts, ends, cu, max_seqlen = layout
    metadata = (
        dict(
            packed_cu_seqlens_q=cu,
            packed_cu_seqlens_k=cu,
            packed_max_seqlen_q=max_seqlen,
            packed_max_seqlen_k=max_seqlen,
        )
        if with_metadata
        else {}
    )
    return dsa_cudnn_kernels._indexer_topk_bshd(
        q,
        k,
        w,
        8,
        varlen_starts=starts,
        varlen_ends=ends,
        return_scores=False,
        return_topk_scores=True,
        use_local_indexer_varlen=True,
        single_packed_thd_sequence=True,
        local_packed_cp_query_len=q.size(1),
        packed_cp_size=cp_size,
        local_packed_cp_rank=cp_rank,
        scorer_options=options,
        **metadata,
    )


@pytest.mark.parametrize("cp_size, cp_rank", [(1, 0), (2, 0), (2, 1)])
def test_mxfp8_scores_single_sequence_packs_with_the_general_scorer(
    mxfp8_runtime, cp_size, cp_rank
):
    """A one-sequence pack leaves the BSHD segment scorer for the MXFP8 THD scorer."""
    layout = fakes.packed_zigzag_layout((256,), cp_size, cp_rank)
    sq = layout[0].numel()
    q, k, w = _inputs(sq, sq * cp_size, seed=cp_size + cp_rank)

    bf16 = _single_sequence_pack(q, k, w, layout, cp_size, cp_rank, BF16)
    assert all(call["cu_seqlens_q"] is None for call in fakes.Fe131DSA.calls)

    fakes.Fe131DSA.reset()
    mxfp8 = _single_sequence_pack(q, k, w, layout, cp_size, cp_rank, MXFP8)
    (call,) = fakes.Fe131DSA.calls
    assert call["precision"] == "mxfp8" and call["cu_seqlens_q"] is not None

    q8 = q.to(torch.float8_e4m3fn).to(torch.bfloat16)
    k8 = k.to(torch.float8_e4m3fn).to(torch.bfloat16)
    reference = _single_sequence_pack(q8, k8, w, layout, cp_size, cp_rank, BF16)
    torch.testing.assert_close(mxfp8[0], reference[0], rtol=0, atol=0)
    torch.testing.assert_close(mxfp8[1], bf16[1], rtol=0, atol=0)


def test_mxfp8_rejects_single_sequence_segments_without_packed_metadata(mxfp8_runtime):
    layout = fakes.packed_zigzag_layout((256,), 2, 0)
    q, k, w = _inputs(128, 256)
    with pytest.raises(RuntimeError, match="packed_single_segments"):
        _single_sequence_pack(q, k, w, layout, 2, 0, MXFP8, with_metadata=False)


def test_mxfp8_rejects_the_dense_indexer_loss(mxfp8_runtime):
    layout = fakes.packed_zigzag_layout((128, 128), 1, 0)
    q, k, w = _inputs(256, 256)
    starts, ends, cu, max_seqlen = layout
    with pytest.raises(RuntimeError, match="dense indexer loss"):
        dsa_cudnn_kernels._indexer_topk_bshd(
            q,
            k,
            w,
            8,
            varlen_starts=starts,
            varlen_ends=ends,
            return_scores=True,
            use_local_indexer_varlen=True,
            packed_cu_seqlens_q=cu,
            packed_cu_seqlens_k=cu,
            packed_max_seqlen_q=max_seqlen,
            packed_max_seqlen_k=max_seqlen,
            scorer_options=MXFP8,
        )


@pytest.mark.parametrize(
    "breakage, message",
    [
        ("legacy_frontend", "lacks precision"),
        ("no_te", "Transformer Engine MXFP8"),
        ("no_triton", "Triton"),
        ("old_gpu", "SM100"),
        ("fp32_weights", "indexer weights must be BF16"),
        ("16_heads", "indexer heads must be 32 or 64"),
    ],
)
def test_mxfp8_runtime_requirements_fail_loudly(mxfp8_runtime, monkeypatch, breakage, message):
    heads = 16 if breakage == "16_heads" else HEADS
    w_dtype = torch.float32 if breakage == "fp32_weights" else torch.bfloat16
    if breakage == "legacy_frontend":
        monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", fakes.BottomRightDefaultDSA)
    elif breakage == "no_te":
        monkeypatch.setattr(indexer_quantization, "HAVE_TE_MXFP8", False)
    elif breakage == "no_triton":
        monkeypatch.setattr(indexer_quantization, "HAVE_TRITON", False)
    elif breakage == "old_gpu":
        monkeypatch.setattr(
            dsa_cudnn_kernels,
            "_mxfp8_scorer_device_problem",
            lambda device: "SM100+ is required, got SM90",
        )
    layout = fakes.packed_zigzag_layout((128, 128), 1, 0)
    q, k, w = _inputs(256, 256, heads=heads, w_dtype=w_dtype)
    with pytest.raises(RuntimeError, match=message):
        _score(q, k, w, layout, 1, 0, 8, MXFP8)


def test_device_check_rejects_cpu_and_pre_blackwell(monkeypatch):
    assert "CUDA" in dsa_cudnn_kernels._mxfp8_scorer_device_problem(torch.device("cpu"))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    assert "SM100+" in dsa_cudnn_kernels._mxfp8_scorer_device_problem(torch.device("cuda", 0))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (10, 3))
    assert dsa_cudnn_kernels._mxfp8_scorer_device_problem(torch.device("cuda", 0)) is None


# ---------------------------------------------------------------------------------------
# Config reaches the scorer through every hook
# ---------------------------------------------------------------------------------------


def test_split_hooks_forward_the_configured_precision(monkeypatch):
    seen = []

    def fake_indexer_topk(q_bshd, _k_bsd, _w_bsh, topk, **kwargs):
        seen.append(kwargs["scorer_options"])
        shape = (q_bshd.size(0), q_bshd.size(1))
        return (
            torch.zeros((*shape, topk), dtype=torch.int32),
            torch.full(shape, topk, dtype=torch.int32),
            torch.zeros((*shape, topk)),
        )

    monkeypatch.setattr(dsa_cudnn_kernels, "_indexer_topk_bshd", fake_indexer_topk)
    config = SimpleNamespace(
        dsa_indexer_precision="mxfp8", dsa_indexer_skip_masked_score_fill=True, kv_lora_rank=512
    )
    q = torch.ones((4, 1, HEADS, HEAD_DIM))
    k = torch.ones((4, 1, HEAD_DIM))
    weights = torch.ones((4, 1, HEADS))
    bounds = dict(starts=torch.zeros(4, dtype=torch.int64), ends=torch.arange(1, 5))
    dsa_cudnn_kernels.run_fused_qk_topk(
        q, k, weights, index_topk=2, block_size=128, config=config, **bounds
    )
    assert seen[-1] == dsa_cudnn_kernels.IndexerScorerOptions(True, "mxfp8")

    applied = {}
    monkeypatch.setattr(
        dsa_cudnn_kernels,
        "FusedQKTopKWithSparseLossFunc",
        SimpleNamespace(apply=lambda *args: applied.setdefault("args", args)),
    )
    monkeypatch.setattr(dsa_cudnn_kernels, "_flash_mla_supports_head_count", lambda query: True)
    dsa_cudnn_kernels.run_fused_qk_topk_with_loss(
        q,
        k,
        weights,
        index_topk=2,
        block_size=128,
        query=torch.ones((4, 1, 64, 576)),
        key=torch.ones((4, 1, 1, 576)),
        softmax_scale=1.0,
        loss_coeff=0.01,
        pg_collection=SimpleNamespace(tp=None),
        config=config,
        **bounds,
    )
    assert applied["args"][-1] == dsa_cudnn_kernels.IndexerScorerOptions(True, "mxfp8")
