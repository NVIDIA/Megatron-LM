# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CSA2 ownership, loss-gradient and packed-layout contracts on small reference inputs."""

from functools import partial

import pytest
import torch
from torch import nn

from megatron.core.process_groups_config import ProcessGroupCollection
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
    CSA2State,
)
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
)
from megatron.core.transformer.transformer_config import MLATransformerConfig


@pytest.fixture(autouse=True)
def loss_scale(monkeypatch):
    """Keep the auxiliary-loss scale local to each reference test."""
    monkeypatch.setattr(DSAIndexerLossAutoScaler, "main_loss_backward_scale", torch.tensor(1.0))


class Group:
    def size(self):
        return 1

    def rank(self):
        return 0


class Linear(nn.Module):
    def __init__(self, input_size, output_size, *, config, init_method, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_size, input_size, dtype=config.params_dtype))
        init_method(self.weight)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight), None

    def backward_dw(self):
        pass


class Norm(nn.Module):
    def __init__(self, *, config, hidden_size, eps, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight


class CpuRoPE(nn.Module):
    def forward(self, n, packed_seq=False):
        frequencies = torch.outer(
            torch.arange(n).float(), 1 / (10000 ** (torch.arange(0, 4, 2).float() / 4))
        )
        return torch.cat((frequencies, frequencies), -1)[:, None, None, :]


def _setup():
    cfg = MLATransformerConfig(
        num_layers=3,
        hidden_size=16,
        num_attention_heads=2,
        params_dtype=torch.float32,
        use_cpu_initialization=True,
        multi_latent_attention=True,
        experimental_attention_variant='dsv4_hybrid',
        dsv4_version='v4.1',
        qk_layernorm=True,
        normalization='RMSNorm',
        q_lora_rank=8,
        v_head_dim=8,
        qk_head_dim=4,
        qk_pos_emb_head_dim=4,
        kv_lora_rank=4,
        csa_compress_ratios=[2, 2, 2],
        csa2_kv_source_layers=[0],
        csa2_index_source_layers=[0, 1],
        csa_window_size=4,
        dsa_indexer_n_heads=2,
        dsa_indexer_head_dim=8,
        dsa_indexer_topk=2,
        dsa_indexer_rotate_activation=False,
        dsa_kernel_backend='none',
        add_bias_linear=False,
        attention_dropout=0,
        hidden_dropout=0,
        dsa_indexer_loss_coeff=0.01,
    )
    g = Group()
    groups = ProcessGroupCollection(tp=g, cp=g, pp=g, dp=g, dp_cp=g, ep=g, expt_dp=g)
    rope = CpuRoPE()
    mods = CompressedSparseAttentionSubmodules(
        compressor=partial(
            CSA2Compressor,
            submodules=CompressorSubmodules(linear_wkv=Linear, linear_wgate=Linear, norm=Norm),
        ),
        indexer=partial(
            CSA2Indexer,
            submodules=CSA2IndexerSubmodules(
                linear_wq_b=Linear, linear_wk=Linear, k_norm=Norm, linear_weights_proj=Linear
            ),
        ),
    )
    layers = [
        CompressedSparseAttention2(
            cfg, mods, i + 1, AttnMaskType.causal, 'self', groups, rotary_pos_emb=rope
        )
        for i in range(3)
    ]
    return cfg, layers


def _forward(layers, state):
    outputs, inputs = [], []
    for layer in layers:
        x = torch.randn(8, 2, 16, requires_grad=True)
        qr = torch.randn(8, 2, 8, requires_grad=True)
        q = torch.randn(8, 2, 2, 8, requires_grad=True)
        kv = torch.randn(8, 2, 1, 8, requires_grad=True)
        inputs.append((x, qr, q, kv))
        outputs.append(layer(q, kv, kv, None, x=x, qr=qr, csa2_state=state))
    return outputs, inputs


def test_shared_owners_receive_main_and_auxiliary_gradients(monkeypatch):
    monkeypatch.setattr(DSAIndexerLossLoggingHelper, "save_loss_to_tracker", lambda **kwargs: None)
    DSAIndexerLossAutoScaler.set_loss_scale(torch.tensor(1.0))
    _, layers = _setup()
    first, first_inputs = _forward(layers, CSA2State())
    second, _ = _forward(layers, CSA2State())
    assert layers[2].indexer is None and layers[1].compressor is None
    sum(output.square().mean() for output in first + second).backward()
    assert layers[0].compressor.linear_wkv.weight.grad.abs().sum() > 0
    assert layers[0].indexer.linear_wk.weight.grad.abs().sum() > 0
    assert layers[1].indexer.linear_wq_b.weight.grad.abs().sum() > 0
    assert torch.isfinite(first_inputs[0][0].grad).all()


@pytest.mark.launch_on_gb200
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "batch,dim,padded", [(1, 128, False), (1, 512, True), (2, 128, True), (2, 512, False)]
)
def test_cudnn_r2_pool_matches_fp32_reference_and_graph(monkeypatch, batch, dim, padded):
    """Exercise fused dispatch, padding gradients and warmed CUDA Graph replay."""
    import copy

    from tests.unit_tests.determinism.kernels.harness import deterministic_algorithms

    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("cuDNN compressor requires SM100 or newer")
    pytest.importorskip("cudnn.csa.compressor")
    cfg, _ = _setup()
    cfg.dsa_kernel_backend = "cudnn"
    cfg.params_dtype = torch.bfloat16
    cfg.v_head_dim = dim
    compressor = CSA2Compressor(
        cfg,
        CompressorSubmodules(linear_wkv=Linear, linear_wgate=Linear, norm=Norm),
        2,
        ProcessGroupCollection(tp=Group()),
    )
    assert compressor._fused_r2_pool is not None
    copy.deepcopy(compressor)
    monkeypatch.setattr(
        compressor,
        "_pool_r2_native",
        lambda *args: pytest.fail("the cuDNN pooling test must use the fused path"),
    )
    torch.manual_seed(790)
    kv = torch.randn(66, batch, dim, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn_like(kv)
    valid = torch.arange(33, device="cuda") % 5 != 2 if padded else None
    if valid is not None:
        invalid = ~valid.repeat_interleave(2)[:, None, None]
        kv = kv.masked_fill(invalid, torch.nan)
        gate = gate.masked_fill(invalid, torch.nan)
    kv.requires_grad_()
    gate.requires_grad_()
    clean_kv = kv if valid is None else kv.masked_fill(invalid, 0)
    clean_gate = gate if valid is None else gate.masked_fill(invalid, 0)
    expected = (
        (
            clean_kv.float().reshape(33, 2, batch, dim)
            * clean_gate.float().reshape(33, 2, batch, dim).softmax(1)
        )
        .sum(1)
        .bfloat16()
    )
    if valid is not None:
        expected = expected.masked_fill(~valid[:, None, None], 0)
    probe = torch.randn_like(expected)
    if valid is not None:
        probe = probe.masked_fill(~valid[:, None, None], torch.nan)
    reference_grads = torch.autograd.grad(expected, (kv, gate), probe)

    def run(keys=kv, gates=gate):
        output = compressor._pool_r2(keys, gates, keys.dtype, valid)
        return output, *torch.autograd.grad(output, (keys, gates), probe)

    with deterministic_algorithms(False):
        actual = run()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            static_kv = kv.detach().clone().requires_grad_()
            static_gate = gate.detach().clone().requires_grad_()
            # Autograd leaves must warm up on the stream used for capture.
            run(static_kv, static_gate)
            run(static_kv, static_gate)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = run(static_kv, static_gate)
        graph.replay()
        torch.cuda.synchronize()
    for observed, reference in zip(actual, (expected, *reference_grads)):
        assert torch.isfinite(observed).all()
        assert observed.dtype == torch.bfloat16
        cosine = torch.nn.functional.cosine_similarity(
            observed.float().flatten(), reference.float().flatten(), dim=0
        )
        assert cosine >= 0.999999
    for observed, reference in zip(captured, actual):
        torch.testing.assert_close(observed, reference, rtol=0, atol=0)


def test_r2_pool_keeps_native_fallback(monkeypatch):
    """Unsupported inputs and strict determinism must not launch the frontend."""
    from megatron.core.ops.attention.csa.cudnn_pooling import _pool_r2

    def forbidden(*args, **kwargs):
        pytest.fail("native fallback must not launch a cuDNN kernel")

    kv = torch.randn(8, 2, 8)
    assert _pool_r2(kv, kv, kv.dtype, forward_fn=forbidden, backward_fn=forbidden) is None
    _, layers = _setup()
    compressor = layers[0].compressor
    compressor._fused_r2_pool = lambda *args: None
    expected = (kv.reshape(4, 2, 2, 8) * kv.reshape(4, 2, 2, 8).softmax(1)).sum(1)
    torch.testing.assert_close(compressor._pool_r2(kv, kv, kv.dtype), expected)


def test_auxiliary_objective_does_not_train_main_compressor(monkeypatch):
    monkeypatch.setattr(DSAIndexerLossLoggingHelper, "save_loss_to_tracker", lambda **kwargs: None)
    _, layers = _setup()
    state = CSA2State(defer_indexer_loss=True)
    _forward(layers[:1], state)
    state.indexer_loss.backward()
    assert layers[0].compressor.linear_wkv.weight.grad is None
    assert layers[0].indexer.linear_wk.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("ratio", [1, 2, 4, 128])
def test_compression_capacity_preserves_main_alignment(ratio):
    import math

    from megatron.core.transformer.experimental_attention_variant.csa_utils.cp_utils import (
        get_cp_compressor_capacity,
    )

    for rows in (1, 7, 128, 257):
        halo = 8 if ratio == 4 else ratio
        alignment = 32 // math.gcd(32, ratio)
        expected = max(1, (rows + halo) // ratio)
        expected = (expected + alignment - 1) // alignment * alignment
        assert get_cp_compressor_capacity(rows, ratio) == expected


def test_candidate_mask_does_not_change_tied_causal_prefix():
    from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_candidates import (
        candidate_blocks_from_scores,
    )

    short = torch.zeros(1, 3, 6)
    visible = torch.tensor([[1, 3, 6]])
    short.masked_fill_(torch.arange(6)[None, None, :] >= visible[..., None], -torch.inf)
    long = torch.nn.functional.pad(short, (0, 5), value=-torch.inf)
    a = candidate_blocks_from_scores(short, visible, 2, 2)
    b = candidate_blocks_from_scores(long, visible, 2, 2)
    assert torch.equal(a.indices, b.indices)
    assert torch.equal(a.lengths, b.lengths)


def test_window_teacher_includes_sink_and_ignores_invalid_keys():
    from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_teacher_lse import (
        fused_csa_window_lse,
    )

    query = torch.ones(3, 2, 4)
    keys = torch.ones(3, 4)
    sink = torch.tensor([0.0, 32.0])
    indices = torch.tensor([[0, -1], [0, 1], [-1, -1]], dtype=torch.int32)
    result = fused_csa_window_lse(query, keys, sink, indices, 0.5)
    expected = torch.stack(
        (
            torch.logaddexp(sink, torch.full_like(sink, 2.0)),
            torch.logaddexp(sink, torch.full_like(sink, 2.0 + torch.log(torch.tensor(2.0)).item())),
            sink,
        )
    )
    torch.testing.assert_close(result, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_window_teacher_gpu_parity_and_bit_exact_replay():
    from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_teacher_lse import (
        fused_csa_window_lse,
    )
    from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact

    query = torch.randn(32, 64, 512, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(32, 512, device="cuda", dtype=torch.bfloat16)
    sink = torch.linspace(0, 32, 64, device="cuda")
    rows = torch.arange(32, device="cuda")[:, None]
    ids = rows - torch.arange(8, device="cuda")[None, :]
    ids = ids.clamp_min(-1).int()
    scale = 512**-0.5
    expected = fused_csa_window_lse(query.cpu(), keys.cpu(), sink.cpu(), ids.cpu(), scale)
    actual = fused_csa_window_lse(query, keys, sink, ids, scale)
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-5)
    assert_replays_bit_exact(
        lambda q, k, s, indices: fused_csa_window_lse(q, k, s, indices, scale),
        (query, keys, sink, ids),
        backward=False,
        replays=3,
        contention=True,
        what="CSA2 window/sink teacher",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_candidate_score_and_compaction_gpu_parity_and_replay():
    pytest.importorskip("cudnn")
    from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_candidates import (
        _select_blocks,
        candidate_blocks_from_scores,
    )
    from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_indexer import (
        _score_chunk,
    )
    from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact

    q = torch.randn(16, 32, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(16, 32, device="cuda", dtype=torch.bfloat16)
    ids = torch.zeros(16, 1, device="cuda", dtype=torch.int32)
    starts = torch.zeros(16, device="cuda", dtype=torch.int32)
    visible = torch.arange(16, device="cuda", dtype=torch.int32) * 2

    def score(query, keys, w):
        return _score_chunk(
            query,
            keys,
            w,
            ids,
            starts,
            starts,
            visible,
            width=32,
            stride=32,
            mode=0,
            block_size=1,
            selection=True,
        )

    scores = score(q, k, weights)
    scaled_weights = (weights.float() / 64).bfloat16().float()
    expected = (
        torch.einsum("qhd,kd->qhk", q.float(), k.float()).relu() * scaled_weights[..., None]
    ).sum(1)
    expected.masked_fill_(torch.arange(32, device="cuda")[None, :] >= visible[:, None], -torch.inf)
    torch.testing.assert_close(scores, expected, rtol=2e-4, atol=2e-4)
    blocks = _select_blocks(scores, visible, 2, 8)
    reference = candidate_blocks_from_scores(expected, visible, 2, 8)
    assert torch.equal(blocks.indices, reference.indices)
    assert torch.equal(blocks.lengths, reference.lengths)
    assert_replays_bit_exact(
        score,
        (q, k, weights),
        backward=False,
        replays=3,
        contention=True,
        what="CSA2 BF16 candidate scoring",
    )
    assert_replays_bit_exact(
        lambda s: (
            _select_blocks(s, visible, 2, 8).indices,
            _select_blocks(s, visible, 2, 8).lengths,
        ),
        (scores,),
        backward=False,
        replays=3,
        what="CSA2 candidate compaction",
    )


def test_cp_layout_uses_main_packed_metadata():
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer.experimental_attention_variant.csa_utils.thd_utils import (
        build_csa2_thd_layout,
    )

    class CPGroup:
        def size(self):
            return 2

        def rank(self):
            return 1

    logical = torch.tensor([0, 5, 12], dtype=torch.int32)
    padded = torch.tensor([0, 8, 16], dtype=torch.int32)
    packed = PackedSeqParams(
        qkv_format='thd',
        cu_seqlens_q=logical,
        cu_seqlens_kv=logical,
        cu_seqlens_q_padded=padded,
        cu_seqlens_kv_padded=padded,
        max_seqlen_q=8,
        max_seqlen_kv=8,
    )
    layout = build_csa2_thd_layout(packed, 8, cp_group=CPGroup())
    assert layout.global_start == 8
    assert layout.valid_tokens.tolist() == [True] * 7 + [False]
    assert layout.position_ids.tolist() == list(range(7)) + [0]
    assert layout.for_compression(2).valid_groups.sum() == 5


@pytest.mark.launch_on_gb200
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("ratio", [1, 2])
def test_fused_shared_attention_and_indexer_gradients_match_native(monkeypatch, ratio):
    """Exercise real Full/Reindex/Reuse kernels with shared producer gradients."""
    import os
    import shutil
    import subprocess
    from dataclasses import replace

    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

    pytest.importorskip("cudnn")
    flash_mla_cuda = pytest.importorskip("flash_mla.cuda")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if torch.cuda.get_device_capability()[0] == 9:
        # CI omits SM90 kernels. Check the binary before launching: a failed
        # FlashMLA call can leave CUDA errors behind even after pytest.skip.
        cuobjdump = shutil.which("cuobjdump")
        if cuobjdump is None:
            pytest.skip("cuobjdump is required to check FlashMLA's SM90 kernel availability")
        kernels = subprocess.check_output(
            [cuobjdump, "--list-elf", flash_mla_cuda.__file__], text=True, timeout=30
        )
        if ".sm_90" not in kernels:
            pytest.skip("FlashMLA was built without SM90 kernels")
    monkeypatch.setattr(DSAIndexerLossLoggingHelper, "save_loss_to_tracker", lambda **kwargs: None)
    cfg, _ = _setup()
    cfg = replace(
        cfg,
        params_dtype=torch.bfloat16,
        bf16=True,
        num_attention_heads=64,
        v_head_dim=512,
        q_lora_rank=128,
        qk_pos_emb_head_dim=64,
        csa_compress_ratios=[ratio] * 3,
        dsa_indexer_n_heads=64,
        dsa_indexer_head_dim=128,
        dsa_indexer_topk=32,
        csa_window_size=8,
        csa2_candidate_source_layer=0,
        csa2_candidate_topk_blocks=4,
        csa2_candidate_block_size=8,
    )
    g = Group()
    groups = ProcessGroupCollection(tp=g, cp=g)
    modules = CompressedSparseAttentionSubmodules(
        compressor=partial(
            CSA2Compressor,
            submodules=CompressorSubmodules(linear_wkv=Linear, linear_wgate=Linear, norm=Norm),
        ),
        indexer=partial(
            CSA2Indexer,
            submodules=CSA2IndexerSubmodules(
                linear_wq_b=Linear, linear_wk=Linear, k_norm=Norm, linear_weights_proj=Linear
            ),
        ),
    )
    rope = RotaryEmbedding(64, rotary_percent=1.0, cp_group=g)
    native = (
        nn.ModuleList(
            [
                CompressedSparseAttention2(
                    cfg, modules, i + 1, AttnMaskType.causal, 'self', groups, rotary_pos_emb=rope
                )
                for i in range(3)
            ]
        )
        .cuda()
        .bfloat16()
    )
    fused = (
        nn.ModuleList(
            [
                CompressedSparseAttention2(
                    replace(cfg, dsa_kernel_backend='cudnn'),
                    modules,
                    i + 1,
                    AttnMaskType.causal,
                    'self',
                    groups,
                    rotary_pos_emb=rope,
                )
                for i in range(3)
            ]
        )
        .cuda()
        .bfloat16()
    )
    # Attention sinks are FP32 in the real module, including BF16 training.
    for layers in (native, fused):
        for layer in layers:
            layer.attn_sink.data = layer.attn_sink.data.float()
    fused.load_state_dict(native.state_dict())
    shapes = ((32, 2, 64, 512), (32, 2, 1, 512), (32, 2, 16), (32, 2, 128))
    inputs = [
        tuple(torch.randn(shape, device='cuda', dtype=torch.bfloat16) for shape in shapes)
        for _ in range(3)
    ]
    results = []
    for layers in (native, fused):
        state, outputs, leaves = CSA2State(), [], []
        for layer, values in zip(layers, inputs):
            q, kv, x, qr = [v.detach().clone().requires_grad_() for v in values]
            leaves.extend((q, kv, x, qr))
            outputs.append(layer(q, kv, kv, None, x=x, qr=qr, csa2_state=state))
        sum(y.float().square().mean() for y in outputs).backward()
        grads = [p.grad for p in layers.parameters()] + [x.grad for x in leaves]
        results.append((outputs, grads))
    for actual, expected in zip(results[1][0], results[0][0]):
        cosine = torch.nn.functional.cosine_similarity(
            actual.float().flatten(), expected.float().flatten(), dim=0
        )
        assert cosine > 0.999
    for actual, expected in zip(results[1][1], results[0][1]):
        if expected is None:
            assert actual is None
            continue
        assert actual is not None and torch.isfinite(actual).all()
        norm = expected.float().norm()
        if norm < 1e-7:
            assert actual.float().norm() < 1e-6
            continue
        cosine = torch.nn.functional.cosine_similarity(
            actual.float().flatten(), expected.float().flatten(), dim=0
        )
        assert cosine > 0.99
        assert (actual.float() - expected.float()).norm() / norm < 0.03


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "ratio,heads,cp", [(r, h, c) for r in (1, 2) for h in (32, 64) for c in (1, 2)]
)
@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
def test_packed_selection_matches_scoring_precision(monkeypatch, ratio, heads, cp, precision):
    """Check compact/source/Reindex selection across empty, padded and CP-cut segments."""
    from dataclasses import replace
    from types import SimpleNamespace

    from megatron.core.ops.attention.csa.indexer_quantization import quantize_indexer_mxfp8_logical
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_candidates import (
        candidate_blocks_from_scores,
        fused_candidate_blocks,
    )
    from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_indexer import (
        fused_candidate_topk,
        fused_compact_topk,
        prepare_csa2_indexer_inputs,
    )
    from megatron.core.transformer.experimental_attention_variant.csa_utils.thd_utils import (
        build_csa2_thd_layout,
    )
    from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact

    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("Compact selection requires SM100")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    torch.manual_seed(487)
    logical = torch.tensor([0, 65, 65, 322], dtype=torch.int32, device="cuda")
    physical = torch.tensor([0, 66, 66, 326], dtype=torch.int32, device="cuda")
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=logical,
        cu_seqlens_kv=logical,
        cu_seqlens_q_padded=physical,
        cu_seqlens_kv_padded=physical,
        max_seqlen_q=260,
        max_seqlen_kv=260,
    )
    group = SimpleNamespace(size=lambda: cp, rank=lambda: cp - 1)
    layout = build_csa2_thd_layout(params, 326 // cp, cp_group=group)
    compressed = layout.for_compression(ratio)
    q = torch.randn(326 // cp, heads, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(compressed.capacity, 128, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(q.shape[0], heads, dtype=torch.bfloat16, device="cuda")
    inputs = prepare_csa2_indexer_inputs(q, k, w, ratio, None, layout, compressed)

    def dequantize(x):
        if precision == "bf16":
            return x.float()
        data, scales = quantize_indexer_mxfp8_logical(x)
        scale = torch.exp2(scales.float() - 127).repeat_interleave(32, -1)
        return (data.float().reshape(-1, 128) * scale).reshape(x.shape)

    weights = (w.float() * (128**-0.5 * heads**-0.5)).bfloat16().float()
    scores = (
        torch.einsum('qhd,kd->qhk', dequantize(q), dequantize(k)).relu() * weights[..., None]
    ).sum(1)
    keys = torch.arange(k.shape[0], device=q.device)[None, :]
    valid = (keys >= inputs.starts[:, None]) & (keys < (inputs.starts + inputs.visible)[:, None])
    scores.masked_fill_(~valid | ~inputs.valid[:, None], -torch.inf)

    def topk(s):
        ids = s.argsort(dim=-1, descending=True, stable=True)[:, :7]
        sentinel = torch.iinfo(torch.int32).max
        ids = ids.masked_fill(~s.gather(1, ids).isfinite(), sentinel).sort(-1).values
        return ids.masked_fill(ids == sentinel, -1).int()

    expected = topk(scores)
    torch.testing.assert_close(fused_compact_topk(inputs, 7, precision), expected)
    selected, blocks = fused_candidate_blocks(
        inputs, topk_blocks=3, block_size=8, precision=precision, token_topk=7
    )
    torch.testing.assert_close(selected, expected)
    local_keys = inputs.starts[:, None] + torch.arange(inputs.max_keys, device=q.device)
    local_scores = scores.gather(1, local_keys.clamp_max(k.shape[0] - 1))
    local_scores.masked_fill_(local_keys >= (inputs.starts + inputs.visible)[:, None], -torch.inf)
    expected_blocks = candidate_blocks_from_scores(local_scores, inputs.visible, 3, 8)
    torch.testing.assert_close(blocks.indices, expected_blocks.indices)
    torch.testing.assert_close(blocks.lengths, expected_blocks.lengths)
    mask = blocks.to_mask(k.shape[0], thd_layout=layout, compressed_layout=compressed)
    torch.testing.assert_close(
        fused_candidate_topk(replace(inputs, candidates=blocks), 7, precision),
        topk(scores.masked_fill(~mask, -torch.inf)),
    )

    def select(query, keys, weights):
        prepared = prepare_csa2_indexer_inputs(
            query, keys, weights, ratio, None, layout, compressed
        )
        source_topk, candidates = fused_candidate_blocks(
            prepared, topk_blocks=3, block_size=8, precision=precision, token_topk=7
        )
        return (
            fused_compact_topk(prepared, 7, precision),
            source_topk,
            candidates.indices,
            fused_candidate_topk(replace(prepared, candidates=candidates), 7, precision),
        )

    assert_replays_bit_exact(
        select, (q, k, w), backward=False, replays=3, what=f"CSA2 {precision} packed selection"
    )
