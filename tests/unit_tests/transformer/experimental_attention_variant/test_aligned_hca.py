# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Aligned HCA dispatch, stock parity, and graph replay checks."""

import sys
from argparse import ArgumentParser
from dataclasses import fields
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.experimental_attention_variant import csa
from megatron.core.transformer.experimental_attention_variant.csa_utils import aligned_hca as hca
from megatron.core.transformer.experimental_attention_variant.csa_utils import (
    fused_sparse_attention as sparse,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils import (
    thd_layout_kernels as layout,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.argument_utils import ArgumentGroupFactory


def test_cli_default_and_opt_in():
    """The public config field supplies the normal CLI flag."""
    parser = ArgumentParser()
    excluded = [f.name for f in fields(TransformerConfig) if f.name != "hca_aligned_backward"]
    ArgumentGroupFactory(TransformerConfig, exclude=excluded).build_group(parser)
    assert parser.parse_args([]).hca_aligned_backward is False
    assert parser.parse_args(["--hca-aligned-backward"]).hca_aligned_backward is True


@pytest.mark.parametrize("missing_api", [None, ModuleType("cudnn")])
def test_optional_api_import_and_cache(missing_api):
    """Missing APIs report the requirement; successful imports are cached."""
    api = ModuleType("cudnn")
    api.AlignedHCABackward = object()
    api.aligned_hca_backward_wrapper = Mock()
    expected = (api.AlignedHCABackward, api.aligned_hca_backward_wrapper)
    hca._get_aligned_hca_backward.cache_clear()
    try:
        with patch.dict(sys.modules, {"cudnn": missing_api}):
            with pytest.raises(ImportError, match="NVIDIA/cudnn-frontend#1198"):
                hca._get_aligned_hca_backward()
        with patch.dict(sys.modules, {"cudnn": api}):
            assert hca._get_aligned_hca_backward() == expected
        with patch.dict(sys.modules, {"cudnn": None}):
            assert hca._get_aligned_hca_backward() == expected
    finally:
        hca._get_aligned_hca_backward.cache_clear()


@pytest.mark.parametrize("cp_rank,cp_size", [(None, 16), (0, 4), (7, 8), (15, 16)])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("reconstruct", [False, True])
def test_backward_dispatch_and_strides(cp_rank, cp_size, deterministic, reconstruct):
    """Autograd routes gradients and passes contiguous inputs to aligned HCA."""
    q = torch.randn(3, 2, 8, device="cpu")[..., ::2].requires_grad_()
    kv = torch.randn(5, 8, device="cpu")[:, ::2].requires_grad_()
    sink = torch.randn(4, device="cpu")[::2].requires_grad_()
    out = torch.randn(3, 2, 8, device="cpu")[..., ::2]
    lse = torch.randn(2, 6, device="cpu")[:, ::2]
    topk = torch.zeros(3, 2, device="cpu", dtype=torch.int32)
    lengths = torch.full((3,), 2, device="cpu", dtype=torch.int32)
    parts = tuple(part.detach() for part in kv.split((1, 3, 1))) if reconstruct else None
    dout = torch.randn(3, 2, 8, device="cpu")[..., ::2]
    grads = dict(
        dq=torch.full_like(q, 1), dkv=torch.full_like(kv, 2), d_sink=torch.full_like(sink, 3)
    )
    aligned = Mock(return_value=grads)
    stock = Mock()
    stock.sparse_attention_backward_wrapper.return_value = grads
    with (
        patch.object(sparse, "_csa_fwd_flash_mla", return_value=(out, lse, None)) as forward,
        patch.object(sparse, "get_flash_mla_topk_alignment", return_value=64),
        patch.object(sparse, "_DSA", stock),
        patch.object(
            sparse,
            "_deterministic_sparse_bwd_kwargs",
            return_value={"deterministic": deterministic},
        ) as deterministic_kwargs,
        patch.object(hca, "_get_aligned_hca_backward", return_value=(Mock(), aligned)) as loader,
    ):
        result = sparse.csa_sparse_attn(
            q,
            kv,
            sink,
            topk,
            0.125,
            topk_length=lengths,
            is_thd=True,
            kv_reconstruction_parts=parts,
            deterministic=deterministic,
            cuda_graph=deterministic,
            hca_cp_rank=cp_rank,
            hca_cp_size=cp_size,
        )
        torch.testing.assert_close(result, out.reshape(3, -1))
        context = result.grad_fn.next_functions[0][0]
        if cp_rank is None:
            assert context.saved_tensors[-3] is forward.call_args.args[2]
            assert context.topk_length is lengths
        else:
            assert context.saved_tensors[-3] is None
            assert context.topk_length is None
        result.backward(dout.reshape_as(result))
    for actual, expected in zip((q.grad, kv.grad, sink.grad), grads.values()):
        torch.testing.assert_close(actual, expected)
    if cp_rank is None:
        loader.assert_not_called()
        aligned.assert_not_called()
        stock.sparse_attention_backward_wrapper.assert_called_once()
        assert (
            stock.sparse_attention_backward_wrapper.call_args.kwargs["deterministic"]
            is deterministic
        )
        assert deterministic_kwargs.call_args.kwargs == dict(
            requested=deterministic, cuda_graph=deterministic
        )
        assert stock.sparse_attention_backward_wrapper.call_args.kwargs["topk_length"] is lengths
    else:
        deterministic_kwargs.assert_not_called()
        stock.sparse_attention_backward_wrapper.assert_not_called()
        aligned.assert_called_once()
        assert aligned.call_args.kwargs == dict(
            cp_rank=cp_rank, cp_size=cp_size, softmax_scale=0.125
        )
        for actual, expected in zip(
            aligned.call_args.args, (q, kv, out, dout, lse, sink), strict=True
        ):
            assert actual.is_contiguous()
            torch.testing.assert_close(actual, expected)


def test_aligned_dispatch_requires_explicit_cp_size():
    """Missing CP size must fail before invoking either attention backend."""
    with patch.object(sparse.CSASparseAttnFunc, "apply") as apply:
        with pytest.raises(ValueError, match="hca_cp_size is required"):
            sparse.csa_sparse_attn(None, None, None, None, 0.125, hca_cp_rank=0)
        apply.assert_not_called()
    with pytest.raises(TypeError, match="cp_size"):
        hca.aligned_hca_backward(None, None, None, None, None, None, cp_rank=0, softmax_scale=0.125)


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("grad_enabled", [False, True])
def test_thd_cp_forward_selects_hca_only_with_grad(training, grad_enabled):
    """Eval and checkpoint no-grad forwards skip backward-only eligibility checks."""
    module = csa.CompressedSparseAttention.__new__(csa.CompressedSparseAttention)
    torch.nn.Module.__init__(module)
    module.use_fused_kernels = True
    module.compress_ratio = module.window_size = 128
    module.softmax_scale = 512**-0.5
    module.attn_sink = torch.nn.Parameter(torch.zeros(128, device="cpu"))
    group = Mock()
    group.size.return_value, group.rank.return_value = 8, 7
    module.pg_collection = SimpleNamespace(cp=group)
    module.config = SimpleNamespace(
        hca_aligned_backward=True,
        dsa_indexer_loss_coeff=0.0,
        dsa_indexer_use_sparse_loss=True,
        recompute_granularity=None,
        deterministic_mode=False,
        cuda_graph_impl="transformer_engine",
    )
    module.indexer = None
    module.compressor = Mock()
    module.compressor._cp_requires_single_projection.return_value = False
    scalar = torch.zeros(1, device="cpu", dtype=torch.bfloat16)
    query = scalar.expand(4096, 128, 512)
    key = scalar.expand(4096, 1, 1, 512)
    x = scalar.expand(4096, 1, 512)
    boundary = scalar.expand(128, 1, 512)
    module.compressor._forward_thd_cp.return_value = (scalar.expand(33, 1, 512), None)
    cu = torch.tensor([0, 32768], device="cpu", dtype=torch.int32)
    params = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, max_seqlen_q=32768)
    params.cp_partition_mode = "contiguous"
    expected = query.reshape(4096, -1)
    module.train(training)
    with (
        torch.set_grad_enabled(grad_enabled),
        patch.object(
            csa, "_OutputInverseRope", return_value=SimpleNamespace(fused_params=object())
        ),
        patch.object(csa, "nvtx_range_push"),
        patch.object(csa, "nvtx_range_pop"),
        patch.object(csa, "get_flash_mla_topk_alignment", return_value=64),
        patch.object(layout, "build_cp_compressor_layout", return_value=Mock()),
        patch.object(layout, "build_attention_indices", return_value=(None,) * 4),
        patch.object(
            csa, "gather_from_sequence_parallel_region", return_value=scalar.expand(264, 512)
        ),
        patch.object(csa, "aligned_hca_cp_rank", return_value=7) as select,
        patch.object(csa, "csa_sparse_attn", return_value=expected) as attention,
    ):
        actual = module._forward_thd_cp(query, key, x, x, boundary, boundary.unsqueeze(2), params)
    assert actual.shape == (4096, 1, 128 * 512)
    assert actual.data_ptr() == expected.data_ptr()
    assert attention.call_args.kwargs["hca_cp_rank"] == (7 if grad_enabled else None)
    assert attention.call_args.kwargs["hca_cp_size"] == 8
    if grad_enabled:
        select.assert_called_once()
    else:
        select.assert_not_called()


def _metadata(cp_rank=0, sequence_length=65536, cp_size=16):
    """Make a padded single-sequence pack without allocating full attention inputs."""
    local_tokens = sequence_length // cp_size
    compressed_rows = sequence_length // 128 + cp_size
    q = torch.empty(1, device="cuda", dtype=torch.bfloat16).expand(local_tokens, 128, 512)
    kv = torch.empty(1, device="cuda", dtype=torch.bfloat16).expand(
        local_tokens + 128 + compressed_rows, 512
    )
    cu = torch.tensor([0] + [sequence_length] * 8, device="cuda", dtype=torch.int32)
    params = PackedSeqParams(
        qkv_format="thd",
        cp_partition_mode="contiguous",
        cu_seqlens_q=torch.tensor([0, sequence_length - 3], device="cuda", dtype=torch.int32),
        cu_seqlens_q_padded=cu,
        max_seqlen_q=sequence_length,
    )
    group = Mock()
    group.size.return_value = cp_size
    group.rank.return_value = cp_rank
    return q, kv, params, group


def _select(q, kv, params, group, **kwargs):
    """Select with the supported HCA geometry."""
    options = dict(window_size=128, compress_ratio=128, boundary_rows=128)
    options.update(kwargs)
    return hca.aligned_hca_cp_rank(q, kv, params, group, **options)


@pytest.mark.parametrize(
    "boundaries,expected",
    [([0, 65536, 65536], True), ([0, 32768, 65536], False), ([0, 0, 65536], False)],
)
def test_prepared_packing_avoids_repeated_readback(boundaries, expected):
    """Batch preparation supplies the same verdict to every eager layer."""
    cu = torch.tensor(boundaries, device="cpu", dtype=torch.int32)
    params = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, max_seqlen_q=65536)
    hca.prepare_aligned_hca_metadata(params)
    with patch.object(torch.Tensor, "item", side_effect=AssertionError("repeated readback")):
        for _ in range(15):
            assert hca._is_single_sequence(params, cu, 65536) is expected


def test_prepared_packing_invalidates_on_changes():
    """Tensor writes, replacements and different lengths cannot reuse a stale verdict."""
    cu = torch.tensor([0, 65536, 65536], device="cpu", dtype=torch.int32)
    params = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, max_seqlen_q=65536)
    hca.prepare_aligned_hca_metadata(params)
    assert not hca._is_single_sequence(params, cu, 32768)
    replacement = torch.tensor([0, 32768, 65536], device="cpu", dtype=torch.int32)
    assert not hca._is_single_sequence(params, replacement, 65536)
    cu[1] = 32768
    assert not hca._is_single_sequence(params, cu, 65536)
    hca.prepare_aligned_hca_metadata(params)
    cu[1] = 65536
    assert hca._is_single_sequence(params, cu, 65536)


def test_prepared_packing_without_version_counter():
    """Inference tensors retain live validation when mutation tracking is unavailable."""
    with torch.inference_mode():
        cu = torch.tensor([0, 65536, 65536], device="cpu", dtype=torch.int32)
        params = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, max_seqlen_q=65536)
        hca.prepare_aligned_hca_metadata(params)
        assert hca._is_single_sequence(params, cu, 65536)
        cu[1] = 32768
        assert not hca._is_single_sequence(params, cu, 65536)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_capture_unsupported_backend_does_not_assert():
    """A backend that declines HCA must not record an HCA-only graph assertion."""
    q, kv, params, group = _metadata()
    params.cu_seqlens_q_padded[1] = 32768
    api = Mock()
    api.supports_configuration.return_value = False
    with (
        patch.object(hca, "_get_aligned_hca_backward", return_value=(api, Mock())),
        patch.object(torch, "_assert_async") as guard,
    ):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            assert _select(q, kv, params, group) is None
        graph.replay()
        guard.assert_not_called()
        api.supports_configuration.assert_called_once()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prepared_dispatch_does_not_synchronize():
    """Prepared metadata removes the per-layer device-to-host synchronization."""
    q, kv, params, group = _metadata()
    hca.prepare_aligned_hca_metadata(params)
    previous = torch.cuda.get_sync_debug_mode()
    try:
        with patch.object(hca, "_get_aligned_hca_backward", return_value=(Mock(), Mock())):
            torch.cuda.set_sync_debug_mode("error")
            for _ in range(15):
                assert _select(q, kv, params, group) == 0
    finally:
        torch.cuda.set_sync_debug_mode(previous)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_packing_changes_are_rechecked():
    """Padding capacity is not a sequence count, and a reused buffer is not a layout proof."""
    q, kv, params, group = _metadata(cp_rank=15)
    with (
        patch.object(torch.cuda, "get_device_capability", return_value=(10, 3)),
        patch.object(hca, "_get_aligned_hca_backward", return_value=(Mock(), Mock())),
    ):
        assert _select(q, kv, params, group) == 15
        params.cu_seqlens_q_padded[1] = 32768
        assert _select(q, kv, params, group) is None
        params.cu_seqlens_q_padded[1] = 65536
        assert _select(q, kv, params, group) == 15
        params.cu_seqlens_q_padded = None
        assert _select(q, kv, params, group) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "change", ["cp", "partition", "window", "ratio", "boundary", "q", "kv", "dtype"]
)
def test_unsupported_layout_falls_back(change):
    """Unsupported layouts never import the optional API."""
    q, kv, params, group = _metadata()
    options = {}
    capability = (10, 3)
    if change == "cp":
        group.size.return_value = 2
    elif change == "partition":
        params.cp_partition_mode = "zigzag"
    elif change == "window":
        options["window_size"] = 64
    elif change == "ratio":
        options["compress_ratio"] = 4
    elif change == "boundary":
        options["boundary_rows"] = 64
    elif change == "q":
        q = q[:2048]
    elif change == "kv":
        kv = kv[:4740]
    elif change == "dtype":
        q = torch.empty(1, device="cuda", dtype=torch.float16).expand_as(q)
    with (
        patch.object(torch.cuda, "get_device_capability", return_value=capability),
        patch.object(hca, "_get_aligned_hca_backward") as loader,
    ):
        assert _select(q, kv, params, group, **options) is None
        loader.assert_not_called()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("prepared", [False, True])
def test_capture_guards_current_boundaries(prepared):
    """Capture emits a tensor guard instead of reusing an eager shape verdict."""
    q, kv, params, group = _metadata()
    if prepared:
        hca.prepare_aligned_hca_metadata(params)
    with (
        patch.object(torch.cuda, "get_device_capability", return_value=(10, 3)),
        patch.object(hca, "_get_aligned_hca_backward", return_value=(Mock(), Mock())),
        patch.object(torch, "_assert_async") as guard,
    ):
        assert _select(q, kv, params, group) == 0
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            assert _select(q, kv, params, group) == 0
        predicate = guard.call_args.args[0]
        graph.replay()
        assert predicate.item()
        params.cu_seqlens_q_padded[1] = 32768
        graph.replay()
        assert not predicate.item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "sequence_length,cp_size,cp_rank",
    [
        (65536, 16, 0),
        (65536, 16, 1),
        (65536, 16, 8),
        (65536, 16, 15),
        (8192, 4, 3),
        (49152, 8, 7),
        (10240, 16, 15),
    ],
)
@pytest.mark.parametrize("reconstruct", [False, True])
def test_stock_gradients_and_bit_exact_graph_replay(sequence_length, cp_size, cp_rank, reconstruct):
    """Exercise MCore-generated indices and its actual autograd backward dispatch."""
    if torch.cuda.get_device_capability() not in ((10, 3), (10, 7)):
        pytest.skip("Aligned HCA requires GB300 or Rubin")
    cudnn = pytest.importorskip("cudnn")
    if not hasattr(cudnn, "aligned_hca_backward_wrapper"):
        pytest.skip("cuDNN Frontend aligned HCA API is not installed")
    pytest.importorskip("flash_mla")
    torch.manual_seed(9100 + cp_rank)
    local_tokens = sequence_length // cp_size
    compressed_rows = sequence_length // 128 + cp_size
    _, _, params, group = _metadata(cp_rank, sequence_length, cp_size)
    cu = params.cu_seqlens_q_padded
    comp = layout.build_cp_compressor_layout(cu, cp_rank * local_tokens, local_tokens, cp_size, 128)
    indices, lengths, _, _ = layout.build_attention_indices(
        cu,
        cp_rank * local_tokens,
        local_tokens,
        128,
        128,
        128,
        sequence_length // 128,
        cu_seqlens_compressed=comp.cu_seqlens_compressed,
        seq_to_rank_row=comp.seq_to_rank_row,
        compressed_rows=compressed_rows,
        output_alignment=64,
    )
    q = torch.randn(local_tokens, 128, 512, device="cuda", dtype=torch.bfloat16).requires_grad_()
    kv = torch.randn(
        local_tokens + 128 + compressed_rows, 512, device="cuda", dtype=torch.bfloat16
    ).requires_grad_()
    sink = torch.randn(128, device="cuda", dtype=torch.float32).requires_grad_()
    grad = torch.randn_like(q).reshape(local_tokens, -1)
    parts = (
        tuple(part.detach() for part in kv.split((128, local_tokens, compressed_rows)))
        if reconstruct
        else None
    )
    rope = None
    if reconstruct:
        angles = torch.randn(sequence_length, 32, device="cuda")
        rope = sparse.OutputRopeParams(
            cos=angles.cos().repeat(1, 2).to(torch.bfloat16).view(sequence_length, 1, 1, 64),
            sin=angles.sin().repeat(1, 2).to(torch.bfloat16).view(sequence_length, 1, 1, 64),
            nope_dim=448,
            pos_dim=64,
            cu_seqlens_q=cu,
            position_ids=torch.arange(
                cp_rank * local_tokens, (cp_rank + 1) * local_tokens, device="cuda"
            ),
        )

    def run(rank):
        output = sparse.csa_sparse_attn(
            q,
            kv,
            sink,
            indices,
            512**-0.5,
            topk_length=lengths,
            is_thd=True,
            kv_reconstruction_parts=parts,
            hca_cp_rank=rank,
            hca_cp_size=cp_size,
            out_rope=rope,
        )
        return torch.autograd.grad(output, (q, kv, sink), grad.clone())

    baseline = run(None)
    assert _select(q, kv, params, group) == cp_rank
    actual = run(cp_rank)
    for got, expected in zip(actual, baseline):
        torch.testing.assert_close(got, expected, atol=0.03, rtol=0.03)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run(cp_rank)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        rank = _select(q, kv, params, group)
        captured = run(rank)
    for _ in range(3):
        graph.replay()
        for got, expected in zip(captured, actual):
            assert torch.equal(
                got.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "sequence_length,cp_size", [(8192, 4), (49152, 8), (10240, 16), (131072, 4)]
)
def test_frontend_capability_controls_dispatch(sequence_length, cp_size):
    q, kv, params, group = _metadata(cp_size - 1, sequence_length, cp_size)
    api = Mock()
    with patch.object(hca, "_get_aligned_hca_backward", return_value=(api, Mock())):
        api.supports_configuration.return_value = True
        assert _select(q, kv, params, group) == cp_size - 1
        api.supports_configuration.assert_called_with(sequence_length // cp_size, cp_size, q.device)
        api.supports_configuration.return_value = False
        assert _select(q, kv, params, group) is None
