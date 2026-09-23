# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Aligned HCA dispatch, stock parity, and graph replay checks."""

from argparse import ArgumentParser
from dataclasses import fields
from unittest.mock import Mock, patch

import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams
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


def _metadata(cp_rank=0):
    """Make a padded single-sequence pack without allocating full attention inputs."""
    q = torch.empty(1, device="cuda", dtype=torch.bfloat16).expand(4096, 128, 512)
    kv = torch.empty(1, device="cuda", dtype=torch.bfloat16).expand(4752, 512)
    cu = torch.tensor([0] + [65536] * 8, device="cuda", dtype=torch.int32)
    params = PackedSeqParams(
        qkv_format="thd",
        cp_partition_mode="contiguous",
        cu_seqlens_q=torch.tensor([0, 65533], device="cuda", dtype=torch.int32),
        cu_seqlens_q_padded=cu,
        max_seqlen_q=65536,
    )
    group = Mock()
    group.size.return_value = 16
    group.rank.return_value = cp_rank
    return q, kv, params, group


def _select(q, kv, params, group, **kwargs):
    """Select with the supported HCA geometry."""
    options = dict(window_size=128, compress_ratio=128, boundary_rows=128)
    options.update(kwargs)
    return hca.aligned_hca_cp_rank(q, kv, params, group, **options)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_packing_changes_are_rechecked():
    """Padding capacity is not a sequence count, and a reused buffer is not a layout proof."""
    q, kv, params, group = _metadata(cp_rank=15)
    with (
        patch.object(torch.cuda, "get_device_capability", return_value=(10, 3)),
        patch.object(hca, "_get_aligned_hca_backward"),
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
    "change", ["cp", "partition", "window", "ratio", "boundary", "q", "kv", "dtype", "device"]
)
def test_unsupported_layout_falls_back(change):
    """Unsupported layouts never import the optional API."""
    q, kv, params, group = _metadata()
    options = {}
    capability = (10, 3)
    if change == "cp":
        group.size.return_value = 8
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
    else:
        capability = (10, 0)
    with (
        patch.object(torch.cuda, "get_device_capability", return_value=capability),
        patch.object(hca, "_get_aligned_hca_backward") as loader,
    ):
        assert _select(q, kv, params, group, **options) is None
        loader.assert_not_called()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_capture_guards_current_boundaries():
    """Capture emits a tensor guard instead of reusing an eager shape verdict."""
    q, kv, params, group = _metadata()
    with (
        patch.object(torch.cuda, "get_device_capability", return_value=(10, 3)),
        patch.object(hca, "_get_aligned_hca_backward"),
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
@pytest.mark.parametrize("cp_rank", [0, 1, 8, 15])
@pytest.mark.parametrize("reconstruct", [False, True])
def test_stock_gradients_and_bit_exact_graph_replay(cp_rank, reconstruct):
    """Exercise MCore-generated indices and its actual autograd backward dispatch."""
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("Aligned HCA requires GB300")
    cudnn = pytest.importorskip("cudnn")
    if not hasattr(cudnn, "aligned_hca_backward_wrapper"):
        pytest.skip("cuDNN Frontend aligned HCA API is not installed")
    pytest.importorskip("flash_mla")
    torch.manual_seed(9100 + cp_rank)
    _, _, params, group = _metadata(cp_rank)
    cu = params.cu_seqlens_q_padded
    comp = layout.build_cp_compressor_layout(cu, cp_rank * 4096, 4096, 16, 128)
    indices, lengths, _, _ = layout.build_attention_indices(
        cu,
        cp_rank * 4096,
        4096,
        128,
        128,
        128,
        512,
        cu_seqlens_compressed=comp.cu_seqlens_compressed,
        seq_to_rank_row=comp.seq_to_rank_row,
        compressed_rows=528,
        output_alignment=64,
    )
    q = torch.randn(4096, 128, 512, device="cuda", dtype=torch.bfloat16).requires_grad_()
    kv = torch.randn(4752, 512, device="cuda", dtype=torch.bfloat16).requires_grad_()
    sink = torch.randn(128, device="cuda", dtype=torch.float32).requires_grad_()
    grad = torch.randn_like(q).reshape(4096, -1)
    parts = tuple(part.detach() for part in kv.split((128, 4096, 528))) if reconstruct else None
    rope = None
    if reconstruct:
        angles = torch.randn(65536, 32, device="cuda")
        rope = sparse.OutputRopeParams(
            cos=angles.cos().repeat(1, 2).to(torch.bfloat16).view(65536, 1, 1, 64),
            sin=angles.sin().repeat(1, 2).to(torch.bfloat16).view(65536, 1, 1, 64),
            nope_dim=448,
            pos_dim=64,
            cu_seqlens_q=cu,
            position_ids=torch.arange(cp_rank * 4096, (cp_rank + 1) * 4096, device="cuda"),
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
