# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Training/decode forward parity, CuTe DSL top-k, and graph refresh."""

import logging

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.dsa_gqa_inference import (
    TritonDSADecodeWorkspace,
)
from megatron.core.transformer.experimental_attention_variant.dsa_min_memory import (
    _sort_topk_support_by_position,
)
from megatron.core.transformer.experimental_attention_variant.dsa_min_memory_triton import (
    triton_simplified_index_scores_block,
    triton_sparse_attention_tile,
)

logger = logging.getLogger(__name__)


def _training_forward(query, index_query, index_keys, keys, values, selected, topk):
    """Call the scoring and attention kernels used by simplified DSA-GQA training."""
    length = index_keys.shape[0]
    index_scale = index_query.shape[-1] ** -0.5
    # Future keys would win routing without the training kernel's causal mask.
    future = index_query.expand(16, -1) * 64
    index_keys = torch.cat((index_keys, future), dim=0)[:, None, None, :]
    keys = torch.cat((keys, query[0].expand(16, -1) * 64), dim=0)[:, None, None, :]
    future_values = values.new_full((16, values.shape[-1]), 32)
    values = torch.cat((values, future_values), dim=0)[:, None, None, :]
    query = query[None, None, :, :]
    scores = triton_simplified_index_scores_block(
        index_query[None, None, None, :], index_keys, index_scale, length - 1, 0
    )
    assert (
        scores is not None
    ), "Training scorer must execute its Triton kernel, not a torch fallback"
    assert bool(torch.isneginf(scores[..., length:]).all())
    selected_scores, indices = scores.topk(min(topk, length), dim=-1)
    _, indices = _sort_topk_support_by_position(selected_scores, indices)
    training_output = triton_sparse_attention_tile(
        query, keys, values, indices, query.shape[-1] ** -0.5, length - 1
    )
    same_support_output = triton_sparse_attention_tile(
        query, keys, values, selected[None, None, :], query.shape[-1] ** -0.5, length - 1
    )
    assert (
        training_output is not None and same_support_output is not None
    ), "Training attention must execute its Triton kernel, not a torch fallback"
    return scores[0, 0, :length], indices[0, 0], training_output[0, 0], same_support_output[0, 0]


@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("seed", [71, 2026, 17])
@pytest.mark.parametrize("context", [32768, 262144])
@pytest.mark.parametrize("topk", [512, 1024, 2048])
def test_native_decode_matches_training_kernels_and_graph_refresh(head_dim, seed, context, topk):
    """Decode matches the training forward kernels on the final causal query row."""
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Native DSA decode requires Blackwell")
    pytest.importorskip("cutlass.cute")
    torch.manual_seed(seed)
    device = torch.device("cuda", torch.cuda.current_device())
    batch, page, heads = 8, 256, 16
    pages = batch * context // page
    query = torch.randn(batch, heads, head_dim, device=device, dtype=torch.bfloat16)
    index_query = torch.randn(batch, 128, device=device, dtype=torch.bfloat16)
    index_keys = torch.randn(pages, page, 128, device=device, dtype=torch.bfloat16)
    keys = torch.randn(pages, page, 1, head_dim, device=device, dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    table = torch.randperm(pages, device=device).to(torch.int32).view(batch, -1)
    requests = torch.randperm(batch, device=device).to(torch.int32)
    lengths_cpu = [0, 1, 31, 63, 511, 2048, 32767, 32768]
    if context > 65536:
        lengths_cpu = [0, 1, 2047, 2048, 65535, 65536, 200000, context]
    lengths = torch.tensor(lengths_cpu, device=device, dtype=torch.int32)
    workspace = TritonDSADecodeWorkspace(
        batch_size=batch,
        num_query_heads=heads,
        head_dim=head_dim,
        index_topk=topk,
        max_context=context,
        page_size=page,
        device=device,
    )

    def run():
        return workspace(
            query=query,
            index_query=index_query,
            index_key_cache=index_keys,
            key_cache=keys,
            value_cache=values,
            block_table=table,
            context_lengths=lengths,
            request_indices=requests,
            softmax_scale=head_dim**-0.5,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    max_score_error = 0.0
    max_output_error = 0.0
    for _ in range(2):
        query.neg_()
        index_query.neg_()
        table.copy_(table.roll(1, dims=0))
        graph.replay()
        actual = workspace.output.clone()
        selected = workspace.indices.clone()
        expected = torch.zeros_like(actual)
        same_support = torch.zeros_like(actual)
        for row, length in enumerate(lengths_cpu):
            if not length:
                assert bool((selected[row] == -1).all())
                continue
            logical = torch.arange(length, device=device)
            physical = table[requests[row].long(), logical // page].long()
            chosen = selected[row][selected[row] >= 0].long()
            assert chosen.numel() == min(topk, length)
            assert chosen.unique().numel() == chosen.numel()
            assert bool((chosen < length).all())
            training_scores, training_indices, training_output, same_support_output = (
                _training_forward(
                    query[row],
                    index_query[row],
                    index_keys[physical, logical % page],
                    keys[physical, logical % page, 0],
                    values[physical, logical % page, 0],
                    chosen,
                    topk,
                )
            )
            expected[row] = training_output
            same_support[row] = same_support_output
            # Decode ranks raw dot products; training applies a positive scalar.
            scaled_scores = workspace.scores[row, :length] * 128**-0.5
            torch.testing.assert_close(scaled_scores, training_scores, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(
                chosen.sort().values, training_indices.sort().values, atol=0, rtol=0
            )
            max_score_error = max(
                max_score_error, (scaled_scores - training_scores).abs().max().item()
            )
        # Blackwell training retains FP32 probabilities; decode rounds BF16 weights
        # before PV. Keep this budget uniform across shapes.
        torch.testing.assert_close(actual, same_support, atol=3e-3, rtol=2e-2)
        torch.testing.assert_close(actual, expected, atol=3e-3, rtol=2e-2)
        max_output_error = max(
            max_output_error, (actual.float() - expected.float()).abs().max().item()
        )
        graph.replay()
        torch.testing.assert_close(workspace.output, actual, atol=0, rtol=0)
        torch.testing.assert_close(workspace.indices, selected, atol=0, rtol=0)

    logger.info(
        "Training/decode parity: seed=%s head_dim=%s max_score_abs=%s max_output_abs=%s",
        seed,
        head_dim,
        max_score_error,
        max_output_error,
    )


@pytest.mark.parametrize("width,topk", [(516, 512), (4096, 1024), (32772, 2048), (65536, 2048)])
def test_cute_topk_cutoffs_ties_and_graph_replay(width, topk):
    """Short/empty rows, ties, infinities, and unaligned lengths select exact values."""
    pytest.importorskip("cutlass.cute")
    torch.manual_seed(71)
    device = torch.device("cuda", torch.cuda.current_device())
    batch = 6
    from megatron.core.transformer.experimental_attention_variant.dsa_gqa_topk import (
        CuteTopKWorkspace,
    )

    alignment = 4
    stride = -(-width // alignment) * alignment
    scores = torch.randn(batch, stride, device=device)
    lengths = torch.tensor([width - 3, 1, 0, width, width, width], device=device, dtype=torch.int32)
    scores[3].zero_()
    scores[4, :width].copy_(torch.arange(width, device=device).float() / width)
    scores[4, 0] = -1e20
    scores[4, 1] = 1e20
    scores[5, ::3] = float("inf")
    scores[5, 1::3] = -float("inf")
    indices = torch.empty(batch, topk, device=device, dtype=torch.int32)

    selector = CuteTopKWorkspace(scores, lengths, indices)

    def run():
        selector()

    run()
    expected = indices.clone()
    for row, length in enumerate(lengths.tolist()):
        count = min(topk, length)
        valid = indices[row][indices[row] >= 0].long()
        assert valid.numel() == count
        assert valid.unique().numel() == count
        assert bool((valid < length).all())
        assert bool((indices[row, count:] == -1).all())
        if count:
            torch.testing.assert_close(
                scores[row, valid].sort().values,
                scores[row, :length].topk(count).values.sort().values,
                atol=0,
                rtol=0,
            )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.testing.assert_close(indices, expected, atol=0, rtol=0)
