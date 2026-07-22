# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import os

import pytest
import torch

from megatron.core.transformer.custom_layers.batch_invariant_kernels import set_batch_invariant_mode
from megatron.rl.rl_utils import selective_log_softmax


def test_selective_log_softmax_batch_invariant():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    B, S, V = 4, 7, 16
    device = torch.device("cuda")
    logits = torch.randn(B, S, V, dtype=torch.float32, device=device)
    labels = torch.randint(low=0, high=V, size=(B, S), device=device)

    # Randomly permute the batch dimension; a batch-invariant implementation should
    # produce outputs that are identical up to the same permutation.
    perm = torch.randperm(B, device=device)

    with set_batch_invariant_mode(True):
        bik_logps = selective_log_softmax(logits, labels)  # [B, S]
        bik_logps_perm = selective_log_softmax(
            logits[perm], labels[perm]
        )  # [B, S] corresponding to permuted batch

    # Undo the permutation on the permuted outputs and compare elementwise.
    # If the kernel is batch invariant, each example's output should not depend
    # on its position in the batch.
    assert torch.equal(bik_logps, bik_logps_perm[perm.argsort()])


def test_moe_unpermute_batch_invariant_inverse_map_rank_tree():
    from megatron.core import parallel_state
    from megatron.core.transformer.moe.moe_utils import unpermute

    hidden = 4
    tokens = torch.tensor(
        [[1e20], [1.0], [-1e20], [1.0]], device="cuda", dtype=torch.float32
    ).expand(4, hidden)
    sorted_indices = torch.zeros(4, device="cuda", dtype=torch.int64)
    routing_map = torch.ones(1, 4, device="cuda", dtype=torch.bool)
    inverse_map = torch.tensor(
        [[[0, 1, 2, 3]], [[0, 1, 2, 3]]], device="cuda", dtype=torch.int64
    )

    parallel_state.set_expert_model_parallel_world_size(2)
    try:
        with set_batch_invariant_mode(True):
            out = unpermute(
                tokens,
                sorted_indices,
                (1, hidden),
                routing_map=routing_map,
                batch_invariant_inverse_map=inverse_map,
            )
    finally:
        parallel_state.set_expert_model_parallel_world_size(None)

    torch.testing.assert_close(out[0], torch.zeros(hidden, device="cuda"), rtol=0.0, atol=0.0)


def test_moe_batch_invariant_permute_unpermute_cuda_graph_non_padded():
    from megatron.core import parallel_state
    from megatron.core.transformer.moe.moe_utils import permute, unpermute

    torch.manual_seed(123)
    num_tokens, hidden, num_experts, topk = 6, 8, 4, 2
    tokens = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    routing_map = torch.zeros(num_tokens, num_experts, device="cuda", dtype=torch.bool)
    routing_map[:, 0] = True
    routing_map[:, 2] = True
    probs = torch.rand(num_tokens, num_experts, device="cuda", dtype=torch.float32)

    def _run():
        permuted, _, sorted_indices, inverse_map, _ = permute(
            tokens,
            routing_map,
            probs=probs,
            num_out_tokens=num_tokens * topk,
            return_batch_invariant_inverse_map=True,
        )
        return unpermute(
            permuted,
            sorted_indices,
            tokens.shape,
            probs=probs,
            routing_map=routing_map,
            batch_invariant_inverse_map=inverse_map,
        )

    parallel_state.set_expert_model_parallel_world_size(2)
    try:
        with torch.no_grad(), set_batch_invariant_mode(True):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    expected = _run()
            torch.cuda.current_stream().wait_stream(stream)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_out = _run()
            graph.replay()
    finally:
        parallel_state.set_expert_model_parallel_world_size(None)

    torch.testing.assert_close(graph_out, expected, rtol=0.0, atol=0.0)
    reference = (
        tokens.float() * probs[:, 0, None] + tokens.float() * probs[:, 2, None]
    ).to(tokens.dtype)
    torch.testing.assert_close(graph_out, reference, rtol=0.0, atol=0.0)
