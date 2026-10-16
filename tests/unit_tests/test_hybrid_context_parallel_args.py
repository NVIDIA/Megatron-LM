# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""validate_args must reject CUDA graphs with dynamic context parallelism.

A captured graph bakes in static communicators, so it cannot follow the per-microbatch
context-parallel group that dynamic context parallelism selects at runtime.
"""

import sys

import pytest

from megatron.training.arguments import parse_args, validate_args

_GUARD = "Dynamic context parallelism not supported with CUDA Graph"


def _hybrid_cp_args(monkeypatch, **overrides):
    monkeypatch.setattr(sys, "argv", ["test_hybrid_context_parallel_args.py"])
    args = parse_args()
    # parse_args reads WORLD_SIZE from the environment; pin it so the test does not depend
    # on how the job was launched.
    args.world_size = 1
    args.num_layers = 2
    args.hidden_size = 128
    args.num_attention_heads = 4
    args.max_position_embeddings = 1024
    args.seq_length = 1024
    args.micro_batch_size = 1
    args.global_batch_size = 1
    args.train_iters = 1
    args.lr = 1e-4
    args.tokenizer_type = "NullTokenizer"
    args.vocab_size = 1024
    args.dynamic_context_parallel = True
    args.max_seqlen_per_dp_cp_rank = 1024
    args.calculate_per_token_loss = True
    args.dataloader_type = "single"
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@pytest.mark.parametrize(
    "cuda_graph_impl, prerequisites",
    [
        ("local", {}),
        ("transformer_engine", {}),
        # full_iteration has its own prerequisite, checked before the hybrid-CP guard.
        ("full_iteration", {"check_for_nan_in_loss_and_grad": False}),
    ],
)
def test_hybrid_context_parallel_rejects_cuda_graph_impl(
    monkeypatch, cuda_graph_impl, prerequisites
):
    args = _hybrid_cp_args(monkeypatch, cuda_graph_impl=cuda_graph_impl, **prerequisites)
    with pytest.raises(AssertionError, match=_GUARD):
        validate_args(args)


@pytest.mark.parametrize("deprecated_flag", ["enable_cuda_graph", "external_cuda_graph"])
def test_hybrid_context_parallel_rejects_deprecated_cuda_graph_flags(monkeypatch, deprecated_flag):
    # validate_args translates these into cuda_graph_impl and deletes them before the
    # hybrid-CP checks run, so the guard must read cuda_graph_impl.
    args = _hybrid_cp_args(monkeypatch, **{deprecated_flag: True})
    with pytest.raises(AssertionError, match=_GUARD):
        validate_args(args)


def test_hybrid_context_parallel_without_cuda_graphs_validates(monkeypatch):
    args = _hybrid_cp_args(monkeypatch)
    validate_args(args)
    assert args.cuda_graph_impl == "none"
