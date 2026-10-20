# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Retained dev THD graph paths require fixed padded token capacity.

Main's blanket hybrid-CP rejection does not apply to the dev implementation. Keep
dev's local-graph rejection, static-metadata prerequisites and legacy normalization.
"""

import sys

import pytest

from megatron.training.arguments import parse_args, validate_args

_GUARD = "THD CUDA Graph requires --pad-packed-seq-alignment"
_LOCAL_GUARD = "Dynamic context parallelism not supported with local CUDA Graph"


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
    args.hybrid_context_parallel = True
    args.max_seqlen_per_dp_cp_rank = 1024
    args.calculate_per_token_loss = True
    args.dataloader_type = "single"
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@pytest.mark.parametrize(
    "cuda_graph_impl, prerequisites",
    [
        ("transformer_engine", {}),
        # full_iteration also requires NaN checking to be disabled during capture.
        ("full_iteration", {"check_for_nan_in_loss_and_grad": False}),
    ],
)
def test_hybrid_context_parallel_cuda_graph_requires_static_padding(
    monkeypatch, cuda_graph_impl, prerequisites
):
    args = _hybrid_cp_args(monkeypatch, cuda_graph_impl=cuda_graph_impl, **prerequisites)
    with pytest.raises(ValueError, match=_GUARD):
        validate_args(args)


@pytest.mark.parametrize("deprecated_flag", ["enable_cuda_graph", "external_cuda_graph"])
def test_hybrid_context_parallel_legacy_cuda_graph_flags_use_canonical_guards(
    monkeypatch, deprecated_flag
):
    # validate_args translates these into cuda_graph_impl and deletes them before the
    # hybrid-CP checks run, so the guard must read cuda_graph_impl.
    args = _hybrid_cp_args(monkeypatch, **{deprecated_flag: True})
    error = AssertionError if deprecated_flag == "enable_cuda_graph" else ValueError
    message = _LOCAL_GUARD if deprecated_flag == "enable_cuda_graph" else _GUARD
    with pytest.raises(error, match=message):
        validate_args(args)


@pytest.mark.parametrize("alignment", [None, "max"])
def test_hybrid_context_parallel_rejects_local_graphs(monkeypatch, alignment):
    args = _hybrid_cp_args(monkeypatch, cuda_graph_impl="local", pad_packed_seq_alignment=alignment)
    with pytest.raises(AssertionError, match=_LOCAL_GUARD):
        validate_args(args)


@pytest.mark.parametrize("alignment", ["max", 1024])
@pytest.mark.parametrize("cuda_graph_impl", ["transformer_engine", "full_iteration"])
def test_hybrid_context_parallel_fixed_padding_validates(monkeypatch, cuda_graph_impl, alignment):
    args = _hybrid_cp_args(
        monkeypatch,
        cuda_graph_impl=cuda_graph_impl,
        pad_packed_seq_alignment=alignment,
        check_for_nan_in_loss_and_grad=False,
    )
    validate_args(args)
    assert args.cuda_graph_impl == cuda_graph_impl
    assert args.dynamic_context_parallel is True


@pytest.mark.parametrize("alignment", [128, 512])
def test_hybrid_context_parallel_cuda_graph_rejects_nonstatic_padding(monkeypatch, alignment):
    args = _hybrid_cp_args(
        monkeypatch, cuda_graph_impl="transformer_engine", pad_packed_seq_alignment=alignment
    )
    with pytest.raises(ValueError, match=_GUARD):
        validate_args(args)


def test_hybrid_context_parallel_without_cuda_graphs_validates(monkeypatch):
    args = _hybrid_cp_args(monkeypatch)
    validate_args(args)
    assert args.cuda_graph_impl == "none"


@pytest.mark.parametrize("capacity", [None, 1024])
def test_dynamic_cp_buffer_capacity_uses_dp_times_cp(monkeypatch, capacity):
    args = _hybrid_cp_args(
        monkeypatch,
        world_size=8,
        global_batch_size=8,
        seq_length=8192,
        max_position_embeddings=8192,
        max_seqlen_per_dp_cp_rank=capacity,
    )
    validate_args(args)
    assert args.data_parallel_size == 8
    assert args.context_parallel_size == 1
    assert args.max_seqlen_per_dp_cp_rank == 1024
    assert args.sequence_packing_scheduler == "default_dynamic_cp"
    assert args.variable_seq_lengths is True


def test_static_packing_buffer_capacity_does_not_include_dp(monkeypatch):
    args = _hybrid_cp_args(
        monkeypatch,
        world_size=8,
        global_batch_size=8,
        seq_length=8192,
        max_position_embeddings=8192,
        hybrid_context_parallel=False,
        sequence_packing_scheduler="dp_balanced",
        max_seqlen_per_dp_cp_rank=1024,
    )
    with pytest.raises(AssertionError, match="Packed sequence buffer size"):
        validate_args(args)
