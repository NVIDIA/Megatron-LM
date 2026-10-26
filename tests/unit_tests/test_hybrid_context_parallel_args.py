# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The config built from args must reject CUDA graphs with dynamic context parallelism.

A captured graph bakes in static communicators, so it cannot follow the per-microbatch
context-parallel group that dynamic context parallelism selects at runtime.
"""

import sys

import pytest

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.arguments import core_transformer_config_from_args, parse_args, validate_args

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
    validate_args(args)
    with pytest.raises(AssertionError, match=_GUARD):
        core_transformer_config_from_args(args, TransformerConfig)


@pytest.mark.parametrize("deprecated_flag", ["enable_cuda_graph", "external_cuda_graph"])
def test_hybrid_context_parallel_rejects_deprecated_cuda_graph_flags(monkeypatch, deprecated_flag):
    # validate_args translates these into cuda_graph_impl and deletes them before the
    # config is built, so the guard must read cuda_graph_impl.
    args = _hybrid_cp_args(monkeypatch, **{deprecated_flag: True})
    validate_args(args)
    with pytest.raises(AssertionError, match=_GUARD):
        core_transformer_config_from_args(args, TransformerConfig)


def test_hybrid_context_parallel_without_cuda_graphs_validates(monkeypatch):
    args = _hybrid_cp_args(monkeypatch)
    validate_args(args)
    assert args.cuda_graph_impl == "none"
    assert core_transformer_config_from_args(args, TransformerConfig).dynamic_context_parallel


def test_hybrid_builder_rejects_dynamic_context_parallel(monkeypatch):
    from hybrid_builders import hybrid_builder

    args = _hybrid_cp_args(monkeypatch)
    with pytest.raises(NotImplementedError, match="not yet supported for HybridModel"):
        hybrid_builder(args, pre_process=True, post_process=True)


def test_deprecated_hybrid_context_parallel_flag_warns_and_maps(monkeypatch):
    args = _hybrid_cp_args(
        monkeypatch, rank=0, dynamic_context_parallel=False, hybrid_context_parallel=True
    )
    with pytest.warns(UserWarning, match="--hybrid-context-parallel is deprecated"):
        validate_args(args)
    assert args.dynamic_context_parallel
    assert not args.hybrid_context_parallel


@pytest.mark.parametrize("use_varlen_dataset,micro_batch_size", [(False, 1), (True, 2)])
def test_static_packed_graph_prerequisites_are_preserved(
    monkeypatch, use_varlen_dataset, micro_batch_size
):
    """Scoping the graph guard must not relax static packing's input requirements."""
    args = _hybrid_cp_args(
        monkeypatch,
        dynamic_context_parallel=False,
        sequence_packing_scheduler="dp_balanced",
        cuda_graph_impl="transformer_engine",
        use_varlen_dataset=use_varlen_dataset,
        micro_batch_size=micro_batch_size,
        global_batch_size=4,
    )
    with pytest.raises(AssertionError, match="Scheduler CUDA graphs require one"):
        validate_args(args)
