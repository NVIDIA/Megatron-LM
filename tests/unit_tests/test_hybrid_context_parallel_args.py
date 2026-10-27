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


@pytest.mark.parametrize("world_size,gtp_size", [(4, 1), (4, 2), (8, 2)])
def test_dynamic_cp_accepts_full_data_domain_minimum(monkeypatch, world_size, gtp_size):
    args = _hybrid_cp_args(
        monkeypatch,
        world_size=world_size,
        context_parallel_size=2,
        tensor_parallel_num_weight_shards=gtp_size,
        global_batch_size=world_size,
        min_dynamic_context_parallel_size=world_size,
    )
    validate_args(args)
    assert args.batch_parallel_size * args.context_parallel_size == world_size
    assert args.data_parallel_size == world_size // (2 * gtp_size)


def test_dynamic_cp_rejects_minimum_larger_than_full_data_domain(monkeypatch):
    args = _hybrid_cp_args(
        monkeypatch,
        world_size=4,
        context_parallel_size=2,
        tensor_parallel_num_weight_shards=2,
        global_batch_size=4,
        min_dynamic_context_parallel_size=8,
    )
    with pytest.raises(ValueError, match="invalid for DPxCP size 4"):
        validate_args(args)


@pytest.mark.parametrize(
    "world_size,tp_size,pp_size,cp_size,gtp_size,sequence_shards",
    [
        (4, 1, 1, 2, 1, None),
        (4, 1, 1, 2, 2, None),
        (8, 1, 1, 2, 2, None),
        (8, 2, 1, 2, 2, None),
        (8, 1, 2, 2, 2, None),
        # Normalized static CP already includes GTP sequence shards; do not
        # multiply those coordinates twice when computing DCP capacity.
        (4, 1, 1, 1, 2, 2),
    ],
)
def test_dynamic_cp_packed_capacity_includes_full_data_domain(
    monkeypatch, world_size, tp_size, pp_size, cp_size, gtp_size, sequence_shards
):
    capacity = world_size // (tp_size * pp_size) * 1024
    args = _hybrid_cp_args(
        monkeypatch,
        world_size=world_size,
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        context_parallel_size=cp_size,
        tensor_parallel_num_weight_shards=tp_size * gtp_size,
        tensor_parallel_num_sequence_shards=sequence_shards,
        global_batch_size=world_size,
        seq_length=capacity,
        max_position_embeddings=capacity,
    )
    validate_args(args)
    assert args.batch_parallel_size * args.context_parallel_size * 1024 == capacity
    assert args.sequence_packing_scheduler == "default_dynamic_cp"


@pytest.mark.parametrize(
    "dynamic,seq_length,valid,sequence_shards",
    [
        (True, 4096, True, None),
        (True, 4100, False, None),
        (False, 2048, True, None),
        (False, 4096, False, None),
        (True, 4100, False, 2),
    ],
)
def test_packed_capacity_rejects_excess_without_expanding_static_cp(
    monkeypatch, dynamic, seq_length, valid, sequence_shards
):
    args = _hybrid_cp_args(
        monkeypatch,
        world_size=4,
        context_parallel_size=1 if sequence_shards == 2 else 2,
        tensor_parallel_num_weight_shards=2,
        tensor_parallel_num_sequence_shards=sequence_shards,
        global_batch_size=4,
        dynamic_context_parallel=dynamic,
        sequence_packing_scheduler="default_dynamic_cp" if dynamic else "dp_balanced",
        seq_length=seq_length,
        max_position_embeddings=seq_length,
    )
    if valid:
        validate_args(args)
    else:
        with pytest.raises(AssertionError, match="Packed sequence buffer size"):
            validate_args(args)
