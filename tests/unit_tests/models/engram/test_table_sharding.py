# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Engram tables sharded over DP x CP (``EngramConfig.parallel_size``) against the EP layout.

Three kinds of cases:

* The configuration, command-line, builder and source checks run in any process.
* Group creation, its reuse and the absence of a warm-up collective each spawn their own 4-rank
  Gloo world on CPU, whatever the launcher's world size.
* The lookup, state-dict and checkpoint cases need exactly four ranks: dp_cp = [0, 1, 2, 3]
  holding the expert grid EP2 x expert-DP2 (``rank = ep + 2 * edp``). They run with NCCL on
  GPUs or with Gloo on CPU, and carry ``launch_on_gb200`` so CI runs them on its 4-GPU nodes::

    torchrun --nproc-per-node 4 -m pytest tests/unit_tests/models/engram/test_table_sharding.py

The EP layout keeps two replicas of every row shard and sums their gradients over expert-DP, as
DDP's expert buffer does. ``parallel_size=4`` gives every row one owner, which receives all four
ranks' gradients through the lookup's backward all-to-all and only applies the 1/dp scaling.
``parallel_size=2`` shards the rows over the EP blocks of dp_cp and replicates them over the
expert-DP pairs: the EP layout's groups. With exactly representable (integer) gradients every
layout agrees bit for bit. With random bf16 gradients, ``parallel_size=4`` differs from the EP
layout only on rows that both expert-DP replicas touch in the same microbatch: the EP layout
rounds each replica's partial sum to bf16 before adding them, the single owner rounds the whole
sum once.

One case sets the expert grid to EP1 or EP4 instead. ``parallel_size=2`` then shards and
replicates the rows over groups that the expert grid does not have.
"""

import hashlib
import os
import shutil
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from megatron.core.models.engram import distributed_embedding, table_sharding
from megatron.core.models.engram.config import EngramConfig
from megatron.core.models.engram.distributed_embedding import EPShardedMultiTableEmbedding
from megatron.core.models.engram.table_sharding import (
    ENGRAM_PARAMETER_DOMAIN,
    clear_engram_parallel_state,
    initialize_engram_parallel_state,
    resolve_engram_groups,
    tag_engram_tables,
)
from megatron.core.parallel_state import RankGenerator
from megatron.core.process_groups_config import (
    ProcessGroupCollection,
    get_parameter_parallel_domain,
)
from megatron.core.utils import get_pg_rank, get_pg_size

WORLD_SIZE = 4
EP_SIZE = 2
EXPERT_DP_SIZE = 2
TABLE_SIZES = (11, 13, 5)
EMBEDDING_DIM = 4
NUM_MICROBATCHES = 2
# parallel_size of the sharded layouts: one owner per row, and the EP layout's groups.
SHARDED_SIZES = (WORLD_SIZE, EP_SIZE)
# Rank order of parallel_state's groups.
ORDER = "tp-cp-ep-dp-pp"


def _engram_config(parallel_size=None):
    """A small configuration; the qwen variant needs no tokenizer artifact."""
    return EngramConfig(
        global_vocab_sizes=(11, 11),
        layer_ids=(1,),
        max_ngram_order=3,
        num_hash_heads=1,
        memory_dim=4,
        kernel_size=4,
        hash_seed=0,
        boundary_token_id=0,
        variant="qwen",
        unigram_vocab_size=32,
        parallel_size=parallel_size,
    )


def _ranks(group):
    return dist.get_process_group_ranks(group)


# ---------------------------------------------------------------------------------------------
# Source, configuration, command line and builders (any process)
# ---------------------------------------------------------------------------------------------

# Git blob id of megatron/core/models/engram/distributed_embedding.py in NVIDIA/Megatron-LM#8225,
# as `git rev-parse <PR head>:megatron/core/models/engram/distributed_embedding.py` prints it.
_PR_8225_BLOB = "f34791982461a6dbd21bcaacceded54a391b25da"


def test_distributed_embedding_is_the_pr_8225_copy():
    """Sharding reuses PR 8225's lookup through its group arguments and must not fork it."""
    source = Path(distributed_embedding.__file__).read_bytes()
    blob = hashlib.sha1(b"blob %d\0" % len(source) + source, usedforsecurity=False).hexdigest()
    assert blob == _PR_8225_BLOB, (
        "megatron/core/models/engram/distributed_embedding.py no longer matches its copy in "
        "NVIDIA/Megatron-LM#8225; change it there and update the blob id here."
    )


@pytest.mark.parametrize("parallel_size", [0, -4, 2.0, True, "4"])
def test_parallel_size_must_be_a_positive_integer(parallel_size):
    with pytest.raises(ValueError, match="engram_parallel_size must be a positive integer"):
        _engram_config(parallel_size)


def _transformer_config(**overrides):
    """The TransformerConfig fields that EngramConfig.validate_startup reads."""
    values = dict(
        num_layers=2,
        tensor_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        gtp_weight_remat_size=1,
        expert_gtp_weight_remat_size=1,
        context_parallel_size=1,
        dynamic_context_parallel=False,
        sequence_parallel=False,
        pipeline_model_parallel_size=1,
        fp16=False,
        fp4=None,
        transformer_impl="transformer_engine",
        overlap_moe_expert_parallel_comm=False,
        wide_residual=None,
        moe_shortcut_connection=False,
        use_mup=False,
        cuda_graph_impl="none",
        cuda_graph_modules=[],
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(tensor_model_parallel_size=2, expert_tensor_parallel_size=2),
        dict(expert_tensor_parallel_size=2),
        dict(gtp_weight_remat_size=2),
        dict(expert_gtp_weight_remat_size=2),
    ],
    ids=["tp", "expert_tp", "gtp", "expert_gtp"],
)
def test_sharded_tables_reject_tensor_and_gtp_sharding(overrides):
    """dp_cp holds the expert grid, and the norms count each row shard once, only without
    tensor, expert-tensor and GTP sharding."""
    _engram_config(WORLD_SIZE).validate_startup(_transformer_config())
    first_field = next(iter(overrides))
    with pytest.raises(ValueError, match=f"{first_field}=2"):
        _engram_config(WORLD_SIZE).validate_startup(_transformer_config(**overrides))
    if overrides.get("expert_tensor_parallel_size", 1) == overrides.get(
        "tensor_model_parallel_size", 1
    ):
        # The EP layout keeps every layout it supported (it needs expert TP equal to TP).
        _engram_config().validate_startup(_transformer_config(**overrides))


_TRAINING_ARGV = (
    "test_table_sharding.py --num-layers 2 --hidden-size 128 --num-attention-heads 4 "
    "--max-position-embeddings 1024 --seq-length 1024 --micro-batch-size 1 --train-iters 1 "
    "--lr 1e-4 --tokenizer-type NullTokenizer --vocab-size 1024"
)
# Engram on EP2 x expert-DP2, which needs no MoE experts.
_ENGRAM_FLAGS = (
    "--engram-vocab-sizes 1009 1009 --engram-layer-ids 2 --engram-variant qwen "
    "--engram-eos-token-id 1 --engram-unigram-vocab-size 2048 --expert-model-parallel-size 2"
)


def _validate_cli(monkeypatch, flags, engram=True, world_size=WORLD_SIZE):
    """Parse and validate the arguments of a run, of four ranks unless told otherwise."""
    from megatron.training.arguments import parse_args, validate_args

    monkeypatch.setenv("WORLD_SIZE", str(world_size))
    argv = f"{_TRAINING_ARGV} {_ENGRAM_FLAGS if engram else ''} {flags}"
    monkeypatch.setattr(sys, "argv", argv.split())
    args = parse_args()
    validate_args(args)
    return args


@pytest.mark.parametrize(
    "flags",
    [
        "--engram-parallel-size 4 --use-distributed-optimizer",
        "--engram-parallel-size 2 --use-distributed-optimizer",
        # Muon with the distributed optimizer becomes the layer-wise optimizer, whose sibling
        # DistributedOptimizers take the tables.
        "--engram-parallel-size 4 --optimizer muon --use-distributed-optimizer",
    ],
    ids=["one_owner", "replicated", "muon"],
)
def test_cli_accepts_the_supported_layouts(monkeypatch, flags):
    args = _validate_cli(monkeypatch, flags)
    assert args.engram_parallel_size == int(flags.split()[1])


@pytest.mark.parametrize(
    ("flags", "error"),
    [
        ("--engram-parallel-size 3 --use-distributed-optimizer", "must divide"),
        ("--engram-parallel-size 4", "requires --use-distributed-optimizer"),
        (
            "--engram-parallel-size 4 --optimizer muon --use-distributed-optimizer "
            "--no-use-layer-wise-param-layout",
            "remove --no-use-layer-wise-param-layout",
        ),
        (
            "--engram-parallel-size 2 --use-distributed-optimizer "
            "--expert-tensor-parallel-size 2",
            "expert_tensor_parallel_size=2",
        ),
        (
            "--engram-parallel-size 4 --use-distributed-optimizer "
            "--num-distributed-optimizer-instances 2",
            "num-distributed-optimizer-instances 1",
        ),
        (
            "--engram-parallel-size 4 --use-distributed-optimizer --overlap-grad-reduce "
            "--overlap-param-gather --overlap-param-gather-with-optimizer-step",
            "overlap-param-gather-with-optimizer-step",
        ),
        (
            "--engram-parallel-size 4 --use-distributed-optimizer "
            "--tensor-metrics global-param-l2:1",
            "does not support --tensor-metrics",
        ),
        (
            "--engram-parallel-size 4 --use-distributed-optimizer --ckpt-format torch",
            "not --ckpt-format torch",
        ),
        # A conversion run replaces --ckpt-format with --ckpt-convert-format after validation.
        (
            "--engram-parallel-size 4 --use-distributed-optimizer --load /nonexistent "
            "--ckpt-convert-format torch --ckpt-convert-save /nonexistent",
            "not --ckpt-convert-format torch",
        ),
        (
            "--engram-parallel-size 4 --use-distributed-optimizer "
            "--dist-ckpt-optim-fully-reshardable --distrib-optim-fully-reshardable-mem-efficient",
            "mem-efficient",
        ),
        # Engram's own FSDP check fires first, in every layout.
        (
            "--engram-parallel-size 4 --use-distributed-optimizer --use-torch-fsdp2",
            "Engram does not support Megatron-FSDP or Torch FSDP2",
        ),
    ],
    ids=[
        "indivisible",
        "no_distributed_optimizer",
        "muon_legacy_layer_wise_layout",
        "expert_tp",
        "optimizer_instances",
        "param_gather_with_optimizer_step",
        "tensor_metrics",
        "torch_checkpoints",
        "torch_checkpoint_conversion",
        "mem_efficient_checkpoints",
        "fsdp",
    ],
)
def test_cli_rejects_unsupported_layouts(monkeypatch, flags, error):
    with pytest.raises(AssertionError, match=error):
        _validate_cli(monkeypatch, flags)


@pytest.mark.parametrize("flag", ["use_megatron_fsdp", "use_torch_fsdp2"])
def test_cli_table_domain_rejects_fsdp(monkeypatch, flag):
    """The sharded layout's own FSDP check, which Engram's general one otherwise hides."""
    from megatron.training.arguments import _validate_engram_parallel_size

    args = _validate_cli(monkeypatch, "--engram-parallel-size 4 --use-distributed-optimizer")
    setattr(args, flag, True)
    with pytest.raises(AssertionError, match="--engram-parallel-size does not support"):
        _validate_engram_parallel_size(args)


def test_cli_fake_process_group_needs_one_pipeline_stage(monkeypatch):
    """A fake process group cannot exchange the dp_cp groups of the other pipeline stages."""
    from megatron.training.arguments import _validate_engram_parallel_size

    args = _validate_cli(
        monkeypatch, "--engram-parallel-size 4 --use-distributed-optimizer --fake-process-group"
    )
    assert args.fake_process_group and args.pipeline_model_parallel_size == 1
    args.pipeline_model_parallel_size = 2
    with pytest.raises(AssertionError, match="--fake-process-group requires"):
        _validate_engram_parallel_size(args)


@pytest.mark.parametrize(
    "field",
    [
        "tensor_model_parallel_size",
        "expert_tensor_parallel_size",
        "gtp_weight_remat_size",
        "expert_gtp_weight_remat_size",
    ],
)
def test_cli_rejects_tensor_and_gtp_sharding(monkeypatch, field):
    """validate_args queries the GPU while it checks tensor-parallel and GTP layouts, so the
    Engram check runs here on otherwise valid arguments."""
    from megatron.training.arguments import _validate_engram_parallel_size

    args = _validate_cli(monkeypatch, "--engram-parallel-size 4 --use-distributed-optimizer")
    setattr(args, field, 2)
    with pytest.raises(AssertionError, match=f"{field}=2"):
        _validate_engram_parallel_size(args)


@pytest.mark.parametrize("flag", ["--engram-parallel-size", "--engram-max-parallel-size"])
def test_cli_parallel_size_requires_engram(monkeypatch, flag):
    with pytest.raises(AssertionError, match=f"{flag} requires Engram"):
        _validate_cli(monkeypatch, f"{flag} 4 --use-distributed-optimizer", False)


@pytest.mark.parametrize(
    ("world_size", "parallel_size"), [(2, 2), (4, 4), (8, 4)], ids=["below", "at", "above"]
)
def test_cli_max_parallel_size_bounds_the_row_shards(monkeypatch, world_size, parallel_size):
    """The same arguments at every job size: one owner per row up to the bound, and above it
    as many row shards as the bound, each replicated."""
    args = _validate_cli(
        monkeypatch,
        "--engram-max-parallel-size 4 --use-distributed-optimizer",
        world_size=world_size,
    )
    assert args.data_parallel_size * args.context_parallel_size == world_size
    assert args.engram_parallel_size == parallel_size
    assert args.engram_max_parallel_size == 4


def test_cli_max_parallel_size_bounds_the_dp_cp_group(monkeypatch):
    """The bound applies to the data- times context-parallel size of one pipeline stage, not to
    the world size or to the data-parallel size alone. validate_args queries the GPU while it
    checks a context-parallel layout, so the second layout is set on validated arguments."""
    from megatron.training.arguments import _resolve_engram_max_parallel_size

    args = _validate_cli(
        monkeypatch, "--engram-max-parallel-size 8 --use-distributed-optimizer", world_size=8
    )
    assert args.engram_parallel_size == 8
    # The same eight ranks as two pipeline stages of DP2 x CP2 each.
    args.engram_parallel_size = None
    args.pipeline_model_parallel_size = 2
    args.data_parallel_size, args.context_parallel_size = 2, 2
    _resolve_engram_max_parallel_size(args)
    assert args.engram_parallel_size == 4


@pytest.mark.parametrize(
    ("flags", "world_size", "error"),
    [
        # Six ranks are more than the bound, which does not divide them.
        (
            "--engram-max-parallel-size 4 --use-distributed-optimizer",
            6,
            r"--engram-max-parallel-size \(4\) must divide .* \(6\)",
        ),
        (
            "--engram-max-parallel-size 4 --engram-parallel-size 4 --use-distributed-optimizer",
            WORLD_SIZE,
            "are mutually exclusive, got 4 and 4",
        ),
        (
            "--engram-max-parallel-size 0 --use-distributed-optimizer",
            WORLD_SIZE,
            "--engram-max-parallel-size must be positive, got 0",
        ),
        (
            "--engram-max-parallel-size=-4 --use-distributed-optimizer",
            WORLD_SIZE,
            "--engram-max-parallel-size must be positive, got -4",
        ),
        # The size it sets goes through the checks of --engram-parallel-size.
        ("--engram-max-parallel-size 4", WORLD_SIZE, "requires --use-distributed-optimizer"),
    ],
    ids=["indivisible", "with_parallel_size", "zero", "negative", "no_distributed_optimizer"],
)
def test_cli_rejects_unsupported_max_parallel_sizes(monkeypatch, flags, world_size, error):
    with pytest.raises(AssertionError, match=error):
        _validate_cli(monkeypatch, flags, world_size=world_size)


@pytest.mark.parametrize(
    "ep, dp, pp, cp", [(2, 4, 1, 1), (4, 4, 1, 1), (2, 4, 2, 1), (2, 2, 1, 2), (4, 4, 1, 2)]
)
def test_dp_cp_holds_the_expert_grid_in_ep_blocks(ep, dp, pp, cp):
    """With TP = 1, every EP group is a block of consecutive ranks of its stage's dp_cp group.

    So create_row_parallel_groups(dp_cp, size(EP)) makes the EP groups the row groups and the
    expert-DP groups their replica groups: parallel_size = size(EP) is the EP layout.
    """
    dense = RankGenerator(tp=1, ep=1, dp=dp, pp=pp, cp=cp, order=ORDER)
    expert = RankGenerator(tp=1, ep=ep, dp=dp * cp // ep, pp=pp, cp=1, order=ORDER)
    for parent in dense.get_ranks("dp-cp"):
        members = set(parent)
        row_groups = [parent[start : start + ep] for start in range(0, len(parent), ep)]
        replica_groups = [parent[offset::ep] for offset in range(ep)]
        assert sorted(row_groups) == sorted(g for g in expert.get_ranks("ep") if set(g) <= members)
        assert sorted(replica_groups) == sorted(
            g for g in expert.get_ranks("dp") if set(g) <= members
        )


def test_builder_sets_up_the_groups_before_building_the_model(monkeypatch):
    """Every rank sets the groups up on the collection that the stages and DDP get, first."""
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.training.models import hybrid

    events = []
    monkeypatch.setattr(
        hybrid,
        "initialize_engram_parallel_state",
        lambda pg_collection, config: events.append(("initialize", pg_collection, config)),
    )

    def build_stages(build_model, transformer_config, pg_collection, *args, **kwargs):
        events.append(("build", pg_collection))
        return [object()]

    monkeypatch.setattr(hybrid, "unimodal_build_distributed_models", build_stages)
    engram_config = _engram_config(WORLD_SIZE)
    model_config = hybrid.HybridModelConfig(
        transformer=TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=1),
        vocab_size=1024,
        engram_config=engram_config,
    )
    pg_collection = object()
    hybrid.HybridModelBuilder(model_config).build_distributed_models(pg_collection)
    assert events == [("initialize", pg_collection, engram_config), ("build", pg_collection)]


def test_legacy_hybrid_builder_rejects_table_sharding():
    from hybrid_builders import hybrid_builder

    args = SimpleNamespace(dynamic_context_parallel=False, engram_parallel_size=WORLD_SIZE)
    with pytest.raises(NotImplementedError, match="--engram-parallel-size"):
        hybrid_builder(args, pre_process=True, post_process=True)


def test_resolve_requires_initialized_groups():
    """The groups are never created lazily: that collective would hang under pipelining."""
    pg_collection = ProcessGroupCollection()
    assert resolve_engram_groups(pg_collection, _engram_config()) == (None, None, "expert")
    with pytest.raises(RuntimeError, match="initialize_engram_parallel_state"):
        resolve_engram_groups(pg_collection, _engram_config(WORLD_SIZE))


def test_destroy_model_parallel_clears_the_engram_state(monkeypatch):
    """destroy_model_parallel finds this module by its name, without importing it; a renamed or
    moved module would leave stale groups behind, so the clean-up is checked end to end."""
    from megatron.core import parallel_state

    stale = object()
    monkeypatch.setitem(table_sharding._ROW_GROUPS, (stale, WORLD_SIZE), (stale, stale))
    parallel_state.destroy_model_parallel()
    assert table_sharding._ROW_GROUPS == {}


# ---------------------------------------------------------------------------------------------
# Group creation in a spawned 4-rank Gloo world (CPU)
# ---------------------------------------------------------------------------------------------

# A rank that blocks is the failure these cases guard against, and groups outlive the world
# timeout, so the parent kills the world and fails once the deadline passes instead of hanging.
_GLOO_DEADLINE_S = 300


def _gloo_worker(rank, store_path, case):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=WORLD_SIZE,
        timeout=timedelta(seconds=_GLOO_DEADLINE_S),
    )
    try:
        case(rank)
    finally:
        dist.destroy_process_group()


def _spawn_gloo_case(tmp_path, case):
    context = mp.start_processes(
        _gloo_worker,
        args=(str(tmp_path / "store"), case),
        nprocs=WORLD_SIZE,
        join=False,
        start_method="spawn",
    )
    deadline = time.monotonic() + _GLOO_DEADLINE_S
    while not context.join(timeout=1):
        if time.monotonic() > deadline:
            for process in context.processes:
                process.kill()
                process.join()
            pytest.fail(f"a rank of the Gloo world was still running after {_GLOO_DEADLINE_S} s")


def _mpu_groups(rank, ep, pp=1, timeout=None):
    """The dp_cp, EP and expert-DP groups that parallel_state builds with TP = CP = 1.

    Every rank creates every group in the same order and keeps the ones it belongs to.
    """
    dp = dist.get_world_size() // pp
    dense = RankGenerator(tp=1, ep=1, dp=dp, pp=pp, cp=1, order=ORDER)
    expert = RankGenerator(tp=1, ep=ep, dp=dp // ep, pp=pp, cp=1, order=ORDER)
    groups = {}
    for name, ranks_list in (
        ("dp_cp", dense.get_ranks("dp-cp")),
        ("ep", expert.get_ranks("ep")),
        ("expt_dp", expert.get_ranks("dp")),
    ):
        for ranks in ranks_list:
            group = dist.new_group(ranks, timeout=timeout)
            if rank in ranks:
                groups[name] = group
    return groups


def _collection(groups):
    return ProcessGroupCollection(dp_cp=groups["dp_cp"], ep=groups["ep"], expt_dp=groups["expt_dp"])


def _assert_all_reduce(rank, group):
    """A group works only if every rank created every group in the same order."""
    probe = torch.tensor([float(rank)])
    dist.all_reduce(probe, group=group)
    assert probe.item() == sum(_ranks(group))


def _row_groups_case(rank):
    # Ranks 2 and 3 are not members, so their group registries end up smaller than those of
    # ranks 0 and 1: the table groups must line up all the same.
    dist.new_group([0, 1])
    for ep, pp in ((2, 1), (4, 1), (1, 1), (2, 2)):
        groups = _mpu_groups(rank, ep, pp)
        dp_cp = _ranks(groups["dp_cp"])
        position = dp_cp.index(rank)
        for parallel_size in (size for size in (1, 2, 4) if len(dp_cp) % size == 0):
            config = _engram_config(parallel_size)
            pg_collection = _collection(groups)
            initialize_engram_parallel_state(pg_collection, config)
            row_group, replica_group = pg_collection.engram, pg_collection.engram_dp
            # Rows over consecutive blocks of dp_cp, replicated at the same block position.
            start = position - position % parallel_size
            assert _ranks(row_group) == dp_cp[start : start + parallel_size]
            assert _ranks(replica_group) == dp_cp[position % parallel_size :: parallel_size]
            assert pg_collection.parameter_replica_groups == {
                ENGRAM_PARAMETER_DOMAIN: replica_group
            }
            assert resolve_engram_groups(pg_collection, config) == (
                row_group,
                replica_group,
                ENGRAM_PARAMETER_DOMAIN,
            )
            if parallel_size == len(dp_cp):
                # One owner per row: dp_cp itself is the row group.
                assert row_group is groups["dp_cp"]
            if parallel_size == 1:
                # Every rank holds whole tables: dp_cp itself is the replica group, with no
                # second communicator over the same ranks.
                assert replica_group is groups["dp_cp"]
            if parallel_size == ep:
                # The EP layout's groups.
                assert _ranks(row_group) == _ranks(groups["ep"])
                assert _ranks(replica_group) == _ranks(groups["expt_dp"])
            _assert_all_reduce(rank, row_group)
            _assert_all_reduce(rank, replica_group)
            # Initializing again, or another collection over the same dp_cp, creates nothing.
            initialize_engram_parallel_state(pg_collection, config)
            other = _collection(groups)
            initialize_engram_parallel_state(other, config)
            assert (other.engram, other.engram_dp) == (row_group, replica_group)
            assert pg_collection.engram is row_group
            # The groups belong to their parallel size.
            with pytest.raises(ValueError, match="parallel_size"):
                initialize_engram_parallel_state(pg_collection, _engram_config(len(dp_cp) * 2))
        # The EP layout uses the expert groups and creates none.
        pg_collection = _collection(groups)
        initialize_engram_parallel_state(pg_collection, _engram_config())
        assert pg_collection.engram is None and not pg_collection.parameter_replica_groups
        assert resolve_engram_groups(pg_collection, _engram_config()) == (
            groups["ep"],
            groups["expt_dp"],
            "expert",
        )

    groups = _mpu_groups(rank, ep=2)
    # A size that divides no dp_cp group raises on every rank (after the exchange), so all of
    # them reach the barrier, and the next call still lines up.
    for parallel_size in (3, 8):
        with pytest.raises(ValueError, match="divisor"):
            initialize_engram_parallel_state(_collection(groups), _engram_config(parallel_size))
        dist.barrier()
    # Tensor parallelism is rejected before any collective, on every rank alike.
    tp_group = None
    for ranks in ([0, 1], [2, 3]):
        group = dist.new_group(ranks)
        if rank in ranks:
            tp_group = group
    pg_collection = _collection(groups)
    pg_collection.tp = tp_group
    with pytest.raises(ValueError, match="tp has 2 ranks"):
        initialize_engram_parallel_state(pg_collection, _engram_config(WORLD_SIZE))
    pg_collection = _collection(groups)
    initialize_engram_parallel_state(pg_collection, _engram_config(EP_SIZE))
    _assert_all_reduce(rank, pg_collection.engram)


def test_row_groups_split_dp_cp_and_reproduce_the_ep_layout(tmp_path):
    _spawn_gloo_case(tmp_path, _row_groups_case)


def _no_warm_up_case(rank):
    groups = _mpu_groups(rank, ep=EP_SIZE)
    existing = [groups["dp_cp"], groups["ep"], groups["expt_dp"]]
    # parallel_size=4 reuses dp_cp as the row group and creates single-rank replica groups;
    # parallel_size=2 creates both; parallel_size=1 creates single-rank row groups and reuses
    # dp_cp as the replica group.
    for parallel_size, num_created in ((WORLD_SIZE, 1), (EP_SIZE, 2), (1, 1)):
        before = [group._get_sequence_number_for_group() for group in existing]
        pg_collection = _collection(groups)
        initialize_engram_parallel_state(pg_collection, _engram_config(parallel_size))
        created = [
            group
            for group in (pg_collection.engram, pg_collection.engram_dp)
            if all(group is not old for old in existing)
        ]
        assert len(created) == num_created
        assert [group._get_sequence_number_for_group() for group in created] == [0] * len(created)
        assert [group._get_sequence_number_for_group() for group in existing] == before


def test_initialization_runs_no_collective_on_the_table_groups(tmp_path):
    """No warm-up probe: only the default group communicates, so a group's communicator is
    created by its first real collective, and a single-rank group never gets one."""
    _spawn_gloo_case(tmp_path, _no_warm_up_case)


def _created_timeout(group):
    """The timeout ``group`` was created with (a private torch API; Gloo on CPU here)."""
    return group._get_backend(torch.device("cpu")).options._timeout


def _timeout_case(rank):
    # parallel_state creates its groups with --distributed-timeout-minutes, which differs from
    # torch's default for new groups (30 minutes for Gloo, 10 for NCCL).
    configured = timedelta(minutes=37)
    groups = _mpu_groups(rank, ep=EP_SIZE, timeout=configured)
    assert _created_timeout(groups["dp_cp"]) == configured
    pg_collection = _collection(groups)
    initialize_engram_parallel_state(pg_collection, _engram_config(EP_SIZE))
    # parallel_size=2 creates both groups, and they take the timeout of dp_cp.
    assert all(
        group is not groups["dp_cp"] for group in (pg_collection.engram, pg_collection.engram_dp)
    )
    assert _created_timeout(pg_collection.engram) == configured
    assert _created_timeout(pg_collection.engram_dp) == configured
    # An explicit timeout wins; parallel_size=4 creates the single-rank replica groups.
    explicit = timedelta(minutes=41)
    pg_collection = _collection(groups)
    initialize_engram_parallel_state(pg_collection, _engram_config(WORLD_SIZE), timeout=explicit)
    assert pg_collection.engram is groups["dp_cp"]
    assert _created_timeout(pg_collection.engram_dp) == explicit


def test_table_groups_take_the_timeout_of_dp_cp(tmp_path):
    """The table groups follow --distributed-timeout-minutes like the groups they split."""
    _spawn_gloo_case(tmp_path, _timeout_case)


def _reuse_case(rank):
    world = dist.group.WORLD
    # The EP layout creates no group, so it runs no collective at all.
    before = world._get_sequence_number_for_group()
    initialize_engram_parallel_state(_collection(_mpu_groups(rank, ep=EP_SIZE)), _engram_config())
    assert world._get_sequence_number_for_group() == before
    # A repeated sharded setup over the same dp_cp groups reuses its groups, without a gather.
    groups = _mpu_groups(rank, ep=EP_SIZE)
    initialize_engram_parallel_state(_collection(groups), _engram_config(WORLD_SIZE))
    before = world._get_sequence_number_for_group()
    initialize_engram_parallel_state(_collection(groups), _engram_config(WORLD_SIZE))
    assert world._get_sequence_number_for_group() == before


def test_ep_layout_and_repeated_setups_run_no_collective(tmp_path):
    _spawn_gloo_case(tmp_path, _reuse_case)


# ---------------------------------------------------------------------------------------------
# Lookups, gradients and checkpoints (torchrun, exactly four ranks)
# ---------------------------------------------------------------------------------------------

_GROUPS = {}


def _distributed(ep=EP_SIZE):
    """Initialize four ranks (NCCL on GPU, Gloo on CPU) and their dp_cp, EP and expert-DP groups.

    The expert grid is EP2 x expert-DP2 unless ``ep`` gives another EP size.
    """
    if int(os.environ.get("WORLD_SIZE", "1")) != WORLD_SIZE:
        pytest.skip(f"requires torchrun with exactly {WORLD_SIZE} ranks")
    if not dist.is_initialized():
        if torch.cuda.is_available():
            from tests.unit_tests.test_utilities import Utils

            Utils.initialize_distributed()
        else:
            dist.init_process_group("gloo")
    rank = dist.get_rank()
    # Built for every case: the groups of the previous one were destroyed with it (conftest.py).
    _GROUPS.clear()
    _GROUPS.update(_mpu_groups(rank, ep=ep))
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    return rank, device


def _full_tables(device, dtype):
    """Global tables with small dyadic values (exact in bf16)."""
    tables = []
    for table_id, rows in enumerate(TABLE_SIZES):
        row = torch.arange(rows, device=device).unsqueeze(1)
        column = torch.arange(EMBEDDING_DIM, device=device).unsqueeze(0)
        tables.append((((table_id * 7 + row * 3 + column) % 17 - 8) / 8.0).to(dtype))
    return tables


def _build(parallel_size, device, dtype, deterministic=False):
    """The tables of one layout (None is the EP layout), set up as the Engram module does."""
    engram_config = _engram_config(parallel_size)
    pg_collection = _collection(_GROUPS)
    initialize_engram_parallel_state(pg_collection, engram_config)
    row_group, replica_group, domain = resolve_engram_groups(pg_collection, engram_config)
    config = SimpleNamespace(
        use_cpu_initialization=device.type == "cpu",
        params_dtype=dtype,
        perform_initialization=False,
        sequence_parallel=False,
        deterministic_mode=deterministic,
    )
    embedding = EPShardedMultiTableEmbedding(
        config=config,
        table_sizes=TABLE_SIZES,
        embedding_dim=EMBEDDING_DIM,
        init_method=lambda _: None,
        ep_group=row_group,
        tp_group=None,
        expt_dp_group=replica_group,
    )
    if domain == ENGRAM_PARAMETER_DOMAIN:
        tag_engram_tables(embedding)
    with torch.no_grad():
        for table, full in zip(embedding.tables, _full_tables(device, dtype)):
            table.weight.copy_(full[table.row_start : table.row_end])
    return embedding


def _hash_ids(rank, microbatch, empty_microbatch):
    """Per-(rank, microbatch) lookups with heavy repetition and one optional empty microbatch."""
    if empty_microbatch and rank == WORLD_SIZE - 1 and microbatch == 0:
        return torch.zeros((1, 0, len(TABLE_SIZES)), dtype=torch.int64)
    generator = torch.Generator().manual_seed(1000 * rank + microbatch)
    num_tokens = 6 + 3 * rank
    ids = torch.stack(
        [torch.randint(0, rows, (num_tokens,), generator=generator) for rows in TABLE_SIZES], dim=-1
    )
    # Hot rows: every rank of both expert-DP replicas hits row 0 of every table.
    ids[:2] = 0
    return ids.view(1, num_tokens, len(TABLE_SIZES))


def _output_gradients(rank, microbatch, num_tokens, exact):
    generator = torch.Generator().manual_seed(5000 + 1000 * rank + microbatch)
    shape = (1, num_tokens, len(TABLE_SIZES), EMBEDDING_DIM)
    if exact:
        return torch.randint(-3, 4, shape, generator=generator).float()
    return torch.randn(shape, generator=generator)


def _train_microbatches(embedding, rank, device, exact, empty_microbatch):
    """Forward/backward every microbatch, accumulating into fp32 main grads as DDP does."""
    main_grads = [
        torch.zeros(table.weight.shape, dtype=torch.float32, device=device)
        for table in embedding.tables
    ]
    outputs = []
    for microbatch in range(NUM_MICROBATCHES):
        hash_ids = _hash_ids(rank, microbatch, empty_microbatch).to(device)
        output = embedding(hash_ids)
        output_grad = _output_gradients(rank, microbatch, hash_ids.shape[1], exact)
        (output * output_grad.to(device=device, dtype=output.dtype)).sum().backward()
        outputs.append(output.detach())
        for main_grad, table in zip(main_grads, embedding.tables):
            main_grad.add_(table.weight.grad)
            table.weight.grad = None
    return outputs, main_grads


def _finish_grad_sync(embedding, main_grads):
    """DDP's reduction: scale by 1/dp, then sum over the tables' replica group.

    That group is expert-DP in the EP layout and engram_dp in the sharded ones. DDP skips a
    single-rank group, and so does this.
    """
    for table, main_grad in zip(embedding.tables, main_grads):
        main_grad.mul_(1.0 / WORLD_SIZE)
        if get_pg_size(table.expt_dp_group) > 1:
            dist.all_reduce(main_grad, group=table.expt_dp_group)


def _assemble(embedding, shards, device):
    """Full tables from the row shards, taking each row from replica 0 of its shard only."""
    full_tables = []
    for table, shard in zip(embedding.tables, shards):
        full = torch.zeros(
            (table.global_num_embeddings, EMBEDDING_DIM), dtype=torch.float32, device=device
        )
        if get_pg_rank(table.expt_dp_group) == 0:
            full[table.row_start : table.row_end] = shard.float()
        dist.all_reduce(full)
        full_tables.append(full)
    return full_tables


def _reference_gradients(device, exact, empty_microbatch):
    """Full-table gradients of every rank's lookups, scaled by 1/dp, in fp32."""
    gradients = [torch.zeros((rows, EMBEDDING_DIM), device=device) for rows in TABLE_SIZES]
    for rank in range(WORLD_SIZE):
        for microbatch in range(NUM_MICROBATCHES):
            hash_ids = _hash_ids(rank, microbatch, empty_microbatch)
            output_grad = _output_gradients(rank, microbatch, hash_ids.shape[1], exact)
            for table_id, gradient in enumerate(gradients):
                gradient.index_add_(
                    0,
                    hash_ids[..., table_id].reshape(-1).to(device),
                    output_grad[..., table_id, :].reshape(-1, EMBEDDING_DIM).to(device),
                )
    return [gradient / WORLD_SIZE for gradient in gradients]


def _rows_shared_by_replicas(empty_microbatch):
    """Per table, the rows both expert-DP replicas look up within one microbatch."""
    shared = [set() for _ in TABLE_SIZES]
    for microbatch in range(NUM_MICROBATCHES):
        for table_id in range(len(TABLE_SIZES)):
            touched = []
            for edp in range(EXPERT_DP_SIZE):
                rows = set()
                for ep in range(EP_SIZE):
                    hash_ids = _hash_ids(ep + EP_SIZE * edp, microbatch, empty_microbatch)
                    rows.update(hash_ids[..., table_id].reshape(-1).tolist())
                touched.append(rows)
            shared[table_id] |= touched[0] & touched[1]
    return shared


def _assert_equal(actual, expected, what):
    """torch.equal with a per-rank diagnostic printed before raising (ranks may abort later)."""
    if torch.equal(actual, expected):
        return
    diff = (actual.float() - expected.float()).abs()
    print(
        f"[rank {dist.get_rank()}] {what}: dtype {actual.dtype}/{expected.dtype} "
        f"shape {tuple(actual.shape)}/{tuple(expected.shape)} "
        f"max|diff| {diff.max().item() if diff.numel() else 0.0}",
        flush=True,
    )
    raise AssertionError(what)


@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("parallel_size", SHARDED_SIZES)
@pytest.mark.parametrize("exact", [True, False], ids=["exact", "random"])
@pytest.mark.parametrize("empty_microbatch", [False, True], ids=["full", "empty-microbatch"])
@pytest.mark.parametrize("deterministic", [False, True], ids=["default", "deterministic"])
def test_lookup_and_gradients_match_ep_layout(
    parallel_size, exact, empty_microbatch, deterministic
):
    rank, device = _distributed()
    dtype = torch.bfloat16
    layouts = {
        layout: _build(layout, device, dtype, deterministic) for layout in (None, parallel_size)
    }
    sharded = layouts[parallel_size]
    if parallel_size == WORLD_SIZE:
        assert _ranks(sharded.ep_group) == list(range(WORLD_SIZE))
    else:
        assert _ranks(sharded.ep_group) == _ranks(_GROUPS["ep"])
    assert get_pg_size(sharded.tables[0].expt_dp_group) == WORLD_SIZE // parallel_size
    for table in sharded.tables:
        assert get_parameter_parallel_domain(table.weight) == ENGRAM_PARAMETER_DOMAIN
    for table in layouts[None].tables:
        assert get_parameter_parallel_domain(table.weight) == "expert"
        assert not hasattr(table.weight, "parallel_domain")

    results = {}
    for layout, embedding in layouts.items():
        outputs, main_grads = _train_microbatches(embedding, rank, device, exact, empty_microbatch)
        _finish_grad_sync(embedding, main_grads)
        results[layout] = (outputs, _assemble(embedding, main_grads, device))

    full_tables = _full_tables(device, dtype)
    for microbatch, (output_ep, output_sharded) in enumerate(
        zip(results[None][0], results[parallel_size][0])
    ):
        hash_ids = _hash_ids(rank, microbatch, empty_microbatch).to(device)
        expected = torch.stack(
            [full[hash_ids[..., t]] for t, full in enumerate(full_tables)], dim=-2
        )
        _assert_equal(output_ep, expected, f"EP layout output, microbatch {microbatch}")
        _assert_equal(output_sharded, expected, f"sharded output, microbatch {microbatch}")

    reference = _reference_gradients(device, exact, empty_microbatch)
    shared_rows = _rows_shared_by_replicas(empty_microbatch)
    for table_id, (grad_ep, grad_sharded) in enumerate(
        zip(results[None][1], results[parallel_size][1])
    ):
        if exact:
            _assert_equal(grad_ep, reference[table_id], f"EP layout gradient, table {table_id}")
            _assert_equal(grad_sharded, reference[table_id], f"sharded gradient, table {table_id}")
            continue
        if parallel_size == EP_SIZE:
            # The EP layout's groups: the same reductions in the same order.
            _assert_equal(grad_sharded, grad_ep, f"gradient, table {table_id}")
            continue
        exclusive = torch.ones(TABLE_SIZES[table_id], dtype=torch.bool, device=device)
        exclusive[list(shared_rows[table_id])] = False
        _assert_equal(
            grad_sharded[exclusive], grad_ep[exclusive], f"exclusive rows, table {table_id}"
        )
        # Shared rows differ by bf16 rounding of the per-replica partial sums.
        torch.testing.assert_close(grad_sharded, reference[table_id], rtol=2**-6, atol=2**-6)
        torch.testing.assert_close(grad_ep, reference[table_id], rtol=2**-6, atol=2**-6)


@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("ep", [1, WORLD_SIZE], ids=["ep1", "ep4"])
@pytest.mark.parametrize("exact", [True, False], ids=["exact", "random"])
def test_replicated_shards_match_ep_layouts_of_other_ep_sizes(ep, exact):
    """``parallel_size=2`` on an expert grid whose EP size is not 2, against its EP layout.

    The tables keep two row shards with two replicas each, over groups of their own: the expert
    grid has no group of two ranks. Its EP layout holds four replicas of the whole tables and
    looks rows up locally (EP1), or gives every row one owner (EP4).
    """
    rank, device = _distributed(ep)
    dtype = torch.bfloat16
    parallel_size = 2
    layouts = {layout: _build(layout, device, dtype) for layout in (None, parallel_size)}
    sharded = layouts[parallel_size]
    start = rank - rank % parallel_size
    assert _ranks(sharded.ep_group) == list(range(start, start + parallel_size))
    assert _ranks(sharded.tables[0].expt_dp_group) == list(
        range(rank % parallel_size, WORLD_SIZE, parallel_size)
    )
    assert get_pg_size(layouts[None].ep_group) == ep
    assert get_pg_size(layouts[None].tables[0].expt_dp_group) == WORLD_SIZE // ep

    results = {}
    for layout, embedding in layouts.items():
        outputs, main_grads = _train_microbatches(embedding, rank, device, exact, False)
        _finish_grad_sync(embedding, main_grads)
        results[layout] = (outputs, _assemble(embedding, main_grads, device))

    full_tables = _full_tables(device, dtype)
    reference = _reference_gradients(device, exact, False)
    for layout, (outputs, gradients) in results.items():
        name = "EP layout" if layout is None else "sharded"
        for microbatch, output in enumerate(outputs):
            hash_ids = _hash_ids(rank, microbatch, False).to(device)
            expected = torch.stack(
                [full[hash_ids[..., t]] for t, full in enumerate(full_tables)], dim=-2
            )
            _assert_equal(output, expected, f"{name} output, microbatch {microbatch}")
        for table_id, gradient in enumerate(gradients):
            if exact:
                # Integer gradients: every partial sum is exact, however a layout groups them.
                _assert_equal(gradient, reference[table_id], f"{name} gradient, table {table_id}")
                continue
            # The output gradients are cast to bf16, and each layout rounds other partial sums of
            # them to bf16 before they are added in fp32: the sharded one per row group, EP1
            # per rank, EP4 over all ranks. The layouts therefore match the fp32 reference, and
            # each other on the rows that several ranks touch (row 0 of every table at least),
            # only to that rounding.
            torch.testing.assert_close(gradient, reference[table_id], rtol=2**-6, atol=2**-6)


@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("parallel_size", (None,) + SHARDED_SIZES)
def test_sharded_state_dict_covers_every_row_once(parallel_size):
    _, device = _distributed()
    embedding = _build(parallel_size, device, torch.float32)
    state = embedding.sharded_state_dict()
    entries = [
        (key, tensor.global_offset[0], tensor.local_shape[0], tensor.replica_id)
        for key, tensor in state.items()
    ]
    gathered = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, entries)
    for table_id, rows in enumerate(TABLE_SIZES):
        key = f"tables.{table_id}.weight"
        intervals = sorted(
            (offset, offset + count)
            for rank_entries in gathered
            for entry_key, offset, count, replica_id in rank_entries
            if entry_key == key and all(index == 0 for index in replica_id)
        )
        assert intervals[0][0] == 0 and intervals[-1][1] == rows
        assert all(end == start for (_, end), (start, _) in zip(intervals, intervals[1:]))
        replicas = [
            entry[3] for rank_entries in gathered for entry in rank_entries if entry[0] == key
        ]
        if parallel_size == WORLD_SIZE:
            # One owner per row: four disjoint main shards, no replicas.
            assert len(intervals) == WORLD_SIZE and all(r == (0, 0, 0) for r in replicas)
        else:
            # Two row shards, each held by two replicas.
            assert len(intervals) == EP_SIZE
            assert sorted(replica_id[2] for replica_id in replicas) == [0, 0, 1, 1]


def _shared_checkpoint_dir():
    path = [tempfile.mkdtemp(prefix="engram-table-sharding-") if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(path, src=0)
    return path[0]


@pytest.mark.launch_on_gb200
@pytest.mark.parametrize(
    "source, target", [(None, WORLD_SIZE), (WORLD_SIZE, None), (EP_SIZE, WORLD_SIZE)]
)
def test_checkpoint_loads_across_layouts(source, target):
    from megatron.core.dist_checkpointing import load, save

    rank, device = _distributed()
    if device.type != "cuda":
        pytest.skip("dist_checkpointing save needs CUDA")
    saved = _build(source, device, torch.float32)
    loaded = _build(target, device, torch.float32)
    with torch.no_grad():
        for table in loaded.tables:
            table.weight.zero_()

    checkpoint_dir = _shared_checkpoint_dir()
    save(saved.sharded_state_dict(), checkpoint_dir)
    dist.barrier()
    loaded.load_state_dict(load(loaded.sharded_state_dict(), checkpoint_dir))

    for table, full in zip(loaded.tables, _full_tables(device, torch.float32)):
        assert torch.equal(table.weight, full[table.row_start : table.row_end])
    dist.barrier()
    if rank == 0:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)


@pytest.mark.launch_on_gb200
def test_table_groups_survive_uneven_group_membership():
    """Group creation must not depend on how many groups each rank already belongs to.

    PyTorch names members-only groups after the size of the rank's group registry; after a
    group with uneven membership, the ranks disagree on that name and a members-only rendezvous
    deadlocks. Every rank must therefore create every table group.

    parallel_size=2 creates new multi-rank row and replica groups, and both must work.
    (parallel_size=4 would reuse dp_cp, created before the uneven group, as the row group and
    add only single-rank replica groups, which never communicate.)
    """
    _, device = _distributed()
    # Ranks 2 and 3 are not members, so their group registries end up smaller.
    dist.new_group([0, 1])
    clear_engram_parallel_state()
    pg_collection = _collection(_GROUPS)
    initialize_engram_parallel_state(pg_collection, _engram_config(EP_SIZE))
    assert all(
        group is not _GROUPS["dp_cp"] for group in (pg_collection.engram, pg_collection.engram_dp)
    )
    probe = torch.ones(1, device=device)
    dist.all_reduce(probe, group=pg_collection.engram)
    assert probe.item() == EP_SIZE
    dist.all_reduce(probe, group=pg_collection.engram_dp)
    assert probe.item() == WORLD_SIZE
