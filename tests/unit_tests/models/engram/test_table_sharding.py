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

Every case looks rows up with NCCL all-to-alls (``communication_backend='nccl'``) unless it
names the symmetric backend. The symmetric parametrization of the lookup case needs GPUs and
the transport's libraries (CuTe DSL >= 4.8.1, nccl4py >= 0.6.0 with a matching NCCL runtime and
device bitcode, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True) and skips without them. It
sums the gradients of duplicate rows on the requester before the owner adds them, so with
random gradients it matches the other layouts to bf16 rounding.
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

from megatron.core.models.engram import distributed_embedding, symmetric_lookup, table_sharding
from megatron.core.models.engram.config import EngramConfig
from megatron.core.models.engram.distributed_embedding import EPShardedMultiTableEmbedding
from megatron.core.models.engram.symmetric_lookup import (
    initialize_engram_transports,
    select_engram_lookup,
)
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

from ._reference_transport import ReferenceEngramTransport
from ._test_utils import (
    closing_symmetric_transports,
    make_module_config,
    make_pg_collection,
    symmetric_transport_or_skip,
)

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


def _engram_config(parallel_size=None, communication_backend="nccl", **overrides):
    """A small configuration; the qwen variant needs no tokenizer artifact.

    The NCCL lookup unless a case names another backend: the symmetric one, the sharded
    layout's default, needs GPUs and the transport's libraries.
    """
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
        communication_backend=communication_backend,
        **overrides,
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


def _validate_cli(monkeypatch, flags, engram=True, backend="nccl"):
    """Parse and validate the arguments of a 4-rank run.

    With Engram, the NCCL lookup unless ``backend`` names another one or is None (the
    layout's default).
    """
    from megatron.training.arguments import parse_args, validate_args

    monkeypatch.setenv("WORLD_SIZE", str(WORLD_SIZE))
    engram_flags = _ENGRAM_FLAGS if engram else ""
    if engram and backend is not None:
        engram_flags += f" --engram-communication-backend {backend}"
    argv = f"{_TRAINING_ARGV} {engram_flags} {flags}"
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


def test_cli_parallel_size_requires_engram(monkeypatch):
    with pytest.raises(AssertionError, match="requires Engram"):
        _validate_cli(monkeypatch, "--engram-parallel-size 4 --use-distributed-optimizer", False)


def test_communication_backend_follows_the_layout(monkeypatch):
    """None looks the EP layout up with NCCL and the sharded one with the symmetric transport,
    except one rank per row group, whose rows either backend looks up locally."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _engram_config(communication_backend=None).communication_backend == "nccl"
    assert _engram_config(WORLD_SIZE).communication_backend == "nccl"
    # Without CUDA too: a single-rank row group asks nothing of the symmetric toolchain.
    assert _engram_config(1, communication_backend=None).communication_backend == "nccl"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    sharded = _engram_config(WORLD_SIZE, communication_backend=None)
    assert sharded.communication_backend == "symmetric"
    assert sharded.symmetric_workspace_mb == 64
    assert _engram_config(EP_SIZE, communication_backend=None).communication_backend == (
        "symmetric"
    )
    # Named explicitly, symmetric stays symmetric (and is checked) with one rank per row group.
    assert _engram_config(1, "symmetric").communication_backend == "symmetric"


@pytest.mark.parametrize(
    ("parallel_size", "overrides", "cuda", "error"),
    [
        (None, dict(communication_backend="symmetric"), True, "requires engram_parallel_size"),
        (WORLD_SIZE, dict(communication_backend="all_to_all"), True, "must be one of"),
        (WORLD_SIZE, dict(communication_backend="symmetric"), False, "needs CUDA"),
        # Without CUDA the sharded layout's default fails too (above one rank per row group); it
        # never turns into NCCL silently.
        (WORLD_SIZE, dict(communication_backend=None), False, "needs CUDA"),
        (WORLD_SIZE, dict(symmetric_workspace_mb=0), True, "positive integer"),
        (WORLD_SIZE, dict(symmetric_workspace_mb=64.0), True, "positive integer"),
    ],
    ids=[
        "ep_layout",
        "unknown_backend",
        "no_cuda",
        "default_without_cuda",
        "empty_workspace",
        "float_workspace",
    ],
)
def test_communication_backend_rejections(monkeypatch, parallel_size, overrides, cuda, error):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    with pytest.raises(ValueError, match=error):
        _engram_config(parallel_size, **overrides)


# A sharded run that validate_args accepts with the symmetric backend's setup.
_SHARDED_FLAGS = "--engram-parallel-size 4 --use-distributed-optimizer"


def _symmetric_setup(
    monkeypatch, cuda=True, alloc_conf="expandable_segments:True", generic_alloc_conf=None
):
    """Pretend CUDA (validate_args queries no GPU for these arguments) and set the allocator:
    ``alloc_conf`` in PYTORCH_CUDA_ALLOC_CONF, ``generic_alloc_conf`` in PYTORCH_ALLOC_CONF."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.delenv("PYTORCH_HIP_ALLOC_CONF", raising=False)
    for name, value in (
        ("PYTORCH_CUDA_ALLOC_CONF", alloc_conf),
        ("PYTORCH_ALLOC_CONF", generic_alloc_conf),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)


def test_cli_resolves_the_communication_backend(monkeypatch):
    """Each layout's default backend reaches EngramConfig, with the workspace size."""
    args = _validate_cli(monkeypatch, "", backend=None)
    assert args.engram_communication_backend == "nccl"
    # One rank per row group defaults to NCCL, without CUDA or the allocator setup.
    _symmetric_setup(monkeypatch, cuda=False, alloc_conf=None)
    args = _validate_cli(
        monkeypatch, "--engram-parallel-size 1 --use-distributed-optimizer", backend=None
    )
    assert args.engram_communication_backend == "nccl"
    assert EngramConfig.from_args(args, _transformer_config()).communication_backend == "nccl"
    # The device-generic variable counts when the CUDA one is unset.
    _symmetric_setup(monkeypatch, alloc_conf=None, generic_alloc_conf="expandable_segments:True")
    args = _validate_cli(monkeypatch, _SHARDED_FLAGS, backend=None)
    assert args.engram_communication_backend == "symmetric"
    _symmetric_setup(monkeypatch, alloc_conf="max_split_size_mb:512,expandable_segments:True")
    args = _validate_cli(
        monkeypatch, f"{_SHARDED_FLAGS} --engram-symmetric-workspace-mb 16", backend=None
    )
    assert args.engram_communication_backend == "symmetric"
    config = EngramConfig.from_args(args, _transformer_config())
    assert (config.parallel_size, config.communication_backend, config.symmetric_workspace_mb) == (
        WORLD_SIZE,
        "symmetric",
        16,
    )
    # The NCCL reference stays available to the sharded layout, without the symmetric setup.
    _symmetric_setup(monkeypatch, cuda=False, alloc_conf=None)
    args = _validate_cli(monkeypatch, _SHARDED_FLAGS)
    assert EngramConfig.from_args(args, _transformer_config()).communication_backend == "nccl"


# Neither CUDA nor expandable segments: the setups the transport cannot serve at all are named
# before its CUDA and allocator requirements.
_BARE = dict(cuda=False, alloc_conf=None)


@pytest.mark.parametrize(
    ("flags", "setup", "error"),
    [
        ("--engram-communication-backend symmetric", {}, "requires --engram-parallel-size"),
        (_SHARDED_FLAGS, dict(cuda=False), "needs CUDA"),
        (_SHARDED_FLAGS, dict(alloc_conf=None), "expandable_segments:True"),
        (_SHARDED_FLAGS, dict(alloc_conf="expandable_segments:False"), "expandable_segments:True"),
        # PyTorch reads the CUDA variable when it is set and ignores the device-generic one.
        (
            _SHARDED_FLAGS,
            dict(alloc_conf="max_split_size_mb:512", generic_alloc_conf="expandable_segments:True"),
            "expandable_segments:True",
        ),
        (f"{_SHARDED_FLAGS} --fake-process-group", _BARE, "fake process group"),
        (
            f"{_SHARDED_FLAGS} --moe-use-upcycling --save /nonexistent --save-interval 1",
            _BARE,
            "--moe-use-upcycling",
        ),
        (f"{_SHARDED_FLAGS} --inprocess-restart", _BARE, "--inprocess-restart"),
        (f"{_SHARDED_FLAGS} --engram-symmetric-workspace-mb 0", {}, "must be positive"),
    ],
    ids=[
        "ep_layout",
        "no_cuda",
        "no_expandable_segments",
        "expandable_segments_off",
        "generic_alloc_conf_hidden",
        "fake_process_group",
        "upcycling",
        "inprocess_restart",
        "empty_workspace",
    ],
)
def test_cli_rejects_unsupported_symmetric_setups(monkeypatch, flags, setup, error):
    """The symmetric transport's requirements are errors at startup, not warnings."""
    _symmetric_setup(monkeypatch, **setup)
    with pytest.raises(AssertionError, match=error):
        _validate_cli(monkeypatch, flags, backend=None)


def test_cli_backend_requires_engram(monkeypatch):
    with pytest.raises(AssertionError, match="requires Engram"):
        _validate_cli(monkeypatch, "--engram-communication-backend nccl", engram=False)


def test_engram_lookup_adds_no_state_dict_keys(monkeypatch):
    """The Engram layer calls its lookup through a bound method or a plain object, which adds
    no submodule and no state-dict key, with either backend."""
    from megatron.core.models.engram import engram as engram_module

    def build(engram_config):
        torch.manual_seed(0)
        return engram_module.Engram(
            config=make_module_config(),
            engram_config=engram_config,
            layer_number=1,
            pg_collection=make_pg_collection(),
        )

    nccl = build(_engram_config())
    # NCCL lookups call the embedding module itself, hooks included, as before.
    assert nccl._lookup == nccl.embedding.__call__
    # The symmetric lookup of a two-rank row group, without its libraries or a process group:
    # the embedding then holds whole tables, which changes none of its keys.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(symmetric_lookup, "require_symmetric_transport", lambda: None)
    monkeypatch.setattr(symmetric_lookup, "select_row_accumulation", lambda: None)
    monkeypatch.setattr(symmetric_lookup, "get_pg_size", lambda group: 2)
    monkeypatch.setattr(
        engram_module,
        "resolve_engram_groups",
        lambda pg_collection, engram_config: (None, None, ENGRAM_PARAMETER_DOMAIN),
    )
    symmetric = build(_engram_config(2, "symmetric"))
    assert isinstance(symmetric._lookup, symmetric_lookup.SymmetricEngramLookup)
    assert symmetric._lookup.embedding is symmetric.embedding
    for module in (nccl, symmetric):
        assert all(name != "_lookup" for name, _ in module.named_modules())
        assert all(not name.startswith("_lookup") for name in module.state_dict())
    assert list(symmetric.state_dict()) == list(nccl.state_dict())
    assert [name for name, _ in symmetric.named_parameters()] == [
        name for name, _ in nccl.named_parameters()
    ]
    # Distributed checkpoints see the same sharded keys and shard geometry too. A stand-in
    # one-rank dp_cp group keeps the default-group lookup out of this single-process test.
    one_rank = SimpleNamespace(rank=lambda: 0, size=lambda: 1)

    def sharded_geometry(module):
        state = module.sharded_state_dict(metadata={"dp_cp_group": one_rank})
        return {key: (tensor.global_shape, tensor.global_offset) for key, tensor in state.items()}

    assert sharded_geometry(symmetric) == sharded_geometry(nccl)
    assert "embedding.tables.0.weight" in sharded_geometry(symmetric)
    # forward looks the rows up through the selected lookup.
    calls = []
    lookup = nccl._lookup
    nccl._lookup = lambda hash_ids: calls.append(tuple(hash_ids.shape)) or lookup(hash_ids)
    input_ids = torch.tensor([[1, 4, 2, 7], [3, 5, 9, 2]])
    nccl(torch.randn(4, 2, 8, dtype=torch.float64), input_ids)
    assert calls == [(2, 4, 2)]


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
    """Every rank sets the groups up on the collection that the stages and DDP get, first, and
    starts the lookup transports on the wrapped stages, before any post-wrap hook."""
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.training.models import hybrid

    events = []
    monkeypatch.setattr(
        hybrid,
        "initialize_engram_parallel_state",
        lambda pg_collection, config: events.append(("initialize", pg_collection, config)),
    )
    stages = [object()]

    def build_stages(build_model, transformer_config, pg_collection, *args, **kwargs):
        events.append(("build", pg_collection))
        return stages

    monkeypatch.setattr(hybrid, "unimodal_build_distributed_models", build_stages)
    monkeypatch.setattr(
        hybrid,
        "initialize_engram_transports",
        lambda model_list, pg_collection, config: events.append(
            ("transports", model_list, pg_collection, config)
        ),
    )
    engram_config = _engram_config(WORLD_SIZE)
    model_config = hybrid.HybridModelConfig(
        transformer=TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=1),
        vocab_size=1024,
        engram_config=engram_config,
        post_wrap_hooks=[lambda model_list: events.append(("post_wrap", model_list)) or model_list],
    )
    pg_collection = object()
    hybrid.HybridModelBuilder(model_config).build_distributed_models(pg_collection)
    assert events == [
        ("initialize", pg_collection, engram_config),
        ("build", pg_collection),
        ("transports", stages, pg_collection, engram_config),
        ("post_wrap", stages),
    ]


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


def test_destroy_model_parallel_aborts_the_engram_transports(monkeypatch):
    """Teardown aborts open symmetric transports without a device barrier (a failed peer would
    never join one), found by module name like the table state; once closed, it is a no-op."""
    from megatron.core import parallel_state

    closed = []
    monkeypatch.setitem(
        symmetric_lookup._TRANSPORTS,
        object(),
        SimpleNamespace(close=lambda clean=False: closed.append(clean)),
    )
    monkeypatch.setattr(symmetric_lookup, "_INITIALIZED", True)
    parallel_state.destroy_model_parallel()
    assert closed == [False]
    assert symmetric_lookup._TRANSPORTS == {} and not symmetric_lookup._INITIALIZED
    parallel_state.destroy_model_parallel()
    assert closed == [False]


def test_symmetric_backend_rejects_a_fake_process_group(monkeypatch):
    """A configuration built in code reaches the core check, which the CLI's does not cover: on
    a fake process group, the transport's first NCCL communicator would wait forever for row
    ranks that do not exist. Every rank raises before the library check and any collective."""
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_backend", lambda group=None: "fake")
    checked = []
    monkeypatch.setattr(
        symmetric_lookup, "require_symmetric_transport", lambda: checked.append("require")
    )
    engram_config = _engram_config(WORLD_SIZE)
    # Built for NCCL, which needs no CUDA; the backend only steers the initialization here.
    engram_config.communication_backend = "symmetric"
    with pytest.raises(ValueError, match="fake process group"):
        initialize_engram_parallel_state(ProcessGroupCollection(), engram_config)
    assert checked == []


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


def _symmetric_initialization_case(rank):
    groups = _mpu_groups(rank, ep=EP_SIZE)

    def config(parallel_size):
        # Built for NCCL, which needs no CUDA; the backend only steers the initialization here.
        engram_config = _engram_config(parallel_size)
        engram_config.communication_backend = "symmetric"
        return engram_config

    # A library missing on rank 1 fails on every rank, before any group is created.
    def require():
        if rank == 1:
            raise ImportError("simulated missing NCCL device bitcode")

    symmetric_lookup.require_symmetric_transport = require
    with pytest.raises(ImportError, match=r"unavailable on ranks \[1\]") as error:
        initialize_engram_parallel_state(_collection(groups), config(WORLD_SIZE))
    assert ("simulated missing NCCL device bitcode" in str(error.value)) == (rank == 1)
    assert table_sharding._ROW_GROUPS == {}

    # With the libraries on every rank: the row and replica groups are created.
    checked = []
    symmetric_lookup.require_symmetric_transport = lambda: checked.append(rank)
    pg_collection = _collection(groups)
    initialize_engram_parallel_state(pg_collection, config(WORLD_SIZE))
    assert checked == [rank]
    assert _ranks(pg_collection.engram) == _ranks(groups["dp_cp"])
    assert pg_collection.parameter_replica_groups == {
        ENGRAM_PARAMETER_DOMAIN: pg_collection.engram_dp
    }
    # The NCCL reference over the same row groups needs no library check.
    initialize_engram_parallel_state(_collection(groups), _engram_config(WORLD_SIZE))
    assert checked == [rank]

    # The EP layout rejects the symmetric backend on every rank alike, before any collective.
    engram_config = _engram_config()
    engram_config.communication_backend = "symmetric"
    with pytest.raises(ValueError, match="requires parallel_size"):
        initialize_engram_parallel_state(_collection(groups), engram_config)
    engram_config.communication_backend = None
    with pytest.raises(ValueError, match="must be resolved"):
        initialize_engram_parallel_state(_collection(groups), engram_config)
    dist.barrier()


def test_symmetric_backend_checks_every_rank(tmp_path):
    _spawn_gloo_case(tmp_path, _symmetric_initialization_case)


def _engram_layer_case(rank):
    """Engram layers of the sharded layout with either backend, wired as the model builder does.

    The symmetric backend runs on the all-to-all reference transport here, so its selection,
    transport start-up and lookup path run on CPU; outputs and dense gradients must equal the
    NCCL lookup's, the table gradients to FP64 round-off (the sums run in another order).
    """
    from megatron.core.models.engram.engram import Engram

    symmetric_lookup.require_symmetric_transport = lambda: SimpleNamespace(
        EngramDeviceTransport=ReferenceEngramTransport,
        create_unique_id=lambda: f"uid-{rank}".encode(),
    )
    groups = _mpu_groups(rank, ep=EP_SIZE)
    generator = torch.Generator().manual_seed(rank)
    input_ids = torch.randint(0, 32, (2, 6), generator=generator)
    input_ids[:, 0] = 1  # repeated n-grams across ranks
    hidden = torch.randn(6, 2, 8, dtype=torch.float64, generator=generator)
    projection = torch.randn(6, 2, 8, dtype=torch.float64, generator=generator)
    results = {}
    for parallel_size, backend in (
        (WORLD_SIZE, "nccl"),
        (WORLD_SIZE, "symmetric"),
        (2, "nccl"),
        (2, "symmetric"),
    ):
        engram_config = _engram_config(parallel_size)
        # Built for NCCL, which needs no CUDA.
        engram_config.communication_backend = backend
        pg_collection = _collection(groups)
        initialize_engram_parallel_state(pg_collection, engram_config)
        torch.manual_seed(0)
        layer = Engram(
            config=make_module_config(),
            engram_config=engram_config,
            layer_number=1,
            pg_collection=pg_collection,
        )
        with torch.no_grad():
            layer.short_conv.weight.normal_(mean=0.0, std=0.03, generator=generator.manual_seed(7))
        initialize_engram_transports([layer], pg_collection, engram_config)
        if backend == "symmetric":
            assert isinstance(layer._lookup, symmetric_lookup.SymmetricEngramLookup)
            transport = symmetric_lookup._TRANSPORTS[layer.embedding.ep_group]
            assert transport.initialized and transport.describes == 1
        else:
            assert layer._lookup == layer.embedding.__call__
            assert symmetric_lookup._TRANSPORTS == {}
        output = layer(hidden, input_ids)
        (output * projection).sum().backward()
        results[parallel_size, backend] = (
            output.detach(),
            [table.weight.grad for table in layer.embedding.tables],
            [p.grad for name, p in layer.named_parameters() if not name.startswith("embedding.")],
        )
        if backend == "symmetric":
            # Every lookup ran one count and two row exchanges, every backward one more.
            assert transport.exchanges == 1 + 4
            symmetric_lookup.close_engram_transports(clean=True)
            assert transport.closed is True and symmetric_lookup._TRANSPORTS == {}
    for parallel_size in (WORLD_SIZE, 2):
        nccl, symmetric = results[parallel_size, "nccl"], results[parallel_size, "symmetric"]
        assert torch.equal(symmetric[0], nccl[0])
        for actual, expected in zip(symmetric[1], nccl[1]):
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        for actual, expected in zip(symmetric[2], nccl[2]):
            assert torch.equal(actual, expected)


def test_engram_layers_look_rows_up_through_the_selected_backend(tmp_path):
    _spawn_gloo_case(tmp_path, _engram_layer_case)


# ---------------------------------------------------------------------------------------------
# Lookups, gradients and checkpoints (torchrun, exactly four ranks)
# ---------------------------------------------------------------------------------------------

_GROUPS = {}


def _distributed():
    """Initialize four ranks (NCCL on GPU, Gloo on CPU) and their dp_cp, EP and expert-DP groups."""
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
    _GROUPS.update(_mpu_groups(rank, ep=EP_SIZE))
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


def _build(parallel_size, device, dtype, deterministic=False, backend="nccl"):
    """The tables of one layout (None is the EP layout) and the lookup the Engram module calls
    on them, set up as the Engram module and the model builder do: the embedding itself for the
    NCCL backend, or the symmetric lookup bound to it once its transport has started."""
    engram_config = _engram_config(parallel_size, backend)
    pg_collection = _collection(_GROUPS)
    initialize_engram_parallel_state(pg_collection, engram_config)
    row_group, replica_group, domain = resolve_engram_groups(pg_collection, engram_config)
    make_lookup = select_engram_lookup(engram_config, row_group)
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
    lookup = make_lookup(embedding)
    # A no-op for the NCCL backend.
    initialize_engram_transports([embedding], pg_collection, engram_config)
    return embedding, lookup


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


def _train_microbatches(embedding, lookup, rank, device, exact, empty_microbatch):
    """Forward/backward every microbatch, accumulating into fp32 main grads as DDP does.

    The tables have no ``main_grad``, so the symmetric lookup returns dense table gradients too.
    """
    main_grads = [
        torch.zeros(table.weight.shape, dtype=torch.float32, device=device)
        for table in embedding.tables
    ]
    outputs = []
    for microbatch in range(NUM_MICROBATCHES):
        hash_ids = _hash_ids(rank, microbatch, empty_microbatch).to(device)
        output = lookup(hash_ids)
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
@pytest.mark.parametrize("backend", ["nccl", pytest.param("symmetric", marks=pytest.mark.internal)])
@pytest.mark.parametrize("parallel_size", SHARDED_SIZES)
@pytest.mark.parametrize("exact", [True, False], ids=["exact", "random"])
@pytest.mark.parametrize("empty_microbatch", [False, True], ids=["full", "empty-microbatch"])
@pytest.mark.parametrize("deterministic", [False, True], ids=["default", "deterministic"])
def test_lookup_and_gradients_match_ep_layout(
    backend, parallel_size, exact, empty_microbatch, deterministic
):
    """The sharded layout, with either backend, against the EP layout's NCCL lookup."""
    rank, device = _distributed()
    if backend == "symmetric":
        if deterministic:
            pytest.skip("deterministic_mode changes only the NCCL lookup's table gather")
        symmetric_transport_or_skip()
    with closing_symmetric_transports():
        _check_lookup_and_gradients(
            backend, parallel_size, exact, empty_microbatch, deterministic, rank, device
        )


def _check_lookup_and_gradients(
    backend, parallel_size, exact, empty_microbatch, deterministic, rank, device
):
    dtype = torch.bfloat16
    layouts = {
        layout: _build(layout, device, dtype, deterministic, layout_backend)
        for layout, layout_backend in ((None, "nccl"), (parallel_size, backend))
    }
    sharded, sharded_lookup = layouts[parallel_size]
    if parallel_size == WORLD_SIZE:
        assert _ranks(sharded.ep_group) == list(range(WORLD_SIZE))
    else:
        assert _ranks(sharded.ep_group) == _ranks(_GROUPS["ep"])
    assert get_pg_size(sharded.tables[0].expt_dp_group) == WORLD_SIZE // parallel_size
    for table in sharded.tables:
        assert get_parameter_parallel_domain(table.weight) == ENGRAM_PARAMETER_DOMAIN
    for table in layouts[None][0].tables:
        assert get_parameter_parallel_domain(table.weight) == "expert"
        assert not hasattr(table.weight, "parallel_domain")
    if backend == "symmetric":
        assert isinstance(sharded_lookup, symmetric_lookup.SymmetricEngramLookup)
        assert sharded.ep_group in symmetric_lookup._TRANSPORTS

    results = {}
    for layout, (embedding, lookup) in layouts.items():
        outputs, main_grads = _train_microbatches(
            embedding, lookup, rank, device, exact, empty_microbatch
        )
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
        if backend == "symmetric":
            # Requesters sum the gradients of their duplicate rows in fp32 before the owner adds
            # them, so the fp32 sums run in another order than the all-to-all's and may round
            # differently before the bf16 cast.
            torch.testing.assert_close(grad_sharded, grad_ep, rtol=2**-6, atol=2**-6)
        elif parallel_size == EP_SIZE:
            # The EP layout's groups: the same reductions in the same order.
            _assert_equal(grad_sharded, grad_ep, f"gradient, table {table_id}")
            continue
        else:
            exclusive = torch.ones(TABLE_SIZES[table_id], dtype=torch.bool, device=device)
            exclusive[list(shared_rows[table_id])] = False
            _assert_equal(
                grad_sharded[exclusive], grad_ep[exclusive], f"exclusive rows, table {table_id}"
            )
        # Shared rows differ by bf16 rounding of the per-replica partial sums.
        torch.testing.assert_close(grad_sharded, reference[table_id], rtol=2**-6, atol=2**-6)
        torch.testing.assert_close(grad_ep, reference[table_id], rtol=2**-6, atol=2**-6)


@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("parallel_size", (None,) + SHARDED_SIZES)
def test_sharded_state_dict_covers_every_row_once(parallel_size):
    _, device = _distributed()
    embedding, _ = _build(parallel_size, device, torch.float32)
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
    saved, _ = _build(source, device, torch.float32)
    loaded, _ = _build(target, device, torch.float32)
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
