# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Checkpoint save and load run on the model's process groups, not on the global parallel grid.

The model-grid tests build a TP2 x PP2 collection on a TP1 x PP1 global grid and replace the
global grid accessors that checkpointing uses with a stand-in that fails on any read. A read of
the global grid therefore fails the test, and a shard named by the global ranks would land in a
different directory.
"""

import contextlib
import os
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core.process_groups_config import (
    MultiModuleProcessGroupCollection,
    ProcessGroupCollection,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training import checkpointing
from megatron.training.checkpointing import get_checkpoint_name, load_checkpoint, save_checkpoint
from megatron.training.global_vars import set_args
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_checkpointing import (
    MockModel,
    create_args,
    create_ckpt_load_args,
    init_model_parallel,
)
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.usefixtures("run_config")

requires_four_rank_multiple = pytest.mark.skipif(
    Utils.world_size < 4 or Utils.world_size % 4 != 0,
    reason="Builds a TP2 x PP2 grid, which needs a multiple of 4 ranks",
)


class _GlobalGridForbidden:
    """Stands in for ``parallel_state`` inside checkpointing: every global grid read fails."""

    def __getattr__(self, name):
        raise AssertionError(f"checkpointing read the global parallel grid: mpu.{name}")


@contextlib.contextmanager
def _no_global_grid():
    """Fail on any read of the global parallel grid made by checkpointing."""
    with (
        mock.patch.object(checkpointing, "mpu", _GlobalGridForbidden()),
        mock.patch.object(
            ProcessGroupCollection,
            "use_mpu_process_groups",
            side_effect=AssertionError("checkpointing built a collection from the global grid"),
        ),
    ):
        yield


class _StandInGroup:
    """A process group stand-in that reports a fixed rank and size."""

    def __init__(self, rank, size):
        self._rank = rank
        self._size = size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


def _grid_unlike_the_global_one():
    """Return a TP2 x PP2 x DP(world/4) collection with the tensor ranks fastest.

    The replica-id group (``dp_cp_gtp_remat``) is a separate communicator over the data-parallel
    ranks, so the tests can tell it apart from the replicate group.
    """
    world_size = torch.distributed.get_world_size()
    dp_size = world_size // 4

    def global_rank(tp_rank, pp_rank, dp_rank):
        return tp_rank + 2 * pp_rank + 4 * dp_rank

    def subgroup(ranks_per_group):
        group, _ = torch.distributed.new_subgroups_by_enumeration(ranks_per_group)
        return group

    dp_ranks = [[global_rank(t, p, d) for d in range(dp_size)] for p in range(2) for t in range(2)]
    dp = subgroup(dp_ranks)
    single = subgroup([[rank] for rank in range(world_size)])
    return ProcessGroupCollection(
        tp=subgroup(
            [[global_rank(t, p, d) for t in range(2)] for d in range(dp_size) for p in range(2)]
        ),
        pp=subgroup(
            [[global_rank(t, p, d) for p in range(2)] for d in range(dp_size) for t in range(2)]
        ),
        cp=single,
        ep=single,
        dp=dp,
        dp_gtp_remat=dp,
        dp_cp=dp,
        dp_cp_gtp_remat=subgroup(dp_ranks),
        expt_dp=dp,
    )


def test_get_checkpoint_name_reads_ranks_from_collection():
    """Ranks that are not given come from the collection, never from the global grid."""
    pg_collection = ProcessGroupCollection(
        tp=_StandInGroup(rank=1, size=2),
        pp=_StandInGroup(rank=2, size=4),
        ep=_StandInGroup(rank=3, size=4),
    )
    with _no_global_grid():
        name = get_checkpoint_name("ckpt", 5, pg_collection=pg_collection)
        explicit_tp_name = get_checkpoint_name(
            "ckpt", 5, tensor_rank=0, expert_parallel=False, pg_collection=pg_collection
        )
        dense_name = get_checkpoint_name(
            "ckpt",
            5,
            pg_collection=ProcessGroupCollection(
                tp=_StandInGroup(rank=1, size=2), pp=_StandInGroup(rank=0, size=1)
            ),
        )

    assert name == os.path.join("ckpt", "iter_0000005", "mp_rank_01_002_003", "model_optim_rng.pt")
    assert explicit_tp_name == os.path.join(
        "ckpt", "iter_0000005", "mp_rank_00_002", "model_optim_rng.pt"
    )
    # A collection without an expert group has no expert parallelism.
    assert dense_name == os.path.join("ckpt", "iter_0000005", "mp_rank_01", "model_optim_rng.pt")


def test_checkpoint_pg_collection_resolution():
    """An explicit collection wins over the model's, and only a model without one reads globals."""
    explicit, owned = ProcessGroupCollection(), ProcessGroupCollection()
    model = [SimpleNamespace(pg_collection=owned)]

    assert checkpointing._checkpoint_pg_collection(model, explicit) is explicit
    assert checkpointing._checkpoint_pg_collection(model) is owned

    global_grid = ProcessGroupCollection()
    with mock.patch.object(
        ProcessGroupCollection, "use_mpu_process_groups", return_value=global_grid
    ) as use_mpu_process_groups:
        assert checkpointing._checkpoint_pg_collection([SimpleNamespace()]) is global_grid
    use_mpu_process_groups.assert_called_once()

    multi_module = MultiModuleProcessGroupCollection(
        module_pgs={"language": owned}, language_model_module_name="language"
    )
    with pytest.raises(ValueError, match="MultiModuleProcessGroupCollection"):
        checkpointing._checkpoint_pg_collection(model, multi_module)


@requires_four_rank_multiple
@pytest.mark.parametrize("pass_collection", [False, True], ids=["model", "argument"])
def test_legacy_checkpoint_round_trip_on_model_grid(
    init_model_parallel, create_ckpt_load_args, tmp_path_dist_ckpt, pass_collection
):
    """Legacy shards are named, written and read by the ranks of the model's grid."""
    args = create_ckpt_load_args
    args.ckpt_format = "torch"
    args.use_dist_ckpt = False
    args.use_distributed_optimizer = False
    args.fp16 = False
    args.bf16 = False

    grid = _grid_unlike_the_global_one()
    tp_rank, pp_rank = grid.tp.rank(), grid.pp.rank()
    config = TransformerConfig(num_layers=1, kv_channels=1)
    model, restored = MockModel(config), MockModel(config)
    with torch.no_grad():
        model.l.weight.fill_(10 * tp_rank + pp_rank + 1)
    if pass_collection:
        kwargs = {"pg_collection": grid}
    else:
        kwargs = {}
        model.pg_collection = grid
        restored.pg_collection = grid

    with TempNamedDir(
        tmp_path_dist_ckpt / "test_legacy_checkpoint_round_trip_on_model_grid", sync=True
    ) as ckpt_dir:
        args.load = ckpt_dir
        args.save = ckpt_dir
        args.save_tokenizer_assets = False
        set_args(args)

        # Data-parallel replicas of one shard share its RNG state.
        torch.manual_seed(1000 + 10 * tp_rank + pp_rank)
        saved_rng = torch.get_rng_state().clone()
        with _no_global_grid():
            save_checkpoint(123, [model], None, None, 456, **kwargs)

        shard_dirs = {
            name for name in os.listdir(ckpt_dir / "iter_0000123") if name.startswith("mp_rank_")
        }
        assert shard_dirs == {f"mp_rank_{t:02d}_{p:03d}" for t in range(2) for p in range(2)}

        torch.manual_seed(0)
        with _no_global_grid():
            loaded = load_checkpoint([restored], None, None, strict=True, **kwargs)

    assert loaded == (123, 456)
    assert torch.equal(restored.l.weight, model.l.weight)
    assert torch.equal(torch.get_rng_state(), saved_rng)


@requires_four_rank_multiple
def test_torch_dist_checkpoint_round_trip_on_model_grid(
    init_model_parallel, create_ckpt_load_args, tmp_path_dist_ckpt
):
    """The RNG shards and the replica ids of a distributed checkpoint use the model's grid."""
    args = create_ckpt_load_args
    args.ckpt_format = "torch_dist"
    args.use_dist_ckpt = True
    args.use_distributed_optimizer = False
    args.fp16 = False
    args.bf16 = False

    grid = _grid_unlike_the_global_one()
    config = TransformerConfig(num_layers=1, kv_channels=1)
    model, restored = MockModel(config), MockModel(config)
    model.pg_collection = grid
    restored.pg_collection = grid

    with TempNamedDir(
        tmp_path_dist_ckpt / "test_torch_dist_checkpoint_round_trip_on_model_grid", sync=True
    ) as ckpt_dir:
        args.load = ckpt_dir
        args.save = ckpt_dir
        args.save_tokenizer_assets = False
        set_args(args)

        torch.manual_seed(1000 + torch.distributed.get_rank())
        saved_rng = torch.get_rng_state().clone()
        with (
            _no_global_grid(),
            mock.patch.object(
                checkpointing, "get_rng_state", wraps=checkpointing.get_rng_state
            ) as save_rng_state,
        ):
            save_checkpoint(123, [model], None, None, 456)

        torch.manual_seed(0)
        with (
            _no_global_grid(),
            mock.patch.object(
                checkpointing, "get_rng_state", wraps=checkpointing.get_rng_state
            ) as load_rng_state,
        ):
            loaded = load_checkpoint([restored], None, None, strict=True)

    assert loaded == (123, 456)
    for rng_state_call in (save_rng_state.call_args, load_rng_state.call_args):
        _, tp_group, pp_group = rng_state_call.args
        assert tp_group is grid.tp and pp_group is grid.pp
        assert rng_state_call.kwargs["dp_group"] is grid.dp
        assert rng_state_call.kwargs["dp_cp_group"] is grid.dp_cp_gtp_remat
    assert model._called_metadata[-1]["dp_cp_group"] is grid.dp_cp_gtp_remat
    assert restored._called_metadata[-1]["dp_cp_group"] is grid.dp_cp_gtp_remat
    # Every rank restores the RNG shard it saved, keyed by its rank in the model's grid.
    assert torch.equal(torch.get_rng_state(), saved_rng)
