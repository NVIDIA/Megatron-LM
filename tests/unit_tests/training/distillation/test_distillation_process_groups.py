# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The cached-logits saver and KD loss run on the process groups of the model they serve.

The distributed tests build a grid unlike the global one and make every read of the global grid
raise, so a group, rank or size taken from the global grid fails the test instead of silently
selecting other ranks.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.training.distillation import cached_logits_loss, logits_saver
from megatron.training.distillation import utils as distillation_utils
from megatron.training.distillation.cached_logits_loss import (
    CachedLogitsKDLoss,
    LossFuncCallable,
    make_teacher_tar_dataset,
    topk_kl_div,
)
from megatron.training.distillation.logits_saver import LogitsSaverHooks
from megatron.training.distillation.utils import resolve_distillation_process_groups
from tests.unit_tests.test_utilities import Utils

_GROUP_NAMES = ('tp', 'cp', 'dp_gtp_remat', 'tp_dp_cp')


def _fake_group(size=1):
    return SimpleNamespace(rank=lambda: 0, size=lambda: size)


def _forbid_global_process_groups(monkeypatch):
    """Make every read of the global grid's groups, ranks and sizes raise."""

    def _global_read(*args, **kwargs):
        raise AssertionError("read the global parallel state instead of the model's groups")

    for name in dir(parallel_state):
        if name.startswith("get_") and name.endswith(
            ("_group", "_groups", "_gloo", "_rank", "_ranks", "_world_size")
        ):
            monkeypatch.setattr(parallel_state, name, _global_read)
    monkeypatch.setattr(ProcessGroupCollection, "use_mpu_process_groups", _global_read)


# ---------------------------------------------------------------------------
# Collection validation (no torch.distributed needed)
# ---------------------------------------------------------------------------


def test_resolve_returns_the_given_collection_without_global_reads(monkeypatch):
    _forbid_global_process_groups(monkeypatch)
    pg_collection = ProcessGroupCollection(**{name: _fake_group() for name in _GROUP_NAMES})

    assert resolve_distillation_process_groups(pg_collection, "owner") is pg_collection


@pytest.mark.parametrize("missing", _GROUP_NAMES)
@pytest.mark.parametrize("explicit_none", [False, True])
def test_resolve_rejects_a_collection_without_a_required_group(missing, explicit_none):
    groups = {name: _fake_group() for name in _GROUP_NAMES if name != missing}
    if explicit_none:
        groups[missing] = None
    pg_collection = ProcessGroupCollection(**groups)

    with pytest.raises(ValueError, match=f"LogitsSaverHooks requires .*'{missing}'"):
        resolve_distillation_process_groups(pg_collection, "LogitsSaverHooks")


# ---------------------------------------------------------------------------
# Distributed: a model grid unlike the global one
# ---------------------------------------------------------------------------


def _strided_grid():
    """Return this rank's groups in a grid laid out unlike the global TP=2 grid.

    The ranks form two pipeline stages, the even and the odd ranks. In each stage, tensor
    parallelism pairs ranks half a stage apart and data parallelism spans the remaining ranks;
    context parallelism is off. ``tp_dp_cp`` is the stage. Every group differs in its ranks from
    the global group of the same role.
    """
    world_size = Utils.world_size
    stages = [list(range(first, world_size, 2)) for first in range(2)]
    pairs = len(stages[0]) // 2
    tp_ranks = [[stage[j], stage[j + pairs]] for stage in stages for j in range(pairs)]
    dp_ranks = [stage[t * pairs : (t + 1) * pairs] for stage in stages for t in range(2)]
    tp, _ = dist.new_subgroups_by_enumeration(tp_ranks)
    dp, _ = dist.new_subgroups_by_enumeration(dp_ranks)
    cp, _ = dist.new_subgroups_by_enumeration([[rank] for rank in range(world_size)])
    stage, _ = dist.new_subgroups_by_enumeration(stages)
    return ProcessGroupCollection(tp=tp, cp=cp, dp_gtp_remat=dp, tp_dp_cp=stage)


def _patch_run_state(monkeypatch, *, global_batch_size=8):
    """Stand in for the training run's arguments and dataset identity."""
    args = SimpleNamespace(
        mtp_num_layers=0,
        micro_batch_size=1,
        global_batch_size=global_batch_size,
        freeze_all_layers=False,
        sft=False,
        dataloader_inter_document_masking=False,
    )
    for module in (logits_saver, cached_logits_loss, distillation_utils):
        monkeypatch.setattr(module, "get_args", lambda: args)
    monkeypatch.setattr(logits_saver, "compute_dataset_hash", lambda: ("hash", {}))
    monkeypatch.setattr(logits_saver, "storage_makedirs", lambda path, exist_ok=True: None)
    # Constructing a saver registers it as the active one; monkeypatch restores the registry.
    monkeypatch.setattr(logits_saver, "_ACTIVE_LOGITS_SAVER", None)
    monkeypatch.setattr(distillation_utils, "_STORAGE_GLOB_CACHE", {})


def _fake_remote_listing(monkeypatch, names):
    """List ``names(rank)`` under a remote root, where ``rank`` is the global rank that lists."""
    monkeypatch.setattr(
        distillation_utils,
        "storage_glob",
        lambda pattern: [f"{pattern.rsplit('/', 1)[0]}/{name}" for name in names(dist.get_rank())],
    )


def _random(*shape, seed):
    """The same tensor on every rank."""
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def _vocab_shard(logits, tp_rank, tp_size):
    shard_size = logits.size(-1) // tp_size
    return logits[..., tp_rank * shard_size : (tp_rank + 1) * shard_size].clone().cuda()


@pytest.mark.skipif(
    Utils.world_size < 4 or Utils.world_size % 4 != 0,
    reason="Requires a world size that is a multiple of 4",
)
class TestDistillationProcessGroups:

    def setup_method(self, method):
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_saver_selects_the_global_groups_without_a_collection(self, monkeypatch):
        _patch_run_state(monkeypatch)

        saver = LogitsSaverHooks(save_dir="unused", k=4)

        assert saver.tp_group is parallel_state.get_tensor_model_parallel_group()
        assert saver.cp_group is parallel_state.get_context_parallel_group()
        assert saver._tp_dp_cp_group is parallel_state.get_tensor_and_data_parallel_group(
            with_context_parallel=True
        )
        assert (saver.tp_rank, saver.tp_size) == (
            parallel_state.get_tensor_model_parallel_rank(),
            parallel_state.get_tensor_model_parallel_world_size(),
        )
        assert (saver.dp_rank, saver.dp_size) == (
            parallel_state.get_data_parallel_rank(),
            parallel_state.get_data_parallel_world_size(),
        )
        assert saver._tp_dst_rank_global == parallel_state.get_tensor_model_parallel_src_rank()

    def test_kd_loss_selects_the_global_groups_without_a_collection(self, monkeypatch):
        _patch_run_state(monkeypatch)

        kd_loss = CachedLogitsKDLoss(logprobs_dir="unused")

        assert kd_loss.tp_group is parallel_state.get_tensor_model_parallel_group()
        assert kd_loss._tp_dp_cp_group is parallel_state.get_tensor_and_data_parallel_group(
            with_context_parallel=True
        )
        assert (kd_loss.dp_rank, kd_loss.dp_size) == (
            parallel_state.get_data_parallel_rank(),
            parallel_state.get_data_parallel_world_size(),
        )
        assert (kd_loss.cp_rank, kd_loss.cp_size) == (
            parallel_state.get_context_parallel_rank(),
            parallel_state.get_context_parallel_world_size(),
        )

    def test_saver_computes_top_k_over_the_model_tensor_parallel_group(self, monkeypatch):
        pg_collection = _strided_grid()
        _patch_run_state(monkeypatch)
        _forbid_global_process_groups(monkeypatch)

        saver = LogitsSaverHooks(
            save_dir="unused", k=4, save_dtype="fp32", pg_collection=pg_collection
        )

        tp_ranks = dist.get_process_group_ranks(pg_collection.tp)
        assert saver.tp_group is pg_collection.tp and saver.cp_group is pg_collection.cp
        assert (saver.tp_rank, saver.tp_size) == (pg_collection.tp.rank(), 2)
        assert (saver.dp_rank, saver.dp_size) == (
            pg_collection.dp_gtp_remat.rank(),
            pg_collection.dp_gtp_remat.size(),
        )
        assert saver._tp_dst_rank_global == min(tp_ranks)
        assert saver._cp_dst_rank_global == dist.get_rank()
        assert saver.metadata_dict["saver"]["dp_size_save"] == pg_collection.dp_gtp_remat.size()

        logits = _random(3, 2, 16, seed=1)
        result = saver._process_single_microbatch(
            _vocab_shard(logits, saver.tp_rank, saver.tp_size)
        )

        if saver.tp_rank != 0:
            assert result is None
            return
        values, indices = result
        expected_values, expected_indices = torch.log_softmax(logits.cuda(), dim=-1).topk(4)
        torch.testing.assert_close(indices, expected_indices)
        torch.testing.assert_close(values, expected_values, rtol=1e-5, atol=1e-5)

    def test_saver_lists_remote_shards_once_per_model_pipeline_stage(self, monkeypatch):
        pg_collection = _strided_grid()
        _patch_run_state(monkeypatch)
        _forbid_global_process_groups(monkeypatch)
        _fake_remote_listing(
            monkeypatch,
            lambda rank: [f"dp{dp}__0-{rank + 8}.tar" for dp in range(Utils.world_size)],
        )
        saver = LogitsSaverHooks(save_dir="msc://bucket/logits", k=4, pg_collection=pg_collection)

        existing_tars = saver.take_pending_data()[4]

        # The first rank of this rank's stage lists for the whole stage.
        stage_first_rank = dist.get_process_group_ranks(pg_collection.tp_dp_cp)[0]
        assert stage_first_rank == dist.get_rank() % 2
        assert existing_tars == [
            f"msc://bucket/logits/dp{saver.dp_rank}__0-{stage_first_rank + 8}.tar"
        ]

    @pytest.mark.parametrize("format_version", [1, 2])
    def test_teacher_loader_lists_remote_shards_once_per_model_pipeline_stage(
        self, monkeypatch, format_version
    ):
        pg_collection = _strided_grid()
        _patch_run_state(monkeypatch)
        _forbid_global_process_groups(monkeypatch)
        root = "msc://bucket/teacher"
        if format_version == 2:
            # The listing rank records its rank in the shard names and the saved DP size.
            _fake_remote_listing(
                monkeypatch, lambda rank: [f"dp{dp}__{rank}-{rank + 8}.tar" for dp in range(2)]
            )
            meta = lambda path: {
                "saver": {
                    "format_version": 2,
                    "mbs_save": 1,
                    "dp_size_save": 1 + dist.get_rank(),
                    "gbs_save": 8,
                }
            }
        else:
            # The listing rank records its rank in the iteration and in the number of DP shards.
            _fake_remote_listing(
                monkeypatch, lambda rank: [f"cp0_dp{dp}__{rank}.tar" for dp in range(1 + rank)]
            )
            meta = lambda path: {"saver": {}}
        monkeypatch.setattr(distillation_utils, "peek_logprobs_metadata", meta)

        dataset = make_teacher_tar_dataset(
            root,
            cp_rank=pg_collection.cp.rank(),
            cp_size=pg_collection.cp.size(),
            dp_rank=pg_collection.dp_gtp_remat.rank(),
            dp_size=pg_collection.dp_gtp_remat.size(),
            ignore_hash=True,
            tp_dp_cp_group=pg_collection.tp_dp_cp,
        )

        stage_first_rank = dist.get_process_group_ranks(pg_collection.tp_dp_cp)[0]
        assert stage_first_rank == dist.get_rank() % 2
        if format_version == 2:
            assert dataset._plan.dp_save == 1 + stage_first_rank
            expected_shard = f"{root}/dp0__{stage_first_rank}-{stage_first_rank + 8}.tar"
        else:
            assert dataset._num_mb_save == 8 // (1 + stage_first_rank)
            expected_shard = f"{root}/cp0_dp0__{stage_first_rank}.tar"
        assert dataset._discover_shards(set(), 0) == [expected_shard]

    def test_kd_loss_builds_the_teacher_loader_on_the_model_groups(self, monkeypatch):
        pg_collection = _strided_grid()
        _patch_run_state(monkeypatch)
        _forbid_global_process_groups(monkeypatch)
        built = {}

        def _make_teacher_tar_dataset(logprobs_dir, cp_rank, cp_size, dp_rank, dp_size, **kwargs):
            built.update(cp=(cp_rank, cp_size), dp=(dp_rank, dp_size), **kwargs)
            return []

        monkeypatch.setattr(
            cached_logits_loss, "make_teacher_tar_dataset", _make_teacher_tar_dataset
        )
        kd_loss = CachedLogitsKDLoss(logprobs_dir="unused", pg_collection=pg_collection)

        kd_loss._init_dataloader(start_sample=0)

        assert built["cp"] == (0, 1)
        assert built["dp"] == (pg_collection.dp_gtp_remat.rank(), pg_collection.dp_gtp_remat.size())
        assert built["tp_dp_cp_group"] is pg_collection.tp_dp_cp

    def test_kd_loss_reduces_over_the_model_tensor_parallel_group(self, monkeypatch):
        pg_collection = _strided_grid()
        _patch_run_state(monkeypatch)
        _forbid_global_process_groups(monkeypatch)
        tp_rank, tp_size = pg_collection.tp.rank(), pg_collection.tp.size()
        student_logits = _random(3, 2, 16, seed=2)
        teacher_values, teacher_indices = torch.log_softmax(_random(3, 2, 16, seed=3), -1).topk(4)
        # The cached teacher top-K of the first iteration, and the student logits of this rank.
        monkeypatch.setattr(
            CachedLogitsKDLoss,
            "_init_dataloader",
            lambda self, start_sample: setattr(
                self, "_dataloader_iter", iter([(0, [teacher_values], [teacher_indices])])
            ),
        )
        monkeypatch.setattr(cached_logits_loss, "get_consumed_train_samples", lambda: 0)
        local_logits = _vocab_shard(student_logits, tp_rank, tp_size)
        monkeypatch.setattr(
            cached_logits_loss,
            "get_student_logits_capture",
            lambda: SimpleNamespace(pop=lambda: local_logits),
        )
        loss_func = LossFuncCallable(logprobs_dir="unused", kd_loss_alpha=1.0)
        model = SimpleNamespace(training=True, pg_collection=pg_collection)
        loss_mask = torch.ones(2, 3, device="cuda")

        _, _, report = loss_func(loss_mask, torch.zeros(2, 3, device="cuda"), model)

        assert loss_func.kd_func.tp_group is pg_collection.tp
        # The whole vocabulary on one rank, without collectives.
        expected = topk_kl_div(
            student_logits.cuda(),
            teacher_values.cuda(),
            teacher_indices.cuda(),
            tp_size=1,
            tp_rank=0,
            tp_group=None,
            add_ghost_token=True,
        ).sum()
        torch.testing.assert_close(
            report["logits distillation loss"][0], expected, rtol=1e-5, atol=1e-5
        )
