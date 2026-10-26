# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The data path uses the process groups of the model or run it serves, not the global grid.

Each test either makes every global parallel_state accessor fail, or gives the code groups whose
layout differs from the global grid, so a read of the global grid cannot pass unnoticed.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import pretrain_gpt
import pretrain_hybrid
from megatron.core import parallel_state, tensor_parallel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.training.utils import is_first_or_last_pipeline_stage
from tests.unit_tests.test_utilities import Utils


class _FakeGroup:
    """Process-group stand-in with a rank, a size and the global ranks it spans."""

    def __init__(self, rank, ranks):
        self._rank = rank
        self.ranks = list(ranks)

    def rank(self):
        return self._rank

    def size(self):
        return len(self.ranks)


class _Wrapper:
    def __init__(self, module):
        self.module = module

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class _HostValue:
    """Batch entry that stays on the host when get_batch moves it to the GPU."""

    def cuda(self, non_blocking=False):
        return self


def _forbid_global_process_groups(monkeypatch):
    """Fail the test on any read of the global groups, ranks, sizes or pipeline stages."""

    def fail(*args, **kwargs):
        pytest.fail("the data path must not read the global parallel state")

    for name, value in list(vars(parallel_state).items()):
        if name.startswith(("get_", "is_pipeline_")) and callable(value):
            monkeypatch.setattr(parallel_state, name, fail)
    monkeypatch.setattr(
        ProcessGroupCollection,
        "use_mpu_process_groups",
        Mock(side_effect=AssertionError("supplied groups must not be rebuilt from globals")),
    )


def _use_fake_groups(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: group.ranks)


@pytest.mark.parametrize(
    "pp_rank,pp_size,vp_stage,vp_size,expected",
    [
        (0, 1, None, None, True),
        (0, 4, None, None, True),
        (3, 4, None, None, True),
        (1, 4, None, None, False),
        (0, 4, 0, 2, True),
        (0, 4, 1, 2, False),
        (3, 4, 1, 2, True),
        (3, 4, 0, 2, False),
        (1, 4, 1, 2, False),
        # Without a virtual pipeline the chunk index does not narrow the stage.
        (3, 4, 0, None, True),
    ],
)
def test_stage_check_reads_the_supplied_pipeline_group(
    monkeypatch, pp_rank, pp_size, vp_stage, vp_size, expected
):
    _forbid_global_process_groups(monkeypatch)
    _use_fake_groups(monkeypatch)
    pp_group = _FakeGroup(pp_rank, range(pp_size))

    assert is_first_or_last_pipeline_stage(vp_stage, pp_group=pp_group, vp_size=vp_size) is expected


@pytest.mark.parametrize("vp_size", [None, 2])
def test_stage_check_matches_the_global_grid(vp_size):
    pp_size = min(Utils.world_size, 4)
    if pp_size < 2 or Utils.world_size % pp_size:
        pytest.skip("needs a world size divisible by a pipeline size of at least 2")
    Utils.initialize_model_parallel(
        pipeline_model_parallel_size=pp_size, virtual_pipeline_model_parallel_size=vp_size
    )
    try:
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        for vp_stage in [None] + list(range(vp_size or 1)):
            assert is_first_or_last_pipeline_stage(
                vp_stage, pp_group=pp_group, vp_size=vp_size
            ) == is_first_or_last_pipeline_stage(vp_stage), f"{vp_stage=}"
    finally:
        Utils.destroy_model_parallel()


@pytest.fixture(params=[pretrain_gpt, pretrain_hybrid], ids=["gpt", "hybrid"])
def batch_entrypoint(request, monkeypatch):
    """A get_batch entry point with stubbed TP and CP helpers that record their groups."""
    module = request.param
    _forbid_global_process_groups(monkeypatch)
    _use_fake_groups(monkeypatch)
    args = SimpleNamespace(
        sequence_packing_scheduler=None,
        dynamic_context_parallel=False,
        context_parallel_size=1,
        sft=False,
        dataloader_inter_document_masking=False,
        create_attention_mask_in_dataloader=False,
        micro_batch_size=1,
        seq_length=8,
        pipeline_model_parallel_size=2,
    )
    config = SimpleNamespace(
        pipeline_model_parallel_layout=None,
        mtp_num_layers=None,
        virtual_pipeline_model_parallel_size=None,
        sequence_parallel=True,
        tensor_model_parallel_size=2,
        linear_cp_layout="zigzag",
        attention_cp_layout="zigzag",
    )
    monkeypatch.setattr(module, "get_args", lambda: args)
    monkeypatch.setattr(module, "core_transformer_config_from_args", lambda args: config)

    def broadcast(batch, **kwargs):
        return {**dict.fromkeys(module.BATCH_KEYS), **batch}

    tp_broadcast = Mock(side_effect=broadcast)
    monkeypatch.setattr(module, "get_batch_on_this_tp_rank", tp_broadcast)
    if module is pretrain_gpt:
        cp_shard = Mock(side_effect=lambda batch, **kwargs: batch)
        monkeypatch.setattr(module, "get_batch_on_this_cp_rank", cp_shard)
    else:
        cp_shard = Mock(return_value=object())
        monkeypatch.setattr(module, "get_batches_on_this_cp_rank", cp_shard)
    return SimpleNamespace(
        module=module, config=config, tp_broadcast=tp_broadcast, cp_shard=cp_shard
    )


def _groups(tp_rank, pp_rank, pp_size):
    """Groups whose global ranks differ from any global grid of the test's world."""
    return ProcessGroupCollection(
        tp=_FakeGroup(tp_rank, [5, 7]),
        pp=_FakeGroup(pp_rank, range(11, 11 + pp_size)),
        cp=_FakeGroup(0, [5]),
        tp_cp=_FakeGroup(tp_rank, [5, 7]),
    )


@pytest.mark.parametrize(
    "tp_rank,pp_rank,pp_size,vp_stage,vp_size",
    [(0, 0, 2, None, None), (1, 1, 2, None, None), (0, 1, 2, 1, 2), (1, 0, 4, 0, 2)],
)
def test_get_batch_broadcasts_and_shards_over_the_supplied_groups(
    batch_entrypoint, tp_rank, pp_rank, pp_size, vp_stage, vp_size
):
    entry = batch_entrypoint
    entry.config.virtual_pipeline_model_parallel_size = vp_size
    pgs = _groups(tp_rank, pp_rank, pp_size)
    sample = {"tokens": _HostValue()}
    data_iterator = iter([sample]) if tp_rank == 0 else None

    result = entry.module.get_batch(data_iterator, vp_stage=vp_stage, pg_collection=pgs)

    batch = entry.tp_broadcast.call_args.args[0]
    if tp_rank == 0:
        assert batch["tokens"] is sample["tokens"]
    else:
        assert batch == {}
    kwargs = entry.tp_broadcast.call_args.kwargs
    assert kwargs["broadcast_group"] is pgs.tp
    assert kwargs["broadcast_src_rank"] == 5
    assert kwargs["tp_rank"] == tp_rank
    assert kwargs["is_pipeline_first_stage"] is (pp_rank == 0)
    assert kwargs["is_pipeline_last_stage"] is (pp_rank == pp_size - 1)
    assert kwargs["mtp_on_this_rank"] is False
    cp_kwargs = entry.cp_shard.call_args.kwargs
    assert cp_kwargs["cp_group"] is pgs.cp
    if entry.module is pretrain_hybrid:
        assert cp_kwargs["tp_group"] is pgs.tp
        assert cp_kwargs["tp_cp_group"] is pgs.tp_cp
        assert result is entry.cp_shard.return_value
    else:
        tokens = result[entry.module.BATCH_KEYS.index("tokens")]
        assert tokens is (sample["tokens"] if tp_rank == 0 else None)


@pytest.mark.parametrize(
    "pp_rank,pp_size,vp_stage,vp_size", [(1, 3, None, None), (1, 2, 0, 2), (0, 2, 1, 2)]
)
def test_get_batch_skips_stages_of_the_supplied_pipeline_without_data(
    batch_entrypoint, pp_rank, pp_size, vp_stage, vp_size
):
    entry = batch_entrypoint
    entry.config.virtual_pipeline_model_parallel_size = vp_size
    data_iterator = iter([{"tokens": _HostValue()}])

    result = entry.module.get_batch(
        data_iterator, vp_stage=vp_stage, pg_collection=_groups(0, pp_rank, pp_size)
    )

    entry.tp_broadcast.assert_not_called()
    entry.cp_shard.assert_not_called()
    assert next(data_iterator) is not None
    if entry.module is pretrain_hybrid:
        assert result.get_batch()["tokens"] is None
    else:
        assert result == [None] * len(entry.module.BATCH_KEYS)


@pytest.mark.parametrize("missing", ["tp", "pp", "cp", "tp_cp"])
def test_get_batch_rejects_a_collection_without_its_groups(batch_entrypoint, missing):
    entry = batch_entrypoint
    groups = {name: _FakeGroup(0, [5]) for name in ("tp", "pp", "cp", "tp_cp")}
    del groups[missing]
    pgs = ProcessGroupCollection(**groups)
    if missing == "tp_cp" and entry.module is pretrain_gpt:
        # GPT shards the batch without the joint TP x CP group.
        entry.module.get_batch(iter([{"tokens": _HostValue()}]), pg_collection=pgs)
        return

    with pytest.raises(ValueError, match=f"pg_collection.{missing}"):
        entry.module.get_batch(iter([{"tokens": _HostValue()}]), pg_collection=pgs)
    entry.tp_broadcast.assert_not_called()


def _strided_tp_group(world_size):
    """Pair rank r with rank r + world_size // 2, unlike the global grid's adjacent TP pairs."""
    rank = torch.distributed.get_rank()
    half = world_size // 2
    tp_group = None
    for first in range(half):
        ranks = [first, first + half]
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            tp_group = group
    return tp_group


def _sample(value, micro_batch_size, seq_length):
    shape = (micro_batch_size, seq_length)
    return {
        "tokens": torch.full(shape, value, dtype=torch.int64),
        "labels": torch.full(shape, value + 1, dtype=torch.int64),
        "loss_mask": torch.ones(shape, dtype=torch.float32),
        "position_ids": torch.arange(seq_length, dtype=torch.int64).repeat(micro_batch_size, 1),
    }


@pytest.mark.parametrize("module", [pretrain_gpt, pretrain_hybrid], ids=["gpt", "hybrid"])
def test_get_batch_reads_on_a_tensor_parallel_grid_unlike_the_global_one(module, monkeypatch):
    world_size = Utils.world_size
    if world_size < 4 or world_size % 2:
        pytest.skip("needs an even world size of at least 4")
    Utils.initialize_model_parallel(tensor_model_parallel_size=2)
    try:
        tp_group = _strided_tp_group(world_size)
        pgs = ProcessGroupCollection(
            tp=tp_group,
            pp=parallel_state.get_pipeline_model_parallel_group(),
            cp=parallel_state.get_context_parallel_group(),
        )
        source_rank = torch.distributed.get_process_group_ranks(tp_group)[0]
        micro_batch_size, seq_length = 2, 8
        args = SimpleNamespace(
            sequence_packing_scheduler=None,
            dynamic_context_parallel=False,
            context_parallel_size=1,
            sft=False,
            dataloader_inter_document_masking=False,
            create_attention_mask_in_dataloader=False,
            micro_batch_size=micro_batch_size,
            seq_length=seq_length,
            pipeline_model_parallel_size=1,
        )
        config = SimpleNamespace(
            pipeline_model_parallel_layout=None,
            mtp_num_layers=None,
            virtual_pipeline_model_parallel_size=None,
            sequence_parallel=False,
            tensor_model_parallel_size=2,
            linear_cp_layout="zigzag",
            attention_cp_layout="zigzag",
        )
        # Only the first rank of each collection TP group holds data. On the global grid the
        # first rank of half of the TP groups has none.
        rank = torch.distributed.get_rank()
        data_iterator = (
            iter([_sample(rank, micro_batch_size, seq_length)]) if tp_group.rank() == 0 else None
        )

        with monkeypatch.context() as patch:
            _forbid_global_process_groups(patch)
            patch.setattr(module, "get_args", lambda: args)
            patch.setattr(module, "core_transformer_config_from_args", lambda args: config)
            result = module.get_batch(data_iterator, pg_collection=pgs)

        if module is pretrain_gpt:
            tokens = result[module.BATCH_KEYS.index("tokens")]
        else:
            tokens = result.get_batch()["tokens"]
        expected = _sample(source_rank, micro_batch_size, seq_length)["tokens"]
        torch.testing.assert_close(tokens.cpu(), expected)
    finally:
        Utils.destroy_model_parallel()


def test_vlm_get_batch_broadcasts_over_the_supplied_tensor_parallel_group(monkeypatch):
    import pretrain_vlm

    _forbid_global_process_groups(monkeypatch)
    tp_group = _FakeGroup(1, [3, 7])
    monkeypatch.setattr(
        pretrain_vlm,
        "get_args",
        lambda: SimpleNamespace(context_parallel_size=1, sequence_parallel=False),
    )
    calls = []

    def broadcast_data(keys, data, datatype, tp_group=None):
        calls.append((tuple(keys), tp_group))
        return {key: torch.zeros(1, 4, dtype=datatype) for key in keys}

    monkeypatch.setattr(tensor_parallel, "broadcast_data", broadcast_data)

    pretrain_vlm.get_batch(None, pg_collection=ProcessGroupCollection(tp=tp_group))

    assert calls == [
        (("tokens", "position_ids", "labels"), tp_group),
        (("image", "loss_mask"), tp_group),
    ]


def test_vlm_forward_step_passes_the_model_groups(monkeypatch):
    import pretrain_vlm

    pgs = ProcessGroupCollection(tp=_FakeGroup(0, [0]))
    output = object()
    model = _Wrapper(Mock(spec=[], pg_collection=pgs, return_value=(output, object())))
    fetch = Mock(return_value=(None,) * 7)
    monkeypatch.setattr(pretrain_vlm, "get_batch", fetch)
    monkeypatch.setattr(pretrain_vlm, "get_timers", lambda: Mock())
    data_iterator = object()

    output_tensor, _ = pretrain_vlm.forward_step(data_iterator, model)

    assert output_tensor is output
    assert fetch.call_args.args == (data_iterator,)
    assert fetch.call_args.kwargs["pg_collection"] is pgs
