# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The post-training loss and quantization summary use the process groups of the model.

Each test either makes every global parallel_state accessor fail, or gives the code groups whose
layout differs from the global grid, so a read of the global grid cannot pass unnoticed.
"""

import warnings
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import megatron.training
from megatron.core import parallel_state, process_groups_config
from megatron.core.process_groups_config import (
    MultiModuleProcessGroupCollection,
    ProcessGroupCollection,
    ProcessGroupFallbackWarning,
)
from megatron.post_training import loss_func as loss_func_module
from tests.unit_tests.test_utilities import Utils


class _FakeGroup:
    """Process-group stand-in with a rank and a size."""

    def __init__(self, rank, size):
        self._rank = rank
        self._size = size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


@pytest.fixture(autouse=True)
def fresh_warning_registry(monkeypatch):
    """Each test observes the first fallback warning of every owner."""
    monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())


def _forbid_global_process_groups(monkeypatch):
    """Fail the test on any read of the global groups, ranks, sizes or pipeline stages."""

    def fail(*args, **kwargs):
        pytest.fail("post-training code must not read the global parallel state")

    for name, value in list(vars(parallel_state).items()):
        if name.startswith(("get_", "is_pipeline_")) and callable(value):
            monkeypatch.setattr(parallel_state, name, fail)
    monkeypatch.setattr(
        ProcessGroupCollection,
        "use_mpu_process_groups",
        Mock(side_effect=AssertionError("supplied groups must not be rebuilt from globals")),
    )


@pytest.fixture
def all_reduce_groups(monkeypatch):
    """Record the group of every all-reduce instead of running it on the fake groups."""
    groups = []
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed, "all_reduce", lambda tensor, group=None, **kwargs: groups.append(group)
    )
    return groups


def test_sequence_parallel_loss_uses_the_supplied_tensor_parallel_group(
    monkeypatch, all_reduce_groups
):
    _forbid_global_process_groups(monkeypatch)
    tp_group = _FakeGroup(rank=1, size=2)
    # The second half of the mask belongs to TP rank 1 of 2.
    loss_mask = torch.tensor([[1.0, 0.0, 1.0, 1.0]])
    output = torch.tensor([[2.0, 3.0]])

    loss = loss_func_module._mask_loss((output, False, True), loss_mask, tp_group=tp_group)

    assert loss.item() == pytest.approx(5.0)
    assert all_reduce_groups == [tp_group]


def test_split_distillation_loss_is_reduced_over_the_supplied_group(monkeypatch, all_reduce_groups):
    _forbid_global_process_groups(monkeypatch)
    tp_group = _FakeGroup(rank=0, size=4)

    loss = loss_func_module._mask_loss(
        (torch.tensor([[1.0, 2.0]]), True, False), torch.ones(1, 2), tp_group=tp_group
    )

    assert loss.item() == pytest.approx(3.0)
    assert all_reduce_groups == [tp_group]


def test_plain_loss_needs_no_group(monkeypatch, all_reduce_groups):
    _forbid_global_process_groups(monkeypatch)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loss = loss_func_module._mask_loss(torch.tensor([[1.0, 2.0, 4.0]]), torch.ones(1, 3))

    assert loss.item() == pytest.approx(7.0)
    assert all_reduce_groups == []


def test_omitted_group_warns_and_uses_the_global_group(monkeypatch, all_reduce_groups):
    global_tp_group = _FakeGroup(rank=0, size=2)
    monkeypatch.setattr(
        parallel_state, "get_tensor_model_parallel_group", lambda *args, **kwargs: global_tp_group
    )

    with pytest.warns(
        ProcessGroupFallbackWarning, match="_mask_loss was called without `tp_group`"
    ) as record:
        loss_func_module._mask_loss((torch.tensor([[1.0, 2.0]]), True, False), torch.ones(1, 2))

    message = str(record[0].message)
    assert "deprecated since Megatron Core 0.21" in message
    assert "removed in 0.23" in message
    assert all_reduce_groups == [global_tp_group]


def test_loss_func_reduces_distillation_losses_over_the_model_group(monkeypatch, all_reduce_groups):
    _forbid_global_process_groups(monkeypatch)
    monkeypatch.setattr(
        loss_func_module, "get_args", lambda: SimpleNamespace(export_kd_teacher_load=True)
    )
    tp_group = _FakeGroup(rank=0, size=2)

    def compute_kd_loss(student_loss, loss_reduction_fn):
        split_loss = loss_reduction_fn((torch.tensor([[4.0, 4.0]]), True, False))
        return {
            "kd_loss": student_loss + split_loss,
            "logits_loss": split_loss,
            "intermediate_loss": torch.tensor(0.0),
        }

    model = SimpleNamespace(
        pg_collection=ProcessGroupCollection(tp=tp_group),
        training=True,
        compute_kd_loss=compute_kd_loss,
    )

    loss, num_tokens, report = loss_func_module.loss_func(
        torch.ones(1, 2), torch.tensor([[1.0, 2.0]]), model
    )

    assert loss.item() == pytest.approx(11.0)
    assert num_tokens.item() == 2
    assert report["logits distillation loss"][0].item() == pytest.approx(8.0)
    # Only the split distillation loss is reduced, over the model's tensor-parallel group.
    assert all_reduce_groups == [tp_group]


def test_loss_func_without_a_model_computes_the_plain_loss(monkeypatch, all_reduce_groups):
    _forbid_global_process_groups(monkeypatch)
    monkeypatch.setattr(
        loss_func_module, "get_args", lambda: SimpleNamespace(export_kd_teacher_load=False)
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loss, num_tokens, report = loss_func_module.loss_func(
            torch.ones(1, 3), torch.tensor([[1.0, 2.0, 4.0]]), None
        )

    assert loss.item() == pytest.approx(7.0)
    assert num_tokens.item() == 3
    assert report["lm loss"][0].item() == pytest.approx(7.0)
    assert all_reduce_groups == []


def test_loss_func_without_a_model_warns_for_a_split_loss(monkeypatch, all_reduce_groups):
    global_tp_group = _FakeGroup(rank=0, size=2)
    monkeypatch.setattr(
        parallel_state, "get_tensor_model_parallel_group", lambda *args, **kwargs: global_tp_group
    )
    monkeypatch.setattr(
        loss_func_module, "get_args", lambda: SimpleNamespace(export_kd_teacher_load=False)
    )

    with pytest.warns(
        ProcessGroupFallbackWarning, match="_mask_loss was called without `tp_group`"
    ):
        loss, _, _ = loss_func_module.loss_func(
            torch.ones(1, 2), (torch.tensor([[1.0, 2.0]]), True, False), None
        )

    assert loss.item() == pytest.approx(3.0)
    assert all_reduce_groups == [global_tp_group]


@pytest.fixture
def quant_summary(monkeypatch):
    """print_distributed_quant_summary on a quantized model stub whose summary is recorded."""
    pytest.importorskip("modelopt", reason="modelopt is not installed")
    from megatron.post_training import utils as post_training_utils

    _forbid_global_process_groups(monkeypatch)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(post_training_utils, "is_quantized", lambda model: True)
    print_summary = Mock()
    monkeypatch.setattr(post_training_utils.mtq, "print_quant_summary", print_summary)
    monkeypatch.setattr(megatron.training, "print_rank_0", lambda *args, **kwargs: None)
    return SimpleNamespace(
        run=post_training_utils.print_distributed_quant_summary, printed=print_summary
    )


def _quantized_model(dp_cp_rank, tp_rank):
    return SimpleNamespace(
        pg_collection=ProcessGroupCollection(
            dp_cp_gtp_remat=_FakeGroup(dp_cp_rank, 2),
            tp=_FakeGroup(tp_rank, 4),
            ep=_FakeGroup(2, 4),
            pp=_FakeGroup(3, 4),
        )
    )


@pytest.mark.parametrize(
    "dp_cp_rank,tp_rank,prints", [(0, 0, True), (0, 1, True), (0, 2, False), (1, 0, False)]
)
def test_quant_summary_selects_ranks_by_the_model_groups(
    quant_summary, capsys, dp_cp_rank, tp_rank, prints
):
    model = _quantized_model(dp_cp_rank, tp_rank)

    quant_summary.run([model])

    assert quant_summary.printed.called is prints
    if prints:
        assert f"TP rank {tp_rank}, EP rank 2, PP rank 3" in capsys.readouterr().out


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(SimpleNamespace(), id="no-collection-attribute"),
        pytest.param(SimpleNamespace(pg_collection=None), id="none"),
        pytest.param(
            SimpleNamespace(pg_collection=ProcessGroupCollection(tp=_FakeGroup(0, 1))),
            id="incomplete-collection",
        ),
        pytest.param(
            SimpleNamespace(
                pg_collection=MultiModuleProcessGroupCollection(
                    module_pgs={"llm": ProcessGroupCollection()}
                )
            ),
            id="multimodule-collection",
        ),
    ],
)
def test_quant_summary_without_the_model_groups_warns_and_uses_the_global_ranks(
    quant_summary, monkeypatch, capsys, model
):
    global_groups = ProcessGroupCollection(
        dp_cp_gtp_remat=_FakeGroup(0, 2),
        tp=_FakeGroup(1, 4),
        ep=_FakeGroup(3, 4),
        pp=_FakeGroup(2, 4),
    )
    monkeypatch.setattr(parallel_state, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(
        ProcessGroupCollection,
        "use_mpu_process_groups",
        staticmethod(lambda required_pgs=None: global_groups),
    )

    with pytest.warns(
        ProcessGroupFallbackWarning,
        match="print_distributed_quant_summary was called without `model.pg_collection`",
    ) as record:
        quant_summary.run(model)

    message = str(record[0].message)
    assert "deprecated since Megatron Core 0.21" in message
    assert "removed in 0.23" in message
    assert quant_summary.printed.called
    assert "TP rank 1, EP rank 3, PP rank 2" in capsys.readouterr().out


def test_quant_summary_with_the_model_groups_does_not_warn(quant_summary):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        quant_summary.run(_quantized_model(dp_cp_rank=0, tp_rank=0))

    assert quant_summary.printed.called


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_split_loss_is_reduced_over_a_tensor_parallel_grid_unlike_the_global_one(monkeypatch):
    world_size = Utils.world_size
    if world_size < 4 or world_size % 2:
        pytest.skip("needs an even world size of at least 4")
    Utils.initialize_model_parallel(tensor_model_parallel_size=2)
    try:
        tp_group = _strided_tp_group(world_size)
        rank = torch.distributed.get_rank()
        output = torch.full((1, 2), float(rank + 1), device="cuda")
        with monkeypatch.context() as patch:
            _forbid_global_process_groups(patch)
            loss = loss_func_module._mask_loss(
                (output, True, False), torch.ones(1, 2, device="cuda"), tp_group=tp_group
            )
        ranks = torch.distributed.get_process_group_ranks(tp_group)
        assert loss.item() == pytest.approx(sum(2.0 * (r + 1) for r in ranks))
    finally:
        Utils.destroy_model_parallel()
