# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Focused coverage for DSA/CSA indexer-loss metric tracking."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.process_groups_config import MultiModuleProcessGroupCollection
from megatron.core.transformer.experimental_attention_variant import dsa as dsa_module
from megatron.core.transformer.experimental_attention_variant import dsa_logging


@pytest.fixture(autouse=True)
def clear_dsa_metric_tracker():
    """Keep process-global tracker state isolated across tests."""
    dsa_logging.DSAIndexerLossLoggingHelper.tracker.clear()
    yield
    dsa_logging.DSAIndexerLossLoggingHelper.tracker.clear()


def test_dsa_reexports_metric_logging_helper():
    """Keep the established DSA import path bound to the shared tracker class."""
    assert dsa_module.DSAIndexerLossLoggingHelper is dsa_logging.DSAIndexerLossLoggingHelper


def test_resolve_dsa_metric_pg_collection_single_module_and_legacy():
    """Single-module callers keep their collection and legacy callers keep fallback mode."""
    model_pg_collection = object()
    schedule_pg_collection = object()

    assert dsa_logging.resolve_dsa_metric_pg_collection(model_pg_collection) == (
        True,
        model_pg_collection,
    )
    assert dsa_logging.resolve_dsa_metric_pg_collection(
        model_pg_collection, schedule_pg_collection=schedule_pg_collection
    ) == (True, model_pg_collection)
    assert dsa_logging.resolve_dsa_metric_pg_collection(None) == (True, None)


def test_resolve_dsa_metric_pg_collection_multi_module():
    """Multi-module collections select the language model or exclude encoder-only ranks."""
    model_pg_collection = object()
    language_pg_collection = object()
    language_schedule = MultiModuleProcessGroupCollection(
        module_pgs={"language": language_pg_collection}, language_model_module_name="language"
    )
    encoder_only_schedule = MultiModuleProcessGroupCollection(
        module_pgs={"encoder": object()}, language_model_module_name=None
    )

    assert dsa_logging.resolve_dsa_metric_pg_collection(
        model_pg_collection, schedule_pg_collection=language_schedule
    ) == (True, language_pg_collection)
    assert dsa_logging.resolve_dsa_metric_pg_collection(language_schedule) == (
        True,
        language_pg_collection,
    )
    assert dsa_logging.resolve_dsa_metric_pg_collection(
        model_pg_collection, schedule_pg_collection=encoder_only_schedule
    ) == (False, None)
    assert dsa_logging.resolve_dsa_metric_pg_collection(encoder_only_schedule) == (False, None)


def test_hybrid_cp_tracker_preserves_nominal_global_batch_weighting(monkeypatch):
    """Mixed effective-CP samples retain equal nominal-global-batch weighting."""
    helper = dsa_logging.DSAIndexerLossLoggingHelper
    configured_cp_size = 4
    parent_size = 8
    configured_dp_size = parent_size // configured_cp_size
    cp1_local_sums = torch.arange(1, 9, dtype=torch.float32)
    cp2_local_sums = torch.arange(1, 9, dtype=torch.float32)
    cp8_first_local_sums = torch.arange(1, 9, dtype=torch.float32)
    cp8_second_local_sums = torch.arange(2, 10, dtype=torch.float32)
    scheduled_local_sums = (
        cp1_local_sums,
        cp2_local_sums,
        cp8_first_local_sums,
        cp8_second_local_sums,
    )
    expected_sample_sums = torch.cat(
        (
            cp1_local_sums,
            cp2_local_sums.reshape(-1, 2).sum(dim=1),
            cp8_first_local_sums.sum().reshape(1),
            cp8_second_local_sums.sum().reshape(1),
        )
    )
    rank_totals = torch.stack(scheduled_local_sums).sum(dim=0)
    nominal_num_microbatches = expected_sample_sums.numel() // configured_dp_size
    pp_group = object()
    parent_dp_cp_group = object()
    calls = []
    logged = []
    helper.tracker.update(
        {"values": rank_totals[:1].clone(), "agreed_size": 1, "agreed_size_pp_group": pp_group}
    )

    def all_reduce(tensor, op=None, group=None):
        calls.append((group, op))
        if group is parent_dp_cp_group:
            torch.testing.assert_close(tensor, rank_totals[:1] * configured_cp_size)
            tensor.fill_(rank_totals.sum() * configured_cp_size / parent_size)

    class Writer:
        @staticmethod
        def add_scalar(name, value, iteration):
            logged.append((name, value.clone(), iteration))

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    helper.track_indexer_metrics(
        loss_scale=1 / nominal_num_microbatches,
        iteration=3,
        writer=Writer(),
        num_layers=1,
        num_indexer_layers=1,
        hybrid_context_parallel=True,
        configured_cp_size=configured_cp_size,
        pp_group=pp_group,
        final_avg_group=parent_dp_cp_group,
    )

    assert calls == [(pp_group, None), (parent_dp_cp_group, torch.distributed.ReduceOp.AVG)]
    assert logged[0][0] == "indexer loss"
    torch.testing.assert_close(logged[0][1], expected_sample_sums.mean())
    assert logged[0][2] == 3
    torch.testing.assert_close(helper.tracker["values"], torch.zeros(1))


def test_static_indexer_logging_preserves_group_updates():
    """Static logging retains its historical last-writer group behavior."""
    helper = dsa_logging.DSAIndexerLossLoggingHelper
    first_group = object()
    second_group = object()
    helper.tracker.update({"values": torch.zeros(1), "agreed_size": 1})

    helper.save_loss_to_tracker(
        loss=torch.tensor(1.0), layer_number=1, num_layers=1, avg_group=first_group
    )
    helper.save_loss_to_tracker(
        loss=torch.tensor(1.0), layer_number=1, num_layers=1, avg_group=second_group
    )

    torch.testing.assert_close(helper.tracker["values"], torch.tensor([2.0]))
    assert helper.tracker["avg_group"] is second_group


def test_hybrid_cp_indexer_reduction_ignores_stale_logical_groups(monkeypatch):
    """Hybrid reduction uses the stable full-data group, not last-writer metadata."""
    helper = dsa_logging.DSAIndexerLossLoggingHelper
    pp_group = object()
    parent_dp_cp_group = object()
    stale_logical_cp_group = object()
    calls = []
    helper.tracker.update(
        {
            "values": torch.tensor([3.0]),
            "agreed_size": 1,
            "agreed_size_pp_group": pp_group,
            "reduce_group": stale_logical_cp_group,
            "avg_group": stale_logical_cp_group,
        }
    )
    monkeypatch.setattr(
        dsa_logging.parallel_state,
        "get_pipeline_model_parallel_group",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected global PP lookup")),
    )
    monkeypatch.setattr(
        dsa_logging.parallel_state,
        "get_data_parallel_group",
        lambda with_context_parallel: (_ for _ in ()).throw(
            AssertionError("unexpected global DP lookup")
        ),
    )
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor, op=None, group=None: calls.append((group, op)),
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    helper.reduce_loss_in_tracker(
        num_layers=1,
        hybrid_context_parallel=True,
        configured_cp_size=4,
        pp_group=pp_group,
        final_avg_group=parent_dp_cp_group,
    )

    torch.testing.assert_close(helper.tracker["values"], torch.tensor([12.0]))
    assert [group for group, _ in calls] == [pp_group, parent_dp_cp_group]
    assert calls[1][1] == torch.distributed.ReduceOp.AVG


def test_static_indexer_reduction_uses_explicit_language_groups(monkeypatch):
    """Static logging does not fall back to globals when language groups are supplied."""
    helper = dsa_logging.DSAIndexerLossLoggingHelper
    pp_group = object()
    final_avg_group = object()
    calls = []
    helper.tracker.update(
        {"values": torch.tensor([2.0]), "agreed_size": 1, "agreed_size_pp_group": pp_group}
    )
    monkeypatch.setattr(
        dsa_logging.parallel_state,
        "get_pipeline_model_parallel_group",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected global PP lookup")),
    )
    monkeypatch.setattr(
        dsa_logging.parallel_state,
        "get_data_parallel_group",
        lambda with_context_parallel: (_ for _ in ()).throw(
            AssertionError("unexpected global DP lookup")
        ),
    )
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor, op=None, group=None: calls.append((group, op)),
    )

    helper.reduce_loss_in_tracker(num_layers=1, pp_group=pp_group, final_avg_group=final_avg_group)
    assert calls == [(pp_group, None), (final_avg_group, torch.distributed.ReduceOp.AVG)]


def test_collection_prefers_gtp_inclusive_full_data_group(monkeypatch):
    """Main metrics include every distinct-data GTP-remat peer in the final average."""
    helper = dsa_logging.DSAIndexerLossLoggingHelper
    pp_group = object()
    dp_group = object()
    dp_cp_group = object()
    full_data_group = object()
    pg_collection = SimpleNamespace(
        pp=pp_group, dp=dp_group, dp_cp=dp_cp_group, dp_cp_gtp_remat=full_data_group
    )
    calls = []
    helper.tracker.update(
        {"values": torch.tensor([2.0]), "agreed_size": 1, "agreed_size_pp_group": pp_group}
    )
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor, op=None, group=None: calls.append((group, op)),
    )

    helper.reduce_loss_in_tracker(pg_collection=pg_collection, num_layers=1)

    assert calls == [(pp_group, None), (full_data_group, torch.distributed.ReduceOp.AVG)]


@pytest.mark.parametrize(
    ("cp_group_key", "cp_op"),
    (("reduce_group", None), ("avg_group", torch.distributed.ReduceOp.AVG)),
)
def test_static_indexer_reduction_normalizes_writer_before_pp(monkeypatch, cp_group_key, cp_op):
    """A writer stage applies its static-CP reduction before sharing values over PP."""
    helper = dsa_logging.DSAIndexerLossLoggingHelper
    cp_group = object()
    pp_group = object()
    final_avg_group = object()
    calls = []
    helper.tracker.update(
        {
            "values": torch.tensor([2.0]),
            "agreed_size": 1,
            "agreed_size_pp_group": pp_group,
            cp_group_key: cp_group,
        }
    )

    def all_reduce(tensor, op=None, group=None):
        calls.append((group, op))
        if group is cp_group:
            tensor.fill_(5.0)
        elif group is pp_group:
            torch.testing.assert_close(tensor, torch.tensor([5.0]))

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    helper.reduce_loss_in_tracker(num_layers=1, pp_group=pp_group, final_avg_group=final_avg_group)
    assert calls == [
        (cp_group, cp_op),
        (pp_group, None),
        (final_avg_group, torch.distributed.ReduceOp.AVG),
    ]


def test_static_indexer_reduction_empty_pp_stage_receives_normalized_peer_values(monkeypatch):
    """An empty PP stage needs no CP metadata after writer stages normalize first."""
    helper = dsa_logging.DSAIndexerLossLoggingHelper
    pp_group = object()
    final_avg_group = object()
    calls = []
    peer_values = torch.tensor([3.0, 7.0])
    helper.tracker.update(
        {
            "values": torch.zeros_like(peer_values),
            "agreed_size": 2,
            "agreed_size_pp_group": pp_group,
        }
    )

    def all_reduce(tensor, op=None, group=None):
        calls.append((group, op))
        if group is pp_group:
            tensor.copy_(peer_values)

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    helper.reduce_loss_in_tracker(num_layers=2, pp_group=pp_group, final_avg_group=final_avg_group)
    assert calls == [(pp_group, None), (final_avg_group, torch.distributed.ReduceOp.AVG)]
    torch.testing.assert_close(helper.tracker["values"], peer_values)


def test_hybrid_cp_indexer_reduction_grows_zero_filled_pp_stage(monkeypatch):
    """A PP stage without local losses grows a correctly shaped zero tensor."""
    helper = dsa_logging.DSAIndexerLossLoggingHelper
    pp_group = object()
    parent_dp_cp_group = object()
    calls = []
    helper.tracker["values"] = torch.zeros(2)

    def all_reduce(tensor, op=None, group=None):
        calls.append((group, op, tuple(tensor.shape)))
        if op == torch.distributed.ReduceOp.MAX:
            tensor.fill_(3)

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    helper.reduce_loss_in_tracker(
        num_layers=2,
        hybrid_context_parallel=True,
        configured_cp_size=4,
        pp_group=pp_group,
        final_avg_group=parent_dp_cp_group,
    )

    assert calls == [
        (pp_group, torch.distributed.ReduceOp.MAX, (1,)),
        (pp_group, None, (3,)),
        (parent_dp_cp_group, torch.distributed.ReduceOp.AVG, (3,)),
    ]
    torch.testing.assert_close(helper.tracker["values"], torch.zeros(3))
    assert helper.tracker["agreed_size"] == 3


def test_zero_capacity_initialization_does_not_poison_later_model(monkeypatch):
    """A writer-free scan leaves no state that can leak into a later model lifecycle."""
    helper = dsa_logging.DSAIndexerLossLoggingHelper
    pp_group = object()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    size = dsa_logging.initialize_dsa_metric_tracker(
        torch.nn.Module(), SimpleNamespace(pp=pp_group)
    )
    assert size == 0
    assert helper.tracker == {}

    class MetricModule(torch.nn.Module):
        logs_dsa_indexer_loss = True

        def __init__(self):
            super().__init__()
            self.layer_number = 1

    replacement_pp_group = object()
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    assert (
        dsa_logging.initialize_dsa_metric_tracker(
            MetricModule(), SimpleNamespace(pp=replacement_pp_group)
        )
        == 1
    )
    assert helper.tracker["agreed_size"] == 1
    assert helper.tracker["agreed_size_pp_group"] is replacement_pp_group
    assert helper.tracker["values"].shape == (1,)


def test_reduction_rejects_initialized_tracker_from_different_pp_group():
    """A fixed-capacity tracker cannot be reduced through another PP domain."""
    helper = dsa_logging.DSAIndexerLossLoggingHelper
    helper.tracker.update(
        {"values": torch.zeros(1), "agreed_size": 1, "agreed_size_pp_group": object()}
    )
    with pytest.raises(RuntimeError, match="cached size belongs to a different PP group"):
        helper.reduce_loss_in_tracker(num_layers=1, pp_group=object())


def test_reduction_accepts_recreated_pp_group_with_same_ranks(monkeypatch):
    """Equivalent process groups remain valid after model-parallel reinitialization."""
    helper = dsa_logging.DSAIndexerLossLoggingHelper
    old_pp_group = object()
    new_pp_group = object()
    final_avg_group = object()
    calls = []
    helper.tracker.update(
        {
            "values": torch.ones(1),
            "agreed_size": 1,
            "agreed_size_pp_group": old_pp_group,
            "agreed_size_pp_ranks": (0, 1),
        }
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: [0, 1])
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor, op=None, group=None: calls.append((group, op)),
    )

    helper.reduce_loss_in_tracker(
        num_layers=1, pp_group=new_pp_group, final_avg_group=final_avg_group
    )
    assert calls == [(new_pp_group, None), (final_avg_group, torch.distributed.ReduceOp.AVG)]


@pytest.mark.parametrize("configured_cp_size", (None, 0))
def test_hybrid_cp_indexer_reduction_rejects_invalid_configured_cp_size(configured_cp_size):
    """The Hybrid-CP normalization factor must be a positive configured CP width."""
    with pytest.raises(ValueError, match="configured_cp_size must be positive"):
        dsa_logging.DSAIndexerLossLoggingHelper.reduce_loss_in_tracker(
            num_layers=1,
            hybrid_context_parallel=True,
            configured_cp_size=configured_cp_size,
            pp_group=object(),
            final_avg_group=object(),
        )


def test_hybrid_cp_indexer_reduction_rejects_nondivisible_full_data_group(monkeypatch):
    """The stable full-data group must contain whole configured-CP domains."""
    parent_dp_cp_group = object()
    pp_group = object()
    dsa_logging.DSAIndexerLossLoggingHelper.tracker.update(
        {"values": torch.ones(1), "agreed_size": 1, "agreed_size_pp_group": pp_group}
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(dsa_logging, "get_pg_size", lambda group: 6)

    with pytest.raises(ValueError, match="must be divisible by configured_cp_size"):
        dsa_logging.DSAIndexerLossLoggingHelper.reduce_loss_in_tracker(
            num_layers=1,
            hybrid_context_parallel=True,
            configured_cp_size=4,
            pp_group=pp_group,
            final_avg_group=parent_dp_cp_group,
        )


@pytest.mark.parametrize("preserve_groups", (None, True))
def test_metric_logging_clears_values_and_optionally_preserves_groups(monkeypatch, preserve_groups):
    """Metric reporting clears values and only graph paths retain reduction groups."""
    helper = dsa_logging.DSAIndexerLossLoggingHelper
    reduce_group = object()
    avg_group = object()
    helper.tracker.update(
        {
            "values": torch.tensor([2.0, 0.0, 6.0, 0.0]),
            "reduce_group": reduce_group,
            "avg_group": avg_group,
        }
    )
    recorded = []
    reduced_with = []

    class Writer:
        @staticmethod
        def add_scalar(name, value, iteration):
            recorded.append((name, value.clone(), iteration))

    monkeypatch.setattr(
        helper,
        "reduce_loss_in_tracker",
        lambda pg_collection=None, num_layers=None, hybrid_context_parallel=False, configured_cp_size=None, pp_group=None, final_avg_group=None: reduced_with.append(
            (pg_collection, hybrid_context_parallel, configured_cp_size, pp_group, final_avg_group)
        ),
    )
    preserve_groups_kwarg = {} if preserve_groups is None else {"preserve_groups": preserve_groups}
    helper.track_indexer_metrics(
        loss_scale=0.5,
        iteration=7,
        writer=Writer(),
        num_indexer_layers=2,
        hybrid_context_parallel=True,
        configured_cp_size=4,
        pp_group=reduce_group,
        final_avg_group=avg_group,
        **preserve_groups_kwarg,
    )

    assert reduced_with == [(None, True, 4, reduce_group, avg_group)]
    name, value, iteration = recorded[0]
    assert name == "indexer loss"
    torch.testing.assert_close(value, torch.tensor(2.0))
    assert iteration == 7
    torch.testing.assert_close(helper.tracker["values"], torch.zeros(4))
    assert helper.tracker["reduce_group"] is (reduce_group if preserve_groups else None)
    assert helper.tracker["avg_group"] is (avg_group if preserve_groups else None)
