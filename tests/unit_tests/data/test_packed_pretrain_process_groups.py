# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch

import pretrain_gpt
import pretrain_hybrid
from megatron.core import parallel_state
from megatron.core.context_parallel import ContextParallelBatch
from megatron.core.process_groups_config import ProcessGroupCollection


def _scheduler_return(module):
    """Build a metadata-free scheduler result for the entrypoint under test.

    NOTE(nightly-sync 26_10_2026): get_batch now calls prepare_packed_seq_params on the
    scheduler's result (dev-only frontend hook, restored on main's structure), so an opaque
    object() sentinel no longer survives the handoff -- gpt reads index 5 and hybrid calls
    get_packed_seq_params(). The metadata is None on purpose: with real PackedSeqParams,
    finalize_packed_seq_params would read parallel_state.get_context_parallel_group and trip
    this module's own "must not read parallel_state" guard. Each test still asserts that
    get_batch returns this exact object.
    """
    if module is pretrain_hybrid:
        return ContextParallelBatch.from_single_layout("contiguous", {}, None)
    return (None,) * 7


class _Wrapper:
    def __init__(self, module):
        self.module = module

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)


@pytest.fixture(params=[pretrain_gpt, pretrain_hybrid])
def packed_entrypoint(request, monkeypatch):
    module = request.param
    pgs = ProcessGroupCollection(
        tp=SimpleNamespace(size=lambda: 1, rank=lambda: 0),
        pp=SimpleNamespace(size=lambda: 2, rank=lambda: 1),
        cp=SimpleNamespace(size=lambda: 1, rank=lambda: 0),
    )
    config = SimpleNamespace(
        sequence_packing_scheduler="default_dynamic_cp",
        dynamic_context_parallel=True,
        virtual_pipeline_model_parallel_size=3,
        pipeline_model_parallel_layout=None,
        mtp_num_layers=1,
    )
    args = SimpleNamespace(sequence_packing_scheduler=None, dynamic_context_parallel=False)
    monkeypatch.setattr(module, "get_args", lambda: args)
    monkeypatch.setattr(
        module, "get_run_config", lambda: SimpleNamespace(model=SimpleNamespace(transformer=config))
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def fail(*args, **kwargs):
        pytest.fail("Explicit process-group paths must not read parallel_state")

    for name, value in list(vars(parallel_state).items()):
        if name.startswith("get_") and callable(value):
            monkeypatch.setattr(parallel_state, name, fail)
    return module, pgs, config


@pytest.mark.parametrize("vp_stage,expected_mtp", [(0, False), (2, True)])
def test_packed_batch_uses_supplied_groups_and_mtp_pipeline_stage(
    packed_entrypoint, monkeypatch, vp_stage, expected_mtp
):
    module, pgs, config = packed_entrypoint
    bootstrap = Mock(side_effect=AssertionError("Explicit groups must not bootstrap globals"))
    monkeypatch.setattr(ProcessGroupCollection, "use_mpu_process_groups", bootstrap)
    result, iterator = _scheduler_return(module), object()
    fetch = Mock(return_value=result)
    monkeypatch.setattr(module, "get_batch_on_this_rank_for_sequence_packing", fetch)

    assert module.get_batch(iterator, vp_stage=vp_stage, pg_collection=pgs) is result
    assert fetch.call_args.args == (iterator,)
    kwargs = fetch.call_args.kwargs
    assert kwargs["pg_collection"] is pgs
    assert kwargs["config"] is config
    assert kwargs["vp_stage"] == vp_stage
    assert kwargs["mtp_on_this_rank"] is expected_mtp
    assert kwargs["dynamic_cp"] is True
    assert "dynamic_cp_group_func" not in kwargs
    assert "dynamic_tp_cp_group_func" not in kwargs
    assert kwargs.get("return_context_parallel_batch", False) is (module is pretrain_hybrid)
    bootstrap.assert_not_called()


def test_direct_packed_entrypoint_bootstraps_collection(packed_entrypoint, monkeypatch):
    module, pgs, _ = packed_entrypoint
    bootstrap = Mock(return_value=pgs)
    monkeypatch.setattr(ProcessGroupCollection, "use_mpu_process_groups", bootstrap)
    result, iterator = _scheduler_return(module), object()
    fetch = Mock(return_value=result)
    monkeypatch.setattr(module, "get_batch_on_this_rank_for_sequence_packing", fetch)

    assert module.get_batch(iterator, vp_stage=2) is result
    assert fetch.call_args.kwargs["pg_collection"] is pgs
    assert set(bootstrap.call_args.kwargs["required_pgs"]) == {
        "tp",
        "pp",
        "cp",
        "tp_cp",
        "dynamic_cp",
        "dynamic_tp_cp",
    }


def test_forward_step_passes_groups_from_wrapped_model(packed_entrypoint, monkeypatch):
    module, pgs, _ = packed_entrypoint
    output, iterator = object(), object()
    model = _Wrapper(_Wrapper(Mock(spec=[], pg_collection=pgs, vp_stage=2, return_value=output)))
    if module is pretrain_gpt:
        batch = (None,) * 7
    else:
        batch = SimpleNamespace(get_batch=lambda: {}, get_packed_seq_params=lambda: None)
    fetch = Mock(return_value=batch)
    monkeypatch.setattr(module, "get_batch", fetch)
    monkeypatch.setattr(module, "get_timers", lambda: Mock())
    stimer = MagicMock()
    stimer.return_value = stimer
    monkeypatch.setattr(module, "stimer", stimer)

    forward_output, loss = module.forward_step(iterator, model)
    assert forward_output is output
    assert callable(loss)
    assert fetch.call_args.args == (iterator, 2)
    assert fetch.call_args.kwargs["pg_collection"] is pgs


@pytest.mark.parametrize(
    ("scheduler", "cli_scheduler"),
    [("default_dynamic_cp", None), (None, "dp_balanced"), (None, None)],
    ids=["scheduler", "stale-cli-scheduler", "no-scheduler"],
)
def test_hybrid_forward_step_leaves_scheduler_batches_to_the_scheduler(
    monkeypatch, scheduler, cli_scheduler
):
    """With a packing scheduler, the hybrid forward step keeps get_batch's packed metadata.

    get_batch already prepares the scheduler's packed metadata, so the forward step does not
    prepare it a second time; without a scheduler it still does. Like get_batch, it reads the
    scheduler from the run config, so a stale CLI value cannot make it skip the preparation.
    The microbatch boundaries always go to update_seqlen_stats_from_cu_seqlens, which ignores
    them once train_step has seeded the global statistics from the scheduler.
    """
    module = pretrain_hybrid
    monkeypatch.setattr(
        module, "get_args", lambda: SimpleNamespace(sequence_packing_scheduler=cli_scheduler)
    )
    monkeypatch.setattr(
        module,
        "get_run_config",
        lambda: SimpleNamespace(
            model=SimpleNamespace(transformer=SimpleNamespace(sequence_packing_scheduler=scheduler))
        ),
    )
    output, iterator = object(), object()
    model = _Wrapper(
        _Wrapper(Mock(spec=[], vp_stage=0, pg_collection=None, config=None, return_value=output))
    )
    cu_seqlens = torch.tensor([[0, 64, 128]], dtype=torch.int32)
    packed = object()
    batch = SimpleNamespace(
        get_batch=lambda: {"cu_seqlens": cu_seqlens}, get_packed_seq_params=lambda: packed
    )
    monkeypatch.setattr(module, "get_batch", Mock(return_value=batch))
    monkeypatch.setattr(module, "get_timers", lambda: Mock())
    stimer = MagicMock()
    stimer.return_value = stimer
    monkeypatch.setattr(module, "stimer", stimer)
    stats, prepare = Mock(), Mock()
    monkeypatch.setattr(module, "update_seqlen_stats_from_cu_seqlens", stats)
    monkeypatch.setattr(module, "prepare_packed_seq_params", prepare)

    forward_output, _ = module.forward_step(iterator, model)
    assert forward_output is output
    stats.assert_called_once()
    if scheduler is None:
        assert prepare.call_args.args[0] is packed
    else:
        prepare.assert_not_called()
