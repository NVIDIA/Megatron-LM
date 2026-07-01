# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Focused tests for checkpoint-save process-group forwarding in ``save_checkpoint_and_time``."""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

from megatron.training import training as training_mod


def _save_patches(args):
    """Patch out the heavy dependencies of ``save_checkpoint_and_time``."""
    return (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock()),
        mock.patch.object(training_mod, "get_energy_monitor", return_value=mock.MagicMock()),
        mock.patch.object(training_mod, "should_disable_forward_pre_hook", return_value=False),
        mock.patch.object(training_mod.one_logger_utils, "track_e2e_metrics"),
        mock.patch.object(training_mod.torch.cuda, "empty_cache"),
        mock.patch.object(training_mod, "save_checkpoint"),
    )


def _run_save(model, *, reset_counter=True):
    """Invoke ``save_checkpoint_and_time`` with report_memory patched and return the mock."""
    args = SimpleNamespace(fp8=False, async_save=False, log_progress=False)
    if reset_counter:
        training_mod.num_checkpoints_memory_reported = 0
    with ExitStack() as stack:
        for patcher in _save_patches(args):
            stack.enter_context(patcher)
        report_memory = stack.enter_context(mock.patch.object(training_mod, "report_memory"))
        training_mod.save_checkpoint_and_time(
            iteration=5,
            model=model,
            optimizer=None,
            opt_param_scheduler=None,
            num_floating_point_operations_so_far=0,
            checkpointing_context={},
        )
    return report_memory


def test_report_memory_receives_model_dp_group():
    """Both report_memory calls get the model PGC dp group as process_group."""
    dp_group = object()
    pgc = SimpleNamespace(tp=object(), pp=object(), dp=dp_group, dp_cp=object(), expt_dp=object())
    model = [SimpleNamespace(pg_collection=pgc)]

    report_memory = _run_save(model)

    assert report_memory.call_count == 2
    before_call, after_call = report_memory.call_args_list
    assert before_call.kwargs["process_group"] is dp_group
    assert after_call.kwargs["process_group"] is dp_group
    assert "before save_checkpoint" in before_call.args[0]
    assert "after save_checkpoint" in after_call.args[0]


def test_report_memory_process_group_none_for_stock_model():
    """A stock model without a pg_collection yields process_group=None (mpu fallback)."""
    model = [SimpleNamespace()]

    report_memory = _run_save(model)

    assert report_memory.call_count == 2
    for call in report_memory.call_args_list:
        assert call.kwargs["process_group"] is None


def test_save_checkpoint_threads_model_groups():
    """save_checkpoint receives the model PGC groups and rng prefix."""
    groups = {name: object() for name in ("tp", "pp", "dp", "dp_cp", "expt_dp")}
    pgc = SimpleNamespace(**groups)
    model = [SimpleNamespace(pg_collection=pgc, rng_state_key_prefix="encoder.")]

    args = SimpleNamespace(fp8=False, async_save=False, log_progress=False)
    training_mod.num_checkpoints_memory_reported = 0
    with ExitStack() as stack:
        for patcher in _save_patches(args):
            stack.enter_context(patcher)
        stack.enter_context(mock.patch.object(training_mod, "report_memory"))
        save_checkpoint = stack.enter_context(mock.patch.object(training_mod, "save_checkpoint"))
        training_mod.save_checkpoint_and_time(
            iteration=5,
            model=model,
            optimizer=None,
            opt_param_scheduler=None,
            num_floating_point_operations_so_far=0,
            checkpointing_context={},
        )

    kwargs = save_checkpoint.call_args.kwargs
    assert kwargs["tp_group"] is groups["tp"]
    assert kwargs["pp_group"] is groups["pp"]
    assert kwargs["dp_group"] is groups["dp"]
    assert kwargs["dp_cp_group"] is groups["dp_cp"]
    assert kwargs["expt_dp_group"] is groups["expt_dp"]
    assert kwargs["rng_state_key_prefix"] == "encoder."
