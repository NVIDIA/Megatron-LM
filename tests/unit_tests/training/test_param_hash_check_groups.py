# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The training loop compares parameter hashes across the model's own data-parallel replicas."""

from types import SimpleNamespace
from unittest import mock

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.training import training as training_mod


def _single_call_kwargs(check, model):
    """Return the keyword arguments of the one call to the mocked check, which got model."""
    assert check.call_count == 1
    (checked_model,), kwargs = check.call_args
    assert checked_model is model
    return kwargs


def _hash_check_kwargs(model):
    """Run the training loop's hash check on model; return the arguments the core check got."""
    check = mock.Mock(return_value=True)
    with mock.patch.object(training_mod, "check_param_hashes_across_dp_replicas", check):
        assert training_mod._param_hashes_match_across_dp_replicas(model)
    return _single_call_kwargs(check, model)


def test_compares_across_the_model_dp_groups():
    pg_collection = ProcessGroupCollection(dp=object(), expt_dp=object())

    kwargs = _hash_check_kwargs([SimpleNamespace(pg_collection=pg_collection)])

    assert kwargs == {
        "cross_check": True,
        "dp_group": pg_collection.dp,
        "expt_dp_group": pg_collection.expt_dp,
    }


def test_collection_without_expert_groups_passes_no_expert_group():
    pg_collection = ProcessGroupCollection(dp=object())

    kwargs = _hash_check_kwargs([SimpleNamespace(pg_collection=pg_collection)])

    assert kwargs["dp_group"] is pg_collection.dp
    assert kwargs["expt_dp_group"] is None


def test_model_without_collection_uses_the_global_groups():
    global_pg_collection = ProcessGroupCollection(dp=object(), expt_dp=object())

    with mock.patch.object(
        ProcessGroupCollection, "use_mpu_process_groups", return_value=global_pg_collection
    ):
        kwargs = _hash_check_kwargs([SimpleNamespace(pg_collection=None)])

    assert kwargs["dp_group"] is global_pg_collection.dp
    assert kwargs["expt_dp_group"] is global_pg_collection.expt_dp


def test_post_training_step_callbacks_check_the_model_dp_groups():
    pg_collection = ProcessGroupCollection(dp=object(), expt_dp=object())
    model = [SimpleNamespace(pg_collection=pg_collection)]
    args = SimpleNamespace(
        train_sync_interval=0,
        log_straggler=False,
        check_weight_hash_across_dp_replicas_interval=2,
        adlr_autoresume=False,
        gpu_sniff_test_interval=None,
        manual_gc=False,
    )
    cfg = SimpleNamespace(
        logger=SimpleNamespace(log_interval=1),
        profiling=SimpleNamespace(use_nsys_profiler=False, use_pytorch_profiler=False),
    )
    check = mock.Mock(return_value=True)

    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_run_config", return_value=cfg),
        mock.patch.object(training_mod, "should_disable_forward_pre_hook", return_value=False),
        mock.patch.object(training_mod, "check_param_hashes_across_dp_replicas", check),
        mock.patch.object(training_mod.torch.distributed, "barrier"),
        mock.patch.object(training_mod, "print_rank_0"),
    ):
        for iteration in (1, 2):
            training_mod.post_training_step_callbacks(model, None, None, iteration, None, 0)

    assert _single_call_kwargs(check, model) == {
        "cross_check": True,
        "dp_group": pg_collection.dp,
        "expt_dp_group": pg_collection.expt_dp,
    }
