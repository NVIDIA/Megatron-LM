# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for explicit objects threaded through ``train_step``."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

from megatron.training import training as training_mod


class _Rerun:
    """Run the forward/backward body once, then ask train_step to exit before optimizer.step."""

    _ran = False

    def should_run_forward_backward(self, data_iterator):
        run, self._ran = not self._ran, True
        return run

    def should_checkpoint_and_exit(self):
        return False, True, 0  # (checkpoint, exit, code)


class _RerunThroughOptimizer(_Rerun):
    """Run one forward/backward body and continue through the optimizer step."""

    def should_checkpoint_and_exit(self):
        return False, False, 0


def _run(**kwargs):
    args = SimpleNamespace(
        save_params_interval=None,
        save_activations_interval=None,
        save_tokens_per_expert_interval=None,
        save_wgrads_interval=None,
        save_dgrads_interval=None,
        reuse_grad_buf_for_mxfp8_param_ag=False,
        overlap_param_gather=False,
        seq_length=8,
        micro_batch_size=1,
        decoder_seq_length=None,
        empty_unused_memory_level=0,
    )
    captured = {}
    model = [SimpleNamespace(force_all_reduce=False, zero_grad_buffer=lambda: None)]
    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock()),
        mock.patch.object(training_mod, "get_rerun_state_machine", return_value=_Rerun()),
        mock.patch.object(training_mod, "get_num_microbatches", return_value=1),
        mock.patch.object(training_mod, "has_nvidia_modelopt", False),
    ):
        training_mod.train_step(
            forward_step_func=lambda *a, **k: None,
            data_iterator=iter([]),
            model=model,
            optimizer=SimpleNamespace(zero_grad=lambda: None),
            opt_param_scheduler=None,
            config=SimpleNamespace(),
            forward_backward_func=lambda **kw: captured.update(kw) or [],
            iteration=0,
            **kwargs,
        )
    return captured


def test_train_step_forwards_schedule_plumbing():
    p2p, pg = object(), object()
    captured = _run(p2p_communicator=p2p, pg_collection=pg)
    assert captured["p2p_communicator"] is p2p and captured["pg_collection"] is pg


def test_train_step_defaults_to_none():
    captured = _run()
    assert captured["p2p_communicator"] is None and captured["pg_collection"] is None


def test_train_step_invokes_tensor_metric_observer_before_optimizer():
    args = SimpleNamespace(
        save_params_interval=None,
        save_activations_interval=None,
        save_tokens_per_expert_interval=None,
        save_wgrads_interval=None,
        save_dgrads_interval=None,
        reuse_grad_buf_for_mxfp8_param_ag=False,
        overlap_param_gather=False,
        seq_length=8,
        micro_batch_size=1,
        decoder_seq_length=None,
        empty_unused_memory_level=0,
        vision_pretraining=False,
        barrier_with_L1_time=False,
        qk_clip=False,
        log_max_attention_logit=False,
        log_num_zeros_in_grad=False,
        data_parallel_size=1,
        gtp_weight_remat_size=1,
    )
    events = []
    pg_collection = SimpleNamespace(mp=object(), pp=object(), dp_cp=object(), dp_cp_gtp_remat=None)
    model = [
        SimpleNamespace(
            force_all_reduce=False, zero_grad_buffer=lambda: None, pg_collection=pg_collection
        )
    ]
    optimizer = SimpleNamespace(
        zero_grad=lambda: None, step=lambda: events.append("optimizer") or (True, 1.0, 0)
    )
    scheduler = mock.MagicMock()

    def observer(**kwargs):
        events.append("observer")
        assert model[0].force_all_reduce is False
        assert kwargs == {
            "model": model,
            "optimizer": optimizer,
            "iteration": 7,
            "pg_collection": pg_collection,
        }

    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock()),
        mock.patch.object(
            training_mod, "get_rerun_state_machine", return_value=_RerunThroughOptimizer()
        ),
        mock.patch.object(training_mod, "get_num_microbatches", return_value=1),
        mock.patch.object(training_mod, "has_nvidia_modelopt", False),
        mock.patch.object(training_mod, "is_pp_last_stage", return_value=False),
        mock.patch.object(
            training_mod,
            "logical_and_across_model_parallel_group",
            side_effect=lambda value, group: value,
        ),
        mock.patch.object(
            training_mod,
            "reduce_max_stat_across_model_parallel_group",
            side_effect=lambda value, group: value,
        ),
    ):
        training_mod.train_step(
            forward_step_func=lambda *a, **k: None,
            data_iterator=iter([]),
            model=model,
            optimizer=optimizer,
            opt_param_scheduler=scheduler,
            config=SimpleNamespace(),
            forward_backward_func=lambda **kw: [],
            iteration=7,
            tensor_metric_observer=observer,
        )

    assert events == ["observer", "optimizer"]
    assert model[0].force_all_reduce is False
    scheduler.step.assert_called_once_with(increment=1)


def test_train_step_scopes_forward_tensor_observation_around_schedule():
    args = SimpleNamespace(
        save_params_interval=None,
        save_activations_interval=None,
        save_tokens_per_expert_interval=None,
        save_wgrads_interval=None,
        save_dgrads_interval=None,
        reuse_grad_buf_for_mxfp8_param_ag=False,
        overlap_param_gather=False,
        seq_length=8,
        micro_batch_size=1,
        decoder_seq_length=None,
        empty_unused_memory_level=0,
        vision_pretraining=False,
        barrier_with_L1_time=False,
        qk_clip=False,
        log_max_attention_logit=False,
        log_num_zeros_in_grad=False,
        data_parallel_size=1,
        gtp_weight_remat_size=1,
    )
    events = []
    pg_collection = SimpleNamespace(mp=object(), pp=object(), dp_cp=object(), dp_cp_gtp_remat=None)
    model = [
        SimpleNamespace(
            force_all_reduce=False, zero_grad_buffer=lambda: None, pg_collection=pg_collection
        )
    ]
    optimizer = SimpleNamespace(
        zero_grad=lambda: None, step=lambda: events.append("optimizer") or (True, 1.0, 0)
    )

    class Observer:
        @contextmanager
        def observe_forward_backward(self, **kwargs):
            assert kwargs == {
                "model": model,
                "iteration": 7,
                "pg_collection": pg_collection,
            }
            events.append("observe-enter")
            yield
            events.append("observe-exit")

        def __call__(self, **kwargs):
            events.append("commit")

    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock()),
        mock.patch.object(
            training_mod, "get_rerun_state_machine", return_value=_RerunThroughOptimizer()
        ),
        mock.patch.object(training_mod, "get_num_microbatches", return_value=1),
        mock.patch.object(training_mod, "has_nvidia_modelopt", False),
        mock.patch.object(training_mod, "is_pp_last_stage", return_value=False),
        mock.patch.object(
            training_mod,
            "logical_and_across_model_parallel_group",
            side_effect=lambda value, group: value,
        ),
        mock.patch.object(
            training_mod,
            "reduce_max_stat_across_model_parallel_group",
            side_effect=lambda value, group: value,
        ),
    ):
        training_mod.train_step(
            forward_step_func=lambda *a, **k: None,
            data_iterator=iter([]),
            model=model,
            optimizer=optimizer,
            opt_param_scheduler=mock.MagicMock(),
            config=SimpleNamespace(),
            forward_backward_func=lambda **kw: events.append("forward-backward") or [],
            iteration=7,
            tensor_metric_observer=Observer(),
        )

    assert events == [
        "observe-enter",
        "forward-backward",
        "observe-exit",
        "commit",
        "optimizer",
    ]
