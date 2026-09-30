# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""train_step forwards p2p_communicator and schedule pg_collection to forward_backward_func."""

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


def _run(*, args_overrides=None, model=None, optimizer=None, **kwargs):
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
    for name, value in (args_overrides or {}).items():
        setattr(args, name, value)
    captured = {}
    model = model or [SimpleNamespace(force_all_reduce=False, zero_grad_buffer=lambda: None)]
    optimizer = optimizer or SimpleNamespace(zero_grad=lambda: None, chained_optimizers=[])
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
            optimizer=optimizer,
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


def test_train_step_uses_optimizer_ddp_config_for_mxfp8_staging():
    class _DistributedOptimizer:
        def __init__(self, overlap_param_gather):
            self.ddp_config = SimpleNamespace(
                reuse_grad_buf_for_mxfp8_param_ag=True, overlap_param_gather=overlap_param_gather
            )
            self._copy_main_params_to_param_buffer = mock.Mock()

    overlapped = _DistributedOptimizer(overlap_param_gather=True)
    nonoverlapped = _DistributedOptimizer(overlap_param_gather=False)
    optimizer = SimpleNamespace(
        zero_grad=lambda: None, chained_optimizers=[overlapped, nonoverlapped]
    )
    model = [
        SimpleNamespace(
            force_all_reduce=False,
            zero_grad_buffer=lambda: None,
            remove_forward_pre_hook_handles={object(): object()},
        )
    ]

    with mock.patch.object(training_mod, "DistributedOptimizer", _DistributedOptimizer):
        # Global args intentionally disagree; the optimizer DDP config is authoritative.
        _run(
            args_overrides={
                "reuse_grad_buf_for_mxfp8_param_ag": False,
                "overlap_param_gather": False,
            },
            model=model,
            optimizer=optimizer,
        )

    overlapped._copy_main_params_to_param_buffer.assert_called_once_with()
    nonoverlapped._copy_main_params_to_param_buffer.assert_not_called()


def test_train_step_supports_bare_distributed_optimizer_for_mxfp8_staging():
    class _DistributedOptimizer:
        def __init__(self):
            self.ddp_config = SimpleNamespace(
                reuse_grad_buf_for_mxfp8_param_ag=True, overlap_param_gather=True
            )
            self._copy_main_params_to_param_buffer = mock.Mock()

        def zero_grad(self):
            pass

    optimizer = _DistributedOptimizer()
    model = [
        SimpleNamespace(
            force_all_reduce=False,
            zero_grad_buffer=lambda: None,
            remove_forward_pre_hook_handles={object(): object()},
        )
    ]

    with mock.patch.object(training_mod, "DistributedOptimizer", _DistributedOptimizer):
        _run(model=model, optimizer=optimizer)

    optimizer._copy_main_params_to_param_buffer.assert_called_once_with()
