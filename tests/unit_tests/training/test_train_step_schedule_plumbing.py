# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Training-loop process-group plumbing tests."""

import ast
import inspect
import textwrap
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


def test_gpu_sniff_uses_explicit_model_groups_without_mpu():
    from megatron.training import gpu_sniff_test

    groups = SimpleNamespace(ep=object(), dp=object(), tp=object())
    with (
        mock.patch.object(training_mod.ProcessGroupCollection, "use_mpu_process_groups") as mpu,
        mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock()),
        mock.patch.object(training_mod, "print_datetime"),
        mock.patch.object(gpu_sniff_test, "run_gpu_sniff_test") as run,
    ):
        training_mod._run_gpu_sniff_test("startup", pg_collection=groups)

    mpu.assert_not_called()
    run.assert_called_once_with("startup", pg_collection=groups)


def test_gpu_sniff_preserves_mpu_fallback_for_legacy_callers():
    from megatron.training import gpu_sniff_test

    groups = object()
    with (
        mock.patch.object(
            training_mod.ProcessGroupCollection, "use_mpu_process_groups", return_value=groups
        ) as mpu,
        mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock()),
        mock.patch.object(training_mod, "print_datetime"),
        mock.patch.object(gpu_sniff_test, "run_gpu_sniff_test") as run,
    ):
        training_mod._run_gpu_sniff_test("legacy")

    mpu.assert_called_once_with(required_pgs=["ep", "dp", "tp"])
    assert run.call_args.kwargs["pg_collection"] is groups


def test_periodic_gpu_sniff_uses_wrapped_model_groups():
    groups = object()
    model = [SimpleNamespace(module=SimpleNamespace(pg_collection=groups))]
    args = SimpleNamespace(
        train_sync_interval=None,
        log_interval=1,
        log_straggler=False,
        check_weight_hash_across_dp_replicas_interval=None,
        adlr_autoresume=False,
        profile=False,
        gpu_sniff_test_interval=10,
        manual_gc=False,
    )
    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "_run_gpu_sniff_test") as run,
    ):
        training_mod.post_training_step_callbacks(model, None, None, 9, None, 0)
        run.assert_not_called()
        training_mod.post_training_step_callbacks(model, None, None, 10, None, 0)

    run.assert_called_once_with("iteration      10", pg_collection=groups)


def test_startup_gpu_sniff_uses_model_groups():
    train_source = textwrap.dedent(inspect.getsource(training_mod.train))
    train_tree = ast.parse(train_source)
    startup_call = next(
        node
        for node in ast.walk(train_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_run_gpu_sniff_test"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "before training"
    )
    pg_keyword = next(
        keyword for keyword in startup_call.keywords if keyword.arg == "pg_collection"
    )
    assert isinstance(pg_keyword.value, ast.Name)
    assert pg_keyword.value.id == "model_pg_collection"
