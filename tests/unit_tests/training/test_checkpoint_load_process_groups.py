# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Focused tests for checkpoint-load process-group forwarding and encoder-rank crash guards."""

import ast
import inspect
import textwrap
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

from megatron.core.enums import ModelType
from megatron.training import training as training_mod


def _setup_args(**overrides):
    """Minimal args to drive setup_model_and_optimizer down the skip-optimizer load path."""
    values = dict(
        skip_train=True,
        perform_rl_step=False,
        logits_save_dir=None,
        logits_load_dir=None,
        moe_use_upcycling=False,
        load="some/dir",
        pretrained_checkpoint=None,
        use_torch_fsdp2=False,
        ckpt_format="torch_dist",
        iteration=1,
        num_floating_point_operations_so_far=0,
        micro_batch_size=1,
        data_parallel_size=4,
        ckpt_convert_format=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _run_setup(model, args, *, mpu_initialized):
    """Invoke setup_model_and_optimizer with the load path mocked; return the load_checkpoint mock."""
    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(training_mod, "get_args", return_value=args))
        stack.enter_context(
            mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock())
        )
        stack.enter_context(mock.patch.object(training_mod, "get_one_logger", return_value=None))
        stack.enter_context(mock.patch.object(training_mod, "get_model", return_value=model))
        stack.enter_context(mock.patch.object(training_mod, "get_num_microbatches", return_value=2))
        stack.enter_context(
            mock.patch.object(training_mod, "get_current_global_batch_size", return_value=8)
        )
        stack.enter_context(
            mock.patch.object(
                training_mod.mpu, "model_parallel_is_initialized", return_value=mpu_initialized
            )
        )
        stack.enter_context(
            mock.patch.object(training_mod.mpu, "get_data_parallel_world_size", return_value=99)
        )
        load_checkpoint = stack.enter_context(
            mock.patch.object(training_mod, "load_checkpoint", return_value=(0, 0))
        )
        training_mod.setup_model_and_optimizer(ModelType.encoder_or_decoder, lambda *a, **k: model)
    return load_checkpoint


def test_load_checkpoint_receives_model_groups():
    """load_checkpoint is threaded the model PGC groups and rng prefix."""
    groups = {name: object() for name in ("tp", "pp", "dp", "dp_cp", "expt_dp")}
    pgc = SimpleNamespace(**groups)
    model = [SimpleNamespace(pg_collection=pgc, rng_state_key_prefix="encoder.")]

    load_checkpoint = _run_setup(model, _setup_args(), mpu_initialized=True)

    kwargs = load_checkpoint.call_args.kwargs
    assert kwargs["tp_group"] is groups["tp"]
    assert kwargs["pp_group"] is groups["pp"]
    assert kwargs["dp_group"] is groups["dp"]
    assert kwargs["dp_cp_group"] is groups["dp_cp"]
    assert kwargs["expt_dp_group"] is groups["expt_dp"]
    assert kwargs["rng_state_key_prefix"] == "encoder."


def test_load_checkpoint_groups_none_for_stock_model():
    """A stock model without a pg_collection threads None groups (mpu fallback) and empty prefix."""
    model = [SimpleNamespace()]

    load_checkpoint = _run_setup(model, _setup_args(), mpu_initialized=True)

    kwargs = load_checkpoint.call_args.kwargs
    for name in ("tp_group", "pp_group", "dp_group", "dp_cp_group", "expt_dp_group"):
        assert kwargs[name] is None
    assert kwargs["rng_state_key_prefix"] == ""


def test_data_parallel_size_falls_back_to_args_when_mpu_uninitialized():
    """With mpu uninitialized, dp size uses args.data_parallel_size (encoder-rank crash guard)."""
    model = [SimpleNamespace()]
    args = _setup_args(data_parallel_size=4)

    # get_data_parallel_world_size would raise if called before mpu init; assert it is skipped.
    with mock.patch.object(
        training_mod.mpu,
        "get_data_parallel_world_size",
        side_effect=AssertionError("must not read mpu dp size when uninitialized"),
    ):
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(training_mod, "get_args", return_value=args))
            stack.enter_context(
                mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock())
            )
            stack.enter_context(
                mock.patch.object(training_mod, "get_one_logger", return_value=None)
            )
            stack.enter_context(mock.patch.object(training_mod, "get_model", return_value=model))
            stack.enter_context(
                mock.patch.object(training_mod, "get_num_microbatches", return_value=2)
            )
            stack.enter_context(
                mock.patch.object(training_mod, "get_current_global_batch_size", return_value=8)
            )
            stack.enter_context(
                mock.patch.object(
                    training_mod.mpu, "model_parallel_is_initialized", return_value=False
                )
            )
            stack.enter_context(
                mock.patch.object(training_mod, "load_checkpoint", return_value=(0, 0))
            )
            # Should not raise: dp size resolves from args, not mpu.
            training_mod.setup_model_and_optimizer(
                ModelType.encoder_or_decoder, lambda *a, **k: model
            )


def test_data_parallel_size_uses_mpu_when_initialized():
    """With mpu initialized, dp size still reads the mpu value (byte-identical to stock)."""
    model = [SimpleNamespace()]
    args = _setup_args()

    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(training_mod, "get_args", return_value=args))
        stack.enter_context(
            mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock())
        )
        stack.enter_context(mock.patch.object(training_mod, "get_one_logger", return_value=None))
        stack.enter_context(mock.patch.object(training_mod, "get_model", return_value=model))
        stack.enter_context(mock.patch.object(training_mod, "get_num_microbatches", return_value=2))
        stack.enter_context(
            mock.patch.object(training_mod, "get_current_global_batch_size", return_value=8)
        )
        stack.enter_context(
            mock.patch.object(training_mod.mpu, "model_parallel_is_initialized", return_value=True)
        )
        get_dp = stack.enter_context(
            mock.patch.object(training_mod.mpu, "get_data_parallel_world_size", return_value=99)
        )
        stack.enter_context(mock.patch.object(training_mod, "load_checkpoint", return_value=(0, 0)))
        training_mod.setup_model_and_optimizer(ModelType.encoder_or_decoder, lambda *a, **k: model)

    get_dp.assert_called()


def _finalize_guard_node():
    """Return the AST assignment guarding config.finalize_model_grads_func in train()."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(training_mod.train)))
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            for child in ast.walk(node):
                if (
                    isinstance(child, ast.Assign)
                    and isinstance(child.targets[0], ast.Attribute)
                    and child.targets[0].attr == "finalize_model_grads_func"
                ):
                    return node
    return None


def test_finalize_hook_assignment_is_guarded_by_is_none():
    """train() only defaults finalize_model_grads_func when it is None (builder hook survives)."""
    guard = _finalize_guard_node()
    assert guard is not None, "expected a conditional guard around finalize_model_grads_func"
    # Condition must be `config.finalize_model_grads_func is None`.
    test = guard.test
    assert isinstance(test, ast.Compare)
    assert isinstance(test.ops[0], ast.Is)
    assert isinstance(test.left, ast.Attribute)
    assert test.left.attr == "finalize_model_grads_func"
    assert isinstance(test.comparators[0], ast.Constant)
    assert test.comparators[0].value is None
