# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the MIMO hooks on the stock optimizer/model setup path:
  (a) ``get_megatron_optimizer`` dispatches a ``MimoModel`` to ``get_mimo_optimizer``;
  (b) ``setup_model_and_optimizer`` forwards an explicit ``pg_collection`` to ``get_model``.

Both are mocked/stubbed so they stay light and need no real distributed build.
"""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

import megatron.training.training as training_mod
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.optimizer import get_megatron_optimizer
from megatron.training.training import setup_model_and_optimizer


def test_get_megatron_optimizer_dispatches_mimo_model():
    """A single MimoModel chunk routes to get_mimo_optimizer."""
    # __new__ gives an isinstance-true MimoModel without running its heavy __init__.
    fake_mimo = MimoModel.__new__(MimoModel)
    sentinel = object()
    with mock.patch(
        "megatron.core.models.mimo.optimizer.get_mimo_optimizer", return_value=sentinel
    ) as get_mimo:
        result = get_megatron_optimizer(SimpleNamespace(optimizer="adam"), [fake_mimo])
    get_mimo.assert_called_once()
    assert result is sentinel


def _make_args():
    return SimpleNamespace(
        skip_train=False,
        perform_rl_step=False,
        no_load_optim=False,
        use_mup=False,
        use_gloo_process_groups=False,
        dump_param_to_param_group_map=False,
        logits_save_dir=None,
        logits_load_dir=None,
        moe_use_upcycling=False,
        fp16=False,
        bf16=False,
        load=None,
        pretrained_checkpoint=None,
        ckpt_convert_format=None,
        micro_batch_size=1,
        iteration=0,
        num_floating_point_operations_so_far=0,
    )


def test_pg_collection_is_forwarded_to_get_model():
    """An explicit pg_collection is threaded through to get_model."""
    args = _make_args()
    fake_model = [mock.MagicMock()]
    sentinel_pg = object()

    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(training_mod, "get_args", return_value=args))
        stack.enter_context(
            mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock())
        )
        stack.enter_context(mock.patch.object(training_mod, "get_one_logger", return_value=None))
        get_model_mock = stack.enter_context(
            mock.patch.object(training_mod, "get_model", return_value=fake_model)
        )
        stack.enter_context(
            mock.patch.object(training_mod, "unwrap_model", return_value=fake_model)
        )
        stack.enter_context(
            mock.patch.object(
                training_mod,
                "get_megatron_optimizer_config",
                return_value=(SimpleNamespace(timers=None, optimizer="adam"), None),
            )
        )
        stack.enter_context(
            mock.patch.object(training_mod, "get_optimizer_param_scheduler", return_value=object())
        )
        stack.enter_context(
            mock.patch.object(training_mod, "get_megatron_optimizer", return_value=object())
        )
        stack.enter_context(
            mock.patch.object(training_mod, "get_num_microbatches", return_value=1)
        )
        stack.enter_context(
            mock.patch.object(training_mod, "get_current_global_batch_size", return_value=1)
        )
        stack.enter_context(
            mock.patch.object(training_mod.mpu, "get_data_parallel_world_size", return_value=1)
        )
        setup_model_and_optimizer(
            model_provider_func=mock.MagicMock(),
            model_type=mock.MagicMock(),
            pg_collection=sentinel_pg,
        )

    assert get_model_mock.call_args.kwargs["pg_collection"] is sentinel_pg
