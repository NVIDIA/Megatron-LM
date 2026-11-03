# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""get_megatron_optimizer dispatches a MimoModel to get_mimo_optimizer."""

from types import SimpleNamespace
from unittest import mock

import pytest

from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.training.training import setup_model_and_optimizer


def test_get_megatron_optimizer_dispatches_mimo_model():
    # __new__ gives an isinstance-true MimoModel without running its heavy __init__.
    fake_mimo = MimoModel.__new__(MimoModel)
    config = OptimizerConfig(optimizer="adam")
    sentinel = object()
    with mock.patch(
        "megatron.core.models.mimo.optimizer.get_mimo_optimizer", return_value=sentinel
    ) as get_mimo:
        result = get_megatron_optimizer(config, [fake_mimo])
    get_mimo.assert_called_once_with(fake_mimo, config)
    assert result is sentinel


class _OptimizerFactoryCalled(Exception):
    """Stop setup immediately after capturing the optimizer-factory arguments."""


@pytest.mark.parametrize(
    ("optimizer_name", "expects_process_groups"),
    [("muon_h", True), ("muon_ht", True), ("adam", False), ("sgd", False)],
)
def test_setup_forwards_process_groups_only_to_constrained_muon(
    optimizer_name, expects_process_groups
):
    """Setup should share topology with MuonH/HT without changing Adam/SGD behavior."""
    args = SimpleNamespace(
        logits_save_dir=None,
        logits_load_dir=None,
        use_mup=False,
        dump_param_to_param_group_map=False,
    )
    config = OptimizerConfig(optimizer=optimizer_name)
    run_config = SimpleNamespace(
        optimizer=config,
        validation=SimpleNamespace(skip_train=False),
        dist=SimpleNamespace(use_gloo_process_groups=True),
    )
    process_groups = object()
    model = [object()]

    with (
        mock.patch("megatron.training.training.get_args", return_value=args),
        mock.patch("megatron.training.training.get_run_config", return_value=run_config),
        mock.patch("megatron.training.training.get_train_state"),
        mock.patch("megatron.training.training.get_timers"),
        mock.patch("megatron.training.training.get_one_logger", return_value=None),
        mock.patch("megatron.training.training.has_nvidia_modelopt", False),
        mock.patch("megatron.training.training.is_gtp_remat_active", return_value=False),
        mock.patch("megatron.training.training.get_model", return_value=model),
        mock.patch("megatron.training.training.unwrap_model", return_value=model),
        mock.patch(
            "megatron.training.training.get_megatron_optimizer", side_effect=_OptimizerFactoryCalled
        ) as optimizer_factory,
        pytest.raises(_OptimizerFactoryCalled),
    ):
        setup_model_and_optimizer(None, model_provider_func=object(), pg_collection=process_groups)

    expected_process_groups = process_groups if expects_process_groups else None
    assert optimizer_factory.call_args.args[0] is config
    assert optimizer_factory.call_args.kwargs["pg_collection"] is expected_process_groups
