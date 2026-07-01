# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Focused tests for the heterogeneous MIMO training entrypoint."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.mimo import pretrain_mimo


def test_parse_and_validate_orders_grid_before_stock_and_checks_validated_optimizer(monkeypatch):
    args = SimpleNamespace(
        use_distributed_optimizer=True,
        world_size=8,
        llm_dp=2,
        llm_tp=2,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        data_parallel_size=8,
        padded_vocab_size=None,
        vocab_size=511,
        make_vocab_size_divisible_by=128,
    )
    calls = []
    reject_after_validation = False

    parser = pretrain_mimo.extra_args_provider(argparse.ArgumentParser())
    local_args = parser.parse_args([])
    assert local_args.dataset_provider == "mock"
    assert local_args.image_token_id == 511
    assert local_args.image_seq_length is None

    def fake_parse_args(extra_args_provider):
        calls.append("parse")
        assert extra_args_provider is pretrain_mimo.extra_args_provider
        return args

    def fake_validate_grid(received, world_size):
        calls.append("grid")
        assert received is args
        assert world_size == 8

    def fake_validate_args(received, defaults):
        calls.append("stock")
        assert received is args
        assert received.world_size == 2
        assert defaults == {"dataloader_type": "external"}
        received.data_parallel_size = received.world_size
        if reject_after_validation:
            received.use_distributed_optimizer = False
        return received

    def fake_calculate(vocab_size, make_divisible_by, tp_size, *, logging_enabled):
        calls.append("vocab")
        assert (vocab_size, make_divisible_by, tp_size, logging_enabled) == (511, 128, 2, False)
        return 512

    monkeypatch.setattr(pretrain_mimo, "parse_args", fake_parse_args)
    monkeypatch.setattr(pretrain_mimo, "validate_hetero_grid_args", fake_validate_grid)
    monkeypatch.setattr(pretrain_mimo, "validate_args", fake_validate_args)
    monkeypatch.setattr(pretrain_mimo, "calculate_padded_vocab_size", fake_calculate)

    assert pretrain_mimo._parse_and_validate() is args
    assert args.world_size == 8
    assert args.data_parallel_size == args.llm_dp
    assert args.padded_vocab_size == 512
    assert calls == ["parse", "grid", "stock", "vocab"]

    reject_after_validation = True
    args.use_distributed_optimizer = True
    args.data_parallel_size = 8
    args.padded_vocab_size = None
    calls.clear()
    with pytest.raises(ValueError, match="--use-distributed-optimizer"):
        pretrain_mimo._parse_and_validate()
    assert calls == ["parse", "grid", "stock"]


def test_main_shuts_down_after_partial_distributed_initialization(mocker):
    mocker.patch.object(
        pretrain_mimo, "_parse_and_validate", return_value=SimpleNamespace(world_size=8)
    )
    mocker.patch.object(pretrain_mimo, "set_global_variables")
    mocker.patch.object(
        pretrain_mimo,
        "initialize_distributed",
        side_effect=RuntimeError("partial initialization failed"),
    )
    shutdown = mocker.patch.object(pretrain_mimo, "shutdown_distributed")

    with pytest.raises(RuntimeError, match="partial initialization failed"):
        pretrain_mimo.main()

    shutdown.assert_called_once_with()


def test_launcher_always_configures_unique_per_rank_logs():
    launcher = (
        Path(__file__).parents[4] / "examples/mimo/scripts/run_hetero_nemotron_20l_mock_train.sh"
    ).read_text()

    assert 'TORCHRUN_LOG_DIR=${TORCHRUN_LOG_DIR:-"${PWD}/logs/torchrun-' in launcher
    assert 'mkdir -p "${TORCHRUN_LOG_DIR}"' in launcher
    assert '--log-dir "${TORCHRUN_LOG_DIR}"' in launcher
    assert "--redirects 3" in launcher
    assert "--tee 3" in launcher
    assert 'if [[ -n "${TORCHRUN_LOG_DIR:-}" ]]' not in launcher
    assert "--moe-flex-dispatcher-backend" not in launcher
