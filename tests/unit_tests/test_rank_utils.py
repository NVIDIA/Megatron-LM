# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from unittest.mock import Mock

from megatron.core import _rank_utils


def test_safe_get_rank_uses_distributed_when_initialized(monkeypatch):
    monkeypatch.setattr(_rank_utils.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(_rank_utils.torch.distributed, "get_rank", lambda: 7)

    assert _rank_utils.safe_get_rank() == 7


def test_safe_get_rank_falls_back_to_env(monkeypatch):
    monkeypatch.setattr(_rank_utils.torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setenv("RANK", "3")

    assert _rank_utils.safe_get_rank() == 3


def test_safe_get_rank_defaults_to_zero_for_missing_or_invalid_env(monkeypatch):
    monkeypatch.setattr(_rank_utils.torch.distributed, "is_initialized", lambda: False)
    monkeypatch.delenv("RANK", raising=False)
    assert _rank_utils.safe_get_rank() == 0

    monkeypatch.setenv("RANK", "not-an-int")
    assert _rank_utils.safe_get_rank() == 0


def test_log_single_rank_only_logs_matching_rank(monkeypatch):
    logger = Mock()

    monkeypatch.setattr(_rank_utils, "safe_get_rank", lambda: 2)
    _rank_utils.log_single_rank(logger, 20, "visible", rank=2, extra={"k": "v"})
    _rank_utils.log_single_rank(logger, 20, "hidden", rank=0)

    logger.log.assert_called_once_with(20, "visible", extra={"k": "v"})
