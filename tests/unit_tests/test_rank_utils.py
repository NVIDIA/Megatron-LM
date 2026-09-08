# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import logging
from unittest.mock import Mock, patch

import pytest

from megatron.core._rank_utils import log_single_rank
from megatron.core.utils import log_on_each_pipeline_stage


def test_log_single_rank_skips_rank_query_when_level_is_disabled():
    logger = Mock(spec=logging.Logger)
    logger.isEnabledFor.return_value = False

    with patch("megatron.core._rank_utils.safe_get_rank") as safe_get_rank:
        log_single_rank(logger, logging.DEBUG, "message")

    logger.isEnabledFor.assert_called_once_with(logging.DEBUG)
    safe_get_rank.assert_not_called()
    logger.log.assert_not_called()


def test_log_single_rank_preserves_keyword_call():
    logger = Mock(spec=logging.Logger)
    logger.isEnabledFor.return_value = True

    with patch("megatron.core._rank_utils.safe_get_rank", return_value=3) as safe_get_rank:
        log_single_rank(
            logger=logger, level=logging.INFO, msg="message", rank=3, extra={"key": "value"}
        )

    logger.isEnabledFor.assert_called_once_with(logging.INFO)
    safe_get_rank.assert_called_once_with()
    logger.log.assert_called_once_with(logging.INFO, "message", extra={"key": "value"})


def test_log_on_each_pipeline_stage_skips_group_queries_when_level_is_disabled():
    logger = Mock(spec=logging.Logger)
    logger.isEnabledFor.return_value = False
    tp_group = Mock()
    dp_cp_group = Mock()

    with patch("megatron.core.utils.torch.distributed.is_initialized", return_value=True):
        log_on_each_pipeline_stage(
            logger, logging.DEBUG, "message", tp_group=tp_group, dp_cp_group=dp_cp_group
        )

    logger.isEnabledFor.assert_called_once_with(logging.DEBUG)
    tp_group.rank.assert_not_called()
    dp_cp_group.rank.assert_not_called()
    logger.log.assert_not_called()


def test_log_on_each_pipeline_stage_validates_group_pair_when_level_is_disabled():
    logger = Mock(spec=logging.Logger)
    logger.isEnabledFor.return_value = False

    with patch("megatron.core.utils.torch.distributed.is_initialized", return_value=True):
        with pytest.raises(
            ValueError, match="tp_group and dp_cp_group must be provided or not provided together"
        ):
            log_on_each_pipeline_stage(logger, logging.DEBUG, "message", tp_group=Mock())

    logger.isEnabledFor.assert_not_called()
    logger.log.assert_not_called()


def test_log_on_each_pipeline_stage_requires_distributed_when_level_is_disabled():
    logger = Mock(spec=logging.Logger)
    logger.isEnabledFor.return_value = False

    with patch("megatron.core.utils.torch.distributed.is_initialized", return_value=False):
        with pytest.raises(AssertionError):
            log_on_each_pipeline_stage(logger, logging.DEBUG, "message")

    logger.isEnabledFor.assert_not_called()
    logger.log.assert_not_called()


def test_log_single_rank_suppresses_log_when_rank_does_not_match():
    logger = Mock(spec=logging.Logger)
    logger.isEnabledFor.return_value = True

    with patch("megatron.core._rank_utils.safe_get_rank", return_value=2):
        log_single_rank(logger, logging.INFO, "message", rank=3)

    logger.isEnabledFor.assert_called_once_with(logging.INFO)
    logger.log.assert_not_called()


def test_log_single_rank_forwards_format_args():
    logger = Mock(spec=logging.Logger)
    logger.isEnabledFor.return_value = True

    with patch("megatron.core._rank_utils.safe_get_rank", return_value=0):
        log_single_rank(logger, logging.INFO, "value=%s", 42)

    logger.log.assert_called_once_with(logging.INFO, "value=%s", 42)


def test_log_on_each_pipeline_stage_logs_and_forwards_arguments_on_emitter():
    logger = Mock(spec=logging.Logger)
    logger.isEnabledFor.return_value = True
    tp_group = Mock()
    tp_group.rank.return_value = 0
    dp_cp_group = Mock()
    dp_cp_group.rank.return_value = 0

    with patch("megatron.core.utils.torch.distributed.is_initialized", return_value=True):
        log_on_each_pipeline_stage(
            logger,
            logging.INFO,
            "value=%s",
            42,
            tp_group=tp_group,
            dp_cp_group=dp_cp_group,
            extra={"key": "value"},
        )

    logger.log.assert_called_once_with(logging.INFO, "value=%s", 42, extra={"key": "value"})


@pytest.mark.parametrize("tp_rank,dp_cp_rank", [(1, 0), (0, 1), (1, 1)])
def test_log_on_each_pipeline_stage_suppresses_log_on_non_emitter(tp_rank, dp_cp_rank):
    logger = Mock(spec=logging.Logger)
    logger.isEnabledFor.return_value = True
    tp_group = Mock()
    tp_group.rank.return_value = tp_rank
    dp_cp_group = Mock()
    dp_cp_group.rank.return_value = dp_cp_rank

    with patch("megatron.core.utils.torch.distributed.is_initialized", return_value=True):
        log_on_each_pipeline_stage(
            logger, logging.INFO, "message", tp_group=tp_group, dp_cp_group=dp_cp_group
        )

    logger.log.assert_not_called()
