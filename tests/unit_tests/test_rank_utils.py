# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import logging
import warnings
from contextlib import contextmanager
from unittest.mock import Mock, patch

import pytest

from megatron.core._rank_utils import (
    get_default_log_ranks,
    log_single_rank,
    set_default_log_ranks,
    warn_single_rank,
)
from megatron.core.utils import log_on_each_pipeline_stage


@pytest.fixture
def restore_default_log_ranks():
    """Put the module-level default back, so one test cannot leak into the next."""
    original = get_default_log_ranks()
    yield
    set_default_log_ranks(original)


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


def test_default_log_ranks_is_rank_zero():
    assert get_default_log_ranks() == (0,)


def test_set_default_log_ranks_sorts_and_deduplicates(restore_default_log_ranks):
    set_default_log_ranks([64, 0, 64])

    assert get_default_log_ranks() == (0, 64)


def test_log_single_rank_logs_on_every_default_rank(restore_default_log_ranks):
    # Non-colocated MIMO logs on the first language-model rank as well as rank 0.
    set_default_log_ranks([0, 64])
    logger = Mock(spec=logging.Logger)
    logger.isEnabledFor.return_value = True

    with patch("megatron.core._rank_utils.safe_get_rank", return_value=64):
        log_single_rank(logger, logging.INFO, "message")

    logger.log.assert_called_once_with(logging.INFO, "message")


def test_log_single_rank_suppresses_log_outside_the_default_ranks(restore_default_log_ranks):
    set_default_log_ranks([0, 64])
    logger = Mock(spec=logging.Logger)
    logger.isEnabledFor.return_value = True

    with patch("megatron.core._rank_utils.safe_get_rank", return_value=5):
        log_single_rank(logger, logging.INFO, "message")

    logger.log.assert_not_called()


def test_explicit_rank_overrides_the_default_ranks(restore_default_log_ranks):
    # A caller that names a rank keeps naming exactly that rank.
    set_default_log_ranks([0, 64])
    logger = Mock(spec=logging.Logger)
    logger.isEnabledFor.return_value = True

    with patch("megatron.core._rank_utils.safe_get_rank", return_value=64):
        log_single_rank(logger, logging.INFO, "message", rank=0)

    logger.log.assert_not_called()


@contextmanager
def _pretend_rank(rank):
    """Make ``warn_single_rank`` see ``rank``, whichever branch it takes.

    It reads ``torch.distributed.get_rank`` directly once distributed is up, and falls back to
    ``safe_get_rank`` only before that, so patching one alone leaves the test at the mercy of
    whether the suite runs under a launcher.
    """
    with (
        patch("megatron.core._rank_utils.torch.distributed.is_initialized", return_value=True),
        patch("megatron.core._rank_utils.torch.distributed.get_rank", return_value=rank),
        patch("megatron.core._rank_utils.safe_get_rank", return_value=rank),
    ):
        yield


def test_warn_single_rank_warns_on_every_default_rank(restore_default_log_ranks):
    set_default_log_ranks([0, 64])

    with _pretend_rank(64):
        with pytest.warns(UserWarning, match="message"):
            warn_single_rank("message")


def test_warn_single_rank_stays_quiet_outside_the_default_ranks(restore_default_log_ranks):
    set_default_log_ranks([0, 64])

    with _pretend_rank(5):
        with warnings.catch_warnings(record=True) as raised:
            warnings.simplefilter("always")
            warn_single_rank("message")

    assert raised == []


def _emit_unrelated_future_warning():
    warnings.warn("unrelated", FutureWarning, stacklevel=2)


def test_warn_single_rank_does_not_rearm_other_warnings():
    """A suppressed call must not re-arm warnings that already printed.

    ``warnings.catch_warnings`` bumps the global filter version on both enter and exit, and
    ``warn_explicit`` clears a module's ``__warningregistry__`` whenever that version moves.
    Entering it on every call therefore re-armed every other warning in the process, so a
    once-per-bucket call site made torch's deprecation notices reprint on every bucket.
    """
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("default")
        with (
            patch("megatron.core._rank_utils.torch.distributed.is_initialized", return_value=True),
            patch("megatron.core._rank_utils.torch.distributed.get_rank", return_value=5),
        ):
            for _ in range(3):
                _emit_unrelated_future_warning()
                # Rank 5 is not a default log rank, so this call itself stays silent.
                warn_single_rank("job-level message")

    assert len([w for w in seen if w.category is FutureWarning]) == 1
