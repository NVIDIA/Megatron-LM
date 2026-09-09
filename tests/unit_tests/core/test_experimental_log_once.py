import logging

import torch

from megatron.core import config
from megatron.core import utils as mcore_utils

# Message emitted by experimental_fn wrapper when EXPERIMENTAL flag is enabled.
_LOG_MSG = "ENABLE_EXPERIMENTAL is True, running experimental code."


def _get_test_logger():
    """Return the same logger instance used inside mcore utils."""
    return logging.getLogger(mcore_utils.__name__)


def test_experimental_fn_logs_once(caplog):
    """Ensure the experimental_fn decorator writes the enable message only once."""

    # Enable experimental features for this test.
    config.set_experimental_flag(True)

    # Define a fresh function with the decorator so it has its own closure state.
    @mcore_utils.experimental_fn(introduced_with_version="0.15.0")
    def sample():  # pragma: no cover
        return 42

    logger = _get_test_logger()

    with caplog.at_level(logging.INFO, logger=logger.name):
        # First invocation should emit the log record.
        assert sample() == 42
        # Second invocation should *not* emit the log record again.
        assert sample() == 42

    # Filter captured records originating from our logger with the expected message.
    records = [
        rec for rec in caplog.records if rec.name == logger.name and _LOG_MSG in rec.getMessage()
    ]
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            assert (
                len(records) == 1
            ), "Expected exactly one log record for experimental fn enable message"
        else:
            assert (
                len(records) == 0
            ), "Expected no log on rank != 0 for experimental fn enable message"
    else:
        assert (
            len(records) == 1
        ), "Expected exactly one log record for experimental fn enable message"

    # Reset flag so it does not leak to other tests.
    config.set_experimental_flag(False)


def test_experimental_cls_logs_once(caplog):
    """Ensure the experimental_cls decorator writes the enable message only once for classes."""

    config.set_experimental_flag(True)

    @mcore_utils.experimental_cls(introduced_with_version="0.15.0")
    class Dummy:
        def foo(self):
            return "bar"

    logger = _get_test_logger()

    with caplog.at_level(logging.INFO, logger=logger.name):
        obj = Dummy()  # Instantiation should trigger logging on attribute access later
        # Access method twice to trigger guard; first should log, second not.
        assert obj.foo() == "bar"
        assert obj.foo() == "bar"

    records = [
        rec for rec in caplog.records if rec.name == logger.name and _LOG_MSG in rec.getMessage()
    ]
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            assert (
                len(records) == 1
            ), "Expected exactly one log record for experimental cls enable message"
        else:
            assert (
                len(records) == 0
            ), "Expected no log on rank != 0 for experimental cls enable message"
    else:
        assert (
            len(records) == 1
        ), "Expected exactly one log record for experimental cls enable message"
    config.set_experimental_flag(False)
