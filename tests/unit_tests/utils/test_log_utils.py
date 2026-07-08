# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Tests for log_utils module."""

import logging
import os
from unittest.mock import patch

from megatron.training.utils.log_utils import (
    add_filter_to_all_loggers,
    append_to_progress_log,
    barrier_and_log,
    module_filter,
    setup_logging,
    warning_filter,
)


def _make_record(
    name: str = "test.logger", level: int = logging.INFO, msg: str = "msg"
) -> logging.LogRecord:
    return logging.LogRecord(
        name=name, level=level, pathname=__file__, lineno=1, msg=msg, args=None, exc_info=None
    )


class TestWarningFilter:
    """Test warning_filter function."""

    def test_filters_out_warning(self):
        record = _make_record(level=logging.WARNING)
        assert warning_filter(record) is False

    def test_allows_info(self):
        record = _make_record(level=logging.INFO)
        assert warning_filter(record) is True

    def test_allows_error(self):
        record = _make_record(level=logging.ERROR)
        assert warning_filter(record) is True

    def test_allows_debug(self):
        record = _make_record(level=logging.DEBUG)
        assert warning_filter(record) is True

    def test_allows_critical(self):
        record = _make_record(level=logging.CRITICAL)
        assert warning_filter(record) is True


class TestModuleFilter:
    """Test module_filter function."""

    def test_filters_matching_prefix(self):
        record = _make_record(name="megatron.core.foo")
        assert module_filter(record, ["megatron.core"]) is False

    def test_allows_non_matching(self):
        record = _make_record(name="other.module")
        assert module_filter(record, ["megatron.core"]) is True

    def test_filters_with_multiple_prefixes(self):
        record = _make_record(name="apex.something")
        assert module_filter(record, ["megatron.core", "apex"]) is False

    def test_empty_filter_list_allows_all(self):
        record = _make_record(name="anything.here")
        assert module_filter(record, []) is True

    def test_exact_match(self):
        record = _make_record(name="megatron")
        assert module_filter(record, ["megatron"]) is False

    def test_partial_prefix_no_match(self):
        # Logger name "megatronics" does start with "megatron" so will be filtered.
        # This documents current (prefix-based) behavior.
        record = _make_record(name="megatronics")
        assert module_filter(record, ["megatron"]) is False


class TestAddFilterToAllLoggers:
    """Test add_filter_to_all_loggers function."""

    def test_adds_filter_to_root_logger(self):
        # Ensure a specific named logger exists before the call.
        logging.getLogger("test_add_filter_module_a")

        def my_filter(record):
            return True

        try:
            add_filter_to_all_loggers(my_filter)
            assert my_filter in logging.getLogger().filters
            assert my_filter in logging.getLogger("test_add_filter_module_a").filters
        finally:
            logging.getLogger().removeFilter(my_filter)
            logging.getLogger("test_add_filter_module_a").removeFilter(my_filter)


class TestSetupLogging:
    """Test setup_logging function."""

    def _restore_loggers(self, original_filters: dict[str, list], original_level: int):
        root = logging.getLogger()
        # Restore filters
        for name, filters in original_filters.items():
            lg = logging.getLogger(name) if name else root
            lg.filters = list(filters)
        root.setLevel(original_level)

    def _snapshot_filters(self) -> dict[str, list]:
        snapshot = {"": list(logging.getLogger().filters)}
        for name in list(logging.root.manager.loggerDict):
            snapshot[name] = list(logging.getLogger(name).filters)
        return snapshot

    @patch.dict(os.environ, {}, clear=True)
    def test_default_logging_level_is_info(self):
        root = logging.getLogger()
        original_level = root.level
        original_filters = self._snapshot_filters()
        try:
            setup_logging(filter_warning=False)
            assert root.level == logging.INFO
        finally:
            self._restore_loggers(original_filters, original_level)

    @patch.dict(os.environ, {}, clear=True)
    def test_explicit_logging_level(self):
        root = logging.getLogger()
        original_level = root.level
        original_filters = self._snapshot_filters()
        try:
            setup_logging(logging_level=logging.DEBUG, filter_warning=False)
            assert root.level == logging.DEBUG
        finally:
            self._restore_loggers(original_filters, original_level)

    @patch.dict(os.environ, {"MEGATRON_LOGGING_LEVEL": str(logging.ERROR)}, clear=True)
    def test_argument_overrides_env_var(self):
        # Per docstring precedence: argument > env var > default.
        root = logging.getLogger()
        original_level = root.level
        original_filters = self._snapshot_filters()
        try:
            setup_logging(logging_level=logging.DEBUG, filter_warning=False)
            assert root.level == logging.DEBUG
        finally:
            self._restore_loggers(original_filters, original_level)

    @patch.dict(os.environ, {}, clear=True)
    def test_warning_filter_added_by_default(self):
        root = logging.getLogger()
        original_level = root.level
        original_filters = self._snapshot_filters()
        try:
            setup_logging()
            assert warning_filter in root.filters
        finally:
            self._restore_loggers(original_filters, original_level)

    @patch.dict(os.environ, {}, clear=True)
    def test_filter_warning_false_does_not_add_filter(self):
        root = logging.getLogger()
        original_level = root.level
        original_filters = self._snapshot_filters()
        try:
            setup_logging(filter_warning=False)
            assert warning_filter not in root.filters
        finally:
            self._restore_loggers(original_filters, original_level)

    @patch.dict(os.environ, {}, clear=True)
    def test_module_filter_added_when_modules_provided(self):
        root = logging.getLogger()
        original_level = root.level
        original_filters = self._snapshot_filters()
        try:
            setup_logging(filter_warning=False, modules_to_filter=["some.module"])
            # A partial-bound module_filter is added — at least one new filter present.
            assert len(root.filters) > len(original_filters[""])
        finally:
            self._restore_loggers(original_filters, original_level)

    @patch.dict(os.environ, {}, clear=True)
    def test_megatron_bridge_logger_level_updated(self):
        root = logging.getLogger()
        original_level = root.level
        bridge_logger = logging.getLogger("megatron.bridge.test")
        original_bridge_level = bridge_logger.level
        original_filters = self._snapshot_filters()
        try:
            setup_logging(logging_level=logging.WARNING, filter_warning=False)
            assert bridge_logger.level == logging.WARNING
        finally:
            bridge_logger.setLevel(original_bridge_level)
            self._restore_loggers(original_filters, original_level)

    @patch.dict(os.environ, {}, clear=True)
    def test_non_bridge_logger_not_updated_by_default(self):
        root = logging.getLogger()
        original_level = root.level
        other_logger = logging.getLogger("some.other.module")
        other_logger.setLevel(logging.CRITICAL)
        original_filters = self._snapshot_filters()
        try:
            setup_logging(logging_level=logging.WARNING, filter_warning=False)
            assert other_logger.level == logging.CRITICAL
        finally:
            other_logger.setLevel(logging.NOTSET)
            self._restore_loggers(original_filters, original_level)

    @patch.dict(os.environ, {}, clear=True)
    def test_set_level_for_all_loggers(self):
        root = logging.getLogger()
        original_level = root.level
        other_logger = logging.getLogger("some.other.module2")
        original_other_level = other_logger.level
        original_filters = self._snapshot_filters()
        try:
            setup_logging(
                logging_level=logging.WARNING, filter_warning=False, set_level_for_all_loggers=True
            )
            assert other_logger.level == logging.WARNING
        finally:
            other_logger.setLevel(original_other_level)
            self._restore_loggers(original_filters, original_level)


class TestAppendToProgressLog:
    """Test append_to_progress_log function."""

    @patch("megatron.training.utils.log_utils.torch.distributed.is_initialized", return_value=False)
    @patch("megatron.training.utils.log_utils.safe_get_rank", return_value=0)
    @patch("megatron.training.utils.log_utils.safe_get_world_size", return_value=4)
    def test_writes_entry_on_rank_0(self, _ws, _rank, _is_init, tmp_path):
        save_dir = str(tmp_path)
        append_to_progress_log(save_dir, "hello world", barrier=False)
        progress_file = tmp_path / "progress.txt"
        assert progress_file.exists()
        contents = progress_file.read_text()
        assert "hello world" in contents
        assert "# GPUs: 4" in contents
        assert "Job ID:" in contents

    @patch("megatron.training.utils.log_utils.torch.distributed.is_initialized", return_value=False)
    @patch("megatron.training.utils.log_utils.safe_get_rank", return_value=1)
    @patch("megatron.training.utils.log_utils.safe_get_world_size", return_value=4)
    def test_no_write_on_non_zero_rank(self, _ws, _rank, _is_init, tmp_path):
        save_dir = str(tmp_path)
        append_to_progress_log(save_dir, "hello", barrier=False)
        assert not (tmp_path / "progress.txt").exists()

    @patch("megatron.training.utils.log_utils.torch.distributed.is_initialized", return_value=False)
    @patch("megatron.training.utils.log_utils.safe_get_rank", return_value=0)
    @patch("megatron.training.utils.log_utils.safe_get_world_size", return_value=2)
    def test_appends_multiple_entries(self, _ws, _rank, _is_init, tmp_path):
        save_dir = str(tmp_path)
        append_to_progress_log(save_dir, "first", barrier=False)
        append_to_progress_log(save_dir, "second", barrier=False)
        contents = (tmp_path / "progress.txt").read_text()
        assert "first" in contents
        assert "second" in contents
        assert contents.count("\n") == 2

    def test_none_save_dir_is_noop(self):
        # Must not raise and must not produce any file.
        append_to_progress_log(None, "anything", barrier=False)  # type: ignore[arg-type]

    @patch("megatron.training.utils.log_utils.torch.distributed.is_initialized", return_value=False)
    @patch("megatron.training.utils.log_utils.safe_get_rank", return_value=0)
    @patch("megatron.training.utils.log_utils.safe_get_world_size", return_value=1)
    @patch.dict(os.environ, {"SLURM_JOB_ID": "987654"}, clear=False)
    def test_includes_slurm_job_id(self, _ws, _rank, _is_init, tmp_path):
        save_dir = str(tmp_path)
        append_to_progress_log(save_dir, "entry", barrier=False)
        contents = (tmp_path / "progress.txt").read_text()
        assert "Job ID: 987654" in contents

    @patch("megatron.training.utils.log_utils.torch.distributed.is_initialized", return_value=True)
    @patch("megatron.training.utils.log_utils.torch.distributed.barrier")
    @patch("megatron.training.utils.log_utils.safe_get_rank", return_value=0)
    @patch("megatron.training.utils.log_utils.safe_get_world_size", return_value=2)
    def test_barrier_called_when_requested(self, _ws, _rank, mock_barrier, _is_init, tmp_path):
        save_dir = str(tmp_path)
        append_to_progress_log(save_dir, "entry", barrier=True)
        mock_barrier.assert_called_once()

    @patch("megatron.training.utils.log_utils.torch.distributed.is_initialized", return_value=True)
    @patch("megatron.training.utils.log_utils.torch.distributed.barrier")
    @patch("megatron.training.utils.log_utils.safe_get_rank", return_value=0)
    @patch("megatron.training.utils.log_utils.safe_get_world_size", return_value=2)
    def test_barrier_skipped_when_disabled(self, _ws, _rank, mock_barrier, _is_init, tmp_path):
        save_dir = str(tmp_path)
        append_to_progress_log(save_dir, "entry", barrier=False)
        mock_barrier.assert_not_called()


class TestBarrierAndLog:
    """Test barrier_and_log function."""

    @patch("megatron.training.utils.log_utils.print_rank_0")
    @patch("megatron.training.utils.log_utils.torch.distributed.is_initialized", return_value=False)
    def test_no_barrier_when_not_initialized(self, _is_init, mock_print):
        barrier_and_log("hello")
        mock_print.assert_called_once()
        msg = mock_print.call_args[0][0]
        assert "[hello]" in msg
        assert "datetime:" in msg

    @patch("megatron.training.utils.log_utils.print_rank_0")
    @patch("megatron.training.utils.log_utils.torch.distributed.barrier")
    @patch("megatron.training.utils.log_utils.torch.distributed.is_initialized", return_value=True)
    def test_barrier_called_when_initialized(self, _is_init, mock_barrier, mock_print):
        barrier_and_log("ready")
        mock_barrier.assert_called_once()
        mock_print.assert_called_once()
