# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""

Per-PE Logger with colored console and file output.



Similar to the C++ Logger implementation, provides:

- Per-PE colored console output

- Per-PE file logging

- Support for TRACE, DEBUG, INFO, SUMMARY, WARN, ERROR levels

"""

import logging
import os
from datetime import datetime
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds color codes for console output."""

    def __init__(self, fmt: str, pe_id: int, use_color: bool = True):
        super().__init__(fmt)
        self.pe_id = pe_id
        self.use_color = use_color

        # ANSI color codes matching C++ implementation
        self.colors = {
            0: "\033[31m",  # Red
            1: "\033[32m",  # Green
            2: "\033[33m",  # Yellow
            3: "\033[34m",  # Blue
            4: "\033[35m",  # Magenta
            5: "\033[36m",  # Cyan
            6: "\033[91m",  # Bright Red
            7: "\033[92m",  # Bright Green
        }
        self.reset = "\033[0m"

    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            s = datetime.fromtimestamp(record.created).strftime(datefmt)
            # For file logs, replace %f with milliseconds
            if "%f" in datefmt:
                s = s.replace("%f", f"{int(record.msecs):03d}")
        else:
            s = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
            s = f"{s}.{int(record.msecs):03d}"
        return s

    def format(self, record):
        # Save original message
        original_msg = record.msg

        if self.use_color and self.pe_id >= 0:
            color = self.colors.get(self.pe_id, "\033[37m")  # White for others
            record.msg = f"{color}{record.msg}{self.reset}"

        result = super().format(record)

        # Restore original message for other handlers
        record.msg = original_msg

        return result


class PELogger:
    """Per-PE logger with colored console and file output."""

    _logger: Optional[logging.Logger] = None
    _pe_id: int = -1
    _level: int = logging.INFO

    @classmethod
    def init(cls, pe_id: int, level: str = "INFO", logs_dir: str = "logs"):
        """
        Initialize logger for this PE.

        Args:
            pe_id: Process element ID
            level: Log level (TRACE, DEBUG, INFO, WARN, ERROR)
            logs_dir: Directory for log files
        """
        cls._pe_id = pe_id

        # Convert level string to logging level
        level_map = {
            "TRACE": logging.DEBUG - 5,  # Custom level below DEBUG
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "SUMMARY": logging.INFO,
            "WARN": logging.WARNING,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        cls._level = level_map.get(level.upper(), logging.INFO)

        # Create logs directory if it doesn't exist
        os.makedirs(logs_dir, exist_ok=True)

        # Create logger
        logger_name = f"PE_{pe_id}"
        cls._logger = logging.getLogger(logger_name)
        cls._logger.setLevel(cls._level)
        cls._logger.propagate = False

        # Remove existing handlers to avoid duplicates
        cls._logger.handlers.clear()

        # 1. Console handler with color
        console_handler = logging.StreamHandler()
        console_handler.setLevel(cls._level)
        console_format = "[PE %d] [%%(asctime)s] [%%(levelname)s] %%(message)s" % pe_id
        console_formatter = ColoredFormatter(console_format, pe_id, use_color=True)
        console_handler.setFormatter(console_formatter)
        cls._logger.addHandler(console_handler)

        # 2. File handler without color
        log_filename = os.path.join(logs_dir, f"pe_{pe_id}.log")
        file_handler = logging.FileHandler(log_filename, mode="w")
        file_handler.setLevel(cls._level)
        file_format = "[PE %d] [%%(asctime)s] [%%(levelname)s] %%(message)s" % pe_id
        file_formatter = ColoredFormatter(file_format, pe_id, use_color=False)
        file_handler.setFormatter(file_formatter)
        cls._logger.addHandler(file_handler)

    @classmethod
    def set_level(cls, level: str):
        """Set the logging level."""
        level_map = {
            "TRACE": logging.DEBUG - 5,
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "SUMMARY": logging.INFO,
            "WARN": logging.WARNING,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        cls._level = level_map.get(level.upper(), logging.INFO)
        if cls._logger:
            cls._logger.setLevel(cls._level)
            for handler in cls._logger.handlers:
                handler.setLevel(cls._level)

    @classmethod
    def trace(cls, msg: str):
        """Log at TRACE level (most detailed)."""
        if cls._logger:
            cls._logger.log(logging.DEBUG - 5, msg)

    @classmethod
    def debug(cls, msg: str):
        """Log at DEBUG level."""
        if cls._logger:
            cls._logger.debug(msg)

    @classmethod
    def info(cls, msg: str):
        """Log at INFO level."""
        if cls._logger:
            cls._logger.info(msg)

    @classmethod
    def summary(cls, msg: str):
        """Log summary information (INFO level with [SUMMARY] prefix)."""
        if cls._logger:
            cls._logger.info(f"[SUMMARY] {msg}")

    @classmethod
    def warn(cls, msg: str):
        """Log at WARNING level."""
        if cls._logger:
            cls._logger.warning(msg)

    @classmethod
    def warning(cls, msg: str):
        """Log at WARNING level (alias for warn)."""
        cls.warn(msg)

    @classmethod
    def error(cls, msg: str):
        """Log at ERROR level."""
        if cls._logger:
            cls._logger.error(msg)

    @classmethod
    def critical(cls, msg: str):
        """Log at CRITICAL level."""
        if cls._logger:
            cls._logger.critical(msg)

    @classmethod
    def shutdown(cls):
        """Shutdown the logger and flush all handlers."""
        if cls._logger:
            for handler in cls._logger.handlers:
                handler.flush()
                handler.close()
            cls._logger.handlers.clear()
            cls._logger = None
