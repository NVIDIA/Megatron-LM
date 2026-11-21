# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import os
from datetime import datetime

LOG_DIR = os.environ.get("LANGRL_LOG_DIR", None)
LOG_PREFIX = os.environ.get("LANGRL_LOG_PREFIX", "LANG_RL")

log_handle = None
_log_opened = False

prefix = f"{LOG_PREFIX}: "


def _open_log():
    """Open the log on first use, so importing this module has no side effects."""
    global log_handle, _log_opened
    if _log_opened:
        return
    _log_opened = True
    print(f"{LOG_PREFIX} Log directory: {LOG_DIR}")
    if LOG_DIR:
        log_handle = open(LOG_DIR + '/lang_rl.log', "w")


def log(message):
    _open_log()
    if log_handle:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_handle.write(f"[{timestamp}] {prefix}{message}\n")
        log_handle.flush()
