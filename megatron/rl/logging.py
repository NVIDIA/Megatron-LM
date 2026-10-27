# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import os
from datetime import datetime

LOG_DIR = os.environ.get("LANGRL_LOG_DIR", None)
LOG_PREFIX = os.environ.get("LANGRL_LOG_PREFIX", "LANG_RL")

print(f"{LOG_PREFIX} Log directory: {LOG_DIR}")

log_handle = None
if LOG_DIR:
    log_handle = open(LOG_DIR + '/lang_rl.log', "w")

prefix = f"{LOG_PREFIX}: "


def log(message):
    if log_handle:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_handle.write(f"[{timestamp}] {prefix}{message}\n")
        log_handle.flush()
