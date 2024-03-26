# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import sys
from logging import LogRecord, StreamHandler

BLACKLISTED_MODULES = ["torch.distributed"]


class CustomHandler(StreamHandler):
    """
    Custom handler to filter out logging from code outside of
    Megatron Core, and dump to stdout.
    """

    def __init__(self):
        super().__init__(stream=sys.stdout)

    def filter(self, record: LogRecord) -> bool:
        # Prevent log entries that come from the blacklisted modules
        # through (e.g., PyTorch Distributed).
        for blacklisted_module in BLACKLISTED_MODULES:
            if record.name.startswith(blacklisted_module):
                return False
        return True
