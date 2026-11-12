# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen benchmark snapshot: preserve the original prediction code body.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

"""Export fresh, authenticated program bodies for manual dataflow review."""

def require(condition, message):
    if not condition:
        raise ValueError(message)
