# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen benchmark snapshot: preserve the original prediction code body.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

"""Decode late-only producer observations; never infer an exact count from a residue."""

def require(condition, message):
    if not condition:
        raise ValueError(message)
