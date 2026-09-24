# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Exceptions for explicitly unavailable optional features."""


class OptionalDependencyUnavailable(ImportError):
    """An optional feature cannot be imported because its dependencies are unavailable."""
