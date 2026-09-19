# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Rank-aware log suppression for training entrypoints.

This module lives directly under the ``megatron`` namespace package and imports
only the standard library, both deliberately. ``megatron.core`` and
``megatron.training`` import torch at the top of their ``__init__``, so a helper
under either would pull torch in before the caller could install any filter, and
torch raises its own deprecations while it is being imported. Keep it that way:
adding an import here that reaches torch silently stops the suppression working,
which ``tests/unit_tests/test_rank_log_setup.py`` checks for.
"""

import logging
import os
import warnings


def suppress_duplicate_logs_off_rank0() -> None:
    """Quiet warnings and torch's pytree notices on every rank but rank zero.

    Deprecation and experimental-API notices describe the job rather than the
    rank that raised them, so every rank emits the same handful and a job with
    hundreds of ranks repeats each one hundreds of times into a shared log. Rank
    0 still reports all of them, so nothing distinct is lost. Warnings that do
    differ by rank are worth giving up here: a rank-specific problem surfaces as
    a traceback, not as a warning.

    Call this before importing torch. A filter installed afterwards cannot reach
    the deprecations torch raises while it is being imported.
    """
    if int(os.environ.get('RANK', 0)) == 0:
        return

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Some libraries (e.g., CUTLASS DSL) use warnings.catch_warnings() with
    # simplefilter("always"), which overrides the filters above. Override
    # showwarning as a fallback to suppress warnings that slip through.
    original_showwarning = warnings.showwarning

    def _rank0_only_showwarning(message, category, filename, lineno, file=None, line=None):
        if issubclass(category, (UserWarning, FutureWarning, DeprecationWarning)):
            return
        original_showwarning(message, category, filename, lineno, file, line)

    warnings.showwarning = _rank0_only_showwarning

    # The pytree notices go through logging rather than the warnings module, so
    # the filters above never see them. Set the emitting logger itself, not the
    # "torch" parent: torch resets the parent's level while it is being imported
    # and would discard this, but it leaves child loggers alone.
    logging.getLogger("torch.utils._pytree").setLevel(logging.ERROR)
