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

import abc
import logging
import os
import warnings

# Module and name of ProcessGroupFallbackWarning. Its module imports torch, so
# the class is matched by name instead of imported.
_PROCESS_GROUP_FALLBACK_WARNING = (
    "megatron.core.process_groups_config",
    "ProcessGroupFallbackWarning",
)


class _ProcessGroupFallbackWarning(Warning, metaclass=abc.ABCMeta):
    """Warning category that matches ProcessGroupFallbackWarning and its subclasses.

    issubclass() consults __subclasshook__, so a warnings filter on this
    category, and the check in the showwarning override below, match the real
    class without importing it.
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        return any(
            (klass.__module__, klass.__qualname__) == _PROCESS_GROUP_FALLBACK_WARNING
            for klass in subclass.__mro__
        )


def suppress_duplicate_logs_off_rank0() -> None:
    """Quiet warnings and torch's pytree notices on every rank but rank zero.

    Deprecation and experimental-API notices describe the job rather than the
    rank that raised them, so every rank emits the same handful and a job with
    hundreds of ranks repeats each one hundreds of times into a shared log. Rank
    0 still reports all of them, so nothing distinct is lost. Warnings that do
    differ by rank are worth giving up here: a rank-specific problem surfaces as
    a traceback, not as a warning. The exception is ProcessGroupFallbackWarning,
    which every rank still reports, because ranks can fall back to the global
    process groups differently, for example on different pipeline stages.

    Filters installed before this call keep precedence over the ones it adds, so
    an ``error`` filter still raises on every rank.

    Call this before importing torch. A filter installed afterwards cannot reach
    the deprecations torch raises while it is being imported.
    """
    if int(os.environ.get('RANK', 0)) == 0:
        return

    # Append, so that filters installed earlier keep precedence. An "error"
    # filter from the user, a strict mode or pytest's -W must raise here too:
    # rank 0 raises either way, so a rank that ignored the warning would wait
    # for rank 0 at its next collective. Filters installed after this call
    # precede these anyway. The first filter keeps ProcessGroupFallbackWarning,
    # a FutureWarning, out of the ignore filters after it.
    warnings.filterwarnings("default", category=_ProcessGroupFallbackWarning, append=True)
    warnings.filterwarnings("ignore", category=UserWarning, append=True)
    warnings.filterwarnings("ignore", category=FutureWarning, append=True)
    warnings.filterwarnings("ignore", category=DeprecationWarning, append=True)

    # Some libraries (e.g., CUTLASS DSL) use warnings.catch_warnings() with
    # simplefilter("always"), which overrides the filters above. Override
    # showwarning as a fallback to suppress warnings that slip through.
    original_showwarning = warnings.showwarning

    def _rank0_only_showwarning(message, category, filename, lineno, file=None, line=None):
        if issubclass(
            category, (UserWarning, FutureWarning, DeprecationWarning)
        ) and not issubclass(category, _ProcessGroupFallbackWarning):
            return
        original_showwarning(message, category, filename, lineno, file, line)

    warnings.showwarning = _rank0_only_showwarning

    # The pytree notices go through logging rather than the warnings module, so
    # the filters above never see them. Set the emitting logger itself, not the
    # "torch" parent: torch resets the parent's level while it is being imported
    # and would discard this, but it leaves child loggers alone.
    logging.getLogger("torch.utils._pytree").setLevel(logging.ERROR)
