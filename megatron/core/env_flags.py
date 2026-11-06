# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Parsing of boolean ``MCORE_*`` environment switches.

Some switches must stay toggleable per run without a configuration change, for example the
kill switch of an optimization that keeps results bit-identical, or opt-in profiling ranges.
They are usually read once, at import. All of them share one syntax, so that a misspelled
value fails loudly instead of silently selecting another code path.
"""

import os

# Normalized (stripped, lower-case) spellings of on and off. They are public so that parsers of
# multi-valued switches accept the same on and off spellings.
TRUE_VALUES = frozenset(("1", "true", "yes", "on"))
FALSE_VALUES = frozenset(("0", "false", "no", "off"))


def env_flag(name: str, default: bool) -> bool:
    """Return the boolean value of the environment variable ``name``.

    An unset variable selects ``default``. ``1``, ``true``, ``yes`` and ``on`` enable the
    switch; ``0``, ``false``, ``no`` and ``off`` disable it. Matching ignores case and
    surrounding whitespace.

    Raises:
        ValueError: if the variable is set to any other value (including an empty string).
    """
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(
        f"Environment variable {name}={value!r} is not a boolean; "
        "use 1/true/yes/on or 0/false/no/off, or unset it for the default."
    )
