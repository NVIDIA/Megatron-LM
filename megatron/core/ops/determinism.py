# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Whether a backend produces bit-exact results run to run.

Every backend declares ``DETERMINISM``. Three values, because "nobody has checked" is a
different statement from "verified safe" and should not be able to masquerade as it:

``DETERMINISTIC``
    Bit-exact under ``--deterministic-mode``, given the environment that mode sets up.
``NONDETERMINISTIC``
    Known not to be. Selecting it under ``--deterministic-mode`` is an error.
``UNKNOWN``
    Nobody has established either way. Usable, but ``--deterministic-mode`` warns, because
    the guarantee it advertises does not hold for that operation.

Declared as plain strings rather than an enum so that a backend module needs no import to
state it -- which is what keeps ``megatron/core/inference/ops/backends.py`` free of any
module-scope ``megatron.core.ops`` import, the invariant that stops the two packages forming
an import cycle.
"""

DETERMINISTIC = "deterministic"
NONDETERMINISTIC = "nondeterministic"
UNKNOWN = "unknown"

VALUES = frozenset({DETERMINISTIC, NONDETERMINISTIC, UNKNOWN})

__all__ = ["DETERMINISTIC", "NONDETERMINISTIC", "UNKNOWN", "VALUES"]
