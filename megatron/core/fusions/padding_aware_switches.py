# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Kill-switch for the padding-aware paths (``MCORE_PADDING_AWARE_SKIP``).

The variable names the paths that are turned off: ``ce`` (row-bounded cross entropy), ``mhc``
(bounded hyper-connection kernels), ``norm`` (row-bounded RMSNorm), ``add`` (bounded three-way
gradient add of the hyper-connection), ``argmax`` (bounded MTP arg-max), ``rope``, ``arena``.

Unset, the default is ``norm,add,argmax``: those three were measured net-negative end to end
(−0.3 … −1.2 % of step time, no memory change) both with every rank full (p = 0) and with 30 %
padding on every rank, because the bound bookkeeping costs more than the skipped tail rows save;
the GEMMs, the cross entropy and the hyper-connection kernels keep their gain. Set the variable
to an empty string to engage everything; name paths explicitly to turn them off.
"""

import os

DEFAULT_SKIP = frozenset({"norm", "add", "argmax"})


def padding_aware_feature_disabled(name: str) -> bool:
    """True when the padding-aware path ``name`` is turned off (see the module docstring)."""
    value = os.environ.get("MCORE_PADDING_AWARE_SKIP")
    if value is None:
        return name in DEFAULT_SKIP
    return name in value.split(",")
