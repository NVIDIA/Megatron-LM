# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Per-module determinism tests.

Bit-exact env vars are set by ``correctness/__init__.py`` at its own
import, so a future subpackage can opt out without contaminating peers.

``CUDA_DEVICE_MAX_CONNECTIONS=1`` is set here (not a determinism knob —
it's the pre-Blackwell async-TP correctness requirement asserted at
``arguments.py:1321``). The driver captures it at CUDA-context creation,
so the gate has to live at package-import time; per-cell writes are
no-ops. This setdefault IS the enforcement in the unit-test CI bucket
(``unit-tests.yaml`` doesn't export it in shell). No-op on Blackwell;
override at launcher with ``=32`` if running MoE-overlap there —
setdefault won't clobber.
"""

import os

os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
