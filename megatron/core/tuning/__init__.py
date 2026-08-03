# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Triton autotune policy: pin the config, and pre-tune it when you can.

Triton chooses a kernel config by benchmarking candidates at first call. The
winner is a property of the machine at that instant, and it fixes ``BLOCK_SIZE``,
``num_warps`` and ``num_stages`` — which is to say it fixes how a reduction is
tiled, and therefore the floating-point accumulation order. Two identical runs
can pick differently and produce different numbers.

Pinning replaces the candidate list with one entry chosen by a rule that never
looks at a clock. Any such rule works; the tuned table only decides whether the
fixed choice is also the fast one.

Typical use::

    from megatron.core.tuning import AutotunePolicy, install

    install(AutotunePolicy.from_env())      # once, during framework init

Recording a table for a new architecture::

    MCORE_AUTOTUNE_RECORD=/tmp/rec  torchrun ... pretrain.py ...
    python -m megatron.core.tuning merge /tmp/rec/*.json -o ~/.mcore/tuning/sm103.json
    MCORE_AUTOTUNE_TABLE_PATH=~/.mcore/tuning  torchrun ... pretrain.py ...

Seeing what a run actually did::

    MCORE_AUTOTUNE_ENUMERATE=1 ...     # every multi-config autotuner reached
    MCORE_AUTOTUNE_VERIFY=1 ...        # assert all ranks chose alike, each step
"""

from megatron.core.tuning.interception import (
    active_policy,
    choice_digest,
    choice_log,
    install,
    install_from_env,
    verify_choices,
)
from megatron.core.tuning.policy import (
    AutotunePolicy,
    set_deterministic_mode,
    use_deterministic_mode,
)
from megatron.core.tuning.selection import autotune_configs

__all__ = [
    "AutotunePolicy",
    "active_policy",
    "autotune_configs",
    "choice_digest",
    "choice_log",
    "install",
    "install_from_env",
    "set_deterministic_mode",
    "use_deterministic_mode",
    "verify_choices",
]
