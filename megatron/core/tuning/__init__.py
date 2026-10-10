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

    install(AutotunePolicy(mode="pinned", verify_every=10))

Megatron's training initializer installs the policy built from the
``--triton-autotune-*`` arguments, or the ``triton_autotune`` YAML section,
before any kernel runs. Without an explicit mode, a recording path, the
``deterministic`` argument or PyTorch's deterministic flag determines it.

Recording a table for a new architecture::

    torchrun ... pretrain_gpt.py ... --triton-autotune-record-path /tmp/rec
    python -m megatron.core.tuning merge /tmp/rec.rank*.json -o ~/.mcore/tuning/sm103.json
    torchrun ... --triton-autotune-mode pinned --triton-autotune-table-path ~/.mcore/tuning

Seeing what a run actually did::

    --triton-autotune-enumerate-autotuners  # every multi-config autotuner reached
    --triton-autotune-verify-every 1        # compare observed choices at each step
"""

from megatron.core.tuning.interception import (
    active_policy,
    choice_digest,
    choice_log,
    install,
    maybe_verify_choices,
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
    "maybe_verify_choices",
    "set_deterministic_mode",
    "use_deterministic_mode",
    "verify_choices",
]
