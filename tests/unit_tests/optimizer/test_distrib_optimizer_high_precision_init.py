# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer


class _ParameterWithHighPrecisionInit:
    def __init__(self, value):
        self.value = value
        self.clear_count = 0

    def get_high_precision_init_val(self):
        return self.value

    def clear_high_precision_init_val(self):
        self.value = None
        self.clear_count += 1


def _config(precision_aware=False):
    return SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=precision_aware)


def test_clear_unowned_high_precision_init_values_preserves_owned_parameters():
    owned = _ParameterWithHighPrecisionInit(object())
    unowned = _ParameterWithHighPrecisionInit(object())
    already_clear = _ParameterWithHighPrecisionInit(None)
    without_high_precision_init = object()
    group_ranges = [
        {"orig_group": {"params": [owned, unowned, already_clear, without_high_precision_init]}}
    ]

    observed, cleared = DistributedOptimizer._clear_unowned_high_precision_init_values(
        group_ranges, {owned: object()}, _config()
    )

    assert observed == 2
    assert cleared == 1
    assert owned.value is not None
    assert owned.clear_count == 0
    assert unowned.value is None
    assert unowned.clear_count == 1
    assert already_clear.clear_count == 0


def test_clear_unowned_high_precision_init_values_skips_precision_aware_optimizer():
    unowned = _ParameterWithHighPrecisionInit(object())
    group_ranges = [{"orig_group": {"params": [unowned]}}]

    observed, cleared = DistributedOptimizer._clear_unowned_high_precision_init_values(
        group_ranges, {}, _config(precision_aware=True)
    )

    assert observed == 0
    assert cleared == 0
    assert unowned.value is not None
    assert unowned.clear_count == 0
