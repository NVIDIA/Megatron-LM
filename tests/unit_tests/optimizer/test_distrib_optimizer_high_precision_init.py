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


def test_clear_high_precision_init_values_clears_all_parameters():
    first = _ParameterWithHighPrecisionInit(object())
    second = _ParameterWithHighPrecisionInit(object())
    already_clear = _ParameterWithHighPrecisionInit(None)
    without_high_precision_init = object()
    model_params = [first, second, already_clear, without_high_precision_init]

    cleared = DistributedOptimizer._clear_high_precision_init_values(model_params, _config())

    assert cleared == 2
    assert first.value is None
    assert first.clear_count == 1
    assert second.value is None
    assert second.clear_count == 1
    assert already_clear.clear_count == 0


def test_clear_high_precision_init_values_skips_precision_aware_optimizer():
    model_param = _ParameterWithHighPrecisionInit(object())

    cleared = DistributedOptimizer._clear_high_precision_init_values(
        [model_param], _config(precision_aware=True)
    )

    assert cleared == 0
    assert model_param.value is not None
    assert model_param.clear_count == 0
