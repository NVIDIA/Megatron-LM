# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Muon with the precision-aware optimizer.

The precision-aware flag applies to the scalar Adam next to Muon; Muon and its LayerWise
wrappers keep plain FP32 main params through a config with the flag off.

Launch: torchrun --nproc-per-node=1 -m pytest
    tests/unit_tests/optimizer/test_muon_precision_aware_optimizer.py
"""

import pytest
import torch

_MUON_CONFIG = dict(
    optimizer='muon',
    bf16=True,
    use_layer_wise_distributed_optimizer=True,
    muon_tp_mode='layer_sharded',
    muon_split_qkv=False,
)


def _config(**kwargs):
    pytest.importorskip("transformer_engine", reason="precision-aware optimizer needs TE")
    from megatron.core.optimizer import OptimizerConfig

    return OptimizerConfig(**{**_MUON_CONFIG, **kwargs})


def test_precision_aware_optimizer_accepted_with_muon():
    config = _config(use_precision_aware_optimizer=True, exp_avg_dtype=torch.bfloat16)
    plain = config.without_precision_aware_optimizer()
    assert not plain.use_precision_aware_optimizer
    assert not plain.use_precision_aware_optimizer_no_fp8_or_ds_fp8
    assert plain.exp_avg_dtype == torch.float32
    assert config.use_precision_aware_optimizer, "the copy must not touch the original"


def test_precision_aware_optimizer_rejected_with_lion_scalar_optimizer():
    with pytest.raises(AssertionError, match="scalar optimizer is adam"):
        _config(use_precision_aware_optimizer=True, muon_scalar_optimizer='lion')


def test_optimizer_cuda_graph_keeps_full_fp32_adam_master_params():
    """TE FusedAdam cannot store param remainders in capturable mode."""
    assert _config(use_precision_aware_optimizer=True).store_param_remainders
    config = _config(use_precision_aware_optimizer=True, optimizer_cuda_graph=True)
    assert not config.store_param_remainders
