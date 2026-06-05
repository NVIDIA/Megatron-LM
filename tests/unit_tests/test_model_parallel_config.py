# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.core.model_parallel_config import ModelParallelConfig


def test_te_cross_entropy_loss_fusion_is_disabled():
    with pytest.raises(AssertionError, match="Transformer Engine cross entropy loss fusion"):
        ModelParallelConfig(cross_entropy_loss_fusion=True, cross_entropy_fusion_impl='te')


def test_native_cross_entropy_loss_fusion_is_allowed():
    config = ModelParallelConfig(cross_entropy_loss_fusion=True, cross_entropy_fusion_impl='native')

    assert config.cross_entropy_loss_fusion
    assert config.cross_entropy_fusion_impl == 'native'
