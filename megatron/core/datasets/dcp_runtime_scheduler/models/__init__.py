# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Cost-model factories, independent of the selected scheduler."""

from .attention.model import AttentionCostModel, CorePredictor
from .attention.predictor import load_parameters

__all__ = ['AttentionCostModel', 'CorePredictor', 'make_model', 'qwen35_new_core_model']


def qwen35_new_core_model(*, fa_remainder=None, non_attention=None, kernel_backend='python'):
    """The exact a55 Qwen3.5 model, with its paired frozen inputs in this package."""
    from .qwen35 import Qwen35CoreCostModel

    if fa_remainder is None:
        fa_remainder = load_parameters('qwen35.json')[0]['fa_remainder']
    predictor = CorePredictor(
        query_heads=32, kv_heads=2, head_dim=256, kernel_backend=kernel_backend
    )
    return Qwen35CoreCostModel(predictor, fa_remainder=fa_remainder, non_attention=non_attention)


def make_model(name, geometry, **options):
    """Resolve an in-tree model name without importing the Megatron package."""
    if name == 'calibrated_attention':
        return AttentionCostModel(geometry, **options)
    if name == 'usp_calib_v1':
        from .usp_calib_v1 import USPCalibV1CostModel

        return USPCalibV1CostModel(geometry, **options)
    if name in ('qwen35_gb200', 'dcp_scheduler.adapters:qwen35_new_core_model'):
        return qwen35_new_core_model(**options)
    raise ValueError(f'Unknown in-tree DCP cost model: {name}')
