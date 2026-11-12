# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""In-tree cost models. The scheduler type selects one; users never name them."""

__all__ = ['make_model', 'measured_dynamic_usp_model', 'model_class']

# Measured calibrations packaged for the DynamicUSP scheduler, by the hardware that the
# runtime file names. The training configuration must match the calibration's domain.
_MEASURED_DYNAMIC_USP = {'h100': 'qwen3_h100_p2p'}


def measured_dynamic_usp_model(hardware):
    """Name the packaged DynamicUSP calibration for ``hardware``; None selects the original."""
    if hardware is None:
        return None
    if not isinstance(hardware, str) or hardware not in _MEASURED_DYNAMIC_USP:
        raise ValueError(
            f"No DynamicUSP calibration is packaged for hardware {hardware!r}; packaged: "
            f"{sorted(_MEASURED_DYNAMIC_USP)}. Omit 'hardware' to use the original DynamicUSP "
            'profile'
        )
    return _MEASURED_DYNAMIC_USP[hardware]


def model_class(name):
    """Resolve an in-tree model name without importing the Megatron package."""
    if name == 'dcp_benchmark':
        from .benchmark import BenchmarkCostModel

        return BenchmarkCostModel
    if name == 'usp_calib_v1':
        from .usp_calib_v1 import USPCalibV1CostModel

        return USPCalibV1CostModel
    if name == 'qwen3_h100_p2p':
        from .qwen3_h100_p2p import Qwen3H100P2PCostModel

        return Qwen3H100P2PCostModel
    if name == 'qwen35_full':
        from .qwen35_full import Qwen35FullCostModel

        return Qwen35FullCostModel
    raise ValueError(f'Unknown in-tree DCP cost model: {name}')


def make_model(name, geometry, **options):
    """Construct an in-tree cost model for one planner geometry."""
    return model_class(name)(geometry, **options)
