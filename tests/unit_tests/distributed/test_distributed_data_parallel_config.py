# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest

from megatron.core.distributed import DistributedDataParallelConfig


class TestDistributedDataParallelConfig:

    def test_overwrite_main_grad_requirements(self):
        DistributedDataParallelConfig(overwrite_main_grad=True)
        for kwargs in (
            {'use_megatron_fsdp': True},
            {'use_custom_fsdp': True},
            {'reuse_grad_buf_for_mxfp8_param_ag': True, 'fp8_param_gather': True},
            {'param_name_patterns_for_fp32_local_accumulation': ('all',)},
        ):
            with pytest.raises(ValueError, match="overwrite_main_grad requires"):
                DistributedDataParallelConfig(overwrite_main_grad=True, **kwargs)
