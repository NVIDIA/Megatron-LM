# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import torch
import torch.nn as nn

from megatron.lite.model.deepseek_v4.vllm import protocol


def test_post_dist_opt_model_load_invalidates_deployment_metadata() -> None:
    class CacheOwner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.weight._fp8_source_scales = torch.ones(1)
            self.weight._fp8_source_scale_version = self.weight._version
            self._fp8_source_scales_by_parameter = {"weight": torch.ones(1)}
            self.cleared = 0

        def clear_deployment_weight_cache(self) -> None:
            self.cleared += 1

    model = nn.Sequential(CacheOwner())
    model._fp8_source_scales_valid = True
    model._fp8_source_scales_by_name = {"0.weight": torch.ones(1)}

    protocol._post_dist_opt_model_load(model)

    assert model._fp8_source_scales_valid is False
    assert model._fp8_source_scales_by_name == {}
    assert not hasattr(model[0].weight, "_fp8_source_scales")
    assert model[0]._fp8_source_scales_by_parameter == {}
    assert model[0].cleared == 1
