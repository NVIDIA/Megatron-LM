# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

import megatron.core.extensions.transformer_engine_int4_fake_qat as int4_qat


def _config(gradient_accumulation_fusion: bool = False):
    return SimpleNamespace(gradient_accumulation_fusion=gradient_accumulation_fusion)


class TestTransformerEngineInt4FakeQAT:
    def test_supported_without_fused_or_fsdp_weight_attributes(self):
        weight = torch.nn.Parameter(torch.empty(4, 4))

        int4_qat._validate_int4_fake_qat_support(_config(), False, [weight])

    def test_rejects_gradient_accumulation_fusion(self):
        weight = torch.nn.Parameter(torch.empty(4, 4))

        with pytest.raises(RuntimeError, match="gradient_accumulation_fusion"):
            int4_qat._validate_int4_fake_qat_support(
                _config(gradient_accumulation_fusion=True), False, [weight]
            )

    def test_rejects_delayed_wgrad_compute(self):
        weight = torch.nn.Parameter(torch.empty(4, 4))

        with pytest.raises(RuntimeError, match="delayed wgrad"):
            int4_qat._validate_int4_fake_qat_support(_config(), True, [weight])

    def test_rejects_fsdp_weight_attributes(self):
        weight = torch.nn.Parameter(torch.empty(4, 4))
        weight.__fsdp_param__ = True
        weight.get_main_grad = lambda: torch.empty_like(weight)

        with pytest.raises(RuntimeError, match="Megatron FSDP"):
            int4_qat._validate_int4_fake_qat_support(_config(), False, [weight])
