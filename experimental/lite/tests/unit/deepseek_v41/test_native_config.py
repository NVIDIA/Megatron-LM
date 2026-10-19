# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Configuration and unsupported candidate/CP combinations fail without allocation."""
import pytest

from megatron.lite.model.deepseek_v41.config import DeepseekV41Config
from megatron.lite.model.deepseek_v41.lite import protocol
from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
from megatron.lite.runtime.backends.mlite.runtime import _build_impl_cfg


@pytest.mark.parametrize("deterministic", [False, True])
def test_native_core_configuration_is_bf16_and_keeps_literal_ratios(deterministic):

    runtime = MegatronLiteConfig(hf_path="release", load_hf_weights=deterministic)
    assert _build_impl_cfg(protocol, runtime).initialize_weights is not deterministic
    with pytest.raises(ValueError, match="at most 40 text layers"):
        protocol._validate(DeepseekV41Config(num_hidden_layers=43), protocol.ImplConfig())
