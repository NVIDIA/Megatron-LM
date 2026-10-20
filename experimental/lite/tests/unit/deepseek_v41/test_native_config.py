# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Configuration and unsupported candidate/CP combinations fail without allocation."""
import pytest

from megatron.lite.model.deepseek_v41.config import DeepseekV41Config
from megatron.lite.model.deepseek_v41.lite import protocol
from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
from megatron.lite.runtime.backends.mlite.runtime import _build_impl_cfg


@pytest.mark.parametrize("deterministic", [False, True])
def test_release_initialization_and_unsupported_topology(deterministic):

    runtime = MegatronLiteConfig(hf_path="release", load_hf_weights=deterministic)
    assert _build_impl_cfg(protocol, runtime).initialize_weights is not deterministic
    with pytest.raises(ValueError, match="at most 40 text layers"):
        protocol._validate(DeepseekV41Config(num_hidden_layers=43), protocol.ImplConfig())


def test_rollout_import_guard_is_active():
    import importlib
    import sys

    importlib.import_module("_ds41_fixture")
    with pytest.raises(AssertionError, match="vllm"):
        importlib.import_module("vllm")
    assert not any(n == "vllm" or n.startswith("vllm.") for n in sys.modules)
