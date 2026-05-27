# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import pytest

from megatron.training.config import CheckpointConfig


@pytest.mark.parametrize("ckpt_format", ["torch_dcp", "fsdp_dtensor"])
def test_checkpoint_config_falls_back_to_mcore_without_nvrx_async_support(ckpt_format, monkeypatch):
    monkeypatch.setattr(
        "megatron.training.utils.has_nvrx_checkpointing_async_support", lambda: False
    )

    config = CheckpointConfig(async_save=True, async_strategy="nvrx", ckpt_format=ckpt_format)

    assert config.async_strategy == "mcore"


@pytest.mark.parametrize("ckpt_format", ["torch_dcp", "fsdp_dtensor"])
def test_checkpoint_config_keeps_nvrx_with_nvrx_async_support(ckpt_format, monkeypatch):
    monkeypatch.setattr(
        "megatron.training.utils.has_nvrx_checkpointing_async_support", lambda: True
    )

    config = CheckpointConfig(async_save=True, async_strategy="nvrx", ckpt_format=ckpt_format)

    assert config.async_strategy == "nvrx"
