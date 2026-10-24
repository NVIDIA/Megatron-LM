# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core.transformer.moe import token_dispatcher_inference
from megatron.core.transformer.moe.token_dispatcher_inference import (
    resolve_inference_moe_dispatcher_type,
)


def _config(dispatcher_type="nvls", ep_size=2, **overrides):
    values = dict(
        inference_moe_token_dispatcher_type=dispatcher_type,
        transformer_impl="inference_optimized",
        expert_model_parallel_size=ep_size,
        batch_invariant_mode=False,
        moe_latent_size=None,
        moe_shared_expert_overlap=False,
        inference_flashinfer_mxfp8_token_capacity=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _on_device(capability):
    return (
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(
            torch.cuda,
            "get_device_properties",
            return_value=SimpleNamespace(major=capability[0], minor=capability[1]),
        ),
        mock.patch.object(torch.cuda, "get_device_name", return_value="mock GPU"),
    )


@pytest.mark.parametrize(
    "requested, capability, expected",
    [
        ("nvls", (8, 0), "nccl"),
        ("nvls", (9, 0), "nvls"),
        ("nvls", (10, 0), "nvls"),
        ("nccl", (8, 0), "nccl"),
        ("nccl", (9, 0), "nccl"),
    ],
)
def test_nvls_falls_back_to_nccl_below_hopper(requested, capability, expected):
    """NVLS multimem collectives need compute capability 9+; older GPUs get the nccl path."""
    config = _config(requested)
    patches = _on_device(capability)
    with patches[0], patches[1], patches[2]:
        assert resolve_inference_moe_dispatcher_type(config) == expected
    # The effective value is written back so every later reader agrees.
    assert config.inference_moe_token_dispatcher_type == expected


def test_fallback_rewrites_the_config_and_warns_once():
    config = _config("nvls")
    patches = _on_device((8, 0))
    with (
        patches[0],
        patches[1],
        patches[2],
        mock.patch.object(token_dispatcher_inference.logger, "warning") as warning,
    ):
        assert resolve_inference_moe_dispatcher_type(config) == "nccl"
        assert resolve_inference_moe_dispatcher_type(config) == "nccl"
    warning.assert_called_once()


def test_requested_type_is_kept_for_other_transformer_impls():
    """The setting is only read by inference_optimized MoE layers; a Transformer Engine
    model on an older GPU keeps it untouched and gets no warning or NVLS-only checks."""
    config = _config("nvls", transformer_impl="transformer_engine", batch_invariant_mode=True)
    patches = _on_device((8, 0))
    with (
        patches[0],
        patches[1],
        patches[2],
        mock.patch.object(token_dispatcher_inference.logger, "warning") as warning,
    ):
        assert resolve_inference_moe_dispatcher_type(config) == "nvls"
    assert config.inference_moe_token_dispatcher_type == "nvls"
    warning.assert_not_called()


def test_requested_type_is_kept_without_expert_parallelism():
    config = _config("nvls", ep_size=1)
    patches = _on_device((8, 0))
    with patches[0], patches[1], patches[2]:
        assert resolve_inference_moe_dispatcher_type(config) == "nvls"


def test_requested_type_is_kept_without_cuda():
    config = _config("nvls")
    with mock.patch.object(torch.cuda, "is_available", return_value=False):
        assert resolve_inference_moe_dispatcher_type(config) == "nvls"


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"batch_invariant_mode": True}, "batch_invariant_mode"),
        ({"moe_latent_size": 64, "moe_shared_expert_overlap": True}, "Shared expert overlap"),
        ({"inference_flashinfer_mxfp8_token_capacity": 1024}, "mxfp8_token_capacity"),
    ],
)
def test_nvls_only_features_raise_instead_of_silently_falling_back(overrides, match):
    config = _config("nvls", **overrides)
    patches = _on_device((8, 0))
    with patches[0], patches[1], patches[2]:
        with pytest.raises(ValueError, match=match):
            resolve_inference_moe_dispatcher_type(config)
    assert config.inference_moe_token_dispatcher_type == "nvls"
