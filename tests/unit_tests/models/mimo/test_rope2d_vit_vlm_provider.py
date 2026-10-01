# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the 2D-RoPE ViT VLM MIMO provider."""

import argparse
from types import SimpleNamespace

import pytest
import torch

from examples.mimo.model_providers import resolve_provider, rope2d_vit_vlm
from examples.mimo.model_providers.rope2d_vit_vlm import (
    ROPE2D_VIT_VLM_MODEL_PROVIDER,
    Rope2dViTModel,
    add_rope2d_vit_args,
    build_rope2d_vit_vlm_communicator,
)
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.vision.vit_model import ViTModel
from megatron.core.transformer.enums import AttnBackend

ENCODER_NAME = "test_vit_encoder"
LANGUAGE_HIDDEN_SIZE = 256
VISION_SIZES = {
    "mimo_vision_num_layers": 2,
    "mimo_vision_hidden_size": 64,
    "mimo_vision_ffn_hidden_size": 128,
    "mimo_vision_num_attention_heads": 4,
}
NON_DEFAULT_ARGS = {
    "mimo_vision_num_query_groups": 2,
    "mimo_vision_kv_channels": 32,
    "mimo_vision_normalization": "RMSNorm",
    "mimo_vision_norm_epsilon": 1e-6,
    "mimo_vision_swiglu": True,
    "mimo_vision_disable_bias_linear": True,
    "mimo_vision_add_qkv_bias": True,
    "mimo_vision_rotary_interleaved": True,
    "mimo_vision_disable_ln_pre": True,
    "mimo_vision_spatial_merge_size": 2,
    "pixel_shuffle": True,
    "mimo_vision_projector_activation": "fast_gelu",
    "mimo_vision_encoder_attention_backend": AttnBackend.flash,
    "mimo_vision_encoder_flash_attention_version": 4,
}
# (vision config, ViT params, projection config) expected from the args above.
DEFAULT_EXPECTED = (
    {
        "num_query_groups": 4,
        "kv_channels": 16,
        "normalization": "LayerNorm",
        "layernorm_epsilon": 1e-5,
        "activation_func": torch.nn.functional.gelu,
        "gated_linear_unit": False,
        "add_bias_linear": True,
        "add_qkv_bias": False,
        "rotary_interleaved": False,
        "attention_backend": AttnBackend.unfused,
        "flash_attention_version": 2,
    },
    {"ln_pre": True, "use_merger": False},
    {"activation_func": torch.nn.functional.gelu, "add_bias_linear": True},
)
NON_DEFAULT_EXPECTED = (
    {
        "num_query_groups": 2,
        "kv_channels": 32,
        "normalization": "RMSNorm",
        "layernorm_epsilon": 1e-6,
        "activation_func": torch.nn.functional.silu,
        "gated_linear_unit": True,
        "add_bias_linear": False,
        "add_qkv_bias": True,
        "rotary_interleaved": True,
        "attention_backend": AttnBackend.flash,
        "flash_attention_version": 4,
    },
    {"ln_pre": False, "use_merger": True, "spatial_merge_size": 2},
    {"activation_func": rope2d_vit_vlm._unfused_fast_gelu, "add_bias_linear": False},
)


def _provider_args(**overrides):
    parser = argparse.ArgumentParser()
    add_rope2d_vit_args(parser)
    values = {
        **vars(parser.parse_args([])),
        **VISION_SIZES,
        "bf16": True,
        "fp16": False,
        "hidden_size": LANGUAGE_HIDDEN_SIZE,
        "image_token_id": 10,
        "img_h": 224,
        "img_w": 224,
        "patch_dim": 16,
        "pixel_shuffle": False,
        "model_provider": ROPE2D_VIT_VLM_MODEL_PROVIDER,
        "mimo_vision_encoder_name": ENCODER_NAME,
        "mimo_bridge_skip_shape_exchange": False,
        "mimo_run_input_projections_on_llm_ranks": False,
        "mimo_vision_encoder_attention_backend": None,
        "mimo_vision_encoder_flash_attention_version": None,
        "params_dtype": torch.bfloat16,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _encoder_spec(monkeypatch, **arg_overrides):
    # The language model's CP, attention backend, and activation must not leak into the ViT.
    base_config = dict(
        activation_func=torch.nn.functional.relu,
        attention_backend=AttnBackend.unfused,
        context_parallel_size=2,
        flash_attention_version=2,
        expert_tensor_parallel_size=None,
        fp32_residual_connection=False,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        recompute_modules=None,
    )
    monkeypatch.setattr(
        rope2d_vit_vlm, "_base_config", lambda _args: SimpleNamespace(**base_config)
    )
    args = _provider_args(**arg_overrides)
    provider = resolve_provider(args)
    return provider, args, provider.encoder_specs[ENCODER_NAME](args, None, None)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [({}, DEFAULT_EXPECTED), (NON_DEFAULT_ARGS, NON_DEFAULT_EXPECTED)],
    ids=["defaults", "every-arg-set"],
)
def test_vision_tower_and_projector_follow_args(monkeypatch, overrides, expected):
    provider, args, spec = _encoder_spec(monkeypatch, **overrides)
    encoder = spec.submodules["encoders"][ENCODER_NAME]
    (projection,) = spec.submodules["input_projections"]
    vision_config = encoder.params["transformer_config"]
    projection_config = projection.params["config"]
    expected_vision, expected_vit, expected_projection = expected

    assert provider.encoder_module_names == (ENCODER_NAME,)
    assert provider.special_token_ids(args) == {ENCODER_NAME: 10}
    assert encoder.module is Rope2dViTModel
    assert encoder.params["pos_emb_type"] == "rope2d"
    sizes = ("num_layers", "hidden_size", "ffn_hidden_size", "num_attention_heads")
    assert tuple(getattr(vision_config, name) for name in sizes) == tuple(VISION_SIZES.values())
    for name, value in expected_vision.items():
        assert getattr(vision_config, name) == value, name
    for name, value in expected_vit.items():
        assert encoder.params[name] == value, name

    # Two-layer MLP from the ViT width to the language hidden size.
    assert projection.params["input_size"] == VISION_SIZES["mimo_vision_hidden_size"]
    assert projection_config.hidden_size == LANGUAGE_HIDDEN_SIZE
    assert projection_config.ffn_hidden_size == LANGUAGE_HIDDEN_SIZE
    assert projection_config.gated_linear_unit is False
    for name, value in expected_projection.items():
        assert getattr(projection_config, name) == value, name

    for config in (vision_config, projection_config):
        assert config.context_parallel_size == 1
        assert config.gtp_weight_remat_size == config.expert_gtp_weight_remat_size == 1


def test_vit_accepts_mimo_encoder_inputs(monkeypatch):
    """MIMO batches name the image tensor x and move imgs_sizes to CUDA."""
    captured = {}

    def fake_forward(self, pixel_values, imgs_sizes=None, packed_seq_params=None):
        captured.update(pixel_values=pixel_values, imgs_sizes=imgs_sizes)
        return pixel_values

    monkeypatch.setattr(ViTModel, "forward", fake_forward)
    model = Rope2dViTModel.__new__(Rope2dViTModel)
    torch.nn.Module.__init__(model)
    x, imgs_sizes = torch.zeros(1, 4, 768), torch.tensor([[32, 32]])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    assert model(x=x, imgs_sizes=imgs_sizes.to(device)) is x
    assert captured["pixel_values"] is x
    assert not captured["imgs_sizes"].is_cuda
    assert torch.equal(captured["imgs_sizes"], imgs_sizes)


def test_bridge_recv_shape_counts_image_tokens(monkeypatch):
    captured = {}
    language_config = SimpleNamespace(hidden_size=LANGUAGE_HIDDEN_SIZE, params_dtype=torch.bfloat16)
    monkeypatch.setattr(
        rope2d_vit_vlm,
        "language_model_spec",
        lambda *_args: SimpleNamespace(params={"config": language_config}),
    )
    monkeypatch.setattr(
        rope2d_vit_vlm, "MultiModulePipelineCommunicator", lambda *_a, **kw: captured.update(kw)
    )
    topology = SimpleNamespace(grids={ENCODER_NAME: object(), MIMO_LANGUAGE_MODULE_KEY: object()})

    build_rope2d_vit_vlm_communicator(
        _provider_args(mimo_bridge_skip_shape_exchange=True), topology, encoder_name=ENCODER_NAME
    )

    # A real batch carries only input_ids; image tokens there size the receive buffer.
    batch = {"input_ids": torch.tensor([[1, 10, 10, 2], [10, 3, 4, 5]])}
    assert captured["bridge_recv_shape_fns"][ENCODER_NAME](batch) == (3, LANGUAGE_HIDDEN_SIZE)
    assert captured["bridge_comm_dtypes"] == {ENCODER_NAME: torch.bfloat16}
