# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Focused tests for the 2D-RoPE ViT VLM MIMO provider."""

import argparse
from types import SimpleNamespace

import pytest
import torch

from examples.mimo.model_providers import resolve_provider
from examples.mimo.model_providers.nemotron_moe_vlm import (
    NEMOTRON_MODEL_PROVIDER,
    add_model_provider_args,
)
from examples.mimo.model_providers.radio_encoder import RADIO_ENCODER_MODULE_NAME
from examples.mimo.model_providers.rope2d_vit_vlm import (
    ROPE2D_VIT_VLM_MODEL_PROVIDER,
    Rope2dViTModel,
    build_rope2d_vit_vlm_communicator,
)
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.models.vision.vit_model import ViTModel
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig

ENCODER_NAME = "test_vit_encoder"
LANGUAGE_HIDDEN_SIZE = 256
VISION_SIZES = {
    "mimo_vision_num_layers": 2,
    "mimo_vision_hidden_size": 64,
    "mimo_vision_ffn_hidden_size": 128,
    "mimo_vision_num_attention_heads": 4,
}


def _provider_args(**overrides):
    values = {
        "bf16": True,
        "fp16": False,
        "hidden_size": LANGUAGE_HIDDEN_SIZE,
        "image_token_id": 10,
        "img_h": 224,
        "img_w": 224,
        "patch_dim": 16,
        "model_provider": ROPE2D_VIT_VLM_MODEL_PROVIDER,
        "mimo_vision_encoder_name": ENCODER_NAME,
        **VISION_SIZES,
        "mimo_bridge_skip_shape_exchange": False,
        "mimo_run_input_projections_on_llm_ranks": False,
        "mimo_vision_encoder_attention_backend": None,
        "mimo_vision_encoder_flash_attention_version": None,
        "params_dtype": torch.bfloat16,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _provider_base_config(**overrides):
    values = {
        "activation_func": torch.nn.functional.gelu,
        "attention_backend": AttnBackend.unfused,
        # The language model's CP; the vision tower and projection must not inherit it.
        "context_parallel_size": 2,
        "flash_attention_version": 2,
        "expert_tensor_parallel_size": None,
        "fp32_residual_connection": False,
        "params_dtype": torch.bfloat16,
        "pipeline_dtype": torch.bfloat16,
        # Read by _make_dense_non_hybrid when it strips language-only recompute modules.
        "recompute_modules": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _assert_gtp_disabled(config, tp_size=1):
    assert config.gtp_weight_remat_size == 1
    assert config.tensor_parallel_num_weight_shards == tp_size
    assert config.expert_gtp_weight_remat_size == 1
    assert config.expert_tensor_parallel_num_weight_shards == tp_size


def _encoder_spec(monkeypatch, **arg_overrides):
    from examples.mimo.model_providers import rope2d_vit_vlm

    monkeypatch.setattr(rope2d_vit_vlm, "_base_config", lambda _args: _provider_base_config())
    args = _provider_args(**arg_overrides)
    provider = resolve_provider(args)
    return provider, args, provider.encoder_specs[ENCODER_NAME](args, None, None)


@pytest.mark.parametrize(
    ("encoder_backend", "encoder_flash_version"),
    ((None, None), (AttnBackend.flash, 4), (AttnBackend.fused, None)),
)
def test_vision_tower_and_projector(monkeypatch, encoder_backend, encoder_flash_version):
    from examples.mimo.model_providers import rope2d_vit_vlm

    provider, args, spec = _encoder_spec(
        monkeypatch,
        mimo_vision_encoder_attention_backend=encoder_backend,
        mimo_vision_encoder_flash_attention_version=encoder_flash_version,
    )
    encoder = spec.submodules["encoders"][ENCODER_NAME]
    vision_config = encoder.params["transformer_config"]
    (projection,) = spec.submodules["input_projections"]
    projection_config = projection.params["config"]

    assert provider.encoder_module_names == (ENCODER_NAME,)
    assert provider.special_token_ids(args) == {ENCODER_NAME: 10}
    assert encoder.module is Rope2dViTModel
    assert issubclass(encoder.module, ViTModel)
    assert (
        vision_config.num_layers,
        vision_config.hidden_size,
        vision_config.ffn_hidden_size,
        vision_config.num_attention_heads,
    ) == tuple(VISION_SIZES.values())
    # MHA with the head dim derived from the hidden size.
    assert vision_config.num_query_groups == 4
    assert vision_config.kv_channels == 16
    assert vision_config.normalization == "RMSNorm"
    assert vision_config.activation_func is torch.nn.functional.silu
    assert vision_config.gated_linear_unit is True
    assert vision_config.add_bias_linear is False
    assert vision_config.add_qkv_bias is False
    assert vision_config.rotary_interleaved is True
    assert vision_config.attention_backend is (encoder_backend or AttnBackend.unfused)
    assert vision_config.flash_attention_version == (encoder_flash_version or 2)
    assert encoder.params["pos_emb_type"] == "rope2d"
    assert encoder.params["patch_dim"] == args.patch_dim
    assert encoder.params["add_class_token"] is False
    assert encoder.params["ln_pre"] is True
    assert encoder.params["use_merger"] is True
    assert encoder.params["spatial_merge_size"] == 2

    # Two-layer MLP from the ViT width to the language hidden size.
    assert projection.module is MultimodalProjector
    assert projection.params["projector_type"] == "mlp"
    assert projection.params["input_size"] == VISION_SIZES["mimo_vision_hidden_size"]
    assert projection_config.hidden_size == LANGUAGE_HIDDEN_SIZE
    assert projection_config.ffn_hidden_size == LANGUAGE_HIDDEN_SIZE
    assert projection_config.activation_func is rope2d_vit_vlm._unfused_fast_gelu
    activation_input = torch.tensor([-2.0, -0.5, 0.0, 1.0, 3.0])
    torch.testing.assert_close(
        projection_config.activation_func(activation_input),
        torch.nn.functional.gelu(activation_input, approximate="tanh"),
    )
    assert projection_config.gated_linear_unit is False
    assert projection_config.add_bias_linear is False
    assert vision_config.context_parallel_size == 1
    assert projection_config.context_parallel_size == 1
    _assert_gtp_disabled(vision_config)
    _assert_gtp_disabled(projection_config)


@pytest.mark.parametrize("missing", sorted(VISION_SIZES))
def test_provider_requires_vision_sizes(missing):
    flag = "--" + missing.replace("_", "-")
    with pytest.raises(ValueError, match=flag):
        resolve_provider(_provider_args(**{missing: None}))


def test_rejects_heads_that_do_not_divide_hidden_size():
    with pytest.raises(ValueError, match="divisible"):
        resolve_provider(_provider_args(mimo_vision_num_attention_heads=5))


def test_rejects_projection_on_language_ranks(monkeypatch):
    with pytest.raises(ValueError, match="encoder ranks"):
        _encoder_spec(monkeypatch, mimo_run_input_projections_on_llm_ranks=True)


def test_vit_keeps_encoder_grid_tp_group(monkeypatch):
    """sharded_state_dict must use the encoder grid's TP group, not the global one."""
    from megatron.core.models.vision import vit_model

    class StubTransformerBlock(torch.nn.Module):
        def __init__(self, **_kwargs):
            super().__init__()

    monkeypatch.setattr(vit_model, "TransformerBlock", StubTransformerBlock)
    tp_group = object()
    model = Rope2dViTModel(
        transformer_config=TransformerConfig(num_layers=1, hidden_size=16, num_attention_heads=4),
        transformer_layer_spec=object(),
        ln_pre=False,
        pos_emb_type="none",
        pg_collection=SimpleNamespace(tp=tp_group),
    )

    assert model.tp_group is tp_group


def test_vit_accepts_mimo_encoder_inputs(monkeypatch):
    """MIMO batches name the image tensor x; ViTModel names it pixel_values."""
    captured = {}

    def fake_forward(
        self, pixel_values, attention_mask=None, imgs_sizes=None, packed_seq_params=None
    ):
        captured.update(pixel_values=pixel_values, imgs_sizes=imgs_sizes)
        return pixel_values

    monkeypatch.setattr(ViTModel, "forward", fake_forward)
    model = Rope2dViTModel.__new__(Rope2dViTModel)
    torch.nn.Module.__init__(model)
    x, imgs_sizes = torch.zeros(1, 4, 768), torch.tensor([[32, 32]])

    assert model(x=x, imgs_sizes=imgs_sizes) is x
    assert captured["pixel_values"] is x and captured["imgs_sizes"] is imgs_sizes

    if torch.cuda.is_available():
        # MIMO moves the batch to CUDA; ViTModel requires host-side imgs_sizes.
        model(x=x, imgs_sizes=imgs_sizes.cuda())
        assert not captured["imgs_sizes"].is_cuda
        assert torch.equal(captured["imgs_sizes"], imgs_sizes)


@pytest.mark.parametrize("skip_shape_exchange", [False, True])
def test_build_communicator_wires_bridge_contract(monkeypatch, skip_shape_exchange):
    from examples.mimo.model_providers import rope2d_vit_vlm

    captured = {}
    communicator = object()
    topology = SimpleNamespace(grids={ENCODER_NAME: object(), MIMO_LANGUAGE_MODULE_KEY: object()})
    language_config = SimpleNamespace(
        hidden_size=LANGUAGE_HIDDEN_SIZE, params_dtype=torch.bfloat16
    )
    monkeypatch.setattr(
        rope2d_vit_vlm,
        "language_model_spec",
        lambda args, pg_collection, grid: SimpleNamespace(params={"config": language_config}),
    )

    def capture_communicator(*args, **kwargs):
        captured.update(kwargs)
        return communicator

    monkeypatch.setattr(rope2d_vit_vlm, "MultiModulePipelineCommunicator", capture_communicator)

    args = _provider_args(mimo_bridge_skip_shape_exchange=skip_shape_exchange)
    assert (
        build_rope2d_vit_vlm_communicator(args, topology, encoder_name=ENCODER_NAME)
        is communicator
    )
    assert captured["bridge_comm_dtypes"] == {ENCODER_NAME: torch.bfloat16}
    shape_fns = captured["bridge_recv_shape_fns"]
    if skip_shape_exchange:
        batch = {"modality_token_indices": {ENCODER_NAME: torch.tensor([2, 5, 9])}}
        assert shape_fns[ENCODER_NAME](batch) == (3, LANGUAGE_HIDDEN_SIZE)
    else:
        assert shape_fns is None


def test_radio_provider_contract_remains_unchanged():
    provider = resolve_provider(SimpleNamespace(model_provider=NEMOTRON_MODEL_PROVIDER))

    assert provider.encoder_module_names == (RADIO_ENCODER_MODULE_NAME,)


@pytest.mark.parametrize(
    ("spec_name", "wide_residual", "expected_name"),
    [
        (None, None, "mamba_stack_spec"),
        (None, 3, "wide_residual_hybrid_stack_spec"),
        ("gated_delta_product_stack_spec", None, "gated_delta_product_stack_spec"),
        ("gated_delta_product_stack_spec", 3, "wide_residual_gated_delta_product_stack_spec"),
        (
            "wide_residual_gated_delta_product_stack_spec",
            3,
            "wide_residual_gated_delta_product_stack_spec",
        ),
    ],
)
def test_language_stack_spec_follows_spec_arg(spec_name, wide_residual, expected_name):
    from examples.mimo.model_providers.nemotron_moe_vlm import _language_stack_spec
    from megatron.core.models.hybrid import hybrid_layer_specs
    from megatron.core.models.mamba import mamba_layer_specs

    module = "megatron.core.models.hybrid.hybrid_layer_specs"
    args = SimpleNamespace(spec=None if spec_name is None else [module, spec_name])
    config = SimpleNamespace(wide_residual=wide_residual)
    specs = {**vars(mamba_layer_specs), **vars(hybrid_layer_specs)}

    assert _language_stack_spec(args, config) is specs[expected_name]


def test_model_provider_args():
    parser = argparse.ArgumentParser()
    add_model_provider_args(parser)

    args = parser.parse_args(
        [
            "--model-provider",
            ROPE2D_VIT_VLM_MODEL_PROVIDER,
            "--mimo-vision-num-layers",
            "2",
            "--mimo-vision-hidden-size",
            "64",
            "--mimo-vision-ffn-hidden-size",
            "128",
            "--mimo-vision-num-attention-heads",
            "4",
            "--mimo-vision-encoder-name",
            ENCODER_NAME,
        ]
    )

    assert args.model_provider == ROPE2D_VIT_VLM_MODEL_PROVIDER
    assert {name: getattr(args, name) for name in VISION_SIZES} == VISION_SIZES
    assert args.mimo_vision_encoder_name == ENCODER_NAME
    assert parser.parse_args([]).mimo_vision_encoder_name == "vision_encoder"
