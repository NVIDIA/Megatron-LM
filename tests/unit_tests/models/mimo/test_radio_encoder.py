# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU-only unit tests for the RADIO vision encoder helpers.

Covers the RADIO-specific arg registration and the encoder ``ModuleSpec`` builder
mapping (the previously write-only ``--pixel-shuffle`` / ``--disable-vision-class-token``
flags must now flow into the wrapper params). A real wrapper forward needs GPU/TE
and is left to a functional (cog) check.
"""

import argparse
from types import SimpleNamespace

from examples.mimo.model_providers.radio_encoder import (
    RADIOEncoderWrapper,
    add_radio_encoder_args,
    radio_vision_encoder_spec,
)


def test_add_radio_encoder_args_registers_flags():
    parser = argparse.ArgumentParser()
    add_radio_encoder_args(parser)

    defaults = parser.parse_args([])
    assert defaults.class_token_len == 8
    assert defaults.pixel_shuffle is False
    assert defaults.disable_vision_class_token is False

    enabled = parser.parse_args(
        ["--class-token-len", "4", "--pixel-shuffle", "--disable-vision-class-token"]
    )
    assert enabled.class_token_len == 4
    assert enabled.pixel_shuffle is True
    assert enabled.disable_vision_class_token is True


def test_radio_vision_encoder_spec_reads_args(monkeypatch):
    # Avoid importing TransformerEngine on the CPU-only path.
    monkeypatch.setattr(
        "examples.mimo.model_providers.radio_encoder.get_vit_layer_with_transformer_engine_spec",
        lambda: "layer-spec",
    )
    args = SimpleNamespace(
        img_h=512,
        img_w=512,
        patch_dim=16,
        class_token_len=8,
        pixel_shuffle=True,
        disable_vision_class_token=True,
        freeze_vit=True,
        dynamic_resolution=False,
    )
    vision_config = object()

    spec = radio_vision_encoder_spec(args, vision_config, pg_collection=None)

    assert spec.module is RADIOEncoderWrapper
    params = spec.params
    assert params["transformer_config"] is vision_config
    assert params["img_h"] == 512 and params["img_w"] == 512
    assert params["patch_dim"] == 16
    assert params["class_token_len"] == 8
    # The previously write-only flags now drive the wrapper params.
    assert params["apply_pixel_shuffle"] is True
    assert params["drop_class_token"] is True
    assert params["force_eval_mode"] is True
    assert params["dynamic_resolution"] is False
