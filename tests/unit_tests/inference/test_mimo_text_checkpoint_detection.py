# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MIMO checkpoints select the text backbone and its checkpoint prefix."""

from argparse import Namespace

import pytest

from megatron.core.inference.text_generation_server.dynamic_text_gen_server import (
    vlm_dynamic_inference,
)


@pytest.mark.parametrize("provider", ["nemotron-moe-vlm", "nemotron-moe-mistral-vit"])
def test_mimo_text_detection_selects_text_backbone(monkeypatch, provider):
    """Model detection selects the text backbone without constructing a VLM."""
    args = Namespace(model_provider="gpt")
    saved = Namespace(model_provider=provider, mimo_llm_tp=1)
    monkeypatch.setattr(vlm_dynamic_inference, "load_args_from_checkpoint", lambda _: (args, saved))
    # Detection also reads tensor metadata on newer main; keep this unit test independent of I/O.
    monkeypatch.setattr(
        vlm_dynamic_inference,
        "_checkpoint_tensor_keys",
        lambda _: [
            "language_model.module.module.decoder.layers.0.self_attention.linear_qkv.weight"
        ],
        raising=False,
    )
    assert not vlm_dynamic_inference._detect_vlm_from_checkpoint(args)
    assert args.model_provider == "hybrid"
    assert args.checkpoint_model_prefix == "language_model.module.module."
