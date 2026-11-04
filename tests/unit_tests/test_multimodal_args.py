# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import sys
from argparse import ArgumentParser
from pathlib import Path

MULTIMODAL_EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "multimodal"
sys.path.insert(0, str(MULTIMODAL_EXAMPLE_DIR))

from multimodal_args import add_multimodal_extra_args  # noqa: E402


def _parse_args(*extra_args):
    parser = add_multimodal_extra_args(ArgumentParser())
    return parser.parse_args(
        [
            "--language-model-type",
            "test-language-model",
            "--tokenizer-prompt-format",
            "nemotron6-moe",
            *extra_args,
        ]
    )


def test_vision_projection_type_defaults_to_mlp():
    assert _parse_args().vision_projection_type == "mlp"


def test_vision_projection_type_accepts_affine():
    assert _parse_args("--vision-projection-type", "affine").vision_projection_type == "affine"
