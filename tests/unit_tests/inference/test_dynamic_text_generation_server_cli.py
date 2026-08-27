# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for configurable sampling defaults on inference server CLIs."""

from argparse import ArgumentParser

import pytest

from examples.inference.launch_inference_server import add_serve_args
from tools.run_dynamic_text_generation_server import add_text_generation_server_args


@pytest.mark.parametrize(
    ("add_args", "required_args"),
    [
        (add_serve_args, []),
        (
            add_text_generation_server_args,
            ["--language-model-type", "placeholder", "--tokenizer-prompt-format", "mistral"],
        ),
    ],
)
@pytest.mark.parametrize(
    ("temperature_args", "expected_temperature"),
    [([], 1.0), (["--default-temperature", "0.4"], 0.4)],
)
def test_default_temperature_flag(add_args, required_args, temperature_args, expected_temperature):
    parser = add_args(ArgumentParser())
    args = parser.parse_args([*required_args, *temperature_args])

    assert args.default_temperature == expected_temperature
