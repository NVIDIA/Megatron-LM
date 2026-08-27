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
    ("serve_args", "expected_defaults"),
    [
        ([], (1.0, 1.0, 0, False)),
        (
            [
                "--default-temperature",
                "0.4",
                "--default-top-p",
                "0.8",
                "--default-top-k",
                "5",
                "--eval-mode",
            ],
            (0.4, 0.8, 5, True),
        ),
    ],
)
def test_sampling_default_flags(add_args, required_args, serve_args, expected_defaults):
    parser = add_args(ArgumentParser())
    args = parser.parse_args([*required_args, *serve_args])

    assert (
        args.default_temperature,
        args.default_top_p,
        args.default_top_k,
        args.eval_mode,
    ) == expected_defaults
