# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Validate Qwen training inputs before checkpoint loading or distributed setup."""

import sys

import pytest

from tools.ssm.qwen_gdn_offload_training import _arguments


@pytest.mark.parametrize(
    "flags,error",
    [
        (["--tp", "2", "--sp", "--seq-length", "129"], "divisible by TP * CP"),
        (["--tp", "4", "--sp", "--cp", "2", "--seq-length", "132"], "divisible by TP * CP"),
        (["--pp", "2", "--vp", "2", "--microbatches", "1"], "divisible by --pp"),
        (["--pp", "2", "--vp", "2", "--microbatches", "3"], "divisible by --pp"),
        (["--tp", "2", "--sp", "--seq-length", "128"], None),
        (["--tp", "4", "--sp", "--cp", "2", "--seq-length", "136"], None),
        (["--tp", "2", "--seq-length", "129"], None),
        (["--pp", "2", "--microbatches", "1"], None),
        (["--pp", "2", "--vp", "2", "--microbatches", "2"], None),
        (["--pp", "2", "--vp", "2", "--microbatches", "4"], None),
    ],
)
def test_qwen_parallel_input_boundaries(
    flags: list[str],
    error: str | None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """Reject incompatible partitions and preserve neighboring valid configurations."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["qwen-training", "--weights", "weights", "--data", "data", "--output", "out", "--dry-run"]
        + flags,
    )
    if error is None:
        assert _arguments().dry_run
    else:
        with pytest.raises(SystemExit) as raised:
            _arguments()
        assert raised.value.code == 2
        assert error in capsys.readouterr().err
