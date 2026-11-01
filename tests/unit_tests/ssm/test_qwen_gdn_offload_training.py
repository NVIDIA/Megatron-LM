# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Validate Qwen training inputs and precision selection."""

import os
import sys
from pathlib import Path

import pytest
import torch

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from tools.ssm.qwen_gdn_offload_training import _arguments, _forward_step


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


@pytest.mark.parametrize("existing", [False, True])
def test_checkpoint_path_boundaries(
    existing: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject overwriting a checkpoint or resuming a missing directory before model loading."""
    path = tmp_path / "checkpoint"
    if existing:
        path.mkdir()
    base = [
        "qwen-training",
        "--weights",
        "weights",
        "--data",
        "data",
        "--output",
        "out",
        "--dry-run",
    ]
    for flag in ("--save-checkpoint", "--load-checkpoint"):
        monkeypatch.setattr(sys, "argv", [*base, flag, str(path)])
        if existing == (flag == "--save-checkpoint"):
            with pytest.raises(SystemExit) as raised:
                _arguments()
            assert raised.value.code == 2
        else:
            args = _arguments()
            assert getattr(args, flag[2:].replace("-", "_")) == path


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for NVFP4.")
def test_forward_step_activates_nvfp4() -> None:
    """Exercise real TE GEMMs when a custom decoder leaves precision to its caller."""
    te = pytest.importorskip("transformer_engine.pytorch")
    from transformer_engine.pytorch.fp8 import check_nvfp4_support  # type: ignore[import-untyped]

    supported, reason = check_nvfp4_support()
    if not supported:
        pytest.skip(reason)
    if torch.cuda.get_device_capability()[0] == 12 and any(
        os.environ.get(name) != "1"
        for name in ("NVTE_NVFP4_DISABLE_RHT", "NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING")
    ):
        pytest.skip("SM120 validation requires the constrained NVFP4 recipe before startup.")
    config = TransformerConfig(
        num_layers=1, hidden_size=32, num_attention_heads=1, bf16=True, fp4="e2m1"
    )

    class LanguageModel(MegatronModule):
        def __init__(self) -> None:
            super().__init__(config=config)
            self.projection = te.Linear(32, 32, params_dtype=torch.bfloat16, device="cuda")

        def forward(
            self, tokens: torch.Tensor, position_ids: None, attention_mask: None, labels: None
        ) -> torch.Tensor:
            return self.projection(tokens)

    model = LanguageModel()
    inputs = torch.randn(128, 32, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    output, _ = _forward_step(iter([dict(tokens=inputs, position_ids=None, labels=None)]), model)
    assert model.projection.fp8 and model.projection.fp8_meta["recipe"].nvfp4()
    output.float().square().mean().backward()
    assert torch.isfinite(inputs.grad).all()
    assert torch.isfinite(model.projection.weight.grad).all()
