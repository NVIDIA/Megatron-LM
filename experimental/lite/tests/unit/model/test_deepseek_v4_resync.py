# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


_RESYNC_PATH = (
    Path(__file__).resolve().parents[3]
    / "megatron"
    / "lite"
    / "model"
    / "deepseek_v4"
    / "lite"
    / "resync.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "deepseek_v4_resync_under_test", _RESYNC_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
export_resync_weights = _MODULE.export_resync_weights


def test_fp8_resync_exports_float32_block_scales() -> None:
    config = SimpleNamespace(
        expert_dtype="fp8",
        quantization_config={"weight_block_size": [128, 128]},
    )
    source = torch.randn(128, 128, dtype=torch.bfloat16)

    exported = dict(
        export_resync_weights(
            [("layers.0.ffn.experts.0.up_proj.weight", source)],
            config,
            resync_config={"expert_dtype": "fp8"},
        )
    )

    assert exported["layers.0.ffn.experts.0.up_proj.scale"].dtype == torch.float32


def test_fp4_resync_uses_mxfp4_only_for_routed_experts(monkeypatch) -> None:
    calls: list[tuple[str, str | None]] = []

    def fake_mxfp4(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(("mxfp4", None))
        return tensor.to(torch.uint8), torch.ones(1, dtype=torch.uint8)

    def fake_block_fp8(
        tensor: torch.Tensor,
        _block_shape: tuple[int, int],
        *,
        scale_format: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(("block_fp8", scale_format))
        return tensor.to(torch.float8_e4m3fn), torch.ones(1, dtype=torch.uint8)

    monkeypatch.setattr(_MODULE, "quantize_mxfp4", fake_mxfp4)
    monkeypatch.setattr(_MODULE, "quantize_block_fp8", fake_block_fp8)
    config = SimpleNamespace(
        expert_dtype="fp4",
        quantization_config={"weight_block_size": [128, 128]},
    )
    source = torch.randn(2, 2, dtype=torch.bfloat16)

    exported = dict(
        export_resync_weights(
            [
                ("layers.0.ffn.experts.0.up_proj.weight", source),
                ("layers.0.attn.out_proj.weight", source),
            ],
            config,
            resync_config={"expert_dtype": "fp4"},
        )
    )

    assert calls == [("mxfp4", None), ("block_fp8", "e8m0")]
    assert exported["layers.0.ffn.experts.0.up_proj.weight"].dtype == torch.uint8
    assert exported["layers.0.attn.out_proj.weight"].dtype == torch.float8_e4m3fn
