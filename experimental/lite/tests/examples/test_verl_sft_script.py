# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import os
import subprocess
from pathlib import Path

LITE_ROOT = Path(__file__).resolve().parents[2]
SFT_SCRIPT = LITE_ROOT / "examples" / "verl" / "scripts" / "run_qwen3moe_sft.sh"


def _run_verl_sft_dry_run(tmp_path: Path, **env_overrides: str) -> str:
    env = {
        **os.environ,
        "MODEL_PATH": "/tmp/mlite-model",
        "TRAIN_FILES": "/tmp/mlite-train.parquet",
        "OUTPUT_ROOT": str(tmp_path),
        "DRY_RUN": "1",
        "NUM_GPUS": "1",
        "NPROC_PER_NODE": "1",
        "TP_SIZE": "1",
        "PP_SIZE": "1",
        "CP_SIZE": "1",
        "EP_SIZE": "1",
        "ETP_SIZE": "1",
        **env_overrides,
    }
    completed = subprocess.run(
        [str(SFT_SCRIPT)], env=env, text=True, capture_output=True, check=True
    )
    return completed.stdout


def test_verl_sft_script_maps_offload_env_to_backend_args(tmp_path):
    command = _run_verl_sft_dry_run(
        tmp_path,
        PARAM_OFFLOAD="True",
        OPTIMIZER_OFFLOAD="True",
        OPTIMIZER_STATE_OFFLOAD_FRACTION="0.75",
    )

    assert "engine.param_offload=True" in command
    assert "engine.optimizer_offload=True" in command
    assert "+optim.override_optimizer_config.offload_fraction=0.75" in command
    assert "+optim.override_optimizer_config.use_precision_aware_optimizer=True" in command


def test_verl_sft_script_does_not_emit_optimizer_state_offload_when_disabled(tmp_path):
    command = _run_verl_sft_dry_run(tmp_path, PARAM_OFFLOAD="False", OPTIMIZER_OFFLOAD="False")

    assert "engine.param_offload=False" in command
    assert "engine.optimizer_offload=False" in command
    assert "override_optimizer_config.offload_fraction" not in command
