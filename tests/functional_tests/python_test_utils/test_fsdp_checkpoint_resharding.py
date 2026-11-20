# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Run with one pytest coordinator on four GPUs, without an outer torchrun."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
import yaml

from tests.functional_tests.python_test_utils.common import read_tb_logs_as_list
from tests.functional_tests.python_test_utils.test_optimizer_grads_match import (
    load_dist_checkpoint_pt,
)

ROOT = Path(__file__).resolve().parents[3]
CONFIG = (
    ROOT
    / "tests/functional_tests/test_cases/hybrid/hybrid_mfsdp_v2_checkpoint_reshard/model_config.yaml"
)


@pytest.mark.parametrize("source,target", [((2, 2), (1, 1)), ((1, 1), (2, 2))])
def test_fsdp_checkpoint_resharding(tmp_path, source, target):
    """Reshard weights and Adam state exactly, then resume the same global batches."""
    assert torch.cuda.device_count() == 4, "This functional test requires four GPUs"
    with CONFIG.open() as stream:
        common_args = yaml.safe_load(stream)["MODEL_ARGS"]

    def train(name, topology, *extra):
        pp, vpp = topology
        # Eight attention/MLP blocks, split into PP * VPP contiguous chunks.
        pattern = "|".join(["*-" * (8 // (pp * vpp))] * (pp * vpp))
        args = {
            **common_args,
            "--pipeline-model-parallel-size": pp,
            "--hybrid-layer-pattern": pattern,
            "--save": str(tmp_path / name),
            "--tensorboard-dir": str(tmp_path / name / "tensorboard"),
            "--data-cache-path": str(tmp_path / "data-cache"),
        }
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=4",
            "--max-restarts=0",
            "pretrain_hybrid.py",
        ]
        for key, value in args.items():
            command.append(key)
            if value is not True:
                command.extend(map(str, value if isinstance(value, list) else [value]))
        command.extend(extra)
        log = tmp_path / f"{name}.log"
        env = {
            **os.environ,
            "CUDA_DEVICE_MAX_CONNECTIONS": "32",
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        }
        with log.open("w") as output:
            result = subprocess.run(
                command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=300
            )
        assert result.returncode == 0, f"{name} failed; see {log}\n{log.read_text()[-12000:]}"

    train("source", source)
    resume_args = (
        "--load",
        str(tmp_path / "source"),
        "--ckpt-step",
        "2",
        "--exit-on-missing-checkpoint",
    )
    # The existing conversion path saves immediately after loading, before an optimizer step.
    train(
        "roundtrip",
        target,
        *resume_args,
        "--ckpt-convert-format",
        "fsdp_dtensor",
        "--ckpt-convert-save",
        str(tmp_path / "converted"),
    )
    saved = tmp_path / "source/iter_0000002"
    restored = tmp_path / "converted/fsdp_dtensor/iter_0000002"
    states = [
        load_dist_checkpoint_pt(path, pattern=r"^(model\d*|optimizer)\.")
        for path in (saved, restored)
    ]
    assert states[0].keys() == states[1].keys()
    for field in ("exp_avg", "exp_avg_sq"):
        assert any(field in name for name in states[0]), f"Missing optimizer coverage: {field}"
    # V2 exposes FP32 main weights directly in the model state dict.
    weights = {name: value for name, value in states[0].items() if name.startswith("model")}
    assert weights and all(value.dtype == torch.float32 for value in weights.values())
    for name in states[0]:
        torch.testing.assert_close(states[0][name], states[1][name], rtol=0, atol=0, msg=name)
    metadata_keys = dcp.FileSystemReader(saved).read_metadata().state_dict_metadata
    metadata_keys = [
        name
        for name, value in metadata_keys.items()
        if name == "iteration"
        or (name.startswith(("optimizer.", "opt_param_scheduler.")) and not hasattr(value, "size"))
    ]
    metadata_states = []
    for path in (saved, restored):
        metadata = dict.fromkeys(metadata_keys)
        dcp.load(metadata, checkpoint_id=path, no_dist=True)
        assert metadata["iteration"] == 2
        metadata_states.append(metadata)
    assert (
        metadata_states[0] == metadata_states[1]
    ), "Optimizer steps and scheduler must survive resharding"
    report = {
        "source_pp_vpp": source,
        "target_pp_vpp": target,
        "iteration": 2,
        "tensor_count": len(states[0]),
        "tensor_elements": sum(value.numel() for value in states[0].values()),
        "fp32_main_weight_count": len(weights),
        "adam_tensor_counts": {
            field: sum(field in name.split(".") for name in states[0])
            for field in ("exp_avg", "exp_avg_sq")
        },
        "tensor_max_abs_error": 0,
        "metadata_exact": True,
        "metadata_keys": metadata_keys,
    }
    report_path = tmp_path / "verification.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    train("resumed", target, *resume_args)
    losses = [
        read_tb_logs_as_list(str(tmp_path / name / "tensorboard"), train_iters=4, step_size=1)[
            "lm loss"
        ].values
        for name in ("source", "resumed")
    ]
    assert set(losses[0]) == {1, 2, 3, 4}
    assert set(losses[1]) == {3, 4}, "Resume must preserve the training iteration"
    # BF16 reductions have a different order after PP/DP resharding.
    torch.testing.assert_close(
        torch.tensor(list(losses[1].values())),
        torch.tensor([losses[0][i] for i in (3, 4)]),
        rtol=1e-3,
        atol=1e-4,
    )
    report["losses"] = {
        step: {
            "source": float(losses[0][step]),
            "resumed": float(losses[1][step]),
            "abs_error": float(abs(losses[1][step] - losses[0][step])),
            "rel_error": float(abs(losses[1][step] - losses[0][step]) / abs(losses[0][step])),
        }
        for step in (3, 4)
    }
    report.update(loss_rtol=1e-3, loss_atol=1e-4, passed=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
