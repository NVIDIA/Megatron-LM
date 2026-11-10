# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise the real GPT train/evaluate loop with static or runtime-CP packed data.

This is a smoke/coverage test, not a deterministic or golden-loss equivalence test.
The worker observes batch metadata without replacing training or collectives.
"""

import argparse
import json
import math
import os
import re
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

CASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CASE_DIR.parents[4]


def run_worker() -> None:
    """Run the unmodified GPT entrypoint and record which CP sizes it consumes."""
    import torch

    from megatron.core.datasets import data_schedule

    original_get_batch = data_schedule.get_batch_on_this_rank_for_sequence_packing
    observed = {"train": set(), "eval": set()}

    def observe_batch(*args, **kwargs):
        result = original_get_batch(*args, **kwargs)
        packed = result[5]
        group = packed.cp_group
        cp_size = 1 if group is None else group.size()
        if kwargs.get("dynamic_cp", False):
            assert group is not None, "Runtime CP1 must retain its singleton process group"
            assert packed.local_cp_size == cp_size
        phase = "train" if torch.is_grad_enabled() else "eval"
        observed[phase].add(cp_size)
        return result

    data_schedule.get_batch_on_this_rank_for_sequence_packing = observe_batch
    sys.argv = [str(REPO_ROOT / "pretrain_gpt.py"), *sys.argv[2:]]
    runpy.run_path(sys.argv[0], run_name="__main__")
    output = Path(os.environ["DCP_TEST_OBSERVATIONS"])
    output.mkdir(parents=True, exist_ok=True)
    (output / f"rank-{os.environ['RANK']}.json").write_text(
        json.dumps({phase: sorted(sizes) for phase, sizes in observed.items()}), encoding="utf-8"
    )


def run_test() -> None:
    """Launch eight ranks, then check real losses, checkpoints, and runtime groups."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packing-mode", choices=("static", "dynamic"), required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    options = parser.parse_args()
    output_root = options.output_path.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    # Preserve failed artifacts while permitting the CI runner to retry a workload.
    output = Path(tempfile.mkdtemp(prefix="attempt-", dir=output_root))
    config = yaml.safe_load((CASE_DIR / "model_config.yaml").read_text(encoding="utf-8"))
    model_args = config["MODEL_ARGS"]
    model_args.update(
        {
            "--sequence-packing-scheduler": (
                "default_dynamic_cp" if options.packing_mode == "dynamic" else "dp_balanced"
            ),
            "--max-seqlen-per-dp-cp-rank": 1024 if options.packing_mode == "dynamic" else 2048,
            "--varlen-mock-dataset-config-json": json.dumps(
                {"mode": "file", "path": str(CASE_DIR / "lengths.csv")}
            ),
            "--save": str(output / "checkpoints"),
        }
    )
    if options.packing_mode == "dynamic":
        model_args["--dynamic-context-parallel"] = True
    arguments = []
    for name, value in model_args.items():
        if value is not False and value is not None:
            arguments.append(name)
            if value is not True:
                arguments.append(str(value))
    environment = dict(os.environ)
    environment.update({name: str(value) for name, value in config["ENV_VARS"].items()})
    environment["DCP_TEST_OBSERVATIONS"] = str(output / "observations")
    environment["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    environment["OMP_NUM_THREADS"] = "1"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=8",
        str(Path(__file__).resolve()),
        "--worker",
        *arguments,
    ]
    log_path = output / "training.log"
    try:
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=900,
            )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        print(f"Training failed; full per-rank output: {log_path}", file=sys.stderr)
        print("\n".join(log_path.read_text(encoding="utf-8").splitlines()[-80:]), file=sys.stderr)
        raise
    log_text = log_path.read_text(encoding="utf-8")
    for pattern, minimum in (
        (r"\blm loss:\s*([^\s|]+)", model_args["--train-iters"]),
        (r"\blm loss value:\s*([^\s|]+)", 2),
        (r"\bgrad norm:\s*([^\s|]+)", model_args["--train-iters"]),
    ):
        values = [float(value) for value in re.findall(pattern, log_text)]
        assert len(values) >= minimum, f"Missing train/eval metrics in {log_path}: {pattern}"
        assert all(math.isfinite(value) for value in values), f"Non-finite metrics in {log_path}"
    tracker = output / "checkpoints" / "latest_checkpointed_iteration.txt"
    assert int(tracker.read_text(encoding="utf-8")) == model_args["--train-iters"]
    expected_sizes = {1, 2} if options.packing_mode == "dynamic" else {1}
    for phase in ("train", "eval"):
        sizes = set()
        for rank in range(8):
            observation = json.loads(
                (output / "observations" / f"rank-{rank}.json").read_text(encoding="utf-8")
            )
            assert observation[phase], f"Rank {rank} did not consume {phase} packed data"
            sizes.update(observation[phase])
        assert sizes == expected_sizes, f"{phase}: expected CP {expected_sizes}, saw {sizes}"
    print(f"Packed {options.packing_mode} GPT train/eval passed; artifacts: {output}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        run_worker()
    else:
        run_test()
