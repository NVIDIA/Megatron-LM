#!/usr/bin/env python3
# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/08035723e7a86f94fc54a7f4473bd0519f2ac941/scripts/conversion/setup_conversion.py
# Source license: Apache-2.0.
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Launch CPU or distributed GPU checkpoint conversion with NeMo Run."""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import sys
from pathlib import Path

import nemo_run as run
from arguments import build_parser, conversion_worker_args
from container_runtime import (
    apply_container_runtime,
    run_from_container_repo_root,
    validate_container_runtime,
)
from nemo_run.config import get_nemorun_home
from slurm_wait import wait_for_slurm_job
from torchx.specs.api import AppState

logger = logging.getLogger(__name__)

CONTAINER_REPO_ROOT = Path("/opt/Megatron-LM")
LOCAL_REPO_ROOT = Path(__file__).resolve().parents[3]


def _parse_env(values: list[str]) -> list[str]:
    """Validate inherited environment variable names without materializing values."""
    env_names: list[str] = []
    for name in values:
        if "=" in name:
            raise ValueError(
                "--env accepts NAME only; export the value before launching to keep it out of arguments."
            )
        if name not in os.environ:
            raise ValueError(
                f"Environment variable '{name}' is not set in the launcher environment."
            )
        if name not in env_names:
            env_names.append(name)
    return env_names


def _parse_mounts(values: list[str]) -> list[str]:
    """Normalize explicit same-path and host-to-container mounts."""
    mounts: list[str] = []
    for value in values:
        if ":" in value:
            host_path, container_path = value.split(":", 1)
        else:
            host_path = value
            container_path = value
        if not host_path or not container_path:
            raise ValueError(f"Invalid --mount value '{value}'; expected HOST or HOST:CONTAINER.")
        mount = f"{host_path}:{container_path}"
        if mount not in mounts:
            mounts.append(mount)
    return mounts


def _validate_args(args: argparse.Namespace) -> None:
    """Validate execution resources and conversion parallelism before launch."""
    validate_container_runtime(args)
    if args.nodes < 1:
        raise ValueError("--nodes must be at least 1.")
    if args.cpu_processes_per_node < 1:
        raise ValueError("--cpu-processes-per-node must be at least 1.")
    if args.cpus_per_task is not None and args.cpus_per_task < 1:
        raise ValueError("--cpus-per-task must be at least 1.")
    distributed_timeout_minutes = getattr(args, "distributed_timeout_minutes", None)
    if distributed_timeout_minutes is not None and distributed_timeout_minutes < 1:
        raise ValueError("--distributed-timeout-minutes must be at least 1.")
    if any(not value.strip() for value in args.srun_args):
        raise ValueError("--srun-arg values must not be empty.")
    for name in ("tp", "pp", "ep", "etp"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name} must be at least 1.")

    if args.executor == "local":
        if args.nodes != 1:
            raise ValueError("Local execution supports exactly one node.")
        if args.detach:
            raise ValueError("--detach is only supported by the Slurm executor.")
        if args.exclusive:
            raise ValueError("--exclusive is only supported by the Slurm executor.")
        if args.srun_args:
            raise ValueError("--srun-arg is only supported by the Slurm executor.")
        if args.additional_slurm_params:
            raise ValueError("--additional-slurm-params is only supported by the Slurm executor.")
        if args.mount:
            raise ValueError(
                "--mount is only supported by the Slurm executor; mount paths before local execution."
            )
    elif not args.account or not args.partition:
        raise ValueError("Slurm execution requires --account and --partition.")
    elif not args.container_image:
        raise ValueError("Slurm execution requires --container-image or CONTAINER_IMAGE.")

    if args.command == "roundtrip" and args.device != "gpu":
        raise ValueError("Round-trip validation requires the GPU backend.")

    distributed_cpu = args.device == "cpu" and args.cpu_processes_per_node > 1
    if args.device == "cpu":
        if not distributed_cpu and args.nodes != 1:
            raise ValueError("Single-process CPU conversion supports exactly one node.")
        if args.gpus_per_node is not None and args.gpus_per_node < 0:
            raise ValueError("--gpus-per-node must not be negative.")
        if distributed_cpu and args.gpus_per_node not in (None, 0):
            raise ValueError("Distributed CPU conversion does not request GPU resources.")
        if args.gres:
            raise ValueError("CPU conversion does not accept --gres.")
        if not distributed_cpu and any(
            getattr(args, name) != 1 for name in ("tp", "pp", "ep", "etp")
        ):
            raise ValueError("CPU conversion requires TP=PP=EP=ETP=1.")
        if distributed_cpu:
            world_size = args.nodes * args.cpu_processes_per_node
            model_parallel_size = args.tp * args.pp
            if world_size % model_parallel_size != 0:
                raise ValueError("nodes*cpu-processes-per-node must be divisible by TP*PP.")
            expert_model_parallel_size = args.etp * args.ep * args.pp
            if world_size % expert_model_parallel_size != 0:
                raise ValueError("nodes*cpu-processes-per-node must be divisible by ETP*EP*PP.")
    else:
        if args.cpu_processes_per_node != 1:
            raise ValueError("--cpu-processes-per-node is only supported by the CPU backend.")
        if args.gpus_per_node is None or args.gpus_per_node < 1:
            raise ValueError("GPU conversion requires --gpus-per-node of at least 1.")
        if args.executor == "local":
            worker_values = [args.hf_model]
            if args.command != "roundtrip":
                worker_values.append(args.megatron_path)
            if args.command == "export":
                worker_values.append(args.hf_path)
            if any(shlex.quote(value) != value for value in worker_values):
                raise ValueError(
                    "Local GPU execution cannot pass model IDs or paths containing whitespace or shell "
                    "metacharacters through NeMo Run 0.10; use shell-safe names and paths."
                )
        world_size = args.nodes * args.gpus_per_node
        model_parallel_size = args.tp * args.pp
        if world_size % model_parallel_size != 0:
            raise ValueError("nodes*gpus-per-node must be divisible by TP*PP.")
        expert_model_parallel_size = args.etp * args.ep * args.pp
        if world_size % expert_model_parallel_size != 0:
            raise ValueError("nodes*gpus-per-node must be divisible by ETP*EP*PP.")

    if args.command == "export":
        if args.save_every_n_ranks < 1:
            raise ValueError("--save-every-n-ranks must be at least 1.")
        distributed_save = (
            args.distributed_save
            if args.distributed_save is not None
            else (args.device == "gpu" or distributed_cpu)
        )
        if args.device == "cpu" and distributed_save and not distributed_cpu:
            raise ValueError(
                "--distributed-save requires distributed CPU export or the GPU backend."
            )
        if distributed_cpu and not distributed_save:
            raise ValueError("Distributed CPU export requires --distributed-save.")
        if not distributed_save and args.save_every_n_ranks != 1:
            raise ValueError("--save-every-n-ranks requires --distributed-save.")
        if args.device == "cpu" and not distributed_cpu and args.export_weight_dtype is not None:
            raise ValueError(
                "--export-weight-dtype requires distributed CPU export or the GPU backend."
            )


def _build_executor(args: argparse.Namespace, env_names: list[str], mounts: list[str]) -> object:
    """Build a Local or Slurm NeMo Run executor."""
    distributed_cpu = args.device == "cpu" and args.cpu_processes_per_node > 1
    task_count = args.gpus_per_node if args.device == "gpu" else args.cpu_processes_per_node
    launcher = (
        run.Torchrun()
        if args.executor == "local" and (args.device == "gpu" or distributed_cpu)
        else None
    )
    if args.executor == "local":
        executor = run.LocalExecutor(
            ntasks_per_node=task_count, launcher=launcher, packager=run.Packager()
        )
        executor.nodes = 1
        return executor

    gpu_kwargs = {}
    if args.gpus_per_node and not args.no_gpu_resource_request:
        gpu_kwargs["gpus_per_node"] = args.gpus_per_node
    cpu_kwargs = {}
    if args.cpus_per_task is not None:
        cpu_kwargs["cpus_per_task"] = args.cpus_per_task
    container_env = [*env_names]
    if "PYTHONPATH" not in container_env:
        container_env.append("PYTHONPATH")
    executor = run.SlurmExecutor(
        poll_estimated_start_time=False,
        account=args.account,
        partition=args.partition,
        job_name_prefix=args.experiment_name,
        nodes=args.nodes,
        ntasks_per_node=task_count,
        mem=args.mem,
        # NeMo Run renders False as ``#SBATCH --exclusive=False``; None omits the directive.
        exclusive=args.exclusive or None,
        time=args.time,
        gres=args.gres,
        launcher=launcher,
        tunnel=run.LocalTunnel(job_dir=os.path.join(get_nemorun_home(), "experiments")),
        packager=run.Packager(),
        container_image=args.container_image,
        container_mounts=mounts,
        container_env=container_env,
        additional_parameters={**args.additional_slurm_params, "export": ",".join(container_env)},
        srun_args=args.srun_args,
        **cpu_kwargs,
        **gpu_kwargs,
    )
    # Values are inherited by Slurm and selected by name for the container;
    # keeping env_vars empty prevents NeMo Run from serializing secrets.
    executor.env_vars = {}
    return executor


def _build_task(args: argparse.Namespace) -> tuple[run.Script, list[str]]:
    """Build the in-job conversion or round-trip task and its display arguments."""
    display_args = conversion_worker_args(args)
    relative_task_path = Path("examples/bridge/conversion/run_conversion.py")
    repo_root = LOCAL_REPO_ROOT if args.executor == "local" else CONTAINER_REPO_ROOT
    task_args = (
        display_args
        if args.executor == "local"
        else [shlex.quote(argument) for argument in display_args]
    )
    if args.executor == "local":
        existing_pythonpath = os.environ.get("PYTHONPATH", "")
        pythonpath = str(repo_root)
        if existing_pythonpath:
            pythonpath = f"{pythonpath}:{existing_pythonpath}"
    else:
        pythonpath = f"{repo_root}:$PYTHONPATH"
    task_env = {"PYTHONPATH": pythonpath}
    distributed_cpu = args.device == "cpu" and args.cpu_processes_per_node > 1
    if args.executor == "local" and (args.device == "gpu" or distributed_cpu):
        # The torchrun console script can belong to a different Python than the
        # uv environment running this launcher. PyTorch honors PYTHON_EXEC for
        # workers, so keep conversion dependencies from this environment.
        task_env["PYTHON_EXEC"] = sys.executable
    task = run.Script(
        path=str(repo_root / relative_task_path),
        # NeMo Run recognizes the literal "python" entrypoint when building a
        # torchrun command. An absolute interpreter path is treated as a
        # non-Python executable and makes torchrun execute this file directly.
        entrypoint="python",
        env=task_env,
        args=task_args,
    )
    return task, display_args


def _raise_on_failed_tasks(experiment: run.Experiment) -> None:
    """Raise when a synchronous NeMo Run task did not complete successfully."""
    unsuccessful = [
        f"{job.id}={job.state}" for job in experiment.jobs if job.state != AppState.SUCCEEDED
    ]
    if unsuccessful:
        raise RuntimeError(f"Conversion failed: {', '.join(unsuccessful)}")


def main(argv: list[str] | None = None) -> None:
    """Parse conversion arguments and launch the selected NeMo Run experiment."""
    args = build_parser(include_execution=True).parse_args(argv)
    _validate_args(args)
    env_names = _parse_env(args.env)
    mounts = _parse_mounts(args.mount)
    executor = _build_executor(args, env_names, mounts)
    task, worker_args = _build_task(args)

    experiment_name = args.experiment_name or f"conversion-{args.command}-{args.device}"
    logger.info("Conversion command: %s", shlex.join(["python", task.path, *worker_args]))
    logger.info("Executor: %s; device: %s", args.executor, args.device)
    logger.info("Forwarded environment variables: %s", ", ".join(env_names) or "none")
    if args.executor == "slurm":
        logger.info("Container mounts: %s", ", ".join(mounts) or "none")

    experiment_options = {"skip_status_at_exit": True} if args.executor == "slurm" else {}
    task = apply_container_runtime(
        args, executor=executor, task=task, mounts=mounts, env_names=env_names
    )
    if args.executor == "slurm" and args.container_runtime == "pyxis":
        task = run_from_container_repo_root(task)
    with run.Experiment(experiment_name, **experiment_options) as experiment:
        experiment.add(task, executor=executor, name=f"{args.command}-{args.device}")
        if args.submission_dry_run:
            experiment.dryrun()
            return
        if args.executor == "slurm":
            experiment.run(detach=True, tail_logs=False)
            if not args.detach:
                wait_for_slurm_job(experiment, poll_interval=args.poll_interval)
        else:
            experiment.run(detach=False, tail_logs=True)
    if not args.detach:
        _raise_on_failed_tasks(experiment)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
