# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/08035723e7a86f94fc54a7f4473bd0519f2ac941/scripts/common/container_runtime.py
# Source license: Apache-2.0.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Optional direct-Enroot execution for the public Slurm launchers."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import nemo_run as run


def _require_lustre(paths: list[str]) -> None:
    """Check submitting-node paths before NeMo-Run can create experiment files."""
    for path in paths:
        try:
            filesystem = subprocess.check_output(
                ["stat", "-f", "-c", "%T", "--", path], text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError(
                f"Direct Enroot requires an existing cluster-side path: {path}"
            ) from exc
        if filesystem != "lustre":
            raise ValueError(
                f"Direct Enroot requires Lustre storage, including NEMORUN_HOME: {path}"
            )


def add_container_runtime_args(parser: argparse._ArgumentGroup) -> None:
    """Add shared runtime selection without changing the default Pyxis path."""
    parser.add_argument(
        "--container-runtime",
        choices=("pyxis", "enroot"),
        default=os.environ.get("BRIDGE_CONTAINER_RUNTIME", "pyxis"),
        help="Slurm container backend (default: pyxis). Enroot requires an existing Lustre image and root.",
    )
    parser.add_argument(
        "--enroot-root",
        default=os.environ.get("BRIDGE_ENROOT_ROOT"),
        help="Existing owned Lustre directory for direct-Enroot runtime, caches and temporary files.",
    )


def validate_container_runtime(args: argparse.Namespace) -> None:
    """Reject incompatible runtime options before creating an experiment."""
    if args.container_runtime not in ("pyxis", "enroot"):
        raise ValueError("--container-runtime must be pyxis or enroot.")
    if args.container_runtime == "pyxis":
        if args.enroot_root:
            raise ValueError("--enroot-root requires --container-runtime enroot.")
        return
    if getattr(args, "executor", "slurm") != "slurm":
        raise ValueError("Direct Enroot requires --executor slurm.")
    for name, value in (
        ("--enroot-root", args.enroot_root),
        ("--container-image", args.container_image),
    ):
        if not value or not Path(value).is_absolute() or ".." in Path(value).parts or value == "/":
            raise ValueError(f"{name} must be an absolute cluster-side path for direct Enroot.")
        if any(char.isspace() or char in ":," for char in value):
            raise ValueError(f"{name} contains unsupported mount/path characters.")
    if not args.container_image.endswith((".sqsh", ".squashfs")):
        raise ValueError(
            "Direct Enroot requires an existing .sqsh or .squashfs image; it never imports images."
        )
    if any("container" in value or "--export" in value for value in args.srun_args):
        raise ValueError(
            "Direct Enroot does not accept Pyxis/container or --export overrides in --srun-arg."
        )


def apply_container_runtime(
    args: argparse.Namespace,
    *,
    executor: run.SlurmExecutor,
    task: run.Script,
    mounts: list[str],
    env_names: list[str],
) -> run.Script:
    """Wrap the unchanged worker in direct Enroot, or return the Pyxis task.

    The host wrapper is packaged by NeMo-Run and executed once per Slurm task.
    No image is downloaded, extracted or copied on the submitting host. Runtime
    filesystem and effective Enroot configuration checks run on the allocation.
    """
    if args.container_runtime == "pyxis":
        return task
    import nemo_run as run
    from nemo_run.config import get_nemorun_home

    validate_container_runtime(args)
    enroot_mounts = []
    host_paths = [args.enroot_root, args.container_image, get_nemorun_home()]
    for mount in mounts:
        parts = mount.split(":")
        if len(parts) not in (2, 3) or any(not Path(p).is_absolute() for p in parts[:2]):
            raise ValueError("Direct Enroot mounts must be absolute HOST:CONTAINER[:ro|rw] paths.")
        source, destination = parts[:2]
        if ".." in Path(source).parts or ".." in Path(destination).parts:
            raise ValueError("Direct Enroot mount paths must not contain parent traversal.")
        source, destination = str(Path(source)), str(Path(destination))
        mode = parts[2] if len(parts) == 3 else "rw"
        if any(char.isspace() or char == "," for char in mount):
            raise ValueError("Direct Enroot mount paths must not contain whitespace or commas.")
        if mode not in ("ro", "rw"):
            raise ValueError("Direct Enroot mount options must be ro or rw.")
        if destination in ("/", "/tmp", "/var/tmp") or destination == args.enroot_root:
            raise ValueError("Direct Enroot reserves root, temporary and runtime mounts.")
        host_paths.append(source)
        # Explicit options suppress Enroot's default destination creation.
        enroot_mounts.append(f"{source}:{destination}:none:bind,{mode},x-create=auto")
    names = sorted(set(env_names) | set(task.env))
    if any(not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name) for name in names):
        raise ValueError("Direct Enroot environment entries must be variable names, not values.")
    if any(name.startswith("ENROOT_") for name in names):
        raise ValueError("Direct Enroot owns ENROOT_* settings; do not forward overrides.")
    _require_lustre(host_paths)
    # Script.args are already individually shell-quoted by all three launchers.
    command = " ".join([shlex.quote(task.entrypoint), shlex.quote(task.path), *task.args])
    prelude = "\n".join(
        [
            "set -euo pipefail",
            f"runtime_parent={shlex.quote(args.enroot_root)}",
            f"image_path={shlex.quote(args.container_image)}",
            f"host_paths=({shlex.join(host_paths)})",
            f"user_mounts=({shlex.join(enroot_mounts)})",
            f"forward_names=({shlex.join(names)})",
        ]
    )
    wrapper = Path(__file__).with_name("enroot_entrypoint.sh").read_text()
    inline = (
        prelude
        + "\n"
        + wrapper
        + "\n"
        + 'exec enroot start "${enroot_args[@]}" "$image_path" bash -c '
        + shlex.quote("set -e\ncd /opt/Megatron-LM\nexec " + command)
        + "\n"
    )
    executor.container_image = None
    executor.container_mounts = []
    executor.container_env = []
    # Enroot needs the original HOME even with home mounting disabled. Slurm
    # also applies this name allowlist to srun, after NeMo-Run sets task.env
    # (including PYTHONPATH) in the batch script. Keep values/secrets inherited.
    executor.additional_parameters["export"] = ",".join(dict.fromkeys(["PATH", "HOME", *names]))
    return run.Script(inline=inline, entrypoint="bash", env=task.env)


def run_from_container_repo_root(task: run.Script) -> run.Script:
    """Run the Pyxis worker from the container checkout instead of NeMo-Run's job directory."""
    import nemo_run as run

    command = " ".join([shlex.quote(task.entrypoint), shlex.quote(task.path), *task.args])
    return run.Script(
        inline=f"set -e\ncd /opt/Megatron-LM\nexec {command}\n", entrypoint="bash", env=task.env
    )
