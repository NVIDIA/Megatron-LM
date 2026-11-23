# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/08035723e7a86f94fc54a7f4473bd0519f2ac941/scripts/conversion/arguments.py
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
"""Argument definitions shared by the conversion launcher and worker."""

import argparse
import os
import sys
from pathlib import Path

COMMON_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "common"
if str(COMMON_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_SCRIPT_DIR))

from container_runtime import add_container_runtime_args  # noqa: E402
from slurm_parameters import add_slurm_parameter_args  # noqa: E402
from slurm_wait import MIN_POLL_INTERVAL, slurm_poll_interval  # noqa: E402

DTYPE_CHOICES = ("bfloat16", "float16", "float32")


def _add_execution_arguments(
    parser: argparse.ArgumentParser, *, default_device: str = "cpu"
) -> None:
    """Add NeMo Run execution arguments to a conversion subcommand."""
    execution = parser.add_argument_group("Execution")
    add_container_runtime_args(execution)
    add_slurm_parameter_args(execution)
    execution.add_argument(
        "--poll-interval",
        type=slurm_poll_interval,
        default=MIN_POLL_INTERVAL,
        help="Seconds between Slurm status checks when waiting (minimum/default: 60).",
    )
    execution.add_argument(
        "--executor",
        choices=("local", "slurm"),
        default="local",
        help="NeMo Run executor (default: local).",
    )
    execution.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default=default_device,
        help=f"Conversion backend (default: {default_device}).",
    )
    execution.add_argument("--nodes", type=int, default=1, help="Number of nodes (default: 1).")
    execution.add_argument(
        "--gpus-per-node",
        "--gpus_per_node",
        type=int,
        dest="gpus_per_node",
        help="GPUs per node; required for the GPU backend and optional as a CPU-backend runtime resource.",
    )
    execution.add_argument(
        "--cpu-processes-per-node",
        type=int,
        default=1,
        help=(
            "CPU conversion processes per node (default: 1). Values above 1 enable distributed CPU import "
            "and export and require model parallelism compatible with nodes*cpu-processes-per-node."
        ),
    )
    execution.add_argument(
        "--cpus-per-task", type=int, help="Slurm CPU cores per conversion process."
    )
    execution.add_argument(
        "--mem", default="0", help="Slurm memory request (default: 0, all node memory)."
    )
    execution.add_argument(
        "--account", default=os.environ.get("SLURM_ACCOUNT"), help="Slurm account."
    )
    execution.add_argument(
        "--partition", default=os.environ.get("SLURM_PARTITION"), help="Slurm partition."
    )
    execution.add_argument(
        "--time", default="04:00:00", help="Slurm time limit (default: 04:00:00)."
    )
    execution.add_argument("--gres", help="Optional Slurm GRES value for GPU jobs.")
    execution.add_argument(
        "--exclusive",
        action="store_true",
        help="Request exclusive Slurm nodes; by default conversion jobs may share nodes.",
    )
    execution.add_argument(
        "--no-gpu-resource-request",
        action="store_true",
        help="Do not emit a Slurm GPU/GRES request when the cluster allocates whole GPU nodes implicitly.",
    )
    execution.add_argument(
        "--container-image",
        default=os.environ.get("CONTAINER_IMAGE"),
        help="Slurm container image; defaults to CONTAINER_IMAGE.",
    )
    execution.add_argument(
        "--mount",
        action="append",
        default=[],
        help="Container mount as HOST or HOST:CONTAINER; may be repeated.",
    )
    execution.add_argument(
        "--env",
        action="append",
        default=[],
        help="Environment variable NAME to inherit; may be repeated. Values are not accepted.",
    )
    execution.add_argument(
        "--srun-arg",
        action="append",
        default=[],
        dest="srun_args",
        metavar="ARG",
        help="Additional cluster-specific argument passed to srun; may be repeated. Use --srun-arg=--flag.",
    )
    execution.add_argument("--experiment-name", help="NeMo Run experiment name.")
    execution.add_argument(
        "--submission-dry-run",
        "--dry-run",
        action="store_true",
        dest="submission_dry_run",
        help="Render the NeMo Run job without executing or submitting it.",
    )
    execution.add_argument(
        "--detach",
        action="store_true",
        help="Return after submitting a Slurm job instead of waiting for conversion to finish.",
    )


def _add_parallelism_arguments(
    parser: argparse.ArgumentParser, *, include_distributed_timeout: bool
) -> None:
    """Add distributed model-parallel arguments."""
    parallelism = parser.add_argument_group("Distributed parallelism")
    parallelism.add_argument(
        "-tp",
        "--tp",
        "--tensor-parallel-size",
        type=int,
        default=1,
        dest="tp",
        help="Tensor parallelism size (default: 1).",
    )
    parallelism.add_argument(
        "-pp",
        "--pp",
        "--pipeline-parallel-size",
        type=int,
        default=1,
        dest="pp",
        help="Pipeline parallelism size (default: 1).",
    )
    parallelism.add_argument(
        "-ep",
        "--ep",
        "--expert-parallel-size",
        type=int,
        default=1,
        dest="ep",
        help="Expert parallelism size (default: 1).",
    )
    parallelism.add_argument(
        "-etp",
        "--etp",
        "--expert-tensor-parallel-size",
        type=int,
        default=1,
        dest="etp",
        help="Expert tensor parallelism size (default: 1).",
    )
    if include_distributed_timeout:
        parallelism.add_argument(
            "--distributed-timeout-minutes",
            type=int,
            help="Distributed process-group timeout in minutes.",
        )


def _add_common_conversion_arguments(
    parser: argparse.ArgumentParser, *, include_execution: bool
) -> None:
    """Add arguments shared by import and export."""
    if include_execution:
        _add_execution_arguments(parser)
    else:
        parser.add_argument("--device", choices=("cpu", "gpu"), required=True)

    conversion = parser.add_argument_group("Conversion")
    conversion.add_argument(
        "--hf-model", required=True, help="Hugging Face model ID or local path."
    )
    conversion.add_argument(
        "--text-only",
        action="store_true",
        help="Convert only the language component of a supported multimodal checkpoint.",
    )
    conversion.add_argument(
        "--hf-revision",
        help="Immutable Hugging Face Hub revision to resolve before conversion (for example, a commit SHA).",
    )
    conversion.add_argument("--megatron-path", required=True, help="Megatron checkpoint path.")
    conversion.add_argument(
        "--torch-dtype",
        choices=DTYPE_CHOICES,
        default="bfloat16",
        help="Model precision (default: bfloat16).",
    )
    conversion.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom code from the Hugging Face model repository.",
    )
    conversion.add_argument(
        "--overwrite", action="store_true", help="Delete a non-empty destination before conversion."
    )

    _add_parallelism_arguments(parser, include_distributed_timeout=True)


def _add_roundtrip_arguments(parser: argparse.ArgumentParser, *, include_execution: bool) -> None:
    """Add round-trip arguments to the launcher or in-job worker parser."""
    if include_execution:
        _add_execution_arguments(parser, default_device="gpu")
    else:
        parser.add_argument("--device", choices=("gpu",), required=True)

    roundtrip = parser.add_argument_group("Round-trip validation")
    roundtrip.add_argument(
        "--hf-model",
        "--hf-model-id",
        required=True,
        dest="hf_model",
        help="Hugging Face model ID or local path.",
    )
    roundtrip.add_argument(
        "--hf-revision",
        help="Immutable Hugging Face Hub revision to resolve before conversion (for example, a commit SHA).",
    )
    roundtrip.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom code from the Hugging Face model repository.",
    )
    _add_parallelism_arguments(parser, include_distributed_timeout=True)


def build_parser(*, include_execution: bool) -> argparse.ArgumentParser:
    """Build the conversion argument parser.

    Args:
        include_execution: Include NeMo Run local and Slurm options for the
            user-facing launcher. The in-job worker omits those options.

    Returns:
        Parser for the user-facing launcher or conversion worker.
    """
    parser = argparse.ArgumentParser(
        description="Convert checkpoints between Hugging Face and Megatron formats.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
        epilog="""
Examples:
  # Local CPU import
  ./examples/bridge/conversion/convert.sh import --executor local --device cpu \\
      --hf-model meta-llama/Llama-3.2-1B --megatron-path /workspace/llama-megatron

  # Slurm distributed GPU export
  ./examples/bridge/conversion/convert.sh export --executor slurm --device gpu \\
      --nodes 1 --gpus-per-node 8 --account ACCOUNT --partition PARTITION \\
      --container-image IMAGE --mount /workspace --env HF_TOKEN \\
      --hf-model Qwen/Qwen3.5-35B-A3B --megatron-path /workspace/qwen/iter_0000000 \\
      --hf-path /workspace/qwen-hf --ep 8

  # Local multi-GPU round-trip validation
  ./examples/bridge/conversion/convert.sh roundtrip --executor local --device gpu \\
      --gpus-per-node 8 --hf-model Qwen/Qwen3.5-35B-A3B \\
      --ep 8

""",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser(
        "import", help="Import a Hugging Face model into a Megatron checkpoint.", allow_abbrev=False
    )
    _add_common_conversion_arguments(import_parser, include_execution=include_execution)

    export_parser = subparsers.add_parser(
        "export", help="Export a Megatron checkpoint into Hugging Face format.", allow_abbrev=False
    )
    _add_common_conversion_arguments(export_parser, include_execution=include_execution)
    export_parser.add_argument(
        "--hf-path", required=True, help="Destination Hugging Face checkpoint path."
    )
    export_parser.add_argument(
        "--no-progress", action="store_true", help="Disable export progress reporting."
    )
    export_parser.add_argument(
        "--not-strict",
        action="store_true",
        help="Allow source and destination checkpoints to have different parameter keys.",
    )
    export_parser.add_argument(
        "--distributed-save",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Let distributed ranks save assigned Hugging Face shards independently "
            "(default: enabled for GPU and distributed CPU export)."
        ),
    )
    export_parser.add_argument(
        "--save-every-n-ranks",
        type=int,
        default=1,
        help="Only every Nth rank writes files during distributed save (default: 1).",
    )
    export_parser.add_argument(
        "--export-weight-dtype",
        choices=DTYPE_CHOICES,
        help="Cast exported Hugging Face weights to this dtype (GPU backend only).",
    )

    roundtrip_parser = subparsers.add_parser(
        "roundtrip",
        help="Validate a Hugging Face to Megatron to Hugging Face conversion on GPUs.",
        allow_abbrev=False,
    )
    _add_roundtrip_arguments(roundtrip_parser, include_execution=include_execution)

    return parser


def conversion_worker_args(args: argparse.Namespace) -> list[str]:
    """Serialize parsed conversion options for the in-job worker.

    Args:
        args: Parsed user-facing launcher arguments.

    Returns:
        Command-line arguments accepted by ``run_conversion.py``.
    """
    if args.command == "roundtrip":
        worker_args = [
            "roundtrip",
            "--device",
            args.device,
            "--hf-model",
            args.hf_model,
            "--tp",
            str(args.tp),
            "--pp",
            str(args.pp),
            "--ep",
            str(args.ep),
            "--etp",
            str(args.etp),
        ]
        if args.hf_revision is not None:
            worker_args.extend(["--hf-revision", args.hf_revision])
        if args.trust_remote_code:
            worker_args.append("--trust-remote-code")
        if args.distributed_timeout_minutes is not None:
            worker_args.extend(
                ["--distributed-timeout-minutes", str(args.distributed_timeout_minutes)]
            )
        return worker_args

    worker_args = [
        args.command,
        "--device",
        args.device,
        "--hf-model",
        args.hf_model,
        "--megatron-path",
        args.megatron_path,
        "--torch-dtype",
        args.torch_dtype,
        "--tp",
        str(args.tp),
        "--pp",
        str(args.pp),
        "--ep",
        str(args.ep),
        "--etp",
        str(args.etp),
    ]
    if args.hf_revision is not None:
        worker_args.extend(["--hf-revision", args.hf_revision])
    if args.trust_remote_code:
        worker_args.append("--trust-remote-code")
    if args.overwrite:
        worker_args.append("--overwrite")
    if args.text_only:
        worker_args.append("--text-only")
    if args.distributed_timeout_minutes is not None:
        worker_args.extend(["--distributed-timeout-minutes", str(args.distributed_timeout_minutes)])

    if args.command == "export":
        worker_args.extend(["--hf-path", args.hf_path])
        if args.no_progress:
            worker_args.append("--no-progress")
        if args.not_strict:
            worker_args.append("--not-strict")
        distributed_cpu = args.device == "cpu" and args.cpu_processes_per_node > 1
        distributed_save = (
            args.distributed_save
            if args.distributed_save is not None
            else (args.device == "gpu" or distributed_cpu)
        )
        worker_args.append("--distributed-save" if distributed_save else "--no-distributed-save")
        worker_args.extend(["--save-every-n-ranks", str(args.save_every_n_ranks)])
        if args.export_weight_dtype is not None:
            worker_args.extend(["--export-weight-dtype", args.export_weight_dtype])
    return worker_args
