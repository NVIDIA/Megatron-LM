# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Arguments shared by the Megatron launcher and unified backend."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass


@dataclass
class Config:
    model: str
    served_model_name: str
    namespace: str
    component: str
    endpoint: str
    discovery_backend: str
    request_plane: str
    event_plane: str | None
    role: str
    nproc_per_node: int
    coordinator_host: str | None
    coordinator_port: int | None
    worker_id_file: str | None
    megatron_root: str
    drain_timeout: float
    megatron_argv: list[str]
    launcher: str = "local"
    nnodes: int = 1
    master_addr: str | None = None
    master_port: int | None = None
    slurm_nodelist: str | None = None
    engine_start_timeout: float = 1800.0
    engine_shutdown_timeout: float = 30.0
    parent_event_host: str = "127.0.0.1"
    endpoint_types: str = "chat,completions"


def _split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" not in argv:
        return argv, []
    separator = argv.index("--")
    return argv[:separator], argv[separator + 1 :]


def add_engine_service_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add arguments shared by the Dynamo parent and Megatron child service."""

    parser.add_argument("--role", choices=["aggregated", "prefill", "decode"], default="aggregated")
    parser.add_argument("--coordinator-host", default=None)
    parser.add_argument("--coordinator-port", type=int, default=None)
    return parser


def parse_args(argv: list[str] | None = None) -> Config:
    dynamo_argv, megatron_argv = _split_argv(list(sys.argv[1:] if argv is None else argv))
    parser = argparse.ArgumentParser(
        prog="python -m megatron.inference.integrations.dynamo",
        description="Launch one DP=1 Megatron rank group as a Dynamo backend worker.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--namespace", default="dynamo")
    parser.add_argument("--component", default=None)
    parser.add_argument("--endpoint", default="generate")
    parser.add_argument(
        "--endpoint-types",
        choices=["chat", "completions", "chat,completions"],
        default="chat,completions",
        help=(
            "OpenAI endpoint surfaces advertised to Dynamo. Use 'completions' "
            "for base models that do not define a chat template."
        ),
    )
    parser.add_argument("--discovery-backend", default="etcd")
    parser.add_argument("--request-plane", default="nats")
    parser.add_argument("--event-plane", default="nats")
    add_engine_service_args(parser)
    parser.add_argument(
        "--launcher",
        choices=["local", "slurm"],
        default="local",
        help="Launch the owned Megatron rank group locally or through a SLURM job step.",
    )
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--nproc-per-node", type=int, required=True)
    parser.add_argument("--master-addr", default=None)
    parser.add_argument("--master-port", type=int, default=None)
    parser.add_argument(
        "--slurm-nodelist",
        default=None,
        help="Optional SLURM node list reserved for this complete Megatron replica.",
    )
    parser.add_argument(
        "--worker-id-file",
        default=None,
        help="Write this worker's assigned Dynamo identity as JSON after engine readiness.",
    )
    parser.add_argument("--megatron-root", default="/opt/megatron-lm")
    parser.add_argument("--drain-timeout", type=float, default=30.0)
    parser.add_argument("--engine-start-timeout", type=float, default=1800.0)
    parser.add_argument("--engine-shutdown-timeout", type=float, default=30.0)
    parser.add_argument(
        "--parent-event-host",
        default="127.0.0.1",
        help="Interface or hostname for the parent-owned engine event socket.",
    )
    args = parser.parse_args(dynamo_argv)

    if args.nproc_per_node < 1:
        parser.error("--nproc-per-node must be at least 1")
    if args.nnodes < 1:
        parser.error("--nnodes must be at least 1")
    if args.launcher == "local" and args.nnodes != 1:
        parser.error("--launcher local only supports --nnodes 1")
    if args.launcher == "slurm" and (not args.master_addr or args.master_port is None):
        parser.error("--launcher slurm requires --master-addr and --master-port")
    if args.master_port is not None and not 1 <= args.master_port <= 65535:
        parser.error("--master-port must be between 1 and 65535")
    if (
        args.launcher == "slurm"
        and args.nnodes > 1
        and args.parent_event_host in {"127.0.0.1", "::1", "localhost"}
    ):
        parser.error(
            "multi-node --launcher slurm requires --parent-event-host to be a routable "
            "address on the Dynamo parent host"
        )
    if args.engine_start_timeout <= 0:
        parser.error("--engine-start-timeout must be positive")
    if args.engine_shutdown_timeout <= 0:
        parser.error("--engine-shutdown-timeout must be positive")
    if not megatron_argv:
        parser.error("Megatron arguments are required after '--'")
    if args.role in ("prefill", "decode") and not args.coordinator_host:
        parser.error("disaggregated roles require a routable --coordinator-host")

    component = args.component
    if component is None:
        component = "prefill" if args.role == "prefill" else "backend"
    return Config(
        model=args.model,
        served_model_name=args.served_model_name or args.model,
        namespace=args.namespace,
        component=component,
        endpoint=args.endpoint,
        discovery_backend=args.discovery_backend,
        request_plane=args.request_plane,
        event_plane=args.event_plane,
        role=args.role,
        launcher=args.launcher,
        nnodes=args.nnodes,
        nproc_per_node=args.nproc_per_node,
        master_addr=args.master_addr,
        master_port=args.master_port,
        slurm_nodelist=args.slurm_nodelist,
        coordinator_host=args.coordinator_host,
        coordinator_port=args.coordinator_port,
        worker_id_file=args.worker_id_file,
        megatron_root=args.megatron_root,
        drain_timeout=args.drain_timeout,
        megatron_argv=megatron_argv,
        engine_start_timeout=args.engine_start_timeout,
        engine_shutdown_timeout=args.engine_shutdown_timeout,
        parent_event_host=args.parent_event_host,
        endpoint_types=args.endpoint_types,
    )
