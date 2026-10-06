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
    nproc_per_node: int | None
    coordinator_host: str | None
    coordinator_port: int | None
    worker_id_file: str | None
    megatron_root: str
    drain_timeout: float
    megatron_argv: list[str]
    engine_launch_mode: str = "local"
    engine_start_timeout: float = 1800.0
    engine_shutdown_timeout: float = 30.0
    parent_event_host: str = "127.0.0.1"
    parent_event_port: int | None = None
    endpoint_types: str = "chat,completions"
    handoff_journal: str | None = None
    handoff_owner: str | None = None


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
        "--engine-launch-mode",
        choices=["local", "external"],
        default="local",
        help=(
            "Launch a one-node Megatron rank group from this worker, or wait for an engine "
            "service launched by the deployment system."
        ),
    )
    parser.add_argument("--nproc-per-node", type=int, default=None)
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
        "--handoff-journal",
        help="Persistent SQLite cleanup journal; required for externally managed decode.",
    )
    parser.add_argument(
        "--handoff-owner",
        help="Supervisor-assigned unique launch-attempt ID; never reuse across restarts.",
    )
    parser.add_argument(
        "--parent-event-host",
        default="127.0.0.1",
        help="Interface or hostname for the parent-owned engine event socket.",
    )
    parser.add_argument(
        "--parent-event-port",
        type=int,
        default=None,
        help="Fixed engine-event port required when --engine-launch-mode=external.",
    )
    args = parser.parse_args(dynamo_argv)

    if args.nproc_per_node is not None and args.nproc_per_node < 1:
        parser.error("--nproc-per-node must be at least 1")
    if args.engine_launch_mode == "local" and args.nproc_per_node is None:
        parser.error("--engine-launch-mode local requires --nproc-per-node")
    if args.engine_launch_mode == "external" and args.parent_event_port is None:
        parser.error("--engine-launch-mode external requires --parent-event-port")
    if bool(args.handoff_journal) != bool(args.handoff_owner):
        parser.error("--handoff-journal and --handoff-owner must be supplied together")
    if args.handoff_journal and args.role != "decode":
        parser.error("--handoff-journal is only supported for decode workers")
    if args.role == "decode" and args.engine_launch_mode == "external" and not args.handoff_journal:
        parser.error("externally managed decode requires --handoff-journal and --handoff-owner")
    if args.parent_event_port is not None and not 1 <= args.parent_event_port <= 65535:
        parser.error("--parent-event-port must be between 1 and 65535")
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
        engine_launch_mode=args.engine_launch_mode,
        nproc_per_node=args.nproc_per_node,
        coordinator_host=args.coordinator_host,
        coordinator_port=args.coordinator_port,
        worker_id_file=args.worker_id_file,
        megatron_root=args.megatron_root,
        drain_timeout=args.drain_timeout,
        megatron_argv=megatron_argv,
        engine_start_timeout=args.engine_start_timeout,
        engine_shutdown_timeout=args.engine_shutdown_timeout,
        parent_event_host=args.parent_event_host,
        parent_event_port=args.parent_event_port,
        endpoint_types=args.endpoint_types,
        handoff_journal=args.handoff_journal,
        handoff_owner=args.handoff_owner,
    )
