# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Command-line interface for the isolated review stages."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .common import ReviewError, parse_trigger
from .context import build_context, serve
from .prepare import prepare_event
from .publisher import publish, report_schema, validate_report_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    trigger_parser = subparsers.add_parser("parse-trigger")
    trigger_parser.add_argument("body")
    prepare_parser = subparsers.add_parser("prepare-event")
    prepare_parser.add_argument("--event", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--acknowledge", action="store_true")
    context_parser = subparsers.add_parser("build-context")
    context_parser.add_argument("--repo", type=Path, required=True)
    context_parser.add_argument("--metadata", type=Path, required=True)
    context_parser.add_argument("--output-dir", type=Path, required=True)
    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument("--context-dir", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--context-dir", type=Path, required=True)
    validate_parser.add_argument("--report", type=Path, required=True)
    validate_parser.add_argument("--output", type=Path)
    publish_parser = subparsers.add_parser("publish")
    publish_parser.add_argument("--context-dir", type=Path, required=True)
    publish_parser.add_argument("--report", type=Path)
    publish_parser.add_argument(
        "--analysis-result",
        choices=["success", "failed", "invalid", "timed_out"],
        default="success",
    )
    subparsers.add_parser("schema")
    args = parser.parse_args(argv)
    try:
        if args.command == "parse-trigger":
            print(json.dumps(parse_trigger(args.body)))
        elif args.command == "prepare-event":
            prepare_event(args.event, args.output, args.acknowledge)
        elif args.command == "build-context":
            print(
                json.dumps(build_context(args.repo, args.metadata, args.output_dir), sort_keys=True)
            )
        elif args.command == "serve":
            serve(args.context_dir)
        elif args.command == "validate":
            validate_report_file(args.report, args.context_dir, args.output)
        elif args.command == "publish":
            print(publish(args.context_dir, args.report, args.analysis_result))
        elif args.command == "schema":
            print(json.dumps(report_schema(), separators=(",", ":")))
    except (
        ReviewError,
        OSError,
        ValueError,
        json.JSONDecodeError,
        subprocess.TimeoutExpired,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0
