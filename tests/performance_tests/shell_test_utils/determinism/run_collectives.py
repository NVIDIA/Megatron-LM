# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""One fresh torchrun arm for actual captured TP/SP operator phase timings."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from datetime import timedelta
from pathlib import Path

from benchmark import _source
from collective_case import (
    ADAPTER,
    configure_policy,
    digest,
    install_head_helpers,
    mode_context,
    mode_signature,
)


def clone_tensor(torch, value):
    """Clone bounded captured storage, retaining offsets and broadcast strides.

    Avoid importing the pytest replay package: its initialization pins the
    deterministic environment and would invalidate the default timing arm.
    """
    if value is None:
        return None
    storage = torch.empty(0, dtype=value.dtype, device=value.device).set_(value.untyped_storage())
    return (
        storage.clone()
        .as_strided(value.shape, value.stride(), value.storage_offset())
        .requires_grad_(value.requires_grad)
    )


def measure(
    torch,
    function,
    local,
    gradient,
    group,
    *,
    phase: str,
    grad_enabled: bool,
    warmup: int,
    steps: int,
) -> list[float]:
    """Isolated CUDA-event intervals, with setup and rank alignment outside them.

    These direct mappings use synchronous c10d calls, which join NCCL work to
    the calling CUDA stream. The end event therefore includes its completion.
    Intervals include launch gaps and rank arrival skew; they are not pure NCCL
    kernel durations. No profiler or CUDA graph runs in these measurements.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for index in range(warmup + steps):
        # Reductions can mutate input/gradient storage. Restore every sample.
        value, upstream = clone_tensor(torch, local), clone_tensor(torch, gradient)
        with torch.set_grad_enabled(grad_enabled):
            output = function(value, group=group) if phase == "backward" else None
            torch.cuda.synchronize()
            torch.distributed.barrier()
            torch.cuda.synchronize()
            start.record()
            if phase == "backward":
                result = torch.autograd.grad(output, (value,), grad_outputs=upstream)
            else:
                result = function(value, group=group)
            end.record()
            end.synchronize()
            elapsed = start.elapsed_time(end)
        if index >= warmup:
            samples.append(elapsed)
        del result, output, value, upstream
    return samples


def _exchange_error(torch, error, world):
    errors = [None] * world
    torch.distributed.all_gather_object(errors, error)
    if any(errors):
        raise ValueError(f"Collective timing preparation failed: {errors}")


def main(argv: list[str] | None = None) -> int:
    """Validate all rank schedules before mapping calls and preserve raw samples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args(argv)
    request = json.loads(args.request.read_text())
    measurement, mode = request["measurement"], request["mode"]
    head = Path(request["head_checkout"])
    capture_root = Path(request["capture"])
    install_head_helpers(head)

    import torch

    from tools.determinism.capture_recipe import runtime_signature, source_context
    from tools.determinism.collective_capture import (
        load_captures,
        prepare_replay,
        restore_group_options,
    )

    captures = load_captures(capture_root, max_bytes=measurement["max_bytes"])
    world, rank = len(captures), int(os.environ["RANK"])
    if int(os.environ["WORLD_SIZE"]) != world or int(os.environ["LOCAL_WORLD_SIZE"]) != world:
        raise ValueError("This adapter requires the original single-node allocation")
    if _source(Path.cwd()) != request["source"]:
        raise ValueError("Selected production checkout changed")
    if any(digest(head / path) != expected for path, expected in measurement["tooling"].items()):
        raise ValueError("Head timing helpers changed")
    first = captures[rank]["events"][measurement["event_indices"][0]]["signature"]
    configure_policy(head, first["runtime"], mode)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=5))
    groups = {}
    try:
        context = source_context(torch)
        error = None
        if context != mode_context(captures[rank]["context"], mode, request["source"]["revision"]):
            error = "Actual source/runtime differs from requested capture policy"
        _exchange_error(torch, error, world)
        specifications = {}
        for capture in captures:
            for index in measurement["event_indices"]:
                collective = capture["events"][index]["signature"]["configuration"]["collective"]
                key = json.dumps(
                    [collective["group_ranks"], collective["group_options"]], sort_keys=True
                )
                specifications[key] = collective
        for key, collective in sorted(specifications.items()):
            group = torch.distributed.new_group(
                ranks=collective["group_ranks"],
                backend="nccl",
                pg_options=restore_group_options(collective["group_options"]),
                timeout=timedelta(minutes=5),
            )
            if rank in collective["group_ranks"]:
                groups[key] = group
        result: dict = {
            "adapter": ADAPTER,
            "rank": rank,
            "mode": mode,
            "measurement": measurement,
            "context": context,
            "rows": {},
            "tooling": measurement["tooling"],
            "capture_manifest_sha256": digest(capture_root / f"rank-{rank}" / "manifest.json"),
        }
        for index in measurement["event_indices"]:
            event = captures[rank]["events"][index]
            signature = mode_signature(event["signature"], mode)
            collective = signature["configuration"]["collective"]
            key = json.dumps(
                [collective["group_ranks"], collective["group_options"]], sort_keys=True
            )
            group = groups[key]
            prepared, error = None, None
            try:
                prepared = prepare_replay(
                    {**event, "signature": signature},
                    capture_root / f"rank-{rank}",
                    group,
                    max_bytes=measurement["max_bytes"],
                )
                mappings = importlib.import_module("megatron.core.tensor_parallel.mappings")
                if (
                    not mappings.__file__
                    or Path(mappings.__file__).resolve()
                    != Path.cwd() / "megatron/core/tensor_parallel/mappings.py"
                ):
                    raise ValueError("Mapping was imported from a different production checkout")
            except (ValueError, RuntimeError, OSError) as caught:
                error = f"{type(caught).__name__}: {caught}"
            _exchange_error(torch, error, world)
            if prepared is None:
                raise ValueError("Collective preparation produced no inputs")
            function, local, gradient = prepared
            phase = "backward" if signature["phase"] == "forward_backward" else "forward"
            actual = {
                **signature,
                "runtime": runtime_signature(torch),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            }
            samples = measure(
                torch,
                function,
                local,
                gradient,
                group,
                phase=phase,
                grad_enabled=collective["grad_enabled"],
                warmup=measurement["warmup"],
                steps=measurement["steps"],
            )
            # Recheck runtime, group and captured blobs after uninstrumented timing.
            prepare_replay(
                {**event, "signature": signature},
                capture_root / f"rank-{rank}",
                group,
                max_bytes=measurement["max_bytes"],
            )
            result["rows"][str(index)] = {
                "capture_signature": event["signature"],
                "actual_signature": actual,
                "signature_after": {
                    **signature,
                    "runtime": runtime_signature(torch),
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                },
                "phase": phase,
                "samples_ms": samples,
            }
        result["context_after"] = source_context(torch)
        if any(
            digest(head / path) != expected for path, expected in measurement["tooling"].items()
        ):
            raise ValueError("Head timing helpers changed during the arm")
        path = args.request.parent / f"rank-{rank}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
        torch.distributed.barrier()
    finally:
        torch.cuda.synchronize()
        for group in groups.values():
            torch.distributed.destroy_process_group(group)
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
