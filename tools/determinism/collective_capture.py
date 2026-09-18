# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Opt-in, bounded tensor snapshots for explicit-group TP/SP recipe replay.

Unlike metadata inventory, this adapter reads tensor contents and synchronizes
device work. Blobs stay beside the capture on the machine running the recipe.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import inspect
import json
import math
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import torch

if TYPE_CHECKING:
    from tools.determinism.capture_recipe import Inventory

MAPPINGS = {
    "copy": "copy_to_tensor_model_parallel_region",
    "reduce": "reduce_from_tensor_model_parallel_region",
    "gather_first": "gather_from_sequence_parallel_region",
    "scatter_first": "reduce_scatter_to_sequence_parallel_region",
    "gather_last": "all_gather_last_dim_from_tensor_parallel_region",
    "scatter_last": "reduce_scatter_last_dim_to_tensor_parallel_region",
}
NCCL_KEYS = (
    "NCCL_ALGO",
    "NCCL_PROTO",
    "NCCL_MAX_NCHANNELS",
    "NCCL_MIN_NCHANNELS",
    "NCCL_NVLS_ENABLE",
    "NCCL_COLLNET_ENABLE",
)
DTYPES = {str(dtype): dtype for dtype in (torch.float32, torch.bfloat16)}


def nccl_environment() -> dict:
    """Retain explicit NCCL/PyTorch communication overrides, including new keys."""
    keys = set(NCCL_KEYS) | {key for key in os.environ if key.startswith(("NCCL_", "TORCH_NCCL_"))}
    return {key: os.environ.get(key) for key in sorted(keys)}


def _options_signature(options) -> dict:
    if getattr(options, "split_from", None) is not None:
        raise ValueError("Split communicators need a separate adapter")
    flags = {}
    for key in ("use_pg_for_symm_mem_rendezvous", "enable_reconfigure"):
        if hasattr(options, key):
            value = getattr(options, key)
            if type(value) is not bool:
                raise ValueError("NCCL group flag is not a boolean")
            flags[key] = value
    config = {}
    for key in dir(options.config):
        if key.startswith("_"):
            continue
        value = getattr(options.config, key)
        if callable(value):
            continue
        if value is not None and type(value) not in (str, bool, int, float):
            raise ValueError("NCCL configuration contains an unsupported option")
        config[key] = value
    if type(options.is_high_priority_stream) is not bool:
        raise ValueError("NCCL stream priority is unavailable")
    signature = {
        "is_high_priority_stream": options.is_high_priority_stream,
        "config": config,
        "flags": flags,
    }
    json.dumps(signature, allow_nan=False)
    return signature


def process_group_options(group: torch.distributed.ProcessGroup, device: torch.device) -> dict:
    """Read options from the actual NCCL backend, not process-wide defaults."""
    try:
        return _options_signature(group._get_backend(device).options)
    except AttributeError as error:
        raise ValueError("NCCL process-group options are unavailable") from error


def restore_group_options(signature: dict) -> torch.distributed.ProcessGroupNCCL.Options:
    """Recreate the captured exposed NCCL configuration before group creation."""
    options = torch.distributed.ProcessGroupNCCL.Options(
        is_high_priority_stream=signature["is_high_priority_stream"]
    )
    defaults = _options_signature(options)
    if set(signature) != set(defaults) or any(
        set(signature[key]) != set(defaults[key]) for key in ('config', 'flags')
    ):
        raise ValueError("Captured NCCL option fields differ from this backend")
    for key, value in signature["config"].items():
        if value != defaults["config"][key]:
            setattr(options.config, key, value)
    for key, value in signature['flags'].items():
        if value != defaults['flags'][key]:
            setattr(options, key, value)
    if json.dumps(_options_signature(options), sort_keys=True) != json.dumps(
        signature, sort_keys=True
    ):
        raise ValueError("NCCL options could not be restored exactly")
    return options


def tensor_metadata(value: torch.Tensor) -> dict:
    """Describe logical bytes and the layout that must be restored for replay."""
    if value.layout != torch.strided or str(value.dtype) not in DTYPES or not value.numel():
        raise ValueError("Collective capture requires nonempty strided FP32/BF16 tensors")
    return {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "dtype": str(value.dtype),
        "requires_grad": value.requires_grad,
        "storage_offset": value.storage_offset(),
    }


def storage_elements(metadata: dict) -> int:
    """Bound restored storage; allow broadcast views but reject other overlap."""
    shape, stride, offset = metadata["shape"], metadata["stride"], metadata["storage_offset"]
    if (
        not shape
        or len(shape) != len(stride)
        or any(type(n) is not int or n < 1 for n in shape)
        or any(type(s) is not int or s < 0 for s in stride)
        or type(offset) is not int
        or offset < 0
        or metadata["dtype"] not in DTYPES
        or type(metadata["requires_grad"]) is not bool
    ):
        raise ValueError("Invalid captured tensor metadata")
    extent = 1
    for step, size in sorted((s, n) for n, s in zip(shape, stride) if n > 1 and s):
        if step < extent:
            raise ValueError("Overlapping tensor strides are unsupported")
        extent += (size - 1) * step
    return offset + extent


def load_tensor(
    root: Path, metadata: dict, *, max_bytes: int, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Verify a content-addressed blob and restore exact logical layout."""
    if set(metadata) != {"shape", "stride", "dtype", "requires_grad", "storage_offset", "sha256"}:
        raise ValueError("Unsupported captured tensor descriptor")
    elements = storage_elements(metadata)
    dtype = DTYPES[metadata["dtype"]]
    itemsize = dtype.itemsize
    logical_bytes = math.prod(metadata["shape"]) * itemsize
    if max(elements * itemsize, logical_bytes) > max_bytes:
        raise ValueError("Captured tensor exceeds the replay byte limit")
    digest = metadata["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("Invalid tensor digest")
    path = root / (digest + ".bin")
    if path.stat().st_size != logical_bytes:
        raise ValueError("Captured tensor size differs")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("Captured tensor digest differs")
    logical = torch.frombuffer(bytearray(payload), dtype=dtype).reshape(metadata["shape"])
    flat = torch.zeros(elements, dtype=dtype, device=device)
    result = flat.as_strided(metadata["shape"], metadata["stride"], metadata["storage_offset"])
    destination, source = result, logical
    for dim, (size, step) in enumerate(zip(metadata["shape"], metadata["stride"])):
        if size > 1 and step == 0:
            first = source.narrow(dim, 0, 1)
            # Validate broadcast bytes as well as values (including NaN payloads).
            expanded = (
                first.expand_as(source)
                .clone(memory_format=torch.contiguous_format)
                .reshape(-1)
                .view(torch.uint8)
            )
            if not torch.equal(
                expanded,
                source.clone(memory_format=torch.contiguous_format).reshape(-1).view(torch.uint8),
            ):
                raise ValueError("Captured bytes disagree with broadcast strides")
            source = first
            destination = destination.narrow(dim, 0, 1)
    destination.copy_(source)
    return result.requires_grad_(metadata["requires_grad"])


def load_captures(root: Path, *, max_bytes: int) -> list[dict]:
    """Validate all rank schedules and blobs before any replay collective starts.

    The initial replay protocol requires the same mapping/phase order on every
    global rank, allowing disjoint equal-sized TP groups. Other schedules need
    an explicit adapter; ranks are never pooled or silently reordered.
    """
    from tools.determinism.capture_recipe import input_signature
    from tools.determinism.recipe_coverage import signature_key

    reports = []
    for path in sorted(root.glob("rank-*/manifest.json")):
        report = json.loads(path.read_text())
        if type(report.get("rank")) is not int or path.parent.name != f"rank-{report['rank']}":
            raise ValueError("Collective manifest rank differs from its directory")
        reports.append(report)
    if not reports:
        raise ValueError("No collective capture manifests found")
    context = reports[0]["context"]
    world = context["world_size"]
    if type(world) is not int or world < 2 or context.get("dirty", True):
        raise ValueError("Collective replay requires a clean multi-rank capture")
    if len(reports) != world or {r["rank"] for r in reports} != set(range(world)):
        raise ValueError("Collective capture ranks are incomplete or duplicated")
    reports.sort(key=lambda report: report["rank"])
    count = len(reports[0]["events"])
    if not count:
        raise ValueError("No collective events were captured")
    for report in reports:
        if (
            report.get("schema_version") != 1
            or report.get("kind") != "collective_recipe_capture"
            or report["context"] != context
            or report.get("context_after") != context
            or report["recipe_id"] != reports[0]["recipe_id"]
            or report.get("complete") is not True
            or report.get("truncated") is not False
            or report.get("capture_issues")
            or len(report["events"]) != count
        ):
            raise ValueError("Incomplete or inconsistent collective capture")
        rank = report["rank"]
        forwards = {}
        bytes_read: dict[str, int] = {}
        for event in report["events"]:
            signature = event["signature"]
            signature_key(signature)
            collective = signature["configuration"]["collective"]
            fields = {
                "capture_schema",
                "case",
                "group_ranks",
                "group_rank",
                "size",
                "backend",
                "nccl_version",
                "nccl_environment",
                "device_uuid",
                "input",
                "grad_enabled",
                "warn_only",
                "group_options",
            }
            if signature["phase"] == "forward_backward":
                fields.update(("gradient", "backward_grad_enabled"))
            if set(collective) != fields or set(signature["configuration"]) != {"collective"}:
                raise ValueError("Unsupported collective capture fields")
            if type(collective["grad_enabled"]) is not bool or collective["warn_only"] is not False:
                raise ValueError("Collective capture requires an explicit strict execution policy")
            case = collective["case"]
            members = collective["group_ranks"]
            if (
                case not in MAPPINGS
                or collective.get("capture_schema") != 1
                or signature["op_id"] != "tensor_parallel_mappings"
                or signature["implementation"] != "mcore:" + MAPPINGS[case]
                or any(type(member) is not int or not 0 <= member < world for member in members)
                or members != sorted(set(members))
                or collective["size"] != len(members)
                or len(members) < 2
                or rank not in members
                or collective["group_rank"] != members.index(rank)
            ):
                raise ValueError("Invalid collective identity or group assignment")
            if type(signature["runtime"].get("fill_uninitialized_memory")) is not bool:
                raise ValueError("Collective capture lacks explicit memory-fill policy")
            if any(
                key in signature
                for key in (
                    'backward_runtime',
                    'backward_deterministic_algorithms',
                    'backward_collective',
                )
            ):
                raise ValueError(
                    "Mixed forward/backward runtime requires a separate replay adapter"
                )
            local = load_tensor(root / f"rank-{rank}", collective["input"], max_bytes=max_bytes)
            if input_signature((local,), torch.Tensor) != signature["inputs"]:
                raise ValueError("Captured input metadata differs from its signature")
            descriptors = [collective["input"]]
            call_id = event["call_id"]
            if type(call_id) is not int or call_id < 0:
                raise ValueError("Invalid collective invocation ID")
            if signature["phase"] == "forward":
                if call_id in forwards or "gradient" in collective:
                    raise ValueError("Duplicate or invalid forward event")
                forwards[call_id] = collective
            else:
                if (
                    call_id not in forwards
                    or {
                        k: v
                        for k, v in collective.items()
                        if k not in ("gradient", "backward_grad_enabled")
                    }
                    != forwards[call_id]
                    or collective.get("backward_grad_enabled") is not False
                    or collective["grad_enabled"] is not True
                    or not local.requires_grad
                ):
                    raise ValueError(
                        "Backward event lacks its forward or uses higher-order autograd"
                    )
                gradient = load_tensor(
                    root / f"rank-{rank}", collective["gradient"], max_bytes=max_bytes
                )
                if gradient.requires_grad or gradient.dtype != local.dtype:
                    raise ValueError("Unsupported collective upstream gradient")
                descriptors.append(collective["gradient"])
            for descriptor in descriptors:
                bytes_read[descriptor["sha256"]] = (
                    math.prod(descriptor["shape"]) * DTYPES[descriptor["dtype"]].itemsize
                )
            if sum(bytes_read.values()) > max_bytes:
                raise ValueError("Collective capture exceeds the replay byte limit")
    for index in range(count):
        events = [report["events"][index] for report in reports]
        signatures = [event["signature"] for event in events]
        collectives = [signature["configuration"]["collective"] for signature in signatures]
        if (
            len(
                {
                    (e["call_id"], s["implementation"], s["phase"])
                    for e, s in zip(events, signatures)
                }
            )
            != 1
        ):
            raise ValueError(
                "Collective replay requires the same invocation/phase order on every rank"
            )
        for collective in collectives:
            members = collective["group_ranks"]
            for member in members:
                peer = collectives[member]
                if any(
                    peer[key] != collective[key]
                    for key in (
                        "case",
                        "group_ranks",
                        "size",
                        "backend",
                        "nccl_version",
                        "nccl_environment",
                        "group_options",
                    )
                ) or any(
                    peer["input"][key] != collective["input"][key] for key in ("shape", "dtype")
                ):
                    raise ValueError("Collective peer contracts differ")
            output_shape = collective["input"]["shape"][:]
            case = collective["case"]
            dim = 0 if case.endswith("first") else len(output_shape) - 1
            if case.startswith("gather"):
                output_shape[dim] *= len(members)
            elif case.startswith("scatter"):
                if output_shape[dim] % len(members):
                    raise ValueError("Collective shards must divide equally")
                output_shape[dim] //= len(members)
            if "gradient" in collective and collective["gradient"]["shape"] != output_shape:
                raise ValueError("Collective upstream gradient shape differs")
    return reports


def prepare_replay(
    event: dict, root: Path, group: torch.distributed.ProcessGroup, *, max_bytes: int
) -> tuple:
    """Materialize one event and require the actual GPU policy/group to match.

    Call after all-rank bundle validation. The distributed caller must exchange
    preparation errors across ranks before entering any mapping collective.
    Runtime mismatches are rejected, never fixed by relabelling the capture.
    """
    from tools.determinism.capture_recipe import runtime_signature

    signature = event["signature"]
    collective = signature["configuration"]["collective"]
    if (
        runtime_signature(torch) != signature["runtime"]
        or torch.are_deterministic_algorithms_enabled() != signature["deterministic_algorithms"]
        or torch.is_deterministic_algorithms_warn_only_enabled()
        or "backward_runtime" in signature
        or "backward_deterministic_algorithms" in signature
        or "backward_collective" in signature
    ):
        raise ValueError("Replay runtime differs from the captured policy")
    actual = {
        "group_ranks": torch.distributed.get_process_group_ranks(group),
        "group_rank": group.rank(),
        "size": group.size(),
        "backend": str(torch.distributed.get_backend(group)),
        "nccl_version": list(torch.cuda.nccl.version()),
        "device_uuid": str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid),
        "nccl_environment": nccl_environment(),
        "group_options": process_group_options(
            group, torch.device("cuda", torch.cuda.current_device())
        ),
    }
    if actual["backend"] != "nccl" or any(
        value != collective[key] for key, value in actual.items()
    ):
        raise ValueError("Replay group or NCCL policy differs from the capture")
    local = load_tensor(root, collective["input"], max_bytes=max_bytes, device="cuda")
    gradient = (
        load_tensor(root, collective["gradient"], max_bytes=max_bytes, device="cuda")
        if signature["phase"] == "forward_backward"
        else None
    )
    mappings = importlib.import_module("megatron.core.tensor_parallel.mappings")
    return getattr(mappings, MAPPINGS[collective["case"]]), local, gradient


class CollectiveCapture:
    """Keep tensor blobs and ordered invocations within explicit capture limits."""

    def __init__(self, root: Path, *, max_bytes: int, max_events: int) -> None:
        if max_bytes < 1 or max_events < 1:
            raise ValueError("Positive collective capture limits are required")
        root.mkdir(parents=True, exist_ok=False)
        self.root = root
        self.max_bytes = max_bytes
        self.max_events = max_events
        self.bytes_written = 0
        self.calls = 0
        self.events: list[dict] = []
        self.issues: set[str] = set()
        self._snapshots = 0
        self._lock = threading.Lock()

    def next_call_id(self) -> int:
        """Assign a unique invocation ID even when Python callers overlap."""
        with self._lock:
            call_id = self.calls
            self.calls += 1
            return call_id

    def snapshot(self, value: torch.Tensor) -> dict:
        """Save logical bytes before a collective can modify the source tensor."""
        with self._lock:
            return self._snapshot(value)

    def _snapshot(self, value: torch.Tensor) -> dict:
        metadata = tensor_metadata(value)
        size = value.numel() * value.element_size()
        if max(size, storage_elements(metadata) * value.element_size()) > self.max_bytes:
            raise ValueError("Collective tensor exceeds the capture byte limit")
        if self._snapshots >= self.max_events or self.bytes_written + size > self.max_bytes:
            raise ValueError("Collective capture limit reached")
        self._snapshots += 1
        payload = (
            value.detach()
            .cpu()
            .clone(memory_format=torch.contiguous_format)
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
        digest = hashlib.sha256(payload).hexdigest()
        path = self.root / (digest + ".bin")
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(payload)
            self.bytes_written += size
        metadata["sha256"] = digest
        return metadata

    def write(self, inventory_report: dict) -> None:
        """Keep unfinished/unsupported captures visibly ineligible for replay."""
        report = {
            key: inventory_report[key]
            for key in ("schema_version", "recipe_id", "rank", "context", "complete", "truncated")
        }
        report.update(
            kind="collective_recipe_capture",
            events=self.events,
            capture_issues=sorted(self.issues),
            bytes_written=self.bytes_written,
        )
        if "context_after" in inventory_report:
            report["context_after"] = inventory_report["context_after"]
        temporary = self.root / "manifest.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.root / "manifest.json")


def wrap_collective(inventory: Inventory, function: Callable, binding: dict) -> Callable:
    """Capture supported mapping calls with their actual group and tensor bytes."""
    from tools.determinism.capture_recipe import input_signature, runtime_signature

    store = inventory.collectives
    if store is None:
        raise ValueError("Collective bindings require --collective-capture")
    mappings = importlib.import_module("megatron.core.tensor_parallel.mappings")
    cases = [case for case, name in MAPPINGS.items() if function is getattr(mappings, name)]
    if len(cases) != 1:
        raise ValueError("Collective bindings must select a supported MCore mapping or its alias")
    case = cases[0]
    if (
        binding["op_id"] != "tensor_parallel_mappings"
        or binding["implementation"] != "mcore:" + MAPPINGS[case]
    ):
        raise ValueError("Collective binding operation/implementation does not match the callable")
    fallback = inventory.wrap(function, {k: v for k, v in binding.items() if k != "adapter"})
    parameters = inspect.signature(function)

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        bound = parameters.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        call_id = store.next_call_id()
        try:
            group = values.get("group")
            if group is None or group.size() < 2:
                raise ValueError("Collective capture requires an explicit multi-rank group")
            for key, expected in (
                ("tensor_parallel_output_grad", True),
                ("output_split_sizes", None),
                ("input_split_sizes", None),
                ("use_global_buffer", False),
            ):
                if key in values and values[key] is not expected:
                    raise ValueError(
                        "Collective capture requires equal shards and default mapping options"
                    )
            ranks = torch.distributed.get_process_group_ranks(group)
            rank = group.rank()
            if (
                len(ranks) != group.size()
                or len(set(ranks)) != len(ranks)
                or not 0 <= rank < len(ranks)
                or ranks[rank] != torch.distributed.get_rank()
            ):
                raise ValueError("Collective group membership is inconsistent")
            local = values["input_"]
            metadata = store.snapshot(local)
            collective = {
                "capture_schema": 1,
                "case": case,
                "group_ranks": ranks,
                "group_rank": rank,
                "size": group.size(),
                "backend": str(torch.distributed.get_backend(group)),
                "nccl_version": list(torch.cuda.nccl.version()) if local.is_cuda else None,
                "device_uuid": (
                    str(torch.cuda.get_device_properties(local.device).uuid)
                    if local.is_cuda
                    else None
                ),
                "nccl_environment": nccl_environment(),
                "group_options": (
                    process_group_options(group, local.device) if local.is_cuda else None
                ),
                "input": metadata,
                "grad_enabled": torch.is_grad_enabled(),
                "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            }
            signature = {
                "op_id": binding["op_id"],
                "implementation": binding["implementation"],
                "phase": "forward",
                "inputs": input_signature((local,), torch.Tensor),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "runtime": runtime_signature(torch),
                "configuration": {"collective": collective},
            }
        except ValueError as error:
            store.issues.add(str(error))
            return fallback(*args, **kwargs)
        result = function(*args, **kwargs)
        if not isinstance(result, torch.Tensor):
            raise TypeError("A collective mapping must return a tensor")
        inventory.record(signature, binding["target"])
        store.events.append({"call_id": call_id, "signature": signature, "site": binding["target"]})

        def backward(gradient):
            try:
                metadata = store.snapshot(gradient)
                backward_signature = {
                    **signature,
                    "phase": "forward_backward",
                    "configuration": {
                        "collective": {
                            **collective,
                            "gradient": metadata,
                            "backward_grad_enabled": torch.is_grad_enabled(),
                        }
                    },
                }
                runtime = runtime_signature(torch)
                mode = torch.are_deterministic_algorithms_enabled()
                communication = {
                    'nccl_environment': nccl_environment(),
                    'group_options': (
                        process_group_options(group, gradient.device) if gradient.is_cuda else None
                    ),
                }
                if any(value != collective[key] for key, value in communication.items()):
                    backward_signature['backward_collective'] = communication
                if runtime != signature["runtime"] or mode != signature["deterministic_algorithms"]:
                    backward_signature.update(
                        backward_runtime=runtime, backward_deterministic_algorithms=mode
                    )
                inventory.record(backward_signature, binding["target"])
                store.events.append(
                    {"call_id": call_id, "signature": backward_signature, "site": binding["target"]}
                )
            except ValueError as error:
                store.issues.add(str(error))
            return gradient

        if result.requires_grad:
            result.register_hook(backward)
        return result

    return wrapped
