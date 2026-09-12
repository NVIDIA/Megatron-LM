# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Audit live Megatron parameters through their native distributed-checkpoint metadata.

Use ``local_parameter_records`` without communication on each model chunk, then merge
the JSON records offline. ``collect_parameter_manifest`` is collective and must be
called once on every rank after all local chunks have been constructed, never from
an interleaved pipeline forward. Neither path reads parameter values or saves weights.
"""

from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch

from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory
from megatron.core.utils import unwrap_model


def _span(tensor: torch.Tensor) -> tuple[str, int, int, int]:
    start = tensor.storage_offset() * tensor.element_size()
    end = (
        start
        + (
            1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()))
            if tensor.numel()
            else 0
        )
        * tensor.element_size()
    )
    return str(tensor.device), tensor.untyped_storage().data_ptr(), start, end


def _parameter_role(model: torch.nn.Module, name: str) -> dict:
    if name.startswith("mtp."):
        return {"category": "mtp", "activation": "excluded"}
    if name.startswith(("embedding.", "output_layer.")):
        return {"category": "embedding_and_head", "activation": "excluded"}
    if ".engram." in name or name.startswith("engram."):
        if ".multi_head_embedding.embedding.weight" in name:
            owner = model.get_submodule(name.rsplit(".embedding.weight", 1)[0])
            return {
                "category": "engram_table",
                "activation": "lookup",
                "table_heads": owner.num_heads,
            }
        return {"category": "engram_fusion", "activation": "all"}
    if ".experts." in name:
        path = name.rsplit(".", 1)[0]
        while path:
            config = getattr(model.get_submodule(path), "config", None)
            if config is not None and getattr(config, "num_moe_experts", None):
                return {
                    "category": "backbone_without_engram",
                    "activation": "routed",
                    "topk": config.moe_router_topk,
                    "routed_experts": config.num_moe_experts,
                }
            path = path.rsplit(".", 1)[0] if "." in path else ""
        raise ValueError(f"Cannot determine routed expert configuration for {name}")
    return {"category": "backbone_without_engram", "activation": "all"}


def local_parameter_records(models: Any) -> dict:
    """Expand checkpoint factories and associate each shard with an actual parameter.

    Buffers and extra state are excluded by parameter identity/storage, rather than
    unreliable suffix guesses. Any trainable parameter missing from checkpoint
    metadata is an error. Shared-storage DDP parameters are matched by byte interval.
    """
    if not isinstance(models, (list, tuple)):
        models = [models]
    entries = []
    parameters = []
    for chunk_index, wrapped in enumerate(models):
        model = unwrap_model(wrapped)
        owners = []
        by_identity = {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            owner = {
                "name": name,
                "span": _span(parameter),
                "numel": parameter.numel(),
                "shape": list(parameter.shape),
                "role": _parameter_role(model, name),
                "represented": parameter.numel() == 0,
            }
            owners.append(owner)
            by_identity[id(parameter)] = owner

        def find_owner(data):
            if not isinstance(data, torch.Tensor):
                return None
            if id(data) in by_identity:
                return by_identity[id(data)]
            device, storage, start, end = _span(data)
            matches = [
                owner
                for owner in owners
                if owner["span"][:2] == (device, storage)
                and owner["span"][2] <= start <= end <= owner["span"][3]
            ]
            if len(matches) > 1:
                raise ValueError("Ambiguous parameter storage in checkpoint metadata")
            return matches[0] if matches else None

        def visit(value, inherited_owner=None):
            if isinstance(value, (ShardedTensor, ShardedTensorFactory)):
                owner = inherited_owner or find_owner(value.data)
                if owner is None:
                    return
                owner["represented"] = True
                if isinstance(value, ShardedTensorFactory):
                    visit(value.build(), owner)
                else:
                    entries.append(
                        {
                            "key": value.key,
                            "global_shape": list(value.global_shape),
                            "global_offset": list(value.global_offset),
                            "local_shape": list(value.local_shape),
                            "replica_id": value.replica_id,
                            "source_parameter": owner["name"],
                            "chunk_index": chunk_index,
                            **owner["role"],
                        }
                    )
            elif isinstance(value, dict):
                for child in value.values():
                    visit(child, inherited_owner)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    visit(child, inherited_owner)

        visit(model.sharded_state_dict())
        missing = [owner["name"] for owner in owners if not owner["represented"]]
        if missing:
            raise ValueError(f"Parameters missing from native checkpoint metadata: {missing}")
        parameters.extend(
            {key: value for key, value in owner.items() if key not in ("span", "role")}
            | {"chunk_index": chunk_index}
            for owner in owners
        )
    return {
        "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
        "world_size": (
            torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        ),
        "parameters": parameters,
        "entries": entries,
    }


def merge_parameter_records(records: list[dict]) -> dict:
    """Count unique global keys once, including expert axes and expanded hash tables."""
    keys = {}
    for local in records:
        for entry in local["entries"]:
            key = entry["key"]
            identity = {
                field: value
                for field, value in entry.items()
                if field
                not in (
                    "global_offset",
                    "local_shape",
                    "replica_id",
                    "source_parameter",
                    "chunk_index",
                )
            }
            if key in keys and keys[key] != identity:
                raise ValueError(f"Inconsistent global parameter metadata for {key}")
            keys[key] = identity
    totals = dict.fromkeys(
        ("backbone_without_engram", "engram_table", "engram_fusion", "embedding_and_head", "mtp"), 0
    )
    activated = Fraction(0)
    for key, entry in keys.items():
        count = math.prod(entry["global_shape"])
        entry["numel"] = count
        totals[entry["category"]] += count
        if entry["activation"] == "all":
            activated += count
        elif entry["activation"] == "routed":
            activated += Fraction(count * entry["topk"], entry["routed_experts"])
        elif entry["activation"] == "lookup":
            # row_a2a factories expose one global key per hash head; local tables
            # concatenate all heads under one key and therefore access H rows.
            heads = 1 if ".table_" in key else entry["table_heads"]
            activated += heads * math.prod(entry["global_shape"][1:])
    if activated.denominator != 1:
        raise ValueError(f"Nonintegral activated parameter count: {activated}")
    totals["engram_total"] = totals["engram_table"] + totals["engram_fusion"]
    totals["backbone_total"] = totals["backbone_without_engram"] + totals["engram_total"]
    totals["all_trainable"] = (
        totals["backbone_total"] + totals["embedding_and_head"] + totals["mtp"]
    )
    totals["activated"] = int(activated)
    return {
        "schema": 1,
        "source": "live parameter identities and expanded native sharded_state_dict global shapes",
        "scope_excludes_from_backbone_and_activated": ["embedding", "lm_head", "mtp"],
        "activated_definition": "all dense/router/fusion weights; top-k routed experts; one row per hash head",
        "totals": totals,
        "global_parameters": [keys[key] for key in sorted(keys)],
        "local_summaries": [
            {
                "rank": record["rank"],
                "parameter_tensors": len(record["parameters"]),
                "parameter_elements_with_replicas": sum(
                    item["numel"] for item in record["parameters"]
                ),
                "checkpoint_shards": len(record["entries"]),
            }
            for record in records
        ],
    }


def collect_parameter_manifest(models: Any) -> dict:
    """Collect metadata on every rank after construction, outside pipeline forward."""
    local = local_parameter_records(models)
    if torch.distributed.is_initialized():
        records = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(records, local)
    else:
        records = [local]
    return merge_parameter_records(records)


def assert_expected_parameters(manifest: dict, expected: dict) -> None:
    """Require exact agreement for the requested scalar totals before training."""
    unknown = set(expected) - set(manifest["totals"])
    if unknown:
        raise ValueError(f"Unknown parameter budget fields: {sorted(unknown)}")
    mismatches = {
        key: {"expected": value, "actual": manifest["totals"].get(key)}
        for key, value in expected.items()
        if manifest["totals"][key] != value
    }
    if mismatches:
        raise ValueError(f"Live model parameter budget mismatch: {mismatches}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = merge_parameter_records([json.loads(path.read_text()) for path in arguments.records])
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")
