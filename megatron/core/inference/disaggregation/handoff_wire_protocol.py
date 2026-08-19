# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Wire payload construction and validation for prefill/decode handoff."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

_NIXL_AGENT_NAME = "agent_name"
_NIXL_AGENT_METADATA = "agent_metadata_b64"


def strip_registered_nixl_agent_metadata(value: Any) -> Any:
    """Remove NIXL agent blobs already registered with the coordinator."""

    if isinstance(value, dict):
        is_agent_record = _NIXL_AGENT_NAME in value
        return {
            key: strip_registered_nixl_agent_metadata(child)
            for key, child in value.items()
            if not (is_agent_record and key == _NIXL_AGENT_METADATA)
        }
    if isinstance(value, list):
        return [strip_registered_nixl_agent_metadata(child) for child in value]
    if isinstance(value, tuple):
        return tuple(strip_registered_nixl_agent_metadata(child) for child in value)
    return value


def _registered_nixl_agents(instance_meta: Any) -> Dict[str, str]:
    """Index agent metadata recursively across TP/PP instance registration."""

    agents: Dict[str, str] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            name = value.get(_NIXL_AGENT_NAME)
            metadata = value.get(_NIXL_AGENT_METADATA)
            if name is not None and metadata is not None:
                previous = agents.setdefault(name, metadata)
                if previous != metadata:
                    raise ValueError(f"NIXL agent {name!r} has conflicting registered metadata")
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(instance_meta)
    return agents


def restore_registered_nixl_agent_metadata(value: Any, instance_meta: Any) -> Any:
    """Restore and validate NIXL agent blobs from engine registration."""

    agents = _registered_nixl_agents(instance_meta)

    def restore(child: Any) -> Any:
        if isinstance(child, dict):
            restored = {key: restore(value) for key, value in child.items()}
            name = restored.get(_NIXL_AGENT_NAME)
            if name is None:
                return restored
            registered = agents.get(name)
            if registered is None:
                raise ValueError(f"NIXL agent {name!r} is absent from instance registration")
            supplied = restored.get(_NIXL_AGENT_METADATA)
            if supplied is not None and supplied != registered:
                raise ValueError(
                    f"NIXL agent {name!r} handoff metadata differs from its registration"
                )
            restored[_NIXL_AGENT_METADATA] = registered
            return restored
        if isinstance(child, list):
            return [restore(value) for value in child]
        if isinstance(child, tuple):
            return tuple(restore(value) for value in child)
        return child

    return restore(value)


def make_submit_request_with_kv_message(
    header_value: int,
    request_id: int,
    prompt: Any,
    sampling_params: dict,
    kv_meta: dict,
    src_block_ids: list,
) -> list:
    """Build a ``SUBMIT_REQUEST_WITH_KV`` message."""

    return [header_value, int(request_id), prompt, sampling_params, kv_meta, list(src_block_ids)]


def parse_submit_request_with_kv_fields(fields: Sequence[Any]) -> Tuple[Any, ...]:
    """Validate and unpack fields following ``SUBMIT_REQUEST_WITH_KV``."""

    if len(fields) != 5:
        raise ValueError(f"SUBMIT_REQUEST_WITH_KV payload must have 5 fields, got {len(fields)}")
    return tuple(fields)


def make_release_kv_message(header_value: int, request_id: int) -> list:
    """Build a ``RELEASE_KV`` message."""

    return [header_value, int(request_id)]
