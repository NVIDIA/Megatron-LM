# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Publication and selective tar reads for offline KD v3."""

import glob
import io
import json
import os
import tarfile
import uuid
from typing import Any

from .v3_format import digest

CACHE_FILE = "_v3_cache.json"
COMPLETE_FILE = "_v3_complete.json"


class Storage:
    """Use local storage or Megatron's configured MSC backend."""

    def __init__(self, root: str):
        self.root = root
        self.remote = root.startswith("msc://")
        self.backend = None
        if self.remote:
            from megatron.core.msc_utils import MultiStorageClientFeature

            MultiStorageClientFeature.enable()
            self.backend = MultiStorageClientFeature.import_package()

    def path(self, name: str) -> str:
        """Resolve a cache-relative filename."""
        return os.path.join(self.root, name)

    def open(self, name: str, mode: str = "rb") -> Any:
        """Open a cache object."""
        path = self.path(name)
        return self.backend.open(path, mode) if self.remote else open(path, mode)

    def exists(self, name: str) -> bool:
        """Check whether a cache object exists."""
        return (
            self.backend.os.path.exists(self.path(name))
            if self.remote
            else os.path.exists(self.path(name))
        )

    def list(self, pattern: str) -> list[str]:
        """List fresh object names; never cache a growing-cache listing."""
        paths = (
            self.backend.glob(self.path(pattern)) if self.remote else glob.glob(self.path(pattern))
        )
        return sorted(os.path.basename(str(path)) for path in paths)

    def read_json(self, name: str) -> dict[str, Any]:
        """Read a published descriptor."""
        with self.open(name) as stream:
            return json.load(stream)

    def move(self, source: str, destination: str) -> None:
        """Retain an unpublished object under a recovery name."""
        if self.remote:
            client, parsed_source = self.backend.resolve_storage_client(self.path(source))
            _, parsed_destination = self.backend.resolve_storage_client(self.path(destination))
            client.copy(parsed_source, parsed_destination)
            self.backend.delete(self.path(source))
        else:
            os.replace(self.path(source), self.path(destination))

    def write_json(self, name: str, payload: dict[str, Any]) -> None:
        """Publish JSON after its dependencies are durable."""
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        self.write_bytes(name, raw)

    def write_bytes(self, name: str, raw: bytes) -> None:
        """Publish a local object atomically, or a complete object-store PUT."""
        if self.remote:
            with self.open(name, "wb") as stream:
                stream.write(raw)
            return
        os.makedirs(self.root, exist_ok=True)
        temporary = f"{name}.{uuid.uuid4().hex}.tmp"
        with self.open(temporary, "wb") as stream:
            stream.write(raw)
        os.replace(self.path(temporary), self.path(name))


def quarantine_unpublished(storage: Storage, published_through: int) -> None:
    """Retire interrupted groups beyond the readable prefix before teacher resume.

    Hide descriptors before moving tar objects so following students never see
    a partially reconstructed group. Retain both files for recovery.
    """
    suffix = f".aborted.{uuid.uuid4().hex}"
    for name in storage.list("dp*__*.tar.ready.json"):
        descriptor = storage.read_json(name)
        if descriptor["records"][0]["start"] < published_through:
            continue
        storage.move(name, name + suffix)
        if storage.exists(descriptor["tar"]):
            storage.move(descriptor["tar"], descriptor["tar"] + suffix)


def write_shard(storage: Storage, name: str, metadata: dict, records: list[dict]) -> None:
    """Write paired members and publish a per-DP completion descriptor.

    A range is readable only after all DP descriptors exist. Existing published
    shards are immutable; restarting a teacher cannot silently replace them.
    """
    descriptor_name = name + ".ready.json"
    descriptions = []
    members = []
    for record in records:
        description = {k: record[k] for k in ("start", "end", "sample_ids", "record_id")}
        for kind in ("inputs", "targets"):
            member_name = f'{record["start"]}-{record["end"]}.{kind}.pt.zst'
            raw = record[kind]
            description[kind] = {"member": member_name, "sha256": digest(raw), "size": len(raw)}
            members.append((member_name, raw))
        descriptions.append(description)
    descriptor = {"metadata": metadata, "tar": name, "records": descriptions}
    if storage.exists(descriptor_name):
        if storage.read_json(descriptor_name) != descriptor:
            raise RuntimeError("Refusing to replace published offline KD v3 records")
        return
    # No ready descriptor exists until the entire tar is closed successfully.
    # Local staging is atomic; remote object visibility is gated by the descriptor.
    staged_name = name if storage.remote else name + ".tmp"
    with storage.open(staged_name, "wb") as stream:
        with tarfile.open(fileobj=stream, mode="w") as archive:
            meta = json.dumps(metadata, sort_keys=True).encode()
            for member_name, raw in [("_meta.json", meta), *members]:
                info = tarfile.TarInfo(member_name)
                info.size = len(raw)
                archive.addfile(info, io.BytesIO(raw))
    if not storage.remote:
        os.replace(storage.path(staged_name), storage.path(name))
    storage.write_json(descriptor_name, descriptor)


def read_members(
    storage: Storage, descriptor: dict, targets: bool, ranges: set[tuple[int, int]] | None = None
) -> dict[tuple[int, int], dict[str, bytes]]:
    """Read only the compressed members needed by this pipeline stage."""
    requested = {}
    for record in descriptor["records"]:
        if ranges is not None and (record["start"], record["end"]) not in ranges:
            continue
        for kind in (("inputs", "targets") if targets else ("inputs",)):
            requested[record[kind]["member"]] = (record, kind)
    result = {}
    with storage.open(descriptor["tar"]) as stream:
        with tarfile.open(fileobj=stream, mode="r:") as archive:
            for name, (record, kind) in list(requested.items()):
                info = archive.getmember(name)
                del requested[name]
                if info.size != record[kind]["size"]:
                    raise ValueError("Offline KD v3 member size mismatch")
                raw = archive.extractfile(info).read()
                result.setdefault((record["start"], record["end"]), {})[kind] = raw
    if requested:
        raise ValueError("Offline KD v3 shard is missing paired members")
    return result
