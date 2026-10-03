# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Durable rollout bank — the queued-rollout (completed-group) path.

Persists completed ``RolloutGroup``s the instant they assemble, so a SIGKILL (the
4h SLURM limit) does not destroy work that is already on disk. This is PR #1 of
the durable rollout bank: it covers only the *queued* path — groups sitting in
the pipeline's ``output_queue`` (assembled, not yet trained-in) plus groups
consumed at a training step that a restart rolls back. In-flight decode state and
partial-group snapshots (design phases B/C) are out of scope here.

Persisted data comes from two separate sources.
Data related to the RL pipeline (rewards, env/problem ids, grouping,
and per-turn completion ids) is extracted from the RL pipeline and stored in
an append-only JSONL ledger (one per collection segment).
Per-request engine-related data is sourced from the inference engine's in-memory
finished-request ledger, which is wrapped by a custom writer in this file and
stored as per-rank backup files.

Layout (manifests are single-writer rank-0; one record backup file per engine rank,
all on Lustre)::

    <bank_dir>/
        PIPELINE.json                 # atomically selects one complete generation
        request_records/              # live per-rank engine backup files
            records-rank00000-gen000123-<tag>.bin
        generations/
            generation-<uuid>/
                consumed.log          # append-only markers {"uid", "iter"}
                gen-<iter>/
                    ledger.log        # append-only JSONL, one manifest per group
                    request_records.bin  # records referenced by this segment

Recovery guarantees: manifest appends are write-through with ``fsync`` per group,
so a kill can damage only the final record, which its checksum drops on read.
Backup files are self-delimiting msgpack streams; a torn tail drops only itself.
A checkpoint compacts survivors — and every request record still owed to a group:
referenced by a survivor, or neither reclaimed with its group nor retired by rank 0
(dropped groups, evaluation) — into a complete new generation before atomically
flipping the manifest, then prunes the consumed (rotated, never open) backup files.
``restore`` replays the active
generation and returns every group that is not already trained through checkpoint
step ``T``:

    marker <= T -> discard (training already in the loaded weights)
    marker  > T -> restore (that training step was erased by the kill)
    no marker   -> restore (never consumed, or lost in the kill)

``recover(T)`` applies those rules once, then publishes the survivors in a new
timeline with an empty marker log. Markers from work after ``T`` therefore cannot
incorrectly suppress a group when the restarted timeline later passes their old
iteration number. A surviving group whose request records did not survive the
crash (lost inside one write-out of the kill) is dropped LOUDLY and excluded from
the new generation — its work is regenerated, never trained without its inference
logprobs.
"""

import dataclasses
import hashlib
import json
import logging
import os
import queue
import re
import shutil
import threading
import uuid
from typing import Iterable, Iterator, Literal, NamedTuple, NotRequired, Optional, TypedDict

import msgpack
import numpy as np

from megatron.core.inference.inference_request import (
    FinishedRequestRecord,
    OffloadedRequestPayload,
    RequestPayloadStageResult,
)
from megatron.rl.types import Rollout, RolloutGroup, TokenRollout

logger = logging.getLogger(__name__)

# Bump whenever the manifest/ledger schema changes.
# Readers intentionally fail closed; cross-version migration must be explicit.
_FORMAT_VERSION = 3
_MANIFEST = "PIPELINE.json"
_GENERATIONS = "generations"
_LEDGER = "ledger.log"
_CONSUMED = "consumed.log"
_REQUEST_RECORDS = "request_records.bin"
_REQUEST_RECORDS_DIRNAME = "request_records"
# Per-run tag: a writer must never append after a crashed run's torn tail.
_REQUEST_RECORD_BACKUP_TAG = uuid.uuid4().hex[:8]
_BACKUP_NAME = re.compile(r"records-rank\d+-gen(\d+)-[0-9a-f]+\.bin$")


class LedgerRecord(TypedDict):
    """One self-describing JSONL manifest record for a completed group.

    Args:
        format_version: The persisted schema version.
        uid: The unique identifier of the group.
        collection_iter: The iteration number of the collection.
        kind: "token" for token groups, whose per-token payloads live in the request records;
            "inline" for groups stored whole.
        group: The group of rollouts.
        checksum: The checksum of the group.
    """

    format_version: int
    uid: str
    collection_iter: int
    kind: Literal["inline", "token"]
    group: dict
    checksum: NotRequired[str]


class Manifest(TypedDict):
    """Bank-level index: how far training got and which segments exist.

    Args:
        format_version: The persisted schema version.
        active_generation: The directory containing the complete active bank state.
        timeline: An opaque identifier for the execution history owning the markers.
        trained_through: The iteration number of the last checkpoint.
        segments: The list of segment names (e.g. ["gen-000000", "gen-000001"]).
    """

    format_version: int
    active_generation: str
    timeline: str
    trained_through: int
    segments: list[str]


class ConsumedMarker(TypedDict):
    """Append-only marker: a group uid was pulled by the trainer at ``iter``."""

    uid: str
    iter: int


def _segment_name(iteration: int) -> str:
    return f"gen-{iteration:06d}"


def _generation_name() -> str:
    return f"generation-{uuid.uuid4().hex}"


def resolve_bank_dir(configured: str | None, save: str | None) -> str:
    """The bank directory: the configured one, else ``<save>/rollout_bank``; shared by the bank
    on rank 0 and every engine's request-record backup writer."""
    return configured or os.path.join(save or ".", "rollout_bank")


def request_records_dir(bank_dir: str) -> str:
    """Directory holding the per-rank finished-request record backup files."""
    return os.path.join(bank_dir, _REQUEST_RECORDS_DIRNAME)


def request_record_backup_path(
    bank_dir: str, rank: int, iteration: int, tag: str = _REQUEST_RECORD_BACKUP_TAG
) -> str:
    """Backup path for one rank and collection window; ``tag`` keeps runs collision-free."""
    return os.path.join(
        request_records_dir(bank_dir), f"records-rank{rank:05d}-gen{iteration:06d}-{tag}.bin"
    )


def _backup_paths(bank_dir: str) -> list[str]:
    """Live per-rank backup files, sorted by name."""
    backup_dir = request_records_dir(bank_dir)
    if not os.path.isdir(backup_dir):
        return []
    return [
        os.path.join(backup_dir, name)
        for name in sorted(os.listdir(backup_dir))
        if _BACKUP_NAME.match(name)
    ]


def _fold_paths(bank_dir: str, manifest: Manifest) -> list[str]:
    """The active generation's folded request-record files (whether or not they exist)."""
    generation_dir = os.path.join(bank_dir, _GENERATIONS, manifest["active_generation"])
    return [os.path.join(generation_dir, seg, _REQUEST_RECORDS) for seg in manifest["segments"]]


def request_record_paths(bank_dir: str) -> list[str]:
    """Every durable record store: live backup files plus the active generation's folds."""
    paths = _backup_paths(bank_dir)
    manifest_path = os.path.join(bank_dir, _MANIFEST)
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path) as f:
                paths += [p for p in _fold_paths(bank_dir, json.load(f)) if os.path.exists(p)]
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            logger.warning(
                "Unreadable rollout-bank manifest at %s; loading backup files only", manifest_path
            )
    return paths


# Token ids are exact; fp32 round-trips the engine's fp32-computed logprobs exactly,
# so restored rollouts train on the same values as live ones.
_TOKEN_DTYPE = np.int32
_LOGPROB_DTYPE = np.float32


def pack_finished_request_record(uid: str, record: FinishedRequestRecord) -> bytes:
    """Encode one record with binary-packed per-token arrays; media and top-N scores are dropped."""

    def packed(values, dtype) -> Optional[bytes]:
        return None if values is None else np.asarray(values, dtype=dtype).tobytes()

    tokens: dict = {}
    if record.payload is not None:
        p = record.payload
        routing = p.routing_indices  # Present only under moe_enable_routing_replay.
        tokens = {
            "prompt_token_ids": packed(p.prompt_token_ids, _TOKEN_DTYPE),
            "generated_token_ids": packed(p.generated_token_ids, _TOKEN_DTYPE),
            "generated_log_probs": packed(p.generated_log_probs, _LOGPROB_DTYPE),
            "prompt_log_probs": packed(p.prompt_log_probs, _LOGPROB_DTYPE),
            "routing_indices": (
                None
                if routing is None
                else {
                    "shape": list(routing.shape),
                    "dtype": str(routing.dtype),
                    "data": routing.tobytes(),
                }
            ),
        }
    payload = {
        "uid": uid,
        "policy_epoch": record.policy_epoch,
        "kv_cache_epoch": record.kv_cache_epoch,
        "num_evictions": record.num_evictions,
        "num_tokens_to_generate": record.num_tokens_to_generate,
        "finish_reason": record.finish_reason,
        **tokens,
    }
    return msgpack.packb(payload, use_bin_type=True)


def _unpack_record(payload: dict) -> tuple[str, FinishedRequestRecord]:
    """Decode one record; epoch stamps round-trip back to lists of tuples."""

    def epochs(value):
        return None if value is None else [tuple(pair) for pair in value]

    def values(raw: Optional[bytes], dtype) -> Optional[list]:
        return None if raw is None else np.frombuffer(raw, dtype=dtype).tolist()

    generated = payload.get("generated_token_ids")
    offloaded = None  # a record no stager pinned a payload on
    if generated is not None:
        routing = payload.get("routing_indices")
        offloaded = OffloadedRequestPayload(
            prompt_token_ids=values(payload.get("prompt_token_ids"), _TOKEN_DTYPE),
            generated_token_ids=values(generated, _TOKEN_DTYPE),
            generated_log_probs=values(payload.get("generated_log_probs"), _LOGPROB_DTYPE),
            prompt_log_probs=values(payload.get("prompt_log_probs"), _LOGPROB_DTYPE),
            routing_indices=(
                None
                if routing is None
                else np.frombuffer(routing["data"], dtype=np.dtype(routing["dtype"])).reshape(
                    routing["shape"]
                )
            ),
        )
    record = FinishedRequestRecord(
        policy_epoch=epochs(payload["policy_epoch"]),
        kv_cache_epoch=epochs(payload["kv_cache_epoch"]),
        num_evictions=payload["num_evictions"],
        num_tokens_to_generate=payload.get("num_tokens_to_generate"),
        finish_reason=payload.get("finish_reason"),
        payload=offloaded,
    )
    return payload["uid"], record


class _Rotate(NamedTuple):
    """Writer control message: close the current file and continue at `path` (None stops)."""

    path: Optional[str]
    done: threading.Event


class FinishedRequestLedgerWriter:
    """Append-only durable writer for finished-request records (one per engine)."""

    def __init__(self, path: str) -> None:
        self._queue: queue.Queue = queue.Queue()
        self._path: Optional[str] = path
        self._file = None
        self._error: Optional[BaseException] = None
        self._thread = threading.Thread(
            target=self._run, name="finished-request-ledger-writer", daemon=True
        )
        self._thread.start()

    def append(self, uid: str, record: FinishedRequestRecord) -> None:
        """Enqueue one record for durable append; returns immediately."""
        self._queue.put((uid, record))

    def rotate(self, path: str) -> None:
        """Flush and close the current file; subsequent appends go to ``path``."""
        self._control(path)

    def close(self) -> None:
        """Flush, close the file, and stop the writer thread."""
        try:
            self._control(None)
        finally:
            self._thread.join()

    def _control(self, path: Optional[str]) -> None:
        done = threading.Event()
        self._queue.put(_Rotate(path, done))
        done.wait()
        if self._error is not None:
            raise RuntimeError("finished-request ledger writer failed") from self._error

    def _sync(self, wrote: bool) -> None:
        if wrote and self._file is not None:
            self._file.flush()
            os.fsync(self._file.fileno())

    def _close_file(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def _run(self) -> None:
        while True:
            batch = [self._queue.get()]
            while True:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            wrote = False
            try:
                for item in batch:
                    if isinstance(item, _Rotate):
                        self._sync(wrote)
                        wrote = False
                        self._close_file()
                        self._path = item.path
                        if item.path is None:
                            return
                    else:
                        if self._file is None:
                            os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
                            self._file = open(self._path, "ab")
                        self._file.write(pack_finished_request_record(*item))
                        wrote = True
                self._sync(wrote)
            except Exception as exc:
                if self._error is None:
                    logger.exception("finished-request ledger writer failed at %s", self._path)
                    self._error = exc
            finally:
                for item in batch:
                    if isinstance(item, _Rotate):
                        item.done.set()


def load_finished_request_records(paths: Iterable[str]) -> dict[str, FinishedRequestRecord]:
    """Load records from backup/fold files, keyed by uid; a torn tail drops only itself."""
    records: dict[str, FinishedRequestRecord] = {}
    for path in sorted(paths):
        try:
            with open(path, "rb") as f:
                for payload in msgpack.Unpacker(f, raw=False):
                    uid, record = _unpack_record(payload)
                    records[uid] = record
        except FileNotFoundError:
            continue
        except Exception:
            logger.warning(
                "Truncated tail in request-record backup %s; kept records before it.", path
            )
    return records


class LedgerPayloadStager:
    """RequestPayloadStager for Megatron RL: pins each finished request's payload on its record."""

    def __init__(self, writer: FinishedRequestLedgerWriter | None = None) -> None:
        # Durable append-log of the staged records (the rollout bank's restore source).
        self.writer = writer
        # Records staged since the last drain(), by uid; stage() runs on worker threads.
        self.records: dict[str, FinishedRequestRecord] = {}
        self._lock = threading.Lock()

    def stage(
        self,
        uid: str,
        payload: OffloadedRequestPayload,
        *,
        finished_metadata: FinishedRequestRecord,
        offload_params: dict | None = None,
    ) -> RequestPayloadStageResult:
        # MRL joins by uid, so offload_params (a Gym admission on the NeMo RL path) has no role.
        finished_metadata.payload = payload
        if self.writer is not None:
            self.writer.append(uid, finished_metadata)
        with self._lock:
            assert uid not in self.records, f"finished-request ledger: duplicate uid {uid!r}"
            self.records[uid] = finished_metadata
        # Custody acknowledged: the reply is served without its per-token payload.
        return RequestPayloadStageResult()

    def drain(self) -> dict[str, FinishedRequestRecord]:
        """Take every record staged since the last drain."""
        with self._lock:
            drained, self.records = self.records, {}
        return drained


def gather_view(record: FinishedRequestRecord) -> FinishedRequestRecord:
    """Keep what the RL joins read: token ids, generated log probs and the staleness stamps."""
    payload = record.payload
    if payload is not None:
        payload = dataclasses.replace(
            payload,
            prompt_log_probs=None,
            prompt_top_n_logprobs=None,
            generated_top_n_logprobs=None,
            routing_indices=None,
            media_tensors=None,
        )
    return dataclasses.replace(record, payload=payload)


# TokenRollout fields rebuilt from the request records at restore, so manifests omit them.
_RECORD_DERIVED_FIELDS = (
    "logprobs",
    "generation_mask",
    "policy_epochs",
    "kv_cache_epochs",
    "num_evictions",
)


def _joinable(record: Optional[FinishedRequestRecord]) -> bool:
    """Whether a record carries everything a restored TokenRollout turn is rebuilt from."""
    p = None if record is None else record.payload
    return p is not None and None not in (
        p.prompt_token_ids,
        p.generated_log_probs,
        record.policy_epoch,
        record.kv_cache_epoch,
    )


def rebuild_turns(turns: list[FinishedRequestRecord]) -> dict[str, list]:
    """TokenRollout fields rebuilt from per-turn records: turn tokens == prompt + generation."""
    payloads = [t.payload for t in turns]
    trajectory = [list(p.prompt_token_ids) + list(p.generated_token_ids) for p in payloads]
    return {
        "trajectory": trajectory,
        "logprobs": [list(p.generated_log_probs) for p in payloads],
        "generation_mask": [
            [i >= len(p.prompt_token_ids) for i in range(len(tokens))]
            for p, tokens in zip(payloads, trajectory)
        ],
        "policy_epochs": [t.policy_epoch for t in turns],
        "kv_cache_epochs": [t.kv_cache_epoch for t in turns],
        "num_evictions": [t.num_evictions for t in turns],
    }


def _checksum(record_wo_checksum: dict) -> str:
    """Digest over the canonical manifest record."""
    h = hashlib.blake2b(digest_size=16)
    h.update(json.dumps(record_wo_checksum, sort_keys=True, separators=(",", ":")).encode())
    return h.hexdigest()


def _validate_format_version(data: dict, source: str) -> None:
    """Reject banks that this implementation cannot decode safely."""
    version = data.get("format_version")
    if version != _FORMAT_VERSION:
        raise ValueError(
            f"Unsupported RolloutBank format_version {version!r} in {source}; "
            f"expected {_FORMAT_VERSION}. Migrate or remove the incompatible rollout bank."
        )


def _fsync_directory(path: str) -> None:
    """Persist directory-entry changes made beneath ``path``."""
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class RolloutBank:
    """Single-writer durable store for completed rollout groups (rank-0 only)."""

    def __init__(self, bank_dir: str, *, max_bytes: int = 0) -> None:
        self.bank_dir = bank_dir
        self.max_bytes = max_bytes
        os.makedirs(bank_dir, exist_ok=True)
        self._lock = threading.RLock()
        self._collection_iter: Optional[int] = None
        self._seg_dir: Optional[str] = None
        self._seq = 0
        self._ledger_f = None  # open ledger handle for the active segment
        self._bytes_written = 0
        self._warned_over_cap = False
        self._retired: set[str] = set()  # uids rank 0 retired, pruned once their records show
        if not os.path.exists(self._manifest_path):
            # The engines may already have created their backup directory; anything
            # else in a manifest-less directory is bank data this run must not overwrite.
            if set(os.listdir(bank_dir)) - {_REQUEST_RECORDS_DIRNAME}:
                raise FileNotFoundError(
                    f"RolloutBank manifest is missing at {self._manifest_path}, but "
                    f"{bank_dir} is not empty. Refusing to overwrite possible bank data; "
                    "recover PIPELINE.json or remove the rollout-bank directory."
                )
            os.makedirs(self._generations_dir)
            _fsync_directory(bank_dir)
            self._write_manifest_atomic(
                {
                    "format_version": _FORMAT_VERSION,
                    "active_generation": self._create_empty_generation(),
                    "timeline": uuid.uuid4().hex,
                    "trained_through": 0,
                    "segments": [],
                }
            )
        manifest = self._read_manifest()
        self._garbage_collect_generations(manifest)
        # Read through the published manifest immediately, both to initialize
        # live-size accounting and to fail closed on malformed existing state.
        self._refresh_bytes_written(manifest)
        self._maybe_warn_over_cap()

    # ------------------------------------------------------------------ paths
    @property
    def _manifest_path(self) -> str:
        return os.path.join(self.bank_dir, _MANIFEST)

    @property
    def _generations_dir(self) -> str:
        return os.path.join(self.bank_dir, _GENERATIONS)

    def _generation_dir(self, manifest: Manifest) -> str:
        return os.path.join(self._generations_dir, manifest["active_generation"])

    def _create_empty_generation(self) -> str:
        generation = _generation_name()
        generation_dir = os.path.join(self._generations_dir, generation)
        os.makedirs(generation_dir)
        with open(os.path.join(generation_dir, _CONSUMED), "w") as f:
            f.flush()
            os.fsync(f.fileno())
        _fsync_directory(generation_dir)
        _fsync_directory(self._generations_dir)
        return generation

    def _garbage_collect_generations(self, manifest: Manifest) -> None:
        """Remove recognized generation directories not selected by ``manifest``."""
        removed = False
        for name in os.listdir(self._generations_dir):
            path = os.path.join(self._generations_dir, name)
            is_generation = name.startswith("generation-")
            is_staging = name.startswith(".generation-") and name.endswith(".tmp")
            if name != manifest["active_generation"] and os.path.isdir(path):
                if is_generation or is_staging:
                    shutil.rmtree(path, ignore_errors=True)
                    removed = True
        if removed:
            _fsync_directory(self._generations_dir)

    def _read_manifest(self) -> Manifest:
        try:
            with open(self._manifest_path) as f:
                manifest = json.load(f)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"RolloutBank manifest is missing at {self._manifest_path}. Refusing to continue; "
                "recover PIPELINE.json or remove the rollout-bank directory."
            ) from exc
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Malformed RolloutBank manifest at {self._manifest_path}: {exc.msg} "
                f"(line {exc.lineno}, column {exc.colno}). Refusing to continue; "
                "recover PIPELINE.json or remove the rollout-bank directory."
            ) from exc
        _validate_format_version(manifest, self._manifest_path)
        active = manifest.get("active_generation")
        if (
            not isinstance(active, str)
            or not active.startswith("generation-")
            or os.path.basename(active) != active
        ):
            raise ValueError(f"Invalid active_generation {active!r} in {self._manifest_path}")
        if not isinstance(manifest.get("timeline"), str) or not manifest["timeline"]:
            raise ValueError(f"Invalid timeline in {self._manifest_path}")
        if not isinstance(manifest.get("segments"), list):
            raise ValueError(f"Invalid segments in {self._manifest_path}")
        if not os.path.isdir(self._generation_dir(manifest)):
            raise ValueError(
                f"Active rollout-bank generation {active!r} does not exist in "
                f"{self._generations_dir}"
            )
        return manifest

    def _write_manifest_atomic(self, manifest: Manifest) -> None:
        manifest = dict(manifest)
        manifest.setdefault("format_version", _FORMAT_VERSION)
        _validate_format_version(manifest, self._manifest_path)
        tmp = self._manifest_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(manifest, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self._manifest_path)  # atomic flip
        _fsync_directory(self.bank_dir)

    # ------------------------------------------------------------- lifecycle
    def set_collection(self, iteration: int) -> None:
        """Point subsequent appends at the ``gen-<iteration>/`` segment."""
        with self._lock:
            if self._collection_iter == iteration and self._seg_dir is not None:
                return
            self._close_handles()
            self._collection_iter = iteration
            seg = _segment_name(iteration)
            manifest = self._read_manifest()
            generation_dir = self._generation_dir(manifest)
            self._seg_dir = os.path.join(generation_dir, seg)
            segment_created = not os.path.exists(self._seg_dir)
            os.makedirs(self._seg_dir, exist_ok=True)
            if segment_created:
                _fsync_directory(generation_dir)
            self._truncate_torn_ledger_tail(self._seg_dir)
            self._seq = self._next_sequence(self._seg_dir, seg)
            if seg not in manifest["segments"]:
                manifest["segments"].append(seg)
                self._write_manifest_atomic(manifest)
            # Window boundary: refresh the durable-footprint accounting.
            self._refresh_bytes_written(manifest)
            self._maybe_warn_over_cap()

    def _next_sequence(self, seg_dir: str, seg: str) -> int:
        """Return the next unused sequence number in ``seg``."""
        next_seq = 0
        for record in self._read_ledger(seg_dir):
            uid_seg, separator, uid_seq = record.get("uid", "").partition("/")
            if separator and uid_seg == seg and uid_seq.isdigit():
                next_seq = max(next_seq, int(uid_seq) + 1)
        return next_seq

    def _truncate_torn_ledger_tail(self, seg_dir: str) -> None:
        """Drop an incomplete final JSONL line so appends never extend a torn record."""
        path = os.path.join(seg_dir, _LEDGER)
        if not os.path.exists(path):
            return
        with open(path, "rb+") as f:
            data = f.read()
            if data and not data.endswith(b"\n"):
                logger.warning("Truncating incomplete final rollout-bank ledger record in %s", path)
                f.truncate(data.rfind(b"\n") + 1)
                f.flush()
                os.fsync(f.fileno())

    def _refresh_bytes_written(self, manifest: Manifest | None = None) -> None:
        """Recompute the durable footprint (ledgers, folded records, live backups)."""
        manifest = manifest or self._read_manifest()
        generation_dir = self._generation_dir(manifest)
        paths = [os.path.join(generation_dir, seg, _LEDGER) for seg in manifest["segments"]]
        paths += _fold_paths(self.bank_dir, manifest) + _backup_paths(self.bank_dir)
        self._bytes_written = sum(os.path.getsize(p) for p in paths if os.path.exists(p))

    def _close_handles(self) -> None:
        if self._ledger_f is not None:
            self._ledger_f.close()
        self._ledger_f = None

    def close(self) -> None:
        with self._lock:
            self._close_handles()

    # ---------------------------------------------------------------- append
    def append(self, group: RolloutGroup) -> str:
        """Write-through one completed group; return its stable uid."""
        with self._lock:
            assert self._seg_dir is not None, "set_collection() must be called before append()"
            uid = f"{_segment_name(self._collection_iter)}/{self._seq}"
            self._seq += 1
            record = self._encode(group, uid)
            record["checksum"] = _checksum(record)
            self._append_ledger(record)
            return uid

    def _append_ledger(self, record: LedgerRecord) -> None:
        created = False
        if self._ledger_f is None:
            path = os.path.join(self._seg_dir, _LEDGER)
            created = not os.path.exists(path)
            self._ledger_f = open(path, "a")
        self._ledger_f.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._ledger_f.flush()
        os.fsync(self._ledger_f.fileno())
        if created:
            _fsync_directory(self._seg_dir)

    def _encode(self, group: RolloutGroup, uid: str) -> LedgerRecord:
        """Build the thin JSONL manifest record for one group."""
        dumped = group.model_dump()
        dumped.pop("uid", None)  # uid lives at the top level of the record
        token_members = [isinstance(member, TokenRollout) for member in group.rollouts]
        if any(token_members) and not all(token_members):
            raise ValueError("RolloutGroup must not mix TokenRollout and Rollout members")
        token_typed = bool(token_members) and all(token_members)
        if token_typed:
            # Thin manifest: discard the arrays reconstructable from the request
            # records; refuse groups that could never be re-joined.
            for member in dumped["rollouts"]:
                trajectory = member.pop("trajectory")  # None: staged, the records hold the turns
                for field in _RECORD_DERIVED_FIELDS:
                    member.pop(field, None)
                completion_ids = member.get("completion_ids") or []
                if trajectory is not None and len(completion_ids) != len(trajectory):
                    raise ValueError(
                        f"token rollout has {len(trajectory)} turns but "
                        f"{len(completion_ids)} completion ids; thin manifests need one "
                        "request record per turn"
                    )
        return {
            "format_version": _FORMAT_VERSION,
            "uid": uid,
            "collection_iter": self._collection_iter,
            "kind": "token" if token_typed else "inline",
            "group": dumped,
        }

    def mark_consumed_many(self, uids: Iterable[str | None], iteration: int) -> None:
        """Durably record that the trainer pulled `uids` at `iteration` (one fsync per call)."""
        markers: list[ConsumedMarker] = [{"uid": uid, "iter": iteration} for uid in uids if uid]
        if not markers:
            return
        with self._lock:
            generation_dir = self._generation_dir(self._read_manifest())
            consumed_path = os.path.join(generation_dir, _CONSUMED)
            created = not os.path.exists(consumed_path)
            with open(consumed_path, "a") as f:
                f.write(
                    "".join(json.dumps(marker, separators=(",", ":")) + "\n" for marker in markers)
                )
                f.flush()
                os.fsync(f.fileno())
            if created:
                _fsync_directory(generation_dir)

    # --------------------------------------------------------------- restore
    def restore(
        self,
        trained_through: int,
        request_records: Optional[dict[str, FinishedRequestRecord]] = None,
    ) -> list[RolloutGroup]:
        """Replay the ledger and return groups not yet trained through ``T``.

        Token groups whose ``request_records`` are missing are dropped loudly.
        """
        with self._lock:
            return [group for _, group in self._survivors(trained_through, request_records)]

    def _survivors(
        self, trained_through: int, request_records: Optional[dict[str, FinishedRequestRecord]]
    ) -> list[tuple[LedgerRecord, RolloutGroup]]:
        """Surviving manifest records paired with their decoded groups (undecodable ones dropped)."""
        request_records = request_records or {}
        records, _, _ = self._restore_state(trained_through)
        pairs = ((record, self._decode(record, request_records)) for record in records)
        return [(record, group) for record, group in pairs if group is not None]

    def _restore_state(
        self, trained_through: int
    ) -> tuple[list[LedgerRecord], list[LedgerRecord], dict[str, int]]:
        """Return survivors, reclaimed records and the active generation's collapsed markers."""
        self._close_handles()  # flush any active writer before reading
        manifest = self._read_manifest()
        generation_dir = self._generation_dir(manifest)
        survivors: list[LedgerRecord] = []
        reclaimed: list[LedgerRecord] = []
        markers = self._read_markers(generation_dir)
        for seg in manifest["segments"]:
            seg_dir = os.path.join(generation_dir, seg)
            for uid, iteration in self._read_markers(seg_dir).items():
                markers[uid] = max(markers.get(uid, iteration), iteration)
            for record in self._read_ledger(seg_dir):
                marker_iter = markers.get(record["uid"])
                if marker_iter is not None and marker_iter <= trained_through:
                    reclaimed.append(record)  # trained into the loaded weights; drop
                else:
                    survivors.append(record)
        return survivors, reclaimed, markers

    def _read_markers(self, seg_dir: str) -> dict[str, int]:
        """Collapse ``consumed.log`` in ``seg_dir`` to ``uid -> latest consumed iteration``."""
        markers: dict[str, int] = {}
        path = os.path.join(seg_dir, _CONSUMED)
        if not os.path.exists(path):
            return markers
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug(f"Invalid JSON in {path}: {line}")
                    continue
                uid, iter = rec.get("uid"), rec.get("iter")
                if uid is not None and iter is not None:
                    markers[uid] = max(markers.get(uid, iter), iter)
        return markers

    def _read_ledger(self, seg_dir: str) -> Iterator[LedgerRecord]:
        """Yield the JSONL manifest records of one segment, skipping a torn final line."""
        path = os.path.join(seg_dir, _LEDGER)
        if not os.path.exists(path):
            return
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug(f"Invalid JSON in {path}: {line}")
                    continue  # torn final record from a mid-append kill
                _validate_format_version(record, path)
                yield record

    def _decode(
        self, record: LedgerRecord, request_records: dict[str, FinishedRequestRecord]
    ) -> Optional[RolloutGroup]:
        """Decode a manifest record; None if it is corrupt or its request records are unusable."""
        if _checksum({k: v for k, v in record.items() if k != "checksum"}) != record.get(
            "checksum"
        ):
            logger.debug(f"Checksum mismatch — dropping record: {record}")
            return None
        group_dict, uid = record["group"], record["uid"]
        if record["kind"] == "inline":
            members = [Rollout.model_validate(m) for m in group_dict["rollouts"]]
        else:
            members = []
            for m in group_dict["rollouts"]:
                turns = [request_records.get(cid) for cid in m.get("completion_ids") or []]
                if not all(_joinable(turn) for turn in turns):
                    logger.warning(
                        "Dropping banked rollout group %s: its request records are missing or "
                        "incomplete; its work will be regenerated.",
                        uid,
                    )
                    return None
                members.append(TokenRollout.model_validate({**m, **rebuild_turns(turns)}))
        group = RolloutGroup(
            rollouts=members,
            batch_id=group_dict.get("batch_id", 0),
            index_in_batch=group_dict.get("index_in_batch", 0),
        )
        group.uid = uid
        return group

    # ------------------------------------------------------------ compaction
    def retire_records(self, uids: Iterable[str]) -> None:
        """Completion ids no group will reference; pruned at the next checkpoint that sees them."""
        with self._lock:
            self._retired.update(uids)

    def checkpoint(self, iteration: int) -> None:
        """Compact survivors and future markers into an atomic new generation.

        Piggybacks the model-checkpoint boundary so the bank's compacted-through T
        tracks the checkpoint. Markers at or before ``iteration`` and their groups
        are reclaimed. Each surviving marker newer than the checkpoint is copied
        once so delayed asynchronous checkpoint finalization remains correct.
        Closed backup files are folded into the new generation and then pruned. A
        record is dropped once its group was reclaimed or rank 0 retired it; every
        other record is kept, so in-flight multi-window groups keep theirs across any
        number of checkpoints. The writer stays closed until the next `set_collection()`.
        """
        with self._lock:
            survivors, reclaimed, markers = self._restore_state(iteration)
            survivor_uids = {record["uid"] for record in survivors}
            retained_markers = {
                uid: marker_iter
                for uid, marker_iter in markers.items()
                if uid in survivor_uids and marker_iter > iteration
            }
            manifest = self._read_manifest()
            closed_backups = self._closed_backup_paths(iteration)
            available = load_finished_request_records(_fold_paths(self.bank_dir, manifest))
            available |= load_finished_request_records(closed_backups)
            released = set(self._referenced_records(reclaimed, available)) | self._retired
            folded = {uid: r for uid, r in available.items() if uid not in released}
            folded |= self._referenced_records(survivors, available)
            self._retired -= set(available)
            self._publish_generation(
                iteration, survivors, retained_markers, folded, timeline=manifest["timeline"]
            )
            for path in closed_backups:
                try:
                    os.remove(path)
                except OSError:
                    logger.warning("Could not prune consumed request-record backup %s", path)

    def recover(
        self,
        trained_through: int,
        request_records: Optional[dict[str, FinishedRequestRecord]] = None,
    ) -> list[RolloutGroup]:
        """Rebase survivors at ``T`` into a new timeline with no old markers.

        Groups whose request records did not survive the crash are dropped loudly
        and excluded; old backup files are pruned at the next checkpoint.
        """
        with self._lock:
            pairs = self._survivors(trained_through, request_records)
            survivors = [record for record, _ in pairs]
            self._publish_generation(
                trained_through,
                survivors,
                {},
                self._referenced_records(survivors, request_records or {}),
                timeline=uuid.uuid4().hex,
            )
            return [group for _, group in pairs]

    def _closed_backup_paths(self, iteration: int) -> list[str]:
        """Backups of windows before ``iteration``; that window's own file may still be open."""
        return [
            path
            for path in _backup_paths(self.bank_dir)
            if int(_BACKUP_NAME.match(os.path.basename(path)).group(1)) < iteration
        ]

    @staticmethod
    def _referenced_records(
        records: list[LedgerRecord], available: dict[str, FinishedRequestRecord]
    ) -> dict[str, FinishedRequestRecord]:
        """Subset of ``available`` referenced by the given manifests' completion ids."""
        referenced: dict[str, FinishedRequestRecord] = {}
        for record in records:
            if record["kind"] != "token":
                continue
            for member in record["group"]["rollouts"]:
                for cid in member.get("completion_ids") or []:
                    if cid in available:
                        referenced[cid] = available[cid]
        return referenced

    def _publish_generation(
        self,
        iteration: int,
        records: list[LedgerRecord],
        markers: dict[str, int],
        request_records: dict[str, FinishedRequestRecord],
        *,
        timeline: str,
    ) -> None:
        """Write and atomically select a complete ledger/marker generation."""
        self._close_handles()
        old_generation = self._read_manifest()["active_generation"]
        generation = _generation_name()
        staging = os.path.join(self._generations_dir, f".{generation}.tmp")
        final_dir = os.path.join(self._generations_dir, generation)
        os.makedirs(staging)
        _fsync_directory(self._generations_dir)

        new_seg = _segment_name(iteration)
        new_seg_dir = os.path.join(staging, new_seg)
        os.makedirs(new_seg_dir)
        _fsync_directory(staging)
        self._rewrite_segment(new_seg_dir, records, request_records)
        self._write_markers(staging, markers)
        _fsync_directory(staging)

        os.replace(staging, final_dir)
        _fsync_directory(self._generations_dir)
        self._write_manifest_atomic(
            {
                "format_version": _FORMAT_VERSION,
                "active_generation": generation,
                "timeline": timeline,
                "trained_through": iteration,
                "segments": [new_seg],
            }
        )

        # The manifest now references only the compacted survivor segment; rebase the
        # footprint accounting and let a future cap crossing warn again.
        self._refresh_bytes_written()
        self._warned_over_cap = False
        self._maybe_warn_over_cap()

        # The old generation remains a valid crash-recovery target until the
        # manifest flip is durable. It is safe to reclaim only afterward.
        shutil.rmtree(os.path.join(self._generations_dir, old_generation), ignore_errors=True)
        _fsync_directory(self._generations_dir)
        self._collection_iter = None
        self._seg_dir = None

    def _write_markers(self, generation_dir: str, markers: dict[str, int]) -> None:
        """Write one durable marker per uid into a not-yet-published generation."""
        path = os.path.join(generation_dir, _CONSUMED)
        with open(path, "w") as f:
            for uid, iteration in sorted(markers.items()):
                marker: ConsumedMarker = {"uid": uid, "iter": iteration}
                f.write(json.dumps(marker, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())
        _fsync_directory(generation_dir)

    def _rewrite_segment(
        self,
        seg_dir: str,
        records: list[LedgerRecord],
        request_records: dict[str, FinishedRequestRecord],
    ) -> None:
        """Write raw records into `seg_dir`."""
        with open(os.path.join(seg_dir, _LEDGER), "w") as f:
            for record in records:
                f.write(json.dumps(record, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())
        if request_records:
            with open(os.path.join(seg_dir, _REQUEST_RECORDS), "wb") as f:
                for uid, record in sorted(request_records.items()):
                    f.write(pack_finished_request_record(uid, record))
                f.flush()
                os.fsync(f.fileno())
        _fsync_directory(seg_dir)

    def _maybe_warn_over_cap(self) -> None:
        if self.max_bytes <= 0 or self._bytes_written <= self.max_bytes:
            return
        if not self._warned_over_cap:
            logger.warning(
                "RolloutBank exceeded --rl-rollout-bank-max-bytes (%d > %d); will compact at the "
                "next checkpoint. Consider a shorter checkpoint interval or lower --rl-generation-lag.",
                self._bytes_written,
                self.max_bytes,
            )
            self._warned_over_cap = True
