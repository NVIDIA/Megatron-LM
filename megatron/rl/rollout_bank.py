# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Durable rollout bank — rollout-granular persistence with incomplete-group replay.

Persists each rollout the instant it is graded, so a SIGKILL (the 4h SLURM limit)
does not destroy finished decode work. A group whose members were only partly
persisted reads back as an *incomplete* group carrying the problem state needed to
regenerate the rest against the same prompt. In-flight decode state (a rollout
killed mid-generation) is still lost.

The ledger has exactly two record kinds:

    problem  {group_uid, problem_state}   one per group, written at prepare
    rollout  {group_uid, rollout_idx, ..} one per graded member

There is deliberately no seal record and no tombstone. A group is complete iff it
has ``rollouts_per_group`` distinct members, which is config (asserted against the
manifest on load). A zero-variance group is re-filtered from its persisted rewards
rather than marked; re-deriving has no crash window between the last member's write
and a marker's, which a tombstone would.

Layout (single-writer, rank-0 only, on Lustre)::

    <bank_dir>/
        MANIFEST.json                 # atomically selects one complete generation
        generations/
            generation-<uuid>/
                consumed.log          # append-only markers {"uid", "iter"}
                gen-<iter>/
                    ledger.log        # append-only JSONL, one self-describing index
                    tokens.bin        # int32 token ids (offset-indexed sidecar)
                    logprobs.bin       # fp16 generation logprobs (offset-indexed sidecar)
                    masks.bin          # uint8 generation masks (offset-indexed sidecar)

Why sidecars: the production ``TokenRollout`` carries a token id and a logprob per
generated token; as JSON text that is ~28 B/token (logprobs alone ~18 B, ~9x their
binary size). The bulk per-token arrays go to append-only binary sidecars and the
JSONL keeps only a small, greppable, self-describing index record that points at
each slice by ``(offset, bytes, lengths)``.

NOTE: ``_append_sidecar`` / ``_load_sidecar`` deliberately mirror the offset-indexed
sidecar format in ``megatron/core/transformer/moe/router_trace.py`` (see
``load_hidden_states_for_record``). That code is inlined into MoE-coupled forward
hooks and hardcodes bf16/filenames, so it is not importable here. Unifying the two
onto a shared parameterized primitive is a tracked follow-up.

This module is a passive, synchronous store: ``write_records`` encodes them and
fsyncs before returning. It owns no thread and no queue. Batching lives in
``RolloutPipeline.stage_bank``, which coalesces records and calls ``write_records``
through ``asyncio.to_thread`` so the blocking fsync never stalls the event loop.
That single-consumer stage is also what serializes access: exactly one write is in
flight at a time, and the pipeline drains it before any segment switch or compaction.

Because writes are queued in the pipeline, a kill can lose one un-drained set of
freshly graded rollout records, which then regenerate. It cannot corrupt: a torn record is
dropped by the tail truncation and per-record checksum, and a missing member only
makes a group look *less* complete — precisely the incomplete-group case this module
already handles.

A checkpoint writes a complete new generation, including a compacted marker log, before
atomically flipping the manifest. ``restore`` replays the active generation and returns
every group that is not already trained through checkpoint step ``T``:

    marker <= T -> discard (training already in the loaded weights)
    marker  > T -> restore (that training step was erased by the kill)
    no marker   -> restore (never consumed, or lost in the kill)

``recover(T)`` applies those rules once, then publishes the survivors in a new
timeline with an empty marker log. Markers from work after ``T`` therefore cannot
incorrectly suppress a group when the restarted timeline later passes their old
iteration number.
"""

import hashlib
import json
import logging
import os
import pathlib
import threading
import uuid
from typing import Iterable, Iterator, Literal, NamedTuple, NotRequired, Optional, TypedDict

import numpy as np

from megatron.rl.types import Rollout, RolloutGroup, TokenRollout

logger = logging.getLogger(__name__)

# Sidecar dtypes. Token ids are exact; logprobs default to fp16 to halve the
# footprint. The IS-correction ratio is exp(old - inference), so logprob
# round-trip error is exponentiated — if fp16 ever drifts the ratio stats, flip
# this one constant to np.float32 (still ~4-5x smaller than JSON text).
_TOKEN_DTYPE = np.int32
_LOGPROB_DTYPE = np.float16
_MASK_DTYPE = np.uint8

# Bump whenever the manifest/ledger schema or any sidecar dtype/layout changes.
# Readers intentionally fail closed; cross-version migration must be explicit.
_FORMAT_VERSION = 3
_ZERO_VARIANCE_TOLERANCE = 1e-6
_MANIFEST = "MANIFEST.json"
_GENERATIONS = "generations"
_LEDGER = "ledger.log"
_CONSUMED = "consumed.log"
_TOKENS_BIN = "tokens.bin"
_LOGPROBS_BIN = "logprobs.bin"
_MASKS_BIN = "masks.bin"


class SidecarMeta(TypedDict):
    """
    Where one record's packed array lives in its sidecar .bin file.

    Args:
        offset: The offset of the packed array in the sidecar .bin file.
        bytes: The number of bytes in the packed array.
        lengths: The lengths of the packed array. This is a list of lists, where the inner list is the lengths of the turns in the member.
    """

    offset: int
    bytes: int
    lengths: list[list[int]]  # [member][turn] token counts


class LedgerRecord(TypedDict):
    """
    One self-describing JSONL index record for a single graded rollout.

    Args:
        format_version: The persisted schema and sidecar-layout version.
        kind: Always ``"rollout"``.
        uid: The rollout's unique identifier, ``"<group_uid>#<rollout_idx>"``.
        group_uid: The owning group's unique identifier.
        rollout_idx: Which slot in the group this member occupies.
        collection_iter: The iteration number of the collection.
        member_type: Whether the member is a ``Rollout`` or a ``TokenRollout``.
        member: The rollout, with bulk per-token arrays stripped into sidecars.
        tok: Token-id sidecar slice. Present only for token-typed members.
        lp: Logprob sidecar slice. Present only when logprobs were provided.
        mask: Generation-mask sidecar slice. Present only when a mask was provided.
        checksum: Digest over the record and its raw sidecar slices.
    """

    format_version: int
    kind: Literal["rollout"]
    uid: str
    group_uid: str
    rollout_idx: int
    collection_iter: int
    member_type: Literal["Rollout", "TokenRollout"]
    member: dict
    tok: NotRequired[SidecarMeta]
    lp: NotRequired[SidecarMeta]
    mask: NotRequired[SidecarMeta]
    checksum: NotRequired[str]


class ProblemRecord(TypedDict):
    """
    One JSONL record carrying the state needed to regenerate a group's members.

    Written once per group, at prepare time, before any of its members exist. The
    writer is FIFO and a member cannot be enqueued until its inference finishes, so
    this record always precedes its members on disk without an explicit barrier.

    Args:
        format_version: The persisted schema and sidecar-layout version.
        kind: Always ``"problem"``.
        group_uid: The owning group's unique identifier.
        collection_iter: The iteration number of the collection.
        problem_state: Agent-owned opaque payload; the bank never inspects it.
        checksum: Digest over the record.
    """

    format_version: int
    kind: Literal["problem"]
    group_uid: str
    collection_iter: int
    problem_state: dict
    checksum: NotRequired[str]


class Manifest(TypedDict):
    """
    Bank-level index: how far training got and which segments exist.

    Args:
        format_version: The persisted schema and sidecar-layout version.
        active_generation: The directory containing the complete active bank state.
        timeline: An opaque identifier for the execution history owning the markers.
        trained_through: The iteration number of the last checkpoint.
        segments: The list of segment names (e.g. ["gen-000000", "gen-000001"]).
        compacted_at: The iteration number of the last compaction.
        rollouts_per_group: Group size this bank's records were written under.
            Completeness is inferred from the member count, so a resume that
            changes it cannot read these records correctly and is rejected.
    """

    format_version: int
    active_generation: str
    timeline: str
    trained_through: int
    segments: list[str]
    compacted_at: int
    rollouts_per_group: int


class ConsumedMarker(TypedDict):
    """Append-only marker: a group uid was pulled by the trainer at ``iter``."""

    uid: str
    iter: int


class EncodedRecord(NamedTuple):
    """Ledger record and packed sidecar payloads for one rollout."""

    record: LedgerRecord
    tok_bytes: bytes
    lp_bytes: bytes
    mask_bytes: bytes


class _PendingProblem(NamedTuple):
    """Queued ``problem`` record awaiting the writer thread."""

    group_uid: str
    problem_state: dict


class _PendingRollout(NamedTuple):
    """Queued ``rollout`` record awaiting the writer thread."""

    group_uid: str
    rollout_idx: int
    rollout: "Rollout | TokenRollout"


def _segment_name(iteration: int) -> str:
    """Generate a segment name for a given iteration."""
    return f"gen-{iteration:06d}"


def _generation_name() -> str:
    """Return a unique immutable generation directory name."""
    return f"generation-{uuid.uuid4().hex}"


def _checksum(record_wo_checksum: dict, *slices: bytes) -> str:
    """Digest over the canonical index record plus its raw sidecar slices."""
    h = hashlib.blake2b(digest_size=16)
    h.update(json.dumps(record_wo_checksum, sort_keys=True, separators=(",", ":")).encode())
    for s in slices:
        h.update(s)
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
    """Single-writer durable store for graded rollouts (rank-0 only).

    Args:
        bank_dir: Directory owning the manifest, generations, and segments.
        max_bytes: Soft cap on live sidecar payload; exceeding it warns.
        rollouts_per_group: Group size. Completeness is inferred from the member
            count, so this is written to the manifest and asserted on load.
        drop_zero_variance: Whether a fully persisted group whose members all
            scored the same is discarded at restore and reclaimed at compaction.
    """

    def __init__(
        self,
        bank_dir: str,
        *,
        max_bytes: int = 0,
        rollouts_per_group: int = 1,
        drop_zero_variance: bool = False,
    ) -> None:
        self.bank_dir = bank_dir
        self.max_bytes = max_bytes
        self.rollouts_per_group = rollouts_per_group
        self.drop_zero_variance = drop_zero_variance
        os.makedirs(bank_dir, exist_ok=True)
        self._lock = threading.RLock()
        self._collection_iter: Optional[int] = None
        self._seg_dir: Optional[str] = None
        self._run_nonce = uuid.uuid4().hex[:12]
        self._seq = 0
        # Open sidecar/ledger handles for the active segment, with running offsets.
        self._ledger_f = None
        self._tok_f = None
        self._lp_f = None
        self._mask_f = None
        self._tok_off = 0
        self._lp_off = 0
        self._mask_off = 0
        self._bytes_written = 0
        self._warned_over_cap = False
        if not os.path.exists(self._manifest_path):
            if os.listdir(self.bank_dir):
                raise FileNotFoundError(
                    f"RolloutBank manifest is missing at {self._manifest_path}, but "
                    f"{self.bank_dir} is not empty. Refusing to overwrite possible bank data; "
                    "recover MANIFEST.json or remove the rollout-bank directory."
                )
            os.makedirs(self._generations_dir)
            _fsync_directory(self.bank_dir)
            generation = self._create_empty_generation()
            self._write_manifest_atomic(
                {
                    "format_version": _FORMAT_VERSION,
                    "active_generation": generation,
                    "timeline": uuid.uuid4().hex,
                    "trained_through": 0,
                    "segments": [],
                    "compacted_at": 0,
                    "rollouts_per_group": rollouts_per_group,
                }
            )
        manifest = self._read_manifest()
        self._garbage_collect_generations(manifest)
        # Read through the published manifest immediately, both to initialize
        # live-size accounting and to fail closed on malformed existing state.
        self._bytes_written = self._manifest_sidecar_bytes(manifest)
        self._maybe_warn_over_cap()

    def active_segment_path(self, name: str) -> pathlib.Path:
        """Path to a file in the segment currently being appended to."""
        assert self._seg_dir is not None, "set_collection() must be called first"
        return pathlib.Path(self._seg_dir) / name

    # ------------------------------------------------------------------ paths
    @property
    def _manifest_path(self) -> str:
        return os.path.join(self.bank_dir, _MANIFEST)

    @property
    def _generations_dir(self) -> str:
        return os.path.join(self.bank_dir, _GENERATIONS)

    def _generation_dir(self, manifest: Manifest) -> str:
        return os.path.join(self._generations_dir, manifest["active_generation"])

    def _active_generation_dir(self) -> str:
        return self._generation_dir(self._read_manifest())

    def _create_empty_generation(self) -> str:
        generation = _generation_name()
        generation_dir = os.path.join(self._generations_dir, generation)
        os.makedirs(generation_dir)
        consumed_path = os.path.join(generation_dir, _CONSUMED)
        with open(consumed_path, "w") as f:
            f.flush()
            os.fsync(f.fileno())
        _fsync_directory(generation_dir)
        _fsync_directory(self._generations_dir)
        return generation

    def _garbage_collect_generations(self, manifest: Manifest) -> None:
        """Remove recognized generation directories not selected by ``manifest``."""
        active = manifest["active_generation"]
        removed = False
        for name in os.listdir(self._generations_dir):
            path = os.path.join(self._generations_dir, name)
            is_generation = name.startswith("generation-")
            is_staging = name.startswith(".generation-") and name.endswith(".tmp")
            if name != active and os.path.isdir(path) and (is_generation or is_staging):
                _rmtree(path)
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
                "recover MANIFEST.json or remove the rollout-bank directory."
            ) from exc
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Malformed RolloutBank manifest at {self._manifest_path}: {exc.msg} "
                f"(line {exc.lineno}, column {exc.colno}). Refusing to continue; "
                "recover MANIFEST.json or remove the rollout-bank directory."
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
        persisted_group_size = manifest.get("rollouts_per_group")
        if persisted_group_size != self.rollouts_per_group:
            raise ValueError(
                f"RolloutBank at {self.bank_dir} was written with rollouts_per_group="
                f"{persisted_group_size!r} but this run uses {self.rollouts_per_group!r}. "
                "Group completeness is inferred from the member count, so these records "
                "cannot be read correctly. Resume with a matching config or clear the "
                "rollout-bank directory."
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
        """Point subsequent appends at the ``gen-<iteration>/`` segment.

        Sidecar offsets are per-segment, so callers must drain queued records first:
        a record written after this returns would use offsets from the old segment.
        """
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
        self._tok_off = self._file_size(_TOKENS_BIN)
        self._lp_off = self._file_size(_LOGPROBS_BIN)
        self._mask_off = self._file_size(_MASKS_BIN)
        if seg not in manifest["segments"]:
            manifest["segments"].append(seg)
            self._write_manifest_atomic(manifest)
            # A directory left behind before manifest publication was not live
            # at startup, but becomes live when this segment is published.
            self._bytes_written += self._segment_sidecar_bytes(self._seg_dir)
            self._maybe_warn_over_cap()

    def reserve_group_uid(self) -> str:
        """Reserve a group's durable identity before any of its members exist.

        Called at prepare time: the first member to finish must already be able
        to name its group, and a regenerated group must not reuse a slot's uid.
        """
        with self._lock:
            uid = f"{self._run_nonce}/{self._seq}"
            self._seq += 1
            return uid

    def _truncate_torn_ledger_tail(self, seg_dir: str) -> None:
        """Remove an incomplete final JSONL line before resuming appends."""
        path = os.path.join(seg_dir, _LEDGER)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return

        with open(path, "rb+") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) == b"\n":
                return

            end = f.tell()
            truncate_at = 0
            while end > 0:
                start = max(0, end - 8192)
                f.seek(start)
                chunk = f.read(end - start)
                newline = chunk.rfind(b"\n")
                if newline >= 0:
                    truncate_at = start + newline + 1
                    break
                end = start

            logger.warning("Truncating incomplete final rollout-bank ledger record in %s", path)
            f.truncate(truncate_at)
            f.flush()
            os.fsync(f.fileno())

    def _file_size(self, name: str) -> int:
        path = os.path.join(self._seg_dir, name)
        return os.path.getsize(path) if os.path.exists(path) else 0

    @staticmethod
    def _segment_sidecar_bytes(seg_dir: str) -> int:
        """Return the on-disk payload bytes in one segment."""
        return sum(
            os.path.getsize(path)
            for name in (_TOKENS_BIN, _LOGPROBS_BIN, _MASKS_BIN)
            if os.path.exists(path := os.path.join(seg_dir, name))
        )

    def _manifest_sidecar_bytes(self, manifest: Manifest | None = None) -> int:
        """Return payload bytes referenced by the currently published manifest."""
        manifest = manifest or self._read_manifest()
        generation_dir = self._generation_dir(manifest)
        return sum(
            self._segment_sidecar_bytes(os.path.join(generation_dir, seg))
            for seg in manifest["segments"]
        )

    def _close_handles(self) -> None:
        for f in (self._ledger_f, self._tok_f, self._lp_f, self._mask_f):
            if f is not None:
                f.close()
        self._ledger_f = self._tok_f = self._lp_f = self._mask_f = None

    def close(self) -> None:
        """Release the segment's file handles."""
        self._close_handles()

    # ---------------------------------------------------------------- append
    def append_problem(self, group_uid: str, problem_state: dict) -> None:
        """Write this group's regeneration state immediately.

        Batching is the caller's job: the pipeline queues records and writes them
        coalesced from ``stage_bank``. This one-record path exists for
        compaction and for direct use in tests.
        """
        self.write_records([_PendingProblem(group_uid, problem_state)])

    def append_rollout(
        self, group_uid: str, rollout_idx: int, rollout: "Rollout | TokenRollout"
    ) -> None:
        """Write one graded member of ``group_uid`` immediately."""
        self.write_records([_PendingRollout(group_uid, rollout_idx, rollout)])

    def append(
        self,
        group: "RolloutGroup",
        uid: str | None = None,
        *,
        problem_state: dict | None = None,
    ) -> str:
        """Write a whole group as its individual member records; return its uid.

        A uid is assigned for a new group or preserved when rewriting an existing
        group during compaction. Member slots come from ``group.member_indices``
        when present, so a restored incomplete group keeps its original slots.
        """
        if uid is None:
            uid = group.uid or self.reserve_group_uid()
        state = problem_state if problem_state is not None else group.problem_state
        if state is not None:
            self.append_problem(uid, state)
        for rollout_idx, rollout in zip(group.member_indices, group.rollouts, strict=True):
            self.append_rollout(uid, rollout_idx, rollout)
        return uid

    # --------------------------------------------------------- writer thread
    def write_records(self, pending: list) -> None:
        """Encode and durably write one set of ledger records.

        Sidecar slices are written before the index records that point at them, so
        a torn write can only ever lose trailing index lines. One fsync per touched
        file per write, rather than per record.

        Blocking. The pipeline calls this through ``asyncio.to_thread`` so the fsync
        never runs on the event loop. Serialization is the caller's responsibility:
        exactly one ``write_records`` may be in flight, and no segment mutation may
        overlap one.
        """
        assert self._seg_dir is not None, "set_collection() must be called before appending"
        records: list[dict] = []
        payload_bytes = 0
        for item in pending:
            if isinstance(item, _PendingProblem):
                records.append(self._encode_problem(item))
                continue
            encoded = self._encode_rollout(item)
            for data, handle, name in (
                (encoded.tok_bytes, "_tok_f", _TOKENS_BIN),
                (encoded.lp_bytes, "_lp_f", _LOGPROBS_BIN),
                (encoded.mask_bytes, "_mask_f", _MASKS_BIN),
            ):
                if data:
                    self._write_sidecar(handle, name, data)
            records.append(encoded.record)
            payload_bytes += (
                len(encoded.tok_bytes) + len(encoded.lp_bytes) + len(encoded.mask_bytes)
            )

        for handle in (self._tok_f, self._lp_f, self._mask_f):
            if handle is not None:
                handle.flush()
                os.fsync(handle.fileno())
        self._append_ledger(records)

        self._bytes_written += payload_bytes
        self._maybe_warn_over_cap()

    def _write_sidecar(self, handle_attr: str, name: str, data: bytes) -> None:
        f = getattr(self, handle_attr)
        created = False
        if f is None:
            path = os.path.join(self._seg_dir, name)
            created = not os.path.exists(path)
            f = open(path, "ab")
            setattr(self, handle_attr, f)
        f.write(data)
        if created:
            _fsync_directory(self._seg_dir)

    def _append_ledger(self, records: list[dict]) -> None:
        """Append index records with a single fsync."""
        if not records:
            return
        created = False
        if self._ledger_f is None:
            path = os.path.join(self._seg_dir, _LEDGER)
            created = not os.path.exists(path)
            self._ledger_f = open(path, "a")
        self._ledger_f.write(
            "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records)
        )
        self._ledger_f.flush()
        os.fsync(self._ledger_f.fileno())
        if created:
            _fsync_directory(self._seg_dir)

    def _encode_problem(self, pending: _PendingProblem) -> dict:
        """Build the JSONL record carrying one group's regeneration state."""
        assert self._collection_iter is not None, "set_collection() must precede encoding"
        record: ProblemRecord = {
            "format_version": _FORMAT_VERSION,
            "kind": "problem",
            "group_uid": pending.group_uid,
            "collection_iter": self._collection_iter,
            "problem_state": pending.problem_state,
        }
        record["checksum"] = _checksum(dict(record))
        return dict(record)

    def _encode_rollout(self, pending: _PendingRollout) -> EncodedRecord:
        """Build the JSONL index record + packed sidecar bytes for one rollout."""
        assert self._collection_iter is not None, "set_collection() must precede encoding"
        rollout = pending.rollout
        dumped = rollout.model_dump()
        token_typed = isinstance(rollout, TokenRollout)
        member_type: Literal["Rollout", "TokenRollout"] = (
            "TokenRollout" if token_typed else "Rollout"
        )
        record: LedgerRecord = {
            "format_version": _FORMAT_VERSION,
            "kind": "rollout",
            "uid": f"{pending.group_uid}#{pending.rollout_idx}",
            "group_uid": pending.group_uid,
            "rollout_idx": pending.rollout_idx,
            "collection_iter": self._collection_iter,
            "member_type": member_type,
            "member": dumped,
        }

        if not token_typed:
            record["checksum"] = _checksum(dict(record))
            return EncodedRecord(record, b"", b"", b"")

        trajectory = dumped.pop("trajectory")
        tok_bytes = np.asarray(
            [token for turn in trajectory for token in turn], dtype=_TOKEN_DTYPE
        ).tobytes()
        record["tok"] = {
            "offset": self._tok_off,
            "bytes": len(tok_bytes),
            "lengths": [len(turn) for turn in trajectory],
        }

        lp_bytes = b""
        logprobs = dumped.pop("logprobs", None)
        if logprobs is not None:
            lp_bytes = np.asarray(
                [value for turn in logprobs for value in turn], dtype=_LOGPROB_DTYPE
            ).tobytes()
            record["lp"] = {
                "offset": self._lp_off,
                "bytes": len(lp_bytes),
                "lengths": [len(turn) for turn in logprobs],
            }

        mask_bytes = b""
        mask = dumped.pop("generation_mask", None)
        if mask is not None:
            mask_bytes = (
                np.asarray([value for turn in mask for value in turn], dtype=bool)
                .astype(_MASK_DTYPE)
                .tobytes()
            )
            record["mask"] = {
                "offset": self._mask_off,
                "bytes": len(mask_bytes),
                "lengths": [len(turn) for turn in mask],
            }

        # Advance running offsets now that this record's slices are placed.
        self._tok_off += len(tok_bytes)
        self._lp_off += len(lp_bytes)
        self._mask_off += len(mask_bytes)

        record["checksum"] = _checksum(dict(record), tok_bytes, lp_bytes, mask_bytes)
        return EncodedRecord(record, tok_bytes, lp_bytes, mask_bytes)

    def mark_consumed(self, uid: str | None, iteration: int) -> None:
        """Record that ``uid`` was pulled by the trainer at ``iteration``.

        Markers append to the active generation. Checkpoint compaction carries
        forward one marker per surviving uid; recovery starts a new timeline
        with no markers from the rolled-back future.

        Args:
            uid: The unique identifier of the group.
            iteration: The iteration number of the training.

        Returns:
            None

        Example:
            uid = "gen-000000/0"
            iteration = 100
            consumed.log:
                {"uid": "gen-000000/0", "iter": 100}
        """
        self.mark_consumed_many([uid], iteration)

    def mark_consumed_many(self, uids: Iterable[str | None], iteration: int) -> None:
        """Durably record one trainer collection's consumption markers.

        All non-empty ``uids`` are appended with one file open and one ``fsync``.
        They are therefore durable before this method returns without paying
        one filesystem synchronization per rollout group.

        Args:
            uids: Unique identifiers of groups pulled by the trainer.
            iteration: The iteration number of the training.

        Returns:
            None
        """
        markers: list[ConsumedMarker] = [{"uid": uid, "iter": iteration} for uid in uids if uid]
        if not markers:
            return
        with self._lock:
            generation_dir = self._active_generation_dir()
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

    def restore(self, trained_through: int) -> list["RolloutGroup"]:
        """Replay the ledger and return groups not yet trained through ``T``."""
        return self._restore_state(trained_through)[0]

    def _restore_state(self, trained_through: int) -> tuple[list["RolloutGroup"], dict[str, int]]:
        """Return survivors and the active generation's collapsed markers.

        Folds the rollout-granular ledger back into groups:

            members == rollouts_per_group -> complete; drop if zero-variance
            0 < members < that            -> incomplete; needs a problem record
            no members                    -> nothing worth restoring

        Callers must have drained any queued records first, so everything appended
        so far is already durable and no write can land mid-read.
        """
        self._close_handles()
        manifest = self._read_manifest()
        generation_dir = self._generation_dir(manifest)
        markers = self._read_markers(generation_dir)
        problems: dict[str, dict] = {}
        members: dict[str, dict[int, "Rollout | TokenRollout"]] = {}
        order: list[str] = []

        for seg in manifest["segments"]:
            seg_dir = os.path.join(generation_dir, seg)
            for uid, iteration in self._read_markers(seg_dir).items():
                markers[uid] = max(markers.get(uid, iteration), iteration)
            for record in self._read_ledger(seg_dir):
                group_uid = record.get("group_uid")
                if group_uid is None:
                    logger.debug(f"Ledger record without a group_uid — dropping: {record}")
                    continue
                marker_iter = markers.get(group_uid)
                if marker_iter is not None and marker_iter <= trained_through:
                    continue  # trained into the loaded weights; drop
                if group_uid not in members:
                    members[group_uid] = {}
                    order.append(group_uid)
                if record["kind"] == "problem":
                    if self._problem_is_intact(record):
                        problems[group_uid] = record["problem_state"]
                    continue
                rollout = self._decode_rollout(record, seg_dir)
                if rollout is not None:
                    members[group_uid][record["rollout_idx"]] = rollout

        restored: list["RolloutGroup"] = []
        for group_uid in order:
            group = self._assemble_restored(group_uid, members[group_uid], problems.get(group_uid))
            if group is not None:
                restored.append(group)
        return restored, markers

    def _assemble_restored(
        self,
        group_uid: str,
        members: dict[int, "Rollout | TokenRollout"],
        problem_state: Optional[dict],
    ) -> Optional["RolloutGroup"]:
        """Turn one group's decoded members into a restorable group, or drop it."""
        if not members:
            return None
        indices = sorted(members)
        rollouts = [members[index] for index in indices]

        if len(indices) >= self.rollouts_per_group:
            if self.drop_zero_variance and _is_zero_variance(rollouts):
                logger.debug(f"Dropping zero-variance restored group {group_uid}")
                return None
        elif problem_state is None:
            logger.debug(
                f"Dropping incomplete restored group {group_uid} "
                f"({len(indices)}/{self.rollouts_per_group} members, no problem state)"
            )
            return None

        return RolloutGroup(
            rollouts=rollouts,
            uid=group_uid,
            member_indices=indices,
            problem_state=problem_state,
        )

    @staticmethod
    def _problem_is_intact(record: dict) -> bool:
        """Verify a problem record's checksum before trusting its payload."""
        expected = _checksum({k: v for k, v in record.items() if k != "checksum"})
        if expected != record.get("checksum"):
            logger.debug(f"Checksum mismatch — dropping problem record: {record}")
            return False
        return True

    def _read_markers(self, seg_dir: str) -> dict:
        """
        Load consumption markers from a directory.
        A marker records that the trainer pulled a group (``uid``) at some
        training iteration (``iter``). Markers live as append-only JSONL in
        ``consumed.log`` and are collapsed to ``uid -> latest (max) consumed
        iteration`` for restore and compaction.

        Args:
            seg_dir: The bank or legacy segment directory containing the markers.

        Returns:
            A dictionary of {uid: latest (max) consumed iteration for this segment}.

        Example:
            seg_dir = "gen-000000"
            consumed.log:
                {"uid": "gen-000000/0", "iter": 100}
                {"uid": "gen-000000/1", "iter": 101}
                {"uid": "gen-000000/0", "iter": 105}
            return:
                {"gen-000000/0": 105, "gen-000000/1": 101}
        """

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
        """
        Read the ledger for a segment and yield the ledger records.
        The ledger is a JSONL file that contains the ledger records for the segment.
        Each line is a JSON object.

        Args:
            seg_dir: The directory of the segment.

        Returns:
            An iterator of ledger records.
        """
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

    def _decode_rollout(
        self, record: LedgerRecord, seg_dir: str
    ) -> "Optional[Rollout | TokenRollout]":
        """
        Decode one ledger record into a single rollout.

        Args:
            record: The ledger record to decode.
            seg_dir: The directory of the segment.

        Returns:
            A Rollout/TokenRollout, or None if the record is corrupted or truncated.
        """
        tok_bytes = self._read_slice(seg_dir, _TOKENS_BIN, record.get("tok"))
        lp_bytes = self._read_slice(seg_dir, _LOGPROBS_BIN, record.get("lp"))
        mask_bytes = self._read_slice(seg_dir, _MASKS_BIN, record.get("mask"))
        if tok_bytes is None or lp_bytes is None or mask_bytes is None:
            logger.debug(f"Sidecar slice was short (truncated) — dropping record: {record}")
            return None

        expected = _checksum(
            {k: v for k, v in record.items() if k != "checksum"}, tok_bytes, lp_bytes, mask_bytes
        )
        if expected != record.get("checksum"):
            logger.debug(f"Checksum mismatch — dropping record: {record}")
            return None

        member = dict(record["member"])
        if record["member_type"] != "TokenRollout":
            return Rollout.model_validate(member)

        tokens = np.frombuffer(tok_bytes, dtype=_TOKEN_DTYPE).tolist()
        member["trajectory"] = self._unflatten(tokens, record["tok"]["lengths"])
        member["logprobs"] = (
            self._unflatten(
                np.frombuffer(lp_bytes, dtype=_LOGPROB_DTYPE).astype(np.float32).tolist(),
                record["lp"]["lengths"],
            )
            if "lp" in record
            else None
        )
        member["generation_mask"] = (
            self._unflatten(
                np.frombuffer(mask_bytes, dtype=_MASK_DTYPE).astype(bool).tolist(),
                record["mask"]["lengths"],
            )
            if "mask" in record
            else None
        )
        # Completion IDs point into the inference engine's process-local metadata
        # ledger. That ledger does not survive a crash, so recovered rollouts must
        # not attempt to join against stale IDs.
        member["completion_ids"] = []
        return TokenRollout.model_validate(member)

    @staticmethod
    def _read_slice(seg_dir: str, name: str, meta: Optional[SidecarMeta]) -> Optional[bytes]:
        """Return the record's sidecar bytes, or None if the slice is truncated."""
        if not meta or meta["bytes"] == 0:
            return b""
        path = os.path.join(seg_dir, name)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            f.seek(meta["offset"])
            data = f.read(meta["bytes"])
        return data if len(data) == meta["bytes"] else None

    @staticmethod
    def _unflatten(flat: list, lengths: list) -> list:
        """Split one rollout's flat array back into per-turn nesting."""
        out, pos = [], 0
        for length in lengths:
            out.append(flat[pos : pos + length])
            pos += length
        return out

    def checkpoint(self, iteration: int) -> None:
        """Compact survivors and future markers into an atomic new generation.

        Piggybacks the model-checkpoint boundary so the bank's compacted-through T
        tracks the checkpoint. Markers at or before ``iteration`` and their groups
        are reclaimed. Each surviving marker newer than the checkpoint is copied
        once so delayed asynchronous checkpoint finalization remains correct. The
        writer remains closed until the next ``set_collection()`` call selects the
        current rollout collection.
        """
        survivors, markers = self._restore_state(iteration)
        survivor_uids = {group.uid for group in survivors}
        retained_markers = {
            uid: marker_iter
            for uid, marker_iter in markers.items()
            if uid in survivor_uids and marker_iter > iteration
        }
        manifest = self._read_manifest()
        self._publish_generation(
            iteration, survivors, retained_markers, timeline=manifest["timeline"]
        )

    def recover(self, trained_through: int) -> list["RolloutGroup"]:
        """Rebase survivors at ``T`` into a new timeline with no old markers."""
        survivors, _ = self._restore_state(trained_through)
        self._publish_generation(trained_through, survivors, {}, timeline=uuid.uuid4().hex)
        return survivors

    def _publish_generation(
        self,
        iteration: int,
        groups: list["RolloutGroup"],
        markers: dict[str, int],
        *,
        timeline: str,
    ) -> None:
        """Write and atomically select a complete ledger/marker generation."""
        self._close_handles()
        old_manifest = self._read_manifest()
        old_generation = old_manifest["active_generation"]
        generation = _generation_name()
        staging = os.path.join(self._generations_dir, f".{generation}.tmp")
        final_dir = os.path.join(self._generations_dir, generation)
        os.makedirs(staging)
        _fsync_directory(self._generations_dir)

        new_seg = _segment_name(iteration)
        new_seg_dir = os.path.join(staging, new_seg)
        os.makedirs(new_seg_dir)
        _fsync_directory(staging)
        self._rewrite_segment(new_seg_dir, iteration, groups)
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
                "compacted_at": iteration,
                "rollouts_per_group": self.rollouts_per_group,
            }
        )

        # The manifest now references only the compacted survivor segment. Rebase
        # the live payload accounting instead of adding the rewritten survivors
        # on top of the segments they replaced.
        self._bytes_written = self._manifest_sidecar_bytes()
        # Compaction establishes a new live bank footprint. Allow a future cap
        # crossing to warn again, and warn immediately if the compacted
        # survivors themselves still exceed the cap.
        self._warned_over_cap = False
        self._maybe_warn_over_cap()

        # The old generation remains a valid crash-recovery target until the
        # manifest flip is durable. It is safe to reclaim only afterward.
        _rmtree(os.path.join(self._generations_dir, old_generation))
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

    def _rewrite_segment(self, seg_dir: str, iteration: int, groups: list["RolloutGroup"]) -> None:
        """Write ``groups`` into ``seg_dir`` as a fresh ledger + sidecars.

        Group uids and member slots are preserved, so a group that survives
        several compactions keeps the identity its records were written under.
        """
        pending: list = []
        for group in groups:
            uid = group.uid or self.reserve_group_uid()
            if group.problem_state is not None:
                pending.append(_PendingProblem(uid, group.problem_state))
            for rollout_idx, rollout in zip(group.member_indices, group.rollouts, strict=True):
                pending.append(_PendingRollout(uid, rollout_idx, rollout))

        live_bytes = self._bytes_written
        warned_over_cap = self._warned_over_cap
        saved = (
            self._seg_dir,
            self._collection_iter,
            self._tok_off,
            self._lp_off,
            self._mask_off,
            self._ledger_f,
            self._tok_f,
            self._lp_f,
            self._mask_f,
        )
        self._seg_dir, self._collection_iter = seg_dir, iteration
        self._tok_off = self._lp_off = self._mask_off = 0
        self._ledger_f = self._tok_f = self._lp_f = self._mask_f = None
        self._bytes_written = 0
        self._warned_over_cap = True  # staging data is not live yet
        try:
            if pending:
                self.write_records(pending)
        finally:
            self._close_handles()
            self._bytes_written = live_bytes
            self._warned_over_cap = warned_over_cap
            (
                self._seg_dir,
                self._collection_iter,
                self._tok_off,
                self._lp_off,
                self._mask_off,
                self._ledger_f,
                self._tok_f,
                self._lp_f,
                self._mask_f,
            ) = saved

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


def _is_zero_variance(rollouts: list) -> bool:
    """Whether every member scored the same, leaving no learning signal.

    Mirrors the same-reward half of ``RolloutPipeline._decide_drop``. Re-deriving
    this from the persisted rewards, rather than persisting a tombstone when the
    pipeline filters a group, removes the crash window between a group's last
    member write and its marker.

    ``_decide_drop`` also drops all-placeholder groups so they can be refilled.
    That half needs no mirror: ``stage_assemble`` never persists a placeholder, so
    a restored group cannot contain one.

    A ``TokenRollout`` reward may be per-turn, so each is collapsed to its
    trajectory total before comparison.
    """
    rewards = [rollout.reward for rollout in rollouts]
    if any(reward is None for reward in rewards):
        return False
    scalars = [float(np.sum(reward)) for reward in rewards]
    return bool(np.std(scalars) <= _ZERO_VARIANCE_TOLERANCE)


def _rmtree(path: str) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)
