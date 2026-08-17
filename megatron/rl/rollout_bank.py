# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Durable rollout bank — the queued-rollout (completed-group) path.

Persists completed ``RolloutGroup``s the instant they assemble, so a SIGKILL (the
4h SLURM limit) does not destroy work that is already on disk. This is PR #1 of
the durable rollout bank: it covers only the *queued* path — groups sitting in
the pipeline's ``output_queue`` (assembled, not yet trained-in) plus groups
consumed at a training step that a restart rolls back. In-flight decode state and
partial-group snapshots (design phases B/C) are out of scope here.

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

Recovery guarantees: append is write-through with ``fsync`` per group, so a kill can
damage only the final record, which its checksum drops on read. A checkpoint writes
a complete new generation, including a compacted marker log, before atomically
flipping the manifest. ``restore`` replays the active generation and returns every
group that is not already trained through checkpoint step ``T``:

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
_FORMAT_VERSION = 2
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
    One self-describing JSONL index record for a completed group.

    Args:
        format_version: The persisted schema and sidecar-layout version.
        uid: The unique identifier of the group.
        collection_iter: The iteration number of the collection.
        member_type: The type of the member. This is the type of the rollouts in the group.
        kind: How the group is stored. "inline" if the group is not token-typed, "token" if the group is token-typed.
        group: The group of rollouts.
        tok: The token meta. This is only present if the group is token-typed.
        lp: The logprob meta. This is only present if the group is token-typed and logprobs are present.
        mask: The mask meta. This is only present if the group is token-typed and generation_mask is present.
        checksum: The checksum of the group.
    """

    format_version: int
    uid: str
    collection_iter: int
    member_type: Literal["Rollout", "TokenRollout"]
    kind: Literal["inline", "token"]
    group: dict
    tok: NotRequired[SidecarMeta]
    lp: NotRequired[SidecarMeta]
    mask: NotRequired[SidecarMeta]
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
    """

    format_version: int
    active_generation: str
    timeline: str
    trained_through: int
    segments: list[str]
    compacted_at: int


class ConsumedMarker(TypedDict):
    """Append-only marker: a group uid was pulled by the trainer at ``iter``."""

    uid: str
    iter: int


class EncodedGroup(NamedTuple):
    """Ledger record and packed sidecar payloads for one rollout group."""

    record: LedgerRecord
    tok_bytes: bytes
    lp_bytes: bytes
    mask_bytes: bytes


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
    """Single-writer durable store for completed rollout groups (rank-0 only)."""

    def __init__(self, bank_dir: str, *, max_bytes: int = 0) -> None:
        self.bank_dir = bank_dir
        self.max_bytes = max_bytes
        os.makedirs(bank_dir, exist_ok=True)
        self._lock = threading.RLock()
        self._collection_iter: Optional[int] = None
        self._seg_dir: Optional[str] = None
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
        self._last_checkpoint_iter = 0
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
                }
            )
        manifest = self._read_manifest()
        self._garbage_collect_generations(manifest)
        # Read through the published manifest immediately, both to initialize
        # live-size accounting and to fail closed on malformed existing state.
        self._bytes_written = self._manifest_sidecar_bytes(manifest)
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

    def _next_sequence(self, seg_dir: str, seg: str) -> int:
        """Return the next unused sequence number in ``seg``."""
        next_seq = 0
        for record in self._read_ledger(seg_dir):
            uid_seg, separator, uid_seq = record.get("uid", "").partition("/")
            if separator and uid_seg == seg and uid_seq.isdigit():
                next_seq = max(next_seq, int(uid_seq) + 1)
        return next_seq

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
        with self._lock:
            self._close_handles()

    # ---------------------------------------------------------------- append
    def append(self, group: "RolloutGroup", uid: str | None = None) -> str:
        """Write-through one completed group; return its stable uid.

        A uid is assigned for a new group or preserved when rewriting an
        existing group during compaction.
        """
        with self._lock:
            assert self._seg_dir is not None, "set_collection() must be called before append()"
            current_seg = _segment_name(self._collection_iter)
            if uid is None:
                uid = f"{current_seg}/{self._seq}"
                self._seq += 1
            else:
                uid_seg, separator, uid_seq = uid.partition("/")
                if separator and uid_seg == current_seg and uid_seq.isdigit():
                    self._seq = max(self._seq, int(uid_seq) + 1)

            record, tok_bytes, lp_bytes, mask_bytes = self._encode(group, uid)
            # Write sidecar slices first, then the index record that points at them,
            # so a torn write can only ever lose the trailing index line.
            if tok_bytes:
                self._write_sidecar("_tok_f", _TOKENS_BIN, tok_bytes)
            if lp_bytes:
                self._write_sidecar("_lp_f", _LOGPROBS_BIN, lp_bytes)
            if mask_bytes:
                self._write_sidecar("_mask_f", _MASKS_BIN, mask_bytes)
            record["checksum"] = _checksum(
                {k: v for k, v in record.items() if k != "checksum"},
                tok_bytes,
                lp_bytes,
                mask_bytes,
            )
            self._append_ledger(record)

            self._bytes_written += len(tok_bytes) + len(lp_bytes) + len(mask_bytes)
            self._maybe_warn_over_cap()
            return uid

    def _write_sidecar(self, handle_attr: str, name: str, data: bytes) -> None:
        f = getattr(self, handle_attr)
        created = False
        if f is None:
            path = os.path.join(self._seg_dir, name)
            created = not os.path.exists(path)
            f = open(path, "ab")
            setattr(self, handle_attr, f)
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
        if created:
            _fsync_directory(self._seg_dir)

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

    def _encode(self, group: "RolloutGroup", uid: str) -> EncodedGroup:
        """Build the JSONL index record + packed sidecar bytes for one group."""
        assert self._collection_iter is not None, "set_collection() must be called before _encode()"
        dumped = group.model_dump()
        dumped.pop("uid", None)  # uid lives at the top level of the record
        members = dumped.get("rollouts", [])
        token_members = [isinstance(member, TokenRollout) for member in group.rollouts]
        if token_members and any(token_members) and not all(token_members):
            raise ValueError("RolloutGroup must not mix TokenRollout and Rollout members")
        token_typed = bool(token_members) and all(token_members)
        member_type: Literal["Rollout", "TokenRollout"] = (
            "TokenRollout" if token_typed else "Rollout"
        )

        if not token_typed:
            record: LedgerRecord = {
                "format_version": _FORMAT_VERSION,
                "uid": uid,
                "collection_iter": self._collection_iter,
                "member_type": member_type,
                "kind": "inline",
                "group": dumped,
            }
            return EncodedGroup(record, b"", b"", b"")

        tok_flat, tok_lengths = [], []
        lp_flat, lp_lengths = [], []
        mask_flat, mask_lengths = [], []
        for m in members:
            traj = m.pop("trajectory")
            tok_lengths.append([len(turn) for turn in traj])
            for turn in traj:
                tok_flat.extend(turn)

            lp = m.pop("logprobs", None)
            if lp is not None:
                lp_lengths.append([len(turn) for turn in lp])
                for turn in lp:
                    lp_flat.extend(turn)

            mask = m.pop("generation_mask", None)
            if mask is not None:
                mask_lengths.append([len(turn) for turn in mask])
                for turn in mask:
                    mask_flat.extend(turn)

        for field, lengths in (("logprobs", lp_lengths), ("generation_mask", mask_lengths)):
            if lengths and len(lengths) != len(members):
                raise ValueError(f"{field} must be present for all or no rollouts in a group")

        tok_bytes = np.asarray(tok_flat, dtype=_TOKEN_DTYPE).tobytes()
        record: LedgerRecord = {
            "format_version": _FORMAT_VERSION,
            "uid": uid,
            "collection_iter": self._collection_iter,
            "member_type": member_type,
            "kind": "token",
            "group": dumped,  # dumped members are now array-stripped
            "tok": {"offset": self._tok_off, "bytes": len(tok_bytes), "lengths": tok_lengths},
        }

        lp_bytes = b""
        if lp_lengths:
            lp_bytes = np.asarray(lp_flat, dtype=_LOGPROB_DTYPE).tobytes()
            record["lp"] = {"offset": self._lp_off, "bytes": len(lp_bytes), "lengths": lp_lengths}

        mask_bytes = b""
        if mask_lengths:
            mask_arr = np.asarray(mask_flat, dtype=bool).astype(_MASK_DTYPE)
            mask_bytes = mask_arr.tobytes()
            record["mask"] = {
                "offset": self._mask_off,
                "bytes": len(mask_bytes),
                "lengths": mask_lengths,
            }

        # Advance running offsets now that this record's slices are placed.
        self._tok_off += len(tok_bytes)
        self._lp_off += len(lp_bytes)
        self._mask_off += len(mask_bytes)

        return EncodedGroup(record, tok_bytes, lp_bytes, mask_bytes)

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
        The batch is therefore durable before this method returns without paying
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
        with self._lock:
            restored, _ = self._restore_state(trained_through)
            return restored

    def _restore_state(self, trained_through: int) -> tuple[list["RolloutGroup"], dict[str, int]]:
        """Return survivors and the active generation's collapsed markers."""
        self._close_handles()  # flush any active writer before reading
        manifest = self._read_manifest()
        generation_dir = self._generation_dir(manifest)
        restored: list["RolloutGroup"] = []
        markers = self._read_markers(generation_dir)
        for seg in manifest["segments"]:
            seg_dir = os.path.join(generation_dir, seg)
            for uid, iteration in self._read_markers(seg_dir).items():
                markers[uid] = max(markers.get(uid, iteration), iteration)
            for record in self._read_ledger(seg_dir):
                uid = record["uid"]
                marker_iter = markers.get(uid)
                if marker_iter is not None and marker_iter <= trained_through:
                    continue  # trained into the loaded weights; drop
                group = self._decode(record, seg_dir)
                if group is not None:
                    restored.append(group)
        return restored, markers

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

    def _decode(self, record: LedgerRecord, seg_dir: str) -> Optional["RolloutGroup"]:
        """
        Decode a ledger record into a RolloutGroup.
        Args:
            record: The ledger record to decode.
            seg_dir: The directory of the segment.

        Returns:
            A RolloutGroup or None if the record is corrupted or truncated.
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

        group_dict = record["group"]
        uid = record["uid"]
        if record["kind"] == "inline":
            cls = TokenRollout if record["member_type"] == "TokenRollout" else Rollout
            members = [cls.model_validate(m) for m in group_dict["rollouts"]]
            group = RolloutGroup(
                rollouts=members,
                batch_id=group_dict.get("batch_id", 0),
                index_in_batch=group_dict.get("index_in_batch", 0),
            )
            group.uid = uid
            return group

        # token kind: re-split flat sidecar arrays back into jagged per-turn lists
        tok = np.frombuffer(tok_bytes, dtype=_TOKEN_DTYPE)
        traj_per_member = self._unflatten(tok.tolist(), record["tok"]["lengths"])
        lp_per_member = None
        if "lp" in record:
            lp = np.frombuffer(lp_bytes, dtype=_LOGPROB_DTYPE).astype(np.float32)
            lp_per_member = self._unflatten(lp.tolist(), record["lp"]["lengths"])
        mask_per_member = None
        if "mask" in record:
            mask = np.frombuffer(mask_bytes, dtype=_MASK_DTYPE).astype(bool)
            mask_per_member = self._unflatten(mask.tolist(), record["mask"]["lengths"])

        members = []
        for i, m in enumerate(group_dict["rollouts"]):
            m = dict(m)
            m["trajectory"] = traj_per_member[i]
            m["logprobs"] = lp_per_member[i] if lp_per_member is not None else None
            m["generation_mask"] = mask_per_member[i] if mask_per_member is not None else None
            # Completion IDs point into the inference engine's process-local
            # metadata ledger. That ledger does not survive a crash, so recovered
            # rollouts must not attempt to join against stale IDs.
            m["completion_ids"] = []
            members.append(TokenRollout.model_validate(m))
        group = RolloutGroup(
            rollouts=members,
            batch_id=group_dict.get("batch_id", 0),
            index_in_batch=group_dict.get("index_in_batch", 0),
        )
        group.uid = uid
        return group

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
        """Split a flat list back into [member][turn] jagged nesting."""
        out, pos = [], 0
        for member_lengths in lengths:
            turns = []
            for n in member_lengths:
                turns.append(flat[pos : pos + n])
                pos += n
            out.append(turns)
        return out

    def checkpoint(self, iteration: int) -> None:
        """Compact survivors and future markers into an atomic new generation.

        Piggybacks the model-checkpoint boundary so the bank's compacted-through T
        tracks the checkpoint. Markers at or before ``iteration`` and their groups
        are reclaimed. Each surviving marker newer than the checkpoint is copied
        once so delayed asynchronous checkpoint finalization remains correct.
        """
        with self._lock:
            active_collection_iter = self._collection_iter
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
            # A delayed async-checkpoint callback can run several collection
            # iterations after ``iteration``. Keep subsequent write-through
            # appends on the collection that was active before compaction.
            if active_collection_iter is not None:
                self.set_collection(active_collection_iter)
            self._last_checkpoint_iter = iteration

    def recover(self, trained_through: int) -> list["RolloutGroup"]:
        """Rebase survivors at ``T`` into a new timeline with no old markers."""
        with self._lock:
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
        """Write ``groups`` into ``seg_dir`` as a fresh ledger + sidecars."""
        live_bytes = self._bytes_written
        warned_over_cap = self._warned_over_cap
        saved = (
            self._seg_dir,
            self._collection_iter,
            self._seq,
            self._tok_off,
            self._lp_off,
            self._mask_off,
            self._ledger_f,
            self._tok_f,
            self._lp_f,
            self._mask_f,
        )
        self._seg_dir, self._collection_iter, self._seq = seg_dir, iteration, 0
        self._tok_off = self._lp_off = self._mask_off = 0
        self._ledger_f = self._tok_f = self._lp_f = self._mask_f = None
        self._bytes_written = 0
        self._warned_over_cap = True  # staging data is not live yet
        try:
            for group in groups:
                self.append(group, uid=group.uid)  # reuses the write-through encoder + fsync
        finally:
            self._close_handles()
            self._bytes_written = live_bytes
            self._warned_over_cap = warned_over_cap
            (
                self._seg_dir,
                self._collection_iter,
                self._seq,
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


def _rmtree(path: str) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)
