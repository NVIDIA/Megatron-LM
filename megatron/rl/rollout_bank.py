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
        MANIFEST.json                 # {"format_version", "trained_through",
                                      #  "segments", "compacted_at"}
        consumed.log                  # append-only markers {"uid", "iter"} on trainer pull
        gen-<iter>/
            ledger.log                # append-only JSONL, one self-describing index
                                      # record per completed group (+ per-record checksum)
            tokens.bin                # int32 token ids (offset-indexed sidecar)
            logprobs.bin              # fp16 generation logprobs (offset-indexed sidecar)
            masks.bin                 # uint8 generation masks (offset-indexed sidecar)

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
damage only the final record, which its checksum drops on read. ``restore`` replays
the ledger and, using the consumption markers, restores every group that is not
already trained through the resumed checkpoint step ``T``:

    marker <= T -> discard (training already in the loaded weights)
    marker  > T -> restore (that training step was erased by the kill)
    no marker   -> restore (never consumed, or lost in the kill)
"""

import hashlib
import json
import logging
import os
from typing import Iterator, Literal, NamedTuple, NotRequired, Optional, TypedDict

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
_FORMAT_VERSION = 1
_MANIFEST = "MANIFEST.json"
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
        trained_through: The iteration number of the last checkpoint.
        segments: The list of segment names (e.g. ["gen-000000", "gen-000001"]).
        compacted_at: The iteration number of the last compaction.
    """

    format_version: int
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
            self._write_manifest_atomic(
                {
                    "format_version": _FORMAT_VERSION,
                    "trained_through": 0,
                    "segments": [],
                    "compacted_at": 0,
                }
            )

    # ------------------------------------------------------------------ paths
    @property
    def _manifest_path(self) -> str:
        return os.path.join(self.bank_dir, _MANIFEST)

    @property
    def _consumed_path(self) -> str:
        return os.path.join(self.bank_dir, _CONSUMED)

    def _read_manifest(self) -> Manifest:
        try:
            with open(self._manifest_path) as f:
                manifest = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logger.warning(f"Manifest file not found at {self._manifest_path}; creating new one.")
            manifest = {
                "format_version": _FORMAT_VERSION,
                "trained_through": 0,
                "segments": [],
                "compacted_at": 0,
            }
        _validate_format_version(manifest, self._manifest_path)
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
        if self._collection_iter == iteration and self._seg_dir is not None:
            return
        self._close_handles()
        self._collection_iter = iteration
        seg = _segment_name(iteration)
        self._seg_dir = os.path.join(self.bank_dir, seg)
        segment_created = not os.path.exists(self._seg_dir)
        os.makedirs(self._seg_dir, exist_ok=True)
        if segment_created:
            _fsync_directory(self.bank_dir)
        self._truncate_torn_ledger_tail(self._seg_dir)
        self._seq = self._next_sequence(self._seg_dir, seg)
        self._tok_off = self._file_size(_TOKENS_BIN)
        self._lp_off = self._file_size(_LOGPROBS_BIN)
        self._mask_off = self._file_size(_MASKS_BIN)
        manifest = self._read_manifest()
        if seg not in manifest["segments"]:
            manifest["segments"].append(seg)
            self._write_manifest_atomic(manifest)

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

    def _close_handles(self) -> None:
        for f in (self._ledger_f, self._tok_f, self._lp_f, self._mask_f):
            if f is not None:
                f.close()
        self._ledger_f = self._tok_f = self._lp_f = self._mask_f = None

    def close(self) -> None:
        self._close_handles()

    # ---------------------------------------------------------------- append
    def append(self, group: "RolloutGroup", uid: str | None = None) -> str:
        """Write-through one completed group; return its stable uid.

        A uid is assigned for a new group or preserved when rewriting an
        existing group during compaction.
        """
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
            {k: v for k, v in record.items() if k != "checksum"}, tok_bytes, lp_bytes, mask_bytes
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
        """Build the JSONL index record + packed sidecar bytes for one group.
        """
        assert self._collection_iter is not None, "set_collection() must be called before _encode()"
        dumped = group.model_dump()
        dumped.pop("uid", None)  # uid lives at the top level of the record
        members = dumped.get("rollouts", [])
        member_type = type(group.rollouts[0]).__name__ if group.rollouts else "Rollout"

        # if the member type is TokenRollout and all the rollouts are token-typed, then set token_typed to True
        token_typed = member_type == "TokenRollout" and all(
            "trajectory" in m and m["trajectory"] and isinstance(m["trajectory"][0], list)
            for m in members
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
            record["mask"] = {"offset": self._mask_off, "bytes": len(mask_bytes), "lengths": mask_lengths}

        # Advance running offsets now that this record's slices are placed.
        self._tok_off += len(tok_bytes)
        self._lp_off += len(lp_bytes)
        self._mask_off += len(mask_bytes)

        return EncodedGroup(record, tok_bytes, lp_bytes, mask_bytes)

    def mark_consumed(self, uid: str, iteration: int) -> None:
        """Record that ``uid`` was pulled by the trainer at ``iteration``.

        Markers are append-only and never deleted (a delete could not be undone;
        a marker can be ignored on restore).

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
        if not uid:
            return
        marker: ConsumedMarker = {"uid": uid, "iter": iteration}
        created = not os.path.exists(self._consumed_path)
        with open(self._consumed_path, "a") as f:
            f.write(json.dumps(marker, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())
        if created:
            _fsync_directory(self.bank_dir)

    def restore(self, trained_through: int) -> list["RolloutGroup"]:
        """Replay the ledger and return groups not yet trained through ``T``."""
        self._close_handles()  # flush any active writer before reading
        manifest = self._read_manifest()
        restored: list["RolloutGroup"] = []
        markers = self._read_markers(self.bank_dir)
        for seg in manifest["segments"]:
            seg_dir = os.path.join(self.bank_dir, seg)
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
        return restored

    def _read_markers(self, seg_dir: str) -> dict:
        """
        Load consumption markers from a directory.
        A marker records that the trainer pulled a group (``uid``) at some
        training iteration (``iter``). Markers live as append-only JSONL in
        ``consumed.log`` and are never deleted; this method collapses them to
        ``uid -> latest (max) consumed iteration`` for ``restore``.

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
        if not meta:
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
        """Compact survivors into a fresh segment and atomically flip the manifest.

        Piggybacks the model-checkpoint boundary so the bank's compacted-through T
        tracks the checkpoint. Everything consumed at marker <= ``iteration`` is
        reclaimed; unconsumed survivors carry forward into ``gen-<iteration>/``.
        """
        self._close_handles()
        survivors = self.restore(iteration)  # groups NOT trained through `iteration`
        old_segments = self._read_manifest()["segments"]

        new_seg = _segment_name(iteration)
        new_dir = os.path.join(self.bank_dir, new_seg)
        # If a same-iteration segment already exists (e.g. re-entrant compaction),
        # stage into a temp dir then swap, to keep the survivor rewrite clean.
        staging = new_dir + ".compact"
        if os.path.exists(staging):
            _rmtree(staging)
        os.makedirs(staging, exist_ok=True)
        _fsync_directory(self.bank_dir)
        self._rewrite_segment(staging, iteration, survivors)

        if os.path.exists(new_dir):
            _rmtree(new_dir)
        os.replace(staging, new_dir)
        _fsync_directory(self.bank_dir)
        self._write_manifest_atomic(
            {
                "format_version": _FORMAT_VERSION,
                "trained_through": iteration,
                "segments": [new_seg],
                "compacted_at": iteration,
            }
        )
        for seg in old_segments:
            if seg != new_seg:
                _rmtree(os.path.join(self.bank_dir, seg))

        # Reopen the (now compacted) active segment for continued appends.
        self._collection_iter = None
        self._seg_dir = None
        self.set_collection(iteration)
        self._last_checkpoint_iter = iteration

    def _rewrite_segment(self, seg_dir: str, iteration: int, groups: list["RolloutGroup"]) -> None:
        """Write ``groups`` into ``seg_dir`` as a fresh ledger + sidecars."""
        saved = (self._seg_dir, self._collection_iter, self._seq,
                 self._tok_off, self._lp_off, self._mask_off,
                 self._ledger_f, self._tok_f, self._lp_f, self._mask_f)
        self._seg_dir, self._collection_iter, self._seq = seg_dir, iteration, 0
        self._tok_off = self._lp_off = self._mask_off = 0
        self._ledger_f = self._tok_f = self._lp_f = self._mask_f = None
        try:
            for group in groups:
                self.append(group, uid=group.uid)  # reuses the write-through encoder + fsync
        finally:
            self._close_handles()
            (self._seg_dir, self._collection_iter, self._seq,
             self._tok_off, self._lp_off, self._mask_off,
             self._ledger_f, self._tok_f, self._lp_f, self._mask_f) = saved

    def _maybe_warn_over_cap(self) -> None:
        if self.max_bytes <= 0 or self._bytes_written <= self.max_bytes:
            return
        if not self._warned_over_cap:
            logger.warning(
                "RolloutBank exceeded --rl-rollout-bank-max-bytes (%d > %d); will compact at the "
                "next checkpoint. Consider a shorter checkpoint interval or lower --rl-generation-lag.",
                self._bytes_written, self.max_bytes,
            )
            self._warned_over_cap = True


def _rmtree(path: str) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)
