# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Wire schema for messages exchanged with the data parallel inference coordinator.

Every message is a list of ZMQ frames: one **metadata** frame followed by zero or
more **payload** frames.

The metadata frame is the only frame the coordinator decodes, so it holds just
what routing needs and nothing whose size follows the sequence length. Payload
frames carry the bulk -- prompts inbound, finished requests outbound -- and are
forwarded as opaque bytes, which keeps the coordinator's per-request cost flat in
prompt length. Payload boundaries come from ZMQ multipart, so splitting a batch
costs the coordinator nothing.

Each message type is declared once, as a :class:`MessageSpec` naming its metadata
fields and payload frames in wire order. ``pack`` and ``parse`` are both derived
from that declaration, so no frame index is written by hand and the two
directions cannot drift apart. Adding a field means editing one tuple.

Which tuple a name goes in is the whole contract: ``metadata_fields`` are decoded
by the coordinator and so must be constant-size, while ``payload_frames`` are
never decoded in transit and may grow with the prompt.
"""

from collections import namedtuple
from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

from megatron.core.inference.headers import Headers

try:
    import msgpack
except ImportError:
    msgpack = None


def header_of(metadata: Sequence) -> Headers:
    """Return the header of a decoded metadata frame."""
    return Headers(metadata[0])


@dataclass(frozen=True)
class MessageSpec:
    """Declares one message's frame layout.

    Attributes:
        name: Name of the named tuple returned by :meth:`parse`.
        header: The header this layout belongs to.
        metadata_fields: Names of the values following the header in the metadata
            frame, in wire order. Decoded by the coordinator, so every one of
            these must be constant-size.
        payload_frames: Names of the opaque frames following the metadata frame,
            in wire order. Never decoded in transit; these may grow with the
            prompt.
    """

    name: str
    header: Headers
    metadata_fields: Tuple[str, ...] = ()
    payload_frames: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_tuple", namedtuple(self.name, self.metadata_fields + self.payload_frames)
        )

    def pack(self, **values: Any) -> List[bytes]:
        """Build the frames for this message from named values."""
        metadata = msgpack.packb(
            [self.header.value, *(values[name] for name in self.metadata_fields)], use_bin_type=True
        )
        return [metadata, *(values[name] for name in self.payload_frames)]

    def parse(self, metadata: Sequence, bodies: Sequence[bytes]) -> Any:
        """Read a received message into a named tuple."""
        return self._tuple(*metadata[1:], *bodies[: len(self.payload_frames)])


@dataclass(frozen=True)
class BatchedMessageSpec:
    """Declares a message carrying N independent entries, one payload frame each.

    Engine replies are batched per engine step but fan out to different clients,
    so each entry keeps its own frame: that is what lets the coordinator route the
    batch without decoding any of it. The metadata frame holds one routing record
    per entry, in the same order as the frames.

    Attributes:
        name: Name of the named tuple returned per entry by :meth:`parse`.
        header: The header this layout belongs to.
        entry_fields: Names of the per-entry routing values held in the metadata
            frame. A single field is packed as a bare scalar per entry; several
            are packed as a list per entry.
        payload_frame: Name of the opaque frame carried per entry.
    """

    name: str
    header: Headers
    entry_fields: Tuple[str, ...]
    payload_frame: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_tuple", namedtuple(self.name, self.entry_fields + (self.payload_frame,))
        )

    def pack(self, entries: Sequence[Any], payloads: Sequence[bytes]) -> List[bytes]:
        """Build the frames for a batch.

        Args:
            entries: One routing record per entry -- a scalar when there is a
                single entry field, otherwise a sequence of that field's values.
            payloads: One opaque frame per entry, in the same order.
        """
        assert len(entries) == len(payloads), (
            f"{self.name}: {len(entries)} entries but {len(payloads)} payload frames; "
            "the metadata frame must describe exactly the frames that follow"
        )
        if len(self.entry_fields) == 1:
            records = list(entries)
        else:
            records = [list(entry) for entry in entries]
        return [msgpack.packb([self.header.value, records], use_bin_type=True), *payloads]

    def parse(self, metadata: Sequence, bodies: Sequence[bytes]) -> List[Any]:
        """Read a batch, pairing each routing record with its frame."""
        records = metadata[1]
        if len(self.entry_fields) == 1:
            return [self._tuple(record, body) for record, body in zip(records, bodies)]
        return [self._tuple(*record, body) for record, body in zip(records, bodies)]


# --- client -> coordinator -> engine ------------------------------------------------

SUBMIT_REQUEST = MessageSpec(
    "SubmitRequest",
    Headers.SUBMIT_REQUEST,
    metadata_fields=("request_id", "sampling_params"),
    payload_frames=("prompt",),
)
"""A plain inference request."""

SUBMIT_REQUEST_WITH_KV = MessageSpec(
    "SubmitRequestWithKV",
    Headers.SUBMIT_REQUEST_WITH_KV,
    # kv_meta is the peer's transfer metadata, bounded by TP size and
    # num_speculative_tokens. src_block_ids names one block per block_size_tokens
    # of prompt, so it grows with the prompt and belongs in a frame.
    metadata_fields=("request_id", "sampling_params", "kv_meta"),
    payload_frames=("prompt", "src_block_ids"),
)
"""A decode request whose KV was computed by a prefill peer."""


# --- engine -> coordinator ----------------------------------------------------------

ENGINE_REPLY = BatchedMessageSpec(
    "FinishedRequest",
    Headers.ENGINE_REPLY,
    # needs_detokenize rides in the metadata so the coordinator can skip decoding
    # the reply entirely for clients that detokenize for themselves.
    entry_fields=("request_id", "needs_detokenize"),
    payload_frame="reply",
)
"""A batch of finished requests, one payload frame each."""

ENGINE_REPLY_PARTIAL = BatchedMessageSpec(
    "PartialReply",
    Headers.ENGINE_REPLY_PARTIAL,
    entry_fields=("request_id",),
    payload_frame="partial",
)
"""A batch of incremental replies, one payload frame each."""


# --- coordinator -> client ----------------------------------------------------------

CLIENT_REPLY = MessageSpec(
    "ClientReply", Headers.ENGINE_REPLY, metadata_fields=("request_id",), payload_frames=("reply",)
)
"""One final reply, split out of an engine batch and re-addressed to its client."""

CLIENT_REPLY_PARTIAL = MessageSpec(
    "ClientPartialReply",
    Headers.ENGINE_REPLY_PARTIAL,
    metadata_fields=("request_id",),
    payload_frames=("partial",),
)
"""One incremental reply, re-addressed to its client."""


# --- control ------------------------------------------------------------------------

KV_HANDOFF_COMPLETE = MessageSpec(
    "KVHandoffComplete", Headers.KV_HANDOFF_COMPLETE, metadata_fields=("request_id", "failed")
)
"""A model-parallel-agreed handoff outcome, distributed over the schedule broadcast."""

SEND_KV = MessageSpec("SendKV", Headers.SEND_KV, metadata_fields=("request_id", "decode_metas"))
"""An instruction to push a pinned handoff's KV to a decode instance."""

SET_GENERATION_EPOCH = MessageSpec(
    "SetGenerationEpoch", Headers.SET_GENERATION_EPOCH, metadata_fields=("generation_epoch",)
)
"""The only control signal carrying an argument."""


def pack_signal(header: Headers, *args: Any) -> List[bytes]:
    """Frame a control signal as a single metadata frame with no payload.

    Covers the messages whose whole content is a header plus at most a couple of
    scalars: the handshake (CONNECT, CONNECT_ACK, DISCONNECT), lifecycle control
    (PAUSE, UNPAUSE, SUSPEND, RESUME, STOP, SHUTDOWN), the profiler signals, and
    the request-scoped controls ABORT_REQUEST and RELEASE_KV.

    The coordinator rebroadcasts decoded metadata verbatim, so args survive the
    hop to the engines.
    """
    return [msgpack.packb([header.value, *args], use_bin_type=True)]


def request_id_of(metadata: Sequence) -> int:
    """Return the request id of a control signal that names exactly one request."""
    return int(metadata[1])
