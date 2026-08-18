# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Byte-level accounting for inference client/coordinator ZMQ traffic.

Every request and reply on the frontend path (HTTP frontend -> InferenceClient ->
coordinator -> engine) travels as a msgpack-packed list over ZMQ. None of those
sites recorded how large the messages were, so payload growth -- a field added to
a reply, prompt log probs left on by default -- only surfaced later as a
throughput regression with no obvious cause.

:class:`WireMetrics` accumulates message counts and byte totals per header at the
send and receive sites. Recording costs one dict lookup and a few integer adds
per message, which is negligible beside the msgpack pack/unpack it accompanies,
so it is always on rather than gated behind a flag.
"""

from .headers import UnknownHeaderError, decode_header

# Indices into the per-header accumulator list.
_SENT_COUNT = 0
_SENT_BYTES = 1
_RECEIVED_COUNT = 2
_RECEIVED_BYTES = 3


def _header_name(header):
    """Best-effort human-readable name for a header wire value.

    Never raises: an unrecognized value is reported rather than masked, because
    this runs on the message path and must not turn a bookkeeping miss into a
    request failure.
    """
    try:
        return decode_header(int(header)).name
    except (UnknownHeaderError, TypeError, ValueError):
        return f"UNKNOWN_{header}"


class WireMetrics:
    """Counts messages and bytes crossing a ZMQ socket, broken down by header.

    Sizes are of the msgpack-serialized payload frame, which is what the socket
    actually moves. ZMQ's own framing and the routing-identity frame are not
    counted, so totals are payload bytes rather than bytes on the NIC.
    """

    __slots__ = ("sent_messages", "sent_bytes", "received_messages", "received_bytes", "per_header")

    def __init__(self):
        self.reset()

    def reset(self):
        """Zero every counter."""
        self.sent_messages = 0
        self.sent_bytes = 0
        self.received_messages = 0
        self.received_bytes = 0
        # header wire value -> [sent_count, sent_bytes, received_count, received_bytes]
        self.per_header = {}

    def _bucket(self, header):
        bucket = self.per_header.get(header)
        if bucket is None:
            bucket = [0, 0, 0, 0]
            self.per_header[header] = bucket
        return bucket

    def record_sent(self, header, num_bytes):
        """Record a payload of ``num_bytes`` sent with ``header``."""
        self.sent_messages += 1
        self.sent_bytes += num_bytes
        bucket = self._bucket(header)
        bucket[_SENT_COUNT] += 1
        bucket[_SENT_BYTES] += num_bytes

    def record_received(self, header, num_bytes):
        """Record a payload of ``num_bytes`` received with ``header``."""
        self.received_messages += 1
        self.received_bytes += num_bytes
        bucket = self._bucket(header)
        bucket[_RECEIVED_COUNT] += 1
        bucket[_RECEIVED_BYTES] += num_bytes

    def mean_sent_bytes(self, header=None):
        """Mean sent payload size, overall or for one header. 0.0 if nothing sent."""
        if header is None:
            count, total = self.sent_messages, self.sent_bytes
        else:
            bucket = self.per_header.get(header)
            if bucket is None:
                return 0.0
            count, total = bucket[_SENT_COUNT], bucket[_SENT_BYTES]
        return total / count if count else 0.0

    def mean_received_bytes(self, header=None):
        """Mean received payload size, overall or for one header. 0.0 if nothing received."""
        if header is None:
            count, total = self.received_messages, self.received_bytes
        else:
            bucket = self.per_header.get(header)
            if bucket is None:
                return 0.0
            count, total = bucket[_RECEIVED_COUNT], bucket[_RECEIVED_BYTES]
        return total / count if count else 0.0

    def snapshot(self):
        """Return the current counters as a plain, JSON-serializable dict.

        Header keys are resolved to names here rather than at record time so the
        message path stays free of enum lookups.
        """
        return {
            "sent_messages": self.sent_messages,
            "sent_bytes": self.sent_bytes,
            "received_messages": self.received_messages,
            "received_bytes": self.received_bytes,
            "mean_sent_bytes": self.mean_sent_bytes(),
            "mean_received_bytes": self.mean_received_bytes(),
            "per_header": {
                _header_name(header): {
                    "sent_messages": bucket[_SENT_COUNT],
                    "sent_bytes": bucket[_SENT_BYTES],
                    "mean_sent_bytes": (
                        bucket[_SENT_BYTES] / bucket[_SENT_COUNT] if bucket[_SENT_COUNT] else 0.0
                    ),
                    "received_messages": bucket[_RECEIVED_COUNT],
                    "received_bytes": bucket[_RECEIVED_BYTES],
                    "mean_received_bytes": (
                        bucket[_RECEIVED_BYTES] / bucket[_RECEIVED_COUNT]
                        if bucket[_RECEIVED_COUNT]
                        else 0.0
                    ),
                }
                for header, bucket in self.per_header.items()
            },
        }

    def format_summary(self):
        """Return a multi-line, human-readable summary for logging."""
        lines = [
            f"sent {self.sent_messages} msgs / {self.sent_bytes} bytes "
            f"(mean {self.mean_sent_bytes():.1f} B), "
            f"received {self.received_messages} msgs / {self.received_bytes} bytes "
            f"(mean {self.mean_received_bytes():.1f} B)"
        ]
        for header, bucket in sorted(self.per_header.items()):
            name = _header_name(header)
            if bucket[_SENT_COUNT]:
                lines.append(
                    f"  {name:24s} sent     {bucket[_SENT_COUNT]:8d} msgs  "
                    f"{bucket[_SENT_BYTES]:12d} B  "
                    f"mean {bucket[_SENT_BYTES] / bucket[_SENT_COUNT]:9.1f} B"
                )
            if bucket[_RECEIVED_COUNT]:
                lines.append(
                    f"  {name:24s} received {bucket[_RECEIVED_COUNT]:8d} msgs  "
                    f"{bucket[_RECEIVED_BYTES]:12d} B  "
                    f"mean {bucket[_RECEIVED_BYTES] / bucket[_RECEIVED_COUNT]:9.1f} B"
                )
        return "\n".join(lines)
