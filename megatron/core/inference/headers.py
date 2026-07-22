# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Message headers for inference coordinator/engine/client communication.

Headers are grouped into category IntEnum classes by concern. Each category
occupies a distinct numeric range so that wire values never collide across
categories. Adding a new class of headers means adding a new category enum in
its own range and listing it in HEADER_ENUMS; the existing categories are left
untouched.

On the wire a header travels as its integer value. decode_header maps an
integer back to the originating category member.
"""

from enum import IntEnum


class Connection(IntEnum):
    """Client <-> coordinator handshake."""

    CONNECT = 0
    CONNECT_ACK = 1


class Request(IntEnum):
    """Inference request submission and completion reply."""

    SUBMIT_REQUEST = 20
    ENGINE_REPLY = 21


class Control(IntEnum):
    """Runtime control signals broadcast to engines."""

    PAUSE = 40
    UNPAUSE = 41
    SUSPEND = 42
    RESUME = 43
    SET_GENERATION_EPOCH = 44
    STOP = 45
    START_CUDA_PROFILER = 46
    STOP_CUDA_PROFILER = 47


class Lifecycle(IntEnum):
    """Process lifecycle signals."""

    DISCONNECT = 60
    SHUTDOWN = 61


class Transport(IntEnum):
    """Low-level transport framing."""

    TP_BROADCAST = 80


# All header categories. To add a new class of headers, define a new IntEnum in
# its own (disjoint) numeric range and append it here; nothing else needs to
# change. Ranges are spaced out to leave room for growth within each category.
HEADER_ENUMS = (Connection, Request, Control, Lifecycle, Transport)


class UnknownHeaderError(Exception):
    """A signal with an unrecognized header was received."""

    def __init__(self, header):
        super().__init__(f"specialize for {header}.")


def _build_tables():
    """Index every header by wire value and by name, asserting no collisions."""
    by_value = {}
    by_name = {}
    for enum_cls in HEADER_ENUMS:
        for member in enum_cls:
            if member.value in by_value:
                raise ValueError(
                    f"duplicate header wire value {member.value}: "
                    f"{by_value[member.value]!r} and {member!r}"
                )
            if member.name in by_name:
                raise ValueError(
                    f"duplicate header name {member.name!r}: "
                    f"{by_name[member.name]!r} and {member!r}"
                )
            by_value[member.value] = member
            by_name[member.name] = member
    return by_value, by_name


_CODE_TO_MEMBER, _NAME_TO_MEMBER = _build_tables()


def decode_header(value):
    """Resolve an integer wire value to its category header member.

    Args:
        value (int): The integer header value read off the wire.

    Returns:
        The corresponding category enum member (e.g. ``Control.PAUSE``).

    Raises:
        UnknownHeaderError: if no header is registered for ``value``.
    """
    try:
        return _CODE_TO_MEMBER[value]
    except KeyError:
        raise UnknownHeaderError(value)


class _Headers:
    """Flat, read-only union over every header category.

    Lets callers use a single name without caring which category a header lives
    in: attribute access (``Headers.PAUSE``) resolves to the underlying category
    member (``Control.PAUSE``), and calling it (``Headers(value)``) decodes a
    wire value via decode_header. Because both return the canonical category
    members, equality and dict-key lookups against the categories work
    unchanged.
    """

    def __call__(self, value):
        return decode_header(value)

    def __getattr__(self, name):
        try:
            return _NAME_TO_MEMBER[name]
        except KeyError:
            raise AttributeError(f"no header named {name!r}")

    def __iter__(self):
        return iter(_NAME_TO_MEMBER.values())


Headers = _Headers()
