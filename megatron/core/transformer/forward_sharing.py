# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""State shared by cooperating components during one forward pass.

Payloads are keyed by type and define their own tensor layout and per-layer indexing.
Cooperating calls must resolve the same carrier: packed-sequence parameters, then
attention mask, then config. Carrier lifetime does not determine payload lifetime.

The outer forward owns cleanup through ``forward_sharing_lifetime``; nested scopes
join it. Direct module callers must establish that scope or explicitly clear state.
Checkpoint and schedule callers may own a separate state and temporarily bind it
with ``use_forward_sharing_state``.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import copy
from functools import wraps
from typing import Any, Callable, Iterator, TypeVar

_FORWARD_SHARING_STATE_ATTR = "_forward_sharing_state"

T = TypeVar("T")


def is_forward_sharing_enabled(config: object) -> bool:
    """Return whether any configured feature needs forward sharing state.

    Add new sharing features here so lifecycle and checkpoint callers do not need
    component-specific predicates. This is an enablement query, not a recompute
    compatibility check; those constraints remain with the feature configuration.
    """
    return (getattr(config, "dsa_indexer_topk_freq", 1) or 1) > 1 or bool(
        getattr(config, "mtp_repeated_layer_shared_components", None)
    )


class ForwardSharingState:
    """One payload per type, independent of carrier lifetime.

    Payloads manage their own per-layer data. Checkpoint snapshots copy the payload
    objects, not tensor storage; payloads with mutable containers implement ``__copy__``.
    """

    def __init__(self) -> None:
        self._entries: dict[type[Any], Any] = {}
        self.active = False

    def get(self, payload_type: type[T]) -> T | None:
        """Return the payload registered for an exact type."""
        self._validate_payload_type(payload_type)
        return self._entries.get(payload_type)

    def get_or_create(self, payload_type: type[T]) -> T:
        """Return a payload, constructing it without arguments if absent."""
        self._validate_payload_type(payload_type)
        payload = self._entries.get(payload_type)
        if payload is None:
            payload = payload_type()
            self._entries[payload_type] = payload
        return payload

    def clear(self, payload_type: type[Any] | None = None) -> None:
        """Clear one payload type, or all payloads when no type is specified."""
        if payload_type is None:
            self._entries.clear()
        else:
            self._validate_payload_type(payload_type)
            self._entries.pop(payload_type, None)

    def snapshot(self) -> ForwardSharingState:
        """Retain independent payload records for a particular checkpoint invocation."""
        state = ForwardSharingState()
        state._entries = {key: copy(payload) for key, payload in self._entries.items()}
        return state

    @staticmethod
    def _validate_payload_type(payload_type: type[Any]) -> None:
        if not isinstance(payload_type, type):
            raise TypeError("payload_type must be a type")


def _get_carrier(packed_seq_params, attention_mask, config):
    carrier = next(
        (
            candidate
            for candidate in (packed_seq_params, attention_mask, config)
            if candidate is not None
        ),
        None,
    )
    if carrier is None:
        raise ValueError("forward sharing requires packed_seq_params, attention_mask, or config")
    return carrier


def get_forward_sharing_state(
    packed_seq_params: object | None = None,
    attention_mask: object | None = None,
    config: object | None = None,
) -> ForwardSharingState:
    """Return the sharing state attached to the highest-priority carrier.

    Carrier priority is ``packed_seq_params``, then ``attention_mask``, then
    ``config``. At least one carrier must be available.
    """
    carrier = _get_carrier(packed_seq_params, attention_mask, config)

    state = getattr(carrier, _FORWARD_SHARING_STATE_ATTR, None)
    if state is None:
        state = ForwardSharingState()
        setattr(carrier, _FORWARD_SHARING_STATE_ATTR, state)
    elif not isinstance(state, ForwardSharingState):
        raise TypeError(f"{_FORWARD_SHARING_STATE_ATTR} must contain a ForwardSharingState")
    return state


@contextmanager
def use_forward_sharing_state(
    state: ForwardSharingState,
    packed_seq_params: object | None = None,
    attention_mask: object | None = None,
    config: object | None = None,
) -> Iterator[None]:
    """Temporarily bind a checkpoint/schedule-owned state to its current input carrier.

    Restore the previous binding on exit without clearing payloads or changing
    ``active``. The checkpoint or schedule remains responsible for state cleanup.
    """
    carrier = _get_carrier(packed_seq_params, attention_mask, config)
    previous = getattr(carrier, _FORWARD_SHARING_STATE_ATTR, None)
    setattr(carrier, _FORWARD_SHARING_STATE_ATTR, state)
    try:
        yield
    finally:
        if previous is None:
            delattr(carrier, _FORWARD_SHARING_STATE_ATTR)
        else:
            setattr(carrier, _FORWARD_SHARING_STATE_ATTR, previous)


@contextmanager
def forward_sharing_lifetime(
    packed_seq_params: object | None = None,
    attention_mask: object | None = None,
    config: object | None = None,
) -> Iterator[ForwardSharingState]:
    """Clear a forward's payloads on entry and exit, including exceptional exits.

    Nested scopes using the same active state leave cleanup to the outer owner.
    The state stays attached to its carrier after exit, but its payloads are cleared.
    """
    state = get_forward_sharing_state(packed_seq_params, attention_mask, config)
    if state.active:
        yield state
        return
    state.clear()
    state.active = True
    try:
        yield state
    finally:
        state.clear()
        state.active = False


def preserve_forward_sharing_for_checkpoint(
    function: Callable, packed_seq_params: object | None, config: object, attention_mask_arg: int
) -> Callable:
    """Pin caller-selected payloads per checkpoint, independent of later forwards.

    Replays temporarily bind a private snapshot to the actual replay carrier (a detached
    mask need not be the original Python tensor). The caller decides whether this mechanism
    is appropriate for its payloads and checkpoint boundary. Snapshots copy payload
    records, not tensor storage or autograd graphs; retained tensors follow the
    checkpoint closure's lifetime. This does not make arbitrary recompute boundaries safe.
    """
    saved_state = None

    @wraps(function)
    def wrapped(*args):
        nonlocal saved_state
        attention_mask = args[attention_mask_arg]
        if saved_state is None:
            result = function(*args)
            saved_state = get_forward_sharing_state(
                packed_seq_params, attention_mask, config
            ).snapshot()
            return result

        replay_state = saved_state.snapshot()
        replay_state.active = True
        try:
            with use_forward_sharing_state(replay_state, packed_seq_params, attention_mask, config):
                return function(*args)
        finally:
            replay_state.clear()

    return wrapped
