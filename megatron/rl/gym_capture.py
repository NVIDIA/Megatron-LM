# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Honor NeMo Gym worker-custody admissions with a Megatron RL-owned prefix store.

NeMo RL answers a Gym ``token_in`` admission by fetching the parent call's exact tokens from
TransferQueue in its prompt preparer and staging the finished call through Gym's capture
helper. Megatron RL has no data plane, so this module provides the store: a ZMQ key-value
service rank 0 runs for the job, fed by every engine's stager and read by every engine's
preparer through the two consumer hooks the engine already exposes.
"""

from __future__ import annotations

import socket
import threading
from collections import OrderedDict
from typing import Any

import msgpack
import zmq

from megatron.core.inference.inference_request import (
    PREFIX_EOS_TOKEN_ID_FIELD,
    PREFIX_TEMPLATE_TOKEN_IDS_FIELD,
    FinishedRequestRecord,
    OffloadedRequestPayload,
    RequestPromptPreparationResult,
)

# Gym's admission rides in offload_params under this key (nemo_gym's NG_CAPTURE_FIELD).
NG_CAPTURE_FIELD = "ng_capture"
NG_COMMIT_COORDS_FIELD = "ng_commit_coords"

HeldCall = tuple[list[int], list[int]]


class PrefixStoreServer:
    """Rank 0's table of finished calls (uid -> prompt and generated ids), served over ZMQ."""

    def __init__(self, host: str | None = None) -> None:
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        port = self._socket.bind_to_random_port("tcp://0.0.0.0")
        self.address = f"tcp://{host or socket.gethostname()}:{port}"
        self._held: dict[str, HeldCall] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, name="prefix-store", daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        while not self._stop.is_set():
            if not poller.poll(timeout=100):
                continue
            op, uid, payload = msgpack.unpackb(self._socket.recv(), raw=False)
            reply = None
            if op == "put":
                self._held[uid] = (payload[0], payload[1])
            elif op == "get":
                reply = self._held.get(uid)
            elif op == "forget":
                for key in payload:
                    self._held.pop(key, None)
            self._socket.send(msgpack.packb(reply, use_bin_type=True))
        self._socket.close(linger=0)
        self._context.term()

    def __len__(self) -> int:
        return len(self._held)

    def close(self) -> None:
        self._stop.set()
        self._thread.join()


class PrefixStoreClient:
    """Thread-safe client of ``PrefixStoreServer``; one REQ socket per calling thread."""

    def __init__(self, address: str) -> None:
        self.address = address
        self._context = zmq.Context.instance()
        self._local = threading.local()

    def _request(self, op: str, uid: str | None = None, payload: Any = None) -> Any:
        sock = getattr(self._local, "socket", None)
        if sock is None:
            sock = self._local.socket = self._context.socket(zmq.REQ)
            sock.connect(self.address)
        sock.send(msgpack.packb([op, uid, payload], use_bin_type=True))
        return msgpack.unpackb(sock.recv(), raw=False)

    def put(self, uid: str, prompt_token_ids: list[int], generated_token_ids: list[int]) -> None:
        self._request("put", uid, [list(prompt_token_ids), list(generated_token_ids)])

    def get(self, uid: str) -> HeldCall | None:
        held = self._request("get", uid)
        return None if held is None else (held[0], held[1])

    def forget(self, uids) -> None:
        uids = list(uids)
        if uids:
            self._request("forget", None, uids)


def replace_prefix_tokens(
    prefix_token_ids, template_prefix_token_ids, template_token_ids, eos_token_ids
):
    """Splice the exact earlier turns in place of their re-rendered form (EOS-count boundary)."""
    eos_token_ids = set(eos_token_ids)
    eos_count = sum(token_id in eos_token_ids for token_id in template_prefix_token_ids)
    if eos_count <= 0:
        raise ValueError("the chat template's turn terminator is not among the EOS token ids")
    seen = 0
    for position, token_id in enumerate(template_token_ids):
        if token_id in eos_token_ids:
            seen += 1
            if seen == eos_count:
                suffix = template_token_ids[position:]
                break
    else:
        raise ValueError(f"expected {eos_count} EOS token(s) before the new turn, found {seen}")
    if prefix_token_ids and prefix_token_ids[-1] in eos_token_ids:
        suffix = suffix[1:]  # the exact turn already ends with its own EOS
    return list(prefix_token_ids) + list(suffix)


class GymPromptPreparer:
    """RequestPromptPreparer that resolves a Gym ``token_in`` admission from the prefix store."""

    def __init__(self, store: PrefixStoreClient, cache_size: int = 256) -> None:
        self.store = store
        # Recently fetched parents, so a conversation's turns cost one round trip, not one per turn.
        self._cache: OrderedDict[str, HeldCall] = OrderedDict()
        self._cache_size = cache_size

    def _parent(self, uid: str) -> HeldCall | None:
        held = self._cache.get(uid)
        if held is None:
            held = self.store.get(uid)
            if held is not None:
                self._cache[uid] = held
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
        return held

    def prepare_prompt(self, prompt, *, offload_params=None) -> RequestPromptPreparationResult:
        admission = (offload_params or {}).get(NG_CAPTURE_FIELD)
        if not isinstance(admission, dict) or admission.get("mode") != "token_in":
            return RequestPromptPreparationResult(prompt=prompt, offload_params=offload_params)
        prefix = list(admission.get("required_prefix_token_ids") or [])
        if not prefix:
            # The parent's prompt already began with its own exact prefix: its whole call is ours.
            chain = admission.get("staging_chain") or []
            held = self._parent(chain[-1]) if chain else None
            if held is None:
                # Unresolvable: leave the prompt re-tokenized; the stager then fails the capture.
                return RequestPromptPreparationResult(prompt=prompt, offload_params=offload_params)
            prefix = held[0] + held[1]
        template_prefix = offload_params.get(PREFIX_TEMPLATE_TOKEN_IDS_FIELD)
        eos_token_ids = offload_params.get(PREFIX_EOS_TOKEN_ID_FIELD)
        if (
            len(prefix) != admission.get("prev_len")
            or template_prefix is None
            or eos_token_ids is None
        ):
            return RequestPromptPreparationResult(prompt=prompt, offload_params=offload_params)
        if isinstance(prompt, (list, tuple)):
            tokens = list(prompt)
        else:
            tokens = prompt.tolist()
        try:
            spliced = replace_prefix_tokens(prefix, template_prefix, tokens, eos_token_ids)
        except ValueError:
            return RequestPromptPreparationResult(prompt=prompt, offload_params=offload_params)
        # Inline the resolved prefix so the stager can prove the prompt began with it.
        offload_params = {
            **offload_params,
            NG_CAPTURE_FIELD: {**admission, "required_prefix_token_ids": prefix},
        }
        return RequestPromptPreparationResult(prompt=spliced, offload_params=offload_params)


def _weight_version(record: FinishedRequestRecord) -> int:
    """The oldest policy epoch that generated the call, as Gym stamps a resumed generation."""
    return min((epoch for epoch, _ in record.policy_epoch or ()), default=0)


def stage_admitted_call(
    uid: str,
    payload: OffloadedRequestPayload,
    finished_metadata: FinishedRequestRecord,
    admission: dict[str, Any],
) -> dict[str, Any]:
    """Run Gym's worker capture over the finished call; return the coordinates it owes Gym."""
    from nemo_gym.token_id_capture.adapters.megatron import MegatronCaptureAdapter
    from nemo_gym.token_id_capture.staging.capture import CaptureError, RolloutTokenCapture
    from nemo_gym.token_id_capture.staging.records import (
        CaptureAdmission,
        CommitCoords,
        StageResult,
    )

    class _LedgerSink:
        """The ledger already holds the full record; its uid is the staging key."""

        def stage(self, record, *, attachments=None):
            return StageResult(ok=True, staging_key=uid)

    parsed = CaptureAdmission.model_validate(admission)
    weight_version = _weight_version(finished_metadata)
    capture = RolloutTokenCapture(
        sink=_LedgerSink(),
        weight_version_fn=lambda: weight_version,
        adapter=MegatronCaptureAdapter(),
    )
    try:
        call = capture.begin_call(
            parsed,
            prefix_token_ids=list(parsed.required_prefix_token_ids) or None,
            weight_version=weight_version,
        )
    except CaptureError:
        # No exact prefix was resolved for a staging-chain admission: fail closed, as Gym expects.
        coords = CommitCoords(
            rollout_id=parsed.rollout_id,
            model_call_id=parsed.model_call_id,
            parent_call_id=parsed.parent_call_id,
            prev_len=parsed.prev_len,
            delta_len=0,
            cum_len=parsed.prev_len,
            weight_version=weight_version,
            disposition="capture_failed",
        )
    else:
        coords = capture.complete_call_from_response(call, payload)
    return {NG_COMMIT_COORDS_FIELD: coords.model_dump(mode="json")}
