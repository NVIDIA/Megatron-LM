# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Honor NeMo Gym worker-custody admissions with a Megatron RL-owned prefix store.

NeMo RL answers a Gym ``token_in`` admission by fetching the parent call's exact tokens from
TransferQueue in its prompt preparer and staging the finished call through Gym's capture
helper. Megatron RL has no data plane, so this module provides the store: a ZMQ key-value
service rank 0 runs for the job, fed by every engine's stager and read by every engine's
preparer through the two consumer hooks the engine already exposes.

Admission modes other than ``token_in`` and ``text`` (Gym's ``candidate`` source-history mode,
generation-cut resumes) are not prepared here and fail closed in the stager.

The stager runs Gym's capture helper in the trainer process, so ``nemo_gym`` must be importable
there whenever Gym sends admissions; Megatron RL does not depend on it otherwise. Without it every
admitted call is acknowledged ``capture_failed`` and the condition is logged once.
"""

from __future__ import annotations

import importlib.util
import logging
import socket
import threading
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import msgpack
import zmq

from megatron.core.inference.inference_request import (
    PREFIX_EOS_TOKEN_ID_FIELD,
    PREFIX_EXPANDED_TOKEN_COUNT_FIELD,
    PREFIX_MEDIA_COUNT_FIELD,
    PREFIX_TEMPLATE_TOKEN_IDS_FIELD,
    FinishedRequestRecord,
    OffloadedRequestPayload,
    RequestPromptPreparationResult,
)

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# Gym's admission rides in offload_params under this key (nemo_gym's NG_CAPTURE_FIELD).
NG_CAPTURE_FIELD = "ng_capture"
NG_COMMIT_COORDS_FIELD = "ng_commit_coords"
# A store round trip that takes longer than this fails the request instead of stalling the engine.
STORE_TIMEOUT_MS = 5000

HeldCall = tuple[list[int], list[int]]


class PrefixStoreServer:
    """Rank 0's table of finished calls, uid -> packed prompt and generated ids, over ZMQ."""

    def __init__(self, host: str | None = None) -> None:
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        port = self._socket.bind_to_random_port("tcp://0.0.0.0")
        self.address = f"tcp://{host or socket.gethostname()}:{port}"
        # Frames stay packed: a few bytes per token instead of a Python int per token.
        self._held: dict[str, bytes] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, name="prefix-store", daemon=True)
        self._thread.start()

    def _handle(self, frame: bytes) -> bytes:
        op, uid, payload = msgpack.unpackb(frame, raw=False)
        if op == "put":
            self._held[uid] = payload
            return msgpack.packb(None)
        if op == "get":
            return self._held.get(uid, msgpack.packb(None))
        if op == "forget":
            for key in payload:
                self._held.pop(key, None)
            return msgpack.packb(None)
        raise ValueError(f"unknown prefix store op {op!r}")

    def _serve(self) -> None:
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        while not self._stop.is_set():
            if not poller.poll(timeout=100):
                continue
            frame = self._socket.recv_multipart()[0]  # REP must consume the whole message
            try:
                reply = self._handle(frame)
            except Exception:  # a bad frame must not take the server down or hang its client
                logger.exception("prefix store rejected a request")
                reply = msgpack.packb({"error": "malformed prefix store request"})
            self._socket.send(reply)
        self._socket.close(linger=0)
        self._context.term()

    def size(self) -> int:
        """Number of calls currently held."""
        return len(self._held)

    def close(self) -> None:
        """Stop the serving thread and release the socket and context."""
        self._stop.set()
        self._thread.join()


class PrefixStoreClient:
    """Client of ``PrefixStoreServer``: one REQ socket per calling thread, no shared lock.

    Stager threads and the engine loop's preparer never wait on one another's exchange, so a
    stalled store costs each caller one timeout instead of queueing them all behind it. The
    client owns its ZMQ context: the engine terminates the process-wide
    ``zmq.Context.instance()`` at shutdown, and that call blocks until every socket in the
    context is closed, so sharing it would hang the teardown; ``close()`` releases this
    client's sockets and context instead.
    """

    def __init__(self, address: str, timeout_ms: int = STORE_TIMEOUT_MS) -> None:
        self.address = address
        self.timeout_ms = timeout_ms
        self._context = zmq.Context()
        self._local = threading.local()  # the calling thread's REQ socket
        self._sockets: set[zmq.Socket] = set()  # every socket opened; closed before term()
        self._sockets_lock = threading.Lock()

    def _socket(self) -> zmq.Socket:
        sock = getattr(self._local, "sock", None)
        if sock is None:
            sock = self._context.socket(zmq.REQ)
            sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(self.address)
            self._local.sock = sock
            with self._sockets_lock:
                self._sockets.add(sock)
        return sock

    def _discard(self, sock: zmq.Socket) -> None:
        """Drop the calling thread's socket; a REQ socket is unusable after a failed exchange."""
        with self._sockets_lock:
            self._sockets.discard(sock)
        sock.close(linger=0)
        self._local.sock = None

    def _request(self, op: str, uid: str | None = None, payload: Any = None) -> Any:
        sock = self._socket()
        try:
            sock.send(msgpack.packb([op, uid, payload], use_bin_type=True))
            reply = msgpack.unpackb(sock.recv(), raw=False)
        except zmq.ZMQError as error:
            self._discard(sock)
            raise RuntimeError(f"prefix store {op} at {self.address} failed: {error}") from error
        if isinstance(reply, dict) and "error" in reply:
            raise RuntimeError(f"prefix store {op} at {self.address} failed: {reply['error']}")
        return reply

    def close(self) -> None:
        """Release every thread's socket and the context; call once nothing stages or prepares."""
        with self._sockets_lock:
            for sock in self._sockets:
                sock.close(linger=0)
            self._sockets.clear()
        self._context.term()

    def ping(self) -> None:
        """Round-trip once; raises when the store is unreachable, so the launch fails early."""
        self._request("get", "")

    def put(self, uid: str, prompt_token_ids: list[int], generated_token_ids: list[int]) -> None:
        """Publish one finished call's prompt and generated token ids under ``uid``."""
        packed = msgpack.packb(
            [list(prompt_token_ids), list(generated_token_ids)], use_bin_type=True
        )
        self._request("put", uid, packed)

    def get(self, uid: str) -> HeldCall | None:
        """The call held under ``uid`` as ``(prompt_token_ids, generated_token_ids)``, or None."""
        held = self._request("get", uid)
        return None if held is None else (held[0], held[1])

    def forget(self, uids: Iterable[str]) -> None:
        """Drop the calls held under ``uids``; unknown ids are ignored."""
        uids = list(uids)
        if uids:
            self._request("forget", None, uids)


def replace_prefix_tokens(
    prefix_token_ids: Sequence[int],
    template_prefix_token_ids: Sequence[int],
    template_token_ids: Sequence[int],
    eos_token_ids: Iterable[int],
) -> list[int]:
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
    """RequestPromptPreparer that resolves a Gym ``token_in`` admission from the prefix store.

    Anything it cannot honor raises: the engine fails the request at admission, which is
    cheaper and clearer than generating a turn whose capture the stager must then poison.
    """

    def __init__(self, store: PrefixStoreClient, cache_size: int = 256) -> None:
        self.store = store
        # Parents fetched so far. Consecutive turns have distinct parents.
        self._cache: OrderedDict[str, HeldCall] = OrderedDict()
        self._cache_size = cache_size

    def _parent(self, uid: str) -> HeldCall:
        held = self._cache.get(uid)
        if held is None:
            held = self.store.get(uid)
            if held is None:
                raise LookupError(
                    f"Gym admission names parent call {uid!r}, which the prefix store does not hold"
                )
            self._cache[uid] = held
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return held

    def prepare_prompt(
        self,
        prompt: str | list[int] | torch.Tensor,
        *,
        offload_params: dict[str, Any] | None = None,
    ) -> RequestPromptPreparationResult:
        admission = (offload_params or {}).get(NG_CAPTURE_FIELD)
        if not isinstance(admission, dict) or admission.get("mode") != "token_in":
            if isinstance(admission, dict) and PREFIX_MEDIA_COUNT_FIELD in offload_params:
                # Nothing is spliced for a text root: the engine expands every media placeholder.
                offload_params = {
                    k: v for k, v in offload_params.items() if k != PREFIX_MEDIA_COUNT_FIELD
                }
            return RequestPromptPreparationResult(prompt=prompt, offload_params=offload_params)
        prefix = list(admission.get("required_prefix_token_ids") or [])
        if not prefix:
            # The parent's prompt already began with its own exact prefix: its whole call is ours.
            chain = admission.get("staging_chain") or []
            if not chain:
                raise ValueError(
                    "Gym token_in admission carries neither a prefix nor a staging chain"
                )
            held = self._parent(chain[-1])
            prefix = held[0] + held[1]
        if len(prefix) != admission.get("prev_len"):
            raise ValueError(
                f"Gym admission expects a prefix of {admission.get('prev_len')} tokens, "
                f"resolved {len(prefix)}"
            )
        template_prefix = offload_params.get(PREFIX_TEMPLATE_TOKEN_IDS_FIELD)
        eos_token_ids = offload_params.get(PREFIX_EOS_TOKEN_ID_FIELD)
        if template_prefix is None or eos_token_ids is None:
            raise ValueError(
                "Gym token_in admission arrived without the endpoint's prefix splice metadata"
            )
        tokens = list(prompt) if isinstance(prompt, (list, tuple)) else prompt.tolist()
        spliced = replace_prefix_tokens(prefix, template_prefix, tokens, eos_token_ids)
        # Inline the resolved prefix so the stager can prove the prompt began with it.
        offload_params = {
            **offload_params,
            NG_CAPTURE_FIELD: {**admission, "required_prefix_token_ids": prefix},
        }
        if PREFIX_MEDIA_COUNT_FIELD in offload_params:
            # The parent's tokens are already media-expanded; the engine expands only the suffix.
            offload_params[PREFIX_EXPANDED_TOKEN_COUNT_FIELD] = len(prefix)
        return RequestPromptPreparationResult(prompt=spliced, offload_params=offload_params)


_GYM_MISSING_LOGGED = False


def _log_gym_missing_once(uid: str) -> None:
    global _GYM_MISSING_LOGGED
    if not _GYM_MISSING_LOGGED:
        _GYM_MISSING_LOGGED = True
        if importlib.util.find_spec("nemo_gym") is None:
            problem = "nemo_gym is not importable in the trainer process"
        else:
            problem = (
                "the installed nemo_gym has no Megatron capture adapter under "
                "nemo_gym.models.token_capture or nemo_gym.token_id_capture (needs >= 0.7.0)"
            )
        logger.error(
            "Gym admitted request %s but %s; every admitted call is acknowledged "
            "capture_failed. Put a compatible Gym checkout or wheel on the trainer's path.",
            uid,
            problem,
        )


def _weight_version(record: FinishedRequestRecord) -> int:
    """The policy epoch the call was admitted under; spans are ``(token position, epoch)``."""
    return record.policy_epoch[0][1] if record.policy_epoch else 0


def _failed_coordinates(admission: dict[str, Any], weight_version: int) -> dict[str, Any]:
    """Gym's ``capture_failed`` acknowledgement, built without Gym types so it cannot raise."""
    prev_len = admission.get("prev_len") or 0
    return {
        "capture_key": admission.get("capture_key"),
        "model_call_id": admission.get("model_call_id"),
        "parent_call_id": admission.get("parent_call_id"),
        "prev_len": prev_len,
        "delta_len": 0,
        "cum_len": prev_len,
        "weight_version": weight_version,
        "disposition": "capture_failed",
    }


def stage_admitted_call(
    uid: str,
    payload: OffloadedRequestPayload,
    finished_metadata: FinishedRequestRecord,
    admission: dict[str, Any],
    prefix_store: PrefixStoreClient | None = None,
) -> dict[str, Any]:
    """Run Gym's worker capture over the finished call; return the coordinates it owes Gym.

    Capture failures never fail the completion: any error here, including a Gym schema the
    installed ``nemo_gym`` no longer accepts or a prefix store that stopped answering, becomes
    ``capture_failed`` coordinates.
    """
    weight_version = _weight_version(finished_metadata)
    try:
        try:
            from nemo_gym.models.token_capture.adapters.megatron import MegatronCaptureAdapter
            from nemo_gym.models.token_capture.staging.capture import RolloutTokenCapture
            from nemo_gym.models.token_capture.staging.records import CaptureAdmission, StageResult
        except ImportError:
            # nemo_gym before 0.8.0, which has not moved token_id_capture yet.
            from nemo_gym.token_id_capture.adapters.megatron import MegatronCaptureAdapter
            from nemo_gym.token_id_capture.staging.capture import RolloutTokenCapture
            from nemo_gym.token_id_capture.staging.records import CaptureAdmission, StageResult
    except ImportError:
        _log_gym_missing_once(uid)
        return {NG_COMMIT_COORDS_FIELD: _failed_coordinates(admission, weight_version)}
    try:
        if prefix_store is not None:
            # Published before the acknowledgement, so a continuation can only name a held parent.
            prefix_store.put(uid, payload.prompt_token_ids, payload.generated_token_ids)

        class _LedgerSink:
            """The ledger already holds the full record; its uid is the staging key."""

            def stage(self, record, *, attachments=None):
                return StageResult(ok=True, staging_key=uid)

        parsed = CaptureAdmission.model_validate(admission)
        capture = RolloutTokenCapture(
            sink=_LedgerSink(),
            weight_version_fn=lambda: weight_version,
            adapter=MegatronCaptureAdapter(),
        )
        # A staging-chain admission the preparer did not resolve raises here and fails closed.
        call = capture.begin_call(
            parsed,
            prefix_token_ids=list(parsed.required_prefix_token_ids) or None,
            weight_version=weight_version,
        )
        coords = capture.complete_call_from_response(call, payload).model_dump(mode="json")
    except Exception:
        logger.exception("Gym capture failed for request %s; acknowledging capture_failed", uid)
        coords = _failed_coordinates(admission, weight_version)
    return {NG_COMMIT_COORDS_FIELD: coords}
