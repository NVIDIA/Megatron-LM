# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import logging
import socket
from collections.abc import Iterable, Sequence
from datetime import timedelta
from typing import Any

import msgpack
import torch

from megatron.core.inference.inference_request import (
    PREFIX_EOS_TOKEN_ID_FIELD,
    PREFIX_EXPANDED_TOKEN_COUNT_FIELD,
    PREFIX_MEDIA_COUNT_FIELD,
    PREFIX_TEMPLATE_TOKEN_IDS_FIELD,
    FinishedRequestRecord,
    OffloadedRequestPayload,
    RequestPromptPreparationResult,
)

logger = logging.getLogger(__name__)

# Gym's admission rides in offload_params under this key (nemo_gym's NG_CAPTURE_FIELD).
NG_CAPTURE_FIELD = "ng_capture"
NG_COMMIT_COORDS_FIELD = "ng_commit_coords"
# A store round trip that takes longer than this fails the request instead of stalling the engine.
STORE_TIMEOUT = timedelta(seconds=5)

HeldCall = tuple[list[int], list[int]]


class PrefixStore:
    """Finished calls by uid, `(prompt token ids, generated token ids)`, in a rank-0 TCPStore.

    Thread-safe, so the stager's worker threads and the engine loop's preparer share one handle
    per rank; connecting to an unreachable master raises here and fails the launch.
    """

    def __init__(self, host: str | None = None, port: int = 0, *, is_master: bool = False) -> None:
        self.host = host or socket.gethostname()
        self._store = torch.distributed.TCPStore(
            self.host, port, is_master=is_master, timeout=STORE_TIMEOUT
        )
        self.port = self._store.port

    def put(self, uid: str, prompt_token_ids: list[int], generated_token_ids: list[int]) -> None:
        """Publish one finished call's prompt and generated token ids under `uid`."""
        packed = msgpack.packb(
            [list(prompt_token_ids), list(generated_token_ids)], use_bin_type=True
        )
        self._store.set(uid, packed)

    def get(self, uid: str) -> HeldCall | None:
        """The call held under `uid` as `(prompt_token_ids, generated_token_ids)`, or None."""
        if not self._store.check([uid]):  # get() alone would wait out the timeout on a miss
            return None
        prompt_token_ids, generated_token_ids = msgpack.unpackb(self._store.get(uid), raw=False)
        return prompt_token_ids, generated_token_ids

    def forget(self, uids: Iterable[str]) -> None:
        """Drop the calls held under `uids`; unknown ids are ignored."""
        for uid in uids:
            self._store.delete_key(uid)


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


class HeldPrefixPreparer:
    """RequestPromptPreparer that resolves a `token_in` admission's prefix from the held calls.

    Anything it cannot honor raises: the engine fails the request at admission, which is
    cheaper and clearer than generating a turn whose capture the stager must then poison.
    """

    def __init__(self, store: PrefixStore) -> None:
        self.store = store

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
            held = self.store.get(chain[-1])
            if held is None:
                raise LookupError(
                    f"Gym admission names parent call {chain[-1]!r}, which the prefix store "
                    "does not hold"
                )
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


def _weight_version(record: FinishedRequestRecord) -> int:
    """The policy epoch the call was admitted under; spans are `(token position, epoch)`."""
    return record.policy_epoch[0][1] if record.policy_epoch else 0


def held_receipt(
    uid: str,
    payload: OffloadedRequestPayload,
    finished_metadata: FinishedRequestRecord,
    admission: dict[str, Any],
) -> dict[str, Any]:
    """Gym's `held` commit coordinates for one finished call, from the ledger record alone.

    The delta is what the call added beyond its parent: the prompt past the admitted prefix and
    the generation. Raises when this worker cannot vouch for the call.
    """
    prompt = payload.prompt_token_ids
    if prompt is None:
        raise ValueError("the finished request carries no prompt token ids")
    mode = admission.get("mode")
    prev_len = int(admission.get("prev_len") or 0)
    if mode == "token_in":
        # The preparer inlined the parent's exact tokens; the engine must have generated from them.
        prefix = list(admission.get("required_prefix_token_ids") or [])
        if not prefix or len(prefix) != prev_len or list(prompt[:prev_len]) != prefix:
            raise ValueError("the generation prompt does not begin with the admitted prefix")
    elif mode != "text" or prev_len:
        raise ValueError(f"unsupported Gym admission mode {mode!r}")
    cum_len = len(prompt) + len(payload.generated_token_ids)
    if cum_len <= prev_len:
        raise ValueError("the finished request adds no tokens beyond its parent")
    return {
        "capture_key": admission["capture_key"],
        "model_call_id": admission["model_call_id"],
        "parent_call_id": admission.get("parent_call_id"),
        "prev_len": prev_len,
        "delta_len": cum_len - prev_len,
        "cum_len": cum_len,
        "weight_version": _weight_version(finished_metadata),
        "disposition": "held",
        "staging_key": uid,
    }


def stage_admitted_call(
    uid: str,
    payload: OffloadedRequestPayload,
    finished_metadata: FinishedRequestRecord,
    admission: dict[str, Any],
    prefix_store: PrefixStore | None = None,
) -> dict[str, Any]:
    """Publish the call for the next turn's prefix and answer Gym with its `held` receipt.

    Capture failures never fail the completion: on any error here (an admission this worker
    cannot vouch for, a prefix store that stopped answering) the reply carries no coordinates
    and Gym fails the call closed on its side.
    """
    try:
        if prefix_store is not None:
            # Published before the acknowledgement, so a continuation can only name a held parent.
            prefix_store.put(uid, payload.prompt_token_ids, payload.generated_token_ids)
        receipt = held_receipt(uid, payload, finished_metadata, admission)
    except Exception:
        logger.exception(
            "Custody receipt failed for request %s; the reply carries no coordinates", uid
        )
        return {}
    return {NG_COMMIT_COORDS_FIELD: receipt}
