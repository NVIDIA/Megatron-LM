# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from asyncio import wait_for

import pytest

from megatron.core.inference.headers import Headers
from tests.unit_tests.inference.coordinator_pairwise_utils import greedy_params, routed_model

pytestmark = [pytest.mark.internal, pytest.mark.asyncio]


async def test_rejected_admission_retires_before_healthy(monkeypatch):
    prompt, params, bad = list(range(4, 20)), greedy_params(), 0
    async with routed_model(monkeypatch, max_sequence_length=256) as h:
        direct = await h.direct(prompt, params)
        await h.start()
        if h.rank == 0:
            client, events = h.clients[0], h.service.events
            failed = await wait_for(client.add_request([4] * 257, params), 60)
            submitted = next(e["after"] for e in events if e["header"] == Headers.SUBMIT_REQUEST)
            failed_id, owner = next(iter(submitted.items()))
            bad = int((failed["request_id"], failed["status"]) != (failed_id, "FAILED"))
            bad |= sum(e["type"] == "FAIL" for e in failed["events"]) != 1
            bad |= not any(
                e["type"] == "ERROR_NONTRANSIENT"
                and e["payload"]["type"] == "MaxSequenceLengthOverflowError"
                for e in failed["events"]
            )
        assert await h.sync.all_reduce_max(
            len(h.engine.failed_request_ids), len(h.engine.requests), len(h.witnesses), bad
        ) == (0, 0, 0, 0)
        h.assert_retired()
        if h.rank == 0:
            final = await wait_for(client.add_request(prompt, params), 60)
            for key in ("status", "generated_tokens"):
                bad |= final[key] != direct[key]
            bad |= final["generated_text"] != h.tokenizer.detokenize(direct["generated_tokens"])
            submits = [e["after"] for e in events if e["header"] == Headers.SUBMIT_REQUEST]
            bad |= submits[-1] != {final["request_id"]: owner}
            replies = [
                item[0]
                for e in events
                if e["header"] == Headers.ENGINE_REPLY
                for item in e["metadata"][1]
            ]
            bad |= replies.count(failed_id) != 1
        assert await h.sync.all_reduce_max(bad, int(bool(h.witnesses))) == (0, 1)
        h.assert_retired()
