# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""How large are the packets on the frontend wire?

Every request and reply between the HTTP frontend, the coordinator and the
engines is a msgpack-packed list on a ZMQ socket. Nothing measured those payloads
before, so growth was invisible until it showed up as a throughput regression
with no obvious cause -- a field added to a reply, or log probs left enabled by
default, silently multiplies the bytes moved per request.

These tests pin the *shape* of the size curve rather than exact byte counts:

* a bounded fixed cost per message, and
* a bounded marginal cost per prompt token and per generated token.

All ceilings are one-sided, so shrinking a payload always passes and only growth
fails. Measured values at the time of writing are quoted next to each ceiling so
the headroom is visible; they were taken with realistic five-digit token ids,
since msgpack's integer encoding makes byte counts depend on token id magnitude.

The last class checks the WireMetrics instrumentation itself against a real
coordinator process, so the counters cannot silently drift from the bytes that
actually cross the socket.
"""

import asyncio
import random

import msgpack
import pytest

from megatron.core.inference.headers import Headers
from megatron.core.inference.inference_client import InferenceClient
from megatron.core.inference.sampling_params import SamplingParams
from tests.unit_tests.inference.frontend_test_utils import (
    ByteLevelFastTokenizer,
    ByteTokenizer,
    build_engine_reply,
    standalone_coordinator,
    synthesize_generated_tokens,
)

# Ceilings, all one-sided. Measured values in comments.
MAX_REQUEST_FIXED_BYTES = 512  # measured 327
MAX_SAMPLING_PARAMS_BYTES = 512  # measured 323
MAX_REQUEST_BYTES_PER_PROMPT_TOKEN = 5.0  # measured 3.8
MAX_REPLY_FIXED_BYTES = 1536  # measured 1011
MAX_REPLY_BYTES_PER_GENERATED_TOKEN = 4.0  # measured 2.0
MAX_LOGPROB_BYTES_PER_GENERATED_TOKEN = 12.0  # measured 9.0


def packed_size(payload) -> int:
    """Serialized size of a payload, exactly as the client and coordinator pack it."""
    return len(msgpack.packb(payload, use_bin_type=True))


def token_ids(count, seed=1234):
    """Realistic five-digit token ids; msgpack sizes depend on id magnitude."""
    rng = random.Random(seed)
    return [rng.randrange(10000, 99999) for _ in range(count)]


def submit_request_payload(prompt_tokens, sampling_params):
    """The SUBMIT_REQUEST payload InferenceClient.add_request builds."""
    return [Headers.SUBMIT_REQUEST.value, 0, prompt_tokens, sampling_params.serialize()]


def engine_reply_payload(prompt_tokens, num_generated, return_prompt_tokens=False, log_probs=False):
    """The ENGINE_REPLY payload the coordinator forwards to a client."""
    tokenizer = ByteTokenizer()
    sampling_params = SamplingParams(
        num_tokens_to_generate=num_generated,
        return_prompt_tokens=return_prompt_tokens,
        return_log_probs=log_probs,
    )
    generated_tokens = synthesize_generated_tokens(num_generated, tokenizer)
    reply = build_engine_reply(
        0,
        prompt_tokens,
        generated_tokens,
        sampling_params,
        generated_log_probs=[-0.512345] * num_generated if log_probs else None,
    )
    reply["generated_text"] = tokenizer.detokenize(generated_tokens)
    return [Headers.ENGINE_REPLY.value, 0, reply]


def marginal_bytes_per_token(size_fn, counts):
    """Largest per-token slope between consecutive measurements of size_fn."""
    slopes = []
    previous_count, previous_size = counts[0], size_fn(counts[0])
    for count in counts[1:]:
        current_size = size_fn(count)
        slopes.append((current_size - previous_size) / (count - previous_count))
        previous_count, previous_size = count, current_size
    return max(slopes)


class TestRequestPacketSize:
    """Client -> coordinator -> engine request payloads."""

    def test_minimal_request_fits_in_budget(self):
        """A request with a one-token prompt is almost entirely fixed overhead."""
        size = packed_size(submit_request_payload([42], SamplingParams(num_tokens_to_generate=16)))
        assert size <= MAX_REQUEST_FIXED_BYTES

    def test_sampling_params_are_the_fixed_cost_of_every_request(self):
        """SamplingParams is serialized in full on every single request.

        All 18 fields ship on the wire whether or not the caller set them, and at
        ~323 bytes they are over 95% of a short request. That is the price of the
        current protocol, not a defect, but it caps how cheap a request can get:
        adding a field here costs bytes on every request forever.
        """
        params_bytes = packed_size(SamplingParams(num_tokens_to_generate=16).serialize())
        assert params_bytes <= MAX_SAMPLING_PARAMS_BYTES

    def test_request_bytes_grow_linearly_with_prompt_length(self):
        """Prompt tokens must cost a bounded number of bytes each.

        Guards against a regression that sends prompts in a wider encoding (a
        float tensor, or ids as strings), which would inflate every prefill.
        """
        sampling_params = SamplingParams(num_tokens_to_generate=16)
        slope = marginal_bytes_per_token(
            lambda n: packed_size(submit_request_payload(token_ids(n), sampling_params)),
            [0, 256, 1024, 4096],
        )
        assert slope <= MAX_REQUEST_BYTES_PER_PROMPT_TOKEN


class TestPayloadDeterminism:
    """Payload sizes have to be reproducible before they can be gated."""

    def test_tokenizer_assigns_the_same_ids_in_every_instance(self):
        """Two ByteLevelFastTokenizers must agree on ids for the same text.

        The tokenizer is rebuilt from scratch in the coordinator and engine
        subprocesses, and the perf test gates request_bytes_per_request. Both
        break if ids are not reproducible: the processes disagree about what a
        token means, and msgpack spends a different number of bytes per id, which
        moved the recorded byte counts by over 20% between two identical runs.
        """
        text = "the quick brown fox jumps over the lazy dog"
        assert ByteLevelFastTokenizer().tokenize(text) == ByteLevelFastTokenizer().tokenize(text)


class TestReplyPacketSize:
    """Engine -> coordinator -> client reply payloads."""

    def test_minimal_reply_fits_in_budget(self):
        size = packed_size(engine_reply_payload([], 0))
        assert size <= MAX_REPLY_FIXED_BYTES

    def test_reply_bytes_grow_linearly_with_generated_tokens(self):
        slope = marginal_bytes_per_token(
            lambda n: packed_size(engine_reply_payload([], n)), [0, 16, 128, 512]
        )
        assert slope <= MAX_REPLY_BYTES_PER_GENERATED_TOKEN

    def test_log_probs_cost_per_token_is_bounded(self):
        """Log probs are the largest opt-in cost on a reply.

        They roughly quintuple the per-token reply cost (~2 -> ~11 bytes), which
        is why the frontend must not enable them by default. Serialized as
        float64; switching to float32 would nearly halve this.
        """
        without = packed_size(engine_reply_payload([], 512, log_probs=False))
        with_log_probs = packed_size(engine_reply_payload([], 512, log_probs=True))
        assert (with_log_probs - without) / 512 <= MAX_LOGPROB_BYTES_PER_GENERATED_TOKEN

    def test_reply_size_is_independent_of_prompt_length(self):
        """A reply must not carry the prompt back unless the caller asked for it.

        DynamicInferenceRequest.serialize drops prompt_tokens when
        return_prompt_tokens is False, to keep the large prompt tensor off the
        engine->coordinator->API path. remaining_prompt_tokens starts life as a
        copy of prompt_tokens, so it has to be dropped too: while it was not, the
        full prompt still shipped at ~3.8 bytes per token, over 90% of the reply
        for a 4k-token prompt.

        Both prompt lengths here are checked against the same fixed budget rather
        than each other, so a reply that grows with the prompt fails no matter how
        large the constant part becomes.
        """
        for num_prompt_tokens in (8, 4096):
            size = packed_size(engine_reply_payload(token_ids(num_prompt_tokens), 16))
            assert size <= MAX_REPLY_FIXED_BYTES + 16 * MAX_REPLY_BYTES_PER_GENERATED_TOKEN

    def test_prompt_tokens_are_returned_when_requested(self):
        """return_prompt_tokens=True must actually put the prompt on the wire."""
        without = packed_size(engine_reply_payload(token_ids(512), 16))
        with_prompt = packed_size(
            engine_reply_payload(token_ids(512), 16, return_prompt_tokens=True)
        )
        assert with_prompt > without


class TestWireMetricsAccounting:
    """The instrumentation must match the bytes that really cross the socket."""

    @pytest.mark.internal
    def test_metrics_match_traffic_through_a_real_coordinator(self):
        """Drive a real coordinator and fake engine, then reconcile the counters.

        This is the end-to-end check that WireMetrics is wired into every send
        and receive path: the totals it reports must equal the payload sizes we
        can compute independently, and both directions must be accounted for.
        """
        num_requests = 8
        num_output_tokens = 12
        prompt = token_ids(64)

        async def run(addr):
            client = InferenceClient(addr)
            client.start(connect_timeout_seconds=30.0)
            try:
                sampling_params = SamplingParams(num_tokens_to_generate=num_output_tokens)
                futures = [
                    client.add_request(list(prompt), sampling_params) for _ in range(num_requests)
                ]
                await asyncio.wait_for(asyncio.gather(*futures), timeout=60.0)
                return client.get_wire_metrics()
            finally:
                client.stop()

        with standalone_coordinator(num_engines=1) as (addr, _engines):
            metrics = asyncio.run(run(addr))

        per_header = metrics["per_header"]

        # Sent: one CONNECT handshake plus one SUBMIT_REQUEST per request.
        assert per_header["CONNECT"]["sent_messages"] == 1
        assert per_header["SUBMIT_REQUEST"]["sent_messages"] == num_requests
        assert metrics["sent_messages"] == num_requests + 1

        # The SUBMIT_REQUEST bytes must equal what msgpack produces for the same
        # payload; add_request assigns ids 0..n-1, which can change the size.
        expected_bytes = sum(
            packed_size(
                [
                    Headers.SUBMIT_REQUEST.value,
                    request_id,
                    list(prompt),
                    SamplingParams(num_tokens_to_generate=num_output_tokens).serialize(),
                ]
            )
            for request_id in range(num_requests)
        )
        assert per_header["SUBMIT_REQUEST"]["sent_bytes"] == expected_bytes

        # Received: one CONNECT_ACK plus one ENGINE_REPLY per request.
        assert per_header["CONNECT_ACK"]["received_messages"] == 1
        assert per_header["ENGINE_REPLY"]["received_messages"] == num_requests
        assert per_header["ENGINE_REPLY"]["mean_received_bytes"] > 0

        # Replies dominate: they carry generated text, tokens and the record.
        assert metrics["received_bytes"] > metrics["sent_bytes"]
