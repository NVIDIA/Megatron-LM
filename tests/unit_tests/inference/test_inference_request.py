# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import warnings

import msgpack
import numpy as np
import pytest
import torch

from megatron.core.inference.inference_request import (
    DynamicInferenceEventType,
    DynamicInferenceRequest,
    DynamicInferenceRequestRecord,
    FinishedRequestRecord,
    InferenceRequest,
    compute_block_hashes_batched,
    deserialize_ndarray,
    deserialize_tensor,
    resolve_multimodal_data_for_engine,
    serialize_multimodal_data,
    serialize_ndarray,
    serialize_tensor,
    unwrap_serialized_tensors,
)
from megatron.core.inference.sampling_params import SamplingParams


def _make_dynamic_request(**kwargs):
    defaults = dict(
        request_id=1,
        prompt_tokens=torch.tensor([1, 2, 3, 4]),
        sampling_params=SamplingParams(num_tokens_to_generate=5, termination_id=0),
    )
    defaults.update(kwargs)
    return DynamicInferenceRequest(**defaults)


def test_serialization_helpers_round_trip():
    """serialize_tensor / serialize_ndarray pair with their deserialize inverses;
    unwrap_serialized_tensors replaces ('tensor', list) sentinels in place and
    leaves other wrappers untouched. The wrapper protocol is the contract every
    higher-level serialize() call depends on."""
    t = torch.tensor([4, 5, 6, 7])
    assert deserialize_tensor(serialize_tensor(t)).tolist() == [4, 5, 6, 7]

    arr = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    arr_out = deserialize_ndarray(serialize_ndarray(arr))
    assert arr_out.dtype == np.float64
    assert np.array_equal(arr_out, arr)

    obj = {
        "a": ("tensor", [1, 2, 3]),
        "b": "plain",
        "c": ("ndarray", {"data": [], "dtype": "int32"}),
    }
    out = unwrap_serialized_tensors(obj)
    assert out["a"] == [1, 2, 3]
    assert out["b"] == "plain"
    assert out["c"] == ("ndarray", {"data": [], "dtype": "int32"})


def test_preexpanded_multimodal_request_round_trip():
    media = {
        "image": {"imgs": torch.ones(1, 2, 4), "imgs_sizes": torch.tensor([[2, 2]])},
        "media_tokens_preexpanded": True,
    }

    wire = serialize_multimodal_data(media)
    assert wire["media_tokens_preexpanded"] is True

    resolved = resolve_multimodal_data_for_engine(wire)
    assert resolved["media_tokens_preexpanded"] is True
    assert torch.equal(resolved["imgs"], media["image"]["imgs"])
    assert torch.equal(resolved["imgs_sizes"], media["image"]["imgs_sizes"])


def test_gym_style_compact_multimodal_request_omits_preexpanded_flag():
    wire = serialize_multimodal_data(
        {"image": {"imgs": torch.ones(1, 2, 4), "imgs_sizes": torch.tensor([[2, 2]])}}
    )
    assert "media_tokens_preexpanded" not in wire
    assert "media_tokens_preexpanded" not in resolve_multimodal_data_for_engine(wire)


def test_multimodal_serialization_generates_stable_content_keys():
    raw_a = serialize_multimodal_data({"image": [b"same-image"]})
    raw_b = serialize_multimodal_data({"image": [b"same-image"]})
    raw_c = serialize_multimodal_data({"image": [b"different-image"]})
    assert raw_a["media_cache_key"] == raw_b["media_cache_key"]
    assert raw_a["media_cache_key"] != raw_c["media_cache_key"]

    tensor_a = serialize_multimodal_data(
        {
            "image": {
                "imgs": torch.arange(8, dtype=torch.float32).reshape(1, 2, 4),
                "imgs_sizes": torch.tensor([[2, 2]]),
            }
        }
    )
    tensor_b = serialize_multimodal_data(
        {
            "image": {
                "imgs": torch.arange(8, dtype=torch.float32).reshape(1, 2, 4),
                "imgs_sizes": torch.tensor([[2, 2]]),
            }
        }
    )
    tensor_c = serialize_multimodal_data(
        {
            "image": {
                "imgs": torch.arange(8, dtype=torch.float32).reshape(2, 1, 4),
                "imgs_sizes": torch.tensor([[2, 2]]),
            }
        }
    )
    assert tensor_a["media_cache_key"] == tensor_b["media_cache_key"]
    # Shape participates in identity even when the flattened bytes are equal.
    assert tensor_a["media_cache_key"] != tensor_c["media_cache_key"]


def test_multimodal_serialization_rejects_user_media_cache_key():
    with pytest.raises(ValueError, match="computed automatically"):
        serialize_multimodal_data({"image": [b"image"], "media_cache_key": "user-provided"})


def test_compute_block_hashes_batched():
    """compute_block_hashes_batched produces one hash per *complete* block and
    chains: the hash of block i depends on block i-1. Single combined test
    because the contract is one function with one shape of behavior."""
    # Sub-block prompt → no hashes.
    assert compute_block_hashes_batched(torch.arange(3, dtype=torch.int64), block_size=4) == []
    # Two complete blocks + 2 leftover tokens → 2 distinct hashes; same input is deterministic.
    h = compute_block_hashes_batched(torch.arange(10, dtype=torch.int64), block_size=4)
    assert len(h) == 2 and h[0] != h[1]
    assert compute_block_hashes_batched(torch.arange(10, dtype=torch.int64), block_size=4) == h
    # Chained: mutating block 0 changes the hash of block 1 (load-bearing for prefix caching).
    prompt_b = torch.arange(8, dtype=torch.int64)
    prompt_b[0] = 99
    h_b = compute_block_hashes_batched(prompt_b, block_size=4)
    assert (
        compute_block_hashes_batched(torch.arange(8, dtype=torch.int64), block_size=4)[1] != h_b[1]
    )
    tokens = torch.arange(8, dtype=torch.int64)
    media_a = compute_block_hashes_batched(tokens, block_size=4, cache_salt="media-a")
    media_b = compute_block_hashes_batched(tokens, block_size=4, cache_salt="media-b")
    assert media_a != media_b
    assert media_a == compute_block_hashes_batched(tokens, block_size=4, cache_salt="media-a")


def test_inference_parameters_alias_warns_and_copies():
    """The legacy `inference_parameters` kwarg emits a deprecation warning and
    is copied into sampling_params. This is real backward-compat behavior."""
    sp = SamplingParams(temperature=0.5)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        req = InferenceRequest(request_id=1, prompt="hi", inference_parameters=sp)
        assert any("renamed" in str(item.message) for item in w)
    assert req.sampling_params is sp


def test_inference_request_serialize_round_trip_through_msgpack():
    """The full serialize → msgpack → deserialize cycle: tensor fields are
    wrapped as ('tensor', list), msgpack converts the tuple to a list, and
    _post_deserialize reconstructs the tensor. Same for ndarray fields on
    DynamicInferenceRequest. status=None must pass through. This is the only
    serialization contract callers actually depend on; the wrapper details
    (tuple shape, key presence) are implementation."""
    sp = SamplingParams(temperature=0.7, top_k=3, num_tokens_to_generate=1, termination_id=0)
    req = InferenceRequest(
        request_id=10,
        prompt="hello",
        sampling_params=sp,
        status=None,
        arrival_time=1.5,
        generated_tokens=torch.tensor([7, 8, 9]),
    )
    data = msgpack.unpackb(msgpack.packb(req.serialize()), raw=False)
    out = InferenceRequest.deserialize(data)
    assert out.request_id == 10 and out.prompt == "hello" and out.status is None
    assert out.arrival_time == 1.5
    assert out.sampling_params.temperature == 0.7 and out.sampling_params.top_k == 3
    assert out.generated_tokens.tolist() == [7, 8, 9]

    # ndarray wrappers handled via routing_indices on the DynamicInferenceRequest.
    # DynamicInferenceRequest.serialize asserts routing_indices.shape[0] == len(prompt) + len(generated) - 1,
    # so size the inputs to match: prompt=[1,2] + generated=[10] → 3 tokens → 2 rows.
    dyn = _make_dynamic_request(
        request_id=22,
        prompt_tokens=torch.tensor([1, 2]),
        generated_tokens=[10],
        routing_indices=np.array([[1, 2], [3, 4]], dtype=np.int32),
    )
    dyn_data = msgpack.unpackb(msgpack.packb(dyn.serialize()), raw=False)
    dyn_out = DynamicInferenceRequest.deserialize(dyn_data)
    assert isinstance(dyn_out.routing_indices, np.ndarray)
    assert dyn_out.routing_indices.tolist() == [[1, 2], [3, 4]]
    # The engine-minted uid (the OpenAI response id / ledger key) survives the wire.
    assert dyn_out.uid == dyn.uid


def test_dynamic_inference_request_post_init_prefix_caching():
    """DynamicInferenceRequest.__post_init__ computes block hashes if and only
    if (a) prefix caching is enabled, (b) block_size_tokens is set, and (c) the
    caller hasn't already supplied them. remaining_prompt_tokens is initialized
    to a copy of prompt_tokens. Both are non-trivial: they gate prefix-cache
    routing on every request submitted."""
    # Without block_size_tokens, no hashes are computed.
    req = _make_dynamic_request(enable_prefix_caching=True, block_size_tokens=None)
    assert req.precomputed_block_hashes == []
    assert torch.equal(req.remaining_prompt_tokens, req.prompt_tokens)
    assert req.remaining_prompt_length == 4

    # With block_size_tokens and no override, hashes are computed.
    req = _make_dynamic_request(
        prompt_tokens=torch.arange(8, dtype=torch.int64),
        block_size_tokens=4,
        enable_prefix_caching=True,
    )
    assert len(req.precomputed_block_hashes) == 2

    # With explicit precomputed_block_hashes, the supplied value wins.
    req = _make_dynamic_request(
        prompt_tokens=torch.arange(8, dtype=torch.int64),
        block_size_tokens=4,
        enable_prefix_caching=True,
        precomputed_block_hashes=[42],
    )
    assert req.precomputed_block_hashes == [42]


def test_dynamic_inference_request_tracked_metadata_defaults_termination_id():
    """Accessing `tracked_metadata` mutates a `termination_id=None` sampling
    param to -1 in-place (the runtime needs an integer sentinel)."""
    sp = SamplingParams(termination_id=None)
    req = _make_dynamic_request(sampling_params=sp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _ = req.tracked_metadata
    assert req.sampling_params.termination_id == -1


def test_dynamic_inference_request_record_checkpoint_and_merge():
    """RequestRecord.checkpoint() rolls the current request forward — prompt
    becomes prompt+generated, num_tokens_to_generate is debited, and the
    prefix-cache configuration is inherited while hashes are recomputed for the
    expanded prompt. The add_engine event is inherited (or created) so
    downstream tooling can find it. RequestRecord.merge() collapses the chain
    back into a single request with concatenated tokens, text, routing_indices,
    and the record's latency. Both are non-trivial state machines."""
    sp = SamplingParams(num_tokens_to_generate=8, termination_id=0)

    # checkpoint() inherits prefix-cache configuration and event_add_engine.
    req = DynamicInferenceRequest(
        request_id=1,
        prompt_tokens=torch.tensor([1, 2, 3, 4, 5, 6]),
        sampling_params=sp,
        generated_tokens=[7, 8],
        block_size_tokens=4,
        enable_prefix_caching=True,
    )
    original_hashes = req.precomputed_block_hashes
    original_event = req.add_event_add_engine()
    record = DynamicInferenceRequestRecord.from_request(req)
    record.checkpoint()
    assert len(record.requests) == 2
    new_req = record.requests[-1]
    assert new_req.prompt_tokens.tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert new_req.sampling_params.num_tokens_to_generate == 6
    assert new_req.block_size_tokens == 4
    assert new_req.enable_prefix_caching
    assert new_req.precomputed_block_hashes == compute_block_hashes_batched(
        new_req.prompt_tokens, new_req.block_size_tokens
    )
    assert new_req.precomputed_block_hashes is not original_hashes
    assert len(new_req.precomputed_block_hashes) == 2
    assert new_req.event_add_engine is original_event

    # A second checkpoint must keep the sticky configuration and extend the hash chain again.
    new_req.generated_tokens = [9, 10, 11, 12]
    previous_hashes = new_req.precomputed_block_hashes
    record.checkpoint()
    assert len(record.requests) == 3
    second_new_req = record.requests[-1]
    assert second_new_req.prompt_tokens.tolist() == list(range(1, 13))
    assert second_new_req.sampling_params.num_tokens_to_generate == 2
    assert second_new_req.block_size_tokens == 4
    assert second_new_req.enable_prefix_caching
    assert second_new_req.precomputed_block_hashes == compute_block_hashes_batched(
        second_new_req.prompt_tokens, second_new_req.block_size_tokens
    )
    assert second_new_req.precomputed_block_hashes is not previous_hashes
    assert len(second_new_req.precomputed_block_hashes) == 3
    assert second_new_req.event_add_engine is original_event

    # checkpoint() creates a new event_add_engine when the previous request had none.
    req2 = DynamicInferenceRequest(
        request_id=2,
        prompt_tokens=torch.tensor([1, 2, 3]),
        sampling_params=sp,
        generated_tokens=[10],
    )
    record2 = DynamicInferenceRequestRecord.from_request(req2)
    record2.checkpoint()
    assert record2.requests[-1].event_add_engine is not None

    # merge() concatenates tokens, text, and ndarray routing_indices; falls back to None.
    a = DynamicInferenceRequest(
        request_id=3,
        prompt_tokens=torch.tensor([1, 2, 3]),
        sampling_params=sp,
        generated_tokens=[10, 11],
    )
    b = DynamicInferenceRequest(
        request_id=3,
        prompt_tokens=torch.tensor([1, 2, 3]),
        sampling_params=sp,
        generated_tokens=[12],
    )
    a.generated_text = "foo"
    b.generated_text = "bar"
    a.routing_indices = np.array([[1, 2]])
    b.routing_indices = np.array([[3, 4]])
    b.policy_epoch = [(0, 3)]
    a.add_event_evict()
    rec = DynamicInferenceRequestRecord(requests=[a, b])
    rec.latency = 4.2
    merged = rec.merge()
    assert merged.generated_tokens == [10, 11, 12]
    assert merged.generated_text == "foobar"
    assert merged.generated_length == 3 and merged.latency == 4.2
    assert merged.routing_indices.tolist() == [[1, 2], [3, 4]]
    # Every request mints a distinct chatcmpl- uid; merge() keeps the FIRST
    # segment's — the id the response and the finished-request ledger key on.
    assert a.uid.startswith("chatcmpl-") and a.uid != b.uid
    assert merged.uid == a.uid
    # FinishedRequestRecord mirrors the merged stamps and counts EVICT events.
    finished = FinishedRequestRecord.from_request(merged)
    assert finished.policy_epoch == [(0, 3)] and finished.kv_cache_epoch is None
    assert finished.num_evictions == 1

    # merge() with both generated_text=None propagates None (rather than "None"+"None").
    c = DynamicInferenceRequest(
        request_id=4,
        prompt_tokens=torch.tensor([1, 2, 3]),
        sampling_params=sp,
        generated_tokens=[10],
    )
    d = DynamicInferenceRequest(
        request_id=4,
        prompt_tokens=torch.tensor([1, 2, 3]),
        sampling_params=sp,
        generated_tokens=[12],
    )
    merged_cd = DynamicInferenceRequestRecord(requests=[c, d]).merge()
    assert merged_cd.generated_text is None
    # Never-stamped requests record None epochs (non-RL serving).
    finished_cd = FinishedRequestRecord.from_request(merged_cd)
    assert finished_cd.policy_epoch is None and finished_cd.num_evictions == 0


def test_dynamic_inference_request_serialize_strips_event_add_engine():
    """DynamicInferenceRequest.serialize() omits `event_add_engine` (it's a
    pointer into `events`, not independent state); on deserialize we get the
    request back with its events list intact. Tested via a record round-trip
    because that's the real caller."""
    req = _make_dynamic_request()
    req.add_event_finish()
    data = req.serialize()
    assert "event_add_engine" not in data
    out = DynamicInferenceRequest.deserialize(unwrap_serialized_tensors(data))
    assert out.request_id == req.request_id
    assert len(out.events) == 1
    assert out.events[0].type == DynamicInferenceEventType.FINISH

    # Record-level serialize/deserialize preserves latency and request ids.
    rec = DynamicInferenceRequestRecord.from_request(_make_dynamic_request(request_id=7))
    rec.latency = 1.0
    rec_out = DynamicInferenceRequestRecord.deserialize(rec.serialize())
    assert rec_out.latency == 1.0
    assert rec_out.requests[0].request_id == 7


@pytest.mark.parametrize(
    ("return_prompt_tokens", "expected_prompt_field"),
    [
        (False, None),  # default: prompt_tokens dropped from payload
        (True, ("tensor", [1, 2, 3, 4])),  # opt-in: prompt_tokens preserved
    ],
)
def test_dynamic_inference_request_serialize_return_prompt_tokens(
    return_prompt_tokens, expected_prompt_field
):
    """DynamicInferenceRequest.serialize() reports prompt_length unconditionally
    (the API uses it for `usage.prompt_tokens` on the response) and drops the
    prompt_tokens tensor from the wire payload unless
    SamplingParams.return_prompt_tokens is True. This is the load-bearing
    wire-cost optimization for long agentic-RL prompts. The same call must
    (a) leave self.prompt_tokens intact on the local instance — the drop is
    wire-only — and (b) keep the routing_indices shape check honest, which
    now relies on the saved prompt_len rather than self.prompt_tokens (which
    is temporarily None during the drop)."""
    sp = SamplingParams(
        num_tokens_to_generate=5, termination_id=0, return_prompt_tokens=return_prompt_tokens
    )
    prompt = torch.tensor([1, 2, 3, 4])
    # prompt_len=4 + generated=[10] → total_tokens=5 → routing_indices.shape[0] must be 4.
    routing = np.zeros((4, 2, 1), dtype=np.int32)
    req = _make_dynamic_request(
        prompt_tokens=prompt, sampling_params=sp, generated_tokens=[10], routing_indices=routing
    )

    obj = req.serialize()

    # prompt_length is always populated (independent of the drop).
    assert obj["prompt_length"] == 4
    # Payload either preserves the tensor wrapper or drops it (present but None).
    assert obj["prompt_tokens"] == expected_prompt_field
    # Local instance is unaffected — the drop is wire-only.
    assert torch.equal(req.prompt_tokens, prompt)
    # routing_indices survives the drop path (shape check would have crashed on
    # the temporarily-None self.prompt_tokens if the fix used self.prompt_tokens).
    assert isinstance(obj["routing_indices"], tuple) and obj["routing_indices"][0] == "ndarray"


def test_dynamic_inference_request_serialize_prompt_length_absent():
    """When prompt_tokens is None on the request, serialize() must not crash
    (the drop path is guarded on `prompt_tokens is not None`) and prompt_length
    must be reported as None. The DP coordinator can dispatch error/finish
    records without prompt_tokens, so this path is real."""
    sp = SamplingParams(num_tokens_to_generate=1, termination_id=0)
    req = DynamicInferenceRequest(request_id=99, prompt_tokens=None, sampling_params=sp)

    obj = req.serialize()

    assert obj["prompt_length"] is None
    assert obj["prompt_tokens"] is None


def test_weight_scoped_salt_partitions_the_hash_space():
    """Block hashes from different weight generations must never match.

    Under PERSIST the prefix cache survives a refit, so without this a request
    admitted after new weights land can match KV the old weights computed.
    """
    from megatron.core.inference.engines.dynamic_engine import _weight_scoped_salt

    tokens = torch.arange(8, dtype=torch.int64)
    gen1 = compute_block_hashes_batched(
        tokens, block_size=4, cache_salt=_weight_scoped_salt(1, None)
    )
    gen2 = compute_block_hashes_batched(
        tokens, block_size=4, cache_salt=_weight_scoped_salt(2, None)
    )
    assert gen1 and gen2
    assert set(gen1).isdisjoint(gen2), "same tokens under different weights must not match"
    # Deterministic within a generation, or a request could not match itself.
    assert gen1 == compute_block_hashes_batched(
        tokens, block_size=4, cache_salt=_weight_scoped_salt(1, None)
    )


def test_weight_scoped_salt_is_inert_before_the_first_resume():
    """Epoch 0 hashes exactly as an unsalted engine did, media key and all."""
    from megatron.core.inference.engines.dynamic_engine import _weight_scoped_salt

    assert _weight_scoped_salt(0, None) is None
    assert _weight_scoped_salt(0, "img-1") == "img-1"

    tokens = torch.arange(8, dtype=torch.int64)
    assert compute_block_hashes_batched(
        tokens, block_size=4, cache_salt=_weight_scoped_salt(0, None)
    ) == compute_block_hashes_batched(tokens, block_size=4)


def test_weight_scoped_salt_keeps_media_identity_distinct():
    """Within one generation, different media must still not share KV."""
    from megatron.core.inference.engines.dynamic_engine import _weight_scoped_salt

    tokens = torch.arange(8, dtype=torch.int64)
    a = compute_block_hashes_batched(
        tokens, block_size=4, cache_salt=_weight_scoped_salt(3, "img-a")
    )
    b = compute_block_hashes_batched(
        tokens, block_size=4, cache_salt=_weight_scoped_salt(3, "img-b")
    )
    text = compute_block_hashes_batched(
        tokens, block_size=4, cache_salt=_weight_scoped_salt(3, None)
    )
    assert set(a).isdisjoint(b)
    assert set(a).isdisjoint(text)


def test_text_request_hashes_are_scoped_to_the_weight_generation():
    """A text-only request must not match blocks hashed under earlier weights.

    The salt is only worth anything if it reaches the path that carries the
    common case; testing the helper alone would pass with the engine never
    applying it.
    """
    from megatron.core.inference.engines.dynamic_engine import _weight_scoped_salt

    tokens = list(range(8))

    def hashes_at(epoch):
        request = DynamicInferenceRequest(
            request_id=0,
            prompt="",
            prompt_tokens=torch.tensor(tokens, dtype=torch.int64),
            block_size_tokens=4,
            enable_prefix_caching=True,
            block_hash_salt=_weight_scoped_salt(epoch, None),
        )
        return request.precomputed_block_hashes

    before, after = hashes_at(1), hashes_at(2)
    assert before and after
    assert set(before).isdisjoint(after), "a refit must make earlier blocks unmatchable"
    assert hashes_at(1) == before, "hashes must be stable within one generation"


def test_supplied_block_hashes_are_not_re_salted():
    """Hashes handed in by a caller are used as-is.

    A disaggregated handoff carries hashes its sender already computed; the
    receiver must adopt them rather than recompute under its own generation, or
    the imported KV would be unreachable.
    """
    request = DynamicInferenceRequest(
        request_id=0,
        prompt="",
        prompt_tokens=torch.tensor(list(range(8)), dtype=torch.int64),
        block_size_tokens=4,
        enable_prefix_caching=True,
        precomputed_block_hashes=[11, 22],
        block_hash_salt="w9",
    )
    assert request.precomputed_block_hashes == [11, 22]
