# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import warnings

import msgpack
import numpy as np
import torch

from megatron.core.inference.inference_request import (
    DynamicInferenceEventType,
    DynamicInferenceRequest,
    DynamicInferenceRequestRecord,
    InferenceRequest,
    compute_block_hashes_batched,
    deserialize_ndarray,
    deserialize_tensor,
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
    assert compute_block_hashes_batched(torch.arange(8, dtype=torch.int64), block_size=4)[1] != h_b[1]


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
    add_engine event is inherited (or created) so downstream tooling can find
    it. RequestRecord.merge() collapses the chain back into a single request
    with concatenated tokens, text, routing_indices, and the record's latency.
    Both are non-trivial state machines."""
    sp = SamplingParams(num_tokens_to_generate=5, termination_id=0)

    # checkpoint() inherits event_add_engine when present.
    req = DynamicInferenceRequest(
        request_id=1, prompt_tokens=torch.tensor([1, 2, 3]), sampling_params=sp,
        generated_tokens=[10, 11],
    )
    original_event = req.add_event_add_engine()
    record = DynamicInferenceRequestRecord.from_request(req)
    record.checkpoint()
    assert len(record.requests) == 2
    new_req = record.requests[-1]
    assert new_req.prompt_tokens.tolist() == [1, 2, 3, 10, 11]
    assert new_req.sampling_params.num_tokens_to_generate == 3
    assert new_req.event_add_engine is original_event

    # checkpoint() creates a new event_add_engine when the previous request had none.
    req2 = DynamicInferenceRequest(
        request_id=2, prompt_tokens=torch.tensor([1, 2, 3]), sampling_params=sp,
        generated_tokens=[10],
    )
    record2 = DynamicInferenceRequestRecord.from_request(req2)
    record2.checkpoint()
    assert record2.requests[-1].event_add_engine is not None

    # merge() concatenates tokens, text, and ndarray routing_indices; falls back to None.
    a = DynamicInferenceRequest(
        request_id=3, prompt_tokens=torch.tensor([1, 2, 3]), sampling_params=sp,
        generated_tokens=[10, 11],
    )
    b = DynamicInferenceRequest(
        request_id=3, prompt_tokens=torch.tensor([1, 2, 3]), sampling_params=sp,
        generated_tokens=[12],
    )
    a.generated_text = "foo"; b.generated_text = "bar"
    a.routing_indices = np.array([[1, 2]]); b.routing_indices = np.array([[3, 4]])
    rec = DynamicInferenceRequestRecord(requests=[a, b])
    rec.latency = 4.2
    merged = rec.merge()
    assert merged.generated_tokens == [10, 11, 12]
    assert merged.generated_text == "foobar"
    assert merged.generated_length == 3 and merged.latency == 4.2
    assert merged.routing_indices.tolist() == [[1, 2], [3, 4]]

    # merge() with both generated_text=None propagates None (rather than "None"+"None").
    c = DynamicInferenceRequest(
        request_id=4, prompt_tokens=torch.tensor([1, 2, 3]), sampling_params=sp,
        generated_tokens=[10],
    )
    d = DynamicInferenceRequest(
        request_id=4, prompt_tokens=torch.tensor([1, 2, 3]), sampling_params=sp,
        generated_tokens=[12],
    )
    assert DynamicInferenceRequestRecord(requests=[c, d]).merge().generated_text is None


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
