# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import warnings

import numpy as np
import pytest
import torch

from megatron.core.inference.inference_request import (
    DynamicInferenceEvent,
    DynamicInferenceEventType,
    DynamicInferenceRequest,
    DynamicInferenceRequestRecord,
    InferenceRequest,
    Status,
    VLMInferenceRequest,
    compute_block_hashes_batched,
    deserialize_ndarray,
    deserialize_tensor,
    serialize_ndarray,
    serialize_tensor,
    unwrap_serialized_tensors,
)
from megatron.core.inference.sampling_params import SamplingParams


class TestSerializationHelpers:

    def test_serialize_tensor_returns_list(self):
        """serialize_tensor converts a tensor to a Python list."""
        t = torch.tensor([1, 2, 3])
        data = serialize_tensor(t)
        assert data == [1, 2, 3]
        assert isinstance(data, list)

    def test_deserialize_tensor_round_trip(self):
        """deserialize_tensor inverts serialize_tensor."""
        t = torch.tensor([4, 5, 6, 7])
        out = deserialize_tensor(serialize_tensor(t))
        assert isinstance(out, torch.Tensor)
        assert out.tolist() == [4, 5, 6, 7]

    def test_serialize_ndarray_records_dtype_and_data(self):
        """serialize_ndarray returns a dict capturing data and dtype."""
        arr = np.array([1, 2, 3], dtype=np.int32)
        out = serialize_ndarray(arr)
        assert out["data"] == [1, 2, 3]
        assert out["dtype"] == "int32"

    def test_deserialize_ndarray_round_trip(self):
        """deserialize_ndarray reconstructs an ndarray with the original dtype."""
        arr = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
        out = deserialize_ndarray(serialize_ndarray(arr))
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float64
        assert np.array_equal(out, arr)

    def test_unwrap_serialized_tensors_strips_tensor_wrapper(self):
        """unwrap_serialized_tensors replaces ('tensor', [...]) with the inner list."""
        obj = {
            "a": ("tensor", [1, 2, 3]),
            "b": "plain",
            "c": ("ndarray", {"data": [], "dtype": "int32"}),
        }
        out = unwrap_serialized_tensors(obj)
        assert out["a"] == [1, 2, 3]
        assert out["b"] == "plain"
        # Non-tensor wrappers are passed through untouched.
        assert out["c"] == ("ndarray", {"data": [], "dtype": "int32"})


class TestStatusEnum:

    def test_status_values_distinct(self):
        """All Status enum values are distinct integers."""
        values = [s.value for s in Status]
        assert len(values) == len(set(values))

    def test_status_lookup_by_name(self):
        """Status[name] returns the corresponding member."""
        assert Status["COMPLETED"] is Status.COMPLETED
        assert Status["FAILED"] is Status.FAILED


class TestComputeBlockHashes:

    def test_empty_when_prompt_shorter_than_block(self):
        """Returns empty list when prompt has no complete blocks."""
        prompt = torch.arange(3, dtype=torch.int64)
        assert compute_block_hashes_batched(prompt, block_size=4) == []

    def test_single_block(self):
        """Single block hashes to one positive int64 value."""
        prompt = torch.arange(4, dtype=torch.int64)
        hashes = compute_block_hashes_batched(prompt, block_size=4)
        assert len(hashes) == 1
        assert 1 <= hashes[0] < 2**63

    def test_multiple_blocks_chain(self):
        """Each complete block produces one hash; partial trailing tokens are ignored."""
        prompt = torch.arange(10, dtype=torch.int64)  # 2 blocks of 4 + 2 leftover
        hashes = compute_block_hashes_batched(prompt, block_size=4)
        assert len(hashes) == 2
        # Different content per block -> different hashes (overwhelmingly likely).
        assert hashes[0] != hashes[1]

    def test_deterministic(self):
        """Same input yields identical hashes across calls."""
        prompt = torch.arange(8, dtype=torch.int64)
        h1 = compute_block_hashes_batched(prompt, block_size=4)
        h2 = compute_block_hashes_batched(prompt, block_size=4)
        assert h1 == h2

    def test_different_prefix_changes_subsequent_hashes(self):
        """Changing the first block changes the chained hashes for later blocks."""
        prompt_a = torch.arange(8, dtype=torch.int64)
        prompt_b = prompt_a.clone()
        prompt_b[0] = 99  # mutate first block
        h_a = compute_block_hashes_batched(prompt_a, block_size=4)
        h_b = compute_block_hashes_batched(prompt_b, block_size=4)
        assert h_a[1] != h_b[1]


class TestInferenceRequest:

    def test_minimal_construction(self):
        """A request can be built with just request_id and prompt."""
        req = InferenceRequest(request_id=1, prompt="hello")
        assert req.request_id == 1
        assert req.prompt == "hello"
        assert req.sampling_params is None

    def test_inference_parameters_alias_warns(self):
        """Passing inference_parameters but no sampling_params emits a deprecation warning and copies the value."""
        sp = SamplingParams(temperature=0.5)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            req = InferenceRequest(request_id=1, prompt="hi", inference_parameters=sp)
            assert any("renamed" in str(item.message) for item in w)
        assert req.sampling_params is sp

    def test_serialize_round_trip_preserves_fields(self):
        """serialize/deserialize preserves dataclass fields."""
        sp = SamplingParams(temperature=0.7, top_k=3)
        req = InferenceRequest(
            request_id=10,
            prompt="hello",
            sampling_params=sp,
            status=Status.WAITING_IN_QUEUE,
            arrival_time=1.5,
        )
        data = req.serialize()
        # Tensor wrappers must be unwrapped before passing to deserialize.
        data = unwrap_serialized_tensors(data)
        out = InferenceRequest.deserialize(data)
        assert out.request_id == 10
        assert out.prompt == "hello"
        assert out.status == Status.WAITING_IN_QUEUE
        assert out.arrival_time == 1.5
        assert out.sampling_params.temperature == 0.7
        assert out.sampling_params.top_k == 3

    def test_post_deserialize_handles_wrapped_tensor_field(self):
        """_post_deserialize converts ('tensor', list) wrappers to torch.Tensors after msgpack."""
        import msgpack

        req = InferenceRequest(request_id=20, prompt="x", generated_tokens=torch.tensor([7, 8, 9]))
        # msgpack converts the ('tensor', [...]) tuple into a list, which is what
        # _post_deserialize matches against.
        data = msgpack.unpackb(msgpack.packb(req.serialize()), raw=False)
        out = InferenceRequest.deserialize(data)
        assert isinstance(out.generated_tokens, torch.Tensor)
        assert out.generated_tokens.tolist() == [7, 8, 9]

    def test_post_deserialize_handles_wrapped_ndarray_field(self):
        """_post_deserialize converts ('ndarray', dict) wrappers to numpy arrays after msgpack."""
        import msgpack

        # Place the ndarray on a real dataclass field. routing_indices on
        # DynamicInferenceRequest accepts ndarrays — but InferenceRequest itself
        # has no ndarray field, so we instead prove the round-trip via
        # DynamicInferenceRequest.
        sp = SamplingParams(num_tokens_to_generate=1, termination_id=0)
        req = DynamicInferenceRequest(
            request_id=22,
            prompt_tokens=torch.tensor([1, 2]),
            generated_tokens=[10],
            sampling_params=sp,
            routing_indices=np.array([[1, 2], [3, 4]], dtype=np.int32),
        )
        data = msgpack.unpackb(msgpack.packb(req.serialize()), raw=False)
        out = DynamicInferenceRequest.deserialize(data)
        assert isinstance(out.routing_indices, np.ndarray)
        assert out.routing_indices.tolist() == [[1, 2], [3, 4]]

    def test_serialize_status_none_passes_through(self):
        """serialize handles status=None without crashing."""
        req = InferenceRequest(request_id=11, prompt="x", status=None)
        data = req.serialize()
        assert data["status"] is None

    def test_serialize_tensor_field_wraps_in_tuple(self):
        """Tensor fields are stored as ('tensor', list) sentinels."""
        req = InferenceRequest(request_id=12, prompt="x", generated_tokens=torch.tensor([1, 2, 3]))
        data = req.serialize()
        assert data["generated_tokens"][0] == "tensor"
        assert data["generated_tokens"][1] == [1, 2, 3]

    def test_serialize_ndarray_field_wraps_in_tuple(self):
        """Numpy array fields are stored as ('ndarray', {data, dtype}) sentinels."""
        # InferenceRequest doesn't have an ndarray field but the serializer is
        # generic; we add one via __dict__ to exercise the branch.
        req = InferenceRequest(request_id=12, prompt="x")
        req.custom_arr = np.array([1, 2], dtype=np.int32)
        data = req.serialize()
        assert data["custom_arr"][0] == "ndarray"
        assert data["custom_arr"][1]["data"] == [1, 2]


class TestDynamicInferenceEvent:

    def test_construction_sets_default_timestamp(self):
        """Timestamp defaults to a positive float when not provided."""
        ev = DynamicInferenceEvent(type=DynamicInferenceEventType.PAUSE)
        assert ev.timestamp is not None
        assert ev.timestamp > 0

    def test_explicit_timestamp_is_preserved(self):
        """Explicit timestamp is kept verbatim."""
        ev = DynamicInferenceEvent(timestamp=123.45, type=DynamicInferenceEventType.PAUSE)
        assert ev.timestamp == 123.45

    def test_invalid_type_asserts(self):
        """Non-enum event type raises AssertionError."""
        with pytest.raises(AssertionError):
            DynamicInferenceEvent(type="not-an-enum")

    def test_generated_token_requires_token_id_payload(self):
        """GENERATED_TOKEN event must carry a payload with token_id."""
        with pytest.raises(AssertionError):
            DynamicInferenceEvent(
                type=DynamicInferenceEventType.GENERATED_TOKEN, payload={"other": 1}
            )
        ev = DynamicInferenceEvent(
            type=DynamicInferenceEventType.GENERATED_TOKEN, payload={"token_id": 7}
        )
        assert ev.payload["token_id"] == 7

    def test_error_event_requires_payload(self):
        """ERROR_TRANSIENT and ERROR_NONTRANSIENT events require non-None payloads."""
        with pytest.raises(AssertionError):
            DynamicInferenceEvent(type=DynamicInferenceEventType.ERROR_TRANSIENT)
        with pytest.raises(AssertionError):
            DynamicInferenceEvent(type=DynamicInferenceEventType.ERROR_NONTRANSIENT)

    def test_other_events_disallow_payload(self):
        """PAUSE / FINISH events must have payload=None."""
        with pytest.raises(AssertionError):
            DynamicInferenceEvent(type=DynamicInferenceEventType.PAUSE, payload="oops")

    def test_str_format_for_generated_token(self):
        """__str__ formats GENERATED_TOKEN events with token id."""
        ev = DynamicInferenceEvent(
            timestamp=1.0, type=DynamicInferenceEventType.GENERATED_TOKEN, payload={"token_id": 11}
        )
        assert "token=11" in str(ev)
        assert "GENERATED_TOKEN" in str(ev)

    def test_str_format_for_payload_none(self):
        """__str__ omits payload section when payload is None."""
        ev = DynamicInferenceEvent(timestamp=2.5, type=DynamicInferenceEventType.PAUSE)
        s = str(ev)
        assert "PAUSE" in s
        # Two consecutive sections separated by a comma exist only when payload is shown.
        assert "{" not in s

    def test_str_format_for_other_payload_uses_type_name(self):
        """__str__ shows payload type name for non-token, non-None payloads."""
        ev = DynamicInferenceEvent(
            timestamp=3.0, type=DynamicInferenceEventType.ERROR_TRANSIENT, payload=ValueError("x")
        )
        s = str(ev)
        assert "ValueError" in s


class TestDynamicInferenceRequest:

    def _make(self, **kwargs):
        defaults = dict(
            request_id=1,
            prompt_tokens=torch.tensor([1, 2, 3, 4]),
            sampling_params=SamplingParams(num_tokens_to_generate=5, termination_id=0),
        )
        defaults.update(kwargs)
        return DynamicInferenceRequest(**defaults)

    def test_remaining_prompt_tokens_initialized_to_prompt(self):
        """After init, remaining_prompt_tokens equals prompt_tokens."""
        req = self._make()
        assert torch.equal(req.remaining_prompt_tokens, req.prompt_tokens)
        assert req.remaining_prompt_length == 4

    def test_prefix_caching_computes_hashes(self):
        """When enabled, block hashes are computed during __post_init__."""
        req = self._make(
            prompt_tokens=torch.arange(8, dtype=torch.int64),
            block_size_tokens=4,
            enable_prefix_caching=True,
        )
        assert len(req.precomputed_block_hashes) == 2

    def test_prefix_caching_skipped_without_block_size(self):
        """No hashes are computed when block_size_tokens is None."""
        req = self._make(enable_prefix_caching=True, block_size_tokens=None)
        assert req.precomputed_block_hashes == []

    def test_prefix_caching_keeps_provided_hashes(self):
        """Pre-supplied precomputed_block_hashes are not overwritten."""
        provided = [42]
        req = self._make(
            prompt_tokens=torch.arange(8, dtype=torch.int64),
            block_size_tokens=4,
            enable_prefix_caching=True,
            precomputed_block_hashes=provided,
        )
        assert req.precomputed_block_hashes == provided

    def test_str_includes_id_and_status(self):
        """__str__ includes the request id, status, and length info."""
        req = self._make()
        req.status = Status.COMPLETED
        s = str(req)
        assert "id 1" in s
        assert "COMPLETED" in s
        assert "prompt len 4" in s

    def test_str_handles_unset_status(self):
        """When status is None, __str__ shows '[NOT ADDED]'."""
        req = self._make()
        s = str(req)
        assert "[NOT ADDED]" in s

    def test_add_event_appends_to_events(self):
        """add_event appends a new DynamicInferenceEvent to the events list."""
        req = self._make()
        ev = req.add_event(DynamicInferenceEventType.PAUSE)
        assert ev is req.events[-1]
        assert ev.type == DynamicInferenceEventType.PAUSE

    def test_event_helpers_use_correct_types(self):
        """The convenience helpers create events of the matching type."""
        req = self._make()
        types = []
        types.append(req.add_event_add_engine().type)
        types.append(req.add_event_add_context().type)
        types.append(req.add_event_pause().type)
        types.append(req.add_event_evict().type)
        types.append(req.add_event_finish().type)
        types.append(req.add_event_fail().type)
        types.append(req.add_event_error_transient(ValueError("t")).type)
        types.append(req.add_event_error_nontransient(RuntimeError("nt")).type)
        assert types == [
            DynamicInferenceEventType.ADD_ENGINE,
            DynamicInferenceEventType.ADD_CONTEXT,
            DynamicInferenceEventType.PAUSE,
            DynamicInferenceEventType.EVICT,
            DynamicInferenceEventType.FINISH,
            DynamicInferenceEventType.FAIL,
            DynamicInferenceEventType.ERROR_TRANSIENT,
            DynamicInferenceEventType.ERROR_NONTRANSIENT,
        ]

    def test_add_event_add_engine_sets_attribute(self):
        """add_event_add_engine stores the resulting event on event_add_engine."""
        req = self._make()
        ev = req.add_event_add_engine()
        assert req.event_add_engine is ev

    def test_add_event_generated_token_with_metadata(self):
        """add_event_generated_token stores all provided metadata in payload."""
        req = self._make()
        ev = req.add_event_generated_token(
            token=5,
            blocks_total=10,
            blocks_hashed_total=8,
            blocks_hashed_active=4,
            blocks_ref_count=12,
            pre_fwd_active_token_count=3,
            pre_fwd_step_count=2,
        )
        p = ev.payload
        assert p["token_id"] == 5
        assert p["blocks_total"] == 10
        assert p["blocks_hashed_total"] == 8
        assert p["blocks_hashed_active"] == 4
        assert p["blocks_ref_count"] == 12
        assert p["pre_fwd_active_token_count"] == 3
        assert p["pre_fwd_step_count"] == 2

    def test_add_event_generated_token_minimal(self):
        """The generated-token helper omits optional metadata if not provided."""
        req = self._make()
        ev = req.add_event_generated_token(token=5)
        assert ev.payload == {"token_id": 5}

    def test_succeeded_failed_status_predicates(self):
        """succeeded() and failed() report on the request status."""
        req = self._make()
        req.status = Status.COMPLETED
        assert req.succeeded() is True
        assert req.failed() is False
        req.status = Status.FAILED
        assert req.succeeded() is False
        assert req.failed() is True

    def test_get_metadata_types_lists_required_fields(self):
        """get_metadata_types yields the canonical sampling-param schema."""
        names = [n for n, _ in DynamicInferenceRequest.get_metadata_types()]
        for required in ["temperature", "top_k", "top_p", "termination_id"]:
            assert required in names

    def test_tracked_metadata_returns_values_in_schema_order(self):
        """tracked_metadata yields one value per metadata field, in the schema order."""
        sp = SamplingParams(temperature=0.7, top_k=3, top_p=0.5, termination_id=99)
        req = self._make(sampling_params=sp)
        values = req.tracked_metadata
        names = [n for n, _ in DynamicInferenceRequest.get_metadata_types()]
        assert len(values) == len(names)
        # Check first three slots line up with the schema we asserted in the previous test.
        assert values[names.index("temperature")] == 0.7
        assert values[names.index("top_k")] == 3
        assert values[names.index("top_p")] == 0.5
        assert values[names.index("termination_id")] == 99

    def test_tracked_metadata_defaults_termination_id(self):
        """tracked_metadata mutates termination_id from None to -1."""
        sp = SamplingParams(termination_id=None)
        req = self._make(sampling_params=sp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ = req.tracked_metadata
        assert req.sampling_params.termination_id == -1

    def test_serialize_round_trip(self):
        """serialize() and deserialize() round-trip a basic dynamic request."""
        sp = SamplingParams(termination_id=0, num_tokens_to_generate=5)
        req = self._make(sampling_params=sp, prompt="hi")
        req.add_event_finish()
        data = req.serialize()
        # event_add_engine should be stripped from the serialized dict.
        assert "event_add_engine" not in data
        data = unwrap_serialized_tensors(data)
        out = DynamicInferenceRequest.deserialize(data)
        assert out.request_id == req.request_id
        assert len(out.events) == 1
        assert out.events[0].type == DynamicInferenceEventType.FINISH


class TestDynamicInferenceRequestRecord:

    def _make_request(self, request_id=1, generated_tokens=None):
        return DynamicInferenceRequest(
            request_id=request_id,
            prompt_tokens=torch.tensor([1, 2, 3]),
            sampling_params=SamplingParams(num_tokens_to_generate=5, termination_id=0),
            generated_tokens=list(generated_tokens or []),
        )

    def test_from_request_initializes_with_one_request(self):
        """from_request returns a record holding a single request."""
        req = self._make_request()
        record = DynamicInferenceRequestRecord.from_request(req)
        assert record.requests == [req]

    def test_indexing_returns_underlying_requests(self):
        """__getitem__ reflects the order of requests appended to the record."""
        req1 = self._make_request(request_id=1)
        req2 = self._make_request(request_id=2)
        record = DynamicInferenceRequestRecord(requests=[req1, req2])
        assert record[0] is req1
        assert record[1] is req2
        assert record[-1] is req2

    def test_request_id_is_first_request_id(self):
        """request_id property returns the id of requests[0]."""
        req = self._make_request(request_id=99)
        record = DynamicInferenceRequestRecord.from_request(req)
        assert record.request_id == 99

    def test_checkpoint_appends_new_request(self):
        """checkpoint() concatenates prompt+generated and appends a new request."""
        req = self._make_request(generated_tokens=[10, 11])
        req.add_event_add_engine()
        record = DynamicInferenceRequestRecord.from_request(req)
        record.checkpoint()
        assert len(record.requests) == 2
        new_req = record[-1]
        # New prompt = old prompt + generated tokens.
        assert new_req.prompt_tokens.tolist() == [1, 2, 3, 10, 11]
        # num_tokens_to_generate is reduced by the number of generated tokens.
        assert new_req.sampling_params.num_tokens_to_generate == 5 - 2

    def test_checkpoint_preserves_event_add_engine(self):
        """The new checkpoint inherits event_add_engine from the previous request."""
        req = self._make_request(generated_tokens=[10])
        req.add_event_add_engine()
        original_event = req.event_add_engine
        record = DynamicInferenceRequestRecord.from_request(req)
        record.checkpoint()
        assert record[-1].event_add_engine is original_event

    def test_checkpoint_creates_event_add_engine_if_missing(self):
        """When the previous request has no add_engine event, the new one creates its own."""
        req = self._make_request(generated_tokens=[10])
        # No add_event_add_engine call.
        record = DynamicInferenceRequestRecord.from_request(req)
        record.checkpoint()
        assert record[-1].event_add_engine is not None

    def test_merge_returns_single_request(self):
        """merge collapses a record into one request with concatenated lists."""
        req1 = self._make_request(generated_tokens=[10, 11])
        req2 = self._make_request(generated_tokens=[12])
        req1.generated_text = "foo"
        req2.generated_text = "bar"
        record = DynamicInferenceRequestRecord(requests=[req1, req2])
        record.latency = 4.2
        merged = record.merge()
        assert merged.generated_tokens == [10, 11, 12]
        assert merged.generated_text == "foobar"
        assert merged.generated_length == 3
        assert merged.latency == 4.2

    def test_merge_handles_none_generated_text(self):
        """merge falls back to generated_text=None when any constituent is None."""
        req1 = self._make_request(generated_tokens=[10])
        req2 = self._make_request(generated_tokens=[12])
        # Both have generated_text=None (default).
        record = DynamicInferenceRequestRecord(requests=[req1, req2])
        merged = record.merge()
        assert merged.generated_text is None

    def test_merge_concatenates_routing_indices(self):
        """merge concatenates ndarray routing_indices from each request."""
        req1 = self._make_request(generated_tokens=[10])
        req2 = self._make_request(generated_tokens=[11])
        req1.routing_indices = np.array([[1, 2]])
        req2.routing_indices = np.array([[3, 4]])
        record = DynamicInferenceRequestRecord(requests=[req1, req2])
        merged = record.merge()
        assert merged.routing_indices.tolist() == [[1, 2], [3, 4]]

    def test_serialize_round_trip(self):
        """serialize/deserialize round-trips a record with a single request."""
        req = self._make_request(generated_tokens=[10, 11])
        record = DynamicInferenceRequestRecord.from_request(req)
        record.latency = 1.0
        data = record.serialize()
        out = DynamicInferenceRequestRecord.deserialize(data)
        assert out.latency == 1.0
        assert out.requests[0].request_id == 1


class TestVLMInferenceRequest:

    def test_constructs_with_required_fields(self):
        """VLMInferenceRequest constructs with image-specific fields."""
        imgs = torch.zeros(1, 3, 8, 8)
        num_tiles = torch.tensor([1])
        req = VLMInferenceRequest(
            request_id=1,
            prompt="describe",
            num_img_embeddings_per_tile=4,
            imgs=imgs,
            num_tiles=num_tiles,
            decoder_seq_length=16,
        )
        assert req.num_img_embeddings_per_tile == 4
        assert req.decoder_seq_length == 16
        assert torch.equal(req.imgs, imgs)
        assert torch.equal(req.num_tiles, num_tiles)
