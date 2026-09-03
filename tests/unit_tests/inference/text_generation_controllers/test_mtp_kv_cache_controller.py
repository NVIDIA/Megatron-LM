# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for the controller half of the MTP KV cache (`MTPInferenceMixin`).

Two things are covered:

  * `_mtp_commit_pass` -- the per-step "first pass" that refreshes every COMMITTED position's
    draft KV from the MAIN model hidden states, so committed KV never carries a stale
    chained-draft-hidden value. It packs one varlen roll-by-one forward covering both decode
    requests (their accepted-draft positions) and prefill requests (their chunk positions).
  * `_compute_serial_mtp_and_sample` -- the draft loop that consumes it: which requests draft,
    which CUDA-graph key is replayed, and the extra D+1th append.

Both are pure tensor assembly around one model call, so they are driven against a fake context
and a fake model that record what they were handed. The assertions target the arithmetic that
is easy to get wrong and expensive to debug on real hardware: the per-request write START
positions and append COUNTS, which differ between a fresh prompt, a continuation chunk, and a
prefix-cache hit.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.transformer.enums import InferenceCudaGraphScope

DEVICE = "cuda"
HIDDEN_SIZE = 8
MAX_REQUESTS = 8
MAX_KV_BLOCK_COUNT = 4
VOCAB_SIZE = 16


def _make_context(
    num_decode_requests: int = 0,
    prefill_query_lengths=(),
    prefill_kv_offsets=(),
    request_ids=None,
    chunked_prefill_request_id: int = -1,
    paused_request_count: int = 0,
    num_speculative_tokens: int = 2,
    max_tokens: int = 64,
):
    """A fake `DynamicInferenceContext` exposing only what `_mtp_commit_pass` reads.

    `_mtp_setup_prefill_step` records its kwargs instead of touching GPU metadata; the real
    bookkeeping it performs is covered separately in
    `tests/unit_tests/inference/contexts/test_dynamic_context_mtp_kv_cache.py`.
    """
    num_prefill = len(prefill_query_lengths)
    num_active = num_decode_requests + num_prefill
    total = paused_request_count + num_active

    # Per-request CPU state, in full max_requests-sized buffers so the active-slice indexing in
    # the implementation is exercised rather than bypassed.
    request_query_lengths = torch.zeros(MAX_REQUESTS, dtype=torch.int32)
    request_kv_length_offsets = torch.zeros(MAX_REQUESTS, dtype=torch.int32)
    ids = torch.full((MAX_REQUESTS,), -1, dtype=torch.int32)
    if request_ids is None:
        request_ids = list(range(100, 100 + num_active))
    for i in range(num_active):
        ids[paused_request_count + i] = request_ids[i]
    for i, (q, off) in enumerate(zip(prefill_query_lengths, prefill_kv_offsets)):
        row = paused_request_count + num_decode_requests + i
        request_query_lengths[row] = q
        request_kv_length_offsets[row] = off

    # Distinct block ids per request so a mis-sliced block table shows up as wrong values.
    request_to_kv_block_ids = torch.full((MAX_REQUESTS, MAX_KV_BLOCK_COUNT), -1, dtype=torch.int32)
    for r in range(MAX_REQUESTS):
        request_to_kv_block_ids[r] = torch.arange(
            r * MAX_KV_BLOCK_COUNT, (r + 1) * MAX_KV_BLOCK_COUNT, dtype=torch.int32
        )

    gpu_view = SimpleNamespace(
        # Non-trivial token ids so a roll-by-one off-by-one is visible in the assertions.
        token_to_input_ids=torch.arange(1000, 1000 + max_tokens, dtype=torch.int64, device=DEVICE),
        mha_block_table=torch.zeros(
            (MAX_REQUESTS, MAX_KV_BLOCK_COUNT), dtype=torch.int32, device=DEVICE
        ),
    )

    context = SimpleNamespace(
        enable_mtp_kv_cache=True,
        paused_request_count=paused_request_count,
        total_request_count=total,
        num_prefill_requests=num_prefill,
        chunked_prefill_request_id=chunked_prefill_request_id,
        num_speculative_tokens=num_speculative_tokens,
        request_query_lengths=request_query_lengths,
        request_kv_length_offsets=request_kv_length_offsets,
        request_ids=ids,
        request_to_kv_block_ids=request_to_kv_block_ids,
        gpu_view=gpu_view,
        _nvls_dispatcher=None,
        inference_cuda_graph_scope=InferenceCudaGraphScope.none,
        mtp_decoder_hidden_states=None,
        setup_prefill_calls=[],
        finalize_prefill_calls=0,
    )

    def _setup_prefill_step(**kwargs):
        context.setup_prefill_calls.append(kwargs)

    def _finalize_prefill_step():
        context.finalize_prefill_calls += 1

    context._mtp_setup_prefill_step = _setup_prefill_step
    context._mtp_finalize_prefill_step = _finalize_prefill_step
    context.using_cuda_graph_this_step = lambda: False
    return context


def _make_model():
    """A fake unwrapped model recording every MTP-layer call.

    `all_forwards` is the EP-relevant log: one entry per forward that drives the MTP layer's
    MoE all-to-all, tagged by which path issued it. Real and idle ranks must produce the same
    number of entries, in the same graph/eager mode, or the collective mismatches.
    """
    calls = []
    all_forwards = []

    def forward_single_position(**kwargs):
        calls.append(kwargs)
        all_forwards.append(("mtp_layer", kwargs))
        return None

    model = SimpleNamespace(
        mtp=SimpleNamespace(
            layers=[SimpleNamespace(forward_single_position=forward_single_position)],
            mtp_use_repeated_layer=True,
        ),
        embedding=object(),
        mtp_layer_calls=calls,
        all_forwards=all_forwards,
    )
    return model


def _make_controller(context, model, sp_enabled: bool = False, tp_size: int = 1):
    controller = TextGenerationController.__new__(TextGenerationController)
    controller.num_speculative_tokens = context.num_speculative_tokens
    controller.num_mtp_depths = context.num_speculative_tokens
    controller.vocab_size = VOCAB_SIZE
    controller.model_config = SimpleNamespace(
        params_dtype=torch.float32,
        hidden_size=HIDDEN_SIZE,
        expert_model_parallel_size=1,
        moe_pad_experts_for_cuda_graph_inference=False,
    )
    controller.inference_wrapped_model = SimpleNamespace(
        inference_context=context, tp_group=None, model=model
    )
    controller._unwrapped_model = model
    controller._sp_enabled = sp_enabled
    controller._tp_size = tp_size
    controller._is_last_pp_stage = True
    controller.model_is_pipeline_parallel = False
    controller._mtp_chunk_boundary_hidden = None
    controller._mtp_chunk_boundary_req_id = -1
    controller._mtp_resolved_padded_count = None
    controller._accepted_token_counts_per_request = torch.zeros(
        MAX_REQUESTS, dtype=torch.int64, device=DEVICE
    )
    return controller


def _hidden(num_tokens: int, fill=None):
    """Packed hidden states shaped [tokens, 1, H]; each row carries its own index as its value."""
    if fill is not None:
        return torch.full((num_tokens, 1, HIDDEN_SIZE), float(fill), device=DEVICE)
    return (
        torch.arange(num_tokens, dtype=torch.float32, device=DEVICE)
        .view(-1, 1, 1)
        .expand(-1, 1, HIDDEN_SIZE)
        .contiguous()
    )


def _row_ids(hidden_states):
    """Recover the per-row index encoded by `_hidden` (identifies which hidden was packed)."""
    return hidden_states[:, 0, 0].cpu().tolist()


class TestMtpCommitPassDecode:
    """Decode requests: refresh the positions whose drafts were accepted this step."""

    def test_refreshes_accepted_draft_positions(self):
        """Request r rewrites its a_r accepted positions starting at `base - 1 - a_r`."""
        context = _make_context(num_decode_requests=2)
        model = _make_model()
        controller = _make_controller(context, model)
        controller._accepted_token_counts_per_request[:2] = torch.tensor(
            [2, 1], device=DEVICE, dtype=torch.int64
        )
        base_position = torch.tensor([10, 7], device=DEVICE)
        stride = context.num_speculative_tokens + 1  # 3 forwarded tokens per decode request

        issued = controller._mtp_commit_pass(
            context,
            model,
            _hidden(2 * stride),
            num_decode_requests=2,
            active_request_count=2,
            base_position=base_position,
        )

        assert issued is True
        call = context.setup_prefill_calls[-1]
        assert call["append_counts"].cpu().tolist() == [2, 1]
        # The LAST accepted position is written by decode depth 0, so the commit pass covers the
        # a_r EARLIER positions: start = base - 1 - a_r.
        assert call["request_start_positions"].cpu().tolist() == [7, 5]

        # Hiddens are the main hiddens for the accepted forwarded tokens: request 0 contributes
        # rows 0,1 of its stride-3 slot; request 1 contributes row 0 of its slot (index 3).
        forward = model.mtp_layer_calls[-1]
        assert _row_ids(forward["hidden_states"]) == [0, 1, 3]
        # Tokens are rolled by one: entry at position p consumes t_{p+1}.
        token_ids = forward["next_token_ids"].view(-1).cpu().tolist()
        assert token_ids == [1001, 1002, 1004]
        assert context.finalize_prefill_calls == 1

    def test_request_with_no_accepted_drafts_contributes_nothing(self):
        """A rejected-everything request still occupies a row, with count 0."""
        context = _make_context(num_decode_requests=2)
        model = _make_model()
        controller = _make_controller(context, model)
        controller._accepted_token_counts_per_request[:2] = torch.tensor(
            [0, 2], device=DEVICE, dtype=torch.int64
        )

        issued = controller._mtp_commit_pass(
            context,
            model,
            _hidden(2 * 3),
            num_decode_requests=2,
            active_request_count=2,
            base_position=torch.tensor([9, 12], device=DEVICE),
        )

        assert issued is True
        call = context.setup_prefill_calls[-1]
        assert call["append_counts"].cpu().tolist() == [0, 2]
        assert call["request_start_positions"].cpu().tolist() == [8, 9]
        # Only request 1's rows (slot base index 3, rows 3 and 4) are packed.
        assert _row_ids(model.mtp_layer_calls[-1]["hidden_states"]) == [3, 4]

    def test_returns_false_when_there_is_nothing_to_commit(self):
        """All drafts rejected and no prefill: the caller must run a dummy slot for EP balance."""
        context = _make_context(num_decode_requests=2)
        model = _make_model()
        controller = _make_controller(context, model)
        controller._accepted_token_counts_per_request[:2] = 0

        issued = controller._mtp_commit_pass(
            context,
            model,
            _hidden(2 * 3),
            num_decode_requests=2,
            active_request_count=2,
            base_position=torch.tensor([9, 12], device=DEVICE),
        )

        assert issued is False
        assert model.mtp_layer_calls == []
        assert context.setup_prefill_calls == []
        assert context.finalize_prefill_calls == 0


class TestMtpCommitPassPrefill:
    """Prefill requests: seed the chunk's positions, with the chunk/prefix-cache distinction."""

    def test_fresh_prompt_seeds_all_but_the_last_position(self):
        """`off == 0`: seed positions 0..q-2; the last position is seeded by the draft loop."""
        context = _make_context(prefill_query_lengths=(5,), prefill_kv_offsets=(0,))
        model = _make_model()
        controller = _make_controller(context, model)

        issued = controller._mtp_commit_pass(
            context,
            model,
            _hidden(5),
            num_decode_requests=0,
            active_request_count=1,
            base_position=torch.tensor([5], device=DEVICE),
        )

        assert issued is True
        call = context.setup_prefill_calls[-1]
        assert call["append_counts"].cpu().tolist() == [4]
        assert call["request_start_positions"].cpu().tolist() == [0]
        forward = model.mtp_layer_calls[-1]
        assert _row_ids(forward["hidden_states"]) == [0, 1, 2, 3]
        assert forward["next_token_ids"].view(-1).cpu().tolist() == [1001, 1002, 1003, 1004]

    def test_continuation_chunk_seeds_the_straddling_position(self):
        """Own prior chunk: the boundary position `off-1` is seeded from the carried hidden."""
        context = _make_context(
            prefill_query_lengths=(3,), prefill_kv_offsets=(8,), request_ids=[42]
        )
        model = _make_model()
        controller = _make_controller(context, model)
        # The previous chunk of THIS request left its last hidden behind.
        controller._mtp_chunk_boundary_hidden = _hidden(1, fill=99.0)
        controller._mtp_chunk_boundary_req_id = 42

        issued = controller._mtp_commit_pass(
            context,
            model,
            _hidden(3),
            num_decode_requests=0,
            active_request_count=1,
            base_position=torch.tensor([11], device=DEVICE),
        )

        assert issued is True
        call = context.setup_prefill_calls[-1]
        # count == q (not q-1) and start == off-1, because the straddling entry is included.
        assert call["append_counts"].cpu().tolist() == [3]
        assert call["request_start_positions"].cpu().tolist() == [7]
        forward = model.mtp_layer_calls[-1]
        # The carried hidden leads, then this chunk's roll-by-one hiddens.
        assert _row_ids(forward["hidden_states"]) == [99.0, 0, 1]
        # The boundary entry consumes this chunk's FIRST token t_off.
        assert forward["next_token_ids"].view(-1).cpu().tolist() == [1000, 1001, 1002]

    def test_prefix_cache_hit_does_not_rewrite_the_shared_divergence_entry(self):
        """`off > 0` with no own prior chunk: the inherited blocks are shared, so start at `off`.

        On a prefix-cache hit the skipped prefix was computed by a DIFFERENT request whose
        activations are gone. Every inherited entry p <= off-2 is already correct (its token
        t_{p+1} is inside the shared prefix). Only entry off-1 differs -- and it lives in a
        ref-counted block shared with the producer and every sibling, so writing this request's
        value would corrupt theirs. It must be left alone.
        """
        context = _make_context(
            prefill_query_lengths=(3,), prefill_kv_offsets=(8,), request_ids=[42]
        )
        model = _make_model()
        controller = _make_controller(context, model)
        assert controller._mtp_chunk_boundary_hidden is None  # no own prior chunk

        issued = controller._mtp_commit_pass(
            context,
            model,
            _hidden(3),
            num_decode_requests=0,
            active_request_count=1,
            base_position=torch.tensor([11], device=DEVICE),
        )

        assert issued is True
        call = context.setup_prefill_calls[-1]
        assert call["append_counts"].cpu().tolist() == [2]
        assert call["request_start_positions"].cpu().tolist() == [
            8
        ], "the prefix-cache hit wrote the shared divergence entry at off-1"
        forward = model.mtp_layer_calls[-1]
        assert _row_ids(forward["hidden_states"]) == [0, 1]
        assert forward["next_token_ids"].view(-1).cpu().tolist() == [1001, 1002]

    def test_boundary_hidden_from_another_request_is_not_reused(self):
        """A carried hidden only applies to the request that produced it."""
        context = _make_context(
            prefill_query_lengths=(3,), prefill_kv_offsets=(8,), request_ids=[42]
        )
        model = _make_model()
        controller = _make_controller(context, model)
        # A stale boundary left by a DIFFERENT request.
        controller._mtp_chunk_boundary_hidden = _hidden(1, fill=99.0)
        controller._mtp_chunk_boundary_req_id = 7

        controller._mtp_commit_pass(
            context,
            model,
            _hidden(3),
            num_decode_requests=0,
            active_request_count=1,
            base_position=torch.tensor([11], device=DEVICE),
        )

        call = context.setup_prefill_calls[-1]
        assert call["append_counts"].cpu().tolist() == [2]
        assert call["request_start_positions"].cpu().tolist() == [8]
        # The other request's hidden must not appear in the packed batch.
        assert 99.0 not in _row_ids(model.mtp_layer_calls[-1]["hidden_states"])

    def test_in_flight_chunk_carries_its_last_hidden_forward(self):
        """The chunked request's final hidden is kept for the next chunk's boundary entry."""
        context = _make_context(
            prefill_query_lengths=(4,),
            prefill_kv_offsets=(0,),
            request_ids=[42],
            chunked_prefill_request_id=42,
        )
        model = _make_model()
        controller = _make_controller(context, model)

        controller._mtp_commit_pass(
            context,
            model,
            _hidden(4),
            num_decode_requests=0,
            active_request_count=1,
            base_position=torch.tensor([4], device=DEVICE),
        )

        assert controller._mtp_chunk_boundary_req_id == 42
        assert controller._mtp_chunk_boundary_hidden is not None
        assert controller._mtp_chunk_boundary_hidden.shape == (1, 1, HIDDEN_SIZE)
        # The LAST hidden of this chunk (row 3) is the one carried.
        assert _row_ids(controller._mtp_chunk_boundary_hidden) == [3]

    def test_boundary_hidden_is_cleared_when_no_chunk_is_in_flight(self):
        """A finished chunk must not leave a boundary a later request could match."""
        context = _make_context(
            prefill_query_lengths=(4,),
            prefill_kv_offsets=(0,),
            request_ids=[42],
            chunked_prefill_request_id=-1,
        )
        model = _make_model()
        controller = _make_controller(context, model)
        controller._mtp_chunk_boundary_hidden = _hidden(1, fill=99.0)
        controller._mtp_chunk_boundary_req_id = 42

        controller._mtp_commit_pass(
            context,
            model,
            _hidden(4),
            num_decode_requests=0,
            active_request_count=1,
            base_position=torch.tensor([4], device=DEVICE),
        )

        assert controller._mtp_chunk_boundary_hidden is None
        assert controller._mtp_chunk_boundary_req_id == -1

    def test_carried_hidden_is_detached_from_the_step_that_produced_it(self):
        """The carry must be a private copy: the next step's hidden buffer is reused/freed."""
        context = _make_context(
            prefill_query_lengths=(4,),
            prefill_kv_offsets=(0,),
            request_ids=[42],
            chunked_prefill_request_id=42,
        )
        model = _make_model()
        controller = _make_controller(context, model)
        gathered = _hidden(4)

        controller._mtp_commit_pass(
            context,
            model,
            gathered,
            num_decode_requests=0,
            active_request_count=1,
            base_position=torch.tensor([4], device=DEVICE),
        )
        carried_before = _row_ids(controller._mtp_chunk_boundary_hidden)
        gathered.fill_(-1.0)  # simulate the buffer being overwritten next step

        assert _row_ids(controller._mtp_chunk_boundary_hidden) == carried_before


class TestMtpCommitPassMixedBatch:
    """Decode and prefill requests packed into one varlen forward, decode-first."""

    def test_decode_and_prefill_are_packed_decode_first(self):
        context = _make_context(
            num_decode_requests=1, prefill_query_lengths=(4,), prefill_kv_offsets=(0,)
        )
        model = _make_model()
        controller = _make_controller(context, model)
        controller._accepted_token_counts_per_request[:1] = 2
        stride = context.num_speculative_tokens + 1

        issued = controller._mtp_commit_pass(
            context,
            model,
            _hidden(stride + 4),
            num_decode_requests=1,
            active_request_count=2,
            base_position=torch.tensor([10, 4], device=DEVICE),
        )

        assert issued is True
        call = context.setup_prefill_calls[-1]
        assert call["append_counts"].cpu().tolist() == [2, 3]
        assert call["request_start_positions"].cpu().tolist() == [7, 0]
        # Decode rows (slot 0: hiddens 0,1) precede prefill rows (offset by decode_len=3).
        assert _row_ids(model.mtp_layer_calls[-1]["hidden_states"]) == [0, 1, 3, 4, 5]
        # The block table covers every active request, in active-slice order.
        assert call["block_table_prefill"].shape[0] == 2
        assert call["padded_request_count"] == 2

    def test_block_table_is_taken_from_the_active_slice(self):
        """Paused requests occupy leading rows; the commit pass must skip them."""
        context = _make_context(
            num_decode_requests=1,
            prefill_query_lengths=(3,),
            prefill_kv_offsets=(0,),
            paused_request_count=2,
        )
        model = _make_model()
        controller = _make_controller(context, model)
        controller._accepted_token_counts_per_request[:1] = 1

        controller._mtp_commit_pass(
            context,
            model,
            _hidden(3 + 3),
            num_decode_requests=1,
            active_request_count=2,
            base_position=torch.tensor([8, 3], device=DEVICE),
        )

        block_table = context.setup_prefill_calls[-1]["block_table_prefill"]
        # Rows 2 and 3 of request_to_kv_block_ids, not rows 0 and 1.
        assert block_table[:, 0].cpu().tolist() == [2 * MAX_KV_BLOCK_COUNT, 3 * MAX_KV_BLOCK_COUNT]

    def test_nvls_dispatcher_is_told_the_unpadded_token_count(self):
        context = _make_context(prefill_query_lengths=(5,), prefill_kv_offsets=(0,))
        context._nvls_dispatcher = object()
        model = _make_model()
        controller = _make_controller(context, model)

        with mock.patch(
            "megatron.core.inference.text_generation_controllers.mtp_inference_mixin."
            "NVLSAllGatherVDispatcher"
        ) as nvls:
            controller._mtp_commit_pass(
                context,
                model,
                _hidden(5),
                num_decode_requests=0,
                active_request_count=1,
                base_position=torch.tensor([5], device=DEVICE),
            )

        nvls.modify_real_token_count_for_mtp.assert_called_once_with(4)


class TestMtpCommitPassSequenceParallel:
    """Under SP the packed batch is padded to a TP multiple and scattered."""

    def test_pads_token_count_to_a_tp_multiple(self):
        context = _make_context(prefill_query_lengths=(4,), prefill_kv_offsets=(0,))
        model = _make_model()
        controller = _make_controller(context, model, sp_enabled=True, tp_size=4)

        with mock.patch(
            "megatron.core.inference.text_generation_controllers.mtp_inference_mixin."
            "scatter_to_sequence_parallel_region",
            side_effect=lambda t, group=None: t,
        ) as scatter:
            issued = controller._mtp_commit_pass(
                context,
                model,
                _hidden(4),
                num_decode_requests=0,
                active_request_count=1,
                base_position=torch.tensor([4], device=DEVICE),
            )

        assert issued is True
        call = context.setup_prefill_calls[-1]
        # 3 real tokens rounded up to the next multiple of 4.
        assert call["padded_token_count"] == 4
        # The write map still describes only the 3 real appends.
        assert call["append_counts"].cpu().tolist() == [3]
        scatter.assert_called_once()
        forward = model.mtp_layer_calls[-1]
        assert forward["hidden_states"].shape[0] == 4
        assert forward["next_token_ids"].shape[-1] == 4
        # Padding slots are zero-filled so the embedding never sees an out-of-range id.
        assert forward["next_token_ids"].view(-1).cpu().tolist() == [1001, 1002, 1003, 0]
        assert forward["position_ids"].view(-1).cpu().tolist() == [0, 0, 0, 0]

    def test_no_padding_when_already_aligned(self):
        context = _make_context(prefill_query_lengths=(5,), prefill_kv_offsets=(0,))
        model = _make_model()
        controller = _make_controller(context, model, sp_enabled=True, tp_size=4)

        with mock.patch(
            "megatron.core.inference.text_generation_controllers.mtp_inference_mixin."
            "scatter_to_sequence_parallel_region",
            side_effect=lambda t, group=None: t,
        ):
            controller._mtp_commit_pass(
                context,
                model,
                _hidden(5),
                num_decode_requests=0,
                active_request_count=1,
                base_position=torch.tensor([5], device=DEVICE),
            )

        # 4 real appends is already a TP multiple, so nothing is padded away.
        assert context.setup_prefill_calls[-1]["padded_token_count"] == 4
        assert context.setup_prefill_calls[-1]["append_counts"].cpu().tolist() == [4]
        forward = model.mtp_layer_calls[-1]
        assert forward["next_token_ids"].view(-1).cpu().tolist() == [1001, 1002, 1003, 1004]


class TestMtpDummyPrefillForward:
    """The EP-balance dummy that idle ranks run in place of a real commit pass."""

    def test_runs_cache_free(self):
        """`inference_context=None` keeps the dummy from appending to the idle rank's KV."""
        context = _make_context()
        model = _make_model()
        controller = _make_controller(context, model)

        controller._mtp_dummy_prefill_forward(context, model)

        call = model.mtp_layer_calls[-1]
        assert call["inference_context"] is None
        assert call["hidden_states"].shape == (1, 1, HIDDEN_SIZE)
        assert call["next_token_ids"].shape == (1, 1)
        assert torch.count_nonzero(call["next_token_ids"]) == 0

    def test_sequence_parallel_dummy_is_tp_sized(self):
        """The dummy's MoE all-to-all footprint must match the real path's."""
        context = _make_context()
        model = _make_model()
        controller = _make_controller(context, model, sp_enabled=True, tp_size=4)

        with mock.patch(
            "megatron.core.inference.text_generation_controllers.mtp_inference_mixin."
            "scatter_to_sequence_parallel_region",
            side_effect=lambda t, group=None: t,
        ):
            controller._mtp_dummy_prefill_forward(context, model)

        call = model.mtp_layer_calls[-1]
        assert call["hidden_states"].shape == (4, 1, HIDDEN_SIZE)
        assert call["next_token_ids"].shape == (1, 4)


def _make_draft_loop_controller(
    context, num_mtp_depths=2, active_request_count=2, graphed=False, mtp_kv_cache_on=True
):
    """A controller wired for `_compute_serial_mtp_and_sample`, recording MTP step calls."""
    model = _make_model()
    steps = []

    def compute_mtp_single_step(**kwargs):
        steps.append(kwargs)
        model.all_forwards.append(("mtp_step", kwargs))
        hidden = kwargs["hidden_states"]
        n = hidden.shape[0] if hidden is not None else active_request_count
        logits = torch.zeros((n, 1, VOCAB_SIZE), device=DEVICE)
        return hidden, logits

    model.compute_mtp_single_step = compute_mtp_single_step
    model.mtp_step_calls = steps

    controller = _make_controller(context, model)
    controller.num_mtp_depths = num_mtp_depths
    controller._mtp_resolved_padded_count = active_request_count if graphed else None
    controller._sampled_tokens_cuda = torch.zeros(MAX_REQUESTS, dtype=torch.int64, device=DEVICE)
    controller._sampled_mtp_tokens_cuda = torch.zeros(
        (num_mtp_depths, MAX_REQUESTS), dtype=torch.int64, device=DEVICE
    )
    controller._mtp_token_ids_buf = torch.zeros((1, MAX_REQUESTS), dtype=torch.int64, device=DEVICE)
    controller._mtp_position_ids_buf = torch.zeros(
        (1, MAX_REQUESTS), dtype=torch.int64, device=DEVICE
    )
    controller._last_accepted_seq_indices = torch.arange(active_request_count, device=DEVICE)
    controller._sample_from_logits_2d = lambda logits_2d: torch.zeros(
        logits_2d.shape[0], dtype=torch.int64, device=DEVICE
    )
    # The commit pass is covered above; stub it so these tests isolate the draft loop.
    controller._mtp_commit_pass = mock.Mock(return_value=True)
    controller._mtp_dummy_prefill_forward = mock.Mock()

    context.enable_mtp_kv_cache = mtp_kv_cache_on
    context.mtp_decoder_hidden_states = _hidden(active_request_count)
    context._mtp_begin_decode = mock.Mock()
    context._mtp_setup_decode_step = mock.Mock()
    context._mtp_advance_decode_step = mock.Mock()
    context._mtp_end_decode = mock.Mock()
    return controller, model, context


class TestSerialMtpDraftLoop:
    """The draft loop's KV-cache-specific behaviour."""

    def test_runs_one_extra_append_beyond_the_depth_loop(self):
        """D depths produce D+1 committed positions next step, so the KV needs D+1 entries."""
        context = _make_context(num_decode_requests=2)
        controller, model, context = _make_draft_loop_controller(context, num_mtp_depths=2)

        controller._compute_serial_mtp_and_sample(
            base_position=torch.tensor([10, 12], device=DEVICE)
        )

        assert len(model.mtp_step_calls) == 3, "expected D depth forwards plus one extra append"
        # Every forward writes KV, and the write position advances once per forward.
        assert context._mtp_setup_decode_step.call_count == 3
        assert context._mtp_advance_decode_step.call_count == 3
        context._mtp_end_decode.assert_called_once()

    def test_begins_decode_at_base_position_minus_one(self):
        """Depth 0 writes the roll-by-one entry for main position `base - 1`."""
        context = _make_context(num_decode_requests=2)
        controller, _, context = _make_draft_loop_controller(context, num_mtp_depths=2)

        controller._compute_serial_mtp_and_sample(
            base_position=torch.tensor([10, 12], device=DEVICE)
        )

        args, kwargs = context._mtp_begin_decode.call_args
        assert args[2].cpu().tolist() == [9, 11]
        assert kwargs["graphed"] is False

    def test_chunked_prefill_request_is_excluded_from_drafting(self):
        """A mid-prompt request must not draft; it becomes a padding row in the write map."""
        context = _make_context(
            num_decode_requests=1,
            prefill_query_lengths=(4,),
            prefill_kv_offsets=(0,),
            chunked_prefill_request_id=101,
        )
        controller, _, context = _make_draft_loop_controller(
            context, num_mtp_depths=2, active_request_count=2
        )

        controller._compute_serial_mtp_and_sample(
            base_position=torch.tensor([10, 4], device=DEVICE)
        )

        args, _ = context._mtp_begin_decode.call_args
        assert args[0] == 1, "the in-flight chunked request was allowed to draft"
        # The graph size and forward count are unchanged, so EP parity is preserved.
        assert args[1] == 2

    def test_all_requests_draft_when_no_chunk_is_in_flight(self):
        context = _make_context(num_decode_requests=2, chunked_prefill_request_id=-1)
        controller, _, context = _make_draft_loop_controller(
            context, num_mtp_depths=2, active_request_count=2
        )

        controller._compute_serial_mtp_and_sample(
            base_position=torch.tensor([10, 12], device=DEVICE)
        )

        args, _ = context._mtp_begin_decode.call_args
        assert args[0] == 2

    def test_graphed_step_replays_the_kv_aware_graph_key(self):
        """KV-aware graphs are captured under a distinct key from the cache-free ones."""
        context = _make_context(num_decode_requests=2)
        controller, model, context = _make_draft_loop_controller(
            context, num_mtp_depths=2, graphed=True
        )
        context.using_cuda_graph_this_step = lambda: True

        controller._compute_serial_mtp_and_sample(
            base_position=torch.tensor([10, 12], device=DEVICE)
        )

        assert context._mtp_begin_decode.call_args.kwargs["graphed"] is True
        for call in model.mtp_step_calls:
            assert call["cache_key"] is not None
            assert call["cache_key"][0] == "mtp_kv"
        # The extra append reuses the last depth's key rather than capturing at runtime.
        assert model.mtp_step_calls[-1]["eager"] is False

    def test_cache_free_graph_key_when_the_kv_cache_is_off(self):
        context = _make_context(num_decode_requests=2)
        controller, model, context = _make_draft_loop_controller(
            context, num_mtp_depths=2, graphed=True, mtp_kv_cache_on=False
        )
        context.using_cuda_graph_this_step = lambda: True

        controller._compute_serial_mtp_and_sample(
            base_position=torch.tensor([10, 12], device=DEVICE)
        )

        assert len(model.mtp_step_calls) == 2, "no extra append when the KV cache is off"
        for call in model.mtp_step_calls:
            assert call["cache_key"][0] == "mtp"
            assert call["mtp_inference_context"] is None
        context._mtp_begin_decode.assert_not_called()
        context._mtp_end_decode.assert_not_called()

    def test_draft_forwards_receive_the_inference_context_when_enabled(self):
        """Without the context the MTP attention runs cache-free and appends nothing."""
        context = _make_context(num_decode_requests=2)
        controller, model, context = _make_draft_loop_controller(context, num_mtp_depths=2)

        controller._compute_serial_mtp_and_sample(
            base_position=torch.tensor([10, 12], device=DEVICE)
        )

        for call in model.mtp_step_calls:
            assert call["mtp_inference_context"] is context

    def test_positions_advance_one_per_depth_then_the_extra_append(self):
        """The extra append is at `base + D`, one past the last depth."""
        context = _make_context(num_decode_requests=2)
        controller, model, context = _make_draft_loop_controller(context, num_mtp_depths=2)
        base_position = torch.tensor([10, 12], device=DEVICE)

        seen = []
        original = model.compute_mtp_single_step

        def recording(**kwargs):
            seen.append(kwargs["position_ids"][0, :2].clone().cpu().tolist())
            return original(**kwargs)

        model.compute_mtp_single_step = recording

        controller._compute_serial_mtp_and_sample(base_position=base_position)

        assert seen == [[10, 12], [11, 13], [12, 14]]

    def test_commit_pass_falls_back_to_a_dummy_slot(self):
        """Every rank runs exactly one commit-pass forward per step, real or dummy."""
        context = _make_context(num_decode_requests=2)
        controller, _, context = _make_draft_loop_controller(context, num_mtp_depths=2)
        controller._mtp_commit_pass = mock.Mock(return_value=False)

        controller._compute_serial_mtp_and_sample(
            base_position=torch.tensor([10, 12], device=DEVICE)
        )

        controller._mtp_dummy_prefill_forward.assert_called_once()

    def test_no_dummy_slot_when_the_commit_pass_issued_a_forward(self):
        context = _make_context(num_decode_requests=2)
        controller, _, context = _make_draft_loop_controller(context, num_mtp_depths=2)

        controller._compute_serial_mtp_and_sample(
            base_position=torch.tensor([10, 12], device=DEVICE)
        )

        controller._mtp_dummy_prefill_forward.assert_not_called()

    def test_legacy_scheduling_derives_base_position_from_post_rewind_cpu_state(self):
        """With no `base_position` the loop reads the ADJUSTED offsets, not the GPU snapshot."""
        context = _make_context(num_decode_requests=2)
        controller, _, context = _make_draft_loop_controller(context, num_mtp_depths=2)
        # Post-rewind CPU state: next position = adjusted offset + processed tokens.
        context.request_kv_length_offsets[:2] = torch.tensor([6, 9], dtype=torch.int32)
        context.request_query_lengths[:2] = torch.tensor([3, 2], dtype=torch.int32)

        controller._compute_serial_mtp_and_sample(base_position=None)

        args, _ = context._mtp_begin_decode.call_args
        assert args[2].cpu().tolist() == [8, 10]  # (6+3)-1 and (9+2)-1
        assert args[2].dtype == torch.int64, "CUDA graph capture expects int64 positions"


def _captured_graph_keys(batch_sizes, num_mtp_depths, mtp_use_repeated_layer=True):
    """Reproduce the key set `DynamicEngine` captures during MTP CUDA-graph warmup.

    Mirrors `dynamic_engine.py`: for each graphed batch size `n` it captures the cache-free
    family under `("mtp", n, depth)` and, when the MTP KV cache is on, the KV-aware family
    under `("mtp_kv", n, depth)`. `depth` is `None` for a repeated layer (the only shape the
    KV cache supports) and `0..D-1` otherwise.
    """
    depths = [None] if mtp_use_repeated_layer else list(range(num_mtp_depths))
    return {
        (prefix, n, depth) for prefix in ("mtp", "mtp_kv") for n in batch_sizes for depth in depths
    }


class TestMtpCudaGraphs:
    """Capture/replay agreement for the KV-aware MTP graphs."""

    def test_every_replayed_key_was_captured_at_warmup(self):
        """A key the draft loop replays but warmup never captured is an illegal runtime capture."""
        context = _make_context(num_decode_requests=2)
        controller, model, context = _make_draft_loop_controller(
            context, num_mtp_depths=2, active_request_count=2, graphed=True
        )
        context.using_cuda_graph_this_step = lambda: True
        captured = _captured_graph_keys(batch_sizes=[2], num_mtp_depths=2)

        controller._compute_serial_mtp_and_sample(
            base_position=torch.tensor([10, 12], device=DEVICE)
        )

        replayed = [call["cache_key"] for call in model.mtp_step_calls]
        assert all(key is not None for key in replayed)
        assert set(replayed) <= captured, f"uncaptured keys: {set(replayed) - captured}"

    def test_extra_append_reuses_the_last_depth_captured_key(self):
        """The D+1th append is a structurally identical forward, so it must not capture anew."""
        context = _make_context(num_decode_requests=2)
        controller, model, context = _make_draft_loop_controller(
            context, num_mtp_depths=2, active_request_count=2, graphed=True
        )
        context.using_cuda_graph_this_step = lambda: True
        captured = _captured_graph_keys(batch_sizes=[2], num_mtp_depths=2)

        controller._compute_serial_mtp_and_sample(
            base_position=torch.tensor([10, 12], device=DEVICE)
        )

        extra_append = model.mtp_step_calls[-1]
        assert extra_append["cache_key"] in captured
        assert extra_append["cache_key"][0] == "mtp_kv"
        assert (
            extra_append["eager"] is False
        ), "leaving eager=True while graphed would capture the extra append at runtime"

    def test_graph_keys_use_the_ep_synced_padded_count(self):
        """The replayed batch size must be the padded/EP-synced one, not the live active count."""
        context = _make_context(num_decode_requests=2)
        controller, model, context = _make_draft_loop_controller(
            context, num_mtp_depths=2, active_request_count=2, graphed=True
        )
        controller._mtp_resolved_padded_count = 4  # graph bucket is larger than the batch
        context.using_cuda_graph_this_step = lambda: True

        controller._compute_serial_mtp_and_sample(
            base_position=torch.tensor([10, 12], device=DEVICE)
        )

        assert all(call["cache_key"][1] == 4 for call in model.mtp_step_calls)
        assert context._mtp_begin_decode.call_args.args[1] == 4
        assert set(call["cache_key"] for call in model.mtp_step_calls) <= _captured_graph_keys(
            batch_sizes=[4], num_mtp_depths=2
        )

    def test_graphed_decision_mirrors_the_main_step_not_the_live_flag(self):
        """`_mtp_resolved_padded_count` is the EP-synced signal; the live flag is clobbered.

        The commit pass runs eager and leaves `using_cuda_graph_this_step()` False. The draft
        loop must still enter graphed mode, because the real context's `_mtp_setup_decode_step`
        restores the flag from the `graphed` argument before each depth forward.
        """
        context = _make_context(num_decode_requests=2)
        controller, _, context = _make_draft_loop_controller(
            context, num_mtp_depths=2, active_request_count=2, graphed=True
        )
        # What the commit pass leaves behind on the real ranks.
        context.using_cuda_graph_this_step = lambda: False

        controller._compute_serial_mtp_and_sample(
            base_position=torch.tensor([10, 12], device=DEVICE)
        )

        assert context._mtp_begin_decode.call_args.kwargs["graphed"] is True

    def test_eager_main_step_never_replays_a_graph(self):
        """An eager main step must keep the whole draft loop (and the extra append) eager."""
        context = _make_context(num_decode_requests=2)
        controller, model, context = _make_draft_loop_controller(
            context, num_mtp_depths=2, active_request_count=2, graphed=False
        )
        context.using_cuda_graph_this_step = lambda: False

        controller._compute_serial_mtp_and_sample(
            base_position=torch.tensor([10, 12], device=DEVICE)
        )

        assert context._mtp_begin_decode.call_args.kwargs["graphed"] is False
        for call in model.mtp_step_calls:
            assert call["cache_key"] is None
            assert call["eager"] is True

    def test_dummy_rank_replays_only_captured_cache_free_keys(self):
        """The idle rank's keys must also exist in the warmup set."""
        context = _make_context(num_decode_requests=2)
        controller, model, context = _make_draft_loop_controller(
            context, num_mtp_depths=2, active_request_count=2, graphed=True
        )
        controller.model_config.expert_model_parallel_size = 2
        controller._mtp_resolved_padded_count = 2
        captured = _captured_graph_keys(batch_sizes=[2], num_mtp_depths=2)

        controller._run_dummy_serial_mtp_forward()

        replayed = [call["cache_key"] for call in model.mtp_step_calls]
        assert set(replayed) <= captured
        assert all(key[0] == "mtp" for key in replayed)

    def test_per_depth_head_is_not_eligible_for_the_kv_cache(self):
        """The KV-aware capture only exists for a repeated layer; guard the assumption.

        `enable_mtp_kv_cache` is gated on `mtp_use_repeated_layer`, so a per-depth head must
        never reach the `mtp_kv` keys -- warmup would not have captured them at `depth=None`.
        """
        context = _make_context(num_decode_requests=2)
        controller, model, context = _make_draft_loop_controller(
            context, num_mtp_depths=2, active_request_count=2, graphed=True, mtp_kv_cache_on=False
        )
        model.mtp.mtp_use_repeated_layer = False
        context.using_cuda_graph_this_step = lambda: True

        controller._compute_serial_mtp_and_sample(
            base_position=torch.tensor([10, 12], device=DEVICE)
        )

        replayed = [call["cache_key"] for call in model.mtp_step_calls]
        assert [key[2] for key in replayed] == [0, 1], "per-depth head must pass a real depth"
        assert set(replayed) <= _captured_graph_keys(
            batch_sizes=[2], num_mtp_depths=2, mtp_use_repeated_layer=False
        )

    def test_block_scope_slices_the_persistent_hidden_buffer(self):
        """Block-scope graphs write a max_tokens-sized buffer; only this step's prefix is valid."""
        context = _make_context(num_decode_requests=2)
        controller, _, context = _make_draft_loop_controller(
            context, num_mtp_depths=2, active_request_count=2
        )
        context.inference_cuda_graph_scope = InferenceCudaGraphScope.block
        context.padded_active_token_count = 3
        # An oversized persistent buffer whose tail holds stale values from a previous step.
        context.mtp_decoder_hidden_states = _hidden(16)

        controller._compute_serial_mtp_and_sample(
            base_position=torch.tensor([10, 12], device=DEVICE)
        )

        passed_hidden = controller._mtp_commit_pass.call_args.args[2]
        assert passed_hidden.shape[0] == 3, "the stale tail of the graph buffer was not sliced off"
        assert _row_ids(passed_hidden) == [0, 1, 2]

    def test_block_scope_keeps_the_persistent_hidden_buffer_alive(self):
        """The block-scope buffer is pre-allocated at a fixed address and must persist."""
        context = _make_context(num_decode_requests=2)
        controller, _, context = _make_draft_loop_controller(
            context, num_mtp_depths=2, active_request_count=2
        )
        context.inference_cuda_graph_scope = InferenceCudaGraphScope.block
        context.padded_active_token_count = 2
        context.mtp_decoder_hidden_states = _hidden(8)

        controller._compute_serial_mtp_and_sample(
            base_position=torch.tensor([10, 12], device=DEVICE)
        )

        assert context.mtp_decoder_hidden_states is not None

    def test_non_block_scope_releases_the_hidden_states(self):
        """In eager/layer scope the attribute holds a live tensor that should be freed."""
        context = _make_context(num_decode_requests=2)
        controller, _, context = _make_draft_loop_controller(
            context, num_mtp_depths=2, active_request_count=2
        )
        context.inference_cuda_graph_scope = InferenceCudaGraphScope.layer

        controller._compute_serial_mtp_and_sample(
            base_position=torch.tensor([10, 12], device=DEVICE)
        )

        assert context.mtp_decoder_hidden_states is None

    @pytest.mark.parametrize("padded_count", [2, 4, 8])
    def test_graph_bucket_sizes_round_trip(self, padded_count):
        """Every graph bucket the engine captures must be replayable by the draft loop.

        `padded_count` is always >= the active request count; the buckets here are the padded
        sizes an EP-synced step would resolve to.
        """
        context = _make_context(num_decode_requests=2)
        controller, model, context = _make_draft_loop_controller(
            context, num_mtp_depths=2, active_request_count=2, graphed=True
        )
        controller._mtp_resolved_padded_count = padded_count
        context.using_cuda_graph_this_step = lambda: True

        controller._compute_serial_mtp_and_sample(
            base_position=torch.tensor([10, 12], device=DEVICE)
        )

        captured = _captured_graph_keys(batch_sizes=[padded_count], num_mtp_depths=2)
        assert set(call["cache_key"] for call in model.mtp_step_calls) <= captured
        assert context._mtp_begin_decode.call_args.args[1] == padded_count


class TestDummySerialMtpForward:
    """The idle EP rank must match the real ranks' forward COUNT and graph/eager MODE."""

    def _make_dummy_controller(self, num_mtp_depths=2, graphed=False, mtp_kv_cache_on=True):
        context = _make_context(num_decode_requests=2)
        controller, model, context = _make_draft_loop_controller(
            context, num_mtp_depths=num_mtp_depths, graphed=graphed, mtp_kv_cache_on=mtp_kv_cache_on
        )
        controller.model_config.expert_model_parallel_size = 2
        controller._mtp_resolved_padded_count = 2 if graphed else None
        return controller, model, context

    def test_forward_count_matches_the_real_path(self):
        """Real ranks run 1 commit + D depths + 1 extra append; the dummy must run D+2 too."""
        controller, model, context = self._make_dummy_controller(num_mtp_depths=2)

        controller._run_dummy_serial_mtp_forward()

        # The commit-pass slot is the MTP-layer forward; the depth loop and extra append are
        # compute_mtp_single_step calls.
        assert controller._mtp_dummy_prefill_forward.call_count == 1
        assert len(model.mtp_step_calls) == 3

    def test_dummy_always_replays_the_cache_free_graph(self):
        """Replaying the KV-aware graph here would append with no valid block table."""
        controller, model, _ = self._make_dummy_controller(num_mtp_depths=2, graphed=True)

        controller._run_dummy_serial_mtp_forward()

        for call in model.mtp_step_calls:
            assert call["cache_key"][0] == "mtp", "the dummy rank replayed the KV-aware graph"
            assert call["eager"] is False

    def test_dummy_mirrors_the_main_step_eager_mode(self):
        """Mode comes from the EP-synced padded count, not the clobbered live graph flag."""
        controller, model, context = self._make_dummy_controller(num_mtp_depths=2, graphed=False)
        # The commit pass clobbers this on the real ranks, so the dummy must ignore it.
        context.using_cuda_graph_this_step = lambda: True

        controller._run_dummy_serial_mtp_forward()

        for call in model.mtp_step_calls:
            assert call["eager"] is True
            assert call["cache_key"] is None

    def test_no_extra_forwards_when_the_kv_cache_is_off(self):
        controller, model, _ = self._make_dummy_controller(num_mtp_depths=2, mtp_kv_cache_on=False)

        controller._run_dummy_serial_mtp_forward()

        controller._mtp_dummy_prefill_forward.assert_not_called()
        assert len(model.mtp_step_calls) == 2

    @pytest.mark.parametrize("num_mtp_depths", [1, 3])
    def test_forward_count_tracks_depth(self, num_mtp_depths):
        controller, model, _ = self._make_dummy_controller(num_mtp_depths=num_mtp_depths)

        controller._run_dummy_serial_mtp_forward()

        assert len(model.mtp_step_calls) == num_mtp_depths + 1


# ---------------------------------------------------------------------------
# Expert-parallel parity and combination stress tests
# ---------------------------------------------------------------------------


def _build_step(
    num_decode_requests=0,
    prefill=(),
    accepted=(),
    chunked_prefill_request_id=-1,
    paused_request_count=0,
    num_mtp_depths=2,
    num_speculative_tokens=2,
    graphed=False,
    sp_enabled=False,
    tp_size=1,
    kv_cache_on=True,
    carried_boundary_req_id=None,
    cuda_graph_scope=InferenceCudaGraphScope.none,
):
    """Wire a controller + context for a FULL real step: commit pass and draft loop, unstubbed.

    Unlike `_make_draft_loop_controller` this leaves `_mtp_commit_pass` and
    `_mtp_dummy_prefill_forward` real, so `model.all_forwards` is the true per-step MoE forward
    log that the idle EP rank has to match.
    """
    prefill_query_lengths = tuple(q for q, _, _ in prefill)
    prefill_kv_offsets = tuple(off for _, off, _ in prefill)
    request_ids = [100 + i for i in range(num_decode_requests)] + [rid for _, _, rid in prefill]

    context = _make_context(
        num_decode_requests=num_decode_requests,
        prefill_query_lengths=prefill_query_lengths,
        prefill_kv_offsets=prefill_kv_offsets,
        request_ids=request_ids,
        chunked_prefill_request_id=chunked_prefill_request_id,
        paused_request_count=paused_request_count,
        num_speculative_tokens=num_speculative_tokens,
    )
    context.enable_mtp_kv_cache = kv_cache_on
    context.inference_cuda_graph_scope = cuda_graph_scope

    model = _make_model()
    steps = []

    def compute_mtp_single_step(**kwargs):
        steps.append(kwargs)
        model.all_forwards.append(("mtp_step", kwargs))
        hidden = kwargs["hidden_states"]
        n = hidden.shape[0] if hidden is not None else 1
        return hidden, torch.zeros((n, 1, VOCAB_SIZE), device=DEVICE)

    model.compute_mtp_single_step = compute_mtp_single_step
    model.mtp_step_calls = steps

    controller = _make_controller(context, model, sp_enabled=sp_enabled, tp_size=tp_size)
    controller.num_mtp_depths = num_mtp_depths
    active = num_decode_requests + len(prefill)
    padded = active
    if sp_enabled:
        padded = ((active + tp_size - 1) // tp_size) * tp_size
    controller._mtp_resolved_padded_count = padded if graphed else None
    controller._sampled_tokens_cuda = torch.zeros(MAX_REQUESTS, dtype=torch.int64, device=DEVICE)
    controller._sampled_mtp_tokens_cuda = torch.zeros(
        (num_mtp_depths, MAX_REQUESTS), dtype=torch.int64, device=DEVICE
    )
    controller._mtp_token_ids_buf = torch.zeros((1, MAX_REQUESTS), dtype=torch.int64, device=DEVICE)
    controller._mtp_position_ids_buf = torch.zeros(
        (1, MAX_REQUESTS), dtype=torch.int64, device=DEVICE
    )
    controller._sample_from_logits_2d = lambda logits_2d: torch.zeros(
        logits_2d.shape[0], dtype=torch.int64, device=DEVICE
    )
    if accepted:
        controller._accepted_token_counts_per_request[: len(accepted)] = torch.tensor(
            accepted, dtype=torch.int64, device=DEVICE
        )
    if carried_boundary_req_id is not None:
        controller._mtp_chunk_boundary_hidden = _hidden(1, fill=99.0)
        controller._mtp_chunk_boundary_req_id = carried_boundary_req_id

    # Packed main hiddens: decode slots first, then each prefill chunk.
    stride = num_speculative_tokens + 1
    total_tokens = num_decode_requests * stride + sum(prefill_query_lengths)
    context.mtp_decoder_hidden_states = _hidden(max(total_tokens, active))
    context.padded_active_token_count = max(total_tokens, active)
    controller._last_accepted_seq_indices = torch.arange(active, device=DEVICE)

    # The real context's `_mtp_setup_decode_step` routes to graph or non-graph metadata and
    # RESTORES the live graph flag (which the eager commit pass clobbered to False). Emulate
    # that here so the depth loop's `eager=`/`cache_key=` decisions are exercised faithfully.
    state = {"graphed": False, "positions": []}

    def begin_decode(active_count, padded_count, start_positions, graphed=False):
        state["graphed"] = graphed
        state["begin"] = (active_count, padded_count, start_positions.clone())
        state["offsets"] = start_positions.clone()
        context._using_cuda_graph_this_step = graphed

    def setup_decode_step():
        context._using_cuda_graph_this_step = state["graphed"]
        state["positions"].append(state["offsets"].cpu().tolist())

    def advance_decode_step():
        state["offsets"] = state["offsets"] + 1

    context._using_cuda_graph_this_step = False
    context.using_cuda_graph_this_step = lambda: context._using_cuda_graph_this_step
    context._mtp_begin_decode = mock.Mock(side_effect=begin_decode)
    context._mtp_setup_decode_step = mock.Mock(side_effect=setup_decode_step)
    context._mtp_advance_decode_step = mock.Mock(side_effect=advance_decode_step)
    context._mtp_end_decode = mock.Mock()
    # The commit pass drives real prefill metadata; keep recording it.
    return controller, model, context, state


def _identity_sp_patches():
    """Patch the SP collectives to identities so shapes/paddings stay observable."""
    return (
        mock.patch(
            "megatron.core.inference.text_generation_controllers.mtp_inference_mixin."
            "scatter_to_sequence_parallel_region",
            side_effect=lambda t, group=None: t,
        ),
        mock.patch(
            "megatron.core.inference.text_generation_controllers.mtp_inference_mixin."
            "gather_from_sequence_parallel_region",
            side_effect=lambda t, group=None: t,
        ),
    )


class TestExpertParallelForwardParity:
    """Idle EP ranks must issue the same MoE forwards as real ranks, or the all-to-all hangs."""

    @staticmethod
    def _dummy_rank_forwards(num_mtp_depths, graphed, kv_cache_on, sp_enabled=False, tp_size=1):
        """Run the idle-rank path in isolation and return its forward log."""
        context = _make_context(num_decode_requests=0)
        context.enable_mtp_kv_cache = kv_cache_on
        model = _make_model()
        steps = []

        def compute_mtp_single_step(**kwargs):
            steps.append(kwargs)
            model.all_forwards.append(("mtp_step", kwargs))
            hidden = kwargs["hidden_states"]
            n = hidden.shape[0] if hidden is not None else 1
            return hidden, torch.zeros((n, 1, VOCAB_SIZE), device=DEVICE)

        model.compute_mtp_single_step = compute_mtp_single_step
        model.mtp_step_calls = steps

        controller = _make_controller(context, model, sp_enabled=sp_enabled, tp_size=tp_size)
        controller.num_mtp_depths = num_mtp_depths
        controller.model_config.expert_model_parallel_size = 2
        controller._mtp_resolved_padded_count = (tp_size if sp_enabled else 2) if graphed else None

        patches = _identity_sp_patches()
        with patches[0], patches[1]:
            controller._run_dummy_serial_mtp_forward()
        return model.all_forwards

    @pytest.mark.parametrize("num_mtp_depths", [1, 2, 3])
    @pytest.mark.parametrize("graphed", [False, True])
    @pytest.mark.parametrize("kv_cache_on", [False, True])
    def test_real_and_dummy_ranks_issue_the_same_forward_count(
        self, num_mtp_depths, graphed, kv_cache_on
    ):
        """The headline EP invariant: forward COUNT must match across ranks, every config."""
        controller, model, context, _ = _build_step(
            num_decode_requests=2,
            accepted=(2, 1),
            num_mtp_depths=num_mtp_depths,
            graphed=graphed,
            kv_cache_on=kv_cache_on,
        )
        controller._compute_serial_mtp_and_sample(
            base_position=torch.tensor([10, 12], device=DEVICE)
        )
        real_forwards = list(model.all_forwards)

        dummy_forwards = self._dummy_rank_forwards(num_mtp_depths, graphed, kv_cache_on)

        expected = num_mtp_depths + (2 if kv_cache_on else 0)
        assert len(real_forwards) == expected
        assert len(dummy_forwards) == len(real_forwards), (
            f"EP mismatch: real rank issued {len(real_forwards)} MTP forwards, "
            f"idle rank issued {len(dummy_forwards)}"
        )

    @pytest.mark.parametrize("graphed", [False, True])
    def test_real_and_dummy_ranks_agree_on_graph_mode(self, graphed):
        """Mode must match too: a graphed replay and an eager launch are different collectives."""
        controller, model, context, _ = _build_step(
            num_decode_requests=2, accepted=(2, 1), num_mtp_depths=2, graphed=graphed
        )
        controller._compute_serial_mtp_and_sample(
            base_position=torch.tensor([10, 12], device=DEVICE)
        )
        real_modes = [kw["eager"] for tag, kw in model.all_forwards if tag == "mtp_step"]

        dummy = self._dummy_rank_forwards(2, graphed, kv_cache_on=True)
        dummy_modes = [kw["eager"] for tag, kw in dummy if tag == "mtp_step"]

        assert real_modes == [not graphed] * len(real_modes)
        assert dummy_modes == real_modes

    def test_dummy_rank_matches_a_real_rank_whose_commit_pass_was_empty(self):
        """All drafts rejected: the real rank runs a dummy slot, so the count still matches."""
        controller, model, context, _ = _build_step(
            num_decode_requests=2, accepted=(0, 0), num_mtp_depths=2
        )

        controller._compute_serial_mtp_and_sample(
            base_position=torch.tensor([10, 12], device=DEVICE)
        )

        # The commit pass issued nothing, so the EP-balance dummy slot ran in its place.
        assert len(model.all_forwards) == 4
        assert model.all_forwards[0][0] == "mtp_layer"
        assert model.all_forwards[0][1]["inference_context"] is None
        dummy = self._dummy_rank_forwards(2, graphed=False, kv_cache_on=True)
        assert len(dummy) == len(model.all_forwards)

    def test_commit_pass_slot_is_issued_exactly_once_per_step(self):
        """Real or dummy, there is exactly one commit-pass forward -- never zero, never two."""
        cases = (((2, 1), ()), ((0, 0), ()), ((0, 0), ((4, 0, 200),)))
        for accepted, prefill in cases:
            controller, model, context, _ = _build_step(
                num_decode_requests=2, prefill=prefill, accepted=accepted, num_mtp_depths=2
            )
            base = torch.tensor([10, 12, 4][: 2 + len(prefill)], device=DEVICE)

            controller._compute_serial_mtp_and_sample(base_position=base)

            slots = [tag for tag, _ in model.all_forwards if tag == "mtp_layer"]
            assert len(slots) == 1, f"accepted={accepted} prefill={prefill} -> {len(slots)} slots"

    @pytest.mark.parametrize("num_mtp_depths", [1, 2])
    def test_sequence_parallel_ranks_stay_matched(self, num_mtp_depths):
        """Under SP both paths pad to the same TP-aligned width."""
        controller, model, context, _ = _build_step(
            num_decode_requests=2,
            accepted=(2, 1),
            num_mtp_depths=num_mtp_depths,
            graphed=True,
            sp_enabled=True,
            tp_size=4,
        )
        patches = _identity_sp_patches()
        with patches[0], patches[1]:
            controller._compute_serial_mtp_and_sample(
                base_position=torch.tensor([10, 12], device=DEVICE)
            )

        dummy = self._dummy_rank_forwards(
            num_mtp_depths, graphed=True, kv_cache_on=True, sp_enabled=True, tp_size=4
        )
        assert len(dummy) == len(model.all_forwards)
        real_widths = {
            kw["next_token_ids"].shape[-1] for tag, kw in model.all_forwards if tag == "mtp_step"
        }
        dummy_widths = {kw["next_token_ids"].shape[-1] for tag, kw in dummy if tag == "mtp_step"}
        assert real_widths == dummy_widths == {4}

    def test_nvls_routing_mask_is_repointed_for_the_request_shaped_forwards(self):
        """MTP forwards are request-count shaped; the mask must follow, else padding rows route."""
        controller, model, context, _ = _build_step(
            num_decode_requests=1, prefill=((5, 0, 200),), accepted=(2,), num_mtp_depths=2
        )
        context._nvls_dispatcher = object()

        with mock.patch(
            "megatron.core.inference.text_generation_controllers.mtp_inference_mixin."
            "NVLSAllGatherVDispatcher"
        ) as nvls:
            controller._compute_serial_mtp_and_sample(
                base_position=torch.tensor([10, 5], device=DEVICE)
            )

        counts = [call.args[0] for call in nvls.modify_real_token_count_for_mtp.call_args_list]
        # First the commit pass's packed token total (2 accepted + 4 prompt), then the draft
        # loop's request count.
        assert counts == [6, 2]


class TestMtpKvCacheCombinations:
    """Stress the optimizations together: prefix caching, chunked prefill, CUDA graphs, SP, EP.

    Each scenario asserts the invariants that must hold in EVERY combination rather than
    scenario-specific values, so new combinations can be added cheaply.
    """

    SCENARIOS = {
        # name: kwargs for _build_step, plus the base positions to drive it with.
        "decode_only_eager": dict(num_decode_requests=2, accepted=(2, 1), base=[10, 12]),
        "decode_only_graphed": dict(
            num_decode_requests=2, accepted=(2, 1), graphed=True, base=[10, 12]
        ),
        "fresh_prefill_only": dict(prefill=((5, 0, 200),), base=[5]),
        "prefix_cache_hit_only": dict(prefill=((3, 8, 200),), base=[11]),
        "continuation_chunk_only": dict(
            prefill=((3, 8, 200),), carried_boundary_req_id=200, base=[11]
        ),
        "mixed_decode_and_fresh_prefill": dict(
            num_decode_requests=1, prefill=((4, 0, 200),), accepted=(2,), base=[10, 4]
        ),
        "mixed_decode_and_prefix_hit": dict(
            num_decode_requests=1, prefill=((3, 8, 200),), accepted=(1,), base=[10, 11]
        ),
        "chunked_prefill_in_flight": dict(
            num_decode_requests=1,
            prefill=((4, 0, 200),),
            accepted=(2,),
            chunked_prefill_request_id=200,
            base=[10, 4],
        ),
        "chunked_continuation_with_decode_graphed": dict(
            num_decode_requests=1,
            prefill=((3, 8, 200),),
            accepted=(2,),
            chunked_prefill_request_id=200,
            carried_boundary_req_id=200,
            graphed=True,
            base=[10, 11],
        ),
        "prefix_hit_plus_chunked_plus_paused": dict(
            num_decode_requests=1,
            prefill=((3, 8, 200),),
            accepted=(1,),
            chunked_prefill_request_id=200,
            paused_request_count=2,
            base=[10, 11],
        ),
        "sp_graphed_mixed": dict(
            num_decode_requests=1,
            prefill=((4, 0, 200),),
            accepted=(2,),
            graphed=True,
            sp_enabled=True,
            tp_size=4,
            base=[10, 4],
        ),
        "block_scope_graphed_decode": dict(
            num_decode_requests=2,
            accepted=(2, 1),
            graphed=True,
            cuda_graph_scope=InferenceCudaGraphScope.block,
            base=[10, 12],
        ),
        "all_rejected_with_prefix_hit": dict(
            num_decode_requests=2, prefill=((3, 8, 200),), accepted=(0, 0), base=[10, 12, 11]
        ),
        "depth_one_chunked_graphed": dict(
            num_decode_requests=1,
            prefill=((4, 0, 200),),
            accepted=(1,),
            chunked_prefill_request_id=200,
            graphed=True,
            num_mtp_depths=1,
            base=[10, 4],
        ),
    }

    @pytest.mark.parametrize("scenario_name", sorted(SCENARIOS))
    def test_invariants_hold(self, scenario_name):
        kwargs = dict(self.SCENARIOS[scenario_name])
        base = kwargs.pop("base")
        num_mtp_depths = kwargs.pop("num_mtp_depths", 2)
        sp_enabled = kwargs.get("sp_enabled", False)
        tp_size = kwargs.get("tp_size", 1)
        graphed = kwargs.get("graphed", False)

        controller, model, context, state = _build_step(num_mtp_depths=num_mtp_depths, **kwargs)
        base_position = torch.tensor(base, device=DEVICE)
        active = kwargs.get("num_decode_requests", 0) + len(kwargs.get("prefill", ()))

        patches = _identity_sp_patches()
        with patches[0], patches[1]:
            controller._compute_serial_mtp_and_sample(base_position=base_position)

        # 1. EP: exactly one commit-pass slot, D depth forwards, one extra append.
        slots = [tag for tag, _ in model.all_forwards if tag == "mtp_layer"]
        steps = [kw for tag, kw in model.all_forwards if tag == "mtp_step"]
        assert len(slots) == 1
        assert len(steps) == num_mtp_depths + 1

        # 2. CUDA graphs: keys are all-or-nothing, and every replayed key was captured.
        padded = controller._mtp_resolved_padded_count or active
        if sp_enabled:
            padded = ((active + tp_size - 1) // tp_size) * tp_size
        keys = [kw["cache_key"] for kw in steps]
        if graphed:
            captured = _captured_graph_keys([padded], num_mtp_depths)
            assert all(k is not None for k in keys)
            assert set(keys) <= captured, f"uncaptured: {set(keys) - captured}"
            assert all(kw["eager"] is False for kw in steps)
        else:
            assert all(k is None for k in keys)
            assert all(kw["eager"] is True for kw in steps)

        # 3. Draft writes: one setup+advance per forward, positions strictly +1 per depth.
        assert context._mtp_setup_decode_step.call_count == num_mtp_depths + 1
        assert context._mtp_advance_decode_step.call_count == num_mtp_depths + 1
        context._mtp_end_decode.assert_called_once()
        positions = state["positions"]
        assert positions[0] == (base_position - 1).cpu().tolist()
        for earlier, later in zip(positions, positions[1:]):
            assert [b - a for a, b in zip(earlier, later)] == [1] * len(earlier)

        # 4. Drafting excludes a mid-prompt chunked request but keeps the graph size.
        begin_active, begin_padded, _ = state["begin"]
        expected_drafters = active - (
            1 if kwargs.get("chunked_prefill_request_id", -1) != -1 else 0
        )
        assert begin_active == expected_drafters
        assert begin_padded == padded

        # 5. The commit pass never emits a negative start or count.
        for call in context.setup_prefill_calls:
            assert (call["append_counts"] >= 0).all()
            if call["request_start_positions"] is not None:
                assert (call["request_start_positions"] >= 0).all()

        # 6. A boundary hidden is carried only while a chunk is genuinely in flight.
        if kwargs.get("chunked_prefill_request_id", -1) == -1:
            assert controller._mtp_chunk_boundary_hidden is None
            assert controller._mtp_chunk_boundary_req_id == -1

    @pytest.mark.parametrize("scenario_name", sorted(SCENARIOS))
    def test_dummy_rank_matches_every_scenario(self, scenario_name):
        """Whatever the real rank does in a scenario, an idle EP rank must match its count."""
        kwargs = dict(self.SCENARIOS[scenario_name])
        base = kwargs.pop("base")
        num_mtp_depths = kwargs.pop("num_mtp_depths", 2)

        controller, model, context, _ = _build_step(num_mtp_depths=num_mtp_depths, **kwargs)
        patches = _identity_sp_patches()
        with patches[0], patches[1]:
            controller._compute_serial_mtp_and_sample(
                base_position=torch.tensor(base, device=DEVICE)
            )

        dummy = TestExpertParallelForwardParity._dummy_rank_forwards(
            num_mtp_depths,
            graphed=kwargs.get("graphed", False),
            kv_cache_on=True,
            sp_enabled=kwargs.get("sp_enabled", False),
            tp_size=kwargs.get("tp_size", 1),
        )

        assert len(dummy) == len(model.all_forwards), (
            f"{scenario_name}: real rank issued {len(model.all_forwards)} MTP forwards, "
            f"idle rank issued {len(dummy)}"
        )
