# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

import torch

from megatron.core.inference.async_txn import (
    AsyncDecodeSlot,
    AsyncDecodeSlotRing,
    AsyncTxnDiagnostics,
    AsyncTxnSkipReason,
    StepTxn,
)
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)


class FakeGPUView:
    def __init__(self):
        self._buf = torch.zeros(32, dtype=torch.uint8)
        self.token_to_input_ids = torch.zeros(4, dtype=torch.int64)
        self.token_to_pos_ids = torch.arange(4, dtype=torch.int64)
        self.token_to_block_idx = torch.zeros(4, dtype=torch.int64)
        self.mha_block_table = torch.zeros((2, 2), dtype=torch.int32)


class FakeKVAllocator:
    block_routing = False

    def store_routing_per_block(self, routing):
        return None


class FakeContext:
    def __init__(self, *, async_scheduling=True, termination_id=-1, use_cuda_graph=False):
        self.async_scheduling = async_scheduling
        self.async_txn_diagnostics = AsyncTxnDiagnostics(enabled=async_scheduling)
        self.async_decode_slot_ring = AsyncDecodeSlotRing(
            (AsyncDecodeSlot(0, FakeGPUView()), AsyncDecodeSlot(1, FakeGPUView()))
        )
        self.active_decode_slot_id = 0
        self.total_request_count = 1
        self.paused_request_count = 0
        self.active_token_count = 1
        self.padded_batch_dimensions = SimpleNamespace(
            token_count=1,
            prefill_req_count=0,
            decode_req_count=1,
            req_count=1,
        )
        self.padded_active_request_count = 1
        self.padded_active_token_count = 1
        self.num_decode_requests = 1
        self.request_ids = torch.tensor([17], dtype=torch.int64)
        self.request_kv_length_offsets = torch.tensor([1], dtype=torch.int32)
        self.request_query_lengths = torch.tensor([1], dtype=torch.int32)
        self.request_output_lengths = torch.tensor([16], dtype=torch.int32)
        self.active_request_metadata = {
            "termination_id": torch.tensor([termination_id, -1, -1, -1], dtype=torch.int64),
            "return_log_probs": torch.tensor([False, False, False, False], dtype=torch.bool),
            "top_n_logprobs": torch.tensor([0, 0, 0, 0], dtype=torch.int32),
            "top_k": torch.tensor([1, 1, 1, 1], dtype=torch.int32),
            "top_p": torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        }
        self.config = SimpleNamespace(
            materialize_only_last_token_logits=True, sampling_backend="flashinfer"
        )
        self.is_hybrid_model = False
        self.mamba_slot_allocator = None
        self.kv_block_allocator = FakeKVAllocator()
        self.request_rng_store = None
        self.prepared = 0
        self.updated = 0
        self.use_cuda_graph = use_cuda_graph
        self.deferred_h2d_prepares = 0
        self.child_graph_shape_syncs = 0
        self.child_graph_shape_order = None
        self.child_forward_graph_flags = []
        self._disable_cuda_graph_replay_this_scope = False

    def is_decode_only(self):
        return True

    def using_cuda_graph_this_step(self):
        return self.use_cuda_graph

    def replay_cuda_graph_this_step(self):
        return self.use_cuda_graph and not self._disable_cuda_graph_replay_this_scope

    def cuda_graph_cache_key(self):
        return self.padded_batch_dimensions

    @contextmanager
    def async_child_forward_graph_replay_disabled_scope(self):
        previous = self._disable_cuda_graph_replay_this_scope
        self._disable_cuda_graph_replay_this_scope = True
        try:
            yield
        finally:
            self._disable_cuda_graph_replay_this_scope = previous

    def last_token_logits(self, logits):
        return logits.squeeze(0)[: self.total_request_count - self.paused_request_count]

    def active_decode_slot(self):
        return self.async_decode_slot_ring.current

    def bind_decode_slot(self, slot=None):
        if slot is None:
            slot = self.async_decode_slot_ring.current
        self.active_decode_slot_id = slot.slot_id
        return slot

    def current_input_and_position_ids(self):
        view = self.async_decode_slot_ring.current.gpu_view
        return view.token_to_input_ids[:1].unsqueeze(0), view.token_to_pos_ids[:1].unsqueeze(0)

    def sync_ep_child_graph_shape(self):
        self.child_graph_shape_syncs += 1
        if self.child_graph_shape_order is not None:
            self.child_graph_shape_order.append("child_graph_shape_sync")

    def prepare_child_from_committed_decode_state(self, *, target_slot=None, defer_h2d=False):
        if not self.async_scheduling:
            return None
        self.prepared += 1
        child = target_slot or self.async_decode_slot_ring.child
        if defer_h2d:
            self.deferred_h2d_prepares += 1
        self.async_txn_diagnostics.record_prepared(under_forward=True)
        return StepTxn(
            step_id=self.prepared,
            request_ids=(17,),
            slot_id=child.slot_id,
            cpu_bookkeeping_buf=torch.zeros_like(child.gpu_view._buf) if defer_h2d else None,
            cuda_graph_key=None,
        )

    def plain_decode_child_needs_terminal_check(self, active_request_count=None):
        active_request_count = active_request_count or 1
        if bool((self.active_request_metadata["termination_id"][:active_request_count] >= 0).any()):
            return True
        return bool(
            torch.ge(
                self.get_active_sequence_lengths()[:active_request_count] + 1,
                self.get_max_sequence_lengths()[:active_request_count],
            ).any()
        )

    def get_active_sequence_lengths(self):
        return self.request_kv_length_offsets + self.request_query_lengths

    def get_max_sequence_lengths(self):
        return self.request_output_lengths

    def update_requests(
        self, active_request_mask, new_sample_copy, sampled_mtp_tokens_cpu=None, async_txn=None
    ):
        self.updated += 1
        self.request_kv_length_offsets.add_(1)
        if async_txn is not None:
            survivors = self.request_ids[active_request_mask.bool()].tolist()
            terminals = self.request_ids[~active_request_mask.bool()].tolist()
            async_txn.mark_committed(survivors, terminal_request_ids=terminals)
        return {}

    def retire_unused_async_kv_leases(self, async_txn):
        return None


def _make_controller(context):
    order = []
    controller = object.__new__(TextGenerationController)
    controller.model_config = SimpleNamespace(
        cuda_graph_impl="none",
        expert_model_parallel_size=1,
        moe_enable_routing_replay=False,
        num_moe_experts=None,
    )
    controller.inference_wrapped_model = SimpleNamespace(
        inference_context=context,
        model=SimpleNamespace(config=controller.model_config),
        config=SimpleNamespace(params_dtype=torch.float32),
    )
    controller.num_speculative_tokens = 0
    controller.model_is_pipeline_parallel = False
    controller._enable_cuda_graph = False
    controller._ep_async_protocol = None
    controller._get_stop_word_finished_ids_callback = None
    controller._has_stop_word_constraints_callback = lambda request_ids: True
    controller._async_prepared_child_txn = None
    controller._async_launched_child_txn = None
    controller._async_deferred_sample_txn = None
    controller._async_deferred_sample_cuda_graph_request_count = None
    controller._accepted_tokens_per_request = None
    controller._accepted_token_counts_per_request = None
    controller._sampled_tokens_cuda = torch.tensor([5], dtype=torch.int64)
    controller._async_sampled_tokens_cpu = torch.empty(4, dtype=torch.int64)
    controller._async_sample_transfer_done_event = None
    controller._async_sample_transfer_count = 0

    def context_init():
        order.append("init")
        return context.current_input_and_position_ids()

    def forward(input_ids, position_ids):
        if "sample" in order:
            order.append("child_forward")
            context.child_forward_graph_flags.append(
                (
                    context.using_cuda_graph_this_step(),
                    context.replay_cuda_graph_this_step(),
                )
            )
        else:
            order.append("forward")

    def sample():
        order.append("sample")
        controller._sampled_tokens_cuda = torch.tensor([7], dtype=torch.int64)

    def transfer(active_request_count):
        order.append("transfer")
        return controller._sampled_tokens_cuda[:active_request_count].cpu(), None

    def record_sample_ready():
        order.append("sample_ready")
        return object()

    def start_async_transfer(active_request_count, sample_ready_event):
        order.append("transfer_start")
        assert sample_ready_event is not None
        assert "sample" in order
        assert "child_forward" not in order
        controller._async_sampled_tokens_cpu[:active_request_count].copy_(
            controller._sampled_tokens_cuda[:active_request_count]
        )
        controller._async_sample_transfer_done_event = sample_ready_event
        controller._async_sample_transfer_count = active_request_count

    def consume_async_transfer(active_request_count):
        if (
            controller._async_sample_transfer_done_event is None
            or controller._async_sample_transfer_count != active_request_count
        ):
            return None
        order.append("transfer")
        sampled_tokens_cpu = controller._async_sampled_tokens_cpu[:active_request_count].clone()
        controller._async_sample_transfer_done_event = None
        controller._async_sample_transfer_count = 0
        return sampled_tokens_cpu

    controller._dynamic_step_context_init = context_init
    controller._dynamic_step_forward_logits = forward
    controller._dynamic_step_sample_logits = sample
    controller._dynamic_step_log_probs_bookkeeping = lambda: (False, False)
    controller._dynamic_step_calculate_log_probs = lambda: (None, None)
    controller._dynamic_step_calculate_top_n_logprobs = lambda log_probs_tensor: None
    controller._router_record_bookkeeping = lambda: None
    controller._transfer_samples_to_cpu = transfer
    controller._record_async_sample_ready_event = record_sample_ready
    controller._start_async_sample_transfer = start_async_transfer
    controller._consume_async_sample_transfer = consume_async_transfer
    return controller, order


def _make_sampling_controller(context):
    controller = object.__new__(TextGenerationController)
    controller.inference_wrapped_model = SimpleNamespace(inference_context=context)
    controller.num_speculative_tokens = 0
    controller._sampling_backend = "flashinfer"
    controller._enable_cuda_graph = False
    controller._greedy_sample_values_cuda = torch.empty(4, dtype=torch.float32)
    controller._greedy_sampled_tokens_cuda = torch.empty(4, dtype=torch.int64)
    controller._sampled_tokens_cuda = None
    return controller


def test_dynamic_sample_logits_uses_argmax_for_greedy_requests():
    context = FakeContext()
    context.total_request_count = 2
    controller = _make_sampling_controller(context)
    controller._sampling = SimpleNamespace(
        sample_kernel=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("greedy requests should not use the generic sampler")
        )
    )
    controller._all_logits_cuda = torch.tensor(
        [[[1.0, 7.0, 3.0, 0.5], [4.0, 2.0, 9.0, 1.0]]], dtype=torch.float32
    )

    controller._dynamic_step_sample_logits()

    assert controller._sampled_tokens_cuda is controller._greedy_sampled_tokens_cuda
    assert controller._sampled_tokens_cuda[:2].tolist() == [1, 2]


def test_dynamic_sample_logits_keeps_generic_sampler_for_non_greedy_requests():
    context = FakeContext()
    context.total_request_count = 2
    context.active_request_metadata["top_p"][1] = 0.8
    controller = _make_sampling_controller(context)
    sampled = torch.tensor([3, 4], dtype=torch.int64)
    calls = []

    def sample_kernel(*args, **kwargs):
        calls.append((args, kwargs))
        return sampled

    controller._sampling = SimpleNamespace(sample_kernel=sample_kernel)
    controller._all_logits_cuda = torch.zeros((1, 2, 4), dtype=torch.float32)

    controller._dynamic_step_sample_logits()

    assert calls
    assert controller._sampled_tokens_cuda is sampled


def test_cpu_sample_transfer_starts_before_child_launch():
    context = FakeContext()
    controller, order = _make_controller(context)

    result = asyncio.run(controller.async_generate_output_tokens_dynamic_batch())

    assert result["sample"].tolist() == [7]
    assert order.index("sample") < order.index("transfer_start")
    assert order.index("transfer_start") < order.index("child_forward")
    assert order.index("child_forward") < order.index("transfer")
    assert context.async_txn_diagnostics.launched == 1
    assert context.async_txn_diagnostics.prepared == 2
    assert context.async_txn_diagnostics.h2d_ready_before_sampling == 1
    assert context.async_txn_diagnostics.sample_to_launch_latency_us > 0.0
    assert context.async_txn_diagnostics.commit_duration_us > 0.0
    assert controller._async_launched_child_txn is not None


def test_prepared_async_child_path_does_not_yield_before_launch(monkeypatch):
    context = FakeContext()
    controller, _ = _make_controller(context)
    sleeps = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    asyncio.run(controller.async_generate_output_tokens_dynamic_batch())

    assert sleeps == []
    assert context.async_txn_diagnostics.launched == 1


def test_child_launch_enters_graph_shape_sync_before_forward():
    context = FakeContext()
    controller, order = _make_controller(context)
    context.child_graph_shape_order = order

    asyncio.run(controller.async_generate_output_tokens_dynamic_batch())

    assert context.child_graph_shape_syncs == 1
    assert order.index("child_graph_shape_sync") < order.index("child_forward")


def test_bookkeeping_uses_supplied_async_cpu_sample_without_default_transfer():
    context = FakeContext()
    controller, _ = _make_controller(context)
    controller._transfer_samples_to_cpu = lambda active_request_count: (_ for _ in ()).throw(
        AssertionError("_transfer_samples_to_cpu should not run for async sample handoff")
    )

    result = controller._dynamic_step_context_bookkeeping(
        sampled_tokens_cpu=torch.tensor([7], dtype=torch.int64)
    )

    assert result["sample"].tolist() == [7]
    assert context.updated == 1


def test_chain_plain_decode_allows_empty_stop_word_state():
    context = FakeContext()
    controller, _ = _make_controller(context)
    controller._has_stop_word_constraints_callback = lambda request_ids: False
    parent_txn = StepTxn(step_id=1, request_ids=(17,), slot_id=0)
    child_txn = StepTxn(step_id=2, request_ids=(17,), slot_id=1)

    assert controller._can_chain_plain_decode_child(
        parent_txn,
        child_txn,
        active_request_count=1,
        return_log_probs=False,
        return_top_n_logprobs=False,
        skip_bookkeeping=False,
    )


def test_chain_plain_decode_defers_active_stop_words():
    context = FakeContext()
    controller, _ = _make_controller(context)
    controller._has_stop_word_constraints_callback = lambda request_ids: True
    parent_txn = StepTxn(step_id=1, request_ids=(17,), slot_id=0)
    child_txn = StepTxn(step_id=2, request_ids=(17,), slot_id=1)

    assert not controller._can_chain_plain_decode_child(
        parent_txn,
        child_txn,
        active_request_count=1,
        return_log_probs=False,
        return_top_n_logprobs=False,
        skip_bookkeeping=False,
    )


def test_consecutive_decode_steps_consume_launched_children():
    context = FakeContext()
    controller, order = _make_controller(context)

    asyncio.run(controller.async_generate_output_tokens_dynamic_batch())
    order.clear()
    asyncio.run(controller.async_generate_output_tokens_dynamic_batch())

    assert "init" not in order
    assert "forward" not in order
    assert order.index("sample") < order.index("child_forward")
    assert context.async_txn_diagnostics.consumed >= 1
    assert context.async_txn_diagnostics.launched == 2


def test_child_launch_allows_ordinary_terminal_rows():
    context = FakeContext(termination_id=7)
    controller, order = _make_controller(context)
    controller._transfer_samples_to_cpu = lambda active_request_count: (
        order.append("transfer") or (controller._sampled_tokens_cuda[:active_request_count], None)
    )

    asyncio.run(controller.async_generate_output_tokens_dynamic_batch())

    assert "child_forward" in order
    assert context.async_txn_diagnostics.launched == 1


def test_cuda_graph_child_launch_replays_current_slot_after_deferred_h2d():
    context = FakeContext(use_cuda_graph=True)
    controller, order = _make_controller(context)
    controller._enable_cuda_graph = True

    asyncio.run(controller.async_generate_output_tokens_dynamic_batch())

    assert "child_forward" in order
    assert context.active_decode_slot_id == 0
    assert controller._async_launched_child_txn.slot_id == 0
    assert controller._async_launched_child_txn.cpu_bookkeeping_buf is None
    assert context.child_forward_graph_flags == [(True, True)]
    assert context.using_cuda_graph_this_step()
    assert context.replay_cuda_graph_this_step()
    assert context.deferred_h2d_prepares == 2
    assert context.async_txn_diagnostics.h2d_ready_before_sampling == 0
    assert context.async_txn_diagnostics.launched == 1


def test_async_disabled_path_does_not_prepare_or_launch():
    context = FakeContext(async_scheduling=False)
    controller, order = _make_controller(context)
    controller._transfer_samples_to_cpu = lambda active_request_count: (
        order.append("transfer") or (controller._sampled_tokens_cuda[:active_request_count], None)
    )

    asyncio.run(controller.async_generate_output_tokens_dynamic_batch())

    assert "child_forward" not in order
    assert context.prepared == 0
    assert context.async_txn_diagnostics.snapshot()["launched"] == 0
