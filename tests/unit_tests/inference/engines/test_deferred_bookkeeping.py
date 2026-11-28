# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bookkeeping of each step behind the next step's forward launch.

`async_forward` leaves the step's record pending. The controller's `after_launch` hook runs
its bookkeeping (`drain_bookkeeping`) once the next forward is launched; the engine loops,
`suspend()` and `shutdown()` run it wherever no forward follows. The stop-word callback
finds the rows the pending record ends before the launch, so a stop word still ends a row
one step after the token that completes it.
"""

import asyncio
import time
from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from itertools import count
from types import SimpleNamespace
from unittest import mock

import msgpack
import pytest
import torch

from megatron.core.inference.config import AsyncScheduleMode, KVCacheManagementMode
from megatron.core.inference.engines.dynamic.loop import StepRecord
from megatron.core.inference.engines.dynamic_engine import (
    DynamicInferenceEngine,
    EngineState,
    RequestEntry,
)
from megatron.core.inference.headers import Headers
from megatron.core.inference.inference_request import (
    DynamicInferenceRequest,
    DynamicInferenceRequestRecord,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    DecodeOnly,
    DynamicBatchControllerStepResult,
    TextGenerationController,
)


def _request(
    generated_tokens=(),
    stop_word_ids=None,
    *,
    request_id=1,
    num_tokens_to_generate=64,
    stop_token_ids=None,
):
    request = DynamicInferenceRequest(
        request_id=request_id,
        prompt_tokens=torch.tensor([1, 2, 3]),
        sampling_params=SamplingParams(
            num_tokens_to_generate=num_tokens_to_generate,
            termination_id=-1,
            detokenize_stop_sequence=True,
            stop_token_ids=stop_token_ids,
        ),
        generated_tokens=list(generated_tokens),
        stop_word_ids=stop_word_ids,
    )
    request.add_event_add_engine()
    return request


def _engine(requests=(), *, num_speculative_tokens=0):
    """An engine holding what the stop-word callback and a bookkeeping pass read: no model, no GPU."""
    controller = TextGenerationController.__new__(TextGenerationController)
    controller.extra_eos_token_id_set = frozenset()
    controller.tokenizer = SimpleNamespace(detokenize=lambda token_ids: str(token_ids[0]))
    engine = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    engine.controller = controller
    engine.num_speculative_tokens = num_speculative_tokens
    engine.stop_word_being_finished_ids = set()
    engine.stop_word_finished_request_ids = set()
    engine._pending_record = None
    engine.context = SimpleNamespace(
        chunked_prefill_request_id=-1,
        enable_prefix_caching=False,
        kv_block_allocator=SimpleNamespace(pool_size=8, pool_avail=8, enable_prefix_caching=False),
        remove_vlm_request_data=mock.Mock(),
    )
    engine.requests = {
        request.request_id: RequestEntry(
            record=DynamicInferenceRequestRecord.from_request(request), future=Future()
        )
        for request in requests
    }
    engine.finished_request_count = engine.evicted_request_count = 0
    engine.track_generated_token_events = engine.track_paused_request_events = False
    engine.is_mp_coordinator = engine.use_coordinator = False
    engine.failed_request_ids = []
    engine.rank = 0
    engine._spec_steps = 0
    engine._spec_tokens_proposed_per_pos = torch.zeros(num_speculative_tokens, dtype=torch.long)
    engine._spec_tokens_accepted_per_pos = torch.zeros(num_speculative_tokens, dtype=torch.long)
    engine._prepare_handoff_metadata_batch = mock.Mock(return_value={})
    engine._promote_recomputed_prompt_scores = mock.Mock()
    return engine


def _record(engine, rows, *, finished=(), forced=(), chunked_prefill_request_id=-1, step=0):
    """Leave a step's record pending, as `async_forward` does.

    `rows` maps each request id to the tokens the step produced for it (accepted drafts
    before the sampled token when a list); `forced` are the rows the step's stop-word
    callback finished.
    """
    bursts = [tokens if isinstance(tokens, list) else [tokens] for tokens in rows.values()]
    depth = max(engine.num_speculative_tokens, max(len(burst) for burst in bursts) - 1)
    accepted_tokens = None
    if depth > 0:
        # Accepted drafts per row, -1 past the accepted ones, as the controller reports them.
        accepted_tokens = torch.tensor(
            [burst[:-1] + [-1] * (depth + 1 - len(burst)) for burst in bursts]
        )
    engine._pending_record = StepRecord(
        step_result={
            "active_request_ids": torch.tensor(list(rows)),
            "finished_request_ids": torch.tensor(list(finished), dtype=torch.int64),
            "sample": torch.tensor([burst[-1] for burst in bursts]),
            "accepted_tokens": accepted_tokens,
            "log_probs": None,
            "top_n_logprobs": None,
            "cuda_graph_request_count": None,
        },
        context_state={
            "chunked_prefill_request_id": chunked_prefill_request_id,
            "active_token_count": len(rows),
            "step_count": step,
            "kv_stats": None,
            "log_this_step": False,
        },
        time=time.time(),
        stop_word_being_finished_ids=set(forced),
        block_stats=None,
        prefix_cache_deltas=None,
    )
    return engine._pending_record


def _run(coro, timeout=10.0):
    """Run a test coroutine; the engine loops exit the process on an error."""
    try:
        return asyncio.run(asyncio.wait_for(coro, timeout))
    except SystemExit as error:
        raise AssertionError("an engine loop exited the process") from error


class _VirtualGPU:
    """The forwards the scripted controller launched and the ones the host has waited for."""

    def __init__(self):
        self.launched = 0
        self.completed = set()

    def launch(self):
        self.launched += 1

    def wait(self, forward):
        self.completed.add(forward)


_STEP_MS = 10.0


def _fake_cuda_event(gpu, order):
    """`torch.cuda.Event` over the virtual GPU: `record()` binds the event to the last launched
    forward, `synchronize()` waits for that forward and records the launches issued by then,
    `elapsed_time()` reports `_STEP_MS` per forward up to the end event's."""

    class FakeEvent:
        def __init__(self, enable_timing=False, **_):
            self.forward = None

        def record(self, stream=None):
            self.forward = gpu.launched - 1

        def synchronize(self):
            gpu.wait(self.forward)
            order.append(("sync", self.forward, gpu.launched))

        def elapsed_time(self, other):
            return _STEP_MS * (other.forward + 1)

    return FakeEvent


def _stepping_engine(
    mode,
    requests,
    script,
    *,
    waiting=(),
    no_overlap_steps=(),
    order=None,
    gpu=None,
    log_every_step=False,
    hold=None,
):
    """An engine whose controller is a script: each step produces each row's next token.

    The scripted controller does what the real one does around the engine's hooks. An overlap
    step (and a legacy step) launches and runs `after_launch`, then waits for the consumed
    forward and asks the stop-word callback which rows finish; a step in `no_overlap_steps`
    waits and asks first, admits the `waiting` rows, then launches and runs the hook. Rows
    finish when forced or at their budget. A wait is a yield, or `hold[step]` when given;
    `gpu` is a virtual GPU whose forward k-1 completes when step k waits for it. `order`
    records the phases and every bookkeeping pass.

    Returns the engine, the live list of active request ids, and the step each row was
    forced at.
    """
    engine = _engine(requests)
    engine.state = EngineState.RUNNING
    engine.logging_step_interval = 1 if log_every_step else 0
    engine.metrics_writer = None
    engine.payload_stager = None
    engine.waiting_request_ids = deque(waiting)
    active = [request.request_id for request in requests if request.request_id not in waiting]
    context = engine.context
    context.config = SimpleNamespace(async_sched_mode=mode)
    context.step_count = 0
    context.prefix_cache_lru_clock = 0
    context.active_token_count = len(active)
    context.num_prefill_requests = 0
    context.dynamo_helper = None
    context.get_active_request_count = lambda: len(active)
    if log_every_step:
        # What a logged step's snapshot and console line read.
        context.max_requests = len(requests)
        context.total_request_count = len(active)
        context.paused_request_count = 0
        context.batch_dimensions = "bd"
        context.padded_batch_dimensions = "pbd"
        context.using_cuda_graph_this_step = lambda: False
        context.kv_block_allocator = SimpleNamespace(
            pool_size=8,
            pool_avail=8,
            paused_limit=0,
            enable_prefix_caching=False,
            get_total_used=lambda: 0,
            get_allocatable_count=lambda: 7,
            get_active_used=lambda: 0,
            get_paused_used=lambda: 0,
        )
    order = [] if order is None else order
    produced = dict.fromkeys(script, 0)
    forced_at = {}

    def budget(request_id):
        return engine.requests[request_id].record[-1].sampling_params.num_tokens_to_generate

    def launch(step, after_launch):
        if gpu is not None:
            gpu.launch()
        order.append(("launch", step))
        after_launch()

    async def wait(step):
        if hold is not None and step in hold:
            await hold[step].wait()
        else:
            await asyncio.sleep(0)
        if gpu is not None and step > 0:
            gpu.wait(step - 1)
        order.append(("wait", step))

    async def scripted_step(
        after_launch, schedule_waiting_requests=None, run_async_overlap=True, **_
    ):
        step = context.step_count
        order.append(("controller", step))
        if run_async_overlap:
            launch(step, after_launch)
        await wait(step)
        rows = list(active)
        sample = [script[request_id][produced[request_id]] for request_id in rows]
        for request_id in rows:
            produced[request_id] += 1
        forced = engine._get_and_clear_stop_word_finished_ids(rows)
        finished = [
            request_id
            for request_id in rows
            if request_id in forced or produced[request_id] >= budget(request_id)
        ]
        for request_id in finished:
            active.remove(request_id)
            if request_id in forced:
                forced_at[request_id] = step
        if not run_async_overlap:
            if schedule_waiting_requests is not None:
                order.append(("admit", step))
                schedule_waiting_requests()
            launch(step, after_launch)
        # The real controller yields once its launch is issued.
        await asyncio.sleep(0)
        order.append(("yield", step))
        output = {
            "active_request_ids": torch.tensor(rows),
            "finished_request_ids": torch.tensor(finished, dtype=torch.int64),
            "sample": torch.tensor(sample),
            "accepted_tokens": None,
            "log_probs": None,
            "top_n_logprobs": None,
            "cuda_graph_request_count": None,
        }
        return DynamicBatchControllerStepResult(
            decode_only=DecodeOnly(consumed=True, launched=run_async_overlap), output=output
        )

    bookkeep = engine._bookkeep

    def recording_bookkeep(record):
        order.append(("pass", record.context_state["step_count"]))
        return bookkeep(record)

    def admit():
        while engine.waiting_request_ids:
            active.append(engine.waiting_request_ids.popleft())

    engine._bookkeep = recording_bookkeep
    engine.controller.async_generate_output_tokens_dynamic_batch = scripted_step
    engine.schedule_waiting_requests = mock.Mock(side_effect=admit)
    engine._should_run_async_sched_overlap = lambda: context.step_count not in no_overlap_steps
    return engine, active, forced_at


def _loop_engine(mode, requests, script, **kwargs):
    """A stepping engine with what `run_engine` reads."""
    engine, active, forced_at = _stepping_engine(mode, requests, script, **kwargs)
    engine._cond = asyncio.Condition()
    engine._initialize_disaggregation_state()
    return engine, active, forced_at


async def _run_engine_until(engine, futures):
    """Run `run_engine` until the futures resolve, then cancel it; returns the results."""
    loop = asyncio.get_running_loop()
    for request_id in futures:
        futures[request_id] = engine.requests[request_id].future = loop.create_future()
    engine_task = asyncio.create_task(DynamicInferenceEngine.run_engine.__wrapped__(engine))
    results = await asyncio.gather(*futures.values())
    # The loop bookkept the last step before idling.
    assert engine._pending_record is None
    engine_task.cancel()
    await engine_task
    return dict(zip(futures, results))


@pytest.mark.parametrize("mode", list(AsyncScheduleMode), ids=lambda mode: mode.value)
def test_run_engine_bookkeeps_each_step_behind_the_next_launch(mode):
    """The loop leaves each step pending; its bookkeeping runs at the next step's hook,
    between that step's launch and its yield, and the last step's when the loop idles. Rows
    end as on main: a stop word or an explicit stop id completed by step k's tokens is forced
    at step k+1 (from the pending record on a no-overlap step, from the channel otherwise),
    the budget by the controller. A row admitted on a no-overlap step runs from the next
    forward. With every step logged, a record's end event is waited on at that hook: the
    consumed forward's, with the next forward already launched, never a later one; each
    token's latency entry is the step time of the record that produced it."""
    requests = [
        _request(stop_word_ids=[[8, 9]], request_id=7),
        _request(request_id=8, num_tokens_to_generate=3),
        _request(request_id=9, stop_token_ids=[42]),
        _request(request_id=10, num_tokens_to_generate=2),
    ]
    script = {7: [5, 8, 9, 10, 11], 8: [1, 2, 3], 9: [4, 42, 43, 44], 10: [6, 7]}
    order, gpu = [], _VirtualGPU()
    engine, _, forced_at = _loop_engine(
        mode,
        requests,
        script,
        waiting=[10],
        no_overlap_steps=[2],
        order=order,
        gpu=gpu,
        log_every_step=True,
    )

    memory_stats = dict.fromkeys(
        ("allocation.all.current", "allocated_bytes.all.current", "reserved_bytes.all.current"), 0
    )
    with (
        mock.patch.object(torch.cuda, "Event", _fake_cuda_event(gpu, order)),
        mock.patch.object(torch.cuda, "memory_stats", lambda: memory_stats),
    ):
        results = _run(_run_engine_until(engine, dict.fromkeys(script)))

    assert {request_id: request.generated_tokens for request_id, request in results.items()} == {
        7: [5, 8, 9],
        8: [1, 2, 3],
        9: [4, 42],
        10: [6, 7],
    }
    assert {request_id: request.finish_reason for request_id, request in results.items()} == {
        7: "stop",
        8: "length",
        9: "stop",
        10: "length",
    }
    assert forced_at == {9: 2, 7: 3}
    steps = 1 + max(step for kind, step in order if kind == "controller")
    for step in range(1, steps):
        # The pass of step k-1 runs after step k's launch and before its yield; on an overlap
        # step before its wait for the GPU. It waits on record k-1's end event with forward
        # k already launched.
        at = order.index(("pass", step - 1))
        before = (
            ("yield", step) if mode == AsyncScheduleMode.ASYNC and step == 2 else ("wait", step)
        )
        assert order.index(("launch", step)) < at < order.index(before), order
        assert order[at + 1] == ("sync", step - 1, step + 1), order[at : at + 2]
    # The last step's pass, when the loop idles, waits on its own forward.
    at = order.index(("pass", steps - 1))
    assert at > order.index(("yield", steps - 1))
    assert order[at + 1] == ("sync", steps - 1, steps)
    if mode == AsyncScheduleMode.ASYNC:
        # The no-overlap step admits before its launch; the previous step's pass follows it.
        at = order.index(("wait", 2))
        assert order[at : at + 4] == [("wait", 2), ("admit", 2), ("launch", 2), ("pass", 1)]
        assert engine.schedule_waiting_requests.call_count == 1
        admitted_at = 3  # row 10's first forward
    else:
        assert engine.schedule_waiting_requests.call_count == steps
        admitted_at = 0

    def step_time(step):
        return _STEP_MS * (step + 1) / 1e3

    assert {request_id: request.tpot for request_id, request in results.items()} == {
        7: [step_time(0), step_time(1), step_time(2)],
        8: [step_time(0), step_time(1), step_time(2)],
        9: [step_time(0), step_time(1)],
        10: [step_time(admitted_at), step_time(admitted_at + 1)],
    }
    assert engine.requests == {}
    assert engine.stop_word_finished_request_ids == engine.stop_word_being_finished_ids == set()


def test_direct_steps_and_a_cancelled_loop_bookkeep_the_pending_step():
    """`async_step`, the synchronous API, bookkeeps a step the loop left pending at its own
    hook and its own step right after, and returns its own result. A loop cancelled while a
    step waits for the GPU with the previous step pending still publishes that step."""
    hold = {3: asyncio.Event()}
    requests = [_request(request_id=7, num_tokens_to_generate=3), _request(request_id=8)]
    order = []
    engine, _, _ = _loop_engine(
        AsyncScheduleMode.ASYNC,
        requests,
        {7: [5, 6, 7], 8: [5, 6, 7, 8]},
        no_overlap_steps=[3],
        order=order,
        hold=hold,
    )

    async def run():
        await engine.async_forward()
        assert [request.generated_tokens for request in requests] == [[], []]
        result = await engine.async_step()
        assert [request.generated_tokens for request in requests] == [[5, 6], [5, 6]]
        assert engine._pending_record is None
        assert (result["active_request_ids"], result["finished_requests"]) == ([7, 8], [])
        # The loop: step 2 finishes row 7 at its budget and is left pending; step 3 holds at
        # its wait for the GPU; the cancel publishes step 2.
        future = engine.requests[7].future = asyncio.get_running_loop().create_future()
        engine_task = asyncio.create_task(DynamicInferenceEngine.run_engine.__wrapped__(engine))
        while order[-1:] != [("controller", 3)]:
            await asyncio.sleep(0)
        assert engine._pending_record is not None and not future.done()
        engine_task.cancel()
        await engine_task
        return future.result()

    finished = _run(run())
    assert finished.generated_tokens == [5, 6, 7]
    assert engine._pending_record is None
    assert order[-2:] == [("controller", 3), ("pass", 2)]


def _coordinator_loop_engine(script, tokens, *, stop_words=None, peer_work=1):
    """A stepping engine under the real coordinator loop.

    The loop's state comes from `script` by iteration. The consensus yields, as the ZMQ
    collective does, and reports the rank's own work, or `peer_work` once it has none. The
    consensus, the dummy forward, the world barrier and the shutdown record (kind, state,
    whether a step is pending) in the returned `order`, next to the bookkeeping passes.
    """
    request = _request(stop_word_ids=stop_words, request_id=7)
    order = []
    engine, _, _ = _stepping_engine(AsyncScheduleMode.ASYNC, [request], {7: tokens}, order=order)
    engine._initialize_disaggregation_state()
    engine.tp_ep_world_size = 2
    engine.ep_consensus_interval = 1
    engine._ep_consensus_loop_counter = 0
    engine._last_ep_consensus = (0, False)
    engine.use_synchronous_zmq_collectives = False
    engine._state_events = {state: asyncio.Event() for state in EngineState}
    engine.step_start_event = mock.Mock()
    engine.step_end_event = mock.Mock()
    iteration = count()

    def record(kind):
        order.append((kind, engine.state.name, engine._pending_record is not None))

    def schedule_requests():
        engine.state = script.get(next(iteration), engine.state)
        return 0

    async def consensus(local_work, signal_consensus):
        record("consensus")
        await asyncio.sleep(0)
        return (local_work or peer_work, engine.state == EngineState.PAUSING)

    async def barrier():
        record("barrier")

    async def shutdown():
        record("shutdown")
        engine.drain_bookkeeping()

    engine.controller.dummy_forward = lambda: record("dummy")
    engine.schedule_requests = schedule_requests
    engine._ep_establish_consensus = consensus
    engine._world_barrier = barrier
    engine.shutdown = shutdown
    return engine, request, order


_STOP = [("barrier", "STOPPING", False), ("shutdown", "STOPPING", False)]


@pytest.mark.parametrize(
    "script, tokens, stop_words, peer_work, expected_order, expected_tokens",
    [
        pytest.param(
            {
                1: EngineState.PAUSING,
                2: EngineState.SUSPENDING,
                3: EngineState.RESUMING,
                4: EngineState.STOPPING,
            },
            [5, 6],
            None,
            1,
            [
                ("consensus", "RUNNING", False),
                ("consensus", "PAUSING", True),
                ("pass", 0),
                ("barrier", "PAUSING", False),
                ("barrier", "SUSPENDING", False),
                ("barrier", "RESUMING", False),
                *_STOP,
            ],
            [5],
            id="pause-suspend-resume-stop",
        ),
        pytest.param(
            {
                1: EngineState.PAUSING,
                2: EngineState.UNPAUSING,
                4: EngineState.PAUSING,
                5: EngineState.STOPPING,
            },
            [5, 6, 7],
            None,
            1,
            [
                ("consensus", "RUNNING", False),
                ("consensus", "PAUSING", True),
                ("pass", 0),
                ("barrier", "PAUSING", False),
                ("barrier", "UNPAUSING", False),
                ("consensus", "RUNNING", False),
                ("consensus", "PAUSING", True),
                ("pass", 1),
                ("barrier", "PAUSING", False),
                *_STOP,
            ],
            [5, 6],
            id="pause-unpause-step-pause-stop",
        ),
        pytest.param(
            {3: EngineState.STOPPING},
            [5, 6],
            [[5]],
            1,
            [
                ("consensus", "RUNNING", False),
                ("consensus", "RUNNING", True),
                ("pass", 0),
                ("consensus", "RUNNING", True),
                ("dummy", "RUNNING", True),
                ("pass", 1),
                *_STOP,
            ],
            [5],
            id="dummy-forward",
        ),
        pytest.param(
            {3: EngineState.STOPPING},
            [5, 6],
            [[5]],
            0,
            [
                ("consensus", "RUNNING", False),
                ("consensus", "RUNNING", True),
                ("pass", 0),
                ("consensus", "RUNNING", True),
                ("pass", 1),
                *_STOP,
            ],
            [5],
            id="idle",
        ),
    ],
)
def test_coordinator_loop_bookkeeps_wherever_no_forward_follows(
    script, tokens, stop_words, peer_work, expected_order, expected_tokens
):
    """PAUSING, SUSPENDING and STOPPING bookkeep the pending step before their world barrier,
    RESUMING and UNPAUSING find none; a rank without work bookkeeps behind its dummy forward,
    or before it idles; the stop word the pending step completes ends the row."""
    engine, request, order = _coordinator_loop_engine(
        script, tokens, stop_words=stop_words, peer_work=peer_work
    )
    future = engine.requests[7].future

    async def run():
        run_loop = DynamicInferenceEngine.run_engine_with_coordinator.__wrapped__
        await run_loop(engine, loop=asyncio.get_running_loop())

    _run(run())

    assert engine.state == EngineState.STOPPING
    kinds = {"consensus", "barrier", "dummy", "shutdown", "pass"}
    assert [entry for entry in order if entry[0] in kinds] == expected_order
    assert request.generated_tokens == expected_tokens
    if stop_words is not None:
        assert future.result().finish_reason == "stop"
    assert engine._pending_record is None


def test_callback_forces_the_rows_the_pending_step_ends_and_each_pass_skips_its_own_forced_rows():
    """Before the launch, the stop-word callback finds the active rows whose pending tokens
    end them (a stop word, an explicit stop id, a stop inside a speculative burst) and skips
    the rows the pending step forced or finished, the partial prefill chunk and a plain token.
    Each pass then appends the tokens of every row but the ones its own step forced, so the
    rows forced meanwhile still get their last tokens, and the channel ends empty."""
    requests = [
        _request(request_id=1),  # plain, one token
        _request([], [[8]], request_id=2),  # a stop word
        _request(request_id=3, stop_token_ids=[42]),  # an explicit stop id
        _request([], [[6, 7]], request_id=4),  # a stop inside a speculative burst
        _request([5], [[7]], request_id=5),  # forced: its 7 is the extra token
        _request([5], request_id=6),  # finished by the controller, at its length
        _request([], [[8]], request_id=7),  # the partial prefill chunk
    ]
    engine = _engine(requests, num_speculative_tokens=2)
    engine.context.chunked_prefill_request_id = 7
    _record(
        engine,
        {1: 5, 2: 8, 3: 42, 4: [5, 6, 7], 5: 7, 6: 8, 7: 8},
        finished=[5, 6],
        forced=[5],
        chunked_prefill_request_id=7,
        step=0,
    )

    forced = engine._get_and_clear_stop_word_finished_ids([1, 2, 3, 4, 7])
    assert forced == {2, 3, 4}
    assert engine.stop_word_being_finished_ids == {2, 3, 4}

    # The pending step's pass, behind the launch: its own forced row keeps its history, the
    # rows forced just now get their tokens, and it reports them once more.
    finished = engine.drain_bookkeeping()["finished_requests"]
    assert [(request.request_id, request.generated_tokens) for request in finished] == [
        (5, [5]),
        (6, [5, 8]),
    ]
    assert [request.finish_reason for request in finished] == ["stop", "length"]
    assert [request.generated_tokens for request in requests[:4]] == [[5], [8], [42], [5, 6, 7]]
    assert requests[6].generated_tokens == []
    assert engine.stop_word_finished_request_ids == {2, 3, 4}

    # The forcing step's record: the forced rows finish on their stop, their extra tokens
    # dropped, and the channel forgets them.
    engine.context.chunked_prefill_request_id = -1
    _record(engine, {1: 9, 2: 99, 3: 99, 4: 99, 7: 9}, finished=[2, 3, 4], forced=[2, 3, 4], step=1)
    finished = engine.drain_bookkeeping()["finished_requests"]
    assert [(request.request_id, request.generated_tokens) for request in finished] == [
        (2, [8]),
        (3, [42]),
        (4, [5, 6, 7]),
    ]
    assert {request.finish_reason for request in finished} == {"stop"}
    assert requests[0].generated_tokens == [5, 9]
    assert requests[6].generated_tokens == [9]
    assert sorted(engine.requests) == [1, 7]
    assert engine.stop_word_finished_request_ids == set()


def test_abort_and_suspend_keep_the_pending_steps_tokens():
    """An abort with a step pending sets each budget to the appended tokens plus the pending
    ones, never more than it was: an evicted row, an active row, a row already at a smaller
    budget. `suspend()` then runs the pending pass before the buffers go: the tokens land
    within the budgets, the eviction requeues its row for the resume with its checkpointed
    tokens, and the stop those tokens complete is kept."""
    evicted = _request([8], [[8, 9]], request_id=1)
    active = _request([5], request_id=2)
    short = _request([5], request_id=3, num_tokens_to_generate=2)
    engine = _engine([evicted, active, short], num_speculative_tokens=1)
    engine.state = EngineState.RUNNING
    engine.unified_memory_level = 0
    engine.waiting_request_ids = deque()
    engine.resume_request_ids = None
    engine._invalidate_vision_state = mock.Mock()
    engine._notify_request_aborted = mock.Mock()
    engine.context.dynamo_helper = None
    engine.context.kv_cache_management_mode = KVCacheManagementMode.PERSIST
    engine.context.static_kv_memory_pointers = True
    engine.context.request_ids = torch.tensor([2, 3])
    engine.context.total_request_count = 2
    engine.context.request_output_lengths = torch.tensor([100, 100])
    engine.context.request_kv_length_offsets = torch.tensor([7, 7])
    engine.context.request_query_lengths = torch.tensor([1, 1])
    order = []
    engine.context.deallocate_inference_state_buffers = lambda: order.append("deallocate")
    record = _record(engine, {1: 9, 2: [6, 7], 3: [6, 7]})
    record.step_result["evict_request_ids"] = torch.tensor([1])
    bookkeep = engine._bookkeep
    engine._bookkeep = lambda record: (order.append("bookkeep"), bookkeep(record))[1]
    aborts = [
        [msgpack.packb([Headers.ABORT_REQUEST.value, request_id])] for request_id in (1, 2, 3, 99)
    ]
    engine.model_parallel_subscriber_socket = SimpleNamespace(
        recv_multipart=lambda: engine._pack_tp_broadcast(aborts)
    )
    engine._pending_signals = deque()
    engine._poll_pending_kv_imports = engine._poll_pending_kv_pushes = mock.Mock()

    assert engine.schedule_requests() == 4

    budgets = [
        request.sampling_params.num_tokens_to_generate for request in (evicted, active, short)
    ]
    assert budgets == [2, 3, 2]
    assert engine.context.request_output_lengths.tolist() == [8, 8]
    assert engine._notify_request_aborted.call_args_list == [
        mock.call(request_id, source_safe=True) for request_id in (1, 2, 3)
    ]
    assert engine._pending_record is record

    with mock.patch.object(
        DynamicInferenceEngine, "suspend_resume_ctx", lambda *args, **kwargs: nullcontext()
    ):
        engine.suspend()

    assert order == ["bookkeep", "deallocate"]
    assert engine._pending_record is None
    assert engine.state == EngineState.SUSPENDED
    assert engine.resume_request_ids == [1]
    assert engine.requests[1].record[0].generated_tokens == [8, 9]
    assert engine.requests[1].record[-1].generated_tokens == []
    assert engine.stop_word_finished_request_ids == {1}
    assert (active.generated_tokens, short.generated_tokens) == ([5, 6, 7], [5, 6])
