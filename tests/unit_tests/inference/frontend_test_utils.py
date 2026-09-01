# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Helpers for standalone frontend/coordinator tests.

The HTTP frontend, the InferenceClient and the coordinator are all independent of
the model: the coordinator needs only a tokenizer, and engines are reached over
ZMQ. These helpers exploit that to drive the entire frontend path with no
checkpoint, no GPU and no torch.distributed -- a real coordinator process plus
fake engines that speak the wire protocol.

Consumed by the OpenAI-compatibility and packet-size unit tests, and by
tests/performance_tests/client/frontend_capacity_benchmark.py.
"""

import asyncio
import heapq
import multiprocessing
import threading
import time
from contextlib import contextmanager

import msgpack
import torch
import zmq

from megatron.core.inference.async_stream import AsyncStream
from megatron.core.inference.data_parallel_inference_coordinator.coordinator import (
    DataParallelInferenceCoordinator,
)
from megatron.core.inference.headers import Headers
from megatron.core.inference.inference_request import DynamicInferenceRequest, Status
from megatron.core.inference.sampling_params import SamplingParams

# Token ids 0-255 are literal bytes; specials live above that.
_EOD_ID = 256
_BOS_ID = 257


class ByteTokenizer:
    """Byte-level tokenizer: reversible, dependency-free, and picklable.

    One token per UTF-8 byte. Being exactly reversible keeps assertions about
    detokenized text meaningful, and a token count that equals the byte count
    makes payload sizes predictable for the packet-size tests. Picklable because
    the coordinator runs in a spawned process and receives the tokenizer as an
    argument.
    """

    eod = _EOD_ID
    bos = _BOS_ID
    eos_id = _EOD_ID
    vocab_size = _BOS_ID + 1

    def tokenize(self, text):
        """Return one token id per UTF-8 byte of ``text``."""
        return list(text.encode("utf-8"))

    def detokenize(self, token_ids):
        """Inverse of tokenize; special ids are dropped."""
        return bytes(t for t in token_ids if t < 256).decode("utf-8", errors="replace")


class _HuggingFaceHandle:
    """The nested handle HuggingFaceFastIncrementalDetokenizer reaches through."""

    include_special_tokens = True

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


class ByteLevelFastTokenizer:
    """A byte-level Hugging Face fast tokenizer for the HTTP frontend.

    Streaming responses go through HuggingFaceFastIncrementalDetokenizer, which
    requires a real PreTrainedTokenizerFast reachable at
    ``tokenizer._tokenizer.tokenizer``. ByteTokenizer cannot satisfy that, so
    streaming tests use this instead; it stays byte-level so detokenization is
    still exactly reversible.

    The vocabulary is built deterministically from the byte-level alphabet, so
    pickling can discard it and rebuild an identical tokenizer on the far side.
    That matters because the coordinator runs in a spawned process and must
    agree with the frontend on token ids, and because the perf benchmark reports
    payload byte counts, which move if a token's id changes width.
    """

    # No chat template: chat_completions then tokenizes the joined message
    # contents, which keeps these tests independent of any model's template.
    chat_template = None
    bos = None
    eod = None

    def __init__(self):
        self._build()

    def _build(self):
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast

        # ByteLevel.alphabet() comes back from a Rust hash set, so its order
        # differs on every call and every process. Sorting is what makes this
        # tokenizer reproducible; without it each process invents its own id for
        # a given byte.
        alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
        backend = Tokenizer(
            models.BPE(vocab={token: i for i, token in enumerate(alphabet)}, merges=[])
        )
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        backend.decoder = decoders.ByteLevel()
        self._huggingface_tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
        self._tokenizer = _HuggingFaceHandle(self._huggingface_tokenizer)
        self.eos_id = self._huggingface_tokenizer.eos_token_id

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        self._build()

    @property
    def vocab_size(self):
        """Number of tokens in the backing vocabulary."""
        return self._huggingface_tokenizer.vocab_size

    def tokenize(self, text):
        """Encode text to token ids without special tokens."""
        return self._huggingface_tokenizer.encode(text, add_special_tokens=False)

    def detokenize(self, token_ids, skip_special_tokens=True):
        """Decode token ids back to text."""
        return self._huggingface_tokenizer.decode(
            list(token_ids), skip_special_tokens=skip_special_tokens
        )


def make_byte_level_fast_tokenizer():
    """Return a byte-level Hugging Face fast tokenizer for the HTTP frontend."""
    return ByteLevelFastTokenizer()


# Sample text used to synthesize generated tokens. Encoding real text with the
# tokenizer under test (rather than picking token ids directly) guarantees the
# ids decode to valid UTF-8, so streamed deltas and batch-decoded text agree.
_SAMPLE_TEXT = "the quick brown fox jumps over the lazy dog "


def synthesize_generated_tokens(num_tokens, tokenizer):
    """Return ``num_tokens`` token ids that decode to valid text."""
    sample = tokenizer.tokenize(_SAMPLE_TEXT)
    if not sample:
        raise ValueError("tokenizer produced no tokens for the sample text")
    return [sample[i % len(sample)] for i in range(num_tokens)]


def build_engine_reply(
    request_id,
    prompt_tokens,
    generated_tokens,
    sampling_params,
    status=Status.COMPLETED,
    generated_log_probs=None,
):
    """Build the serialized reply an engine would send for a finished request.

    Goes through the real DynamicInferenceRequest.serialize() rather than
    hand-writing a dict, so tests fail if the reply schema drifts.

    Args:
        request_id (int): Coordinator-side request id, echoed back for routing.
        prompt_tokens (list[int]): The request's prompt tokens.
        generated_tokens (list[int]): Tokens to report as generated.
        sampling_params (SamplingParams): The request's sampling params.
        status (Status): Terminal status to report.
        generated_log_probs (list[float] | None): Optional per-token log probs.

    Returns:
        dict: A msgpack-serializable reply payload.
    """
    request = DynamicInferenceRequest(
        request_id=request_id,
        prompt_tokens=(
            None if not prompt_tokens else torch.tensor(list(prompt_tokens), dtype=torch.int64)
        ),
        sampling_params=sampling_params,
        status=status,
        generated_tokens=list(generated_tokens),
        generated_length=len(generated_tokens),
        generated_log_probs=generated_log_probs,
    )
    return request.serialize()


class FakeEngine:
    """A ZMQ DEALER that impersonates a data parallel engine.

    Registers with a coordinator, answers SUBMIT_REQUEST with a canned reply of
    ``num_output_tokens`` tokens, and honours control signals well enough to shut
    down cleanly. Deliberately has no model, no torch.distributed and no CUDA:
    the point is to measure the frontend and coordinator in isolation, at a rate
    a real engine could never sustain.
    """

    def __init__(
        self,
        coordinator_addr,
        identity,
        num_output_tokens=8,
        context=None,
        reply_log_probs=False,
        tokenizer=None,
        reply_delay_s=0.0,
    ):
        self.coordinator_addr = coordinator_addr
        self.identity = identity if isinstance(identity, bytes) else identity.encode()
        self.num_output_tokens = num_output_tokens
        self.reply_log_probs = reply_log_probs
        self.tokenizer = tokenizer or ByteTokenizer()
        # Holding replies for a while lets many requests be genuinely in flight
        # at once, which is what makes a concurrency sweep meaningful. Replies are
        # queued and released by the poll loop, so one thread still serves many
        # overlapping requests.
        self.reply_delay_s = reply_delay_s
        self._pending_replies = []
        self._reply_sequence = 0
        self._owns_context = context is None
        self.context = context or zmq.Context()
        self.socket = None
        self._thread = None
        self._stop = threading.Event()
        # Requests served, for capacity accounting.
        self.num_requests_served = 0

    def start(self):
        """Connect, register with the coordinator, and start serving."""
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.IDENTITY, self.identity)
        self.socket.setsockopt(zmq.SNDHWM, 0)
        self.socket.setsockopt(zmq.RCVHWM, 0)
        self.socket.connect(self.coordinator_addr)
        # An empty payload registers this engine with the coordinator.
        self.socket.send(b"")
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        while not self._stop.is_set():
            # Poll briefly while replies are queued so they are released on time.
            timeout_ms = 1 if self._pending_replies else 50
            if dict(poller.poll(timeout=timeout_ms)):
                # Drain everything already queued rather than one message per
                # iteration, so the engine never becomes the bottleneck in a
                # capacity sweep.
                while True:
                    try:
                        raw = self.socket.recv(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    self._handle(msgpack.unpackb(raw, raw=False))
                    if self._stop.is_set():
                        return
            self._flush_due_replies()

    def _flush_due_replies(self):
        now = time.monotonic()
        # Min-heap on due time: only the frames that are actually due are
        # touched, so a large in-flight backlog costs nothing per iteration.
        while self._pending_replies and self._pending_replies[0][0] <= now:
            _, _, payload, is_final = heapq.heappop(self._pending_replies)
            self._send_frame(payload, is_final)

    def _handle(self, payload):
        header = Headers(payload[0])
        if header == Headers.SUBMIT_REQUEST:
            request_id, prompt, serialized_params = payload[1:4]
            self._reply(request_id, prompt, serialized_params)
        elif header in (Headers.STOP, Headers.SHUTDOWN):
            self._stop.set()
        # Everything else (pause/resume/profiler signals) needs no engine-side
        # state for these tests.

    def _reply(self, request_id, prompt, serialized_params):
        sampling_params = SamplingParams.deserialize(serialized_params)
        num_tokens = sampling_params.num_tokens_to_generate or self.num_output_tokens
        generated_tokens = synthesize_generated_tokens(num_tokens, self.tokenizer)
        prompt_tokens = prompt if isinstance(prompt, list) else []
        reply = build_engine_reply(
            request_id,
            prompt_tokens,
            generated_tokens,
            sampling_params,
            generated_log_probs=(
                [-0.5] * len(generated_tokens)
                if (self.reply_log_probs or sampling_params.return_log_probs)
                else None
            ),
        )
        # Streaming requests get one partial per token, spread over the reply
        # delay, so the frontend pays its real per-token streaming cost (a
        # detokenize step and an SSE flush) instead of seeing one lump reply.
        frames = []
        if sampling_params.streaming:
            step = self.reply_delay_s / max(1, len(generated_tokens))
            frames.extend(
                (
                    step * (index + 1),
                    [
                        Headers.ENGINE_REPLY_PARTIAL.value,
                        [{"request_id": request_id, "new_tokens": [token]}],
                    ],
                    False,
                )
                for index, token in enumerate(generated_tokens)
            )
        frames.append((self.reply_delay_s, [Headers.ENGINE_REPLY.value, [reply]], True))

        for delay, payload, is_final in frames:
            if delay > 0:
                # The counter breaks ties so heapq never has to order the payloads.
                self._reply_sequence += 1
                heapq.heappush(
                    self._pending_replies,
                    (time.monotonic() + delay, self._reply_sequence, payload, is_final),
                )
            else:
                self._send_frame(payload, is_final)

    def _send_frame(self, payload, is_final):
        self.socket.send(msgpack.packb(payload, use_bin_type=True))
        if is_final:
            self.num_requests_served += 1

    def stop(self):
        """Stop serving and release the socket."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self.socket is not None and not self.socket.closed:
            self.socket.close(linger=0)
            self.socket = None
        if self._owns_context:
            self.context.term()


@contextmanager
def standalone_coordinator(max_requests=256, port=None, tokenizer=None, num_engines=1, **kwargs):
    """Run a real coordinator process with fake engines attached.

    The coordinator is spawned with ``data_parallel_size=0`` so it does not block
    waiting for engines; the fake engines then register through the
    empty-payload path, exactly as real engines do when they reconnect.

    Args:
        max_requests (int): Coordinator's per-rank max request count.
        port (int | None): Port to bind; None picks a random free port.
        tokenizer: Tokenizer for the coordinator; defaults to ByteTokenizer.
        num_engines (int): Number of fake engines to attach.
        **kwargs: Forwarded to FakeEngine.

    Yields:
        tuple[str, list[FakeEngine]]: The coordinator address and the engines.
    """
    spawn_context = multiprocessing.get_context('spawn')
    pipe_parent, pipe_child = spawn_context.Pipe()
    ready_event = spawn_context.Event()
    proc = spawn_context.Process(
        target=DataParallelInferenceCoordinator.entrypoint,
        kwargs={
            "pipe_connection": pipe_child,
            "ready_event": ready_event,
            "data_parallel_size": 0,
            "tokenizer": tokenizer or ByteTokenizer(),
            "max_requests": max_requests,
            "inference_coordinator_port": port,
            "hostname": "127.0.0.1",
        },
    )
    proc.start()

    addr = None
    engines = []
    try:
        while not pipe_parent.poll(timeout=0.1):
            assert proc.is_alive(), "Coordinator process died during init"
        addr = pipe_parent.recv()
        pipe_parent.close()
        assert ready_event.wait(timeout=30.0), "Coordinator never signalled ready"

        for i in range(num_engines):
            engine = FakeEngine(addr, f"fake-engine-{i}", **kwargs)
            engine.start()
            engines.append(engine)

        yield addr, engines
    finally:
        for engine in engines:
            engine.stop()
        _shutdown_coordinator(proc, addr)


def _shutdown_coordinator(proc, addr):
    """Ask the coordinator to exit its loop, then reap the process."""
    if proc is None or not proc.is_alive():
        return
    if addr is not None:
        context = zmq.Context()
        socket = context.socket(zmq.DEALER)
        socket.connect(addr)
        socket.send(msgpack.packb([Headers.CONNECT.value], use_bin_type=True))
        if socket.poll(timeout=5000):
            socket.recv()  # CONNECT_ACK
            socket.send(msgpack.packb([Headers.SHUTDOWN.value], use_bin_type=True))
        socket.close(linger=1000)
        context.term()
    proc.join(timeout=10.0)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5.0)


class FakeInferenceClient:
    """Stands in for InferenceClient so HTTP endpoints can be tested alone.

    Implements the two methods the endpoints call (add_request and
    add_request_streaming) and records what it was asked for, so tests can assert
    on how request fields were translated into SamplingParams. Replies come from
    the same serializer the engine uses.
    """

    def __init__(
        self,
        num_output_tokens=4,
        generated_log_probs=False,
        status=Status.COMPLETED,
        tokenizer=None,
    ):
        self.num_output_tokens = num_output_tokens
        self.generated_log_probs = generated_log_probs
        self.status = status
        # Used to fill generated_text, which the coordinator normally detokenizes.
        # Sharing the app's tokenizer keeps streamed deltas and the non-streaming
        # text consistent.
        self.tokenizer = tokenizer or ByteTokenizer()
        self.next_request_id = 0
        # (prompt, sampling_params) for every submitted request, in order.
        self.submissions = []
        # Set to override the next reply, e.g. to simulate a failed request.
        self.reply_override = None

    def _build_reply(self, prompt, sampling_params):
        request_id = self.next_request_id
        self.next_request_id += 1
        self.submissions.append((prompt, sampling_params))
        if self.reply_override is not None:
            return self.reply_override
        num_tokens = sampling_params.num_tokens_to_generate or self.num_output_tokens
        generated_tokens = synthesize_generated_tokens(num_tokens, self.tokenizer)
        reply = build_engine_reply(
            request_id,
            prompt if isinstance(prompt, list) else [],
            generated_tokens,
            sampling_params,
            status=self.status,
            generated_log_probs=(
                [-0.5] * len(generated_tokens)
                if (self.generated_log_probs or sampling_params.return_log_probs)
                else None
            ),
        )
        # The coordinator detokenizes final replies before the client sees them.
        reply["generated_text"] = self.tokenizer.detokenize(generated_tokens)
        return reply

    def add_request(self, prompt, sampling_params, multi_modal_data=None):
        """Return an already-resolved future holding a canned reply."""
        del multi_modal_data
        future = asyncio.get_running_loop().create_future()
        future.set_result(self._build_reply(prompt, sampling_params))
        return future

    def add_request_streaming(self, prompt, sampling_params, multi_modal_data=None):
        """Return a stream that yields one partial per token, then the final reply."""
        del multi_modal_data
        sampling_params.streaming = True
        reply = self._build_reply(prompt, sampling_params)
        request_id = reply["request_id"]
        stream = AsyncStream(request_id, lambda: None)
        for index, token in enumerate(reply["generated_tokens"]):
            stream.put({"partial": {"request_id": request_id, "new_tokens": [token]}})
        stream.put({"final": reply})
        stream.finish()
        return stream
