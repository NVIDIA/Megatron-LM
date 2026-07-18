# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Bounded read-ahead for a frozen MIMO encoder."""

from __future__ import annotations

import argparse
import logging
import threading
import time
from collections import deque
from collections.abc import Callable

import torch

from examples.mimo.training.batch import map_batch_tensors, move_batch_to_cuda

PREFETCHED_FEATURES_KEY = "_mimo_prefetched_encoder_features"
PROJECTION_TIMER_KEY = "_mimo_encoder_prefetch_projection_timer"

logger = logging.getLogger(__name__)
_debug_logger = logger.getChild("debug")


def add_encoder_prefetch_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("mimo encoder prefetch")
    group.add_argument(
        "--mimo-encoder-prefetch",
        action="store_true",
        help="Prefetch completed features from a frozen encoder on encoder ranks.",
    )
    group.add_argument(
        "--mimo-encoder-prefetch-depth",
        type=int,
        default=2,
        help="Target number of completed encoder-feature batches kept ready.",
    )
    group.add_argument(
        "--mimo-encoder-prefetch-debug",
        action="store_true",
        help="Log per-batch encoder-prefetch timing and queue diagnostics.",
    )
    return parser


def validate_encoder_prefetch_args(args) -> None:
    if not args.mimo_encoder_prefetch:
        return
    if not args.freeze_vit:
        raise ValueError("encoder prefetch requires --freeze-vit")
    if args.freeze_projection:
        raise ValueError("encoder prefetch requires a trainable projection")
    for field, label in (
        ("encoder_tp", "TP"),
        ("encoder_cp", "CP"),
        ("encoder_pp", "PP"),
        ("encoder_ep", "EP"),
    ):
        if getattr(args, field, 1) != 1:
            raise ValueError(f"encoder prefetch requires encoder {label}=1")
    if args.mimo_encoder_prefetch_depth <= 0:
        raise ValueError("encoder prefetch depth must be positive")
    if args.rerun_mode != "disabled":
        raise ValueError("encoder prefetch does not support rerun modes")


def prefetch_frozen_features(
    module: torch.nn.Module, encoder_inputs: dict[str, object]
) -> torch.Tensor:
    with torch.no_grad():
        return module.combine_embeddings(module.encode(encoder_inputs))


def _record_batch_stream(value, stream: torch.cuda.Stream) -> None:
    def record(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_cuda:
            tensor.record_stream(stream)
        return tensor

    map_batch_tensors(value, record)


def _log_producer_debug(
    batch_id: int,
    data_fetch_ms: float,
    encode_start: torch.cuda.Event,
    encode_end: torch.cuda.Event,
) -> None:
    _debug_logger.info(
        "encoder-prefetch-debug producer batch=%d data_fetch_ms=%.3f encode_ms=%.3f",
        batch_id,
        data_fetch_ms,
        encode_start.elapsed_time(encode_end),
    )


def _log_consumer_debug(
    batch_id: int, ready_at_request: int, depth: int, claimed_pending: bool, wait_start: float
) -> None:
    _debug_logger.info(
        "encoder-prefetch-debug consumer batch=%d ready_at_request=%d/%d "
        "claimed_pending=%d pop_wait_ms=%.3f",
        batch_id,
        ready_at_request,
        depth,
        claimed_pending,
        (time.perf_counter() - wait_start) * 1000,
    )


def _log_encoder_wait_debug(
    batch_id: int, start_event: torch.cuda.Event, end_event: torch.cuda.Event
) -> None:
    _debug_logger.info(
        "encoder-prefetch-debug consumer-wait batch=%d encoder_wait_ms=%.3f",
        batch_id,
        start_event.elapsed_time(end_event),
    )


def _log_projection_debug(
    batch_id: int, start_event: torch.cuda.Event, end_event: torch.cuda.Event
) -> None:
    _debug_logger.info(
        "encoder-prefetch-debug projection batch=%d projection_ms=%.3f",
        batch_id,
        start_event.elapsed_time(end_event),
    )


class _ProjectionTimer:
    def __init__(self, loader: EncoderPrefetchLoader, batch_id: int) -> None:
        self._loader = loader
        self._batch_id = batch_id
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self._start_event.record(torch.cuda.current_stream())
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self._end_event.record(torch.cuda.current_stream())
        self._loader._queue_projection_timing(self._batch_id, self._start_event, self._end_event)


class EncoderPrefetchLoader:
    """Keep completed encoder features ready without delaying projection."""

    def __init__(
        self,
        *,
        source,
        encoder_name: str,
        feature_producer: Callable[[dict[str, object]], torch.Tensor],
        depth: int,
        stream: torch.cuda.Stream | None = None,
        worker_join_timeout_s: float = 30.0,
        debug: bool = False,
    ) -> None:
        if depth <= 0:
            raise ValueError("encoder prefetch depth must be positive")
        if worker_join_timeout_s <= 0:
            raise ValueError("worker_join_timeout_s must be positive")
        self._source = iter(source)
        self._encoder_name = encoder_name
        self._feature_producer = feature_producer
        self._depth = depth
        self._stream = stream
        self._worker_join_timeout_s = worker_join_timeout_s
        self._debug = debug
        if debug:
            _debug_logger.setLevel(logging.INFO)
        self._condition = threading.Condition()
        self._ready: deque[dict[str, object]] = deque()
        self._pending: tuple[dict[str, object], torch.cuda.Event] | None = None
        self._in_flight = False
        self._encoder_wait_timings: deque[tuple[int, torch.cuda.Event, torch.cuda.Event]] = deque()
        self._projection_timings: deque[tuple[int, torch.cuda.Event, torch.cuda.Event]] = deque()
        self._produced_batches = 0
        self._consumed_batches = 0
        self._producer_error: BaseException | None = None
        self._source_exhausted = False
        self._stop = False
        self._worker: threading.Thread | None = None
        self._device: int | None = None
        self._closed = False

    def __iter__(self):
        return self

    def start(self) -> None:
        with self._condition:
            if self._worker is not None:
                raise RuntimeError("encoder prefetch loader is already started")
            if self._closed:
                raise RuntimeError("cannot start a closed encoder prefetch loader")
            self._device = torch.cuda.current_device()
            if self._stream is None:
                self._stream = torch.cuda.Stream()
            setup_event = torch.cuda.Event()
            setup_event.record(torch.cuda.current_stream())
            self._stream.wait_event(setup_event)
            self._worker = threading.Thread(
                target=self._producer_main, name=f"mimo-{self._encoder_name}-prefetch", daemon=True
            )
            self._worker.start()

    def _producer_main(self) -> None:
        staged_batch = None
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._stop
                    or self._producer_error is not None
                    or self._source_exhausted
                    or (not self._in_flight and len(self._ready) < self._depth)
                )
                if self._stop or self._producer_error is not None or self._source_exhausted:
                    return
                self._in_flight = True

            source_exhausted = False
            source_error = None
            data_fetch_ms = 0.0
            try:
                if staged_batch is None:
                    batch = next(self._source)
                else:
                    batch = staged_batch
                    staged_batch = None
                item, completion_event, encode_start = self._enqueue_batch(batch)

                with self._condition:
                    if not self._stop:
                        self._pending = (item, completion_event)
                        self._condition.notify_all()

                with self._condition:
                    should_read_ahead = not self._stop
                if should_read_ahead:
                    # Stage one CPU batch while this GPU encode runs; enqueue it later.
                    # This advances the source by one batch beyond the trained cursor.
                    data_fetch_start = time.perf_counter() if self._debug else 0.0
                    try:
                        staged_batch = next(self._source)
                    except StopIteration:
                        source_exhausted = True
                    except BaseException as error:
                        source_error = error
                    if self._debug:
                        data_fetch_ms = (time.perf_counter() - data_fetch_start) * 1000
                completion_event.synchronize()
            except StopIteration:
                with self._condition:
                    self._source_exhausted = True
                    self._in_flight = False
                    self._condition.notify_all()
                return
            except BaseException as error:
                with self._condition:
                    if not self._stop:
                        self._producer_error = error
                    self._in_flight = False
                    self._pending = None
                    self._condition.notify_all()
                return

            with self._condition:
                self._in_flight = False
                if self._stop:
                    return
                if self._pending is not None:
                    self._ready.append(self._pending[0])
                    self._pending = None
                batch_id = self._produced_batches
                self._produced_batches += 1
                if source_exhausted:
                    self._source_exhausted = True
                if source_error is not None:
                    self._producer_error = source_error
                self._condition.notify_all()
                terminate = source_exhausted or source_error is not None
            if self._debug:
                assert encode_start is not None
                _log_producer_debug(batch_id, data_fetch_ms, encode_start, completion_event)
                self._drain_encoder_wait_timings()
                self._drain_projection_timings()
            if terminate:
                return

    def _enqueue_batch(
        self, batch: dict[str, object]
    ) -> tuple[dict[str, object], torch.cuda.Event, torch.cuda.Event | None]:
        if not isinstance(batch, dict):
            raise TypeError("encoder prefetch source must return a batch dictionary")
        modality_inputs = batch.get("modality_inputs")
        if not isinstance(modality_inputs, dict) or self._encoder_name not in modality_inputs:
            raise ValueError(f"batch has no inputs for encoder {self._encoder_name!r}")

        with torch.cuda.device(self._device), torch.cuda.stream(self._stream):
            # Encoder ranks intentionally retain only fields consumed by their forward step.
            output_batch = {"input_ids": batch["input_ids"]}
            encoder_inputs = move_batch_to_cuda(modality_inputs[self._encoder_name])
            encode_start = torch.cuda.Event(enable_timing=True) if self._debug else None
            if encode_start is not None:
                encode_start.record(self._stream)
            encoded = self._feature_producer(encoder_inputs)
            if not isinstance(encoded, torch.Tensor):
                raise TypeError("feature_producer must return one combined tensor")
            output_batch[PREFETCHED_FEATURES_KEY] = {self._encoder_name: encoded}
            completion_event = torch.cuda.Event(enable_timing=self._debug)
            completion_event.record(self._stream)
        return output_batch, completion_event, encode_start

    def __next__(self) -> dict[str, object]:
        if self._worker is None:
            raise RuntimeError("encoder prefetch loader must be started before use")
        with self._condition:
            ready_at_request = len(self._ready)
            wait_start = time.perf_counter() if self._debug else 0.0
            self._condition.wait_for(
                lambda: self._stop
                or self._producer_error is not None
                or self._ready
                or self._pending is not None
                or (self._source_exhausted and not self._ready and self._pending is None)
            )
            if self._stop:
                raise StopIteration
            completion_event = None
            if self._ready:
                item = self._ready.popleft()
                batch_id = self._consumed_batches
                self._consumed_batches += 1
                self._condition.notify_all()
            elif self._pending is not None:
                item, completion_event = self._pending
                self._pending = None
                batch_id = self._consumed_batches
                self._consumed_batches += 1
                self._condition.notify_all()
            elif self._producer_error is not None:
                raise RuntimeError("encoder prefetch producer failed") from self._producer_error
            else:
                raise StopIteration

        current_stream = torch.cuda.current_stream()
        if completion_event is not None:
            wait_start_event = torch.cuda.Event(enable_timing=True) if self._debug else None
            wait_end_event = torch.cuda.Event(enable_timing=True) if self._debug else None
            if wait_start_event is not None:
                wait_start_event.record(current_stream)
            current_stream.wait_event(completion_event)
            if wait_end_event is not None:
                wait_end_event.record(current_stream)
                self._queue_encoder_wait_timing(batch_id, wait_start_event, wait_end_event)
        _record_batch_stream(item, current_stream)
        if self._debug:
            item[PROJECTION_TIMER_KEY] = _ProjectionTimer(self, batch_id)
            _log_consumer_debug(
                batch_id, ready_at_request, self._depth, completion_event is not None, wait_start
            )
        return item

    def _queue_encoder_wait_timing(
        self, batch_id: int, start_event: torch.cuda.Event, end_event: torch.cuda.Event
    ) -> None:
        with self._condition:
            self._encoder_wait_timings.append((batch_id, start_event, end_event))

    def _drain_encoder_wait_timings(self) -> None:
        ready = []
        with self._condition:
            while self._encoder_wait_timings and self._encoder_wait_timings[0][2].query():
                ready.append(self._encoder_wait_timings.popleft())
        for batch_id, start_event, end_event in ready:
            _log_encoder_wait_debug(batch_id, start_event, end_event)

    def _queue_projection_timing(
        self, batch_id: int, start_event: torch.cuda.Event, end_event: torch.cuda.Event
    ) -> None:
        with self._condition:
            self._projection_timings.append((batch_id, start_event, end_event))

    def _drain_projection_timings(self) -> None:
        ready = []
        with self._condition:
            while self._projection_timings and self._projection_timings[0][2].query():
                ready.append(self._projection_timings.popleft())
        for batch_id, start_event, end_event in ready:
            _log_projection_debug(batch_id, start_event, end_event)

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._stop = True
            self._ready.clear()
            self._pending = None
            self._condition.notify_all()
            worker = self._worker
        if worker is not None:
            worker.join(timeout=self._worker_join_timeout_s)
            if worker.is_alive():
                logger.warning(
                    "encoder prefetch worker did not stop within %.2f seconds",
                    self._worker_join_timeout_s,
                )
        if self._debug:
            self._drain_encoder_wait_timings()
            self._drain_projection_timings()
