# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Private engine-event transport between Megatron rank zero and its parent."""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable

import msgpack
import zmq


class EngineEventReporter:
    """Queue rank-zero engine events for the parent-owned socket."""

    def __init__(self, engine, parent_address: str) -> None:
        self.engine = engine
        self.parent_address = parent_address
        # The forward path must never wait for the event transport. SimpleQueue is
        # thread-safe and unbounded; the publisher thread owns all ZMQ operations.
        self._events = queue.SimpleQueue()
        self._ready = queue.Queue(maxsize=1)
        self._thread = None

    def start(self) -> None:
        if self.engine.rank != 0:
            return
        self.engine.context.dynamo_helper.add_kv_event_listener(self.observe)
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="megatron-engine-event-reporter"
        )
        self._thread.start()
        result = self._ready.get()
        if isinstance(result, BaseException):
            self._thread.join()
            self._thread = None
            raise RuntimeError("Failed to start engine event reporter") from result

    def observe(self, kind: str, payload: dict) -> None:
        self._events.put((kind, payload))

    def _run(self) -> None:
        context = None
        socket = None
        # `start()` takes the readiness item, so the queue's emptiness cannot tell a
        # startup failure from a later one.
        started = False
        try:
            context = zmq.Context()
            socket = context.socket(zmq.PUSH)
            socket.setsockopt(zmq.LINGER, 0)
            socket.connect(self.parent_address)
            self._ready.put(True)
            started = True
            while True:
                event = self._events.get()
                if event is None:
                    break
                kind, payload = event
                try:
                    frame = msgpack.packb([kind, payload], use_bin_type=True)
                except Exception:
                    # One event the parent cannot receive must not silence later ones.
                    logging.exception("Dropping engine event %r that cannot be serialized", kind)
                    continue
                socket.send(frame)
        except Exception as exc:
            if started:
                logging.exception("Engine event reporter failed")
            else:
                self._ready.put(exc)
        finally:
            if socket is not None:
                socket.close()
            if context is not None:
                context.term()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._events.put(None)
        self._thread.join(timeout=1.0)
        if self._thread.is_alive():
            logging.warning("Engine event reporter did not stop within 1 second")
        self._thread = None


class EngineEventReceiver:
    """Own the parent endpoint receiving rank-zero engine events."""

    def __init__(
        self, callback: Callable[[str, dict], None], bind_host: str, bind_port: int | None = None
    ) -> None:
        self.callback = callback
        self.bind_host = bind_host
        self.bind_port = bind_port
        self._stop = threading.Event()
        self._ready = queue.Queue(maxsize=1)
        self._thread = None

    def start(self) -> str:
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="megatron-engine-event-receiver"
        )
        self._thread.start()
        result = self._ready.get()
        if isinstance(result, BaseException):
            self._thread.join()
            self._thread = None
            raise RuntimeError("Failed to start engine event receiver") from result
        return result

    def _run(self) -> None:
        context = None
        socket = None
        # `start()` takes the readiness item, so the queue's emptiness cannot tell a
        # startup failure from a later one.
        started = False
        try:
            context = zmq.Context()
            socket = context.socket(zmq.PULL)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVTIMEO, 100)
            if self.bind_port is None:
                socket.bind_to_random_port(f"tcp://{self.bind_host}")
            else:
                socket.bind(f"tcp://{self.bind_host}:{self.bind_port}")
            self._ready.put(socket.getsockopt_string(zmq.LAST_ENDPOINT))
            started = True
            while not self._stop.is_set():
                try:
                    frame = socket.recv()
                except zmq.Again:
                    continue
                # A malformed message, or a failure handling one, must not stop the
                # readiness, KV and progress events that follow it.
                try:
                    kind, payload = msgpack.unpackb(frame, raw=False)
                except Exception:
                    logging.exception("Ignoring a malformed engine event")
                    continue
                try:
                    self.callback(str(kind), payload)
                except Exception:
                    logging.exception("Failed to handle engine event %r", kind)
        except Exception as exc:
            if started:
                logging.exception("Engine event receiver failed")
            else:
                self._ready.put(exc)
        finally:
            if socket is not None:
                socket.close()
            if context is not None:
                context.term()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
