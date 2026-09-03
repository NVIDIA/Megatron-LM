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
            target=self._run,
            daemon=True,
            name="megatron-engine-event-reporter",
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
        try:
            context = zmq.Context()
            socket = context.socket(zmq.PUSH)
            socket.setsockopt(zmq.LINGER, 0)
            socket.connect(self.parent_address)
            self._ready.put(True)
            while True:
                event = self._events.get()
                if event is None:
                    break
                kind, payload = event
                socket.send(msgpack.packb([kind, payload], use_bin_type=True))
        except Exception as exc:
            if self._ready.empty():
                self._ready.put(exc)
            else:
                logging.exception("Engine event reporter failed")
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

    def __init__(self, callback: Callable[[str, dict], None], bind_host: str) -> None:
        self.callback = callback
        self.bind_host = bind_host
        self._stop = threading.Event()
        self._ready = queue.Queue(maxsize=1)
        self._thread = None

    def start(self) -> str:
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="megatron-engine-event-receiver",
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
        try:
            context = zmq.Context()
            socket = context.socket(zmq.PULL)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVTIMEO, 100)
            socket.bind_to_random_port(f"tcp://{self.bind_host}")
            self._ready.put(socket.getsockopt_string(zmq.LAST_ENDPOINT))
            while not self._stop.is_set():
                try:
                    kind, payload = msgpack.unpackb(socket.recv(), raw=False)
                except zmq.Again:
                    continue
                self.callback(str(kind), payload)
        except Exception as exc:
            if self._ready.empty():
                self._ready.put(exc)
            else:
                logging.exception("Engine event receiver failed")
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
