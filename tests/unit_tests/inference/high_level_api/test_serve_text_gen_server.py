# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Contract tests between the high-level ``serve()`` APIs and
``start_text_gen_server``.

``test_apis.py`` replaces ``start_text_gen_server`` with a ``**kwargs`` lambda,
which accepts any keyword. These tests bind both ``serve()`` calls to the real
signature with ``create_autospec`` and exercise the pre-bound socket path, so a
caller and the server cannot drift apart unnoticed.
"""

import inspect
import socket
from unittest.mock import MagicMock, create_autospec

import pytest

from megatron.core.inference.apis.async_llm import MegatronAsyncLLM
from megatron.core.inference.apis.llm import MegatronLLM
from megatron.core.inference.apis.serve_config import ServeConfig

tgs = pytest.importorskip(
    "megatron.core.inference.text_generation_server.dynamic_text_gen_server"
    ".text_generation_server"
)


def _make_primary_instance(cls):
    """Build a coordinator-mode primary-rank instance without running the
    constructor's engine/runtime setup."""
    obj = cls.__new__(cls)
    obj._engine = MagicMock()
    obj._context = MagicMock()
    obj._controller = MagicMock()
    obj._use_coordinator = True
    obj._is_primary_rank = True
    obj._loop_manager = None
    obj._coord_runtime = MagicMock()
    obj._coord_runtime.coord_addr = "tcp://coord:5555"
    obj._shutdown_called = False
    obj._serve_started = False
    return obj


@pytest.fixture
def autospec_start(monkeypatch):
    """Replace ``start_text_gen_server`` with a mock that enforces its real signature."""
    import torch.distributed as dist

    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    start = create_autospec(tgs.start_text_gen_server)
    monkeypatch.setattr(tgs, "start_text_gen_server", start)
    return start


@pytest.fixture
def fake_mp(monkeypatch):
    """Record spawn-context replicas without starting processes or sharing server state."""
    fake = MagicMock()

    def make_process(**_kwargs):
        process = MagicMock()
        process.is_alive.return_value = False
        return process

    fake.Process.side_effect = make_process
    monkeypatch.setattr(tgs, "_SERVER_PROCESS_CONTEXT", fake)
    monkeypatch.setattr(tgs, "_SERVER_PROCESSES", [])
    return fake


class TestServeMatchesStartTextGenServer:
    """Both ``serve()`` implementations call the server with accepted arguments."""

    @pytest.mark.parametrize("sock", [None, MagicMock(spec=socket.socket)])
    def test_sync_serve(self, autospec_start, sock):
        llm = _make_primary_instance(MegatronLLM)
        llm.serve(ServeConfig(port=1234, sock=sock), blocking=False)
        assert llm._serve_started is True
        autospec_start.assert_called_once()
        kwargs = autospec_start.call_args.kwargs
        assert kwargs["coordinator_addr"] == "tcp://coord:5555"
        assert kwargs["server_port"] == 1234
        assert kwargs["sock"] is sock

    @pytest.mark.asyncio
    @pytest.mark.parametrize("sock", [None, MagicMock(spec=socket.socket)])
    async def test_async_serve(self, autospec_start, sock):
        llm = _make_primary_instance(MegatronAsyncLLM)
        await llm.serve(ServeConfig(port=1234, sock=sock), blocking=False)
        assert llm._serve_started is True
        autospec_start.assert_called_once()
        kwargs = autospec_start.call_args.kwargs
        assert kwargs["coordinator_addr"] == "tcp://coord:5555"
        assert kwargs["server_port"] == 1234
        assert kwargs["sock"] is sock


class TestPreboundSocket:
    """The caller supplies the port; each spawned worker owns its own listener."""

    def test_replicas_rebind_the_prebound_port(self, fake_mp):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            bound_port = sock.getsockname()[1]
            make_process = fake_mp.Process.side_effect

            def make_replica(**kwargs):
                assert sock.fileno() == -1, "release the parent listener before starting workers"
                return make_process(**kwargs)

            fake_mp.Process.side_effect = make_replica
            url = tgs.start_text_gen_server(
                coordinator_addr="tcp://coord:5555",
                tokenizer=MagicMock(),
                rank=0,
                server_port=9999,
                num_replicas=2,
                hostname="127.0.0.1",
                sock=sock,
            )
            assert url == f"http://127.0.0.1:{bound_port}"
            assert sock.fileno() == -1
            assert fake_mp.Process.call_count == 2
            processes = list(tgs._SERVER_PROCESSES)
            assert len(processes) == 2
            for call in fake_mp.Process.call_args_list:
                assert call.kwargs["target"] is tgs._server_process_worker
                worker_call = inspect.signature(tgs._server_process_worker).bind(
                    *call.kwargs["args"]
                )
                assert worker_call.arguments["server_port"] == bound_port
                assert worker_call.arguments["hostname"] == "127.0.0.1"
            for process in processes:
                process.start.assert_called_once_with()

            tgs.stop_text_gen_server()
            assert tgs._SERVER_PROCESSES == []
            for process in processes:
                process.join.assert_called_once_with(timeout=3)

    def test_worker_listeners_bind_the_same_port_independently(self):
        with tgs._bind_reuseport_socket(0, "127.0.0.1") as first:
            first.listen(1)
            port = first.getsockname()[1]
            with tgs._bind_reuseport_socket(port, "127.0.0.1") as second:
                second.listen(1)
                assert first.fileno() != second.fileno()
                assert second.getsockname()[1] == port
                for listener in (first, second):
                    assert listener.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT) == 1
                    assert listener.getblocking() is False

    def test_unbound_socket_is_rejected(self, fake_mp):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            with pytest.raises(ValueError, match="bound to a real port"):
                tgs.start_text_gen_server(
                    coordinator_addr="tcp://coord:5555",
                    tokenizer=MagicMock(),
                    rank=0,
                    server_port=0,
                    sock=sock,
                )
        fake_mp.Process.assert_not_called()
        assert tgs._SERVER_PROCESSES == []
