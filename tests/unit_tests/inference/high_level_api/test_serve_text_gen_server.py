# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Contract tests between the high-level ``serve()`` APIs and
``start_text_gen_server``.

``test_apis.py`` replaces ``start_text_gen_server`` with a ``**kwargs`` lambda,
which accepts any keyword. These tests bind both ``serve()`` calls to the real
signature with ``create_autospec`` and exercise the pre-bound socket path, so a
caller and the server cannot drift apart unnoticed.
"""

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
    """Record frontend replicas instead of spawning processes, and isolate the
    module-level server globals."""
    fake = MagicMock()
    fake.Process.return_value.is_alive.return_value = False
    monkeypatch.setattr(tgs, "mp", fake)
    monkeypatch.setattr(tgs, "_SERVER_PROCESSES", [])
    monkeypatch.setattr(tgs, "_SHARED_SOCKET", None)
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
    """``start_text_gen_server(sock=...)`` serves on the caller's bound socket."""

    def test_replicas_share_the_prebound_socket(self, fake_mp):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            bound_port = sock.getsockname()[1]
            tgs.start_text_gen_server(
                coordinator_addr="tcp://coord:5555",
                tokenizer=MagicMock(),
                rank=0,
                server_port=9999,
                num_replicas=2,
                sock=sock,
            )
            assert tgs._SHARED_SOCKET is sock
            assert sock.getblocking() is False
            assert sock.get_inheritable() is True
            assert fake_mp.Process.call_count == 2
            for call in fake_mp.Process.call_args_list:
                args = call.kwargs["args"]
                # (coordinator_addr, tokenizer, rank, server_port, parsers, verbose, fd, hostname)
                assert args[3] == bound_port
                assert args[6] == sock.fileno()

            tgs.stop_text_gen_server()
            assert tgs._SHARED_SOCKET is None
            assert sock.fileno() == -1

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
