# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio

import pytest

quart = pytest.importorskip("quart")
from quart import Quart


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_app(client):
    """Build a Quart app with the health blueprint and a 'client' in config."""
    from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints import (
        health,
    )

    app = Quart(__name__)
    app.config['client'] = client
    app.register_blueprint(health.bp)
    return app


class TestHealthEndpoint:

    def test_health_returns_ready_when_client_present(self):
        """GET /health returns 200 + ready=True when a client is configured."""
        app = _make_app(client=object())  # any non-None client
        client = app.test_client()
        response = _run(client.get("/health"))
        assert response.status_code == 200
        body = _run(response.get_json())
        assert body["ready"] is True
        assert body["status"] == "ok"

    def test_health_returns_503_when_client_missing(self):
        """GET /health returns 503 + status=error when no client is configured."""
        app = _make_app(client=None)
        client = app.test_client()
        response = _run(client.get("/health"))
        assert response.status_code == 503
        body = _run(response.get_json())
        assert body["status"] == "error"
        assert body["ready"] is False
        assert "Inference client not initialized" in body["details"]

    def test_v1_health_alias_works(self):
        """GET /v1/health is an alias for /health and returns the same response."""
        app = _make_app(client=object())
        client = app.test_client()
        response = _run(client.get("/v1/health"))
        assert response.status_code == 200
        body = _run(response.get_json())
        assert body["ready"] is True

    def test_health_returns_500_on_unexpected_exception(self):
        """GET /health returns 500 if `current_app.config.get` raises unexpectedly."""
        from unittest.mock import patch

        app = _make_app(client=object())
        # Patch dict.get so that calling current_app.config.get('client') raises.
        # This must take effect AFTER blueprint registration (which iterates config).
        with patch.object(type(app.config), "get", side_effect=RuntimeError("config exploded")):
            client = app.test_client()
            response = _run(client.get("/health"))
        assert response.status_code == 500
        body = _run(response.get_json())
        assert body["status"] == "error"
        assert "config exploded" in body["details"]
