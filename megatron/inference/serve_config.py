# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ServeConfig:
    """Programmatic configuration for ``MegatronAsyncLLM.serve(...)``.

    This dataclass also serves as the future source of truth for a
    ``megatron serve`` CLI. It controls only the HTTP serving surface; engine
    construction and coordinator addressing are configured separately via the
    ``MegatronLLM`` / ``MegatronAsyncLLM`` constructor.
    """

    host: str = "0.0.0.0"
    """HTTP bind host for the OpenAI-compatible frontend.

    Distinct from the ``MegatronLLM`` / ``MegatronAsyncLLM`` constructor's
    ``coordinator_host`` argument: ``coordinator_host`` is the internal/routable
    address used for coordinator ZMQ traffic, whereas ``host`` is the
    externally-visible interface where the HTTP server accepts client
    connections.
    """

    port: int = 5000
    """HTTP bind port for the OpenAI-compatible frontend."""

    model_name: str = "megatron-model"
    """Served OpenAI model name.

    Echoed in HTTP responses regardless of ``strict_model_name``. The
    ``/v1/models`` endpoint always returns this value as the single advertised
    model id.
    """

    strict_model_name: bool = True
    """Whether to validate the request ``model`` field against ``model_name``.

    If True, requests whose ``model`` field does not match ``model_name`` are
    rejected with HTTP 400 in OpenAI's error shape. If False, the request is
    accepted regardless of the supplied ``model`` value.
    """

    role: Literal["primary", "worker", "auto"] = "auto"
    """Per-rank role selector for the serving frontend.

    - ``"primary"``: this rank exposes the HTTP frontend.
    - ``"worker"``: this rank does not expose HTTP; it participates in the
      dynamic engine loop only.
    - ``"auto"``: automatically picks ``"primary"`` on global rank 0 and
      ``"worker"`` elsewhere.
    """

    parsers: list[str] = field(default_factory=list)
    """Response parser names to enable on the HTTP frontend.

    Examples include ``["json", "tool_use"]``. Values are passed through to the
    underlying text-generation server unchanged.
    """

    verbose: bool = False
    """Whether the HTTP frontend should log per-request detail."""

    frontend_replicas: int = 4
    """Number of HTTP frontend processes spawned on the primary rank.

    The default of 4 matches the existing ``start_text_gen_server`` default of
    ``num_replicas=4``.
    """
