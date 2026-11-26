# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Out-of-band launch secrets for the source coordinators a decode adapter hands off from.

A source (prefill) coordinator that authenticates its peers hands its launch secret only to
the deployment, never to requests: the handoff metadata that travels through Dynamo carries
just the coordinator's endpoint and instance ID. A decode adapter therefore learns each
source's secret from a file that the supervisor provides (the source adapter writes it with
``--coordinator-address-file``), and it only connects to the sources listed there.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from megatron.core.inference.data_parallel_inference_coordinator.auth import (
    authenticated_address,
    split_authenticated_address,
)


class UnknownSourceCoordinator(ValueError):
    """The handoff names a coordinator that this adapter was not configured to trust."""


class SourceCredentials:
    """Resolve a handoff's source endpoint to the address to connect to, secret included.

    With no files configured, any endpoint is accepted and none is authenticated: this suits
    sources that run without ``--inference-authenticate-coordinator``. With files, the
    endpoint must be one of theirs, so a request cannot make the adapter connect elsewhere.
    The files are read again each time a connection is opened, so a source that restarted with
    a new secret is picked up by the reconnect.
    """

    def __init__(self, paths: Sequence[str | Path] = ()) -> None:
        self._paths = [Path(path) for path in paths]

    @property
    def restricted(self) -> bool:
        """Whether handoffs are limited to the configured sources."""
        return bool(self._paths)

    def connect_address(self, endpoint: str) -> str:
        """Return ``endpoint`` with its secret attached, or raise if it is not a source."""
        # A secret in the metadata is never used: only the deployment's files are trusted.
        endpoint, _ = split_authenticated_address(endpoint)
        if not self._paths:
            return endpoint
        secrets_by_endpoint: dict[str, bytes | None] = {}
        for path in self._paths:
            source, token = split_authenticated_address(path.read_text().strip())
            secrets_by_endpoint[source] = token
        if endpoint not in secrets_by_endpoint:
            raise UnknownSourceCoordinator(
                f"Handoff source {endpoint} is not one of the configured source coordinators"
            )
        token = secrets_by_endpoint[endpoint]
        return endpoint if token is None else authenticated_address(endpoint, token)
