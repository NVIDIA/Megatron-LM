# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Transport-level authentication for the inference coordinator's ZMQ socket.

The coordinator's ROUTER socket is reachable over TCP, so every peer is untrusted
until proven otherwise. Authentication is opt-in
(``InferenceConfig.authenticate_coordinator``); without it the coordinator accepts any
peer that can reach its port. When enabled, it happens in ZeroMQ itself, using the PLAIN
mechanism with a ZAP handler: a peer that does not present the coordinator's
launch-time secret is refused during the connection handshake and none of its
messages ever reach the coordinator. The wire protocol is unchanged, and no
handler has to remember to check who is calling it.

PLAIN does not encrypt traffic. It keeps unauthorized peers out; confidentiality
on an untrusted network needs a secure transport (e.g. a VPN or TLS tunnel).

The secret travels with the coordinator address, as ``<endpoint>#auth=<hex>``,
so existing code that passes the address around keeps working. Only the code
that opens the socket needs to split it. Without authentication the address is
a plain endpoint. Treat an authenticated address as a credential: do not log it.
"""

import secrets
from typing import Optional

try:
    import zmq
    from zmq.auth.thread import ThreadAuthenticator

    HAVE_ZMQ = True
except ImportError:
    HAVE_ZMQ = False

_AUTH_FRAGMENT = "#auth="
_AUTH_TOKEN_BYTES = 32
_PLAIN_USERNAME = b"megatron"


def generate_auth_token() -> bytes:
    """Return a cryptographically random coordinator secret."""
    return secrets.token_bytes(_AUTH_TOKEN_BYTES)


def authenticated_address(address: str, auth_token: bytes) -> str:
    """Attach the secret to a coordinator address for transport between trusted components.

    The fragment is stripped by ``split_authenticated_address`` before the endpoint is
    handed to ZeroMQ, so it is never part of the transport endpoint.
    """
    return f"{address}{_AUTH_FRAGMENT}{auth_token.hex()}"


def split_authenticated_address(address: str) -> tuple[str, Optional[bytes]]:
    """Split a coordinator address into its ZeroMQ endpoint and secret (None if absent)."""
    endpoint, separator, token_hex = address.rpartition(_AUTH_FRAGMENT)
    if not separator:
        return address, None
    try:
        token = bytes.fromhex(token_hex)
    except ValueError as exc:
        raise ValueError("Invalid inference coordinator authentication token") from exc
    if len(token) != _AUTH_TOKEN_BYTES:
        raise ValueError("Invalid inference coordinator authentication token")
    return endpoint, token


def start_server_authentication(context: "zmq.Context", auth_token: bytes) -> "ThreadAuthenticator":
    """Start a ZAP handler on ``context`` that admits only peers presenting ``auth_token``.

    Must run before any server socket on ``context`` binds. Call ``stop()`` on the
    returned authenticator before terminating the context.
    """
    authenticator = ThreadAuthenticator(context)
    authenticator.start()
    authenticator.configure_plain(
        domain="*", passwords={_PLAIN_USERNAME.decode(): auth_token.hex()}
    )
    return authenticator


def require_authentication(socket: "zmq.Socket") -> None:
    """Make ``socket`` a PLAIN server: peers without valid credentials cannot connect.

    Must be called before the socket binds.
    """
    socket.plain_server = True


def present_credentials(socket: "zmq.Socket", auth_token: Optional[bytes]) -> None:
    """Make ``socket`` authenticate to a coordinator with ``auth_token``.

    Must be called before the socket connects. A missing token leaves the socket
    unauthenticated, which a coordinator that requires authentication will refuse.
    """
    if auth_token is None:
        return
    socket.plain_username = _PLAIN_USERNAME
    socket.plain_password = auth_token.hex().encode()
