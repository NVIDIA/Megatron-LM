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

The same mechanism, with the same secret, protects the sockets that the engine ranks open
among themselves (collectives, request fan-out, KV-handoff completion), see
``engine_socket_auth``. One secret covers the whole deployment: whoever holds the
coordinator's secret can already submit requests and control the engines, so a separate
rank-to-rank secret would add little.

PLAIN does not encrypt traffic. It keeps unauthorized peers out; confidentiality
on an untrusted network needs a secure transport (e.g. a VPN or TLS tunnel).

The secret travels with the coordinator address, as ``<endpoint>#auth=<hex>``,
so existing code that passes the address around keeps working. Only the code
that opens the socket needs to split it. Without authentication the address is
a plain endpoint. Treat an authenticated address as a credential: do not log it.
"""

import hashlib
import secrets
import threading
from dataclasses import dataclass, field
from typing import Optional

try:
    import zmq
    from zmq.auth.thread import ThreadAuthenticator

    HAVE_ZMQ = True
except ImportError:
    HAVE_ZMQ = False

_AUTH_FRAGMENT = "#auth="
# 32 random bytes is 256 bits of entropy: the stdlib's own default for ``secrets``
# (``secrets.DEFAULT_ENTROPY``) and well above the 128 bits usually recommended for a
# shared secret. The token is also the PLAIN password, so there is no benefit to making
# it shorter.
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


def require_authentication(socket: "zmq.Socket", domain: Optional[str] = None) -> None:
    """Make ``socket`` a PLAIN server: peers without valid credentials cannot connect.

    Must be called before the socket binds. ``domain`` selects the ZAP domain whose secret
    the socket checks (see ``engine_socket_auth``); the default domain is the coordinator's.
    """
    if domain is not None:
        socket.zap_domain = domain.encode()
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


@dataclass(frozen=True)
class SocketAuth:
    """The secret and ZAP domain that the engine ranks' own sockets authenticate with."""

    domain: str
    token: bytes = field(repr=False)


# A ZeroMQ context has a single ZAP endpoint, so all of its authenticated sockets share one
# handler, which tells secrets apart by ZAP domain.
_context_authenticators: dict = {}
_context_authenticators_lock = threading.Lock()


def engine_socket_auth(context: "zmq.Context", token: bytes) -> SocketAuth:
    """Make the sockets that the engine ranks bind on ``context`` accept only peers with ``token``.

    ``token`` is the deployment's one secret, the coordinator's. Call it on every rank that
    binds such a socket, before it binds; calling it again with the same token is harmless.
    The domain is derived from the token, so engines in one process that hold different
    secrets (as in tests) do not share a domain. Pair with ``stop_context_authentication``
    before terminating ``context``.
    """
    domain = "engine-" + hashlib.sha256(token).hexdigest()[:16]
    with _context_authenticators_lock:
        authenticator = _context_authenticators.get(context)
        if authenticator is None:
            authenticator = ThreadAuthenticator(context)
            authenticator.start()
            _context_authenticators[context] = authenticator
        authenticator.configure_plain(
            domain=domain, passwords={_PLAIN_USERNAME.decode(): token.hex()}
        )
    return SocketAuth(domain, token)


def stop_context_authentication(context: "zmq.Context") -> None:
    """Stop the ZAP handler that ``engine_socket_auth`` started on ``context``, if any.

    Must run before ``context.term()``, which otherwise waits on the handler forever.
    """
    with _context_authenticators_lock:
        authenticator = _context_authenticators.pop(context, None)
    if authenticator is not None:
        authenticator.stop()


def secure_server(socket: "zmq.Socket", auth: Optional[SocketAuth]) -> None:
    """Require ``auth`` from every peer of a socket that is about to bind. None: no-op."""
    if auth is not None:
        require_authentication(socket, auth.domain)


def secure_client(socket: "zmq.Socket", auth: Optional[SocketAuth]) -> None:
    """Present ``auth`` from a socket that is about to connect. None: no-op."""
    if auth is not None:
        present_credentials(socket, auth.token)
