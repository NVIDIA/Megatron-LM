# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from megatron.core.inference.data_parallel_inference_coordinator.auth import (
    authenticated_address,
    generate_auth_token,
    present_credentials,
    split_authenticated_address,
)
from megatron.core.inference.inference_client import InferenceClient

ENDPOINT = "tcp://127.0.0.1:5555"


def test_address_round_trips_the_secret():
    token = generate_auth_token()
    assert len(token) == 32 and token != generate_auth_token()
    assert split_authenticated_address(authenticated_address(ENDPOINT, token)) == (ENDPOINT, token)


def test_address_without_a_secret_is_left_alone():
    assert split_authenticated_address(ENDPOINT) == (ENDPOINT, None)


@pytest.mark.parametrize("fragment", ["#auth=zz", "#auth=abcd", "#auth="])
def test_malformed_secret_is_rejected(fragment):
    with pytest.raises(ValueError, match="authentication token"):
        split_authenticated_address(ENDPOINT + fragment)


def test_credentials_are_presented_only_when_there_is_a_secret():
    socket = SimpleNamespace()
    present_credentials(socket, None)
    assert vars(socket) == {}

    token = bytes(range(32))
    present_credentials(socket, token)
    assert socket.plain_username == b"megatron"
    assert socket.plain_password == token.hex().encode()


def test_client_authenticates_with_the_secret_in_the_coordinator_address():
    token = generate_auth_token()
    fake_socket = MagicMock(name="zmq_socket")
    fake_context = MagicMock(name="zmq_context")
    fake_context.socket.return_value = fake_socket
    with patch("megatron.core.inference.inference_client.zmq.Context", return_value=fake_context):
        InferenceClient(authenticated_address(ENDPOINT, token))

    # The secret is never part of the transport endpoint.
    fake_socket.connect.assert_called_once_with(ENDPOINT)
    assert fake_socket.plain_username == b"megatron"
    assert fake_socket.plain_password == token.hex().encode()
