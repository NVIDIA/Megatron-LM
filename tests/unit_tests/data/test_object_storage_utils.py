# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from megatron.core.datasets import object_storage_utils


class _ClientError(Exception):
    """Minimal botocore ClientError replacement for unit tests."""

    def __init__(self, code: str):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


@pytest.fixture(autouse=True)
def _patch_client_error(monkeypatch):
    monkeypatch.setattr(
        object_storage_utils,
        "exceptions",
        SimpleNamespace(ClientError=_ClientError),
        raising=False,
    )


def test_s3_object_exists():
    client = Mock()

    assert object_storage_utils._s3_object_exists(client, "s3://bucket/path/to/data.bin")
    client.head_object.assert_called_once_with(Bucket="bucket", Key="path/to/data.bin")


def test_s3_object_does_not_exist():
    client = Mock()
    client.head_object.side_effect = _ClientError("404")

    assert not object_storage_utils._s3_object_exists(client, "s3://bucket/missing.idx")
    client.head_object.assert_called_once_with(Bucket="bucket", Key="missing.idx")


def test_s3_object_exists_reraises_non_404_errors():
    client = Mock()
    error = _ClientError("403")
    client.head_object.side_effect = error

    with pytest.raises(_ClientError) as exc_info:
        object_storage_utils._s3_object_exists(client, "s3://bucket/private.idx")

    assert exc_info.value is error
    client.head_object.assert_called_once_with(Bucket="bucket", Key="private.idx")
