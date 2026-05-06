# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.core.inference.headers import Headers, UnknownHeaderError


class TestHeaders:

    def test_headers_are_unique(self):
        """All Header enum members have unique values."""
        values = [h.value for h in Headers]
        assert len(values) == len(set(values))

    def test_headers_contain_expected_members(self):
        """The Headers enum exposes all coordinator protocol values."""
        names = {h.name for h in Headers}
        expected = {
            "CONNECT",
            "CONNECT_ACK",
            "SUBMIT_REQUEST",
            "ENGINE_REPLY",
            "PAUSE",
            "UNPAUSE",
            "SUSPEND",
            "RESUME",
            "SET_GENERATION_EPOCH",
            "STOP",
            "DISCONNECT",
            "SHUTDOWN",
            "TP_BROADCAST",
        }
        assert expected.issubset(names)

    def test_headers_lookup_by_name(self):
        """Headers can be retrieved by name via __getitem__."""
        assert Headers["CONNECT"] is Headers.CONNECT
        assert Headers["SHUTDOWN"] is Headers.SHUTDOWN


class TestUnknownHeaderError:

    def test_unknown_header_error_is_exception(self):
        """UnknownHeaderError is an Exception subclass."""
        assert issubclass(UnknownHeaderError, Exception)

    def test_unknown_header_error_message_contains_header(self):
        """The error message identifies which header was unrecognized."""
        err = UnknownHeaderError("BAD_HEADER")
        assert "BAD_HEADER" in str(err)

    def test_unknown_header_error_can_be_raised_and_caught(self):
        """The exception can be raised and caught as Exception."""
        with pytest.raises(UnknownHeaderError):
            raise UnknownHeaderError(Headers.CONNECT)

    def test_unknown_header_error_with_enum_payload(self):
        """The error formats enum members correctly into the message."""
        err = UnknownHeaderError(Headers.PAUSE)
        assert "PAUSE" in str(err)
