# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Tests for the architecture-independent residual-connection protocol."""

import pytest
import torch

from megatron.core.transformer.residual_connection import ResidualConnection


class _ProtocolResidualConnection(ResidualConnection):
    """Minimal configurable connection for exercising the abstract runtime contract."""

    def __init__(
        self,
        *,
        residual_stream_hidden_size=12,
        branch_hidden_size=4,
        read_behavior="valid",
        write_behavior="valid",
    ):
        super().__init__(residual_stream_hidden_size, branch_hidden_size)
        self.read_behavior = read_behavior
        self.write_behavior = write_behavior

    def _read(self, hidden_states):
        if self.read_behavior == "non_tensor":
            return object(), ()
        if self.read_behavior == "wrong_leading_shape":
            return torch.empty(3, self.branch_hidden_size), ()
        if self.read_behavior == "wrong_hidden_size":
            return torch.empty(*hidden_states.shape[:-1], self.branch_hidden_size + 1), ()
        if self.read_behavior == "non_tuple_state":
            return hidden_states[..., : self.branch_hidden_size], []
        if self.read_behavior == "non_tensor_state":
            return hidden_states[..., : self.branch_hidden_size], (object(),)
        return hidden_states[..., : self.branch_hidden_size], ()

    def _write(self, branch_output, state, *, dropout_probability, training):
        del branch_output, dropout_probability, training
        if self.write_behavior == "non_tensor":
            return object()
        if self.write_behavior == "wrong_shape":
            return state[0][..., :-1]
        return state[0]


class TestResidualConnectionContract:
    @pytest.mark.parametrize(
        ("residual_stream_hidden_size", "branch_hidden_size"), [(0, 4), (12, 0)]
    )
    def test_constructor_requires_positive_hidden_sizes(
        self, residual_stream_hidden_size, branch_hidden_size
    ):
        with pytest.raises(ValueError, match="hidden size must be positive"):
            _ProtocolResidualConnection(
                residual_stream_hidden_size=residual_stream_hidden_size,
                branch_hidden_size=branch_hidden_size,
            )

    def test_forward_rejects_operation_argument_mismatches(self):
        connection = _ProtocolResidualConnection()
        residual = torch.randn(2, 12)
        branch = torch.randn(2, 4)

        with pytest.raises(TypeError, match="read expects a tensor"):
            connection((residual, None), operation="read")
        with pytest.raises(TypeError, match="write-only arguments"):
            connection(residual, operation="read", state=())
        with pytest.raises(TypeError, match="read-only arguments"):
            connection(
                branch,
                operation="write",
                state=(residual,),
                fp32_residual_connection=True,
                dropout_probability=0.0,
                training=False,
            )
        with pytest.raises(TypeError, match="requires connection state"):
            connection(branch, operation="write", dropout_probability=0.0, training=False)
        with pytest.raises(TypeError, match="requires dropout_probability and training"):
            connection(branch, operation="write", state=(residual,))
        with pytest.raises(ValueError, match="Unsupported residual connection operation"):
            connection(residual, operation="invalid")

    @pytest.mark.parametrize(
        ("read_behavior", "expected_exception", "expected_error"),
        [
            ("non_tensor", TypeError, "expected a tensor"),
            ("wrong_leading_shape", ValueError, "changed non-hidden dimensions"),
            ("wrong_hidden_size", ValueError, "expected branch hidden size"),
            ("non_tuple_state", TypeError, "possibly empty tuple of tensors"),
            ("non_tensor_state", TypeError, "contain only tensors"),
        ],
    )
    def test_read_validates_implementation_outputs(
        self, read_behavior, expected_exception, expected_error
    ):
        connection = _ProtocolResidualConnection(read_behavior=read_behavior)

        with pytest.raises(expected_exception, match=expected_error):
            connection(torch.randn(2, 12), operation="read")

    def test_read_validates_input_width(self):
        with pytest.raises(ValueError, match="expected residual-stream hidden size"):
            _ProtocolResidualConnection()(torch.randn(2, 11), operation="read")

    def test_write_validates_state_branch_output_and_implementation_output(self):
        connection = _ProtocolResidualConnection()
        residual = torch.randn(2, 12)
        branch = torch.randn(2, 4)
        write_kwargs = dict(
            operation="write", state=(residual,), dropout_probability=0.0, training=False
        )

        with pytest.raises(TypeError, match="non-empty tuple of tensors"):
            connection(branch, **{**write_kwargs, "state": ()})
        with pytest.raises(TypeError, match="contain only tensors"):
            connection(branch, **{**write_kwargs, "state": (object(),)})
        with pytest.raises(TypeError, match="tensor or an .* tensor tuple"):
            connection((branch,), **write_kwargs)
        with pytest.raises(TypeError, match="tensor and an optional tensor"):
            connection((object(), None), **write_kwargs)
        with pytest.raises(ValueError, match="incompatible non-hidden dimensions"):
            connection(torch.randn(3, 4), **write_kwargs)
        with pytest.raises(ValueError, match="expected branch output hidden size"):
            connection(torch.randn(2, 5), **write_kwargs)

        for write_behavior, expected_error in (
            ("non_tensor", "expected a tensor"),
            ("wrong_shape", "changed the residual-stream shape"),
        ):
            invalid_connection = _ProtocolResidualConnection(write_behavior=write_behavior)
            with pytest.raises(
                TypeError if write_behavior == "non_tensor" else ValueError, match=expected_error
            ):
                invalid_connection(branch, **write_kwargs)
