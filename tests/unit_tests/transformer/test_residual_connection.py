# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Tests for the architecture-independent residual-connection protocol."""

import pytest
import torch

from megatron.core.transformer.residual_connection import ResidualConnection
from megatron.core.transformer.residual_recompute import (
    checkpoint_residual_read,
    checkpoint_residual_write,
)


class _ProtocolResidualConnection(ResidualConnection):
    """Minimal configurable connection for exercising the abstract runtime contract."""

    def __init__(
        self,
        *,
        operation,
        residual_stream_hidden_size=12,
        branch_hidden_size=4,
        read_behavior="valid",
        write_behavior="valid",
    ):
        super().__init__(residual_stream_hidden_size, branch_hidden_size, operation=operation)
        self.read_behavior = read_behavior
        self.write_behavior = write_behavior

    def _read(self, hidden_states, *, fp32_residual_connection, branch_input_dtype):
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
        branch = hidden_states[..., : self.branch_hidden_size]
        if branch_input_dtype is not None:
            branch = branch.to(branch_input_dtype)
        residual = hidden_states.float() if fp32_residual_connection else hidden_states
        return branch, (residual,)

    def _write(self, branch_output, state, *, dropout_probability, training):
        del branch_output, dropout_probability, training
        self.received_state = state
        if self.write_behavior == "non_tensor":
            return object()
        if self.write_behavior == "wrong_shape":
            return state[0][..., :-1]
        if self.write_behavior == "use_extra":
            return torch.cat(
                (
                    state[0][..., : self.branch_hidden_size] + state[1],
                    state[0][..., self.branch_hidden_size :],
                ),
                dim=-1,
            )
        return state[0]


class _SelectedStateConnection(_ProtocolResidualConnection):
    """Select a complete state instead of relying on the shared input tensor."""

    def __init__(self, *, state_behavior="alias"):
        super().__init__(operation="read")
        self.state_behavior = state_behavior
        self.selected_state = None

    def _read(self, hidden_states, *, fp32_residual_connection, branch_input_dtype):
        del fp32_residual_connection, branch_input_dtype
        branch = hidden_states[..., : self.branch_hidden_size]
        if self.state_behavior == "non_tuple":
            state = []
        elif self.state_behavior == "empty":
            state = ()
        elif self.state_behavior == "non_tensor":
            state = (hidden_states, object())
        elif self.state_behavior == "wrong_leading_shape":
            state = (hidden_states[:1],)
        elif self.state_behavior == "wrong_hidden_size":
            state = (hidden_states[..., :-1],)
        else:
            residual = (
                hidden_states * 2
                if self.state_behavior == "replacement"
                else hidden_states.view_as(hidden_states)
            )
            state = (residual, branch.square())
        self.selected_state = state
        return branch, state


class _IncomingResidualReplayConnection(_ProtocolResidualConnection):
    """Explicit policy for a reader whose carried stream is its incoming tensor."""

    def _checkpoint_read(
        self, hidden_states, context, *, fp32_residual_connection, branch_input_dtype
    ):
        def read_branch(stream):
            branch, _ = self(
                stream, fp32_residual_connection=False, branch_input_dtype=branch_input_dtype
            )
            return branch

        branch = context.checkpoint(read_branch, hidden_states)
        residual = hidden_states.float() if fp32_residual_connection else hidden_states
        return branch, (residual,)


class _ImmediateContext:
    def checkpoint(self, function, *args):
        return function(*args)


class TestResidualConnectionContract:
    def test_construction_operation_dispatches(self):
        reader = _ProtocolResidualConnection(operation="read")
        writer = _ProtocolResidualConnection(operation="write")
        residual = torch.randn(2, 12)
        branch, state = reader(residual)
        assert writer(branch, state=state, dropout_probability=0.0, training=False) is residual

    def test_constructor_rejects_invalid_mode(self):
        with pytest.raises(ValueError, match="Unsupported residual connection operation"):
            _ProtocolResidualConnection(operation="invalid")

    @pytest.mark.parametrize(
        ("residual_stream_hidden_size", "branch_hidden_size"), [(0, 4), (12, 0)]
    )
    def test_constructor_requires_positive_hidden_sizes(
        self, residual_stream_hidden_size, branch_hidden_size
    ):
        with pytest.raises(ValueError, match="hidden size must be positive"):
            _ProtocolResidualConnection(
                operation="read",
                residual_stream_hidden_size=residual_stream_hidden_size,
                branch_hidden_size=branch_hidden_size,
            )

    def test_forward_rejects_mode_argument_mismatches(self):
        reader = _ProtocolResidualConnection(operation="read")
        writer = _ProtocolResidualConnection(operation="write")
        residual = torch.randn(2, 12)
        branch = torch.randn(2, 4)

        with pytest.raises(TypeError, match="read expects a tensor"):
            reader((residual, None))
        with pytest.raises(TypeError, match="write-only arguments"):
            reader(residual, state=())
        with pytest.raises(TypeError, match="read-only arguments"):
            writer(
                branch,
                state=(residual,),
                fp32_residual_connection=True,
                dropout_probability=0.0,
                training=False,
            )
        with pytest.raises(TypeError, match="requires connection state"):
            writer(branch, dropout_probability=0.0, training=False)
        with pytest.raises(TypeError, match="requires dropout_probability and training"):
            writer(branch, state=(residual,))

    @pytest.mark.parametrize(
        ("read_behavior", "expected_exception", "expected_error"),
        [
            ("non_tensor", TypeError, "expected a tensor"),
            ("wrong_leading_shape", ValueError, "changed non-hidden dimensions"),
            ("wrong_hidden_size", ValueError, "expected branch hidden size"),
            ("non_tuple_state", TypeError, "non-empty tuple of tensors"),
            ("non_tensor_state", TypeError, "contain only tensors"),
        ],
    )
    def test_read_validates_implementation_outputs(
        self, read_behavior, expected_exception, expected_error
    ):
        connection = _ProtocolResidualConnection(operation="read", read_behavior=read_behavior)

        with pytest.raises(expected_exception, match=expected_error):
            connection(torch.randn(2, 12))

    def test_read_validates_input_width(self):
        with pytest.raises(ValueError, match="expected residual-stream hidden size"):
            _ProtocolResidualConnection(operation="read")(torch.randn(2, 11))

    @pytest.mark.parametrize("state_behavior", ["alias", "replacement"])
    def test_read_preserves_implementation_selected_full_state(self, state_behavior):
        connection = _SelectedStateConnection(state_behavior=state_behavior)
        hidden_states = torch.randn(2, 12, requires_grad=True)

        branch, state = connection(hidden_states)

        assert state is connection.selected_state
        assert connection.residual_stream(state) is not hidden_states
        if state_behavior == "alias":
            assert state[0].data_ptr() == hidden_states.data_ptr()
        else:
            assert state[0].data_ptr() != hidden_states.data_ptr()
            torch.testing.assert_close(state[0], hidden_states * 2, rtol=0, atol=0)
        assert state[1].grad_fn is not None
        assert len(state) == 2
        writer = _ProtocolResidualConnection(operation="write", write_behavior="use_extra")
        output = writer(branch, state=state, dropout_probability=0.0, training=False)
        assert writer.received_state is state
        assert writer.received_state[1] is connection.selected_state[1]
        output.sum().backward()

        expected_grad = torch.full_like(hidden_states, 2 if state_behavior == "replacement" else 1)
        expected_grad[..., : connection.branch_hidden_size] += (
            2 * hidden_states.detach()[..., : connection.branch_hidden_size]
        )
        torch.testing.assert_close(hidden_states.grad, expected_grad, rtol=0, atol=0)

    def test_shared_validation_preserves_implementation_selected_state_dtype(self):
        connection = _SelectedStateConnection()
        hidden_states = torch.randn(2, 12, dtype=torch.bfloat16)

        _, state = connection(hidden_states, fp32_residual_connection=True)

        assert state is connection.selected_state
        assert state[0].dtype == torch.bfloat16
        assert state[0].data_ptr() == hidden_states.data_ptr()

    def test_default_read_replay_requires_an_explicit_policy_for_selected_state(self):
        with pytest.raises(NotImplementedError, match="must implement _checkpoint_read"):
            checkpoint_residual_read(
                _SelectedStateConnection(),
                torch.randn(2, 12),
                _ImmediateContext(),
                fp32_residual_connection=False,
            )

    def test_default_write_replay_requires_an_explicit_policy_for_aliased_output(self):
        connection = _ProtocolResidualConnection(operation="write")
        residual = torch.randn(2, 12, requires_grad=True)
        branch = torch.randn(2, 4, requires_grad=True)
        state = (residual,)
        assert connection(branch, state=state, dropout_probability=0.0, training=True) is residual

        with pytest.raises(NotImplementedError, match="must implement _checkpoint_write"):
            checkpoint_residual_write(
                connection,
                branch,
                state,
                _ImmediateContext(),
                dropout_probability=0.0,
                training=True,
            )

    def test_explicit_incoming_replay_accepts_module_backward_hook_input_views(self):
        connection = _IncomingResidualReplayConnection(operation="read")
        calls = []
        connection.register_full_backward_hook(lambda *args: calls.append("read"))
        hidden_states = torch.randn(2, 12, requires_grad=True)

        branch, state = checkpoint_residual_read(
            connection, hidden_states, _ImmediateContext(), fp32_residual_connection=False
        )
        branch.sum().backward()

        assert state[0] is hidden_states
        assert calls == ["read"]
        expected_grad = torch.zeros_like(hidden_states)
        expected_grad[..., : connection.branch_hidden_size] = 1
        torch.testing.assert_close(hidden_states.grad, expected_grad, rtol=0, atol=0)

    @pytest.mark.parametrize(
        ("state_behavior", "expected_exception", "expected_error"),
        [
            ("non_tuple", TypeError, "non-empty tuple of tensors"),
            ("empty", TypeError, "non-empty tuple of tensors"),
            ("non_tensor", TypeError, "contain only tensors"),
            ("wrong_leading_shape", ValueError, "incompatible carried residual-stream shape"),
            ("wrong_hidden_size", ValueError, "incompatible carried residual-stream shape"),
        ],
    )
    def test_read_validates_implementation_selected_full_state(
        self, state_behavior, expected_exception, expected_error
    ):
        connection = _SelectedStateConnection(state_behavior=state_behavior)

        with pytest.raises(expected_exception, match=expected_error):
            connection(torch.randn(2, 12))

    def test_write_validates_state_branch_output_and_implementation_output(self):
        connection = _ProtocolResidualConnection(operation="write")
        residual = torch.randn(2, 12)
        branch = torch.randn(2, 4)
        write_kwargs = dict(state=(residual,), dropout_probability=0.0, training=False)

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
            invalid_connection = _ProtocolResidualConnection(
                operation="write", write_behavior=write_behavior
            )
            with pytest.raises(
                TypeError if write_behavior == "non_tensor" else ValueError, match=expected_error
            ):
                invalid_connection(branch, **write_kwargs)
