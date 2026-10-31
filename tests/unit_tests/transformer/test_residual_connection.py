# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Tests for the architecture-independent residual-connection protocol."""

from unittest.mock import patch

import pytest
import torch

from megatron.core.transformer.residual_connection import ResidualConnection


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

    def _eager_read(self, hidden_states, **kwargs):
        return self._read_result(hidden_states, **kwargs)

    def _read_result(
        self, hidden_states, *, fp32_residual_connection, branch_input_dtype, execution_context
    ):
        self.received_execution_context = execution_context
        self.received_input = hidden_states
        if self.read_behavior == "non_tensor":
            return object(), ()
        if self.read_behavior == "wrong_leading_shape":
            return torch.empty(3, self.branch_hidden_size), ()
        if self.read_behavior == "wrong_hidden_size":
            return torch.empty(*hidden_states.shape[:-1], self.branch_hidden_size + 1), ()
        branch = hidden_states[..., : self.branch_hidden_size]
        if branch_input_dtype is not None:
            branch = branch.to(branch_input_dtype)
        if self.read_behavior == "wrong_dtype":
            branch = branch.double()
        residual = hidden_states.float() if fp32_residual_connection else hidden_states
        return branch, (residual,)

    def _eager_write(self, branch_output, state, **kwargs):
        return self._write_result(branch_output, state, **kwargs)

    def _write_result(
        self, branch_output, state, *, dropout_probability, training, execution_context
    ):
        del dropout_probability, training
        self.received_execution_context = execution_context
        self.received_input = branch_output
        self.received_state = state
        if self.write_behavior == "add_branch":
            return torch.cat(
                (
                    state[0][..., : self.branch_hidden_size] + branch_output,
                    state[0][..., self.branch_hidden_size :],
                ),
                dim=-1,
            )
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


class _DirectCheckpointResidualConnection(_ProtocolResidualConnection):
    """Direct checkpoint arithmetic with validation supplied by the public dispatcher."""

    def _checkpoint_read(self, hidden_states, context, **kwargs):
        return self._read_result(hidden_states, **kwargs)

    def _checkpoint_write(self, branch_output, state, context, **kwargs):
        return self._write_result(branch_output, state, **kwargs)


class _ForwardOverrideConnection(_DirectCheckpointResidualConnection):
    def forward(self, value, **kwargs):
        self.events.append("forward")
        result = super().forward(value * 2, **kwargs)
        if self.operation == "read":
            branch, state = result
            return branch + 1, state
        return result + 1


class _SelectedStateConnection(_DirectCheckpointResidualConnection):
    """Select a complete state instead of relying on the shared input tensor."""

    def __init__(self, *, state_behavior="alias"):
        super().__init__(operation="read")
        self.state_behavior = state_behavior
        self.selected_state = None

    def _read_result(
        self, hidden_states, *, fp32_residual_connection, branch_input_dtype, execution_context
    ):
        del fp32_residual_connection, branch_input_dtype, execution_context
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


class _NoCheckpointSelectedStateConnection(_SelectedStateConnection):
    _checkpoint_read = ResidualConnection._checkpoint_read


def _execute(connection, value, *, checkpointed, **kwargs):
    return connection(value, recompute_context=object() if checkpointed else None, **kwargs)


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

    @pytest.mark.parametrize("checkpointed", [False, True])
    def test_forward_rejects_mode_argument_mismatches(self, checkpointed):
        reader = _ProtocolResidualConnection(operation="read")
        writer = _ProtocolResidualConnection(operation="write")
        residual = torch.randn(2, 12)
        branch = torch.randn(2, 4)

        with pytest.raises(TypeError, match="read expects a tensor"):
            _execute(reader, (residual, None), checkpointed=checkpointed)
        with pytest.raises(TypeError, match="write-only arguments"):
            _execute(reader, residual, state=(), checkpointed=checkpointed)
        with pytest.raises(TypeError, match="read-only arguments"):
            _execute(
                writer,
                branch,
                checkpointed=checkpointed,
                state=(residual,),
                fp32_residual_connection=True,
                dropout_probability=0.0,
                training=False,
            )
        with pytest.raises(TypeError, match="requires connection state"):
            _execute(
                writer, branch, checkpointed=checkpointed, dropout_probability=0.0, training=False
            )
        with pytest.raises(TypeError, match="requires dropout_probability and training"):
            _execute(writer, branch, state=(residual,), checkpointed=checkpointed)

    @pytest.mark.parametrize(
        ("read_behavior", "expected_exception", "expected_error"),
        [
            ("non_tensor", TypeError, "expected a tensor"),
            ("wrong_leading_shape", ValueError, "changed non-hidden dimensions"),
            ("wrong_hidden_size", ValueError, "expected branch hidden size"),
            ("wrong_dtype", TypeError, "returned branch dtype"),
        ],
    )
    @pytest.mark.parametrize("checkpointed", [False, True])
    def test_read_validates_implementation_outputs(
        self, read_behavior, expected_exception, expected_error, checkpointed
    ):
        connection = _DirectCheckpointResidualConnection(
            operation="read", read_behavior=read_behavior
        )

        with pytest.raises(expected_exception, match=expected_error):
            _execute(
                connection,
                torch.randn(2, 12),
                checkpointed=checkpointed,
                branch_input_dtype=torch.float32,
            )

    @pytest.mark.parametrize("checkpointed", [False, True])
    def test_read_validates_input_width(self, checkpointed):
        with pytest.raises(ValueError, match="expected residual-stream hidden size"):
            _execute(
                _DirectCheckpointResidualConnection(operation="read"),
                torch.randn(2, 11),
                checkpointed=checkpointed,
            )

    @pytest.mark.parametrize("state_behavior", ["alias", "replacement"])
    @pytest.mark.parametrize("checkpointed", [False, True])
    def test_read_preserves_implementation_selected_full_state(self, state_behavior, checkpointed):
        connection = _SelectedStateConnection(state_behavior=state_behavior)
        hidden_states = torch.randn(2, 12, requires_grad=True)

        branch, state = _execute(connection, hidden_states, checkpointed=checkpointed)

        assert state is connection.selected_state
        assert connection.residual_stream(state) is not hidden_states
        if state_behavior == "alias":
            assert state[0].data_ptr() == hidden_states.data_ptr()
        else:
            assert state[0].data_ptr() != hidden_states.data_ptr()
            torch.testing.assert_close(state[0], hidden_states * 2, rtol=0, atol=0)
        assert state[1].grad_fn is not None
        assert len(state) == 2
        writer = _DirectCheckpointResidualConnection(operation="write", write_behavior="use_extra")
        output = _execute(
            writer,
            branch,
            checkpointed=checkpointed,
            state=state,
            dropout_probability=0.0,
            training=False,
        )
        assert writer.received_state is state
        assert writer.received_state[1] is connection.selected_state[1]
        output.sum().backward()

        expected_grad = torch.full_like(hidden_states, 2 if state_behavior == "replacement" else 1)
        expected_grad[..., : connection.branch_hidden_size] += (
            2 * hidden_states.detach()[..., : connection.branch_hidden_size]
        )
        torch.testing.assert_close(hidden_states.grad, expected_grad, rtol=0, atol=0)

    @pytest.mark.parametrize("checkpointed", [False, True])
    def test_shared_validation_preserves_implementation_selected_state_dtype(self, checkpointed):
        connection = _SelectedStateConnection()
        hidden_states = torch.randn(2, 12, dtype=torch.bfloat16)

        _, state = _execute(
            connection, hidden_states, checkpointed=checkpointed, fp32_residual_connection=True
        )

        assert state is connection.selected_state
        assert state[0].dtype == torch.bfloat16
        assert state[0].data_ptr() == hidden_states.data_ptr()

    def test_default_read_replay_requires_an_explicit_policy_for_selected_state(self):
        connection = _NoCheckpointSelectedStateConnection()
        with pytest.raises(NotImplementedError, match="must implement _checkpoint_read"):
            connection(
                torch.randn(2, 12), recompute_context=object(), fp32_residual_connection=False
            )

    def test_default_write_replay_requires_an_explicit_policy_for_aliased_output(self):
        connection = _ProtocolResidualConnection(operation="write")
        residual = torch.randn(2, 12, requires_grad=True)
        branch = torch.randn(2, 4, requires_grad=True)
        state = (residual,)
        assert connection(branch, state=state, dropout_probability=0.0, training=True) is residual

        with pytest.raises(NotImplementedError, match="must implement _checkpoint_write"):
            connection(
                branch,
                state=state,
                recompute_context=object(),
                dropout_probability=0.0,
                training=True,
            )

    @pytest.mark.parametrize("checkpointed", [False, True])
    @pytest.mark.parametrize("operation", ["read", "write"])
    def test_backward_hook_preserves_selected_input_view_and_scales_gradients(
        self, checkpointed, operation
    ):
        connection = (
            _DirectCheckpointResidualConnection(operation="read")
            if operation == "read"
            else _DirectCheckpointResidualConnection(operation="write", write_behavior="add_branch")
        )
        calls = []

        def scale_gradient(module, grad_input, grad_output):
            calls.append(operation)
            return (grad_input[0] * 3,)

        connection.register_full_backward_hook(scale_gradient)
        width = 12 if operation == "read" else 4
        value = torch.randn(2, width, requires_grad=True)
        if operation == "read":
            output, state = _execute(connection, value, checkpointed=checkpointed)
            # The selected state is exactly the module-input view, not the helper's input.
            assert state[0] is connection.received_input
            expected_grad = torch.zeros_like(value)
            expected_grad[..., : connection.branch_hidden_size] = 3
        else:
            residual = torch.randn(2, 12, requires_grad=True)
            output = _execute(
                connection,
                value,
                checkpointed=checkpointed,
                state=(residual,),
                dropout_probability=0.0,
                training=True,
            )
            expected_grad = torch.full_like(value, 3)
        assert connection.received_input is not value
        assert connection.received_input.data_ptr() == value.data_ptr()
        assert connection.received_input.grad_fn is not None
        output.sum().backward()
        assert calls == [operation]
        torch.testing.assert_close(value.grad, expected_grad, rtol=0, atol=0)
        if operation == "write":
            torch.testing.assert_close(residual.grad, torch.ones_like(residual), rtol=0, atol=0)

    @pytest.mark.parametrize("checkpointed", [False, True])
    @pytest.mark.parametrize("operation", ["read", "write"])
    def test_forward_override_and_value_changing_hooks_apply_in_both_modes(
        self, checkpointed, operation
    ):
        connection = _ForwardOverrideConnection(operation=operation, write_behavior="add_branch")
        connection.events = []
        execution_context = object()

        def prepare(module, args):
            module.events.append("pre")
            return (args[0] + 3,)

        def finish(module, args, output):
            module.events.append("post")
            if operation == "read":
                branch, state = output
                return branch * 0.5, state
            return output * 0.5

        connection.register_forward_pre_hook(prepare)
        connection.register_forward_hook(finish)
        value = torch.randn(2, 12 if operation == "read" else 4, requires_grad=True)
        reference = value.detach().clone().requires_grad_()
        transformed = (reference + 3) * 2
        if operation == "read":
            output, state = _execute(
                connection, value, checkpointed=checkpointed, execution_context=execution_context
            )
            expected = (transformed[..., :4] + 1) * 0.5
            assert state[0] is connection.received_input
            torch.testing.assert_close(state[0], transformed, rtol=0, atol=0)
        else:
            residual = torch.randn(2, 12, requires_grad=True)
            reference_residual = residual.detach().clone().requires_grad_()
            output = _execute(
                connection,
                value,
                checkpointed=checkpointed,
                state=(residual,),
                execution_context=execution_context,
                dropout_probability=0.0,
                training=True,
            )
            expected = (
                torch.cat(
                    (reference_residual[..., :4] + transformed, reference_residual[..., 4:]), dim=-1
                )
                + 1
            ) * 0.5
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        output.square().sum().backward()
        expected.square().sum().backward()
        assert connection.events == ["pre", "forward", "post"]
        assert connection.received_execution_context is execution_context
        torch.testing.assert_close(value.grad, reference.grad, rtol=0, atol=0)
        if operation == "write":
            torch.testing.assert_close(residual.grad, reference_residual.grad, rtol=0, atol=0)

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
    @pytest.mark.parametrize("checkpointed", [False, True])
    def test_read_validates_implementation_selected_full_state(
        self, state_behavior, expected_exception, expected_error, checkpointed
    ):
        connection = _SelectedStateConnection(state_behavior=state_behavior)

        with pytest.raises(expected_exception, match=expected_error):
            _execute(connection, torch.randn(2, 12), checkpointed=checkpointed)

    @pytest.mark.parametrize("checkpointed", [False, True])
    def test_write_validates_state_branch_output_and_implementation_output(self, checkpointed):
        connection = _DirectCheckpointResidualConnection(operation="write")
        residual = torch.randn(2, 12)
        branch = torch.randn(2, 4)
        write_kwargs = dict(state=(residual,), dropout_probability=0.0, training=False)

        with pytest.raises(TypeError, match="non-empty tuple of tensors"):
            _execute(connection, branch, checkpointed=checkpointed, **{**write_kwargs, "state": ()})
        with pytest.raises(TypeError, match="contain only tensors"):
            _execute(
                connection,
                branch,
                checkpointed=checkpointed,
                **{**write_kwargs, "state": (object(),)},
            )
        with pytest.raises(TypeError, match="tensor or an .* tensor tuple"):
            _execute(connection, (branch,), checkpointed=checkpointed, **write_kwargs)
        with pytest.raises(TypeError, match="tensor and an optional tensor"):
            _execute(connection, (object(), None), checkpointed=checkpointed, **write_kwargs)
        with pytest.raises(ValueError, match="incompatible non-hidden dimensions"):
            _execute(connection, torch.randn(3, 4), checkpointed=checkpointed, **write_kwargs)
        with pytest.raises(ValueError, match="expected branch output hidden size"):
            _execute(connection, torch.randn(2, 5), checkpointed=checkpointed, **write_kwargs)

        for write_behavior, expected_error in (
            ("non_tensor", "expected a tensor"),
            ("wrong_shape", "changed the residual-stream shape"),
        ):
            invalid_connection = _DirectCheckpointResidualConnection(
                operation="write", write_behavior=write_behavior
            )
            with pytest.raises(
                TypeError if write_behavior == "non_tensor" else ValueError, match=expected_error
            ):
                _execute(invalid_connection, branch, checkpointed=checkpointed, **write_kwargs)

    def test_checkpoint_inputs_are_validated_before_algorithm_dispatch(self):
        reader = _DirectCheckpointResidualConnection(operation="read")
        writer = _DirectCheckpointResidualConnection(operation="write")
        stream, branch = torch.randn(2, 12), torch.randn(2, 4)
        context = object()
        with (
            patch.object(reader, "_checkpoint_read", side_effect=AssertionError("read dispatched")),
            patch.object(
                writer, "_checkpoint_write", side_effect=AssertionError("write dispatched")
            ),
        ):
            with pytest.raises(TypeError, match="floating-point torch.dtype"):
                reader(
                    stream,
                    recompute_context=context,
                    fp32_residual_connection=False,
                    branch_input_dtype=torch.int64,
                )
            with pytest.raises(ValueError, match="expected branch output hidden size"):
                writer(
                    branch[..., :-1],
                    state=(stream,),
                    recompute_context=context,
                    dropout_probability=0.0,
                    training=True,
                )
