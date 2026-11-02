# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Runtime protocol for architecture-specific residual connections."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Literal, NamedTuple, Optional, TypeAlias, TypeVar, overload

import torch
from torch import Tensor, nn

from megatron.core.transformer.residual_recompute import ResidualStreamRecomputeContext

ResidualBranchOutput: TypeAlias = Tensor | tuple[Tensor, Optional[Tensor]]
ResidualConnectionState: TypeAlias = tuple[Tensor, ...]
ResidualConnectionOperation: TypeAlias = Literal["read", "write"]

_StateT = TypeVar("_StateT")


class ResidualBranchInput(NamedTuple, Generic[_StateT]):
    """Common preparation result for residual implementations sharing a branch forward.

    The shared forward runs the branch on ``branch_input``, then passes ``state``
    unchanged to the matching residual-update hook. Each implementation controls input
    preparation (normalization and dtype) and the saved state needed to apply its update.

    Ordinary Mamba stores its residual tensor; current WR stores ``(wide_residual,)``.
    The contract allows future token-dependent WR variants to save per-token coefficients
    needed at writeback, and an mHC adapter to save predicted mixing/write maps. These
    implementations can reuse the same branch forward: it passes ``state`` as a
    single value without unpacking it; the update hook extracts the tensors it needs.

    This container preserves tensor references for one branch execution; cross-layer carry
    is a separate contract.
    """

    branch_input: Tensor
    state: _StateT


class ResidualConnection(nn.Module, ABC):
    """Read one branch input from, then write its update to, a residual stream.

    ``forward`` validates and dispatches both eager and checkpointed execution. The first state
    tensor is the carried residual stream, which may be a selected autograd alias of the input.
    Concrete connections own bias, dropout, mapping, and residual-update semantics.

    ``operation`` fixes an instance to one operation at construction.

    Both modes call this module once per logical operation. Replay runs the computation
    without repeating this module's ``forward`` or hooks. Subclasses must keep eager and
    checkpointed computations equivalent.

    Hooks and forward overrides must not modify checkpoint-owned output storage in place:
    replay restores the original values.
    """

    def __init__(
        self,
        residual_stream_hidden_size: int,
        branch_hidden_size: int,
        *,
        operation: ResidualConnectionOperation,
    ):
        super().__init__()
        if operation not in ("read", "write"):
            raise ValueError(f"Unsupported residual connection operation: {operation}.")
        self.operation = operation
        if residual_stream_hidden_size <= 0:
            raise ValueError("Residual-stream hidden size must be positive.")
        if branch_hidden_size <= 0:
            raise ValueError("Branch hidden size must be positive.")
        self._residual_stream_hidden_size = residual_stream_hidden_size
        self._branch_hidden_size = branch_hidden_size

    @property
    def residual_stream_hidden_size(self) -> int:
        """Hidden width accepted and returned by this connection."""

        return self._residual_stream_hidden_size

    @property
    def branch_hidden_size(self) -> int:
        """Hidden width produced by the ``read`` operation for the wrapped branch."""

        return self._branch_hidden_size

    @overload
    def forward(
        self,
        value: Tensor,
        *,
        fp32_residual_connection: bool = False,
        branch_input_dtype: torch.dtype | None = None,
        execution_context: object | None = None,
        recompute_context: ResidualStreamRecomputeContext | None = None,
    ) -> tuple[Tensor, ResidualConnectionState]: ...

    @overload
    def forward(
        self,
        value: ResidualBranchOutput,
        *,
        state: ResidualConnectionState,
        dropout_probability: float,
        training: bool,
        execution_context: object | None = None,
        recompute_context: ResidualStreamRecomputeContext | None = None,
    ) -> Tensor: ...

    def forward(
        self,
        value: ResidualBranchOutput,
        *,
        state: ResidualConnectionState | None = None,
        fp32_residual_connection: bool = False,
        branch_input_dtype: torch.dtype | None = None,
        dropout_probability: float | None = None,
        training: bool | None = None,
        execution_context: object | None = None,
        recompute_context: ResidualStreamRecomputeContext | None = None,
    ) -> Tensor | tuple[Tensor, ResidualConnectionState]:
        """Execute one residual operation through the standard module call path.

        ``branch_input_dtype`` is read-only. ``None`` preserves the connection's natural read
        dtype; otherwise compatible connections may produce that dtype directly in a fused read.

        ``execution_context`` supplies per-call dependencies, such as mHC's parameter-owning
        module and fusion setting. Supplied modules remain registered with their caller;
        this connection uses them without registering them as children.

        ``recompute_context=None`` selects eager execution. Otherwise, the connection uses
        its checkpoint implementation, which chooses what to replay.
        """

        match self.operation:
            case "read":
                if not torch.is_tensor(value):
                    raise TypeError("Residual connection read expects a tensor.")
                if state is not None or dropout_probability is not None or training is not None:
                    raise TypeError("Residual connection read received write-only arguments.")
                self._validate_read_input(value, branch_input_dtype)
                if recompute_context is None:
                    branch_input, connection_state = self._eager_read(
                        value,
                        fp32_residual_connection=fp32_residual_connection,
                        branch_input_dtype=branch_input_dtype,
                        execution_context=execution_context,
                    )
                else:
                    branch_input, connection_state = self._checkpoint_read(
                        value,
                        recompute_context,
                        fp32_residual_connection=fp32_residual_connection,
                        branch_input_dtype=branch_input_dtype,
                        execution_context=execution_context,
                    )
                self._validate_read_output(
                    value, branch_input, connection_state, branch_input_dtype
                )
                return branch_input, connection_state
            case "write":
                if fp32_residual_connection or branch_input_dtype is not None:
                    raise TypeError("Residual connection write received read-only arguments.")
                if state is None:
                    raise TypeError("Residual connection write requires connection state.")
                if dropout_probability is None or training is None:
                    raise TypeError(
                        "Residual connection write requires dropout_probability and training."
                    )
                self._validate_write_input(value, state)
                if recompute_context is None:
                    hidden_states = self._eager_write(
                        value,
                        state,
                        dropout_probability=dropout_probability,
                        training=training,
                        execution_context=execution_context,
                    )
                else:
                    hidden_states = self._checkpoint_write(
                        value,
                        state,
                        recompute_context,
                        dropout_probability=dropout_probability,
                        training=training,
                        execution_context=execution_context,
                    )
                self._validate_write_output(hidden_states, state)
                return hidden_states
            case _:
                raise ValueError(f"Unsupported residual connection operation: {self.operation}.")

    def _validate_read_input(
        self, hidden_states: Tensor, branch_input_dtype: torch.dtype | None
    ) -> None:
        """Validate inputs shared by eager and checkpointed reads without changing tensors."""

        if not torch.is_tensor(hidden_states):
            raise TypeError("Residual connection read expects a tensor.")
        if branch_input_dtype is not None and (
            not isinstance(branch_input_dtype, torch.dtype)
            or not branch_input_dtype.is_floating_point
        ):
            raise TypeError(
                "branch_input_dtype must be a floating-point torch.dtype or None, got "
                f"{branch_input_dtype!r}."
            )
        if hidden_states.shape[-1] != self.residual_stream_hidden_size:
            raise ValueError(
                f"{type(self).__name__} expected residual-stream hidden size "
                f"{self.residual_stream_hidden_size}, got {hidden_states.shape[-1]}."
            )

    def _validate_read_output(
        self,
        hidden_states: Tensor,
        branch_input: Tensor,
        state: ResidualConnectionState,
        branch_input_dtype: torch.dtype | None,
    ) -> None:
        """Validate either read implementation while preserving its selected state."""

        if not torch.is_tensor(branch_input):
            raise TypeError(
                f"{type(self).__name__} read returned {type(branch_input).__name__}, "
                "expected a tensor."
            )
        if branch_input.shape[:-1] != hidden_states.shape[:-1]:
            raise ValueError(
                f"{type(self).__name__} changed non-hidden dimensions while reading: "
                f"{tuple(hidden_states.shape)} -> {tuple(branch_input.shape)}."
            )
        if branch_input.shape[-1] != self.branch_hidden_size:
            raise ValueError(
                f"{type(self).__name__} expected branch hidden size "
                f"{self.branch_hidden_size}, got {branch_input.shape[-1]}."
            )
        if branch_input_dtype is not None and branch_input.dtype != branch_input_dtype:
            raise TypeError(
                f"{type(self).__name__} returned branch dtype {branch_input.dtype}, expected "
                f"the requested {branch_input_dtype}."
            )
        self._validate_state(state)
        if state[0].shape != hidden_states.shape:
            raise ValueError(
                f"{type(self).__name__} returned an incompatible carried residual-stream "
                f"shape: {tuple(hidden_states.shape)} -> {tuple(state[0].shape)}."
            )

    def _validate_write_input(
        self, branch_output: ResidualBranchOutput, state: ResidualConnectionState
    ) -> None:
        """Validate inputs shared by eager and checkpointed writes without changing tensors."""

        self._validate_branch_output(branch_output)
        self._validate_state(state)
        output = branch_output[0] if isinstance(branch_output, tuple) else branch_output
        if output.shape[:-1] != state[0].shape[:-1]:
            raise ValueError(
                f"{type(self).__name__} received incompatible non-hidden dimensions while "
                f"writing: residual stream {tuple(state[0].shape)}, branch output "
                f"{tuple(output.shape)}."
            )
        if output.shape[-1] != self.branch_hidden_size:
            raise ValueError(
                f"{type(self).__name__} expected branch output hidden size "
                f"{self.branch_hidden_size}, got {output.shape[-1]}."
            )

    def _validate_write_output(self, hidden_states: Tensor, state: ResidualConnectionState) -> None:
        """Validate either write implementation's output."""

        if not torch.is_tensor(hidden_states):
            raise TypeError(
                f"{type(self).__name__} write returned {type(hidden_states).__name__}, "
                "expected a tensor."
            )
        if hidden_states.shape != state[0].shape:
            raise ValueError(
                f"{type(self).__name__} changed the residual-stream shape while writing: "
                f"{tuple(state[0].shape)} -> {tuple(hidden_states.shape)}."
            )

    def _checkpoint_read(
        self,
        hidden_states: Tensor,
        context: ResidualStreamRecomputeContext,
        *,
        fp32_residual_connection: bool,
        branch_input_dtype: torch.dtype | None,
        execution_context: object | None,
    ) -> tuple[Tensor, ResidualConnectionState]:
        """Require the algorithm to choose which read results can be discarded and replayed.

        The carried residual may alias the input, and predicted maps may need to stay live.
        Checkpointing or reconstructing the complete state here would make that choice
        implicitly for every algorithm.
        """

        raise NotImplementedError(
            f"{type(self).__name__} must implement _checkpoint_read to select its replay policy."
        )

    def _checkpoint_write(
        self,
        branch_output: ResidualBranchOutput,
        state: ResidualConnectionState,
        context: ResidualStreamRecomputeContext,
        *,
        dropout_probability: float,
        training: bool,
        execution_context: object | None,
    ) -> Tensor:
        """Require the algorithm to choose which write result can be discarded and replayed.

        A write result may alias an input or require a smaller replay boundary. The concrete
        connection must explicitly select whole-write replay when its output is safe to discard.
        """

        raise NotImplementedError(
            f"{type(self).__name__} must implement _checkpoint_write to select its replay policy."
        )

    @staticmethod
    def residual_stream(state: ResidualConnectionState) -> Tensor:
        """Return the carried residual stream from a validated connection state."""

        ResidualConnection._validate_state(state)
        return state[0]

    @staticmethod
    def _validate_state(state: ResidualConnectionState) -> None:
        if not isinstance(state, tuple) or not state:
            raise TypeError("Connection state must be a non-empty tuple of tensors.")
        if not all(torch.is_tensor(tensor) for tensor in state):
            raise TypeError("Connection state must contain only tensors.")

    @staticmethod
    def _validate_branch_output(branch_output: ResidualBranchOutput) -> None:
        if torch.is_tensor(branch_output):
            return
        if not isinstance(branch_output, tuple) or len(branch_output) != 2:
            raise TypeError(
                "A residual branch must return a tensor or an (output, bias) tensor tuple."
            )
        output, bias = branch_output
        if not torch.is_tensor(output) or (bias is not None and not torch.is_tensor(bias)):
            raise TypeError(
                "A residual branch (output, bias) tuple must contain a tensor and an "
                "optional tensor."
            )

    @abstractmethod
    def _eager_read(
        self,
        hidden_states: Tensor,
        *,
        fp32_residual_connection: bool,
        branch_input_dtype: torch.dtype | None,
        execution_context: object | None,
    ) -> tuple[Tensor, ResidualConnectionState]:
        """Read without registering replay, returning ``(branch, (residual, *algorithm_state))``.

        The implementation selects the carried residual, any additional tensors, and their
        precision. The shared forward validates the result and passes the state unchanged.
        """

    @abstractmethod
    def _eager_write(
        self,
        branch_output: ResidualBranchOutput,
        state: ResidualConnectionState,
        *,
        dropout_probability: float,
        training: bool,
        execution_context: object | None,
    ) -> Tensor:
        """Apply the branch update without registering replay, using the complete read state."""
