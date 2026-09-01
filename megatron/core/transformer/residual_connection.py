# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Runtime protocol for architecture-specific residual connections."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Optional, TypeAlias, overload

import torch
from torch import Tensor, nn

ResidualBranchOutput: TypeAlias = Tensor | tuple[Tensor, Optional[Tensor]]
ResidualConnectionState: TypeAlias = tuple[Tensor, ...]
ResidualConnectionWriteState: TypeAlias = tuple[Tensor, ...]
ResidualConnectionOperation: TypeAlias = Literal["read", "write"]


class ResidualConnection(nn.Module, ABC):
    """Read one branch input from, then write its update to, a residual stream.

    ``forward`` validates the common contract and retains the incoming residual
    stream as the first tensor in ``ResidualConnectionState``. Concrete connections
    own bias, dropout, mapping, and residual-update semantics.
    """

    def __init__(self, residual_stream_hidden_size: int, branch_hidden_size: int):
        super().__init__()
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
        self, value: Tensor, *, operation: Literal["read"], fp32_residual_connection: bool = False
    ) -> tuple[Tensor, ResidualConnectionState]: ...

    @overload
    def forward(
        self,
        value: ResidualBranchOutput,
        *,
        operation: Literal["write"],
        state: ResidualConnectionState,
        dropout_probability: float,
        training: bool,
    ) -> Tensor: ...

    def forward(
        self,
        value: ResidualBranchOutput,
        *,
        operation: ResidualConnectionOperation,
        state: ResidualConnectionState | None = None,
        fp32_residual_connection: bool = False,
        dropout_probability: float | None = None,
        training: bool | None = None,
    ) -> Tensor | tuple[Tensor, ResidualConnectionState]:
        """Execute one residual operation through the standard module call path."""

        if operation == "read":
            if not torch.is_tensor(value):
                raise TypeError("Residual connection read expects a tensor.")
            if state is not None or dropout_probability is not None or training is not None:
                raise TypeError("Residual connection read received write-only arguments.")
            return self._read_with_validation(
                value, fp32_residual_connection=fp32_residual_connection
            )
        if operation == "write":
            if fp32_residual_connection:
                raise TypeError("Residual connection write received read-only arguments.")
            if state is None:
                raise TypeError("Residual connection write requires connection state.")
            if dropout_probability is None or training is None:
                raise TypeError(
                    "Residual connection write requires dropout_probability and training."
                )
            return self._write_with_validation(
                value, state, dropout_probability=dropout_probability, training=training
            )
        raise ValueError(f"Unsupported residual connection operation: {operation}.")

    def _read_with_validation(
        self, hidden_states: Tensor, *, fp32_residual_connection: bool
    ) -> tuple[Tensor, ResidualConnectionState]:
        if hidden_states.shape[-1] != self.residual_stream_hidden_size:
            raise ValueError(
                f"{type(self).__name__} expected residual-stream hidden size "
                f"{self.residual_stream_hidden_size}, got {hidden_states.shape[-1]}."
            )
        branch_input, write_state = self._read(hidden_states)
        if not torch.is_tensor(branch_input):
            raise TypeError(
                f"{type(self).__name__}._read returned {type(branch_input).__name__}, "
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
        self._validate_tensor_state(write_state, state_name="write state", allow_empty=True)

        residual_stream = hidden_states.float() if fp32_residual_connection else hidden_states
        return branch_input, (residual_stream, *write_state)

    def _write_with_validation(
        self,
        branch_output: ResidualBranchOutput,
        state: ResidualConnectionState,
        *,
        dropout_probability: float,
        training: bool,
    ) -> Tensor:
        self._validate_branch_output(branch_output)
        self._validate_tensor_state(state, state_name="connection state", allow_empty=False)
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
        hidden_states = self._write(
            branch_output, state, dropout_probability=dropout_probability, training=training
        )
        if not torch.is_tensor(hidden_states):
            raise TypeError(
                f"{type(self).__name__}._write returned {type(hidden_states).__name__}, "
                "expected a tensor."
            )
        if hidden_states.shape != state[0].shape:
            raise ValueError(
                f"{type(self).__name__} changed the residual-stream shape while writing: "
                f"{tuple(state[0].shape)} -> {tuple(hidden_states.shape)}."
            )
        return hidden_states

    @staticmethod
    def residual_stream(state: ResidualConnectionState) -> Tensor:
        """Return the carried residual stream from a validated connection state."""

        ResidualConnection._validate_tensor_state(
            state, state_name="connection state", allow_empty=False
        )
        return state[0]

    @staticmethod
    def _validate_tensor_state(
        state: tuple[Tensor, ...], *, state_name: str, allow_empty: bool
    ) -> None:
        if not isinstance(state, tuple) or (not allow_empty and not state):
            qualifier = "possibly empty" if allow_empty else "non-empty"
            raise TypeError(f"{state_name.capitalize()} must be a {qualifier} tuple of tensors.")
        if not all(torch.is_tensor(tensor) for tensor in state):
            raise TypeError(f"{state_name.capitalize()} must contain only tensors.")

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
    def _read(self, hidden_states: Tensor) -> tuple[Tensor, ResidualConnectionWriteState]:
        """Return the branch input and state needed only by the later write."""

    @abstractmethod
    def _write(
        self,
        branch_output: ResidualBranchOutput,
        state: ResidualConnectionState,
        *,
        dropout_probability: float,
        training: bool,
    ) -> Tensor:
        """Implement the architecture-specific residual-stream update."""
