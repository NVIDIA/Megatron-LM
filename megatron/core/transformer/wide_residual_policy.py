# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
"""Wide-residual connection, replay, and checkpoint-loading policy."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.residual_connection import (
    ResidualBranchOutput,
    ResidualConnection,
    ResidualConnectionOperation,
    ResidualConnectionState,
)
from megatron.core.transformer.residual_recompute import ResidualStreamRecomputeContext
from megatron.core.transformer.streamwise_residual_ops import (
    streamwise_sigmoid_read,
    streamwise_sigmoid_writeback,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.wide_residual import (
    LearnedWideResidualRetention,
    StreamwiseSigmoidMap,
)
from megatron.core.typed_torch import apply_module


class StreamwiseSigmoidWideResidualConnection(ResidualConnection):
    """Streamwise residual maps restricted to a read or write at construction."""

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        branch_name: str,
        pg_collection: ProcessGroupCollection | None = None,
        name: str | None = None,
        *,
        operation: ResidualConnectionOperation,
    ) -> None:
        del pg_collection, name
        if config.wide_residual is None:
            raise ValueError(
                "StreamwiseSigmoidWideResidualConnection requires wide_residual config."
            )
        wr = config.wide_residual
        super().__init__(
            residual_stream_hidden_size=wr.num_streams * config.hidden_size,
            branch_hidden_size=config.hidden_size,
            operation=operation,
        )
        self.layer_number = layer_number
        self.branch_name = branch_name
        self.num_streams = wr.num_streams
        if operation == "read":
            self.read_map = StreamwiseSigmoidMap(config, map_kind="read")
        else:
            self.write_map = StreamwiseSigmoidMap(config, map_kind="write")
        self.retention = (
            LearnedWideResidualRetention(
                config, layer_number, branch_name, num_streams=self.num_streams
            )
            if operation == "write" and wr.learned_retention
            else None
        )

    def _read(
        self,
        hidden_states: Tensor,
        *,
        fp32_residual_connection: bool,
        branch_input_dtype: torch.dtype | None,
    ) -> tuple[Tensor, ResidualConnectionState]:
        """Read in branch precision before promoting the carried residual."""

        branch_input = streamwise_sigmoid_read(
            hidden_states,
            self.read_map(return_logits=True),
            self.num_streams,
            output_dtype=branch_input_dtype,
        )
        # Full-model FP32-residual inputs are normally already FP32. Preserve the defensive
        # promotion for direct low-precision inputs, after the read has consumed its input.
        residual_stream = hidden_states.float() if fp32_residual_connection else hidden_states
        return branch_input, (residual_stream,)

    def _write(
        self,
        branch_output: ResidualBranchOutput,
        state: ResidualConnectionState,
        *,
        dropout_probability: float,
        training: bool,
    ) -> Tensor:
        residual_stream = state[0]
        if isinstance(branch_output, tuple):
            branch_update, bias = branch_output
        else:
            branch_update, bias = branch_output, None

        dropout_is_active = training and dropout_probability > 0.0
        defer_update_cast = (
            residual_stream.dtype == torch.float32
            and branch_update.dtype in (torch.bfloat16, torch.float16)
            and bias is None
            and not dropout_is_active
        )
        # Bias addition and active dropout currently run in residual precision. Only defer the
        # update cast when neither operation can observe its placement; the mixed-dtype Triton
        # write then performs the BF16/FP16 -> FP32 conversion while loading the update.
        if not defer_update_cast:
            branch_update = branch_update.to(dtype=residual_stream.dtype)
        if bias is not None:
            branch_update = branch_update + bias.to(
                device=branch_update.device, dtype=branch_update.dtype
            )
        branch_update = F.dropout(branch_update, p=dropout_probability, training=training)
        retention_logits = (
            self.retention(return_logits=True) if self.retention is not None else None
        )
        return streamwise_sigmoid_writeback(
            residual_stream,
            branch_update,
            self.write_map(return_logits=True),
            self.num_streams,
            retention_logits=retention_logits,
            retention_max_forget=(self.retention.max_forget if self.retention is not None else 0.0),
        )

    def _checkpoint_read(
        self,
        hidden_states: Tensor,
        context: ResidualStreamRecomputeContext,
        *,
        fp32_residual_connection: bool,
        branch_input_dtype: torch.dtype | None = None,
    ) -> tuple[Tensor, ResidualConnectionState]:
        """Replay WR's read while retaining its incoming residual stream.

        An optional branch-input dtype request is captured by the replay closure so eager execution
        and backward reconstruction produce the same branch dtype.
        """

        def run_read(stream: Tensor) -> tuple[Tensor, ...]:
            branch_input, state = apply_module(self)(
                stream, fp32_residual_connection=False, branch_input_dtype=branch_input_dtype
            )
            if state[0].shape != stream.shape:
                raise ValueError(
                    "Residual connection read returned an incompatible carried stream."
                )
            return (branch_input, *state[1:])

        outputs = context.checkpoint(run_read, hidden_states)
        if torch.is_tensor(outputs):
            outputs = (outputs,)
        if not isinstance(outputs, tuple) or not outputs:
            raise TypeError("Checkpointed residual read must return a non-empty tensor tuple.")
        if not all(torch.is_tensor(output) for output in outputs):
            raise TypeError("Checkpointed residual read state must contain only tensors.")

        # This mirrors WR's defensive state promotion. Normal full-model
        # FP32-residual inputs are already FP32, making .float() an alias rather than a cast kernel;
        # direct/custom low-precision inputs still need the conversion to preserve replay state.
        residual_stream = hidden_states.float() if fp32_residual_connection else hidden_states
        return outputs[0], (residual_stream, *outputs[1:])

    def _checkpoint_write(
        self,
        branch_output: ResidualBranchOutput,
        state: ResidualConnectionState,
        context: ResidualStreamRecomputeContext,
        *,
        dropout_probability: float,
        training: bool,
    ) -> Tensor:
        """Replay WR's whole write, whose output has independent storage."""

        if isinstance(branch_output, tuple):
            output, bias = branch_output
            return_tuple = True
        else:
            output, bias = branch_output, None
            return_tuple = False

        def run_write(
            branch_update: Tensor, branch_bias: Tensor | None, *connection_state: Tensor
        ) -> Tensor:
            value = (branch_update, branch_bias) if return_tuple else branch_update
            return apply_module(self)(
                value,
                state=connection_state,
                dropout_probability=dropout_probability,
                training=training,
            )

        hidden_states = context.checkpoint(run_write, output, bias, *state)
        if not torch.is_tensor(hidden_states):
            raise TypeError("Checkpointed residual write must return a tensor.")
        return hidden_states


_RESIDUAL_STATE_PREFIXES = tuple(
    (f"residual_{operation}{suffix}.{parameter}.", f"residual_connection{suffix}.{parameter}.")
    for suffix in ("", "_self_attn", "_mlp")
    for operation, parameter in (
        ("read", "read_map"),
        ("write", "write_map"),
        ("write", "retention"),
    )
)


def load_residual_state(
    module, state_dict, prefix, local_metadata, strict, missing, unexpected, errors
):
    """Accept existing residual model keys with strict loading."""
    del module, local_metadata, strict, missing, unexpected, errors
    for key in list(state_dict):
        for current, legacy in _RESIDUAL_STATE_PREFIXES:
            if key.startswith(prefix + legacy):
                state_dict[prefix + current + key[len(prefix + legacy) :]] = state_dict.pop(key)
                break


def wide_residual_recompute_enabled(config: TransformerConfig, training: bool) -> bool:
    """Return whether selective residual-stream replay is active for this forward."""

    return bool(
        training
        and torch.is_grad_enabled()
        and config.recompute_granularity == "selective"
        and config.recompute_modules is not None
        and "residual_stream" in config.recompute_modules
    )
