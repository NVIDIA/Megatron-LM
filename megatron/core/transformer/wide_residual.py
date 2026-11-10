# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
"""Wide-residual parameters, connection policy, and model-boundary helpers."""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from megatron.core.fusions.wide_residual_shortcut import (
    PendingWideResidualWrites,
    shortcut_add_wide_residual_write_read_rmsnorm_quant,
    shortcut_add_wide_residual_write_two_reader_rmsnorm_quant,
)
from megatron.core.fusions.wide_residual_shortcut.api import reader_for
from megatron.core.fusions.wide_residual_shortcut.residual_replay import ReplayInterval
from megatron.core.inference.utils import InferenceMode
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

_MIN_CONTROLLER_NUMEL = 128


def _make_padded_vector(values: Tensor) -> nn.Parameter:
    """Create a parameter vector padded for small distributed-optimizer shards."""

    if values.ndim != 1:
        raise ValueError(f"Expected one-dimensional initial values, got {values.shape}.")
    if values.numel() >= _MIN_CONTROLLER_NUMEL:
        return nn.Parameter(values.clone())
    padded = torch.zeros(_MIN_CONTROLLER_NUMEL, dtype=values.dtype)
    padded[: values.numel()] = values
    return nn.Parameter(padded)


def _active_vector(param: Tensor, length: int) -> Tensor:
    """Return the active prefix of a potentially padded vector parameter."""

    return param[:length]


def _mark_residual_map_parameter(param: nn.Parameter, config: TransformerConfig) -> None:
    """Attach optimizer and TP-gradient metadata to a replicated residual map."""

    setattr(param, "is_residual_map_parameter", True)
    setattr(param, "use_muon", False)
    setattr(param, "sequence_parallel", config.sequence_parallel)


def _mark_replicated_tp_parameter(param: nn.Parameter, config: TransformerConfig) -> None:
    """Mark a replicated controller for the correct TP gradient reduction."""

    setattr(param, "allreduce", True)
    setattr(param, "tensor_model_parallel", False)
    setattr(param, "average_gradients_across_tp_domain", not config.sequence_parallel)


def _mark_retention_parameter(param: nn.Parameter, config: TransformerConfig) -> None:
    """Attach optimizer and TP-gradient metadata to a replicated retention parameter."""

    setattr(param, "is_wide_residual_retention_parameter", True)
    setattr(param, "sequence_parallel", config.sequence_parallel)
    _mark_replicated_tp_parameter(param, config)


class LearnedWideResidualRetention(nn.Module):
    """Bounded learned retention with one controller per full-width stream."""

    def __init__(
        self, config: TransformerConfig, layer_number: int, branch_name: str, *, num_streams: int
    ) -> None:
        super().__init__()
        if config.wide_residual is None:
            raise ValueError("LearnedWideResidualRetention requires wide_residual config.")
        wr = config.wide_residual
        if num_streams != wr.num_streams:
            raise ValueError(
                "Retention must define one controller per full-width stream, got "
                f"{num_streams} controllers for {wr.num_streams} streams."
            )
        self.layer_number = layer_number
        self.branch_name = branch_name
        self.num_streams = num_streams
        self.max_forget = wr.retention_max_forget
        forget_ratio = (1.0 - wr.retention_init) / self.max_forget
        initial_logit = math.log((1.0 - forget_ratio) / forget_ratio)
        self.retention_logit = _make_padded_vector(
            torch.full((num_streams,), initial_logit, dtype=torch.float32)
        )
        _mark_retention_parameter(self.retention_logit, config)

    def factors(self) -> Tensor:
        """Return one FP32 retention factor for every full-width stream."""

        logits = _active_vector(self.retention_logit, self.num_streams).float()
        return 1.0 - self.max_forget * torch.sigmoid(-logits)

    def forward(self, *, return_logits: bool = False) -> Tensor:
        """Return padded logits or materialized retention factors."""

        return self.retention_logit if return_logits else self.factors()


class StreamwiseSigmoidMap(nn.Module):
    """Positive scalar controller for each contiguous full-width stream."""

    def __init__(self, config: TransformerConfig, *, map_kind: Literal["read", "write"]) -> None:
        super().__init__()
        if config.wide_residual is None:
            raise ValueError("StreamwiseSigmoidMap requires wide_residual config.")
        wr = config.wide_residual

        self.map_kind = map_kind
        self.num_streams = wr.num_streams

        if map_kind == "read":
            initial_factor = 1.0 / self.num_streams
            initial_logit = math.log(initial_factor / (1.0 - initial_factor))
            initial_logits = torch.full((self.num_streams,), initial_logit, dtype=torch.float32)
        elif map_kind == "write":
            initial_logits = torch.linspace(
                -wr.streamwise_sigmoid_init_scale,
                wr.streamwise_sigmoid_init_scale,
                self.num_streams,
                dtype=torch.float32,
            )
        else:
            raise ValueError(f"Unsupported streamwise sigmoid map kind: {map_kind!r}.")

        self.logit = _make_padded_vector(initial_logits)
        _mark_residual_map_parameter(self.logit, config)
        _mark_replicated_tp_parameter(self.logit, config)

    def factors(self) -> Tensor:
        """Return one positive factor for every full-width residual stream."""

        factors = torch.sigmoid(_active_vector(self.logit, self.num_streams).float())
        if self.map_kind == "write":
            factors = 2.0 * factors
        return factors

    def forward(self, *, return_logits: bool = False) -> Tensor:
        """Return padded logits or one positive factor per full-width stream."""

        return self.logit if return_logits else self.factors()


class StreamwiseSigmoidWideResidualRead(nn.Module):
    """Learn an independent ordinary-width read from a wide residual stream."""

    def __init__(self, config: TransformerConfig, layer_number: int, branch_name: str) -> None:
        super().__init__()
        if config.wide_residual is None:
            raise ValueError("StreamwiseSigmoidWideResidualRead requires wide_residual config.")
        wr = config.wide_residual
        self.layer_number = layer_number
        self.branch_name = branch_name
        self.num_streams = wr.num_streams
        self.residual_stream_hidden_size = wr.num_streams * config.hidden_size
        self.branch_hidden_size = config.hidden_size
        self.branch_input_dtype = config.params_dtype
        self.read_map = StreamwiseSigmoidMap(config, map_kind="read")

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Read in branch precision; the operator fixes the output shape and dtype."""

        if hidden_states.shape[-1] != self.residual_stream_hidden_size:
            raise ValueError(
                "StreamwiseSigmoidWideResidualRead expected residual-stream hidden size "
                f"{self.residual_stream_hidden_size}, got {hidden_states.shape[-1]}."
            )
        return streamwise_sigmoid_read(
            hidden_states,
            self.read_map(return_logits=True),
            self.num_streams,
            output_dtype=self.branch_input_dtype,
        )


class StreamwiseSigmoidWideResidualConnection(ResidualConnection):
    """Streamwise residual maps restricted to a read or write at construction.

    Override ``_compute_read`` or ``_compute_write`` to customize arithmetic shared by
    eager execution and checkpoint replay. State or replay-policy changes belong in
    the corresponding eager and checkpoint methods.
    """

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

    def _eager_read(
        self,
        hidden_states: Tensor,
        *,
        fp32_residual_connection: bool,
        branch_input_dtype: torch.dtype | None,
        execution_context: object | None,
    ) -> tuple[Tensor, ResidualConnectionState]:
        """Read in branch precision before promoting the carried residual."""

        del execution_context
        branch_input = self._compute_read(hidden_states, branch_input_dtype=branch_input_dtype)
        # Full-model FP32-residual inputs are normally already FP32. Preserve the defensive
        # promotion for direct low-precision inputs, after the read has consumed its input.
        residual_stream = hidden_states.float() if fp32_residual_connection else hidden_states
        return branch_input, (residual_stream,)

    def _compute_read(
        self, hidden_states: Tensor, *, branch_input_dtype: torch.dtype | None
    ) -> Tensor:
        """Compute the branch input for eager execution and checkpoint replay."""

        return streamwise_sigmoid_read(
            hidden_states,
            self.read_map(return_logits=True),
            self.num_streams,
            output_dtype=branch_input_dtype,
        )

    def _eager_write(
        self,
        branch_output: ResidualBranchOutput,
        state: ResidualConnectionState,
        *,
        dropout_probability: float,
        training: bool,
        execution_context: object | None,
    ) -> Tensor:
        del execution_context
        output, bias = branch_output if isinstance(branch_output, tuple) else (branch_output, None)
        return self._compute_write(
            output, bias, *state, dropout_probability=dropout_probability, training=training
        )

    def _compute_write(
        self,
        branch_update: Tensor,
        bias: Tensor | None,
        *state: Tensor,
        dropout_probability: float,
        training: bool,
    ) -> Tensor:
        """Compute the residual update for eager execution and checkpoint replay."""

        residual_stream = state[0]
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
        execution_context: object | None,
    ) -> tuple[Tensor, ResidualConnectionState]:
        """Replay WR's read while retaining its incoming residual stream.

        Bind the requested branch-input dtype so eager execution and replay use the same dtype.
        Adapter hooks surround the outer module call, not this arithmetic reconstruction.
        """

        del execution_context

        branch_input = context.checkpoint(
            partial(self._compute_read, branch_input_dtype=branch_input_dtype), hidden_states
        )
        if not torch.is_tensor(branch_input):
            raise TypeError("Checkpointed residual read must return a tensor.")

        # This mirrors WR's defensive state promotion. Normal full-model
        # FP32-residual inputs are already FP32, making .float() an alias rather than a cast kernel;
        # direct/custom low-precision inputs still need the conversion to preserve replay state.
        residual_stream = hidden_states.float() if fp32_residual_connection else hidden_states
        return branch_input, (residual_stream,)

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
        """Replay WR's whole write, whose output has independent storage."""

        del execution_context
        output, bias = branch_output if isinstance(branch_output, tuple) else (branch_output, None)
        hidden_states = context.checkpoint(
            partial(
                self._compute_write, dropout_probability=dropout_probability, training=training
            ),
            output,
            bias,
            *state,
        )
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


def map_residual_sharded_keys(sharded_state_dict, prefix: str) -> None:
    """Give the residual read/write parameters their existing checkpoint keys.

    ``load_residual_state`` covers plain ``load_state_dict``; distributed checkpoints are keyed by
    ``sharded_state_dict``, so without this a dist-checkpoint load reports every read map, write
    map and retention as unexpected and leaves them at their initial values.
    """
    from megatron.core.dist_checkpointing.utils import apply_prefix_mapping

    apply_prefix_mapping(
        sharded_state_dict,
        {f"{prefix}{current}": f"{prefix}{legacy}" for current, legacy in _RESIDUAL_STATE_PREFIXES},
    )


class StreamwiseSigmoidResidualReadout(nn.Module):
    """Learned streamwise readout from D' to D before final normalization."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        if config.wide_residual is None:
            raise ValueError("StreamwiseSigmoidResidualReadout requires wide_residual config.")
        wr = config.wide_residual
        self.input_hidden_size = wr.num_streams * config.hidden_size
        self.output_hidden_size = config.hidden_size
        self.exit_map = StreamwiseSigmoidMap(config, map_kind="read")

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Mix the full-width streams into one backbone-width activation."""

        if hidden_states.shape[-1] != self.input_hidden_size:
            raise ValueError(
                "Streamwise sigmoid readout expected hidden size "
                f"{self.input_hidden_size}, got {hidden_states.shape[-1]}."
            )
        return streamwise_sigmoid_read(
            hidden_states, self.exit_map(return_logits=True), self.exit_map.num_streams
        )


def build_wide_residual_readout(
    config: TransformerConfig,
) -> StreamwiseSigmoidResidualReadout | None:
    """Build the streamwise model-boundary readout when wide residuals are enabled."""

    if config.wide_residual is None:
        return None
    return StreamwiseSigmoidResidualReadout(config)


def expand_wide_residual_stream(hidden_states: Tensor, num_streams: int) -> Tensor:
    """Replicate an ordinary-width activation into contiguous residual streams."""

    if num_streams <= 1:
        raise ValueError(f"num_streams must be greater than one, got {num_streams}.")
    repeat_shape = [1] * hidden_states.ndim
    repeat_shape[-1] = num_streams
    return hidden_states.repeat(*repeat_shape)


def wide_residual_recompute_enabled(config: TransformerConfig, training: bool) -> bool:
    """Return whether selective residual-stream replay is active for this forward."""

    return bool(
        training
        and torch.is_grad_enabled()
        and config.recompute_granularity == "selective"
        and config.recompute_modules is not None
        and "residual_stream" in config.recompute_modules
    )


# Fused residual seams: a layer's last write is a DeferredWrite linked to the next layer's
# WriteReadNorm. The DeferredWrite hands its write to that (empty) read instead of applying it, and
# the read takes it and applies it with its read and RMSNorm in one kernel.


@dataclass(frozen=True)
class SeamInput:
    """A layer input whose residual read and RMSNorm a fused seam kernel already applied."""

    residual: Tensor
    branch_input: Tensor


class WriteReadNorm(nn.Module):
    """Applies the previous layer's pending residual write, then this layer's read and RMSNorm, in
    one fused kernel; owns the read logits and the norm weight.

    A fused layer spec selects this as the layer's read connection (with an identity input norm);
    its predecessor's DeferredWrite always hands its write here, and the tensor this read is called
    with is only that write's unwritten input residual. Reader zero is MXFP8 for an eligible
    consumer, otherwise BF16. A SeamInput (a fused shortcut block's two-reader kernel already
    applied the write) passes through. Returns ``(branch_input, (residual,))`` like a
    residual-connection read.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        branch_name: str,
        operation: ResidualConnectionOperation = "read",
    ) -> None:
        del branch_name
        if operation != "read":
            raise ValueError(f"WriteReadNorm is a read connection, got operation={operation!r}")
        super().__init__()
        self.read_map = StreamwiseSigmoidMap(config, map_kind="read")
        device = "cpu" if config.use_cpu_initialization else torch.cuda.current_device()
        self.weight = nn.Parameter(
            torch.ones(config.hidden_size, dtype=config.params_dtype, device=device)
        )
        setattr(self.weight, "sequence_parallel", config.sequence_parallel)
        self.eps = config.layernorm_epsilon
        self.num_streams = config.wide_residual.num_streams
        self.residual_stream_hidden_size = self.num_streams * config.hidden_size
        self.branch_hidden_size = config.hidden_size
        # The Linear that consumes reader zero as MXFP8, if any (set when the stack is wired).
        self.consumer = None
        # The write handed here by the previous layer's DeferredWrite, until this read takes it.
        self.pending = None
        # Checkpoint replay (training): every N consecutive seams form one interval. The stack's
        # second layer has its first seam, and one seam follows per layer.
        interval = config.residual_stream_recompute_num_layers
        self.replay_every = interval if interval is not None and interval > 1 else None
        self.starts_replay = self.replay_every is not None and (
            (layer_number - 2) % self.replay_every == 0
        )

    def hand(self, item) -> None:
        """Hand ``item`` (PendingWideResidualWrites or SeamInput) to this empty read."""
        if self.pending is not None:
            raise RuntimeError("a deferred residual write for this read was not applied")
        self.pending = item

    def take(self):
        """Take the item handed to this read, leaving it empty."""
        item, self.pending = self.pending, None
        if item is None:
            raise RuntimeError("no deferred residual write was handed to this fused read")
        return item

    def _replay_interval(self, pending):
        """The checkpoint-replay interval this read's seam records into (None unless training
        with replay): a new one at an interval start, else the one tagged on the input residual
        by the previous seam."""
        if self.replay_every is None or not (self.training and torch.is_grad_enabled()):
            return None
        if self.starts_replay:  # linked to the interval whose last write this seam applies
            return ReplayInterval(getattr(pending.residual, "_replay_interval", None))
        interval = getattr(pending.residual, "_replay_interval", None)
        if interval is None:
            raise RuntimeError("a replayed seam's input residual carries no replay interval")
        return interval

    def _hooked_weight(self) -> Tensor:
        """The norm weight after the forward pre-hooks calling this read would run (DDP's
        overlapped parameter all-gather waits for the weight there), for a kernel applying it
        before this read is called."""
        for hook_id, hook in tuple(self._forward_pre_hooks.items()):
            if hook_id in self._forward_pre_hooks_with_kwargs:
                hook(self, (), {})
            else:
                hook(self, ())
        return self.weight

    def apply_write(self, pending, shortcut: WriteReadNorm | None = None):
        """Write ``pending`` and return ``(residual, branch_input)`` (plus ``shortcut``'s
        normalized input, for a shortcut block's second reader) from one fused kernel. Reader zero
        (this read) goes as MXFP8 to its consumer Linear if it has one (chosen at construction),
        and the write is recorded into this seam's replay interval, which the output residual
        carries on to the next seam."""
        replay = self._replay_interval(pending)
        # Inference builds no autograd graph (and uses the inference MXFP8 eligibility) even with
        # gradients enabled.
        with torch.no_grad() if InferenceMode.is_active() else nullcontext():
            if self.consumer is not None:
                check_mxfp8_call(pending.routed)  # never falls back to BF16
            options = dict(
                mxfp8_consumer=self.consumer,
                num_streams=self.num_streams,
                epsilon=self.eps,
                replay=replay,
            )
            if shortcut is None:
                outputs = shortcut_add_wide_residual_write_read_rmsnorm_quant(
                    pending, self.read_map(return_logits=True), self.weight, **options
                )
            else:
                # This read's weight is used before its layer calls it: run its hooks now.
                outputs = shortcut_add_wide_residual_write_two_reader_rmsnorm_quant(
                    pending,
                    (self.read_map(return_logits=True), shortcut.read_map(return_logits=True)),
                    (self._hooked_weight(), shortcut.weight),
                    **options,
                )
        if replay is not None:
            outputs[0]._replay_interval = replay
        return outputs

    def forward(
        self,
        hidden_states,
        *,
        fp32_residual_connection=False,
        branch_input_dtype=None,
        execution_context=None,
        recompute_context=None,
    ):
        """Apply the write handed to this read with this read and norm."""
        # The kernel keeps the residual's dtype; ``hidden_states`` is the unwritten input residual.
        # Fusion rejects residual-stream recomputation (seams replay through their intervals).
        del hidden_states, fp32_residual_connection, branch_input_dtype
        del execution_context, recompute_context
        item = self.take()
        if isinstance(item, SeamInput):
            return item.branch_input, (item.residual,)
        residual, branch_input = self.apply_write(item)
        return branch_input, (residual,)

    @staticmethod
    def residual_stream(state) -> Tensor:
        """The carried residual stream of a read's state."""
        return state[0]


class ShortcutAddWideResidualWriteReadNorm(WriteReadNorm):
    """A fused shortcut block's shortcut read and pre-MLP RMSNorm (owning their parameters).

    Linked to the compute layer's WriteReadNorm (``link``), it applies the write handed to that
    read with both reads and norms in one two-reader kernel; the compute read's own call then
    returns its normalized input. Returns ``(residual, shortcut_input)``.
    """

    compute_read: WriteReadNorm | None = None

    def link(self, compute_read: WriteReadNorm) -> None:
        """Serve ``compute_read``, the block's compute layer read (a reference, not a submodule)."""
        object.__setattr__(self, "compute_read", compute_read)

    def forward(self) -> tuple[Tensor, Tensor]:  # pylint: disable=arguments-differ
        compute_read = self.compute_read
        # This seam is the compute read's (its replay interval, consumer and epsilon).
        residual, branch_input, shortcut_input = compute_read.apply_write(
            compute_read.take(), shortcut=self
        )
        compute_read.hand(SeamInput(residual, branch_input))
        return residual, shortcut_input


class DeferredWrite(StreamwiseSigmoidWideResidualConnection):
    """A residual write that its linked WriteReadNorm (the next layer's read) applies in its fused
    kernel. Returns the unwritten input residual; only that read may consume it.

    A fused layer spec selects this as every write but the stack's last (built like a write
    connection); the stack links each one to its successor's read (``link``).
    """

    target: nn.Module | None = None

    def link(self, target: nn.Module) -> None:
        """Hand this write to ``target``, the next layer's read (a reference, not a submodule)."""
        object.__setattr__(self, "target", target)

    def forward(  # pylint: disable=arguments-differ
        self,
        branch_output,
        *,
        state,
        dropout_probability,
        training,
        shared=None,
        execution_context=None,
        recompute_context=None,
    ):
        """Hand this write (plus an optional shared-expert update: a shortcut add) to the target
        read as PendingWideResidualWrites; return the unwritten residual."""
        del dropout_probability, training  # fusion requires hidden_dropout == 0
        # Fusion rejects residual-stream recomputation (seams replay through their intervals).
        del execution_context, recompute_context
        update, bias = branch_output
        if bias is not None:
            raise ValueError("fused residual seams require a bias-free branch projection")
        retain = None if self.retention is None else self.retention(return_logits=True)
        write_logits = self.write_map(return_logits=True)
        pending = PendingWideResidualWrites(state[0], update, shared, write_logits, retain)
        self.target.hand(pending)
        return state[0]


def seam_connections(layer) -> tuple[nn.Module, nn.Module]:
    """A wide-residual layer's ``(read, write)`` connection modules, as its class names them."""
    read, write = type(layer).get_wide_residual_modules(layer.submodules_config)
    return getattr(layer, read), getattr(layer, write)


# MXFP8 handoff: a fused read writes reader zero only as rowwise MXFP8, the operand of its
# consumer Linear's forward GEMM. The consumer's weight-gradient GEMM needs the columnwise operand:
# after the consumer's forward, a hook on TE's own saved slot for that operand returns, at unpack,
# the columnwise operand replayed bit-exactly from the seam (its replay interval runs before any of
# its GEMMs). TE's backward, including weight-gradient accumulation and distributed-weight
# finalization, runs unchanged.


def mxfp8_linear_eligible(linear, config) -> bool:
    """Construction time: whether ``linear`` may take a seam's reader zero directly as MXFP8.

    Its layer's precision is MXFP8 (Megatron fixes precision per layer) at a hidden size the
    kernel quantizes, and it is a TELinear taking its input whole (no TP, no userbuffer overlap
    or quantized parameters).
    """
    from megatron.core.extensions.transformer_engine import TELinear
    from megatron.core.fusions.wide_residual_shortcut.kernel_utils import WIDTHS, qualified_device

    return bool(
        config.fp8
        and config.fp8_recipe == "mxfp8"
        and config.hidden_size in WIDTHS
        and qualified_device(torch.cuda.get_device_properties(torch.cuda.current_device()))
        and isinstance(linear, TELinear)
        and getattr(linear, "te_quant_params", None) is None
        and getattr(linear, "tp_size", 1) == 1
        and not any(
            getattr(linear, name, False)
            for name in (
                "ub_overlap_ag_fprop",
                "ub_overlap_ag_dgrad",
                "ub_overlap_rs_dgrad",
                "ub_bulk_dgrad",
                "ub_bulk_wgrad",
            )
        )
    )


def mxfp8_consumer(linear, config):
    """``linear``, registered as its read's MXFP8 consumer, if it takes that read directly as
    MXFP8 (decided once, at construction): after each forward on a seam's operand, its saved copy
    is handed over to the replayed columns (``_install_mxfp8_unpack``)."""
    if linear is None or not mxfp8_linear_eligible(linear, config):
        return None
    if not getattr(linear, "_shortcut_mxfp8_consumer", False):
        linear.register_forward_hook(_after_mxfp8_consumer_forward, with_kwargs=True)
        linear._shortcut_mxfp8_consumer = True
    return linear


def _after_mxfp8_consumer_forward(linear, args, kwargs, output):
    _install_mxfp8_unpack(args[0] if args else next(iter(kwargs.values())), output, linear)


def _install_mxfp8_unpack(tensor, output, linear):
    """After ``linear``'s forward on a seam's MXFP8 operand: its backward unpacks the replayed
    columnwise operand in place of the saved rowwise one (TE's own save slot only; a scope-wide
    saved-tensor hook would detach the saved weight and lose GTP metadata)."""
    reader = reader_for(tensor)
    if reader is None or not torch.is_grad_enabled():
        return
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

    recipe = FP8GlobalStateManager.get_fp8_recipe()
    if not FP8GlobalStateManager.is_fp8_enabled() or not recipe.mxfp8():
        raise RuntimeError("shortcut MXFP8 input requires an MXFP8 TE linear consumer")
    if getattr(recipe, "backward_override", None) is not None:
        raise RuntimeError("shortcut MXFP8 handoff requires ordinary MXFP8 backward")
    if reader.consumer() is not linear:
        raise RuntimeError("a seam's MXFP8 operand must be consumed only by its declared Linear")
    output = output[0] if isinstance(output, tuple) else output
    node = output.grad_fn
    key = tensor._rowwise_data.untyped_storage()._cdata
    matched = 0
    for value, slot in zip(node.saved_tensors, node._raw_saved_tensors):
        data = getattr(value, "_rowwise_data", None)
        if data is not None and data.untyped_storage()._cdata == key:
            slot.register_hooks(torch.Tensor.size, lambda size: reader.columns().view(size))
            matched += 1
    if matched != 1:
        raise RuntimeError("TE did not save exactly one shortcut MXFP8 activation")


def check_mxfp8_call(tensor) -> None:
    """Per call, for a consumer chosen at construction: raise unless its FP8 context is active
    and the kernel takes this token count with an MXFP8 reader. Precision never changes per
    call: training shapes are fixed for a run (the tuned set), and inference rows must be
    divisible by 32, as TE's MXFP8 GEMM requires too (Megatron's inference pads to 64)."""
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

    from megatron.core.fusions.wide_residual_shortcut.kernel_utils import SHAPES

    if not FP8GlobalStateManager.is_fp8_enabled():
        raise RuntimeError("an MXFP8 layer's fused seam ran outside its FP8 context")
    rows, hidden = tensor.numel() // tensor.shape[-1], tensor.shape[-1]
    training = torch.is_grad_enabled()
    if (rows, hidden) not in SHAPES if training else rows % 32:
        raise ValueError(
            f"the MXFP8 seam handoff does not support {rows} tokens at hidden size {hidden} in "
            + (
                "training (supported: multiples of 1024 from 2048 tokens, up to 16384 above "
                "hidden 4096)"
                if training
                else "inference (rows must be divisible by 32)"
            )
        )


def _input_projection(layer):
    """The Linear a layer's input RMSNorm was fused into (as its class locates it), or None when
    the norm had its own slot."""
    location = type(layer).get_input_layernorm_module(layer.submodules_config)
    if len(location) != 2:
        return None
    branch, projection = location
    # A fused MoE layer's emptied norm slot points at mlp.linear_fc1, which an MoELayer lacks (its
    # router and experts all take the read: no single consumer).
    return getattr(getattr(layer, branch), projection, None)


def _unfused_error(layer) -> ValueError:
    """The error rejecting ``layer``, which was not built as a fused-seam layer, with the reason."""
    layer_class = type(layer)
    if not hasattr(layer_class, "get_input_layernorm_module"):
        reason = f"{layer_class.__name__} is not a wide-residual layer"
    else:
        try:
            location = layer_class.get_input_layernorm_module(layer.submodules_config)
            reason = (
                f"its input RMSNorm ({'.'.join(location)}) was not moved into its read: build the "
                "stack from hybrid_layer_specs.shortcut_fusion_stack_spec"
            )
        except ValueError as error:
            reason = str(error)
    return ValueError(
        f"moe_shortcut_wide_residual_norm_fusion cannot fuse layer {layer.layer_number} "
        f"({layer_class.__name__}): {reason}"
    )


def link_residual_seams(layers) -> None:
    """Check and wire a fused stack's residual seams (``layers``: physical layers in order).

    The fused spec makes every write but the last a DeferredWrite and every read but the first a
    WriteReadNorm. Each deferred write is linked to the next layer's fused read, whose layer's
    input projection takes it as MXFP8 when eligible; a layer the spec could not fuse is rejected
    with its reason, and so is a one-layer stack.
    """
    connections = []
    for layer in layers:
        if not hasattr(type(layer), "get_wide_residual_modules"):
            raise _unfused_error(layer)
        try:
            connections.append(seam_connections(layer))
        except ValueError:
            raise _unfused_error(layer) from None
    for index, (layer, (_, write)) in enumerate(zip(layers, connections)):
        if index + 1 == len(layers):
            if isinstance(write, DeferredWrite):
                raise ValueError(
                    "moe_shortcut_wide_residual_norm_fusion requires at least two layers: the "
                    f"write of layer {layer.layer_number} is deferred to no following read"
                )
            continue  # the stack's last write stays plain
        successor, read = layers[index + 1], connections[index + 1][0]
        if not isinstance(write, DeferredWrite):
            raise _unfused_error(layer)
        if not isinstance(read, WriteReadNorm):
            raise _unfused_error(successor)
        write.link(read)
        read.consumer = mxfp8_consumer(_input_projection(successor), successor.config)
