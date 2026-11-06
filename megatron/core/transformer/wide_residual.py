# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
"""Wide-residual parameters, connection policy, and model-boundary helpers."""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from megatron.core.fusions.wide_residual_shortcut.api import (
    PendingWideResidualWrites,
    _apply_boundary,
)
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
from megatron.core.typed_torch import apply_module

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


def put_pending_write(read: nn.Module, item) -> None:
    """Hand ``item`` (PendingWideResidualWrites or SeamInput) to the empty ``read``."""
    if read.pending is not None:
        raise RuntimeError("a deferred residual write for this read was not applied")
    read.pending = item


def take_pending_write(read: nn.Module):
    """Take the write handed to ``read``, leaving it empty."""
    item, read.pending = read.pending, None
    if item is None:
        raise RuntimeError("no deferred residual write was handed to this fused read")
    return item


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

    def __init__(self, config: TransformerConfig, layer_number: int, branch_name: str) -> None:
        del layer_number, branch_name
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
        # (branch, input projection) that may consume reader zero as MXFP8; set by the layer.
        self.consumer = None
        # The write handed here by the previous layer's DeferredWrite, until this read takes it.
        self.pending = None

    def forward(self, hidden_states, *, fp32_residual_connection=False, branch_input_dtype=None):
        """Apply the write handed to this read with this read and norm."""
        # The kernel keeps the residual's dtype; ``hidden_states`` is the unwritten input residual.
        del hidden_states, fp32_residual_connection, branch_input_dtype
        item = take_pending_write(self)
        if isinstance(item, SeamInput):
            return item.branch_input, (item.residual,)
        residual, branch_input = apply_pending_write(
            item, (self,), (self.weight,), self.eps, self.consumer
        )
        return branch_input, (residual,)

    def apply_with(self, read, norm) -> tuple[Tensor, Tensor]:
        """Run a fused shortcut block's two-reader seam: apply the write handed to this read with
        this read/norm and the block's shortcut ``read``/``norm`` in one kernel.

        This read's own call then returns its normalized input. Returns ``(residual,
        shortcut_input)``.
        """
        pending = take_pending_write(self)
        gammas = (hooked_weight(self), hooked_weight(norm))  # neither module is called
        residual, branch_input, shortcut_input = apply_pending_write(
            pending, (self, read), gammas, self.eps, self.consumer
        )
        put_pending_write(self, SeamInput(residual, branch_input))
        return residual, shortcut_input

    @staticmethod
    def residual_stream(state) -> Tensor:
        """The carried residual stream of a read's state."""
        return state[0]


class DeferredWrite(StreamwiseSigmoidWideResidualConnection):
    """A residual write that its linked WriteReadNorm (the next layer's read) applies in its fused
    kernel. Returns the unwritten input residual; only that read may consume it.

    A fused layer spec selects this as every write but the stack's last; the stack links each one
    to its successor's read (``link``).
    """

    target: nn.Module | None = None

    def __init__(self, config: TransformerConfig, layer_number: int, branch_name: str) -> None:
        super().__init__(config, layer_number, branch_name, operation="write")

    def link(self, target: nn.Module) -> None:
        """Hand this write to ``target``, the next layer's read (a reference, not a submodule)."""
        object.__setattr__(self, "target", target)

    def forward(  # pylint: disable=arguments-differ
        self, branch_output, *, state, dropout_probability, training, shared=None
    ):
        """Hand this write (plus an optional shared-expert update: a shortcut add) to the target
        read; return the unwritten residual."""
        del dropout_probability, training  # fusion requires hidden_dropout == 0
        put_pending_write(self.target, defer_write(state[0], branch_output, self, shared))
        return state[0]


def defer_write(residual: Tensor, branch_output, write, shared: Tensor | None = None):
    """A branch's residual write as PendingWideResidualWrites (plus an optional shared update)."""
    update, bias = branch_output
    if bias is not None:
        raise ValueError("fused residual seams require a bias-free branch projection")
    retain = None if write.retention is None else write.retention(return_logits=True)
    return PendingWideResidualWrites(
        residual, update, shared, write.write_map(return_logits=True), retain
    )


def hooked_weight(module: nn.Module) -> Tensor:
    """``module.weight`` after the forward pre-hooks calling ``module`` would run (DDP's overlapped
    parameter all-gather waits for the weight there), for norms a kernel applies uncalled."""
    for hook_id, hook in tuple(module._forward_pre_hooks.items()):
        if hook_id in module._forward_pre_hooks_with_kwargs:
            hook(module, (), {})
        else:
            hook(module, ())
    return module.weight


def apply_pending_write(pending, reads, gammas, epsilon, consumer=None):
    """Write ``pending`` and return ``(residual, *rmsnorm(read(residual)))`` from one fused kernel;
    reader zero goes as MXFP8 to the ``(branch, input projection)`` ``consumer`` if eligible."""
    # Inference builds no autograd graph (and uses the inference MXFP8 eligibility) even with
    # gradients enabled.
    with torch.no_grad() if InferenceMode.is_active() else nullcontext():
        return _apply_pending_write(pending, reads, gammas, epsilon, consumer)


def _apply_pending_write(pending, reads, gammas, epsilon, consumer):
    from megatron.core.fusions.wide_residual_shortcut import mxfp8_handoff

    streams = reads[0].num_streams
    linear = None
    if consumer is not None:
        branch, linear = consumer
        # A checkpointed input projection (GDP's recompute_in_proj) keeps its weight hooks in an
        # inner autograd graph the deferred weight gradient cannot order: with fine-grained
        # offloading it stays BF16, and training needs GTP's explicit weight finalization.
        recompute = getattr(branch, "recompute_in_proj", False)
        if (
            linear is None
            or (recompute and branch.config.fine_grained_activation_offloading)
            or not mxfp8_handoff.eligible(linear, pending.routed, streams)
        ):
            linear = None
        elif (
            recompute
            and torch.is_grad_enabled()
            and not getattr(linear.weight, "is_gtp_weight_remat", False)
        ):
            raise ValueError(
                "direct shortcut MXFP8 with GDP input-projection recompute requires GTP weight "
                "finalization; disable gdp_in_proj recompute or use a BF16 handoff"
            )
    logits = tuple(read.read_map(return_logits=True) for read in reads)
    return _apply_boundary(
        pending, logits, gammas, mxfp8_consumer=linear, num_streams=streams, epsilon=epsilon
    )
