# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
"""Wide-residual parameters, connection policy, and model-boundary helpers."""

from __future__ import annotations

import math
import weakref
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.enums import Fp8Recipe
from megatron.core.fusions.wide_residual_shortcut import (
    ColumnSlot,
    PendingWideResidualWrites,
    ReplayInterval,
    check_mxfp8_call,
    fused_seam,
    install_columns,
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


def has_combined_residual_keys(checkpoint_keys) -> bool:
    """Whether a distributed checkpoint stores each residual branch as one combined module.

    Such checkpoints key the read map, write map and retention of a branch under
    residual_connection{suffix} rather than residual_read{suffix} / residual_write{suffix}.

    Args:
        checkpoint_keys: ShardedTensor keys of the checkpoint, e.g.
            dist_checkpointing.load_tensors_metadata(checkpoint_dir).keys().
    """
    combined_prefixes = tuple(combined for _, combined in _RESIDUAL_STATE_PREFIXES)
    return any(
        key.startswith(combined) or f".{combined}" in key
        for key in checkpoint_keys
        for combined in combined_prefixes
    )


def apply_combined_residual_keys(
    sharded_state_dict: ShardedStateDict, prefix: str, metadata: dict | None
) -> None:
    """Address a layer's residual tensors by their combined residual_connection{suffix} keys.

    The loader sets metadata['combined_residual_keys'] for checkpoints that store each residual
    branch as one combined module; other checkpoints keep the residual_read{suffix} /
    residual_write{suffix} keys.

    Args:
        sharded_state_dict: The layer's sharded state dict, renamed in place.
        prefix: The layer's state dict prefix.
        metadata: Sharded state dict metadata from the checkpoint load, or None.
    """
    if not metadata or not metadata.get("combined_residual_keys"):
        return
    apply_prefix_mapping(
        sharded_state_dict,
        {f"{prefix}{split}": f"{prefix}{combined}" for split, combined in _RESIDUAL_STATE_PREFIXES},
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
    with is only that write's unwritten input residual. Every module consuming the read is linked
    to it (``link_consumer``), and the stack fixes each consumer's format (``link_residual_seams``):
    MXFP8 if it quantizes the read, else BF16. A read with two consumers is returned as one tensor
    per consumer, in link order. A SeamInput (a fused shortcut block's two-reader kernel already
    applied the write) passes through.
    Returns ``(branch_input, (residual,))`` like a residual-connection read.
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
        # Linked when the stack is wired (references, not submodules): every module consuming
        # this read with its layer, the layer whose seam emits the read (a block's shortcut read is
        # emitted by its compute layer's), and the stack's previous fused read.
        self.consumers, self.seam_layer, self.previous_read = [], None, None
        self.consumer_formats = ()  # fixed by the stack (``resolve_formats``)
        # This forward's replay interval and ColumnSlot (a weak reference, for the consumers'
        # saved-slot hooks).
        self.interval = self.columns_slot = None
        # The write handed here by the previous layer's DeferredWrite, until this read takes it.
        self.pending = None
        # Checkpoint replay: every N consecutive seams form one interval (without replay, every
        # seam its own). The stack's second layer has its first seam, which starts the first
        # interval, and one seam follows per layer.
        every = config.residual_stream_recompute_num_layers
        self.starts_replay = every is None or every <= 1 or (layer_number - 2) % every == 0

    def link_consumer(self, module, layer) -> None:
        """Link ``module`` of ``layer`` as a consumer of this read."""
        self.consumers.append((module, layer))

    def resolve_formats(self, runs_mxfp8) -> None:
        """Fix each consumer's format, in link order (the seam returns one output per consumer,
        whose gradients its backward kernel sums): MXFP8 if it quantizes the read, an eligible
        TELinear whose layer and the seam's both run in MXFP8 (``runs_mxfp8(layer)``, as the
        stack's forward enters their quantization contexts), else BF16."""
        if not 1 <= len(self.consumers) <= 2:
            raise RuntimeError("a fused read feeds one or two linked consumers")
        formats = []
        for module, layer in self.consumers:
            quantizes = runs_mxfp8(self.seam_layer) and runs_mxfp8(layer)
            if quantizes and mxfp8_linear_eligible(module):
                if getattr(module, "_shortcut_mxfp8_read", None) is not None:
                    raise RuntimeError("a Linear takes at most one fused read as MXFP8")
                object.__setattr__(module, "_shortcut_mxfp8_read", self)
                module.register_forward_hook(
                    partial(_after_mxfp8_consumer_forward, self), with_kwargs=True
                )
                formats.append("mxfp8")
            else:
                formats.append("bf16")
        self.consumer_formats = tuple(formats)

    @property
    def format(self) -> str:
        """The seam kernel's output format for this read."""
        formats = set(self.consumer_formats)
        return formats.pop() if len(formats) == 1 else "both"

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

    def _open_interval(self) -> ReplayInterval:
        """This forward's replay interval: a new one where an interval starts (following the
        previous read's, which its first seam's backward replays), else the previous read's."""
        if self.starts_replay:
            previous = None if self.previous_read is None else self.previous_read.interval
            self.interval = ReplayInterval(previous)
        else:
            self.interval = self.previous_read.interval
        return self.interval

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
        normalized input, for a shortcut block's second reader) from one fused kernel. Each read
        is emitted in its consumers' formats (one output per consumer). With gradients the write
        is recorded into this forward's replay interval, which fills each MXFP8 read's
        ColumnSlot."""
        from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
            is_batch_invariant_mode_enabled,
        )

        reads = (self,) if shortcut is None else (self, shortcut)
        # Inference builds no autograd graph (and uses the inference MXFP8 eligibility) even with
        # gradients enabled.
        with torch.no_grad() if InferenceMode.is_active() else nullcontext():
            if any(read.format != "bf16" for read in reads):
                check_mxfp8_call(pending.routed)  # never falls back to BF16
            interval, slots = None, (None,) * len(reads)
            if torch.is_grad_enabled():
                interval = self._open_interval()
                slots = tuple(ColumnSlot() if read.format != "bf16" else None for read in reads)
            for read, slot in zip(reads, slots):
                read.columns_slot = None if slot is None else weakref.ref(slot)
            # A shortcut seam runs before this read's layer calls it: run its weight's hooks now.
            weight = self.weight if shortcut is None else self._hooked_weight()
            return fused_seam(
                pending,
                tuple(read.read_map(return_logits=True) for read in reads),
                (weight, *(read.weight for read in reads[1:])),
                outputs=tuple(read.consumer_formats for read in reads),
                columns=slots,
                interval=interval,
                num_streams=self.num_streams,
                epsilon=self.eps,
                batch_invariant=is_batch_invariant_mode_enabled(),
            )

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
    read with both reads and norms in one two-reader kernel (each in its own use's format); the
    compute read's own call then returns its normalized input. Returns
    ``(residual, shortcut_input)``.
    """

    compute_read: WriteReadNorm | None = None

    def link(self, compute_read: WriteReadNorm) -> None:
        """Serve ``compute_read``, the block's compute layer read (references, not submodules)."""
        object.__setattr__(self, "compute_read", compute_read)
        object.__setattr__(compute_read, "shortcut", self)

    def forward(self) -> tuple[Tensor, Tensor]:  # pylint: disable=arguments-differ
        compute_read = self.compute_read
        # This seam is the compute read's (its replay interval and epsilon).
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


# MXFP8 handoff: a fused read's eligible consumer Linears take it as rowwise MXFP8 from the seam
# kernel, which emits it directly (format "mxfp8") or next to the BF16 output other modules read
# ("both", one output per consumer). Their weight-gradient GEMMs need the columnwise operand: after
# each consumer's forward, TE's own saved slot for that operand unpacks the read's ColumnSlot,
# which the read's replay interval fills bit-exactly before any of its GEMMs. TE's backward,
# including weight-gradient accumulation and distributed-weight finalization, runs unchanged.


def mxfp8_linear_eligible(linear) -> bool:
    """Whether ``linear`` can take a read directly as MXFP8 when its layer runs in MXFP8: a
    TELinear without a per-module quantization override on a qualified GB300 (TransformerConfig
    rejects MXFP8 fusion at unsupported hidden sizes, with TP or userbuffer overlap)."""
    from megatron.core.extensions.transformer_engine import TELinear
    from megatron.core.fusions.wide_residual_shortcut.kernel_utils import qualified_device

    return bool(
        isinstance(linear, TELinear)
        and getattr(linear, "te_quant_params", None) is None
        and qualified_device(torch.cuda.get_device_properties(torch.cuda.current_device()))
    )


def _after_mxfp8_consumer_forward(read, linear, args, kwargs, output):
    del linear
    if not torch.is_grad_enabled():
        return
    slot = None if read.columns_slot is None else read.columns_slot()
    if slot is None:
        raise RuntimeError("an MXFP8 consumer took a fused read emitted without gradients")
    install_columns(output, args[0] if args else next(iter(kwargs.values())), slot)


def _per_layer_fp8(config) -> bool:
    """Whether ``HybridStack.forward`` enters each layer's FP8 context itself (delayed scaling
    wraps the whole pass in one context instead)."""
    return bool(config.fp8) and config.fp8_recipe != Fp8Recipe.delayed


def runs_mxfp8(stack_config, layer) -> bool:
    """Whether ``HybridStack.forward`` runs ``layer`` under an MXFP8 autocast: the per-layer
    context ``get_fp8_context`` builds from the layer's own config."""
    from megatron.core.fp8_utils import is_first_last_bf16_layer

    config = layer.config
    return (
        _per_layer_fp8(stack_config)
        and bool(config.fp8)
        and config.fp8_recipe == Fp8Recipe.mxfp8
        and not is_first_last_bf16_layer(config, layer.layer_number - 1)
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


def link_residual_seams(layers, stack_config) -> tuple:
    """Check and wire a fused stack's residual seams (``layers``: physical layers in order, after
    shortcut grouping, whose blocks have linked their own MoE and shortcut reads); returns the
    stack's fused reads.

    The fused spec makes every write but the last a DeferredWrite and every read but the first a
    WriteReadNorm. Each deferred write is linked to the next layer's fused read, and each read not
    yet linked to its consumer: a layer's input projection, or for an MoE layer outside a shortcut
    block the MoE layer itself. Its read reaches the router, shared expert and dispatcher (or
    latent projection) as one BF16 tensor, so under MXFP8 its Linears quantize it themselves, and
    autograd sums the three consumers' gradients. Every read's formats are then fixed with
    ``runs_mxfp8``, whether the stack's forward runs a layer in MXFP8. Each read follows the
    previous one's replay interval, and the stack's last write closes the last interval: its
    output, the stack's residual, replays it before its readers' consumers' backward. A layer the
    spec could not fuse is rejected with its reason, and so is a one-layer stack.
    """
    from megatron.core.transformer.moe.moe_layer import MoELayer

    connections = []
    for layer in layers:
        if not hasattr(type(layer), "get_wide_residual_modules"):
            raise _unfused_error(layer)
        try:
            connections.append(seam_connections(layer))
        except ValueError:
            raise _unfused_error(layer) from None
    reads = []
    for index, (layer, (_, write)) in enumerate(zip(layers, connections)):
        if index + 1 == len(layers):
            if isinstance(write, DeferredWrite):
                raise ValueError(
                    "moe_shortcut_wide_residual_norm_fusion requires at least two layers: the "
                    f"write of layer {layer.layer_number} is deferred to no following read"
                )
            if reads:  # the stack's last write stays plain and closes the last interval
                last = reads[-1]
                write.register_forward_hook(
                    lambda module, args, output: (
                        output if last.interval is None else last.interval.replay_before(output)
                    )
                )
            continue
        successor, read = layers[index + 1], connections[index + 1][0]
        if not isinstance(write, DeferredWrite):
            raise _unfused_error(layer)
        if not isinstance(read, WriteReadNorm):
            raise _unfused_error(successor)
        write.link(read)
        object.__setattr__(read, "seam_layer", successor)
        object.__setattr__(read, "previous_read", reads[-1] if reads else None)
        if not reads and not read.starts_replay:
            raise ValueError("a fused stack's first seam must start a replay interval")
        reads.append(read)
        if not read.consumers:  # else a shortcut block's MoE read, linked by its block
            moe = isinstance(getattr(successor, "mlp", None), MoELayer)
            read.link_consumer(successor.mlp if moe else _input_projection(successor), successor)
        for fused in (read, getattr(read, "shortcut", None)):
            if fused is not None:
                fused.resolve_formats(partial(runs_mxfp8, stack_config))
    return tuple(reads)
