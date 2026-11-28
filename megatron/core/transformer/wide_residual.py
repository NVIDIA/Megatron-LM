# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
"""Wide-residual parameters, connection policy, and model-boundary helpers."""

from __future__ import annotations

import math
from contextlib import nullcontext
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
from megatron.core.fusions.wide_residual_shortcut.residual_replay import column_slot
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
from megatron.core.transformer.transformer_config import (
    TransformerConfig,
    _validate_shortcut_fusion_config,
)

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


class _FusedWideResidualConnection(StreamwiseSigmoidWideResidualConnection):
    def __init__(self, config, *args, **kwargs):
        if not config.moe_shortcut_wide_residual_norm_fusion:
            raise ValueError("a fused residual spec requires moe_shortcut_wide_residual_norm_fusion")
        super().__init__(config, *args, **kwargs)


class WriteReadNorm(_FusedWideResidualConnection):
    """Apply a deferred write, streamwise read and RMSNorm in one kernel.

    Each linked consumer receives its own BF16 or MXFP8 operand. A shortcut block
    can apply both reads early and hand the compute layer its completed read result.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        branch_name: str,
        operation: ResidualConnectionOperation = "read",
    ) -> None:
        if operation != "read":
            raise ValueError(f"WriteReadNorm is a read connection, got operation={operation!r}")
        super().__init__(config, layer_number, branch_name, operation=operation)
        device = "cpu" if config.use_cpu_initialization else torch.cuda.current_device()
        self.weight = nn.Parameter(
            torch.ones(config.hidden_size, dtype=config.params_dtype, device=device)
        )
        setattr(self.weight, "sequence_parallel", config.sequence_parallel)
        self.eps = config.layernorm_epsilon
        self.fuse_wgrad_accumulation = config.gradient_accumulation_fusion
        # Stack links must not register neighboring layers as checkpoint submodules.
        self.consumers, self.previous_read = (), None
        self.consumer_formats = ()
        self.interval = self.pending = None
        # The first seam applies layer one's write at the read of layer two.
        every = config.residual_stream_recompute_num_layers
        self.starts_replay = every is None or every <= 1 or (layer_number - 2) % every == 0

    def resolve_formats(self, gemm_format) -> None:
        """Choose formats in consumer order and hook Linears that need replayed columns."""
        if not 1 <= len(self.consumers) <= 2:
            raise RuntimeError("a fused read feeds one or two linked consumers")
        self.consumer_formats = tuple(gemm_format(*consumer) for consumer in self.consumers)
        self.compact_scales_for_tp = False
        for (module, _), fmt in zip(self.consumers, self.consumer_formats):
            if fmt != "mxfp8":
                continue
            if getattr(module, "_shortcut_replays_columns", False):
                raise RuntimeError("a Linear takes at most one fused read as MXFP8")
            # TE gathers the input (and its replayed columns for wgrad) along tokens.
            self.compact_scales_for_tp |= (
                module.parallel_mode == "column" and module.sequence_parallel
            )
            module._shortcut_replays_columns = True
            module.register_forward_hook(self._install_replayed_columns_for_wgrad, with_kwargs=True)

    @staticmethod
    def _install_replayed_columns_for_wgrad(module, args, kwargs, output):
        if not torch.is_grad_enabled():
            return
        operand = args[0] if args else next(iter(kwargs.values()))
        slot = column_slot(operand)
        if slot is None:
            raise RuntimeError("an MXFP8 consumer took a fused read emitted without gradients")
        install_columns(output, operand, slot)

    def hand(self, item) -> None:
        """Hand a deferred write or a completed (branch_input, state) pair to this read."""
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
        """Start a linked replay interval or continue the previous read's interval."""
        previous = self.previous_read.interval if self.previous_read is not None else None
        self.interval = ReplayInterval(previous) if self.starts_replay else previous
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
        """Apply the write and emit normalized operands for one or two reads."""
        from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
            is_batch_invariant_mode_enabled,
        )

        reads = (self,) if shortcut is None else (self, shortcut)
        # Inference builds no autograd graph (and uses the inference MXFP8 eligibility) even with
        # gradients enabled.
        with torch.no_grad() if InferenceMode.is_active() else nullcontext():
            outputs = tuple(read.consumer_formats for read in reads)
            if any("mxfp8" in formats for formats in outputs):
                check_mxfp8_call(pending.routed)  # never falls back to BF16
            interval, slots = None, (None,) * len(reads)
            if torch.is_grad_enabled():
                interval = self._open_interval()
                slots = tuple(ColumnSlot() if "mxfp8" in formats else None for formats in outputs)
            # A shortcut seam runs before this read's layer calls it: run its weight's hooks now.
            weight = self.weight if shortcut is None else self._hooked_weight()
            return fused_seam(
                pending,
                tuple(read.read_map(return_logits=True) for read in reads),
                (weight, *(read.weight for read in reads[1:])),
                outputs=outputs,
                compact_scales_for_tp=tuple(read.compact_scales_for_tp for read in reads),
                columns=slots,
                interval=interval,
                num_streams=self.num_streams,
                epsilon=self.eps,
                batch_invariant=is_batch_invariant_mode_enabled(),
                fuse_wgrad_accumulation=self.fuse_wgrad_accumulation,
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
        if isinstance(item, tuple):
            return item
        residual, branch_input = self.apply_write(item)
        return branch_input, (residual,)


class ShortcutAddWideResidualWriteReadNorm(WriteReadNorm):
    """Apply both shortcut-block reads and hand the compute read its normalized input."""

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
        compute_read.hand((branch_input, (residual,)))
        return residual, shortcut_input


class DeferredWrite(_FusedWideResidualConnection):
    """Hand branch updates to the next layer's read and return the unwritten residual."""

    target: nn.Module | None = None
    # Shortcut adds supply both producers in forward argument order (routed, shared).
    producers: tuple = ()
    update_formats: tuple | None = None

    def link(self, target, layer, gemm_format) -> None:
        """Bind the next read and resolve each update's gradient format."""
        object.__setattr__(self, "target", target)
        producers = self.producers or (layer.seam_output_projection(),)
        self.update_formats = tuple(gemm_format(p, layer, gradients=True) for p in producers)
        # Shared updates use one gradient. Compact scales also work for a local GEMM.
        self.compact_grad_scales_for_tp = any(
            fmt == "mxfp8" and p.parallel_mode == "row" and p.sequence_parallel
            for p, fmt in zip(producers, self.update_formats)
        )

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
        """Defer this write, including the shared expert for a shortcut add."""
        update, bias = branch_output
        if bias is not None:
            raise ValueError("fused residual seams require a bias-free branch projection")
        retain = None if self.retention is None else self.retention(return_logits=True)
        write_logits = self.write_map(return_logits=True)
        pending = PendingWideResidualWrites(
            state[0],
            update,
            shared,
            write_logits,
            retain,
            update_formats=self.update_formats,
            compact_grad_scales_for_tp=self.compact_grad_scales_for_tp,
        )
        self.target.hand(pending)
        return state[0]


def takes_mxfp8(
    linear, autocast_recipe, fp8_format="e4m3", *, gradients=False, recipe_attrs=None
) -> bool:
    """Resolve a TE Linear's MXFP8 handoff using its inherited recipe and module override.
    An MXFP8 gradient handoff also requires E4M3 grad_output."""

    from megatron.core.extensions.transformer_engine import TELinear

    if not isinstance(linear, TELinear):
        return False
    params = getattr(linear, "te_quant_params", None)
    if params is not None and params.training_recipe is not None:
        recipe = params.training_recipe
        if params.evaluation_recipe not in (None, recipe):
            return False
        applies = (
            recipe.override_quantized_autocast
            if autocast_recipe
            else recipe.override_nonquantized_autocast
        )
        if applies:
            if recipe.fp4_quantization_recipe is not None or recipe.recipe_attrs:
                return False
            autocast_recipe, fp8_format = recipe.fp8_quantization_recipe, recipe.fp8_format
            recipe_attrs = recipe.recipe_attrs
    if autocast_recipe != Fp8Recipe.mxfp8 or (gradients and fp8_format != "e4m3"):
        return False

    from transformer_engine.common.recipe import MXFP8BlockScaling

    # TE initializes this default from NVTE_BACKWARD_OVERRIDE at import. An override
    # can require the original BF16 input for wgrad and a BF16 output gradient for dgrad.
    default_backward = getattr(MXFP8BlockScaling, "backward_override", None)
    return (recipe_attrs or {}).get("backward_override", default_backward) is None


def _gemm_autocast_recipe(stack_config, layer):
    """The recipe a GEMM inherits from HybridStack.forward before its own TE override."""
    from megatron.core.fp8_utils import is_first_last_bf16_layer
    from megatron.core.ssm.kda_layer_config import KDALayerConfig

    config = layer.config
    if isinstance(config, KDALayerConfig) and config.kda_disable_fp8:
        return None
    if stack_config.fp8 and stack_config.fp8_recipe == Fp8Recipe.delayed:
        return Fp8Recipe.delayed
    if is_first_last_bf16_layer(config, layer.layer_number - 1):
        return None
    if stack_config.fp8 and config.fp8:
        return config.fp8_recipe
    if not stack_config.fp8 and stack_config.fp4 and config.fp4:
        return config.fp4_recipe
    return None


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


def link_residual_seams(stack, *, prefix: str) -> None:
    """Link adjacent writes and reads in physical layer order, then close the replay chain.

    Standalone MoE reads serve the whole MoE as one BF16 consumer. Shortcut blocks
    have already linked their individual consumers.
    """
    from megatron.core.quantization.utils import get_quant_config_or_none
    from megatron.core.transformer.moe.moe_layer import MoELayer

    # Grouping changes recipe paths. Apply the stack recipe to the final names before
    # binding formats, including clearing any override inherited from a layer config.
    recipe = stack.config.quant_recipe
    if recipe is not None or any(c.quant_recipe is not None for c in stack.layer_config_list):
        for path, module in stack.named_modules(prefix=prefix):
            if hasattr(module, "finish_init"):
                module.finish_init(get_quant_config_or_none(path, recipe))

    connections = []
    for layer in stack.physical_layers():
        if not hasattr(layer, "seam_connections"):
            raise _unfused_error(layer)
        try:
            connections.append((layer, *layer.seam_connections()))
        except ValueError:
            raise _unfused_error(layer) from None

    def gemm_format(linear, layer, *, gradients=False):
        if linear is not None and takes_mxfp8(
            linear,
            _gemm_autocast_recipe(stack.config, layer),
            layer.config.fp8,
            gradients=gradients,
            recipe_attrs=layer.config.fp8_recipe_attrs,
        ):
            if getattr(linear, "te_quant_params", None) is not None:
                # An override can opt in on a BF16 stack, after initial config validation.
                _validate_shortcut_fusion_config(layer.config, mxfp8_handoff=True)
            return "mxfp8"
        return "bf16"

    last_read = None
    for (layer, _, write), (successor, read, _) in zip(connections, connections[1:]):
        if not isinstance(write, DeferredWrite):
            raise _unfused_error(layer)
        if not isinstance(read, WriteReadNorm):
            raise _unfused_error(successor)
        write.link(read, layer, gemm_format)
        object.__setattr__(read, "previous_read", last_read)
        if last_read is None and not read.starts_replay:
            raise ValueError("a fused stack's first seam must start a replay interval")
        last_read = read
        if not read.consumers:
            if isinstance(getattr(successor, "mlp", None), MoELayer):
                consumer = successor.mlp
            else:
                location = successor.get_input_layernorm_module(successor.submodules_config)
                if len(location) != 2:
                    raise _unfused_error(successor)
                branch, projection = location
                consumer = getattr(getattr(successor, branch), projection)
            read.consumers = ((consumer, successor),)
        for fused in (read, getattr(read, "shortcut", None)):
            if fused is not None:
                fused.resolve_formats(gemm_format)
    if connections:
        layer, _, write = connections[-1]
        if isinstance(write, DeferredWrite):
            raise ValueError(
                "moe_shortcut_wide_residual_norm_fusion requires at least two layers: the "
                f"write of layer {layer.layer_number} is deferred to no following read"
            )
        if last_read is not None:
            write.register_forward_hook(
                lambda module, args, output: (
                    output if last_read.interval is None else last_read.interval.replay_before(output)
                )
            )
