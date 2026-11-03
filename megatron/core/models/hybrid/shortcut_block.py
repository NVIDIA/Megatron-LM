# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Implement shortcut-connected MoE blocks based on the ScMoE architecture introduced in
`Shortcut-connected Expert Parallelism for Accelerating Mixture of Experts
<https://arxiv.org/abs/2404.05019>`_.

ScMoE routes the preceding layer's representation through sparse experts while the paired layer
computes attention and the shared expert, breaking the usual sequential dependency so
expert-parallel All-to-All communication can overlap with computation.
Our variant adds two LayerNorms to the paper's design: `shortcut_pre_mlp_layernorm` normalizes
the shortcut representation before routing and expert dispatch, while `shortcut_post_norm`
normalizes the merged routed- and shared-expert result before the MoE residual/BDA step.
Also supports selective activation recomputation and activation offloading on
shortcut_pre_mlp_layernorm and shortcut_post_norm, respectively, to reduce the additional
device-memory pressure from enabling Shortcut-MoE.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Sequence

import torch

from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.fusions.wide_residual_shortcut.api import PendingShortcutAdd, validate_input
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols as LayerSymbols
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.tensor_parallel.random import CheckpointWithoutOutput
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule, TwoStageAttentionLayer
from megatron.core.transformer.moe.shared_experts import set_tensor_grad_fn_sequence_sr
from megatron.core.transformer.residual_recompute import ResidualStreamRecomputeContext
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.wide_residual import (
    StreamwiseSigmoidWideResidualRead,
    fused_norm_weight,
    is_fusable_rmsnorm,
    pending_write,
)
from megatron.core.typed_torch import apply_module

# Layer symbols that may precede a shortcut MoE
SUPPORTED_SHORTCUT_PREDECESSORS = frozenset(
    {LayerSymbols.MAMBA, LayerSymbols.GDN}
    | {
        symbol
        for symbol, layer_config in LayerSymbols.LAYER_CONFIG_MAP.items()
        if layer_config in LayerSymbols.ATTENTION_LAYER_CONFIGS
    }
)


def _get_offloading_interface():
    """Get the fine-grained activation offloading interface lazily."""
    from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
        FineGrainedActivationOffloadingInterface,
    )

    return FineGrainedActivationOffloadingInterface


def group_layers_into_shortcut_blocks(
    layers: torch.nn.ModuleList,
    layer_type_list: Sequence[str],
    config: TransformerConfig,
    pp_layer_offset: int = 0,
) -> torch.nn.ModuleList:
    """Group physical layers into their registered shortcut-block hierarchy.

    Grouping updates the layer names through the returned `ModuleList` hierarchy. Layers not
    followed by an MoE remain direct children of the returned `ModuleList`.

    Args:
        layers: Physical layers in execution order.
        layer_type_list: Physical layer symbols in execution order.
        config: Transformer configuration controlling shortcut scheduling.
        pp_layer_offset: Global offset of this pipeline stage's first physical layer.

    Returns:
        The registered logical layers.

    Raises:
        ValueError: If an MoE has an unsupported predecessor or its shortcut pair crosses a
            pipeline-stage boundary.
    """

    if pp_layer_offset > 0 and layer_type_list and layer_type_list[0] == LayerSymbols.MOE:
        raise ValueError(
            "Shortcut MoE pairs cannot cross pipeline-stage boundaries; this pipeline stage "
            f"begins with an MoE layer (global layer {pp_layer_offset + 1}). Move the stage "
            "boundary so each MoE layer stays on the same stage as the layer before it."
        )

    grouped_layers = torch.nn.ModuleList()
    physical_index = 0
    while physical_index < len(layers):
        next_is_moe = (
            physical_index + 1 < len(layers)
            and layer_type_list[physical_index + 1] == LayerSymbols.MOE
        )
        if not next_is_moe:
            grouped_layers.append(layers[physical_index])
            physical_index += 1
            continue

        compute_layer = layers[physical_index]
        paired_type = layer_type_list[physical_index]
        if paired_type not in SUPPORTED_SHORTCUT_PREDECESSORS:
            raise ValueError(
                "Shortcut MoE must be preceded by a Mamba, GDN, or supported attention layer"
            )

        supports_two_stage = (
            isinstance(compute_layer, TwoStageAttentionLayer)
            and compute_layer.supports_two_stage_attention()
        )
        if not supports_two_stage:
            raise ValueError(
                f"Shortcut attention layer {paired_type!r} does not support two-stage attention"
            )
        moe_layer = layers[physical_index + 1]
        grouped_layers.append(
            ShortcutMoEBlock(
                compute_layer,
                moe_layer,
                overlap_a2a=config.moe_shortcut_parallel,
                attn_local_idx=physical_index,
                moe_local_idx=physical_index + 1,
            )
        )
        physical_index += 2

    if config.moe_shortcut_wide_residual_norm_fusion and not any(
        getattr(layer, "using_shortcut_moe_fused_residual_kernel", False)
        for layer in grouped_layers
    ):
        if not all(getattr(layer, "is_mtp_layer", False) for layer in layers):
            raise ValueError(
                "shortcut kernel fusion requires an adjacent Mamba/GDN/attention and MoE pair"
            )
    if config.moe_shortcut_wide_residual_norm_fusion:
        _link_residual_seams(grouped_layers)
    return grouped_layers


def _link_residual_seams(grouped_layers) -> None:
    """Fuse each residual write into the seam kernel of the layer that reads it next.

    A layer defers its write (``defers_residual_write``) when its successor applies it: a fused
    ShortcutMoEBlock (two-reader kernel) or a lone layer with a ``pending_write_reader`` (one-reader
    kernel). A fused block defers its MoE write (routed + shared); a lone layer defers its
    ``pending_write_writer`` write. The deferred write travels to the successor as its
    ``hidden_states`` (a PendingShortcutAdd).
    """

    def fused_block(layer):
        return (
            isinstance(layer, ShortcutMoEBlock) and layer.using_shortcut_moe_fused_residual_kernel
        )

    def applies_pending_write(layer):
        return (
            fused_block(layer) or getattr(layer, "pending_write_reader", lambda: None)() is not None
        )

    def can_defer_write(layer):
        return (
            fused_block(layer) or getattr(layer, "pending_write_writer", lambda: None)() is not None
        )

    for layer, successor in zip(grouped_layers, grouped_layers[1:]):
        if applies_pending_write(successor) and can_defer_write(layer):
            layer.defers_residual_write = True


class ShortcutMoEBlock(MegatronModule):
    """Own and execute one attention-layer/shortcut-MoE pair."""

    _parallel_stream: torch.cuda.Stream | None = None
    # Set by group_layers_into_shortcut_blocks when the next layer applies this block's MoE
    # write (fused kernels only): forward returns a PendingShortcutAdd instead.
    defers_residual_write: bool = False

    @classmethod
    def _get_a2a_overlap_stream(cls) -> torch.cuda.Stream:
        """Return the process-wide high-priority shortcut stream."""
        if cls._parallel_stream is None:
            cls._parallel_stream = torch.cuda.Stream(priority=-1)
        return cls._parallel_stream

    def __init__(
        self,
        compute_layer,
        moe_layer,
        overlap_a2a: bool,
        attn_local_idx: int | None = None,
        moe_local_idx: int | None = None,
    ):
        super().__init__(compute_layer.config)

        self.overlap_mode = overlap_a2a
        self.layer_number = compute_layer.layer_number
        self.attn_layer_idx = compute_layer.layer_number - 1
        self.attn_local_idx = attn_local_idx
        self.moe_layer_idx = moe_layer.layer_number - 1
        self.moe_local_idx = moe_local_idx

        self.is_first_layer = getattr(compute_layer, "is_first_layer", False)
        self.is_last_layer = getattr(moe_layer, "is_last_layer", False)
        self.tp_group = moe_layer.mlp.tp_group
        self.compute_layer = compute_layer
        self.moe_layer = moe_layer
        self.recompute_shortcut_pre_mlp_layernorm = (
            self.config.recompute_granularity == "selective"
            and "shortcut_pre_mlp_layernorm" in (self.config.recompute_modules or [])
        )
        self.offload_shortcut_post_norm = (
            self.config.fine_grained_activation_offloading
            and "shortcut_post_norm" in (self.config.offload_modules or [])
        )
        self.off_interface = _get_offloading_interface()
        self.shortcut_pre_mlp_layernorm_checkpoint = None

        compute_is_mtp = bool(getattr(compute_layer, "is_mtp_layer", False))
        moe_is_mtp = bool(getattr(moe_layer, "is_mtp_layer", False))
        if compute_is_mtp != moe_is_mtp:
            raise ValueError("A ShortcutMoE pair must agree on whether it belongs to MTP.")
        self.is_mtp_layer = moe_is_mtp
        if not self.is_mtp_layer and (
            (compute_layer.config.wide_residual is None) != (moe_layer.config.wide_residual is None)
        ):
            raise ValueError(
                "A ShortcutMoE pair must agree on whether it carries a wide residual stream."
            )

        self.shortcut_residual_read = None
        if moe_layer.config.wide_residual is not None and not self.is_mtp_layer:
            self.shortcut_residual_read = StreamwiseSigmoidWideResidualRead(
                config=moe_layer.config,
                layer_number=moe_layer.layer_number,
                branch_name="shortcut_routed",
            )
            outer_read = moe_layer._get_mlp_residual_connection(operation="read")
            if outer_read is None:
                raise ValueError(
                    "A wide-residual shortcut read requires an outer MoE residual connection."
                )
            if (
                self.shortcut_residual_read.residual_stream_hidden_size
                != outer_read.residual_stream_hidden_size
            ):
                raise ValueError(
                    "The shortcut read and outer MoE residual connection must consume the "
                    "same residual-stream width."
                )
            predecessor_stream_width = getattr(compute_layer, "residual_stream_hidden_size", None)
            if predecessor_stream_width != self.shortcut_residual_read.residual_stream_hidden_size:
                raise ValueError(
                    "The shortcut predecessor and routed read must carry the same "
                    "residual-stream width."
                )

        shortcut_norm_spec = TENorm
        self.shortcut_pre_mlp_layernorm = build_module(
            shortcut_norm_spec,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
            has_residual=False,
        )
        self.shortcut_post_norm = (
            build_module(
                shortcut_norm_spec,
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
                has_residual=False,
            )
            if self.config.moe_shortcut_post_norm
            else IdentityOp()
        )
        self.route_ready_event = torch.cuda.Event() if self.overlap_mode else None

        # The fused kernels own the pair's residual seams: the two-reader kernel applies the
        # previous layer's write with this block's compute and shortcut norms, and
        # wide_residual_write_read_rmsnorm_quant writes the compute update with the MoE pre-MLP
        # norm.
        self.using_shortcut_moe_fused_residual_kernel = bool(
            self.config.moe_shortcut_wide_residual_norm_fusion and not self.is_mtp_layer
        )
        if self.using_shortcut_moe_fused_residual_kernel:
            self._validate_fused_residual_kernel()
            compute_layer.using_shortcut_moe_fused_residual_kernel = True

    def _validate_fused_residual_kernel(self):
        """Require the residual structure that the fused residual kernels implement."""
        from megatron.core.transformer.wide_residual import StreamwiseSigmoidWideResidualConnection

        parts_of = getattr(self.compute_layer, "shortcut_compute_parts", None)
        if parts_of is None or parts_of() is None:
            raise ValueError(
                "shortcut kernel fusion requires a wide-residual Mamba/GDN/attention compute-only "
                "predecessor"
            )
        parts = self.compute_layer.shortcut_compute_parts()
        compute_read, compute_write, compute_norm = parts.read, parts.write, parts.input_norm
        connections = (
            compute_read,
            compute_write,
            self.moe_layer._get_mlp_residual_connection("read"),
            self.moe_layer._get_mlp_residual_connection("write"),
        )
        if any(not isinstance(c, StreamwiseSigmoidWideResidualConnection) for c in connections):
            raise ValueError("shortcut kernel fusion requires streamwise sigmoid connections")
        if any(c.num_streams != self.config.wide_residual.num_streams for c in connections):
            raise ValueError("shortcut kernel fusion connections must use the same stream count")
        for name, norm in (
            ("the compute layer's input norm", compute_norm),
            ("pre_mlp_layernorm", self.moe_layer.pre_mlp_layernorm),
            ("shortcut_pre_mlp_layernorm", self.shortcut_pre_mlp_layernorm),
        ):
            if not is_fusable_rmsnorm(norm):
                raise ValueError(
                    f"shortcut kernel fusion requires {name} to be a standalone RMSNorm, got "
                    f"{type(norm).__name__}. Build the model with "
                    "hybrid_layer_specs.shortcut_fusion_stack_spec(<wide-residual stack spec>), "
                    "or split input norms from their projections in a custom spec."
                )
        if compute_norm.eps != self.shortcut_pre_mlp_layernorm.eps:
            # One two-reader kernel normalizes both inputs with a single epsilon.
            raise ValueError(
                "shortcut kernel fusion requires the compute layer's input norm and "
                "shortcut_pre_mlp_layernorm to share an epsilon"
            )
        if not self.moe_layer.mlp.use_shared_expert:
            raise ValueError("shortcut kernel fusion requires a shared expert")

    def _shortcut_add(self, pending, quant_context_factory):
        """Apply the pending write and norm this block's two inputs.

        The pending write is the previous fused block's MoE output (a shortcut add of routed and
        shared) or a preceding lone layer's single update.

        Returns ``(residual, compute_input, shortcut_input)`` from
        ``shortcut_add_wide_residual_write_two_reader_rmsnorm_quant``.
        """
        from megatron.core.fusions import wide_residual_shortcut as kernels
        from megatron.core.fusions.wide_residual_shortcut import mxfp8_handoff

        streams = self.config.wide_residual.num_streams
        parts = self.compute_layer.shortcut_compute_parts()
        compute_read, compute_norm = parts.read, parts.input_norm
        mixer, linear = parts.branch, parts.input_projection
        with quant_context_factory(self.compute_layer.config, self.attn_layer_idx):
            # A GDP input-projection checkpoint keeps its weight hooks in an inner autograd
            # graph that the boundary's weight edge cannot order; only GTP's explicit
            # finalization follows the deferred weight GEMM. With activation offload, its
            # weight-ready notifications are not preserved either, so that combination stays
            # on the qualified BF16 handoff.
            recompute = getattr(mixer, "recompute_in_proj", False)
            offload = recompute and getattr(
                mixer.config, "fine_grained_activation_offloading", False
            )
            direct = (
                linear
                if not offload and mxfp8_handoff.eligible(linear, pending.routed, streams)
                else None
            )
            if (
                torch.is_grad_enabled()
                and direct is not None
                and recompute
                and not getattr(direct.weight, "is_gtp_weight_remat", False)
            ):
                raise ValueError(
                    "direct shortcut MXFP8 requires deferred weight gradients; "
                    "GDP input-projection recompute requires GTP weight finalization. "
                    "Disable gdp_in_proj recompute or use a BF16 handoff."
                )
            return kernels.shortcut_add_wide_residual_write_two_reader_rmsnorm_quant(
                pending,
                (compute_read.read_logits(), self.shortcut_residual_read.read_logits()),
                (
                    fused_norm_weight(compute_norm),
                    fused_norm_weight(self.shortcut_pre_mlp_layernorm),
                ),
                mxfp8_handoff=direct is not None,
                mxfp8_consumer=direct,
                num_streams=streams,
                epsilon=compute_norm.eps,  # equal to the shortcut norm's (validated at init)
            )

    def _wide_residual_write_read(self, residual, update):
        """Write the compute update and emit the shared-expert input (one-reader kernel)."""
        from megatron.core.fusions import wide_residual_shortcut as kernels
        from megatron.core.fusions.wide_residual_shortcut import mxfp8_handoff

        streams = self.config.wide_residual.num_streams
        compute_write = self.compute_layer.shortcut_compute_parts().write
        shared_expert = getattr(self.moe_layer.mlp, "shared_experts", None)
        consumer = None if shared_expert is None else shared_expert.direct_input_consumer()
        return kernels.wide_residual_write_read_rmsnorm_quant(
            residual,
            update,
            *compute_write.write_logits(),
            self.moe_layer._get_mlp_residual_connection("read").read_logits(),
            fused_norm_weight(self.moe_layer.pre_mlp_layernorm),
            mxfp8_consumer=(
                consumer
                if consumer is not None and mxfp8_handoff.eligible(consumer, update, streams)
                else None
            ),
            num_streams=streams,
            epsilon=self.moe_layer.pre_mlp_layernorm.eps,
        )

    def _fused_compute_input(self, residual):
        """Read and norm the compute layer's input when no pending shortcut add precedes it."""
        parts = self.compute_layer.shortcut_compute_parts()
        compute_input, _ = apply_module(parts.read)(
            residual,
            fp32_residual_connection=self.config.fp32_residual_connection,
            branch_input_dtype=self.config.params_dtype,
        )
        return apply_module(parts.input_norm)(compute_input.to(dtype=self.config.params_dtype))

    def _read_shortcut_hidden(
        self,
        hidden_states: torch.Tensor,
        *,
        recompute_context: ResidualStreamRecomputeContext | None,
    ) -> torch.Tensor:
        """Read the pair input used only by shortcut routing."""

        if self.shortcut_residual_read is None:
            return hidden_states
        if recompute_context is not None:
            return recompute_context.checkpoint(
                apply_module(self.shortcut_residual_read), hidden_states
            )
        return apply_module(self.shortcut_residual_read)(hidden_states)

    def _moe_router_preprocess(
        self,
        shortcut_hidden,
        padding_mask=None,
        packed_seq_params=None,
        recompute_context: ResidualStreamRecomputeContext | None = None,
        normalized: bool = False,
    ):
        """Run shortcut normalization, routing, and dispatch preprocessing.

        ``normalized`` input already went through the shortcut norm (fused two-reader kernel).
        """
        if normalized:
            shortcut_input = shortcut_hidden
        elif recompute_context is not None:
            shortcut_input = recompute_context.checkpoint(
                apply_module(self.shortcut_pre_mlp_layernorm), shortcut_hidden
            )
        elif self.recompute_shortcut_pre_mlp_layernorm and not InferenceMode.is_active():
            self.shortcut_pre_mlp_layernorm_checkpoint = CheckpointWithoutOutput()
            shortcut_input = self.shortcut_pre_mlp_layernorm_checkpoint.checkpoint(
                apply_module(self.shortcut_pre_mlp_layernorm), shortcut_hidden
            )
        else:
            shortcut_input = apply_module(self.shortcut_pre_mlp_layernorm)(shortcut_hidden)
        if recompute_context is not None and self.config.fine_grained_activation_offloading:
            # Replay reconstructs this norm output; exclude it from saved-tensor offloading.
            self.off_interface.mark_not_offload(shortcut_input)
        shortcut_input, padding_mask, _, _ = self.moe_layer._maybe_unflatten_for_moe(
            shortcut_input, padding_mask, input_ids=None, packed_seq_params=packed_seq_params
        )
        probs, routing_map = self.moe_layer.mlp.route(
            shortcut_input, padding_mask, packed_seq_params=packed_seq_params
        )
        return self.moe_layer.mlp.preprocess(
            shortcut_input, probs, routing_map, padding_mask, packed_seq_params=packed_seq_params
        )

    def _moe_shared_experts(
        self,
        hidden_states,
        padding_mask=None,
        packed_seq_params=None,
        recompute_context: ResidualStreamRecomputeContext | None = None,
        fused_residual: torch.Tensor | None = None,
    ):
        """Run the paired MoE layer's pre-MLP norm and shared experts.

        With ``fused_residual``, ``hidden_states`` is the compute layer's raw update and
        ``wide_residual_write_read_rmsnorm_quant`` writes it and emits the shared-expert input.

        Returns:
            `(shared_expert_output, moe_unflatten_mbs, residual, mlp_state)`.
        """
        if fused_residual is not None:
            residual, pre_mlp_output = self._wide_residual_write_read(fused_residual, hidden_states)
            mlp_state = (residual,)
        elif recompute_context is None:
            pre_mlp_output, residual, mlp_state = self.moe_layer._pre_mlp_layernorm_and_residual(
                hidden_states
            )
        else:
            pre_mlp_output, residual, mlp_state = self.moe_layer._pre_mlp_layernorm_and_residual(
                hidden_states, residual_stream_recompute_context=recompute_context
            )
        pre_mlp_output, _, _, moe_unflatten_mbs = self.moe_layer._maybe_unflatten_for_moe(
            pre_mlp_output, padding_mask, input_ids=None, packed_seq_params=packed_seq_params
        )
        shared_expert_output = self.moe_layer.mlp.shared_experts_compute(pre_mlp_output)
        return shared_expert_output, moe_unflatten_mbs, residual, mlp_state

    def _postprocess(
        self,
        residual,
        combined_output,
        shared_expert_output,
        packed_seq_params=None,
        moe_unflatten_mbs=None,
        mlp_state=(),
        recompute_context: ResidualStreamRecomputeContext | None = None,
    ):
        """Join routed/shared output, apply shortcut post-norm, and finish residual/BDA.

        With ``defers_residual_write`` the MoE write (routed + shared) is returned as a
        PendingShortcutAdd for the next layer's fused seam kernel instead.
        """
        if self.defers_residual_write:
            return pending_write(
                residual,
                (self.moe_layer.mlp.postprocess(combined_output, None), None),
                self.moe_layer._get_mlp_residual_connection("write"),
                shared=shared_expert_output,
            )
        output = self.moe_layer.mlp.postprocess(combined_output, shared_expert_output)
        post_norm_input = output
        post_norm_manager = self.off_interface(
            self.offload_shortcut_post_norm
            and recompute_context is None
            and not InferenceMode.is_active(),
            post_norm_input,
            "shortcut_post_norm",
        )
        with post_norm_manager as post_norm_input:
            if recompute_context is not None and not isinstance(
                self.shortcut_post_norm, IdentityOp
            ):
                output = recompute_context.checkpoint(
                    apply_module(self.shortcut_post_norm), post_norm_input
                )
            else:
                output = apply_module(self.shortcut_post_norm)(post_norm_input)
        output = post_norm_manager.group_offload(output, forced_released_tensors=[post_norm_input])
        if recompute_context is not None and self.config.fine_grained_activation_offloading:
            # Residual replay manages this write input; exclude it from saved-tensor offloading.
            self.off_interface.mark_not_offload(output)
        output = self.moe_layer._maybe_reflatten_from_moe(
            output, packed_seq_params, moe_unflatten_mbs
        )
        if recompute_context is None:
            output = self.moe_layer._apply_mlp_bda_step((output, None), residual, mlp_state)
        else:
            output = self.moe_layer._apply_mlp_bda_step(
                (output, None),
                residual,
                mlp_state,
                residual_stream_recompute_context=recompute_context,
            )
        return output[0] if isinstance(output, tuple) else output

    def _launch_dispatch(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, async_op: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Launch dispatch on the current stream or the shortcut side stream."""
        if not async_op:
            return self.moe_layer.mlp.dispatch(hidden_states, probs)

        assert self.route_ready_event is not None
        dispatch_stream = self._get_a2a_overlap_stream()
        dispatch_stream.wait_event(self.route_ready_event)
        hidden_states.record_stream(dispatch_stream)
        probs.record_stream(dispatch_stream)

        with torch.cuda.stream(dispatch_stream):
            return self.moe_layer.mlp.dispatch(hidden_states, probs)

    def _wait_dispatch(
        self, dispatched_input: torch.Tensor, dispatched_probs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Wait for dispatch and return its outputs on the main stream."""
        assert self.overlap_mode

        torch.cuda.current_stream().wait_stream(self._get_a2a_overlap_stream())
        dispatched_input.record_stream(torch.cuda.current_stream())
        dispatched_probs.record_stream(torch.cuda.current_stream())
        return dispatched_input, dispatched_probs

    def _dispatch_and_compute_routed_experts(
        self, route_input: torch.Tensor, route_probs: torch.Tensor, overlap: bool
    ) -> torch.Tensor:
        """Dispatch tokens and compute local experts within one ownership scope."""
        dispatched_input, dispatched_probs = self._launch_dispatch(
            route_input, route_probs, async_op=overlap
        )
        if overlap:
            dispatched_input, dispatched_probs = self._wait_dispatch(
                dispatched_input, dispatched_probs
            )
        output, _ = self.moe_layer.mlp.routed_experts_compute(dispatched_input, dispatched_probs)
        return output

    def _launch_combine(self, output: torch.Tensor, async_op: bool = False) -> torch.Tensor:
        """Launch combine on the current stream or the shortcut side stream."""
        if not async_op:
            return self.moe_layer.mlp.combine(output)

        combine_stream = self._get_a2a_overlap_stream()
        combine_stream.wait_stream(torch.cuda.current_stream())
        output.record_stream(combine_stream)
        with torch.cuda.stream(combine_stream):
            return self.moe_layer.mlp.combine(output)

    def _wait_combine(self, combined_output: torch.Tensor) -> torch.Tensor:
        """Wait for the asynchronous combine and return its output on the main stream."""
        torch.cuda.current_stream().wait_stream(self._get_a2a_overlap_stream())
        combined_output.record_stream(torch.cuda.current_stream())
        return combined_output

    def _forward_compute_atomic(
        self,
        hidden_states,
        *,
        attention_mask,
        inference_context,
        rotary_pos_emb,
        sequence_len_offset,
        packed_seq_params,
        padding_mask,
    ):
        """Run the paired layer's ordinary forward, which updates its KV cache or recurrent state.

        Uses the same per-layer arguments as HybridStack.forward passes to unpaired layers.
        """
        if isinstance(self.compute_layer, MambaLayer):
            return self.compute_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
            )
        if self.using_shortcut_moe_fused_residual_kernel:
            # A compute-only attention layer ends after its attention branch; the flagged
            # layer returns the raw (output, bias) there. This bypasses the layer's __call__, so
            # its own forward hooks and per-layer CUDA-graph manager do not run; the block (or the
            # stack's graph) is the captured unit.
            hidden_states, _ = self.compute_layer._forward_attention(
                hidden_states,
                attention_mask=attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                sequence_len_offset=sequence_len_offset,
                packed_seq_params=packed_seq_params,
                padding_mask=padding_mask,
            )
            return hidden_states
        hidden_states, _ = self.compute_layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            sequence_len_offset=sequence_len_offset,
            packed_seq_params=packed_seq_params,
            padding_mask=padding_mask,
        )
        return hidden_states

    def forward(
        self,
        hidden_states,
        attention_mask,
        inference_context,
        rotary_pos_emb,
        sequence_len_offset,
        packed_seq_params,
        padding_mask,
        quant_context_factory,
        cp_layout_state=None,
        packed_sequence_cp_metadata=None,
        attn_recompute_context: ResidualStreamRecomputeContext | None = None,
        moe_recompute_context: ResidualStreamRecomputeContext | None = None,
        padding_mask_by_layout=None,
    ):
        """Run the eager schedule with each physical layer's quantization context.

        In inference the paired layer runs its ordinary forward instead of the two-stage split,
        which never updates KV-cache or recurrent state, and communication stays on the current
        stream.

        With the fused residual kernels ``hidden_states`` may be a PendingShortcutAdd: the
        previous layer's unwritten write, applied here with this block's two input norms. With
        ``defers_residual_write`` the block returns its own MoE write the same way.
        """
        fused = self.using_shortcut_moe_fused_residual_kernel
        if fused and inference_context is not None and not InferenceMode.is_active():
            raise ValueError("shortcut inference_context requires InferenceMode.active()")
        if not fused and (
            isinstance(hidden_states, PendingShortcutAdd) or self.defers_residual_write
        ):
            raise ValueError("deferred residual writes require the fused residual kernels")
        if fused:
            unsupported = [
                name
                for name, value in (
                    ("packed_seq_params", packed_seq_params),
                    ("padding_mask", padding_mask),
                    ("cp_layout_state", cp_layout_state),
                    ("packed_sequence_cp_metadata", packed_sequence_cp_metadata),
                    ("attn_recompute_context", attn_recompute_context),
                    ("moe_recompute_context", moe_recompute_context),
                    # HybridModel passes per-layout masks even without CP; only real masks count.
                    (
                        "padding_mask_by_layout",
                        next(
                            (m for m in (padding_mask_by_layout or {}).values() if m is not None),
                            None,
                        ),
                    ),
                )
                if value is not None
            ]
            if unsupported:
                # Fused residuals use checkpoint replay, not residual-stream recompute.
                raise ValueError(
                    "shortcut kernel fusion does not support packed sequences, padding masks, "
                    "context parallelism or residual-stream recompute contexts; got "
                    + ", ".join(unsupported)
                )
        # Fused inference never builds an autograd graph.
        grad_context = (
            torch.set_grad_enabled(torch.is_grad_enabled() and not InferenceMode.is_active())
            if fused
            else nullcontext()
        )
        with grad_context:
            return self._forward(
                hidden_states,
                attention_mask,
                inference_context,
                rotary_pos_emb,
                sequence_len_offset,
                packed_seq_params,
                padding_mask,
                quant_context_factory,
                cp_layout_state,
                packed_sequence_cp_metadata,
                attn_recompute_context,
                moe_recompute_context,
                padding_mask_by_layout,
            )

    def _forward(
        self,
        hidden_states,
        attention_mask,
        inference_context,
        rotary_pos_emb,
        sequence_len_offset,
        packed_seq_params,
        padding_mask,
        quant_context_factory,
        cp_layout_state,
        packed_sequence_cp_metadata,
        attn_recompute_context,
        moe_recompute_context,
        padding_mask_by_layout,
    ):
        fused = self.using_shortcut_moe_fused_residual_kernel
        select_token_dispatcher = getattr(self.moe_layer.mlp, "select_token_dispatcher", None)
        if select_token_dispatcher is not None:
            select_token_dispatcher()
        inference = InferenceMode.is_active()
        if inference:
            if cp_layout_state is not None:
                raise RuntimeError("Shortcut-MoE inference does not support context parallelism.")
        else:
            assert inference_context is None, (
                "Shortcut-MoE received an inference context outside inference mode; the two-stage "
                "schedule does not update KV-cache or recurrent inference state."
            )
        overlap = self.overlap_mode and not inference

        if (attn_recompute_context is None) != (moe_recompute_context is None):
            raise ValueError("Shortcut replay requires contexts for both layers in the pair.")
        if attn_recompute_context is not None:
            assert moe_recompute_context is not None
            if attn_recompute_context.manager is not moe_recompute_context.manager:
                raise ValueError("A shortcut pair must share one residual replay manager.")
            if attn_recompute_context.is_block_end:
                raise ValueError("A residual replay block cannot end inside a ShortcutMoE pair.")

        attn_config = self.compute_layer.config
        moe_config = self.moe_layer.config
        if cp_layout_state is not None:
            assert self.attn_local_idx is not None and self.moe_local_idx is not None
            hidden_states, packed_seq_params = cp_layout_state.prepare_layer(
                self.attn_local_idx, hidden_states
            )
        moe_hidden_states = hidden_states
        moe_packed_seq_params = packed_seq_params
        moe_padding_mask = padding_mask
        if cp_layout_state is not None:
            moe_hidden_states, moe_packed_seq_params = cp_layout_state.prepare_layer(
                self.moe_local_idx, moe_hidden_states
            )
            moe_padding_mask = cp_layout_state.get_layer_padding_mask(
                self.moe_local_idx, padding_mask, padding_mask_by_layout
            )

        # The fused kernels own the residual seams; the flagged compute layer skips its own read,
        # norm and write. A pending shortcut add already produced both normalized inputs.
        fused_residual = None
        shortcut_add = isinstance(hidden_states, PendingShortcutAdd)
        if shortcut_add:
            fused_residual, hidden_states, moe_hidden_states = self._shortcut_add(
                hidden_states, quant_context_factory
            )
        elif fused:
            if self.config.fp32_residual_connection:
                hidden_states = hidden_states.float()
            # The kernels carry an FP32 or a BF16 residual stream in its own dtype.
            fused_residual = moe_hidden_states = hidden_states
            validate_input(
                fused_residual,
                self.config.hidden_size,
                self.config.wide_residual.num_streams,
                inference=not torch.is_grad_enabled(),
            )

        # Launch the moe_router
        with quant_context_factory(moe_config, self.moe_layer_idx):
            if not shortcut_add:
                moe_hidden_states = self._read_shortcut_hidden(
                    moe_hidden_states, recompute_context=attn_recompute_context
                )
            route_input, route_probs = self._moe_router_preprocess(
                shortcut_hidden=moe_hidden_states,
                padding_mask=moe_padding_mask,
                packed_seq_params=moe_packed_seq_params,
                recompute_context=attn_recompute_context,
                normalized=shortcut_add,
            )
            if overlap:
                self.route_ready_event.record(torch.cuda.current_stream())

        # Launch the input and attn of the attention layer
        paired_state = None  # initialize variable early to make pylint happy
        with quant_context_factory(attn_config, self.attn_layer_idx):
            if fused and not shortcut_add:
                hidden_states = self._fused_compute_input(fused_residual)
            if inference:
                attn_layer_output = self._forward_compute_atomic(
                    hidden_states,
                    attention_mask=attention_mask,
                    inference_context=inference_context,
                    rotary_pos_emb=rotary_pos_emb,
                    sequence_len_offset=sequence_len_offset,
                    packed_seq_params=packed_seq_params,
                    padding_mask=padding_mask,
                )
            elif attn_recompute_context is None:
                paired_state = self.compute_layer.forward_pre_attn_and_core_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    packed_seq_params=packed_seq_params,
                    packed_sequence_cp_metadata=packed_sequence_cp_metadata,
                )
            else:
                paired_state = self.compute_layer.forward_pre_attn_and_core_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    packed_seq_params=packed_seq_params,
                    packed_sequence_cp_metadata=packed_sequence_cp_metadata,
                    residual_stream_recompute_context=attn_recompute_context,
                )

        # Launch the dispatch, experts, and combine
        with quant_context_factory(moe_config, self.moe_layer_idx):
            output = self._dispatch_and_compute_routed_experts(route_input, route_probs, overlap)
            combined_output = self._launch_combine(output, async_op=overlap)
            if self.shortcut_pre_mlp_layernorm_checkpoint is not None:
                self.shortcut_pre_mlp_layernorm_checkpoint.discard_output_and_register_recompute(
                    combined_output
                )
                self.shortcut_pre_mlp_layernorm_checkpoint = None

        # launch the output layer of the attention layer
        if not inference:
            assert paired_state is not None
            with quant_context_factory(attn_config, self.attn_layer_idx):
                if attn_recompute_context is None:
                    attn_layer_output = self.compute_layer.forward_post_core_attn(*paired_state)
                else:
                    attn_layer_output = self.compute_layer.forward_post_core_attn(
                        *paired_state, residual_stream_recompute_context=attn_recompute_context
                    )
                if isinstance(attn_layer_output, tuple) and not fused:
                    attn_layer_output = attn_layer_output[0]

        if fused:
            # The flagged compute layer returns its raw output: (update, bias) for Mamba/GDN,
            # ((update, bias), context) for attention.
            if isinstance(attn_layer_output[0], tuple):
                attn_layer_output = attn_layer_output[0]
            attn_layer_output, bias = attn_layer_output
            if bias is not None:
                raise ValueError("shortcut kernel fusion requires a bias-free mixer projection")

        if cp_layout_state is not None:
            attn_layer_output, moe_packed_seq_params = cp_layout_state.prepare_layer(
                self.moe_local_idx, attn_layer_output
            )

        # launch the moe shared experts and combine attn and moe layer outputs
        with quant_context_factory(moe_config, self.moe_layer_idx):
            shared_expert_output, moe_unflatten_mbs, mlp_residual, mlp_state = (
                self._moe_shared_experts(
                    hidden_states=attn_layer_output,
                    padding_mask=moe_padding_mask,
                    packed_seq_params=moe_packed_seq_params,
                    recompute_context=moe_recompute_context,
                    fused_residual=fused_residual,
                )
            )
            if overlap:
                combined_output = self._wait_combine(combined_output)

            # Ensure the combine autograd node is scheduled first before shared_experts
            set_tensor_grad_fn_sequence_sr(combined_output, torch.iinfo(torch.int).max)
            output = self._postprocess(
                residual=mlp_residual,
                combined_output=combined_output,
                shared_expert_output=shared_expert_output,
                packed_seq_params=moe_packed_seq_params,
                moe_unflatten_mbs=moe_unflatten_mbs,
                mlp_state=mlp_state,
                recompute_context=moe_recompute_context,
            )
        if cp_layout_state is not None:
            output = cp_layout_state.finalize_layer(self.moe_local_idx, output)
        return output
