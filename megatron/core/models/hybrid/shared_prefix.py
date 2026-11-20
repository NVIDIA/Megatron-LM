# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Explicit shared-prefix adapter for a HybridStack and the HybridModel that owns it.

This module deliberately does not alter :class:`HybridStack`'s normal ``forward`` path. Call
``forward_hybrid_stack_shared_prefix`` with a shared-prefix layout (stars ``[P, C_1, ..., C_G]``)
to opt in. This module validates the stack and runs its layers in order; the layer-specific
execution lives with the layers. Mamba layers run
:func:`megatron.core.ssm.mamba_shared_prefix.forward_mamba_layer_shared_prefix`, which calls the
layer and scans each prefix once, forking its state into the completion branches. Attention
layers run the exact-backward forest attention in :mod:`megatron.core.ops.attention.shared_prefix`
after :mod:`megatron.core.transformer.shared_prefix_attention` validates them.
Unsupported topology or model features fail before executing a partial forward.

The ``HybridModel``-level pieces live here too, so ``HybridModel.forward`` only dispatches to
them: ``validate_hybrid_model_shared_prefix`` checks the arguments of a forward call,
``hybrid_shared_prefix_rotary_pos_emb`` builds the prefix-continued RoPE rows, and
``forward_hybrid_mtp_shared_prefix`` runs the MTP heads once over the dense branches.

CP>1 uses the model's standard zigzag sequence shards and sequence-to-head all-to-alls. The
implemented surface covers TP1 and TP>1 with sequence parallelism, CP1 and CP>1, explicit physical
padding, MoE expert-bias accounting, and full uniform activation recomputation. The capability
tokens at the end of this module name these code paths so integrations can require the exact
conjunction they use; they are contract identifiers, not evidence that a topology has been
qualified at full-model scale.
"""

from typing import TYPE_CHECKING, Optional

import torch
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.inference.utils import InferenceMode
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.shared_prefix_layout import SharedPrefixForestLayout, SharedPrefixLayout
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.ssm.mamba_shared_prefix import (
    forward_mamba_layer_shared_prefix,
    validate_mamba_layer_shared_prefix,
)
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.moe.moe_utils import router_gating_token_blocks
from megatron.core.transformer.multi_token_prediction import process_mtp_loss
from megatron.core.transformer.shared_prefix_attention import validate_shared_prefix_attention
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import make_viewless_tensor

if TYPE_CHECKING:
    from megatron.core.models.hybrid.hybrid_model import HybridModel

# MoE router load-balancing types that leave top-k routing unchanged; any loss they add is
# rejected separately through its coefficient.
_AUX_LOSS_LOAD_BALANCING_TYPES = frozenset({"none", "aux_loss", "seq_aux_loss", "global_aux_loss"})


def _has_nonzero_config_value(value) -> bool:
    """Return whether a scalar or per-layer configuration contains a nonzero value."""
    if value is None:
        return False
    if isinstance(value, (list, tuple)):
        return any(float(item) != 0.0 for item in value)
    return float(value) != 0.0


def _validate_shared_prefix_physical_length(
    layout: SharedPrefixLayout | SharedPrefixForestLayout,
    physical_len: int,
    *,
    tp_size: int,
    cp_size: int,
    sequence_parallel: bool,
) -> None:
    """Validate the global star length against its negotiated topology/padding contract."""
    physical_len = int(physical_len)
    if tp_size > 1:
        topology_multiple = 2 * tp_size * cp_size
    elif cp_size > 1:
        topology_multiple = 2 * cp_size
    else:
        topology_multiple = 1
    if isinstance(layout, SharedPrefixForestLayout):
        multiple = layout.padding_multiple or topology_multiple
        if physical_len % multiple or not 0 <= physical_len - layout.total_len < multiple:
            raise ValueError("shared-prefix forest must use minimal global topology padding")
        for root in layout.roots:
            _validate_shared_prefix_physical_length(
                root,
                ((root.total_len + multiple - 1) // multiple) * multiple,
                tp_size=tp_size,
                cp_size=cp_size,
                sequence_parallel=sequence_parallel,
            )
        return
    padding = physical_len - layout.total_len

    if layout.padding_multiple is not None:
        padding_multiple = layout.padding_multiple
        if padding_multiple % topology_multiple:
            raise ValueError(
                "shared-prefix padding_multiple must be divisible by the topology quantum: "
                f"M={padding_multiple}, Q={topology_multiple}, TP={tp_size}, CP={cp_size}"
            )
        for branch, (physical_completion, logical_completion) in enumerate(
            zip(layout.completion_lens, layout.logical_completion_lens, strict=True)
        ):
            if (layout.prefix_len + physical_completion) % padding_multiple or not (
                0 <= physical_completion - logical_completion < padding_multiple
            ):
                raise ValueError(
                    "shared-prefix physical completion span must use the minimal per-branch "
                    "padding to padding_multiple: "
                    f"branch={branch}, prefix={layout.prefix_len}, "
                    f"logical={logical_completion}, physical={physical_completion}, "
                    f"M={padding_multiple}"
                )
        if physical_len % padding_multiple:
            raise ValueError(
                "shared-prefix physical length must be divisible by padding_multiple: "
                f"physical={physical_len}, M={padding_multiple}"
            )
        if not 0 <= padding < padding_multiple:
            raise ValueError(
                "shared-prefix input must use the minimal trailing pad to padding_multiple: "
                f"physical={physical_len}, layout={layout.total_len}, M={padding_multiple}"
            )
        return

    if tp_size > 1:
        if physical_len % topology_multiple:
            raise ValueError(
                "shared-prefix TP/SP physical length must be divisible by 2 * tensor parallel "
                "size * context parallel size"
            )
        if not 0 <= padding < topology_multiple:
            raise ValueError(
                "shared-prefix TP/SP input must use the minimal trailing pad to a 2*TP*CP "
                f"multiple: physical={physical_len}, layout={layout.total_len}, "
                f"TP={tp_size}, CP={cp_size}"
            )
    elif cp_size == 1:
        if physical_len != layout.total_len:
            raise ValueError(
                f"packed sequence length {physical_len} does not match layout {layout.total_len}"
            )
    else:
        if physical_len % topology_multiple:
            raise ValueError(
                "shared-prefix CP physical length must be divisible by 2 * context parallel size"
            )
        if not 0 <= padding < topology_multiple:
            raise ValueError(
                "shared-prefix CP input must use the minimal trailing pad to a 2*CP multiple: "
                f"physical={physical_len}, layout={layout.total_len}, CP={cp_size}"
            )


def resolve_exclude_sequence_padding_from_expert_bias(config, requested: bool | None) -> bool:
    """Return whether per-branch padding rows are excluded from MoE expert-bias counts.

    ``requested`` is the caller's dense convention. ``None`` excludes padding with the flex
    dispatcher's HybridEP backend, which masks per-sequence padding in a dense packed forward,
    and counts it otherwise. The backbone routers and the MTP routers both follow this rule.
    """
    if requested is not None:
        return requested
    return (
        getattr(config, "moe_token_dispatcher_type", None) == "flex"
        and getattr(config, "moe_flex_dispatcher_backend", None) == "hybridep"
    )


def _validate_hybrid_stack(
    stack, layout: SharedPrefixLayout | SharedPrefixForestLayout, *, physical_len: int
) -> None:
    """Reject unsupported shared-prefix stack features before any layer executes.

    Only the configuration, process groups, layer types and the global physical star length are
    inspected, so a caller can run this before the embedding or any other partial forward.
    """
    # New upstream features must not silently bypass their ordinary execution.
    # These combinations need dedicated shared-layout integrations and GPU tests.
    if getattr(stack.config, "moe_num_hash_layers", 0):
        raise NotImplementedError("shared-prefix Hybrid does not yet support hash MoE routing")
    if getattr(stack.config, "quant_recipe", None) is not None:
        raise NotImplementedError("shared-prefix Hybrid does not yet support quantization recipes")
    if getattr(stack.config, "wide_residual", None) is not None:
        raise NotImplementedError("shared-prefix Hybrid does not yet support wide residual streams")
    if getattr(stack.config, "enable_mhc_connections", False):
        raise NotImplementedError("shared-prefix Hybrid does not yet support mHC connections")
    if getattr(stack.config, "attn_logit_softcapping", None) is not None:
        raise NotImplementedError("shared-prefix attention does not yet support logit softcapping")
    # The shared forward runs on the build-time CP group and takes no packed_seq_params, so it
    # cannot follow the group a dynamic-CP scheduler picks for a microbatch.
    if getattr(stack.config, "dynamic_context_parallel", False):
        raise NotImplementedError(
            "shared-prefix Hybrid does not yet support dynamic context parallelism"
        )
    # A custom layer is called through its own adapter with inputs the shared forward lacks.
    if getattr(stack, "_has_custom_layers", False):
        raise NotImplementedError("shared-prefix Hybrid does not yet support custom hybrid layers")
    # Batch-invariant mode promises bitwise training/inference agreement. Shared-prefix attention
    # runs FlashAttention 2 passes although this mode pins version 3 or 4, and shared-prefix Mamba
    # applies its gate to the rounded scan output instead of inside the scan.
    if getattr(stack.config, "batch_invariant_mode", False):
        raise NotImplementedError(
            "shared-prefix Hybrid does not yet support batch-invariant mode: its attention runs "
            "FlashAttention 2 and its Mamba layers gate after the scan output is rounded"
        )
    tp_size = stack.tp_group.size()
    sequence_parallel = bool(stack.config.sequence_parallel)
    if tp_size > 1 and not sequence_parallel:
        raise NotImplementedError("shared-prefix Hybrid TP>1 requires sequence parallelism")
    if tp_size == 1 and sequence_parallel:
        raise NotImplementedError("shared-prefix Hybrid sequence parallelism requires TP>1")
    if stack.config.tensor_model_parallel_size != tp_size:
        raise RuntimeError(
            "shared-prefix Hybrid tensor-parallel config does not match its process group"
        )
    if stack.pg_collection.tp.size() != tp_size:
        raise RuntimeError("shared-prefix Hybrid tensor-parallel groups disagree on their size")
    if stack.pp_group.size() != 1:
        raise NotImplementedError("shared-prefix Hybrid adapter currently supports PP1 only")
    cp_group = stack.pg_collection.cp
    cp_size = cp_group.size()
    if stack.config.context_parallel_size != cp_size:
        raise RuntimeError(
            "shared-prefix Hybrid context-parallel config does not match its process group"
        )
    if stack.config.recompute_granularity == "full":
        if stack.config.recompute_method != "uniform":
            raise NotImplementedError(
                'shared-prefix Hybrid full recomputation currently supports only the uniform '
                'method'
            )
        if not isinstance(stack.config.recompute_num_layers, int) or (
            stack.config.recompute_num_layers < 1
        ):
            raise ValueError(
                "shared-prefix Hybrid uniform recomputation requires recompute_num_layers >= 1"
            )
    if stack.config.fine_grained_activation_offloading:
        raise NotImplementedError(
            "shared-prefix Hybrid adapter does not support fine-grained activation offloading"
        )
    if stack.config.cuda_graph_impl != "none":
        raise NotImplementedError("shared-prefix Hybrid adapter does not support CUDA graphs")
    if stack.config.fp8 or stack.config.fp4:
        raise NotImplementedError("shared-prefix Hybrid adapter currently supports fp16/bf16 only")
    _validate_shared_prefix_physical_length(
        layout, physical_len, tp_size=tp_size, cp_size=cp_size, sequence_parallel=sequence_parallel
    )
    if stack.config.attention_dropout != 0.0 or stack.config.hidden_dropout != 0.0:
        raise NotImplementedError("shared-prefix Hybrid adapter currently requires zero dropout")
    window_size = stack.config.window_size
    if window_size is not None and tuple(window_size) != (-1, -1):
        raise NotImplementedError(
            "shared-prefix Hybrid adapter does not support sliding-window attention"
        )
    if stack.config.softmax_type != "vanilla":
        raise NotImplementedError("shared-prefix forest attention supports only vanilla softmax")

    num_moe_experts = stack.config.num_moe_experts
    expert_bias_enabled = bool(getattr(stack.config, "moe_router_enable_expert_bias", False))
    if num_moe_experts is not None and num_moe_experts > 0:
        if stack.config.moe_router_force_load_balancing:
            raise NotImplementedError(
                "shared-prefix Hybrid adapter does not support randomized forced MoE routing"
            )
        if stack.config.moe_router_force_biased is not None:
            raise NotImplementedError(
                "shared-prefix Hybrid adapter does not support randomized forced MoE router bias"
            )
        load_balancing = stack.config.moe_router_load_balancing_type
        load_balancing_types = (
            [load_balancing] if isinstance(load_balancing, str) else load_balancing
        )
        # Sinkhorn and quantile balancing replace top-k routing itself. The auxiliary-loss
        # types only add a loss term, which the coefficient check below rejects when active.
        if any(item not in _AUX_LOSS_LOAD_BALANCING_TYPES for item in load_balancing_types):
            raise NotImplementedError(
                "shared-prefix Hybrid adapter does not support MoE router load balancing type "
                f"{load_balancing!r}"
            )
        aux_loss_coeffs = stack.config.moe_aux_loss_coeff
        if isinstance(load_balancing, str):
            aux_loss_coeffs = [aux_loss_coeffs]
        # As in TopKRouter.get_aux_loss_coeff, a coefficient paired with "none" adds no loss.
        if any(
            load_balancing_type != "none" and _has_nonzero_config_value(coeff)
            for load_balancing_type, coeff in zip(load_balancing_types, aux_loss_coeffs)
        ):
            raise NotImplementedError(
                "shared-prefix Hybrid adapter does not support MoE auxiliary router loss"
            )
        if _has_nonzero_config_value(stack.config.moe_z_loss_coeff):
            raise NotImplementedError(
                "shared-prefix Hybrid adapter does not support MoE router z-loss"
            )
        if _has_nonzero_config_value(stack.config.moe_input_jitter_eps):
            raise NotImplementedError(
                "shared-prefix Hybrid adapter does not support MoE input jitter"
            )
        if expert_bias_enabled and any(
            root.logical_completion_lens is None for _, root in layout.iter_roots()
        ):
            raise NotImplementedError(
                "shared-prefix MoE expert-bias accounting requires explicit physical "
                "branch padding and logical completion lengths"
            )
        if stack.config.moe_expert_capacity_factor is not None:
            raise NotImplementedError(
                'shared-prefix Hybrid adapter does not support MoE expert capacity or token '
                'dropping'
            )
        # Both flex backends can drop tokens over the per-rank budget. A dropped prompt row
        # would remove its expert contribution from every completion that shares it.
        if stack.config.moe_expert_rank_capacity_factor is not None:
            raise NotImplementedError(
                "shared-prefix Hybrid adapter does not support MoE expert rank capacity or "
                "token dropping: the per-rank budget counts physical star rows, and a dropped "
                "prompt row is lost for every completion that shares it"
            )
        # Without THD packing, NCCL EP sizes its buffers once, from the first microbatch, and
        # needs the same size on every rank. Shared-prefix forwards are unpacked, and their
        # length changes with every microbatch and differs across ranks.
        if (
            stack.config.moe_token_dispatcher_type == "flex"
            and stack.config.moe_flex_dispatcher_backend == "ncclep"
        ):
            raise NotImplementedError(
                "shared-prefix Hybrid adapter does not support the flex dispatcher's 'ncclep' "
                "backend: without THD packing it sizes its buffers from the first microbatch, "
                "while shared-prefix microbatches change length"
            )
        # Replay targets are recorded per dense row; shared-prefix routes physical star rows.
        if stack.config.moe_enable_routing_replay:
            raise NotImplementedError(
                "shared-prefix Hybrid adapter does not support MoE routing replay: replay "
                "targets are indexed by dense rows, while a shared-prefix forward routes "
                "physical star rows that store each prompt once"
            )
        if getattr(stack.config, "mlp_chunks_for_training", 1) != 1:
            raise NotImplementedError(
                "shared-prefix Hybrid MoE adapter does not support training MLP chunking"
            )

    for layer in stack.layers:
        if isinstance(layer, MambaLayer):
            validate_mamba_layer_shared_prefix(layer, tp_size=tp_size, cp_size=cp_size)
        elif isinstance(layer, TransformerLayer):
            if not isinstance(layer.self_attention, (IdentityOp, SelfAttention)):
                raise NotImplementedError(
                    'shared-prefix Hybrid adapter supports standard self-attention and MLP/MoE '
                    'layers'
                )
            if isinstance(layer.self_attention, SelfAttention):
                # Also binds flash-attn, so a missing or old install fails before any layer runs.
                validate_shared_prefix_attention(
                    layer.self_attention, tp_size=tp_size, cp_size=cp_size
                )
            if expert_bias_enabled and getattr(layer, "is_moe_layer", False):
                from megatron.core.transformer.moe.moe_layer import MoELayer
                from megatron.core.transformer.moe.router import TopKRouter

                if not isinstance(layer.mlp, MoELayer) or not isinstance(
                    layer.mlp.router, TopKRouter
                ):
                    raise NotImplementedError(
                        "shared-prefix expert-bias accounting requires an MCore "
                        "MoELayer with TopKRouter"
                    )
        else:
            raise NotImplementedError(
                f"shared-prefix state forking is not implemented for {type(layer).__name__}"
            )


def forward_hybrid_stack_shared_prefix(
    stack,
    hidden_states: Tensor,
    layout: SharedPrefixLayout | SharedPrefixForestLayout,
    *,
    rotary_pos_emb: Tensor | tuple[Tensor, Tensor] | None = None,
    position_embedding_type: str = "rope",
    exclude_sequence_padding_from_expert_bias: bool | None = None,
    validated_physical_len: int | None = None,
) -> Tensor:
    """Explicit exact-prompt star forward for a supported ``HybridStack`` topology.

    Normal ``HybridStack.forward`` behavior is unchanged. Each layer receives its shared-prefix
    input as a keyword argument of its own call: self-attention layers ``shared_prefix_forest``,
    Mamba layers ``shared_prefix_call`` (built by :func:`forward_mamba_layer_shared_prefix`), and,
    with expert bias, MoE layers ``shared_prefix_token_multiplicities``. No module state is set,
    so recomputation replays a layer with the arguments of its own call.

    ``exclude_sequence_padding_from_expert_bias`` states the caller's dense expert-bias
    convention for per-branch padding rows. ``True`` excludes them, as a dense forward with the
    packed-sequence padding mask does; ``False`` counts them, as a dense forward with
    ``padding_mask=None`` does. ``None`` (the default) excludes them with the flex dispatcher's
    HybridEP backend and counts them otherwise. Trailing topology padding is never counted.

    ``validated_physical_len`` is the global physical length for which the caller has already
    validated this stack and layout, as ``HybridModel.forward`` does before its embedding. The
    stack validation is then not repeated; the length implied by ``hidden_states`` must match.
    """
    if hidden_states.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("shared-prefix forest attention requires fp16 or bf16 hidden states")
    if hidden_states.ndim != 3 or hidden_states.shape[1] != 1:
        raise ValueError(
            "shared-prefix Hybrid input must have shape [sequence/(TP*CP), 1, hidden] "
            "when sequence parallelism is enabled"
        )
    cp_group = stack.pg_collection.cp
    tp_group = stack.pg_collection.tp
    tp_size = tp_group.size()
    sequence_shards = tp_size if stack.config.sequence_parallel else 1
    physical_len = hidden_states.shape[0] * cp_group.size() * sequence_shards
    if validated_physical_len is None:
        _validate_hybrid_stack(stack, layout, physical_len=physical_len)
    elif validated_physical_len != physical_len:
        raise RuntimeError(
            f"shared-prefix hidden states imply physical length {physical_len}, but the stack "
            f"was validated for {validated_physical_len}"
        )
    has_attention = any(
        isinstance(layer, TransformerLayer) and isinstance(layer.self_attention, SelfAttention)
        for layer in stack.layers
    )
    if position_embedding_type not in ("rope", "none"):
        raise NotImplementedError(
            "shared-prefix attention supports only RoPE or positionless Hybrid models"
        )
    if has_attention and position_embedding_type == "rope" and rotary_pos_emb is None:
        raise ValueError("position-aware rotary_pos_emb is required for shared-prefix attention")
    if position_embedding_type == "none" and rotary_pos_emb is not None:
        raise ValueError("positionless shared-prefix attention must not receive rotary_pos_emb")

    token_multiplicities = None
    expert_bias_enabled = bool(getattr(stack.config, "moe_router_enable_expert_bias", False))
    if expert_bias_enabled:
        exclude_sequence_padding_from_expert_bias = (
            resolve_exclude_sequence_padding_from_expert_bias(
                stack.config, exclude_sequence_padding_from_expert_bias
            )
        )
        token_multiplicities = layout.padded_token_multiplicities(
            physical_len,
            hidden_states.device,
            exclude_sequence_padding=exclude_sequence_padding_from_expert_bias,
        )
        if cp_group.size() > 1:
            token_multiplicities = token_multiplicities.index_select(
                0,
                layout.cp_local_indices(
                    physical_len, cp_group.size(), cp_group.rank(), hidden_states.device
                ),
            )
        if tp_size > 1:
            if token_multiplicities.numel() % tp_size:
                raise RuntimeError(
                    "shared-prefix CP-local token multiplicities must divide evenly over TP"
                )
            token_multiplicities = torch.chunk(token_multiplicities, tp_size, dim=0)[
                tp_group.rank()
            ].contiguous()
        if token_multiplicities.numel() != hidden_states.shape[0]:
            raise RuntimeError(
                "shared-prefix token multiplicity ownership does not match the local hidden rows"
            )

    # Self-attention takes the layout as a forest; a single star is a forest with one root.
    attention_forest = (
        layout
        if isinstance(layout, SharedPrefixForestLayout)
        else SharedPrefixForestLayout((layout,))
    )

    @router_gating_token_blocks()
    def forward_layer_range(hidden_states: Tensor, start: int, end: int) -> Tensor:
        for layer in stack.layers[start:end]:
            if isinstance(layer, MambaLayer):
                # Runs the layer through MambaLayer.__call__ and MambaMixer.__call__, like the
                # other layers below, so module hooks fire as on the dense path. Every
                # topology and root count honors NRL_SP_MAMBA_IMPL.
                hidden_states = forward_mamba_layer_shared_prefix(layer, hidden_states, layout)
            else:
                layer_kwargs = {}
                if not isinstance(layer.self_attention, IdentityOp):
                    layer_kwargs["rotary_pos_emb"] = rotary_pos_emb
                    layer_kwargs["shared_prefix_forest"] = attention_forest
                if expert_bias_enabled and getattr(layer, "is_moe_layer", False):
                    layer_kwargs["shared_prefix_token_multiplicities"] = token_multiplicities
                hidden_states = layer(hidden_states, attention_mask=None, **layer_kwargs)

            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
        return hidden_states

    if stack.config.recompute_granularity == "full" and stack.training:
        chunk_size = stack.config.recompute_num_layers
        for start in range(0, len(stack.layers), chunk_size):
            end = min(start + chunk_size, len(stack.layers))

            def custom_forward(hidden_states: Tensor, start=start, end=end) -> Tensor:
                # The router scope and the layer calls, with their shared-prefix arguments, live
                # inside the callable, so backward replay reconstructs them.
                return forward_layer_range(hidden_states, start, end)

            hidden_states = tensor_parallel.checkpoint(
                custom_forward, stack.config.distribute_saved_activations, hidden_states
            )
    else:
        hidden_states = forward_layer_range(hidden_states, 0, len(stack.layers))

    if stack.post_process and stack.post_layer_norm:
        hidden_states = stack.final_norm(hidden_states)
    return make_viewless_tensor(
        inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
    )


# Capability tokens are contract identifiers: each names a code path that MCore implements and
# guards. They are not qualification evidence for a topology or a model.
# The legacy scalar remains the CP1 negotiation surface. CP-aware integrations must negotiate
# against the collection so installing the CP track cannot regress existing CP1 runs.
# ``SHARED_PREFIX_CP_TRAINING_CAPABILITY`` is advertised independently so integrations can retain
# the CP1 fast path while requiring the CP>1 Hybrid forward/backward contract.
SHARED_PREFIX_TRAINING_CAPABILITY = "hybrid_star_cp1_tp1_v1"
SHARED_PREFIX_CP_TRAINING_CAPABILITY = "hybrid_star_cp_v1"
SHARED_PREFIX_EXPLICIT_PHYSICAL_PADDING_CAPABILITY = "hybrid_star_explicit_physical_padding_v1"
SHARED_PREFIX_MOE_EXPERT_BIAS_CAPABILITY = "hybrid_star_moe_expert_bias_v1"
SHARED_PREFIX_FULL_RECOMPUTE_CAPABILITY = "hybrid_star_full_uniform_recompute_v1"
SHARED_PREFIX_TP_SP_TRAINING_CAPABILITY = "hybrid_star_cp1_tp_sp_v1"
SHARED_PREFIX_CP_TP_SP_TRAINING_CAPABILITY = "hybrid_star_cp_tp_sp_v1"
# Target-model feature: Nemotron-H attention is positionless while Mamba remains
# state-positioned by sequence order.
SHARED_PREFIX_POSITIONLESS_ATTENTION_CAPABILITY = "hybrid_star_positionless_attention_v1"
# MTP predictor feature: reconstruct dense attention/MLP or attention/MoE heads from
# the shared-prefix physical layout on the implemented TP/CP topologies.
SHARED_PREFIX_MTP_DENSE_HEADS_CAPABILITY = "hybrid_star_mtp_dense_heads_v1"
SHARED_PREFIX_GROUP_EXECUTION_CAPABILITY = "hybrid_forest_stable_router_v1"
# Topology and feature capabilities are independent so integrations can negotiate their exact
# conjunction without inferring support from a broader aggregate token.
SHARED_PREFIX_TRAINING_CAPABILITIES = frozenset(
    {
        SHARED_PREFIX_TRAINING_CAPABILITY,
        SHARED_PREFIX_CP_TRAINING_CAPABILITY,
        SHARED_PREFIX_EXPLICIT_PHYSICAL_PADDING_CAPABILITY,
        SHARED_PREFIX_MOE_EXPERT_BIAS_CAPABILITY,
        SHARED_PREFIX_FULL_RECOMPUTE_CAPABILITY,
        SHARED_PREFIX_TP_SP_TRAINING_CAPABILITY,
        SHARED_PREFIX_CP_TP_SP_TRAINING_CAPABILITY,
        SHARED_PREFIX_POSITIONLESS_ATTENTION_CAPABILITY,
        SHARED_PREFIX_MTP_DENSE_HEADS_CAPABILITY,
        SHARED_PREFIX_GROUP_EXECUTION_CAPABILITY,
    }
)


def validate_hybrid_model_shared_prefix(
    model: "HybridModel",
    layout: SharedPrefixLayout | SharedPrefixForestLayout,
    *,
    input_ids: Optional[Tensor],
    decoder_input: Optional[Tensor],
    attention_mask: Optional[Tensor],
    labels: Optional[Tensor],
    loss_mask: Optional[Tensor],
    mtp_input_mask: Optional[Tensor],
    packed_seq_params: Optional[PackedSeqParams],
    padding_mask: Optional[Tensor],
    cp_batch,
    inference_context,
    inference_params,
    output_processor,
    compute_mtp_loss: bool,
    runtime_gather_output: Optional[bool],
) -> int:
    """Validate the arguments of a shared-prefix ``HybridModel.forward`` call.

    Runs before any preprocessing, so an unsupported request fails without a partial forward.
    Returns the global physical length of the shared-prefix sequence.

    The forward's contract with a ``layout``:

    * The global packed input is ``[prefix, completion_1, ..., completion_G,
      optional_topology_padding]`` with batch size one. CP1 receives it whole; CP>1 receives the
      standard two-chunk zigzag sequence shards. With TP>1, input IDs remain CP-local and
      replicated over TP; sequence parallelism starts at the embedding output.
    * The layout owns the exact tree mask and the prefix-continued RoPE positions.
    * With ``labels=None`` and parallel output, logits remain
      ``[1, physical_len/CP, padded_vocab/TP]`` in the input shard's zigzag token order: the
      model gathers neither the sequence across CP nor the vocabulary across TP.
    * MTP runs only in training mode and never changes the returned logits. It rebuilds the
      dense branches from ``input_ids`` and requires ``loss_mask``; it rejects ``decoder_input``
      and ``mtp_input_mask``, and a forest layout also requires ``calculate_per_token_loss``.
    * Models built from ``hybrid_layer_config_list`` are rejected.
    """
    if cp_batch is not None:
        raise ValueError("Shared prefix owns CP token layout; cp_batch must be None")
    # The shared-prefix guards and MTP predictor checks read the model-level config and
    # hybrid_layer_pattern; config-list layers carry their own independent configs.
    if model.hybrid_layer_config_list is not None:
        raise NotImplementedError(
            "shared-prefix Hybrid forward does not yet support hybrid_layer_config_list; "
            "use hybrid_layer_pattern"
        )
    if InferenceMode.is_active() or inference_context is not None or inference_params is not None:
        raise NotImplementedError("shared-prefix Hybrid forward is training-only")
    if attention_mask is not None:
        raise ValueError(
            'shared-prefix Hybrid forward owns its tree mask and requires attention_mask=None'
        )
    if labels is not None:
        raise NotImplementedError(
            "shared-prefix Hybrid forward requires external next-token loss computation"
        )
    # The post-decoder hook would receive hidden states in the shared-prefix token layout
    # (stars, CP zigzag), not the dense rows its callers expect.
    if output_processor is not None:
        raise NotImplementedError("shared-prefix Hybrid forward does not support output_processor")
    if packed_seq_params is not None:
        raise ValueError("shared_prefix_layout and packed_seq_params are mutually exclusive")
    if padding_mask is not None:
        raise ValueError("shared-prefix Hybrid forward does not accept padding_mask")
    if not model.pre_process or not model.post_process or model.vp_stage is not None:
        raise NotImplementedError(
            "shared-prefix HybridModel forward currently requires a complete PP1 model"
        )
    # Shared-prefix MTP runs only in training; see ``compute_mtp_loss``.
    if model.mtp_process and model.training and compute_mtp_loss:
        _validate_shared_prefix_mtp_pattern(model.mtp_pattern)
        _validate_shared_prefix_mtp_attention_backend(model.config.attention_backend)
        if loss_mask is None:
            raise ValueError("shared-prefix Hybrid MTP requires an explicit loss mask")
        # The dense branches are rebuilt from input_ids and loss_mask only, so
        # reject the inputs that the dense MTP path would otherwise consume.
        if decoder_input is not None:
            raise NotImplementedError(
                "shared-prefix Hybrid MTP re-embeds input_ids and does not support a "
                "precomputed decoder_input; pass compute_mtp_loss=False to skip MTP"
            )
        if mtp_input_mask is not None:
            raise NotImplementedError(
                "shared-prefix Hybrid MTP does not support mtp_input_mask; pass "
                "compute_mtp_loss=False to skip MTP"
            )
        # Every forest normalizes MTP per loss group (one group per root by
        # default), which process_mtp_loss supports only with per-token loss.
        if (
            isinstance(layout, SharedPrefixForestLayout)
            and not model.config.calculate_per_token_loss
        ):
            raise NotImplementedError(
                "shared-prefix forest MTP loss groups require calculate_per_token_loss=True"
            )
    if model.position_embedding_type not in ('rope', 'none'):
        raise NotImplementedError(
            "shared-prefix Hybrid forward supports only RoPE or positionless attention"
        )
    if model.config.multi_latent_attention:
        raise NotImplementedError(
            "shared-prefix Hybrid forward does not support multi-latent attention"
        )
    tp_size = model.pg_collection.tp.size()
    cp_size = model.pg_collection.cp.size()
    if decoder_input is None:
        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("shared-prefix Hybrid input_ids must have shape [1, physical_len/CP]")
        physical_len = input_ids.shape[1] * cp_size
    else:
        if decoder_input.ndim != 3 or decoder_input.shape[1] != 1:
            raise ValueError(
                "shared-prefix Hybrid decoder input must have shape "
                "[physical_len/(TP*CP), 1, hidden] when sequence parallelism is enabled"
            )
        sequence_shards = tp_size if model.config.sequence_parallel else 1
        physical_len = decoder_input.shape[0] * cp_size * sequence_shards
    # Topology (TP/SP/CP/PP), layer types, MoE features and the physical length.
    _validate_hybrid_stack(model.decoder, layout, physical_len=physical_len)
    if tp_size > 1 and (not model.parallel_output or runtime_gather_output):
        raise NotImplementedError(
            "shared-prefix HybridModel TP/SP requires TP-sharded parallel output logits"
        )
    return physical_len


def hybrid_shared_prefix_rotary_pos_emb(
    model: "HybridModel", layout: SharedPrefixLayout | SharedPrefixForestLayout, physical_len: int
) -> Tensor:
    """RoPE rows for this CP rank's shard: every branch continues its prefix's positions."""
    cp_group = model.pg_collection.cp
    cp_size = cp_group.size()
    rotary_table = model.rotary_pos_emb.get_emb(max(layout.dense_branch_lengths))
    global_position_ids = layout.padded_position_ids(physical_len, rotary_table.device)
    if cp_size > 1:
        local_indices = layout.cp_local_indices(
            physical_len, cp_size, cp_group.rank(), rotary_table.device
        )
        global_position_ids = global_position_ids.index_select(0, local_indices)
    return rotary_table.index_select(0, global_position_ids)


def _canonicalize_shared_prefix_cp_sequence(
    local_tensor: Tensor,
    layout: SharedPrefixLayout | SharedPrefixForestLayout,
    physical_len: int,
    cp_group: torch.distributed.ProcessGroup,
    *,
    reduce_scatter_grad: bool,
) -> Tensor:
    """Gather one zigzag CP shard and restore canonical global star order."""
    cp_size = cp_group.size()
    if local_tensor.shape[0] * cp_size != physical_len:
        raise ValueError(
            "shared-prefix CP-local tensor length does not match the physical star: "
            f"{local_tensor.shape[0]} * {cp_size} != {physical_len}"
        )
    if cp_size == 1:
        return local_tensor

    rank_order_tensor = gather_from_sequence_parallel_region(
        local_tensor, tensor_parallel_output_grad=reduce_scatter_grad, group=cp_group
    )
    rank_order_indices = torch.cat(
        [
            layout.cp_local_indices(physical_len, cp_size, rank, local_tensor.device)
            for rank in range(cp_size)
        ]
    )
    inverse_order = torch.empty_like(rank_order_indices)
    inverse_order[rank_order_indices] = torch.arange(
        physical_len, device=local_tensor.device, dtype=torch.long
    )
    return rank_order_tensor.index_select(0, inverse_order)


def _validate_shared_prefix_mtp_star(
    global_hidden_states: Tensor,
    global_input_ids: Tensor,
    global_loss_mask: Tensor,
    layout: SharedPrefixLayout | SharedPrefixForestLayout,
    *,
    cp_size: int,
    cp_rank: int,
) -> None:
    """Validate one canonical star before its dense MTP branches are reconstructed.

    The loss-mask contract (no loss on the prompt, on ordinary per-branch padding,
    or on topology-only padding) is checked with one device-side reduction and a
    single host sync.  The offending region class is re-derived, with extra
    syncs, only on the error path so the messages stay specific.
    """
    physical_len = global_hidden_states.shape[0]
    if global_hidden_states.ndim != 3 or global_hidden_states.shape[1] != 1:
        raise ValueError("shared-prefix MTP hidden states must have shape [tokens, 1, hidden]")
    if global_input_ids.shape != (1, physical_len):
        raise ValueError("shared-prefix MTP input IDs must have shape [1, physical_tokens]")
    if global_loss_mask.shape != (1, physical_len):
        raise ValueError("shared-prefix MTP loss mask must have shape [1, physical_tokens]")
    if physical_len < layout.total_len:
        raise ValueError(
            f"shared-prefix MTP physical length {physical_len} is shorter than layout "
            f"length {layout.total_len}"
        )
    if any(
        root.logical_completion_lens is None or root.padding_multiple is None
        for _, root in layout.iter_roots()
    ):
        raise NotImplementedError(
            "shared-prefix MTP requires explicit logical completion lengths and physical padding"
        )
    if cp_size < 1 or not 0 <= cp_rank < cp_size:
        raise ValueError("shared-prefix MTP received an invalid CP size/rank")

    # Every region below is defined by the layout's Python ints, so the mask is
    # built with plain kernels and the whole contract costs one ``.item()``.
    device = global_loss_mask.device
    must_be_zero = torch.ones(physical_len, dtype=torch.bool, device=device)
    for offset, root in layout.iter_roots():
        for branch, logical_len in zip(
            root.completion_slices(), root.logical_completion_lens, strict=True
        ):
            must_be_zero[offset + branch.start : offset + branch.start + logical_len] = False
    if ((global_loss_mask[0] != 0) & must_be_zero).any().item():
        for offset, root in layout.iter_roots():
            if torch.count_nonzero(global_loss_mask[:, offset : offset + root.prefix_len]).item():
                raise ValueError("shared-prefix MTP loss mask must exclude every prompt token")
        if torch.count_nonzero(global_loss_mask[:, layout.total_len :]).item():
            raise ValueError("shared-prefix MTP loss mask must exclude topology-only padding")
        raise ValueError("shared-prefix MTP loss mask must exclude ordinary per-sequence padding")


def _shared_prefix_mtp_branch_indices(
    layout: SharedPrefixLayout | SharedPrefixForestLayout,
    device: torch.device | str,
    *,
    cp_size: int = 1,
    cp_rank: int = 0,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    """Return per-branch ``(star_indices, dense_positions)`` in this rank's CP-local order.

    Branch ``i`` is the conventional dense sequence ``[prompt, completion_i]``,
    including that completion's ordinary per-sequence padding.  ``star_indices[i]``
    selects those tokens from the canonical global star and ``dense_positions[i]``
    is each selected token's ``arange`` position inside its own dense branch.  With
    CP enabled both are composed with the branch's standard two-chunk zigzag
    ownership, so the full dense branch is never allocated.
    """
    star_indices = []
    dense_positions = []
    for indices in layout.dense_branch_indices(device):
        if cp_size > 1:
            positions = SharedPrefixLayout.cp_local_indices(
                indices.numel(), cp_size, cp_rank, device
            )
            indices = indices.index_select(0, positions)
        else:
            positions = torch.arange(indices.numel(), device=device, dtype=torch.long)
        star_indices.append(indices)
        dense_positions.append(positions)
    return tuple(star_indices), tuple(dense_positions)


def _fixed_order_copy_contributors(indices: Tensor, num_rows: int, copies: int) -> Tensor:
    """Return a ``[copies, num_rows]`` map from each source row to the positions copying it.

    Entry ``[k, row]`` is the position in ``indices`` of the ``k``-th copy of ``row`` (in
    ``indices`` order), or ``indices.numel()`` when ``row`` has fewer than ``k + 1`` copies.
    Built on device without a host sync.
    """
    order = torch.argsort(indices, stable=True)
    sorted_rows = indices.index_select(0, order)
    copy = torch.arange(indices.numel(), device=indices.device) - torch.searchsorted(
        sorted_rows, sorted_rows
    )
    contributors = torch.full(
        (copies, num_rows), indices.numel(), device=indices.device, dtype=torch.long
    )
    contributors[copy, sorted_rows] = order
    return contributors


class _FixedOrderRowGather(torch.autograd.Function):
    """``index_select`` over rows whose backward sums repeated rows in a fixed FP32 order.

    The backward of ``index_select`` scatters with atomic adds in the input dtype, so
    the prompt rows that every dense MTP branch copies would get run-to-run different
    BF16 gradients. Only ``multi_rows`` can have more than one copy: each adds its
    copies in copy order in FP32 and rounds once. Every other row has at most one
    copy, which is its exact gradient, gathered in the input dtype.
    """

    @staticmethod
    def forward(ctx, value, indices, first_copies, multi_rows, multi_copies):
        """Gather ``value[indices]``."""
        ctx.save_for_backward(first_copies, multi_rows, multi_copies)
        return value.index_select(0, indices)

    @staticmethod
    def backward(ctx, gradient):
        """Gather single-copy rows, then overwrite ``multi_rows`` with their ordered sum."""
        first_copies, multi_rows, multi_copies = ctx.saved_tensors
        sentinel = gradient.shape[0]
        if sentinel == 0:
            empty = gradient.new_zeros((first_copies.numel(), *gradient.shape[1:]))
            return empty, None, None, None, None
        missing_shape = (-1, *(1,) * (gradient.dim() - 1))

        def gather(positions):
            # The sentinel position marks a missing copy and contributes an exact zero.
            rows = gradient.index_select(0, positions.clamp(max=sentinel - 1))
            return rows.masked_fill_((positions == sentinel).view(missing_shape), 0)

        result = gather(first_copies)
        if multi_rows.numel() > 0:
            accumulation_dtype = torch.float64 if gradient.dtype == torch.float64 else torch.float32
            accumulation = gather(multi_copies[0]).to(accumulation_dtype)
            for positions in multi_copies[1:].unbind(0):
                # In-place type promotion upcasts each copy exactly before adding.
                accumulation.add_(gather(positions))
            result.index_copy_(0, multi_rows, accumulation.to(gradient.dtype))
        return result, None, None, None, None


def _gather_shared_prefix_mtp_rows(
    global_hidden_states: Tensor,
    packed_indices: Tensor,
    layout: SharedPrefixLayout | SharedPrefixForestLayout,
) -> Tensor:
    """Gather dense MTP branch rows of one canonical star with a reproducible backward."""
    # A prompt row is copied at most once per dense branch of its root; every other
    # row belongs to at most one branch.
    copies = max(len(root.completion_lens) for _, root in layout.iter_roots())
    contributors = _fixed_order_copy_contributors(
        packed_indices, global_hidden_states.shape[0], copies
    )
    multi_rows = torch.cat(
        [
            torch.arange(offset, offset + root.prefix_len, device=packed_indices.device)
            for offset, root in layout.iter_roots()
        ]
    )
    return _FixedOrderRowGather.apply(
        global_hidden_states,
        packed_indices,
        contributors[0].clone(),
        multi_rows,
        contributors.index_select(1, multi_rows),
    )


def _pack_shared_prefix_mtp_branches(
    global_hidden_states: Tensor,
    global_input_ids: Tensor,
    global_loss_mask: Tensor,
    layout: SharedPrefixLayout | SharedPrefixForestLayout,
    *,
    cp_size: int = 1,
    cp_rank: int = 0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Pack every dense MTP branch of one canonical star into a single THD sequence.

    Returns ``(hidden_states, input_ids, loss_mask, position_ids)`` for the
    branch-major concatenation ``[prompt + completion_1 | ... | prompt +
    completion_G]`` in this rank's CP-local order, gathered with exactly one
    ``index_select`` per tensor. Positions restart at zero for every branch,
    exactly as in a conventional dense batch. The hidden-state gradient sums each
    prompt row's branch copies in a fixed FP32 order, so it is reproducible.
    """
    _validate_shared_prefix_mtp_star(
        global_hidden_states,
        global_input_ids,
        global_loss_mask,
        layout,
        cp_size=cp_size,
        cp_rank=cp_rank,
    )
    star_indices, dense_positions = _shared_prefix_mtp_branch_indices(
        layout, global_hidden_states.device, cp_size=cp_size, cp_rank=cp_rank
    )
    packed_indices = torch.cat(star_indices)
    return (
        _gather_shared_prefix_mtp_rows(global_hidden_states, packed_indices, layout),
        global_input_ids.index_select(1, packed_indices),
        global_loss_mask.index_select(1, packed_indices),
        torch.cat(dense_positions).unsqueeze(0),
    )


def _shared_prefix_mtp_padding_mask(
    layout: SharedPrefixLayout | SharedPrefixForestLayout,
    packed_position_ids: Tensor,
    *,
    cp_size: int = 1,
) -> Tensor:
    """Per-sequence padding of the packed dense MTP branches, ``True`` at padding rows.

    A dense branch ``[prompt, completion_i]`` is padded after its logical completion length,
    so a row is padding when its position in the branch is at or past
    ``prefix_len + logical_completion_len``. ``packed_position_ids`` is the branch-major
    CP-local position tensor built with the same branch order.
    """
    limits = []
    lengths = []
    for _, root in layout.iter_roots():
        logical_lens = root.logical_completion_lens or root.completion_lens
        for branch_len, logical_len in zip(root.dense_branch_lengths, logical_lens):
            limits.append(root.prefix_len + logical_len)
            lengths.append(branch_len // cp_size)
    limit = torch.tensor(limits, dtype=torch.long).repeat_interleave(torch.tensor(lengths))
    return packed_position_ids >= limit.to(packed_position_ids.device).unsqueeze(0)


def _validate_shared_prefix_mtp_pattern(mtp_pattern: Optional[str]) -> None:
    """Keep the promoted predictor scope narrower than the Hybrid backbone scope."""
    if mtp_pattern is not None and 'M' in mtp_pattern:
        raise NotImplementedError(
            "shared-prefix MTP supports a Mamba Hybrid backbone but does not yet "
            "support a Mamba layer inside the MTP predictor; use an attention/MLP "
            "or attention/MoE MTP pattern such as '*-' or '*E'"
        )


def _validate_shared_prefix_mtp_attention_backend(attention_backend: AttnBackend) -> None:
    """Reject the one attention backend that can never run the packed MTP branches.

    Shared-prefix MTP packs every dense branch into a single THD sequence.  mcore's
    local ``DotProductAttention`` asserts ``packed_seq_params is None``, so it cannot
    serve that pack under any version.  Transformer Engine backends are left to TE's
    own per-release backend selection, which raises when the installed release has
    no THD-capable kernel for the requested backend.
    """
    if attention_backend == AttnBackend.local:
        raise NotImplementedError(
            "shared-prefix MTP packs every dense branch into one THD sequence, which "
            "mcore's local DotProductAttention does not support; use a Transformer Engine "
            "attention backend (flash/fused/unfused/auto)"
        )


def forward_hybrid_mtp_shared_prefix(
    model: "HybridModel",
    *,
    hidden_states: Tensor,
    input_ids: Tensor,
    loss_mask: Tensor,
    layout: SharedPrefixLayout | SharedPrefixForestLayout,
    output_weight: Optional[Tensor],
    exclude_sequence_padding_from_expert_bias: Optional[bool] = None,
) -> Tensor:
    """Run the MTP heads once over all dense branches, preserving the backbone graph.

    The star is expanded into the branch-major dense sequence
    ``[prompt + completion_1 | ... | prompt + completion_G]`` and the MTP block runs
    a single time on it as a THD packed batch.  Attention, RoPE and the MTP token
    shifts all respect the ``cu_seqlens`` branch boundaries, so this equals running
    the block once per branch while issuing one set of kernels and collectives.

    With expert bias enabled, ``exclude_sequence_padding_from_expert_bias`` selects the
    same dense padding convention as the backbone: when it resolves to ``True``, the
    branches' per-sequence padding is passed to the MTP block as ``padding_mask``.
    """
    tp_group = model.pg_collection.tp
    cp_group = model.pg_collection.cp
    tp_size = tp_group.size()
    cp_size = cp_group.size()
    physical_len = input_ids.shape[1] * cp_size

    # The shared backbone is SP-sharded over TP.  Gather without reducing
    # gradients because the packed branches are scattered over TP again
    # below, so each sequence position has one downstream owner.
    # Detach before gathering and expansion when MTP cannot update the backbone.
    # The loss anchor otherwise retains zero-gradient backward communication.
    cp_local_hidden = hidden_states.detach() if model.config.mtp_detach_heads else hidden_states
    if tp_size > 1:
        cp_local_hidden = gather_from_sequence_parallel_region(
            cp_local_hidden, tensor_parallel_output_grad=False, group=tp_group
        )

    global_hidden = _canonicalize_shared_prefix_cp_sequence(
        cp_local_hidden, layout, physical_len, cp_group, reduce_scatter_grad=True
    )
    global_input_ids = _canonicalize_shared_prefix_cp_sequence(
        input_ids.transpose(0, 1).contiguous(),
        layout,
        physical_len,
        cp_group,
        reduce_scatter_grad=False,
    ).transpose(0, 1)
    global_loss_mask = _canonicalize_shared_prefix_cp_sequence(
        loss_mask.transpose(0, 1).contiguous(),
        layout,
        physical_len,
        cp_group,
        reduce_scatter_grad=False,
    ).transpose(0, 1)
    global_branch_lengths = layout.dense_branch_lengths
    if any(branch_len % cp_size for branch_len in global_branch_lengths):
        raise ValueError("shared-prefix MTP branch length must be divisible by CP size")
    combined_cp_length = sum(global_branch_lengths) // cp_size
    if combined_cp_length % tp_size:
        raise ValueError(
            "shared-prefix MTP combined CP-local sequence must be divisible by TP size"
        )
    packed_sp_length = combined_cp_length // tp_size
    required_alignment = tp_size if cp_size == 1 else 2 * cp_size * tp_size
    for branch_len in global_branch_lengths:
        if branch_len % required_alignment:
            raise ValueError(
                "shared-prefix MTP physical branch length must be divisible by "
                f"the CP/TP sequence quantum {required_alignment}, got {branch_len}"
            )

    # Build the dense branch-major sequence [P+C_1 | ... | P+C_G] in this
    # rank's CP-local zigzag order with one index_select per tensor.  Under
    # THD every branch is its own attention sequence and its own roll_tensor
    # segment, so one MTP call over the pack equals G independent dense calls.
    packed_hidden, packed_input_ids, packed_loss_mask, packed_position_ids = (
        _pack_shared_prefix_mtp_branches(
            global_hidden,
            global_input_ids,
            global_loss_mask,
            layout,
            cp_size=cp_size,
            cp_rank=cp_group.rank(),
        )
    )
    del cp_local_hidden, global_hidden, global_input_ids, global_loss_mask
    packed_padding_mask = None
    if getattr(model.config, "moe_router_enable_expert_bias", False):
        if resolve_exclude_sequence_padding_from_expert_bias(
            model.config, exclude_sequence_padding_from_expert_bias
        ):
            # As in a dense forward with padding_mask: the MTP routers exclude padding.
            packed_padding_mask = _shared_prefix_mtp_padding_mask(
                layout, packed_position_ids, cp_size=cp_size
            )
    if packed_hidden.shape[0] != combined_cp_length:
        raise RuntimeError(
            "shared-prefix MTP packed CP-local sequence has an invalid length: "
            f"{packed_hidden.shape[0]} != {combined_cp_length}"
        )
    if tp_size > 1:
        # Scatter the whole pack once.  This rank's SP shard is its contiguous
        # slice of the branch-major sequence, which is the exact per-depth
        # layout process_mtp_loss consumes below, so no TP gathers are needed.
        packed_hidden = tensor_parallel.scatter_to_sequence_parallel_region(
            packed_hidden, group=tp_group
        )
        if packed_padding_mask is not None:
            # The MTP block takes an SP-sharded mask, like the dense forward's.
            packed_padding_mask = (
                tensor_parallel.scatter_to_sequence_parallel_region(
                    packed_padding_mask.transpose(0, 1).contiguous(), group=tp_group
                )
                .transpose(0, 1)
                .contiguous()
            )

    # Global (pre-CP) cumulative branch lengths, as in the packed non-shared
    # path: TE attention, THD RoPE, and roll_tensor divide by CP internally and
    # map each branch onto this rank's two zigzag chunks.  cp_group/local_cp_size
    # stay unset exactly like trainer-built PackedSeqParams, so TE keeps the CP
    # group it was constructed with instead of re-binding it every microbatch.
    cumulative_lengths = [0]
    for branch_len in global_branch_lengths:
        cumulative_lengths.append(cumulative_lengths[-1] + branch_len)
    cu_seqlens = torch.tensor(cumulative_lengths, device=packed_input_ids.device, dtype=torch.int32)
    packed_seq_params = PackedSeqParams(
        qkv_format='thd',
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max(global_branch_lengths),
        max_seqlen_kv=max(global_branch_lengths),
    )

    # Mirror the packed non-shared forward: one table covering the longest
    # branch, not CP-sliced (packed_seq=True).  THD RoPE selects each branch's
    # positions from cu_seqlens, taking this CP rank's front/back zigzag chunks
    # of every branch, which is what RotaryEmbedding(branch_len) did per branch.
    if model.position_embedding_type == 'rope':
        rotary_seq_len = model.rotary_pos_emb.get_rotary_seq_len(
            None, None, None, model.config, packed_seq_params
        )
        packed_rotary_pos_emb = model.rotary_pos_emb(rotary_seq_len, packed_seq=True)
    elif model.position_embedding_type == 'none':
        packed_rotary_pos_emb = None
    else:
        raise NotImplementedError(
            "shared-prefix MTP supports only RoPE or positionless Hybrid models"
        )
    # Use the backbone's fixed router GEMM row blocks for MoE layers inside the
    # predictor too, so star and forest packing give the same MTP routing.
    with router_gating_token_blocks():
        packed_mtp_hidden = model.mtp(
            input_ids=packed_input_ids,
            position_ids=packed_position_ids,
            hidden_states=packed_hidden,
            attention_mask=None,
            inference_params=None,
            rotary_pos_emb=packed_rotary_pos_emb,
            packed_seq_params=packed_seq_params,
            embedding=model.embedding,
            padding_mask=packed_padding_mask,
        )
    del packed_hidden
    # MultiTokenPredictionBlock returns depth-major chunks [depth_0 | ... |
    # depth_D], each the SP shard of the branch-major pack, which is the layout
    # process_mtp_loss chunks by depth.
    depth_count = 1 + model.config.mtp_num_layers
    if packed_mtp_hidden.shape[0] != depth_count * packed_sp_length:
        raise RuntimeError(
            "shared-prefix MTP returned an invalid depth-major SP shard length: "
            f"{packed_mtp_hidden.shape[0]} != {depth_count} * {packed_sp_length}"
        )
    processed_mtp_hidden = process_mtp_loss(
        hidden_states=packed_mtp_hidden,
        labels=None,
        loss_mask=packed_loss_mask,
        output_layer=model.output_layer,
        output_weight=output_weight,
        is_training=model.training,
        compute_language_model_loss=model.compute_language_model_loss,
        config=model.config,
        cp_group=cp_group,
        tp_group=model.tp_group,
        packed_seq_params=packed_seq_params,
        scale_logits_fn=model._scale_logits if model.config.use_mup else None,
        input_ids=packed_input_ids,
        metric_avg_group=(
            getattr(model.pg_collection, "dp_cp_gtp_remat", None) or model.pg_collection.dp_cp
        ),
        loss_group_lengths=(
            tuple(length // cp_size for length in layout.mtp_loss_group_lengths)
            if isinstance(layout, SharedPrefixForestLayout)
            else None
        ),
    )

    # The external RL loss consumes star logits.  A zero-valued attachment
    # retains process_mtp_loss's MTPLossAutoScaler nodes without changing
    # those logits; its backward hook supplies the auxiliary-loss gradient.
    mtp_loss_anchor = processed_mtp_hidden.reshape(-1)[0] * 0.0
    return hidden_states + mtp_loss_anchor
