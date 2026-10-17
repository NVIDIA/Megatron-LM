# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Explicit shared-prefix adapter for a HybridStack.

This module deliberately does not alter :class:`HybridStack`'s normal ``forward`` path. Call
``forward_hybrid_stack_shared_prefix`` with a shared-prefix layout (stars ``[P, C_1, ..., C_G]``)
to opt in. This module validates the stack and runs its layers in order; the layer-specific
execution lives with the layers. Mamba layers run
:func:`megatron.core.ssm.mamba_shared_prefix.forward_mamba_layer_shared_prefix`, which scans each
prefix once and forks its state into the completion branches. Attention layers run the
exact-backward forest attention in :mod:`megatron.core.transformer.shared_prefix_attention`.
Unsupported topology or model features fail before executing a partial forward.

CP>1 uses the model's standard zigzag sequence shards and sequence-to-head all-to-alls. The
implemented surface covers TP1 and TP>1 with sequence parallelism, CP1 and CP>1, explicit physical
padding, MoE expert-bias accounting, and full uniform activation recomputation. The capability
tokens at the end of this module name these code paths so integrations can require the exact
conjunction they use; they are contract identifiers, not evidence that a topology has been
qualified at full-model scale.
"""

import torch
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.shared_prefix_layout import SharedPrefixForestLayout, SharedPrefixLayout
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.ssm.mamba_shared_prefix import (
    forward_mamba_layer_shared_prefix,
    validate_mamba_layer_shared_prefix,
)
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.moe.moe_utils import router_gating_token_blocks
from megatron.core.transformer.shared_prefix_attention import (
    require_shared_prefix_flash_attn,
    validate_shared_prefix_attention,
)
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import make_viewless_tensor

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
        raise NotImplementedError("shared-prefix fused attention supports only vanilla softmax")

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

    has_attention = False
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
                has_attention = True
                validate_shared_prefix_attention(
                    layer.self_attention, stack.config, tp_size=tp_size, cp_size=cp_size
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
    if has_attention:
        # The attention entry repeats this check; failing here precedes the first Mamba layer.
        require_shared_prefix_flash_attn()


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

    Normal ``HybridStack.forward`` and ``Attention.forward`` behavior is unchanged unless this
    function installs its scoped private forest descriptor. The descriptor is always removed in a
    ``finally`` block, including when a layer raises.

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
        raise TypeError("fused shared-prefix attention requires fp16 or bf16 hidden states")
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

    @router_gating_token_blocks()
    def forward_layer_range(hidden_states: Tensor, start: int, end: int) -> Tensor:
        for layer in stack.layers[start:end]:
            moe_layer = (
                layer.mlp if expert_bias_enabled and getattr(layer, "is_moe_layer", False) else None
            )
            if moe_layer is not None:
                if moe_layer._shared_prefix_token_multiplicities is not None:
                    raise RuntimeError("nested shared-prefix MoE dispatch is not supported")
                moe_layer._shared_prefix_token_multiplicities = token_multiplicities
            try:
                if isinstance(layer, MambaLayer):
                    # Every topology and root count honors NRL_SP_MAMBA_IMPL.
                    hidden_states = forward_mamba_layer_shared_prefix(layer, hidden_states, layout)
                elif isinstance(layer.self_attention, IdentityOp):
                    hidden_states = layer(hidden_states=hidden_states, attention_mask=None)
                else:
                    attention = layer.self_attention
                    if attention._shared_prefix_forest is not None:
                        raise RuntimeError(
                            "nested shared-prefix attention dispatch is not supported"
                        )
                    attention._shared_prefix_forest = layout.forest
                    try:
                        hidden_states = layer(
                            hidden_states=hidden_states,
                            attention_mask=None,
                            rotary_pos_emb=rotary_pos_emb,
                        )
                    finally:
                        del attention._shared_prefix_forest
            finally:
                if moe_layer is not None:
                    del moe_layer._shared_prefix_token_multiplicities

            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
        return hidden_states

    if stack.config.recompute_granularity == "full" and stack.training:
        chunk_size = stack.config.recompute_num_layers
        for start in range(0, len(stack.layers), chunk_size):
            end = min(start + chunk_size, len(stack.layers))

            def custom_forward(hidden_states: Tensor, start=start, end=end) -> Tensor:
                # Scopes live inside the callable so backward replay reconstructs them.
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
