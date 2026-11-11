# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Backend-neutral hooks for optional fused DeepSeek sparse attention kernels."""

from __future__ import annotations

import logging
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
from torch import Tensor

from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.utils import _UNFUSED_DSA_KERNEL_BACKENDS, filter_kwargs_for_callable

if TYPE_CHECKING:
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.transformer_config import TransformerConfig

_BACKEND_MODULE_NAME_BY_BACKEND = {
    "tilelang": "megatron.core.transformer.experimental_attention_variant.dsa_tilelang_kernels",
    "cudnn": "megatron.core.transformer.experimental_attention_variant.dsa_cudnn_kernels",
}
_BACKEND: Optional[ModuleType] = None
_BACKEND_SELECTION: Optional[str] = None
_LOGGER = logging.getLogger(__name__)


def _get_dsa_kernel_backend(config: TransformerConfig) -> str:
    """Return the configured DSA kernel backend."""
    backend = config.dsa_kernel_backend
    # The DSA-over-GQA values select implementations that load no fused backend module, so
    # every helper in this module treats them like "none"; TransformerConfig validates the
    # value itself.
    if backend not in _UNFUSED_DSA_KERNEL_BACKENDS and backend not in (
        _BACKEND_MODULE_NAME_BY_BACKEND
    ):
        raise ValueError(
            "dsa_kernel_backend must be one of: "
            f"{', '.join(_UNFUSED_DSA_KERNEL_BACKENDS + tuple(_BACKEND_MODULE_NAME_BY_BACKEND))}."
        )
    return backend


def _get_backend_module_name(config: TransformerConfig) -> Optional[str]:
    """Return the optional DSA backend module selected by config."""
    backend = _get_dsa_kernel_backend(config)
    if backend in _UNFUSED_DSA_KERNEL_BACKENDS:
        return None
    return _BACKEND_MODULE_NAME_BY_BACKEND[backend]


def _load_backend(config: TransformerConfig) -> Optional[ModuleType]:
    """Import the configured optional DSA kernel backend."""
    global _BACKEND, _BACKEND_SELECTION
    module_name = _get_backend_module_name(config)
    if module_name is None:
        _BACKEND = None
        _BACKEND_SELECTION = None
        return None
    if _BACKEND is not None and _BACKEND_SELECTION == module_name:
        return _BACKEND

    try:
        _BACKEND = import_module(module_name)
    except (ImportError, OSError) as exc:
        raise RuntimeError(f"Failed to import DSA kernel backend {module_name}.") from exc
    _BACKEND_SELECTION = module_name
    return _BACKEND


def _log_declined_hook(config: TransformerConfig, hook_name: str, reason: str) -> None:
    """Emit one debug breadcrumb when a selected fused hook falls back."""
    if not _LOGGER.isEnabledFor(logging.DEBUG):
        return
    _LOGGER.debug(
        "DSA fused backend %s %s declined; falling back (%s).",
        _get_dsa_kernel_backend(config),
        hook_name,
        reason,
    )


def _packed_layout_hook_kwargs(
    fn,
    *,
    varlen_is_plain_causal: bool,
    packed_thd_causal_identity_layout: bool,
    packed_thd_single_sequence: bool,
    opaque_fallback_names: Tuple[str, ...] = (),
):
    """Offer CP-independent layout facts only to hooks that declare support."""
    return filter_kwargs_for_callable(
        fn,
        {
            "varlen_is_plain_causal": varlen_is_plain_causal,
            "packed_thd_causal_identity_layout": packed_thd_causal_identity_layout,
            "packed_thd_single_sequence": packed_thd_single_sequence,
        },
        signature_unavailable_fallback=opaque_fallback_names,
    )


def _legacy_packed_cp_flags(
    *, cp_size: int, use_local_indexer_varlen: bool, single_packed_thd_sequence: bool
) -> Tuple[bool, bool]:
    """Return packed-layout backend hook flags with their CP-only contract."""
    return (use_local_indexer_varlen and cp_size > 1, single_packed_thd_sequence and cp_size > 1)


def _resolve_fused_hook(config: TransformerConfig, hook_name: str):
    """Return the selected backend's ``hook_name`` callable, or None if unavailable.

    Returns None when no backend is configured, or (logging a decline breadcrumb) when
    the configured backend does not implement the hook.
    """
    backend = _load_backend(config)
    if backend is None:
        return None
    fn = getattr(backend, hook_name, None)
    if fn is None:
        _log_declined_hook(config, hook_name, "hook is not implemented")
    return fn


def use_fused_dsa_kernels(config: TransformerConfig) -> bool:
    """Return whether DSA should attempt optional fused kernels before falling back."""
    backend = config.attention_backend
    if backend == AttnBackend.unfused or backend == "unfused":
        return False
    return _get_dsa_kernel_backend(config) not in _UNFUSED_DSA_KERNEL_BACKENDS


def _deterministic_dsa_requested(config: TransformerConfig) -> bool:
    """Return whether DSA must use bitwise-repeatable implementations."""
    return bool(
        getattr(config, "deterministic_mode", False) or torch.are_deterministic_algorithms_enabled()
    )


def run_fused_qk_topk(
    config: TransformerConfig,
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    index_topk: int,
    starts: Tensor,
    ends: Tensor,
    block_size: int,
    use_relu: bool = True,
    use_local_indexer_varlen: bool = False,
    single_packed_thd_sequence: bool = False,
    local_packed_cp_rank: int = 0,
    local_packed_cp_query_start: int = 0,
    local_packed_cp_query_len: Optional[int] = None,
    packed_seq_params: Optional[PackedSeqParams] = None,
    cp_size: int = 1,
    varlen_is_plain_causal: bool = False,
) -> Optional[Tuple[Tensor, Optional[Tensor]]]:
    """Optional fused indexer hook for backend-specific implementations.

    ``single_packed_thd_sequence`` and ``use_local_indexer_varlen`` are CP-only
    backend kwargs. CP-independent layout facts are offered as optional,
    signature-filtered kwargs.
    """
    fn = _resolve_fused_hook(config, "run_fused_qk_topk")
    if fn is None:
        return None
    legacy_local_varlen, legacy_single_sequence = _legacy_packed_cp_flags(
        cp_size=cp_size,
        use_local_indexer_varlen=use_local_indexer_varlen,
        single_packed_thd_sequence=single_packed_thd_sequence,
    )
    result = fn(
        q=q,
        k=k,
        weights=weights,
        index_topk=index_topk,
        starts=starts,
        ends=ends,
        block_size=block_size,
        use_relu=use_relu,
        use_local_indexer_varlen=legacy_local_varlen,
        single_packed_thd_sequence=legacy_single_sequence,
        local_packed_cp_rank=local_packed_cp_rank,
        local_packed_cp_query_start=local_packed_cp_query_start,
        local_packed_cp_query_len=local_packed_cp_query_len,
        packed_seq_params=packed_seq_params,
        cp_size=cp_size,
        **filter_kwargs_for_callable(
            fn, {"deterministic": _deterministic_dsa_requested(config), "config": config}
        ),
        **_packed_layout_hook_kwargs(
            fn,
            varlen_is_plain_causal=varlen_is_plain_causal,
            packed_thd_causal_identity_layout=use_local_indexer_varlen,
            packed_thd_single_sequence=single_packed_thd_sequence,
        ),
    )
    if result is None:
        _log_declined_hook(config, "run_fused_qk_topk", "backend returned None")
    return result


def run_kpool_pool_topk(
    config: TransformerConfig,
    q: Tensor,
    k_pooled: Tensor,
    weights: Tensor,
    pool_topk: int,
    pool_size: int,
    query_runs: Optional[Tuple[Tuple[int, int, int], ...]],
    hf_scoring: bool,
    use_relu: bool = True,
    return_scores: bool = False,
) -> Optional[Union[Tensor, Tuple[Tensor, Optional[Tensor]]]]:
    """Optional fused KPool pool selection (``dsa_indexer_kpool_scorer_backend='cudnn'``).

    Returns ``[b, sq, pool_topk]`` int64 pool ids with ``-1`` for unused slots, or None when
    the backend has no such hook or declines the call; the caller then keeps the PyTorch
    KPool scorer. See ``dsa_cudnn_kernels.run_kpool_pool_topk`` for the contract.

    With ``return_scores=True`` a backend result is ``(pool_ids, pool_scores)``, the FP32
    selector score of each selected pool aligned with its id. A hook that takes no
    ``return_scores`` argument is called as without it, and its ids come back with ``None``
    scores.
    """
    fn = _resolve_fused_hook(config, "run_kpool_pool_topk")
    if fn is None:
        return None
    optional_kwargs = {"deterministic": _deterministic_dsa_requested(config), "config": config}
    if return_scores:
        # Offered only when requested, so the frozen selector's call is unchanged.
        optional_kwargs["return_scores"] = True
    hook_kwargs = filter_kwargs_for_callable(fn, optional_kwargs)
    result = fn(
        q=q,
        k_pooled=k_pooled,
        weights=weights,
        pool_topk=pool_topk,
        pool_size=pool_size,
        query_runs=query_runs,
        hf_scoring=hf_scoring,
        use_relu=use_relu,
        **hook_kwargs,
    )
    if result is None:
        _log_declined_hook(config, "run_kpool_pool_topk", "backend returned None")
        return None
    if return_scores and "return_scores" not in hook_kwargs:
        return result, None
    return result


def kpool_sparse_indexer_loss_decline_reason(
    config: TransformerConfig, query: Tensor, key: Tensor
) -> Optional[str]:
    """Why the backend cannot run the trainable KPool loss of this forward, or None if it can.

    The backend's ``kpool_sparse_indexer_loss_decline_reason`` hook decides from host facts
    (see ``dsa_cudnn_kernels``); a backend without the hook cannot run it.
    """
    fn = _resolve_fused_hook(config, "kpool_sparse_indexer_loss_decline_reason")
    if fn is None:
        return (
            f"DSA kernel backend {_get_dsa_kernel_backend(config)!r} has no fused KPool "
            "indexer loss"
        )
    return fn(
        query=query,
        key=key,
        v_channels=int(getattr(config, "kv_lora_rank", 0) or 0),
        **filter_kwargs_for_callable(fn, {"deterministic": _deterministic_dsa_requested(config)}),
    )


def run_kpool_attention_token_target(
    config: TransformerConfig,
    query: Tensor,
    key: Tensor,
    lse: Tensor,
    token_topk: Tensor,
    softmax_scale: float,
) -> Tensor:
    """Main-attention mass of each attended KPool token slot, from the backend's kernel.

    Returns FP32 ``[b, sq, width]``: per row, the head-summed attention probability of each
    slot of ``token_topk`` divided by the local head count, zero at ``-1`` slots. ``lse`` is
    the fused sparse attention's own LSE over those ids. See
    ``dsa_cudnn_kernels.run_kpool_attention_token_target``. The caller chose the fused KPool
    loss for this forward, so a missing hook or a declined call raises.
    """
    fn = _resolve_fused_hook(config, "run_kpool_attention_token_target")
    if fn is None:
        raise RuntimeError(
            f"DSA kernel backend {_get_dsa_kernel_backend(config)!r} has no "
            "run_kpool_attention_token_target hook for the fused KPool indexer loss."
        )
    result = fn(query=query, key=key, lse=lse, token_topk=token_topk, softmax_scale=softmax_scale)
    if result is None:
        raise RuntimeError(
            f"DSA kernel backend {_get_dsa_kernel_backend(config)!r} declined the KPool "
            "attention target of a forward that chose the fused KPool indexer loss."
        )
    return result


def run_kpool_sparse_indexer_loss(
    config: TransformerConfig,
    q: Tensor,
    weights: Tensor,
    k_pooled: Tensor,
    pool_ids: Tensor,
    pool_scores: Tensor,
    target: Tensor,
    row_mask: Tensor,
    row_count: Union[float, Tensor],
    loss_coeff: float,
    hf_scoring: bool,
    use_relu: bool = True,
) -> Tensor:
    """The trainable KPool indexer loss on the backend's kernels, graph-connected to ``q``,
    ``weights`` and ``k_pooled``.

    See ``dsa_cudnn_kernels.run_kpool_sparse_indexer_loss``. The caller chose the fused KPool
    loss for this forward, so a missing hook or a declined call raises.
    """
    fn = _resolve_fused_hook(config, "run_kpool_sparse_indexer_loss")
    if fn is None:
        raise RuntimeError(
            f"DSA kernel backend {_get_dsa_kernel_backend(config)!r} has no "
            "run_kpool_sparse_indexer_loss hook for the fused KPool indexer loss."
        )
    result = fn(
        q=q,
        weights=weights,
        k_pooled=k_pooled,
        pool_ids=pool_ids,
        pool_scores=pool_scores,
        target=target,
        row_mask=row_mask,
        row_count=row_count,
        loss_coeff=loss_coeff,
        hf_scoring=hf_scoring,
        use_relu=use_relu,
        **filter_kwargs_for_callable(fn, {"deterministic": _deterministic_dsa_requested(config)}),
    )
    if result is None:
        raise RuntimeError(
            f"DSA kernel backend {_get_dsa_kernel_backend(config)!r} declined the KPool "
            "indexer loss of a forward that chose the fused KPool indexer loss."
        )
    return result


def run_fused_qk_topk_with_scores(
    config: TransformerConfig,
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    index_topk: int,
    starts: Tensor,
    ends: Tensor,
    block_size: int,
    use_relu: bool = True,
    use_local_indexer_varlen: bool = False,
    single_packed_thd_sequence: bool = False,
    local_packed_cp_rank: int = 0,
    local_packed_cp_query_start: int = 0,
    local_packed_cp_query_len: Optional[int] = None,
    packed_seq_params: Optional[PackedSeqParams] = None,
    cp_size: int = 1,
    varlen_is_plain_causal: bool = False,
    query: Optional[Tensor] = None,
    key: Optional[Tensor] = None,
) -> Optional[Tuple[Tensor, Tensor, Tensor]]:
    """Optional fused indexer hook that also returns the selected indexer scores.

    Returns the same index-sorted ``(topk_indices, topk_length)`` as ``run_fused_qk_topk``
    plus fp32 ``topk_scores`` aligned slot-for-slot with the indices, for
    ``run_sparse_indexer_loss_from_topk`` to consume in a later checkpoint replay. The
    attention ``query``/``key`` let the backend decline layouts that replay cannot consume.
    """
    fn = _resolve_fused_hook(config, "run_fused_qk_topk_with_scores")
    if fn is None:
        return None
    legacy_local_varlen, legacy_single_sequence = _legacy_packed_cp_flags(
        cp_size=cp_size,
        use_local_indexer_varlen=use_local_indexer_varlen,
        single_packed_thd_sequence=single_packed_thd_sequence,
    )
    result = fn(
        q=q,
        k=k,
        weights=weights,
        index_topk=index_topk,
        starts=starts,
        ends=ends,
        block_size=block_size,
        use_relu=use_relu,
        use_local_indexer_varlen=legacy_local_varlen,
        single_packed_thd_sequence=legacy_single_sequence,
        local_packed_cp_rank=local_packed_cp_rank,
        local_packed_cp_query_start=local_packed_cp_query_start,
        local_packed_cp_query_len=local_packed_cp_query_len,
        packed_seq_params=packed_seq_params,
        cp_size=cp_size,
        **filter_kwargs_for_callable(
            fn,
            {
                "deterministic": _deterministic_dsa_requested(config),
                "query": query,
                "key": key,
                "config": config,
            },
        ),
        **_packed_layout_hook_kwargs(
            fn,
            varlen_is_plain_causal=varlen_is_plain_causal,
            packed_thd_causal_identity_layout=use_local_indexer_varlen,
            packed_thd_single_sequence=single_packed_thd_sequence,
        ),
    )
    if result is None:
        _log_declined_hook(config, "run_fused_qk_topk_with_scores", "backend returned None")
    return result


def run_sparse_indexer_loss_from_topk(
    config: TransformerConfig,
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    topk_indices: Tensor,
    topk_length: Tensor,
    topk_scores: Tensor,
    query: Tensor,
    key: Tensor,
    softmax_scale: float,
    loss_coeff: float,
    pg_collection: ProcessGroupCollection,
    query_valid_rows: Optional[Tensor] = None,
    calculate_per_token_loss: bool = False,
) -> Optional[Tensor]:
    """Optional sparse indexer-loss hook for an already selected top-k set.

    Consumes the output of ``run_fused_qk_topk_with_scores`` instead of rerunning the
    dense indexer and top-k. Returns a graph-connected loss whose backward produces the
    indexer ``q``/``k``/``weights`` gradients, or None when the backend declines.
    """
    fn = _resolve_fused_hook(config, "run_sparse_indexer_loss_from_topk")
    if fn is None:
        return None
    result = fn(
        config=config,
        q=q,
        k=k,
        weights=weights,
        topk_indices=topk_indices,
        topk_length=topk_length,
        topk_scores=topk_scores,
        query=query,
        key=key,
        softmax_scale=softmax_scale,
        loss_coeff=loss_coeff,
        pg_collection=pg_collection,
        query_valid_rows=query_valid_rows,
        calculate_per_token_loss=calculate_per_token_loss,
        **filter_kwargs_for_callable(fn, {"deterministic": _deterministic_dsa_requested(config)}),
    )
    if result is None:
        _log_declined_hook(config, "run_sparse_indexer_loss_from_topk", "backend returned None")
    return result


def run_fused_qk_topk_with_loss(
    config: TransformerConfig,
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    index_topk: int,
    starts: Tensor,
    ends: Tensor,
    block_size: int,
    query: Tensor,
    key: Tensor,
    softmax_scale: float,
    loss_coeff: float,
    pg_collection: ProcessGroupCollection,
    query_valid_rows: Optional[Tensor] = None,
    calculate_per_token_loss: bool = False,
    use_relu: bool = True,
    use_local_indexer_varlen: bool = False,
    single_packed_thd_sequence: bool = False,
    local_packed_cp_rank: int = 0,
    local_packed_cp_query_start: int = 0,
    local_packed_cp_query_len: Optional[int] = None,
    packed_seq_params: Optional[PackedSeqParams] = None,
    cp_size: int = 1,
    varlen_is_plain_causal: bool = False,
) -> Optional[Tuple[Tensor, Optional[Tensor], Tensor]]:
    """Optional fused indexer+loss hook for backend-specific implementations.

    Packed-layout backend kwargs are CP-only; CP-independent facts use optional,
    signature-filtered kwargs.
    """
    fn = _resolve_fused_hook(config, "run_fused_qk_topk_with_loss")
    if fn is None:
        return None
    legacy_local_varlen, legacy_single_sequence = _legacy_packed_cp_flags(
        cp_size=cp_size,
        use_local_indexer_varlen=use_local_indexer_varlen,
        single_packed_thd_sequence=single_packed_thd_sequence,
    )
    result = fn(
        config=config,
        q=q,
        k=k,
        weights=weights,
        index_topk=index_topk,
        starts=starts,
        ends=ends,
        block_size=block_size,
        query=query,
        key=key,
        softmax_scale=softmax_scale,
        loss_coeff=loss_coeff,
        pg_collection=pg_collection,
        query_valid_rows=query_valid_rows,
        calculate_per_token_loss=calculate_per_token_loss,
        use_relu=use_relu,
        use_local_indexer_varlen=legacy_local_varlen,
        single_packed_thd_sequence=legacy_single_sequence,
        local_packed_cp_rank=local_packed_cp_rank,
        local_packed_cp_query_start=local_packed_cp_query_start,
        local_packed_cp_query_len=local_packed_cp_query_len,
        packed_seq_params=packed_seq_params,
        cp_size=cp_size,
        **filter_kwargs_for_callable(fn, {"deterministic": _deterministic_dsa_requested(config)}),
        **_packed_layout_hook_kwargs(
            fn,
            varlen_is_plain_causal=varlen_is_plain_causal,
            packed_thd_causal_identity_layout=use_local_indexer_varlen,
            packed_thd_single_sequence=single_packed_thd_sequence,
        ),
    )
    if result is None:
        _log_declined_hook(config, "run_fused_qk_topk_with_loss", "backend returned None")
    return result


def run_fused_absorbed_sparse_attention(
    config: TransformerConfig,
    query: Tensor,
    key: Tensor,
    topk_indices: Tensor,
    softmax_scale: float,
    v_channels: int,
    topk_length: Optional[Tensor] = None,
    all_topk_rows_nonempty: bool = False,
    return_lse: bool = False,
) -> Optional[Union[Tensor, Tuple[Tensor, Tensor]]]:
    """Optional fused sparse-attention hook for backend-specific implementations.

    Returns the backend's latent output, or None when no backend is selected or it declines.
    With ``return_lse=True`` a backend result is ``(output, lse)``, where ``lse`` is the
    non-differentiable FP32 ``[sq, b, np]`` natural-log attention LSE described in
    ``dsa_cudnn_kernels.run_fused_absorbed_sparse_attention``. A selected backend whose hook
    does not declare ``return_lse`` raises instead of running, and so does a hook that then
    returns no LSE.
    """
    fn = _resolve_fused_hook(config, "run_fused_absorbed_sparse_attention")
    if fn is None:
        return None
    optional_kwargs = {
        "all_topk_rows_nonempty": all_topk_rows_nonempty,
        "deterministic": _deterministic_dsa_requested(config),
    }
    if return_lse:
        # Offered only when requested, so hooks without the argument keep their old call.
        optional_kwargs["return_lse"] = True
    hook_kwargs = filter_kwargs_for_callable(fn, optional_kwargs)
    if return_lse and "return_lse" not in hook_kwargs:
        raise RuntimeError(
            f"DSA kernel backend {_get_dsa_kernel_backend(config)!r} cannot return the "
            "sparse-attention LSE: its run_fused_absorbed_sparse_attention hook takes no "
            "return_lse argument."
        )
    result = fn(query, key, topk_indices, softmax_scale, v_channels, topk_length, **hook_kwargs)
    if result is None:
        _log_declined_hook(config, "run_fused_absorbed_sparse_attention", "backend returned None")
        return None
    if return_lse and not (
        isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], Tensor)
    ):
        raise RuntimeError(
            f"DSA kernel backend {_get_dsa_kernel_backend(config)!r} "
            "run_fused_absorbed_sparse_attention was asked for the LSE (return_lse=True) but "
            "did not return an (output, lse) pair."
        )
    return result


def run_fused_dsa_attention(
    *,
    config: TransformerConfig,
    query: Tensor,
    key: Tensor,
    value: Optional[Tensor],
    up_v_weight: Optional[Tensor],
    q_indexer: Tensor,
    k_indexer: Tensor,
    indexer_weights: Tensor,
    indexer_topk: int,
    softmax_scale: float,
    loss_coeff: float,
    sparse_loss: bool,
    calculate_per_token_loss: bool,
    absorbed_mla: bool,
    cp_size: int,
    attn_mask_type: Optional[AttnMaskType],
    packed_seq_params: Optional[PackedSeqParams],
    varlen_starts: Optional[Tensor],
    varlen_ends: Optional[Tensor],
    key_positions: Optional[Tensor],
    query_valid_rows: Optional[Tensor],
    varlen_is_plain_causal: bool = False,
    use_relu: bool,
    use_local_indexer_varlen: bool = False,
    single_packed_thd_sequence: bool = False,
    local_packed_cp_rank: int = 0,
    local_packed_cp_query_start: int = 0,
    local_packed_cp_query_len: Optional[int] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> Optional[Tuple[Tensor, Tensor]]:
    """Optional full fused DSA hook for backends that fuse indexer and attention together.

    Packed-layout backend kwargs are CP-only; CP-independent facts use optional,
    signature-filtered kwargs.
    """
    if _deterministic_dsa_requested(config):
        # The combined autograd function precomputes indexer gradients and runs sparse-attention
        # backward internally. Keep deterministic mode on the split route so each backward can
        # independently select a bitwise-repeatable implementation.
        _log_declined_hook(config, "run_fused_dsa_attention", "deterministic mode")
        return None
    fn = _resolve_fused_hook(config, "run_fused_dsa_attention")
    if fn is None:
        return None
    legacy_local_varlen, legacy_single_sequence = _legacy_packed_cp_flags(
        cp_size=cp_size,
        use_local_indexer_varlen=use_local_indexer_varlen,
        single_packed_thd_sequence=single_packed_thd_sequence,
    )
    result = fn(
        config=config,
        query=query,
        key=key,
        value=value,
        up_v_weight=up_v_weight,
        q_indexer=q_indexer,
        k_indexer=k_indexer,
        indexer_weights=indexer_weights,
        indexer_topk=indexer_topk,
        softmax_scale=softmax_scale,
        loss_coeff=loss_coeff,
        sparse_loss=sparse_loss,
        calculate_per_token_loss=calculate_per_token_loss,
        absorbed_mla=absorbed_mla,
        cp_size=cp_size,
        attn_mask_type=attn_mask_type,
        packed_seq_params=packed_seq_params,
        varlen_starts=varlen_starts,
        varlen_ends=varlen_ends,
        key_positions=key_positions,
        query_valid_rows=query_valid_rows,
        **_packed_layout_hook_kwargs(
            fn,
            varlen_is_plain_causal=varlen_is_plain_causal,
            packed_thd_causal_identity_layout=use_local_indexer_varlen,
            packed_thd_single_sequence=single_packed_thd_sequence,
            # The full-hook contract includes this kwarg even for opaque callables.
            opaque_fallback_names=("varlen_is_plain_causal",),
        ),
        use_relu=use_relu,
        use_local_indexer_varlen=legacy_local_varlen,
        single_packed_thd_sequence=legacy_single_sequence,
        local_packed_cp_rank=local_packed_cp_rank,
        local_packed_cp_query_start=local_packed_cp_query_start,
        local_packed_cp_query_len=local_packed_cp_query_len,
        pg_collection=pg_collection,
    )
    if result is None:
        _log_declined_hook(config, "run_fused_dsa_attention", "backend returned None")
    return result


__all__ = [
    "run_fused_absorbed_sparse_attention",
    "run_fused_dsa_attention",
    "run_fused_qk_topk",
    "run_fused_qk_topk_with_loss",
    "run_fused_qk_topk_with_scores",
    "run_sparse_indexer_loss_from_topk",
    "use_fused_dsa_kernels",
]
