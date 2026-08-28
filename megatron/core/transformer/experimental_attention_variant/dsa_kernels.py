# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Backend-neutral hooks for optional fused DeepSeek sparse attention kernels."""

from __future__ import annotations

import inspect
import logging
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Optional, Tuple

from torch import Tensor

from megatron.core.transformer.enums import AttnBackend, AttnMaskType

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
_HOOK_KWARG_SIGNATURE_CACHE: dict[int, tuple[object, Optional[frozenset[str]], bool]] = {}


def _get_dsa_kernel_backend(config: TransformerConfig) -> str:
    """Return the configured DSA kernel backend."""
    backend = config.dsa_kernel_backend
    if backend != "none" and backend not in _BACKEND_MODULE_NAME_BY_BACKEND:
        raise ValueError("dsa_kernel_backend must be one of: none, tilelang, cudnn")
    return backend


def _get_backend_module_name(config: TransformerConfig) -> Optional[str]:
    """Return the optional DSA backend module selected by config."""
    backend = _get_dsa_kernel_backend(config)
    if backend == "none":
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


def _hook_kwargs_accepting(fn, *, opaque_fallback_names: Tuple[str, ...] = (), **candidate_kwargs):
    """Keep only the kwargs ``fn`` can accept.

    Backend hooks are resolved dynamically and may live out of tree; passing a
    keyword an older backend does not know would fail with a TypeError at the
    first fused call rather than a clean decline. Opaque C/pybind hooks receive
    no newly introduced kwargs by default, but callers can name kwargs that
    predate this filtering so an unavailable signature does not break an
    existing hook contract.
    """
    cache_key = id(fn)
    cached = _HOOK_KWARG_SIGNATURE_CACHE.get(cache_key)
    if cached is not None and cached[0] is fn:
        accepted_names = cached[1]
        signature_is_opaque = cached[2]
    else:
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            # Opaque C/pybind hooks cannot safely receive a newly introduced keyword.
            accepted_names = frozenset()
            signature_is_opaque = True
        else:
            signature_is_opaque = False
            if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                accepted_names = None
            else:
                accepted_names = frozenset(
                    name
                    for name, parameter in sig.parameters.items()
                    if parameter.kind
                    in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
                )
        # Retain the callable so Python object-id reuse cannot apply a stale signature.
        _HOOK_KWARG_SIGNATURE_CACHE[cache_key] = (fn, accepted_names, signature_is_opaque)

    if accepted_names is None:
        return candidate_kwargs
    if signature_is_opaque:
        accepted_names = frozenset(opaque_fallback_names)
    return {k: v for k, v in candidate_kwargs.items() if k in accepted_names}


def _packed_layout_hook_kwargs(
    fn,
    *,
    varlen_is_plain_causal: bool,
    packed_thd_causal_identity_layout: bool,
    packed_thd_single_sequence: bool,
    opaque_fallback_names: Tuple[str, ...] = (),
):
    """Offer CP-independent layout facts only to hooks that declare support."""
    return _hook_kwargs_accepting(
        fn,
        opaque_fallback_names=opaque_fallback_names,
        varlen_is_plain_causal=varlen_is_plain_causal,
        packed_thd_causal_identity_layout=packed_thd_causal_identity_layout,
        packed_thd_single_sequence=packed_thd_single_sequence,
    )


def _legacy_packed_cp_flags(
    *, cp_size: int, use_local_indexer_varlen: bool, single_packed_thd_sequence: bool
) -> Tuple[bool, bool]:
    """Preserve the historical CP-only meaning of existing backend hook flags."""
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
    return _get_dsa_kernel_backend(config) != "none"


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

    The existing ``single_packed_thd_sequence`` and ``use_local_indexer_varlen``
    backend kwargs retain their historical CP-only meaning. CP-independent layout
    facts are offered as optional, signature-filtered kwargs so older out-of-tree
    hooks cannot silently enter a CP path for a CP=1 pack.
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

    The existing packed-layout kwargs retain their historical CP-only meaning;
    CP-independent facts use optional, signature-filtered kwargs.
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
) -> Optional[Tensor]:
    """Optional fused sparse-attention hook for backend-specific implementations."""
    fn = _resolve_fused_hook(config, "run_fused_absorbed_sparse_attention")
    if fn is None:
        return None
    result = fn(
        query,
        key,
        topk_indices,
        softmax_scale,
        v_channels,
        topk_length,
        **_hook_kwargs_accepting(fn, all_topk_rows_nonempty=all_topk_rows_nonempty),
    )
    if result is None:
        _log_declined_hook(config, "run_fused_absorbed_sparse_attention", "backend returned None")
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

    The existing packed-layout kwargs retain their historical CP-only meaning;
    CP-independent facts use optional, signature-filtered kwargs.
    """
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
            # ``varlen_is_plain_causal`` was already part of the full-hook
            # contract before optional kwarg filtering was introduced.
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
    "use_fused_dsa_kernels",
]
