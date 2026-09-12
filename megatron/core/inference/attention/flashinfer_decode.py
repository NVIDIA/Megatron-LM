# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""Blackwell-native paged decode attention via flashinfer's trtllm-gen kernel.

Megatron's decode path runs FlashAttention-2's `flash_attn_with_kvcache`, whose kernels
predate Blackwell. A matched per-category comparison against vLLM at BS256/OSL1024
(GAP-S18) found attention to be the largest remaining device-time gap -- 0.409 ms/step at
an *identical* launch count, which rules out anything Megatron does around the call and
points at the kernel generation itself. vLLM reaches `fmhaSm100f` through flashinfer, and
flashinfer is already installed alongside flash-attn here, so the same kernel is
available to us: at our shapes it is 24% faster than FA2 with matching numerics.

Two properties make it usable at decode without restructuring anything:

  * It takes the paged layout Megatron already has. With ``kv_layout="NHD"`` the k and v
    caches are ``[num_pages, page_size, num_kv_heads, head_dim]``, exactly FA2's, and it
    accepts Megatron's 256-token pages (FA2 in fact *requires* pages be a multiple of
    256, so this is the stricter of the two).
  * It needs no host-side `plan()` call, unlike flashinfer's wrapper APIs, so it is
    capturable in the decode CUDA graph.

The one operational constraint is the scratch buffer, which must be zeroed before first
use. Zeroing it inside a graph capture would record a 128 MiB memset into the graph and
replay it every step, so the buffer is allocated eagerly and this path declines to engage
if it is asked to do so for the first time mid-capture.
"""

import logging
import os
import sys
from typing import Optional, Tuple

import torch

try:
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache

    HAVE_FLASHINFER = True
except ImportError:
    trtllm_batch_decode_with_kv_cache = None
    HAVE_FLASHINFER = False

logger = logging.getLogger(__name__)

ENABLED: bool = os.environ.get("MCORE_FLASHINFER_DECODE", "0") == "1"

if ENABLED and not HAVE_FLASHINFER:
    # Say so loudly once. Asking for this path and silently getting FA2 is the failure
    # this module is easiest to get wrong: flashinfer is commonly present in the run
    # venv and absent from the container's system python, so the same command can
    # engage or not depending only on which interpreter launched it.
    logger.warning(
        "MCORE_FLASHINFER_DECODE=1 but flashinfer is not importable under %s; "
        "decode will use the flash-attention path instead",
        sys.executable,
    )

# Programmatic Dependent Launch lets this kernel's prologue start while its predecessor
# drains. 855 of mcore's 934 per-step GPU gaps are sub-microsecond launch overhead
# (GAP-S18), so it is worth an arm; left as a knob because PDL interacts with graph
# capture and its benefit is hardware- and neighbour-dependent. None = flashinfer decides.
_PDL_ENV = os.environ.get("MCORE_FLASHINFER_PDL", "")
ENABLE_PDL: Optional[bool] = bool(int(_PDL_ENV)) if _PDL_ENV else None

# vLLM sizes this at 128 MiB for the same kernel family.
_WORKSPACE_BYTES: int = 128 * 1024 * 1024

_workspace: Optional[torch.Tensor] = None
_declined: bool = False


def _get_workspace(device: torch.device) -> Optional[torch.Tensor]:
    """The zeroed scratch buffer, or None if it cannot be created safely right now."""
    global _workspace, _declined
    if _workspace is not None and _workspace.device == device:
        return _workspace
    if torch.cuda.is_current_stream_capturing():
        # Allocating here would bind the buffer to the graph's pool and, worse, record
        # the 128 MiB zero-fill as a graph node replayed every step. Decline instead;
        # the caller falls back to FA2 for this graph.
        if not _declined:
            _declined = True
            logger.warning(
                "flashinfer decode declined: no workspace yet and stream is capturing; "
                "falling back to FA2 for this graph"
            )
        return None
    _workspace = torch.zeros(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    return _workspace


def can_use(
    query: torch.Tensor,
    block_table: Optional[torch.Tensor],
    need_lse: bool,
    window_size: Tuple[int, int],
    tokens_per_request: int,
) -> bool:
    """Whether the trtllm-gen decode kernel can serve this call.

    Deliberately narrow: every condition below is one this module has not verified
    against FA2, and a wrong answer here is a silent accuracy bug rather than a crash.
    """
    if not (ENABLED and HAVE_FLASHINFER):
        return False
    if block_table is None:  # non-paged KV cache
        return False
    if need_lse:  # attention sinks need an LSE correction pass; not wired up
        return False
    if window_size != (-1, -1):  # sliding window maps to window_left; untested
        return False
    if tokens_per_request != 1:  # speculative decoding needs q_len_per_req plumbing
        return False
    if query.dtype not in (torch.bfloat16, torch.float16):
        return False
    return _get_workspace(query.device) is not None


def decode(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seqlens_k: torch.Tensor,
    max_seqlen_k: int,
    softmax_scale: float,
) -> torch.Tensor:
    """Paged decode attention. ``query`` is ``(B, 1, H, D)``; returns the same shape."""
    num_requests, tokens_per_request = query.shape[0], query.shape[1]
    out = trtllm_batch_decode_with_kv_cache(
        query=query.reshape(-1, query.shape[-2], query.shape[-1]),
        kv_cache=(k_cache, v_cache),
        workspace_buffer=_workspace,
        block_tables=block_table,
        seq_lens=seqlens_k,
        max_seq_len=max_seqlen_k,
        # bmm2 is the P@V product, which needs no rescaling for a bf16 cache.
        bmm1_scale=softmax_scale,
        bmm2_scale=1.0,
        kv_layout="NHD",
        enable_pdl=ENABLE_PDL,
    )
    return out.reshape(num_requests, tokens_per_request, *out.shape[1:])
