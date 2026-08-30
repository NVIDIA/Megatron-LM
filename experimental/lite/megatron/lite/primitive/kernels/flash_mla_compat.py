"""Adapt the pinned vLLM FlashMLA interface to mLite's DSA contract."""

from vllm.v1.attention.ops.flashmla import flash_mla_sparse_fwd as _sparse_fwd


def flash_mla_sparse_fwd(
    q,
    kv,
    indices,
    sm_scale,
    d_v=512,
    attn_sink=None,
    topk_length=None,
    out=None,
    indexer_topk=0,
):
    result = _sparse_fwd(
        q,
        kv,
        indices,
        sm_scale,
        d_v=d_v,
        attn_sink=attn_sink,
        topk_length=topk_length,
        out=out,
    )
    if not indexer_topk:
        return result
    if indexer_topk < indices.shape[-1]:
        raise RuntimeError(
            "vLLM FlashMLA can only synthesize lse_indexer when indexer_topk "
            "covers the complete sparse top-k"
        )
    output, max_logits, lse = result
    return output, max_logits, lse, lse.clone()


__all__ = ["flash_mla_sparse_fwd"]
