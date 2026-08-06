# Experimental NVSHMEM Context-Parallel Attention

Megatron can select an opt-in NVSHMEM context-parallel attention backend with:

```text
--context-parallel-attention-backend nvshmem
```

The default remains unchanged. The experimental backend publishes K/V into
NVSHMEM symmetric workspaces, consumes peer K/V from those workspaces, and
returns remote K/V gradients to their owners. Workspaces are allocated
collectively before pipeline execution so that allocation order cannot diverge
between ranks.

## Supported Contract

The implementation fails closed outside the configuration that has been
qualified:

- context parallel size 4;
- tensor parallel size 1;
- `cp_comm_type=p2p` as the comparison contract;
- multi-head attention with head dimension 128;
- causal self-attention training;
- one in-flight microbatch;
- BF16 Q/K/V for the selected fused path.

Inference, cross-attention, packed sequences, rotary embeddings at this
boundary, attention output gates, activation offload, and overlapping pipeline
microbatches are not supported yet.

## Dependency Stack

This Megatron integration depends on two experimental lower-level changes that
are not part of this patch:

1. one Transformer Engine CP=4 global dK/dV owner-return primitive with a
   stream-ordered epoch protocol;
2. FlashAttention 4 two-section causal backward for two query sections that
   share one K/V prefix.

Megatron owns symmetric allocation, peer mappings, K/V publication, and backend
selection. FlashAttention owns attention math, and Transformer Engine only
performs the final symmetric owner-gradient return.

The backend also requires `nvshmem4py-cu13`, `cuda-core`, CuPy, and an NVSHMEM
runtime visible to the dynamic linker. Deployments must set an explicit
`NVSHMEM_SYMMETRIC_SIZE` large enough for all per-layer K/V and gradient-return
workspaces. CUDA VMM policy is deployment-specific and is deliberately not set
by Megatron.

Do not enable the selector against unmodified upstream Transformer Engine or
FlashAttention builds. Missing extension symbols cause a fail-closed error.

## Four-GPU Evidence

The backend was measured on four GB200 GPUs at TP=1, PP=1, DP=1, EP=1, CP=4,
sequence length 262144, hidden size 2560, 20 heads, head dimension 128, four
layers, vocabulary 157184, and global batch size 1.

The initial research stack measured 1831.75 ms for TE p2p and 1416.00 ms for
the candidate, or 22.68% lower iteration time over four counterbalanced pairs.
That result motivated the dependency reduction, but it is not the clean-branch
scoreboard.

The reduced Megatron, Transformer Engine, and FlashAttention branches were then
built from clean checkouts and rerun as one baseline/candidate pair:

| Metric | TE p2p | NVSHMEM | Change |
|---|---:|---:|---:|
| Median training step, iterations 4-7 | 1707.45 ms | 1413.65 ms | 17.21% lower |
| Strict checkpoint comparison | reference | max abs 1.90735e-6 | 73/73 tensors passed |
| Steady allocated memory | 23885.75 MiB | 33501.38 MiB | +9615.63 MiB |
| Peak allocated memory | 104005.82 MiB | 112986.44 MiB | +8980.62 MiB |

Both runs used eight optimizer updates, mock data, selective activation
recompute, BF16, and identical seeds and optimizer settings. The clean result
establishes training correctness and an integrated speedup, but a single pair
is not sufficient for a final performance claim.

This is production-shape sequence evidence, not production-topology evidence.
It does not cover TP=1, PP=4, DP=8, EP=8, CP=4 on 128 GPUs.

## Review and Promotion Gates

Completed review-branch gates:

1. Reduced and pinned Transformer Engine and FlashAttention dependency branches.
2. Built all three repositories from clean checkouts into isolated wheels/runtime trees.
3. Passed a tiny deterministic two-update checkpoint comparison: 22/22 tensors,
   worst max-abs 9.20e-8.
4. Passed the strict four-GPU long-sequence gate reported above.

Remaining promotion gates:

1. Repeat the clean long-sequence pair in counterbalanced order.
2. Reduce the approximately 9 GiB memory premium.
3. Measure attention phases and rank skew without perturbing the clean scoreboard.
4. Qualify the full production topology when the required hardware is available.
