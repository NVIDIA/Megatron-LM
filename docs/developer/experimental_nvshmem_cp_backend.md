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

1. Transformer Engine NVSHMEM workspace, synchronization, fused attention, and
   owner-gradient-return primitives.
2. FlashAttention 4 two-section causal backward with saved-state and direct
   owner dK/dV return.

The backend also requires `nvshmem4py-cu13`, `cuda-core`, CuPy, and an NVSHMEM
runtime visible to the dynamic linker. Deployments must set an explicit
`NVSHMEM_SYMMETRIC_SIZE` large enough for all per-layer K/V and gradient-return
workspaces. CUDA VMM policy is deployment-specific and is deliberately not set
by Megatron.

Do not enable the selector against unmodified upstream Transformer Engine or
FlashAttention builds. Missing extension symbols cause a fail-closed error.

## Existing Four-GPU Evidence

The research build was measured on four GB200 GPUs at TP=1, PP=1, DP=1, EP=1,
CP=4, sequence length 262144, hidden size 2560, 20 heads, head dimension 128,
four layers, vocabulary 157184, and global batch size 1.

| Metric | TE p2p | NVSHMEM | Change |
|---|---:|---:|---:|
| Median training step | 1831.75 ms | 1416.00 ms | 22.68% faster |
| Strict checkpoint comparison | reference | max abs 1.90735e-6 | 73 tensors passed |
| Allocated memory | reference | approximately +9 GiB | requires reduction |

The timing result is the median of four isolated counterbalanced pairs; each
pair ran eight updates and used updates 4-7 as its steady window. The observed
speedups ranged from 22.62% to 22.82%.

This is production-shape sequence evidence, not production-topology evidence.
It does not cover TP=1, PP=4, DP=8, EP=8, CP=4 on 128 GPUs. It was also produced
from the research dependency stack; the reduced review branch must be rebuilt
and rerun from clean dependency checkouts before the performance result can be
attached to an upstream PR.

## Review and Promotion Gates

1. Review and pin the Transformer Engine and FlashAttention dependency changes.
2. Build all three repositories from clean checkouts without path overrides.
3. Run tiny deterministic forward/backward equivalence tests.
4. Repeat the strict four-GPU long-sequence checkpoint and timing gate.
5. Measure rank skew, attention phases, and peak memory.
6. Qualify the full production topology when the required hardware is available.
