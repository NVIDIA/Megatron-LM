<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Mamba2 State-Passing Context Parallelism Benchmark

Compares the Mamba2 context-parallel paths against each other so the
`--mamba-state-passing-cp-load-balancing` mode can be chosen from measurements
rather than by guessing. See
[the feature guide](../../docs/user-guide/features/mamba_state_passing_context_parallel.md)
for what the paths do.

## What is measured

Only the post-projection part of `MambaMixer` — the causal convolution and the
SSD scan — is timed. That is where the CP paths differ; `in_proj`, the output
projection, and RMSNorm are identical across paths and would dilute the
comparison, so they are excluded.

Four paths are measured:

| Path | Description |
|---|---|
| `a2a` | The existing Mamba CP path: activations are redistributed so every rank holds the whole sequence for a subset of heads and groups |
| `permute-p2p` | State passing; the balanced shard is exchanged for a contiguous causal shard point-to-point |
| `permute-a2a` | State passing; the same exchange in one unequal-split `all_to_all_single` |
| `virtual` | State passing with no activation exchange; each balanced half is treated as an independent causal segment |

## Running it

One process per GPU; the world size is the CP size. No tensor-parallel group is
created — `--tp-size` only divides the head and group counts so that a TP-local
shard shape can be measured with one rank per CP position.

```bash
# CP=4, 32K sequence
PYTHONPATH=. torchrun --standalone --nproc_per_node=4 \
  examples/mamba_state_passing_context_parallel/benchmark_mamba_state_passing_cp.py \
  --L 32768 --iters 20

# Sweep sequence lengths and batch sizes in one run
PYTHONPATH=. torchrun --standalone --nproc_per_node=8 \
  examples/mamba_state_passing_context_parallel/benchmark_mamba_state_passing_cp.py \
  --sequence-lengths 32768 131072 --batch-sizes 1 3

# Forward only, and just the two paths worth comparing for a decision
PYTHONPATH=. torchrun --standalone --nproc_per_node=4 \
  examples/mamba_state_passing_context_parallel/benchmark_mamba_state_passing_cp.py \
  --forward-only --paths a2a virtual

# TP2-local shard shape under CP4
PYTHONPATH=. torchrun --standalone --nproc_per_node=4 \
  examples/mamba_state_passing_context_parallel/benchmark_mamba_state_passing_cp.py \
  --tp-size 2 --L 32768 --batch 3
```

Defaults are Nemotron-3 Nano's mixer shape: 64 heads of 64, 8 groups, state 128,
SSD chunk 128, `d_conv` 4, BF16 activations.

## Timing method

Each iteration is timed with CUDA events and reduced across ranks with `MAX`,
because a CP path is only as fast as its slowest rank. Rank alignment happens on
a barrier outside the timed interval, so waiting for stragglers is not counted
twice. Warmup iterations are discarded, and peak allocated memory is reported
alongside latency.

Output is one human-readable line per path plus a machine-readable `RESULT,...`
line per path for scripted collection:

```text
Mamba CP paths (fwd+bwd) L=32768 batch=1 local_L=8192 cp=4 ...
               a2a:    12.34 ms +/-   0.12 p50=   12.30  peak=  1.23 GiB
       permute-p2p:     9.87 ms +/-   0.09 p50=    9.85  peak=  0.98 GiB
p50 speedup vs a2a: permute-p2p=1.249x, ...
RESULT,cp=4,L=32768,batch=1,tp_size=1,path=a2a,mean_ms=...,p50_ms=...,peak_gib=...
```

## Constraints

The sequence length must divide the CP size, each rank's local length must be
even (balanced CP gives each rank two chunks), and each local half must be a
multiple of the SSD chunk size. The script asserts all three before measuring.

---

## Single-GPU Virtual Shape Kernel Benchmark

To isolate and compare the pure computation kernel latency of `(Batch = B, SeqLen = L)` versus `(Batch = 2*B, SeqLen = L/2)` (the shape used by `virtual` CP mode), run:

```bash
python examples/mamba_state_passing_context_parallel/benchmark_mamba2_virtual_shape.py
```

This measures pure kernel execution (Fused Conv1d + SSD Scan, SSD Chunk Scan Only, and Causal Conv1d Only) on a single GPU across various batch sizes and sequence lengths without distributed communication overhead.

