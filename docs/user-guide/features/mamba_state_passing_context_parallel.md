<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Mamba2 State-Passing Context Parallelism

Sequence-sharded, state-passing context parallelism (CP) for Mamba2 training.
The production implementation lives in
[`megatron/core/ssm/ops/ssd_state_passing_cp.py`](../../../megatron/core/ssm/ops/ssd_state_passing_cp.py)
and is opt-in; the existing all-to-all Mamba CP path remains the default.

## Overview

The existing Mamba CP path redistributes activations with an all-to-all (A2A) so
that every rank holds the full sequence for a subset of heads and groups.
State-passing CP instead keeps the sequence shard local and communicates only
what the causal boundary requires: the convolution halo and the SSD state
summary.

| Path | Sequence handling | Dominant communication |
|---|---|---|
| No CP | one rank processes the whole sequence | none |
| Mamba A2A CP | full sequence, head/group shard | activation A2A proportional to sequence length |
| State-passing CP | local sequence shard is kept | conv halo and SSD boundary all-gather |

Approximate rank-local payloads:

```text
A2A:
  O(local_sequence * hidden)

State-passing boundary:
  Conv: O(batch * channels * (d_conv - 1))
  SSD : O(batch * heads * headdim * dstate) + O(batch * heads) for decay
```

The boundary collective is an all-gather. Its size does not scale with sequence
length, but the received volume grows with the CP size.

## Configuration

- `--use-mamba-state-passing-cp` enables the path.
- `--mamba-state-passing-cp-load-balancing {none,permute_p2p,permute_a2a,virtual}`
  selects how the standard balanced CP layout is handled.

`none` expects an already-contiguous causal shard and is only valid for direct
calls into the state-passing kernels. Standard Megatron CP hands the mixer a
front/back balanced layout, so `MambaMixer` requires `permute_p2p`,
`permute_a2a`, or `virtual`.

`MambaMixer` dispatch:

```text
hidden_states
  -> in_proj -> zxBCdt
  -> MambaMixer._use_mamba_state_passing_cp()
       |
       +-- cp_size == 1
       |     -> standard fused Mamba
       |
       +-- state-passing OFF
       |     -> MambaContextParallel.pre_conv_ssm()
       |     -> fused Mamba
       |     -> MambaContextParallel.post_conv_ssm()
       |
       `-- state-passing ON
             -> MambaStatePassingCPAdapter.forward()
             -> out_proj
```

## Load-balancing modes

Megatron's balanced CP assigns one front chunk and one back chunk to each rank:

```text
CP=3 balanced ownership:
  rank 0: [1, 6]
  rank 1: [2, 5]
  rank 2: [3, 4]

causal order:
  [1, 2, 3, 4, 5, 6]
```

### Permute

`permute_p2p` and `permute_a2a` physically exchange the balanced activations for
a contiguous causal shard:

```text
balanced:
  rank 0 [1,6], rank 1 [2,5], rank 2 [3,4]

contiguous:
  rank 0 [1,2], rank 1 [3,4], rank 2 [5,6]
```

`undo_state_passing_cp_load_balancing()` runs on the forward input and
`redo_state_passing_cp_load_balancing()` on the output. A custom autograd
function restores gradient ownership with the inverse permutation in the
backward pass.

- `permute_p2p` sends and receives the required remote chunks point-to-point.
- `permute_a2a` exchanges them in one unequal-split `all_to_all_single`.

Both backends produce the same layout and the same mathematics; only the
communication implementation differs.

### Relation to `context_parallel_layout`

Megatron Core already models the balanced/contiguous distinction as
`CpPartitionMode = Literal["zigzag", "contiguous"]` in
`megatron.core.context_parallel_layout`, and Gated DeltaNet uses it at its module
entry point through `convert_module_input_tensors_cp_partition_mode()` when it
runs its chunkwise (sequence-sharded) CP path. `permute_p2p` and `permute_a2a`
perform the same conversion, so the two overlap.

They are implemented separately here for two reasons: the permutation is driven
from inside the fused Conv+SSD autograd function rather than at the module entry
point, and `permute_p2p` adds a point-to-point backend that the shared helper
does not provide. Consolidating onto the shared helper is intended follow-up
work.

### Virtual

`virtual` moves no activations. Each physical rank's front and back chunks are
interpreted as two independent virtual causal segments, and the input is viewed
as an interleaved virtual batch:

```text
[local_L, batch, ...] -> [2 * batch, local_L / 2, ...]
```

Boundary routing then selects the predecessor and successor virtual segment out
of the rank-major gather result.

#### Computational Kernel Efficiency of Virtual Mode

Because Mamba-2 (SSD) decomposes the sequence into fixed-size chunks (e.g. `chunk_size = 128`), the dominant chunk-level GEMM/BMM FLOPs depend strictly on the total token count (`batch * local_L`). Halving the sequence length and doubling the batch size (`[2 * batch, local_L / 2]`) preserves the total token count and chunk count:
- **Inter-chunk recurrence latency**: Sequential scan loops across chunks per sequence are halved (`(local_L / 2) / chunk_size`), reducing critical path latency while doubling grid parallelism (beneficial for SM occupancy on small micro-batches).
- **Pure kernel throughput**: Benchmarking confirms that `(2 * batch, local_L / 2)` executes at parity (`1.00x` speedup) with `(batch, local_L)` for fused Conv+SSD kernels, with negligible boundary routing overhead.
- **End-to-end advantage**: By eliminating the `O(local_L * hidden)` activation permutation communication entirely, `virtual` mode is the most communication- and memory-efficient load-balancing strategy for long sequence training.


## Production flow

The path fuses the convolution and the SSD scan into a single custom autograd
function, `MambaSplitConv1dScanCombinedStatePassingCPFn`. Its public API keeps
the upstream `MambaSplitConv1dScanCombinedFn` argument order and appends
`state_passing_cp_group` and `state_passing_cp_virtual`.

### Forward

```text
zxBCdt [local_L, batch, packed]
  -> load-balancing handling
  -> virtual: view as [2 * batch, local_L / 2, packed]
  -> split z / xBC / dt

Conv
  -> all-gather local tail
  -> select predecessor tail
  -> causal_conv1d(initial_states=predecessor_tail)
  -> split x / B / C

SSD
  -> dt cumsum and local chunk contribution
  -> packed summary [S_ext, a_block]
  -> async all-gather(summary) || local CB computation
  -> exclusive causal boundary scan
  -> S_in
  -> local state passing and chunk scan

Output
  -> virtual unpack or permutation redo
  -> [local_L, batch, d_inner]
```

The SSD transform of one causal segment is affine:

```text
S_out = a_block * S_in + S_ext
```

- `S_ext` is the final state the local segment produces from a zero initial state.
- `a_block` is the per-head decay across the whole segment.
- `S_in` is the exclusive prefix composition of the preceding segment summaries.

`_state_passing_summary_fwd_kernel()` writes `S_ext` and the decay straight into
an FP32 packed buffer. `_state_passing_boundary_scan_kernel()` handles both the
forward and the backward causal order through the compile-time `VIRTUAL_CP` and
`REVERSE` flags.

### Convolution boundary

The causal convolution needs the last `d_conv - 1` tokens of the preceding
segment:

```text
forward:
  local tail -> all-gather -> predecessor tail
             -> causal_conv1d_fwd_function(initial_states=...)

backward:
  causal_conv1d_bwd_function(return_dinitial_states=True)
             -> all-gather d(initial_state)
             -> select successor gradient
             -> in-place add into the local dx tail
```

### Backward

```text
re-split xBC from zxbcdt
  -> recompute the causal conv forward
  -> recompute dt cumsum, CB, chunk contribution, recurrence states
  -> local chunk dstates
  -> local dS_in summary
  -> async all-gather(dS_in) || local dC / dCB / ddA computation
  -> exclusive reverse boundary scan
  -> dS_ext from the successor
  -> one _state_passing_bwd(dfinal_states=dS_ext)
  -> dx / dB / dC / ddt / dA / dD / dz
  -> causal conv backward
  -> apply the reverse conv boundary gradient to the dx tail
```

The post-conv gradient the SSD backward produces is written into a preallocated
`dzxBCdt` view and consumed directly by the convolution backward; there is no
separate autograd accumulation to re-join the split gradients.

### Saved tensors and dtypes

The forward context keeps the upstream tensor order and appends
`state_passing_conv_initial_states`, `state_passing_initial_states`, and
`state_passing_gathered_decays`. Quantities that scale with sequence length
(`xBC_conv`, the split `x`/`B`/`C`, `dt_proc`, the chunk states, and `CB`) are
recomputed in the backward pass instead of being saved.

| Value | dtype |
|---|---|
| `S_ext`, `a_block` | FP32 |
| `S_in`, recurrence states | FP32 |
| `dS_in`, `dS_ext` | FP32 |
| scan/BMM boundary tensors | upstream input dtype |

## Supported scope

Supported:

- the training memory-efficient path
- fixed-length, non-packed sequences
- the standard balanced CP input layout
- `permute_p2p`, `permute_a2a`, and `virtual`
- BF16 activations with FP32 boundary state

Not supported yet:

- inference
- `seq_idx` and packed sequences
- hybrid, dynamic, or otherwise variable-length CP
- an externally supplied SSD `initial_states`
- `--mamba-training-ssm-states-dtype`
- fused RMSNorm and output projection inside the custom function

The mixer adapter runs RMSNorm and the output projection outside the custom
function, so both remain available on the full `MambaMixer` training path.

Shape constraints: each virtual segment length must be a multiple of the SSD
`chunk_size`, and each convolution segment must hold at least `d_conv - 1`
tokens.

## Tests

- [`tests/unit_tests/ssm/ops/test_ssd_state_passing_cp.py`](../../../tests/unit_tests/ssm/ops/test_ssd_state_passing_cp.py)
  checks the load-balancing permutation for exactness and the convolution and
  SSD kernels against a full-sequence reference.
- [`tests/unit_tests/ssm/test_mamba_mixer_state_passing_cp.py`](../../../tests/unit_tests/ssm/test_mamba_mixer_state_passing_cp.py)
  checks every mode end to end against a full-sequence mixer and against the
  A2A CP path.
- [`tests/unit_tests/ssm/test_mamba_state_passing_cp_cuda_graph.py`](../../../tests/unit_tests/ssm/test_mamba_state_passing_cp_cuda_graph.py)
  covers the Megatron local and Transformer Engine CUDA Graph backends.
