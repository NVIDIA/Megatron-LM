# AllGatherV MoE Dispatcher — Design Document

> Living design document. Update as decisions are finalized.

---

## 1. Overview

Standard EP all-gather dispatchers assume every rank contributes the same number of tokens per step. Real inference workloads have variable batch sizes across EP ranks (decode differs per request, prefill chunks differ per sequence). This document designs a variable-count (`AllGatherV`) MoE dispatcher that:

- Uses NVLS multicast all-gather for activation tensors (hidden_size always 128-bit aligned)
- Handles routing metadata (probs, expert IDs) which may not be 128-bit aligned
- Supports CUDA graph capture
- Keeps the old dispatcher intact with a flag to opt into the new one

**Scope:** EP-parallel AllGather path only. TP reduce-scatter is out of scope here. Torch grouped GEMM backend only.

---

## 2. Symmetric Memory Allocation

Three tensors are packed into a single symmetric memory buffer, allocated once at model load time:

| Tensor | Shape | Dtype | Notes |
|---|---|---|---|
| Activations | `[max_tokens * ep_size, hidden_size]` | bf16 | Always 128-bit aligned (assert on hidden_size) |
| Routing probs | `[max_tokens * ep_size, topk]` | fp32 | **Alignment issue — see §2.1** |
| Expert indices | `[max_tokens * ep_size, topk]` | int64 | Expert IDs; not a comm bottleneck |

`max_tokens` is the absolute engine-level maximum (known at model load). Buffers never resize.

### 2.1 Routing Probs Alignment Problem

`topk` is 6 or 22 for nanov3 and superv3. A row of fp32 probs:
- `topk=6`: 6 × 4 = 24 bytes — **not 128-bit (16-byte) aligned**
- `topk=22`: 22 × 4 = 88 bytes — **not 128-bit aligned**

**Decision: 64-bit fallback kernel.**

NCCL cannot express AllGatherV (variable per-rank counts), so routing metadata must also go through our custom kernel path. Padding wastes memory and complicates downstream indexing. Instead, add a 64-bit multicast variant:

- Use `multimem.st.relaxed.sys.global.v2.f32` (64-bit) in `multimem_asm.py` alongside the existing 128-bit `st_128`.
- fp32 probs: `topk=6` → 24 bytes (not 16-byte aligned) → 3 × 64-bit per row ✓
- fp32 probs: `topk=22` → 88 bytes (not 16-byte aligned) → 11 × 64-bit per row ✓
- int64 indices: `topk=6` → 48 bytes, `topk=22` → 176 bytes — both 16-byte aligned → **128-bit path**, no fallback needed

The `multimem_all_gather_v` kernel gains a `BITS: tl.constexpr` parameter (128 or 64) that selects the appropriate load/store path. Python wrapper detects alignment and dispatches accordingly.

### 2.2 Expert Indices: int64

Kept as int64 (matches PyTorch topk output dtype; not a communication bottleneck).

---

## 3. Per-Step Preprocessing (Outside the Model CUDA Graph)

Each step, before the model graph is replayed, the host runs a lightweight preprocessing pass that computes and publishes metadata into pre-allocated GPU tensors. The model graph reads from these fixed addresses at replay time but never writes or computes them itself.

### Pre-allocated GPU metadata tensors (fixed address, int32)

```
local_tokens_per_rank  [ep_size]   # how many tokens each rank contributes
prefix_sum             [ep_size]   # cumulative sum: prefix_sum[r] = sum(local_tokens[:r])
total_tokens           [1]         # sum of all local_tokens_per_rank
max_local_tokens       [1]         # max over ranks (drives CTA grid size)
rank_token_offset      [1]         # = prefix_sum[this_rank]; passed to AGV kernel
```

### Steps (per engine step, not captured in model CUDA graph)

1. **Engine writes** `local_tokens` (scalar) for this rank.
2. **NCCL AllGather** `local_tokens_per_rank` across the EP group (standard all-gather on a 1-D int32 tensor — supported by NCCL).
3. **On-device preprocessing kernel**: given `local_tokens_per_rank` in device memory, computes prefix sum, total sum, max and writes them to the fixed-address tensors above. This is a tiny single-block Triton kernel, *not* part of the model graph.
4. **Dispatcher `set_metadata` call** (see §7.3): copies the computed tensors into the dispatcher's state before graph replay.

This design keeps the model CUDA graph clean: it only reads from fixed addresses, never computes routing metadata.

---

## 4. AllGatherV Collective

### 4.1 Activation Tensor (NVLS multicast)

Already implemented in `variable_collectives.py::multimem_all_gather_v`.

- One CTA per token (persistent over `local_tokens` with stride `num_blocks`)
- Grid sized to `min(max_local_tokens, MAX_NUM_BLOCKS)` — same on all ranks (required for barrier symmetry)
- CTAs with `pid >= local_tokens` skip data movement but still participate in `symm_mem_sync`
- `rank_token_offset` loaded at kernel entry (fixed-address GPU tensor, graph-friendly)

**Assert (must stay):** `hidden_size * element_size % 16 == 0`

### 4.2 Routing Metadata (64-bit fallback kernel)

NCCL has no AllGatherV, so probs and indices also go through the NVLS kernel path using the 64-bit variant.

- Probs `[local_tokens, topk]` → `[total_tokens, topk]` via `multimem_all_gather_v(..., BITS=64)`
- Indices `[local_tokens, topk]` → `[total_tokens, topk]` via `multimem_all_gather_v(..., BITS=64)`
- These can be fused with the activation all-gather into a single kernel with one barrier (same pattern as `_multimem_all_gather_3_kernel` in `collectives.py`), provided all three writes complete before the sync.

### 4.3 Barrier Symmetry Requirement

All ranks must launch the same number of CTAs. This is why `max_local_tokens` (not `local_tokens`) drives the grid. Ranks with fewer tokens have idle CTAs that still call `symm_mem_sync` and exit. This is already handled correctly in the current implementation.

---

## 5. ReduceScatterV (Combine Path)

**This is missing from the current design and must be planned.**

The combine path is the reverse of dispatch:

1. Each rank holds a `[total_tokens, hidden_size]` output buffer (expert outputs summed over local experts).
2. **ReduceScatterV**: each rank reads its owned slice `[prefix_sum[r] : prefix_sum[r] + local_tokens[r], :]` from the all-gathered output, performs the NVLS `multimem.ld_reduce` reduction across EP peers, and writes `[local_tokens[r], hidden_size]` locally.

### Key design point

In standard reduce-scatter, each rank owns a contiguous equal-sized shard. Here, rank `r` owns tokens `[prefix_sum[r], prefix_sum[r] + local_tokens[r])` in the global buffer. The kernel must:
- Know `rank_token_offset` (prefix_sum[this_rank]) to find its shard in the reduction buffer
- Know `local_tokens` to know how much to read
- Grid size again driven by `max_local_tokens` for barrier symmetry

A `multimem_reduce_scatter_v` kernel mirrors `multimem_all_gather_v`:
- Each CTA processes one token in this rank's shard
- Reads from `multicast_ptr + (rank_token_offset + token_idx) * numel_per_token` using `multimem.ld_reduce`
- Writes to local output buffer

### Routing metadata combine

Routing probs/indices do **not** need to be reduce-scattered — they were used as read-only inputs to expert selection and are no longer needed after the forward pass.

---

## 6. Grouped GEMM Backend Changes

The torch grouped GEMM backend currently assumes a static token count. Changes needed:

### 6.1 Valid token count parameter

Pass a scalar int32 GPU tensor `valid_tokens` (= `total_tokens`) to the grouped GEMM entry point. This tensor has a fixed address (allocated once) and its value is updated each step.

### 6.2 Internal kernel changes

All internal operations must respect `valid_tokens`:

| Operation | Change |
|---|---|
| `permute` | Operate on `[valid_tokens, hidden_size]` slice only |
| `unpermute` | Operate on `[valid_tokens, hidden_size]` slice only |
| Expert GEMM | Expert token counts derived from `valid_tokens` slice of routing map |
| Activation | Applied only within `valid_tokens` rows |

The workspace is sized `[max_tokens * ep_size, hidden_size]` but only the first `valid_tokens` rows are valid. Rows beyond `valid_tokens` are stale from a previous step — they must not be read by any kernel.

### 6.3 Routing map

The grouped GEMM backend receives routing decisions for the full `[valid_tokens, topk]` slice. The `mcore_fused_moe` API already accepts an int routing map (expert IDs); this extends naturally.

---

## 7. Dispatcher Design

### 7.1 New class: `MoEAllGatherVTokenDispatcher`

Inherits from (TBD — likely a new base or minimal shared base, not `MoEAlltoAllTokenDispatcher`).

Interface mirrors existing dispatcher but accepts variable token counts:

```python
class MoEAllGatherVTokenDispatcher:
    def dispatch(self, hidden_states, probs, routing_map) -> (all_gathered_hidden, all_gathered_probs, all_gathered_routing)
    def combine(self, expert_output) -> local_output
```

### 7.3 Metadata Propagation via `set_metadata`

Rather than threading `valid_tokens`, `max_tokens`, and `rank_token_offset` through every function signature, the dispatcher exposes a `classmethod` (or `staticmethod`) that the engine calls once per step, before graph replay:

```python
class MoEAllGatherVTokenDispatcher:
    # Class-level state: fixed-address GPU tensors, same across all instances
    _valid_tokens: torch.Tensor      # int32 scalar, device
    _max_local_tokens: torch.Tensor  # int32 scalar, device
    _rank_token_offset: torch.Tensor # int32 scalar, device
    _prefix_sum: torch.Tensor        # int32 [ep_size], device

    @classmethod
    def set_metadata(cls, valid_tokens, max_local_tokens, rank_token_offset, prefix_sum):
        """Called by engine after preprocessing, before graph replay.
        Writes values into the fixed-address tensors that the CUDA graph reads."""
        cls._valid_tokens.copy_(valid_tokens)
        cls._max_local_tokens.copy_(max_local_tokens)
        cls._rank_token_offset.copy_(rank_token_offset)
        cls._prefix_sum.copy_(prefix_sum)

    @classmethod
    def init_metadata_buffers(cls, ep_size: int, device: torch.device):
        """Called once at model init to allocate fixed-address metadata tensors."""
        cls._valid_tokens = torch.zeros(1, dtype=torch.int32, device=device)
        cls._max_local_tokens = torch.zeros(1, dtype=torch.int32, device=device)
        cls._rank_token_offset = torch.zeros(1, dtype=torch.int32, device=device)
        cls._prefix_sum = torch.zeros(ep_size, dtype=torch.int32, device=device)
```

Using class-level state means all MoE layers in the model share the same metadata tensors — consistent with the fact that `valid_tokens` is global to the step, not per-layer. The CUDA graph captures the fixed tensor addresses; `set_metadata` updates their values before each replay without re-capturing.

### 7.2 Enable flag

Add to `TransformerConfig`:
```python
moe_use_allgather_v_dispatcher: bool = False  # enable AllGatherV dispatcher
```

Consistent with existing patterns (`moe_disable_fuse_quant`, etc.).

### 7.3 Keep old dispatcher

Old dispatcher remains untouched. Flag selects which dispatcher is instantiated in `experts.py` or the MoE layer init.

### 7.4 CUDA graph considerations (resolved)

The new dispatcher is CG-safe by construction:

- All metadata (`valid_tokens`, `rank_token_offset`, `prefix_sum`) lives in **fixed-address GPU tensors**; `set_step_metadata()` updates them *before* graph replay — no host-device sync inside the graph.
- The `local_tokens_per_rank` NCCL all-gather and `dist.barrier(ep_group)` are called from the preprocessing path (context + dummy forward), fully **outside** the graph boundary.
- `symm_mem_sync` signal-pad barriers inside AGV/RSV kernels are self-resetting → safe to replay without re-initialization.

---

## 8. Open Questions

| # | Question | Status |
|---|---|---|
| 1 | Routing metadata alignment strategy? | **Resolved: 64-bit fallback kernel** |
| 2 | Expert index dtype? | **Resolved: int64** |
| 3 | Preprocessing in model CUDA graph? | **Resolved: outside graph, on-device kernel, fixed tensors** |
| 4 | Metadata propagation mechanism? | **Resolved: `classmethod set_metadata` on dispatcher** |
| 5 | `multimem_reduce_scatter_v` kernel design? | **Resolved: implemented, 10/10 tests passing.** |
| 6 | What does the new dispatcher inherit from? | TBD. |
| 7 | CUDA graph capture strategy (ep sync points)? | **Resolved: fixed-address tensors + preprocessing outside graph + self-resetting symm_mem_sync.** |
| 8 | 64-bit multicast store (`multimem.st v2.f32`) verified on Blackwell? | **Resolved: verified, all tests pass.** |
| 9 | Fuse AGV for all 3 tensors (activations + probs + indices) into one kernel? | TBD — check signal pad reuse. |

---

## 9. Implementation Order (Suggested)

1. ✅ **64-bit ASM primitives** — `ld_64` / `st_64` added to `multimem_asm.py`
2. ✅ **`multimem_all_gather_v` with `BITS` param** — auto-selects 128/64-bit path from row alignment; 22/22 tests passing (4 distributions × 5 tensor specs + multi-call + ep_max edge case)
3. ✅ **`multimem_reduce_scatter_v` kernel** — mirrors AGV, uses `rank_token_offset` to locate this rank's shard; 10/10 tests passing (4 distributions × 2 tensor specs + multi-call + ep_max edge case)
4. **On-device preprocessing kernel** — prefix sum + total + max from all-gathered `local_tokens_per_rank`
5. **`MoEAllGatherVTokenDispatcher` skeleton** — `init_metadata_buffers`, `set_metadata`, dispatch (AGV for all 3 tensors)
6. **Grouped GEMM `valid_tokens` plumbing** — pass fixed-address scalar through `mcore_fused_moe` → permute/unpermute
7. **Combine path** — `multimem_reduce_scatter_v` + unpermute with `valid_tokens`
8. ✅ **CUDA graph capture hardening** — fixed-address metadata tensors, preprocessing outside graph, self-resetting symm_mem_sync
9. **End-to-end integration test** — nanov3 config (EP=4, topk=8, hidden=2688)
