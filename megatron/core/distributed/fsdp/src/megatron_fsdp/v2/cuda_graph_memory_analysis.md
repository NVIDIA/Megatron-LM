# CUDA Graph Memory Overhead — Analysis & Findings

> **Status:** experimental findings from profiling the per-layer graph
> capture path.  Numbers come from a QwenImage diffusers model wrapped
> with Megatron FSDP v2 + `enable_cuda_graph=True`.  The qualitative
> conclusions hold for any per-layer graph capture that pops torch
> forward hooks before capture.

## TL;DR

CG capture costs **+1.222 GB / captured layer** vs no-CG, in steady
state.  There are **two independent sources**, both rooted in the same
root cause: **CG capture pops all module hooks (including torch._dynamo
inductor hooks), so the captured forward sees *un-fused* python-level
ops**.

| Source | Per-layer | % of total |
|---|---:|---:|
| Framed activations (inductor fusion skipped → one SavedVariable per python op instead of one per fused region) | +0.524 GB | 43 % |
| No-frame cuBLAS / cuDNN workspaces (no fusion → one workspace per matmul, never recycled across layers in the graph pool) | +0.698 GB | 57 % |
| **Total** | **+1.222 GB** | **100 %** |

Neither source is "saved activations that backward would have freed" —
that was the wrong framing in earlier drafts.  Both sources survive in
the graph pool forever because graph memory is pinned for replay, but
their *magnitude* would be much smaller if CG capture went through
inductor's fused kernels rather than the raw python ops.

## How we measured (apples-to-apples)

Two env-var gated debug modes in `FSDPCudaGraphRunner`
(see `cuda_graph_runner.py`):

| `MFSDP_CG_MEM_DEBUG` | Behaviour |
|---|---|
| `cg`    | Capture CG normally. For the first `MFSDP_CG_MEM_SNAP_LAYERS` layers, record stack-aware `torch.cuda.memory._record_memory_history` around the forward-graph capture and dump `cg_layer{N}_rank{R}.pickle` immediately after capture. |
| `nocg`  | Skip CG capture entirely. `capture_forward` just stores debug state; `install()` **patches `module.forward`** with a wrapper (`_debug_recorded_forward`) that records + snapshots the **real** forward — the one that runs after `forward_pre_hook` returns. So `nocg_layer{N}_rank{R}.pickle` captures the actual training forward, with hooks (and inductor) intact. |
| _(unset)_ | Normal production. |

Run separately:

```bash
MFSDP_CG_MEM_DEBUG=cg    MFSDP_CG_MEM_SNAP_LAYERS=3 MFSDP_CG_MEM_SNAPSHOT_DIR=/tmp/cg_mem python train.py
MFSDP_CG_MEM_DEBUG=nocg  MFSDP_CG_MEM_SNAP_LAYERS=3 MFSDP_CG_MEM_SNAPSHOT_DIR=/tmp/cg_mem python train.py
```

Per-layer peak (`peak_alloc`, `peak_reserved`, `post`) is logged for
each captured layer in both modes.  Snapshots are loadable at
https://pytorch.org/memory_viz.

### Why the nocg path patches `module.forward`

Earlier versions of the debug code ran an *extra* forward inside
`capture_forward` to take the snapshot.  That created a measurement
artifact: the extra forward's autograd tape was destroyed when
`_debug_run_eager` returned, freeing its activations into the caching
allocator's free list.  The real forward (the one that runs after the
pre-hook returns) then *reused* those freed addresses, making no-CG
appear to grow at ~0.5 GB/layer instead of its true ~1.2 GB/layer
forward-only rate.

The current design patches `module.forward` itself so the snapshot is
taken during the actual training forward.  This means:

- The autograd tape accumulates naturally across layers (as in real
  training).
- Inductor is active (its dynamo hooks survive because `capture_forward`
  returns early in `nocg` mode, before `_pop_hooks_recursive` runs).
- The snapshot captures true per-layer forward cost.

## The numbers

### Per-layer live memory (total active bytes across all pools)

| Layer | CG total | NOCG total | CG − NOCG |
|---:|---:|---:|---:|
| 0 | 52.055 GB | 51.450 GB | **+0.606 GB** |
| 1 | 53.771 GB | 51.943 GB | **+1.828 GB** |
| 2 | 55.487 GB | 52.437 GB | **+3.050 GB** |

### Per-layer growth rate (averaged over L0→L1 and L1→L2)

| Component | CG | NOCG | Savings |
|---|---:|---:|---:|
| Framed activations (graph pool in CG, caching pool in NOCG) | +1.176 GB | +0.652 GB | **0.524 GB** |
| No-frame workspaces (caching pool, both modes) | +0.540 GB | −0.158 GB | **0.698 GB** |
| **Total** | **+1.716 GB** | **+0.494 GB** | **1.222 GB** |

Note: NOCG's no-frame caching pool actually **shrinks** by 158 MB/layer.
That is the inductor-fused forward releasing workspace back to the
caching allocator after each fused kernel completes; the only thing
that persists is autograd-saved activations.

## Source A — framed activations (0.524 GB/layer)

### Per-operator-family breakdown at layer 0

| Family | CG graph pool | NOCG caching pool | Diff |
|---|---:|---:|---:|
| Linear (`linear.py:134`) | 0.397 GB | 0.000 GB | +0.397 |
| LayerNorm (`functional.py:2935`, `normalization.py:554/555/560`) | 0.454 GB | 0.000 GB | +0.454 |
| GELU (`activations.py:85`) | 0.129 GB | 0.000 GB | +0.129 |
| QwenImage attention processor (`transformer_qwenimage.py:558/559/560/582/733/739`) | 0.163 GB | 0.000 GB | +0.163 |
| Attention (`flash_attn_interface.py:96`) | 0.034 GB | 0.034 GB | 0.000 |
| Inductor triton kernel (`ctjbu2*.py`) | 0.000 GB | 0.618 GB | −0.618 |
| **Total** | **1.176 GB** | **0.652 GB** | **+0.524 GB** |

### Why the discrepancy

CG's call stacks show raw python ops:
```
linear.py:134 forward
  ↑ module.py:1789 _call_impl
  ↑ activations.py:88 forward         (diffusers GEGLU)
  ↑ attention.py:1741 forward
  ↑ transformer_qwenimage.py:732 forward
  ↑ cuda_graph_runner.py:516 capture_forward
  ↑ hooks.py:84 forward_pre_hook
```

NOCG's call stacks show inductor triton kernels:
```
ctjbu2xj2tikjgdogi372mldv5x4h2vgi5fvncjsn3yscjbwxopj.py:1631 call   ← inductor-fused kernel
  ↑ activations.py:88 forward
  ↑ attention.py:1741 forward
  ↑ transformer_qwenimage.py:732 forward
  ↑ utils.py:3331 run              ← inductor runtime
  ↑ output_code.py:638 __call__    ← inductor output code
  ↑ runtime_wrappers.py:930 inner_fn
```

100 % of NOCG framed bytes (45/45 blocks, 0.652 GB) come from inductor
kernels.  0 % of CG framed bytes touch inductor (0/68 blocks).  This is
because `_pop_hooks_recursive` (called at `cuda_graph_runner.py:438`)
removes the dynamo hooks that drive inductor compilation.  CG capture
runs the un-fused python-level forward; each python op saves its own
input tensor for backward, producing the per-call-site breakdown above.

The inductor path, by contrast, fuses sequences of ops (Linear +
LayerNorm + GELU + ...) into a single triton kernel.  Saved-tensor
points become fusion-region boundaries rather than per-op boundaries,
so far fewer SavedVariables are allocated.

## Source B — no-frame workspaces (0.698 GB/layer)

### Layer-over-layer delta in no-frame caching blocks

| Mode | L0→L1 gone | L0→L1 new | Net | L1→L2 gone | L1→L2 new | Net |
|---|---|---|---:|---|---|---:|
| NOCG | 1 block / −0.158 GB | 0 / 0 GB | **−0.158 GB** | 1 / −0.158 GB | 0 / 0 GB | **−0.158 GB** |
| CG | 2 blocks / −0.032 GB | 52 / +0.706 GB | **+0.674 GB** | 2 / −0.032 GB | 52 / +0.707 GB | **+0.674 GB** |

- **NOCG** actually **frees** one ~158 MB no-frame block per layer —
  inductor's fused kernel allocates a single workspace and releases it
  when the kernel completes.  Inductor's per-call workspace is not
  captured by autograd, so it returns to the caching allocator's free
  list.
- **CG** adds **+52 no-frame blocks per layer (0.706 GB)** that never
  get recycled.  These are cuBLAS / cuDNN workspaces — every captured
  python-level matmul and convolution grabs its own workspace.  The
  graph capture prevents workspace reuse across layers (the kernels
  baked into the graph reference those workspace addresses for replay).

The 52-blocks-per-layer pattern is highly regular (predictable block
sizes: 2 × 96 MB, 2 × 36 MB, 4 × 33 MB, 9 × 24 MB, 9 × 9 MB, ...),
suggesting one workspace per matmul/conv per layer.  With inductor
fusion, far fewer separate matmuls happen, so far fewer workspaces are
needed.

## Why CG capture bypasses inductor

`hooks.py:84` triggers CG capture inside the FSDP `forward_pre_hook`
(which is `@torch.compiler.disable`d, so dynamo treats it as a graph
break).  Inside the pre-hook, `capture_forward` (line 438) does two
things:

1. **Pops all hooks recursively** via `_pop_hooks_recursive`.  This
   removes the FSDP `forward_pre_hook` / `forward_hook` that triggered
   capture — necessary, otherwise they would re-fire during the
   captured forward and race with the training loop.

2. **Invokes the forward body directly** via `_call_module` →
   `self._module.forward(**kwargs)`.  This bypasses `nn.Module.__call__`
   (and thus `_call_impl`), which is where PyTorch 2.x installs the
   dynamo dispatch.

The previous version of this document blamed step 1 — claiming that
`_pop_hooks_recursive` removes "the dynamo/inductor hooks." **This is
incorrect.**  In PyTorch 2.4, `module.compile()` does NOT install any
hooks; it sets `module._compiled_call_impl = torch.compile(_call_impl)`
(a `__call__`-level dispatch).  Hooks popping is therefore orthogonal
to inductor: removing `_forward_pre_hooks` / `_forward_hooks` etc.
has no effect on whether dynamo fires.

The actual cause is step 2 — `_call_module` calls `.forward()`
directly, never going through `_wrapped_call_impl` /
`_compiled_call_impl`.  Furthermore, even if the call site were
switched to `self._module(**kwargs)`, the dispatch happens INSIDE a
dynamo graph break (the FSDP pre-hook is `@torch.compiler.disable`),
so dynamo would not have an active trace to fall back into.  The
captured forward therefore runs the raw python body, emitting eager
aten ops + cuBLAS workspaces (Source B) and per-op SavedVariables
(Source A).  No inductor fusion, no triton kernels.

## Practical levers

### Lever 1 — Capture a torch.compile()'d forward body (biggest win)

Compile `module.forward` directly inside `capture_forward` so the
warmup below populates dynamo's cache for the forward BODY (not
`_call_impl`), and the capture (inside `torch.cuda.graph(pool=...)`)
runs the cached inductor code.  Triton kernel launches are
CUDA-graph-capturable, so the resulting CG graph contains the same
fused kernels as the no-CG path — eliminating essentially all of the
1.222 GB/layer overhead (both Source A and Source B).

Open questions:
- Does dynamo's cached lookup fire inside `torch.cuda.graph()` context?
  Initial source inspection suggests yes (no global "skip during
  capture" guard in `torch/_dynamo/eval_frame.py`); needs on-GPU
  confirmation.
- Does inductor's allocator play nice with `torch.cuda.graph(pool=...)`?
  Inductor-allocated workspaces would land in the graph pool and be
  reused on replay — desirable.

Implementation sketch (CURRENT FIX, env-gated on
`MFSDP_CG_COMPILE_FWD=1`):

```python
# Inside capture_forward, after _pop_hooks_recursive and gc.freeze,
# BEFORE the warmup loop:
if _CG_COMPILE_FWD and not hasattr(self._module.forward, "get_compiler_config"):
    self._orig_fwd_body = self._module.forward
    self._module.forward = torch.compile(self._orig_fwd_body)
    self._captured_fwd_was_compiled = True

# Warmup (3 iters): _call_module → self._module.forward (compiled).
# Iter 1 triggers dynamo compile; iters 2–3 use the cache.

# Capture: _call_module inside torch.cuda.graph(pool=...).
# Compiled code runs from dynamo cache → triton kernels fire → captured.

# Finally: restore self._module.forward = self._orig_fwd_body so
# install() sees the user-written body and substitutes _patched_fwd
# (the CG replay function) over it.
```

User-visible effect:
- `blk(*args)` → dynamo's compiled `_call_impl` (user's `blk.compile()`)
  → forwards call to `_patched_fwd` → `_CudaGraphFunction.apply`
  → `fwd_graph.replay()` fires the captured triton kernels.

The user's outer `blk.compile()` is still useful: it fuses the
pre/post-hook graph breaks, the autograd-Function boundary, and any
ops outside the FSDP block.

### Lever 2 — Activation recomputation on the captured block

The 0.524 GB/layer of framed activations (Source A) is forward
intermediates saved for backward.  If `module.forward` is wrapped in
`torch.utils.checkpoint.checkpoint`, the captured forward graph can
free its intermediates before end-of-capture.  Per-layer pinned
footprint from Source A drops from 0.524 GB to ~50 MB (just block I/O).

This does NOT help Source B (cuBLAS workspaces) — those are captured
regardless of whether the forward is checkpointed.  And it adds
compute (forward runs twice).

### Lever 3 — Replace diffusers RMSNorm + Linear + GELU with TE fused kernels

TE fuses norm + linear + activation into one kernel that uses
kernel-private workspace.  The Linear (0.397 GB) + LayerNorm (0.454 GB)
+ GELU (0.129 GB) buckets in Source A collapse into a handful of TE
fused ops; pinned bytes drop proportionally.  Also reduces Source B
because there are fewer separate matmul calls.

### Lever impact summary

| Lever | Source A | Source B | Total | Notes |
|---|---:|---:|---:|---|
| Lever 1 (capture through inductor) | −0.524 GB | −0.698 GB | **−1.222 GB** | Recovers essentially all overhead. Implementation risk in graph-pool / inductor-allocator interaction. |
| Lever 2 (activation recompute) | −0.474 GB | 0 | **−0.474 GB** | Adds compute. Standard pattern in `te.fp8_checkpoint`. |
| Lever 3 (TE fused ops) | −0.5 to −0.6 GB | −0.3 to −0.4 GB | **−0.8 to −1.0 GB** | Direct replacement; works alongside Lever 1 or 2. |
| L1 + L2 combined | −0.524 GB* | −0.698 GB** | **−1.222 GB** | *recompute on fused kernels saves less per layer, but the per-fused-region count is what matters; **inductor workspace recycling still helps. |

Lever 1 is the clearly preferred direction — it fixes the root cause
rather than treating the symptoms.

## What is *not* the problem

Earlier drafts of this analysis attributed CG overhead to "saved
activations that backward would normally free."  That framing is
incomplete:

- **NOCG also saves activations for backward** — its 0.652 GB/layer of
  framed caching-pool bytes are SavedVariables too.  Per-layer growth
  in NOCG is real (0.494 GB/layer total); CG's overhead is on top of
  that, not instead of it.
- After backward, NOCG's framed activations are freed by the caching
  allocator and the addresses are reused next microbatch.  CG's framed
  activations are pinned for replay.  But the *magnitude* of CG's
  pinned bytes is inflated by the inductor-bypass: if both paths
  captured the same fused-kernel forward, CG's per-layer pin would be
  0.652 GB, not 1.176 GB.
- The dominant saving (0.698 GB/layer, 57 %) is **not about
  SavedVariables at all** — it is about cuBLAS/cuDNN workspaces that
  accumulate because un-fused python-level matmuls each grab their own.

## Methodology artefacts removed

The current nocg measurement path (patching `module.forward` with
`_debug_recorded_forward`) addresses three artefacts that affected
earlier measurements:

1. **Tape destruction before snapshot (early `_debug_run_eager`).**
   The first version did `del out, flat_out` before taking the
   snapshot, which collapsed the autograd tape and made NOCG look like
   it held 0 activations.  Fixed by keeping `out` alive in scope until
   the function returns.
2. **Extra forward before real forward (later `_debug_run_eager`).**
   The second version ran an extra forward inside `capture_forward`,
   which polluted the caching allocator's free list and made NOCG look
   like it grew only 0.5 GB/layer.  Fixed by patching `module.forward`
   so the snapshot is taken during the real training forward.
3. **`empty_cache()` in the debug path.**  An earlier version called
   `torch.cuda.empty_cache()` after each layer's snapshot, which
   perturbed the caching allocator's free list between layers.
   Removed.

## Files & tooling reference

| Artifact | Location / how to produce |
|---|---|
| Debug mode implementation | `cuda_graph_runner.py` — `_CG_MEM_MODE`, `_debug_recorded_forward`, the nocg branch in `capture_forward`, the nocg branch in `install`. |
| Snapshot dumps | `cg_layer{N}_rank{R}.pickle`, `nocg_layer{N}_rank{R}.pickle` |
| Snapshot visualization | https://pytorch.org/memory_viz (drag pickles in) |
| Env vars | `MFSDP_CG_MEM_DEBUG={cg,nocg}`, `MFSDP_CG_MEM_SNAP_LAYERS=N`, `MFSDP_CG_MEM_SNAPSHOT_DIR=<dir>` |
| Snapshot analysis scripts | ad-hoc Python (see git history of this doc for `rigorous_compare.py`, `confirm_theory.py`, `layer_delta.py`). |

## Open questions

- Can `_pop_hooks_recursive` be made to selectively pop only FSDP
  hooks, leaving dynamo/inductor hooks intact?  If so, Lever 1 becomes
  a single-line fix.
- Does `torch.compile(module)` before CG capture produce a forward that
  plays nicely with `torch.cuda.graph(pool=...)`?  Specifically, does
  inductor's allocator honor the pool argument?
- The 52-blocks-per-layer no-frame pattern in CG is highly regular —
  cataloguing which python ops each block corresponds to would let us
  estimate Lever 3's impact per-op rather than per-family.
- For non-attention modules (e.g. MLP-only blocks), is the
  inductor-bypass effect smaller (fewer separate matmuls to fuse)?

---

## Full-snapshot analysis (post-Lever-1)

After enabling `MFSDP_CG_COMPILE_FWD=1` (Lever 1 — `torch.compile` the
forward body during capture), a full end-of-training-step snapshot was
captured for both NOCG and CG+compile.  These snapshots include the
optimizer state, gradient buffers, and the captured graphs for all 60
transformer layers, giving an apples-to-apples comparison of the
*total* memory cost.

Files:
- `memory_snapshot_nocg.pickle` — no CG (caching allocator only)
- `memory_snapshot_cg_compile.pickle` — CG + `torch.compile` fusion

### Top-line: CG+compile costs +50 GB vs NOCG (101 GB vs 51 GB)

| Category (pool) | NOCG | CG+C | Δ GB |
|---|---:|---:|---:|
| inductor:attn/blk (graph) | 0.000 | 21.124 | **+21.124** |
| inductor:activations (graph) | 0.000 | 18.741 | **+18.741** |
| no_frames (caching) — model weights/optimizer | 20.441 | 20.441 | 0 |
| adam.py (caching) — optimizer state | 20.431 | 20.431 | 0 |
| param_group.py (caching) — FSDP grad buffers | 10.216 | 10.215 | 0 |
| cg_runner (caching) — static_inputs | 0.000 | 4.166 | **+4.166** |
| fsdp_allocator (caching) — gradient bucket coloring | 0.000 | 3.561 | **+3.561** |
| flash_attn (graph) | 0.000 | 2.115 | **+2.115** |
| inductor (caching) — warmup allocations | 0.000 | 0.272 | +0.272 |
| **TOTAL** | **51.115** | **101.093** | **+49.979** |

Model + optimizer + grad buffers (the three "no Δ" rows) account for
51 GB in both runs — that's the irreducible baseline.  CG+compile
adds **+50 GB on top**, all in 5 categories broken down by source
below.

### Big bucket 1: inductor graph-pool capture — +39.865 GB (+0.664 GB/layer)

By far the largest CG overhead.  All inductor triton-kernel allocations
during the captured forward go to the graph pool `(0, 1)`, where they
get pinned for replay.  The 60-layer model accumulates 2552 such
blocks, totalling ~40 GB.  Inductor aggregations grouped by user-code
call site:

| Role (user-code frame) | Blocks | GB total | MB/layer |
|---|---:|---:|---:|
| activations.py:88 (GELU/SiLU outputs saved for backward) | 600 | 18.741 | 312.4 |
| attention linear / QKV / proj (transformer_qwenimage.py:520/558-586) | 1020 | 16.716 | 278.6 |
| transformer block fwd (transformer_qwenimage.py:689/733) | 332 | 2.254 | 37.6 |
| attention module (attention.py:1741) | 120 | 2.082 | 34.7 |
| apply_rotary_emb + modulation | 480 | 0.071 | 1.2 |
| **Total inductor at (0, 1)** | **2552** | **39.865** | **664.4** |

**Why these accumulate per layer.**  In NOCG, inductor's triton kernels
allocate intermediate workspaces that get released to the caching
allocator's free list as soon as each kernel completes; the next layer
reuses the same addresses.  In CG, the captured graph *references*
every intermediate for replay, so the graph pool can't recycle them
between layers — the per-layer activations/workspaces accumulate
linearly with layer count.

**The 18.7 GB GELU/SiLU activation row** is the dominant contributor.
These are activation outputs saved on the autograd tape for backward
(inductor emits them as its own kernel outputs).  Two inductor call
sites generate 60 blocks each of 100 MB/layer (= 6 GB per call site,
12 GB total for the pair).  Activation checkpointing inside the
compiled forward (Lever 2 below) would free these at end-of-capture,
reducing the row to ~0.

### Big bucket 2: cg_runner static_inputs — +4.166 GB (caching pool)

Allocated at `cuda_graph_runner.py:419`:
```python
t.clone().detach().requires_grad_(t.requires_grad) for t in flat_live
```

These are the per-layer **static input buffers** — one set per captured
layer, ~70 MB per layer × 60 layers = 4.2 GB.  They live in the caching
pool (allocated outside `torch.cuda.graph`), so the per-layer snapshots
correctly show no graph-pool growth from these.

Each static input is a clone of the user's actual forward input, used
to feed the captured graph during replay (`_CudaGraphFunction.forward`
copies live values into them before `fwd_graph.replay()`).

**Possible reduction:** if multiple layers' static inputs share the
same shape/dtype, they could share a single static buffer per
(shape, dtype) pair via the graph pool — but this would require
reordering captures and is invasive.

### Big bucket 3: FSDP gradient bucket coloring — +3.561 GB (caching pool)

Allocated at `allocator.py:438` (`_color_and_allocate_slots`).  Seven
huge blocks (~508 MB each) allocated only in CG mode.

These are the FSDP `TracePoolAllocator` reduce-scatter gradient
buckets, allocated during the post-backward final callback.  In CG mode
the trace→optimized transition happens after the first backward, and
the bucket allocation happens then; in NOCG mode (or pre-CG) the
allocator either hasn't planned yet or uses a different allocation
strategy that doesn't materialise these blocks.

**Worth investigating:** the same bucket allocation may be avoidable
under CG by re-using the existing grad buffer storage as the bucket,
rather than allocating dedicated reduce-scatter buckets.

### Big bucket 4: flash_attn graph pool — +2.115 GB

180 blocks at ~35 MB/layer (graph pool).  Flash attention's forward
intermediates (V matrix accumulators, softmax workspace) are captured
into the graph pool like inductor's allocations.  These are
forward-only and don't need to persist for backward — but once the
graph captures them, they're pinned for replay.

### Small bucket 5: inductor warmup — +0.272 GB (caching pool)

Inductor allocations during the warmup iterations (before graph
capture starts) that did NOT get released to the free list.  These
show up at `ctjbu2*.py` (30 blocks, 0.144 GB) and a second inductor
code file `cwy3f5*.py` (15 blocks, 0.128 GB).  Tiny compared to the
graph-pool bucket, but indicates inductor keeps some persistent state
across the warmup→capture transition.

### Per-layer vs total: reconciliation

Earlier per-layer snapshots (cg_layer{0,1,2}_rank0_compile.pickle)
showed **+1.286 GB/layer** growth, but the full snapshot shows
**+50 GB / 60 layers = +0.833 GB/layer**.  The discrepancy comes from
capturing different points in time:

| Captured at | Per-layer snapshots | Full snapshots |
|---|---|---|
| When | end of forward capture for layer N | end of full training step (after backward + bucket coloring) |
| Layer 0..N forward graphs in pool | yes (cumulative) | yes (cumulative) |
| Backward graphs | no | yes (lazy, first backward) |
| Bucket coloring | no | yes |
| Optimizer state | no | yes (same in both) |

The +1.286 GB/layer figure represents **forward-capture** growth only;
the +0.833 GB/layer figure amortises the one-time backward + bucket
costs across the whole step.  Both are correct for their respective
points in time.

## Updated lever priorities

Given the full-snapshot breakdown, the lever priority order shifts:

### Lever 2 (was Lever 2): no_grad forward capture — biggest single win

**Expected savings: ~18.7 GB (the inductor activations row).**

The captured forward graph accumulates SavedVariables for backward —
the 18.7 GB inductor-activations row. These are dead weight at replay
time because `_capture_backward_and_run` already re-runs the forward
with grad enabled (via `replay_inputs = t.detach().clone().requires_grad_(...)`)
to build a fresh autograd tape for the bwd_graph.

Wrapping the capture call with `torch.no_grad()` drops all
SavedVariables from the fwd graph. This is strictly better than
`torch.utils.checkpoint` (Lever 2's original form):
- same fwd-graph savings as checkpoint
- no extra recompute in bwd capture (checkpoint's unpack hook
  would re-run the forward during the runner's bwd-capture re-run,
  doubling the recompute)
- no bwd-graph growth
- bonus: capture-time behavior matches replay-time behavior.
  `_CudaGraphFunction.forward` is a `torch.autograd.Function`, and
  PyTorch runs autograd Function forward methods with grad
  disabled — so `fwd_graph.replay()` already executes under no_grad
  at runtime. The legacy grad-enabled capture was inconsistent.

**Status: IMPLEMENTED (default ON).** Gated by
`MFSDP_CG_NO_GRAD_FWD` (default `1`; set to `0` for legacy
behavior). The wrap is applied only to the capture call, not to
the warmup (which still needs grad to settle FP8 scales via
`torch.autograd.grad`):

```python
# Inside capture_forward, around the CUDA graph capture:
no_grad_ctx = torch.no_grad() if _CG_NO_GRAD_FWD else contextlib.nullcontext()
with torch.cuda.graph(self.fwd_graph, pool=..., stream=...):
    with no_grad_ctx:
        out = self._call_module(static_inputs, tensor_names, frozen_kwargs)
```

Backward graph capture (`_capture_backward_and_run`) is unchanged:
it still re-runs the forward with `requires_grad_(True)` and grad
enabled, building its own autograd tape that gets captured into
`bwd_graph`. The bwd graph size is unaffected.

For the historical `torch.utils.checkpoint`-based approach (kept
for A/B testing), see `cuda_graph_checkpoint_design.md` and the
`MFSDP_CG_USE_CHECKPOINT` env var.

### Lever 4 (new): Share static input buffers across layers

**Expected savings: ~3-4 GB (the cg_runner row).**

If the same shapes recur across layers (true for transformer blocks
with identical attention dims), allocate one set of static I/O buffers
per (shape, dtype) pair instead of per layer.  Live inputs get copied
into the shared buffer before each replay.

### Lever 5 (new): Avoid duplicate gradient bucketing in CG

**Expected savings: ~3.5 GB (the fsdp_allocator row).**

Investigate why the FSDP `TracePoolAllocator` materialises 7 large
reduce-scatter buckets only in CG mode.  If the existing grad buffer
storage can be used directly for reduce-scatter (without a separate
bucket allocation), this disappears.

### Lever 6 (new): flash_attn recompute

**Expected savings: ~2 GB (the flash_attn row).**

Flash attention's forward intermediates account for 2.1 GB in the
graph pool.  A recompute variant (re-run flash-attn forward in
backward) would let the captured graph free them.

### Lever 3 (carried forward): TE fused ops

Still relevant for the inductor attention-linear bucket (16.7 GB), but
Lever 2 + Lever 4 + Lever 5 together would already close 25 GB of the
50 GB gap.
