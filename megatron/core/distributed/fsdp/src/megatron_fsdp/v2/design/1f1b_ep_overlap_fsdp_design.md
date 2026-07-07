# 1F1B (EP) Overlap — FSDP Integration Design

This document describes the FSDP-side contract required by the 1F1B EP-overlap
schedule (`combined_1f1b`), and how Megatron FSDP v2 fulfils it.
v1 behaviour is documented in [§7 — v1 reference](#7-v1-reference-megatron_fsdp).

---

## 0. Key concepts: `--overlap-moe-expert-parallel-comm` and `--delay-wgrad-compute`

### `--overlap-moe-expert-parallel-comm` (EP overlap)

Enables the **combined 1F1B schedule** (`combined_1f1b.py`).  Instead of running
each `TransformerLayer` forward and backward sequentially, the schedule
interleaves the **sub-module operations** (attn, mlp, moe_dispatch, moe_combine)
of two different micro-batches:

```
comm_stream:  combine_bwd  │  dispatch_fwd → dispatch_bwd  │  combine_fwd
comp_stream:  attn_fwd     │  mlp_bwd → mlp_bwd_dw → mlp_fwd │  attn_bwd → attn_bwd_dw
               ── micro-batch N ──►    ◄── micro-batch N-1 ──
```

- Forward micro-batch `N` (attn_fwd, mlp_fwd) runs **concurrently** with backward
  micro-batch `N-1` (mlp_bwd, attn_bwd).
- MoE all-to-all communication (dispatch/combine) overlaps with computation on
  the other stream.
- Global batch size and micro-batch size semantics are unchanged.  You have
  `mbs=4` — each forward or backward still operates on `mbs=4` tokens; the
  "parallelism" is between **different** micro-batches.

### `--delay-wgrad-compute` (delayed weight gradient)

Instructs Transformer Engine layers to **postpone weight gradient computation**
until an explicit `backward_dw()` call.  During `backward()`, only activation
gradients flow; weight gradients are computed later in `backward_dw()`.

**Requires** `--overlap-moe-expert-parallel-comm` (asserted in
`transformer_config.py:2802-2804`).  This lets the schedule overlap the delayed
wgrad kernels with other work (e.g., P2P communication):

```
comp_stream:  ... mlp_bwd (act grads only)  │  mlp_bwd_dw (wgrad)  │  ...
              ─── activation-grad-only ──►  ─── weight grad ──►
```

Without `delay_wgrad_compute`, wgrads are computed inline during `backward()`.
With it, they move to `backward_dw()`, creating additional overlapping
opportunities but also requiring FSDP's gradient reduction to wait until after
`backward_dw()` — the central challenge for FSDP integration.

### How they relate

| Flag | Requires | Relationship |
|---|---|---|
| `overlap_moe_expert_parallel_comm` | — | Enables the combined 1F1B schedule.  Without it, normal 1F1B or naive pipeline runs. |
| `delay_wgrad_compute` | `overlap_moe_expert_parallel_comm` | Moves wgrad computation from `backward()` to `backward_dw()`.  FSDP must defer `reduce_grad()` until after `backward_dw()`. |

---

## 1. Why the overlap schedule needs special FSDP handling

**Normal FSDP flow** (hooks fire on `TransformerLayer`):

```
TransformerLayer.forward()
  → pre-forward hook:  unshard params
  → actual compute
  → post-forward hook: reshard params
backward()
  → pre-backward hook:  unshard params
  → compute grads
  → post-backward hook: reshard params + reduce-scatter grads
```

**EP overlap flow** (calls sub-modules directly, bypassing `TransformerLayer.forward()`):

```
combined_forward_backward_step()
  → f_layer.attn.forward()     ← no TransformerLayer hook fires
  → b_layer.mlp.backward()     ← no TransformerLayer hook fires
  → f_layer.moe_dispatch.forward()
  → ...
```

Because the schedule invokes sub-modules (`attn`, `moe_dispatch`, `mlp`,
`moe_combine`) directly, the FSDP hooks registered on `TransformerLayer` are
**never triggered**.  FSDP must therefore expose a set of manual management
APIs that the schedule calls explicitly at the right moments.

---

## 2. FSDP API contract

### 2.1 Required attributes on the FSDP wrapper

| Hook | Type | Required for | Semantics | v2 impl |
|---|---|---|---|---|
| Root backward-phase setup | callable | All sharding strategies | Set `backward_phase = True`, unshard root params for backward. Called once before the overlapped forward+backward run. | `mfsdp_pre_backward_setup` |
| Root backward finalization | callable | All sharding strategies | Handle any modules whose per-module post-backward was skipped, drain async reduce-grad events, reset state. Called once after the overlapped run. | `mfsdp_post_backward_final_callback` |
| Per-layer post-forward | callable | `optim_grads_params` only | Reshard (release all-gathered parameters) for one layer after its forward ops complete. Called on the last forward node. | `mfsdp_post_forward_hook` |
| Per-layer post-backward | callable | `optim_grads_params` only | **Reshard + reduce gradients** for one layer after its backward ops complete. Called on the last backward node. Copies `.grad` → main grad buffer → reduce-scatter → installs `dist_param.grad` as DTensor. | `mfsdp_post_backward_hook` |
| Gradient sync suppression | context manager | All sharding strategies | Suppress gradient reduce-scatter for inner micro-batches. | `no_sync` (nullcontext in v2) |
| Sharding strategy access | object | All | Provides `data_parallel_sharding_strategy` for the schedule to decide per-layer hooks. | `ddp_config` / `_fsdp_param_groups` |

> **v1 vs v2 note**: In v1, the per-layer post-backward hook only releases
> parameters — gradient reduction happens separately via per-param
> `post_accumulate_grad_hook`.  In v2, `mfsdp_post_backward_hook` performs
> both `reshard()` **and** `reduce_grad()` in a single call.  Both v1 and v2
> have a per-layer post-forward hook that only does reshard (no grad work).
>
> `_replace_param_with_raw_if_needed()` is **not required** in v2.  See §2.2.

### 2.2 Required hooks for fine-grained sub-module management

When `overlap_moe_expert_parallel_comm=True`, two additional hook modes must be
enabled:

| Hook mode | Required for | Effect |
|---|---|---|
| Fine-grained pre-forward unshard | All strategies | Register `mfsdp_forward_pre_hook` on **every sub-module** (not just FSDP units), because the schedule calls sub-modules directly. |
| Fine-grained pre-backward unshard | All strategies | Register `mfsdp_pre_backward_setup` via `register_multi_grad_hook` on each sub-module's output tensor, ensuring params are unsharded before backward compute. |

> **Why `_replace_param_with_raw_if_needed()` is not required in v2**: v2 keeps
> DTensor params directly on the module and manages them via
> `unshard()`/`reshard()`.  Fine-grained pre-forward hooks ensure parameters are
> all-gathered before each sub-module compute.  v1 required the swap because its
> optimizer-facing params were separate from module params.

### 2.3 Per-layer post-forward and post-backward hooks (`optim_grads_params` only)

The schedule plan calls `set_fsdp_reshard_hooks(post_forward_hook, post_backward_hook)`
on each `TransformerLayerSchedulePlan` to wire:

- **Post-forward**: attached to the **last forward node**
  (`moe_combine` for MoE, `mlp` otherwise).  Calls `mfsdp_post_forward_hook(layer)`,
  which reshard parameters (release all-gathered buffer, install DTensor dist_params).

- **Post-backward**: attached to the **last backward node** (`attn`).
  Calls `mfsdp_post_backward_hook(layer)`, which does **both**:
  1. Reshard parameters (release all-gathered buffer, install DTensor dist_params).
  2. Reduce gradients: copy `.grad` → main grad buffer → reduce-scatter →
     install `dist_param.grad = dist_grad` (DTensor).

These are needed because the overlap schedule bypasses `TransformerLayer.forward()`,
so the normal forward/backward hooks never fire.

---

## 3. Runtime call sequence (v2)

### 3.1 Overall flow

The schedule in `combined_1f1b_schedule_for_no_pipelining()` orchestrates four
phases per training step (file: `megatron/core/pipeline_parallel/combined_1f1b.py`):

```
combined_1f1b_schedule_for_no_pipelining():
│
├─ 1. combined_forward_backward_step(  # first microbatch, fwd only
│       f_model=model, b_model=None, ...
│     )
│     No FSDP root-level calls for this phase.  Fine-grained pre-forward
│     hooks (registered on every sub-module) handle per-sub-module unshard.
│
├─ 2. with no_sync_func():   # inner micro-batches
│     │
│     └─ combined_forward_backward_step(
│          f_model=model, b_model=model, ...
│        )
│        │
│        ├─ 2a. pre_backward_fn(root_module)                # mfsdp_pre_backward_setup
│        │       → ctx.backward_phase = True
│        │       → root_module.unshard(bwd_pass=True)
│        │       → skip_final_callback=True (no auto-enqueue)
│        │
│        ├─ 2b. for each layer: layer_plan.set_fsdp_reshard_hooks(
│        │         mfsdp_post_forward_hook, mfsdp_post_backward_hook
│        │       )
│        │
│        ├─ 2c. TransformerModelChunkSchedulePlan.run(f, b, ...)
│        │       │
│        │       │  During the run, per-layer hooks fire:
│        │       │   - After last fwd node: mfsdp_post_forward_hook(layer)
│        │       │       → layer.reshard()
│        │       │   - After last bwd node: mfsdp_post_backward_hook(layer)
│        │       │       → layer.reshard() + layer.reduce_grad()
│        │       │
│        │       │  Also, the normal autograd post-backward hook
│        │       │  (RegisterFSDPBackwardFunction) is DISABLED for all FSDP
│        │       │  modules when delay_wgrad_compute=True.  See §3.4.
│        │
│        └─ 2d. mfsdp_post_backward_final_callback(root_module)
│                → Handles ALL fsdp modules whose per-module post-backward
│                  was skipped (e.g., nested TEGroupedMLP expert modules).
│                  Runs after all backward_dw() calls complete.
│
└─ 3. combined_forward_backward_step(  # last batch, bwd only
       f_model=None, b_model=model, ...
     )
     │
     ├─ 3a. pre_backward_fn(root_module)
     ├─ 3b. TransformerModelChunkSchedulePlan.run(None, b, ...)
     └─ 3c. mfsdp_post_backward_final_callback(root_module)
```

### 3.2 Single-layer execution order

Inside `TransformerLayerSchedulePlan.run()` (file:
`megatron/core/models/common/model_chunk_schedule_plan.py`), each overlapped
layer pair executes:

```
line 263: b_layer.moe_combine.backward(grad)     ← MoE combine backward
line 267: f_layer.attn.forward(f_input)          ← attention forward
line 270: b_layer.mlp.backward(grad)             ← autograd backward:
                                                   1. fine-grained pre-bwd hook → unshard MLP
                                                   2. TE mlp backward (act grads only if delay_wgrad)
                                                   3. autograd post-bwd hook → SKIPPED (delay_wgrad)
line 274: f_layer.moe_dispatch.forward(f_input)  ← MoE dispatch forward
line 277: b_layer.mlp.backward_dw()              ← delayed TE mlp wgrad
line 278: b_layer.moe_dispatch.backward(grad)    ← MoE dispatch backward
line 285: f_layer.mlp.forward(f_input)           ← MLP forward
line 289: f_layer.moe_combine.forward(f_input)   ← MoE combine forward
line 292: b_layer.attn.backward(grad)            ← autograd backward:
                                                   1. fine-grained pre-bwd hook → unshard attn
                                                   2. TE attn backward (act grads only if delay_wgrad)
                                                   3. autograd post-bwd hook → SKIPPED (delay_wgrad)
line 301: b_layer.attn.backward_dw()             ← delayed TE attn wgrad
                                                   → fires mfsdp_post_backward_hook(layer)
                                                   → layer.reshard() + layer.reduce_grad()
```

### 3.3 Fine-grained pre-backward hook registration

In v2, fine-grained hooks are registered via the `fine_grained_hooks` parameter
of `fully_shard()` (wired to `config.overlap_moe_expert_parallel_comm` in
`mcore_fsdp_adapter.py:276`).

**Pre-forward** (`_register_forward_pre_hook(fine_grained=True)` in `fully_shard.py:124`):
- Registers `mfsdp_forward_pre_hook` on **every sub-module** of each FSDP unit.
- When the schedule calls `f_layer.attn.forward()`, the hook on the `attn`
  sub-module fires → resolves parent FSDPModule → `unshard()`.

**Pre-backward** (`_register_backward_pre_hook(fine_grained=True)` in `fully_shard.py:131`):
- Registers `mfsdp_pre_backward_setup` via `register_multi_grad_hook` on
  every sub-module's output tensor.
- When backward reaches a sub-module, the hook fires → resolves parent
  FSDPModule → `unshard(bwd_pass=True)` → resets `post_backward_issued`.

### 3.4 Critical design constraint: `_register_backward_hook` and `delay_wgrad_compute`

The normal autograd post-backward hook (`_register_backward_hook` at `fully_shard.py:141`)
inserts `RegisterFSDPBackwardFunction` into the autograd graph.  During
backward, this function fires and calls `mfsdp_post_backward_hook(module)`,
which does `reshard()` + `reduce_grad()`.

When `delay_wgrad_compute=True`, this hook fires **before** `backward_dw()`
completes the weight gradient computation.  This causes `reduce_grad()` to
reduce only activation gradients, but not weight gradients.  The subsequent
`backward_dw()` then writes to the already-resharded DTensor params,
corrupting `.grad`.

**Fix**: When `skip_backward_callback=True` (wired to `config.delay_wgrad_compute`
in `mcore_fsdp_adapter.py:303`), the autograd `RegisterFSDPBackwardFunction` is
**skipped** for this module.  Per-layer `reshard()` + `reduce_grad()` still fires
manually for the TransformerLayer via `set_fsdp_reshard_hooks` →
`mfsdp_post_backward_hook`.  Remaining nested modules (e.g., TEGroupedMLP, root)
are handled by `mfsdp_post_backward_final_callback` (called at
`combined_1f1b.py:629`), which runs after all `backward_dw()` calls complete.

For EP overlap **without** `delay_wgrad_compute`, the autograd hook fires at
the right time (all grads are ready inline during backward), so it is
**not** skipped.

### 3.5 Hook fire count — what to expect

`mfsdp_forward_pre_hook` fires on **every** `Module.__call__()`, not just on
the FSDP unit modules.  Because the EP overlap schedule invokes sub-modules
directly, and fine-grained hooks are registered on all sub-modules, the **root
pre-forward hook fires multiple times per forward micro-batch** — once for each
root-child sub-module entered via `__call__`.

In a typical GPT model, these root-child calls happen in:

- `PreProcessNode.forward()` → `gpt_model._preprocess()`:
  - `self.embedding(...)` — calls `embedding.__call__()`
  - `self.rotary_pos_emb(...)` — calls `rotary_pos_emb.__call__()`
- `PostProcessNode.forward()` → `gpt_model._postprocess()`:
  - `self.decoder.final_layernorm(...)` — calls `final_layernorm.__call__()`

That is **3** root forward pre-hook fires per forward micro-batch.

Similarly, the fine-grained **pre-backward** hook fires on each sub-module
whose backward is entered.  For a root-child sub-module (e.g., `embedding`,
`final_layernorm`), this triggers `mfsdp_pre_backward_setup` which deduplicates
via `_fsdp_pre_backward_done` — so the root backward setup fires **once per
backward micro-batch**, on the first root-child sub-module whose backward runs.

**Example**: World size `W=4`, `EP=2`, `mbs=4`, `global_batch_size=32`:

```
DP = W / EP = 2 (dense DP = 4, but micro-batching uses the inter-EP DP)
num_microbatches = 32 / (4 × 2) = 4
```

The 1F1B schedule produces 5 phases — 4 forward + 4 backward.  Expected hook
fires (root-level only):

```
Phase 0 (warmup):   FWD[0]                              → F F F
Phase 1 (overlap):  FWD[1] ─ BWD[0]                    → F F F  B
Phase 2 (overlap):  FWD[2] ─ BWD[1]                    → F F F  B
Phase 3 (overlap):  FWD[3] ─ BWD[2]                    → F F F  B
Phase 4 (cleanup):           BWD[3]                     →        B

Total per step: 12 F, 4 B
```

A typical log snippet would show:

```
=== Starting forward pass ===  (×3, phase 0)
=== Starting forward pass ===  (×3, phase 1)
=== Starting backward pass ===
=== Starting forward pass ===  (×3, phase 2)
=== Starting forward pass ===  (×3, phase 3)
=== Starting backward pass ===
=== Starting backward pass === (×2, phases 1-2-3-4 depending on dedup timing)
...
```

The **6 forward per 1 backward** pattern in the log happens naturally because
a typical print grouping captures ~2 forward phases worth of hooks before
each backward.  This is expected and does **not** mean anything is wrong.  The
root pre-forward hook is designed to be called repeatedly — it is idempotent
(both the root-level bookkeeping and the per-module `unshard()` are safe to
repeat).

---

## 4. Hook behavior — v2 implementation

The EP overlap schedule wires four hook functions defined in
`megatron/core/distributed/fsdp/src/megatron_fsdp/v2/hooks.py`.
Their schedule-side wiring is in `combined_1f1b.py` (see §3.1).

### 4.1 Root backward-phase setup — `mfsdp_pre_backward_setup`

- **File**: `megatron/core/distributed/fsdp/src/megatron_fsdp/v2/hooks.py:194`
- **When**: called once in `combined_forward_backward_step()` before the
  schedule plan `.run()` (line 468 in `combined_1f1b.py`), with
  `skip_final_callback=True`.
- **What it does**:
  1. Resolves target FSDPModule via `_find_fsdp_target(hook_module)`.
  2. Deduplicates via `_fsdp_pre_backward_done` flag.
  3. If root: sets `ctx.backward_phase = True`, advances backward module tracker.
  4. Does **not** auto-enqueue `mfsdp_post_backward_final_callback`
     (`skip_final_callback=True`) — the schedule calls it manually.
  5. Calls `module.unshard(bwd_pass=True)` — all-gathers params for backward compute.
  6. Sets `module.post_backward_issued = False` — resets per-module bookkeeping.
  7. Resets TE gradient-accumulation fusion flags.

### 4.2 Root backward finalization — `mfsdp_post_backward_final_callback`

- **File**: `megatron/core/distributed/fsdp/src/megatron_fsdp/v2/hooks.py:252`
- **When**: called once in `combined_forward_backward_step()` after the
  schedule plan `.run()` (line 629 in `combined_1f1b.py`).
- **What it does**:
  1. Iterates `reversed(ctx.forward_order)` — all FSDP modules in the tree.
  2. For each module with `post_backward_issued = False`:
     - `module.reshard()` — release all-gathered buffers, install DTensor params.
     - `module.reduce_grad()` — copy grads → main grad buffer → reduce-scatter →
       set `dist_param.grad = dist_grad` (DTensor).
  3. Drains pending async reduce-grad events (`event.wait()` + release buffers).
  4. Resets root/context state (`backward_phase = False`, clears
     `backward_done_modules`, `_fsdp_pre_backward_done` flags).
  5. Transitions bucket allocator from trace → optimized plan (first micro-batch).

### 4.3 Per-layer post-forward (reshard only) — `mfsdp_post_forward_hook`

- **File**: `megatron/core/distributed/fsdp/src/megatron_fsdp/v2/hooks.py:133`
- **When**: wired via `set_fsdp_reshard_hooks()`, fires after the last
  forward node of each layer.
- **What it does**:
  1. Calls `module.reshard()` — release all-gathered params after forward.
  2. Has a backward-phase guard: if `ctx.backward_phase` is active and this
     module is the current backward module, skips reshard (activation
     recomputation case — params are still needed).

### 4.4 Per-layer post-backward (reshard + reduce_grad) — `mfsdp_post_backward_hook`

- **File**: `megatron/core/distributed/fsdp/src/megatron_fsdp/v2/hooks.py:225`
- **When**: wired via `set_fsdp_reshard_hooks()`, fires after the last
  backward node (`attn`) of each layer.  Also potentially fired by
  `RegisterFSDPBackwardFunction` during autograd backward (but see §3.4).
- **What it does**:
  1. Adds module to `backward_done_modules`.
  2. Advances the backward module tracker.
  3. Calls `module.reshard()` — release all-gathered params.
  4. If sharding strategy is `optim_grads` or `optim_grads_params`:
     calls `module.reduce_grad()` — copy grads → reduce-scatter → install DTensor grads.
  5. Sets `module.post_backward_issued = True`.

### 4.5 Gradient sync suppression — `no_sync()`

- **v2 status**: Returns `nullcontext` (no-op).
  This means gradient reduce-scatter fires on every micro-batch, including
  inner ones.  The standard 1F1B pattern (suppress sync for inner
  micro-batches, sync on the last) is not yet implemented for v2.

---

## 5. Constraints enforced at init time

When `overlap_moe_expert_parallel_comm=True`, the following constraints apply:

| Constraint | Enforced? | Rationale |
|---|---|---|
| `overlap_moe_expert_parallel_comm` required by `delay_wgrad_compute` | ✅ `transformer_config.py:2802-2804` | Delayed wgrad only works in the 1F1B overlap schedule. |
| `delay_wgrad_compute` mutually exclusive with `overlap_dispatch_backward_with_experts_wgrad` | ✅ `transformer_config.py:2819-2822` | Two different wgrad-deferral strategies; choose one. |
| CUDA graph scope on `moe` and `mlp` blocked | ✅ `transformer_config.py:2790-2798` | Partial CUDA graph scopes conflict with fine-grained schedule. |
| TE >= 2.3.0 for `delay_wgrad_compute` | ✅ `megatron/training/arguments.py:1937` | `delay_wgrad_compute` kernel support requires TE >= 2.3.0. |
| Only `GPTModel` supported | ✅ `combined_1f1b.py:505` | Schedule plan build checks `isinstance(unwrapped_model, GPTModel)`. |
| Interleaved PP + FSDP blocked | ✅ `combined_1f1b.py:317-321` | Multi-chunk models not yet supported with EP overlap. |
| `fsdp_double_buffer = False` | ❌ not enforced in v2 | Double buffering is incompatible with per-sub-module parameter management. |
| `fsdp_unit_modules` compatibility | ❌ not enforced in v2 | The schedule expects each layer to be an FSDPModule (wrapped by `fully_shard()`), but this is not validated at init time. |

---

## 6. Key code locations

| Component | File |
|---|---|
| **v2 implementation** | |
| FSDP API definitions | `megatron/core/distributed/fsdp/src/megatron_fsdp/v2/hooks.py` |
| FSDP module (`FSDPModule`) | `megatron/core/distributed/fsdp/src/megatron_fsdp/v2/fsdp_module.py` |
| `fully_shard()` entry point | `megatron/core/distributed/fsdp/src/megatron_fsdp/v2/fully_shard.py` |
| Parameter groups & buffers | `megatron/core/distributed/fsdp/src/megatron_fsdp/v2/param_group.py` |
| FSDP adapter (config wiring) | `megatron/core/distributed/fsdp/mcore_fsdp_adapter.py` |
| **Shared (v1 + v2)** | |
| Schedule orchestration | `megatron/core/pipeline_parallel/combined_1f1b.py` |
| Schedule plan (layer-level) | `megatron/core/models/common/model_chunk_schedule_plan.py` |
| Schedule plan (chunk-level) | `megatron/core/models/common/model_chunk_schedule_plan.py` |
| Schedule dispatch | `megatron/core/pipeline_parallel/schedules.py` |
| Fine-grained callables | `megatron/core/models/gpt/fine_grained_callables.py` |
| Config definitions | `megatron/core/model_parallel_config.py`, `megatron/core/transformer/transformer_config.py` |
| **Tests** | |
| M-FSDP v2 EP overlap e2e | `tests/unit_tests/distributed/megatron_fsdp/v2/test_mcore_nd_parallel.py` |
| delay_wgrad_compute unit test | `tests/unit_tests/a2a_overlap/test_delay_wgrad_compute.py` |
| FSDP 1F1B overlap test | `tests/unit_tests/a2a_overlap/test_fsdp_1f1b_overlap.py` |

---

## 7. v1 reference (`megatron_fsdp`)

> This section describes the Megatron FSDP v1 (`megatron_fsdp`) implementation
> for reference.  v1 uses different internal APIs than v2.

### 7.1 Hook behavior — v1

> All v1 hooks live in `megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py`.

#### 7.1.1 Root backward-phase setup — `pre_backward()`

- **V1 impl**: `_root_pre_backward(module=None, skip_backward_hook=True)`
- **When**: called once in `combined_forward_backward_step()` before the
  schedule plan `.run()`.
- **What it does**:
  1. Sets `_root_pre_backward_hook_issued = True` (idempotency guard).
  2. For `optim_grads_params`: sets all sub-module `_training_state` to
     `PRE_BACKWARD`, marks all AG buckets as releasable.
  3. Tracks params that require gradient handling via
     `_params_require_handle_grad`.
  4. Does NOT auto-enqueue `_root_post_backward` — the schedule calls it
     manually via `post_backward()`.

#### 7.1.2 Root backward finalization — `post_backward()`

- **V1 impl**: `_root_post_backward()`
- **What it does**:
  1. Processes any remaining unhandled gradients.
  2. Launches async reduce-scatter for gradient-sharding strategies.
  3. Resets root state: `_root_pre_backward_hook_issued = False`,
     increments `microbatch_count`.

#### 7.1.3 Per-layer post-forward (reshard only) — `post_forward_release_module()`

- **V1 impl**: `_post_forward(module, input=None, output=None)`
- **When**: wired via `set_fsdp_reshard_hooks()`, fires after the last
  forward node of each layer.
- **What it does**:
  1. If `_training_state == PRE_BACKWARD`: lazy release (activation
     recomputation case).
  2. Otherwise: release params via `release_module_parameters(module, bwd=False)`,
     transition to `IDLE`.

#### 7.1.4 Per-layer post-backward (release only, no grad reduction) — `post_backward_release_module()`

- **V1 impl**: `_post_backward_release_module(module)`
- **When**: wired via `set_fsdp_reshard_hooks()`, fires after the last
  backward node (`attn`) of each layer.
- **What it does**:
  - Releases params for both backward and forward passes:
    `release_module_parameters(module, bwd=True)` and
    `release_module_parameters(module, bwd=False)`.
  - Transitions all sub-modules to `IDLE` state.

- **Key difference from v2**: v1's per-layer post-backward hook does **not**
  perform gradient reduction.  Gradient processing (reduce-scatter) is handled
  separately by the per-param `post_accumulate_grad_hook` that fires
  independently.  In contrast, v2's `mfsdp_post_backward_hook` combines
  **both** reshard and gradient reduction in a single call.

- With `delay_wgrad_compute`: gradient processing for params with
  `skip_backward_post_hook=True` is deferred to `backward_dw()` via
  `setup_delayed_wgrad_acc_hook` (`megatron_fsdp.py:76-101`).

#### 7.1.5 Gradient sync suppression — `no_sync()`

- **V1 impl**: `MegatronFSDP.no_sync()` (context manager, sets
  `is_last_microbatch = False` on enter, `True` on exit).
- **When**: wraps the inner micro-batches (all except first forward-only
  and last backward-only).
- **Effect**: prevents gradient reduce-scatter for non-final micro-batches.

#### 7.1.6 `_replace_param_with_raw_if_needed()`

- **V1 impl**: `MegatronFSDP._replace_param_with_raw_if_needed()`
- **When**: called once at the start of the schedule (`combined_1f1b.py:179`),
  before any layer access.
- **Effect**: swaps the distributed (optimizer-managed) `DTensor` parameters
  back to raw `nn.Parameter` tensors so the schedule can call sub-modules
  directly.  In v1, optimizer-facing params are separate from module params,
  so this swap is mandatory.  v2 does not require this because DTensor
  params live on the module directly and are managed via `unshard()`/`reshard()`.

### 7.2 Key code locations (v1)

| Component | File |
|---|---|
| FSDP API definitions | `megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py` |
| Delayed wgrad hook setup | `megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py:76-101` |
| Param attribute preservation | `megatron/core/distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py:2743-2748` |
