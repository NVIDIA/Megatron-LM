# Mamba / SSM Inference and Triton Production Rules

Hybrid Mamba models (Nemotron-H, Nano) push most inference time through hand-
written Triton kernels, so kernel hygiene matters more here than anywhere else in
the stack. The Triton rules in this document are not Mamba-specific — they apply
to any kernel on the inference path.

Source commits: `ab2b33d5` (#4397), `f29c747f` (#5608), `9b4074b5` (#4764),
`411a5d8b` (#5863), `648bc011` (#5866).

## The Triton specialization trap

This is the highest-value rule in the skill relative to how easy it is to get
wrong. Commit `f29c747f` is four lines:

```diff
 def _tensor_get_slice_after_kernel(
-    INPUT_BATCH_SIZE: tl.constexpr,
-    OUTPUT_BATCH_SIZE: tl.constexpr,
+    INPUT_BATCH_SIZE,
+    OUTPUT_BATCH_SIZE,
     ROW_SIZE: tl.constexpr,
     BLOCK_SIZE: tl.constexpr,
```

A `tl.constexpr` parameter is **baked into the compiled kernel**, so every
distinct value produces a separate compilation. `INPUT_BATCH_SIZE` is the number
of active requests, which changes essentially every step. The kernel was
JIT-compiling on the hot path, on nearly every step.

`ROW_SIZE` and `BLOCK_SIZE` correctly stay `constexpr` — they depend only on the
fixed state shape and are needed for `tl.arange` bounds and loop unrolling.

Current state in
[tensor_ops.py](megatron/core/inference/contexts/attention_context/triton/tensor_ops.py):

```python
@triton.jit
def _tensor_get_slice_after_kernel(
    INPUT_TENSOR,
    OUTPUT_TENSOR,
    POS_ON_DEVICE,
    INPUT_BATCH_SIZE,
    OUTPUT_BATCH_SIZE,
    ROW_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
```

### The full decision rule

| Value | How to pass it |
|---|---|
| Block/tile size, `tl.arange` bound, unroll count, `d_conv` | `tl.constexpr` |
| Per-step count, but you want one compilation | plain arg + `do_not_specialize` |
| Per-step count, and the kernel is CUDA-graphed | fixed-address GPU tensor, `tl.load` it |

The subtle middle case: **even plain `int` arguments get specialized** by
`@triton.jit` on divisibility-by-16 and `== 1`. So demoting from `constexpr` is
not always sufficient. If the value varies per step, be explicit:

```python
# megatron/core/ssm/ops/mamba_ssm.py
@triton.jit(do_not_specialize=["batch"])
def _selective_scan_update_kernel(
```

The third case is strictly better when applicable: passing the value as a 0-d
tensor and loading it inside the kernel avoids specialization entirely *and* keeps
CUDA graphs valid, since the address is fixed while the value may change between
replays.

## Never autotune on the hot path

`@triton.autotune` times every config on first sight of each new key. In
production that is a multi-second stall and a source of nondeterminism.

Two mitigations, both in use:

**Keep config lists short.** Commit `9b4074b5` trimmed the varlen causal conv from
four configs to two, keeping one from each regime. The comment explains why both
regimes exist, which is the useful part:

```python
# megatron/core/ssm/ops/causal_conv1d_varlen.py
# Two block-dim regimes:
#   1. vLLM-style: small BLOCK_T, large BLOCK_C, pipelined. Many small programs maximize
#      GPU occupancy, BLOCK_C=256 fully fills HBM transactions, sequence-pure programs
#      avoid in-block boundary branching. Usually wins at moderate-to-large conv_dim.
#   2. Large-block fallback: bigger tiles, fewer programs. Can win for small conv_dim
#      where vLLM's regime over-parallelizes, or when launch overhead dominates.
```

**Route everything through `autotune_configs`.** In
[determinism.py](megatron/core/ssm/ops/determinism.py), this is the single funnel:
in deterministic mode it either enables cached autotuning
(`TRITON_CACHE_AUTOTUNING=1`, Triton >= 3.4.0) or picks the single cheapest config
by `block_product * stages`, tie-broken on warp count — so exactly one
compilation and no timing sweep.

```python
def autotune_configs(configs):
    if not configs or not use_deterministic_mode():
        return configs
    if TRITON_HAS_CACHE_RESULTS and os.environ.get("TRITON_CACHE_AUTOTUNING") == "1":
        return configs
    ...
    return [min(configs, key=_estimate_config_cost)]
```

Wrap any new autotuned kernel in `autotune_configs([...])`, not a bare
`configs=[...]`.

## Fused gather-plus-scatter instead of materialized intermediates

Prefix caching needs to extract Mamba states at block boundaries. The original
implementation was a three-step tensor dance: the chunk scan gathered requested
chunks internally, `mamba_mixer` copied the result into scratch, and conv windows
were extracted with a separate PyTorch gather plus `clamp_` plus `transpose` plus
`copy_`. That materialized an intermediate tensor and made two passes over HBM.

Commit `648bc011` replaced it with two kernels in
[intermediate_extraction.py](megatron/core/ssm/ops/intermediate_extraction.py).
The module docstring states the design contract:

```
These replace the two-step ``states[indices]`` (dense gather) + ``.copy_()``
(scratch write) pattern with a single kernel that:

1. Reads a runtime ``real_count`` from a fixed-address GPU tensor.
2. For each slot ``i < real_count``, gathers the source row indexed by the
   per-slot index/position and writes it directly into the destination scratch.
3. For each slot ``i >= real_count``, returns immediately (no work, no write).

This is CUDA-graph safe: the launch grid is sized at capture time to the maximum
possible slot count, but per-program execution is data-conditional on the
runtime ``real_count``, so padded slots cost almost nothing.
```

The gating idiom, which generalizes to any graphed kernel with a variable count:

```python
pid_slot = tl.program_id(0)
pid_col = tl.program_id(1)

real_count = tl.load(real_count_ptr).to(tl.int32)
if pid_slot >= real_count:
    return
```

The conv kernel adds two more techniques worth stealing:

**Fold the transpose into the write address.** The old code did
`.transpose(1, 2).copy_()`; the kernel just indexes the destination differently:

```python
for j in tl.static_range(D_CONV):
    p_raw = abs_pos - D_CONV + j
    p = tl.maximum(0, tl.minimum(p_raw, seq_len - 1))
    src = src_ptr + p.to(tl.int64) * src_stride_s + c_idxs.to(tl.int64) * src_stride_c
    dst = out_ptr + slot_base + c_idxs.to(tl.int64) * D_CONV + j
    val = tl.load(src, mask=c_mask)
    tl.store(dst, val, mask=c_mask)
```

**`D_CONV` is `constexpr` so the window loop is `tl.static_range`** — fully
unrolled, no dynamic loop. This is the legitimate use of `constexpr`: `d_conv` is
a model constant.

The scan correspondingly got simpler. `ssd_combined.py` no longer gathers
internally; `return_raw_states` hands the caller the full state tensor and the
caller extracts what it needs.

### Publishing the count

```python
# megatron/core/inference/contexts/attention_context/mamba_metadata.py
# Publish real_count to the fixed-address GPU tensor the scatter
# kernels consult. fill_ is async (no host sync) and keeps the tensor
# at the same address captured graphs reference.
self._intermediate_real_count_buffer.fill_(self.intermediate_count)
self.intermediate_real_count = self._intermediate_real_count_buffer
```

## Right-size scratch to the real per-step bound

The extraction scratch was sized `MAX_INTERMEDIATE_OFFSETS_PER_REQUEST (=3) *
max_requests`. But a step processes at most `max_tokens` tokens, and a state can
only be extracted at a block boundary — one per `block_size_tokens`. So there are
two independent bounds and the truth is the tighter one:

```python
# megatron/core/inference/contexts/dynamic_context.py
# Per-step upper bound on Mamba intermediate-state extractions, shared with
# MambaMetadata and MambaSlotAllocator so scratch/metadata buffers and the
# budget accounting agree. Bounded both by the token budget (one block
# boundary per block_size_tokens) and by the request budget
# (MAX_INTERMEDIATE_OFFSETS_PER_REQUEST per request);
token_based_count = math.ceil(self.max_tokens / self.block_size_tokens)
request_based_count = MAX_INTERMEDIATE_OFFSETS_PER_REQUEST * self.max_requests
self.max_mamba_intermediate_states_per_step = min(token_based_count, request_based_count)
```

For low-concurrency long-context configs this is an order of magnitude: 1 request
with 16384 tokens once reserved 65 scratch slots a single request could never
fill. Since scratch is reserved *before* the durable prefix cache, every wasted
slot directly shrinks the cache.

Two structural points that make this pattern work:

**One value, shared everywhere.** `max_mamba_intermediate_states_per_step` flows
into `MambaMetadata(max_intermediate_count=...)`,
`MambaSlotAllocator.max_intermediate_count`, and the byte budget. When a bound is
duplicated it drifts; when it is shared, a wrong bound fails loudly.

**Fail at config time, not at OOM.** If reserving scratch leaves fewer than one
durable slot, the context raises `ValueError` naming the budget and advising which
knob to reduce.

The same commit fixed a hardcoded `mamba_chunk_size = 128`, now read from config.
States can only be extracted at multiples of the chunk size the kernel actually
runs with, so a hardcoded value silently skipped valid boundaries for any other
chunk size.

## Bound per-step work by the bucket, not the global max

Commit `9b4074b5` is essentially this rule applied to Nemotron prefill. The
extraction bookkeeping looped over the *global* `max_intermediate_count`, so every
prefill step did work proportional to the largest possible batch even in a tiny
graph bucket:

```diff
-        max_count = self.max_intermediate_count
+        max_count = padded_prefill_count * MAX_INTERMEDIATE_OFFSETS_PER_REQUEST
...
-                    self._intermediate_chunk_indices_buffer[real_count:].fill_(0)
+                    self._intermediate_chunk_indices_buffer[real_count:max_count].fill_(0)
```

Same for the scratch writes: `intermediate_ssm_out.copy_(...)` became
`intermediate_ssm_out[:n].copy_(...)`. Whenever you see a `fill_` or `copy_` over
a whole preallocated buffer on the per-step path, ask what the real bound is.

## Inner-loop micro-optimizations

From `ab2b33d5`, in the single-token decode SSM update:

**Use `exp2` for `exp`.** Hardware has a native `exp2`:

```python
@triton.jit
def fast_exp(x):
    """
    Fast calculation of exponent via exponent of 2.
    """
    LOG2E = tl.constexpr(1.4426950408889634)
    return tl.math.exp2(LOG2E * x)
```

**`torch.empty` when the kernel overwrites every row.** Zero-init is wasted HBM
bandwidth. Applied to the squared-ReLU output in
[activations.py](megatron/core/inference/moe/activations.py).

**Specialize the launch shape on compute capability.** The tile and warp choice
was derived from `dstate` alone; Blackwell wants a different point:

```python
# megatron/core/ssm/ops/mamba_ssm.py
is_blackwell = torch.cuda.get_device_capability(x.device)[0] >= 10
...
else:
    # dstate > 64
    if is_blackwell:
        # Optimized for B200 with dstate>64
        BLOCK_SIZE_M, num_warps = 32, 8
    elif dstate <= 128:
        BLOCK_SIZE_M, num_warps = 4, 4
```

## File map

| File | Role |
|---|---|
| [mamba_mixer.py](megatron/core/ssm/mamba_mixer.py) | The mixer. Prefill: conv, chunk scan, state update, extraction. Decode: `selective_state_update` + `causal_conv1d_update`. |
| [ssd_combined.py](megatron/core/ssm/ops/ssd_combined.py) | `mamba_chunk_scan_combined_varlen`, the packed varlen chunk scan. `return_raw_states` exposes intermediates. |
| [mamba_ssm.py](megatron/core/ssm/ops/mamba_ssm.py) | `selective_state_update`, the decode-path kernel. |
| [intermediate_extraction.py](megatron/core/ssm/ops/intermediate_extraction.py) | Fused conditional gather+scatter for prefix-cache extraction. |
| [causal_conv1d_varlen.py](megatron/core/ssm/ops/causal_conv1d_varlen.py) | Varlen depthwise causal conv with fused SiLU (prefill). |
| [determinism.py](megatron/core/ssm/ops/determinism.py) | `autotune_configs` — the single autotuning funnel. |
| [mamba_metadata.py](megatron/core/inference/contexts/attention_context/mamba_metadata.py) | Per-step Mamba metadata in fixed-address buffers. |
| [mamba_slot_allocator.py](megatron/core/inference/contexts/mamba_slot_allocator.py) | Durable state cache plus extraction scratch. |

## How these are tested

The pattern is consistent and worth imitating:

**Compare against an obvious reference.** `_ref_conv` is a plain nested-loop
PyTorch implementation; the kernel is checked against it with
`torch.testing.assert_close`. A fast kernel with no slow reference has no test.

**Assert the gating with a sentinel.** Fill the output with `12345.0`, run with
`real_count < max_count` where the trailing indices are *valid but must not be
gathered*, then assert both that `out[:real_count]` matches the reference and that
`torch.all(out[real_count:] == sentinel)`. The second half proves padded slots
produced no HBM writes — the actual performance claim.

**Re-derive formulas independently.** `test_max_intermediate_states_per_step_formula`
recomputes `min(token_based, request_based)` in the test, asserts it matches, and
asserts the same value is shared by the allocator, the metadata, and the buffer
shape. It covers both regimes explicitly, including one where the request bound is
strictly tighter.

**Cover the boundary.** `test_scatter_conv_sub_dconv_clamp` uses `pos=2` with
`d_conv=4` so the window start goes negative, checking all three out-of-range
positions clamp to token 0.

**Test the error path.** A too-small budget must raise `ValueError`, not silently
OOM later.

Tests: `tests/unit_tests/ssm/ops/test_ssd_combined.py`,
`tests/unit_tests/inference/contexts/test_dynamic_prefix_caching.py`.
