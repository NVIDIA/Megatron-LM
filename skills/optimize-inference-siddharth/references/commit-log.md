# Commit Provenance

Every commit by `sidsingh@nvidia.com` on `main` since 2026-01-01, grouped by the
pattern it demonstrates. Use this to read the primary example of any technique:

```bash
git show <sha>
git show --stat <sha>          # scope first
git log -1 --format=%B <sha>   # message and co-authors
```

Regenerate this list:

```bash
git log --format='%h|%ad|%s' --date=short --since=2026-01-01 --author='sidsingh' main
```

## MoE inference stack

| SHA | PR | Pattern |
|---|---|---|
| `7d1c01685` | #3496 | **Foundational.** Forks a parallel inference MoE hierarchy: `InferenceTopKRouter`, `InferenceGroupedMLP`, graph-safe AllGather dispatcher, GPU-resident offsets, fused NVLS collectives, centralized `are_tensors_nvls_eligible`. |
| `905c0e386` | #3851 | Lazy `SymmetricMemoryManager`; lazy non-destructive weight concatenation via `param.data` views to preserve TE `Parameter` identity. Fixes the RL integration. |
| `589cd9e12` | #3858 | `megatron/core/inference/moe/` package: `mcore_fused_moe`, Triton permute, padding-aware activations, fused activation+MXFP8-quantize+swizzle. Backend enum replaces a boolean. |
| `bfd45740c` | #4258 | **Architectural rewrite.** Variable-count AllGather-V / ReduceScatter-V so EP ranks carry different token counts; `fused_metadata_update` collapses 5 kernels into 1; buffers allocated once at init; dispatcher swap moves into `MoELayer.train()`. |
| `442a936a1` | #4570 | Shared-expert overlap on `SharedExpertMLP.stream`, with the AGV capped to 16 CTAs so it does not starve the side stream. |
| `c817dad28` | #4587 | Cuts the EP graph-size sync back to the minimum (max token count + is-anyone-non-decode) after it accumulated dead complexity. |
| `20f09364e` | #4603 | Replaces 25 `@triton.autotune` configs with vLLM's host-side `_get_default_config` heuristic keyed on a token-count *hint*; persistent-grid `moe_sum` that stops zeroing rows past `valid_tokens`; latent-MoE shared-expert overlap; `ep_consensus_interval`. |
| `3a253ac5c` | #4922 | `mask_routing_padding` writes `-1` into padded rows' topk slots so CUDA-graph padding activates no expert. Fixed-address `real_token_count` scalar. |

Start with #3496 for the structure and #4258 for the current design. Read them in
that order — #4258 only makes sense as a response to #3496's equal-token-count
limitation.

## CUDA graphs

| SHA | PR | Pattern |
|---|---|---|
| `32efeffd2` | #3250 | Splits inference graph scope from the training `full_iteration` scope; adds block-scope graphs for the Mamba block. Establishes the `_should_call_local_cudagraph` predicate. |
| `fde3b90a8` | #3527 | `num_cuda_graphs=-1` auto-sizing with a dense small-batch ladder, `[1,2,4] + range(8,256,8) + range(256,max,16)` — fine granularity where decode batches actually live. |
| `60a25aa67` | #3525 | `add_dummy_requests_for_expert_parallel_step`: idle EP ranks `fill_` preallocated tensors instead of constructing request objects. Rewrote 1448 lines of golden values because bucketization changed. |
| `35f76df3f` | #4440 | Moves hybrid graph ownership from `HybridStack` up to `HybridModel` so embedding and output layers fall inside the capture. Widening scope as an optimization. |
| `740c16e6b` | #5797 | `cuda_graph_max_tokens` default 512, so prefill and mixed steps up to 512 tokens get a graph instead of falling back to eager. |

## Mamba / SSM and Triton

| SHA | PR | Pattern |
|---|---|---|
| `ab2b33d54` | #4397 | `do_not_specialize=["batch"]`; `fast_exp` via `exp2`; capability-specialized `BLOCK_SIZE_M`/`num_warps` for Blackwell; `torch.zeros` to `torch.empty` where the kernel overwrites every row. |
| `9b4074b51` | #4764 | Bounds per-step Mamba work by the padded bucket rather than the global max; retunes the varlen conv autotune list to two regimes; removes the AGV CTA cap from #4570 after measuring it net-negative. |
| `f29c747f2` | #5608 | **Four lines, high value.** Demotes per-step batch sizes from `tl.constexpr` to runtime args, ending per-step JIT recompilation. |
| `411a5d8b2` | #5863 | Scratch sized `min(ceil(max_tokens / block_size_tokens), 3 * max_requests)` — an order of magnitude at low concurrency. Also fixes a hardcoded `mamba_chunk_size = 128`. |
| `648bc011f` | #5866 | Fused gather-plus-conditional-scatter Triton kernels replacing dense gather + `copy_`; runtime `real_count` gate; transpose folded into the write address. |

## Host path, scheduling, observability

| SHA | PR | Pattern |
|---|---|---|
| `53a2b19a2` | #2920 | Drops `dataclasses.asdict` (recursive deepcopy) and `torch.save` from request IPC; moves detokenization to the coordinator so it overlaps the next engine step; adds the NVTX ranges that make the host critical path visible. |
| `faced5128` | #3034 | MoE routing replay: records per-token expert choices into a static graph-safe buffer for load-imbalance analysis. |
| `eadbaa618` | #5611 | `/start_profile` and `/stop_profile` endpoints relaying `cudaProfilerStart/Stop` to every engine, to pair with `nsys --capture-range=cudaProfilerApi`. |
| `edd45620b` | #5609 | Prefix-cache hit reporting as `usage.prompt_tokens_details.cached_tokens`. |
| `bcf4c8fb5` | #5607 | Load-aware DP routing: `alpha * match + (1-alpha) * free_capacity`, replacing round-robin. Vectorized numpy because it runs per request. |
| `602fad039` | #5918 | Drops `prompt_tokens` from the wire by default, keeping `prompt_length` for the usage contract. |

## Test and CI maintenance

Not optimization patterns, but useful for seeing what breaks when the above lands.

| SHA | PR | Note |
|---|---|---|
| `369e0eba7` | #3071 | Fixes functional tests broken by #2920 |
| `66ec17eac` | #3357 | Inference functional test fixes |
| `a1165fabc` | #4454 | Fixes GitLab functional tests |

## Non-performance work

Included for completeness; these are product features on the same subsystem, not
hot-path optimizations.

| SHA | PR | Note |
|---|---|---|
| `a79f49d37` | #5276 | Reasoning-token retention delegated to the chat template |
| `82e9dc69c` | #5634 | `nemotron_v3` reasoning parser |

## Superseded APIs — do not copy from these

Several commits above contain code that has since been replaced. If you read them
directly, know what no longer exists.

**Dispatchers.** `InferenceCUDAGraphTokenDispatcher` (#3496) and the manual
`set_inference_cuda_graphed_iteration` / `unset_...` plumbing were replaced in #4258
by `NCCLAllGatherDispatcher` / `NVLSAllGatherVDispatcher` plus the automatic
`MoELayer.train()` swap.

**Grouped GEMM backend.** `inference_disable_torch_grouped_mm` (#3496) →
`inference_grouped_gemm_backend: 'auto'|'torch'|'te'` (#3858) →
`'flashinfer'|'torch'` (#4258) → `'flashinfer'|'torch'|'vllm'` with `vllm` as the
default (#4603).

**Deleted modules.** `megatron/core/inference/moe/pad.py` and the `skip_permute`
branches (added #3858, removed #4258). `smallest_non_decode_cuda_graph_size`
(removed #4587).

**Symmetric memory globals.** `_GLOBAL_SYMMETRIC_MEMORY_BUFFER_TP/_EP` in
`parallel_state.py` (#3496) → `SymmetricMemoryManager` in
`megatron/core/inference/symmetric_memory.py` (#3851).

**CUDA graph scope.** The `cuda_graph_scope` list holding
`CudaGraphScope.full_iteration_inference` (#3250, #4440) → `cuda_graph_impl` plus
`inference_cuda_graph_scope` (`none`/`layer`/`block`). `CudaGraphScope` survives
only for checkpoint deserialization; its docstring carries the migration table.

## Related work on branches, not `main`

`44334edad` "Fix flashinfer sampling kernels to return int64 instead of int32"
lives on the unmerged `fix_flashinfer_sampling` branch. The same `.long()` casts
reached `main` via `650b7838` (#5791, Keshav Santhanam, co-authored by Siddharth),
which also concluded that FlashInfer sampling must run **eagerly** — its kernel
choice is data-dependent and FlashInfer bakes the philox RNG state into a graph as
a by-value constant. See [host-path.md](host-path.md).
