# vLLM Navigation Map

Verified path index for `/Users/shanmugamr@nvidia.com/vllm`.

**Verified at HEAD `b8160878f`** (`v0.19.2rc0-219-gb8160878f`). Every path below
was confirmed to exist at that revision. Re-verify before quoting if the checkout
has advanced; vLLM moves files frequently. Line numbers are approximate anchors —
grep the symbol rather than trusting the number.

Paths are relative to the checkout root.

## MoE and expert parallelism

The area this campaign asks about most.

### Layer and dispatch abstraction

| Path | What it is |
|---|---|
| `vllm/model_executor/layers/fused_moe/layer.py` | `FusedMoE` module (~219), `forward` (~1545) |
| `vllm/model_executor/layers/fused_moe/modular_kernel.py` | The modular-kernel abstraction: `FusedMoEPrepareAndFinalize`, `FusedMoEExperts`, `FusedMoEKernel`, `TopKWeightAndReduce`, `ExpertTokensMetadata` |
| `vllm/model_executor/layers/fused_moe/config.py` | MoE parallel and quant config resolution |
| `vllm/model_executor/layers/fused_moe/fused_moe_method_base.py` | Base quant-method interface for MoE |
| `vllm/model_executor/layers/fused_moe/fused_moe_modular_method.py` | Modular-kernel quant-method wiring |
| `vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py` | The bf16/fp16 path — **this is the one Qwen3-30B-A3B bf16 takes** |
| `vllm/model_executor/layers/fused_moe/oracle/` | Backend *selection* per quant type: `unquantized.py`, `fp8.py`, `nvfp4.py`, `mxfp4.py`, `mxfp8.py`, `int8.py`, `int_wna16.py` |

`oracle/` answers "which kernel does vLLM actually pick", which is usually the
question behind "how does vLLM do MoE".

### Prepare/finalize — dispatch and combine

`vllm/model_executor/layers/fused_moe/prepare_finalize/`

| File | Backend |
|---|---|
| `deepep_ht.py` | DeepEP high-throughput |
| `deepep_ll.py` | DeepEP low-latency |
| `no_dp_ep.py` | Single-rank, no DP/EP |
| `naive_dp_ep.py` | Allgather-reducescatter DP/EP |
| `batched.py` | Batched per-expert padded layout |
| `flashinfer_nvlink_one_sided.py`, `flashinfer_nvlink_two_sided.py` | FlashInfer NVLink all2all |
| `mori.py`, `nixl_ep.py` | MORI (ROCm) and NIXL transports |

### All2all manager selection

| Path | What it is |
|---|---|
| `vllm/distributed/device_communicators/all2all.py` | All manager classes: `AgRsAll2AllManager` (~41), `DeepEPHTAll2AllManager` (~197), `DeepEPLLAll2AllManager` (~261), `FlashInferNVLinkTwoSidedManager` (~449), `FlashInferNVLinkOneSidedManager` (~556), `MoriAll2AllManager` (~672) |
| `vllm/distributed/device_communicators/cuda_communicator.py` (~118-166) | The if/elif chain mapping the `all2all_backend` string to a manager |
| `vllm/config/parallel.py` (~40-51) | `All2AllBackend` Literal — the valid names |
| `vllm/config/parallel.py` (~418) | Where `pplx` and `naive` are warned about and rewritten to `allgather_reducescatter` |
| `vllm/model_executor/layers/fused_moe/all2all_utils.py` | Shared helpers |

### Triton kernel and tuned configs

| Path | What it is |
|---|---|
| `vllm/model_executor/layers/fused_moe/fused_moe.py` | The Triton fused MoE kernel. `get_moe_configs` (~1035), `fused_experts` (~1605), `fused_experts_impl` (~1682) |
| `vllm/model_executor/layers/fused_moe/configs/` | ~316 tuned tile-config JSONs, `E=<experts>,N=<intermediate>,device_name=<gpu>[,dtype=...].json`. Grep for the specific shape; do not list. |
| `moe_align_block_size.py`, `moe_permute_unpermute.py`, `topk_weight_and_reduce.py` | Scatter/gather/reduce support ops (same directory) |

### Expert implementations

`vllm/model_executor/layers/fused_moe/experts/`

| File | Implementation |
|---|---|
| `cutlass_moe.py` | CUTLASS grouped GEMM |
| `deep_gemm_moe.py`, `batched_deep_gemm_moe.py` | DeepGEMM contiguous and masked-batched |
| `flashinfer_cutedsl_moe.py`, `flashinfer_cutedsl_batched_moe.py` | FlashInfer CuteDSL |
| `trtllm_bf16_moe.py`, `trtllm_fp8_moe.py`, `trtllm_nvfp4_moe.py`, `trtllm_mxfp4_moe.py` | TRT-LLM-gen. `trtllm_bf16_moe.py` is the one probed and rejected for mcore over its 4-D pre-shuffled weight-layout requirement. |
| `gpt_oss_triton_kernels_moe.py`, `nvfp4_emulation_moe.py`, `ocp_mx_emulation_moe.py`, `xpu_moe.py` | Others |

Still flat at the `fused_moe/` root: `fused_batched_moe.py`,
`flashinfer_cutlass_moe.py`, `triton_cutlass_moe.py`, `triton_deep_gemm_moe.py`,
`fused_marlin_moe.py`, `rocm_aiter_fused_moe.py`, `fallback.py`.

### Routing and top-k

`vllm/model_executor/layers/fused_moe/router/`

| Path | What it is |
|---|---|
| `grouped_topk_router.py` | `fused_grouped_topk` (~29), `grouped_topk` (~81), `GroupedTopk` CustomOp (~167), `GroupedTopKRouter` (~247) |
| `router_factory.py` | `create_fused_moe_router` (~35) — the dispatch point |
| `fused_moe_router.py` | `FusedMoERouter` ABC |
| `fused_topk_router.py`, `fused_topk_bias_router.py`, `custom_routing_router.py`, `zero_expert_router.py`, `routing_simulator_router.py`, `gate_linear.py` | Other routers |
| `csrc/moe/topk_softmax_kernels.cu`, `csrc/moe/grouped_topk_kernels.cu`, `csrc/moe/moeTopKFuncs.cuh` | The CUDA routing kernels |

This is where vLLM's one-kernel router lives — the counterpart to mcore's
four-to-five-kernel routing chain in the differential.

### Shared experts and overlap

| Path | What it is |
|---|---|
| `vllm/model_executor/layers/fused_moe/runner/shared_experts.py` | `SharedExpertsOrder` IntEnum (~27): `NONE`, `NO_OVERLAP`, `MK_INTERNAL_OVERLAPPED` (overlapped with dispatch/combine inside the modular kernel), `MULTI_STREAM_OVERLAPPED` (aux stream, overlapped with gate/router/experts). `SharedExperts` (~41), `_disable_shared_experts_overlap` heuristic (~82) |
| `vllm/model_executor/layers/fused_moe/runner/moe_runner.py`, `moe_runner_interface.py` | The runner invoking experts |

### EPLB — expert load balancing

`vllm/distributed/eplb/`: `eplb_state.py` (`EplbState` ~210),
`rebalance_execute.py` (physical weight movement), `eplb_communicator.py`,
`async_worker.py`, `eplb_utils.py`, `policy/`. Runner-side hooks in
`vllm/v1/worker/gpu/eplb_utils.py`.

## Qwen3 models

| Path | What it is |
|---|---|
| `vllm/model_executor/models/qwen3_moe.py` | `Qwen3MoeMLP` (~92), `Qwen3MoeSparseMoeBlock` (~137), `Qwen3MoeAttention` (~261), `Qwen3MoeDecoderLayer` (~364), `Qwen3MoeModel` (~440), `Qwen3MoeForCausalLM` (~675) — **the benchmarked model** |
| `vllm/model_executor/models/qwen3.py` | Dense Qwen3 |
| `qwen3_next.py`, `qwen3_next_mtp.py`, `qwen3_vl_moe.py`, `qwen3_omni_moe_thinker.py` | Hybrid/SSM, MTP, and multimodal variants |

## V1 decode loop and host path

| Path | What it is |
|---|---|
| `vllm/v1/worker/gpu_model_runner.py` | **The production GPU model runner** |
| `vllm/v1/worker/gpu/model_runner.py` | Experimental "Model Runner V2" — see `vllm/v1/worker/gpu/README.md`. Its `vllm/v1/worker/gpu/input_batch.py` defines a *different* `InputBatch`. Do not confuse the two. |
| `vllm/v1/worker/gpu_worker.py` | GPU worker: device init, memory profiling, KV cache allocation |
| `vllm/v1/worker/gpu_input_batch.py` | The persistent batch: `CachedRequestState` (~30), `InputBatch` (~81) |
| `vllm/v1/worker/block_table.py` | Per-request device block table across steps |
| `vllm/v1/core/sched/scheduler.py` | The scheduler |
| `vllm/v1/core/sched/async_scheduler.py` | `AsyncScheduler` (~12) — schedules step N+1 while N runs on GPU |
| `vllm/v1/core/sched/output.py`, `interface.py`, `request_queue.py` | Scheduler output struct, ABC, queue policies |
| `vllm/v1/engine/core.py` | `EngineCore` (~91), `EngineCoreProc` (~806), `run_busy_loop` (~1164, and a DP variant ~1731) |
| `vllm/v1/engine/async_llm.py` | Async front end |
| `vllm/forward_context.py` | Per-forward context threaded into the model: attention metadata, DP metadata, cudagraph runtime mode |

CPU/GPU overlap beyond async scheduling: `vllm/v1/worker/ubatching.py` and
`vllm/v1/worker/gpu_ubatch_wrapper.py` (DBO-style micro-batch overlap),
`vllm/v1/worker/gpu/async_utils.py` (async output copy, V2 runner).

## CUDA graphs and compilation

| Path | What it is |
|---|---|
| `vllm/v1/cudagraph_dispatcher.py` | `CudagraphDispatcher` — picks runtime mode and key per step |
| `vllm/compilation/cuda_graph.py` | `CUDAGraphWrapper` (~145) which captures and replays; `CUDAGraphEntry`, `CUDAGraphOptions` |
| `vllm/config/compilation.py` | `CUDAGraphMode` (~53) with `FULL_AND_PIECEWISE` (~63, the v1 default), `cudagraph_capture_sizes` (~622), `resolve_cudagraph_mode_and_sizes` (~1301) |
| `vllm/config/vllm.py` (~1423) | `_set_cudagraph_sizes` — **where the default bucket list is generated**: `[1, 2, 4] + list(range(8, 256, 8)) + ...` (~1432). Read this for bucket policy, not `compilation.py`. |
| `vllm/compilation/backends.py` | `VllmBackend` (~798), `PiecewiseCompileInterpreter` (~680), `CompilerManager` (~123) |
| `vllm/compilation/piecewise_backend.py` | `PiecewiseBackend` (~86) |
| `vllm/compilation/decorators.py` | `@support_torch_compile`, dynamic-dim marking |
| `vllm/compilation/partition_rules.py`, `passes/` | Graph-partition rules and fusion passes |

## Attention

`vllm/attention/` does not exist at this revision.

| Path | What it is |
|---|---|
| `vllm/v1/attention/selector.py` | `get_attn_backend` (~53) — the selection entry point |
| `vllm/v1/attention/backends/registry.py` | `AttentionBackendEnum` (~34), `register_backend` (~211) |
| `vllm/platforms/cuda.py` | Platform-level `get_attn_backend` override applying CUDA eligibility rules |
| `vllm/v1/attention/backend.py` | The abstract base file: `AttentionBackend`, `AttentionMetadataBuilder`, `CommonAttentionMetadata` all live here — **not** in `backends/utils.py`, which is helpers only |
| `vllm/v1/attention/backends/flash_attn.py` | FlashAttention |
| `vllm/v1/attention/backends/flashinfer.py` | FlashInfer — the backend behind the `trtllm-gen` Blackwell decode kernel in the differential |
| `vllm/v1/attention/backends/triton_attn.py` | Triton unified attention |
| `vllm/v1/attention/backends/mla/` | MLA variants |
| `vllm/v1/attention/ops/` | Attention op kernels |
| `vllm/model_executor/layers/attention/attention.py` | The `Attention` nn.Module used inside models |
| `vllm/vllm_flash_attn/flash_attn_interface.py` | Vendored FA Python shim |

## KV cache and paging

`vllm/v1/kv_cache_interface.py` (specs and layout), `vllm/v1/core/kv_cache_manager.py`
(paged manager), `vllm/v1/core/block_pool.py` (allocation and prefix-cache hash
table), `vllm/v1/core/kv_cache_utils.py` (block hashing),
`vllm/v1/core/kv_cache_coordinator.py` and `single_type_kv_cache_manager.py`
(multi-group, hybrid models), `csrc/cache_kernels.cu` and
`csrc/cache_kernels_fused.cu` (reshape-and-cache).

## Distributed and communication

| Path | What it is |
|---|---|
| `vllm/distributed/parallel_state.py` | Process group setup and accessors for TP/PP/DP/EP |
| `vllm/distributed/device_communicators/base_device_communicator.py` | `DeviceCommunicatorBase` |
| `vllm/distributed/device_communicators/cuda_communicator.py` | CUDA communicator; also selects the all2all manager |
| `vllm/distributed/device_communicators/custom_all_reduce.py` | Custom all-reduce; kernel in `csrc/custom_all_reduce.cu` |
| `vllm/distributed/device_communicators/symm_mem.py` | `SymmMemCommunicator` (~25) — PyTorch symmetric memory, the closest thing to an NVLS/multicast path. Multicast pointer check ~105 falls back when `multicast_ptr == 0`. |
| `vllm/distributed/device_communicators/flashinfer_all_reduce.py`, `mnnvl_compat.py` | FlashInfer all-reduce, MNNVL compat |
| `vllm/distributed/device_communicators/pynccl.py`, `pynccl_wrapper.py`, `pynccl_allocator.py` | NCCL bindings |
| `vllm/distributed/communication_op.py` | High-level TP collectives |

## Config and environment

| Path | What it is |
|---|---|
| `vllm/envs.py` | Every `VLLM_*` env var and its default |
| `vllm/config/vllm.py` | `VllmConfig` — stitches sub-configs and runs cross-config resolution, including cudagraph sizes |
| `vllm/config/parallel.py` | `ParallelConfig`: TP/PP/DP/EP, all2all backend |
| `vllm/config/compilation.py` | `CompilationConfig`, `CUDAGraphMode` |
| `vllm/config/cache.py` | `CacheConfig`: block size, KV dtype, GPU memory utilization |
| `vllm/config/model.py`, `scheduler.py` | `ModelConfig`; `SchedulerConfig` (max num seqs, chunked prefill, async scheduling) |
| `vllm/config/attention.py`, `kernel.py`, `quantization.py`, `speculative.py`, `mamba.py`, `load.py` | The rest |

## Benchmarks

Thin CLI scripts over a library. `benchmarks/benchmark_throughput.py`,
`benchmark_serving.py`, `benchmark_latency.py` are entry points;
`vllm/benchmarks/throughput.py`, `serve.py`, `latency.py` are the implementations
behind `vllm bench {throughput,serve,latency}`. Microbenchmarks worth knowing
about for kernel comparison: `benchmarks/kernels/`,
`benchmarks/attention_benchmarks/`, `benchmarks/fused_kernels/`,
`benchmarks/cutlass_benchmarks/`.

## csrc

88 `.cu` files. Notable: `csrc/moe/` (topk softmax, grouped topk,
permute/unpermute, marlin wna16, mxfp8, DSv3 router GEMM), `csrc/attention/`
including `csrc/attention/mla/`, `csrc/quantization/` (machete, marlin, w8a8,
awq, gptq, gguf, hadamard), `csrc/mamba/mamba_ssm/`, `csrc/cutlass_extensions/`,
`csrc/quickreduce/`, `csrc/rocm/`, `csrc/cpu/`.

## External vs vendored

All imported, none vendored:

| Package | Where imported |
|---|---|
| `flashinfer` | 13 files, incl. `vllm/v1/attention/backends/flashinfer.py`, `experts/trtllm_{bf16,fp8,nvfp4}_moe.py`, `flashinfer_all_reduce.py` |
| `deep_ep` | Exactly 3 files: `device_communicators/all2all.py`, `prepare_finalize/deepep_{ht,ll}.py` |
| `deep_gemm` | Wrapped behind `vllm/utils/deep_gemm.py`; direct import only in `vllm/model_executor/warmup/kernel_warmup.py`. MoE code imports the wrapper (`fused_moe/deep_gemm_utils.py`). |
| `pplx_kernels` | **Not present.** Removed from the tree. |

Vendored in-tree: `vllm/vllm_flash_attn/` and `vllm/third_party/`
(`flashmla`, `pynvml.py`).

## Known absent

Do not send a search after these; they do not exist at this revision.

- `vllm/attention/` — moved, see the attention section.
- `pplx_kernels` and any pplx all2all backend — removed. The `pplx` config string
  is accepted and silently rewritten.
- A naive all2all implementation — same treatment. The nearest real code is
  `AgRsAll2AllManager` in `all2all.py` and `prepare_finalize/naive_dp_ep.py`.
- `vllm/_version.py` — generated at build time, absent in this source tree.
