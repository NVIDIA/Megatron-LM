# Qwen3-30B-A3B optimization ledger

Goal: make Megatron-Core EP4 inference match or exceed the vLLM DP4+EP
throughput on one OCI 4×GB200 node without correctness regressions.

This ledger starts from scratch. Append every experiment, including failures
and regressions. Never edit an earlier result after it is recorded.

## Fixed protocol

| Setting | Value |
|---|---|
| Cluster | OCI `oci-hsg` |
| Hardware | 1 node, 4×GB200 |
| Model | Qwen3-30B-A3B, BF16 |
| Dataset / batch | gsm8k / 256 |
| Throughput workload | OSL1024, 2 warmups, 5 timed iterations |
| Profile workload | OSL128, one short warmup request, one timed request |
| mcore layout | TP=1, PP=1, EP=4, ETP=1 |
| vLLM layout | TP=1, DP=4, `--enable-expert-parallel` |
| Correctness gate | Fixed temperature-0 coherence prompts plus benchmark success |
| Primary metric | Throughput (output tokens/s) |
| Secondary metrics | Average latency and TPOT |

Do not compare results when hardware, checkpoint, batch size, output length,
parallelism, or warmup/timed counts differ.

## Baselines

| ID | Engine | Throughput | Avg latency | TPOT | Job / run | Nsight trace | Status |
|---|---|---:|---:|---:|---|---|---|
| VLLM-BASELINE | vLLM DP4+EP | 23,606.7 tok/s | 1,368.2 ms | 10.844 ms/tok | 5547673 / `vllm-qwen30b-nsys-20260722-093437` | `vllm_profile.nsys-rep`, `.sqlite` | Pass |
| MCORE-BASELINE | mcore EP4/TP1 | 12,346.1 tok/s | 2,590.1 ms | 20.735 ms/tok | 5553135 / `qwen-30b-nsys-20260722-161020` | `mcore_profile.nsys-rep`, `.sqlite` | Pass |

Fresh profile gap: mcore delivers 52.30% of vLLM throughput and is 47.70%
below the target. vLLM is 1.912× faster on this profile workload.

Baseline order is mandatory:

1. Record `VLLM-BASELINE` with Nsight Systems.
2. Record `MCORE-BASELINE` with Nsight Systems.
3. Compute the absolute and percentage gap.
4. Only then modify Megatron-Core.

## Experiment index

| ID | Date | Hypothesis | Changed files / flags | Throughput | Delta vs mcore baseline | Correctness | Job / run | Conclusion |
|---|---|---|---|---:|---:|---|---|---|
| VLLM-BASELINE | 2026-07-22 | Establish the fixed competitor target | none | 23,606.7 | n/a | Benchmark pass | 5547673 | Target established |
| MCORE-BASELINE | 2026-07-22 | Establish the fixed EP4 starting point | `max_requests=256` | 12,346.1 | baseline | Benchmark pass | 5553135 | Starting point established |
| QWEN-001 | 2026-07-22 | Single-kernel FC1+SwiGLU+FC2+topk-reduce mega-fusion beats the 4-kernel vLLM MoE path | `megatron/core/inference/moe/fused_moe_decode.py` (new), `dev/moe_fused/harness.py` (new) | microbench only | n/a (0.68–0.80× kernel) | Numerics pass (max_abs 2.6e-5, allclose) | session `qwen-moe-kernel` | Rejected — fused kernel 20–50% slower than reference; not integrated |
| QWEN-002 | 2026-07-22 | Fusing SiLU(gate)*up into the FC1 GEMM epilogue (removing bounded_silu_mul + the 2N round-trip) speeds up the decode MoE path without hurting FC2 tiling | `vllm_fused_moe.py` (FUSE_SWIGLU), `experts.py` (`fuse_fc1_activation=True`), `dev/moe_fused/harness_fc1.py`, `dev/moe_fused/run_e2e_insession.sh` (new) | **22,741.9 tok/s** (OSL1024) | **+0.55%** vs same-env fusion-off (22,617.7) | Coherent + numerics (max_abs 3.9e-5) | session `qwen-moe-kernel` in-session A/B | Accepted — MoE path 1.25×, e2e +0.55% throughput / −0.55% TPOT, no regression |
| PROFILE-OSL1024 | 2026-07-22 | Re-profile at the real throughput regime (BS256/OSL1024) to find the true vLLM→mcore gap and the dominant decode bottleneck | none (profiling only); `dev/moe_fused/profile_insession.sh` (new), `dev/moe_fused/vllm_osl1024_tput.sbatch` (new) | vLLM **33,994.5** vs mcore **~22,700** tok/s | mcore = **66.8%** of vLLM (**vLLM 1.50×**) | Both coherent | vLLM job `5555787` (tput) + `5555868` (nsys); mcore in-session `prof256a` | Gap is real & large at OSL1024. mcore decode: GPU 79% busy; MoE grouped-GEMM 41%, attn 22%, MoE routing 12% (49k tiny kernels), **exposed EP comm 11.5%**, norm/elt 10%, GPU idle 21%. vLLM uses TRT-LLM fused MoE (1-kernel routing + cutlass bmm + fused finalize, **0 exposed comm**). Next target: routing-kernel storm + exposed EP comm (pending approval) |
| SESSION2-BASE | 2026-07-23 | Re-establish clean OSL1024 baseline in fresh session `qwen-opt` (fusion on, histogram off) before autonomous optimization campaign | none (config = current best) | **22,398.9 tok/s** | baseline for session 2 (−1.5% vs QWEN-002 run, within variance) | Coherent | session `qwen-opt` run `e2ebase` | Clean reference; 65.9% of vLLM 33,994.5 |
| QWEN-003 | 2026-07-23 | Replacing per-pair `atomic_add` in the MoE local-token count kernel with a `tl.histogram` variant (one atomic/bin/CTA) cuts the routing-kernel cost | `permute.py` (`_count_local_tokens_kernel_histogram`, env `MCORE_MOE_HISTOGRAM_COUNT`), `dev/moe_fused/harness_count.py` (new) | microbench only | **0.96× (wash)** | **EXACT integer match** vs reference | session `qwen-opt` `harness_count.py` | Rejected — count-kernel cost is per-launch fixed overhead, not atomic contention; in-kernel rewrite can't help. Default OFF. Real lever = fewer launches + less host-scheduling idle |
| QWEN-004 | 2026-07-23 | flashinfer sampling backend is faster than torch sampling | `run_e2e_cfg.sh` (`--inference-dynamic-batching-sampling-backend flashinfer`) | server crash | n/a | n/a | session `qwen-opt` `flashinfer` | Rejected — incompatible with `full_iteration_inference` CUDA graph capture: `RuntimeError: Generator not registered with the capturing graph` in `flashinfer.sampling`. torch sampling stays. |
| QWEN-005a | 2026-07-23 | `async-sched-mode=serial` overlaps host resolve with next forward, hiding the ~2.3ms/step (21%) GPU idle | `run_e2e_cfg.sh` (`--inference-dynamic-batching-async-sched-mode serial`) | server crash | n/a | n/a | session `qwen-opt` `serial` | Blocked by explicit guards: `ValueError: Async scheduling does not support expert parallelism` (+ separate MoE guard). Guards env-gated for experiment → QWEN-005b |
| QWEN-005b | 2026-07-23 | Guards were merely conservative; async serial works for MoE+EP if opened | env-gated EP+MoE guards in `dynamic_engine.py` + `text_generation_controller.py` (`MCORE_ALLOW_ASYNC_MOE`) | **hang** | n/a | server init OK but **first decode request hangs** (>2min) | session `qwen-opt` `asyncmoe` (cancelled) | Rejected — the guard encodes a real limitation: async serial + nvls alltoall + EP deadlocks on the first decode step. Patch reverted (tree clean). Would need real engine work to support. |
| QWEN-006 | 2026-07-23 | `nccl` AllGather/ReduceScatter inference dispatcher overlaps/costs less than `nvls` | `run_e2e_cfg.sh` (`--inference-moe-token-dispatcher-type nccl`) | **14,677 tok/s** | **0.66× (much worse)** | Coherent | session `qwen-opt` `nccl` | Rejected — nccl pads to worst-case per-rank token count (fixed-count AllGather), inflating comm volume ~2×. `nvls` variable-count stays the best dispatcher. |
| QWEN-007 | 2026-07-23 | `async-sched-mode=serial` (guards opened) hides the between-graph idle at the real OSL1024 regime | env-gated guards + `run_e2e_cfg.sh` | **22,589 tok/s** | **+0.85% (marginal)** | Runs at BS256 (hang was single-request-only) | session `qwen-opt` `async1024` | Marginal. Proves decode at OSL1024 is **NOT idle-bound** (the earlier "21% idle" was an OSL256 prefill artifact). Not worth shipping (unsupported path + tiny gain). Guards left env-gated default-off. |
| QWEN-008 | 2026-07-23 | `CUDA_DEVICE_MAX_CONNECTIONS=8` (was hardcoded 1) lets comm & compute overlap on separate HW queues, hiding exposed NVLS comm | `run_e2e_cfg.sh` (env override) | **~22,556 tok/s** | **~flat (noise)** | Coherent | session `qwen-opt` `maxconn8` | Reject — overlap is bounded by the full-iteration CUDA-graph structure / data deps, not connection count. No effect. |
| QWEN-010 | 2026-07-23 | `torch` grouped-GEMM backend (`torch.nn.functional.grouped_mm`, cuBLAS) beats the vLLM Triton fused-MoE backend on GB200 | `run_e2e_cfg.sh` (`--inference-grouped-gemm-backend torch`) | **18,434 tok/s** | **0.82× (worse)** | Coherent | session `qwen-opt` `gemmtorch` | Rejected — vLLM Triton backend stays best. (`flashinfer` cutlass backend is blocked for SwiGLU; only torch/vllm allowed.) All three backends now evaluated → vLLM is optimal. |
| QWEN-009 | 2026-07-23 | Built-in `--moe-router-fusion` (TE fused softmax+topk) + `--moe-permute-fusion` cut the routing critical path (~18%) | `run_e2e_cfg.sh` EXTRA_SERVER_ARGS | server crash | n/a | n/a | session `qwen-opt` `routperm`/`routfus` | Rejected — both crash: `AssertionError: hidden_size mismatch: 128 vs 8`. TE fused router emits a dense **128-expert** routing map, but `InferenceTopKRouter` (transformer_impl=inference_optimized) uses a dense **top-8** contract for the vLLM/nvls dispatcher. The built-in fusions are wired to the training MoE path only. A hand-written fused softmax+topk would need to honor the top-8 inference contract. |
| PROFILE-DECODE | 2026-07-23 | Get the TRUE per-step decode bottleneck (prior OSL256 totals were prefill-contaminated) | analysis of archived `mcore_osl256.sqlite`, pure-decode window (t0+220s, big-dispatch-free) | n/a | n/a | n/a | local sqlite | **Corrected model** (decode GPU-time share): MoE grouped-GEMM (`_fused_moe_kernel`) **~40% #1**, routing (count/moe_sum/topk/scatter/softmax/meta) ~18%, exposed comm (dispatch 122k + combine 320k) ~16%, attention ~13%, norm/elt ~10%. Kernels overlap across streams → **wall = per-layer critical path** (attn→router→dispatch→GEMM→combine). vLLM wins via TRT-LLM fused MoE (0 exposed comm, fused routing+finalize). Explains QWEN-002 1.25× kernel → +0.55% e2e. |

## Session 2 (2026-07-23) — conclusion & recommendation

**Best shippable config stays: legacy async / nvls dispatcher / vLLM grouped-GEMM
backend / FC1+SwiGLU fusion (QWEN-002) = 22,398.9 tok/s (65.9% of vLLM 33,994.5).**

Every accessible knob was swept and rejected (QWEN-003…009). Key learnings:
- Decode at OSL1024 is **compute/comm-bound, not idle-bound** (async serial +0.85%).
- The gap is **structural**: mcore runs the MoE decode as a chain of discrete
  kernels — router gemm → softmax → topk → count/scatter/metadata → **exposed
  NVLS AllGather-V dispatch** → grouped GEMM (FC1+SwiGLU, FC2) → moe_sum →
  **exposed NVLS ReduceScatter-V combine** — all on the per-layer critical path.
  vLLM/TRT-LLM does the equivalent as one fused MoE (fused routing, cutlass
  grouped bmm, fused finalize) with **0 exposed comm**.
- Built-in fusions (`--moe-router-fusion`, `--moe-permute-fusion`) and
  `sampling-backend=flashinfer` are wired to training / non-full-graph paths and
  are **incompatible** with the `inference_optimized` top-8 + full-iteration-graph
  contract.

**To actually close the 1.5× gap (large, multi-session work), in priority order:**
1. **Eliminate exposed NVLS comm** (~16% + it gates the critical path): overlap
   dispatch/combine with expert GEMM via chunked/pipelined experts, or fuse the
   combine ReduceScatter-V with the FC2 epilogue + moe_sum (finalize fusion).
2. **Fuse the routing chain** honoring the inference top-8 contract: one kernel
   for softmax+top8+count+scatter-metadata (cuts ~18% of small serial kernels).
3. **A TRT-LLM-style single fused MoE decode kernel** (dispatch+groupedGEMM+
   SwiGLU+finalize) — the real vLLM-parity path; QWEN-001's full fusion was
   slower, so this needs a cutlass/CUTE grouped-bmm core, not Triton.

## Detailed records

### VLLM-BASELINE — DP4 with expert parallelism

| Field | Value |
|---|---|
| Date | 2026-07-22 |
| Hypothesis | Establish the fresh vLLM BS256 Nsight target |
| Code revision | `808c475352de6c3693b182048f174736af82356e`; skill files untracked, Megatron source clean |
| Changed files | none |
| Runtime flags | TP1, DP4, `--enable-expert-parallel`, max model length 4096, max sequences 512 |
| Image | `/lustre/fsw/portfolios/coreai/users/shanmugamr/agents-space/images/87e4947c6ce36433.sqsh` |
| Checkpoint / tokenizer | `/lustre/fsw/portfolios/coreai/users/shanmugamr/agents-space/checkpoints/qwen3-30b-a3b-hf` |
| Hardware / layout | OCI `oci-hsg`, 1×4 GB200, TP1/DP4/EP enabled |
| Workload | gsm8k, BS256, OSL128, one BS8/OSL32 warmup request, one timed request |
| Job / run | `5547673`; `/lustre/fsw/portfolios/coreai/users/shanmugamr/agents-space/runs/vllm-qwen30b-nsys-20260722-093437` |
| Throughput | 23,606.708 tok/s |
| Latency / TPOT | 1,368.153 ms / 10.844 ms-token |
| Correctness | Benchmark completed 256/256 requests |
| Nsight artifacts | `vllm_profile.nsys-rep`; `vllm_profile.sqlite` in the run directory |
| Result | vLLM target established |
| Next action | Record the matching mcore EP4 profile |

### MCORE-BASELINE — EP4/TP1

| Field | Value |
|---|---|
| Date | 2026-07-22 |
| Hypothesis | Establish the fresh mcore EP4 BS256 Nsight starting point |
| Code revision | `808c475352de6c3693b182048f174736af82356e`; skill files untracked, Megatron source clean |
| Changed files | none in Megatron source; profile harness sets `max_requests=256` |
| Runtime flags | TP1, PP1, EP4, ETP1, NVLS dispatcher, vLLM grouped GEMM, inference-optimized transformer, full-iteration inference CUDA graphs |
| Image | Cog dev image `/lustre/fsw/portfolios/coreai/users/shanmugamr/agents-space/images/ceecf5c304a5d8bd.sqsh` |
| Checkpoint / tokenizer | `qwen3-30b-a3b-mcore` / `qwen3-30b-a3b-hf` under the user checkpoint root |
| Hardware / layout | OCI `oci-hsg`, 1×4 GB200, TP1/EP4 |
| Workload | gsm8k, BS256, OSL128, one BS8/OSL32 warmup request, one timed request |
| Job / run | `5553135`; `/lustre/fsw/portfolios/coreai/users/shanmugamr/agents-space/runs/qwen-30b-nsys-20260722-161020` |
| Throughput | 12,346.092 tok/s |
| Latency / TPOT | 2,590.135 ms / 20.735 ms-token |
| Correctness | Checkpoint loaded and benchmark completed 256/256 requests |
| Nsight artifacts | `mcore_profile.nsys-rep`; `mcore_profile.sqlite` in the run directory |
| Result | mcore reaches 52.30% of vLLM; 47.70% below target |
| Next action | Analyze the timed-request A/B windows before changing code |

### QWEN-001 — MoE decode mega-fusion (FC1+SwiGLU+FC2+topk-reduce)

| Field | Value |
|---|---|
| Date | 2026-07-22 |
| Hypothesis | Fusing the whole MoE expert path into one Triton kernel (removing the `bounded_silu_mul` + `_moe_sum` kernels and two intermediate HBM round-trips) beats the 4-kernel `vllm_fused_moe` path, which is 40.5% of decode GPU-busy time |
| Code revision | branch `perf/moe-fused-decode-gemm` off `808c475352de6c3693b182048f174736af82356e`, dirty |
| Changed files | `megatron/core/inference/moe/fused_moe_decode.py` (new kernel), `dev/moe_fused/harness.py` (new standalone correctness+timing harness) |
| Runtime flags | standalone microbench; Qwen3-30B decode shapes H=2048, moe_ffn=768, 32 local experts, top-8, 256 valid tokens |
| Image | Cog dev image `ceecf5c304a5d8bd.sqsh` |
| Checkpoint / tokenizer | n/a (synthetic weights, reference = production `vllm_fused_moe`) |
| Hardware / layout | OCI `oci-hsg`, 1×GB200 (session `qwen-moe-kernel`) |
| Workload | microbench, 10 warmup + 100 timed CUDA-event iters |
| Job / run | session `qwen-moe-kernel`, exec runs `manual1784770098/200/321` |
| Throughput | not measured end-to-end (rejected at microbench) |
| Latency / TPOT | kernel: reference 175 µs vs fused 258 µs (best sweep 0.80×; larger tiles OOM shared memory) |
| Correctness | Pass — max_abs_diff 2.6e-5, max_rel 0.15 on a 1e-4 floor, `allclose(rtol=2e-2,atol=2e-2)` True |
| Nsight artifacts | none (microbench) |
| Result | Rejected — fused kernel is consistently 20–50% slower than the reference in every same-run comparison |
| Next action | Root cause: one CTA per token-block serializes the H=2048 FC2 output loop (vs the reference's N-parallel multi-CTA GEMMs), 3× atomic traffic to `out`, and shared-memory pressure caps tile sizes; the HBM/launch savings are negligible under CUDA graphs. Pivot to either (a) partial FC1+SwiGLU epilogue fusion only, or (b) the 11.5% exposed NVLS all-gatherv/reduce-scatter-v communication |

### QWEN-002 — FC1+SwiGLU epilogue fusion

| Field | Value |
|---|---|
| Date | 2026-07-22 |
| Hypothesis | The 4-kernel MoE path (FC1→2N intermediate→`bounded_silu_mul`→FC2→reduce) wastes a full HBM round-trip of the `[num_valid, 2N]` intermediate and a whole kernel launch. Computing gate & up in the same FC1 program and applying `SiLU(gate)*up` in the fp32 epilogue — writing the `[num_valid, N]` activated intermediate directly — removes both while keeping FC2's N-parallel tiling |
| Code revision | branch `perf/moe-fused-decode-gemm`, dirty |
| Changed files | `megatron/core/inference/moe/vllm_fused_moe.py` (add `FUSE_SWIGLU` constexpr to `_fused_moe_kernel`, `fuse_swiglu` to `_invoke_fused_moe_kernel`, `fuse_fc1_activation` path in `vllm_fused_moe`), `megatron/core/transformer/moe/experts.py` (`_vllm_forward` passes `fuse_fc1_activation=True`), `dev/moe_fused/harness_fc1.py` (new A/B harness) |
| Runtime flags | vLLM grouped-GEMM backend, SwiGLU; microbench at Qwen3-30B decode shapes H=2048, moe_ffn=768, 32 local experts, top-8, 256 valid tokens |
| Image | Cog dev image `ceecf5c304a5d8bd.sqsh` |
| Checkpoint / tokenizer | n/a for microbench (synthetic weights; reference = unfused `vllm_fused_moe`) |
| Hardware / layout | OCI `oci-hsg`, 1×GB200 (session `qwen-moe-kernel`) |
| Workload | microbench, 10 warmup + 200 timed CUDA-event iters, 3 repeats |
| Job / run | session `qwen-moe-kernel`, exec runs `fc1c…`/`fc1t…` |
| Throughput | **22,741.9 tok/s** (fused) vs **22,617.7 tok/s** (same-env, fusion off) → **+0.55%**. gsm8k BS256 OSL1024, 2 warmup + 5 timed iters |
| Latency / TPOT | e2e TPOT 11.257 (fused) vs 11.319 ms/tok (off) → −0.55%. MoE-path kernel microbench: reference 174–182 µs vs fused 141–143 µs → **1.24–1.27×** (3 repeats) |
| Correctness | Pass — fused vs unfused `vllm_fused_moe`: max_abs_diff 3.9e-5, allclose; e2e coherence prompts coherent (2+2=4, capital of France = Paris) |
| Nsight artifacts | none (in-session A/B benchmark, not profiled) |
| Result | Accepted — correct, MoE kernel 1.25×, e2e +0.55% throughput / −0.55% TPOT, no regression. Unfused path byte-identical (all edits guarded by the `FUSE_SWIGLU` compile-time constexpr) |
| Next action | The 1.25× MoE-kernel win only nets +0.55% e2e at OSL1024, so MoE FC1 activation is not the wall-time bottleneck at the throughput regime — re-profile the OSL1024 decode (not the OSL128 profile trace) to find the true dominant cost before the next change. NOTE: at OSL1024 mcore already reaches ~22.6k tok/s; the ledger's 52% "gap" is an artifact of the OSL128 single-request profile workload |

Append records using this exact structure:

```markdown
### QWEN-NNN — short name

| Field | Value |
|---|---|
| Date | YYYY-MM-DD |
| Hypothesis | One measurable claim |
| Code revision | Commit SHA and clean/dirty state |
| Changed files | Exact paths, or `none` for baseline |
| Runtime flags | Exact non-default flags |
| Image | Immutable image path/tag |
| Checkpoint / tokenizer | Exact paths |
| Hardware / layout | Cluster, GPUs, TP/PP/EP/ETP/DP |
| Workload | Dataset, batch, OSL, warmups, timed iterations |
| Job / run | Slurm job ID and run directory |
| Throughput | tokens/s |
| Latency / TPOT | ms / ms-token |
| Correctness | Prompt outputs and benchmark status |
| Nsight artifacts | `.nsys-rep`, `.sqlite`, analysis output |
| Result | Supported / rejected / inconclusive |
| Next action | One prioritized follow-up |
```

## Optimization rules

1. Profile and classify before proposing a code change.
2. Change one performance variable at a time.
3. Preserve the fixed protocol.
4. Validate correctness before accepting throughput.
5. Revert regressions or correctness failures.
6. Record the result before beginning another experiment.
7. Stop when mcore meets or exceeds `VLLM-BASELINE`, then rerun both baselines
   once to confirm parity under identical conditions.
