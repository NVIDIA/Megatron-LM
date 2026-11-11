# MoE dispatcher study: running report

Living notes for the `nvls` (AllGather-V / ReduceScatter-V over NVLS multicast) vs `nvl_a2a`
(all-to-all-V: unicast dispatch, pull combine) comparison. Newest section first. Run directories
are under `inference-bench/experiments/` (git-ignored); analysis scripts and plots under
`inference-bench/analysis/`.

## Setup (Ultra v3 wide-EP runs)

- Model: Nemotron 3 Ultra 550B-A55B. 108 layers = **48 MoE** + 48 Mamba + 12 attention (+1 MTP,
  unused in plain decode). 512 experts, top-22, latent 2048, hidden 8192.
  (Super v3 for reference: 88 layers = 40 MoE + 40 Mamba + 8 attention.)
- Parallelism: 16 GB200 nodes (64 GPUs), TP4 x EP64 (ETP1), so 16 data-parallel groups.
- Server: `BUFFER_GB=80`, `MAMBA_RATIO=0.4` (332 requests per group), async scheduling, CUDA
  graphs, FlashInfer sampling, bf16 combine for **both** dispatchers (verified in `config.env`, the
  server's parsed args, and `dynamic_context.py`, which allocates the RSV buffer in bf16).
- Weights per GPU at TP4 x EP64: 18.2 B params (~34 GiB): routed experts 15.3 GiB + dense / TP.
  Checkpoint totals: routed experts 526 B params (980 GiB), Mamba 20.2 B, shared experts 8.2 B,
  router + latent 1.8 B, attention 1.7 B, embedding + LM head 2.1 B.

## 2026-10-27: Megatron Core inference (MInf, NVLS) vs vLLM 0.32 on the math blend

Same workload, client and metrics as the math-blend steady runs: closed loop at c users,
natural EOS, no output cap, 60 s warmup + 300 s window per ascending level; throughput from each
server's `/metrics`, interactivity = 1000 / median TPOT (TTFT included). Prefix caching off for
both. vLLM 0.32.0 (`launch_vllm.sh`), one engine over all of a run's GPUs with
`--enable-expert-parallel` and the configs' exact TP / DP; its expert GEMM is FlashInfer's
TRT-LLM BF16 MoE, the same kernel family as MInf's `trtllm_bf16_routed`. vLLM's EP dispatch:

- EP > TP (DP > 1): `--all2all-backend flashinfer_nvlink_one_sided` (FlashInfer's NVLink one-sided
  all-to-all, what InferenceX's vLLM GB200 recipes use); vLLM's default all-gather /
  reduce-scatter (AG/RS) shown for Super narrow.
- EP = TP (DP = 1, Ultra narrow): no dispatch; tokens replicated across TP and the TP all-reduce
  (FlashInfer MNNVL, fused with RMSNorm) combines. vLLM ignores the all-to-all setting here.

The 1-request-per-GPU level is dropped (few completions; first level after startup). Plots:
`analysis/pareto/mlsys-minf/`. tok/s/GPU, tok/s/user; last column = MInf throughput vs best vLLM.

**Super v3, 8 GPUs: non-expert TP4 x DP2, expert EP8**

| c | MInf (NVLS) | vLLM (FlashInfer A2A) | vLLM (AG/RS) | MInf vs vLLM A2A |
|---|---|---|---|---|
| 16 | 278.7, 140.4 | 225.4, 110.8 | 222.2, 113.3 | **+24%** |
| 32 | 443.4, 111.6 | 390.5, 99.1 | 370.8, 92.3 | **+14%** |
| 64 | 703.4, 88.5 | 648.7, 82.2 | 598.6, 75.7 | **+8%** |
| 128 | 1118.8, 71.4 | 1031.1, 64.8 | 928.4, 58.9 | **+9%** |
| 256 | 1612.2, 50.8 | 1561.6, 49.6 | 1285.1, 40.7 | +3% |

**Ultra v3, 8 GPUs: non-expert TP8, expert EP8 (EP = TP)**

| c | MInf (NVLS) | vLLM (no dispatch) | MInf vs vLLM |
|---|---|---|---|
| 16 | 118.0, 59.3 | 126.7, 63.8 | -7% |
| 32 | 180.5, 45.4 | 187.9, 48.6 | -4% |
| 64 | 285.0, 35.8 | 293.5, 37.1 | -3% |
| 128 | 446.0, 27.9 | 433.0, 27.4 | **+3%** |

**Ultra v3, 64 GPUs: non-expert TP8 x DP8, expert EP64**

| c | MInf (NVLS) | vLLM (FlashInfer A2A) | MInf vs vLLM |
|---|---|---|---|
| 128 | 173.7, 87.6 | 146.2, 73.5 | **+19%** |
| 256 | 287.5, 73.4 | 242.8, 61.7 | **+18%** |
| 512 | 418.1, 53.4 | 380.0, 48.2 | **+10%** |
| 1024 | 547.5, 34.6 | 550.2, 34.8 | -0.5% |

Super v3 wide (TP4 x DP16, EP64) on vLLM is queued.

**Reading.**

- With EP > TP, MInf leads at every level below the top, by up to ~20-25% on both axes, and
  the gap closes as the batch grows (tied by 16 requests per GPU). With EP = TP, vLLM leads at
  small batches by up to ~7% and MInf passes it by c = 128.
- Output lengths are not the cause: per-level median output tokens are similar across the two
  servers and move in no consistent direction (finished requests per window are small, so the
  medians are noisy).
- The expert GEMM is the same kernel, and the gap shrinks as the step becomes GEMM-bound, so
  the difference is a fixed per-step cost on vLLM's EP > TP path. Candidates, not yet measured:
  (1) vLLM synchronizes token counts across DP ranks every step (a CPU all-reduce under async
  scheduling) and idle ranks run dummy steps; (2) under CUDA graphs it pads every DP rank's whole
  forward (attention, Mamba, dense) to the largest rank's token count, where MInf pads only the
  MoE rows; (3) collective latency per MoE layer at small token counts (NVLS multicast all-gather
  / reduce-scatter vs the one-sided all-to-all). An nsys trace of one decode step on each side
  would split these.
- vLLM's dispatch backend matters at scale: the one-sided all-to-all beats its default AG/RS by
  21% at c = 256 on Super narrow, but not at c = 16.
- The EP = TP result is the replicate-tokens path: one fused all-reduce + norm per layer beats
  sequence parallelism's all-gather + reduce-scatter while latency dominates, and loses once the
  replicated non-expert activation traffic grows. A per-step switch between the two in MInf was
  prototyped and stashed (incorrect output; not debugged).

Runs: `20261024_175142_math_superv3_narrow_ep_nvls`,
`20261027_162013_superv3_narrow_ep_vllm_fi_one_sided`, `20261027_154044_superv3_narrow_ep_vllm`,
`20261024_171149_math_ultrav3_narrow_ep_nvls`, `20261027_154043_ultrav3_narrow_ep_vllm`,
`20261024_184036_math_ultrav3_wide_ep_nvls`, `20261027_165152_ultrav3_wide_ep_vllm`.

## 2026-10-17: A long-decode dataset: natural generation lengths of Super v3 on math reasoning

The wide-EP sweeps so far force OSL 8192 with `ignore_eos` on ~60-token GSM8K prompts, so most
decode steps continue past the model's natural end (and routing may not look like real
generation). To get realistic long decodes, Super v3 (TP4 x EP4, one node, vLLM MoE backend) was
asked for natural-length rollouts: `/v1/chat/completions`, chat template with reasoning on, the
model's default sampling (`generation_config.json`; no sampling parameters sent), no `ignore_eos`,
`max_tokens` 131072 only as a safety cap; 64 requests in flight (closed loop). Tooling:
`rollouts/collect_rollouts.py` + `rollouts/run_rollouts.sh` (`DATASET=math500 | hard_math`);
plot `rollouts/superv3_lengths.png` (`rollouts/plot_lengths.py`); data in
`20261016_183040_rollouts_superv3_math500/rollouts{,_hard_math}.jsonl`.

| Set | Rollouts | p10 | p50 | p90 | p99 | Max | Correct |
|---|---|---|---|---|---|---|---|
| MATH-500 (`HuggingFaceH4/MATH-500`) | 1,165 | 207 | **703** | 4,313 | 13.6k | 31k | 86% |
| Hard math: AIME 2025 + HMMT Feb 2025 + BeyondAIME, 8 samples each | 1,129 | 3.1k | **18.9k** | 103.8k | 131k (cap) | 131k, 48 capped | 78% |

| Group | p50 | p90 | Correct |
|---|---|---|---|
| MATH level 1 / 2 / 3 / 4 / 5 | 261 / 402 / 527 / 974 / 1.6k | 0.7k / 1.3k / 2.6k / 4.3k / 8.5k | 91 / 83 / 88 / 87 / 85% |
| AIME 2025 (`math-ai/aime25`) | 10.7k | 69k | 92% |
| HMMT Feb 2025 (`MathArena/hmmt_feb_2025`) | 16.4k | 66k | 83% |
| BeyondAIME (`ByteDance-Seed/BeyondAIME`) | 28.9k | 109k | 72% |

- **The hard-math set is the long-decode workload to use:** median ~19k generated tokens, p90 ~100k,
  so contexts grow to tens of thousands of tokens during decode, with natural routing. MATH-500 is
  too short (median 0.7k) for long-decode studies.
- Length is set by difficulty: each MATH level roughly doubles the median, and the competition sets
  are ~10x beyond MATH level 5. BeyondAIME, the hardest and least accurate, is the longest (38 of
  its rollouts hit the cap), as reported for other reasoning models (unsolved problems run long).
- Heavy tails: p90 / p50 is ~6x for both sets, so a fixed batch stays partly busy long after most
  requests finish; a closed-loop or arrival-rate client is the right way to load a server with it.
- **The tails are under-counted.** Both runs lost their in-flight rollouts at the end (MATH-500 when
  its client was cancelled; the hard set when the server hit its 4-hour limit), and in-flight
  rollouts at a cutoff are the long ones; the 48 capped rollouts may have run longer. A clean rerun
  needs a longer server limit (or resuming the unfinished items).

## 2026-10-17: Does wide EP help? Super v3 TP4 x EP4 vs TP4 x EP64 (EP16 pending)

Same method as the Ultra comparison below: fixed concurrency c per GPU (total = 4c x DP groups;
each TP4 group decodes 4c requests), GSM8K prompts, OSL 8192 with `ignore_eos`, TRT-LLM BF16 MoE,
`nvls`, bf16 combine, `BUFFER_GB=50`, `MAMBA_RATIO=0.4`, code `6bfbca9`. No waiting requests at any
point. Runs `20261016_182545_wideep_tpot_superv3_tp4_ep4_osl8192` and
`20261016_182832_wideep_tpot_superv3_tp4_ep64_osl8192`; plot `analysis/superv3_wideep_tpot.png`.

| c per GPU | Total on 64 GPUs | TP4 x EP4 TPOT | TP4 x EP64 TPOT | EP64 vs EP4 |
|---|---|---|---|---|
| 1 | 64 | 6.56 ms | 6.09 ms | -7% |
| 2 | 128 | 7.60 ms | 6.44 ms | -15% |
| 4 | 256 | 9.04 ms | 6.78 ms | -25% |
| 8 | 512 | 11.10 ms | 7.19 ms | **-35%** |
| 16 | 1024 | 13.17 ms | 8.39 ms | **-36%** |
| 32 | 2048 | 15.35 ms | 11.08 ms | -28% |

- Wide EP helps Super too, and by more than Ultra (up to 36%): an EP4 GPU holds 128 experts (vs 64
  for Ultra's EP8), so its expert-weight reads keep growing with load; EP64's 8 experts per GPU are
  all hit from c = 1 and its TPOT is nearly flat to c = 8.
- The gap narrows at c = 32: EP64 jumps 8.4 -> 11.1 ms while EP4 rises 2.2 ms. Likely EP4's experts
  are saturated by then while EP64's token-proportional costs (16x more dispatch / combine rows per
  GPU) keep growing; not yet confirmed with a trace.
- TP4 x EP16 failed at first (its 4 nodes landed in two NVL72 racks; symmetric-memory init fails
  across racks), was resubmitted with `--segment 4`, and is pending.

## 2026-10-16: Does wide EP help? Ultra v3 TP8 x EP8 vs TP8 x EP64, end to end

Same per-GPU load, same TP8 groups, same code (`0345cd18e` + freeing the canonical expert weights
after the TRT-LLM conversion), TRT-LLM BF16 MoE, `nvls`, bf16 combine. Fixed concurrency c per
GPU: total = 8c on EP8 (1 DP group, 64 experts per GPU) and 64c on EP64 (8 DP groups, 8 experts
per GPU), so every DP group decodes the same 8c requests. GSM8K prompts, OSL 8192 with
`ignore_eos` (prefill amortized away), TPOT = batch wall / 8192, one timed batch after a warmup.
No waiting requests at any point (server logs). Runs
`20261016_165554_wideep_tpot_ultrav3_tp8_ep8_osl8192` and
`20261016_164611_wideep_tpot_ultrav3_tp8_ep64_osl8192`; plot `analysis/ultrav3_wideep_tpot.png`
(`analysis/plot_wideep.py`).

| c per GPU | EP8 TPOT | EP64 TPOT | TPOT | EP8 tok/s/GPU | EP64 tok/s/GPU | Throughput |
|---|---|---|---|---|---|---|
| 1 | 12.45 ms | 10.81 ms | -13% | 80 | 92 | +15% |
| 2 | 12.93 ms | 11.90 ms | -8% | 155 | 168 | +9% |
| 4 | 18.13 ms | 12.57 ms | **-31%** | 221 | 318 | **+44%** |
| 8 | 20.50 ms | 14.57 ms | **-29%** | 390 | 549 | **+41%** |
| 16 | 25.73 ms | 18.69 ms | **-27%** | 622 | 856 | **+38%** |

- **Yes, wide EP helps: 27-31% lower TPOT and ~40% more throughput per GPU from c = 4 up.**
- Mechanism: expert activation. Each GPU does the same number of (token, expert) pairs in both
  configs, but EP8's GPUs hold 64 experts and stream the weights of every one that is hit; the
  expected number hit grows with load (~19 / 32 / 48 / 60 / 64 at c = 1-16), while EP64's 8
  experts are all hit from c = 1. EP8's jump from c = 2 to 4 (+5.2 ms) matches ~16 more experts
  per layer (~32 GB more per step, ~4.6 ms at HBM bandwidth); past c = 8 it is saturated and both
  configs grow with token-proportional work.
- Open: c = 1 -> 2 does not fit the byte model (EP8 should slow by ~3 ms, it slowed 0.5 ms and the
  gap narrowed to 8%). A trace of EP8 at c = 1 and 2 would show the expert-GEMM time directly.
- EP64 pays 8x more dispatch / combine rows per GPU at equal load; that cost is not yet broken out.

## 2026-10-16: Ultra v3 TP8 x EP64 with the TRT-LLM BF16 routed MoE

Same setup as the Ultra trace log (concurrency 128, 128 output tokens, bf16 combine), with
`GEMM_BACKEND=trtllm_bf16_routed` and the per-step EP token-sum consensus; run
`20261016_101531_trace_ultrav3_tp8_ep64_nvls_bs128_trtllm` vs the vLLM baseline `152634`. This
run predates the pre-dispatch routing pack (it still packs with torch ops after the 3-tensor AGV).

- Decode step **12.00 -> 10.65 ms (-1.35 ms, -11.2%)**, identical on ranks 4 and 44.
- Rank 4, us per decode step: expert GEMMs 3435 (`_fused_moe_kernel`) vs 2236 (`bmm_*`), **1.54x**;
  GEMM + alignment / routing + top-k reduction 4331 vs 2816 (TRT-LLM routing 383, our finalize
  196; vLLM alignment 580, `_moe_sum` 316), 1.54x. Ultra's shapes (8 local experts, K=2048,
  N=5120) favour TRT-LLM's GEMM far more than Super's at TP4 x EP4 (~1.1x).
- The torch packing chain still costs +373 us/step here (435 on rank 44); the pre-dispatch pack
  removed the equivalent ~0.4-0.5 ms/step on Super, so a rerun should land near ~10.2 ms.
- Rerun with the pre-dispatch pack (`20261016_115006_..._trtllm_packed`): **10.06 ms** (-0.59 ms,
  -5.5%; identical on ranks 4 and 44), 28.5% below the TP4 `nvls` baseline. Rank 4, us/step:
  elementwise 828 -> 505, routing + bookkeeping 231 -> 153 (no `mask_routing_padding`), expert
  GEMMs unchanged. Its profiled-batch wall time (3.63 s vs 2.67 s) includes prefill, which now
  runs eager on every EP rank; not yet looked at.

## 2026-10-16: Super v3, routing packed before the dispatch

`20261016_103607_trace_superv3_tp4_ep4_nvls_sweep_trtllm_packed`: the routing is packed on each
rank before the dispatch by one Triton kernel (`pack_routing`, which also masks CUDA-graph padding
rows, replacing `mask_routing_padding`), and the AGV gathers hidden + packed int32
(`multimem_all_gatherv_2tensor`) instead of hidden + int64 ids + fp32 probs.

| Concurrency | vLLM | TRT-LLM | TRT-LLM packed | Packed vs vLLM |
|---|---|---|---|---|
| 1 | 6.05 ms | 5.75 ms | 5.35 ms | -11.5% |
| 4 | 7.15 ms | 6.77 ms | 6.39 ms | -10.6% |
| 16 | 9.98 ms | 9.30 ms | 8.85 ms | -11.3% |
| 64 | 14.18 ms | 13.25 ms | 12.82 ms | -9.6% |
| 256 | 16.32 ms | 15.21 ms | 14.68 ms | -10.0% |

- At concurrency 16 the torch elementwise kernels drop 424 -> 77 us/step for a 70 us pack kernel,
  and `mask_routing_padding` (55 us) goes away.
- The gather does not get cheaper (2-tensor 272 vs 3-tensor 244 us/step): latency-bound at these
  sizes, so a third fewer bytes buys nothing.
- Rank 0's TP reduce-scatter roughly doubles (530 -> 1205 us/step at concurrency 16). Likely it now
  waits on slower TP peers (the step period still improves); needs ranks 1-3 to confirm.

## 2026-10-16: Super v3 end to end, TRT-LLM BF16 routed MoE vs vLLM Triton (TP4 x EP4)

Super v3 on one GB200 node, TP4 x EP4 (one DP group, so the TRT-LLM row bound is the batch
itself), `nvls`, bf16 combine, decode CUDA graphs. One nsys capture per backend over a GSM8K
sweep (1 timed batch of 128 output tokens per concurrency, 10 s idle between concurrencies;
`client.py --profile-all --pause 10`). Rank 0, median decode step period and mean kernel time per
decode step, from `analysis/sweep_steps.py`. Runs `20261016_095242_trace_superv3_tp4_ep4_nvls_sweep_vllm`
and `20261016_095241_trace_superv3_tp4_ep4_nvls_sweep_trtllm_bf16_routed`.

| Concurrency | vLLM step | TRT-LLM step | Delta |
|---|---|---|---|
| 1 | 6.05 ms | 5.75 ms | -0.29 ms (-4.9%) |
| 4 | 7.15 ms | 6.77 ms | -0.38 ms (-5.3%) |
| 16 | 9.98 ms | 9.30 ms | -0.68 ms (-6.8%) |
| 64 | 14.18 ms | 13.25 ms | -0.93 ms (-6.6%) |
| 256 | 16.32 ms | 15.21 ms | -1.11 ms (-6.8%) |

Per decode step (us), the MoE-related categories:

| Concurrency | Routed GEMMs vLLM / TRT | Routing + bookkeeping vLLM / TRT | Elementwise vLLM / TRT |
|---|---|---|---|
| 1 | 768 / 972 | 729 / 183 | 335 / 606 |
| 4 | 1754 / 1899 | 802 / 183 | 361 / 627 |
| 16 | 3633 / 3663 | 853 / 193 | 386 / 680 |
| 64 | 6514 / 7121 | 943 / 182 | 429 / 798 |
| 256 | 6592 / 6032 | 916 / 192 | 493 / 817 |

(TRT-LLM's routed GEMMs include its routing kernel and our finalize; the bookkeeping row is the
sigmoid top-k router, ~136 us/step in both, plus vLLM's alignment / indirection kernels.)

- TRT-LLM is 5-7% faster per decode step at every concurrency.
- Correction: the table above is not like-for-like. Its TRT-LLM "routed GEMMs" include the routing
  kernel and our finalize, while vLLM's equivalents sit elsewhere (alignment kernels under
  bookkeeping, the `_moe_sum` top-k reduction under elementwise). Split by kernel (us/step):

  | Concurrency | vLLM GEMM | TRT GEMM (`bmm_*`) | GEMM ratio | vLLM GEMM + align + moe_sum | TRT GEMM + routing + finalize | Expert-path ratio |
  |---|---|---|---|---|---|---|
  | 1 | 768 | 697 | 1.10x | 1331 | 972 | 1.37x |
  | 4 | 1754 | 1575 | 1.11x | 2390 | 1899 | 1.26x |
  | 16 | 3633 | 3270 | 1.11x | 4324 | 3663 | 1.18x |
  | 64 | 6514 | 6514 | 1.00x | 7295 | 7121 | 1.02x |
  | 256 | 6592 | 5516 | 1.20x | 7344 | 6033 | 1.22x |

  The pure GEMM (`bmm_Bfloat16_..._t128x8` / `t128x16`, swap-AB with 8- or 16-token tiles) is
  ~10% faster on Super (20% at 256, equal at 64); the whole expert path is 1.2-1.4x faster at low
  concurrency, in line with the Ultra microbenchmark (which also timed the whole layer, so it never
  isolated the GEMM).
- TRT-LLM pays +270-390 us/step in elementwise kernels: the per-layer -1 routing fill, the
  `id << 16 | bf16 prob` packing and casts. Fusing these into the dispatch (already listed below)
  is worth ~0.3 ms/step here.
- Both backends' routed-GEMM time grows ~linearly with concurrency up to 64: at TP4 x EP4 the batch
  per expert is still small, so more tokens means proportionally more weight-tile passes for both.
- Caveat: the kernel-category sums at concurrency 64 do not fully account for the period delta
  (TRT-LLM's sums are higher, its period lower); the decode-step counts also differ (145 vs 127).
  Worth a second look before quoting the concurrency-64 number.

## 2026-10-15: routed-expert grouped GEMM: TRT-LLM vs Triton, and does wide EP help decode?

Microbenchmark of one routed MoE layer on one GB200 at Nemotron 3 Ultra's shape (512 experts,
top-22, K=2048 latent, N=5120, squared ReLU, bf16 combine buffer), assuming TP = EP (one DP group,
so every EP rank holds the same batch and the gathered row count is host-known; rows are the batch
padded to a multiple of TP, padding routed to expert -1). Both kernels write the bf16 combine
buffer directly. `microbench/tp_eq_ep_moe_bench.py`, results `tp_eq_ep_moe_bench.{log,csv}`.

- `vllm_triton`: megatron `vllm_fused_moe` (today's default).
- `trtllm`: FlashInfer `trtllm_bf16_routed_moe` (what vLLM uses for Ultra) with the tricks from
  nvcsathe/Megatron-LM@trtllm-fused-moe: -1 routing for padding rows, `do_finalize=False`, and a
  padding-aware Triton finalize (`microbench/finalize_valid_loop.py`: the branch's kernel with the
  token loop bounded by `valid_tokens` and a bf16 output allowed).

All 120 configurations match an fp32 reference.

**Data point 1: TRT-LLM is the better kernel.** `vllm_triton` time / `trtllm` time per layer:

| Concurrency (total) | EP4 | EP8 | EP16 | EP32 | EP64 |
|---|---|---|---|---|---|
| 1-4 | 1.22-1.25 | 1.03-1.30 | 1.20-1.21 | 0.84 (no local work) | 0.79 (no local work) |
| 8-64 | 1.06-1.12 | 1.08-1.17 | 1.09-1.21 | 1.01-1.14 | 1.07-1.33 |
| 128-512 | 1.14-1.16 | 1.18-1.23 | 1.23-1.30 | 1.16-1.32 | 1.30-1.47 |
| 1024-2048 | 1.10-1.31 | 1.20-1.38 | 1.29-1.41 | 1.44 | 1.71-1.77 |

TRT-LLM wins 54/60 configurations; it only loses (~20.5 vs ~16-17.5 us) when this GPU's experts
get no tokens, where another rank is the bottleneck anyway. Its lead grows with batch and EP.

**Data point 2: at a fixed load per GPU, wide EP makes the expert compute much faster.** Per-GPU
MoE layer time (trtllm) at c requests per GPU (total batch = c x EP); [tokens per local expert]:

| Requests/GPU | EP4 | EP8 | EP16 | EP32 | EP64 | EP4 -> EP64 |
|---|---|---|---|---|---|---|
| 1 | 111.6 us [0.2] | 136.2 [0.3] | 112.0 [0.7] | 93.9 [1.4] | 72.4 [2.8] | 1.5x |
| 2 | 250.4 [0.3] | 204.8 [0.7] | 161.2 [1.4] | 113.4 [2.8] | 79.6 [5.5] | 3.1x |
| 4 | 399.5 [0.7] | 302.6 [1.4] | 194.0 [2.8] | 125.1 [5.5] | 82.4 [11] | 4.8x |
| 8 | 539.3 [1.4] | 364.5 [2.8] | 216.2 [5.5] | 128.1 [11] | 85.7 [22] | 6.3x |
| 16 | 673.6 [2.8] | 384.7 [5.5] | 221.6 [11] | 132.1 [22] | 90.7 [44] | 7.4x |
| 32 | 737.9 [5.5] | 402.2 [11] | 227.8 [22] | 141.6 [44] | 121.9 [88] | 6.1x |

- At equal load the (token, expert) pairs per GPU stay ~22c while the expert weights each GPU reads
  shrink as 1/EP, i.e. each expert sees EP-times more tokens per step. The grouped GEMM goes from
  weight-bandwidth bound (EP4 streams 5.4 GB/GPU at ~7.2-7.3 TB/s, ~91% of peak) to fixed-cost
  bound (EP64: 0.34 GB/GPU, ~20-35 us of fixed cost per layer). So wide EP improves both decode
  latency and per-GPU throughput for the expert compute: ~5-7x from EP4 to EP64 at >= 4
  requests/GPU; diminishing per doubling (EP32 -> EP64 at 8/GPU: 128 -> 86 us).
- 48 layers at 8 requests/GPU: ~26 ms of MoE per step at EP4 vs ~4.1 ms at EP64. EP64's dispatch +
  combine costs ~1.3 ms/step in the traces, far below the ~20 ms saved.
- Correction: an earlier framing compared EP sizes at the same *total* concurrency (which hands wide
  EP more GPUs per request) and concluded "wide EP buys latency, not per-GPU throughput". That was
  wrong; the fixed-per-GPU-load comparison above is the right one.

Caveats: rank 0 only (the real layer time is the busiest EP rank, which matters most at low load
and small EP; a busiest-rank rerun would tighten those rows); communication not included; TP = EP
above 8 is not runnable for Ultra today (Mamba `ngroups % tp`).

**Why TP = EP (or another host-known bound) is required for TRT-LLM.** In Megatron the NVLS
all-gather buffer is sized for the worst case, max_tokens / TP x EP rows (65,536 for Ultra TP8 x
EP64). The TRT-LLM kernel's cost grows with the rows it is given even when they are routed to -1
(~60 us + ~6-7 us per 1000 rows): on the full buffer it takes ~470 us/layer vs ~66 us on the
valid rows (`microbench/mlm_moe_bench.py`, Ultra TP8 x EP64):

| Concurrency | vllm_triton | trtllm, full 65,536-row buffer | trtllm, tight bound (rows = local x EP) | trtllm, fixed bound 512 x EP / TP (4096) |
|---|---|---|---|---|
| 128 | 108.2 us | 515.4 | 83.2 | 94.8 |
| 1024 | 163.7 | 519.2 | 86.5 | 99.5 |

The branch's original finalize loops over all buffer rows (28-37 us); bounding the loop by
`valid_tokens` makes it 2-10 us. Any bound is only safe if every EP rank is in a decode graph
(see the bounded-FlashInfer bug below); TP = EP makes it true by construction.

## 2026-10-15: optimization 2, fused tl.topk router (commit d2ec3f55e)

`megatron/core/inference/moe/sigmoid_topk.py`: one Triton launch for sigmoid + bias, top-k and
normalization. Selection is Triton's bitonic `tl.topk` on packed int64 keys (order-preserving
score bits << 32 | reversed expert index), top next_power_of_2(k) then keep k, so the cost does
not grow with k. Used by InferenceTopKRouter for sigmoid routing with an expert bias.
(A port of the router kernel from NVIDIA/Megatron-LM#8114 was tried first and reverted: it runs
k serial max-reductions on one CTA and took 57.9 us/layer at Ultra's top-22/512, a 1.6 ms/step
regression.)

Microbenchmark (`microbench/router_bench.py`, GB200, per call, 48 calls per CUDA graph):

| Tokens/GPU | Ultra (512, top-22) compiled | Ultra fused | nano (128, top-6) compiled | nano fused |
|---|---|---|---|---|
| 1-16 | 9.1-11.3 us | 2.9 us | 7.4-8.7 us | 2.0 us |
| 32-128 | 11.7-12.2 | 2.7-3.0 | 8.3-8.9 | 1.9 |
| 1024 | 37.9 | 9.1 | 12.1 | 3.0 |
| 4096 | 125.7 | 28.0 | 29.5 | 7.3 |

Same expert sets as the compiled path at every size; probabilities within 9e-8. Unit tests
(18) pass; nano smoke test passes (0 failing checks).

Ultra TP8 x EP64, concurrency 128 (`*_tltopk` vs `*_rmsnorm`, `analysis/decode_steps.py`):

| Per decode step | TP8 + FlashInfer norm | + tl.topk router | Delta |
|---|---|---|---|
| Routing + MoE bookkeeping | 1589 us | 1160 us | -429 |
| GPU busy | 11.82 ms | 11.35 ms | -0.47 |
| **Step period** | **12.52 ms** | **12.00 ms** | **-0.52 (-4.2%)** |

The router is 3.10 us/layer in the model (was gatherTopK ~10.5 us + sigmoid + normalize ~3 us).
Running total for Ultra decode at concurrency 128: 14.07 ms (TP4 baseline) -> 12.00 ms.

## 2026-10-15: GEMM bandwidth vs speed of light (Ultra TP8 x EP64, concurrency 128)

Run `20261015_134511_trace_ultrav3_tp8_ep64_nvls_bs128_rmsnorm`, rank 4 (rank 44 within ~2%),
medians over 128 decode steps; `analysis/gemm_bandwidth.py`. At ~2-16 rows per GPU every GEMM is
a weight read, so speed of light (SOL) = weight bytes / 7 TB/s (GB200 peak 8 TB/s).

| GEMM | Calls/step | Weight/GPU | Time/call (incl. split-K reduce) | Achieved | % of peak | SOL/call | Gap/step |
|---|---|---|---|---|---|---|---|
| Mamba in_proj | 48 | 71.8 MB | 16.3 us (13.0 + 3.3) | 4.41 TB/s | 55% | 10.3 us | 289 us |
| Mamba out_proj | 48 | 33.6 MB | 7.6 us | 4.44 TB/s | 56% | 4.8 us | 132 us |
| Attention qkv | 12 | 17.8 MB | 10.6 us (7.9 + 2.8) | 1.68 TB/s | 21% | 2.5 us | 97 us |
| Attention proj | 12 | 16.8 MB | 5.3 us | 3.14 TB/s | 39% | 2.4 us | 35 us |
| Router gate (fp32) | 48 | 16.8 MB | 6.9 us (4.5 + 2.5) | 2.42 TB/s | 30% | 2.4 us | 218 us |
| Latent fc1 (not TP-sharded) | 48 | 33.6 MB | 11.3 us (8.5 + 2.8) | 2.97 TB/s | 37% | 4.8 us | 312 us |
| Shared expert fc1 | 48 | 21.0 MB | 11.1 us (8.5 + 2.7) | 1.88 TB/s | 24% | 3.0 us | 391 us |
| Shared expert fc2 | 48 | 21.0 MB | 5.8 us | 3.62 TB/s | 45% | 3.0 us | 134 us |
| Routed experts fc1 (8 local) | 48 | 167.8 MB | 31.0 us | 5.41 TB/s | 68% | 24.0 us | 338 us |
| Routed experts fc2 (8 local) | 48 | 167.8 MB | 47.0 us | 3.57 TB/s | 45% | 24.0 us | 1,106 us |
| Latent fc2 (not TP-sharded) | 48 | 33.6 MB | 7.7 us | 4.37 TB/s | 55% | 4.8 us | 139 us |
| LM head | 1 | 268.4 MB | 40.4 us | 6.65 TB/s | 83% | 38.3 us | 2 us |
| **Total** | | **27.9 GB/step** | **7.19 ms/step** | **3.88 TB/s** | **48%** | **3.98 ms/step** | **3.20 ms/step** |

- Only the LM head is near SOL. The 3.2 ms gap (~26% of the 12.5 ms step) is mostly routed fc2
  (1.1 ms; 832 CTAs vs fc1's 2080), the small 17-34 MB GEMMs plus their split-K reduces
  (~1.5 ms; router, latent fc1 and shared fc1 read the same input and could be one GEMM), and the
  Mamba projections (0.4 ms).
- Attention also runs a 17-19 us NCCL all-gather of the qkv output per layer (KV heads < TP and
  interleaved qkv weights; `attention.py:1930`), ~0.23 ms/step; reordering the qkv weight per rank
  (+3 MB reads) would remove it.

## 2026-10-14: optimization 1, RMSNorm on FlashInfer

All ~108 per-step RMSNorms ran TE's untuned `rmsnorm_fwd_general_kernel` (8.2 us/call in the
trace, 0.88 ms/step). Speed of light at decode sizes is latency, ~1.5-2.5 us (80 KB per call at
2 rows x 8192).

Microbenchmark (`microbench/rmsnorm_bench.py`, GB200, 108 calls per CUDA graph, us per call):

| Rows (hidden 8192) | TE | FlashInfer | Triton (16 warps) |
|---|---|---|---|
| 1-2 | 5.3-5.6 | **1.68-1.69** | 2.18-2.20 |
| 4-16 | 2.90-2.96 | **1.68-1.70** | 2.18-2.19 |
| 32-128 | 2.95-4.33 | **2.56-2.95** | 3.01-3.45 |
| 512 | 7.24 | **4.91** | 5.20 |
| 2048 | 16.98 | **12.76** | 13.29 |

FlashInfer is best or tied at hidden 4096 and 2688 too.

Change (`megatron/core/tensor_parallel/inference_layers.py`): `_rms_norm_kernel` (was
`_te_rms_norm_kernel`) calls `flashinfer.norm.rmsnorm` for bf16/fp16 inputs with a same-dtype
weight and 16-byte rows, else TE; batch-invariant mode is unchanged. `InferenceTENorm` builds
TENorm and switches a plain TE RMSNorm to a subclass that uses the same kernel in eval (same
state dict); the hybrid inference spec's MoE `pre_mlp_layernorm` uses it. Covers Mamba in_proj
(48) + attention qkv (12) + MoE pre-norm (48) per step; the final norm (1) stays on TE.

Validation (`microbench/check_inference_rmsnorm.py`): bit-identical to TE at 1-2 rows, <= 1 bf16
ulp (0.0156) at 128-2048 rows, exact under CUDA-graph replay; InferenceTENorm state-dict keys
match TENorm. Nano end-to-end: FlashInfer path active on all ranks, graphs captured, generation
OK. Expected: ~0.6 ms/step (~4%) on Ultra at concurrency 128; to be confirmed by a re-trace.

Nano sweep (`*_rmsnorm_check_nano3`, `nvls` fp32 combine, GSM8K, 1024 output tokens, 3 timed
iterations, iterations within ~2%) vs the 2026-10-12 `nvls` fp32 sweep. Not a pure A/B: the
branch also gained #8002, #8030 and the `nvl_a2a` grid fix in between.

| Concurrency | ms/token now | 2026-10-12 | Change |
|---|---|---|---|
| 1 | 3.28 | 3.42 | -4.1% |
| 8 | 4.12 | 4.53 | -9.1% |
| 32 | 5.43 | 5.54 | -2.0% |
| 128 | 7.33 | 7.46 | -1.7% |

## 2026-10-14: per-step kernel breakdown at concurrency 128 (nsys)

Runs `*_trace_ultrav3_ep64_{nvl_a2a,nvls}_bs128`: client on its own node, `LOG_INTERVAL=1000`,
128 requests x 128 output tokens, every rank under nsys. Wall time for the profiled batch: 3.44 s
(`nvl_a2a`) vs 3.29 s (`nvls`).

**One `nvls` decode step, all kernels grouped (14.8 ms of kernel time):**

| Category | Kernels | Time per step | Share |
|---|---|---|---|
| Dense GEMMs (Mamba / attention / shared-expert / latent projections) | `nvjet_*` (4 variants), `cublasLt::splitKreduce` x2 | 4.31 ms | 29.1% |
| Routed expert GEMMs | `_fused_moe_kernel` | 3.81 ms | 25.7% |
| Routing + MoE bookkeeping | `gatherTopK`, `_moe_sum`, `_scatter_token_indices`, `_count_local_tokens`, `_init_sorted_ids`, `_mask_routing_padding`, `_fill_expert_block_ids`, `_prefix_sum`, fill, 2 router `triton_*` | 1.94 ms | 13.1% |
| TP communication | `_multimem_all_gather` + `_multimem_reduce_scatter` (215 calls), NCCL ring all-gather (13) | 1.54 ms | 10.4% |
| EP communication | AGV-3tensor + RSV + `_fused_metadata` | 1.28 ms | 8.7% |
| Norms | `rmsnorm_fwd` (107 x 8.2 us), Mamba gated `layer_norm` | 0.98 ms | 6.6% |
| Mamba recurrence | `_selective_scan_update`, `causal_conv1d_update` | 0.46 ms | 3.1% |
| Elementwise / copies | `add_copy`, add, `pow_relu`, gather, copy | 0.31 ms | 2.1% |
| Attention | flash forward + combine, KV append, cat | 0.17 ms | 1.1% |

**EP collectives per decode step** (median x 48 calls, ranks 4 and 44, decode steps only):

| | `nvls` | `nvl_a2a` | Delta |
|---|---|---|---|
| Dispatch | 536-588 us (11.2-12.3 us/call) | 909-948 us (18.9-19.7 us/call) | ~+7 us/call |
| Combine | 679-779 us (14.1-16.2 us/call) | 799-850 us (16.6-17.7 us/call) | ~+2 us/call |
| EP total | 1.22-1.37 ms | 1.71-1.80 ms | **+0.43-0.49 ms/step (~3% of the step)** |

TP collectives: ~5.5-6.5 us/call median with very little spread, 1.2-1.4 ms per step. The
0.6-1 us/call TP difference between the two runs is most likely node-placement noise: TP and EP
collectives run in sequence, not concurrently.

**Matches the microbenchmark.** The GB300 microbenchmark (EP64, hidden 1024, 88 layers) gives
1.15 vs 1.95 ms per step at B=128, a 0.80 ms gap. Scaled by layer count (48/88), that predicts
**0.44 ms** for Ultra; the traces show 0.43-0.49 ms. At ~2 tokens per rank both collectives are
latency-bound, so the 2x wider latent rows barely matter; the layer count scales the gap linearly.

**Skew (ranks 4 and 44 aligned call by call).** Each rank's trace uses its own node's clock;
removing the per-call end-time offset leaves residuals within +-2 us on 93-100% of calls, so the
alignment is valid.

| | `nvls` dispatch | `nvls` combine | `nvl_a2a` dispatch | `nvl_a2a` combine |
|---|---|---|---|---|
| Median \|arrival skew\| r4 vs r44 | 1.2 us | 3.7 us | 1.1 us | 3.1 us |
| Arrival skew p5 / p95 | -1.6 / +3.7 | -7.0 / +10.8 | -1.5 / +3.9 | -7.1 / +9.3 |
| Duration on the later arriver, p5 / p50 / p95 | 8.5 / 10.8 / 14.9 | 7.6 / 12.8 / 19.3 | 15.7 / 18.6 / 21.3 | 9.2 / 15.1 / 22.4 |

Combine is more skewed than dispatch: it follows the uneven routed-expert GEMMs. The later of two
ranks still runs above the microbenchmark (~6 us `nvls`, ~10 us `nvl_a2a` per call, from an even
dispatch/combine split of the plot), but that is an upper bound on the no-wait cost: the last of
64 ranks can be later than both. Skew inflates both dispatchers roughly equally, so it does not
explain the gap between them.

**Takeaways**

- The dispatcher choice is worth ~4.5 us per call (~0.45 ms, ~3% per step at 128). All of EP
  communication is <9% of the step; the barrier alone is worth less.
- Bigger levers at low/medium concurrency:
  1. Routing + MoE bookkeeping, 1.94 ms (13%): ~11 small kernels per MoE layer (~40 us/layer);
     fusing them could recover ~1.5 ms. `gatherTopK` alone is 0.44 ms.
  2. GEMMs at roughly half of HBM bandwidth: dense + routed take 8.1 ms vs an estimated ~4.2 ms to
     read ~16 GiB of dense and ~16 GB of expert weights at ~8 TB/s (rough estimate). Split-K
     reduces add 0.5 ms.
  3. RMSNorm 0.88 ms (6%): 8.2 us per call for a few rows of 8192 should be ~2-3 us, or be fused
     into the TP collective.
  4. TP communication 1.54 ms (10%), already ~6 us per call; the 13 NCCL ring all-gathers could
     move to symmetric memory (~0.1 ms).
- Rough ceiling: ~14.8 -> ~10-11 ms per step (25-30%) from the levers above, vs ~3% from the
  dispatcher.

Open: the end-to-end gap at concurrency 128 (~0.26-0.29 ms) is smaller than the traced 0.45 ms
(single runs on different node sets; async overlap). A same-allocation A/B would settle it. Export
all 64 ranks to get the true last arriver per call.

## 2026-10-14: decode sweep with the client on its own node, `LOG_INTERVAL=1000`

Runs `*_decode_ultrav3_ep64_{nvl_a2a,nvls}_li1000`: `server/client.sh` on a separate node,
GSM8K prompts, 1024 output tokens, 1 warmup + 1 timed iteration per concurrency.
Plot: `analysis/ultrav3_ep64_client_sweep_li1000.png` (`analysis/plot_client_sweep.py`).

| Concurrency | `nvl_a2a` wall | `nvls` wall | `nvls` speedup | `nvl_a2a` tok/s | `nvls` tok/s |
|---|---|---|---|---|---|
| 1 | 12.89 s | 12.65 s | +1.9% | 79 | 81 |
| 2 | 12.93 | 12.87 | +0.5% | 158 | 159 |
| 4 | 13.99 | 13.83 | +1.1% | 293 | 296 |
| 8 | 13.83 | 13.65 | +1.3% | 592 | 600 |
| 16 | 13.98 | 13.76 | +1.6% | 1,172 | 1,191 |
| 32 | 14.75 | 14.28 | +3.3% | 2,221 | 2,295 |
| 64 | 15.37 | 18.10 | -15.1%* | 4,264 | 3,621 |
| 128 | 16.00 | 15.71 | +1.9% | 8,191 | 8,342 |
| 256 | 16.70 | 16.50 | +1.2% | 15,700 | 15,885 |
| 512 | 19.45 | 19.55 | -0.5% | 26,961 | 26,820 |
| 1,024 | 23.06 | 23.18 | -0.5% | 45,470 | 45,239 |
| 2,048 | 32.85 | 33.26 | -1.2% | 63,845 | 63,062 |
| 4,096 | 55.13 | 52.68 | +4.6% | 76,086 | 79,616 |

\* Likely an `nvls` host-side stall (single iteration).

- 15-30% faster than the earlier runs (per-step logging on, client on rank 0's node); the
  concurrency-4 outlier is gone and throughput keeps climbing to 76-80k tok/s at 4096 (old peak
  59k).
- `nvls` leads by 0.5-3.3% at 1-256; `nvl_a2a` by 0.5-1.2% at 512-2048.
- Use `BENCH_ITERS=3` (median) for publication numbers: one stall flips a single-iteration point.

## 2026-10-14: engine step time vs global batch (per-step logs)

Runs `20261014_135642_decode_ultrav3_ep64` (`nvl_a2a`) and `*_decode_ultrav3_ep64_nvls`,
`LOG_INTERVAL=1`, in-job client. Steps from the 16 group leads joined on wall-clock second; global
tokens = sum of each group's tokens; pure-decode seconds only; seconds with a host stall (any step
> max(200 ms, 4x the bucket median)) dropped. Plot: `analysis/ultrav3_ep64_step_time_16_2048.png`
(`analysis/step_time_vs_tokens.py`).

| Global tokens | `nvl_a2a` | `nvls` | Delta |
|---|---|---|---|
| 16 | 15.40 ms | 15.21 ms | +1.3% |
| 32 | 15.93 | 15.54 | +2.5% |
| 64 | 16.65 | 16.24 | +2.5% |
| 128 | 17.18 | 16.91 | +1.6% |
| 256 | 17.57 | 17.28 | +1.7% |
| 512 | 20.15 | 19.97 | +0.9% |
| 1,024 | 23.18 | 23.10 | +0.3% |
| 2,048 | 31.97 | 32.24 | -0.9% |

- Nearly flat to 256 tokens (fixed per-step cost), bending upward from ~512.
- Logged `time:` excludes host time between steps: the step rate implies a ~12% longer period.

## 2026-10-14: host-side stalls in the per-step-logging runs

- Two patterns, both host-side gaps between steps (not GPU work):
  1. Multi-second freezes at concurrency 4-16 from rank 0 (which also hosted the HTTP server,
     coordinator and client), clustered at batch-size transitions. This caused the old
     concurrency-4 outlier.
  2. ~1 s hiccups at 4096 (256 requests per group), mostly with group 44 as the late arriver.
- A rank that causes a stall logs a normal `time:` with a long gap before its next log line; every
  other group logs one inflated step (it waited inside the EP collective).
- Stalls explain every end-to-end difference between the dispatchers in those runs to within
  ~0.2 s. Moving the client off rank 0's node and turning off per-step logging removed most of
  them.

## Earlier results (summary)

- Super v3 (TP4 x EP4): after the shared-expert-overlap fix, `nvl_a2a` ~= `nvls` (~1% slower at
  1k-4k tokens, 5% faster at 8k).
- Ultra v3 at TP8 x EP8 (2 nodes): `nvl_a2a` ~= `nvls` on prefill (+-2%) and decode (~10.9 ms at
  batch 1). At EP8 a token reaches ~90% of ranks, so unicast saves little.
- At EP64 a token reaches ~19 of 64 ranks (30%), so `nvl_a2a` moves 3.4x fewer bytes. That only
  matters once steps carry thousands of tokens (prefill, very large decode batches).

## Next steps

- [ ] Export all 64 ranks' traces; per call, find the last arriver and its duration (true no-wait
      cost); compare with the microbenchmark run on our GB200 nodes.
- [ ] Same-allocation A/B (both dispatchers back to back on the same 16 nodes), `BENCH_ITERS=3`.
- [ ] EP64 prefill sweep, where `nvl_a2a`'s 3.4x byte saving should show.
- [ ] Prototype: fuse the per-layer routing/bookkeeping kernels; check RMSNorm fusion into the TP
      collectives; look at small-M GEMM efficiency.
- [ ] Cheaper barrier (epoch / multicast counter): applies to ~300 collectives per step (TP + EP).
- [ ] (Parked) TP16/TP32 for Ultra: Mamba asserts `ngroups % tp_size == 0` (8 groups, so max
      TP8). Simplest unblock: shard in_proj naively and all-gather B/C per layer, as attention
      already does for qkv when KV heads < TP. SSM state still shards cleanly by head.
- [ ] BUG (to fix): bounded FlashInfer MXFP8 MoE (`inference_flashinfer_mxfp8_token_capacity`) decides
      per rank. NVLS skips the EP graph consensus (`match_ep_token_counts` only for NCCL/training
      dispatchers), so a decode-only rank truncates to `capacity` rows while a peer EP rank prefills;
      rows past it combine stale buffer data (peer's prompt and later ranks' decodes corrupted). Fix:
      EP consensus every step when the capacity is set, and bound only on CUDA-graph steps. Repro
      scaffold (nano TP1 x EP4, needle + counting checks): `repro_bounded_fi/` (`run.sh` + `repro_client.py`
      on a worktree of upstream main `ee6baa2e3`; not yet run).
- [x] Router: fused sigmoid + bias + tl.topk kernel (done, -0.52 ms/step on Ultra TP8).
- [ ] Integrate the TRT-LLM BF16 routed MoE for TP = EP (rows = host-known batch padded to TP;
      -1 routing; padding-aware finalize with the valid-token loop). Measured 1.03-1.77x faster than
      vllm_triton per layer (see "routed-expert grouped GEMM" section). Rerun the TP = EP sweep with
      the busiest EP rank instead of rank 0. Later: fuse the per-step -1 routing fill and the
      id<<16 | bf16-prob packing into the dispatch (AGV) instead of separate full-buffer passes.
- [ ] Reintroduce the parallel expert-alignment kernel on its own (~4 us/layer, ~0.2 ms/step).
- [ ] GEMMs: ~3.2 ms/step gap to speed of light; routed fc2 (1.1 ms) first.
- [ ] TP1 (memory fits: ~79 GiB weights per GPU) is a throughput-only experiment: it removes
      ~1.5 ms of TP communication but adds ~6.5 ms of dense weight reads per step at low batch.
