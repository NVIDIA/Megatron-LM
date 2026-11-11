# KV-parallel decode attention for Nemotron 3 Ultra (high TP without KV duplication)

Status: design note, 2026-10-15. No code, no runs. Each number is labelled with one of these
tags:
- **[M]**: measured (from `inference-bench/REPORT.md`, nsys TP4 x EP64, concurrency 128).
- **[C]**: computed exactly from the model config.
- **[E]**: my estimate, with its assumptions stated.

The TP8 x EP64 trace (`trace_ultrav3_tp8_ep64_nvls_bs128_rmsnorm`) is still running. It replaces
every [E] TP8 step-time number below.

## The idea

At high TP with only 2 KV heads, standard Megatron replicates each KV head across TP/2 ranks
(`num_attention_heads_per_partition = 1` when `num_query_groups < tp`, see
`dynamic_context.py:433`). Instead, the TP ranks that share a KV head would each store a disjoint
slice of the tokens. Each rank computes partial attention over its slice, and the partials are
merged with online softmax (o, lse) over NVLink. In the literature this is KV parallelism (Medha),
Decode Context Parallelism (DCP in vLLM/SGLang), or the attention half of Helix.

## TL;DR

1. **Memory.** KV duplication only matters above roughly 8-17k tokens. Below that, Mamba state
   (98 MiB per request per GPU at TP4) dominates per-request memory [C]. Above it, the scheme
   gives these concurrency gains [C]:
   - TP4: 1.2x at 8k, 1.5x at 32k, 1.8x at 128k, 2.0x at 1M.
   - TP8: 1.6x at 8k, 2.5x at 32k, 3.4x at 128k, 3.9x at 1M.
2. **Clean invariant.** At a fixed global concurrency, perfect KV sharding makes peak concurrency
   independent of TP, because Mamba state and ideal KV both scale as 1/TP. Duplication costs
   exactly a factor of TP/2 on the KV term. So "go TP8 for decode" without KV parallelism cuts the
   long-context capacity of the deployment in half. With KV parallelism, TP8 is free in memory
   (TP8 even frees ~9 GB per GPU of weights) [C/E].
3. **Bandwidth.** Duplication also doubles the KV bytes each GPU reads per decode step at TP8, at
   a fixed global batch. At 128k context and global concurrency 128 this is about +1 ms per step
   [E], half of the dense-GEMM saving TP8 buys. KV parallelism turns it into a 0.5 ms saving
   instead.
4. **Cost.** For 12 attention layers: about 1-2 extra small NVLink collectives per layer, so
   ~0.1-0.2 ms per step (~1% of a ~13 ms step) [E]. The cost is independent of context length.
5. **TP16 is impossible for Mamba as written.** `mamba_mixer.py:275` asserts
   `ngroups % tp_size == 0`, and ngroups = 8. TP8 is the ceiling unless Mamba runs at its own,
   lower TP.
6. **Verdict.** Worth building if the target workload is long context (≥32k: agentic SWE, RL
   rollouts) at high concurrency, and only after TP8 itself is shown to win on step time. Not
   worth it for GSM8K-length decode, where memory is not binding (8 requests per group in use vs
   ~520-740 that fit).

## 1. Does higher TP help decode?

Model the change at a **fixed global concurrency of 128** with 64 GPUs:
- TP4 = 16 DP groups x 8 requests, 2 tokens per rank after sequence parallel.
- TP8 = 8 DP groups x 16 requests, still 2 tokens per rank.

This framing matters. Per-rank token counts for routing, EP, and norms do not change. Only the
work that is sharded by TP changes.

| Component | TP4 [M] | TP8 [E] | Reasoning |
|---|---|---|---|
| Dense GEMMs (Mamba in/out proj, shared expert, attn, latent) | 4.31 ms | 2.4-2.8 ms | Weight bytes per GPU halve (Mamba 20.2B + shared 8.2B + attn 1.7B dominate). M goes 8 -> 16, still weight-bound. Smaller N per rank lowers HBM efficiency (fewer CTAs, more split-K), so not a clean 2x. Router/latent may stay replicated (floor). |
| Routed expert grouped GEMM | 3.81 ms | ~3.8 ms | ETP=1, so each GPU still reads its 8 experts. Same global tokens, so same per-GPU work. |
| Routing / MoE bookkeeping | 1.94 ms | ~1.94 ms | Latency-bound on 2 tokens per rank; unchanged. |
| TP collectives (~215 calls) | 1.54 ms (~6.3 us/call, 4 GPUs, 1 tray) | 1.7-2.0 ms | Same call count. 8 participants across 2 trays (still NVLink/NVLS inside the NVL72). Guess +1-3 us per call. |
| EP dispatch/combine | 1.28 ms | ~1.28 ms | Same EP64, same tokens per rank. |
| Norms | 0.98 ms (-> ~0.3 with FlashInfer) | same | Per-rank rows unchanged. |
| Mamba recurrence | 0.46 ms | ~0.46 ms | Twice the requests per group x half the heads per rank = same state bytes per GPU. |
| Attention, short context | 0.17 ms | ~0.2 ms | Launch-bound. |
| **Step** | **14.8 ms** | **~12.9-13.5 ms** | Roughly 9-13% faster [E] |

Where higher TP stops helping:
- **TP8 -> TP16.**
  - Blocked by Mamba ngroups = 8.
  - Even with Mamba at TP8 and the rest at TP16, the only weights left to shrink are shared
    expert and attention (~10B params, so ~1.3 GB less to read per GPU per step, ~0.25 ms [E]).
  - Collectives then span 4 trays: 215 calls x (+1-3 us) = +0.2-0.6 ms [E].
  - Net: roughly zero.
- **The irreducible part.** About 9 ms of the 14.8 ms step does not shrink with TP at all:
  routed experts, routing, EP, norms, recurrence. That is the Amdahl ceiling. TP8 is probably the
  sweet spot. The bigger levers are still the ones in REPORT.md (fused routing ~1.5 ms, expert
  GEMM efficiency ~2 ms).
- **Prefill.** Your intuition is right: prefill is compute-bound with large M, and higher TP only
  adds collective volume, which scales with tokens. Low TP is better for prefill.

## 2. Memory per request: KV vs Mamba state

The KV cache per token for the whole model is 12 layers x 2 KV heads x 128 x (K+V) x bf16 =
**12,288 B = 12 KiB** [C].

Mamba state per request (whole model) [C]:
- SSM state: 48 layers x 256 heads x 64 x 128 x fp32 = 384 MiB.
- Conv state: 48 x (16384 + 2x8x128 channels) x 4 x bf16 = 6.75 MiB.
- Total ≈ **391 MiB**, sharded by TP.
- The TP4 figure (97.7 MiB) matches the server's 31.7 GiB / 332 requests.

| TP | Mamba per request per GPU | KV per token per GPU, duplicated | KV per token per GPU, ideal | Crossover (KV = Mamba), duplicated | Crossover, ideal |
|---|---|---|---|---|---|
| 1 | 391 MiB | 12 KiB | 12 KiB | 33k tokens | 33k |
| 2 | 195 MiB | 6 KiB | 6 KiB | 33k | 33k |
| 4 | 97.7 MiB | 6 KiB (2x) | 3 KiB | **17k** | 33k |
| 8 | 48.8 MiB | 6 KiB (4x) | 1.5 KiB | **8.3k** | 33k |
| 16 | n/a (Mamba ngroups) | 6 KiB (8x) | 0.75 KiB | n/a | n/a |

With duplication, KV per token per GPU stays stuck at 6 KiB for every TP ≥ 2. That is the whole
problem in one number.

### Peak concurrency

Assumptions: 80 GB buffer per GPU, perfectly packed Mamba + KV (ignores the `MAMBA_RATIO` split,
block rounding, and the paused-block reserve), all requests at context L [C].

| L | TP4, duplicated | TP4, KV-parallel | TP8, duplicated | TP8, KV-parallel | TP8, KV-parallel, +9 GB freed weights [E] |
|---|---|---|---|---|---|
| 1k | 11.8k | 12.1k | 11.2k | 12.1k | 13.5k |
| 8k | 8.4k | 10.0k | 6.3k | 10.0k | 11.2k |
| 32k | 4.2k | 6.3k | 2.5k | 6.3k | 7.0k |
| 128k | 1.41k | 2.53k | 0.75k | 2.53k | 2.82k |
| 1M | 196 | 385 | 99 | 385 | 428 |

All values are global concurrency (64 GPUs); per group = global / (64 / TP).

Read the table as follows:
- Below ~8k tokens, the scheme is worth ≤ 20%. Mamba state is the binding constraint and already
  shards perfectly.
- At 32-128k it is worth 1.5-1.8x at TP4 and 2.5-3.4x at TP8.
- These are capacity ceilings. They only matter if the workload actually runs that many requests
  at that context. GSM8K uses 8 per group out of ~700 that fit.

## 3. Design: KV-parallel decode inside the TP group

Notation: TP = 8, Hkv = 2. Each KV head has a **KV group** of G = TP / Hkv = 4 ranks. Each rank
holds 8 of the 32 query heads that attend to its KV head. This is Helix's N = KVP x TPA with
TPA = 2 and KVP = 4.

### Token placement

| Option | Balance | Prefix cache / block hashing | Writes | Verdict |
|---|---|---|---|---|
| Round-robin per token (vLLM default, interleave = 1) | Perfect: within 1 token per request | A "virtual block" of G x block_size tokens maps to one physical block per rank. Hashes stay per virtual block and are identical on every rank. | Every rank computes K and V (the KV projection is replicated at TP > 2 anyway). Owner rank = pos % G stores. No communication. | **Recommended** |
| Per block (interleave = block_size) | Off by up to 1 block per request; short requests sit on one rank | Simplest; whole blocks. | Same | Fine for long context; poor at short context and in mixed batches |
| Contiguous chunks | Poor while a request grows; the last chunk is hot | Hard | Same | No |

Megatron specifics:
- The block allocator stays rank-local but must allocate in lockstep across the KV group:
  - same virtual block IDs on every rank;
  - each rank's physical block holds tokens {r, r+G, r+2G, ...} of the virtual block.
- The per-request KV length for rank r is ceil((L - r) / G), computed on the host, so it is
  deterministic and needs no communication.
- Prefix caching works unchanged at virtual-block granularity.
- Mamba state is unaffected: it is already sharded by head.

### Getting queries to the owners

Every rank in a KV group needs all 32 query heads of that group. With sequence parallel, the
hidden state is all-gathered before QKV anyway. There are two options:
- **(a) All-gather Q** inside the KV group after the QKV projection. That is R x 32 x 128 x bf16 =
  8 KB per request, ~6-7 us per layer [E].
- **(b) Replicate q_proj inside the KV group**, so each rank computes all 32 heads. No collective,
  but an extra 0.75 x (64 x 128 x 8192 x 12 layers) / 2 ≈ 0.3B params ≈ 0.6 GB of weight reads
  per step, ~0.1 ms [E]. That is about the same as (a) and costs memory, so prefer (a).

### Merge (online softmax)

Each rank produces partial outputs o_r for 32 heads (fp32, 16 KB per request) plus lse_r for
32 heads. Two variants:
- **(A) All-to-all of (o, lse) inside the KV group.** Each rank receives its 8 heads' partials
  from the other 3 ranks, then merges locally (FlashInfer `merge_states`). The normal row-parallel
  o_proj and the existing reduce-scatter follow unchanged. Cost: 1 collective of ~R x 12 KB per
  rank. **SGLang's `cp_lse_ag_out_rs` is AG + RS; one all-to-all is strictly better.**
- **(B) Fuse the merge into the existing post-attention reduce-scatter.** All-gather only the lse
  (32 floats per request, ~6 us). Each rank rescales its partial
  o_r <- exp(lse_r - lse_global) * o_r. The output projection is linear:
  sum_r (w_r o_r) W_o = (sum_r w_r o_r) W_o. So if every rank applies o_proj to all 32 heads of
  its group, the **existing** SP reduce-scatter performs the merge for free.
  - Cost: o_proj replicated across G (+0.6 GB of reads per step, ~0.1 ms, plus memory).
  - Collectives: 1 tiny lse AG.
  - It also needs a numerically matching rescale, and fp32 vs bf16 partials before the RS.

Extra communication per decode step (12 layers), with 6-8 us per small NVLS collective [E] and
message size irrelevant at decode batch sizes:

| Variant | Extra collectives per step | Extra time [E] | Extra weight reads per step |
|---|---|---|---|
| Q-AG + o/lse all-to-all | 24 | 0.15-0.2 ms | 0 |
| Q-AG + lse-AG + fused RS (B) | 24 | 0.15-0.2 ms + ~0.1 ms of o_proj reads | ~0.6 GB |
| Replicated q_proj + o/lse all-to-all | 12 | 0.08-0.1 ms + ~0.1 ms of q_proj reads | ~0.6 GB |
| One fused kernel: Q-AG -> attention -> push partials via symmetric memory -> merge (Helix HOP-B style, overlap batch halves) | ~12 effective | ~0.1 ms, partly hidden | 0 |

Against that, at TP8 and global concurrency 128 the KV bytes read per GPU per step for all 12
attention layers are [E, at ~6 TB/s effective]:

| Context | Duplicated | KV-parallel |
|---|---|---|
| 32k | 3 GiB (~0.5 ms) | 0.75 GiB (~0.13 ms) |
| 128k | 12 GiB (~2.1 ms) | 3 GiB (~0.5 ms) |

So KV parallelism also **wins on latency** above roughly 10-15k tokens at this batch. It is
memory **and** bandwidth, not just memory.

### Other concerns

- **Load balance.** Token round-robin is balanced to within 1 token. Imbalance across requests
  does not matter because every rank sees every request. Mixed short/long batches are still
  balanced, unlike attention-DP or request-level CP.
- **CUDA graphs.** Fine:
  - Per-rank KV lengths and page tables are host-built metadata, the same way they work today.
  - Collective sizes depend only on the batch bucket, not on context.
  - The symmetric-memory NVLS/a2a kernels in `inference/communication/torch_symm_triton/` are
    already graph-safe and are the natural place to put the (o, lse) exchange.
- **Chunked prefill.** Writes go to owner ranks. Chunk queries attend to the distributed prefix
  with the same partial-plus-LSE merge. The Q-AG volume is chunk x 32 heads x 256 B (8k chunk:
  64 MB per layer, ~0.1 ms) against roughly 50+ ms of prefill compute per chunk, so it is
  negligible [E]. Alternatively, all-gather the prefix KV for prefill only (more bytes, simpler
  kernel).
- **Speculative decoding / MTP.** Draft KV must use the same interleave. vLLM had exactly this bug
  (draft config dropped the interleave size and acceptance collapsed).
- **Kernels.** FA2/FA3/FA4 and FlashInfer decode kernels already return LSE. The merge kernel
  exists (`flashinfer.cascade.merge_states`). The new code is the allocator interleave, the
  metadata, and the exchange.

## 4. Prior art

| Work | What it does | Closeness | What to borrow |
|---|---|---|---|
| **Helix Parallelism** (NVIDIA, arXiv 2507.07120) | Attention runs on N = KVP x TPA GPUs with TPA ≤ #KV heads, so there is no duplication. The same GPUs then run TP/TPxEP for the FFN. All-to-all of partials plus LSE merge. HOP-B overlaps the communication with the next batch's compute. Up to 32x more users and 1.5x interactivity at multi-million context. | **Essentially your idea**, generalised. | The TPA/KVP split, a2a merge, HOP-B overlap. https://arxiv.org/abs/2507.07120, vLLM RFC https://github.com/vllm-project/vllm/issues/34018 |
| **vLLM Decode Context Parallel (DCP)** | Shards KV inside the TP group. Q all-gathered, partial attention per rank, LSE-corrected combine. `cp_kv_cache_interleave_size` (formerly `dcp_kv_cache_interleave_size`): 1 = token round-robin, = block_size for disaggregation. GQA rule: TP / num_kv_heads ≥ DCP. | Same | Interleave semantics; virtual-block mapping (vLLM-Ascend doc). https://docs.vllm.ai/en/latest/api/vllm/config/parallel/, https://docs.vllm.ai/projects/ascend/en/main/developer_guide/feature_guide/context_parallel.html |
| **SGLang DCP** (`--dcp-size`) | Position p goes to rank p mod c. The slot allocator widens by dcp_size. `cp_lse_ag_out_rs_mha` merges partials. Roadmap includes Helix. Ascend port: 8.66x context for the same HBM. | Same | Allocator trick (slot // dcp_size); GQA caveat that qkv sharding must match. https://lmsysorg.mintlify.app/docs/advanced_features/dcp, https://github.com/sgl-project/sglang/issues/29736 |
| **Medha / KVP** (arXiv 2409.17264) | KV parallelism across servers. Q is copied to every GPU, online-softmax merge. Communication is independent of KV length. Combined with TP and sequence pipeline parallelism up to 10M tokens. | Same core | Argument that the communication is O(q tokens). https://arxiv.org/abs/2409.17264 |
| **Flash-Decoding / split-KV** | The same LSE merge, but across SMs inside one GPU. | Intra-GPU version | It is the proof that the merge is exact and cheap; the existing FA4 num_splits path. https://crfm.stanford.edu/2023/10/12/flashdecoding.html |
| **Tree Attention** (Zyphra, arXiv 2408.04093) | Tree-reduce the logsumexp across devices in log(P) steps; up to 8x faster than ring for decode. | Merge topology | With G = 4 on NVLink, one a2a or multimem beats a tree. Relevant only at large KVP. https://arxiv.org/abs/2408.04093 |
| **Ring attention for decode / Meta's CP for million-token inference** (arXiv 2411.01783) | Ring pass-Q / pass-KV. Decode uses pass-Q plus an all-to-all merge. | Close | Pass-Q for decode, pass-KV for prefill heuristics. https://arxiv.org/abs/2411.01783 |
| **LoongServe** (SOSP'24, arXiv 2404.09526) | Elastic sequence parallelism. KV is spread over instances and decode merges partials; the degree of parallelism scales per request. | Cross-instance | Dynamic KVP degree per request (overkill here). https://arxiv.org/abs/2404.09526 |
| **Star Attention** (NVIDIA, arXiv 2411.17116) | Block-sparse prefill; decode via a global LSE merge across hosts. | Decode half is the same | Merge formulation. https://arxiv.org/abs/2411.17116 |
| **NanoCP** (arXiv 2605.21100) | Request-level dynamic CP for DP+EP decode. | Adjacent | Turn CP on only for long requests. https://arxiv.org/html/2605.21100 |

## 5. Alternatives

1. **Higher TP for dense/Mamba GEMMs, attention left as is.** This is where the step-time win is:
   - Attention weights are only 1.7B of ~34B dense params. Mamba (20.2B) and shared experts
     (8.2B) are what TP8 shrinks.
   - The ~10% TP8 gain is independent of attention.
   - KV duplication is a capacity and long-context bandwidth tax on top. **Measure TP8 first.**
2. **Attention TP decoupled from MLP/Mamba TP (attention at TP2 inside a TP8 group = attention
   DP x 4).**
   - Upside: no duplication and no LSE merge.
   - Costs:
     - Tokens must be all-to-all redistributed by request before and after each attention layer.
     - Attention weights at TP2 mean +1.5 GB per GPU of reads per step (~0.25 ms [E]) and of
       memory.
     - Per-request load imbalance across replicas, which is bad with long-tail agentic contexts.
   - KV parallelism dominates on weights and balance; attention-DP is simpler to build.
3. **Helix-style KVP x TPA** (= your idea framed generally). This is the right abstraction:
   TPA = 2, KVP = TP / 2.
4. **FP8 KV cache.**
   - 2x KV capacity and bandwidth with no new communication. FA3/FA4/FlashInfer support it. Main
     risk is accuracy (GQA with 32 q per kv head concentrates error).
   - Orthogonal and multiplicative with KVP: TP8 + KVP + fp8 gives 8x less KV per GPU than
     TP8-duplicated.
   - Cheapest first step for capacity.
5. **"Replicate only for the 12 attention layers."** That is already the situation: Mamba and MoE
   have no duplication. The duplicated bytes are 6 KiB per token, all of it attention.
6. **Mamba state in bf16, or fewer state slots.** At ≤ 8k context, Mamba (fp32 SSM, 98 MiB per
   request per GPU at TP4) is the binding constraint. bf16 SSM state would be worth more than KVP
   there, but correctness and accuracy are unverified.

## 6. Verdict and next steps

**Is it worth it?** Yes, conditionally. It pays off when both hold:
- contexts are ≥ ~16k (TP4) or ≥ ~8k (TP8), **and**
- the deployment is concurrency-limited by memory, or attention bandwidth shows up in the step
  (≥ ~32k at moderate batch).

For SWE-agent / RL rollouts at 32-128k on TP8 it is worth 2.5-3.4x capacity plus ~1.5 ms per step
at 128k [C/E]. For GSM8K-style short decode it is worth ~0.

Ranked recommendations:
1. **Get the TP8 step time first** (the trace already running). If TP8 is not ≥ ~8% faster than
   TP4 per step at equal global concurrency, then stay at TP4, where KVP is worth at most 1.5-2x
   at long context and fp8 KV gets most of that.
2. **Microbenchmark the merge.** On 4 and 8 GPUs (1 and 2 trays), CUDA-graphed, for decode batches
   of 1-64 requests x 32 heads, time:
   - Q all-gather + (o, lse) all-to-all + `merge_states`;
   - against the current single reduce-scatter.

   Target: ≤ 15 us per layer combined (≤ 0.2 ms per step). Also measure FA decode at 1/4 of the
   KV length (32 q per kv head) against the full length to confirm the bandwidth saving. This
   about 1 day of work de-risks the communication side.
3. **Measure the long-context decode step on Ultra** at TP4 and TP8 (e.g. 32k and 128k,
   ignore_eos, concurrency 64-128) to get the real attention share. The TP4 0.17 ms is
   short-context only. Combined with fp8 KV (cheap flag-level experiment, check accuracy), this
   tells you how much is left for KVP.

If 1-3 come out favourable, implement following vLLM/SGLang DCP:
- token-interleaved virtual blocks in `dynamic_context.py`;
- owner-only KV append;
- per-rank lengths and page tables;
- Q-AG + (o, lse) a2a over the symmetric-memory path;
- later, Helix HOP-B overlap or fusion variant (B).

### Open questions for Sid

- What context lengths and concurrency are the real targets? This decides everything above.
- Does the Ultra server actually hit `max_requests` or block-pool exhaustion today on agentic
  runs?
- Is a different TP for Mamba than for attention/shared-expert acceptable (needed for anything
  beyond TP8)?
