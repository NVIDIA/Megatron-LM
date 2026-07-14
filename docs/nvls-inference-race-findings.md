# NVLS inference dispatcher: inter-step race causing async IMA under agentic workloads

**For:** Teo (tde/logging-inference-mr66 / PR #5 lineage)
**From:** Jorge (SWE-RL grid, HSG GB200 NVL72)
**Date:** 2026-07-14
**TL;DR:** The nvls inference MoE dispatcher has a pre-existing step-boundary ordering gap. PR #5's config changes (max-tokens 8192→16384 default, max-requests 128→derived 696, NCG 2→32, decode-only dropped) don't introduce it but widen the timing window from "rare wave-tail crash" to "crashes nearly every SWE rollout wave." A per-forward `cuda.synchronize()` probe suppresses it completely (7/7 unprobed links crashed, 0/4 probed links crashed, same code, same params, same hour) — i.e., it is a concurrency race between consecutive engine steps, not a deterministic OOB. Single-step blend workloads essentially never open the window, which is why your runs are clean.

---

## 1. Setup

- Model: NemotronH 3B hybrid mamba/attention/MoE ("nano v3"), TP2 CP4 EP16 PP1, 16 GB200 nodes (64 GPUs), 8 DP inference engines, EP group spans engines.
- Params: exactly PR #5 — `--inference-dynamic-batching-num-cuda-graphs 32`, buffer 80 GB, no max-requests (derives to 696), no max-tokens (default 16384), `inference_moe_token_dispatcher_type=nvls`, unified-memory-level 1, `--rl-persist-cuda-graphs`, kv recompute mode, `--enable-chunked-prefill`, SL 131072.
- Workload (the trigger): SWE multi-turn RL (OpenHands agents, 128 episodes/wave). Engines constantly alternate idle↔busy as episodes wait on tool calls, so **EP dummy forwards run continuously** in lock-step with peers doing **multi-hundred-ms eager 16k-token chunked prefills**. Maximum step-time asymmetry across the EP group.
- Blend workloads (single-step, short prompts, no chunked prefill) keep all engines doing similar-sized steps in near-lockstep → window effectively never opens → clean runs. Confirmed by Jorge's production experience with the same code+params.

## 2. Symptom

Async `CUDA error: an illegal memory access was encountered` on engine ranks, surfacing at the EP dummy-forward hard sync:

```
File ".../megatron/core/inference/engines/dynamic_engine.py", line 3083, in run_engine_with_coordinator
    self.step_end_event.synchronize()
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

The location is a red herring — the dummy step's per-iteration sync is the only host sync in the engine loop, so any sticky async fault lands there. Typically 1–3 engines die within one rollout wave (~1–35 min after "Collecting rollouts"), coordinator removes them, link dies. Under the pre-PR config (max-tokens 8192, max-requests 128, NCG 2, decode-only) the same signature occurred at ~0.8 crashes/tail-hour at near-empty occupancy; under PR-config SWE load it fires nearly every wave on lag ≤ 2 chains.

## 3. The decisive experiment

`MRL_SYNC_AFTER_FORWARD=1` (env-gated `torch.cuda.synchronize()` after every forward — both real steps and dummy forwards; the dummy-path probe was added in `26335c7db`) exported to all grid submissions after 2026-07-14 04:14 UTC-7.

| Cohort (same hour, same code, same params, same cluster) | Links | IMA crashes |
|---|---|---|
| No probe (submitted < 04:14) | 7 (f0×2, f1×2, f2×2, f8, u8) | **7/7 crashed** in first real wave |
| Probe on (submitted > 04:14) | 4 (f8, u8, u2, f1) | **0/4** — deep in real waves (f8: 20,500 engine steps, 1,273 episodes finished) |

A fault that disappears under per-forward serialization requires overlap between consecutive steps' device work. Probe cost is negligible for this workload: decode replays measure 7.8–8.0 ms *with* the sync; iteration time is episode-wall-time dominated.

## 4. Mechanism (confirmed class, exact instruction unproven)

The NVLS dispatcher (`NVLSAllGatherVDispatcher`, `megatron/core/transformer/moe/token_dispatcher_inference.py` + `variable_collectives.py`) moves tokens through class-level symmetric-memory windows with one-sided multicast writes, coordinated by **anonymous, tag-less symm-mem barriers** (`barrier.py:105-115`). Correctness of window reuse rests on 1:1 EP-wide pairing of those barriers and on every rank reading `_step_metadata` from the same paired invocation. There is no event/ordering dependency that prevents rank A's step-N+1 window writes from overlapping rank B's still-running step-N reads when step times diverge wildly across the group — exactly what SWE creates (8 ms dummy replays lock-stepped against 16k-token eager prefill chunks).

Things we **ruled out** along the way:

- **AGV/RSV offset math**: verified in-bounds by construction (per-rank post-TP tokens capped at `max_tokens/tp_size` = per-rank window rows; both old and new configs are exactly-tight).
- **FlashInfer routing-buffer `fill_(-1)` race** (captured fill between metadata barrier and AGV end-barrier, `token_dispatcher_inference.py:438-439`): plausible member of the same family but not our instance — production runs `inference_grouped_gemm_backend=vllm`.
- **Padded block-table `-1` sentinel** (separate bug, fixed): pad rows `[real_bs:padded_bs]` were `-1` + seqlen 0 and handed to the external paged FA decode kernel; any address formed from row 0 of a zero-length row lands one KV page below the buffer base, which under UVM level 1 is unmapped VA. Fixed by filling pad rows with `kv_block_allocator.dummy_block_idx` (matches capture-time state; commit `1aa727b0a`). Post-fix, a chain replayed the [24]-slot min graph at 6/696 active for 14k steps clean — so near-empty replay per se is now safe. Some pre-fix "wave-tail IMA" incidents may in hindsight have been the ordering race; the two aren't separable on old evidence.
- **Graph-vs-eager mixed rendezvous as the sole cause**: we imposed nccl-style EP-wide agreement on nvls (`match_ep_token_counts` now includes an `_nvls_ep_dispatcher` flag; any EP rank in a non-decode step → whole group eager that step; capture excluded — commits `44828d2d8` + `b0b4be493`). This moved crashes later into the wave but did not eliminate them — and it deviates from your per-rank-selection design, so treat it as a band-aid to revisit, not a conclusion.

## 5. Repro recipe

On any ≥2-engine-per-EP-group deployment (we use 8 engines/EP16, 16 GB200 nodes):

1. PR #5 params verbatim (NCG 32, 80 GB buffer, no max-requests/max-tokens overrides, nvls, persist graphs, UVM 1, recompute KV) + `--enable-chunked-prefill`, SL 131072.
2. Drive it with an agentic multi-turn workload that (a) sends prompts ≫ max_tokens so chunked prefill saturates 16384-token chunks, and (b) leaves engines intermittently idle so EP dummy forwards run against peers' prefills (any OpenHands-style SWE env does this; 128 episodes/wave, 8–9 parallel generation batches accelerates it).
3. Expected: 1–3 engines fault with the §2 signature within the first wave (minutes), on lag/parallel-batch configs that drain or stagger engines; ~100% within-wave reproduction on our lag ≤ 2 chains.
4. Confirm the race property: set `MRL_SYNC_AFTER_FORWARD=1` → crash vanishes; unset → returns. (Probe hooks: real-step `text_generation_controller.py:~1761`, dummy-step added in `26335c7db`.)

Evidence artifacts on HSG (test.log per run dir, `runs/nanov3_rl_complete_swe_131072SL_TP2_CP4_16nodes_*_jalbericiola/logs/`): unprobed crashes — jobs 5074989/5074994 (all-type graphs, ~60 s into wave), 5053651/5053663 (decode-only, ~23 min), 5129612/5129613, 5130987/5130990, 5131302/5131306; probed clean — 5132294, 5132621, 5134318, 5141594.

## 6. Suggested fix directions (your call, it's your dispatcher)

1. **Ordering at window reuse**: record an event at each collective's end-barrier and make the *next* write into the same symmetric window wait on it (per-window, per-step event chain). Kills the race without global serialization; keeps per-rank step freedom.
2. **Tagged/sequenced barriers**: stamp the anonymous barriers with a step sequence number so pairing skew becomes detectable (fail loud) instead of silently corrupting.
3. **Decide the mixed-mode contract**: either bless captured-replay ↔ eager coexistence in one collective (and enforce its preconditions with the above), or adopt EP-wide step-type agreement for nvls like nccl (our interim `44828d2d8` does this; costs graph replay whenever any peer prefills).
4. Keep the cheap tripwire we added in `token_dispatch` (`local_tokens ≤ per_rank_window_rows` assert) — converts a whole family of silent corruptions into clean exceptions.

## 7. Where our fixes live

Branch `jalbericiola/offpolicy-on-main` (github.com/jalbericiola/Megatron-LM), on top of the PR #5 merge:

- `1aa727b0a` block-table pad rows → dummy block (real bug, keep)
- `44828d2d8` EP-wide graph agreement for nvls + dispatch tripwire (band-aid + tripwire)
- `b0b4be493` skip EP agreement during capture (needed by the above)
- `52c3e345f`, `e77726967` RL metrics guards for empty/placeholder rollouts (unrelated to the race; needed for agentic waves)
- `26335c7db` dummy-forward probe hook
- Mirror lineage with identical content: `hsg-live-merge-20260713` (tip `7f454edc8`+).

Happy to run any candidate fix on the grid — we reproduce within one wave, so validation is fast.
