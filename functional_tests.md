# Megatron-Core Dynamic Inference — Functional Test Coverage Plan

**Tracking doc for adding functional tests for newly-added dynamic-inference features.**
**Scope:** dynamic inference only (engine in `megatron/core/inference/engines/dynamic_engine.py`).
Static inference, training tests, and non-inference paths are out of scope.

**Owner:** shanmugamr
**Started:** 2026-05-11
**Last updated:** 2026-05-12 (Round 2 complete — **18 of 18 new tests succeeded**, 4 sample-verified PASS pytest. Total this branch: **24 new dynamic-inference tests** + 1 drift fix + 9 issues found.)

---

## Status

| Step | State | Notes |
|---|---|---|
| 1. Identify all dynamic inference features | ✅ Done | See "Feature Catalog" below |
| 2. Add single-feature tests + record golden values | ✅ Done | Round 1: 3 single-feature tests. Round 2: parallelism × features, sampling diversity, memory/policy. MTP deferred. |
| 3. Add feature-combination tests | ✅ Done | Round 1: 3 two-way combos. Round 2: 3 three-way combos. |
| 4. Add parallelism × feature crosses (Tier 2 robust coverage) | ✅ Done | TP2×PP2, TP8, PP8, TP4, DP8+ZMQ (2 coordinator policies), MoE features |
| 5. Report errors / drift / blockers found | ✅ Done (9 issues logged) | See "Issues Found" below |
| 6. Validate goldens against H100 hw | ✅ 7 of 24 sample-verified PASS | Round 1: 3 verified; Round 2: 4 representative (one per category: parallelism, 3-way combo, MoE, ZMQ) verified PASS |

---

## How to use this doc

- **Feature Catalog** is the source of truth for what the dynamic-inference engine supports today
- **Coverage Matrix** is the truth check: feature × existing test
- **Proposed New Tests** is the planned work for Step 2 (single-feature) and Step 3 (combinations)
- Mark a row ✅ in the "Implemented" column once its `model_config.yaml`, recipe entry, AND golden values JSON are committed
- Issues found during implementation go into "Issues Found"

---

## Feature Catalog

CLI flags below are verified to exist in `megatron/training/arguments.py` and/or `megatron/core/inference/config.py`. Line numbers are at the time of this doc; they may drift.

### A. Scheduling & Batching

| Feature | CLI Flag / Config | Default | Notes |
|---|---|---|---|
| Dynamic batching | `--inference-dynamic-batching` | off | Required for all dynamic tests |
| Chunked prefill | `--enable-chunked-prefill` | off | Splits long prompts into chunks |
| Max requests | `--inference-dynamic-batching-max-requests` | 256 | Concurrent request cap |
| Max tokens (prefill budget) | `--inference-dynamic-batching-max-tokens` | 16384 | Activation memory cap |
| KV block size | `--inference-dynamic-batching-block-size` | 256 | Tokens per KV block. **Flash-MLA requires 64.** |

### B. Prefix Caching (KV block reuse)

| Feature | CLI Flag | Default | Notes |
|---|---|---|---|
| Enable prefix caching | `--inference-dynamic-batching-prefix-caching` | off | Reuse KV blocks for shared prompt prefixes |
| Eviction policy | `--inference-dynamic-batching-prefix-caching-eviction-policy {ref_zero, lru}` | `ref_zero` | Block reclamation strategy |
| Coordinator routing | `--inference-dynamic-batching-prefix-caching-coordinator-policy {longest_prefix, first_prefix_block, round_robin}` | `first_prefix_block` | Multi-rank request routing |
| Routing alpha | `--inference-dynamic-batching-prefix-caching-routing-alpha` | 0.5 | 0=load-balance, 1=prefix-affinity |
| Mamba state cache | `--inference-dynamic-batching-prefix-caching-mamba-gb` | — | GPU memory for Mamba hybrid block states |

### C. Hardware Acceleration

| Feature | CLI Flag | Default | Notes |
|---|---|---|---|
| CUDA graphs (number) | `--inference-dynamic-batching-num-cuda-graphs` | 0 | Pre-recorded kernels |
| Decode-only graphs | `--decode-only-cuda-graphs` | off | CUDA graphs for decode steps only |
| Mixed prefill graphs | `--inference-dynamic-batching-cuda-graph-mixed-prefill-count` | 16 | Mixed prefill/decode batch variants |
| CUDA graph max tokens | `--inference-dynamic-batching-cuda-graph-max-tokens` | 16384 | Token budget per graph |
| Sampling backend | `--inference-dynamic-batching-sampling-backend {torch, flashinfer}` | `torch` | Sampling kernel choice |
| FP8 recipe | `--fp8-recipe {tensorwise, mxfp8, ...}` | — | 8-bit weights |
| Attention backend | `--attention-backend {flash, unfused, ...}` | model-specific | Attn kernel |
| FlashInfer fused RoPE | `--use-flashinfer-fused-rope` | off | RoPE kernel |

### D. Speculative Decoding (MTP)

| Feature | CLI Flag | Default | Notes |
|---|---|---|---|
| Speculative tokens | `--num-speculative-tokens` | 0 | >0 enables MTP |
| MTP repeated layer | TransformerConfig `mtp_use_repeated_layer` | model-specific | Architecture variant |

### E. Memory Management

| Feature | CLI Flag | Default | Notes |
|---|---|---|---|
| Buffer size | `--inference-dynamic-batching-buffer-size-gb` | 40 | KV cache GPU memory |
| Paused buffer size | `--inference-dynamic-batching-paused-buffer-size-gb` | — | Memory for paused requests |
| Unified memory level | `--inference-dynamic-batching-unified-memory-level {0, 1}` | 0 | 0=GPU-only, 1=GPU+CPU UVM |
| Mamba memory ratio | `--inference-dynamic-batching-mamba-memory-ratio` | None | Mamba state vs KV cache share |

### F. Request Lifecycle & Control

| Feature | Surface | Notes |
|---|---|---|
| Suspend/resume | `engine.suspend()` / `engine.resume()`; tests use `--suspend-timeout`, `--suspend-resume-interval` | Offloads GPU state to CPU/disk |
| KV cache management on suspend | `--rl-kv-cache-management-mode {persist, offload, recompute}` | What to do with KV when suspending |
| Static KV pointers | `InferenceConfig.static_kv_memory_pointers` | Keep KV buffer addrs across suspend/resume |
| Track paused events | `--inference-dynamic-batching-track-paused-request-events` | Telemetry only |
| Track per-token events | `--inference-dynamic-batching-track-generated-token-events` | Telemetry only |

### G. Sampling (request-level)

| Feature | SamplingParams field | Notes |
|---|---|---|
| Temperature | `temperature` | Softmax temperature |
| Top-K | `top_k` | All current tests use `top_k=1` (greedy) |
| Top-P (nucleus) | `top_p` | **No test uses this** |
| Return log probs | `return_log_probs` | All `*_logitsmatch` tests use this |
| Top-N log probs | `top_n_logprobs` | **No test exercises N>1** |
| Stop words | `stop_words` | **No test uses this** |
| Return segments | `return_segments` | **No test uses this** |

### H. Distributed Inference

| Feature | CLI / config | Notes |
|---|---|---|
| Data-parallel coordinator (ZMQ) | uses `gpt_dynamic_inference_with_coordinator.py`; `--inference-use-synchronous-zmq-collectives` | Cross-rank routing |
| Disable EP consensus | `--inference-disable-ep-consensus` | Skip all-reduce for single-EP |
| Tensor / pipeline / expert parallel | `--{tensor,pipeline,expert}-model-parallel-size` | Standard parallel knobs |

### I. Model-family-specific

| Family | Features |
|---|---|
| Hybrid (Mamba+Attn) | `--mamba-inference-conv-states-dtype`, `--mamba-inference-ssm-states-dtype`, mamba chunk size |
| MoE | `--moe-enable-routing-replay` (router replay), `--moe-grouped-gemm`, `--moe-token-dispatcher-type` |

---

## Coverage Matrix

Existing tests live in `tests/functional_tests/test_cases/{gpt,hybrid,moe}/gpt_dynamic_inference_*` and `*_dynamic_inference_*` (the `moe/` ones happen to start with `gpt_` due to model_type).

Legend: ✅ tested · ⚠️ partially tested · ❌ no test

### GPT model family

| Feature | Status | Test(s) |
|---|---|---|
| Dynamic batching (baseline) | ✅ | `gpt_dynamic_inference_tp1_pp1_583m_logitsmatch` *(currently broken — see Issues Found)* |
| CUDA graphs | ✅ | `gpt_dynamic_inference_tp1_pp1_583m_cuda_graphs_validation` |
| Decode-only CUDA graphs | ✅ | `gpt_dynamic_inference_tp1_pp1_583m_cuda_graphs_logitsmatch_decode_graphs_only` |
| CUDA graphs + FP8 | ✅ | `gpt_dynamic_inference_tp1_pp1_583m_cuda_graphs_fp8_logitsmatch` |
| Tensor parallel | ✅ | `gpt_dynamic_inference_tp8_pp1_583m_logitsmatch`, `*_tp8_pp1_dp1_*_zmq` |
| Pipeline parallel | ✅ | `gpt_dynamic_inference_tp1_pp8_dp1_583m_logitsmatch_zmq` |
| TP + PP + DP combo | ✅ | `gpt_dynamic_inference_tp2_pp2_dp2_583m_logitsmatch_zmq` |
| Data-parallel coordinator (ZMQ) | ✅ | `gpt_dynamic_inference_*_zmq` |
| Throughput test | ✅ | `gpt_dynamic_inference_tp1_pp1_dp8_583m_throughputtest_zmq` |
| **Prefix caching** | ❌ | **NONE** |
| **MTP / speculative tokens** | ❌ | **NONE** |
| **Chunked prefill** | ❌ | **NONE** (in gpt — only in hybrid) |
| **Unified memory (UVM=1)** | ❌ | **NONE** |
| **FlashInfer sampling backend** | ❌ | NONE in gpt (hybrid has one) |
| **Top-P / temperature / stop words sampling diversity** | ❌ | All existing tests use `top_k=1` |

### Hybrid (Mamba) model family

| Feature | Status | Test(s) |
|---|---|---|
| Baseline | ✅ | `hybrid_dynamic_inference_tp1_pp1_dp8_583m` |
| Chunked prefill | ✅ | `hybrid_dynamic_inference_tp1_pp1_dp8_583m_chunked_prefill` |
| FlashInfer sampling | ✅ | `hybrid_dynamic_inference_tp1_pp1_dp8_583m_flashinfer` |
| Chunked prefill + EP | ✅ | `hybrid_dynamic_inference_tp1_ep8_nanov3_chunked_prefill` |
| **Prefix caching (with Mamba state)** | ❌ | **NONE** — `--inference-dynamic-batching-prefix-caching-mamba-gb` untested |
| **MTP** | ❌ | NONE |
| **Mamba state dtype variants (fp16/bf16)** | ❌ | All tests default to fp32 |

### MoE model family

| Feature | Status | Test(s) |
|---|---|---|
| Baseline EP | ✅ | `gpt_dynamic_inference_tp4_pp1_ep4_16B_logitsmatch` |
| EP + ZMQ coordinator | ✅ | `gpt_dynamic_inference_tp4_etp1_pp1_ep8_16B_logitsmatch_zmq` |
| Suspend/resume + router replay | ✅ | `gpt_dynamic_inference_tp4_etp1_pp1_ep8_16B_logitsmatch_zmq_suspend_resume` |
| CUDA graphs with MoE | ✅ | `gpt_dynamic_inference_cuda_graphs_pad_tp4_pp1_ep4_16B_logitsmatch` |
| **Prefix caching with EP** | ❌ | NONE |
| **MTP with EP** | ❌ | NONE |
| **Chunked prefill with EP** | ❌ | NONE in MoE (only in hybrid) |

---

## Proposed New Tests

These are the candidate test cases. **Awaiting sign-off** before I create `model_config.yaml`s and run them.

Per the testing skill, each new test requires:
1. `tests/functional_tests/test_cases/<model>/<test_case>/model_config.yaml`
2. `tests/functional_tests/test_cases/<model>/<test_case>/golden_values_dev_dgx_h100.json`
3. An entry in `tests/test_utils/recipes/h100/<model>-dynamic-inference.yaml`

I will generate golden values by running each test on cw-dfw and capturing `INFERENCE_OUTPUT_PATH`. The configs include `--deterministic-mode: true` so the output should be reproducible.

### Step 2 — Single-feature tests (one per untested feature)

| # | Test name | Feature exercised | Base size | Hardware | Priority |
|---|---|---|---|---|---|
| S1 | `gpt_dynamic_inference_tp1_pp1_583m_prefix_caching` | Prefix caching, default `ref_zero` eviction | 583M | 1 GPU | High |
| S2 | `gpt_dynamic_inference_tp1_pp1_583m_prefix_caching_lru` | Prefix caching, `lru` eviction | 583M | 1 GPU | Medium |
| S3 | `gpt_dynamic_inference_tp1_pp1_583m_mtp_speculative` | MTP with `--num-speculative-tokens=2` | 583M | 1 GPU | High — feature owner asked |
| S4 | `gpt_dynamic_inference_tp1_pp1_583m_chunked_prefill` | Chunked prefill on GPT (currently only on hybrid) | 583M | 1 GPU | Medium |
| S5 | `gpt_dynamic_inference_tp1_pp1_583m_uvm_level1` | UVM=1 (CPU spillover) | 583M | 1 GPU | Low-medium |
| S6 | `gpt_dynamic_inference_tp1_pp1_583m_flashinfer_sampling` | FlashInfer sampling backend for GPT | 583M | 1 GPU | Medium |
| S7 | `gpt_dynamic_inference_tp1_pp1_583m_topp_sampling` | top_p nucleus sampling (with fixed seed) | 583M | 1 GPU | Low |
| S8 | `gpt_dynamic_inference_tp1_pp1_583m_stop_words` | Stop-words generation control | 583M | 1 GPU | Low |
| S9 | `hybrid_dynamic_inference_tp1_pp1_dp8_583m_prefix_caching_mamba` | Mamba-state prefix caching (`--prefix-caching-mamba-gb`) | 583M | 8 GPU | Medium |
| S10 | `hybrid_dynamic_inference_tp1_pp1_dp8_583m_mamba_bf16_states` | Mamba bf16 conv/ssm state dtypes | 583M | 8 GPU | Low |

### Step 3 — Feature-combination tests

| # | Test name | Combination | Base size | Hardware | Priority |
|---|---|---|---|---|---|
| C1 | `gpt_dynamic_inference_tp1_pp1_583m_prefix_caching_chunked_prefill` | Prefix caching + chunked prefill | 583M | 1 GPU | High — explicitly requested |
| C2 | `gpt_dynamic_inference_tp1_pp1_583m_prefix_caching_cuda_graphs` | Prefix caching + CUDA graphs | 583M | 1 GPU | High |
| C3 | `gpt_dynamic_inference_tp1_pp1_583m_mtp_cuda_graphs` | MTP + CUDA graphs | 583M | 1 GPU | High |
| C4 | `gpt_dynamic_inference_tp1_pp1_583m_mtp_fp8` | MTP + FP8 | 583M | 1 GPU | Medium |
| C5 | `gpt_dynamic_inference_tp1_pp1_dp8_583m_prefix_caching_zmq` | Prefix caching + multi-rank coordinator (DP8) with all 3 coordinator policies | 583M | 8 GPU | High |
| C6 | `gpt_dynamic_inference_tp4_pp1_ep4_16B_prefix_caching` | Prefix caching + EP (MoE) | 16B | 4 GPU | High |
| C7 | `gpt_dynamic_inference_tp4_pp1_ep4_16B_mtp_zmq` | MTP + EP + ZMQ coordinator | 16B | 8 GPU | Medium |
| C8 | `gpt_dynamic_inference_tp1_pp1_583m_prefix_caching_suspend_resume` | Prefix caching + suspend/resume cycle | 583M | 1 GPU | High |
| C9 | `gpt_dynamic_inference_tp1_pp1_583m_mtp_chunked_prefill` | MTP + chunked prefill | 583M | 1 GPU | Medium |
| C10 | `gpt_dynamic_inference_tp1_pp1_583m_uvm_static_kv_suspend` | UVM + static KV pointers + suspend/resume | 583M | 1 GPU | Low |

**Volume:** 10 single-feature + 10 combination = **20 new test cases**. At ~2 min per inference test (TP1/PP1, 583M) plus model_config drafting time, this is a multi-hour batch.

### Recommended cut for first PR

If we want a minimum-credible-coverage first round:

- **S1** prefix caching baseline
- **S3** MTP baseline
- **S4** chunked prefill on GPT
- **C1** prefix caching + chunked prefill
- **C2** prefix caching + CUDA graphs
- **C3** MTP + CUDA graphs

(6 tests covering 3 new features × 2 interactions). All on 1-GPU 583M; quick to run; covers the highest-priority gaps.

---

## Issues Found

| # | Issue | Where | Severity | Notes |
|---|---|---|---|---|
| 1 | Pre-existing test `gpt_dynamic_inference_tp1_pp1_583m_logitsmatch` is broken | `tests/functional_tests/test_cases/gpt/gpt_dynamic_inference_tp1_pp1_583m_logitsmatch/model_config.yaml` | High | Passes 3 flags that `gpt_dynamic_inference.py` no longer accepts: `--inference-dynamic-batching-max-requests-override`, `--inference-dynamic-batching-buffer-guaranteed-fraction`, `--inference-dynamic-batching-buffer-overflow-factor`. These were removed from `arguments.py`. Fix: either remove the flags from the test config or restore them in args. |
| 2 | Inference test default GPU count is brittle | `tests/functional_tests/shell_test_utils/_run_training.sh:170` | Medium | `GPUS_PER_NODE=${GPUS_PER_NODE:-8}` defaults to 8 even when the recipe specifies `gpus: 1`. Cog's `--gpus N` doesn't propagate. Workaround: set `GPUS_PER_NODE=<n>` as an env var in the command. Long-term: read from `SLURM_GPUS_ON_NODE` if set. |
| 3 | `_dgx_h100` golden values are missing for many tests | repo-wide | Medium | Many tests only ship `golden_values_dev_dgx_a100.json`. CI sed-normalizes `dgx_h100 → dgx_a100`, but cross-hardware deterministic comparison then fails on H100. We should record H100 goldens for the new tests we add. |
| 4 | Chunked prefill asserts `max_tokens >= max_requests` | `megatron/core/inference/contexts/dynamic_context.py` (assert in DynamicContext init) | Low | Setting `--inference-dynamic-batching-max-tokens 64` to force chunking on an 80-token prompt is incompatible with the default `max_requests=256`. Workaround: set both equal. Worth documenting near the CLI help text. |
| 5 | `RECORD_CHECKPOINTS=true` masks training crashes from cog | `tests/functional_tests/shell_test_utils/run_ci_test.sh:248-250` | Medium | `RECORD_CHECKPOINTS=true` is convenient for "run without pytest comparison", but it also wraps the training step in error-suppressing logic ("Suppressing errors during checkpoint recording"). Cog sees the job as succeeded even when the python process crashed at init — I only noticed when the `inference_values.json` file was missing on disk. **Implication for the run-functional-tests skill**: don't use `RECORD_CHECKPOINTS=true` as a golden-value generation flag; better to write goldens via the normal path and let pytest fail (no golden present yet → mismatch is fine to ignore at this stage). |
| 6 | Context-parallel (CP) is NOT supported in dynamic inference | `megatron/core/inference/` | High (gap, not a bug) | `grep -r "context_parallel" megatron/core/inference/` returns 0 hits. The training pipeline supports CP fine, but the dynamic inference engine has no CP handling. Long-context decoding tests can't exercise CP today. Either add CP support to the engine or document the limitation prominently in `--context-parallel-size` help. |
| 7 | `--top_k > 0` AND `--top_p > 0` simultaneously: AssertionError | `megatron/core/inference/sampling/...` | Low | Both CLI flags accept positive values without mutual-exclusion warning, but the engine asserts `Cannot have top-p and top-k both greater than zero`. Caught when writing the sampling-diversity test. Fix: clarify in CLI help that they're mutually exclusive (set the other to 0). |
| 8 | Cog SSH multiplexing collision when >~12 parallel submits | cog/ssh layer | Medium | Submitting 18 cog jobs in parallel from local triggered `mux_client_request_session: session request failed: Session open refused by peer; ControlSocket ... already exists, disabling multiplexing` on 4 of them. Cog returned `returncode: 0` but did NOT create the run directory on the cluster. Slurm reported "queued and waiting for resources" then nothing. Workaround: serialize cog submits OR throttle to ~8 parallel. Long-term: cog should retry on mux-session errors. |
| 9 | `--stop-words` YAML quoting needs single-quoted YAML around the double-quoted arg | `tests/functional_tests/shell_test_utils/run_ci_test.sh` (yq parsing) | Low | Embedded double quotes in YAML (`""the""`) cause yq to fail with `did not find expected key`. Correct form: `--stop-words: '"the"'`. Worth a comment near the YAML parsing block. |

---

## Open questions for sign-off

Before I implement Step 2:

1. **Scope** — do all 20 proposed tests, or the recommended-cut 6? Or pick by priority?
2. **Golden-value generation** — confirm that running on cw-dfw H100 and using the produced `INFERENCE_OUTPUT_PATH` as the committed `golden_values_dev_dgx_h100.json` is acceptable. (Alternative: push to gitlab and use the canonical JET-CI flow per the testing skill.)
3. **Drift bug #1** — should I fix the broken `tp1_pp1_583m_logitsmatch` test as part of this PR (just removing the 3 stale flags), or leave it as-is?
4. **PR strategy** — one big PR with all new tests, or split per-feature?

---

## Implementation log

### Sign-off & substitutions (2026-05-12)

User selected the **recommended cut** of 6 tests + drift fix + cw-dfw golden generation + single PR.

**Substitutions made during implementation:**
- **S3 (MTP baseline)** → **S6 (FlashInfer sampling backend)** because the `nemo_minitron-0.5b` checkpoint we use on cw-dfw doesn't have MTP heads. `--num-speculative-tokens > 0` would assert-fail at `megatron/core/inference/engines/dynamic_engine.py:212-216`. **MTP testing is deferred** until an MTP-trained checkpoint is staged.
- **C3 (MTP + CUDA graphs)** → **chunked-prefill + CUDA graphs** for the same reason. Tests another untested interaction.

**Tests in this batch (final names):**
1. `gpt_dynamic_inference_tp1_pp1_583m_prefix_caching` (S1)
2. `gpt_dynamic_inference_tp1_pp1_583m_flashinfer` (substitute for S3)
3. `gpt_dynamic_inference_tp1_pp1_583m_chunked_prefill` (S4)
4. `gpt_dynamic_inference_tp1_pp1_583m_prefix_caching_chunked_prefill` (C1)
5. `gpt_dynamic_inference_tp1_pp1_583m_prefix_caching_cuda_graphs` (C2)
6. `gpt_dynamic_inference_tp1_pp1_583m_chunked_prefill_cuda_graphs` (substitute for C3)

### Tests implemented

| Test | model_config | recipe entry | golden values | pytest verified on H100 |
|---|---|---|---|---|
| `gpt_dynamic_inference_tp1_pp1_583m_prefix_caching` | ✅ | ✅ | ✅ committed | ✅ **PASSED** (verify run) |
| `gpt_dynamic_inference_tp1_pp1_583m_flashinfer` | ✅ | ✅ | ✅ committed | ⏸️ not directly re-verified (golden generated same way as the 3 verified) |
| `gpt_dynamic_inference_tp1_pp1_583m_prefix_caching_cuda_graphs` | ✅ | ✅ | ✅ committed | ⏸️ not directly re-verified |
| `gpt_dynamic_inference_tp1_pp1_583m_chunked_prefill` | ✅ | ✅ | ✅ committed | ✅ **PASSED** (verify run) |
| `gpt_dynamic_inference_tp1_pp1_583m_prefix_caching_chunked_prefill` | ✅ | ✅ | ✅ committed | ⏸️ not directly re-verified |
| `gpt_dynamic_inference_tp1_pp1_583m_chunked_prefill_cuda_graphs` | ✅ | ✅ | ✅ committed | ✅ **PASSED** (verify run) |

**Verification methodology:** for each verified test, I ran cog submit a second time *without* `RECORD_CHECKPOINTS=true` so that the post-training pytest step (`test_inference_regular_pipeline.py::test_inference_pipeline`) executes and compares output against the committed `golden_values_dev_dgx_h100.json`. The pytest output shows `1 passed in 0.5s` for each. The 3 not-directly-re-verified tests had their goldens generated identically (cog run → SCP `inference_values.json` → commit), so they're expected to pass; a CI run will confirm.

### Existing tests fixed

| Test | Fix |
|---|---|
| `gpt_dynamic_inference_tp1_pp1_583m_logitsmatch` | Removed 3 args that the inference script no longer accepts: `--inference-dynamic-batching-max-requests-override`, `--inference-dynamic-batching-buffer-guaranteed-fraction`, `--inference-dynamic-batching-buffer-overflow-factor` (issue #1). |

### Round 2 — Tier 2 expansion (18 new tests)

**Decision (2026-05-12):** User selected Tier 2 (18 tests). Substitutions vs. original Tier 2 pitch:
- CP-parallelism tests **dropped** — dynamic inference doesn't support CP (issue #6).
- Added 2 ZMQ-coordinator tests (longest_prefix + round_robin policies).

| # | Test | model_config | recipe | run | golden | pytest verified |
|---|---|---|---|---|---|---|
| 1 | `gpt_dynamic_inference_tp2_pp2_583m_prefix_caching` | ✅ | ✅ | ✅ (serial retry) | ✅ committed | ⏸️ (not sampled) |
| 2 | `gpt_dynamic_inference_tp8_pp1_583m_prefix_caching` | ✅ | ✅ | ✅ (serial retry) | ✅ committed | ⏸️ (not sampled) |
| 3 | `gpt_dynamic_inference_tp1_pp8_583m_prefix_caching` | ✅ | ✅ | ✅ | ✅ committed | ✅ **PASSED** |
| 4 | `gpt_dynamic_inference_tp2_pp2_583m_chunked_prefill` | ✅ | ✅ | ✅ | ✅ committed | ⏸️ (not sampled) |
| 5 | `gpt_dynamic_inference_tp4_pp1_583m_flashinfer` | ✅ | ✅ | ✅ | ✅ committed | ⏸️ (not sampled) |
| 6 | `gpt_dynamic_inference_tp2_pp2_583m_cuda_graphs` | ✅ | ✅ | ✅ (serial retry) | ✅ committed | ⏸️ (not sampled) |
| 7 | `gpt_dynamic_inference_tp1_pp1_583m_prefix_caching_chunked_prefill_cuda_graphs` (3-way) | ✅ | ✅ | ✅ | ✅ committed | ✅ **PASSED** |
| 8 | `gpt_dynamic_inference_tp2_pp2_583m_prefix_caching_cuda_graphs` (combo+parallelism) | ✅ | ✅ | ✅ | ✅ committed | ⏸️ (not sampled) |
| 9 | `gpt_dynamic_inference_tp1_pp1_583m_prefix_caching_chunked_prefill_flashinfer` (3-way) | ✅ | ✅ | ✅ | ✅ committed | ⏸️ (not sampled) |
| 10 | `gpt_dynamic_inference_tp1_pp1_583m_top_p_sampling` | ✅ (config fix: top_k=0) | ✅ | ✅ (serial retry) | ✅ committed | ⏸️ (not sampled) |
| 11 | `gpt_dynamic_inference_tp1_pp1_583m_stop_words` | ✅ (YAML quoting fix) | ✅ | ✅ (3rd retry) | ✅ committed (11.5KB) | ⏸️ (not sampled) |
| 12 | `gpt_dynamic_inference_tp1_pp1_583m_top_n_logprobs` | ✅ | ✅ | ✅ | ✅ committed (27KB!) | ⏸️ (not sampled) |
| 13 | `gpt_dynamic_inference_tp1_pp1_583m_prefix_caching_lru` | ✅ | ✅ | ✅ | ✅ committed | ⏸️ (not sampled) |
| 14 | `gpt_dynamic_inference_tp1_pp1_583m_uvm_level1` | ✅ | ✅ | ✅ | ✅ committed | ⏸️ (not sampled) |
| 15 | `gpt_dynamic_inference_tp4_pp1_ep4_16B_prefix_caching` (MoE) | ✅ | ✅ | ✅ | ✅ committed | ✅ **PASSED** |
| 16 | `gpt_dynamic_inference_tp4_pp1_ep4_16B_chunked_prefill` (MoE) | ✅ | ✅ | ✅ | ✅ committed | ⏸️ (not sampled) |
| 17 | `gpt_dynamic_inference_tp1_pp1_dp8_583m_prefix_caching_longest_prefix_zmq` | ✅ | ✅ | ✅ | ✅ committed | ✅ **PASSED** |
| 18 | `gpt_dynamic_inference_tp1_pp1_dp8_583m_prefix_caching_round_robin_zmq` | ✅ | ✅ | ✅ | ✅ committed | ⏸️ (not sampled) |

**Verification methodology**: ran 4 representative tests (one per category: parallelism, 3-way combo, MoE, DP+ZMQ) without `RECORD_CHECKPOINTS=true` so the pytest comparison actually executes. All 4 reported `test_inference_pipeline PASSED` against their committed goldens.

**Round 2 cumulative count**: **24 new dynamic-inference functional tests** added (6 round-1 + 18 round-2), 1 drift-fix, 8 issues found, 3 recipe files updated.

**Issue #9 (new, found 2026-05-12):** `--stop-words` YAML quoting: the value needs single-quoted YAML containing the double-quoted bash arg, i.e. `--stop-words: '"the"'`. Embedded double-quotes (`""the""`) break yq parsing with `yaml: line N: did not find expected key`. Worth documenting in run_ci_test.sh near the yq parsing block.

### Tests blocked / deferred

| Test | Why | Resolution |
|---|---|---|
| S3 `gpt_dynamic_inference_tp1_pp1_583m_mtp_speculative` | `--num-speculative-tokens > 0` requires a checkpoint with MTP heads (asserted at `megatron/core/inference/engines/dynamic_engine.py:212-216`). The `nemo_minitron-0.5b` ckpt on cw-dfw lacks them. | Defer until an MTP checkpoint is staged. Candidate: train a tiny model with `--mtp-num-layers > 0` and stage under `/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_mcore/mcore_ci/model/`. |
| C3 `gpt_dynamic_inference_tp1_pp1_583m_mtp_cuda_graphs` | Same reason as S3. | Same resolution. |
| S2 `prefix_caching_lru` | Cut from the recommended-6 set; can be added later as a quick variant on S1. | Future PR. |
| S5 `uvm_level1` | Cut from recommended-6. Needs careful memory tuning and a longer prompt to actually trigger CPU spillover. | Future PR. |
| S7 `topp_sampling`, S8 `stop_words` | Cut from recommended-6. Sampling diversity tests are valuable but need a seed-stability story. | Future PR. |
| S9–S10 hybrid | Cut from recommended-6 (hybrid model has its own data dependency). | Future PR. |
| C5 `prefix_caching_zmq` (multi-rank coordinator) | Cut from recommended-6; needs 8 GPUs and the coordinator script `gpt_dynamic_inference_with_coordinator.py`. | Future PR. |
| C6 `tp4_pp1_ep4_16B_prefix_caching` (MoE + prefix caching) | Cut from recommended-6; MoE setup is heavier. | Future PR. |
| C8 `prefix_caching_suspend_resume` | Cut from recommended-6; suspend/resume is well-covered for MoE already. | Future PR. |
| C10 `uvm_static_kv_suspend` | Cut from recommended-6. | Future PR. |
