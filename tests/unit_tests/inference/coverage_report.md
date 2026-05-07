# Coverage Report: megatron/core/inference

## Target

90 % per file (adjusted downward for files dominated by CUDA/Triton/native code).

## Baseline

The full-folder baseline run (`pytest tests/unit_tests/inference --cov=megatron/core/inference`) **crashed at 16 %** with SIGABRT inside `tests/unit_tests/inference/engines/test_dynamic_engine.py`. No coverage table was produced for the broader folder.

Strategy switch: rather than fighting that suite, this pass writes targeted tests for top-level source files that had **no existing test file**, then validates each batch in isolation. Files inside subdirectories (engines, contexts, sampling, etc.) and existing-test files are out of scope for this iteration. They contain heavy GPU/distributed dependencies that are not unit-testable without the full inference pipeline.

### Top-level source files (in scope)

| File | Stmts | Existing test? | Action |
|------|-------|----------------|--------|
| `async_stream.py` | 37 | NO | wrote `test_async_stream.py` |
| `headers.py` | 18 | NO | wrote `test_headers.py` |
| `inference_client.py` | 100 | NO | wrote `test_inference_client.py` |
| `inference_request.py` | 304 | NO | wrote `test_inference_request.py` |
| `sampling_params.py` | 39 | thin (1 test) | wrote comprehensive `test_sampling_params.py` |
| `symmetric_memory.py` | 85 | NO | wrote `test_symmetric_memory.py` |
| `unified_memory.py` | 216 | NO | **deferred** — requires runtime CUDA C extension compilation |

## File Queue

- [x] `headers.py` — 0 % → **100 %**
- [x] `async_stream.py` — 0 % → **100 %**
- [x] `sampling_params.py` — 0 % → **100 %**
- [x] `inference_request.py` — 0 % → **98 %**
- [x] `inference_client.py` — 0 % → **94 %**
- [x] `symmetric_memory.py` — 31 % → **94 %**
- [ ] `unified_memory.py` — 18 % → deferred (CUDA-bound)

## Per-File Progress

### headers.py

- Baseline: 0 %
- After tests: **100 %** (18/18 stmts)
- Missing lines: none
- Test file: `tests/unit_tests/inference/test_headers.py`

### async_stream.py

- Baseline: 0 %
- After tests: **100 %** (37/37 stmts)
- Missing lines: none
- Test file: `tests/unit_tests/inference/test_async_stream.py`
- Notes: One initial test (`test_generator_calls_cancel_on_generator_exit`) was failing because `aclose()` re-raised `asyncio.CancelledError` after invoking the cancel callback. Fixed by wrapping `aclose()` in a try/except to swallow the expected `CancelledError`.

### sampling_params.py

- Baseline: 0 % (existing test had a single trivial assertion)
- After tests: **100 %** (39/39 stmts)
- Missing lines: none
- Test file: `tests/unit_tests/inference/test_sampling_params.py`
- Coverage of: defaults, custom-init, the deprecated `return_prompt_top_n_logprobs` warning + assertion, `add_attributes`, `serialize`/`deserialize` round-trip including extra keys.

### inference_request.py

- Baseline: 0 %
- After tests: **98 %** (304 stmts, 6 missed)
- Missing lines: 318-324, 344-350 — both are the `ContextErrorFactory` serialization/deserialization paths inside `DynamicInferenceEvent`. Calling these requires importing `megatron.core.inference.contexts.dynamic_context.ContextErrorFactory` which transitively pulls in `transformer_engine` and CUDA kernel imports. Skipped — not unit-testable without bringing up the full context machinery.
- Test file: `tests/unit_tests/inference/test_inference_request.py`
- Notes: Initial post-deserialize tests assumed `serialize()` produced lists; in fact it produces tuples (`("tensor", [...])`) which msgpack converts to lists during transport. Tests were updated to msgpack-roundtrip data before calling `deserialize()`.

### inference_client.py

- Baseline: 0 %
- After tests: **94 %** (100 stmts, 6 missed)
- Missing lines: 18-19 (zmq `ImportError` fallback), 25-26 (msgpack `ImportError` fallback), 151-152 (`KeyboardInterrupt` break in `_recv_task`). All three are platform/install-flavor branches that can't be hit without uninstalling `zmq`/`msgpack` or sending a real keyboard signal mid-loop.
- Test file: `tests/unit_tests/inference/test_inference_client.py`
- Notes: The receive-loop tests use `MagicMock` for the zmq socket. Setting `side_effect = [reply, zmq.Again()]` will raise `StopIteration` on a third call — the listener loops forever, so use a callable side_effect that returns the reply once then keeps raising `zmq.Again()`. Also: `task.cancel()` followed by `await task` raises `asyncio.CancelledError`, which is a `BaseException` (not `Exception`) — must be caught as `(Exception, asyncio.CancelledError)`.

### symmetric_memory.py

- Baseline: 31 %
- After tests: **94 %** (85 stmts, 5 missed)
- Missing lines: 23-24, 30-31 (`ImportError` fallback when `torch.distributed._symmetric_memory` or `triton` are unavailable; in this container both are present), 56 (`enable_symm_mem_for_group` exception branch).
- Test file: `tests/unit_tests/inference/test_symmetric_memory.py`
- Notes: `SymmetricMemoryBuffer.__init__` rendezvouses against a real process group. To exercise the byte-packing logic without GPUs we used `SymmetricMemoryBuffer.__new__` to bypass `__init__`, then assigned a real CPU `torch.uint8` tensor as `symm_buffer` and a `MagicMock` as `symm_mem_hdl`. The class-level `_buffers` dict on `SymmetricMemoryManager` requires `setup_method`/`teardown_method` cleanup so tests don't leak state.

## Validation Runs

| Run | Request ID | Result |
|-----|-----------|--------|
| Baseline (full inference) | `e3dcf165...` | SIGABRT at 16 % inside `test_dynamic_engine.py`; coverage not produced |
| Batch 1 (4 files) | `9750aa83...` | 83 passed, 1 failed (`test_generator_calls_cancel_on_generator_exit`) |
| Batch 2 (6 files) | `b60abe06...` | 116 passed, 3 failed (post-deserialize wrappers, recv-task StopIteration) |
| Batch 3 after first fixes | `fbe31f58...` | 119 passed, 1 failed (recv-task CancelledError unhandled) |
| Final | `6e76edbe...` | **120 passed, 0 failed** |

## Final Summary

| File | Baseline | Final | Delta | Test File |
|------|----------|-------|-------|-----------|
| async_stream.py | 0% | **100%** | +100% | test_async_stream.py |
| headers.py | 0% | **100%** | +100% | test_headers.py |
| sampling_params.py | 0% | **100%** | +100% | test_sampling_params.py |
| inference_request.py | 0% | **98%** | +98% | test_inference_request.py |
| inference_client.py | 0% | **94%** | +94% | test_inference_client.py |
| symmetric_memory.py | 31% | **94%** | +63% | test_symmetric_memory.py |

**Six previously-untested top-level files now have 94–100 % coverage.** Six new test files (~970 lines of test code, 120 test cases) added to `tests/unit_tests/inference/`.

## Out of Scope (deferred)

- `unified_memory.py` (216 stmts) — requires runtime CUDA C++ extension compilation via `torch.utils.cpp_extension.load_inline`. The non-CUDA paths are mostly trivial type/None guards — modest coverage improvement only.
- `data_parallel_inference_coordinator.py`, `communication_utils.py`, `scheduler.py`, etc. — have existing test files but each requires tests with proper torch.distributed setup; out of scope for the file-by-file untested-source-files pass.

---

# Subdirectory Pass

## Scope

Walked every subdirectory of `megatron/core/inference/`:

| Subdir | Total stmts | Status |
|--------|-------------|--------|
| `engines/` | 1303 | small files tested; `dynamic_engine.py` (1069) and `static_engine.py` (132) require full inference pipeline |
| `contexts/` | ~2500 | abstract bases tested; `dynamic_context.py` (1388), `mamba_*` (614), `kv_block_allocator.py` (198) require full context machinery |
| `communication/torch_symm_triton/` | 522 | dominated by Triton kernels — Python coverage cannot trace `@triton.jit` bodies |
| `model_inference_wrappers/` | 187 | each wrapper requires a built `MegatronModule` — full GPU model setup |
| `moe/` | 582 | dominated by Triton kernels |
| `quantization/` | 226 | tractable utilities tested; `mxfp8_quantize.py` (60) is Triton-bound |
| `sampling/` | 122 | abstract base tested; concrete samplers require flashinfer / CUDA |
| `text_generation_controllers/` | 1132 | inherit from 869-line `text_generation_controller.py` requiring full GPU pipeline |
| `text_generation_server/` | 996 | HTTP endpoints (quart) requiring an ASGI test client + live `InferenceClient` |

The tractable-without-GPU subset is small. This pass writes tests for all such files.

## Files Targeted

| File | Baseline | After tests | Stmts/Miss | Test File |
|------|----------|-------------|------------|-----------|
| `engines/abstract_engine.py` | 0% | **86%** | 7/1 | `engines/test_abstract_and_mcore_engine.py` |
| `engines/mcore_engine.py` | 0% | **100%** | 1/0 | `engines/test_abstract_and_mcore_engine.py` |
| `contexts/base_context.py` | 53% | **95%** | 19/1 | `contexts/test_base_context.py` |
| `contexts/attention_context/metadata_base.py` | 29% | **100%** | 21/0 | `contexts/attention_metadata/test_metadata_base.py` |
| `quantization/mxfp8_tensor.py` | 49% | **76%** | 37/9 | `quantization/test_mxfp8_tensor.py` |
| `sampling/base.py` | 0% | **100%** | 15/0 | `sampling/test_base.py` |
| `sampling/__init__.py` | 0% | **100%** | 4/0 | (covered transitively) |

**36 new tests, 0 failures, 0 retries** — all green on first run.

## Subdirectory Per-File Notes

### engines/abstract_engine.py

- Missing line 17: `pass` body of the abstract `generate()` method — Python coverage marks abstractmethod bodies as missing because they're never executed.
- Test file: `tests/unit_tests/inference/engines/test_abstract_and_mcore_engine.py`

### engines/mcore_engine.py

- Tiny re-export module. Test verifies `MCoreEngine is StaticInferenceEngine`.

### contexts/base_context.py

- Missing line 24: same abstractmethod-body limitation as above (`is_static_batching` body is `pass`).
- Tests use two minimal concrete subclasses (`_StaticContext`, `_DynamicContext`) to exercise each branch of `is_dynamic_batching`, `increment_sequence_len_offset`, `increment_batch_size_offset`, `reset_batch_size_offset`.

### contexts/attention_context/metadata_base.py

- 100 % coverage. The `tensor_copy_and_pad` method has cumulative / non-cumulative branches and assertion guards — all covered with CPU-only tensors.

### quantization/mxfp8_tensor.py

- Missing lines 11 (`flashinfer ImportError` fallback — flashinfer is installed in this container) and 61-69 (`from_bf16` actual quantization — requires CUDA + bf16 + the flashinfer/triton kernels). Tested everything pure-Python: `_ceil_div`, `size`, `scale_2d` (1-D and 2-D paths, with explicit and inferred `K`), and the unknown-backend error path.

### sampling/base.py

- 100 % coverage. `sample_speculative` builds a per-token request mapping; tests verify the mapping for 3 cases (mixed decode+prefill, prefill-only, with `gather_indices` forwarded).

## Subdirectories Not Touched (and why)

- **`engines/dynamic_engine.py` (1069 stmts), `static_engine.py` (132 stmts)** — Both require an instantiated `MegatronModule` plus full `DynamicInferenceContext` / process-group setup. Existing functional tests in the repo are precisely what the SIGABRT-prone baseline was running.
- **`contexts/dynamic_context.py` (1388 stmts), `mamba_*` (614 stmts), `kv_block_allocator.py` (198 stmts), `mamba_slot_allocator.py` (288 stmts)** — All require a running CUDA inference context with KV-cache buffers allocated. Not tractable as unit tests; covered by existing functional tests.
- **`communication/torch_symm_triton/` (522 stmts across 6 files)** — Every public function either requires real symmetric-memory rendezvous or invokes Triton kernels (whose bodies are invisible to coverage.py).
- **`model_inference_wrappers/`** — Each wrapper takes a built `MegatronModule`. Construction requires the full transformer config + GPU.
- **`moe/permute.py` (229), `moe/activations.py` (89), `moe/vllm_fused_moe.py` (170)** — Triton-heavy; even a perfect harness can't get coverage past the ~55-65% Triton ceiling documented in `run-tests.md` Known Quirks.
- **`text_generation_controllers/text_generation_controller.py` (869 stmts)** — Full inference orchestration; needs the entire stack live.
- **`text_generation_server/dynamic_text_gen_server/endpoints/*.py` (~600 stmts)** — Quart HTTP endpoints. Testing requires an ASGI test client plus a live `InferenceClient`. Practical via `quart.testing` but each endpoint then needs the full inference coordinator running, which is beyond unit-test scope.
- **`text_generation_server/text_generation_server.py` (152), `dynamic_text_gen_server/text_generation_server.py` (114), `tokenization.py` files** — All entry-point glue that wires Quart + tokenizer + InferenceClient together. Not tractable without bringing up the full server.

---

# GPU-Required Pass (Round 3)

After the user authorized GPU-using tests on the cw-dfw cluster, we picked up files that
need `torch.cuda.current_device()`, real CUDA tensors, or HTTP test clients.

## Files Targeted (Round 3)

| File | Before | After | Stmts/Miss | Test File |
|------|--------|-------|------------|-----------|
| `contexts/routing_metadata.py` | 26% | **100%** | 35/0 | `contexts/test_routing_metadata.py` |
| `contexts/attention_context/mha_metadata.py` | 32% | **100%** | 28/0 | `contexts/attention_metadata/test_mha_metadata.py` |
| `contexts/static_context.py` | 24% | **69%** | 51/16 | `contexts/test_static_context.py` |
| `text_generation_controllers/encoder_decoder_*.py` | 0% | **100%** | 12/0 | `text_generation_controllers/test_encoder_decoder_and_vlm_controllers.py` |
| `text_generation_controllers/vlm_*.py` | 0% | **100%** | 14/0 | (same file) |
| `text_generation_server/dynamic_text_gen_server/endpoints/common.py` | 0% | **100%** | 7/0 | `text_generation_server/dynamic_text_gen_server/endpoints/test_common.py` |
| `text_generation_server/dynamic_text_gen_server/endpoints/health.py` | 0% | **91%** | 23/2 | `text_generation_server/dynamic_text_gen_server/endpoints/test_health.py` |
| `text_generation_server/tokenization.py` | 0% | **58%** | 38/16 | `text_generation_server/test_tokenization.py` |
| `text_generation_server/dynamic_text_gen_server/tokenization.py` | 0% | **50%** | 38/19 | `text_generation_server/dynamic_text_gen_server/test_dynamic_tokenization.py` |
| `quantization/utils.py` | 18% | **36%** | 129/82 | `quantization/test_utils.py` |

**60 new tests, 0 failures.** Validation run: `c79dc681e9e1412cbde67fade8a46e70`.

## Round 3 Per-File Notes

### contexts/static_context.py (69%)

- Missing lines 73, 85-119 are `__str__` and `__eq__` — both reference
  `self.materialize_only_last_token_logits` which is on `self.config`, not on the context.
  The methods would `AttributeError` if called on an in-spec context. Likely-buggy code;
  not covered until the source is fixed.

### text_generation_server/{,dynamic_text_gen_server/}tokenization.py (58% / 50%)

- The public `tokenize_prompts` function (lines 22-67) requires `torch.distributed`
  to be initialised so `broadcast_int_list` / `broadcast_tensor` can run. We test the
  internal `_tokenize_prompts_and_batch` helper exhaustively (all eod/eos/no-token branches,
  add_BOS, padding lengths). The broadcast wrapper is covered by existing functional tests
  that bring up a ProcessGroup.

### text_generation_server/dynamic_text_gen_server/endpoints/health.py (91%)

- Missing lines 37-38 are the `except ImportError` branch for quart. With quart installed
  in the run's venv (now part of the install line), this branch is unreachable.

### quantization/utils.py (36%)

- Tested all CPU-tractable predicates: `_should_quantize_param` (5 input cases), `_to_bf16`
  (TE / wrapped TE / plain), `collect_mxfp8_param_metadata` (empty model + plain + TE param).
  Missing lines (164-210, 222-236, 245-264) are the actual quantization paths
  (`quantize_params_to_mxfp8`, `_mm_mxfp8_flashinfer`, `_mm_mxfp8_torch`, `mm_mxfp8`) which
  call into FlashInfer / `torch.nn.functional.scaled_mm` and need bf16 → mxfp8 round-tripping
  on real CUDA. Out of scope for unit tests.

## Combined Final Summary (top-level + subdirectory + GPU passes)

| File | Baseline | Final | Delta |
|------|----------|-------|-------|
| async_stream.py | 0% | **100%** | +100% |
| headers.py | 0% | **100%** | +100% |
| sampling_params.py | 0% | **100%** | +100% |
| inference_request.py | 0% | **98%** | +98% |
| inference_client.py | 0% | **94%** | +94% |
| symmetric_memory.py | 31% | **94%** | +63% |
| engines/abstract_engine.py | 0% | **86%** | +86% |
| engines/mcore_engine.py | 0% | **100%** | +100% |
| contexts/base_context.py | 53% | **95%** | +42% |
| contexts/attention_context/metadata_base.py | 29% | **100%** | +71% |
| contexts/routing_metadata.py | 26% | **100%** | +74% |
| contexts/attention_context/mha_metadata.py | 32% | **100%** | +68% |
| contexts/static_context.py | 24% | **69%** | +45% |
| quantization/mxfp8_tensor.py | 49% | **76%** | +27% |
| quantization/utils.py | 18% | **36%** | +18% |
| sampling/base.py | 0% | **100%** | +100% |
| text_generation_controllers/encoder_decoder_*.py | 0% | **100%** | +100% |
| text_generation_controllers/vlm_*.py | 0% | **100%** | +100% |
| text_generation_server/tokenization.py | 0% | **58%** | +58% |
| text_generation_server/dynamic_text_gen_server/tokenization.py | 0% | **50%** | +50% |
| text_generation_server/dynamic_text_gen_server/endpoints/common.py | 0% | **100%** | +100% |
| text_generation_server/dynamic_text_gen_server/endpoints/health.py | 0% | **91%** | +91% |

**Across all three passes: 22 source files brought to 36–100 % coverage, ~216 new test cases, 19 new test files, 0 final failures.**

---

# Round 4 — Heavier Subdirectory Files

This round picks up files that need real ZMQ socket plumbing, CUDA C++ extension
compilation, or are dominated by Triton kernels — the parts where coverage is
constrained more by what coverage.py can see than by test difficulty.

## Files Targeted (Round 4)

| File | Before | After | Stmts/Miss | Test File |
|------|--------|-------|------------|-----------|
| `engines/async_zmq_communicator.py` | 0% | **68%** | 92/29 | `engines/test_async_zmq_communicator.py` |
| `moe/fused_moe.py` | 34% | **40%** | 53/32 | `moe/test_fused_moe.py` |
| `unified_memory.py` | 18% | **56%** | 216/96 | `test_unified_memory.py` |

**46 new tests, 0 failures.** Validation run: `3ffb68b54fe943668acfd7b7ad04123e`.

## Round 4 Per-File Notes

### engines/async_zmq_communicator.py (68%)

- Tests construct the communicator via `__new__` + injected mocks for the gather/bcast
  ZMQ sockets (rendezvous in `__init__` requires a real ProcessGroup). Both async
  (`all_reduce_max`) and sync (`sync_all_reduce_max`) leader/follower paths covered,
  plus `zmq.Again` retry loops, the value-count assertion, single-rank short-circuits,
  and `close(linger=0)`.
- Missing lines 13-17 (`zmq` ImportError fallback), 44-73 (the `__init__` rendezvous
  body that binds sockets and broadcast_object_list-s addresses), and a few branch
  edges are unreachable without a real ProcessGroup or without uninstalling `zmq`.

### moe/fused_moe.py (40%)

- Pure-Python parts covered: `ActivationType` enum, `_get_activation_func` (both
  fused/unfused branches and the unsupported-activation `ValueError`).
- Missing lines 125-193 are the `mcore_fused_moe` orchestration which calls
  `permute_tokens` / `permute_and_quantize_mxfp8` / `grouped_mm` /
  `scaled_grouped_mm` — all real Triton/CUDA kernels. Out of scope for unit tests.

### unified_memory.py (56%)

- All trivial guards covered: `prefetch_managed_tensor`, `advise_managed_tensor_*`
  with `None`, non-tensor `TypeError`, 0-element `numel`, CPU-tensor `ValueError`.
- Module-walking helpers `prefetch_managed_module_parameters` and
  `advise_managed_module_parameters_preferred_location` covered with: None module,
  CPU-only model, CUDA model, shared-storage dedup, `include_buffers=True`, and
  the `RuntimeError` propagation when the underlying lib returns non-zero.
- Enums and exception classes covered.
- `compile_allocator` covered for the short-circuit (already-attempted) and the
  no-MemPool failure path.
- `create_unified_mempool` covered for both failure messages (specific reason +
  fallback "Unknown reason").
- Missing lines 103-257 are the `_mempool_c_src` triple-quoted CUDA C source
  string and the `load_inline` build path — hard to test in isolation. Lines
  280, 339-349, 372-378, 401-407 are the success branches of the CUDA-mem-prefetch
  helpers and require a real UVM-allocated CUDA tensor; covered by the existing
  functional/integration tests instead.

## Combined Cumulative Summary (Rounds 1-4)

| File | Baseline | Final |
|------|----------|-------|
| async_stream.py | 0% | **100%** |
| headers.py | 0% | **100%** |
| sampling_params.py | 0% | **100%** |
| inference_request.py | 0% | **98%** |
| inference_client.py | 0% | **94%** |
| symmetric_memory.py | 31% | **94%** |
| **unified_memory.py** | **18%** | **56%** |
| engines/abstract_engine.py | 0% | **86%** |
| engines/mcore_engine.py | 0% | **100%** |
| **engines/async_zmq_communicator.py** | **0%** | **68%** |
| contexts/base_context.py | 53% | **95%** |
| contexts/attention_context/metadata_base.py | 29% | **100%** |
| contexts/routing_metadata.py | 26% | **100%** |
| contexts/attention_context/mha_metadata.py | 32% | **100%** |
| contexts/static_context.py | 24% | **69%** |
| **moe/fused_moe.py** | **34%** | **40%** |
| quantization/mxfp8_tensor.py | 49% | **76%** |
| quantization/utils.py | 18% | **36%** |
| sampling/base.py | 0% | **100%** |
| text_generation_controllers/encoder_decoder_*.py | 0% | **100%** |
| text_generation_controllers/vlm_*.py | 0% | **100%** |
| text_generation_server/tokenization.py | 0% | **58%** |
| text_generation_server/dynamic_text_gen_server/tokenization.py | 0% | **50%** |
| text_generation_server/dynamic_text_gen_server/endpoints/common.py | 0% | **100%** |
| text_generation_server/dynamic_text_gen_server/endpoints/health.py | 0% | **91%** |

**Across all four rounds: 25 source files brought to 36–100 % coverage, ~262 new test cases, 22 new test files, 0 final failures.**

---

# Round 5 — Tractable Dynamic-Path Files

This round covers the dynamic-path files where mocking the surrounding context
makes unit testing tractable. The user authorized skipping static engine code
entirely, and asked us to either mock the dynamic paths if possible or
document them as out-of-scope.

## Files Targeted (Round 5)

| File | Before | After | Stmts/Miss | Test File |
|------|--------|-------|------------|-----------|
| `contexts/gpu_view.py` | 3% | **100%** | 99/0 | `contexts/test_gpu_view.py` |
| `contexts/kv_block_allocator.py` | 14% | **50%** | 198/99 | `contexts/test_kv_block_allocator.py` |

**27 new tests, 0 failures.** Validation run: `8941032aad4d4461b47bcadd3b0db6cf`.

## Round 5 Per-File Notes

### contexts/gpu_view.py (100%)

- `ContextGPUView` is a single class whose `__init__` carves a `uint8` buffer
  into typed views. Tests cover both Mamba and non-Mamba layouts, verify the
  zero-init guarantee, view aliasing, and the `assert off == total_bytes`
  layout-bug guard. CUDA-required (`torch.cuda.current_device()` and CUDA
  device inputs).

### contexts/kv_block_allocator.py (50%)

- Tests cover the no-prefix-caching path (init, `__str__`, allocate/release,
  reset, get_*_used / get_*_avail) and the prefix-caching path's ref-count
  bookkeeping, hash registration, REF_ZERO eviction policy.
- Missing lines (208-225, 252-258, 289-316, 324-329, 337-338, 352-365,
  383-419, 439-457, 469-474, 485) are LRU eviction bookkeeping
  (`update_timestamps`, `evict_lru_blocks`, `_deregister_blocks` callback
  paths), routing-data persistence (`get_block_routing` / `set_block_routing`),
  and the `lookup_kv_blocks_by_hashes` prefix discovery API. These all touch
  fields on the surrounding `DynamicInferenceContext`
  (`prefix_cache_lru_clock`, `request_to_kv_block_ids`,
  `request_kv_block_counts`, etc.) in ways that are tightly coupled to the
  full bookkeeping cycle. Out of scope for unit tests; covered by integration
  tests on the live inference engine.

---

# Out of Scope (Dynamic-Path Files)

After five rounds of unit-test additions, the following files within
`megatron/core/inference/` remain at low coverage. Each is documented here
with the specific reason a unit test cannot meaningfully cover it.

## Massive orchestrators (need full inference pipeline)

| File | Stmts | Reason |
|------|-------|--------|
| `engines/dynamic_engine.py` | 1069 | Massive orchestrator. Requires a built `MegatronModule` + `DynamicInferenceContext` + KV-cache buffers + a real `ProcessGroup`. The existing `tests/unit_tests/inference/engines/test_dynamic_engine.py` SIGABRTs midway when run with `--cov`, suggesting it requires multi-GPU + specific memory layouts to even start. Covered by integration tests, not unit tests. |
| `contexts/dynamic_context.py` | 1388 | Massive context machinery (KV cache, slot allocator, GPU view, transfer-bookkeeping). Public API is consumed by `dynamic_engine.py`. To exercise any non-trivial branch, you need the full engine + a model that produces tokens. Not unit-testable. |
| `text_generation_controllers/text_generation_controller.py` | 869 | Base controller orchestrating prefill / decode / sampling against a real model wrapper. Every method requires a built `inference_wrapped_model`, tokenizer, and active inference context. |

## Mamba-specific dynamic state (need mamba model + chunk config)

| File | Stmts | Reason |
|------|-------|--------|
| `contexts/mamba_slot_allocator.py` | 288 | Mamba-specific slot allocator. Needs `MambaInferenceStateConfig` derived from a built mamba/hybrid model and live request tracking. |
| `contexts/attention_context/mamba_metadata.py` | 326 | Mamba kernel metadata (chunked prefill cu_seqlens, conv state indices). Needs a hybrid model and the surrounding dynamic context's mamba bookkeeping fields populated. |

## HTTP endpoints / live ZMQ services

| File | Stmts | Reason |
|------|-------|--------|
| `data_parallel_inference_coordinator.py` | 297 | Long-running ZMQ coordinator daemon driving real inference engines via PUB/SUB + DEALER/ROUTER sockets. Testing a coordinator's coordination logic requires standing up real engines on fake addresses; the result is a functional test, not a unit test. |
| `text_generation_server/dynamic_text_gen_server/text_generation_server.py` | 114 | Hypercorn ASGI server driven by `multiprocessing.Process` workers. Requires live processes + ports + a coordinator. |
| `text_generation_server/dynamic_text_gen_server/endpoints/completions.py` | 153 | Quart HTTP endpoint that drives a live `InferenceClient` against a coordinator. Each request awaits a real inference future. The test client setup we used for `health.py` works because `health` only reads `current_app.config['client'] is not None`; `completions` actually exercises the client. |
| `text_generation_server/dynamic_text_gen_server/endpoints/chat_completions.py` | 412 | Same as above, plus chat-template formatting and tool/parser dispatch — a small unit test of one branch wouldn't move the needle. |

## Triton-bound files (hard coverage.py ceiling)

`@triton.jit` kernel bodies execute as compiled GPU code; Python's trace hooks
don't fire for them. These files cap out somewhere between ~30 % and ~70 %
no matter how many tests you write — the missing percentage is the kernel
source itself.

| File | Stmts | Reason |
|------|-------|--------|
| `contexts/fused_kv_append_kernel.py` | 55 | Single Triton kernel + a Python wrapper that asserts `HAVE_TRITON`. |
| `moe/permute.py` | 229 | Multiple Triton kernels for token permutation. |
| `moe/activations.py` | 89 | Triton kernels for activation + fused quantize. |
| `moe/vllm_fused_moe.py` | 170 | vLLM-style fused MoE Triton kernels. |
| `text_generation_controllers/mtp_utils_triton.py` | 139 | Triton kernels for MTP. |
| `communication/torch_symm_triton/barrier.py` | 30 | Symmetric-memory barrier Triton kernel. |
| `communication/torch_symm_triton/collectives.py` | 95 | NVLS collectives Triton kernels. |
| `communication/torch_symm_triton/fused_collectives.py` | 113 | Fused NVLS collectives. |
| `communication/torch_symm_triton/multimem_asm.py` | 43 | Inline-PTX multimem store/load. |
| `communication/torch_symm_triton/utils.py` | 33 | Triton utility helpers (`sync_threads`, etc.). |
| `communication/torch_symm_triton/variable_collectives.py` | 208 | Variable-sized NVLS collectives. |

## Other model-bound files

| File | Stmts | Reason |
|------|-------|--------|
| `model_inference_wrappers/abstract_model_inference_wrapper.py` | 82 | Constructor calls `get_model_config(self.model)` and walks pipeline-parallel groups. Requires a real `MegatronModule`. |
| `model_inference_wrappers/gpt/gpt_inference_wrapper.py` | 38 | Subclass of the above, takes a built `GPTModel`. |
| `model_inference_wrappers/t5/t5_inference_wrapper.py` | 67 | Subclass of the above, takes a built `T5Model`. |
| `text_generation_controllers/mtp_utils_pytorch.py` | 98 | MTP (multi-token-prediction) helpers tied to a built MTP model. |
| `engines/static_engine.py` | 132 | User instructed to skip static-engine related files. |

---

# Combined Cumulative Summary (Rounds 1-5)

| File | Baseline | Final |
|------|----------|-------|
| async_stream.py | 0% | **100%** |
| headers.py | 0% | **100%** |
| sampling_params.py | 0% | **100%** |
| inference_request.py | 0% | **98%** |
| inference_client.py | 0% | **94%** |
| symmetric_memory.py | 31% | **94%** |
| unified_memory.py | 18% | **56%** |
| engines/abstract_engine.py | 0% | **86%** |
| engines/mcore_engine.py | 0% | **100%** |
| engines/async_zmq_communicator.py | 0% | **68%** |
| contexts/base_context.py | 53% | **95%** |
| contexts/attention_context/metadata_base.py | 29% | **100%** |
| contexts/routing_metadata.py | 26% | **100%** |
| contexts/attention_context/mha_metadata.py | 32% | **100%** |
| contexts/static_context.py | 24% | **69%** |
| **contexts/gpu_view.py** | **3%** | **100%** |
| **contexts/kv_block_allocator.py** | **14%** | **50%** |
| moe/fused_moe.py | 34% | **40%** |
| quantization/mxfp8_tensor.py | 49% | **76%** |
| quantization/utils.py | 18% | **36%** |
| sampling/base.py | 0% | **100%** |
| text_generation_controllers/encoder_decoder_*.py | 0% | **100%** |
| text_generation_controllers/vlm_*.py | 0% | **100%** |
| text_generation_server/tokenization.py | 0% | **58%** |
| text_generation_server/dynamic_text_gen_server/tokenization.py | 0% | **50%** |
| text_generation_server/dynamic_text_gen_server/endpoints/common.py | 0% | **100%** |
| text_generation_server/dynamic_text_gen_server/endpoints/health.py | 0% | **91%** |

**Across all five rounds: 27 source files brought to 36–100 % coverage, ~289 new test cases, 24 new test files, 0 final failures.**

---

# Round 6 — dynamic_engine.py Mock-Based Unit Tests

After previously listing `engines/dynamic_engine.py` (1069 stmts) as out-of-scope
because it requires a built MegatronModule + KV cache + ProcessGroup, this
round adds a complementary mock-based test file that exercises the
unit-testable surface (standalone helpers, enums, and the engine methods that
only touch `self.*` state).

## Files Targeted (Round 6)

| File | Before (alone) | After (combined) | Stmts/Miss | Test File |
|------|----------------|-------------------|------------|-----------|
| `engines/dynamic_engine.py` | 14% (new tests alone) | **61%** (existing + new at nproc=1) | 1069/417 | `engines/test_dynamic_engine_unit.py` |

**33 new mock-based tests, 0 failures.** Validation runs:
- New tests alone: `bfce3c9347074bceafcda34715ef97fd` — 33 passed, 14% on dynamic_engine.py
- Combined (existing `test_dynamic_engine.py` + new): `679f04a21f43436aae38e99c94a67c02` — 123 passed, 87 multi-GPU tests fail at nproc=1 (expected — they need TP/PP > 1), 93 skipped, **61% on dynamic_engine.py**.

## Round 6 Per-File Notes

### engines/dynamic_engine.py (61% combined)

The existing `test_dynamic_engine.py` exercises the engine end-to-end on a
real model + KV cache + ProcessGroup. Most of its parameterised cases require
TP/PP > 1 GPUs (`TestDynamicInferenceEngineParallel::test_parallel_inference`,
`test_sequence_parallel_fp8_inference`, `TestChunkedPrefillCudaGraphs`) — they
fail at `nproc-per-node=1` with NCCL/world-size errors, but the single-GPU
subset still produces ~47% coverage.

The new `test_dynamic_engine_unit.py` complements this with 33 mock-based
tests that need no GPU, no model, no `ProcessGroup`. Specifically tested:

- **Module-level helpers (no engine instance):**
  - `EngineState` enum (member uniqueness, presence of all protocol states,
    membership in `_STATE_EVENTS` stable subset)
  - `EngineSuspendedError` exception
  - `format_mem_bytes` for all five suffix branches (bytes / kb / mb / gb / tb)
    plus the trailing fallthrough for 0 bytes
  - `RequestEntry` kw-only dataclass

- **Engine methods (via `__new__` + injected mock attributes):**
  - `has_unfinished_requests` (3 cases — context, waiting queue, both empty)
  - `get_request` (existing id + KeyError on unknown id)
  - `get_prefix_coordination_metrics`
  - `_get_and_clear_stop_word_finished_ids` (empty / intersect-and-clear /
    disjoint)
  - `_check_stop_words_for_request_post_append` (no stop words, empty list,
    too-few tokens, end-of-tokens hit, `detokenize_stop_sequence=True`,
    speculative-decoding mid-sequence trim, no match)
  - `_find_mamba_match_count` (empty hashes, farthest-match, no match,
    first-block-only match)

The 417 lines still uncovered fall into three groups:
- **Cuda-graph capture / suspend / resume** (lines 333-456, 728-770, 776-830):
  require real `torch.cuda.graph` capture and a live model.
- **`schedule_*` and `step_*` orchestration** (lines 1544-2003, 2097-2440):
  require a populated `DynamicInferenceContext` + active requests mid-flight.
- **`post_process_requests`** (lines 1088-1360): needs a real text generation
  controller producing logits each step.

These remain covered by the existing functional tests (when run on multi-GPU).

## Learnings Fed Back

- Added to `run-tests.md` Known Quirks: `omegaconf` is required for `tests/unit_tests/conftest.py` to load — pip install `pytest-cov omegaconf` together when running coverage.
- Added to `run-tests.md` Known Quirks: `MagicMock.side_effect = [a, b]` exhausts after 2 calls and raises `StopIteration` on the 3rd; `_recv_task`-style infinite loops need a callable `side_effect` that returns the first value once then keeps raising the "no data" exception.
- Added to `run-tests.md` Known Quirks: `task.cancel()` followed by `await task` propagates `asyncio.CancelledError` (a `BaseException`, not `Exception` since Py 3.8) — `except Exception` will not catch it.
- Added to `run-tests.md` Known Quirks: When testing `serialize`/`deserialize` round-trips that use the `("tensor", [...])` / `("ndarray", {...})` tuple wrappers, msgpack transport converts the tuples to lists. To exercise the deserializer's wrapper-handling branches, msgpack-roundtrip the serialized output before calling `deserialize()`; otherwise the `isinstance(v, list)` check in `_post_deserialize` does not match.
