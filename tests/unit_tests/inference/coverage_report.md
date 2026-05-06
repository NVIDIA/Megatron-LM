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

## Combined Final Summary (top-level + subdirectory pass)

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
| quantization/mxfp8_tensor.py | 49% | **76%** | +27% |
| sampling/base.py | 0% | **100%** | +100% |

**Total across both passes: 12 source files brought to 76–100 % coverage, 156 new test cases, 11 new test files, 0 final failures.**

## Learnings Fed Back

- Added to `run-tests.md` Known Quirks: `omegaconf` is required for `tests/unit_tests/conftest.py` to load — pip install `pytest-cov omegaconf` together when running coverage.
- Added to `run-tests.md` Known Quirks: `MagicMock.side_effect = [a, b]` exhausts after 2 calls and raises `StopIteration` on the 3rd; `_recv_task`-style infinite loops need a callable `side_effect` that returns the first value once then keeps raising the "no data" exception.
- Added to `run-tests.md` Known Quirks: `task.cancel()` followed by `await task` propagates `asyncio.CancelledError` (a `BaseException`, not `Exception` since Py 3.8) — `except Exception` will not catch it.
- Added to `run-tests.md` Known Quirks: When testing `serialize`/`deserialize` round-trips that use the `("tensor", [...])` / `("ndarray", {...})` tuple wrappers, msgpack transport converts the tuples to lists. To exercise the deserializer's wrapper-handling branches, msgpack-roundtrip the serialized output before calling `deserialize()`; otherwise the `isinstance(v, list)` check in `_post_deserialize` does not match.
