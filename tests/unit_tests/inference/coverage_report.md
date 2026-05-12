# Coverage Report: `megatron/core/inference`

Generated: 2026-05-07
Run: 3 batches with `--cov-append`, `nproc=8`, on `cw-dfw` cluster.
Session: `test-session` · Request: `6afad7a500b045b59910657f7785dd5b` · Exit code: 0

**Overall: 9583 statements, 2952 missed → 69% covered**

## Test run summary

| Batch | Scope | Passed | Failed | Skipped |
|---|---|---:|---:|---:|
| 1 | Non-engines (top-level + contexts + moe + quantization + sampling + text_generation_*) | 1924 | 75 | 35 |
| 2 | Engines (safe: abstract, static, async_zmq, dynamic_events, dynamic_engine_unit, dynamic_engine_coordinator) | 102 | 0 | 8 |
| 3 | Engines (risky: dynamic_engine, hybrid_prefix_caching_e2e, prefix_caching_cuda_graphs) | ~153 | ~21 | 108 |

Test failures don't affect the coverage table — coverage is recorded for every line executed, regardless of whether the assertion passed.

---

## Top-level (`megatron/core/inference/*.py`)

| File | Stmts | Miss | Cover | Missing Lines |
|---|---:|---:|---:|---|
| `__init__.py` | 0 | 0 | 100% | — |
| `async_stream.py` | 37 | 0 | **100%** | — |
| `batch_dimensions_utils.py` | 142 | 8 | 94% | 94, 98, 112, 128, 171, 178, 258, 368 |
| `common_inference_params.py` | 1 | 1 | 0% | 2 (single import line) |
| `communication_utils.py` | 77 | 22 | 71% | 14, 23, 152-166, 176-187, 197, 209 |
| `config.py` | 135 | 5 | 96% | 332, 338-341 |
| `data_parallel_inference_coordinator.py` | 297 | 138 | 54% | 28-29, 35-36, 151-158, 163, 175-177, 182, 233, 245-251, 260-261, 273-280, 362, 393-396, 403-406, 420-472, 489-534, 538-555, 567-568, 571-576, 589-596, 658-659, 668-675 |
| `headers.py` | 18 | 0 | **100%** | — |
| `inference_client.py` | 100 | 6 | 94% | 18-19, 25-26, 151-152 (ImportError guards) |
| `inference_request.py` | 304 | 0 | **100%** | — |
| `sampling_params.py` | 39 | 0 | **100%** | — |
| `scheduler.py` | 70 | 12 | 83% | 80-85, 106-109, 114-115, 191-193 |
| `symmetric_memory.py` | 85 | 4 | 95% | 23-24, 30-31 (ImportError guards) |
| `unified_memory.py` | 216 | 96 | 56% | 22, 24-25, 62-84, 103-257, 280, 286-314, 339-349, 372-378, 401-407, 438, 460, 462, 465, 470, 498, 501, 504, 516-527 |
| `utils.py` | 137 | 79 | 42% | 22-37, 85, 94-96, 117-119, 138-182, 195-218, 244-248, 268-269, 273-280, 284-288, 297-304 |

## `communication/torch_symm_triton/` (Triton-heavy — kernel bodies cannot be covered by Python coverage.py)

| File | Stmts | Miss | Cover | Missing Lines |
|---|---:|---:|---:|---|
| `__init__.py` | 4 | 0 | **100%** | — |
| `barrier.py` | 30 | 18 | 40% | 12-15, 22, 44, 101-115 |
| `collectives.py` | 95 | 70 | 26% | 14-18, 21-22, 41-64, 80-84, 117-148, 174-204, 223-227, 242-267, 289-303, 335-368 |
| `fused_collectives.py` | 113 | 97 | 14% | 12-19, 35-38, 53-70, 81-111, 129-200, 233-280 |
| `multimem_asm.py` | 43 | 22 | 49% | 14-17, 52-92, 144-162, 196, 228, 266-284, 316, 354-372, 395 |
| `utils.py` | 33 | 14 | 58% | 14-17, 34-36, 46, 65, 84-86, 93, 105 |
| `variable_collectives.py` | 208 | 135 | 35% | 24-28, 32-33, 85-143, 195-253, 419-518, 569-628 |

## `contexts/`

| File | Stmts | Miss | Cover | Missing Lines |
|---|---:|---:|---:|---|
| `__init__.py` | 6 | 0 | **100%** | — |
| `attention_context/mamba_metadata.py` | 326 | 2 | 99% | 697, 714 |
| `attention_context/metadata_base.py` | 21 | 0 | **100%** | — |
| `attention_context/mha_metadata.py` | 28 | 0 | **100%** | — |
| `attention_context/triton/tensor_ops.py` | 165 | 85 | 48% | 12-20, 35-50, 72-97, 113-137, 157-193, 216-259, 264 |
| `base_context.py` | 19 | 1 | 95% | 24 |
| `dynamic_context.py` | 1388 | 128 | 91% | 56-57, 62, 69-70, 156-157, 223, 286, 295-296, 318, 325, 357, 370, 379-380, 422-429, 555, 598-601, 662, 742, 747, 841-858, 870, 1203, 1205-1209, 1233, 1241-1254, 1270, 1277, 1280-1290, 1313, 1327, 1465-1488, 1507, 1553-1559, 1595-1609, 1634-1645, 1666-1684, 1698, 1727, 1759, 1844, 1949-1954, 2185-2186, 2223-2226, 2630, 2634, 2640, 2671, 2708, 2831, 3100, 3131-3136, 3172, 3288, 3355, 3742 |
| `fused_kv_append_kernel.py` | 55 | 25 | 55% | 10-18, 60-90, 131 |
| `gpu_view.py` | 99 | 0 | **100%** | — |
| `kv_block_allocator.py` | 198 | 29 | 85% | 173, 291, 328, 356, 386-419, 449, 455 |
| `mamba_slot_allocator.py` | 288 | 38 | 87% | 133, 164, 194-226, 237, 255, 272, 277, 451, 470, 485-493 |
| `routing_metadata.py` | 35 | 0 | **100%** | — |
| `static_context.py` | 51 | 16 | 69% | 73, 85-119 |

## `engines/`

| File | Stmts | Miss | Cover | Missing Lines |
|---|---:|---:|---:|---|
| `__init__.py` | 3 | 0 | **100%** | — |
| `abstract_engine.py` | 7 | 1 | 86% | 17 |
| `async_zmq_communicator.py` | 92 | 8 | 91% | 13-17, 101, 109, 123, 126 |
| `dynamic_engine.py` | 1069 | 310 | 71% | 71-72, 78-79, 85-86, 91, 100-101, 197, 237-258, 318-321, 340, 385, 392, 399-402, 512-658, 706, 728, 752-755, 766, 777, 803, 807-809, 863-867, 874, 913-915, 944, 963-964, 967-968, 982-986, 1020-1021, 1029, 1039, 1102-1103, 1160, 1194-1197, 1223-1225, 1259, 1348-1350, 1470-1472, 1651-1653, 1671, 1687, 1710, 1713-1715, 1724-1739, 1782-1783, 1801-1802, 1807-1809, 1840-1850, 1864-1899, 1908-1975, 2016-2018, 2033-2041, 2231-2248, 2271-2297, 2308-2313, 2328-2440 |
| `mcore_engine.py` | 1 | 0 | **100%** | — |
| `static_engine.py` | 132 | 37 | 72% | 27-31, 82, 121-130, 134, 169-173, 181, 196-199, 231, 234, 238-243, 284, 294-296, 363, 375-376, 396-397, 401-403 |

## `model_inference_wrappers/`

| File | Stmts | Miss | Cover | Missing Lines |
|---|---:|---:|---:|---|
| `__init__.py` | 0 | 0 | **100%** | — |
| `abstract_model_inference_wrapper.py` | 82 | 2 | 98% | 96, 107 |
| `gpt/__init__.py` | 0 | 0 | **100%** | — |
| `gpt/gpt_inference_wrapper.py` | 38 | 1 | 97% | 81 |
| `multimodal/vlm_inference_wrapper.py` | 72 | 8 | 89% | 31, 50, 152, 179, 187, 196, 204, 209 |
| `t5/__init__.py` | 0 | 0 | **100%** | — |
| `t5/t5_inference_wrapper.py` | 67 | 3 | 96% | 115-117 |

## `moe/`

| File | Stmts | Miss | Cover | Missing Lines |
|---|---:|---:|---:|---|
| `__init__.py` | 7 | 0 | **100%** | — |
| `activations.py` | 89 | 53 | 40% | 19-20, 23-25, 48-60, 103-149 |
| `fused_moe.py` | 53 | 12 | 77% | 28-31, 37-38, 57, 135-142, 155, 179, 189 |
| `metadata.py` | 41 | 24 | 41% | 30-34, 38-39, 72-103 |
| `permute.py` | 229 | 127 | 45% | 23-24, 27-29, 65-73, 92-109, 174-182, 197-201, 258-284, 385-395, 417-432, 517-572 |
| `vllm_fused_moe.py` | 170 | 88 | 48% | 24-25, 28-30, 181-244, 266-271, 291-304, 320-326, 484-503 |

## `quantization/`

| File | Stmts | Miss | Cover | Missing Lines |
|---|---:|---:|---:|---|
| `__init__.py` | 0 | 0 | **100%** | — |
| `mxfp8_quantize.py` | 60 | 36 | 40% | 22-30, 99-157 |
| `mxfp8_tensor.py` | 37 | 3 | 92% | 11, 63-64 |
| `utils.py` | 129 | 82 | 36% | 13-14, 19, 28-29, 35-50, 64-94, 104, 164-210, 215, 222-236, 245-264 |

## `sampling/`

| File | Stmts | Miss | Cover | Missing Lines |
|---|---:|---:|---:|---|
| `__init__.py` | 4 | 0 | **100%** | — |
| `base.py` | 15 | 0 | **100%** | — |
| `flashinfer_sampling.py` | 36 | 24 | 33% | 23-33, 67-101 |
| `torch_sampling.py` | 71 | 0 | **100%** | — |

## `text_generation_controllers/`

| File | Stmts | Miss | Cover | Missing Lines |
|---|---:|---:|---:|---|
| `__init__.py` | 0 | 0 | **100%** | — |
| `encoder_decoder_text_generation_controller.py` | 12 | 0 | **100%** | — |
| `mtp_utils_pytorch.py` | 98 | 0 | **100%** | — |
| `mtp_utils_triton.py` | 139 | 86 | 38% | 12-20, 55-102, 185-216, 283-307, 385-410 |
| `text_generation_controller.py` | 869 | 113 | 87% | 59-60, 93, 116, 129, 178, 372, 468, 591-595, 598-603, 787, 818, 870-877, 925-926, 1100-1137, 1191, 1236-1242, 1341-1351, 1474-1495, 1513, 1515, 1524, 1533-1534, 1574, 1665-1670, 1717, 1731, 1777-1778, 1814, 1941, 1986-1997, 2041, 2048, 2084, 2139-2140, 2203, 2223, 2230-2233, 2247-2249, 2369, 2401-2475 |
| `vlm_text_generation_controller.py` | 14 | 0 | **100%** | — |

## `text_generation_server/`

| File | Stmts | Miss | Cover | Missing Lines |
|---|---:|---:|---:|---|
| `__init__.py` | 1 | 0 | **100%** | — |
| `dynamic_text_gen_server/__init__.py` | 1 | 0 | **100%** | — |
| `dynamic_text_gen_server/endpoints/__init__.py` | 7 | 2 | 71% | 10-11 |
| `dynamic_text_gen_server/endpoints/chat_completions.py` | 412 | 371 | **10%** | 36-38, 43-45, 50-69, 74-76, 80-88, 93-110, 115-129, 134-150, 155-192, 200-204, 217-225, 243-290, 302-317, 326-333, 346-360, 366, 378-400, 406-775 |
| `dynamic_text_gen_server/endpoints/common.py` | 7 | 0 | **100%** | — |
| `dynamic_text_gen_server/endpoints/completions.py` | 153 | 140 | **8%** | 23-296 |
| `dynamic_text_gen_server/endpoints/health.py` | 23 | 2 | 91% | 37-38 |
| `dynamic_text_gen_server/text_generation_server.py` | 114 | 90 | 21% | 16-17, 33-39, 57-111, 125-141, 158-186, 194-215 |
| `dynamic_text_gen_server/tokenization.py` | 38 | 19 | 50% | 22-67, 82-85 |
| `endpoints/common.py` | 7 | 2 | 71% | 13-14 |
| `endpoints/completions.py` | 89 | 75 | **16%** | 14, 26-43, 50-51, 55-212 |
| `run_mcore_engine.py` | 42 | 35 | **17%** | 25-106 |
| `text_generation_server.py` | 152 | 131 | **14%** | 12, 31-33, 37-189, 196-202, 206 |
| `tokenization.py` | 38 | 16 | 58% | 22-67 |

---

## Lowest-coverage files (biggest opportunities, sorted by missed statements)

| File | Stmts | Miss | Cover |
|---|---:|---:|---:|
| `text_generation_server/dynamic_text_gen_server/endpoints/chat_completions.py` | 412 | 371 | 10% |
| `engines/dynamic_engine.py` | 1069 | 310 | 71% |
| `text_generation_server/dynamic_text_gen_server/endpoints/completions.py` | 153 | 140 | 8% |
| `data_parallel_inference_coordinator.py` | 297 | 138 | 54% |
| `communication/torch_symm_triton/variable_collectives.py` | 208 | 135 | 35% |
| `text_generation_server/text_generation_server.py` | 152 | 131 | 14% |
| `contexts/dynamic_context.py` | 1388 | 128 | 91% |
| `moe/permute.py` | 229 | 127 | 45% |
| `text_generation_controllers/text_generation_controller.py` | 869 | 113 | 87% |
| `communication/torch_symm_triton/fused_collectives.py` | 113 | 97 | 14% |
| `unified_memory.py` | 216 | 96 | 56% |
| `moe/vllm_fused_moe.py` | 170 | 88 | 48% |
| `text_generation_server/dynamic_text_gen_server/text_generation_server.py` | 114 | 90 | 21% |
| `text_generation_controllers/mtp_utils_triton.py` | 139 | 86 | 38% |
| `contexts/attention_context/triton/tensor_ops.py` | 165 | 85 | 48% |
| `quantization/utils.py` | 129 | 82 | 36% |
| `utils.py` | 137 | 79 | 42% |
| `endpoints/completions.py` | 89 | 75 | 16% |

## Notes on the low-coverage clusters

- **`communication/torch_symm_triton/*`, `moe/*`, `contexts/attention_context/triton/tensor_ops.py`, `text_generation_controllers/mtp_utils_triton.py`, `quantization/mxfp8_quantize.py`** — these are Triton-heavy. `@triton.jit` kernel bodies execute as compiled GPU code; Python's coverage.py trace hooks do not fire for them, so they always show as missing. Effective ceilings here are ~40–75 % regardless of test quality.
- **`text_generation_server/dynamic_text_gen_server/endpoints/chat_completions.py` and `completions.py`** — ~10 % coverage. These are Quart HTTP route handlers; existing tests likely import the module but don't exercise the routes (or `quart` was missing in some test paths). High-impact target for new tests.
- **`engines/dynamic_engine.py`** — 71 % across 1069 statements. Mock-based unit tests (`test_dynamic_engine_unit.py`, `test_dynamic_engine_coordinator.py` from recent commits) hit the standalone surface, but large blocks (512-658, 1864-1975, 2231-2440) — likely engine-loop / coordinator paths — remain.
- **`unified_memory.py`** (56 %) and **`utils.py`** (42 %) at the top level — both have large unbroken missing ranges (e.g. 103-257, 138-182), suggesting whole code paths with no tests.
