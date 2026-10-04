# MFSDP v2 test placement

Choose a focused file by the behavior the test asserts. Extend an existing
test or parametrization when it covers the same behavior. Use
`test_fully_shard.py` only as a last resort when none of the focused files fit.

| File | Put new tests here when they check |
| --- | --- |
| `test_parameter.py` | Parameter ownership and lifecycle: nested ownership, tied or frozen parameters, CPU/meta initialization, and parameter-view validity. |
| `test_optimizer.py` | Optimizer adapters, parameter/gradient dtype compatibility, post-step synchronization, and visibility of updated weights. |
| `test_context.py` | Context construction, sharing and scoping, module traversal and prefetch order, and microbatch context state. |
| `test_memory.py` | Persistent or peak allocation sizes, buffer lifetime, and storage release. |
| `test_overlap.py` | Communication/computation overlap and the effect of prefetch settings on overlap. |
| `test_dbuffer.py` | DBuffer layout, padding, views, casting, redistribution, and collective operations. |
| `test_quantized_dbuffer.py` | Quantized buffer data and scales, tensor views, quantization, redistribution, and GEMM compatibility. |
| `test_quantization.py` | Quantized MFSDP training correctness and mixed-precision integration. |
| `test_owner_planning.py` | Owner-compute layouts, work assignment, and gather/scatter packing plans without distributed communication. |
| `test_checkpoint.py` | Checkpoint save/load, metadata, chunk layout, and round-trip correctness. Activation recomputation belongs with the behavior it exercises. |
| `test_cuda_graph.py` | CUDA graph capture and replay. |
| `test_symmetric_memory.py` | Symmetric-memory staging and communication backend behavior. |
| `test_annotation.py` | NVTX ranges and other profiling annotations. |
| `test_mcore_adapter.py` | MCore adapter configuration and integration with MCore models and optimizers. |
| `test_expert_parallel.py` | MFSDP composition with expert parallelism and comparison against a full-batch reference. |
| `test_pipeline_parallel.py` | Pipeline-parallel model wrapping and context sharing across virtual pipeline chunks. |
| `test_fully_shard.py` | Last resort: tests that do not fit any focused file above. |

Keep parametrized tests together in the appropriate focused file, including
tests that cover multiple placement or mesh configurations.
A memory test parametrized over sharding strategies still belongs in
`test_memory.py`. Training parity or coverage of multiple sharding strategies
alone is not a reason to use `test_fully_shard.py`.

Use `conftest.py` for shared fixtures and `profiler_utils.py` for shared profiler
helpers. When moving tests, preserve their decorators and parametrization, and
move any required helpers, imports, and constants with them.
