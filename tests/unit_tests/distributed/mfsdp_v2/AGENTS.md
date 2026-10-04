# MFSDP v2 test placement

Choose a file by the behavior the test asserts. Extend an existing test or
parametrization when it covers the same behavior.

| File | Put new tests here when they check |
| --- | --- |
| `test_fully_shard.py` | General `fully_shard` correctness: training parity, gradient reduction and accumulation, placement validation, and interactions between features. This includes unsharded, ZeRO, HSDP, and HFSDP configurations. |
| `test_parameter.py` | Parameter ownership and lifecycle: nested ownership, tied or frozen parameters, CPU/meta initialization, and parameter-view validity. |
| `test_optimizer.py` | Optimizer adapters, parameter/gradient dtype compatibility, post-step synchronization, and visibility of updated weights. |
| `test_context.py` | Context construction, sharing and scoping, module traversal and prefetch order, and microbatch context state. |
| `test_memory.py` | Persistent or peak allocation sizes, buffer lifetime, and storage release. |
| `test_overlap.py` | Communication/computation overlap and the effect of prefetch settings on overlap. |
| `test_dbuffer.py` | DBuffer layout, padding, views, casting, redistribution, and collective operations. |
| `test_checkpoint.py` | Checkpoint save/load, metadata, chunk layout, and round-trip correctness. Activation recomputation belongs with the behavior it exercises. |
| `test_cuda_graph.py` | CUDA graph capture and replay. |
| `test_symmetric_memory.py` | Symmetric-memory staging and communication backend behavior. |
| `test_annotation.py` | NVTX ranges and other profiling annotations. |
| `test_mcore_adapter.py` | MCore adapter configuration and integration with MCore models and optimizers. |
| `test_expert_parallel.py` | MFSDP composition with expert parallelism and comparison against a full-batch reference. |

There is no perfect boundary between files. When no focused file is a natural
fit, use `test_fully_shard.py`.

Keep parametrized tests together. Sharding strategy is a test configuration,
not a file boundary: do not create a separate `test_hybrid.py` or split a test
across files just because it covers multiple placement or mesh configurations.
A memory test parametrized over sharding strategies still belongs in
`test_memory.py`; a training-parity test belongs in `test_fully_shard.py`.

Use `conftest.py` for shared fixtures and `profiler_utils.py` for shared profiler
helpers. When moving tests, preserve their decorators and parametrization, and
move any required helpers, imports, and constants with them.
