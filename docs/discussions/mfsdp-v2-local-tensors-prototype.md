# MFSDP v2 local optimizer tensors prototype

Issue: https://github.com/NVIDIA/Megatron-LM/issues/7264

MFSDP v2 uses local optimizer tensors unconditionally. No configuration flag is needed.

## Runtime

Optimizer-facing parameters are ordinary persistent local `nn.Parameter` views.
Backward creates local gradient views instead of rebuilding a DTensor per
parameter per microbatch. Mesh, placement, and logical-shape metadata remain in
the owning parameter group's buffers. MCore gradient norms and zero counts use
that metadata, including replication factors.

Deferred reductions need special handling: the accumulation buffer can hold more
rows than an optimizer parameter. `.grad` remains bound to the optimizer-sized
view; the final reduction fills it before the step. Gradient statistics use the
optimizer layout after reduction. Both `zero_grad(set_to_none=True)` and in-place zeroing
are covered, as are full-iteration and optimizer CUDA graphs.

## Checkpoint boundary

Use MFSDP v2's native `save_checkpoint` / `load_checkpoint` helpers. They construct
DTensor views for model weights and parameter-shaped optimizer state, attach the
existing uneven-shard metadata, and unwrap loaded state before installing it.
Optimizer parameter identities survive loading. Checkpoints interchange with
the existing DTensor runtime in both directions.

Bare `state_dict()` calls expose local shards; they are not a
distributed checkpoint API. MCore's `FullyShardedOptimizer` checkpoint methods
remain unsupported, as on the base branch. This does not add support for Muon,
factored optimizer states, or other non-elementwise optimizer algorithms.

## Reproduce

```bash
PYTHONPATH=$PWD torchrun --standalone --nproc-per-node=2 -m pytest -q --experimental \
  tests/unit_tests/distributed/mfsdp_v2/test_local_tensors.py \
  tests/unit_tests/distributed/mfsdp_v2/test_mcore_adapter.py::TestMcoreAdapterDense::test_build_train_and_step \
  tests/unit_tests/distributed/mfsdp_v2/test_mcore_adapter.py::TestMcoreAdapterDense::test_gradient_clipping_reaches_global_norm \
  tests/unit_tests/distributed/mfsdp_v2/test_mcore_adapter.py::TestMcoreAdapterCudaGraph

PYTHONPATH=$PWD torchrun --standalone --nproc-per-node=2 \
  tools/benchmark_mfsdp_local_tensors.py
```

The benchmark uses 32 small independent Linear modules, two microbatches, FP32
Adam, three warmup steps, and 20 measured steps. It reports synchronized wall
latency separately from a single profiled step's host ranges. It is deliberately
sensitive to per-parameter overhead and does not reproduce the DeepSeek proxy.
Compare separate baseline and prototype worktrees; short timings on a shared machine are noisy.

The local path eliminates all 128 `_FromTorchTensor` events observed per benchmark
step. DeepSeek proxy throughput and eight-rank HSDP/EP measurements remain to be
run on the cluster.

Before removing the runtime switch and gradient-view cache, two paired runs on 2x RTX A6000 gave these
rank-0 median step latencies:

| Run order | DTensor | Local tensors |
| --- | ---: | ---: |
| DTensor first | 103.62 ms | 102.40 ms |
| Local first | 123.08 ms | 106.45 ms |

Final losses were identical. The separately profiled optimizer host range fell
from 2.0–2.3 ms to about 1.1 ms, and roughly 5 ms of DTensor-wrap events disappeared.
Wall-time gains varied substantially with run order; an earlier exploratory run
even showed a small regression. These measurements establish reduced wrapper
work, not a reliable end-to-end speedup for the workload in #7264.
