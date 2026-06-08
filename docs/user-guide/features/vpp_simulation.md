<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# VPP Training Simulation

VPP training simulation estimates one global-batch step for an interleaved
pipeline-parallel training recipe without launching the full pipeline world. It
is intended for performance sampling and schedule analysis of large PP/VPP
models when running the complete trainer is expensive.

## Execution Model

Simulation launches an executor world and folds pipeline parallelism out during
initialization. After Megatron initialization, the simulator restores the target
pipeline layout and sequentially samples the work that would have run on each
virtual pipeline rank.

For a target recipe with `PP = pipeline_model_parallel_size`, the executor
world is typically the target world with PP removed:

```text
executor_world_size = launched WORLD_SIZE
vtrainer_world_size = executor_world_size * PP
```

For example, a 256-GPU target recipe with `TP2/PP8/CP1` can be sampled on
32 executor GPUs. The simulator restores `PP=8`, samples all PP stages, and
reconstructs a full 256-GPU global-batch timeline.

## Data Parallel Size

Megatron derives the non-expert data-parallel size from the full virtual
trainer shape:

```text
data_parallel_size = vtrainer_world_size / (TP * PP * CP)
```

During simulation initialization, `PP` is temporarily set to 1 so the smaller
executor world can initialize process groups. After initialization, simulation
restores the original PP layout and expands `data_parallel_size` by the
restored PP size. The global-batch microbatch count therefore follows the
target trainer:

```text
num_microbatches = global_batch_size / (micro_batch_size * data_parallel_size)
```

MoE expert data parallelism is a separate expert-group dimension. It is derived
from the expert tensor/pipeline/expert-parallel shape and should not be confused
with the non-expert `data_parallel_size`.

## Task DAG And Timing

The simulator builds forward and backward tasks for every tuple of
`(pp_rank, microbatch_id, model_chunk_id)`. Dependencies are created from:

* the pipeline schedule for each PP rank,
* forward-to-backward dependencies for the same microbatch and chunk,
* cross-PP forward and backward edges.

Each task stores a measured duration in milliseconds. The analyzer computes the
global-batch time from the reconstructed DAG:

```text
gbs_time = latest_task_end_time - earliest_task_start_time
```

The primary throughput metric is normalized by the full virtual trainer size,
because the reconstructed time models the target full-world global-batch step:

```text
throughput_per_gpu = FLOPs(one GBS) / (gbs_time * 1e12 * vtrainer_world_size)
```

The executor-normalized throughput is also reported as a sampling diagnostic:

```text
executor_normalized_throughput =
    FLOPs(one GBS) / (gbs_time * 1e12 * executor_world_size)
```

## Memory And Allocator Lifetime

Simulation should time the same compute region that the full trainer executes.
Per-repeat cleanup must not include allocator-level cache eviction. In
particular, `torch.cuda.empty_cache()` is kept at model-chunk teardown
boundaries but is not part of the per-repeat timing release path.

Gradient tensors are also kept allocated across timing repeats and zeroed
instead of being set to `None`. This keeps the CUDA caching allocator and CUDA
graph pools stable after warmup, avoiding repeated large allocations and making
sampled chunk timing closer to full training.

## Unsupported Features

Simulation currently samples individual forward and backward tasks directly
instead of running the full training loop's CUDA graph capture and replay
lifecycle. It therefore rejects CUDA graph modes while `--simulate-global-step`
is enabled:

* `--cuda-graph-impl` values other than `none`,
* non-empty `--cuda-graph-modules`,
* `--optimizer-cuda-graph`.

These guards prevent reporting eager task timings as if they were equivalent to
full training with captured CUDA graphs. CUDA graph support should be added only
after the simulator can run the same warmup, capture, and replay lifecycle as
the corresponding training path.

## Unit Test Coverage

`tests/unit_tests/simulation/test_vpp_simulator_e2e.py` is a single-node E2E
smoke test. It initializes the real NCCL process group on the local ranks and
uses a tiny MCore GPT model to run the simulator's real forward/backward timing
functions:

* task creation and scheduling,
* model creation and static-memory collection,
* warmup and measured CUDA task execution,
* result serialization,
* analyzer load and throughput reporting.

The test is expected to run under the regular Megatron unit-test launcher with
one node and eight processes. All ranks exercise the simulator execution path,
and rank 0 validates the saved result files, positive measured task durations,
and the `executor_world_size` versus `vtrainer_world_size` throughput
accounting.
