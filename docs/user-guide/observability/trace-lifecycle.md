<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA Corporation is strictly prohibited.
-->

# Trainer trace lifecycle

Megatron emits short traces for startup, each training-loop pass, and terminal
checkpoint finalization. It does not keep a run-long training root open.
Each root uses a fresh trace ID; Resource attributes identify the rank and run.

The `megatron.job` group controls these roots and startup spans. The default
preset includes this group without enabling individual training-iteration or
MCore-detail spans. Other children require their own
[span groups](span-groups.md).

## Startup

`nv.dl.training.startup` is a fresh root, independent of an ambient launcher
or resiliency span. It stays open through initialization and the training
preamble, then ends immediately before the first loop pass.

Already-elapsed children use explicit timestamps and the startup root's parent
context. Intervals with missing, non-finite, negative, reversed, or out-of-root
timestamps are omitted; valid zero-duration intervals remain valid.

| Startup child span | Measured interval |
| --- | --- |
| `nv.dl.training.startup.python` | OS process-creation estimate to the first entrypoint timestamp |
| `nv.dl.training.startup.imports` | First entrypoint timestamp to the entry into `__main__` |
| `nv.dl.training.startup.arg_parse` | Entry into `__main__` to entry into `pretrain()` |
| `nv.dl.training.startup.in_job_setup` | Existing in-process setup end to in-job setup end |
| `nv.dl.training.startup.initialize_megatron` | In-job setup end to initialization end |
| `nv.dl.training.startup.jit_fusion_options` | Initialization end to JIT-fusion setup end |
| `nv.dl.training.startup.model_init` | Model, optimizer, and learning-rate scheduler setup |
| `nv.dl.training.startup.dataloader` | Training, validation, and test iterator setup |
| `nv.dl.training.startup.weight_hash_check` | Initial data-parallel parameter hash check |
| `nv.dl.resiliency.gpu_sniff.startup` | Requested startup GPU benchmark |

The Python interval is a process-start estimate, not an isolated measurement
of CPython initialization. It is considered only on the first startup root
in a process. When OS process time is unavailable, the root can use valid
entrypoint timestamps or its normal creation time; no Python child is invented.
Launcher/container timing belongs outside these trainer startup spans.

## Loop passes and child parenting

Each pass creates `nv.dl.training.iter_block` before its training work,
including passes that skip an iteration or exit before committing it.
Its `nv.dl.training.step` is the same one-based attempted-step identity
used by that pass's training instrumentation; committed checkpoint progress
retains its separate meaning.

The first pass links to the ended startup root. Each later pass links to the
previous pass, while keeping a fresh trace ID. Links are not parent edges.
The chain resets at the next training invocation and is unrelated to the
checkpoint-save interval.

For the usual pretraining path, enabled children retain these parent edges:

| Parent span | Child span |
| --- | --- |
| `nv.dl.training.startup` | `nv.dl.training.startup.model_init` |
| `nv.dl.training.startup.model_init` | `nv.dl.training.checkpoint.load` |
| `nv.dl.training.checkpoint.load` | `nv.dl.training.checkpoint.load.io_read` |
| `nv.dl.training.startup` | `nv.dl.training.startup.dataloader` |
| `nv.dl.training.iter_block` | `nv.dl.training.iteration` |
| `nv.dl.training.iter_block` | `nv.mlm.train.iteration_report` |
| `nv.dl.training.iter_block` | `nv.dl.training.evaluate` |
| `nv.dl.training.evaluate` | `nv.dl.training.evaluate.step` |
| `nv.dl.training.iter_block` | `nv.dl.training.checkpoint.exposed_save` |
| `nv.dl.training.checkpoint.exposed_save` | `nv.dl.training.checkpoint.save` |

There is no enclosing training root after the loop: post-training evaluation
or a final save can therefore begin a separate trace.

## Terminal finalization and teardown

The final loop-pass root ends before the terminal checkpoint drain.
`nv.mlm.checkpoint.exit_finalize` is a separate fresh root, with
`nv.dl.training.checkpoint.save.finalize` enclosing the complete queue drain.
On ordinary completion, this happens in `pretrain()` after any final save.
On an early exit from `train()`, that path performs the terminal drain.
A nonterminal return from `train()` drains pending work without terminating
the worker. Queue callbacks and blocking/termination decisions remain
application-owned.

The telemetry shim owns active spans, context tokens, and telemetry-only exit
hooks. Its managed handle ends active loop/startup spans before Lens shutdown,
then restores the exact incoming child Resource carrier. Repeated teardown is
idempotent, including normal completion followed by the exit fallback.

Hooks install once, only while trainer telemetry is exporting. A graceful
SIGTERM drain flushes already-ended spans without ending the active tree;
hard termination closes the tree and shuts down telemetry. Both preserve
the prior signal handler's behavior. Unsupported signal installation leaves
the exit fallback available. Missing Lens or disabled telemetry adds no hooks,
span initialization, GPU work, or synchronization.
