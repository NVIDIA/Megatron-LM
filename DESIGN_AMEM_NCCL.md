# Design: AMem NCCL Integration for RL Training

## Overview

This PR integrates optional AMem (Asynchronous Memory) NCCL-based offloading support for Megatron RL training workflows. The feature enables releasing NCCL-allocated communicator buffers during RL rollout phases to reduce GPU memory pressure.

## When AMem Hooks Are Enabled

AMem NCCL integration is activated when:
- `--rl-amem-offload-during-rollout` flag is set
- Training is in RL mode (rollout generation phase)
- NCCL backend is available for CPU-GPU memory transfers

The integration is **completely optional** and disabled by default. No impact on existing training workflows.

## Why RL Rollout Benefits

RL training has two distinct phases:
1. **Rollout Phase**: Generate responses/trajectories (inference-heavy, memory-intensive)
2. **Training Phase**: Update model parameters (compute-heavy)

During rollout:
- Large batches of sequences are generated
- Activation tensors accumulate rapidly
- GPU memory becomes a bottleneck before compute saturation
- NCCL buffer memory (communicator allocations) is idle during rollout generation

AMem enables:
- Releasing GPU memory allocated by NCCL (communication buffers)
- Restoring NCCL memory before training collective operations resume
- Reducing peak GPU memory during rollout phases where NCCL buffers are otherwise idle

## What Memory Is Offloaded

AMem enables:
- Offloading GPU memory allocated by NCCL (communication buffers)
- Restoring NCCL memory before training collective operations resume
- Reducing peak GPU memory during rollout phases where NCCL buffers are otherwise idle

AMem does NOT offload:
- Model parameters
- Optimizer states
- Gradient buffers
- Activations

Parameter and optimizer offloading are handled separately via
`--rl-offload-optimizer-during-inference` and related flags.

## Implementation Details

### Entry Point
`megatron/training/initialize.py`:
- Checks `--rl-amem-offload-during-rollout` flag
- Initializes AMem NCCL backend if enabled
- Sets up environment variables (NCCL_ALGO=Ring for stability)

### Core Logic
`megatron/core/amem_nccl.py`:
- Wraps NCCL operations for CPU-GPU memory transfers
- Manages memory pinning and buffer registration
- Provides async offload/prefetch primitives

### RL Integration
`megatron/rl/rl_utils.py`:
- Hooks into rollout phase entry/exit
- Triggers offload before rollout starts
- Triggers prefetch before training phase begins

### Configuration
`megatron/training/arguments.py`:
- Single flag: `--rl-amem-offload-during-rollout`
- No complex tuning parameters exposed initially
- Defaults designed for safety and stability

## Non-Goals / Out of Scope

- **No performance claims**: This PR establishes the integration. Performance tuning and benchmarks will be addressed separately.
- **No automatic memory management**: Offload is explicit, triggered by RL phase transitions.
- **No impact on non-RL training**: Code paths are isolated to RL workflows.

## Testing Strategy

- Unit tests validate NCCL offload/prefetch operations
- Functional tests ensure RL training completes successfully with flag enabled
- Backward compatibility: existing RL scripts work unchanged (flag defaults to off)

## Future Work

- Fine-tune offload granularity (which layers, when)
- Benchmark memory savings vs. transfer overhead
- Explore overlapping strategies for multi-stage pipelines
- Extend to other memory-constrained scenarios beyond RL

---

**Note**: This design focuses on integration correctness and safety. Performance optimization will be data-driven based on real RL workloads.
