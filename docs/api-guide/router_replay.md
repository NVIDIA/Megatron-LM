<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Design Document: MoE Router Replay Feature

### 1. Overview

This document provides a detailed description of the "Router Replay" feature implemented within the Megatron-LM Core for Mixture-of-Experts (MoE) models.

This feature is designed to enhance determinism and analyzability in MoE model training and inference. It enables the model to load routing decisions from a predefined file and enforce their use during the forward pass, thereby bypassing the real-time routing computation.

### 2. Motivation

*   **Determinism & Reproducibility**: In distributed training, MoE routing decisions can exhibit minor variations due to factors like floating-point precision. By replaying a fixed routing table, the MoE computation path is guaranteed to be identical across runs, which facilitates debugging and reproducing experimental results.
*   **Performance Profiling**: The router's own computation (e.g., logits calculation, top-k selection) incurs overhead. In replay mode, this part of the computation can be completely skipped, allowing for more precise isolation and profiling of performance bottlenecks within the Expert Layers themselves.
*   **Debugging Aid**: When issues arise in the model, fixing the routing decisions helps to isolate variables, making it easier to determine whether the problem lies with the routing mechanism or the expert computations.

### 3. Design and Architecture

The design follows the principles of being non-intrusive and on-demand, with the core idea of activating the replay logic only when explicitly requested by the user.

*   **Core Components**:
    *   `RouterReplay` (located in `megatron/core/transformer/moe/router_replay.py`): A utility class for replaying MoE routing decisions. When enabled via the `moe_enable_routing_replay` flag, a separate instance of `RouterReplay` is created for each MoE layer's router. Each instance is responsible for loading routing data and providing the deterministic routing decisions for its corresponding layer during the forward pass.
    *   `moe_enable_routing_replay` (located in `megatron/core/transformer/transformer_config.py`): A boolean global configuration flag that serves as the sole entry point for enabling this feature.

*   **Workflow**:
    The feature supports different modes, such as recording and replaying, controlled by a `RouterReplayAction`.

    1.  **Enabling the Feature**: The user sets `moe_enable_routing_replay` to `True` in the model configuration.
    2.  **Initialization**: When `moe_enable_routing_replay` is true, each `TopKRouter` creates its own `RouterReplay` instance.
    3.  **Mode Configuration**: The user must programmatically set the desired router replay action (e.g., `record`, `forward_replay`, `backward_replay`) on the `RouterReplay` instances.
    4.  **Execution Flow (within a mini-batch)**:
        *   **Forward Pass**:
            *   For each micro-batch, the `topk_routing_with_score_function` checks the `router_replay_action`.
            *   **In `record` mode**: The dynamically computed `top-k` expert indices are captured and stored.
            *   **In `forward_replay` mode**: The function retrieves pre-loaded expert indices from `target_topk_idx`. These indices are used for the forward computation and are also appended to the `replay_backward_list` to prepare for the backward pass.
        *   **Backward Pass**:
            *   For each micro-batch (processed in reverse order in pipeline parallelism), the `router_replay_action` is checked again.
            *   **In `backward_replay` mode**: The function retrieves the expert indices for the corresponding micro-batch by popping them from the `replay_backward_list`. This mode is intended for training recomputation (e.g., activation checkpointing and pipeline recompute) so the same routing decisions are used during recompute/backward as in forward, ensuring determinism and correctness.

### 4. Implementation Details

The implementation cleanly separates the replay logic from the router's core computation.

*   **`megatron/core/transformer/transformer_config.py`**:
    *   Adds the configuration option `moe_enable_routing_replay: bool = False`.

*   **`megatron/core/transformer/moe/moe_utils.py`**:
    *   Introduces the `RouterReplay` class to manage the state for recording and replaying routing decisions for a single MoE layer.
        *   `target_topk_idx`: An attribute holding the expert indices for the current micro-batch during forward replay mode.
        *   `recorded_topk_idx`: An attribute for storing the computed expert indices when in record mode.
        *   `replay_backward_list`: A list that accumulates the `top-k` indices used during the forward passes of a mini-batch. This list is consumed in FIFO order during the backward pass to ensure correctness under pipeline parallelism.
        *   `set_target_indices()`: A method to load the replay indices into `target_topk_idx` for the forward pass.
        *   `record_indices()`: A method to save the computed indices.
    *   The `topk_routing_with_score_function` is modified to contain the core logic. It checks the `router_replay_action` on the `router_replay` instance and accordingly performs one of the following actions: computes and records indices, replays indices from `target_topk_idx` (for forward), replays indices from `replay_backward_list` (for backward), or falls through to the default dynamic routing.

#### Training recompute usage
- During forward replay, `set_target_indices()` prepares `replay_backward_list` so each micro-batch’s indices are available for recomputation.
- During recompute/backward, set action to `REPLAY_BACKWARD` so indices are consumed in FIFO order to mirror the forward sequence.

### 5. Usage Guide

1.  **Enable & Instantiate**
    - Create one `RouterReplay` instance per MoE router layer when building the model.
    - Optionally use the global helpers to set/clear actions across all layers.
2.  **Record Routing Decisions**
    - Set action: `RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)`.
    - Run the model; retrieve per-layer indices via `RouterReplay.get_recorded_data()` and persist.
3.  **Forward Replay**
    - Load indices and distribute: `RouterReplay.set_replay_data(list_of_tensors)`.
    - Set action: `RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)`.
    - Run the model; dynamic top‑k is bypassed and target indices are used.
4.  **Backward Replay**
    - For training recomputation (activation checkpointing or pipeline recompute), set action: `REPLAY_BACKWARD` during recomputation.
    - Per micro‑batch indices are consumed from `replay_backward_list` in FIFO order.
5.  **Cleanup**
    - Use `RouterReplay.clear_global_indices()`, `RouterReplay.clear_global_router_replay_action()`, and `RouterReplay.clear_global_router_replay_instances()` to restore default behavior and prevent memory leaks.

#### Quick usage with `topk_routing_with_score_function`

```python
import torch
from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction
from megatron.core.transformer.moe.moe_utils import topk_routing_with_score_function

rr = RouterReplay()

# Record
RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)
logits = torch.randn(8, 16)
probs_rec, routing_map_rec = topk_routing_with_score_function(
    logits=logits, topk=2, use_pre_softmax=False, score_function="softmax", router_replay=rr,
)
recorded = rr.get_recorded_indices()
torch.save(recorded, "/tmp/replay.pt")

# Forward replay
rr.clear_router_replay_action()
rr.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
target = torch.load("/tmp/replay.pt")
rr.set_target_indices(target)
probs_rep, routing_map_rep = topk_routing_with_score_function(
    logits=logits, topk=2, use_pre_softmax=False, score_function="softmax", router_replay=rr,
)

RouterReplay.clear_global_router_replay_action()
RouterReplay.clear_global_indices()
RouterReplay.clear_global_router_replay_instances()
```

### 6. Minimal Demo

Here is a minimal code example showing how to use RouterReplay for recording and replaying:

```python
import torch
import torch.distributed as dist
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction


# Initialize distributed training
if not dist.is_initialized():
    dist.init_process_group(backend="nccl")

# Create a transformer config with RouterReplay enabled
config = TransformerConfig(
    num_experts=8,
    expert_model_parallel_size=1,
    num_top_k=2,
    moe_enable_routing_replay=True
)

# Create a TopKRouter instance
router = TopKRouter(config)

# Generate sample input (batch_size, sequence_length, hidden_size)
logits = torch.randn(16, 32, 8).to(torch.cuda.current_device())

# -----------------
# 1. Recording Mode
# -----------------
print("=== Recording Mode ===")
# Set global router replay action to RECORD
RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)

# Perform routing
routing_output = router.forward(logits)
print(f"Recorded top-k indices shape: {routing_output.top_k_idx.shape}")

# -----------------
# 2. Forward Replay Mode
# -----------------
print("\n=== Forward Replay Mode ===")
# Save recorded indices to a file
torch.save(routing_output.top_k_idx, "/tmp/replay.pt")

# Load indices from file and set as target for replay
replay_indices = torch.load("/tmp/replay.pt")
for router_instance in RouterReplay.global_router_replay_instances:
    router_instance.target_topk_idx = replay_indices

# Set global router replay action to REPLAY_FORWARD
RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

# Perform routing again - this will use the replayed indices
replay_routing_output = router.forward(logits)
print(f"Replayed top-k indices shape: {replay_routing_output.top_k_idx.shape}")
print(f"Are indices the same? {torch.equal(routing_output.top_k_idx, replay_routing_output.top_k_idx)}")


# Clean up
RouterReplay.clear_global_router_replay_action()
RouterReplay.clear_global_indices()
RouterReplay.clear_global_router_replay_instances()
if dist.is_initialized():
    dist.destroy_process_group()
```
