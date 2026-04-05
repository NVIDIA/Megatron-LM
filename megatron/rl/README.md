# Megatron-RL

## Status
08/27/2025: Megatron-RL is actively under development. While it is functional internally at NVIDIA, it is not yet usable by external users because not all required code has been released. The available code and examples may change as development progresses. For a current roadmap of planned Megatron-RL features please see [#1776](https://github.com/NVIDIA/Megatron-LM/issues/1776).

## Overview
Megatron-RL is adding native reinforcement learning (RL) based post-training to Megatron-LM. It provides a flexible library for defining RL environments and agents, extending the Megatron-LM training loop with RL algorithm support.

The bulk of the new library code is located in `megatron/rl`. However:
- Significant modifications have been made to the Megatron Core inference code.
- Minor changes were made to the Megatron-LM training loop to enable Megatron-RL.

Example environments for Megatron-RL can be found in `examples/rl`.

Megatron-RL is designed for research teams exploring RL post-training of LLMs at scale on state-of-the-art foundation models with cutting-edge performance on the latest NVIDIA hardware.

It is **not** intended as an enterprise framework and won't necessarily provide out-of-the-box support for any given open model. For those capabilities please refer to [Nemo RL](https://github.com/NVIDIA-NeMo/RL).

## Design

The design philosophy of Megatron RL is to keep the agent/environment design as decoupled as possible from the underlying RL implementation.
- The environment design defines the "Agent Environment" which takes as input a handle to an inference interface (i.e. something supporting `.generate(prompt, **generation_args)`) and must return experience rollouts along with rewards.
- The RL training system handles batching inference requests, hosting inference, training and other orchestration tasks.

Below we describe the different conceptual components and how they divide responsibility.

### Agent and Environment (referred to as an Agent)
- Takes a handle to an `InferenceInterface`.
- Returns `Rollout` or `EvaluationResponse` objects.
- Responsible for sampling parameters, custom generation arguments (e.g., stop conditions, inline evaluation), etc.

### Trainer/Evaluator
- Manages the control flow for rollout generation and evaluation.
- Coordinates with (or creates) the `InferenceInterface` and `Agent`s.

### Inference Interface
- Provides the endpoint the `AgenticEnvironment` uses to run `.generate(prompt, **generation_args)`.
- Can take many forms (e.g. Megatron, OpenAI, HF) and supports many configuration options.

## Off-Policy Generation

Pure on-policy RL runs rollout generation and training sequentially: each training step has to wait for fresh rollouts from the current policy, and then the inference engine sits idle while training runs. This wastes hardware.

Off-policy generation overlaps the two. While training is busy with step *N*, the inference engine keeps generating rollouts that will be consumed by steps *N+1*, *N+2*, … The rollouts come from slightly older policy weights, so they are *stale* by some amount. We call this the **collection lag**:

> A lag of *L* means the rollouts consumed by the current training step were generated *L* training steps earlier, i.e. by a policy that is *L* updates behind the one we're training now.

Higher lag means more overlap (throughput) at the cost of more staleness (sample quality / stability). Finding the right point on that tradeoff is the main thing this knob exists for.

### Turning it on

Off-policy generation requires `--rl-partial-rollouts`. The lag itself is set with `--rl-desired-lag`:

```
--rl-partial-rollouts --rl-desired-lag 2
```

Useful values:
- `L = -1`: the minimum — only 1 prompt group in flight at a time (effectively serial, mostly useful for testing).
- `L = 0`: fill exactly one training step's worth of generation capacity. The inference engine is never ahead of the trainer.
- `L = 1, 2, …`: the engine runs this many steps ahead. Higher L = more throughput, more staleness.

### Autotuning

If you omit `--rl-desired-lag`, the system sets it automatically, and prints what it picked at startup.    
Alternatively, if you provide your own `--rl-desired-lag`, the same log line will inform you whether *your* `--rl-desired-lag` oversubscribes the inference engine —
i.e. whether you're asking for more in-flight work than the engine can actually serve.    
This is why it is important to allow `L` to turn negative. Negative values of `L` occur when there are so few inference resources that even fully-synchronous RL will oversubscribe the engine.
Given such information, users can utilize the formula below to appropriately scale their inference resources:

```
max_effective_lag = DP * engine.max_requests / (G * P) - 1
```

where `DP` is the inference data-parallel size, `G` is `--grpo-group-size`, `P` is `--grpo-prompts-per-step`, and `engine.max_requests` is the per-rank request capacity of the inference engine (set via `--inference-max-requests` or derived from KV-cache memory).

### Strict vs. non-strict lag

By default, lag is *non-strict*: rollouts arrive as soon as generation finishes, so any given training step consumes a mix of rollouts at slightly different staleness, averaging out to the requested *L*. This gives the best throughput.

With `--rl-use-strict-lag`, the system enforces that every training step consumes rollouts from exactly *L* steps ago. This trades a small amount of throughput for determinism, which is useful for debugging, reproducibility, or ablations where lag-variance would be a confound.

### How it maps to the internal knobs

Under the hood, the lag controls how many prompt groups are in flight simultaneously:

```
tasks  = round((L + 1) * P)             # total prompt groups in flight
rollouts_in_flight = tasks * G          # total concurrent inference requests
```

In strict mode, those `tasks` are grouped into ordered batches of `gcd(tasks, P)` groups; in non-strict mode every group is its own batch. You generally don't need to think about this — just pick an `L` (or let autotune pick one) and check the startup log to confirm the resulting task count is what you expected.

## Examples
See `examples/rl` for demonstrations of:
- Custom `InferenceInterface` endpoints
- Example `Agents`
