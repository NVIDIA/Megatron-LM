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

Off-policy generation requires `--rl-partial-rollouts`. The lag itself is set with `--rl-generation-lag`:

```
--rl-partial-rollouts --rl-generation-lag 2
```

Useful values:
- `L = -1`: the minimum — a single unit of generation work in flight at a time (effectively serial, mostly useful for testing; only distinguishable from `L = 0` at submission granularity `G` or `R`, see below).
- `L = 0`: fill exactly one training step's worth of generation capacity. The inference engine is never ahead of the trainer.
- `L = 1, 2, …`: the engine runs this many steps ahead. Higher L = more throughput, more staleness. Fractional values are allowed.

### Autotuning

If you omit `--rl-generation-lag`, the system sets it automatically, and prints what it picked at startup.
Alternatively, if you provide your own `--rl-generation-lag`, the same log line will inform you whether *your* `--rl-generation-lag` oversubscribes the inference engine —
i.e. whether you're asking for more in-flight work than the engine can actually serve.
This is why it is important to allow `L` to turn negative. Negative values of `L` occur when there are so few inference resources that even fully-synchronous RL will oversubscribe the engine.
Given such information, users can utilize the formula below to appropriately scale their inference resources:

```
max_effective_lag = DP * engine.max_requests / (G * P) - 1
```

where `DP` is the inference data-parallel size, `G` is `--grpo-group-size`, `P` is `--grpo-prompts-per-step`, and `engine.max_requests` is the per-rank request capacity of the inference engine (set via `--inference-max-requests` or derived from KV-cache memory).

### Submission and consumption granularity

Two further flags shape the overlap by picking the pipeline's *units of work*: a single rollout (`R`), a prompt group (`G` — one prompt's `--grpo-group-size` samples), or a whole training batch (`B` — all `P` groups of a step):

- `--rl-submission-granularity {R,G,B}` sets the unit the engine's capacity is parceled out in. A unit occupies its share of the engine from admission until it completes; only then is that capacity released for new work.
- `--rl-consumption-granularity {G,B}` sets the unit the trainer waits for: `G` consumes prompt groups in completion order, `B` consumes whole batches in submission order. Consumption can't be finer than submission (`B`-submission forces `B`-consumption), and per-rollout consumption is impossible under GRPO — advantages are relative to the group's rewards, so the group is the atomic training unit.

Writing modes as submission/consumption, the supported modes are `B/B` (the default), `G/G`, `G/B`, `R/G`, and `R/B`. Their behavior is governed by one fact: a unit is only finished when its *slowest* rollout is finished. With long-tailed generation lengths, a unit's expected completion time therefore exceeds the mean rollout time by a *tail factor* `τ = E[slowest rollout in unit] / E[rollout]`, which grows with unit size: `τ_rollout = 1 ≤ τ_group ≤ τ_batch`. Two laws follow (for an engine right-sized to the lag gate — which is exactly what autotuning arranges):

- **Utilization is set by submission alone:** `U = 1 / τ_submit`. A coarse unit holds engine slots while its stragglers finish, idling the engine a fraction `1 − 1/τ_submit` of the time; finer submission backfills freed slots immediately, and `R`-submission keeps the engine fully busy.
- **Staleness is set by the ratio:** mean first-token staleness `≈ (τ_consume / τ_submit) · (1 + L)`. Consuming coarser than you submit multiplies staleness by every tail factor spanned in between, while buying *zero* utilization — so at fixed submission the matched mode dominates (`G/G` over `G/B`, `R/G` over `R/B`).

Consumption granularity also sets the staleness *distribution*. Under `G`-consumption a group's staleness scales with its own completion time — long groups always train proportionally staler. `B/B` is the special corner: releasing capacity only at consumption phase-locks generation to the training clock, so every group in every step trains at *exactly* `L` versions of staleness — deterministic, and one version below the `(1 + L)` that release-on-completion modes pay. At `L = 0`, `B/B` is fully synchronous on-policy training. Its price is the idle engine (`1 − 1/τ_batch`), which it uniquely does *not* convert into run-ahead staleness.

Rules of thumb:
- `B/B` (default): the on-policy corner — smallest and deterministic staleness, lowest throughput; best for debugging, reproducibility, and ablations where lag-variance would confound.
- `G/G`: `τ_batch/τ_group`× the utilization of `B/B`, for one extra version of mean staleness paid as a length-correlated spread.
- `R/G`: the only route to a fully-utilized engine, at `τ_group`× the staleness of `G/G`. On its own a par trade; with importance-sampling correction (`--rl-inference-logprobs-is-correction`) it becomes a win — corrected staleness is a statistical cost, while an idle engine is unrecoverable wall-clock.
- `G/B`, `R/B`: dominated — same utilization as their matched mode with strictly more staleness; use only if you specifically need batch-ordered consumption together with streaming submission.

### How it maps to the internal knobs

Under the hood, the lag controls how many submission units ("tasks") are in flight simultaneously:

```
tasks              = round((L + 1) * P * G / unit)   # in-flight units of generation work
rollouts_in_flight = tasks * unit                    # total concurrent inference requests
```

where `unit` is the number of rollouts per submission unit (`P * G` for `B`, `G` for `G`, `1` for `R`). You generally don't need to think about this — pick an `L` (or let autotune pick one) and check the startup log to confirm the resulting task count is what you expected.

## Examples
See `examples/rl` for demonstrations of:
- Custom `InferenceInterface` endpoints
- Example `Agents`
