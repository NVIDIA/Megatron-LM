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

## Examples
See `examples/rl` for demonstrations of:
- Custom `InferenceInterface` endpoints
- Example `Agents`