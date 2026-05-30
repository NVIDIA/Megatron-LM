---
name: mcore-experiment-design
description: Design bounded Megatron training experiments with controls, metrics, and thresholds.
---

# Skill: MCore Experiment Design

## Purpose
Turn a Megatron training goal or suspected failure into bounded experiments with controls, metrics, acceptance criteria, and artifact capture.

## Trigger Conditions
- A user asks how to validate a model, optimizer, data, or distributed change.
- A training signal needs an ablation plan.
- A benchmark claim needs repeatable evidence.

## Method
1. State the hypothesis in one sentence.
2. Define the control run and treatment run.
3. Change one variable per experiment unless the user explicitly accepts a factorial plan.
4. Specify hardware, topology, seed, data slice, checkpoint source, and runtime budget.
5. Identify metrics: loss, grad norm, LR, memory, tokens/sec, accuracy, logits, golden values, or restart success.
6. Define pass/fail thresholds and stop conditions.
7. Add a small smoke run before expensive multi-node validation.
8. Capture logs, configs, git SHA, container, environment, and output paths.

## Output
Experiment plan sections for `jobs/current/outputs/training-review.md`.

## Edge Cases
- If two variables changed, require rollback or paired ablation.
- If hardware is unknown, state assumptions and provide the exact setting to confirm.
