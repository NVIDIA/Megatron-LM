---
name: mcore-training-signal-diagnosis
description: Diagnose loss, NaN, throughput, memory, and rank-failure signals from training artifacts.
---

# Skill: MCore Training Signal Diagnosis

## Purpose
Interpret Megatron logs, scalar metrics, tracebacks, and benchmark signals to isolate likely causes of instability or regression.

## Trigger Conditions
- Loss spikes, NaNs, infs, divergence, throughput regression, memory regression, or rank-specific failure.
- A user provides TensorBoard summaries, stdout/stderr, rank logs, or metric tables.

## Method
1. Read logs and configs before diagnosing.
2. Locate the first bad step and earliest traceback.
3. Compare loss, grad norm, LR, loss scale, memory, tokens/sec, data batch, and rank-local status around the event.
4. Classify the cause as data, optimizer, precision, memory, distributed, or code regression.
5. Check whether multiple changes landed at once.
6. Pair every conclusion with an artifact path or labeled assumption.
7. Propose the smallest reproduction or one-variable ablation.
8. Define success metrics and stop conditions.

## Output
Signal diagnosis notes for `jobs/current/working/optimization-stability-notes.md` and final findings for `jobs/current/outputs/training-review.md`.

## Edge Cases
- If scalar history is missing, ask for loss, grad norm, LR, loss scale, memory, throughput, and rank tracebacks.
- If only NCCL errors appear, inspect earlier rank-local Python failures first.
