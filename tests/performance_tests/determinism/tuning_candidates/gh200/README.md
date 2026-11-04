# GH200 sm90 tuning candidate

This candidate is **not qualified for promotion to the packaged defaults**.
It is kept outside `megatron/core/tuning/tables` so normal deterministic-mode
runs do not select it automatically.

The finite sweep covers 63 MLA RoPE, mHC and Mamba SSD shapes at 2048, 8192 and
32768 tokens, reaching 25 multi-config kernels. The table has 42 entries.
MLA RoPE and mHC choose a wildcard minimizing worst-case slowdown and per-key
overrides when they improve that worst case by more than 5%; SSD chooses one
minimax configuration per key. Invalid/nonpositive measurements are excluded,
and every compared shape must have a valid measurement for an eligible
candidate. Raw observations, the exact builder and replay checks are in the
[reproduction record](https://github.com/0z5a/Megatron-LM/tree/hopper-validation-evidence/validation/hopper-20261020).

The source field and version metadata bind the measured GH200 environment.
A separate H100 recording used a different runtime and is not merged here.
The 25-kernel scope is not an inventory of every kernel in Megatron-LM.

## Qualification results

All 63 component shapes passed two independent processes with two ordered
replays each: output, input-gradient and selected-configuration digests were
equal. Pinning without ordered reductions failed six SSD gradient cases.

A subsequent synthetic GPT/HybridModel forward/backward/SGD comparison used
three model families, three modes (auto, this table, cheapest), three processes
per mode, four warmup and ten measured steps, with rotated mode order:

| Family | Table-mode cross-process loss/parameter/gradient digests | Table/auto median-step ratio, paired 95% interval |
| --- | --- | --- |
| MLA | One of three processes differs | 1.120 [0.495, 2.531] |
| mHC | Equal | 1.484 [0.642, 3.429] |
| SSD | Equal | 0.972 [0.697, 1.356] |

All three table-mode configuration digests were stable. The first MLA mismatch
therefore is not explained by a different recorded table choice. Its cause is
not established. The auto SSD control also varies numerically. All process
exits were zero and losses finite; those facts do not erase the digest failure.

The intervals use Student-t on paired log median-step ratios (n=3, df=2).
None meets the preregistered upper-confidence-bound <= 1.05 timing gate.
These small/noisy samples do not establish model performance or convergence.
Retain these results when investigating or recording a replacement candidate.

## Reproduce explicitly

Use the recorded dependency versions and a fitting finite workload. Select
this directory only for a diagnostic run:

```bash
--triton-autotune-mode pinned \
--triton-autotune-table-path tests/performance_tests/determinism/tuning_candidates/gh200 \
--triton-autotune-on-miss error
```

Keep strict deterministic algorithms, `MAMBA_DETERMINISTIC=1`, the required
causal-conv1d behavior and the documented NCCL/cuBLAS/TE environment controls.
For an auto/table/cheapest comparison, use a source tree without a packaged
sm90 table so that the cheapest control actually exercises the fallback.
The candidate should move into the packaged tables only after its model
correctness and performance gates are satisfied with preserved evidence.
