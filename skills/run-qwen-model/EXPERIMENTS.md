# Qwen3-30B-A3B optimization ledger

Goal: make Megatron-Core EP4 inference match or exceed the vLLM DP4+EP
throughput on one OCI 4×GB200 node without correctness regressions.

This ledger starts empty. Append every experiment, including failures and
regressions. Never edit an earlier result after it is recorded — supersede it
with a new row.

## Fixed protocol

| Setting | Value |
|---|---|
| Cluster | OCI `oci-hsg` |
| Hardware | 1 node, 4×GB200 |
| Model | Qwen3-30B-A3B, BF16 |
| Dataset / batch | gsm8k / 256 |
| Throughput workload | OSL1024, 2 warmups, 5 timed iterations |
| Profile workload | OSL128, one short warmup request, one timed request |
| mcore layout | TP=1, PP=1, EP=4, ETP=1 |
| vLLM layout | TP=1, DP=4, `--enable-expert-parallel` |
| Correctness gate | Fixed temperature-0 coherence prompts plus benchmark success |
| Primary metric | Throughput (output tokens/s) |
| Secondary metrics | Average latency and TPOT |

Do not compare results when hardware, checkpoint, batch size, output length,
parallelism, or warmup/timed counts differ.

## Baselines

Nothing recorded yet. Baseline order is mandatory:

1. Record `VLLM-BASELINE` with Nsight Systems.
2. Record `MCORE-BASELINE` with Nsight Systems.
3. Compute the absolute and percentage gap.
4. Only then modify Megatron-Core.

| ID | Engine | Throughput | Avg latency | TPOT | Job / node | Nsight trace | Status |
|---|---|---:|---:|---:|---|---|---|
| VLLM-BASELINE | vLLM DP4+EP | — | — | — | — | — | not run |
| MCORE-BASELINE | mcore EP4/TP1 | — | — | — | — | — | not run |

Both baselines must be run with no optimization flags enabled, and the gap
computed, before any code change.

## Experiment index

| ID | Date | Hypothesis | Changed files / flags | Throughput | Delta vs baseline | Correctness | Job / run | MR | Conclusion |
|---|---|---|---|---:|---:|---|---|---|---|
| — | — | _no experiments recorded yet_ | — | — | — | — | — | — | — |

## Optimization rules

1. Profile and classify before proposing a code change.
2. Quantify the lever's ceiling before building it; record gates that come out
   negative.
3. Change one performance variable at a time.
4. Preserve the fixed protocol.
5. A/B in one allocation, arms back to back and alternating; accept on
   distribution separation, not mean delta.
6. Validate correctness before accepting throughput.
7. Revert regressions or correctness failures.
8. Record the result — including rejections, with their root cause — before
   beginning another experiment.
9. Open a separate draft MR for each accepted change, and link it from its row.
10. Stop when mcore meets or exceeds `VLLM-BASELINE`, then rerun both baselines
    once to confirm parity under identical conditions.
