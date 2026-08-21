# Qwen3-30B-A3B optimization ledger

Goal: make Megatron-Core EP4 inference match or exceed the vLLM DP4+EP
throughput on one OCI 4×GB200 node without correctness regressions.

This ledger starts empty. Append every experiment, including failures and
regressions. Never edit an earlier result after it is recorded — supersede it
with a new row.

## The merge stack

**Accepted changes form a single linear stack, not a set of independent PRs.**
`main` is used exactly once, to cut the branch for stack position 1. Every
accepted change after that is branched off the previous accepted change, whether
or not the two are logically related.

```
main ──● (baselines measured here)
       └──● 01  ──● 02  ──● 03  ──● …
```

The reviewer then merges strictly in ascending `#`. Once `01` is merged, `02` is
rebased onto the new `main` and its PR narrows to only its own diff, so each
review is one mechanism at a time even though the work was done cumulatively.

| # | ID | Mechanism | Branch | Cut from | PR | PR state | Marginal gain | Cumulative tok/s | Kill switch |
|---|---|---|---|---|---|---|---:|---:|---|
| — | — | _nothing accepted yet; position 1 is cut from `main`_ | — | — | — | — | — | — | — |

**Merge sequence is the `#` column, ascending. No other order is valid.**

### Rules for this table

- A row appears **only** after the change passed its A/B on distribution
  separation and its correctness gate. Nothing provisional, nothing planned.
- `#` is assigned when the row is created and never reused or reordered. If a
  change is later reverted, strike its row through and leave the number retired —
  renumbering silently invalidates every `Cut from` below it.
- `Cut from` is `main` for `#1` and the branch of `#N-1` for every other row. This
  column is what makes the stack reconstructable after the fact; fill it even
  when it looks redundant.
- `Marginal gain` is the percentage measured against the same-session OFF arm
  **on top of everything below it in the stack** — not against `MCORE-BASELINE`.
- `Cumulative tok/s` is the absolute throughput with all rows up to and including
  this one enabled. Marginal gains do not sum to the cumulative gain, and the
  gap between them is real information: two changes attacking the same kernel
  overlap, so the second one's marginal gain is smaller than it would have been
  alone. Record both and do not reconcile them by arithmetic.
- `PR state` is draft, open, merged, or closed. Refresh it whenever you touch the
  ledger; a stale "draft" against a merged PR is what this column exists to catch.

### Rebase discipline as the stack merges

After `#N` merges, every row above it must be replayed onto the new `main` with
its parent's commits dropped. A plain `git rebase main` will try to reapply the
already-merged commits and conflict, particularly when the parent was
squash-merged upstream:

```bash
git fetch origin main
git rebase --onto origin/main <parent-branch> <this-branch>
git push --force-with-lease <fork> <this-branch>
```

Force-push only to your own fork's stack branches. If a row is rejected upstream
rather than merged, the rows above it are rebased with `--onto` past it in exactly
the same way, and its row here is struck through with the reason.

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
2. Record `MCORE-BASELINE` with Nsight Systems, at `main`.
3. Compute the absolute and percentage gap.
4. Only then modify Megatron-Core.

| ID | Engine | Throughput | Avg latency | TPOT | Job / node | Nsight trace | Status |
|---|---|---:|---:|---:|---|---|---|
| VLLM-BASELINE | vLLM DP4+EP | — | — | — | — | — | not run |
| MCORE-BASELINE | mcore EP4/TP1 | — | — | — | — | — | not run |

Both baselines must be run with no optimization flags enabled, and the gap
computed, before any code change. These are the only measurements taken at
`main`; every later number is measured on a stack branch.

## Experiment index

Every experiment, accepted or not. `Measured on` records the stack position the
experiment was built on, without which its delta cannot be interpreted.

| ID | Date | Hypothesis | Changed files / flags | Measured on | Throughput | Marginal delta | Correctness | Job / run | PR | Conclusion |
|---|---|---|---|---|---:|---:|---|---|---|---|
| — | — | _no experiments recorded yet_ | — | — | — | — | — | — | — | — |

## Optimization rules

1. Profile and classify before proposing a code change.
2. Quantify the lever's ceiling before building it; record gates that come out
   negative.
3. Change one performance variable at a time.
4. Preserve the fixed protocol.
5. Build every new experiment on the **stack tip** — all accepted changes
   applied — never on `main`. `main` is only for the two baselines.
6. A/B in one allocation, arms back to back and alternating, with the stack below
   the change held on in both arms; accept on distribution separation, not mean
   delta.
7. Validate correctness before accepting throughput.
8. Revert regressions or correctness failures.
9. Record the result — including rejections, with their root cause — before
   beginning another experiment.
10. Each accepted change gets its own draft PR, branched off its predecessor and
    recorded as a new row in *The merge stack* with its `#`, branch, `Cut from`,
    and PR link. An accepted change missing from that table is not recorded,
    however thorough its detailed write-up is.
11. Stop when mcore meets or exceeds `VLLM-BASELINE`, then rerun both baselines
    once to confirm parity under identical conditions.
