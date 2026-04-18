# Leaderboards

Per-size ranked lists of training runs on this fork of Megatron-LM. Each
entry points to a self-contained, re-runnable `.sbatch` and a W&B run.

## Structure

```
leaderboards/
  <size>/
    README.md       # ranked table + wandb links + one-line description
    runs/
      01-*.sbatch   # self-contained, no env-var branching, all hparams pinned
      02-*.sbatch
      ...
```

A self-contained sbatch runs with `sbatch runs/<NN>-<name>.sbatch` — no
`SWEEP_*` env vars, no harness dependencies. The header of each file
records the rank at entry time, the git sha the run was executed at,
the W&B URL, and the final + min training loss.

To reproduce an entry bitwise: `git checkout <sha>` then
`sbatch leaderboards/<size>/runs/NN-<name>.sbatch`.

## Sizes

| leaderboard | tokens | nodes | purpose |
| --- | ---: | ---: | --- |
| [`350m-ablation`](350m-ablation/README.md) | 1B | 1 | quick sweeps, optimizer ablations (~30 min/run) |
| [`350m`](350m/README.md) | 15B | 1 | 350M full baseline (~8 h/run) |
| [`760m`](760m/README.md) | 30B | 4 | 760M full baseline (~7 h/run) |
| [`1.3b`](1.3b/README.md) | 100B | 8 | 1.3B full baseline (~19 h/run) |
| [`2.7b`](2.7b/README.md) | 300B | 16 | 2.7B full baseline (~60 h/run) |

## Related folders

- `_research/launch/` — canonical baseline launch scripts (one `-adamw`
  and one `-muon` per size). These are the *recommended* configs.
- `_research/sweeps/` — sweep harnesses with `SWEEP_*` env-var
  branching. Use these to explore new ablation axes; once a variant
  wins, snapshot a self-contained copy into the matching leaderboard.
