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
env-var branching, no harness dependencies. The header of each file
records the rank at entry time, the git sha the run was executed at,
the W&B URL, and the final + min training loss.

To reproduce an entry bitwise: `git checkout <sha>` then
`sbatch leaderboards/<size>/runs/NN-<name>.sbatch`.

## Promoting a run to a leaderboard

When a run wins (or is otherwise worth tracking), add it as an entry:

1. Copy the executed sbatch into `leaderboards/<size>/runs/`. Pick the
   next free numeric prefix (`NN-`); the rest of the filename is the
   stable **slug** for that entry (e.g. `08-aurora-lr1e-2.sbatch` has
   slug `aurora-lr1e-2`).
2. Pin every hparam in the file: no env-var branching, no harness
   dependencies. The header should carry rank, sha, W&B URL, final +
   min loss.
3. Add a row to the leaderboard's `README.md` table with these stable
   columns:
   - `entry`: the slug from step 1.
   - `parent`: the slug of the entry this run builds on, or empty if
     it's a new root.
   - `change`: a one-line delta vs the parent (e.g. `LR 3.6e-4 -> 6e-4`,
     `add --qk-layernorm (RMSNorm on Q, K)`, `FP8 e4m3 blockwise
     (DeepSeek-V3)`). Roots use this column to describe the recipe
     family.
4. Reorder the table by `min loss` (or whatever metric matters for
   that leaderboard) and update the `rank` column. Rank is unstable
   over time; the slug + parent + change triple is the durable record
   of lineage.

## Sizes

| leaderboard | tokens | nodes | purpose |
| --- | ---: | ---: | --- |
| [`350m-ablation`](350m-ablation/README.md) | 1B | 1 | quick sweeps, optimizer ablations (~30 min/run) |
| [`350m`](350m/README.md) | 15B | 1 | 350M full baseline (~8 h/run) |
| [`760m`](760m/README.md) | 30B | 4 | 760M full baseline (~7 h/run) |
| [`1.3b`](1.3b/README.md) | 100B | 8 | 1.3B full baseline (~19 h/run) |
| [`2.7b`](2.7b/README.md) | 300B | 16 | 2.7B full baseline (~60 h/run) |

## Related folders

- `_research/launch/` — launchable sbatches: baseline full-runs
  (`transformer-pp-<size>-<adamw|muon>.sbatch`) plus a 1B-token quick
  reference (`-ablation.sbatch`). All hparams pinned. To try a variant,
  copy an existing file and edit; once it wins, snapshot it into the
  matching leaderboard under `leaderboards/<size>/runs/`.
