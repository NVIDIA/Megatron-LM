# Agents Guide

Technical orientation for AI assistants working in this repo. Read this
once on first entry; use it as a reference, not a script.

## What this is

A fork of [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) with
a `_research/` scaffold for optimizer and architecture ablations on dense
and MoE transformers. Upstream is untouched except flag-gated edits in
`megatron/core/optimizer/` and `megatron/training/arguments.py`.

## Repo layout

| path | contents |
| --- | --- |
| `_research/launch/` | launchable sbatches: baselines (`transformer-pp-<size>-<adamw\|muon>.sbatch`) and ablation harnesses (`-ablation.sbatch`) with `SWEEP_*` env-var branching |
| `_research/leaderboards/<size>/README.md` | ranked result table + W&B links for that size |
| `_research/leaderboards/<size>/runs/NN-*.sbatch` | frozen, self-contained sbatches (no env vars, hparams pinned) |
| `_research/logging_patch/` | JSONL + wandb telemetry, activated by a two-line hook in `pretrain_gpt.py` |
| `_research/data/` | tokenizer files (GPT-2 BPE) |
| `_research/results/` | gitignored run outputs (logs, tensorboard, wandb) |
| `megatron/core/optimizer/emerging_optimizers.py` | Muon / AdaMuon / NorMuon glue |
| `megatron/core/optimizer/ademamix.py` | AdEMAMix port |
| `megatron/core/activations.py` | xIELU activation |
| `megatron/training/arguments.py` | CLI flags (upstream + our additions) |
| `megatron/core/optimizer/optimizer_config.py` | `OptimizerConfig` fields (mirror each new CLI flag here) |
| `pretrain_gpt.py` | entrypoint + logging-patch hook |

Everything not listed is upstream Megatron — treat as read-only unless
you're adding a flag-gated code path.

## How experiments flow

```
launch/<size>-ablation.sbatch (with SWEEP_* env vars)
        ↓  a config wins
snapshot a self-contained sbatch into
leaderboards/<size>/runs/NN-*.sbatch  +  add a row to that README.md
```

Every leaderboard sbatch header carries the git sha, W&B URL, and final +
min loss at time of entry. Bitwise reproduction:

```
git checkout <sha> && sbatch _research/leaderboards/<size>/runs/NN-*.sbatch
```

## Good practices

- **Flag-gate everything new**, default off. Add the CLI flag in
  `arguments.py` and the field in `optimizer_config.py`; don't change
  existing default behaviour.
- **Don't remove the logging-patch hook** in `pretrain_gpt.py` — every
  run downstream of it depends on the JSONL/wandb telemetry.
- **Muon + `--overlap-param-gather` is broken** (the per-bucket async
  gather corrupts Newton-Schulz). Keep `--use-distributed-optimizer` and
  `--overlap-grad-reduce`, drop `--overlap-param-gather` for any
  `--optimizer muon|adaptive_muon`. The ablation sweep harness does
  this automatically.
- **No SLURM job without a stopping condition** (`--train-iters`,
  `--train-samples`, or `--exit-duration-in-mins`).
- **Cluster specifics are hardcoded** in `#SBATCH` headers. To port to a
  new site, edit the header block and `--data-path`; the rest is
  portable.
- **Commits are not signed with AI attributions**. Don't push to `main`;
  open a PR.
- **Sync with upstream**: `git fetch upstream && git merge upstream/main`.
  Expect conflicts only in the files listed above.

## When to pause and ask the human

- Adding cluster-wide or repo-wide dependencies.
- Launching jobs above whatever node cap the human has set.
- Merging to `main` or any action with blast radius beyond this repo.
- Touching upstream source outside the flag-gated exceptions.
- Deleting results, branches, or worktrees you didn't create.
