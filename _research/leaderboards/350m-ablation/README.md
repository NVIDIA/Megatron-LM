# 350M Ablation Leaderboard

Fast optimizer-ablation track on 350M dense Transformer++ (24L / 1024H /
2560F / 16h / 4kv GQA), 1B tokens on ClimbMix, GBS=128 (524K tokens/step),
seq_len=4096, 1 GH200 node, ~30 min/run, WSD schedule.

All entries share architecture, data, schedule, seed (42); only the
optimizer block differs. W&B project:
[megatron-lm-research-baseline](https://wandb.ai/ischlag/megatron-lm-research-baseline).

| rank | optimizer | matrix LR | final loss | min loss | sbatch |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | NorMuon | 3.6e-4 | **2.225** | **2.061** | [`01-normuon-lr3.6e-4.sbatch`](runs/01-normuon-lr3.6e-4.sbatch) |
| 2 | NorMuon | 2e-4 | 2.226 | 2.063 | [`02-normuon-lr2e-4.sbatch`](runs/02-normuon-lr2e-4.sbatch) |
| 3 | Muon (shape_scaling) | 2e-2 | 2.232 | 2.069 | [`03-muon-kj-lr2e-2.sbatch`](runs/03-muon-kj-lr2e-2.sbatch) |
| 4 | NorMuon | 6e-4 | 2.244 | 2.082 | [`04-normuon-lr6e-4.sbatch`](runs/04-normuon-lr6e-4.sbatch) |
| 5 | AdamW | 1e-3 | 2.314 | 2.148 | [`05-adamw-lr1e-3.sbatch`](runs/05-adamw-lr1e-3.sbatch) |
| 6 | AdamW | 5e-4 | 2.366 | 2.204 | [`06-adamw-lr5e-4.sbatch`](runs/06-adamw-lr5e-4.sbatch) |

*W&B links pending — will be added once the currently running repro jobs finish.*

## Notes on entries

- **Rank 1** is the recommended NorMuon recipe: matrix LR 3.6e-4 (NorMuon
  paper default for 124M), scalar LR 1.5e-3 on embeddings / norms / biases,
  scalar WD 0, spectral scale mode, Nesterov + momentum 0.95, QKV split on.
  Two additional flag-config variants of the same LR — with/without
  `--use-distributed-optimizer` and `--overlap-grad-reduce` — landed at
  final loss 2.219 and 2.224 (within numerical noise). Dropped from table.
- **Rank 3** is the Keller Jordan recipe: plain Muon, `shape_scaling`
  scale mode, LR 2e-2. Works competitively out of the box on the 350M.
- **AdamW ranks** use `--optimizer adam` which in Megatron is decoupled-WD
  AdamW; the naming in the table reflects that.
- `--overlap-param-gather` is **on** for AdamW rows and **off** for Muon
  rows (it corrupts Muon's Newton-Schulz, see the main README).
