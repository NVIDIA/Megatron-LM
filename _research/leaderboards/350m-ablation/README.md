# 350M Ablation Leaderboard

Fast optimizer-ablation track on 350M dense Transformer++ (24L / 1024H /
2560F / 16h / 4kv GQA), 1B tokens on ClimbMix, GBS=128 (524K tokens/step),
seq_len=4096, 1 GH200 node, ~30 min/run, WSD schedule.

All entries share architecture, data, schedule, seed (42); only the
optimizer block differs. W&B project:
[megatron-lm-research-baseline](https://wandb.ai/ischlag/megatron-lm-research-baseline).

| rank | optimizer | matrix LR | final loss | min loss | sbatch | wandb |
| ---: | --- | ---: | ---: | ---: | --- | --- |
| 1 | NorMuon | 3.6e-4 | **2.221** | **2.058** | [`01-normuon-lr3.6e-4.sbatch`](runs/01-normuon-lr3.6e-4.sbatch) | [ispegm6g](https://wandb.ai/ischlag/megatron-lm-research-baseline/runs/ispegm6g) |
| 2 | Muon (shape_scaling) | 2e-2 | 2.226 | 2.068 | [`03-muon-kj-lr2e-2.sbatch`](runs/03-muon-kj-lr2e-2.sbatch) | [f7uvsbai](https://wandb.ai/ischlag/megatron-lm-research-baseline/runs/f7uvsbai) |
| 3 | NorMuon | 2e-4 | 2.227 | 2.063 | [`02-normuon-lr2e-4.sbatch`](runs/02-normuon-lr2e-4.sbatch) | [heevf919](https://wandb.ai/ischlag/megatron-lm-research-baseline/runs/heevf919) |
| 4 | NorMuon | 6e-4 | 2.244 | 2.081 | [`04-normuon-lr6e-4.sbatch`](runs/04-normuon-lr6e-4.sbatch) | [zu96uyts](https://wandb.ai/ischlag/megatron-lm-research-baseline/runs/zu96uyts) |
| 5 | AdamW | 1e-3 | 2.316 | 2.154 | [`05-adamw-lr1e-3.sbatch`](runs/05-adamw-lr1e-3.sbatch) | [6qecfvwc](https://wandb.ai/ischlag/megatron-lm-research-baseline/runs/6qecfvwc) |
| 6 | AdamW | 5e-4 | 2.363 | 2.199 | [`06-adamw-lr5e-4.sbatch`](runs/06-adamw-lr5e-4.sbatch) | [zzywif5m](https://wandb.ai/ischlag/megatron-lm-research-baseline/runs/zzywif5m) |

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
