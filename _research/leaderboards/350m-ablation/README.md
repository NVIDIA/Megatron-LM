# 350M Ablation Leaderboard

Fast optimizer-ablation track on 350M dense Transformer++ (24L / 1024H /
2560F / 16h / 4kv GQA), 1B tokens on ClimbMix, GBS=128 (524K tokens/step),
seq_len=4096, 1 GH200 node, ~30 min/run, WSD schedule.

All entries share architecture, data, schedule, seed (42); only the
optimizer block differs. W&B project:
[megatron-lm-research-baseline](https://wandb.ai/ischlag/megatron-lm-research-baseline).

The `commit` column records the git tag (and short SHA) of the
megatron-lm-research-baseline state at which each row was last executed,
so results stay reproducible across upstream syncs. Tag list:
[`baseline-*`](https://github.com/ischlag/megatron-lm-research-baseline/tags).

The `entry` slug is the stable identifier for each row (the sbatch
filename minus the numeric prefix). `parent` references the entry this
one builds on; `change` is the one-line delta. `rank` is just current
ordering and shifts as new entries land, so it is not safe as a
reference. Roots have an empty `parent`.

| rank | entry | parent | change | optimizer | matrix LR | final | min | sbatch | wandb | commit |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | aurora-qkn | aurora-lr1e-2 | add `--qk-layernorm` (RMSNorm on Q, K) | Aurora + QK norm | 1e-2 | **2.185** | **2.025** | [`09-aurora-qkn.sbatch`](runs/09-aurora-qkn.sbatch) | [s5e3c4bm](https://wandb.ai/ischlag/megatron-lm-research-baseline/runs/s5e3c4bm) | [`f3816b24e`](https://github.com/ischlag/megatron-lm-research-baseline/tree/f3816b24e) |
| 2 | aurora-lr1e-2 | normuon-lr3.6e-4 | Aurora polar (Tilde): row-uniform Stiefel; matrix LR rescaled to 1e-2 | Aurora | 1e-2 | 2.200 | 2.038 | [`08-aurora-lr1e-2.sbatch`](runs/08-aurora-lr1e-2.sbatch) | [czxm4be0](https://wandb.ai/ischlag/megatron-lm-research-baseline/runs/czxm4be0) | [`c51a272dd`](https://github.com/ischlag/megatron-lm-research-baseline/tree/c51a272dd) |
| 3 | normuon-lr3.6e-4 | muon-kj-lr2e-2 | adaptive_muon (NorMuon per-row 2nd moment); LR rescaled to 3.6e-4 | NorMuon | 3.6e-4 | 2.224 | 2.061 | [`01-normuon-lr3.6e-4.sbatch`](runs/01-normuon-lr3.6e-4.sbatch) | [0l47egjv](https://wandb.ai/ischlag/megatron-lm-research-baseline/runs/0l47egjv) | [`baseline-2026-05-08-0b2385c`](https://github.com/ischlag/megatron-lm-research-baseline/tree/baseline-2026-05-08-0b2385c) |
| 4 | muon-kj-lr2e-2 |  | root: plain Muon, Keller Jordan recipe (`shape_scaling`, LR 2e-2) | Muon (shape_scaling) | 2e-2 | 2.226 | 2.068 | [`03-muon-kj-lr2e-2.sbatch`](runs/03-muon-kj-lr2e-2.sbatch) | [f7uvsbai](https://wandb.ai/ischlag/megatron-lm-research-baseline/runs/f7uvsbai) | [`baseline-2026-05-08-a76e5bc`](https://github.com/ischlag/megatron-lm-research-baseline/tree/baseline-2026-05-08-a76e5bc) |
| 5 | normuon-lr2e-4 | normuon-lr3.6e-4 | LR 3.6e-4 -> 2e-4 | NorMuon | 2e-4 | 2.227 | 2.063 | [`02-normuon-lr2e-4.sbatch`](runs/02-normuon-lr2e-4.sbatch) | [heevf919](https://wandb.ai/ischlag/megatron-lm-research-baseline/runs/heevf919) | [`baseline-2026-05-08-a76e5bc`](https://github.com/ischlag/megatron-lm-research-baseline/tree/baseline-2026-05-08-a76e5bc) |
| 6 | normuon-fp8 | normuon-lr3.6e-4 | add `--fp8-format e4m3 --fp8-recipe blockwise` (DeepSeek-V3) | NorMuon (FP8) | 3.6e-4 | 2.229 | 2.065 | [`07-normuon-fp8.sbatch`](runs/07-normuon-fp8.sbatch) | [7j84uy3j](https://wandb.ai/ischlag/megatron-lm-research-baseline/runs/7j84uy3j) | [`598cbe616`](https://github.com/ischlag/megatron-lm-research-baseline/tree/598cbe616) |
| 7 | normuon-lr6e-4 | normuon-lr3.6e-4 | LR 3.6e-4 -> 6e-4 | NorMuon | 6e-4 | 2.244 | 2.081 | [`04-normuon-lr6e-4.sbatch`](runs/04-normuon-lr6e-4.sbatch) | [zu96uyts](https://wandb.ai/ischlag/megatron-lm-research-baseline/runs/zu96uyts) | [`baseline-2026-05-08-a76e5bc`](https://github.com/ischlag/megatron-lm-research-baseline/tree/baseline-2026-05-08-a76e5bc) |
| 8 | adamw-lr1e-3 |  | root: AdamW baseline (decoupled WD), LR 1e-3 | AdamW | 1e-3 | 2.316 | 2.154 | [`05-adamw-lr1e-3.sbatch`](runs/05-adamw-lr1e-3.sbatch) | [6qecfvwc](https://wandb.ai/ischlag/megatron-lm-research-baseline/runs/6qecfvwc) | [`baseline-2026-05-08-a76e5bc`](https://github.com/ischlag/megatron-lm-research-baseline/tree/baseline-2026-05-08-a76e5bc) |
| 9 | adamw-lr5e-4 | adamw-lr1e-3 | LR 1e-3 -> 5e-4 | AdamW | 5e-4 | 2.363 | 2.199 | [`06-adamw-lr5e-4.sbatch`](runs/06-adamw-lr5e-4.sbatch) | [zzywif5m](https://wandb.ai/ischlag/megatron-lm-research-baseline/runs/zzywif5m) | [`baseline-2026-05-08-a76e5bc`](https://github.com/ischlag/megatron-lm-research-baseline/tree/baseline-2026-05-08-a76e5bc) |

## Notes on entries

- **`aurora-qkn`**: per-head RMSNorm on Q and K (TENorm under
  `--normalization RMSNorm`). WD stays off on the new q/k norm gains
  (default skip-WD-on-shape-1 rule); a paired ablation with
  `--apply-wd-to-qk-layernorm` landed at final 2.187 / min 2.025,
  indistinguishable from no-WD-on-gains.
- **`aurora-lr1e-2`**: Aurora (Tilde Research, 2026), leverage-uniform
  polar update from
  [tilderesearch.com/blog/aurora](https://blog.tilderesearch.com/blog/aurora).
  Matrix LR 1e-2, scalar LR 1.5e-3, spectral scale, Nesterov + momentum
  0.95, polar bf16, pp_iterations=2, pp_beta=0.5, 12-step simple-quintic
  Newton-Schulz. Aurora is strikingly LR-robust: a sweep at 5e-3 / 8e-3 /
  2e-2 / 3e-2 all land within 0.05 nats. At matched LR, Aurora wins by
  > 2 nats over NorMuon, which diverges (final > 4.5) outside its narrow
  tuned window.
- **`normuon-lr3.6e-4`**: the recommended NorMuon recipe. Matrix LR
  3.6e-4 (NorMuon paper default for 124M), scalar LR 1.5e-3 on
  embeddings / norms / biases, scalar WD 0, spectral scale mode,
  Nesterov + momentum 0.95, QKV split on. Two flag-config variants of
  the same LR (with/without `--use-distributed-optimizer` and
  `--overlap-grad-reduce`) landed at final 2.219 and 2.224 (within
  numerical noise); dropped from table.
- **`muon-kj-lr2e-2`**: Keller Jordan recipe, plain Muon with
  `shape_scaling` and LR 2e-2. Works competitively out of the box on
  350M.
- **`normuon-fp8`**: FP8 e4m3 blockwise (DeepSeek-V3 recipe,
  Hopper-compatible). FP8 matmuls only (params and master weights stay
  bf16; no `--fp8-param-gather`, no `fp8_param`). Final +0.005 vs bf16
  NorMuon, within numerical noise. Steady throughput ~313 TFLOP/s/GPU.
- **AdamW entries** use `--optimizer adam`, which in Megatron is
  decoupled-WD AdamW.
- `--overlap-param-gather` is on for AdamW rows and off for Muon /
  Aurora rows (it corrupts Newton-Schulz; see the main README).
