# 350M Ablation Leaderboard

Fast optimizer-ablation track on 350M dense Transformer++ (24L / 1024H /
2560F / 16h / 4kv GQA), 1B tokens on ClimbMix, GBS=128 (524K tokens/step),
seq_len=4096, 1 GH200 node, ~30 min/run, WSD schedule.

Most entries share architecture, data, schedule, seed (42); the optimizer
block differs. Architecture variants (linear-attention layers in place of
softmax attention) are flagged in the variant column. W&B project:
[megatron-lm-research-baseline](https://wandb.ai/ischlag/megatron-lm-research-baseline).

The `commit` column records the git tag (and short SHA) of the
megatron-lm-research-baseline state at which each row was last executed,
so results stay reproducible across upstream syncs. Tag list:
[`baseline-*`](https://github.com/ischlag/megatron-lm-research-baseline/tags).

| rank | optimizer | matrix LR | final loss | min loss | sbatch | wandb | commit |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | Aurora | 1e-2 | **2.200** | **2.038** | [`08-aurora-lr1e-2.sbatch`](runs/08-aurora-lr1e-2.sbatch) | [czxm4be0](https://wandb.ai/ischlag/megatron-lm-research-baseline/runs/czxm4be0) | [`c51a272dd`](https://github.com/ischlag/megatron-lm-research-baseline/tree/c51a272dd) |
| 2 | NorMuon | 3.6e-4 | 2.224 | 2.061 | [`01-normuon-lr3.6e-4.sbatch`](runs/01-normuon-lr3.6e-4.sbatch) | [0l47egjv](https://wandb.ai/ischlag/megatron-lm-research-baseline/runs/0l47egjv) | [`baseline-2026-05-08-0b2385c`](https://github.com/ischlag/megatron-lm-research-baseline/tree/baseline-2026-05-08-0b2385c) |
| 3 | Muon (shape_scaling) | 2e-2 | 2.226 | 2.068 | [`03-muon-kj-lr2e-2.sbatch`](runs/03-muon-kj-lr2e-2.sbatch) | [f7uvsbai](https://wandb.ai/ischlag/megatron-lm-research-baseline/runs/f7uvsbai) | [`baseline-2026-05-08-a76e5bc`](https://github.com/ischlag/megatron-lm-research-baseline/tree/baseline-2026-05-08-a76e5bc) |
| 4 | NorMuon | 2e-4 | 2.227 | 2.063 | [`02-normuon-lr2e-4.sbatch`](runs/02-normuon-lr2e-4.sbatch) | [heevf919](https://wandb.ai/ischlag/megatron-lm-research-baseline/runs/heevf919) | [`baseline-2026-05-08-a76e5bc`](https://github.com/ischlag/megatron-lm-research-baseline/tree/baseline-2026-05-08-a76e5bc) |
| 5 | NorMuon (FP8) | 3.6e-4 | 2.229 | 2.065 | [`07-normuon-fp8.sbatch`](runs/07-normuon-fp8.sbatch) | [7j84uy3j](https://wandb.ai/ischlag/megatron-lm-research-baseline/runs/7j84uy3j) | [`598cbe616`](https://github.com/ischlag/megatron-lm-research-baseline/tree/598cbe616) |
| 6 | NorMuon | 6e-4 | 2.244 | 2.081 | [`04-normuon-lr6e-4.sbatch`](runs/04-normuon-lr6e-4.sbatch) | [zu96uyts](https://wandb.ai/ischlag/megatron-lm-research-baseline/runs/zu96uyts) | [`baseline-2026-05-08-a76e5bc`](https://github.com/ischlag/megatron-lm-research-baseline/tree/baseline-2026-05-08-a76e5bc) |
| 7 | AdamW | 1e-3 | 2.316 | 2.154 | [`05-adamw-lr1e-3.sbatch`](runs/05-adamw-lr1e-3.sbatch) | [6qecfvwc](https://wandb.ai/ischlag/megatron-lm-research-baseline/runs/6qecfvwc) | [`baseline-2026-05-08-a76e5bc`](https://github.com/ischlag/megatron-lm-research-baseline/tree/baseline-2026-05-08-a76e5bc) |
| 8 | AdamW | 5e-4 | 2.363 | 2.199 | [`06-adamw-lr5e-4.sbatch`](runs/06-adamw-lr5e-4.sbatch) | [zzywif5m](https://wandb.ai/ischlag/megatron-lm-research-baseline/runs/zzywif5m) | [`baseline-2026-05-08-a76e5bc`](https://github.com/ischlag/megatron-lm-research-baseline/tree/baseline-2026-05-08-a76e5bc) |
| 9 | NorMuon (DN+carry-v2+cap200) | 3.6e-4 | 2.207 | 2.045 | [`09-deltanet-perchgate-carry-v2-cap200.sbatch`](runs/09-deltanet-perchgate-carry-v2-cap200.sbatch) | (dev) | [`91920d23d`](https://github.com/ischlag/megatron-lm-research-baseline/tree/91920d23d) (feat/deltanet-variants) |

## Notes on entries

- **Rank 1** is Aurora (Tilde Research, 2026): leverage-uniform polar
  update from [tilderesearch.com/blog/aurora](https://blog.tilderesearch.com/blog/aurora).
  Matrix LR 1e-2, scalar LR 1.5e-3, spectral scale, Nesterov + momentum
  0.95, polar bf16, pp_iterations=2, pp_beta=0.5, 12-step simple-quintic
  Newton-Schulz. Beats the prior NorMuon leader by 0.024 nats final.
  Aurora is also strikingly LR-robust: a sweep at 5e-3 / 8e-3 / 2e-2 /
  3e-2 all land within 0.05 nats of rank 1, while NorMuon at 8e-3 or
  1e-2 diverges (final loss > 4.5). At matched LR, Aurora wins by
  > 2 nats.
- **Rank 2** is the recommended NorMuon recipe: matrix LR 3.6e-4 (NorMuon
  paper default for 124M), scalar LR 1.5e-3 on embeddings / norms / biases,
  scalar WD 0, spectral scale mode, Nesterov + momentum 0.95, QKV split on.
  Two additional flag-config variants of the same LR — with/without
  `--use-distributed-optimizer` and `--overlap-grad-reduce` — landed at
  final loss 2.219 and 2.224 (within numerical noise). Dropped from table.
- **Rank 3** is the Keller Jordan recipe: plain Muon, `shape_scaling`
  scale mode, LR 2e-2. Works competitively out of the box on the 350M.
- **Rank 5** is the FP8 variant of rank 2 — same NorMuon recipe, but
  with `--fp8-format e4m3 --fp8-recipe blockwise` (DeepSeek-V3 recipe,
  Hopper-compatible). FP8 matmuls only (params and master weights stay
  bf16; no `--fp8-param-gather`, no `fp8_param`). Final loss +0.005 vs
  bf16 NorMuon, within numerical noise. Steady throughput ~313
  TFLOP/s/GPU.
- **AdamW ranks** use `--optimizer adam` which in Megatron is decoupled-WD
  AdamW; the naming in the table reflects that.
- `--overlap-param-gather` is **on** for AdamW rows and **off** for Muon /
  Aurora rows (it corrupts Newton-Schulz; see the main README).
- **Rank 9** is the first architecture variant: hybrid 3:1 (Schlag DeltaNet
  + softmax attention every 4th layer), per-channel output gate, full
  per-batch carry-v2 of the recurrent state across batches, and a hard
  Frobenius cap of 200 on the per-element carried state. NorMuon optimizer
  to match the canonical recipe. Cap=200 binds late (~step 950) and keeps
  the carried state norm bounded for length-gen safety without measurably
  hurting iso-token loss vs the no-cap variant. Runs on
  `feat/deltanet-variants` branch (DeltaNet flag-gated path); will be
  re-tagged on main once the branch lands.
