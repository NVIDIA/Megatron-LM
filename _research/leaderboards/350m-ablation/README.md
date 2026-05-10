# 350M Ablation Leaderboard

Fast optimizer-ablation track on 350M dense Transformer++ (24L / 1024H /
2560F / 16h / 4kv GQA), 1B tokens on ClimbMix, GBS=128 (524K tokens/step),
seq_len=4096, 1 GH200 node, ~30 min/run, WSD schedule.

All entries share data, schedule, seed (42), and **active param count
(~355M)** per token; the `change` column carries the diff that produced
the entry. Most rows are dense (active = total = 355M). MoE rows
add total params via routed experts while keeping active fixed
(`topk * moe_ffn_hidden = ffn_hidden = 2560`). Linear-attention rows
swap softmax attention for an SSM/linear-attention variant in 18/24
layers (3:1 hybrid) and keep the dense FFN. W&B project:
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
| 1 | aurora-qkn-moe-32e-tk3-sh1 | aurora-qkn | replace dense FFN with 32 routed experts top-3 + 1 shared expert (all width 640); active 355M, total ~1.73B (DeepSeek-V3-style fine-grained MoE) | Aurora + QK norm + MoE | 1e-2 | **2.098** | **1.934** | [`16-aurora-qkn-moe-32e-tk3-sh1.sbatch`](runs/16-aurora-qkn-moe-32e-tk3-sh1.sbatch) | [5ttumu3u](https://wandb.ai/ischlag/lm-research-baseline-dev/runs/5ttumu3u) | [`c664f35d2`](https://github.com/ischlag/megatron-lm-research-baseline/tree/c664f35d2) |
| 2 | aurora-qkn-xsa | aurora-qkn | add `--exclusive-self-attention` (XSA, GQA-aware + torch.compile) | Aurora + QK norm + XSA | 1e-2 | 2.167 | 2.009 | [`12-aurora-qkn-xsa.sbatch`](runs/12-aurora-qkn-xsa.sbatch) | [mdmnb7mc](https://wandb.ai/ischlag/megatron-lm-research-baseline/runs/mdmnb7mc) | [`435b3cf2d`](https://github.com/ischlag/megatron-lm-research-baseline/tree/435b3cf2d) |
| 3 | aurora-xsa | aurora-lr1e-2 | add `--exclusive-self-attention` (XSA) | Aurora + XSA | 1e-2 | 2.183 | 2.023 | [`13-aurora-xsa.sbatch`](runs/13-aurora-xsa.sbatch) | [w6vimrkm](https://wandb.ai/ischlag/lm-research-baseline-dev/runs/w6vimrkm) | [`e994bb632`](https://github.com/ischlag/megatron-lm-research-baseline/tree/e994bb632) |
| 4 | aurora-qkn | aurora-lr1e-2 | add `--qk-layernorm` (RMSNorm on Q, K) | Aurora + QK norm | 1e-2 | 2.185 | 2.025 | [`09-aurora-qkn.sbatch`](runs/09-aurora-qkn.sbatch) | [s5e3c4bm](https://wandb.ai/ischlag/megatron-lm-research-baseline/runs/s5e3c4bm) | [`f3816b24e`](https://github.com/ischlag/megatron-lm-research-baseline/tree/f3816b24e) |
| 5 | aurora-qkn-fp8 | aurora-qkn | add `--fp8-format e4m3 --fp8-recipe blockwise` (DeepSeek-V3) | Aurora + QK norm (FP8) | 1e-2 | 2.188 | 2.027 | [`11-aurora-qkn-fp8.sbatch`](runs/11-aurora-qkn-fp8.sbatch) | [guwuunre](https://wandb.ai/ischlag/lm-research-baseline-dev/runs/guwuunre) | [`6db3a8f8b`](https://github.com/ischlag/megatron-lm-research-baseline/tree/6db3a8f8b) |
| 6 | aurora-lr1e-2 | normuon-lr3.6e-4 | Aurora polar (Tilde): row-uniform Stiefel; matrix LR rescaled to 1e-2 | Aurora | 1e-2 | 2.200 | 2.038 | [`08-aurora-lr1e-2.sbatch`](runs/08-aurora-lr1e-2.sbatch) | [czxm4be0](https://wandb.ai/ischlag/megatron-lm-research-baseline/runs/czxm4be0) | [`c51a272dd`](https://github.com/ischlag/megatron-lm-research-baseline/tree/c51a272dd) |
| 7 | aurora-qkn-learned-sink | aurora-qkn | add `--softmax-type learnable` (gpt-oss per-head learned sink bias) | Aurora + QK norm + learned sink | 1e-2 | 2.201 | 2.040 | [`15-aurora-qkn-learned-sink.sbatch`](runs/15-aurora-qkn-learned-sink.sbatch) | [9n4abgdu](https://wandb.ai/ischlag/lm-research-baseline-dev/runs/9n4abgdu) | [`435b3cf2d`](https://github.com/ischlag/megatron-lm-research-baseline/tree/435b3cf2d) |
| 8 | aurora-qkn-softmax1 | aurora-qkn | add `--softmax-type off-by-one` (Evan Miller softmax1 sink) | Aurora + QK norm + softmax1 | 1e-2 | 2.203 | 2.027 | [`14-aurora-qkn-softmax1.sbatch`](runs/14-aurora-qkn-softmax1.sbatch) | [tz4vr1ng](https://wandb.ai/ischlag/lm-research-baseline-dev/runs/tz4vr1ng) | [`bd77a2f32`](https://github.com/ischlag/megatron-lm-research-baseline/tree/bd77a2f32) |
| 9 | deltanet-perchgate-carry-v2-cap200 | normuon-lr3.6e-4 | swap 18/24 softmax-attention layers for Schlag DeltaNet (3:1 hybrid) with per-channel output gate, full per-batch carry-v2 of recurrent state, hard Frobenius cap 200 on per-element carried state | NorMuon (DN+carry-v2+cap200) | 3.6e-4 | 2.207 | 2.045 | [`17-deltanet-perchgate-carry-v2-cap200.sbatch`](runs/17-deltanet-perchgate-carry-v2-cap200.sbatch) | (dev) | [`91920d23d`](https://github.com/ischlag/megatron-lm-research-baseline/tree/91920d23d) |
| 10 | normuon-lr3.6e-4 | muon-kj-lr2e-2 | adaptive_muon (NorMuon per-row 2nd moment); LR rescaled to 3.6e-4 | NorMuon | 3.6e-4 | 2.224 | 2.061 | [`01-normuon-lr3.6e-4.sbatch`](runs/01-normuon-lr3.6e-4.sbatch) | [0l47egjv](https://wandb.ai/ischlag/megatron-lm-research-baseline/runs/0l47egjv) | [`baseline-2026-05-08-0b2385c`](https://github.com/ischlag/megatron-lm-research-baseline/tree/baseline-2026-05-08-0b2385c) |
| 11 | muon-kj-lr2e-2 |  | root: plain Muon, Keller Jordan recipe (`shape_scaling`, LR 2e-2) | Muon (shape_scaling) | 2e-2 | 2.226 | 2.068 | [`03-muon-kj-lr2e-2.sbatch`](runs/03-muon-kj-lr2e-2.sbatch) | [f7uvsbai](https://wandb.ai/ischlag/megatron-lm-research-baseline/runs/f7uvsbai) | [`baseline-2026-05-08-a76e5bc`](https://github.com/ischlag/megatron-lm-research-baseline/tree/baseline-2026-05-08-a76e5bc) |
| 12 | normuon-lr2e-4 | normuon-lr3.6e-4 | LR 3.6e-4 -> 2e-4 | NorMuon | 2e-4 | 2.227 | 2.063 | [`02-normuon-lr2e-4.sbatch`](runs/02-normuon-lr2e-4.sbatch) | [heevf919](https://wandb.ai/ischlag/megatron-lm-research-baseline/runs/heevf919) | [`baseline-2026-05-08-a76e5bc`](https://github.com/ischlag/megatron-lm-research-baseline/tree/baseline-2026-05-08-a76e5bc) |
| 13 | normuon-fp8 | normuon-lr3.6e-4 | add `--fp8-format e4m3 --fp8-recipe blockwise` (DeepSeek-V3) | NorMuon (FP8) | 3.6e-4 | 2.229 | 2.065 | [`07-normuon-fp8.sbatch`](runs/07-normuon-fp8.sbatch) | [7j84uy3j](https://wandb.ai/ischlag/megatron-lm-research-baseline/runs/7j84uy3j) | [`598cbe616`](https://github.com/ischlag/megatron-lm-research-baseline/tree/598cbe616) |
| 14 | normuon-lr6e-4 | normuon-lr3.6e-4 | LR 3.6e-4 -> 6e-4 | NorMuon | 6e-4 | 2.244 | 2.081 | [`04-normuon-lr6e-4.sbatch`](runs/04-normuon-lr6e-4.sbatch) | [zu96uyts](https://wandb.ai/ischlag/megatron-lm-research-baseline/runs/zu96uyts) | [`baseline-2026-05-08-a76e5bc`](https://github.com/ischlag/megatron-lm-research-baseline/tree/baseline-2026-05-08-a76e5bc) |
| 15 | adamw-lr1e-3 |  | root: AdamW baseline (decoupled WD), LR 1e-3 | AdamW | 1e-3 | 2.316 | 2.154 | [`05-adamw-lr1e-3.sbatch`](runs/05-adamw-lr1e-3.sbatch) | [6qecfvwc](https://wandb.ai/ischlag/megatron-lm-research-baseline/runs/6qecfvwc) | [`baseline-2026-05-08-a76e5bc`](https://github.com/ischlag/megatron-lm-research-baseline/tree/baseline-2026-05-08-a76e5bc) |
| 16 | adamw-lr5e-4 | adamw-lr1e-3 | LR 1e-3 -> 5e-4 | AdamW | 5e-4 | 2.363 | 2.199 | [`06-adamw-lr5e-4.sbatch`](runs/06-adamw-lr5e-4.sbatch) | [zzywif5m](https://wandb.ai/ischlag/megatron-lm-research-baseline/runs/zzywif5m) | [`baseline-2026-05-08-a76e5bc`](https://github.com/ischlag/megatron-lm-research-baseline/tree/baseline-2026-05-08-a76e5bc) |
| -- | aurora-qkn-derf-rmsqk | aurora-qkn | swap RMSNorm for Derf at block norms (input/pre-MLP/final); keep RMSNorm at QK (`APERTUS_DERF_QK_RMSNORM=1`) | Aurora + QK norm + Derf | 1e-2 | 2.217 | 2.057 | [`10-aurora-qkn-derf-rmsqk.sbatch`](runs/10-aurora-qkn-derf-rmsqk.sbatch) | [94h3m6bs](https://wandb.ai/ischlag/lm-research-baseline-dev/runs/94h3m6bs) | [`dffead810`](https://github.com/ischlag/megatron-lm-research-baseline/tree/dffead810) |

## Notes on entries

- **`aurora-qkn-moe-32e-tk3-sh1`**: 32 routed experts at width 640 (top-3) plus 1
  always-on shared expert at width 640. Active params per token are
  exactly matched to dense aurora-qkn (`(3 + 1) * 3 * 1024 * 640 = 7.86M`
  per-layer FFN, same as `3 * 1024 * 2560 = 7.86M`). Total ~1.73B (4.9×
  dense). Mixtral-style aux loss (`--moe-aux-loss-coeff 1e-2`) and
  Switch/DeepSeek-style z-loss (`--moe-z-loss-coeff 1e-3`); grouped
  GEMM, alltoall dispatcher, EP=1 / DP=4 (matches dense parallelism).
  3D expert weights bypass Aurora's 2D predicate and fall through to
  AdamW; attention/embeds/output/router still get Aurora. MBS halved
  16 -> 8 to fit MoE permute/dispatch activations + replicated
  expert weights (~1 GB/layer at 32E × 640).
  Steady throughput ~87 TFLOP/s/GPU vs dense ~265 — the small-model MoE
  penalty (per-token all-to-all dispatch dominates at hidden 1024;
  vanishes at hidden ≥ 4K). Promoted from a sweep across `num_experts`
  ∈ {8, 16, 32, 64}, `topk` ∈ {2, 4}, ±1 shared expert; full sweep
  results in the merging PR. `tk3-sh1 32E` matches `tk2 64E` quality
  (final 2.098 vs 2.095) at ~3.6× fewer total params and ~2× the
  throughput.
- **`aurora-qkn-xsa`**: Exclusive Self-Attention (XSA, arXiv 2603.09078) on top
  of the leader. After core attention, subtracts each head's output along its
  L2-normalized value direction: `out -= (out . v_hat) * v_hat`. Zero parameters.
  GQA-aware implementation broadcasts `v_hat` against query heads of each KV
  group (no V materialization), wrapped in `@torch.compile`. Replicates the
  apertus_two_ablations 125M result (XSA was the best attention variant there
  at min loss). Throughput ~234.8 TFLOP/s/GPU; the GQA-aware rewrite eliminates
  the ~9% throughput tax of the naive `repeat_interleave` version.
- **`aurora-xsa`**: XSA without QK-norm. Apples-to-apples sanity check with the
  apertus 125M sweep (XSA standalone wins over its parent without QK-norm).
- **`aurora-qkn-softmax1`**: `--softmax-type off-by-one`, Evan Miller's
  attention-is-off-by-one sink (built-in upstream; no code change). Adds an
  implicit zero-key column to the softmax denominator. Marginal (-0.018 final,
  +0.002 min) on top of the leader -- QK-norm already addresses the related
  logit-blow-up pathology.
- **`aurora-qkn-learned-sink`**: `--softmax-type learnable`, gpt-oss-style
  per-head learnable scalar in the softmax denominator (built-in upstream; no
  code change). HURTS by 0.015 nats min loss vs the leader -- the explicit
  sink-bias mechanism is redundant with QK-norm at this scale and seems to add
  optimization noise.
- **`aurora-qkn-derf-rmsqk`** (representative entry from a `feat/dyt-derf-norm`
  ablation series): Derf normalisation at block sites, RMSNorm at the per-head
  QK sites. Out-of-the-box Derf+QKN at all sites lands at 2.278 (+0.093 vs
  RMSNorm leader); keeping RMSNorm at QK closes most of that gap to +0.032.
  Hyperparameter sweeps on Derf (`alpha_init` 0.2 / 0.5 / 0.8, Aurora LR
  8e-3 / 1e-2) move final loss only ~0.02 nats and don't recover the gap.
  DyT under the same recipe lands at 2.307 (+0.122). Wired via
  `--normalization Derf` + `APERTUS_DERF_OPTIM=compile` (torch.compile
  fusion, `_research/derf_optim/option1_compile.py`) which recovers most
  of the throughput lost from unfusing TE's `LayerNormColumnParallelLinear`
  (270 vs 217 TFLOP/s/GPU at this shape).
- **`aurora-qkn`**: per-head RMSNorm on Q and K (TENorm under
  `--normalization RMSNorm`). WD stays off on the new q/k norm gains
  (default skip-WD-on-shape-1 rule); a paired ablation with
  `--apply-wd-to-qk-layernorm` landed at final 2.187 / min 2.025,
  indistinguishable from no-WD-on-gains.
- **`aurora-qkn-fp8`**: same recipe as `aurora-qkn` plus DeepSeek-V3 FP8
  (e4m3 + blockwise scaling, FP8 matmuls only; params and master weights
  stay bf16; no `--fp8-param-gather`, no `fp8_param`). Loss penalty is
  noise (+0.003 final / +0.002 min). **Throughput drops** ~7% (246 vs
  ~265 TFLOP/s/GPU). Below 350M / hidden 1024, the FP8 matmul win doesn't
  amortise the per-tensor amax-tracking overhead, and `--qk-layernorm`
  adds 48 bf16 norm sites that don't speed up. Compare to `normuon-fp8`
  (no QKN, slightly larger relative matmul share): there FP8 was a small
  net throughput **gain** (~313 vs ~310 bf16). Going to larger hidden /
  longer sequence reverses the sign.
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
- **Rank 9** is the first architecture variant: hybrid 3:1 (Schlag DeltaNet
  + softmax attention every 4th layer), per-channel output gate, full
  per-batch carry-v2 of the recurrent state across batches, and a hard
  Frobenius cap of 200 on the per-element carried state. NorMuon optimizer
  to match the canonical recipe. Cap=200 binds late (~step 950) and keeps
  the carried state norm bounded for length-gen safety without measurably
  hurting iso-token loss vs the no-cap variant. Runs on
  `feat/deltanet-variants` branch (DeltaNet flag-gated path); will be
  re-tagged on main once the branch lands.
