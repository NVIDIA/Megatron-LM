# Megatron-LM Research Baseline

A fork of [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) for
reproducible research comparisons of dense and MoE language models at various
scales. Configurations are tuned for the Swiss AI Alps supercomputer (Clariden
cluster, GH200 Grace-Hopper nodes with 4 GPUs per node, Slingshot-11
interconnect).

Training data is [ClimbMix](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix)
in Megatron binary format with the GPT-2 BPE tokenizer (50257 vocab).

See [README.nvidia.md](README.nvidia.md) for the original NVIDIA Megatron-LM
documentation.

**AI assistants**: read [AGENTS.md](AGENTS.md) before making changes — it
documents the repo layout, experiment flow, and conventions.

## Changes from upstream

## Configurations

Transformer++ baselines (SwiGLU, RMSNorm, RoPE, GQA, AdamW, WSD schedule,
bf16) on ClimbMix with GPT-2 BPE tokenizer. All configs use GBS=128
sequences (524K tokens/step) and are tuned for GH200 nodes with 4 GPUs each.

| config | params | tokens | nodes | GPUs | DP | MBS | est. wall | GPU-h |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `transformer-pp-350m-ablation` | 350M | 1B | 1 | 4 | 4 | 16 | ~30 min | 2 |
| `transformer-pp-350m-{adamw,muon}` | 350M | 15B | 1 | 4 | 4 | 16 | ~8 h | 31 |
| `transformer-pp-760m-adamw` | 760M | 30B | 4 | 16 | 16 | 4 | ~7 h | 115 |
| `transformer-pp-1.3b-adamw` | 1.3B | 100B | 8 | 32 | 32 | 2 | ~19 h | 596 |
| `transformer-pp-2.7b-adamw` | 2.7B | 300B | 16 | 64 | 64 | 1 | ~60 h | 3,834 |

The `-muon` variant at 350M uses NorMuon (adaptive_muon + normuon) with
matrix LR 3.6e-4 and scalar LR 1.5e-3; it differs from `-adamw` only in
the optimizer block (validated on `-ablation`: NorMuon beats AdamW by
~0.10 nats final loss).

### Architectures

| | 350M | 760M | 1.3B | 2.7B |
| --- | ---: | ---: | ---: | ---: |
| hidden | 1024 | 1536 | 2048 | 2560 |
| layers | 24 | 24 | 24 | 32 |
| heads / kv_heads | 16 / 4 | 24 / 8 | 32 / 8 | 32 / 8 |
| ffn (SwiGLU) | 2560 | 4096 | 5632 | 7680 |
| peak LR | 3e-4 | 2.5e-4 | 2e-4 | 1.6e-4 |

### Leaderboards

Ranked per-size run lists live under
[`_research/leaderboards/`](_research/leaderboards/README.md). Each entry
has a self-contained sbatch (no env-var branching, all hparams pinned)
plus a W&B link and the git sha it was executed at. Current leaderboards:

- [`350m-ablation`](_research/leaderboards/350m-ablation/README.md) —
  1B-token optimizer ablations. Top entry: NorMuon @ 3.6e-4, final loss
  2.225 (beats best AdamW @ 1e-3 by ~0.09 nats).
- `350m`, `760m`, `1.3b`, `2.7b` — placeholder dirs; no full runs yet.

### Launch vs sweeps vs leaderboards

| folder | purpose |
| --- | --- |
| `_research/launch/` | canonical baseline scripts, one `-adamw` and (at 350m) one `-muon` per size. The *recommended* config. |
| `_research/sweeps/` | sweep harnesses with `SWEEP_*` env-var branching. Use these to explore new ablation axes. |
| `_research/leaderboards/` | historical ranked runs. Each entry is a frozen, reproducible sbatch + W&B link. |

Workflow: sweep in `sweeps/` → once a variant wins, snapshot a
self-contained sbatch into the matching `leaderboards/<size>/runs/` and
add a row to that size's `README.md` table.

All scripts live in `_research/launch/` and accept `SWEEP_MBS` and
`SWEEP_MOCK` env-var overrides for quick sweeps.

## Changes from upstream

### Logging patch (`_research/logging_patch/`)

A monkey-patch layer that hooks into Megatron's `training_log` and
`setup_model_and_optimizer` without modifying upstream source files. All
configuration is via environment variables; the patch is activated by a
two-line import in `pretrain_gpt.py`.

Per-step metrics written to an append-only JSONL log (`<run>.jsonl`) plus a
metadata sidecar (`<run>.meta.json`):

| metric | env var | default |
| --- | --- | --- |
| `train_loss`, `lr`, `grad_norm`, `params_norm`, `tput` | always on | -- |
| MFU (model FLOPs utilization) | always on | -- |
| Per-layer activation norms and max | `APERTUS_LOG_ACT_STATS` | on |
| Top-1 next-token accuracy (TP-aware) | `APERTUS_LOG_TOP1_ACC` | on |
| Per-parameter gradient norms | `APERTUS_LOG_PER_LAYER_GRADS` | off |
| Loss spike detection (rolling z-score) | `APERTUS_LOG_LOSS_SPIKES` | off |
| Startup phase timeline (sbatch, srun, container, dist init, model build, first iters) | always on | -- |

The JSONL writer scales O(1) per step (no full-file rewrite). An analysis
loader at `_research/analyse/load_runs.py` reads both the new JSONL format and
legacy single-file JSON for backward compatibility.

### Muon / NorMuon (`--optimizer muon`, `--optimizer adaptive_muon`)

Muon uses Newton-Schulz orthogonalization on the momentum matrix to take
spectrally-normalized steps. Routing: 2D matrix params use Muon; scalar
params (embeddings, norms, biases) use AdamW via the upstream
`_is_nonlinear_or_embedding` predicate. Relevant flags (we added three
on top of upstream):

- `--muon-scalar-lr` (decouples the AdamW-group LR from the Muon-group LR;
  NorMuon recipe uses matrix LR 3.6e-4 and scalar LR 1.5e-3).
- `--muon-scalar-weight-decay` (set to 0 to keep WD on the Muon matrix
  group only).
- `--adaptive-muon-moment2-method` (`adamuon` default, or `normuon` for
  per-row second-moment rescale, arXiv 2510.05491).

**Must omit `--overlap-param-gather` when training with Muon.** This flag
is safe for AdamW because Adam's update is elementwise: each weight moves
by `lr * m / (sqrt(v) + eps)` regardless of when other weights are
gathered, so pipelining the per-bucket all-gather with the next forward
pass is correctness-neutral. Muon's update is spectral: Newton-Schulz
orthogonalizes the full momentum matrix, and every row's update
direction depends on every other row's magnitude (via the `X Xᵀ X` terms
inside NS5). If `--overlap-param-gather` assembles the matrix from shards
gathered at inconsistent async states, NS5 sees a corrupted matrix, its
output isn't an orthogonal projection of the true momentum, and the step
has the wrong spectrum. Empirically this manifests as monotonically
growing `params_norm` and `grad_norm` from step 1, not just slower
convergence. Keep `--overlap-grad-reduce` and `--use-distributed-optimizer`
(both verified safe for Muon). Throughput cost of dropping
`--overlap-param-gather` at 350M / 1 node is ~2-3%; may grow at larger
scale where param-gather overlap matters more.

### AdEMAMix optimizer (`--optimizer ademamix`)

Ported from the [swiss-ai/Megatron-LM](https://github.com/swiss-ai/Megatron-LM)
fork. AdEMAMix adds a slow EMA on top of Adam's fast EMA for better
convergence at scale (Pagliardini et al., 2025). Relevant flags:

- `--ademamix-alpha` (default 2.0)
- `--ademamix-beta3` (default 0.9999)
- `--ademamix-beta3-warmup` (warmup steps for beta3 in half-life space)
- `--ademamix-alpha-warmup` (warmup steps for alpha)

### xIELU activation (`--xielu`)

Learnable per-layer activation from the Apertus architecture (two parameters
per layer: `alpha_p`, `alpha_n`). Non-gated 2-matrix MLP (up projection,
activation, down projection). Use with `--ffn-hidden-size` set to the full
intermediate dimension (no 2/3 scaling needed unlike SwiGLU).

### Goldfish loss (`--goldfish-loss`)

Token-level memorization suppression (Hans et al., 2024). Masks ~1/k of
target labels using a deterministic hash of the preceding h tokens. Relevant
flags:

- `--goldfish-k` (drop fraction, default 50)
- `--goldfish-h` (context width for hashing, default 50)

### Determinism (`APERTUS_DETERMINISTIC=1`)

Optional flag that enables `torch.use_deterministic_algorithms`,
deterministic cuDNN, and `CUBLAS_WORKSPACE_CONFIG=:4096:8` for bitwise
reproducibility (at a compute cost).

## Syncing with upstream

```bash
git fetch upstream
git merge upstream/main
```

The `upstream` remote points at `NVIDIA/Megatron-LM`. Our changes are
confined to `_research/`, `megatron/core/optimizer/ademamix.py`,
`megatron/core/activations.py` (XIELU class), and small edits to
`arguments.py`, `optimizer/__init__.py`, `optimizer_config.py`, `mlp.py`,
`gpt_dataset.py`, and `pretrain_gpt.py`. Upstream merges should be
low-conflict.

## License

Same as upstream NVIDIA Megatron-LM. See [LICENSE](LICENSE).
