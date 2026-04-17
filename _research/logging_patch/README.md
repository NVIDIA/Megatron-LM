# logging_patch

Reproducibility + per-run JSON logging for Apertus 2 ablations. Hooks into Megatron's `training_log` and `setup_model_and_optimizer` via monkey-patches so we never touch the upstream sources. All configuration is via environment variables so the patch stays decoupled from Megatron's argparse.

## Wiring

Already wired in `pretrain_gpt.py` just before `pretrain(...)`:

```python
from _apertus.logging_patch import install as _apertus_install
_apertus_install()
```

Because `PYTHONPATH=$WORKDIR` is set in the sbatch scripts, the package is importable as `_apertus.logging_patch`.

## Environment variables

| var                              | default                              | effect |
| ---                              | ---                                  | --- |
| `APERTUS_LOG_DIR`                | `_apertus/results/performance`       | Where to write `<run_name>.json` |
| `APERTUS_RUN_NAME`               | `<branch>-<sha>-<ts>-<jobid>`        | Override the generated run name |
| `APERTUS_TRACK`                  | unset                                | `throughput` or `performance` (used by the analyse notebook to filter) |
| `APERTUS_FEATURE`                | git branch                           | Feature label for the run (override if different from branch) |
| `APERTUS_DETERMINISTIC`          | unset                                | `1` enables deterministic cudnn + `torch.use_deterministic_algorithms` |
| `APERTUS_LOG_PER_LAYER_GRADS`    | unset                                | `1` emits `per_layer_grad_norm` dict per step (~1 float per parameter tensor) |
| `APERTUS_LOG_ACT_STATS`          | `1` (on)                             | `0` disables the forward hooks on every attention + mlp submodule; otherwise emits `act_stats` per step |
| `APERTUS_LOG_LOSS_SPIKES`        | unset                                | `1` emits a boolean `loss_spike` per step based on rolling z-score |
| `APERTUS_LOG_TOP1_ACC`           | `1` (on)                             | `0` disables top-1 next-token accuracy; otherwise emits `top1_accuracy` per log interval and mirrors to wandb |

## JSON schema

One file per run: `<log_dir>/<run_name>.json`. Rewritten atomically every log interval (tmp + rename). Top-level keys:

```json
{
  "name": "feat-foo-abc1234-20260411-123456-1824563",
  "feature": "feat/foo",
  "track": "throughput",
  "git_sha": "abc1234...",
  "start_time": 1744369200.1,
  "env": { "git": {...}, "host": {...}, "framework": {...}, "slurm": {...}, "argv": [...], "config": {...} },
  "n_params_total": 8012345678,
  "n_params_active": 8012345678,
  "tokens_per_sec_per_gpu": 12345.6,
  "mfu": 0.41,
  "steps": 30,
  "tokens": 245760,
  "series": {
    "step":        [1, 2, ...],
    "wall":        [1744369201.3, ...],
    "train_loss":  [11.2, 10.8, ...],
    "lr":          [3e-4, ...],
    "grad_norm":   [1.3, ...],
    "params_norm": [123.4, ...],
    "tput":        [12345.6, ...],
    "per_layer_grad_norm": [ {"decoder.layers.0.self_attention...": 0.12, ...}, ... ],
    "act_stats":   [ {"decoder.layers.0.self_attention": {"norm_mean": 42.1, "max": 128.0}, ...}, ... ],
    "loss_spike":  [false, false, ...],
    "top1_accuracy": [0.01, 0.02, ...]
  }
}
```

Only `step`, `wall`, and whatever scalars Megatron passed through (`train_loss`, `lr`, `grad_norm`, `params_norm`, `tput`) are always populated. The three per-layer/act/spike columns only appear when their env flag is set. The loader in `_apertus/analyse/load_runs.py` tolerates missing keys.

## What the monkey-patch does

1. `patch_setup_model_and_optimizer` — wraps Megatron's setup call. After the original runs, stores a reference to the returned model list, counts parameters (all-reducing across the world and dividing out DP×CP replicating factors), and (if `APERTUS_LOG_ACT_STATS` is on) walks the module tree and registers forward hooks on every `*.self_attention` and `*.mlp` submodule.
2. `patch_training_log` — wraps Megatron's `training_log`. After the original logs to stdout/wandb/tensorboard, we extract the same arguments, compute tokens/sec/GPU from a wall-clock delta, compute MFU via a standard `6N + 6*L*S*H` formula (`mfu.py`), drain the activation accumulators, optionally compute per-layer grad norms, check the loss spike heuristic, all-reduce accumulated top-1 counts, mirror the result to wandb, and append one row to the JSON log.
3. `patch_compute_language_model_loss` — wraps `LanguageModule.compute_language_model_loss` on the last pipeline stage. After the original returns, computes top-1 predictions from the TP-vocab-parallel logits: each rank finds its local argmax, all-reduces the max logit across the TP group, and masks to recover the global argmax. Only TP rank 0 accumulates (correct, total) into module state so the later all-reduce in `training_log` yields a correctly-scaled ratio.

The MFU formula uses `--num-layers`, `--hidden-size`, `--seq-length`, and the local param count. We parse the first three from `argv` before Megatron's argparse runs (so they're available at `install()` time). If any is missing, MFU is silently skipped.

## Determinism

Setting `APERTUS_DETERMINISTIC=1` flips on the following before any CUDA kernel launches:

- `CUBLAS_WORKSPACE_CONFIG=:4096:8` (required by `torch.use_deterministic_algorithms`)
- `torch.use_deterministic_algorithms(True, warn_only=True)`
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`

Megatron's own `--seed` is respected. We do not override it.

## Rank-0 only

The writer short-circuits on `RANK != 0` so only rank 0 emits the JSON. Other ranks call through to the original `training_log` unchanged.

## Notes / known limitations

- `n_params_active` currently equals `n_params_total`. For MoE we'll refine this once we hook up the real 600B config and know how many experts fire per token.
- Tokens/sec is computed from wall-clock deltas, not from Megatron's internal `interval-time` timer, so it picks up a little noise on step 1. The MFU number agrees with `--log-throughput` within ~1% once the run is past the first few steps.
- Checkpointing is disabled for now (known SIGSEGV on GH200 ARM64). When we re-enable it, add a `writer.set(checkpoint=...)` call in the resume path.
