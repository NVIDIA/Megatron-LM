---
name: run-qwen-model
description: Run Qwen3-30B-A3B inference with Megatron-Core or vLLM on one OCI 4×GB200 node at batch size 256, and capture matching Nsight Systems profiles. Use for Qwen 30B mcore runs, vLLM runs, baseline benchmarks, or nsys profile capture. Performance optimization belongs to the qwen-model-optimizer agent.
---

# Run Qwen3-30B-A3B on OCI GB200

This skill only runs the fixed comparison workload. Use the
`qwen-model-optimizer` subagent for performance investigation or code changes.

## Fixed comparison

| Setting | Megatron-Core | vLLM |
|---|---|---|
| Cluster | OCI `oci-hsg`, one 4×GB200 node | same |
| Model | Qwen3-30B-A3B, BF16 | same HF weights |
| Batch | 256 gsm8k requests | same |
| Throughput OSL | 1024 | 1024 |
| Profile OSL | 128 (bounded trace) | 128 |
| Parallelism | **TP=1, PP=1, EP=4** | **TP=1, DP=4, expert parallel enabled** |
| Warmup / timed | 2 / 5 | 2 / 5 |

Checkpoint paths:

```bash
export QWEN30B_CKPT=/lustre/fsw/portfolios/coreai/users/shanmugamr/agents-space/checkpoints/qwen3-30b-a3b-mcore
export QWEN30B_TOKENIZER=/lustre/fsw/portfolios/coreai/users/shanmugamr/agents-space/checkpoints/qwen3-30b-a3b-hf
export QWEN30B_HF="$QWEN30B_TOKENIZER"
```

Before running:

Run these from inside the checkout you want to benchmark — cog deploys the
tree you are standing in:

```bash
source ~/.cog/setup.env.oci-hsg
cog prepare-image --cluster-name "$COG_CLUSTER_NAME"
cog ensure-env \
  --cluster-name "$COG_CLUSTER_NAME" \
  --run-name qwen30b-env --gpus 4 --time 00:30:00 \
  --partition "$COG_BATCH_PARTITION"
```

Never download checkpoints. If either path is missing, ask the user.

**Confirm which checkout you are benchmarking.** With more than one Megatron-LM
checkout on the machine, deploying the wrong one is silent — the run succeeds
and measures code you never changed. The scripts here default to the checkout
they live in and warn on a mismatch, and `COG_MEGATRON_REPO` is deliberately
absent from `~/.cog/setup.env*` so nothing machine-wide can override that. If
you set it by hand, it wins; check before a measurement run:

```bash
echo "${COG_MEGATRON_REPO:-<unset — cog will use cwd>}"; git rev-parse --show-toplevel
```

Cross-check afterwards: the `CODE_REVISION` recorded by the run must equal your
local `HEAD`.

## If a cog command hangs, switch to `sbatch` after the second attempt

These runs are long enough that a stuck launch is expensive. Every cog command
that touches your code syncs the workspace first; when that sync hangs there is
no job ID and no Slurm log, so it is indistinguishable from a slow start except
by the missing `job_id`.

> **Two strikes, then hand-write the job.** After the second consecutive hang
> or broken pipe from the same client, stop retrying and submit `sbatch`
> directly against a tree staged on lustre. On the rebase measurement this
> converted a day of failed cog attempts into a completed benchmark in under 7
> minutes. Full recipe — image path via `cog profile` (local-only, so it still
> works), staging to a fresh directory, container mounts, venv activation — is
> in `skills/cog-setup-and-help/SKILL.md`, section "Escape hatch: when
> workspace sync hangs".

Three rules for any directly-submitted job here, all learned the hard way:

- **Always pass an explicit `--qos`.** Omitting it takes the default `normal`
  (priority 100) and buries you behind a queue whose head sits near 350k, even
  when hundreds of nodes are idle: two jobs submitted without it sat `PENDING
  (Priority)` for 14 hours and 7 hours, then both started *within seconds* of
  resubmission under a correct QOS. **Pick by node count, not by walltime**, and
  the name `interactive` is misleading — a QOS constrains resources, not how you
  submitted, so it applies to `sbatch` exactly as it does to `salloc`:
  `--qos=interactive` (priority 700, ≤4 nodes/user, **no walltime cap**) for
  anything up to 4 nodes, which is every job in this skill; `--qos=short`
  (priority 200, ≤2 h, ≤64 nodes) only when you need more than 4 nodes. For a
  1-node job `interactive` strictly dominates: 3.5× the priority and no time
  limit. All three carry `DenyOnLimit`, so exceeding a limit is rejected at
  submit rather than pending forever. Check your grants with
  `sacctmgr -nP show assoc user=$USER format=Account,QOS` and the limits with
  `sacctmgr -nP show qos format=Name,Priority,MaxWall,MaxTRESPU`. Diagnose a
  stalled job by comparing `squeue -j <id> -o '%Q'` against
  `squeue -p batch -t PD -S -Q -o '%.8Q' -h | head`; if idle nodes exist and
  your number is far below the head, it is QOS, not contention.
- **Gate the allocation on an `os.path.isfile` filesystem health check** over a
  few of your changed files plus something deep in the shared venv, and `exit`
  before the model load if any are unreadable. A node with a broken Lustre
  client lists files it cannot open, and a 4×GB200 allocation is far too
  expensive to discover that after checkpoint load.
- **A cascade of `ModuleNotFoundError` across unrelated packages is a bad node,
  not a bad venv.** Exclude it (`--exclude=<node>`) and resubmit; do not
  pip-install overlay copies to work around it.
- **A job that dies in seconds with exit 141 and empty logs is SIGPIPE, and
  under `set -euo pipefail` the writer is what failed.** `ls … | head -1`
  is the usual culprit: `head` closes the pipe, `ls` takes SIGPIPE (128+13),
  `pipefail` promotes it, and `set -e` aborts before a single line is logged.
  This is latent until the glob grows, so it appears long after the script was
  written. Drain the pipe (`sed -n 1p`) rather than closing it early. It bit
  `run_qwen_vllm.sh` once the workspace glob reached 142 entries.

The cog skill's inlined template is the source of truth for the sbatch pattern.

## 1. Megatron-Core inference

Runs `examples.inference.launch_inference_server` with
`transformer_impl=inference_optimized`, full-iteration CUDA graphs, and the
fixed EP4/TP1 layout.

```bash
EXPERIMENT_ID=MCORE-BASELINE \
EXPERIMENT_HYPOTHESIS="Fresh EP4 mcore baseline" \
QWEN30B_TP=1 QWEN30B_EP=4 QWEN30B_ETP=1 \
BENCH_SIZES_OVERRIDE=256 BENCH_OUTPUT_TOKENS=1024 \
NUM_WARMUP_ITERS=2 NUM_TIMED_ITERS=5 \
bash skills/run-qwen-model/run_qwen_inference.sh \
  qwen3-30b-a3b --checkpoint "$QWEN30B_CKPT"
```

Do not add optimization flags to a baseline run.

## 2. vLLM inference

Runs `vllm serve` with TP1/DP4 and `--enable-expert-parallel`.

```bash
EXPERIMENT_ID=VLLM-BASELINE \
EXPERIMENT_HYPOTHESIS="Fresh vLLM DP4+EP baseline" \
BENCH_BS=256 BENCH_OUTPUT_TOKENS=1024 \
NUM_WARMUP_ITERS=2 NUM_TIMED_ITERS=5 \
bash skills/run-qwen-model/run_qwen_vllm.sh
```

## Nsight Systems profiles

Profiles use BS256 and OSL128 to bound trace size while preserving the same
parallel layouts. Both scripts export `.nsys-rep` and `.sqlite`.

```bash
# mcore EP4/TP1
PROFILE_BS=256 PROFILE_OSL=128 \
QWEN30B_CKPT="$QWEN30B_CKPT" \
bash skills/run-qwen-model/profile_qwen_mcore.sh

# vLLM DP4+EP
PROFILE_BS=256 PROFILE_OSL=128 \
bash skills/run-qwen-model/profile_qwen_vllm.sh
```

Run the profiles sequentially to avoid node contention. Record the run
directory, Slurm job, throughput, latency, TPOT, and trace paths in
`EXPERIMENTS.md`.

For analysis, use `skills/nsight-system-analysis/SKILL.md` — Workflow C
(`scripts/forward_pass.py`) for the single-decode-step mcore/vLLM comparison, and
its Steps 1–6 for deeper interval-union attribution. Optimization decisions
belong to `skills/optimize-inference-siddharth/SKILL.md`.

## Records

`EXPERIMENTS.md` is the sole performance ledger. The first two records must be
the fresh vLLM and mcore nsys baselines. Append every later attempt, including
failures and regressions; never rewrite prior records.
