# Nano-v3 MMLU-Pro disaggregation evaluation

This directory runs one MMLU-Pro configuration against four serving modes on
the same four OCI-HSG GPUs:

| Mode | GPU topology | Control/transfer path |
| --- | --- | --- |
| `no_disagg` | one EP=4 engine | regular Megatron coordinator |
| `dynamo` | EP=2 prefill + EP=2 decode | Dynamo frontend/router + NIXL handoff |
| `native_nccl` | EP=2 prefill + EP=2 decode | native coordinator + NCCL handoff |
| `native_nixl` | EP=2 prefill + EP=2 decode | native coordinator + NIXL handoff |

All modes use the same Nano checkpoint, tokenizer, MMLU-Pro task
(`mmlu_pro_cot_mini_5_shot_base`, five-shot), cache sizing, and inference
limits. Servers are launched and evaluated sequentially so they can reuse a
single four-GPU allocation. The Megatron endpoints retain the harness's
default batch size of 512. Dynamo uses `DYNAMO_BATCH_SIZE=128` because its
OpenAI frontend accepts at most 128 choices in one completion request.

CUDA graphs remain enabled with the local implementation and block scope in
every mode. `--inference-cuda-graph-all-prefills` prevents the scheduler from
admitting new prefill work that cannot use a captured graph. The matrix defaults
to `INFERENCE_NUM_CUDA_GRAPHS=13`, which captures exponential prefill/mixed
token-count buckets from 8,192 through 4 plus 1, as well as decode buckets from
32 through 1. The explicit count trims the two-token prefill bucket while
retaining the two-token decode graph. Auto-sizing (`-1`) captures a two-token
pure-prefill shape that currently triggers a device-side assert in the
fused-MoE Triton kernel for Nano with expert parallelism, so it should not be
used for this matrix.
Every inference step is logged, and the matrix fails immediately if a component
log contains `cuda graph OFF`.
Native NCCL uses `NATIVE_NCCL_CUDA_DEVICE_MAX_CONNECTIONS=8` because its
point-to-point KV handoff must make progress concurrently with EP collectives;
the other modes retain `CUDA_DEVICE_MAX_CONNECTIONS=1`.

All four modes have completed full 1,203-example MMLU-Pro evaluations with
72/72 CUDA-graph shapes on every participating rank and no eager fallbacks.
The verified run IDs and metrics are recorded in
[ERROR_LOG.md](ERROR_LOG.md).

The server runs in the Dynamo/NIXL container. Following the cluster workflow,
the outer launcher waits for server readiness, SSHes from the login host to the
allocated node, and runs the eval client in the host environment. This is
necessary because the shared lm-eval harness uses Python 3.10 while the server
image uses Python 3.12.

## Prerequisites

- Run from an OCI-HSG login node with access to the `nemotron_sw_pre` account.
- The default Dynamo/NIXL image is
  `/lustre/fsw/portfolios/nemotron/users/csathe/chaitrasathe+dynamo-megatron+mamba.sqsh`.
- The default checkpoint, tokenizer, HF cache, and `adlr/nemo5` lm-eval harness
  paths are centralized in `config.sh`. Override any of them through the
  environment if needed.
- The supplied Dynamo/NIXL image does not include Quart. On first launch the
  matrix installs the repo-locked `quart==0.20.0` and its dependencies into
  `EVAL_PYTHON_DEPS_DIR` (under the evaluation output base by default) and
  reuses that overlay for subsequent runs.
- The image runs PyTorch with CUDA 13.2 but contains only the
  `nixl-cu12==0.10.1` backend. For `dynamo` and `native_nixl`, the launcher
  detects this mismatch and installs the matching `nixl-cu13==0.10.1` wheel
  into the same reusable overlay. It selects that wheel's NIXL plugin and UCX
  CUDA modules as one stack. `UCX_TLS` includes `tcp` for NIXL's active-message
  control path plus `cuda_copy` and `cuda_ipc` for VRAM. The launcher then
  verifies an actual GPU-to-GPU loopback transfer before model loading. This
  prevents UCX from silently registering CUDA buffers as host memory.
- MMLU-Pro defines four explicit stop strings. Dynamo currently requires fewer
  than four, so the Dynamo client omits only `</s>` from the explicit list.
  This tokenizer defines `</s>` as its EOD token, and Megatron applies that EOD
  as the default termination ID; `Q:`, `Question:`, and `<|im_end|>` remain
  explicit stop strings.
- `HF_DATASETS_CACHE` defaults to an evaluation-owned cache under the output
  base. This avoids reading dataset metadata from another user's `HF_HOME`
  that may have been produced by an incompatible `datasets` version.
- `HF_HUB_CACHE` is derived from `HF_HOME` even when the login environment has
  another hub-cache value. Override it explicitly with `EVAL_HF_HUB_CACHE`.
- The worktree must be clean enough that the commit recorded in the result
  manifest identifies the code being evaluated; uncommitted changes are not
  copied elsewhere.

## Run

Start with a short four-mode smoke test from the repository root:

```bash
EVAL_LIMIT=5 \
  bash examples/inference/nano_v3_mmlu_pro_eval/launch_oci.sh
```

For an integer `EVAL_LIMIT`, the launcher caps the client batch size to that
limit so lm-eval does not pad a five-example smoke test into 512 duplicate
server requests. Full runs retain the normal 512/128 endpoint batch sizes.

Run the complete matrix:

```bash
bash examples/inference/nano_v3_mmlu_pro_eval/launch_oci.sh
```

To keep each allocation comfortably below the interactive time limit, modes
can also be run separately while preserving the same configuration:

```bash
MODES=no_disagg bash examples/inference/nano_v3_mmlu_pro_eval/launch_oci.sh
MODES=dynamo bash examples/inference/nano_v3_mmlu_pro_eval/launch_oci.sh
MODES=native_nccl bash examples/inference/nano_v3_mmlu_pro_eval/launch_oci.sh
MODES=native_nixl bash examples/inference/nano_v3_mmlu_pro_eval/launch_oci.sh
```

Common overrides include `BATCH_SIZE`, `DYNAMO_BATCH_SIZE`, `EVAL_LIMIT`,
`INFERENCE_NUM_CUDA_GRAPHS`, `INFERENCE_LOGGING_STEP_INTERVAL`, `SLURM_TIME`, `HF_HOME`,
`LM_EVAL_HARNESS_PATH`, `EVAL_OUTPUT_BASE`, `NIXL_UCX_TLS`, and
`NATIVE_NCCL_CUDA_DEVICE_MAX_CONNECTIONS`.
`launch_oci.sh` can also be run from an existing allocation; it detects
`SLURM_JOB_ID` and starts the container as an overlapping job step.

## Results and logs

Each invocation creates a timestamped directory under
`eval_results/nano_v3_mmlu_pro/` containing:

- one lm-eval result directory per mode;
- `matrix.log` for container orchestration and dependency setup;
- `nixl-cuda-preflight.log` with the selected CUDA/NIXL variant and UCX GPU
  transfer check when a NIXL-backed mode is selected;
- `runtime/<mode>/*.log` for every server and Dynamo component;
- `runtime/<mode>/client-handoff.log` for the host SSH/evaluator handoff;
- `manifest.txt` with the commit, paths, modes, and evaluation settings;
- `gpus.txt` with the allocated GPU identities; and
- `summary.txt` with numeric metrics extracted from lm-eval result JSON files.

On startup or evaluation failure, the launcher prints the last 100 lines from
every component log for that mode and terminates the complete process group.

See [ERROR_LOG.md](ERROR_LOG.md) for the chronological pipe-cleaning journal,
including primary error signatures, root causes, fixes, and verification runs.
