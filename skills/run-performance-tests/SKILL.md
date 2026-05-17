---
name: run-performance-tests
description: Run Megatron-LM inference performance tests on the cw-dfw cluster via cog. Spins up an OpenAI-compatible completions server with a chosen model, sweeps batch sizes with synthetic prompts, records throughput / latency / TPOT, and compares against a checked-in baseline_values.json with a configurable tolerance.
when_to_use: Running an inference perf test; bootstrapping or refreshing a perf baseline; investigating an inference perf regression flagged by CI; 'run perf test', 'inference benchmark', 'throughput regression', 'baseline_values.json', 'performance_tests'.
---

# Run Inference Performance Tests

Run Megatron-LM inference performance tests on the cw-dfw Slurm cluster via
`cog submit`. The harness spins up the dynamic text-generation server, fires
concurrent OpenAI-style `/v1/completions` requests at sweep batch sizes, and
compares throughput / latency / TPOT against `baseline_values.json` with a 10%
tolerance.

## TL;DR — happy-path invocation that works on cw-dfw today

```bash
COG_EXTRA_MOUNTS=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_mcore:/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_mcore \
cog submit \
  --repo /Users/shanmugamr/Megatron-LM \
  --cluster-name cw-dfw \
  --run-name perf-<model>-verify \
  --base-image gitlab-master.nvidia.com/adlr/megatron-lm/mcore_ci_dev:main \
  --gpus <gpus> --nodes 1 --ntasks-per-node 1 --time <HH:MM:SS> --partition batch \
  --command 'GPUS_PER_NODE=<gpus> bash tests/performance_tests/shell_test_utils/run_perf_test.sh \
    "CONFIG_PATH=tests/performance_tests/test_cases/<model_family>/<test_case>/model_config.yaml" \
    "CHECKPOINT_LOAD_PATH=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_mcore/mcore_ci" \
    "RESULTS_ROOT=/tmp/perf_results"'
```

Critical flags (all identical to functional tests on cw-dfw):
- `--base-image gitlab-master.nvidia.com/adlr/megatron-lm/mcore_ci_dev:main` — `yq` and TE present.
- `--ntasks-per-node 1` — `run_perf_test.sh` launches `torch.distributed.run` itself.
- `COG_EXTRA_MOUNTS=...` — required to mount the mcore_ci data project into the container.
- `GPUS_PER_NODE=<n>` — must match `--gpus`; the script reads it for torchrun's `--nproc-per-node`.

## How it works

1. `run_perf_test.sh` parses `KEY=VALUE` positional args, loads `model_config.yaml` via `yq`.
2. Spawns `tools.run_dynamic_text_generation_server` in the background via `torch.distributed.run`, passing model-specific args from `tests/performance_tests/server/model_args/<MODEL>.args`.
3. Polls `GET /v1/health` until the server is up (15 min timeout).
4. For each batch size in `BATCH_SIZES`, runs `client/static_benchmark.py`:
   - Synthetic prompt of `NUM_INPUT_TOKENS` tokens.
   - `NUM_WARMUP_ITERS` un-timed batches, then `NUM_TIMED_ITERS` timed batches.
   - All requests in a batch fire simultaneously via `asyncio.gather`.
5. Writes `results.json` to `RESULTS_ROOT` with one entry per batch size:

   ```json
   {
     "batch_8": {
       "batch_size": 8,
       "num_input_tokens": 512,
       "num_output_tokens": 128,
       "num_iters": 5,
       "throughput_tok_per_sec": 351.4,
       "avg_latency_ms": 2912,
       "p50_latency_ms": 2915,
       "p99_latency_ms": 2921,
       "tpot_ms_per_tok": 22.8
     },
     ...
   }
   ```

6. Runs `compare_to_baseline.py`: fail when `throughput_tok_per_sec < baseline * 0.9` or any latency metric `> baseline * 1.1`. Exit `1` on any regression, `0` otherwise.
7. Kills the server (trap on EXIT).

## Modes

| Env var | Effect |
|---------|--------|
| _(none)_ | Default: bench → compare. Fails on regression. |
| `RECORD_BASELINE=1` | Bench → overwrite `baseline_values.json` in the test case dir. Skip comparison. Use for first-time bootstrap. |
| `SKIP_COMPARE=1` | Bench → emit `results.json`, skip comparison. Use to capture numbers without asserting. |

Note: `RECORD_BASELINE=1` writes to the synced workspace on cluster scratch
(`/lustre/.../workspaces/.../repo/tests/performance_tests/test_cases/...`).
`scp` the resulting file back into your local checkout to commit it.

## Available test cases (validated 2026-05-13 on cw-dfw H100)

| Test case | Model | GPUs | Batch sizes | Wallclock | Baseline throughput at largest batch |
|-----------|-------|------|-------------|-----------|--------------------------------------|
| `gpt/gpt_583m_perf` | mcore-mistral 583M | 1 | 1, 8, 32, 128 | ~5 min | batch=128 → 5171 tok/s, 25 ms/tok |
| `hybrid/hybrid_2b_perf` | mamba hybrid 2B | 1 | 1, 8, 32, 128 | ~8 min | batch=128 → 3583 tok/s, 36 ms/tok |
| `gpt/gpt_16b_perf` | deepseek-style 16B MoE (64 experts, ~2.5B active) | 1 | 1, 8, 32 | ~13 min | batch=32 → 422 tok/s, 76 ms/tok |

## Bootstrap workflow (new test case)

1. Add `tests/performance_tests/test_cases/<family>/<name>/model_config.yaml`.
2. Add `tests/performance_tests/server/model_args/<MODEL>.args` if it's a new model.
3. Submit with `RECORD_BASELINE=1`:

   ```bash
   COG_EXTRA_MOUNTS=... cog submit ... --run-name perf-<name>-bootstrap \
     --command '... "RECORD_BASELINE=1"'
   ```

4. After the run completes, pull the baseline back into your local repo:

   ```bash
   scp <ssh_host>:/lustre/fsw/portfolios/coreai/users/shanmugamr/agents-space/workspaces/megatron_lm/<workspace_hash>/repo/tests/performance_tests/test_cases/<family>/<name>/baseline_values.json \
     tests/performance_tests/test_cases/<family>/<name>/baseline_values.json
   ```

   (`<workspace_hash>` is in the submit JSON at `workspace_sync.workspace_hash`.)

5. Submit a second time without `RECORD_BASELINE` to verify the compare step passes.
6. Commit `model_config.yaml`, `baseline_values.json`, and the args file.

## Refreshing an existing baseline

When a deliberate code change improves perf and the existing baseline is now
artificially low (so future regression detection is weaker than it should be),
refresh:

1. Re-run `RECORD_BASELINE=1` on the affected test case.
2. `scp` the new baseline back.
3. Commit with a message explaining the perf delta and the change that justifies it.
4. Reviewer should sanity-check the new throughput numbers in the PR.

## Investigating a regression

When `compare_to_baseline.py` exits 1:

1. Read the per-metric diff in the slurm stdout — it shows measured vs baseline vs floor/ceiling for every (batch_size, metric) pair.
2. Common false positives:
   - **Run-to-run noise** (~3-5% on the H100s). If the regression is just over the line, re-run once. We currently do not auto-retry — that's intentional, since perf noise this large should be investigated.
   - **Cold caches** (uv venv first-time, sqsh first-import). Pre-warm with `cog prepare-image` + `cog ensure-env` (see run-functional-tests skill, "One-time cog setup → B").
3. Real perf regressions: bisect via `git bisect` against the perf harness. Use the same model config and `SKIP_COMPARE=1` to gather numbers without baseline assertions.

## Error handling reference

| Situation | How to handle |
|-----------|---------------|
| `[run_perf_test] error: server did not become ready within 15 min` | Server crashed during init. Check `<RESULTS_ROOT>/server_logs/server.log` for the actual traceback. Common causes: bad `--load` path, checkpoint format mismatch, TE/CUDA mismatch with the image. |
| `aiohttp.client_exceptions.ServerDisconnectedError` mid-benchmark | The server died under load. Look for OOM in `server.log` (e.g. `--inference-dynamic-batching-buffer-size-gb` too high, or the model + KV cache won't fit the chosen batch size). Reduce `BATCH_SIZES` or buffer size. |
| `port 5000 already in use` | A previous run's server didn't get killed. Re-run — the script's `trap cleanup EXIT` should not leak, but a `kill -9` of the cog job can. SSH to the node and `pkill -f run_dynamic_text_generation_server` if it persists. |
| `assert actual_output == num_output_tokens, ...` | Server cut generation short — likely a sampling config issue. We pass `ignore_eos: true, stop_token_ids: []` to force exactly `NUM_OUTPUT_TOKENS` outputs; if the server ignores those it's a server-side regression. |
| Baseline file missing on a non-bootstrap run | `run_perf_test.sh` exits 3 with a clear error. Run once with `RECORD_BASELINE=1` first. |
| `yq: not found` | Base image is bare NGC PyTorch instead of `mcore_ci_dev`. Switch base image (see TL;DR). |
| `ImportError: MambaSSM is not installed` (hybrid/mamba models) | cog's `uv sync --extra dev --extra mlm` excludes the `ssm` extra, so the cog venv lacks `mamba-ssm`. The runner adds `/opt/venv/lib/python3.12/site-packages` to `PYTHONPATH` for `hybrid_*`/`mamba_*` MODELs to surface the image's preinstalled copy. If you see this error, check that the runner reached the line `adding /opt/venv/lib/python3.12/site-packages to PYTHONPATH (mamba-ssm shim)`. |
| `AssertionError: Flash decode is only applicable to static batching.` | Your model_args/`<MODEL>`.args contains `--flash-decode`, but the perf harness uses dynamic batching. Remove `--flash-decode` from the args file. (We use dynamic batching even for "static-style" inference tests because the throughput-test endpoint requires it.) |

## Defaults reference

| Setting | Value |
|---------|-------|
| Repo | `/Users/shanmugamr/Megatron-LM` |
| Cluster | `cw-dfw` |
| SSH host | `cw-dfw-cs-001-login-01.cw-dfw-cs-001.hpc.nvidia.com` |
| Scratch root | `/lustre/fsw/portfolios/coreai/users/shanmugamr/agents-space` |
| Default `CHECKPOINT_LOAD_PATH` | `/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_mcore/mcore_ci` |
| Default base image | `gitlab-master.nvidia.com/adlr/megatron-lm/mcore_ci_dev:main` |
| Default partition | `batch` |
| Default `--ntasks-per-node` | `1` (script invokes its own torchrun) |
| Required env var | `COG_EXTRA_MOUNTS=...:...` (same as functional tests) |
| Default tolerance | 10% per metric |
| Default warmup iters | 2 |
| Default timed iters | 5 |
| Default ISL × OSL | 512 × 128 |

## What this skill does NOT do

- It does not yet auto-poll for job completion the way `run-functional-tests` does. For now, do a foreground `cog submit` and read the slurm stdout when it returns. (Schedule-wakeup polling can be added later — the perf jobs are short enough that foreground works.)
- It does not run vLLM benchmarks. The mcore server is the only backend supported.
- It does not yet support multi-node inference. All current test cases are single-GPU.
