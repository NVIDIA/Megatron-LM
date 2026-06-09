# P4 Async Scheduling Validation

Date: 2026-06-09 UTC

Validated code commit before this note: `5644230c0973c4fd010a853347cfecd711b78765`.

## Final Checks

- `python -m py_compile megatron/core/inference/async_transaction.py megatron/core/inference/text_generation_controllers/async_decode_coordinator.py megatron/core/inference/text_generation_controllers/text_generation_controller.py megatron/core/inference/contexts/dynamic_context.py`
- `uv run pytest -q tests/unit_tests/inference/test_async_transaction_architecture.py`
  - 3 passed
  - `EXPECTED_VIOLATIONS = {}`
- `uv run pytest -q tests/unit_tests/inference/test_async_scheduling_compact.py`
  - 133 passed
- `uv run pytest -q tests/unit_tests/inference/contexts/test_dynamic_context.py`
  - 84 passed, 4 skipped

## Nano Coordinator Parity

The DFW direct coordinator Nano parity workflow was not run in this Slurm job.
The checkout includes `hybrid_dynamic_inference_tp1_ep8_nanov3_chunked_prefill`,
but its recipe is marked `mr-broken` / `mr-github-broken` with a documented
hang, and this job exposes one H100 while that NanoV3 EP=8 case requires eight
GPUs. The required CI checkpoint path `/mnt/artifacts` is also absent here.

No local script was found that runs the requested three-branch comparison
(`main` async off, PR 4604/weekend async on, and P4 async on), so
`branch_parity_diffs` was not produced.

Closest repo-owned H100 perf harness once an 8-GPU CI allocation and artifacts
are available:

```bash
GPUS_PER_NODE=8 bash ./tests/performance_tests/shell_test_utils/run_perf_test.sh \
  CONFIG_PATH=tests/performance_tests/test_cases/hybrid/hybrid_nanov3_3b_perf/model_config.yaml \
  CHECKPOINT_LOAD_PATH=/mnt/artifacts/ \
  RESULTS_ROOT=<assets_dir>/perf_results
```

## HSG Inference Bench

HSG inference-bench was not run from this job because the HSG wrapper is not in
this checkout and the job has neither the required checkpoint artifacts nor an
8-GPU allocation for NanoV3 EP=8.

Required HSG command to run in the proper HSG environment:

```bash
MODEL=nanov3 OSL=32768 BATCH_SIZES="1 1" \
  inference-bench --inference-dynamic-batching-async-scheduling
```

Expected result: no correctness failure and no regression from the current
`4.14-4.16 ms/output token` baseline.
