# Megatron-LM Tests

## A2A Overlap Test Isolation

The two-chunk schedule tests in `unit_tests/a2a_overlap/test_schedule_chunk_1f1b.py`
reuse one parallel topology per class (and per TP size for padding-mask tests).
This bounds communicator and backend initialization across the parameter matrix.
MCore's
`destroy_model_parallel()` clears NCCL group references without unregistering those
groups from PyTorch, so each topology scope explicitly destroys the groups it
creates. The fixture preserves the default group and pre-existing groups,
and finalizes NCCL EP before
destroying its borrowed communicator. Cached flex buffers are also released before
their groups. A per-case fixture collects model/plan cycles after the
test helpers return; topology teardown checks that no new registered groups remain. This keeps
communication memory bounded across the full dispatcher, precision, MTP, and
recompute matrix.

## CUDA Graph Test Isolation

Reference outputs retained across capture must be detached after eager backward so
their autograd nodes do not retain warmup-stream state. Capture errors restore the
process-wide warmup/capture flags before propagating the error; otherwise later DDP
backwards can incorrectly suppress gradient accumulation. The MFSDP v2 parity test
records actual graph `replay()` calls and checks both their order and training loss,
independent of profiler support for CUDA runtime versus driver launch events.

## Updating Functional Test Golden Values

When adding new functional tests, it may be necessary to update the golden values used to verify if the test is
passing as expected.

1. Add the new functional test case with the scope set to `mr-github`
2. Open a PR with the new test. Ensure the label `Run functional tests` is added
3. Run the PR CI tests
4. Run the script to download golden values from a Github CI run
    a. Ensure click, requests, and python-gitlab are installed in your environment
    b. Ensure a Github access token is set as an environment variable `GITHUB_TOKEN`
    c. Run the script `python tests/test_utils/python_scripts/download_golden_values.py --source github --pipeline-id <github-workflow-run-id>`
    d. Optionally pass in `--only-failing` to only download golden values for failing tests only
    e. Ensure you are only checking-in golden values for tests are you updating

### Golden-value precision

New training golden files preserve the full scalar precision available in TensorBoard and mark each metric with
`"value_precision": "full"`. Deterministic checks compare these values without rounding. Approximate checks round
both the golden and actual values to five decimal places before applying their configured tolerances.

Existing golden files do not need to be regenerated. A metric without `value_precision` is treated as a legacy
five-decimal golden, so deterministic checks retain their previous behavior until that golden is deliberately
regenerated.

The Github CI infra may not be appropriate for Perf tests. Perf tests may be more appropriate for nightly jobs on other infra.
