---
orphan: true
---

# Determinism performance measurements

`perf_breakdown.sh` reports repeated **unprofiled** training timings; its
uncalibrated CI pilot does not enforce performance budgets. Nsight is an
optional separate diagnostic. JSON and Markdown artifacts retain all paired
runs, per-step samples, revisions, GPU identifiers, driver/package versions,
and effective mode settings. A timing pass does not establish determinism.

## Protocol

The default is three deterministic/default pairs, each with 20 warmup steps and
50 measured steps. Every arm starts a fresh process; pair order alternates to
reduce order bias. Both arms use identical model dimensions, input seed,
parallelism, and Python dependencies. Determinism-owned environment variables
are explicitly reset before launch, including in the default arm.
`TRITON_CACHE_AUTOTUNING` is unset in both arms: deterministic SSM uses its
pinned-config fallback. An inherited cache directory or explicit
`TRITON_AUTOTUNE_BLOCK_*` overrides are retained equally and recorded. Cached
SSM autotuning is a separate policy that needs its own controlled benchmark.

The report uses the median iteration time of each run, then ratios within each
pair. A paired bootstrap interval describes run-to-run uncertainty. Fewer than
three pairs or an interval that straddles the limit produces `inconclusive`
(exit 2), not a pass. A confirmed regression or invalid/incomplete timing log
exits 1. A valid pass exits 0. Three pairs are an initial CI budget, not a
guarantee that rare noise is characterized; increase the count when calibrating.

The existing 1.35 deterministic/default limit is retained. The optional
base-to-head comparison defaults to 1.05. Thresholds are configurable through
the Python CLI and should be calibrated on the target recipe and hardware.

The current metric is Megatron's logged iteration duration on its logging rank.
The reader requires a single timing stream and every expected step, and rejects
duplicate/restarted iterations, missing samples, nonfinite values, and zeros.
It does not introduce barriers or synchronization inside the training schedule.

## Run a recipe

Run inside the normal GPU test environment from the checkout being measured:

```bash
uv run --no-sync python tests/performance_tests/shell_test_utils/determinism/benchmark.py \
  --output /tmp/dense-paired --recipe dense --gpus 8
```

Presets are `dense`, `moe`, and `hybrid`, using BF16 mock-data training. The
hybrid preset includes Mamba, attention, and MLP layers. Presets require an even
GPU count because TP is 2; MoE requires a multiple of four for TP=2 and EP=2.
Additional workloads can supply a launcher command
after `--`; it must honor `DETERMINISM_PERF_MODE`, `DETERMINISM_PERF_LOG_DIR`, and
`DETERMINISM_PERF_TRAIN_ITERS`, and emit the same iteration log contract.

To measure the PR against a base source checkout on the **same allocation**:

```bash
uv run --no-sync python tests/performance_tests/shell_test_utils/determinism/benchmark.py \
  --output /tmp/dense-base-head --recipe dense --gpus 8 \
  --base-checkout /path/to/base-checkout
```

The head's launcher and Python environment run both revisions. This isolates
source changes; dependency/image changes need a separately controlled comparison.
The base checkout is read only. `head_overhead` measures head det/default;
`det_regression` measures head det/base det; `default_regression` measures head
default/base default. This catches slowdowns shared by both execution modes.
`base_overhead` is reported for context and does not gate the proposed change.

Outputs must use a fresh directory. Failed attempts are retained, never
overwritten by a retry. Dirty source trees or unavailable GPU provenance yield
an inconclusive result. Optional profiling through
`DETERMINISM_PERF_PROFILE=1 bash .../perf_breakdown.sh OUT LOGS` runs afterward;
the NVTX range table describes host annotations and must not be summed as GPU
kernel time or used as the latency gate.

The H100 dense recipe remains in L1. Broader MoE/hybrid and GB200 rows are
scheduled at nightly cadence within L1, which the current workflow selects.
Labels that bypass cadence also select those rows. The GPU presets and initial
budgets require runtime validation before treating their reports as baselines.

## Kernel leaderboard pilot

The operator pilot covers `bias_swiglu`, `weighted_swiglu`, and
`weighted_squared_relu` from the production fusion modules. It produces six
rows per precision: separate forward and backward measurements for each case.
BF16 and FP32 activations produce twelve rows in total, with 4096 tokens,
hidden size 8192 and FP32 token weights.
This small selection is not a leaderboard of every registered kernel.

```bash
uv run --no-sync python tests/performance_tests/shell_test_utils/determinism/kernel_leaderboard.py \
  --output /tmp/kernel-leaderboard
```

The H100 and GB200 `determinism-kernel-perf.yaml` recipes select this pilot at
nightly cadence in L1 (and when labels bypass cadence). They use one GPU per
measurement and upload `leaderboard.json`, `leaderboard.md`, full per-case
reports, raw samples and logs through the existing assets directory. A cadence
selects eligible jobs; a scheduled or explicit workflow trigger must still run.

Each arm starts a fresh process with the requested policy before operator imports.
CUDA events time all GPU work in the operator call. Compilation and warmup are
excluded; backward also excludes forward/graph construction and upstream-gradient
allocation. `autograd.grad` avoids accumulating leaf gradients. The interval can
include launch gaps; it is operator latency, not a sum of individual device-kernel
durations. The report records input shapes/strides/dtypes, CUDA/driver/GPU/package
versions, source revisions and the uninitialized-memory-fill setting. The shared
`kernel_case.py` adapter generates native-dtype inputs with seed 1234 for both
timing and author replay; backward uses an all-ones upstream gradient in both.
SHA-256 fingerprints include every input's exact bytes, shape, stride, dtype
and gradient requirement, retaining positional `None` arguments. Input hashing
and context capture happen before timing. UUIDs identify the actual timing GPU.

Kernel comparisons default to **report only** (`reported`, exit 0), with no
performance pass claim. Missing/invalid timings fail; insufficient pairs, dirty
source or missing GPU provenance are inconclusive. A failing row stays visible
while later rows still run. A new run needs a fresh output directory.

After calibrating a specific case, phase, dtype, shape and GPU, an explicit budget
can gate that row. Base/head runs use the same driver and allocation:

```bash
uv run --no-sync python tests/performance_tests/shell_test_utils/determinism/benchmark.py \
  --output /tmp/swiglu-backward --kernel-case weighted_swiglu --phase backward --gpus 1 \
  --base-checkout /path/to/base-checkout \
  --max-overhead-ratio <calibrated-ratio> --max-regression-ratio <calibrated-ratio>
```

`--tokens`, `--hidden-size`, and `--dtype` select another case configuration.
Keep H100 and GB200 baselines separate. The GB200 recipe uses
`CUDA_DEVICE_MAX_CONNECTIONS=32`, matching its replay bucket; H100 uses 1.
Historical baseline bundles are described below. Enforcing calibrated
changed-kernel budgets remains separate from publishing measurements.

### Diagnose a timing difference

Use the separate Nsight breakdown described above for attribution. Benchmark
acceptance uses unprofiled event or training-step samples. An integrated
telemetry sampler, compiler-cache inventory and post-timing Torch profiler are
outside this benchmark's scope. Reports marked diagnostic-only remain ineligible
for author performance evidence or baseline publication.

## Join author checks and phase timings

Run the coverage producer and performance driver from the **same clean source
revision**, containing both features. Reports from separate PR heads cannot be
joined. The shared adapter records its own source hash, input fingerprints,
strict Torch policy (including warn-only and memory-fill settings), autocast/TF32/
cuDNN settings, CUDA/driver/GPU details, package versions and environment overrides.
Default and deterministic timing arms must differ only in the declared policy;
old reports lacking this contract remain ineligible for author evidence.

After collecting the coverage report and the twelve-row leaderboard:

```bash
uv run --no-sync python tests/performance_tests/shell_test_utils/determinism/author_evidence.py \
  --coverage /tmp/replay/coverage.json \
  --performance /tmp/kernel-leaderboard/leaderboard.json \
  --revision "$(git rev-parse HEAD)" \
  --output /tmp/author-evidence.json
```

Repeat `--performance` to supply separate `benchmark.json` files. Use one attempt
and one hardware/configuration context per bundle. The join retains every
manifest-required case, requires passing reference/sensitivity/replay evidence
on every replay rank, and requires one matching forward and one backward report.
It rejects duplicates instead of selecting a favorable retry. Raw samples,
medians and paired comparisons are checked again, and every arm of a phase must
use the same GPU UUID. Local activation replay may be replicated across multiple
ranks while timing uses one GPU; this adapter contains no collectives and makes
no inference about distributed performance. Other operators need explicit adapters.

`evidence_complete` and performance status answer different questions. Complete
head-only or unbudgeted timings are `not_gated`, even if det/default overhead met
a configured limit. A performance pass also requires a common base revision for
both phases, complete base/head arms, and passing explicit overhead and revision
limits. This catches slowdowns shared by default and deterministic modes. The
limits still need independently reviewed calibration; their presence is not proof
that the budgets were calibrated.

The CLI writes JSON and Markdown plus hashes of its input artifacts before
returning. Exit 0 means complete, nonfailing evidence; exit 1 means a numerical or
performance failure; exit 2 means missing/incompatible/uncertain evidence. Add
`--require-performance-pass` to also return 2 for an unbudgeted bundle. Existing
reports are not overwritten. Combining compatible coverage/performance CI jobs
and calibrated PR-wide enforcement still needs the combined GPU workflow.

## Publish and verify a historical baseline

The existing CI log uploader already includes the kernel leaderboard, separate
benchmark reports, raw timing files and launcher logs. Download the coverage and
performance artifacts from the same clean source revision and compatible runtime
context. Keep one attempt per directory; do not combine retries or allocations.

```bash
python tests/performance_tests/shell_test_utils/determinism/baseline.py publish \
  --coverage /tmp/coverage-logs/determinism-coverage.json \
  --leaderboard /tmp/perf-logs/kernel-leaderboard/leaderboard.json \
  --revision <full-source-revision> \
  --origin <CI-run-or-execution-reference> \
  --store /shared/determinism-baselines
```

Publication recomputes the author/timing join for every declared requirement and
requires a matching forward and backward report. It checks every separate
`benchmark.json` against the leaderboard and every raw `kernel.json` against its
embedded measurements. All rows must share the source, measurement protocol and
timing GPU. Missing files, duplicate attempts, extra rows, incomplete evidence
and failed numerical or performance checks prevent publication. No missing phase
is filled from a nearby configuration or a different run.

The store must be outside the downloaded artifact directory. Each baseline lives
under the SHA-256 of its `baseline.json` manifest. It contains the original
coverage aggregate, full performance reports, raw timing files and logs, with
relative file names, sizes and hashes. Original runner paths inside reports are
retained as provenance; verification does not require those paths to exist.
The manifest also retains the recomputed author evidence. Identical publication
reuses a verified bundle; existing baselines are never refreshed in place.

After copying or archiving a bundle, verify it against the saved identifier:

```bash
python tests/performance_tests/shell_test_utils/determinism/baseline.py verify \
  /downloaded/baseline --expected-id <saved-manifest-sha256>
```

Verification checks the complete file inventory and recomputes the numerical
evidence/timing join again. Hashes check integrity, not execution authenticity;
the origin is supplied by the publisher and should reference the actual CI run.
The coverage aggregate retains observations and checks; this tool does not
reconstruct it from pytest shards or independently rerun GPU comparisons.

Unbudgeted bundles remain `not_gated`. Publication never assigns limits, promotes
historical timings into a performance pass, or makes cross-allocation timings
equivalent to paired base/head measurements. H100 and GB200 remain separate
contexts. Retain the content-addressed store in durable storage; the ordinary CI
log artifact's retention period alone does not provide permanent publication.

### Compare retained calibration runs

Use pinned, verified, unbudgeted bundles to review variation across runs:

```bash
python tests/performance_tests/shell_test_utils/determinism/calibration.py \
  --baseline /store/first-run <first-manifest-sha256> \
  --baseline /store/second-run <second-manifest-sha256> \
  --output /reports/calibration.json
```

The command rechecks every bundle's files and author/timing join. It rejects
duplicate identifiers, identical measurements republished with different metadata,
and budgeted results whose selection could hide failed measurements. Invalid input
fails the report instead of silently selecting the remaining passing records.

Comparison groups retain exact source revisions, cases, shapes, inputs, precision,
adapter, runtime settings and measurement protocol. H100/GB200, changed drivers,
software versions, base revisions and phase protocols remain separate. Host names,
GPU UUIDs and checkout locations remain in each run's provenance; they do not
split a group. Counts of distinct timing GPUs do not establish statistical
independence or prove separate scheduler allocations.

Cache paths match by default. For runs using different explicit cache directories,
`--compare-cache-locations` groups those locations while preserving the actual
paths in every run. Unset and explicit cache locations still remain separate.
This option does not establish equal cache contents, generated code or dispatch.
Other runtime settings continue to require exact equality.

For captured collectives, select the distinct pinned bundle format explicitly:

```bash
python tests/performance_tests/shell_test_utils/determinism/calibration.py \
  --collective-baseline /store/first-collective-run <first-manifest-sha256> \
  --collective-baseline /store/second-collective-run <second-manifest-sha256> \
  --output /reports/collective-calibration.json
```

Activation and collective inputs cannot be mixed in one invocation. The
collective path rechecks every capture blob, replay/reference record, worker
request and rank timing file before using the paired estimates. It also rejects
reused raw timing arms under different publication metadata, including a
head-only bundle that reuses arms from another base/head bundle. Repeated
publication does not create another observation.

Collective cohorts retain the **entire captured workload on every rank**, not
only the row being displayed. Recipe identity, call order, input and upstream
gradient byte hashes, shape/stride/storage offset, precision, rank membership,
communicator options, source revisions, runtime and measurement settings must
match. A changed peer input or neighboring captured call splits the cohort.
Different event indices and groups remain separate even when their timings are
identical. The tool does not infer equivalence across recipes, reorder events to
make a match, or erase source/tooling differences to create repeated runs.

The recorded hardware inventory and rank-to-device indices participate in the
comparison. Host names, GPU UUIDs and checkout paths remain in run provenance;
they do not split otherwise matching cohorts. Missing runtime/hardware metadata,
inconsistent rank devices and ambiguous inventory entries fail calibration.
The captured logical groups and hardware inventory do not prove physical fabric
equivalence. Distinct device assignments do not prove independent allocations.

The same `--compare-cache-locations` opt-in is available for collectives. JSON
retains the original paths, full workload metadata, baseline identifiers, each
run's paired ratios/intervals and arm medians. Markdown shows every event/group,
including singleton cohorts. Hardware identifiers and environment values are
part of the detailed provenance; retain and share reports according to the
same access rules as their input artifacts.

Each bundle contributes one median paired ratio. JSON retains each run's paired
ratios and within-run intervals; Markdown shows the minimum, median, maximum and
observed range across runs. Default/deterministic overhead stays separate from
paired head/base regression. Raw event samples are never pooled to manufacture
more independent observations, and the observed range is not a confidence or
prediction bound. One-run groups are explicitly marked in JSON.

The result remains `report_only` / `not_gated`. The tool does not choose budgets,
approve promotion or replace production-recipe measurements. Inputs are selected
published bundles, not a complete survey of executions. Preserve unsuccessful
attempts alongside them when reviewing calibration and unexplained variation.

CPU contract tests use explicitly synthetic GPU metadata and timings. They do
not establish hardware latency, correctness, replay coverage or usable budgets.

## Time actual captured collectives

`benchmark_collectives.py` measures the six direct TP/SP mappings supported by
the [recipe capture adapter](recipe-coverage.md), using its actual rank-local
input and upstream-gradient bytes. This optional adapter needs the capture and
replay producer (#7260/#7317) and the early startup API (#7419). It runs on the
capture's original single-node allocation; a capture from a different physical
GPU assignment cannot supply these measurements.

After capture and replay have completed at the clean head revision, run the
parent outside `torchrun`, from that head checkout:

```bash
python tests/performance_tests/shell_test_utils/determinism/benchmark_collectives.py \
  --capture /results/recipe/capture \
  --evidence /results/recipe/coverage.json \
  --base-checkout /checkouts/base \
  --output /results/recipe/collective-timing \
  --pairs 3 --warmup 20 --steps 50
```

The capture must have matching passing replay, reference and sensitivity
observations on every rank for each selected event. `--event-indices 0 1`
selects explicit entries from the manifests; omission measures every event.
A `forward` event measures forward, and its `forward_backward` event measures
backward with the captured upstream gradient. The latter's forward graph setup
is outside timing. Both phases restore the input and gradient before every
sample, including their original strides and storage offsets.

Each source/policy arm starts fresh processes and process groups; pair order
alternates. Explicit group membership, stream priority, NCCL options, input
hashes, UUIDs and non-policy settings stay fixed. The deterministic arm retains
the captured policy. The default arm disables Torch/cuDNN deterministic
selection, removes NCCL_ALGO/CUBLAS_WORKSPACE_CONFIG and enables the existing
default TE/Mamba/causal-convolution settings. Memory fill, TF32, Triton caching
and other NCCL overrides stay fixed and are recorded. Autocast and cuDNN
benchmark captures currently require a different adapter. The head's timing
helpers run both sources; production mapping imports are checked against the
selected checkout. Matching correctness evidence applies to deterministic head
execution, not to default or baseline accuracy.

Each communicator is initialized with a group barrier before operator warmup.
The report retains requested options and the actual settings before and after
initialization. PyTorch resolves NCCL's undefined `blocking` field to `0` or `1`
according to the explicit configuration/environment; only that declared
resolution is accepted. Stream priority, CTA limits and every other option stay
exact. Measured signatures record the resolved settings alongside the original
capture signature. See [ProcessGroupNCCL initialization](https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp).

Rank alignment barriers, input restoration, graph setup and warmup are excluded
from CUDA-event intervals. These direct synchronous c10d mappings join NCCL
completion to the calling stream, so its end event includes completion; see
[PyTorch's collective stream semantics](https://docs.pytorch.org/docs/2.14/distributed.html#synchronous-and-asynchronous-collective-operations).
Intervals include host launch gaps and arrival skew. They measure isolated
operator phases, including identity phases, not pure NCCL kernel durations,
overlap or whole-model throughput.

Every arm retains raw per-rank JSON and a log. For each event and captured
group, the report takes the maximum rank latency at each aligned sample index,
then its median. Paired bootstrap intervals use these independent process-pair
ratios; ranks and samples are never pooled as independent repetitions. JSON and
Markdown retain each group's result, head overhead and optional base/head
regressions. Incomplete ranks, changed source/runtime, modified captures or
missing accuracy evidence fail and leave the partial attempt for diagnosis.

Ratios are report-only unless explicit reviewed `--max-overhead-ratio` and/or
`--max-regression-ratio` limits are supplied; fewer than three pairs remain
inconclusive. Local-activation leaderboard and calibration reports retain their
own format. Use the explicit collective publication and artifact-consumer paths
below for this report kind. Reviewed baseline promotion and production-recipe
performance acceptance remain separate work.

### Run the collective CI producer

The H100 and GB200 nightly `determinism_collective_perf` recipes run the complete
capture/replay/reference/timing pipeline on one node (eight H100 or four GB200
ranks). They require the collective capture/replay and early startup dependencies.
An ordinary PR or merge-group cadence does not select this pilot; the existing
explicit cadence bypass can select it. The existing workflow stamps the actual
producer outcome, uploads its logs, and selects the corresponding
`--collective-platform` CPU consumer. Missing or unsuccessful selected producers
fail artifact verification; existing activation selection remains independent.

To run the same producer in an allocated, clean source checkout:

```bash
python tests/performance_tests/shell_test_utils/determinism/collective_pipeline.py \
  --output /results/collective-performance --gpus 8
```

Use `--gpus 4` for the GB200 recipe. Launch the parent outside torchrun. The
producer launches its own fresh worker groups, preserves every stage log and
requires a fresh output directory. A failed capture, replay, accuracy check,
recipe join or timing arm stops the pipeline and retains the incomplete attempt.
The final CPU verification requires the complete matrix before marking the
producer complete.

This bounded synthetic workload covers six direct TP/SP mappings in FP32/BF16,
forward/backward, with explicit rank-local inputs/upstream gradients, nonzero
offsets and noncontiguous three-dimensional tensors. Pair-sized and world-sized
groups exercise distinct priority/CTA options, producing 96 events per rank.
The default three fresh-process policy pairs use 20 warmup and 50 measured
samples per event. CI measures current-head deterministic/default overhead;
base/head comparisons remain available through the separate
`benchmark_collectives.py --base-checkout` entrypoint. No budgets are imposed by
this pilot, and complete results remain `not_gated`. It does not establish
production recipe, multi-node, overlap or full-state/restart acceptance. Recipe
and workflow implementation alone does not establish a successful protected CI
run.

### Publish and transport captured collective evidence

Publish a complete capture, its matching replay/accuracy report and timing tree:

```bash
python tests/performance_tests/shell_test_utils/determinism/baseline.py publish-collectives \
  --capture /results/recipe/capture \
  --coverage /results/recipe/coverage.json \
  --benchmark /results/recipe/timing/benchmark.json \
  --revision <measured-head-revision> --origin <original-run-reference> \
  --store /store/collectives

python tests/performance_tests/shell_test_utils/determinism/baseline.py verify \
  /downloaded/collective-bundle --expected-id <published-baseline-id>
```

The distinct `determinism_collective_baseline` format retains original capture
manifests and tensor blobs, coverage evidence, per-rank samples, worker requests,
logs and timing reports. It checks every captured event, its all-rank replay,
reference and sensitivity evidence, source/runtime and NCCL options; a selection
of favorable events cannot form a complete baseline. It recomputes aligned group
maxima, medians and paired comparisons from the separate rank files. Missing,
changed or undeclared files, duplicate attempts, diagnostic records, failed
budgets and inconclusive measurements are rejected. Unbudgeted results stay
`not_gated`.

Publication is immutable and verifies the copied bundle before exposing its
content identifier. Repeated publication of identical content reuses the same
identifier. Verification works on CPU without Torch, GPUs or the original
checkout paths; recorded commands are data and are never executed. Supply the
original identifier after transport to bind verification to the intended bundle.
The verifier checks recorded numerical evidence and captured-byte integrity; it
does not rerun GPU numerical references or establish model/restart equality.
Tensor blobs and logs may contain recipe data, so select their destination and
access policy explicitly.

The CPU CI consumer supports a separate `determinism_collective_perf` producer.
Its uploaded tree must contain exactly one dataset with sibling `capture/`,
`coverage.json` and `timing/benchmark.json` paths. Stamp its actual outcome using
`ci_artifacts.py stamp --test-case determinism_collective_perf` and the same
repository, checked-out revision, run, attempt and platform arguments used below.
After downloading separately named artifacts, select collective platforms
explicitly:

```bash
python tests/performance_tests/shell_test_utils/determinism/ci_artifacts.py collect \
  --artifacts /downloaded/logs --output /reports/collective-candidates \
  --repository NVIDIA/Megatron-LM --revision <measured-head-revision> \
  --run-id <run-id> --attempt <run-attempt> \
  --collective-platform dgx_h100 --collective-platform dgx_gb200
```

Each selected platform requires one successful collective artifact containing
matching capture/replay/timing evidence from its original allocation. Activation
artifacts cannot substitute for it. `--platform` continues to select the existing
activation producers; both kinds can be requested together and are reported
separately. A missing, failed, stale, wrong-platform or ambiguous selected producer
fails verification. The derived bundles retain the source records and unbudgeted
status. The nightly producer recipes and workflow selection described above
provide the automatic path. Protected CI acceptance, reviewed performance limits
and durable baseline promotion remain separate rollout requirements.

## Consume CI artifacts

The shared GitHub test action stamps only the latest kernel replay bucket and
kernel performance pilot. Their existing log artifacts retain the repository,
actual checked-out revision, run ID, run attempt, platform, test case and raw
producer outcome/exit code. Their names include the run and attempt so a rerun
does not silently consume an earlier attempt. Other log names are unchanged.

When the actual integration matrix selects `determinism_kernel_perf`, the CPU
`cicd-determinism-baselines` job downloads those current-attempt artifacts into
separate named directories. For each selected platform it requires exactly one
successful replay producer and one successful timing producer, checks all source
identities and the observed GPU, then invokes the baseline publisher above.
Failed/missing producers, mismatched sources or duplicate uploads cannot pass.
A lone upload retry is accepted; two available copies remain ambiguous and must
be investigated instead of selecting the more favorable report. Data from a
platform whose performance pilot was not selected receive no verification credit.
The consumer executes the checked-out verifier, never code from log artifacts.

The job retains `report.json`, `report.md` and each successfully verified scoped
bundle in `determinism-baselines-<run>-a<attempt>`, including diagnostic reports on
failure. The final CI gate waits for artifact verification. A complete artifact
check does not turn `not_gated` timing into a performance pass. The consumer needs
the measured coverage producer from the companion coverage change (#7317);
registration-only reports cannot substitute for it.

To reproduce the consumer after downloading the original named artifacts:

```bash
python tests/performance_tests/shell_test_utils/determinism/ci_artifacts.py collect \
  --artifacts /tmp/determinism-inputs --output /tmp/determinism-candidates \
  --repository NVIDIA/Megatron-LM --revision <full-source-revision> \
  --run-id <run-id> --attempt <run-attempt> --platform dgx_h100 --platform dgx_gb200
```

Pass only platforms selected by that run's actual performance matrix. The output
directory must be empty and outside the downloaded inputs. A partial rerun that
does not rerun both required producers cannot reuse older evidence and remains
unverified. Run both producers in the new attempt to obtain a complete candidate.
GitHub artifact retention is temporary; durable storage, reviewed promotion and
calibrated performance budgets remain separate responsibilities. CPU transport
tests do not establish successful protected CI execution or GPU acceptance.

CPU tests:

```bash
PYTHONPATH=. python -m pytest --confcutdir=tests/unit_tests/determinism_reporting \
  tests/unit_tests/determinism_reporting/test_paired_performance.py
```

### CI rollout and retry behavior

Land this PR after #7419, #7317, and #7260. The operator pilots intentionally fail
on missing or incompatible evidence, including on label-triggered runs. Ordinary
unlabelled PR cadence does not select the operator pilots. The training benchmark
wrapper uses `--report-only` until per-recipe budgets are calibrated: confidence
intervals and limit violations remain in the report, while malformed measurements
and missing provenance still fail. Three pairs have an empirical interval bounded
by the minimum and maximum observed ratios; use more pairs for calibration.

Each training/kernel attempt has a separate output directory. A byte-identical
GitHub upload retry with the same producer ID is deduplicated with both receipts;
different contents or independent producers remain ambiguous and cannot pass.
Coverage stamps select the dev environment. Stamping failures do not fail ordinary
unit jobs, but a selected pilot's consumer still requires the artifact.

Full reporting tests require CPU PyTorch and run with:

```bash
python -m pytest --confcutdir=tests/unit_tests/determinism_reporting tests/unit_tests/determinism_reporting
```

Runtime/driver/SKU matching stays strict so timing is joined to evidence from the
same configuration. Heterogeneous nodes need their own coverage producers.

The kernel pilot receives a 150-minute action timeout inside a 180-minute job
budget; other integration jobs keep their existing limits. Actual calibration
wall time still needs current CI measurement.

Replay reports distinguish requested side-stream contention from effective
contention. With `CUDA_DEVICE_MAX_CONNECTIONS=1`, the serialized replay remains
eligible only for that same runtime policy and is not evidence of concurrent
stream stress.
