# Megatron Lite Local Validation

MLite uses the standard Megatron-LM development/CI container and exposes one
local-validation entry:

~~~bash
experimental/lite/tests/run_tests.sh
~~~

With no arguments, the script discovers and runs the complete test set for the
visible hardware. It also accepts one or more test directories or test files:

~~~bash
experimental/lite/tests/run_tests.sh experimental/lite/tests/runtime
experimental/lite/tests/run_tests.sh \
  experimental/lite/tests/model/test_qwen_config_unit.py \
  experimental/lite/tests/primitive/test_parallel_unit.py
~~~

Relative paths are interpreted from the directory where the script is invoked.
Targets must stay under model/, primitive/, runtime/, or examples/. The script
accepts paths, not arbitrary pytest options.

An explicit CPU-only subset does not probe CUDA or require a supported GPU
topology. An explicit GPU subset accepts any positive number of homogeneous
Hopper or Blackwell GPUs, as long as enough GPUs are visible for every selected
test. Exact profile topology is required only for the no-argument workflow.

## Development Container

Build the standard dev image from the repository root. The Dockerfile has an
internal-only jet stage, so local and public builds must select main explicitly:

~~~bash
docker build \
  --target main \
  --build-arg FROM_IMAGE_NAME="$(cat docker/.ngc_version.dev)" \
  --build-arg IMAGE_TYPE=dev \
  -f docker/Dockerfile.ci.dev \
  -t megatron-lm:local \
  .
~~~

The image uses the repository-level pyproject.toml and uv.lock; MLite does not
maintain a separate container dependency set. Mount the checkout so validation
uses the current source:

~~~bash
docker run --rm -it --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$PWD:/workspace/Megatron-LM" \
  -w /workspace/Megatron-LM \
  megatron-lm:local \
  bash
~~~

Run validation explicitly inside the container:

~~~bash
experimental/lite/tests/run_tests.sh
~~~

The test workflow uses repository code, synthetic inputs, and temporary local
output. After the image is built, it does not require network access, shared
storage, credentials, or a particular scheduler.

## Default Hardware Profiles

| Visible hardware | No-argument behavior |
| --- | --- |
| 8 homogeneous SM90 GPUs (H100 reference) | Runs CPU tests and tests whose minimum architecture is Hopper |
| 4 homogeneous SM100 GPUs (GB200 reference) | Runs only tests whose minimum architecture is Blackwell |
| Anything else, including mixed capabilities | Reports unsupported hardware and exits 2 |

No Blackwell-only tests are registered yet, so the default GB200 run reports
NOT_RUN and exits 3. When an explicit subset is requested on Blackwell, Hopper
tests are allowed because Blackwell satisfies their minimum architecture.

## Discovery and Layout

The entry automatically discovers normal pytest files below:

- experimental/lite/tests/model/
- experimental/lite/tests/primitive/
- experimental/lite/tests/runtime/
- experimental/lite/tests/examples/

Adding a normal test does not require editing run_tests.sh or _test_harness/.
Stable roots, exclusions, pytest arguments, and hardware profiles are defined
once in _test_harness/runner.py; tests and suites are never enumerated there.
VERL example tests named test_verl_*.py remain outside this workflow and belong
to a downstream or future optional validation flow.

## Execution Markers

Marker registration, validation, defaults, and architecture ordering are
centralized in _test_harness/markers.py. pytest --markers displays the same
contracts, and the workflow uses --strict-markers.

- No gpus marker means a CPU test.
- pytest.mark.gpus(count) requests GPUs and defaults to minimum Hopper support.
- pytest.mark.gpus(count, min_architecture="blackwell") declares a
  Blackwell-or-newer requirement.
- pytest.mark.env(NAME="value") sets an allowlisted per-test environment
  variable. A value of None explicitly unsets it. The harness sanitizes managed
  variables before applying these overrides.
- pytest.mark.timeout(seconds=N) sets a positive per-suite timeout. Without it,
  the timeout is 1800 seconds.
- pytest.mark.optional excludes a test from the no-argument workflow. Explicit
  path selection includes optional tests.

Module, class, function, and parameter markers are supported. A more specific
marker overrides a broader default. CUDA_DEVICE_MAX_CONNECTIONS is currently
the only allowlisted per-test environment variable.

CPU tests with the same execution contract share one pytest process. GPU tests
with the same execution contract and source file share one isolated pytest or
torchrun process. The harness hides CUDA from CPU tests, limits each GPU process
to its requested device count, and rejects tests that request incompatible
hardware. Distributed rank reports must be present and agree on test outcomes.

## Result Policy

The standard development container provides the dependencies required by the
MLite tests. The harness does not pre-import every package: collection and the
selected tests report missing or broken dependencies at the point of use.

A collection error, failed test, unexpected skip or XPASS, timeout, missing rank
report, or inconsistent rank result fails the run. Repeated warning summaries
are suppressed in the standard output.

| Code | Meaning |
| --- | --- |
| 0 | The selected tests completed successfully |
| 1 | Dependency, collection, test, timeout, or result-integrity failure |
| 2 | Invalid target, unsupported default topology, or insufficient subset hardware |
| 3 | The hardware/profile is valid but no selected test is runnable |

The final summary contains only the source revision with a dirty suffix when
applicable, safe software and GPU metadata, suite durations, result counts, and
overall status. It does not print hostnames, container identifiers, environment
values, credentials, or infrastructure paths.
