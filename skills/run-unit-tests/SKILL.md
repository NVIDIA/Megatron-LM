---
name: run-unit-tests
description: How to run Megatron-LM unit tests on a GPU node. Covers environment setup with uv, launching tests through torch.distributed.run, marker filters, CI parity, and common gotchas. All Megatron-LM unit tests initialize a torch distributed group, so every invocation requires GPU access and is launched through torch.distributed.run.
when_to_use: Running unit tests; debugging a unit test failure; reproducing a CI test failure locally; setting up the unit-test environment; invoking pytest on Megatron-LM; 'run unit tests', 'pytest fails', 'reproduce test failure'.
---

# Run Megatron-LM Unit Tests

## Prerequisites

- Megatron-LM checked out; `pyproject.toml` and `uv.lock` present at the repo root.
- `uv` installed (https://docs.astral.sh/uv/).
- A working PyTorch + CUDA environment matching the repo's required versions. The supported NGC PyTorch base images are pinned in `docker/.ngc_version.dev` (development stack) and `docker/.ngc_version.lts` (long-term-support stack).
- A node with one or more visible NVIDIA GPUs. All Megatron-LM unit tests initialize a torch distributed group, so every invocation requires GPU access and is launched through `torch.distributed.run`.

## Set up the environment

From the repo root:

```bash
uv sync --extra training --extra dev
```

Use `--extra lts` for the long-term-support image stack. This materializes a `.venv` with the Megatron-LM dependencies. All commands below run through `uv run`, which executes inside that venv.

## Run the unit suite

```bash
uv run python -m torch.distributed.run --nproc-per-node N -m pytest -q tests/unit_tests
```

Set `--nproc-per-node N` to the GPU count your tests need. Use `8` (or whatever your node provides) for distributed-feature tests; `1` is enough for tests that do not fan out beyond rank 0 but still require an initialized process group.

## Run a specific test

Single file:

```bash
uv run python -m torch.distributed.run --nproc-per-node 8 -m pytest -q \
  tests/unit_tests/models/test_gpt_model.py
```

Single test:

```bash
uv run python -m torch.distributed.run --nproc-per-node 8 -m pytest -q \
  tests/unit_tests/models/test_gpt_model.py::TestGPTModel::test_constructor
```

Filter by name substring:

```bash
uv run python -m torch.distributed.run --nproc-per-node 8 -m pytest -q \
  tests/unit_tests -k optimizer
```

## Marker filters

Megatron-LM uses pytest markers `internal`, `flaky`, and `flaky_in_dev` (declared in `pyproject.toml`). To exclude unstable tests during development:

```bash
uv run python -m torch.distributed.run --nproc-per-node 8 -m pytest -q \
  tests/unit_tests -m "not flaky and not flaky_in_dev"
```

Experimental tests are gated behind a flag:

```bash
uv run python -m torch.distributed.run --nproc-per-node 8 -m pytest -q \
  tests/unit_tests --experimental
```

## CI parity

The CI bucket runner is `tests/unit_tests/run_ci_test.sh`. It expands buckets, applies marker filters by environment, and writes coverage. Use it to reproduce a CI bucket failure locally; otherwise prefer the direct `torch.distributed.run` invocations above.

## Common gotchas

- The default pytest config in `pyproject.toml` sets `addopts = --durations=15 -s -rA`. That means stdout is not captured (`-s`), the slowest 15 tests are reported, and a short summary of all outcomes is printed at the end. Override with explicit pytest flags if you need different defaults.
- Because `addopts` includes `-s`, stdout from every rank interleaves in the same terminal during multi-rank runs. When debugging a specific rank, override with `--capture=fd` so each process captures its own stdout.
- `tests/unit_tests/conftest.py` looks for test data under `/opt/data` and attempts a download if it is missing. If you are running outside the canonical container, supply the data manually or skip data-dependent tests.
