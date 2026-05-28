---
name: build-and-dependency
description: Container-based dev environment setup and dependency management for Megatron-LM. Covers acquiring and launching the CI container, uv package management, and updating uv.lock.
license: Apache-2.0
when_to_use: Adding, removing, or updating a dependency; editing pyproject.toml or uv.lock; uv.lock merge conflict; setting up a dev environment; pulling or building the CI container; container build errors; uv errors; 'how do I install', 'uv sync fails', 'ModuleNotFoundError'.
---

# Build & Dependency Guide

The core principle: **build and develop inside containers** — the CI container
ships the correct CUDA toolkit, PyTorch build, and pre-compiled native extensions
(TransformerEngine, DeepEP, …) that cannot be reproduced on a bare host.

---

## Why Containers

Megatron-LM depends on CUDA, NCCL, PyTorch with GPU support, TransformerEngine,
and optional components like ModelOpt and DeepEP. Installing these on a bare host
is fragile and hard to reproduce. The project ships Dockerfiles that pin every
dependency.

**Use the container as your development environment.** This guarantees:

- Identical CUDA / NCCL / cuDNN versions across all developers and CI.
- `uv.lock` resolves the same way locally and in CI.
- GPU-dependent operations (training, testing) work out of the box.

---

## dev vs lts

Two image variants exist, each with its own Dockerfile, selected by the
`container::lts` PR label:

| Variant | Base image pin | Dockerfile | Where deps live | When used |
|---------|---------------|------------|-----------------|-----------|
| **`dev`** | `docker/.ngc_version.dev` | `docker/Dockerfile.ci.dev` | `pyproject.toml` `dev` extra (uv-resolved) | Default — CI, local development, most PRs |
| **`lts`** | `docker/.ngc_version.lts` | `docker/Dockerfile.ci.lts` | `docker/lts/requirements.txt` (pinned, sourced from main's `uv.lock` at AUT-479) | Stability testing; excludes ModelOpt and other bleeding-edge extras |

> LTS deps used to live in `[project.optional-dependencies].lts` in
> `pyproject.toml`. They were moved into `docker/lts/requirements.txt` so
> `pyproject.toml` can host meaningful module-level extras without colliding
> with the LTS pin set. To bump an LTS dependency, edit the version in
> `docker/lts/requirements.txt` and rebuild `docker/Dockerfile.ci.lts`.

**Use `dev` for everything unless you have a specific reason to test `lts`.**
CI runs `dev` by default; attach `container::lts` to a PR only when verifying
compatibility with the stable stack (e.g. a dependency upgrade that must not
break LTS users). The `@pytest.mark.flaky_in_dev` marker skips tests in the
`dev` environment; `@pytest.mark.flaky` skips them in `lts`.

---

## Step 1 — Acquire an Image

**Option A — NVIDIA-internal: pull a CI-built image**

> ⚠️ Requires access to the internal GitLab instance.
> See @tools/trigger_internal_ci.md for setup (adding the git remote, obtaining a token).

The internal GitLab CI publishes images to its container registry.
Derive the registry host from your configured `gitlab` remote — the same
host you use for `trigger_internal_ci.py`:

```bash
# Derive host from your 'gitlab' remote:
GITLAB_HOST=$(git remote get-url gitlab | sed 's/.*@\(.*\):.*/\1/')

docker pull ${GITLAB_HOST}/adlr/megatron-lm/mcore_ci_dev:main
```

**Option B — Build from scratch (works for everyone)**

> ⚠️ `Dockerfile.ci.dev` has two stages: `main` and `jet`. The `jet` stage
> requires an internal build secret and will fail without it. Always pass
> `--target main` to stop at the public stage.

```bash
# dev image (default)
docker build \
  --target main \
  --build-arg FROM_IMAGE_NAME=$(cat docker/.ngc_version.dev) \
  --build-arg IMAGE_TYPE=dev \
  -f docker/Dockerfile.ci.dev \
  -t megatron-lm:local .

# lts image (uses a dedicated Dockerfile; no IMAGE_TYPE arg)
docker build \
  --target main \
  --build-arg FROM_IMAGE_NAME=$(cat docker/.ngc_version.lts) \
  -f docker/Dockerfile.ci.lts \
  -t megatron-lm:local-lts .
```

Which image variant is used is controlled by the PR label `container::lts`;
absent that label, `dev` is used.

---

## Step 2 — Launch the Container

**Option A — Local Docker runtime**

```bash
docker run --rm --gpus all \
  -v $(pwd):/workspace \
  -w /workspace \
  megatron-lm:local \
  bash -c "<your command>"
```

**Option B — Slurm cluster (for those without a local Docker runtime)**

NVIDIA clusters typically use [Pyxis](https://github.com/NVIDIA/pyxis) +
[enroot](https://github.com/NVIDIA/enroot). Request an interactive session:

```bash
srun \
  --nodes=1 --gpus-per-node=8 \
  --container-image megatron-lm:local \
  --container-mounts $(pwd):/workspace \
  --container-workdir /workspace \
  --pty bash
```

For clusters that require a `.sqsh` archive first:

```bash
enroot import -o megatron-lm.sqsh dockerd://megatron-lm:local
srun \
  --nodes=1 --gpus-per-node=8 \
  --container-image $(pwd)/megatron-lm.sqsh \
  --container-mounts $(pwd):/workspace \
  --container-workdir /workspace \
  --pty bash
```

---

## Dependency Management

Dependencies are declared in `pyproject.toml`. The venv lives at `/opt/venv`
inside the container (already on `PATH`).

> **All `uv` operations must be run inside the container.**
> Never run `uv sync` / `uv pip install` on the host.

### uv Dependency Groups

| Group | Purpose |
|-------|---------|
| `training` | Runtime training extras |
| `dev` | Full dev environment (TransformerEngine, ModelOpt, …) |
| `test` | pytest, coverage, nemo-run |
| `linting` | ruff, black, isort, pylint |
| `build` | Cython, pybind11, nvidia-mathdx |

> The previous `lts` extra has been emptied. LTS deps are pinned in
> `docker/lts/requirements.txt` rather than `pyproject.toml`. Do not add new
> packages under `[project.optional-dependencies].lts`.

Install commands (inside the container):

```bash
# Full dev + test environment
uv sync --locked --group dev --group test

# Linting only
uv sync --locked --only-group linting
```

The LTS environment is reproduced by building `docker/Dockerfile.ci.lts`
end-to-end; there is no `uv sync`-only equivalent because the LTS deps no
longer live in `pyproject.toml`. The LTS top-level pin set is in
`docker/lts/requirements.txt`; bump versions there and rebuild the image.

Several dependencies are sourced directly from git (TransformerEngine, nemo-run,
FlashMLA, Emerging-Optimizers, nvidia-resiliency-ext). The locked `uv.lock` file
pins exact revisions; update it with `uv lock` when changing `pyproject.toml`.

### Adding a New Dependency

Follow this three-step workflow:

1. **Acquire a container image** — see [Step 1](#step-1--acquire-an-image) above.
2. **Launch the container interactively** — see [Step 2](#step-2--launch-the-container) above.
3. **Update the lock file inside the container**, then commit it:

   ```bash
   # Inside the container:
   uv add <package>          # adds to pyproject.toml and resolves
   uv lock                   # regenerates uv.lock
   # Exit the container, then on the host:
   git add pyproject.toml uv.lock
   git commit -S -s -m "build: add <package> dependency"
   ```

### Resolving a merge conflict in uv.lock

`uv.lock` is machine-generated; never resolve conflicts manually. Instead:

```bash
git checkout origin/main -- uv.lock   # take main's version as the base
# then inside the container:
uv lock                               # re-resolve on top of your pyproject.toml changes
```

---

## Common Pitfalls

| Problem | Cause | Fix |
|---------|-------|-----|
| `uv sync --locked` fails | Dependency conflict or stale `uv.lock` | Re-run `uv lock` inside the container and commit updated lock |
| `ModuleNotFoundError` after pip install | pip installed outside the uv-managed venv | Use `uv add` and `uv sync`, never bare `pip install` |
| `uv: command not found` inside container | Wrong container image | Use the `megatron-lm` image built from `Dockerfile.ci.dev` |
| `No space left on device` during uv ops | Cache fills container's `/root/.cache/` | Mount a host cache dir via `-v $HOME/.cache/uv:/root/.cache/uv` |
| `docker build` fails with secret-related error | `Dockerfile.ci.dev` has a `jet` stage that requires an internal secret | Add `--target main` to stop before the `jet` stage |
| `access forbidden` when pulling | Registry URL includes an explicit port (e.g. `:5005`) | Use `${GITLAB_HOST}/adlr/...` with no port — the sed extracts the hostname only |
