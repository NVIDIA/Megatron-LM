# CLI Guide

Agent-facing reference for `cog` — a CLI for running Megatron-LM workloads on Slurm clusters.

## Install

Preferred:

```bash
./scripts/install-dev-cli.sh
```

Portable wheel install:

```bash
./scripts/install-wheel.sh
```

Repo-local fallback:

```bash
uv run cog --help
```

After install, agents should call the real CLI:

```bash
cog ...
```

## Contract

- Most commands print a single JSON object to stdout. Agents should parse it, not scrape text.
- Every success JSON payload starts with `"schema_version": 1`.
- Errors are emitted on stderr as structured JSON: `{"schema_version":1,"error":{"code":"UPPER_SNAKE","message":"...","details":{...}}}`.
- `logs slurm` and `logs app` are the exceptions — they stream plain text.
- All compute paths stay under cluster scratch.
- Reuse the same `--session-handle` to keep using the same live interactive allocation.
- For Megatron-LM v1, assume the `dev` image flavor unless explicitly overridden.

## Global Options

| Flag | Description |
|------|-------------|
| `--pretty` | Indent JSON output on stdout (and stderr for errors) for readability. |
| `--debug` | On uncaught exceptions, include `traceback` in the `INTERNAL_ERROR` error details. |

## Shared Options

Many commands share a common set of options:

| Flag | Description | Default |
|------|-------------|---------|
| `--repo` | Path to the Megatron-LM git checkout. If omitted, resolves from the current working directory via `git rev-parse --show-toplevel`. | cwd |
| `--run-name` / `--run-root` | Identify a run. Exactly one is required where applicable. `--run-name` resolves to `<scratch>/runs/<name>`. Names must match `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`. | — |
| `--cluster-config` | Path to a cluster YAML file. Overrides `--cluster-name`. | None |
| `--cluster-name` | Name of a registered cluster (from `cluster add`). If neither cluster flag is given, the default cluster is used. | None |
| `--base-image-flavor` | Base container image flavor (`dev` or `lts`). | `dev` |
| `--base-image` | Explicit base image override (bypasses flavor resolution). | None |
| `--dry-run` | Preview the execution plan without side-effects (available on `submit`, `ensure-env`, `prepare-image`). | `false` |

## Environment variables

Cog reads a small set of optional environment variables on the machine where you invoke the CLI. They are not per-command flags.

| Variable | Description |
|----------|-------------|
| `COG_SSH_OPTS` | Space-separated extra `ssh` arguments appended **after** Cog's defaults. Because they come last, they win when `ssh` sees duplicate options (for example `export COG_SSH_OPTS='-o ConnectTimeout=30'`). Whitespace-only values are ignored. |
| `COG_EXTRA_MOUNTS` | Comma-separated `host:container` bind mounts appended to Slurm `--container-mounts` for one-shot container `srun` jobs (`submit`, `ensure-env`). The cluster scratch root is always mounted first as `<scratch>:<scratch>`. Example: `export COG_EXTRA_MOUNTS='/lustre/shared:/lustre/shared,/data:/data'`. Empty entries are ignored. Does **not** apply to persistent `session start` allocations. |

### SSH connection multiplexing

Every cluster SSH invocation (for example `doctor`, `sync-workspace`, `logs slurm`) uses connection multiplexing to amortize handshake cost across step-heavy workflows:

- `ControlMaster=auto`
- `ControlPersist=60s`
- `ControlPath=~/.ssh/cm-%C` — `%C` is a hash of the connection parameters; the literal `~` is expanded by `ssh`, not the shell

Override or extend these via `COG_SSH_OPTS`.

---

## Command Reference

### `cog doctor`

**Purpose:** Validate local prerequisites and cluster connectivity. Run this first to verify that the environment is correctly set up before attempting any compute operations.

**When to use:** At the start of any workflow, after initial setup, or when troubleshooting connectivity or configuration issues.

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--repo` | Path to the Megatron-LM checkout. | cwd |
| `--cluster-config` | Cluster YAML override. | None |
| `--cluster-name` | Registered cluster name. | None |

**Behavior:** Runs a series of checks: Python version, package version, optional repo profile validation. If a cluster resolves, also runs SSH connectivity, scratch write, and `squeue --version` on the cluster. Cluster resolution failure is **not** fatal — it is recorded as a degraded or failed check.

**Output (JSON):**

```json
{
  "schema_version": 1,
  "checks": [
    {"name": "python_version", "status": "ok", "detail": "3.11.5", "duration_ms": 2},
    {"name": "cluster_resolved", "status": "ok", "duration_ms": 15},
    {"name": "ssh_reachable", "status": "ok", "duration_ms": 340}
  ],
  "overall": "ok",
  "cluster": {"name": "draco", "ssh_host": "login.cluster.example.com", "...": "..."},
  "cluster_source": "registry"
}
```

| Field | Description |
|-------|-------------|
| `checks[]` | List of individual check results, each with `name`, `status` (`ok`, `degraded`, `fail`), optional `detail`, and `duration_ms`. |
| `overall` | Aggregate status: `ok`, `degraded`, or `fail`. |
| `cluster` | Resolved cluster dict (if available). |
| `cluster_source` | Where the cluster config came from (`registry`, `config_file`, etc.). |

**Exit code:** `0` if `overall` is `ok` or `degraded`. `1` if `overall` is `fail`.

**Example:**

```bash
cog doctor --repo /Users/shanmugamr/Megatron-LM --cluster-name cw-dfw
```

---

### `cog profile`

**Purpose:** Show the fully resolved Megatron profile, run layout, base image, `.sqsh` path, and cluster scratch paths — without touching the cluster. Useful for validating configuration before running expensive operations.

**When to use:** Before running `prepare-image`, `ensure-env`, or `submit` to verify that paths, image, and layout are correct.

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--repo` | Path to the Megatron-LM checkout. | cwd |
| `--run-name` / `--run-root` | Run target (exactly one required). | — |
| `--base-image-flavor` | Image flavor (`dev` or `lts`). | `dev` |
| `--base-image` | Explicit base image override. | None |
| `--cluster-config` | Cluster YAML override. | None |
| `--cluster-name` | Registered cluster name. | None |

**Behavior:** Resolves the Megatron repo profile, materializes the run layout on cluster scratch (in plan form), resolves the base image, and computes the `.sqsh` plan and enroot import plan. No cluster mutation occurs.

**Output (JSON):**

```json
{
  "schema_version": 1,
  "profile": "megatron-lm",
  "required_repo_inputs": ["..."],
  "base_image": "nvcr.io/nvidia/pytorch:24.01-py3",
  "base_image_flavor": "dev",
  "cluster": {"name": "draco", "...": "..."},
  "cluster_source": "registry",
  "run_layout": {
    "run_root": "/scratch/.../runs/demo",
    "slurm_dir": "/scratch/.../runs/demo/slurm",
    "logs_dir": "/scratch/.../runs/demo/logs"
  },
  "default_exec_env": {"...": "..."},
  "sqsh_plan": {
    "cache_key": "abc123",
    "sqsh_path": "/scratch/.../images/abc123.sqsh"
  },
  "import_plan": {
    "account": "coreai_dlalgo_llm",
    "partition": "interactive",
    "source_ref": "nvcr.io/nvidia/pytorch:24.01-py3",
    "output_path": "/scratch/.../images/abc123.sqsh",
    "command": "srun ... enroot import ..."
  }
}
```

**Exit code:** `0`.

**Example:**

```bash
cog profile \
  --repo /Users/shanmugamr/Megatron-LM \
  --cluster-name cw-dfw \
  --run-name unit-test-smoke
```

---

### `cog prepare-image`

**Purpose:** Ensure the container `.sqsh` image exists on cluster scratch. This converts the base Docker/NGC image into an enroot squashfs file, or reuses an existing cached one.

**When to use:** Before `ensure-env` or `submit` to pre-warm the image cache. This avoids import latency during actual compute jobs. Also useful to validate image resolution with `--dry-run`.

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--repo` | Path to the Megatron-LM checkout. | cwd |
| `--base-image-flavor` | Image flavor (`dev` or `lts`). | `dev` |
| `--base-image` | Explicit base image override. | None |
| `--dry-run` | Plan without importing. | `false` |
| `--cluster-config` | Cluster YAML override. | None |
| `--cluster-name` | Registered cluster name. | None |

**Behavior:**
- **Dry-run:** Resolves the image and returns the import plan without contacting the cluster.
- **Real run:** Checks for an existing `.sqsh` on scratch. If absent, submits an `srun` job to run `enroot import`, waits for completion, and returns the result.

**Output (JSON):**

| Field | Description |
|-------|-------------|
| `dry_run` | `true` or `false`. |
| `profile` | Resolved Megatron profile name. |
| `base_image` | Fully qualified image reference. |
| `base_image_flavor` | `dev` or `lts`. |
| `cluster`, `cluster_source` | Resolved cluster and origin. |
| `cache_key` | Content-addressed key for the `.sqsh`. |
| `sqsh_path` | Absolute path to the `.sqsh` on cluster scratch. |
| `import_plan` | Planned Slurm job details (account, partition, command, stdout/stderr paths). |
| `cache_hit` | *(real run only)* Whether the `.sqsh` already existed. |
| `uv_cache_dir` | *(real run only)* UV cache directory used. |
| `import_stdout`, `import_stderr` | *(real run only)* Content of import job logs. |

**Exit code:** `0` on success. `1` on `AppError` (e.g., `IMAGE_IMPORT_FAILED`).

**Example — real run (reuses cached image):**

```bash
cog prepare-image \
  --repo /Users/shanmugamr/Megatron-LM \
  --cluster-name cw-dfw
```

**Example — dry-run to preview without importing:**

```bash
cog prepare-image \
  --repo /Users/shanmugamr/Megatron-LM \
  --cluster-name cw-dfw \
  --dry-run
```

---

### `cog sync-workspace`

**Purpose:** Synchronize the local Megatron-LM git worktree to the cluster scratch directory. This stages the code that container jobs will execute against.

**When to use:** Before `submit` or `session exec` when you've made local code changes and need them reflected on the cluster. Note: `submit`, `ensure-env`, and `session start`/`session exec` call this internally, so you typically don't need to invoke it directly unless you want to sync without running anything.

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--repo` | Path to the Megatron-LM checkout. | cwd |
| `--cluster-config` | Cluster YAML override. | None |
| `--cluster-name` | Registered cluster name. | None |

**Behavior:** Uses an rsync-style transfer to sync the local worktree to the cluster scratch location determined by the repo profile. Computes a workspace hash to detect changes.

**Output (JSON):**

| Field | Description |
|-------|-------------|
| `profile` | Resolved Megatron profile name. |
| `cluster`, `cluster_source` | Resolved cluster and origin. |
| `workspace_hash` | Hash of the synced workspace content. |
| `file_count` | Number of files transferred. |
| `remote_root` | Root path on cluster scratch. |
| `remote_repo_path` | Path to the synced repo on cluster. |
| `manifest_path` | Path to the sync manifest on cluster. |
| `cache_hit` | Whether the workspace was already up-to-date. |
| `transfer_strategy` | How the sync was performed (e.g., rsync). |

**Exit code:** `0`.

**Example:**

```bash
cog sync-workspace \
  --repo /Users/shanmugamr/Megatron-LM \
  --cluster-name cw-dfw
```

---

### `cog ensure-env`

**Purpose:** Materialize or reuse the shared Python virtual environment (`.venv`) on cluster scratch for the current dependency and image recipe. This runs `uv sync` inside the container to install all dependencies.

**When to use:** After `prepare-image` and before `submit` or `session` workflows. Ensures subsequent jobs skip the environment setup step. Best used once at the start of a session to amortize the install cost.

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--repo` | Path to the Megatron-LM checkout. | cwd |
| `--run-name` / `--run-root` | Run target (exactly one required). | — |
| `--gpus` | Number of GPUs for the env-build job. | `1` |
| `--time` | Slurm time limit. | `00:10:00` |
| `--partition` | Slurm partition. | `interactive` |
| `--nodes` | Number of nodes. | `1` |
| `--ntasks-per-node` | Tasks per node. | `1` |
| `--job-name` | Slurm job name. | `cog:env` |
| `--base-image-flavor` | Image flavor. | `dev` |
| `--base-image` | Explicit base image override. | None |
| `--dry-run` | Plan without running. | `false` |
| `--cluster-config` | Cluster YAML override. | None |
| `--cluster-name` | Registered cluster name. | None |

**Behavior:** Builds the full execution context: run root, image, workspace sync, and env recipe key. If the env is already ready (marker file present with matching recipe key), returns immediately with `ready_cache_hit: true`. Otherwise, submits an `srun` job that runs `uv sync` + validation inside the container, writes an env marker, and waits for completion.

**Output (JSON):**

| Field | Description |
|-------|-------------|
| `dry_run` | `true` or `false`. |
| `profile` | Resolved profile name. |
| `cluster`, `cluster_source` | Resolved cluster and origin. |
| `workspace_sync` | Workspace sync result (hash, file count, etc.). |
| `image` | Image resolution result (sqsh path, cache hit, etc.). |
| `env.env_root` | Absolute path to the `.venv` on scratch. |
| `env.marker_path` | Path to the env readiness marker file. |
| `env.ready_cache_hit` | Whether the env was already up-to-date. |
| `env.reason` | Why the env was built or reused. |
| `env.recipe_key` | Hash of the dependency recipe. |
| `env.dependency_inputs_key` | Hash of dependency input files. |
| `run_layout` | Paths for logs, artifacts, etc. |
| `job` | Slurm job result (or `null` if cache hit). Contains `job_id`, `returncode`, `state`, `stdout`, `stderr`. |
| `planned_job` | *(dry-run only)* Planned Slurm job parameters and paths. |

**Exit code:** `0` if no job was needed or `job.returncode == 0`. Otherwise, the Slurm job's return code. May also exit `1` with `ENV_LOCK_TIMEOUT` if another env build holds the lock.

**Example:**

```bash
cog ensure-env \
  --repo /Users/shanmugamr/Megatron-LM \
  --cluster-name cw-dfw \
  --run-name env-warmup \
  --gpus 1 \
  --time 00:20:00 \
  --partition interactive
```

---

### `cog submit`

**Purpose:** Run a one-shot, non-interactive `srun` job inside the container. Blocks until the job completes and returns the full result with logs and artifact paths.

**When to use:** For batch verification jobs, test runs, or any workload that should run to completion without interactive intervention. Prefer this over `session exec` when you don't need to run multiple commands in the same allocation.

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--repo` | Path to the Megatron-LM checkout. | cwd |
| `--run-name` / `--run-root` | Run target (exactly one required). | — |
| `--command` | The command to execute inside the container. **Required.** | — |
| `--gpus` | Number of GPUs. | `1` |
| `--time` | Slurm time limit. | `00:10:00` |
| `--partition` | Slurm partition. | `interactive` |
| `--nodes` | Number of nodes. | `1` |
| `--ntasks-per-node` | Tasks per node. | `1` |
| `--job-name` | Slurm job name. | `cog:run` |
| `--skip-uv-sync` | Skip the inline `uv sync` step. | `false` |
| `--base-image-flavor` | Image flavor. | `dev` |
| `--base-image` | Explicit base image override. | None |
| `--dry-run` | Plan without submitting. | `false` |
| `--cluster-config` | Cluster YAML override. | None |
| `--cluster-name` | Registered cluster name. | None |

**Behavior:** Prepares the full execution context (image, workspace sync, env), lints the command for common mistakes (e.g., bare `torchrun` instead of `python -m torch.distributed.run`), then submits the job via `srun`. For multi-node jobs, Cog automatically populates `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK`, and `LOCAL_RANK` inside the container. Blocks until the job terminates.

**Output (JSON):**

| Field | Description |
|-------|-------------|
| `profile` | Resolved profile name. |
| `cluster`, `cluster_source` | Resolved cluster and origin. |
| `workspace_sync` | Workspace sync result. |
| `image` | Image resolution result. |
| `uv_sync` | UV sync result (skipped if `--skip-uv-sync`). |
| `job.job_id` | Slurm job ID. |
| `job.command` | The full command that was executed. |
| `job.returncode` | Process return code. |
| `job.submission_returncode` | Slurm submission return code. |
| `job.state` | Normalized job state (e.g., `COMPLETED`, `FAILED`). |
| `job.raw_state` | Raw Slurm state string. |
| `job.slurm_exit_code` | Slurm-reported exit code. |
| `job.node_list` | Allocated node list. |
| `job.stdout`, `job.stderr` | Captured output. |
| `job.stdout_path`, `job.stderr_path` | Resolved paths on cluster scratch. |
| `artifacts.run_root` | Root of the run directory. |
| `artifacts.app_log_dir` | Application log directory. |
| `artifacts.torchrun_log_dir` | Torchrun log directory. |
| `artifacts.manifest_path` | Artifacts manifest path. |
| `artifacts.slurm_stdout_path` | `<run-root>/slurm/<jobid>.out` |
| `artifacts.slurm_stderr_path` | `<run-root>/slurm/<jobid>.err` |
| `warnings` | *(optional)* List of lint warnings (e.g., bare `torchrun` usage). |

**Exit code:** `0` if `job.returncode == 0`. Otherwise, the job's return code is propagated.

**Example — run a simple import smoke test (no GPU/distributed needed):**

```bash
cog submit \
  --repo /Users/shanmugamr/Megatron-LM \
  --cluster-name cw-dfw \
  --run-name test-basic \
  --command 'pytest tests/unit_tests/test_basic.py -v -o addopts=' \
  --gpus 1 \
  --time 00:10:00 \
  --partition interactive
```

**Example — run a distributed unit test with torchrun on 8 GPUs:**

```bash
cog submit \
  --repo /Users/shanmugamr/Megatron-LM \
  --cluster-name cw-dfw \
  --run-name test-gpt-constructor \
  --command 'python -m torch.distributed.run --nproc-per-node 8 --log-dir "$TORCHRUN_LOG_DIR" -m pytest -xvs tests/unit_tests/models/test_gpt_model.py::TestGPTModel::test_constructor' \
  --gpus 8 \
  --nodes 1 \
  --ntasks-per-node 8 \
  --time 01:00:00 \
  --partition interactive
```

**Example — dry-run to preview the Slurm job without submitting:**

```bash
cog submit \
  --repo /Users/shanmugamr/Megatron-LM \
  --cluster-name cw-dfw \
  --run-name test-basic-dry \
  --command 'pytest tests/unit_tests/test_basic.py -v -o addopts=' \
  --dry-run
```

---

### `cog session start`

**Purpose:** Start (or reuse) a persistent interactive Slurm allocation. The allocation stays alive so you can run multiple commands in it without re-queuing.

**When to use:** For iterative development and debugging workflows where you need to run many commands against the same GPU allocation. Start once, then use `session exec` repeatedly.

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--repo` | Path to the Megatron-LM checkout. | cwd |
| `--session-handle` | Unique handle for the session. Must match `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`. **Required.** | — |
| `--gpus` | Number of GPUs. | `1` |
| `--time` | Slurm time limit. | `04:00:00` |
| `--partition` | Slurm partition. | `interactive` |
| `--ntasks-per-node` | Tasks per node. | `1` |
| `--job-name` | Slurm job name. | None |
| `--base-image-flavor` | Image flavor. | `dev` |
| `--base-image` | Explicit base image override. | None |
| `--cluster-config` | Cluster YAML override. | None |
| `--cluster-name` | Registered cluster name. | None |

**Behavior:** Prepares the image and syncs the workspace, then starts a persistent interactive allocation via the `PersistentSessionManager`. If an allocation with the same handle already exists and is still running, it is reused (indicated by `reused: true`).

**Output (JSON):**

| Field | Description |
|-------|-------------|
| `profile` | Resolved profile name. |
| `cluster`, `cluster_source` | Resolved cluster and origin. |
| `workspace_sync` | Workspace sync result. |
| `image` | Image resolution result. |
| `env` | Environment root paths and keys. |
| `session.session_handle` | The handle you provided. |
| `session.job_id` | Slurm job ID for the allocation. |
| `session.state` | Session state (e.g., `RUNNING`). |
| `session.node_list` | Allocated nodes. |
| `session.controller_running` | Whether the session controller process is alive. |
| `session.session_root` | Root directory for this session on scratch. |
| `session.workdir` | Working directory inside the container. |
| `reused` | `true` if an existing session was reused, `false` if newly created. |

**Exit code:** `0`.

**Example:**

```bash
cog session start \
  --repo /Users/shanmugamr/Megatron-LM \
  --cluster-name cw-dfw \
  --session-handle megatron-dev-01 \
  --gpus 8 \
  --time 02:00:00 \
  --partition interactive
```

---

### `cog session status`

**Purpose:** Inspect the current state of a persistent session, including Slurm job state and controller health.

**When to use:** To check if a session is still running before executing commands in it, or to diagnose why a session might not be responding.

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--session-handle` | Handle of the session to inspect. **Required.** | — |
| `--cluster-config` | Cluster YAML override. | None |
| `--cluster-name` | Registered cluster name. | None |

**Behavior:** Looks up the session in local state, resolves the cluster (from the stored session metadata or flags), refreshes the remote Slurm state via SSH, and updates the local database.

**Output (JSON):**

| Field | Description |
|-------|-------------|
| `cluster`, `cluster_source` | Resolved cluster and origin. |
| `session` | Full session payload including `session_handle`, `job_id`, `state`, `raw_state`, `node_list`, `controller_running`, `controller_pid`, stdout/stderr paths, and all session root paths. |

**Exit code:** `0`. Errors (e.g., `SESSION_NOT_FOUND`, `SESSION_CLUSTER_MISMATCH`) exit `1` with structured error JSON on stderr.

**Example:**

```bash
cog session status --session-handle megatron-dev-01
```

---

### `cog session exec`

**Purpose:** Execute a single command inside an existing persistent session. The workspace is re-synced before execution to pick up local code changes.

**When to use:** After `session start`, to run tests, training steps, or debugging commands within the live allocation. Use repeatedly for iterative workflows.

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--repo` | Path to the Megatron-LM checkout (used for workspace sync). | cwd |
| `--session-handle` | Handle of the target session. **Required.** | — |
| `--command` | The command to execute. **Required.** | — |
| `--wait-timeout` | Maximum seconds to wait for completion. | `3600` |
| `--detach` | Submit the command and return immediately without waiting. | `false` |
| `--cluster-config` | Cluster YAML override. | None |
| `--cluster-name` | Registered cluster name. | None |

**Behavior:** Syncs the workspace to the cluster, then dispatches the command to the persistent session. In foreground mode (default), blocks until the command completes or the timeout is reached. In detached mode, returns immediately with a `request_id` that can be polled via `session exec-status`.

The command is linted for common issues (e.g., bare `torchrun`), and warnings are included in the output.

**Output (JSON):**

| Field | Description |
|-------|-------------|
| `cluster`, `cluster_source` | Resolved cluster and origin. |
| `session` | Session state snapshot. |
| `execution.request_id` | Unique identifier for this execution. |
| `execution.completed` | Whether the command finished within the timeout. |
| `execution.exit_code` | Remote process exit code (or `null` if incomplete/detached). |
| `execution.failure_reason` | Error reason if the execution failed. |
| `execution.stdout_path` | Path to stdout log on cluster. |
| `execution.stderr_path` | Path to stderr log on cluster. |
| `execution.wait_timeout` | The timeout that was applied. |
| `execution.detached_sentinel` | `"EXEC_DETACHED"` if `--detach` was used. |
| `artifacts.exec_root` | Root directory for this execution's artifacts. |
| `artifacts.run_root` | Run root within the execution. |
| `artifacts.torchrun_log_dir` | Torchrun log directory for this execution. |
| `warnings` | *(optional)* Lint warnings. |

**Exit code:**
- **Detached:** always `0`.
- **Foreground:** `124` if the wait timed out. `1` if incomplete or `exit_code` is `null`. Otherwise, the remote command's exit code.

**Example — run a standalone unit test inside the session:**

```bash
cog session exec \
  --session-handle megatron-dev-01 \
  --repo /Users/shanmugamr/Megatron-LM \
  --command 'pytest tests/unit_tests/test_basic.py -v -o addopts=' \
  --wait-timeout 300
```

**Example — run a distributed test with torchrun:**

```bash
cog session exec \
  --session-handle megatron-dev-01 \
  --repo /Users/shanmugamr/Megatron-LM \
  --command 'python -m torch.distributed.run --nproc-per-node 8 --log-dir "$TORCHRUN_LOG_DIR" -m pytest -xvs tests/unit_tests/models/test_gpt_model.py::TestGPTModel::test_constructor' \
  --wait-timeout 1800
```

**Example — fire-and-forget with `--detach`:**

```bash
cog session exec \
  --session-handle megatron-dev-01 \
  --repo /Users/shanmugamr/Megatron-LM \
  --command 'python -m torch.distributed.run --nproc-per-node 8 --log-dir "$TORCHRUN_LOG_DIR" -m pytest -xvs tests/unit_tests/transformer/' \
  --detach
```

---

### `cog session exec-status`

**Purpose:** Check the status of a previously submitted (typically detached) execution within a session.

**When to use:** After `session exec --detach` to poll whether the command has finished and retrieve its exit code.

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--session-handle` | Handle of the target session. **Required.** | — |
| `--request-id` | The `request_id` from the original `session exec`. **Required.** | — |
| `--cluster-config` | Cluster YAML override. | None |
| `--cluster-name` | Registered cluster name. | None |

**Output (JSON):**

| Field | Description |
|-------|-------------|
| `cluster`, `cluster_source` | Resolved cluster and origin. |
| `request` | Execution state: `request_id`, `state`, timestamps, `exit_code`, stdout/stderr paths. |

**Exit code:** `0`. `REQUEST_NOT_FOUND` error if the request directory doesn't exist.

**Example:**

```bash
cog session exec-status \
  --session-handle megatron-dev-01 \
  --request-id a1b2c3d4
```

---

### `cog session exec-cancel`

**Purpose:** Cancel a running execution within a session.

**When to use:** To abort a long-running or stuck command that was submitted via `session exec`.

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--session-handle` | Handle of the target session. **Required.** | — |
| `--request-id` | The `request_id` to cancel. **Required.** | — |
| `--cluster-config` | Cluster YAML override. | None |
| `--cluster-name` | Registered cluster name. | None |

**Output (JSON):**

| Field | Description |
|-------|-------------|
| `cluster`, `cluster_source` | Resolved cluster and origin. |
| `cancel.was_terminal` | Whether the request had already completed before the cancel. |
| `cancel.sentinel_written` | Whether a cancel sentinel was written. |

**Exit code:** `0`. Errors: `REQUEST_NOT_FOUND`, `REQUEST_ALREADY_TERMINAL`.

**Example:**

```bash
cog session exec-cancel \
  --session-handle megatron-dev-01 \
  --request-id a1b2c3d4
```

---

### `cog session requests ls`

**Purpose:** List recent execution requests within a session.

**When to use:** To review the history of commands executed in a session, including their states and exit codes.

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--session-handle` | Handle of the target session. **Required.** | — |
| `--limit` | Maximum number of requests to return. | `50` |
| `--cluster-config` | Cluster YAML override. | None |
| `--cluster-name` | Registered cluster name. | None |

**Output (JSON):**

| Field | Description |
|-------|-------------|
| `cluster`, `cluster_source` | Resolved cluster and origin. |
| `session_handle` | The session handle. |
| `limit` | The limit applied. |
| `requests[]` | List of requests, each with `request_id`, `state`, timestamps, and `exit_code`. |

**Exit code:** `0`.

**Example:**

```bash
cog session requests ls --session-handle megatron-dev-01 --limit 10
```

---

### `cog session stop`

**Purpose:** Cancel the Slurm job and stop the controller process for a persistent session.

**When to use:** When you're done with an interactive session and want to release the GPU allocation.

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--session-handle` | Handle of the session to stop. **Required.** | — |
| `--cluster-config` | Cluster YAML override. | None |
| `--cluster-name` | Registered cluster name. | None |

**Output (JSON):**

| Field | Description |
|-------|-------------|
| `cluster`, `cluster_source` | Resolved cluster and origin. |
| `session` | Session payload extended with `cancel_requested`, `controller_was_running`, `controller_signalled`. |

**Exit code:** `0`.

**Example:**

```bash
cog session stop --session-handle megatron-dev-01
```

---

### `cog cluster add`

**Purpose:** Register a new cluster in the local registry so subsequent commands can reference it by name.

**When to use:** Once per cluster, during initial setup. Use `--set-default` to avoid passing `--cluster-name` on every command.

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--name` | Cluster name. Required unless `--from-file`. | — |
| `--ssh-host` | SSH hostname for the cluster login node. Required unless `--from-file`. | — |
| `--scratch-root` | Absolute path to the scratch root on the cluster. Required unless `--from-file`. | — |
| `--import-account` | Slurm account for image import jobs. Required unless `--from-file`. | — |
| `--runtime-account` | Slurm account for runtime jobs. | Same as `--import-account` |
| `--import-partition` | Slurm partition for image imports. Required unless `--from-file`. | — |
| `--import-job-name` | Slurm job name for import jobs. Required unless `--from-file`. | — |
| `--from-file` | Path to a YAML onboarding file containing all cluster fields. | None |
| `--set-default` | Set this cluster as the default. | `false` |
| `--overwrite` | Overwrite if a cluster with the same name exists. | `false` |

**Output (JSON):**

| Field | Description |
|-------|-------------|
| `action` | `"added"`. |
| `cluster` | The registered cluster dict. |
| `path` | Path to the registry file. |
| `default` | Current default cluster name. |

**Exit code:** `0`. Errors: `CLUSTER_CONFIG_INVALID`, `CLUSTER_ALREADY_EXISTS`.

**Example — from a YAML onboarding file:**

```bash
cog cluster add \
  --from-file cluster-cw-dfw.yaml \
  --set-default
```

**Example — manual registration:**

```bash
cog cluster add \
  --name cw-dfw \
  --ssh-host cw-dfw-cs-001-vscode-01 \
  --scratch-root /lustre/fsw/portfolios/coreai/users/shanmugamr/agents-space \
  --import-account coreai_dlalgo_llm \
  --runtime-account coreai_dlalgo_llm \
  --import-partition cpu \
  --import-job-name cog:import \
  --set-default
```

---

### `cog cluster ls`

**Purpose:** List all registered clusters.

**When to use:** To see available clusters and which one is the default.

**Output (JSON):**

| Field | Description |
|-------|-------------|
| `default` | Name of the default cluster (or `null`). |
| `clusters` | List of cluster dicts. |
| `invalid_entries` | List of registry entries that failed to parse. |

**Exit code:** `0`.

**Example:**

```bash
cog cluster ls
```

---

### `cog cluster show`

**Purpose:** Show details of a specific registered cluster.

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--name` | Cluster name. **Required.** | — |

**Output (JSON):**

| Field | Description |
|-------|-------------|
| `cluster` | Full cluster dict. |
| `default` | Current default cluster name. |
| `is_default` | Whether this cluster is the default. |

**Exit code:** `0`. Error: `CLUSTER_NOT_FOUND`.

**Example:**

```bash
cog cluster show --name cw-dfw
```

---

### `cog cluster remove`

**Purpose:** Remove a cluster from the local registry.

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--name` | Cluster name. **Required.** | — |

**Output (JSON):**

| Field | Description |
|-------|-------------|
| `action` | `"removed"`. |
| `name` | The removed cluster name. |
| `default` | Current default cluster name (may change if the removed cluster was the default). |

**Exit code:** `0`. Error: `CLUSTER_NOT_FOUND`.

**Example:**

```bash
cog cluster remove --name old-cluster
```

---

### `cog cluster use`

**Purpose:** Set an existing registered cluster as the default.

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--name` | Cluster name. **Required.** | — |

**Output (JSON):**

| Field | Description |
|-------|-------------|
| `action` | `"use"`. |
| `default` | The new default cluster name. |
| `cluster` | The cluster dict. |

**Exit code:** `0`.

**Example:**

```bash
cog cluster use --name cw-dfw
```

---

### `cog sessions ls`

**Purpose:** List all known sessions from the local state database.

**When to use:** To find active or past sessions across clusters.

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--cluster` | Filter by cluster name. | None (all clusters) |

**Output (JSON):**

| Field | Description |
|-------|-------------|
| `sessions[]` | List of session records: `session_handle`, `job_id`, `mode`, `state`, `cluster_name`, `node_list`, `created_at`, `updated_at`. |

**Exit code:** `0`.

**Example:**

```bash
cog sessions ls --cluster cw-dfw
```

---

### `cog runs ls`

**Purpose:** List all known runs from the local state database.

**When to use:** To review past submit jobs and their outcomes.

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--cluster` | Filter by cluster name. | None (all clusters) |

**Output (JSON):**

| Field | Description |
|-------|-------------|
| `runs[]` | List of run records: `run_id`, `kind`, `cluster_name`, `run_name`, `run_root`, `job_id`, `state`, `stdout_path`, `stderr_path`, `created_at`, `updated_at`. |

**Exit code:** `0`.

**Example:**

```bash
cog runs ls --cluster cw-dfw
```

---

### `cog jobs get`

**Purpose:** Look up a Slurm job by ID, combining cluster-side `sacct`/`squeue` data with the local run record (if any).

**When to use:** To check the status of a specific job after submission, or to retrieve job details for debugging.

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--job-id` | Slurm job ID. **Required.** | — |
| `--cluster-config` | Cluster YAML override. | None |
| `--cluster-name` | Registered cluster name. | None |

**Output (JSON):**

| Field | Description |
|-------|-------------|
| `cluster`, `cluster_source` | Resolved cluster and origin. |
| `job.job_id` | The job ID. |
| `job.state` | Normalized job state. |
| `job.exit_code` | Exit code reported by Slurm. |
| `job.elapsed` | Elapsed time string. |
| `job.node_list` | Allocated nodes. |
| `job.start`, `job.end` | Start and end timestamps. |
| `run` | Local run record (or `null` if no matching local record). |

**Exit code:** `0`. Error: `JOB_NOT_FOUND` if neither `sacct` nor `squeue` reports the job.

**Example:**

```bash
cog jobs get --job-id 12345 --cluster-name cw-dfw
```

---

### `cog logs slurm`

**Purpose:** Read Slurm stdout/stderr log files from cluster scratch via SSH. Unlike other commands, this outputs **plain text**, not JSON.

**When to use:** After `submit` or `session exec` to inspect job output. Use `--follow` for live tailing during active jobs.

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--run-name` / `--run-root` | Run target (exactly one required). | — |
| `--job-id` | Slurm job ID. **Required.** | — |
| `--stream` | Which log stream to read: `stdout`, `stderr`, or `both`. | `both` |
| `--lines` | Number of lines to show (passed to `tail`). | `100` |
| `--follow` | Continuously tail the log (`tail -f`). | `false` |
| `--cluster-config` | Cluster YAML override. | None |
| `--cluster-name` | Registered cluster name. | None |

**Behavior:** Resolves log paths as `<run_root>/slurm/<job_id>.out` and `<run_root>/slurm/<job_id>.err`, then runs `tail` over SSH.

**Output:** Raw log text on stdout (not JSON).

**Exit code:** The `tail`/SSH return code. `130` on `KeyboardInterrupt` (e.g., Ctrl+C during `--follow`).

**Example — read both stdout and stderr:**

```bash
cog logs slurm \
  --run-name test-basic \
  --job-id 12345 \
  --stream both \
  --cluster-name cw-dfw
```

**Example — live-tail stdout during an active job:**

```bash
cog logs slurm \
  --run-name test-basic \
  --job-id 12345 \
  --stream stdout \
  --follow \
  --cluster-name cw-dfw
```

---

### `cog logs app`

**Purpose:** Read the local control-plane event log (`~/.cog/events.jsonl`). Unlike other commands, this outputs **plain text** (or JSONL), not structured JSON.

**When to use:** To debug cog's own behavior — what commands were run, session lifecycle events, errors, etc.

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--lines` | Number of lines to show. | `100` |
| `--follow` | Continuously tail the log. | `false` |
| `--kind` | Filter events by kind. | None (all) |
| `--agent-id` | Filter events by agent ID. | None (all) |
| `--session-handle` | Filter events by session handle. | None (all) |
| `--format` | Output format: `pretty` or `jsonl`. | `pretty` |

**Output:**
- **pretty:** `<timestamp> <kind> [agent=<id>] [session=<handle>] <payload-json>`
- **jsonl:** One JSON object per line with `ts`, `kind`, `payload`, optional `agent_id`, `session_handle`.

**Exit code:** `0`. `130` on `KeyboardInterrupt` during `--follow`.

**Example — recent events in pretty format:**

```bash
cog logs app --lines 50 --format pretty
```

**Example — filter events for a specific session:**

```bash
cog logs app --session-handle megatron-dev-01 --format jsonl
```

---

## Common Workflows

All examples below use:
- **Cluster:** `cw-dfw` (registered as default)
- **Scratch root:** `/lustre/fsw/portfolios/coreai/users/shanmugamr/agents-space/`
- **Repo:** `/Users/shanmugamr/Megatron-LM`

### 1. Initial setup — register the cluster

```bash
cog cluster add \
  --name cw-dfw \
  --ssh-host cw-dfw-cs-001-vscode-01 \
  --scratch-root /lustre/fsw/portfolios/coreai/users/shanmugamr/agents-space \
  --import-account coreai_dlalgo_llm \
  --runtime-account coreai_dlalgo_llm \
  --import-partition cpu \
  --import-job-name cog:import \
  --set-default
```

Verify:

```bash
cog cluster ls
```

### 2. Inspect the environment

```bash
cog doctor --repo /Users/shanmugamr/Megatron-LM

cog profile \
  --repo /Users/shanmugamr/Megatron-LM \
  --run-name unit-test-smoke
```

### 3. Warm image + env once

```bash
cog prepare-image --repo /Users/shanmugamr/Megatron-LM

cog ensure-env \
  --repo /Users/shanmugamr/Megatron-LM \
  --run-name env-warmup \
  --gpus 1 \
  --time 00:20:00 \
  --partition interactive
```

Use this when you want later jobs or sessions to avoid rebuilding the shared env.

### 4. Run a standalone unit test (one-shot)

A simple import smoke test that needs no GPUs or distributed setup:

```bash
cog submit \
  --repo /Users/shanmugamr/Megatron-LM \
  --run-name test-basic \
  --command 'pytest tests/unit_tests/test_basic.py -v -o addopts=' \
  --gpus 1 \
  --time 00:10:00 \
  --partition interactive
```

### 5. Run a distributed unit test (one-shot)

A GPU test that requires `torch.distributed.run` with 8 processes:

```bash
cog submit \
  --repo /Users/shanmugamr/Megatron-LM \
  --run-name test-gpt-constructor \
  --command 'python -m torch.distributed.run --nproc-per-node 8 --log-dir "$TORCHRUN_LOG_DIR" -m pytest -xvs tests/unit_tests/models/test_gpt_model.py::TestGPTModel::test_constructor' \
  --gpus 8 \
  --nodes 1 \
  --ntasks-per-node 8 \
  --time 01:00:00 \
  --partition interactive
```

For multi-node distributed jobs, use `--nodes` with `--ntasks-per-node`
and let Cog populate `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK`, and
`LOCAL_RANK` inside the container before your command runs. For example, a
two-node job can pass `--nodes 2 --gpus 8 --ntasks-per-node 8` and keep the
command focused on the application entry point.

### 6. Start one persistent session and iterate

Start an 8-GPU interactive session:

```bash
cog session start \
  --repo /Users/shanmugamr/Megatron-LM \
  --session-handle megatron-dev-01 \
  --gpus 8 \
  --time 02:00:00 \
  --partition interactive
```

Check it is running:

```bash
cog session status --session-handle megatron-dev-01
```

Run a quick standalone test:

```bash
cog session exec \
  --session-handle megatron-dev-01 \
  --repo /Users/shanmugamr/Megatron-LM \
  --command 'pytest tests/unit_tests/test_basic.py -v -o addopts=' \
  --wait-timeout 300
```

Run a distributed test in the same allocation:

```bash
cog session exec \
  --session-handle megatron-dev-01 \
  --repo /Users/shanmugamr/Megatron-LM \
  --command 'python -m torch.distributed.run --nproc-per-node 8 --log-dir "$TORCHRUN_LOG_DIR" -m pytest -xvs tests/unit_tests/models/test_gpt_model.py::TestGPTModel::test_constructor' \
  --wait-timeout 1800
```

Run another test in the same allocation:

```bash
cog session exec \
  --session-handle megatron-dev-01 \
  --repo /Users/shanmugamr/Megatron-LM \
  --command 'python -m torch.distributed.run --nproc-per-node 8 --log-dir "$TORCHRUN_LOG_DIR" -m pytest -xvs tests/unit_tests/models/test_gpt_model.py::TestGPTModel::test_set_input_tensor' \
  --wait-timeout 1800
```

List execution history:

```bash
cog session requests ls --session-handle megatron-dev-01
```

Stop the session and release GPUs:

```bash
cog session stop --session-handle megatron-dev-01
```

### 7. Read logs after a job

Read both stdout and stderr (replace `12345` with the actual `job_id` from the submit/exec output):

```bash
cog logs slurm \
  --run-name test-basic \
  --job-id 12345 \
  --stream both
```

Tail logs live during an active job:

```bash
cog logs slurm \
  --run-name test-gpt-constructor \
  --job-id 12345 \
  --stream stdout \
  --follow
```

Check local cog events:

```bash
cog logs app --lines 50 --format pretty
```

### 8. Look up a specific job

```bash
cog jobs get --job-id 12345 --cluster-name cw-dfw
```

### 9. Review past runs and sessions

```bash
cog runs ls --cluster cw-dfw
cog sessions ls --cluster cw-dfw
```

## Logs and Artifacts

### One-shot `submit`

- Slurm logs:
  - `<run-root>/slurm/<jobid>.out`
  - `<run-root>/slurm/<jobid>.err`
- Application artifacts:
  - `<run-root>/logs/app`
  - `<run-root>/logs/torchrun`
  - `<run-root>/checkpoints`
  - `<run-root>/tensorboard`
  - `<run-root>/artifacts.json`

### Persistent `session exec`

- Exec logs:
  - `.../sessions/<handle>/exec/runs/<request-id>/stdout.log`
  - `.../sessions/<handle>/exec/runs/<request-id>/stderr.log`
- Exec artifacts:
  - `.../sessions/<handle>/exec/runs/<request-id>/run/logs/app`
  - `.../sessions/<handle>/exec/runs/<request-id>/run/logs/torchrun`
  - `.../sessions/<handle>/exec/runs/<request-id>/run/checkpoints`
  - `.../sessions/<handle>/exec/runs/<request-id>/run/tensorboard`
  - `.../sessions/<handle>/exec/runs/<request-id>/run/artifacts.json`

## Error Handling

All errors are emitted on stderr as structured JSON:

```json
{
  "schema_version": 1,
  "error": {
    "code": "CLUSTER_NOT_FOUND",
    "message": "No cluster 'draco' found in registry and no default set.",
    "details": {}
  }
}
```

Common error codes (defined in `core/errors.py`):

| Code | Meaning |
|------|---------|
| `CLUSTER_NOT_FOUND` | No matching cluster in registry and no default set. |
| `CLUSTER_CONFIG_INVALID` | Cluster configuration is malformed or has invalid paths. |
| `CLUSTER_ALREADY_EXISTS` | `cluster add` without `--overwrite` on an existing name. |
| `REPO_NOT_FOUND` | `--repo` does not point to a valid git repository. |
| `RUN_ROOT_OUTSIDE_SCRATCH` | `--run-root` is not under the cluster's scratch root. |
| `SESSION_NOT_FOUND` | No session with the given handle exists locally. |
| `SESSION_CLUSTER_MISMATCH` | Session belongs to a different cluster than the one resolved. |
| `ENV_LOCK_TIMEOUT` | Another `ensure-env` process holds the env build lock. |
| `IMAGE_IMPORT_FAILED` | The enroot image import job failed. |
| `SUBMIT_FAILED` | The `srun` submission itself failed. |
| `EXEC_TIMEOUT` | `session exec` foreground wait exceeded `--wait-timeout`. |
| `REQUEST_NOT_FOUND` | No execution request with the given ID. |
| `REQUEST_ALREADY_TERMINAL` | Attempted to cancel a request that already finished. |
| `SSH_CONNECTION_FAILED` | SSH connection to the cluster failed. |
| `SCRATCH_UNWRITABLE` | Cannot write to the cluster scratch directory. |
| `SLURM_UNREACHABLE` | Slurm CLI (`squeue`, `sacct`) is not accessible on the cluster. |
| `JOB_NOT_FOUND` | No Slurm job with the given ID in `sacct` or `squeue`. |
| `INTERNAL_ERROR` | Unexpected error. With `--debug`, includes `traceback` in `details`. |

## Recommended Agent Rules

- Use installed `cog`, not `uv run`, unless running from a fresh checkout without install.
- Treat returned JSON as the source of truth — every payload starts with `"schema_version": 1`.
- Register a cluster once with `cluster add --set-default`, then refer to runs via `--run-name`.
- Prefer `ensure-env` once, then `submit` or `session` commands.
- Prefer `session start` + repeated `session exec` for iterative debugging.
- Prefer `submit` for one-shot verification jobs.
- Use `--dry-run` on `submit`, `ensure-env`, `prepare-image` to preview the plan without side-effects.
- Errors on stderr carry a structured `error.code` you can match on (see `core/errors.py`).
- Use returned log and artifact paths instead of guessing file locations.
- Use `python -m torch.distributed.run` instead of bare `torchrun` to avoid venv visibility issues.
- Use `--skip-uv-sync` on `submit` if you've already run `ensure-env` for the same recipe.
