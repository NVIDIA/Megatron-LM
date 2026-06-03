---
name: cog-setup-and-help
description: Canonical reference for everything `cog` (Claude-first local control plane for Megatron-LM workloads on Slurm) — installation, cluster registration, verification, the full CLI command catalogue (submit, session start/exec/exec-status/exec-cancel/stop, ensure-env, prepare-image, doctor, profile, logs slurm, logs app, jobs get, sessions ls, runs ls, cluster add/ls/show/use/remove), environment variables, error codes, and the iterative session-vs-submit workflow. Every other skill in this repo that uses cog defers to this one — read it first instead of the upstream cli-guide.
when_to_use: Setting up cog for the first time on a new machine; cloning and installing cog; registering a cluster with cog; verifying cog by running a Megatron-LM unit test on the cluster; looking up the flags / output shape / error code for any `cog` subcommand; deciding between `cog submit` and `cog session`; debugging an `error.code` from a cog command; 'install cog', 'set up cog', 'configure cog', 'cog cluster add', 'cog doctor fails', 'cog session', 'cog submit failed', 'what does cog X do', 'cog error code'.
---

# `cog` — setup and help

This is the canonical in-repo reference for cog: how to install it,
how to register a cluster, the full command catalogue, error codes,
and the iterative-workflow guidance. Other skills that drive cog
(`run-inference-unit-tests`, `run-inference-functional-tests`,
`run-inference-performance-tests`, …) link here for every cog
question — read this skill *before* opening the upstream
`docs/cli-guide.md`. The reference sections below mirror the cli-guide
but are kept in-tree so an offline / context-only agent can resolve
every cog question without a web fetch.

`cog` is a CLI that runs Megatron-LM workloads on a Slurm cluster from your
laptop: it imports the container image to scratch as `.sqsh`, syncs your
worktree, materializes a shared `.venv`, and submits jobs via `srun` over SSH.

This skill takes you from nothing to "I ran a Megatron-LM unit test on the
cluster via cog" in four steps:

1. Clone cog
2. Install the `cog` CLI
3. **3a. Collect & persist cluster defaults to `~/.cog/setup.env`** →
   3b. Register the cluster with cog and verify connectivity
4. Submit a Megatron-LM unit test through cog

> **🔁 Future agent invocations:** anything that touches cog (`cog
> submit`, `cog session exec`, `cog ensure-env`, …) **must `source
> ~/.cog/setup.env` first** and use the `$COG_*` variables it defines
> (`$COG_SSH_HOST`, `$COG_RUNTIME_ACCOUNT`, `$COG_INTERACTIVE_PARTITION`,
> `$COG_BATCH_PARTITION`, `$COG_SCRATCH_ROOT`, `$COG_MEGATRON_REPO`, …).
> Never assume hardcoded hostnames, accounts, partitions, or repo
> paths — read the env file. If the file is missing, run Step 3a to
> populate it (asking the user for each value with the documented
> defaults) before proceeding.

## Prerequisites

- **`uv`** installed on your local machine. If missing:
  `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **SSH access** to the cluster login node, with key-based auth working
  non-interactively. Test with `ssh -o BatchMode=yes <ssh-host> true`.
- **A local Megatron-LM checkout** on disk (cog syncs *this* directory to
  cluster scratch — it is not the same as the cog repo).
- **A cluster scratch directory** you can write to, with enough capacity for
  the container `.sqsh` (~10 GB) plus a `.venv` and run artifacts.
- **Slurm account + partition** with permission to submit `srun` jobs that
  use Pyxis / enroot container mounts. For NVIDIA `cw-dfw`, the usual
  account is `coreai_dlalgo_llm` (yours may differ — confirm with the
  cluster owner).

---

## Step 1 — Clone cog

The repo is on the internal NVIDIA GitLab. Two clone forms — pick the
one that matches how you authenticate to `gitlab-master.nvidia.com`:

**SSH (recommended if you already have a key uploaded to GitLab):**

```bash
# gitlab-master.nvidia.com uses a non-standard SSH port (12051), not 22
ssh-keyscan -H -p 12051 gitlab-master.nvidia.com >> ~/.ssh/known_hosts
git clone ssh://git@gitlab-master.nvidia.com:12051/shanmugamr/cog.git ~/cog
cd ~/cog
```

**HTTPS (needs a Personal Access Token in a credential helper):**

```bash
git clone https://gitlab-master.nvidia.com/shanmugamr/cog.git ~/cog
cd ~/cog
```

> **⚠️ The internal GitLab uses SSH port 12051, not the default 22.**
> If you try `git@gitlab-master.nvidia.com:shanmugamr/cog.git` (the
> shorthand SSH form) you'll get the banner *"If you are trying to
> clone, you are using the incorrect port, use 12051"*. The
> `ssh://...:12051/...` URL form above is the fix.
>
> If the HTTPS form fails with *"could not read Username for
> https://gitlab-master.nvidia.com: Device not configured"*, your shell
> is non-interactive and has no cached credential. Either configure a
> Personal Access Token in `~/.git-credentials` / a credential helper,
> or use the SSH form above.

---

## Step 2 — Install the `cog` CLI

From inside the clone, install as an editable `uv` tool so future
`git pull` updates are picked up without reinstalling:

```bash
uv tool install --editable . --force
```

Verify the binary is on `PATH` and prints help:

```bash
cog --help
```

If `cog: command not found`, add `uv tool`'s bin dir to `PATH`:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc   # or ~/.bashrc
exec $SHELL -l
```

> Alternative: `uv run cog --help` from inside the cog repo also works
> without a global install, but every command then requires
> `--project ~/cog` or being run from `~/cog`. The `uv tool install` path
> above is the recommended one for daily use.

---

## Step 3a — Collect & persist cluster defaults

This skill stores user-specific defaults in **`~/.cog/setup.env`** so
later steps (and later agent turns / later skill invocations) don't
have to re-prompt. The exact same file is the single source of truth
for SSH host, accounts, partitions, scratch root, and the local
Megatron-LM checkout path.

**Procedure (the agent running this skill should do this — don't
hardcode anything):**

1. Check whether `~/.cog/setup.env` exists.
2. **If it exists**, run `source ~/.cog/setup.env` and skip ahead to
   Step 3b. Confirm each variable is set (`echo $COG_SSH_HOST`, …) and
   only re-prompt for any that come up empty.
3. **If it does not exist**, ask the user for each variable in the
   table below using `AskUserQuestion` (or equivalent). For every
   variable except `COG_MEGATRON_REPO`, *offer the listed default* —
   accept the default if the user doesn't override. `COG_MEGATRON_REPO`
   has no default; require an answer.

   | Variable | Prompt to the user | Default |
   |---|---|---|
   | `COG_CLUSTER_NAME` | Name to register this cluster under in the cog registry | `cw-dfw` |
   | `COG_SSH_HOST` | SSH login host for the cluster | `cw-dfw-cs-001-vscode-01` |
   | `COG_RUNTIME_ACCOUNT` | Slurm account for GPU runtime jobs (`submit`, `session start`) | `coreai_dlalgo_llm` |
   | `COG_IMPORT_ACCOUNT` | Slurm account for the CPU-side `enroot import` job | same value as `COG_RUNTIME_ACCOUNT` |
   | `COG_INTERACTIVE_PARTITION` | Partition for interactive / short jobs (used by Step 4 and most `cog submit` examples) | `interactive` |
   | `COG_BATCH_PARTITION` | Partition for long-running batch jobs | `batch` |
   | `COG_IMPORT_PARTITION` | Partition that runs the CPU `enroot import` job | `cpu` |
   | `COG_SCRATCH_ROOT` | Absolute path to your scratch root on the cluster (cog will store `.sqsh`, `.venv`, workspaces, run logs under this) | `/lustre/fsw/portfolios/coreai/users/${USER}/agents-space` |
   | `COG_MEGATRON_REPO` | Absolute path to your local Megatron-LM checkout | **no default — must come from the user** |

4. Write the answers to `~/.cog/setup.env`. **Each line must use
   `export KEY="value"`** — without the `export`, the variables exist
   only in the sourcing shell and won't be visible to `cog`, `ssh`,
   `srun`, or any other subprocess, which silently produces empty
   `--ssh-host`, `--partition`, `--repo`, etc. arguments. Example:

   ```bash
   mkdir -p ~/.cog
   cat > ~/.cog/setup.env <<'EOF'
   # Populated by the cog-setup-and-help skill. Source this file before invoking cog.
   # Variables are `export`-ed so subprocesses (cog, ssh, srun) see them.
   export COG_CLUSTER_NAME="cw-dfw"
   export COG_SSH_HOST="cw-dfw-cs-001-vscode-01"
   export COG_RUNTIME_ACCOUNT="coreai_dlalgo_llm"
   export COG_IMPORT_ACCOUNT="coreai_dlalgo_llm"
   export COG_INTERACTIVE_PARTITION="interactive"
   export COG_BATCH_PARTITION="batch"
   export COG_IMPORT_PARTITION="cpu"
   export COG_SCRATCH_ROOT="/lustre/fsw/portfolios/coreai/users/${USER}/agents-space"
   export COG_MEGATRON_REPO="/Users/${USER}/Megatron-LM"
   EOF
   source ~/.cog/setup.env
   # Sanity check — both lines must show the value, not blank:
   echo "$COG_SSH_HOST"
   env | grep ^COG_ | sort
   ```

5. From this point on, every `cog` invocation in this skill (and in
   later turns) references `$COG_*` rather than literal values.

---

## Step 3b — Register the cluster with cog

With `~/.cog/setup.env` sourced, register the cluster once. `cog
cluster add --set-default` makes later commands omit `--cluster-name`:

```bash
source ~/.cog/setup.env
cog cluster add \
  --name "$COG_CLUSTER_NAME" \
  --ssh-host "$COG_SSH_HOST" \
  --scratch-root "$COG_SCRATCH_ROOT" \
  --import-account "$COG_IMPORT_ACCOUNT" \
  --runtime-account "$COG_RUNTIME_ACCOUNT" \
  --import-partition "$COG_IMPORT_PARTITION" \
  --import-job-name cog:import \
  --set-default
```

Field notes:

- **`--scratch-root`** must be a path that exists and is writable by you on
  the cluster. Cog stores `.sqsh` images, the shared `.venv`, synced
  workspaces, and run logs under this path. Create it ahead of time:
  `ssh "$COG_SSH_HOST" mkdir -p "$COG_SCRATCH_ROOT"`.
- **`--import-account` / `--runtime-account`** can be the same Slurm
  account or different ones if your site separates CPU import jobs from
  GPU runtime jobs.
- **`--import-partition`** is a CPU partition; image import does not need
  GPUs and reserving them just to run `enroot import` is wasteful.

Verify and run `doctor`:

```bash
cog cluster ls
cog doctor --repo "$COG_MEGATRON_REPO"
```

`cog doctor` checks: local Python / cog version, SSH reachability,
scratch writability, and `squeue --version` on the cluster. Status `ok`
or `degraded` exits `0`; `fail` exits `1`. Read the `checks[]` array in
the JSON output to see which probe failed before retrying.

Common `doctor` failures:

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `SSH_CONNECTION_FAILED` | Key not loaded / host unknown | `ssh-add`, accept host key, set `COG_SSH_OPTS='-o StrictHostKeyChecking=accept-new'` |
| `SCRATCH_UNWRITABLE` | Scratch root missing or wrong owner | `ssh <host> mkdir -p <scratch-root> && touch <scratch-root>/.cog-test` |
| `SLURM_UNREACHABLE` | `squeue` not on PATH on login node | Confirm you're pointing at a login node, not a worker |

---

## Step 4 — Verify by running a Megatron-LM unit test

The cheapest end-to-end smoke test is `tests/unit_tests/test_basic.py` —
it imports the package and runs a few CPU-only assertions, so it
exercises the full cog pipeline (image import, workspace sync,
`uv sync`, `srun`) without needing real GPU compute.

### 4a. Pre-warm the image and env (optional but recommended)

This avoids paying the ~5-minute import latency inside the actual test
job. Skip if you'd rather see one combined job:

```bash
source ~/.cog/setup.env
cog prepare-image --repo "$COG_MEGATRON_REPO"

cog ensure-env \
  --repo "$COG_MEGATRON_REPO" \
  --run-name env-warmup \
  --gpus 1 \
  --time 00:20:00 \
  --partition "$COG_INTERACTIVE_PARTITION"
```

`prepare-image` runs `enroot import` on the cluster and caches the
`.sqsh` keyed on the resolved base image. `ensure-env` runs `uv sync`
inside that image and writes a marker file so subsequent jobs reuse the
`.venv`. Both are idempotent: a second invocation with no changes
returns `cache_hit: true` instantly.

### 4b. Submit the unit test

```bash
source ~/.cog/setup.env
cog submit \
  --repo "$COG_MEGATRON_REPO" \
  --run-name verify-cog-basic \
  --command 'python -m pytest tests/unit_tests/test_basic.py -v -o addopts=' \
  --gpus 1 \
  --time 00:10:00 \
  --partition "$COG_INTERACTIVE_PARTITION"
```

> **Use `python -m pytest`, not bare `pytest`.** The shared `.venv` is
> built under a `.partial.<hash>` directory and then `mv`-d to its
> canonical recipe path; this breaks any bin-script (like `pytest`) that
> has the original path baked into its shebang, so a bare `pytest` call
> exits 127 (`pytest: command not found`). `python` itself is resolved
> via the `PATH` cog sets before sourcing the venv's `activate`, so
> `python -m pytest ...` works. Same rule as the `torchrun` →
> `python -m torch.distributed.run` substitution below.

What this does:

- Syncs your local Megatron-LM worktree to scratch.
- Allocates 1 GPU on the `interactive` partition for up to 10 minutes.
- Runs the test inside the container at the cached `.sqsh`, with the
  shared `.venv` mounted.
- Streams `srun` to completion, then prints a JSON payload to stdout
  including `job.job_id`, `job.returncode`, `job.stdout`, and
  `artifacts.slurm_stdout_path`.

Exit code `0` and `job.returncode: 0` means cog is fully wired up. If
the unit test prints "passed", you're done.

> `-o addopts=` clears any `addopts` from `pyproject.toml` that would
> otherwise pull in distributed-only options. Keep it for the smoke test.

### 4c. (Optional) Run a real distributed unit test

Once the basic test passes, validate the multi-GPU + `torch.distributed`
path:

```bash
source ~/.cog/setup.env
cog submit \
  --repo "$COG_MEGATRON_REPO" \
  --run-name verify-cog-gpt \
  --command 'python -m torch.distributed.run --nproc-per-node 8 --log-dir "$TORCHRUN_LOG_DIR" -m pytest -xvs tests/unit_tests/models/test_gpt_model.py::TestGPTModel::test_constructor' \
  --gpus 8 \
  --nodes 1 \
  --ntasks-per-node 8 \
  --time 01:00:00 \
  --partition "$COG_INTERACTIVE_PARTITION"
```

> Use `$COG_INTERACTIVE_PARTITION` for short / iterative jobs (test
> verification, debugging, `cog session` allocations). Use
> `$COG_BATCH_PARTITION` for long batch jobs (full training runs,
> sweeps). Pick based on intent, never hardcode the literal.

Note: **`python -m torch.distributed.run`**, not bare `torchrun` — the
shared venv lives at a path where `torchrun`'s shebang sometimes
resolves to the wrong interpreter. Cog lints `--command` for bare
`torchrun` usage and surfaces a warning, but using `python -m
torch.distributed.run` from the start avoids the trap.

---

## Iterating: prefer `cog session start` + `cog session exec` over repeated `cog submit`

> **🔁 IMPORTANT — iterative workflow.** If you keep hitting errors,
> retrying the same job, or running short commands in a loop, **stop
> using `cog submit` and switch to `cog session`**. Every `cog submit`
> requests a fresh Slurm allocation, which means queue time +
> container start + venv activate on every attempt — minutes of latency
> per iteration.
>
> Instead, allocate one node for a few hours and exec into it
> repeatedly:
>
> ```bash
> source ~/.cog/setup.env
>
> # Hold an 8-GPU node for 3 hours.
> cog session start \
>   --repo "$COG_MEGATRON_REPO" \
>   --run-name iter-debug \
>   --gpus 8 --nodes 1 --ntasks-per-node 1 \
>   --time 03:00:00 \
>   --partition "$COG_INTERACTIVE_PARTITION"
>
> # Now run as many commands as you want — each starts in ~1 second
> # because the allocation, container, and venv are already up.
> cog session exec --run-name iter-debug --command 'python -m pytest tests/unit_tests/test_basic.py -v'
> cog session exec --run-name iter-debug --command 'python -c "import megatron; print(megatron.__file__)"'
>
> # When done, release the allocation.
> cog session stop --run-name iter-debug
> ```
>
> Rule of thumb: if you'll run more than 2 commands against the same
> code, use a session. The full session command set
> (`session start` / `exec` / `exec-status` / `exec-cancel` / `stop`,
> `sessions ls`, `session requests ls`) is documented in
> `~/cog/docs/cli-guide.md` and online at
> <https://gitlab-master.nvidia.com/shanmugamr/cog/-/blob/main/docs/cli-guide.md>.

---

## Reading logs

`cog submit` prints `job.stdout` / `job.stderr` inline by default, but for
larger jobs use `cog logs slurm`:

```bash
cog logs slurm \
  --run-name verify-cog-basic \
  --job-id <job_id> \
  --stream both \
  --lines 200
```

For cog's own behavior (what commands ran, when sessions started, errors
cog raised), use the local app log:

```bash
cog logs app --lines 50
```

---

## Common pitfalls

- **`uv tool install` succeeded but `cog: command not found`.** Add
  `$HOME/.local/bin` to `PATH` (see Step 2). `uv tool dir --bin` prints
  the exact directory.
- **`RUN_ROOT_OUTSIDE_SCRATCH` error on `submit`.** `--run-root` (or the
  path that `--run-name` resolves to) must be under the registered
  cluster's `scratch_root`. Prefer `--run-name <name>` and let cog
  compose the path.
- **First `submit` is slow (~5-10 min).** That is the one-time
  `enroot import` + `uv sync`. Subsequent runs with the same image and
  `pyproject.toml` hit cache and start in seconds. Pre-warm with
  `prepare-image` + `ensure-env` if you want the slow path out of the
  critical path.
- **`cog doctor` shows `cluster_resolved: ok` but `ssh_reachable: fail`.**
  The registry entry is fine; SSH itself is the problem. Reproduce with
  `ssh -o BatchMode=yes <ssh-host> true` and fix at the SSH layer.
- **Editing cog itself.** Because Step 2 installed editable
  (`--editable`), any `git pull` in `~/cog` is picked up immediately —
  no reinstall needed. If you ever switch to a wheel install
  (`uv build && uv tool install --force dist/cog-*.whl`), you must
  reinstall after each change.
- **Pointing cog at the wrong repo.** `--repo` is the **Megatron-LM**
  checkout, not the cog checkout. Get this wrong and `doctor` will fail
  the repo-profile check with a clear message — but it is the most
  common day-one confusion.
- **`srun ... task 0: Exited with exit code 127` from `cog submit`,
  Slurm log ends in `pytest: command not found` (or any other bin
  script not found).** The shared `.venv` is built in a `.partial.<hash>`
  directory, then renamed into place; bin-script shebangs still point at
  the old path. Invoke via `python -m <module>` instead of the bin
  wrapper (e.g. `python -m pytest`, `python -m torch.distributed.run`).
  `python` itself works because cog explicitly sets `PATH` to the
  canonical venv before sourcing `activate`.
- **Stale cog install: `cog` on `PATH` but raises
  `ModuleNotFoundError: No module named 'cog'`.** Happens when an
  earlier `uv tool install --editable` pointed at a clone that has
  since been deleted. Fix with `uv tool uninstall cog`, re-clone (Step 1),
  then `uv tool install --editable . --force` again.
- **`~/.cog/setup.env` is missing or has stale values.** Re-run Step 3a
  to re-prompt and re-write the file. If only one variable is wrong
  (e.g. user moved their Megatron-LM checkout), edit just that line in
  the file rather than wiping it. Sourcing the file is cheap — always
  do it at the start of any turn that runs a `cog` command.
- **`cog` runs but `--repo` / `--partition` / `--ssh-host` flags
  receive empty strings.** The env file was written without `export`,
  so `source ~/.cog/setup.env` populated the variables in the current
  shell only — they were not visible to the `cog` subprocess. Fix:
  prepend `export ` to every line in `~/.cog/setup.env` (the Step 3a
  template already does this; older files written before this fix may
  not). Verify with `source ~/.cog/setup.env && env | grep ^COG_` —
  if the grep is empty, the file is missing `export` keywords.
- **Hardcoded hostnames / accounts / partitions / repo paths in agent
  output.** This is a guidance failure, not a tool failure. Every cog
  command this skill emits should reference `$COG_*` variables from
  `~/.cog/setup.env`. If you find yourself typing literal `cw-dfw…`,
  `coreai_dlalgo_llm`, or `interactive` into a `cog` flag, stop and
  source the env file instead.

---

# Cog command reference

The rest of this skill is the in-tree mirror of cog's
`docs/cli-guide.md`. **Read this section before falling back to the
upstream guide.** The upstream guide
(<https://gitlab-master.nvidia.com/shanmugamr/cog/-/blob/main/docs/cli-guide.md>,
or `~/cog/docs/cli-guide.md` locally) is the source of truth for
anything not covered here; this skill stays current with the commands,
flags, output fields, and error codes other skills in this repo
actually use.

## Contract

- Most commands print a single JSON object to stdout. Parse it; don't
  scrape text.
- Every success JSON payload starts with `"schema_version": 1`.
- Errors are emitted on stderr as structured JSON:
  `{"schema_version":1,"error":{"code":"UPPER_SNAKE","message":"...","details":{...}}}`.
- `logs slurm` and `logs app` are the exceptions — they stream plain
  text.
- All compute paths stay under cluster scratch.
- Re-use the same `--session-handle` to keep using the same live
  interactive allocation.
- For Megatron-LM v1, assume the `dev` image flavor unless explicitly
  overridden.

## Global options

| Flag | Description |
|------|-------------|
| `--pretty` | Indent JSON output on stdout (and stderr for errors). |
| `--debug` | Include `traceback` in `INTERNAL_ERROR` error details on uncaught exceptions. |

## Shared options (used by most commands)

| Flag | Description | Default |
|------|-------------|---------|
| `--repo` | Path to the Megatron-LM git checkout. Resolves from cwd via `git rev-parse --show-toplevel` if omitted. | cwd |
| `--run-name` / `--run-root` | Identify a run. Exactly one is required where applicable. `--run-name` resolves to `<scratch>/runs/<name>`. Names must match `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`. | — |
| `--cluster-config` | Path to a cluster YAML file. Overrides `--cluster-name`. | None |
| `--cluster-name` | Name of a registered cluster (from `cluster add`). Falls back to the default cluster. | None |
| `--base-image-flavor` | Base container image flavor (`dev` or `lts`). | `dev` |
| `--base-image` | Explicit base image override (bypasses flavor resolution). | None |
| `--dry-run` | Preview the execution plan without side-effects (available on `submit`, `ensure-env`, `prepare-image`). | `false` |

## Environment variables

Read by cog on the machine where you invoke the CLI — not per-command flags.

| Variable | Description |
|----------|-------------|
| `COG_SSH_OPTS` | Space-separated extra `ssh` arguments appended **after** cog's defaults (later args win on duplicates). Example: `export COG_SSH_OPTS='-o ConnectTimeout=30'`. |
| `COG_EXTRA_MOUNTS` | Comma-separated `host:container` bind mounts appended to Slurm `--container-mounts` for one-shot container `srun` jobs (`submit`, `ensure-env`). The cluster scratch root is always mounted first as `<scratch>:<scratch>`. **Does NOT apply to persistent `session start` allocations.** Example: `export COG_EXTRA_MOUNTS='/lustre/shared:/lustre/shared,/data:/data'`. |

SSH connection multiplexing is on by default
(`ControlMaster=auto`, `ControlPersist=60s`,
`ControlPath=~/.ssh/cm-%C`). Override via `COG_SSH_OPTS`.

---

## Command catalogue

The columns below are: what the command does · the flags you'll actually
use · the output fields agents read most often.

### `cog doctor` — verify environment + cluster

**Use first** when troubleshooting. Checks Python version, package
version, optional repo profile validation, then (if a cluster
resolves) SSH connectivity, scratch writability, and `squeue
--version`.

| Flag | Default |
|------|---------|
| `--repo` | cwd |
| `--cluster-name` / `--cluster-config` | default cluster |

Output: `checks[]` (each with `name`, `status` ∈ {`ok`,`degraded`,`fail`}, `duration_ms`), `overall`, `cluster`, `cluster_source`.

Exit `0` if `overall` is `ok` or `degraded`; `1` if `fail`.

```bash
cog doctor --repo "$COG_MEGATRON_REPO"
```

### `cog profile` — resolve the run plan without touching the cluster

Shows the resolved Megatron profile, run layout, base image, `.sqsh`
path, scratch paths.

| Flag | Default |
|------|---------|
| `--repo` | cwd |
| `--run-name` / `--run-root` (one required) | — |
| `--base-image-flavor` | `dev` |
| `--base-image` | None |

Output: `profile`, `base_image`, `run_layout.{run_root,slurm_dir,logs_dir}`, `sqsh_plan.{cache_key,sqsh_path}`, `import_plan.{account,partition,command,output_path}`.

### `cog prepare-image` — ensure the `.sqsh` exists on scratch

Runs `enroot import` on the cluster (or reuses cache). Pre-warm before
`submit` / `ensure-env` to avoid in-job import latency.

| Flag | Default |
|------|---------|
| `--repo` | cwd |
| `--base-image-flavor` | `dev` |
| `--base-image` | None |
| `--dry-run` | `false` |

Output: `dry_run`, `base_image`, `cache_key`, `sqsh_path`, `import_plan`, `cache_hit` (real run), `import_stdout/stderr` (real run).

Exit `0` on success; `1` on `IMAGE_IMPORT_FAILED`.

### `cog sync-workspace` — push the local worktree to scratch

`submit`, `ensure-env`, and `session start`/`exec` call this internally —
only invoke directly to sync without running anything.

Output: `workspace_hash`, `file_count`, `remote_repo_path`, `cache_hit`, `transfer_strategy`.

### `cog ensure-env` — materialize the shared `.venv` for this dep recipe

Submits an `srun` job that runs `uv sync` inside the container, writes
an env marker, and waits. Idempotent: a second call with the same
recipe key returns `ready_cache_hit: true` instantly.

| Flag | Default |
|------|---------|
| `--repo` | cwd |
| `--run-name` / `--run-root` (one required) | — |
| `--gpus` | `1` |
| `--time` | `00:10:00` |
| `--partition` | `interactive` |
| `--nodes` / `--ntasks-per-node` | `1` / `1` |
| `--base-image-flavor` / `--base-image` | `dev` / None |
| `--dry-run` | `false` |

Output: `env.env_root`, `env.marker_path`, `env.ready_cache_hit`, `env.reason`, `env.recipe_key`, `env.dependency_inputs_key`, `job.{job_id,returncode,state,stdout,stderr}` (or `null` on cache hit).

Exit `0` if no job ran or `job.returncode == 0`; otherwise the job's return code. `1` on `ENV_LOCK_TIMEOUT`.

### `cog submit` — one-shot `srun` job, block to completion

The default workhorse for **single** verification jobs.

| Flag | Default |
|------|---------|
| `--repo` | cwd |
| `--run-name` / `--run-root` (one required) | — |
| `--command` (required) | — |
| `--gpus` | `1` |
| `--time` | `00:10:00` |
| `--partition` | `interactive` |
| `--nodes` / `--ntasks-per-node` | `1` / `1` |
| `--skip-uv-sync` | `false` — set if you've already `ensure-env`'d the same recipe |
| `--base-image-flavor` / `--base-image` | `dev` / None |
| `--dry-run` | `false` |

Cog lints `--command` (warns on bare `torchrun`), then submits via
`srun`. For multi-node jobs, cog auto-populates `MASTER_ADDR`,
`MASTER_PORT`, `WORLD_SIZE`, `RANK`, `LOCAL_RANK` inside the container.

Output: `job.{job_id,command,returncode,state,raw_state,slurm_exit_code,node_list,stdout,stderr,stdout_path,stderr_path}`, `artifacts.{run_root,app_log_dir,torchrun_log_dir,manifest_path,slurm_stdout_path,slurm_stderr_path}`, optional `warnings`.

Exit propagates `job.returncode`.

### `cog session start` — persistent interactive allocation

**Prefer this whenever you'll run more than 2 commands on the same
code.** Allocation stays alive for `--time`; you `session exec` into
it repeatedly.

| Flag | Default |
|------|---------|
| `--repo` | cwd |
| `--session-handle` (required, regex same as run names) | — |
| `--gpus` | `1` |
| `--time` | `04:00:00` |
| `--partition` | `interactive` |
| `--ntasks-per-node` | `1` |
| `--base-image-flavor` / `--base-image` | `dev` / None |

Output: `session.{session_handle,job_id,state,node_list,controller_running,session_root,workdir}`, `reused` (true if an existing handle was matched).

### `cog session status` — inspect a live session

| Flag |
|------|
| `--session-handle` (required) |

Output: `session.{session_handle,job_id,state,raw_state,node_list,controller_running,controller_pid}` and stdout/stderr/root paths.

Errors: `SESSION_NOT_FOUND`, `SESSION_CLUSTER_MISMATCH`.

### `cog session exec` — run one command inside an existing session

Re-syncs the workspace, then dispatches the command. Lints
`--command` like `submit`. Foreground by default; `--detach` returns a
`request_id` for later polling.

| Flag | Default |
|------|---------|
| `--repo` | cwd |
| `--session-handle` (required) | — |
| `--command` (required) | — |
| `--wait-timeout` | `3600` seconds |
| `--detach` | `false` |

Output: `execution.{request_id,completed,exit_code,failure_reason,stdout_path,stderr_path,wait_timeout,detached_sentinel}`, `artifacts.{exec_root,run_root,torchrun_log_dir}`.

Exit code:
- Detached: always `0`.
- Foreground: `124` on `--wait-timeout`; `1` if incomplete / `exit_code` null; otherwise the remote command's exit code.

### `cog session exec-status` — poll a (typically detached) exec

| Flag |
|------|
| `--session-handle` (required) |
| `--request-id` (required) |

Output: `request.{request_id,state,exit_code,...}`. Error: `REQUEST_NOT_FOUND`.

### `cog session exec-cancel` — kill a running exec

| Flag |
|------|
| `--session-handle` (required) |
| `--request-id` (required) |

Output: `cancel.{was_terminal,sentinel_written}`. Errors: `REQUEST_NOT_FOUND`, `REQUEST_ALREADY_TERMINAL`.

### `cog session requests ls` — recent execs in a session

| Flag | Default |
|------|---------|
| `--session-handle` (required) | — |
| `--limit` | `50` |

Output: `requests[]` with `request_id`, `state`, `exit_code`, timestamps.

### `cog session stop` — cancel the Slurm job, kill the controller

| Flag |
|------|
| `--session-handle` (required) |

Output: `session` payload extended with `cancel_requested`, `controller_was_running`, `controller_signalled`.

### `cog cluster add` — register a cluster

| Flag | Required unless `--from-file` |
|------|-------------------------------|
| `--name` | yes |
| `--ssh-host` | yes |
| `--scratch-root` | yes |
| `--import-account` | yes |
| `--runtime-account` | defaults to `--import-account` |
| `--import-partition` | yes |
| `--import-job-name` | yes |
| `--from-file` | YAML onboarding file (sets all of the above) |
| `--set-default` | makes this the default cluster |
| `--overwrite` | overwrite an existing entry of the same name |

Errors: `CLUSTER_CONFIG_INVALID`, `CLUSTER_ALREADY_EXISTS`.

### `cog cluster ls` / `show` / `use` / `remove`

- `ls`: lists registered clusters; output has `default`, `clusters[]`, `invalid_entries[]`.
- `show --name <n>`: full cluster dict + `is_default`.
- `use --name <n>`: set the default.
- `remove --name <n>`: delete the registry entry.

Common error: `CLUSTER_NOT_FOUND`.

### `cog sessions ls` / `runs ls`

- `sessions ls [--cluster <n>]`: list session records (`session_handle`, `job_id`, `mode`, `state`, `cluster_name`, `node_list`, timestamps).
- `runs ls [--cluster <n>]`: list run records (`run_id`, `kind`, `cluster_name`, `run_name`, `run_root`, `job_id`, `state`, `stdout_path`, `stderr_path`, timestamps).

### `cog jobs get` — Slurm job lookup by ID

Combines `sacct`/`squeue` data with the local run record (if any).

| Flag |
|------|
| `--job-id` (required) |

Output: `job.{job_id,state,exit_code,elapsed,node_list,start,end}`, `run` (local record or `null`). Error: `JOB_NOT_FOUND`.

### `cog logs slurm` — read Slurm stdout/stderr (plain text)

| Flag | Default |
|------|---------|
| `--run-name` / `--run-root` (one required) | — |
| `--job-id` (required) | — |
| `--stream` | `both` (or `stdout` / `stderr`) |
| `--lines` | `100` |
| `--follow` | `false` |

Resolves `<run_root>/slurm/<job_id>.{out,err}` and `tail`s over SSH.
Output is raw log text, not JSON. `--follow` is `tail -f`; Ctrl+C exits
`130`.

### `cog logs app` — read the local control-plane event log (plain text / JSONL)

Reads `~/.cog/events.jsonl`.

| Flag | Default |
|------|---------|
| `--lines` | `100` |
| `--follow` | `false` |
| `--kind` / `--agent-id` / `--session-handle` | filters |
| `--format` | `pretty` (or `jsonl`) |

---

## Error code reference

Cog emits `{"error":{"code": ..., "message": ..., "details": ...}}` on
stderr. Match on `code`, not the prose. Codes you'll actually see:

| Code | Meaning · How to fix |
|------|---------------------|
| `CLUSTER_NOT_FOUND` | No matching cluster in registry and no default set. Run `cog cluster add --set-default`. |
| `CLUSTER_CONFIG_INVALID` | Cluster config malformed / paths invalid. Fix the YAML or the `cluster add` flags. |
| `CLUSTER_ALREADY_EXISTS` | `cluster add` without `--overwrite` on an existing name. Pass `--overwrite` or pick a new name. |
| `REPO_NOT_FOUND` | `--repo` is not a git repository. Point at the Megatron-LM checkout (not the cog clone). |
| `RUN_ROOT_OUTSIDE_SCRATCH` | `--run-root` is outside the cluster's `scratch_root`. Prefer `--run-name` and let cog compose the path. |
| `SESSION_NOT_FOUND` | No local session record for that handle. Did you `session start` it on this machine? |
| `SESSION_CLUSTER_MISMATCH` | Session was started against a different cluster. Use `--cluster-name` matching the session's origin. |
| `ENV_LOCK_TIMEOUT` | Another `ensure-env` holds the build lock. Wait, or remove a stale lock file on scratch. |
| `IMAGE_IMPORT_FAILED` | `enroot import` failed. Inspect `import_stderr` in the JSON output. |
| `SUBMIT_FAILED` | `srun` submission itself failed (before the job ran). Check Slurm account / partition / quota. |
| `EXEC_TIMEOUT` | `session exec` foreground wait exceeded `--wait-timeout`. Either raise it or use `--detach` and poll. |
| `REQUEST_NOT_FOUND` | No execution request with that ID in the session. |
| `REQUEST_ALREADY_TERMINAL` | Tried to cancel a finished request — no-op. |
| `SSH_CONNECTION_FAILED` | SSH to the cluster failed. Reproduce with `ssh -o BatchMode=yes <host> true`. |
| `SCRATCH_UNWRITABLE` | Can't write to scratch root. `mkdir -p` it; check ownership. |
| `SLURM_UNREACHABLE` | `squeue`/`sacct` not on the cluster host's PATH. Are you pointed at a login node? |
| `JOB_NOT_FOUND` | Slurm has no record of the job ID. Probably too old (sacct purged) or a typo. |
| `INTERNAL_ERROR` | Unexpected cog crash. Re-run with `--debug` to get a `traceback` in `details`. |

---

## Logs and artifacts paths

### One-shot `submit`

- Slurm: `<run-root>/slurm/<jobid>.{out,err}`
- App: `<run-root>/logs/app`, `<run-root>/logs/torchrun`, `<run-root>/checkpoints`, `<run-root>/tensorboard`, `<run-root>/artifacts.json`

### Persistent `session exec`

- Per-exec logs: `.../sessions/<handle>/exec/runs/<request-id>/{stdout,stderr}.log`
- Per-exec artifacts: `.../sessions/<handle>/exec/runs/<request-id>/run/{logs/app,logs/torchrun,checkpoints,tensorboard,artifacts.json}`

---

## Common workflows (in this repo)

### 1. Initial setup

See Steps 1-3 above (`cluster add --set-default`, populate
`~/.cog/setup.env`, `cog cluster ls` to verify).

### 2. Warm image + env once per recipe

```bash
source ~/.cog/setup.env
cog prepare-image --repo "$COG_MEGATRON_REPO"
cog ensure-env --repo "$COG_MEGATRON_REPO" \
  --run-name env-warmup --gpus 1 --time 00:20:00 \
  --partition "$COG_INTERACTIVE_PARTITION"
```

### 3. One-shot verification — use `cog submit`

```bash
cog submit \
  --repo "$COG_MEGATRON_REPO" \
  --run-name test-basic \
  --command 'python -m pytest tests/unit_tests/test_basic.py -v -o addopts=' \
  --gpus 1 --time 00:10:00 \
  --partition "$COG_INTERACTIVE_PARTITION"
```

### 4. Iterating on the same code — use `cog session`

See the "Iterating" section earlier in this skill. Rule of thumb: if
you'll run **more than 2 commands** against the same code, start a
session instead of submitting again.

```bash
cog session start --repo "$COG_MEGATRON_REPO" \
  --session-handle iter-debug --gpus 8 --time 03:00:00 \
  --partition "$COG_INTERACTIVE_PARTITION"

cog session exec --session-handle iter-debug --repo "$COG_MEGATRON_REPO" \
  --command 'python -m pytest tests/unit_tests/test_basic.py -v'
# … re-run as many times as needed …

cog session stop --session-handle iter-debug
```

### 5. Reading logs

```bash
cog logs slurm --run-name <run> --job-id <id> --stream both --lines 200
cog logs app --lines 50
```

---

## Agent rules (cargo-culted from the upstream cli-guide)

- Use installed `cog`, not `uv run`, unless running from a fresh checkout without install.
- Treat returned JSON as the source of truth — every payload starts with `"schema_version": 1`.
- Register a cluster once with `cluster add --set-default`, then refer to runs via `--run-name`.
- Prefer `ensure-env` once, then `submit` or `session` commands.
- Prefer `session start` + repeated `session exec` for iterative debugging.
- Prefer `submit` for one-shot verification jobs.
- Use `--dry-run` on `submit`, `ensure-env`, `prepare-image` to preview the plan without side-effects.
- Errors on stderr carry a structured `error.code` — match on that, not on prose.
- Use returned log and artifact paths instead of guessing file locations.
- Use `python -m torch.distributed.run` instead of bare `torchrun` (venv visibility).
- Use `--skip-uv-sync` on `submit` if you've already run `ensure-env` for the same recipe.

---

## Fall-through: when this skill isn't enough

The upstream cli-guide is at
`~/cog/docs/cli-guide.md` (local clone) or
<https://gitlab-master.nvidia.com/shanmugamr/cog/-/blob/main/docs/cli-guide.md>
(online). Open it only if you need:

- A field-by-field JSON schema for a less common command this skill
  summarizes only at a high level.
- Verification that a flag default hasn't changed in a newer cog
  release than this skill was written against.

For everything else, this skill is canonical.
