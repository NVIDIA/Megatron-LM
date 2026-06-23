# Persistent Cache

First-iteration acceleration across job restarts. Caches Triton autotune results,
TorchInductor FX graphs, CUDA PTX→SASS, cuDNN frontend heuristics, NCCL topology,
hybrid_ep NVCC JIT, and dataset index across job boundaries.

On a warm restart the cold compile/autotune storm is paid once and reloaded, so
first-iteration time drops substantially, with sub-second bootstrap extract overhead.

## Architecture (three storage tiers)

```
${PERSISTENT_CACHE}/                         ◄── Lustre (durable)
├── cache_read/<scope>.tar.zst                  read-only at job time
└── cache_write/
    ├── <scope>/<files>                         shared rsync sink, complete only
    └── _nodes/<job>_<host>/<scope>/<files>     per-node cp fallback snapshots

/tmp/mcore_cache_${SLURM_JOB_ID}/<scope>/    ◄── per-node, ephemeral
                                                live working cache
```

**Read = tarball, write = directory tree.** Splitting them avoids Lustre MDT
invalidation storms from concurrent reader/writer access on the same dir, and
lets every job-start be one sequential tarball read instead of thousands of
small-file opens.

## Scripts

| File                  | When                              | What                                              |
| --------------------- | --------------------------------- | ------------------------------------------------- |
| `lib.sh`              | sourced by all                    | shared functions + env defaults                   |
| `bootstrap.sh`        | sourced per rank in rank entrypoint | per-node seed of `/tmp` + per-rank env export    |
| `sidecar.sh`          | spawned `&` once per node         | long-running periodic writeback loop + EXIT trap |
| `writeback.sh`        | invoked by sidecar / save-hook / atexit | one-shot writeback from `/tmp/<scope>/`; rsync writes shared scopes, cp fallback writes per-node snapshots |
| `promote_tarballs.sh` | launcher's `srun -p cpu` between jobs | merge complete writeback sources, then tar to `cache_read/<scope>.tar.zst` |

`writeback.sh` marks a scope dirty after a successful copy. This is intentional:
`rsync -a` preserves cache file mtimes, so a newly copied artifact can otherwise
look older than an existing tarball on Lustre and be missed by mtime-only
promotion.

Partial writebacks are fail-closed. Direct rsync creates
`cache_write/<scope>/.mcore_cache_partial` before copying and removes it only
after a clean `rsync` exit. If `rsync` is missing, the fallback copies into a
job+node-private temp directory, publishes it under `cache_write/_nodes/` only
after the copy succeeds, and leaves `<scope>.partial` on failure. Promotion skips
any source with a partial marker and exits nonzero if only partial sources exist.

## Required env

Set by the launcher before sourcing `bootstrap.sh`:

```bash
export PERSISTENT_CACHE=/path/to/shared_fs/<user>/.cache/mcore_persistent
# Derived (optional override):
export CACHE_READ_DIR="${PERSISTENT_CACHE}/cache_read"
export CACHE_WRITE_DIR="${PERSISTENT_CACHE}/cache_write"
```

Tunables (defaults shown):

```bash
export MCORE_CACHE_SYNC_DELAY_SECONDS=600     # don't rsync before compile is likely done
export MCORE_CACHE_SYNC_FREQUENCY=300         # sidecar loop interval
export MCORE_CACHE_SYNC_JITTER_SECONDS=120    # spread initial rsync across 16 nodes
export MCORE_CACHE_MAX_PERIODIC_WRITEBACKS=0  # default 0 = end-of-job consolidated writeback only
export MCORE_CACHE_SIDECAR_ENABLED=0          # default 0 = sidecar off; opt in with 1 for in-job snapshots
export MCORE_CACHE_SCOPES="triton inductor cuda_ptx hybrid_ep cudnn_fe nccl_topo dataset_idx megatron_fused_kernels"
```

## How to use (launcher integration)

```bash
# 1. In sbatch wrapper, before srun:
export PERSISTENT_CACHE=/path/to/shared_fs/<user>/.cache/mcore_persistent
mkdir -p "${PERSISTENT_CACHE}/cache_read" "${PERSISTENT_CACHE}/cache_write"

# 2. Refresh stale tarballs out-of-band on a CPU partition:
srun -N1 -n1 -p cpu -q cpu-short -A "${SLURM_ACCOUNT}" -t 00:15:00 --quiet \
    bash "${MEGATRON_LM_DIR}/tools/persistent_cache/promote_tarballs.sh" \
    || echo "[CACHE] promote step failed (first job will compile cold)"

# 3. Add the new CLI flags to your training command:
OPTIONS+=" --persistent-cache-read-dir ${PERSISTENT_CACHE}/cache_read"
OPTIONS+=" --persistent-cache-write-dir ${PERSISTENT_CACHE}/cache_write"

# 4. In your rank entrypoint, source bootstrap and spawn sidecar:
if [[ -n "${PERSISTENT_CACHE:-}" ]]; then
    # shellcheck disable=SC1091
    source "${MEGATRON_LM_DIR}/tools/persistent_cache/bootstrap.sh"
    if [[ "${MCORE_CACHE_SIDECAR_ENABLED:-0}" == "1" ]]; then
        bash "${MEGATRON_LM_DIR}/tools/persistent_cache/sidecar.sh" </dev/null &
    fi
fi

# 5. Exec the training command — env vars from bootstrap stick:
exec python pretrain_mamba.py …
```

## Cache scopes

| Scope        | Env var                       | What's cached                                         |
| ------------ | ----------------------------- | ----------------------------------------------------- |
| `triton`     | `TRITON_CACHE_DIR`            | Triton compiled cubins + autotune choices             |
| `inductor`   | `TORCHINDUCTOR_CACHE_DIR`     | torch._dynamo FX graphs + autograd cache              |
| `cuda_ptx`   | `CUDA_CACHE_PATH`             | CUDA driver PTX→SASS                                  |
| `hybrid_ep`  | `HYBRID_EP_JIT_DIR`           | NVCC-JIT compiled .so for DeepEP hybrid_ep dispatcher |
| `cudnn_fe`   | `CUDNN_FRONTEND_CACHE_DIR`    | cuDNN heuristic results                               |
| `nccl_topo`  | `NCCL_TOPO_DUMP_FILE`         | NCCL topology dump (informational; read-side reload unreliable at scale) |
| `dataset_idx`| `--data-cache-path`           | Megatron dataset blend / index files                  |
| `megatron_fused_kernels`| `TORCH_EXTENSIONS_DIR`| Megatron "compiling and loading fused kernels" `torch.utils.cpp_extension.load` build dir (saves 5-17 s/restart) |

## Triton autotune `pre_hook` disk-cache patch

Upstream Triton skips disk-cache lookup for any autotune `Config` carrying a
`pre_hook`, so kernels that register a `pre_hook` (e.g. some Mamba backward
kernels) re-autotune on every warm restart. Two files fix this without
serializing the hook itself:

| File | Role |
|---|---|
| `triton_autotune_disk_cache.py` | Monkey-patches `Autotuner.check_disk_cache` to key configs by a JSON-friendly dict (kwargs + `pre_hook.__qualname__`), persist `pre_hook` configs, and reconstruct + run the local hook on a cache hit. `arm()` registers a `MetaPathFinder` so the patch lands on the first `import triton.runtime.autotuner` regardless of import order; `install()` covers the already-imported case. |
| `_triton_patch_shim/sitecustomize.py` | Auto-loaded by Python's `site` module when its directory is on `PYTHONPATH`; calls `arm()`+`install()`, then chains to the container's apport `sitecustomize.py`. |

Wiring (in your rank entrypoint): prepend the shim dir to `PYTHONPATH` so it arms
before Triton is imported:

```bash
export PYTHONPATH="${MEGATRON_LM_DIR}/tools/persistent_cache/_triton_patch_shim:${PYTHONPATH:-}"
```

Kill switch: `TRITON_AUTOTUNE_PREHOOK_PATCH_DISABLE=1`. Debug: `TRITON_AUTOTUNE_PATCH_DEBUG=1`.

## Failure modes

| Failure | What happens | Recovery |
| ------- | ------------ | -------- |
| Tarball missing | `[CACHE SEED] <scope>: no tarball` | Cold compile, sidecar populates `cache_write/`, next promote builds tarball |
| Tarball corrupt | `tar --zstd -xf` fails non-fatally | Cold compile; bash logs warning |
| `/dev/shm` noexec | exec-probe falls back to `/tmp` | Automatic |
| `rsync` not in container image | sidecar uses per-node cp snapshot fallback | Automatic, logged once; promotion merges completed snapshots later |
| Writeback fails midway | Partial marker is left beside the source | Promotion skips it and returns nonzero if no complete source exists |
| Job crashes before writeback | `cache_write/` may be empty for this job | No corruption; next job compiles cold |
| Sidecar hangs | atexit terminates it after 60 s wait | Bounded exit delay |
| Lustre temporarily unreachable | Tar/rsync fails | Logged non-fatal; training unaffected |

## How to opt out

Don't pass `--persistent-cache-read-dir` or `--persistent-cache-write-dir`. The
Python side becomes a no-op; the bash side never sources without `PERSISTENT_CACHE`.
