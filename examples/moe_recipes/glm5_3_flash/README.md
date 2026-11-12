# GLM-5.3-Flash GB300 recipes

These native recipes package full-model GLM-5.3-Flash benchmark workloads at 4K, 128K and 1M
tokens per sequence on 64 GB300 GPUs. They use mock data and force-balanced routing, not a
convergence training configuration. Besides Slurm with Pyxis and NVIDIA mlperf-common's
`bindpcie` script, no launch toolkit is required.

## Workloads

All three recipes build the full model: 90 hybrid layers (KDA linear attention, DSA on absorbed
MLA, 288-expert MoE with one shared expert, 3 leading dense MLPs) with 4-stream manifold
hyper-connections (mHC), one MTP layer, hidden size 4096, FP8 blockwise GEMMs with FP8
parameter gather, the precision-aware distributed optimizer (FP32 main parameters and
gradients, BF16 moments), the HybridEP flex dispatcher on 32 SMs and cuDNN DSA kernels.
Parallelism is TP1 and EP32 with expert TP1 on 64 GPUs across 16 four-GPU nodes:

| Recipe | Sequence | PP / CP / DP | GBS (microbatches per pipeline) | Recompute | CUDA graphs | Iterations |
| --- | ---: | --- | ---: | --- | --- | ---: |
| [4K](gb300/fp8_SL4K_64GPU_TP1PP2EP32.yaml) | 4,096 | 2 / 1 / 32 | 2,560 (80) | selective `mhc` | TE, `attn` with the mHC recompute split | 20 |
| [128K](gb300/fp8_SL128K_64GPU_TP1PP2EP32CP8.yaml) | 131,072 | 2 / 8 / 4 | 64 (16) | selective `mhc gdn moe mla_up_proj` | off | 16 |
| [1M](gb300/fp8_SL1M_64GPU_TP1PP1EP32CP64.yaml) | 1,048,576 | 1 / 64 / 1 | 16 (16) | selective `mhc gdn moe mla_up_proj` | off | 12 |

- The pipeline split is 48 | 42 hybrid layers (DSA layers 6 | 5, MTP on the last stage) at 4K
  and 128K; at 1M every GPU holds all 90 layers and MTP.
- Context parallelism splits each sequence zigzag (`cp_comm_type: all_gather`, which DSA
  requires); a rank holds 16,384 tokens at 128K and at 1M.
- The KDA core runs in FP32 at 4K (the model's profile) and in BF16 at 128K and 1M, where the FP32
  core is not supported with CP > 1.
- At 1M, linear cross-entropy (`cross_entropy_fusion_impl: linear`) never materializes the
  logits, and chunked optimizer-state offload keeps the FP32 main parameters and BF16 moments in
  pinned host memory between steps (about 149 GiB per node).
- Long-sequence graphs are off: a non-packed CP DSA forward synchronizes with the host.

**DSA indexer.** GLM-5.3-Flash selects keys by pool (`dsa_indexer_kpool: 4`): the indexer
compresses every four keys into one pooled key and selects 512 pools (2,048 keys) per query plus
the query's own incomplete pool, with the Hugging Face scoring and key-compression arithmetic
(`dsa_indexer_hf_scoring`, `dsa_indexer_kpool_key_dtype_compression`).
`dsa_indexer_kpool_scorer_backend: cudnn` scores the pooled keys with the cuDNN indexer kernel at
ratio 4 instead of the PyTorch FP32 reference scorer; its selection equals the reference's up to
pools whose scores tie at the selection boundary. `dsa_indexer_topk_backend: deepselect` picks
the pools with DeepSelect where a call is wide enough (at 1M, rows of up to 262,144 pools); every
other call, including all calls at 4K and 128K, keeps the cuDNN top-k.

Each recipe runs the indexer frozen (`dsa_indexer_loss_coeff: 0.0`, the released checkpoint's
indexer) or, with `--variant trainable_indexer`, trains it with the sparse pool-level indexer
loss (`dsa_indexer_loss_coeff: 1e-2`, `dsa_indexer_use_sparse_loss`). With cuDNN Frontend 1.31,
TP1, HF scoring and the cuDNN KPool scorer, that loss runs on cuDNN Frontend kernels (score
recompute for the teacher, the scorer's own scores as the student, the sparse indexer backward);
below FE 1.31 it would run in PyTorch, which costs about 136 ms more per 1M call.

## Required runtime artifacts

The base image and sealed artifacts below are external prerequisites, not included in this
example. The base image alone is insufficient. Reproduction is blocked without access to the
internal registry and the qualified artifacts; no complete public bootstrap or public artifact
download is supplied.

Use this exact aarch64 image, preserving its CUDA, nvcc, TransformerEngine and cuDNN:

```text
gitlab-master.nvidia.com:5005/arch_moe_exploration/agentic-mcore-dev/dsv4-gb200-torch2604-nvrx060-tepr2831@sha256:e598e2a568fbf34e4471bb8e84c9f260e3698982eb3b06bd87bfee877e9b2455
```

Pyxis runs it from a SquashFS file on a filesystem that every node mounts. The recipe runs used
an export with SHA256 `0c46ff6b56c1d683eaf8ab996718e7235662133328720cd7fb7fff17bda0af4b`
(24,117,202,944 bytes). That file and the three trees below are distributed internally; request
them from the recipes' author. In a full (not shallow) clone, this prints the author:
`git log --diff-filter=A --format='%an <%ae>' -- ':/examples/moe_recipes/glm5_3_flash/README.md'`.
Exporting the image yourself needs enroot 4.0 or newer, because enroot 3 cannot import by
digest, and a registry token: put
`machine gitlab-master.nvidia.com login <user> password <token with read_registry>` in
`${ENROOT_CONFIG_PATH:-$HOME/.config/enroot}/.credentials`. Run the import on a node with
several tens of GB free under `ENROOT_CACHE_PATH` and `ENROOT_TEMP_PATH`. Enroot reaches the
registry over HTTPS, so the reference drops `:5005` and separates the registry with `#`. A new
export need not have identical bytes.

```bash
enroot import --output /absolute/path/to/glm53-flash.sqsh \
  'docker://gitlab-master.nvidia.com#arch_moe_exploration/agentic-mcore-dev/dsv4-gb200-torch2604-nvrx060-tepr2831@sha256:e598e2a568fbf34e4471bb8e84c9f260e3698982eb3b06bd87bfee877e9b2455'
```

Three sealed directory trees supply what the image lacks. Expose each at the same absolute path
inside every node's container and set one variable to each root:

```text
GLM53_FLASH_DEPENDENCIES/    # deps-fe131-dsl46: env/ (sitecustomize.py entry point), site/, MANIFEST.sha256
GLM53_FLASH_DEEPSELECT/      # deps-deepselect: site/ds_tune, build inputs, MANIFEST.sha256
GLM53_FLASH_FLA/             # fla-core-0.5.1: python/fla, python/fla_core-0.5.1.dist-info, MANIFEST.sha256
```

| Artifact | Required SHA256 |
| --- | --- |
| `GLM53_FLASH_DEPENDENCIES/MANIFEST.sha256` | `cfab368e02095de972c50f2834c642aea78d999f96a75893a59f68f0db56ad84` |
| cuDNN Frontend 1.31.0 package payload (command below) | `be284950746367cb85ebe0cdfe56fc0898cc7bdad6263ea6088a3b99fe5ab077` |
| `GLM53_FLASH_DEEPSELECT/MANIFEST.sha256` | `63f253bcfc5f511e09c706ce1ab5fe440039e56f01e2301f04e41962a14de918` |
| `ds_tune` extension, `site/ds_tune/ds_tune_cuda.cpython-312-aarch64-linux-gnu.so` | `d393ce337024901a1d0599f7c532acb29207b1d1d877592dcc8cdba837ef7033` |
| `GLM53_FLASH_FLA/MANIFEST.sha256` | `e4284e126fae00d828956159094b7eb721468fcff30ff723f5354ac4ba254a00` |

Check each root with `sha256sum MANIFEST.sha256` against the table, then its contents with
`sha256sum --quiet -c MANIFEST.sha256` from inside `GLM53_FLASH_DEPENDENCIES` and
`GLM53_FLASH_DEEPSELECT` and with `sha256sum --quiet -c ../MANIFEST.sha256` from inside
`GLM53_FLASH_FLA/python`, where the FLA manifest's paths start. Hash the cuDNN payload from
`GLM53_FLASH_DEPENDENCIES/site` exactly as follows; the `./` prefix is part of the hashed
listing:

```bash
find ./cudnn -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum
```

- `GLM53_FLASH_DEPENDENCIES` is the `deps-fe131-dsl46` tree of the GLM-5.2 recipes. These
  recipes use its `env/` and `site/`: cuDNN Frontend 1.31.0 (the official aarch64 wheel's package with three
  `deepseek_sparse_attention/indexer_forward` files patched to add `clean_logits=False`) and the
  unmodified PyPI wheels of CUTLASS DSL 4.6.2, quack-kernels 0.6.4, flash-attn-4 4.0.0b29 and
  apache-tvm-ffi 0.1.12. Python imports `env/sitecustomize.py` at startup. It puts `site/` ahead
  of the image's packages and stops the process (exit 78) if another `cudnn` package, such as an
  older cuDNN Frontend 1.21.1 overlay, is on the path. Do not run Python with `-I`, `-E` or `-S`.
  The tree also holds Mixture-of-Kittens and an optimizer wheel, which these recipes do not use.
- `GLM53_FLASH_DEEPSELECT` holds `ds_tune 0.1.0+bfa4507.p3`: DeepSelect commit
  `bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18` with three patches (a 2048 top-k tier,
  per-configuration window tables, private kernel namespaces). DeepSelect has no release.
- `GLM53_FLASH_FLA` holds the package of the pure-Python
  [fla-core 0.5.1](https://pypi.org/project/fla-core/0.5.1/) wheel (flash-linear-attention's
  Triton kernels, which the KDA layers use) under `python/`, ahead of the image's own FLA.

Public source repositories are [DeepSelect](https://github.com/deepseek-ai/DeepSelect) and
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention). Rebuilding
`ds_tune` from source does not reproduce the measured bytes and needs its own qualification. Do
not overlay a different `cudnn`, CUTLASS or `fla` installation.

The recipes set `PYTHONPATH` to FLA, the stack entry point, DeepSelect and the current Megatron
root, in that order. `run.py` infers `MEGATRON_ROOT`; `CUDNN_PATH` is `/usr/lib/aarch64-linux-gnu`
in the pinned image. The launcher only translates recipe arguments and environment; it does not
acquire or hash-validate dependencies. Verify the sealed identities, and the import directories
that step 2 below prints inside the container, before starting the 16-node run.

## Run from the Megatron-LM root

Use Python 3.10 or newer with PyYAML for the local dry-run; Torch is not needed. Render arguments
and environment without starting distributed training:

```bash
export GLM53_FLASH_DEPENDENCIES=/absolute/path/to/deps-fe131-dsl46
export GLM53_FLASH_DEEPSELECT=/absolute/path/to/deps-deepselect
export GLM53_FLASH_FLA=/absolute/path/to/fla-core-0.5.1
python3 examples/moe_recipes/glm5_3_flash/run.py \
  examples/moe_recipes/glm5_3_flash/gb300/fp8_SL1M_64GPU_TP1PP1EP32CP64.yaml --dry-run
```

A dry-run is a command audit, not runtime or dependency qualification. Use the 4K or 128K
filename to render the other workloads, and add `--variant trainable_indexer` to render the
trainable indexer.

The steps below use Slurm with Pyxis. Work from the Megatron-LM root, on a filesystem that every
node mounts, in a clean shell. `srun` forwards the shell's environment into every rank, but
values that the image or enroot's configuration set take precedence (the `environ.d` files and
`hooks.d` scripts of the cluster's and your own enroot configuration); enroot's PyTorch hook, for
example, sets `MASTER_ADDR` and `MASTER_PORT`, so step 1 exports the rendezvous as
`GLM53_FLASH_MASTER_ADDR` and `GLM53_FLASH_MASTER_PORT`. The recipes and `srun_rank.sh` set the
NCCL, CUDA, PyTorch, Transformer Engine and OpenMP variables they need, so export no other
`NCCL_*`, `CUDA_*`, `TORCH_*`, `NVTE_*` or `OMP_*` variables from the shell.

Allocate 16 nodes exclusively inside one NVL72 domain, on a partition and QoS that admit them,
for example `--nodes=16 --ntasks-per-node=4 --gpus-per-node=4 --segment=16 --exclusive
--time=4:00:00`. On partitions without GPU GRES, leave out `--gpus-per-node` here and
`--gpus-per-task` in step 2. Keep the three variables above set and define:

```bash
export GLM53_FLASH_RECIPE=examples/moe_recipes/glm5_3_flash/gb300/fp8_SL1M_64GPU_TP1PP1EP32CP64.yaml
# Logs and per-rank JIT caches, on a filesystem that every node mounts and that permits exec.
export GLM53_FLASH_RUN_DIR=/absolute/path/to/run-directory
export GLM53_FLASH_BINDPCIE=/absolute/path/to/bindpcie  # executable, see step 3
mkdir -p "${GLM53_FLASH_RUN_DIR}"
CPUS_PER_TASK=31  # a quarter of the node's schedulable cores (31 on the measured nodes)
# Mount each filesystem that holds the paths above at the same path; $HOME is not mounted.
CONTAINER=(--container-image=/absolute/path/to/glm53-flash.sqsh
           --container-mounts=/absolute/path/to/filesystem:/absolute/path/to/filesystem
           --container-workdir="${PWD}" --no-container-mount-home)
LAYOUT=(--exact --nodes=16 --ntasks-per-node=4 --distribution=block:block
        --cpus-per-task="${CPUS_PER_TASK}")
```

1. **Check rank placement and the NVLink fabric.** All 64 ranks must occupy one NVLink domain
   (the 1M context-parallel group spans every rank), and GPU count and node names alone do not
   prove this topology. Every rank reports its node, the node's IMEX channels and its GPU's
   `nvidia-smi -q`; [`check_placement.py`](check_placement.py) then requires four ranks per node
   in block order on one driver version, every GPU's fabric registration completed with no
   fabric health problem, an IMEX channel on every node and one fabric clique for all ranks. Its
   `PLACEMENT OK` line names the driver version and the GPUs' power limits, and it prints rank
   0's node.

   ```bash
   srun -l "${LAYOUT[@]}" bash -c 'echo "node=${SLURMD_NODENAME} localid=${SLURM_LOCALID}" \
       "imex=$(ls /dev/nvidia-caps-imex-channels 2>/dev/null | tr "\n" " ")"
     nvidia-smi -q -i "${SLURM_LOCALID}"' > "${GLM53_FLASH_RUN_DIR}/placement-${SLURM_JOB_ID}.txt"
   GLM53_FLASH_MASTER_ADDR=$(python3 examples/moe_recipes/glm5_3_flash/check_placement.py \
     "${GLM53_FLASH_RUN_DIR}/placement-${SLURM_JOB_ID}.txt")
   export GLM53_FLASH_MASTER_ADDR GLM53_FLASH_MASTER_PORT=29500  # any free port on that node
   ```

   If the check reports a problem, do not train on this allocation; with
   `GLM53_FLASH_MASTER_ADDR` empty, `srun_rank.sh` refuses to start.

2. **Prepare the run.** Build Megatron's dataset helpers once in the container (otherwise rank 0
   compiles them at start-up, which needs a writable checkout and races when two jobs start from
   one checkout), then print where the runtime artifacts are imported from. `fla` must lie under
   `GLM53_FLASH_FLA/python`, `cudnn` and `cutlass` under `GLM53_FLASH_DEPENDENCIES/site`, and
   `ds_tune` under `GLM53_FLASH_DEEPSELECT/site`; `cudnn` must report 1.31.0 and `fla` 0.5.1.

   ```bash
   srun --nodes=1 --ntasks=1 "${CONTAINER[@]}" make -C megatron/core/datasets
   srun --nodes=1 --ntasks=1 --gpus-per-task=1 "${CONTAINER[@]}" \
     python3 examples/moe_recipes/glm5_3_flash/run.py "${GLM53_FLASH_RECIPE}" --check-imports
   ```

3. **Train** with one task per GPU. Add `--variant trainable_indexer` after
   `"${GLM53_FLASH_RECIPE}"` to train the indexer. The log goes to
   `${GLM53_FLASH_RUN_DIR}/train-<job id>.<Slurm step id>.log`.

   ```bash
   srun "${LAYOUT[@]}" --mpi=pmix --kill-on-bad-exit=1 --unbuffered "${CONTAINER[@]}" \
     --output="${GLM53_FLASH_RUN_DIR}/train-%j.%s.log" \
     bash examples/moe_recipes/glm5_3_flash/srun_rank.sh "${GLM53_FLASH_RECIPE}"
   ```

   Before it starts `run.py`, [`srun_rank.sh`](srun_rank.sh) maps Slurm's task ids to `RANK`,
   `WORLD_SIZE`, `LOCAL_RANK` and `LOCAL_WORLD_SIZE` and `GLM53_FLASH_MASTER_ADDR` and
   `GLM53_FLASH_MASTER_PORT` to `MASTER_ADDR` and `MASTER_PORT`; sets `OMP_NUM_THREADS=1` and a
   node-local `TMPDIR`; gives each rank fresh Triton, CuTe DSL and Inductor caches under
   `${GLM53_FLASH_RUN_DIR}/jit-cache/<job id>.<Slurm step id>/` (delete them after the job); and
   runs `run.py` under `bindpcie`, which binds each rank's CPUs and memory to its GPU's NUMA node.
   The first iteration compiles and autotunes FLA's Triton kernels for every new shape, which
   takes several minutes.

   `bindpcie` is `client/bindpcie` of [NVIDIA mlperf-common](https://github.com/NVIDIA/mlperf-common)
   at commit `b42fb7306ae42f40d6a317b40ae99161ea47d6fc` (SHA256
   `d7d78a16d86a0d2a69ec9faa074922236f97dfaf48a4b1242fe87835e106311a`). Make it executable. It
   runs with its defaults (`--cpu=node --mem=node`) and uses `numactl` from the pinned image.

4. **Check the run.**
   - The log must contain `DSA KPool selection uses the cuDNN indexer kernel` and must not
     contain `dsa_indexer_topk_backend='deepselect' is unavailable`. The 1M log also contains
     `DSA indexer top-k uses DeepSelect`; at 4K and 128K every call is narrower than DeepSelect's
     minimum width and keeps the cuDNN top-k.
   - With `--variant trainable_indexer`, the log must contain `The trainable KPool indexer loss
     runs on the cuDNN frontend kernels` and must not contain `The trainable KPool indexer loss
     runs in PyTorch`.

   Otherwise a slower path ran, and the run is not the measured configuration. Every rank
   also logs a PyTorch warning with a Triton `CompilationError` traceback, `Encountered an
   exception in identify_accessed_tensors, assuming every input is mutated`, for FLA's
   `layer_norm_gated_fwd_kernel`; it is harmless, because `torch.compile` then treats every
   input of that kernel as written.

   The step time is the median of the last rank's `elapsed time per iteration (ms)` over the
   timing window (4K: iterations 6–20, after the CUDA graph capture; 128K: 5–16; 1M: 4–12).
   Tokens per GPU per second are the step's tokens divided by 64 GPUs and that median.
   TFLOP/s/GPU is the median of Megatron's per-iteration throughput. The snippet also prints
   the step time's p95 (`ceil(0.95 n)`-th of the window's n sorted samples) and its
   population coefficient of variation (CV).

   ```bash
   python3 - "${GLM53_FLASH_RUN_DIR}/train-<job id>.<Slurm step id>.log" <<'EOF'
   import math, re, statistics, sys
   step = re.compile(r'iteration\s+(\d+)/\s*(\d+) \|.*?elapsed time per iteration \(ms\): ([0-9.]+)'
                     r'.*?throughput per GPU \(TFLOP/s/GPU\): ([0-9.]+)')
   rows = {int(i): (int(n), float(ms) / 1e3, float(tf)) for i, n, ms, tf in step.findall(open(sys.argv[1]).read())}
   # Recipe by its iteration count: tokens per step and the first timed iteration.
   tokens, first = {20: (2560 * 4096, 6), 16: (64 * 131072, 5), 12: (16 * 1048576, 4)}[next(iter(rows.values()))[0]]
   window = [i for i in sorted(rows) if i >= first]
   seconds = [rows[i][1] for i in window]
   median = statistics.median(seconds)
   p95 = sorted(seconds)[math.ceil(0.95 * len(seconds)) - 1]
   cv = 100 * statistics.pstdev(seconds) / statistics.mean(seconds)
   tflops = statistics.median(rows[i][2] for i in window)
   print(f'{median:.3f} s/step (p95 {p95:.3f} s, CV {cv:.2f}%) over iterations {window[0]}-{window[-1]}, '
         f'{tokens / 64 / median:.1f} tokens/s/GPU, {tflops:.1f} TFLOP/s/GPU')
   EOF
   ```

## Validation scope

The table gives the recipe runs: steps 1–4 of this README as written, in one allocation of 16
nodes per context length on the MIS-CDG cluster, inside one NVL72 domain (driver 580.173.02, the
default 1,400 W GPU power limit): Slurm jobs 167436 (4K), 167437 (128K) and 167438 (1M) on
2026-11-12. Each job ran steps 1 and 2 once and step 3 twice, with the frozen indexer and with
`--variant trainable_indexer`. The jobs used the SquashFS, the three artifact trees and the
upstream `bindpcie` named above, all of which matched this README's hashes when checked after
the jobs, and the cluster's enroot environment: `OMP_NUM_THREADS=1` from enroot's PyTorch hook,
which `srun_rank.sh` sets anyway, and the site's NCCL network and logging settings from enroot's
`environ.d` (`NCCL_IB_HCA==mlx5_0,mlx5_1,mlx5_2,mlx5_3`, `NCCL_IB_SL=1`, `NCCL_IB_TIMEOUT=20`,
`NCCL_SOCKET_IFNAME=eth0`, `NCCL_MNNVL_ENABLE=1`, `NCCL_CUMEM_ENABLE=1`, `NCCL_DEBUG=WARN`).
Their checkout, an earlier revision of the commit that adds these recipes with no local changes,
was identical to that commit except for this README and the recipe index
(`examples/moe_recipes/README.md`), so they ran its code and recipes.

Every log passed the checks of step 4: all 64 ranks logged the cuDNN KPool selection, all 64
ranks of both 1M runs logged DeepSelect, all 64 ranks of every trained run logged the cuDNN
frontend indexer loss, and no log contains either warning. No iteration was skipped or had a
NaN. The step-4 snippet gives the timing values below. Max allocated is the largest
`max allocated` (MiB) that Megatron logs, in GiB.

| Recipe | Indexer | Step time (s) | p95 (s) | CV (%) | Tokens/s/GPU | TFLOP/s/GPU | Max allocated (GiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4K | trained | 60.169 | 61.271 | 0.71 | 2,723.0 | 310.7 | 219.7 |
| 4K | frozen | 59.171 | 60.803 | 0.92 | 2,768.9 | 315.3 | 219.1 |
| 128K | trained | 45.507 | 46.073 | 0.51 | 2,880.2 | 339.9 | 158.1 |
| 128K | frozen | 44.800 | 45.373 | 0.58 | 2,925.7 | 344.5 | 156.1 |
| 1M | trained | 97.313 | 98.194 | 0.43 | 2,693.8 | 348.5 | 180.3 |
| 1M | frozen | 95.398 | 95.971 | 0.35 | 2,747.9 | 354.8 | 177.6 |

Training the indexer costs 1.7% (4K), 1.6% (128K) and 2.0% (1M) of the frozen step time. At 4K
and 128K the p95 and CV come mostly from iterations 6, 11 and 16, which absorb the garbage
collection that `manual_gc_interval: 5` runs after iterations 5, 10 and 15; without them the
CVs fall to 0.15–0.38%. At 1M the spread does not concentrate on those iterations.

Each row was measured in one allocation, so the spread between allocations is not known. A
median well off the table's points to a different code path (see step 4), different Megatron
code, a different enroot environment or different nodes, such as another driver, power limit
or placement; the `PLACEMENT OK` line of step 1 names the driver and the GPU power limits. Step 1
does not check the environment, so compare the `NCCL_*` variables that your ranks start with
against the list above: run `env` under `srun -l` with step 3's `LAYOUT`, `--mpi=pmix` and
`CONTAINER` options. The adapter and interface names (`NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`) are
this cluster's own and may differ on yours, but other differences can matter: with
`NCCL_MNNVL_ENABLE=0` or `NCCL_CUMEM_ENABLE=0`, for example, NCCL sends traffic between the
nodes of an NVL72 domain over the network instead of NVLink.

Before a 64-GPU run on a new cluster, verify the artifact identities and check the import
directories that step 2 prints.
