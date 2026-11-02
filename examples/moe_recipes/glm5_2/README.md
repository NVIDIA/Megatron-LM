# GLM-5.2 GB300 MoK recipes

These native recipes package the full-model Mixture-of-Kittens (MoK) benchmark
workloads. They use mock data and force-balanced routing, not a convergence
training configuration. Besides Slurm with Pyxis and NVIDIA mlperf-common's
`bindpcie` script, no launch toolkit is required.

## Workloads

All three recipes use 78 transformer layers (3 dense and 75 MoE), 256 experts,
hidden size 6144, repeated MTP depth 7, MXFP8 GEMMs and parameter gather, cuDNN DSA, and
full-iteration CUDA Graphs. Parallelism is TP1/PP4/EP64/configured CP64 with
expert TP1, on 256 GB300 GPUs across 64 four-GPU nodes. With dynamic context
parallelism (`dynamic_context_parallel`), CP64 is the largest group: the scheduler
runs each sample on a CP group no smaller than the smallest power of two that
keeps each GPU within `max_seqlen_per_dp_cp_rank` tokens (4,096 at 4K and 128K,
16,384 at 1M), and it may use a larger group to balance work or to leave no GPU
idle. At that smallest size, these recipes' fixed-length samples fill all 64 GPUs
of the CP64 group in every microbatch, so it is the realized CP that the table
below gives.

The DSA indexer scores keys in MXFP8 with DeepGEMM (`dsa_indexer_precision: mxfp8`,
`dsa_indexer_scorer_backend: deepgemm`) and selects them with DeepSelect
(`dsa_indexer_topk_backend: deepselect`). Megatron uses DeepSelect on these BF16
scores only for top-k calls at least 16,384 keys wide; a call is as wide as the
longest key span among its rows. At 128K and 1M every call uses DeepSelect. At 4K
every training call is narrower (realized CP1, at most 4,096 keys), so the cuDNN
top-k runs in every iteration; only Megatron's pipeline prewarm
(`pipeline_model_parallel_prewarm`), on one synthetic 262,144-token sequence over the
configured CP64, uses DeepSelect there. The prewarm runs before the first iteration,
and Megatron logs its time as part of iteration 1, outside the timing samples
(iterations 13–32). At 1M
the full-recompute replay reuses the top-k of the first forward
(`dsa_indexer_reuse_checkpoint_topk`). MXFP8 scoring selects some keys
differently from BF16 scoring; it needs its own training evidence before production
use. Each recipe runs the indexer trained (`dsa_indexer_loss_coeff 1e-2`, the
default) or frozen (`--variant frozen_indexer`: coefficient 0, so the indexer parameters
take no gradient and the indexer loss is not computed).

The recipes explicitly enable `muon_split_mla_qkv` together with
`muon_split_qkv_per_head`. The separate MLA opt-in preserves the benchmark's
per-head up-projection optimizer behavior on current dev; omitting it would
orthogonalize those weights as whole matrices.

| Recipe | Valid tokens/sample | Realized CP | GBS | Microbatches/step | Recompute | MoK macro |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| [THD 4K](gb300/mxfp8_THD4K_256GPU_TP1PP4EP64CP64_full_cg_mok.yaml) | 3,968 | 1 | 2,048 | 32 | Selective layernorm | 40,960 |
| [THD 128K](gb300/mxfp8_THD128K_256GPU_TP1PP4EP64CP64_full_cg_mok.yaml) | 126,976 | 32 | 64 | 32 | Selective layernorm | 40,960 |
| [THD 1M](gb300/mxfp8_THD1M_256GPU_TP1PP4EP64CP64_full_cg_mok.yaml) | 1,015,808 | 64 | 8 | 8 | Full uniform-1 | 131,072 |

Every step carries 8,126,464 valid global tokens (31,744 per GPU). The pipeline layouts are:

- 4K and 128K: `Et*22|t*24|t*24|t*8m*7L`
- 1M: `Et*10|t*12|t*12|t*12|t*12|t*12|t*8|m*7L`, an interleaved schedule with two
  virtual stages per PP rank (22/24/20/12 decoder layers per rank) and
  `no_overlap_p2p_communication`, because full-iteration CUDA Graphs are
  validated with batched PP send/recv only.

With DSA index sharing every fourth layer, every pipeline stage or virtual stage
must start at a layer that computes its own top-k, so layouts move in steps of
four layers. In same-allocation A/B runs on 256 GB300 with these recipes' settings
and runtime stack (the measured runs described under
[Validation scope](#validation-scope)), the layouts above cut the median step time by 8.2% (4K), 3.7% (128K) and 13.7% (1M)
against the previous `Et*18|t*24|t*24|t*12m*7L` (4K, 128K) and
`Et*22|t*24|t*24|t*8m*7L` (1M). MoK's minibatch size is 4096. 4K and 128K share one
layout; 1M differs in layout, recompute policy and schedule, so the cases are not a
controlled context-length comparison.

The fixed-length mock samples produce a fixed realized dynamic-CP schedule.
`cuda_graph_static_dynamic_cp` allows full-iteration capture of that schedule;
these recipes do not establish support for arbitrary changing dynamic-CP
shapes under one captured graph.

Each recipe runs 32 iterations: iterations 1–3 warm up, iteration 4 captures,
iterations 5–12 warm up replay, and iterations 13–32 are the 20 timing samples.


## Required runtime artifacts

The base image and sealed artifacts below are external prerequisites, not
included in this example. The base image alone is insufficient. Reproduction
is blocked without access to the internal registry and the qualified artifacts;
no complete public bootstrap or public artifact download is supplied.

Use this exact aarch64 image, preserving its CUDA, nvcc, TransformerEngine and
cuDNN:

```text
gitlab-master.nvidia.com:5005/arch_moe_exploration/agentic-mcore-dev/dsv4-gb200-torch2604-nvrx060-tepr2831@sha256:e598e2a568fbf34e4471bb8e84c9f260e3698982eb3b06bd87bfee877e9b2455
```

Pyxis runs it from a SquashFS file on a filesystem that every node mounts. The
measured runs used an export with SHA256
`0c46ff6b56c1d683eaf8ab996718e7235662133328720cd7fb7fff17bda0af4b`
(24,117,202,944 bytes). That file and the three trees below are distributed
internally; request them from the recipes' author. In a full (not shallow)
clone, this prints the author:
`git log --diff-filter=A --format='%an <%ae>' -- ':/examples/moe_recipes/glm5_2/README.md'`.

Exporting the image yourself needs enroot 4.0 or newer, because enroot 3 cannot
import by digest, and a registry token: put
`machine gitlab-master.nvidia.com login <user> password <token with read_registry>`
in `${ENROOT_CONFIG_PATH:-$HOME/.config/enroot}/.credentials`. Run the import on
a node with several tens of GB free under `ENROOT_CACHE_PATH` and
`ENROOT_TEMP_PATH`. Enroot reaches the registry over HTTPS, so the reference
drops `:5005` and separates the registry with `#`. A new export need not have
identical bytes.

```bash
enroot import --output /absolute/path/to/glm52.sqsh \
  'docker://gitlab-master.nvidia.com#arch_moe_exploration/agentic-mcore-dev/dsv4-gb200-torch2604-nvrx060-tepr2831@sha256:e598e2a568fbf34e4471bb8e84c9f260e3698982eb3b06bd87bfee877e9b2455'
```

Three sealed directory trees supply what the image lacks. Obtain them complete:
the DeepSelect and DeepGEMM manifests also list their trees' documents, build
inputs, sources and wheels (`PROVENANCE.md`, `USAGE.md`, `VERSION`, `build/`,
`src/`, `wheels/` and DeepSelect's `patches/`). The `GLM52_DEPENDENCIES`
manifest covers only the runtime payload (`env/`, `site/`,
`mixture-of-kittens/` and the optimizer wheel); that tree's `build/` and
`PROVENANCE.md` are not hashed. Expose each tree at the same absolute path
inside every node's container and set one variable to each root:

```text
GLM52_DEPENDENCIES/          # deps-fe131-dsl46
├── env/                     # sitecustomize.py entry point
├── site/                    # cuDNN Frontend, CUTLASS DSL, quack-kernels, flash-attn-4, apache-tvm-ffi
├── mixture-of-kittens/
│   └── mok/                 # MoK package with _C.cpython-312-aarch64-linux-gnu.so
├── emerging_optimizers-0.5.0a0-py3-none-any.whl
└── MANIFEST.sha256
GLM52_DEEPSELECT/            # deps-deepselect: site/ds_tune, build inputs, MANIFEST.sha256
GLM52_DEEPGEMM/              # deps-deepgemm: site/deep_gemm, build inputs, MANIFEST.sha256
```

| Artifact | Required SHA256 |
| --- | --- |
| `GLM52_DEPENDENCIES/MANIFEST.sha256` | `cfab368e02095de972c50f2834c642aea78d999f96a75893a59f68f0db56ad84` |
| cuDNN Frontend 1.31.0 package payload (command below) | `be284950746367cb85ebe0cdfe56fc0898cc7bdad6263ea6088a3b99fe5ab077` |
| MoK CPython 3.12/aarch64 SM103 extension | `722573258dad5e54df92621f1fac36e618a3eb98b4cc7569694842d863ff3607` |
| Emerging Optimizers wheel | `dc143b4ab0cf3a0e6d945a695dadc710f8338f08a5abc6a107d80044c8b7ce1e` |
| `GLM52_DEEPSELECT/MANIFEST.sha256` | `63f253bcfc5f511e09c706ce1ab5fe440039e56f01e2301f04e41962a14de918` |
| `ds_tune` extension, `site/ds_tune/ds_tune_cuda.cpython-312-aarch64-linux-gnu.so` | `d393ce337024901a1d0599f7c532acb29207b1d1d877592dcc8cdba837ef7033` |
| `GLM52_DEEPGEMM/MANIFEST.sha256` | `0ebbc0493e45bf19626838f6bbbbc980479675d13800d21da1748ba70b596d06` |
| `deep_gemm` extension, `site/deep_gemm/_C.cpython-312-aarch64-linux-gnu.so` | `a5e58cc6fb09d6fb48f687c3ba7e138b932ceb694a391be39e9be2f4d28e4048` |

Check each root with `sha256sum MANIFEST.sha256` against the table and
`sha256sum --quiet -c MANIFEST.sha256` from inside it. Hash the cuDNN payload
from `GLM52_DEPENDENCIES/site` exactly as follows; the `./` prefix is part of
the hashed listing:

```bash
find ./cudnn -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum
```

The builds come from these sources. The shipped trees cannot verify these
identities; they identify the builds:

| Build | Source identity |
| --- | --- |
| MoK | commit `a2ad2d0ce2366c1817d15498cdbaf8ead5966117`, committed tree `7d70aa508a8c49abb70fb611221413bcc21158be` |
| ThunderKittens (MoK submodule) | gitlink `1c3920d993404dd49a6d4c7267ea11d583bd5c68`, committed tree `0d4c1fef62c3322428ea2c0bbaea2c4a86006766`, MXFP8 MN-major patch SHA256 `e001b2d0b6e4088bcedb5cef49bf6811d1757b9e9e9835ae50744ee54b8f633d` |
| Emerging Optimizers | commit `a44d1f83a4950b445f2a77ca71177c5f033d45d5`, committed tree `5eae22a77e27a115942395d9d77e7f14726e1d63` |
| DeepSelect | commit `bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18` plus three patches |
| DeepGEMM | commit `5081f8513af57e61696de4f4f04b0e707ffb503e` |

Do not overlay a different `cudnn` or CUTLASS installation.

- `GLM52_DEPENDENCIES` holds cuDNN Frontend 1.31.0, the official aarch64 wheel's
  package with three files of `deepseek_sparse_attention/indexer_forward` patched
  to add `clean_logits=False` (no dense `-inf` score prefill). It also holds the
  unmodified PyPI wheels of CUTLASS DSL 4.6.2, quack-kernels 0.6.4, flash-attn-4
  4.0.0b29 and apache-tvm-ffi 0.1.12, next to MoK and the optimizer wheel. Python
  imports `env/sitecustomize.py` at startup. It puts `site/` ahead of the image's
  packages and stops the process (exit 78) if another `cudnn` package, such as
  the `cudnnfe-1.21.1-qoffset-gridx` overlay of older GLM-5.2 dependency trees, is
  on the path. Do not run Python with `-I`, `-E` or `-S`.
- `GLM52_DEEPSELECT` holds `ds_tune 0.1.0+bfa4507.p3`: DeepSelect `bfa4507` with
  three GLM-5.2 patches (a 2048 top-k tier, per-configuration window tables,
  private kernel namespaces). DeepSelect has no release.
- `GLM52_DEEPGEMM` holds `deep_gemm 2.8.2+5081f85`, an untagged build whose module
  reports version 2.8.2, which step 2 prints. DeepGEMM compiles its kernels at first
  use with the image's nvcc (see below).

Public source repositories are
[MoK's MCore integration fork](https://github.com/QiZhangNV/mixture-of-kittens/tree/qizhang/mcore-integration),
[ThunderKittens](https://github.com/HazyResearch/ThunderKittens),
[Emerging Optimizers](https://github.com/NVIDIA-NeMo/Emerging-Optimizers),
[DeepSelect](https://github.com/deepseek-ai/DeepSelect) and
[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM). Rebuilding an extension from
source does not reproduce the measured bytes (a fresh MoK `make -B ARCH=SM103`
did not) and needs its own qualification. The optimizer wheel must retain
distribution metadata, so a source-only import is not a substitute.

The recipes set `PYTHONPATH` to the stack entry point, MoK, the optimizer wheel,
DeepSelect, DeepGEMM and the current Megatron root, in that order. `run.py`
infers `MEGATRON_ROOT`; `CUDNN_PATH` is `/usr/lib/aarch64-linux-gnu` in the pinned
image. The launcher only translates recipe arguments and environment; it does
not acquire or hash-validate dependencies. Verify the sealed identities, and
the import directories that step 2 below prints inside the container, before
starting the 64-node run.

## Run from the Megatron-LM root

Use Python 3.10 or newer with PyYAML for the local dry-run; Torch is not needed. Render
arguments and environment without starting distributed training:

```bash
export GLM52_DEPENDENCIES=/absolute/path/to/deps-fe131-dsl46
export GLM52_DEEPSELECT=/absolute/path/to/deps-deepselect
export GLM52_DEEPGEMM=/absolute/path/to/deps-deepgemm
export DG_JIT_CACHE_DIR=/absolute/path/to/deepgemm-jit-cache
python3 examples/moe_recipes/glm5_2/run.py \
  examples/moe_recipes/glm5_2/gb300/mxfp8_THD128K_256GPU_TP1PP4EP64CP64_full_cg_mok.yaml \
  --dry-run
```

A dry-run is a command audit, not runtime or dependency qualification. Use the
4K or 1M filename to render the other workloads, and add `--variant frozen_indexer`
to render the frozen indexer.

The steps below use Slurm with Pyxis, as the measured runs did. Work from the
Megatron-LM root, on a filesystem that every node mounts, in a clean shell.
`srun` forwards the shell's environment into every rank, but values that the
image or enroot's configuration set take precedence: the `environ.d` files and
`hooks.d` scripts in the cluster's enroot directory (by default `/etc/enroot`)
and in your own `${ENROOT_CONFIG_PATH:-$HOME/.config/enroot}`, which enroot
applies even though the container does not mount `$HOME`. Enroot's PyTorch
hook, for example, sets `MASTER_ADDR` and `MASTER_PORT`, so step 1 exports the
rendezvous as `GLM52_MASTER_ADDR` and `GLM52_MASTER_PORT`. The recipes and
`srun_rank.sh` set the NCCL, CUDA, PyTorch, Transformer Engine and OpenMP
variables they need, so export no other `NCCL_*`, `CUDA_*`, `TORCH_*`, `NVTE_*`
or `OMP_*` variables from the shell, and set none in your own enroot
`environ.d` or `hooks.d`.

Allocate 64 nodes exclusively as four 16-node segments, each inside one NVL72
domain, on a partition and QoS that admit 64 exclusive nodes, for example
`--nodes=64 --ntasks-per-node=4 --gpus-per-node=4 --segment=16 --exclusive
--time=2:00:00`. On partitions without GPU GRES, leave out `--gpus-per-node`
here and `--gpus-per-task` in step 2; every step then sees all four GPUs. In the
measured runs and in the recipe run (both under [Validation scope](#validation-scope)),
each training run took 11–13.5 minutes at 4K and 128K and 16–17.5 minutes at 1M, and
logged its first iteration 3.1–4.1 minutes after it started. Steps 1 and 2 take 1–2
minutes, so two hours cover all six runs (the recipe run, which ran all six in one
job, took 1 hour 23 minutes) and one hour covers one context length's two runs, with
the JIT caches deleted after the job (see step 3). Keep the four variables above set
and define:

```bash
export GLM52_RECIPE=examples/moe_recipes/glm5_2/gb300/mxfp8_THD128K_256GPU_TP1PP4EP64CP64_full_cg_mok.yaml
# Logs and per-rank JIT caches, on a filesystem that every node mounts and that permits exec.
export GLM52_RUN_DIR=/absolute/path/to/run-directory
export GLM52_BINDPCIE=/absolute/path/to/bindpcie  # executable, see step 3
mkdir -p "${GLM52_RUN_DIR}" "${DG_JIT_CACHE_DIR}"
CPUS_PER_TASK=31  # a quarter of the node's schedulable cores (31 on the measured nodes)
# Mount each filesystem that holds the paths above (the Megatron root, the three trees,
# DG_JIT_CACHE_DIR, GLM52_RUN_DIR and bindpcie) at the same path. The measured runs, whose paths
# were under /scratch, mounted /lustre:/lustre,/scratch:/scratch. $HOME is not mounted.
CONTAINER=(--container-image=/absolute/path/to/glm52.sqsh
           --container-mounts=/absolute/path/to/filesystem:/absolute/path/to/filesystem
           --container-workdir="${PWD}" --no-container-mount-home)
LAYOUT=(--exact --nodes=64 --ntasks-per-node=4 --distribution=block:block
        --cpus-per-task="${CPUS_PER_TASK}")
```

1. **Check rank placement and the NVLink fabric.** Each PP stage's 64
   consecutive ranks must occupy one NVLink domain, and GPU count and node names
   alone do not prove this topology. The probe below runs with the training
   step's layout: every rank reports its node, the node's IMEX channels and its
   GPU's `nvidia-smi -q`. [`check_placement.py`](check_placement.py) then
   requires four ranks per node in block order on one driver version, every GPU's
   fabric registration completed with no fabric health problem (a degraded
   NVLink route trains slower while everything else works), an IMEX channel on
   every node, and one fabric clique per PP stage. Its `PLACEMENT OK` line names
   the driver version and the GPUs' power limits, and it prints rank 0's node.

   ```bash
   srun -l "${LAYOUT[@]}" bash -c 'echo "node=${SLURMD_NODENAME} localid=${SLURM_LOCALID}" \
       "imex=$(ls /dev/nvidia-caps-imex-channels 2>/dev/null | tr "\n" " ")"
     nvidia-smi -q -i "${SLURM_LOCALID}"' > "${GLM52_RUN_DIR}/placement-${SLURM_JOB_ID}.txt"
   GLM52_MASTER_ADDR=$(python3 examples/moe_recipes/glm5_2/check_placement.py \
     "${GLM52_RUN_DIR}/placement-${SLURM_JOB_ID}.txt")
   export GLM52_MASTER_ADDR GLM52_MASTER_PORT=29500  # any free port on that node
   ```

   If the check reports a problem, do not train on this allocation; with
   `GLM52_MASTER_ADDR` empty, `srun_rank.sh` refuses to start. The measured
   runs used Slurm's own block order, and reordering ranks (for example with
   `SLURM_HOSTFILE`) also changes their CPU binding. Keep the probe output as
   evidence.

   Step 1 checks placement and the NVLink fabric, not the network between the
   NVL72 domains. In Slurm job 50915, the first attempt at the recipe run under
   [Validation scope](#validation-scope), step 1 passed, but the first two training
   runs (4K, indexer trained and frozen) failed 2.4 and 3.4 minutes after they
   started, before their first iteration, with NCCL warnings
   `NET/IB ... IBV_WC_RETRY_EXC_ERR(12)` and then `ncclRemoteError`. Each `NCCL WARN`
   line names the reporting node and the IP address of its peer, and every failing
   connection had one end in the same NVL72 domain. NCCL prints these warnings only
   at `NCCL_DEBUG=WARN` or a more verbose level, which the measured cluster's enroot
   `environ.d` sets. Without them, each `ncclRemoteError` message still repeats its
   rank's last warning, with the peer's address, after `Last error:`, and the
   `[rank<N>]:` prefix of PyTorch's `ProcessGroupNCCL` error line gives that rank,
   whose node step 1's probe output lists. If this happens, cancel the job, report
   the failing connections to the cluster's administrators and allocate again
   without that domain's nodes (Slurm's `--exclude`); the recipe run's allocation
   excluded them.

2. **Prepare the run.** Build Megatron's dataset helpers once in the
   container. Otherwise rank 0 compiles them at start-up, which needs a writable
   checkout and races when two jobs start from one checkout. Then fill the
   DeepGEMM JIT cache on one GPU. DeepGEMM compiles its indexer kernels at first
   use into `DG_JIT_CACHE_DIR`, so point that at a persistent, writable directory
   shared by all nodes; later runs reuse it. If a training rank cannot compile or
   write there, its indexer falls back to the slower cuDNN MXFP8 scorer with a
   warning (see step 4, Check the run). This DeepGEMM prewarm first prints the version and import directory
   of `cudnn`, `cutlass`, `mok`, `emerging_optimizers`, `ds_tune` and `deep_gemm`.
   Each must lie under its root above, `cudnn` and `cutlass` under
   `GLM52_DEPENDENCIES/site`.

   ```bash
   srun --nodes=1 --ntasks=1 "${CONTAINER[@]}" make -C megatron/core/datasets
   srun --nodes=1 --ntasks=1 --gpus-per-task=1 "${CONTAINER[@]}" \
     python3 examples/moe_recipes/glm5_2/run.py "${GLM52_RECIPE}" --prewarm-deepgemm
   ```

3. **Train** with one task per GPU. Add `--variant frozen_indexer` after
   `"${GLM52_RECIPE}"` to train with a frozen indexer. The log goes to
   `${GLM52_RUN_DIR}/train-<job id>.<Slurm step id>.log`.

   ```bash
   srun "${LAYOUT[@]}" --mpi=pmix --kill-on-bad-exit=1 --unbuffered "${CONTAINER[@]}" \
     --output="${GLM52_RUN_DIR}/train-%j.%s.log" \
     bash examples/moe_recipes/glm5_2/srun_rank.sh "${GLM52_RECIPE}"
   ```

   Before it starts `run.py`, [`srun_rank.sh`](srun_rank.sh):

   - maps Slurm's task ids to `RANK`, `WORLD_SIZE`, `LOCAL_RANK` and `LOCAL_WORLD_SIZE`,
     and `GLM52_MASTER_ADDR` and `GLM52_MASTER_PORT` to `MASTER_ADDR` and `MASTER_PORT`;
   - sets `OMP_NUM_THREADS=1`, as torchrun does;
   - sets a node-local `TMPDIR`;
   - gives each rank fresh Triton, CuTe DSL and Inductor caches under
     `${GLM52_RUN_DIR}/jit-cache/<job id>.<Slurm step id>/`. Every run gets its own
     directory, so nothing needs deleting between runs; delete these directories after the
     job has ended, because removing one run's caches took 5–7 minutes on the measured
     cluster while all 64 nodes sat idle (4–6 minutes from a login node after the job);
   - runs `run.py` under `bindpcie`, which binds each rank's CPUs and memory to its GPU's NUMA node.

   `bindpcie` is `client/bindpcie` of
   [NVIDIA mlperf-common](https://github.com/NVIDIA/mlperf-common) at commit
   `b42fb7306ae42f40d6a317b40ae99161ea47d6fc` (SHA256
   `d7d78a16d86a0d2a69ec9faa074922236f97dfaf48a4b1242fe87835e106311a`). The
   measured runs used a copy that differs only by an added comment block (SHA256
   `6682d552aea229ce2ee72c4cda64841f95f3b17b99ce5b511a7ee8cdac4ad0a9`). Make it
   executable. It runs with its defaults (`--cpu=node --mem=node`) and uses
   `numactl` from the pinned image.

4. **Check the run.**
   - The log must not contain `dsa_indexer_scorer_backend='deepgemm' is unavailable`.
   - It must also not contain `dsa_indexer_topk_backend='deepselect' is unavailable`. At 4K
     only Megatron's pipeline prewarm (`pipeline_model_parallel_prewarm`, logged as part of
     iteration 1) uses DeepSelect, so there this warning leaves the step time unchanged,
     but it still means that DeepSelect, as installed, cannot run.

   Either warning means a slower cuDNN kernel ran, and the run is not the measured
   configuration. Megatron reports DeepSelect's other fallbacks, such as a call narrower
   than its minimum width (every training call at 4K) or a misaligned score buffer, only
   at INFO level, which these recipes do not log. In each measured run at 128K and 1M,
   the internal runner's observers (see [Validation scope](#validation-scope)) saw
   DeepSelect take every top-k call they counted: 180,224 in each 128K run and 45,056 in
   each 1M run. The step time is the median, over iterations 13–32, of the last rank's
   `elapsed time per iteration (ms)`. Tokens per GPU per second are 31,744 divided by
   that median. TFLOP/s/GPU is the median of Megatron's per-iteration throughput. The
   snippet also prints the step time's p95 (the 19th of the 20 sorted samples) and its
   population coefficient of variation (CV).

   ```bash
   python3 - "${GLM52_RUN_DIR}/train-<job id>.<Slurm step id>.log" <<'EOF'
   import math, re, statistics, sys
   step = re.compile(r'iteration\s+(\d+)/\s*32 \|.*?elapsed time per iteration \(ms\): ([0-9.]+)'
                     r'.*?throughput per GPU \(TFLOP/s/GPU\): ([0-9.]+)')
   rows = {int(i): (float(ms) / 1e3, float(tf)) for i, ms, tf in step.findall(open(sys.argv[1]).read())}
   seconds = [rows[i][0] for i in range(13, 33)]
   median = statistics.median(seconds)
   p95 = sorted(seconds)[math.ceil(0.95 * len(seconds)) - 1]
   cv = 100 * statistics.pstdev(seconds) / statistics.mean(seconds)
   tflops = statistics.median(rows[i][1] for i in range(13, 33))
   print(f'{median:.3f} s/step (p95 {p95:.3f} s, CV {cv:.2f}%), {31744 / median:.1f} tokens/s/GPU, '
         f'{tflops:.1f} TFLOP/s/GPU')
   EOF
   ```

## Validation scope

The table gives the measured runs on 256 GB300 GPUs of the MIS-CDG cluster
(driver 580.173.02, the default 1,400 W GPU power limit), one allocation per
context length: Slurm jobs 11713 (4K) and 11749 (1M) on 2026-10-23 and 12015
(128K) on 2026-10-24. The step-4 snippet gives every value from those runs'
logs. The runs had:

- the arguments and environment that these recipes render, checked flag for
  flag, under an internal runner that also wrapped `pretrain_gpt.py` in
  read-only observers;
- the same SquashFS, artifacts, launch layout, CPU binding and DeepGEMM cache
  preparation, and placement and fabric checks equivalent to step 1;
- the cluster's enroot environment, as `/etc/enroot` on a login node shows (their
  ranks' environment was not recorded, and the account that ran them has no
  enroot configuration directory of its own): `OMP_NUM_THREADS=1` from enroot's
  PyTorch hook, which `srun_rank.sh` now sets on every cluster, and the site's
  NCCL network and logging settings from enroot's `environ.d`
  (`NCCL_IB_HCA==mlx5_0,mlx5_1,mlx5_2,mlx5_3`, `NCCL_IB_SL=1`, `NCCL_IB_TIMEOUT=20`,
  `NCCL_SOCKET_IFNAME=eth0`, `NCCL_MNNVL_ENABLE=1`, `NCCL_CUMEM_ENABLE=1`,
  `NCCL_DEBUG=WARN`). The recipe run described below the table recorded the
  environment that ranks 0 and 255 start with in step 3's layout and container,
  before `srun_rank.sh` runs. Their `OMP_*` and `NCCL_*` variables were
  `OMP_NUM_THREADS=1`, these NCCL settings, the image's `NCCL_VERSION` and
  `NCCL_INSPECTOR_DUMP_DIR=/var/lib/node_exporter/textfile_collector`, whose source
  is not known (neither the login node's `/etc/enroot` nor the submitting shell sets
  it). That variable only sets where NCCL's Inspector profiler plugin writes; the
  Inspector stays disabled unless `NCCL_INSPECTOR_ENABLE=1`, and neither the
  recorded environment nor the recipes set that or `NCCL_PROFILER_PLUGIN`. Whether
  the measured runs' ranks had `NCCL_INSPECTOR_DUMP_DIR` is not known;
- source `2c9bd94108a2a8220fa68ab8f607a1061fe3389b` (4K, 1M) or
  `2d7d052aee28fad7f71793c18d926b9265594e9a` (128K), earlier revisions of the
  commit that added these recipes. Outside `examples/`, `2d7d052aee` is
  identical to `67512b59adec3348ee7c993fb43408f654d1ce1d`, the parent of the
  commit that adds these recipes, and `2c9bd94108` differs from it only in
  tests, docs and, in `megatron/`, comments, a cuDNN Frontend version gate that
  FE 1.31 passes either way, and a configuration-time pipeline-layout check.
  From the Megatron-LM root, `git diff --stat 67512b59ad -- . ':(exclude)examples'`
  lists the files in which your checkout differs from it outside `examples/` and
  prints nothing when they match. If Git reports `bad revision`, your clone
  lacks that commit (a shallow clone, or a later, rebased revision of this
  branch); fetch it by its full ID from the repository you cloned.

| Recipe | Indexer | Step time (s) | p95 (s) | CV (%) | Tokens/s/GPU | TFLOP/s/GPU |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4K | trained | 13.505 | 13.897 | 0.93 | 2,350.5 | 818.8 |
| 4K | frozen | 12.593 | 12.993 | 1.02 | 2,520.8 | 875.8 |
| 128K | trained | 15.728 | 16.134 | 0.80 | 2,018.3 | 762.5 |
| 128K | frozen | 14.504 | 14.895 | 0.86 | 2,188.6 | 824.4 |
| 1M | trained | 25.023 | 25.350 | 0.48 | 1,268.6 | 581.5 |
| 1M | frozen | 23.580 | 23.881 | 0.47 | 1,346.2 | 615.6 |

The recipe run, Slurm job 50951 on 2026-11-01, ran the recipes with this README's
steps 1–4, without the internal runner or its observers, and reproduced the step time
of every row within 0.11%. It ran steps 1 and 2 once and step 3 for each row in one
allocation of 64 nodes of the same cluster, in four NVL72 domains with the same
driver and power limit. It used the upstream `bindpcie` and the measured runs'
SquashFS and artifacts, checked beforehand against the hashes above; its allocation
excluded the nodes of the NVL72 domain whose network connections failed in an earlier
attempt (see step 1). Its checkout was identical to `67512b59ad` outside `examples/`
(the source check above printed nothing), and Megatron logged the same arguments as
in the measured runs. For the rows in table order, the step-4 snippet gives step
times of 13.497, 12.598, 15.728, 14.508, 25.041 and 23.605 s, which differ from the
table by -0.06%, +0.04%, 0.00%, +0.03%, +0.07% and +0.11%; no log contains either
warning of step 4.

On nodes like these, a run reproduces a row when its median step time is
within 1% of the table's. The previous layouts' trained runs, each measured in
two allocations, differed by 0.26% at 4K, 0.13% at 128K (one run at each source)
and 0.03% at 1M; their other allocations were Slurm jobs 11495 (4K, 128K) and
11497 (1M) on 2026-10-23. A median further off points to a different code path
(see step 4), different Megatron code (see the source check above), a different
enroot environment or different nodes, such as another driver, power limit or
placement; the `PLACEMENT OK` line of step 1 names the driver and the GPU power
limits. Step 1 does not check the environment, so compare the `NCCL_*` variables that
your ranks start with against the list above, not only the files in your cluster's
enroot `environ.d` and `hooks.d` on a login node: the recipe run's ranks had
`NCCL_INSPECTOR_DUMP_DIR`, which a login node's files do not set. To list your ranks'
variables, run `env` under `srun -l` with step 3's `LAYOUT`, `--mpi=pmix` and
`CONTAINER` options, as the recipe run did. The adapter and interface names
(`NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`) are this cluster's own and may differ on yours,
but other differences can matter: with `NCCL_MNNVL_ENABLE=0` or `NCCL_CUMEM_ENABLE=0`,
for example, NCCL sends traffic between the nodes of an NVL72 domain over the network
instead of NVLink.

In the measured runs, and in five of the recipe run's six training runs, the p95 and
CV come mostly from iterations 21 and 31, which absorb the garbage collection that
`manual_gc_interval: 10` runs after iterations 20 and 30. In the recipe run's 1M
frozen run, iterations 27 and 29 took 0.44 and 0.36 s longer than the median, for no
reason that the log shows; they set its p95 (23.963 s) and raised its CV to 0.67%
(0.50% without them), but moved its median by only 4 ms.

Before a 256-GPU run on a new cluster, verify the artifact identities and check
the import directories that step 2 prints.
