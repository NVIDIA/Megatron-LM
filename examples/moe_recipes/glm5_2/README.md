# GLM-5.2 GB300 MoK recipes

These native recipes package the full-model P4 (Mixture-of-Kittens) benchmark
workloads. They use mock data and force-balanced routing, not a convergence
training configuration. No external launch toolkit is required.

## Workloads

All three recipes use 78 transformer layers (3 dense and 75 MoE), 256 experts,
hidden size 6144, repeated MTP depth 7, MXFP8 GEMMs and parameter gather, cuDNN DSA, and
full-iteration CUDA Graphs. Parallelism is TP1/PP4/EP64/configured CP64 with
expert TP1, on 256 GB300 GPUs across 64 four-GPU nodes.

The recipes explicitly enable `muon_split_mla_qkv` together with
`muon_split_qkv_per_head`. The separate MLA opt-in preserves the benchmark's
per-head up-projection optimizer behavior on current dev; omitting it would
orthogonalize those weights as whole matrices. Historical timings remain
attributed to their original measured source, not to this refreshed stack.

| Recipe | Valid tokens/sample | Realized CP | GBS | Microbatches/step | Recompute | MoK macro |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| [THD 4K](gb300/mxfp8_THD4K_256GPU_TP1PP4EP64CP64_full_cg_mok.yaml) | 3,968 | 1 | 2,048 | 32 | Selective layernorm | 40,960 |
| [THD 128K](gb300/mxfp8_THD128K_256GPU_TP1PP4EP64CP64_full_cg_mok.yaml) | 126,976 | 32 | 64 | 32 | Selective layernorm | 40,960 |
| [THD 1M](gb300/mxfp8_THD1M_256GPU_TP1PP4EP64CP64_full_cg_mok.yaml) | 1,015,808 | 64 | 8 | 8 | Full uniform-1 | 131,072 |

Every step carries 8,126,464 valid global tokens. The pipeline layouts are:

- 4K and 128K: `Et*22|t*24|t*24|t*8m*7L`
- 1M: `Et*10|t*12|t*12|t*12|t*12|t*12|t*8|m*7L`, an interleaved schedule with two
  virtual stages per PP rank (22/24/20/12 decoder layers per rank) and
  `no_overlap_p2p_communication`, because full-iteration CUDA Graphs are
  validated with batched PP send/recv only.

With DSA index sharing every fourth layer, every pipeline stage or virtual stage
must start at a layer that computes its own top-k, so layouts move in steps of
four layers. In same-allocation A/B runs on 256 GB300 (4K and 1M at source
`2c9bd94108`, 128K at `2d7d052aee`), the layouts above cut the median step time
by 8.2% (4K), 3.7% (128K) and 13.7% (1M) against the previous
`Et*18|t*24|t*24|t*12m*7L` (4K, 128K) and `Et*22|t*24|t*24|t*8m*7L` (1M).
MoK's minibatch size is 4096. 4K and 128K share one layout; 1M differs in
layout, recompute policy and schedule, so the cases are not a controlled
context-length comparison.

The fixed-length mock samples produce a fixed realized dynamic-CP schedule.
`cuda_graph_static_dynamic_cp` allows full-iteration capture of that schedule;
these recipes do not establish support for arbitrary changing dynamic-CP
shapes under one captured graph.

Each recipe runs 32 iterations: steps 1–3 warm up, step 4 captures, steps 5–12
warm up replay, and steps 13–32 are the 20 timing samples.

## Required runtime artifacts

The base image and sealed artifacts below are external prerequisites, not
included in this example. The base image alone is insufficient. Reproduction
is blocked without access to the internal registry and the qualified artifacts;
no complete public bootstrap or public artifact download is supplied.

Use this exact aarch64 image, preserving its CUDA, TransformerEngine, cuDNN,
and CUTLASS stack:

```text
gitlab-master.nvidia.com:5005/arch_moe_exploration/agentic-mcore-dev/dsv4-gb200-torch2604-nvrx060-tepr2831@sha256:e598e2a568fbf34e4471bb8e84c9f260e3698982eb3b06bd87bfee877e9b2455
```

The original exported SquashFS SHA256 was
`0c46ff6b56c1d683eaf8ab996718e7235662133328720cd7fb7fff17bda0af4b`.
This is distinct from the OCI image digest; a newly exported archive need not
have identical bytes.

Expose the following layout at the same absolute path inside every node's
container and set `GLM52_DEPENDENCIES` to its root:

```text
GLM52_DEPENDENCIES/
├── mixture-of-kittens/
│   ├── mok/_C.cpython-312-aarch64-linux-gnu.so
│   └── third_party/ThunderKittens/
├── emerging_optimizers-0.5.0a0-py3-none-any.whl
└── cudnnfe-1.21.1-qoffset-gridx/
    └── cudnn/  # complete package, including its compiled module
```

| Artifact | Required identity |
| --- | --- |
| MoK commit | `a2ad2d0ce2366c1817d15498cdbaf8ead5966117` |
| MoK committed tree | `7d70aa508a8c49abb70fb611221413bcc21158be` |
| ThunderKittens gitlink | `1c3920d993404dd49a6d4c7267ea11d583bd5c68` |
| ThunderKittens committed tree | `0d4c1fef62c3322428ea2c0bbaea2c4a86006766` |
| ThunderKittens MXFP8 MN-major patch SHA256 | `e001b2d0b6e4088bcedb5cef49bf6811d1757b9e9e9835ae50744ee54b8f633d` |
| MoK CPython 3.12/aarch64 SM103 extension SHA256 | `722573258dad5e54df92621f1fac36e618a3eb98b4cc7569694842d863ff3607` |
| Emerging Optimizers source commit | `a44d1f83a4950b445f2a77ca71177c5f033d45d5` |
| Emerging Optimizers committed tree | `5eae22a77e27a115942395d9d77e7f14726e1d63` |
| Emerging Optimizers wheel SHA256 | `dc143b4ab0cf3a0e6d945a695dadc710f8338f08a5abc6a107d80044c8b7ce1e` |
| cuDNN Frontend q-offset/GridX payload SHA256 | `8a8dc72734dbe76b97bfa8a0047083784570073b6493f67b04d58122456862ea` |
| cuDNN `_compiled_module.cpython-312-aarch64-linux-gnu.so` SHA256 | `e48aedababaeb72397073d041eadf1e512892eb754d99b9c0c8d06e979964c28` |

The overlay retains cuDNN Frontend 1.21.1 and CUTLASS DSL 4.4.2. Its q-offset
ABI and GridX fix are required; a q-offset-only package is not equivalent.
Do not overlay a different CUTLASS installation. Compute its payload hash
from the overlay root, including the compiled module and excluding only
`PROVENANCE.md`:

```bash
find . -type f ! -name PROVENANCE.md -print0 \
  | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum
```

Public source repositories are
[MoK's MCore integration fork](https://github.com/QiZhangNV/mixture-of-kittens/tree/qizhang/mcore-integration),
[ThunderKittens](https://github.com/HazyResearch/ThunderKittens), and
[Emerging Optimizers](https://github.com/NVIDIA-NeMo/Emerging-Optimizers).
To prepare the pinned MoK source independently of any launch toolkit:

```bash
git clone https://github.com/QiZhangNV/mixture-of-kittens.git \
  "${GLM52_DEPENDENCIES}/mixture-of-kittens"
git -C "${GLM52_DEPENDENCIES}/mixture-of-kittens" checkout --detach \
  a2ad2d0ce2366c1817d15498cdbaf8ead5966117
git -C "${GLM52_DEPENDENCIES}/mixture-of-kittens" submodule update --init \
  third_party/ThunderKittens
"${GLM52_DEPENDENCIES}/mixture-of-kittens/scripts/prepare_thunderkittens.sh"
```

Verify the committed trees, gitlink, patch file, and patched ThunderKittens
`git diff --binary` against the identities above. Obtain the sealed extension,
optimizer wheel, and complete GridX overlay from the experiment maintainer.
Install the extension at the shown path and verify its SHA256. A fresh
`make -B ARCH=SM103` rebuild of the same source did not reproduce the measured
extension bytes; it requires separate qualification. The optimizer wheel must
retain distribution metadata, so a source-only import is not a substitute.

The recipes set `PYTHONPATH` to MoK source, the optimizer wheel, the cuDNN
overlay, and the current Megatron root, in that order. `run.py` infers
`MEGATRON_ROOT`; `CUDNN_PATH` is `/usr/lib/aarch64-linux-gnu` in the pinned image.
The launcher only translates recipe arguments and environment; it does not
acquire or hash-validate dependencies. Verify the sealed identities and actual
module import origins inside the container before requesting an allocation.

## Run from the Megatron-LM root

Use Python 3.10 or newer with PyYAML for the local dry-run; Torch is not needed. Render
arguments and environment without starting distributed training:

```bash
export GLM52_DEPENDENCIES=/absolute/path/to/qualified-dependencies
python3 examples/moe_recipes/glm5_2/run.py \
  examples/moe_recipes/glm5_2/gb300/mxfp8_THD128K_256GPU_TP1PP4EP64CP64_full_cg_mok.yaml \
  --dry-run
```

A dry-run is a command audit, not runtime or dependency qualification. Use the
4K or 1M filename to render the other workloads.

For training, arrange the allocation and container mounts yourself. Each PP
stage's 64 consecutive ranks must occupy one independently verified NVLink
domain; GPU count and node names alone do not prove this topology. Retain
rank-to-node/GPU and NVLink-domain evidence before starting MoK EP64.

Run the following once on each of the 64 nodes inside the pinned container.
Set `NODE_RANK` uniquely to 0–63, ordered by the verified PP/NVLink placement;
use the same reachable `MASTER_ADDR`, available job-unique `MASTER_PORT`, and
`RUN_ID` on every node. Static rendezvous preserves this node-rank assignment:

```bash
export GLM52_DEPENDENCIES=/absolute/path/to/qualified-dependencies
torchrun --nnodes=64 --nproc-per-node=4 --node-rank="${NODE_RANK}" \
  --rdzv-backend=static --master-addr="${MASTER_ADDR}" --master-port="${MASTER_PORT}" \
  --rdzv-id="${RUN_ID}" \
  examples/moe_recipes/glm5_2/run.py \
  examples/moe_recipes/glm5_2/gb300/mxfp8_THD128K_256GPU_TP1PP4EP64CP64_full_cg_mok.yaml
```

No Slurm submission or GPU allocation automation is provided.

## Validation scope

Historical full-model validation used Megatron commit
`6fd9341d529f51e5c259f6511e1c809cf86e0747` (tree
`35aa86d1d77551ec8879abaf21309caccc507513`) and the exact dependency stack above.
Those results do not qualify this entry point on a newer source revision, and
this example makes no new performance claim.

On this source revision one data-loading default differs from the historical
runs. With a sequence-packing scheduler set (these recipes use
`default_dynamic_cp` with the default 2 workers), the training data loader now
keeps about one step of samples in flight instead of PyTorch's default
`prefetch_factor=2`, which the historical timings used. That costs about
+224 MiB (4K), +210 MiB (128K) and +112 MiB (1M) of pinned host memory per
rank; the effect on step time has never been measured at scale.
`--dataloader-prefetch-factor 2` restores the historical window.

Configuration packaging alone does not require retiming the historical
benchmark. Before a general release, validate artifact identities and import
origins, run a small import/operator smoke, then exercise this new entry point
through eager and full-CG warmup, capture, and replay. Check finite LM/MTP/DSA
losses, a changing replay trajectory, and stable memory before any formal
256-GPU performance run.
