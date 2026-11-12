# Megatron-FSDP Examples

Example scripts for training and checkpoint conversion using [Megatron-FSDP](../../docs/user-guide/features/megatron_fsdp.md). These demonstrate recommended configurations for Llama 3 8B and DeepSeek-V3 671B models, a DeepSeek-V4 Pro proxy with the Muon optimizer, as well as checkpoint format conversion between `torch_dist` (N-D parallel) and `fsdp_dtensor` formats.

## Scripts

### `train_llama3_8b_fsdp_h100_fp8.sh`

Single-node training script for **Llama 3 8B** using Megatron-FSDP with FP8 precision on H100 GPUs. Uses `torchrun` for local distributed training and supports both mock data (for benchmarking) and real datasets.

#### Usage

Run from the root of the Megatron-LM repository:

```bash
# With mock data (default, for benchmarking)
bash examples/megatron_fsdp/train_llama3_8b_fsdp_h100_fp8.sh

# With real data
bash examples/megatron_fsdp/train_llama3_8b_fsdp_h100_fp8.sh \
    checkpoints/llama3_8b_fsdp_fp8 \
    /path/to/data_prefix \
    /path/to/tokenizer \
    nsys_profiles/llama3_8b_fsdp_fp8 \
    tensorboard_logs/llama3_8b_fsdp_fp8

# With Nsight Systems profiling (steps 4–6 on rank 0)
NSYS_PROFILE=1 bash examples/megatron_fsdp/train_llama3_8b_fsdp_h100_fp8.sh

# Without uv (use the ambient `python`)
USE_UV=0 bash examples/megatron_fsdp/train_llama3_8b_fsdp_h100_fp8.sh
```

| Positional Argument | Default | Description |
|---------------------|---------|-------------|
| `$1` — Checkpoint Path | `checkpoints/llama3_8b_fsdp_fp8` | Directory for saving and loading checkpoints. |
| `$2` — Data Path | `MOCK` | Data prefix for training data, or `MOCK` for mock data. |
| `$3` — Tokenizer | `MOCK` | Path to a tokenizer model, or `MOCK` for `NullTokenizer`. |
| `$4` — NSight Profiling Path | `nsys_profiles/llama3_8b_fsdp_fp8` | Output path (without extension) for the `.nsys-rep` file when `NSYS_PROFILE=1`. |
| `$5` — TensorBoard Path | `tensorboard_logs/llama3_8b_fsdp_fp8` | Directory for TensorBoard logs. |

#### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_UV` | `1` | Set to `1` to launch via `uv run` (project venv). Set to `0` to use the ambient `python`. |
| `NSYS_PROFILE` | `0` | Set to `1` to wrap the launch in `nsys profile`. Captures steps 4–6 on rank 0 via `--capture-range=cudaProfilerApi`, with CUDA graph node tracing and CUDA memory usage enabled. Output goes to the path in `$4`. |
| `USE_MEGATRON_FSDP` | `1` | Set to `1` to enable Megatron-FSDP. Set to `0` to train with standard DDP. |
| `SHARDING_STRATEGY` | `optim_grads_params` | FSDP sharding strategy (ZeRO-3). Options: `no_shard`, `optim`, `optim_grads`, `optim_grads_params`. |
| `OUTER_SHARDING_STRATEGY` | `no_shard` | DP-Outer sharding strategy for HSDP/HFSDP. Options: `no_shard`, `optim`. |
| `MASTER_ADDR` | `localhost` | Master node address for distributed training. |
| `MASTER_PORT` | `29500` | Master node port. |
| `NODE_RANK` | `0` | Rank of the current node. |

#### Configuration Summary

- **Model**: Llama 3 8B (GQA with 32 heads / 8 KV groups, RoPE, SwiGLU, RMSNorm)
- **Parallelism**: TP=1, CP=1, PP=1, 8 GPUs per node, FSDP ZeRO-3
- **Precision**: FP8 (hybrid format) with BF16 training and BF16 gradient reduction
- **Batch size**: micro-batch=1, global-batch=128, sequence length=8192
- **Optimizations**: NCCL user buffers, FSDP double buffering, manual registration, meta-device initialization, per-token loss, overlapped grad-reduce and param-gather
- **Launch**: `[uv run] [nsys profile ...] python -m torch.distributed.run ... pretrain_gpt.py ...` — the `uv run` and `nsys profile` prefixes are toggled by `USE_UV` and `NSYS_PROFILE` respectively.

---

### `sbatch_mfsdp_deepseek_v3.sh`

Multi-node SLURM training script for **DeepSeek-V3** (671B MoE) using Megatron-FSDP. Submits an `sbatch` job with containerized execution via `srun`.

#### Usage

Set the required configuration variables and submit:

```bash
export MEGATRON_PATH=/path/to/Megatron-LM
export CONTAINER_IMAGE=/path/to/container.sqsh   # or docker image URL
export OUTPUT_PATH=/path/to/output
export DATA_PATH=/path/to/training/data

bash examples/megatron_fsdp/sbatch_mfsdp_deepseek_v3.sh
```

Before running, update the `#SBATCH` directives and `--container-mounts` in the script to match your cluster configuration.

#### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MEGATRON_PATH` | *(required)* | Path to the Megatron-LM repository. |
| `CONTAINER_IMAGE` | *(required)* | Container image (`.sqsh` file or Docker URL). |
| `OUTPUT_PATH` | *(required)* | Base directory for checkpoints, TensorBoard logs, SLURM logs, and Nsight profiles. |
| `DATA_PATH` | *(required)* | Training data prefix path. |
| `USE_MEGATRON_FSDP` | `1` | Enable Megatron-FSDP. Set to `0` for standard DDP. |
| `SHARDING_STRATEGY` | `optim_grads_params` | FSDP sharding strategy (ZeRO-3). |
| `TP` | `1` | Tensor parallel size. |
| `EP` | `8` | Expert parallel size. |
| `MBS` | `4` | Micro-batch size. |
| `GBS` | `2048` | Global batch size. |
| `PROFILE` | `0` | Set to `1` to enable Nsight Systems profiling (steps 10–12). |
| `WANDB` | `1` | Set to `1` to enable Weights & Biases logging. Requires `WANDB_API_KEY`. |
| `COMMENT` | N/A | Tag appended to W&B experiment names and Nsight profile filenames. |

#### Configuration Summary

- **Model**: DeepSeek-V3 (61 layers, 256 routed experts, top-8 routing, Multi-Latent Attention, MTP)
- **Parallelism**: TP=1, EP=8, CP=1, FSDP ZeRO-3
- **Precision**: BF16
- **MoE**: Flex dispatcher with HybridEP backend, grouped GEMM, sigmoid routing with expert bias, auxiliary sequence loss
- **Recomputation**: Selective recomputation of `mlp`, `moe`, `mla_up_proj`, and `layernorm` modules
- **Optimizations**: NCCL user buffers, FSDP double buffering, meta-device initialization, per-token loss, overlapped grad-reduce and param-gather
- **Tokenizer**: `deepseek-ai/DeepSeek-V3` via HuggingFace

---

### `sbatch_mfsdp_muon_deepseek_v4_pro.sh`

SLURM batch script for a **DeepSeek-V4 Pro** proxy (Pro width and expert layout, 7 decoder blocks) trained with Megatron-FSDP v2 and the **Muon** optimizer on 64 GB200 GPUs (16 nodes x 4). It runs 12 iterations on mock data and is meant for measuring the Muon optimizer step.

#### Usage

```bash
export MEGATRON_PATH=/path/to/Megatron-LM
export CONTAINER_IMAGE=/path/to/container.sqsh   # or docker image URL
export CONTAINER_MOUNTS=/lustre:/lustre          # must include MEGATRON_PATH and OUTPUT_PATH
export OUTPUT_PATH=/path/to/output

sbatch --account=<account> --partition=<partition> --segment=16 \
    examples/megatron_fsdp/sbatch_mfsdp_muon_deepseek_v4_pro.sh
```

On NVL72 systems, `--segment=16` keeps the 16 nodes in one NVLink domain. Rank logs, Nsight reports, and caches go to `$OUTPUT_PATH/mfsdp-muon-deepseek-v4-pro-<job id>/`.

#### Requirements

A container with PyTorch 2.13 (NGC PyTorch 26.07) and Transformer Engine 2.18, plus the DeepSeek-V4 attention kernels, installed for example into a virtual environment on shared storage whose Python is passed as `PYTHON`:

```bash
python -m venv --system-site-packages /path/to/dsv4-env
/path/to/dsv4-env/bin/python -m pip install --no-deps nvidia-cudnn-frontend==1.29.1 \
    apache-tvm-ffi==0.1.11 nvidia-cutlass-dsl==4.8.0 nvidia-cutlass-dsl-libs-base==4.8.0 \
    nvidia-cutlass-dsl-libs-core==4.8.0 nvidia-cutlass-dsl-libs-cu13==4.8.0
FLASH_MLA_DISABLE_SM90=1 /path/to/dsv4-env/bin/python -m pip install --no-deps --no-build-isolation \
    'git+https://github.com/deepseek-ai/FlashMLA.git@b7643bd54521f563b839b98289b5cd048c062ba2'
TORCH_CUDA_ARCH_LIST=10.0 /path/to/dsv4-env/bin/python -m pip install --no-deps --no-build-isolation \
    'git+https://github.com/Dao-AILab/fast-hadamard-transform.git@f134af63deb2df17e1171a9ec1ea4a7d8604d5ca'
```

The tokenizer, `unsloth/DeepSeek-V3`, is downloaded from Hugging Face into `HF_HOME` (default `$OUTPUT_PATH/cache/huggingface`).

#### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MEGATRON_PATH` | *(required)* | Path to the Megatron-LM repository. |
| `CONTAINER_IMAGE` | *(required)* | Container image (`.sqsh` file or Docker URL). |
| `CONTAINER_MOUNTS` | *(required)* | Container mounts; they must include `MEGATRON_PATH` and `OUTPUT_PATH`. |
| `OUTPUT_PATH` | *(required)* | Shared directory for logs, Nsight reports, and caches. |
| `PYTHON` | `python` | Python interpreter with the kernels above. |
| `PROFILE` | `0` | Set to `1` to record Nsight Systems reports on `PROFILE_RANKS` for steps 9–11. |
| `PROFILE_RANKS` | `0 32 48 63` | Ranks to profile. |
| `EXTRA_ARGS` | N/A | Extra training flags, e.g. `--train-iters 20`. |

#### Measuring the optimizer step

With `PROFILE=1`, each profiled rank writes an Nsight Systems report of steps 9–11 to `nsys/` in the run directory. In each step, the optimizer's GPU work follows the last backward kernels; on these ranks it is mostly Muon's Newton-Schulz GEMMs, overlapped with the gather and scatter of the next and previous parameter groups.

#### Configuration Summary

- **Model**: DeepSeek-V4 Pro width (hidden 7168, 128 heads, 384 routed experts, top-6 routing) with 7 hybrid decoder blocks (HCA/CSA attention and MoE), mHC with 4 residual streams, and one MTP depth
- **Parallelism**: TP=1, PP=1, CP=1, EP=64; dense parameters in Megatron-FSDP v2 ZeRO-3 over 64 ranks
- **Precision**: BF16 with MXFP8 (FP8 parameter gather)
- **MoE**: Flex dispatcher with HybridEP backend, grouped GEMM, `sqrtsoftplus` routing with expert bias
- **Optimizer**: Muon (`deepseekv4` coefficients, 10 Newton-Schulz steps, momentum 0.95) for matrices, Adam for the rest
- **Batch size**: micro-batch=1, global-batch=64, sequence length=4096

---

### `sbatch_checkpoint_convert.sh`

SLURM batch script for converting checkpoints from **`torch_dist`** (N-D parallel) format to **`fsdp_dtensor`** (Megatron-FSDP) format. This enables resuming training under Megatron-FSDP from checkpoints originally saved with tensor/pipeline/expert parallelism.

#### Prerequisites

Before converting, you need a `param_to_param_group_map.json` file. Generate it by running a `torch_dist` training job with the `--dump-param-to-param-group-map` flag, then converting the output:

```bash
# 1. Run a training job (or trivial experiment) with the dump flag
--dump-param-to-param-group-map /path/to/param_to_param_group_map

# 2. Convert the dumped map to JSON
python tools/checkpoint/checkpoint_inspector.py \
    print-torch-dcp-in-json /path/to/param_to_param_group_map
```

See the [Checkpoint Conversion](../../docs/user-guide/features/megatron_fsdp.md#checkpoint-conversion) section in the Megatron-FSDP docs for details.

#### Usage

Set the required configuration variables, update the checkpoint paths in `RUN_CMD`, and submit:

```bash
export MEGATRON_PATH=/path/to/Megatron-LM
export CONTAINER_IMAGE=/path/to/container.sqsh
export OUTPUT_PATH=/path/to/output

bash examples/megatron_fsdp/sbatch_checkpoint_convert.sh
```

Before running, you must edit the script to fill in:
- The input `torch_dist` checkpoint path
- The output `fsdp_dtensor` checkpoint path
- The path to `param_to_param_group_map.json`
- The `#SBATCH` directives and `--container-mounts` for your cluster

#### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MEGATRON_PATH` | *(required)* | Path to the Megatron-LM repository. |
| `CONTAINER_IMAGE` | *(required)* | Container image (`.sqsh` file or Docker URL). |
| `OUTPUT_PATH` | *(required)* | Base directory for SLURM logs. |

#### Conversion Command

The script runs `checkpoint_inspector.py convert-torch-dist-to-fsdp-dtensor` with the `--swiglu` flag (for models using SwiGLU activations). Remove `--swiglu` if converting a non-SwiGLU model.

## Further Reading

- [Megatron-FSDP User Guide](../../docs/user-guide/features/megatron_fsdp.md) — full feature guide, API reference, and sharding strategy documentation.
- [Megatron-FSDP on PyPI](https://pypi.org/project/megatron-fsdp/) — standalone `fully_shard` API.
- [Megatron-FSDP Source](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/distributed/fsdp/src) — implementation source code.
