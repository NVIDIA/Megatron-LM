# Megatron-FSDP Examples

Example scripts for training and checkpoint conversion using [Megatron-FSDP](../../docs/user-guide/features/megatron_fsdp.md). These demonstrate recommended configurations for Llama 3 8B and DeepSeek-V3 671B models, as well as checkpoint format conversion between `torch_dist` (N-D parallel) and `fsdp_dtensor` formats.

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
    tensorboard_logs/llama3_8b_fsdp_fp8 \
    /path/to/tokenizer \
    /path/to/data_prefix
```

| Positional Argument | Default | Description |
|---------------------|---------|-------------|
| `$1` — Checkpoint path | `checkpoints/llama3_8b_fsdp_fp8` | Directory for saving and loading checkpoints. |
| `$2` — TensorBoard path | `tensorboard_logs/llama3_8b_fsdp_fp8` | Directory for TensorBoard logs. |
| `$3` — Tokenizer | `MOCK` | Path to a tokenizer model, or `MOCK` for `NullTokenizer`. |
| `$4` — Data path | `MOCK` | Data prefix for training data, or `MOCK` for mock data. |

#### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_MEGATRON_FSDP` | `1` | Set to `1` to enable Megatron-FSDP. Set to `0` to train with standard DDP. |
| `SHARDING_STRATEGY` | `optim_grads_params` | FSDP sharding strategy (ZeRO-3). Options: `no_shard`, `optim`, `optim_grads`, `optim_grads_params`. |
| `OUTER_SHARDING_STRATEGY` | `no_shard` | DP-Outer sharding strategy for HSDP/HFSDP. Options: `no_shard`, `optim`. |
| `MASTER_ADDR` | `localhost` | Master node address for distributed training. |
| `MASTER_PORT` | `6000` | Master node port. |
| `NODE_RANK` | `0` | Rank of the current node. |

#### Configuration Summary

- **Model**: Llama 3 8B (GQA with 32 heads / 8 KV groups, RoPE, SwiGLU, RMSNorm)
- **Parallelism**: TP=1, CP=1, PP=1, 8 GPUs per node, FSDP ZeRO-3
- **Precision**: FP8 (hybrid format) with BF16 training and BF16 gradient reduction
- **Batch size**: micro-batch=1, global-batch=128, sequence length=8192
- **Optimizations**: NCCL user buffers, FSDP double buffering, manual registration, meta-device initialization, per-token loss, overlapped grad-reduce and param-gather

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
