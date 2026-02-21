# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Megatron-LM** is NVIDIA's reference implementation for training large language models at scale using advanced parallelism strategies. It consists of two main components:

- **Megatron Core** (`megatron/core/`): Production library providing composable building blocks, GPU-optimized kernels, and parallelism strategies (TP, PP, DP, EP, CP)
- **Megatron-LM** (top-level): Reference training scripts and examples for GPT, LLaMA, DeepSeek, Qwen, Mamba, and other models

The codebase supports training models from 2B to 462B+ parameters across thousands of GPUs with state-of-the-art performance (up to 47% MFU on H100 clusters).

## Current Branch: nonuniform-tp

This branch implements **Nonuniform Tensor Parallelism (NTP)**, a fault tolerance mechanism that allows training to continue when GPU failures occur within a tensor-parallel group.

### Key Changes

**New Module**: `megatron/core/distributed/nonuniform_tp.py` (404 lines)
- Implements nonuniform TP where a subset of TP ranks ("spares") provide fault tolerance
- Supports arbitrary non-contiguous GPU failures across all parallelism dimensions (DP, CP, PP)
- Core ranks handle computation; spare ranks enable recovery from failures

**Modified Files**:
- `megatron/core/parallel_state.py`: Added NTP configuration support to `initialize_model_parallel()`
- `megatron/core/distributed/distributed_data_parallel_config.py`: New fields for NTP config
  - `tp_base`: Base tensor parallel size (e.g., 8)
  - `tp_spares`: Number of spare ranks (e.g., 2 for reduced TP=6)
  - `num_reduced_tp_dp_ranks`: How many DP ranks use reduced TP
  - `non_active_ranks_per_dp`: Mapping of (DP, CP, PP) rank to list of non-active local TP ranks
- `megatron/core/distributed/param_and_grad_buffer.py`: Parameter resharding for NTP
- `megatron/core/optimizer/optimizer.py`: Optimizer integration

### NTP Concepts

- **tp_base**: Original/healthy tensor parallel size (e.g., 8 GPUs)
- **tp_spares**: Number of spare GPUs for fault tolerance (e.g., 2)
- **Reduced TP**: Actual working size = tp_base - tp_spares (e.g., 6 active GPUs)
- **Healthy ranks**: DP replicas using full tp_base (no failures)
- **Unhealthy ranks**: DP replicas with failures, using reduced TP
- **send_splits/recv_splits**: Parameter resharding metadata for synchronizing between healthy and unhealthy ranks

### Example NTP Configuration

```python
from megatron.core.distributed import DistributedDataParallelConfig

# Configure NTP with 2 spare ranks out of 8
ddp_config = DistributedDataParallelConfig(
    tp_base=8,              # Original TP size
    tp_spares=2,            # 2 spares = 6 active ranks
    num_reduced_tp_dp_ranks=1,  # First DP rank uses reduced TP
    non_active_ranks_per_dp={
        (0, 0, 0): [2, 5],  # DP=0, CP=0, PP=0 has GPU 2,5 failed
        (0, 1, 0): [1, 3],  # DP=0, CP=1, PP=0 has GPU 1,3 failed
    }
)
```

## Development Setup

### Installation

```bash
# Install with dev dependencies
pip install "setuptools<80.0.0,>=77.0.0" "packaging>=24.2"
pip install --no-build-isolation .[mlm,dev]

# Or use Docker (recommended)
docker run --runtime=nvidia --gpus all -it --rm \
  -v $(pwd):/workspace/megatron \
  -e PIP_CONSTRAINT= \
  nvcr.io/nvidia/pytorch:25.04-py3
```

### Running Tests

```bash
# Run all unit tests
pytest tests/unit_tests/

# Run specific test file
pytest tests/unit_tests/test_optimizer.py

# Run specific test
pytest tests/unit_tests/test_optimizer.py::TestOptimizer::test_specific_case

# Run with verbose output
pytest tests/unit_tests/ -v -s

# Run functional tests (requires GPUs)
pytest tests/functional_tests/
```

### Linting and Formatting

```bash
# Run pre-commit hooks (black, pylint, isort)
pre-commit run --all-files

# Format code with black
black megatron/core/ tests/unit_tests/

# Run pylint
pylint megatron/core/

# Sort imports
isort megatron/core/
```

Black formatting uses `--skip-magic-trailing-comma` and `--skip-string-normalization` options.

### Training Examples

```bash
# Simple training example (2 GPUs, mock data)
torchrun --nproc_per_node=2 examples/run_simple_mcore_train_loop.py

# LLaMA-3 8B with FP8 (8 GPUs)
./examples/llama/train_llama3_8b_fp8.sh

# GPT-3 pretraining
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 2 \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --micro-batch-size 4 \
    --global-batch-size 32 \
    --seq-length 1024 \
    --train-iters 100000 \
    --data-path /path/to/data
```

## Architecture Overview

### Directory Structure

```
megatron/core/              # Megatron Core library (composable components)
├── models/                 # Model architectures (GPT, LLaMA, Mixtral, etc.)
├── transformer/            # Transformer building blocks (attention, MLP, layers)
├── tensor_parallel/        # Tensor parallelism implementations
├── pipeline_parallel/      # Pipeline parallelism implementations
├── distributed/            # Distributed training (FSDP, DDP, NTP)
│   ├── distributed_data_parallel.py
│   ├── nonuniform_tp.py    # NEW: Nonuniform TP for fault tolerance
│   └── param_and_grad_buffer.py
├── optimizer/              # Distributed optimizers
├── datasets/               # Dataset utilities
├── inference/              # Inference engines
└── export/                 # Model export (TensorRT-LLM, etc.)

megatron/training/          # Training utilities
megatron/inference/         # Inference server
megatron/legacy/            # Legacy components

Top-level scripts:          # Entry points for training
├── pretrain_gpt.py
├── pretrain_bert.py
├── pretrain_mamba.py
├── pretrain_vlm.py
└── train_rl.py

examples/                   # Training recipes and tutorials
tools/                      # Data preprocessing, checkpoint conversion
tests/                      # Unit and functional tests
```

### Key Architecture Concepts

#### Parallelism Strategies

Megatron supports multiple parallelism dimensions that can be combined:

1. **Tensor Parallelism (TP)**: Splits individual layers across GPUs
   - `--tensor-model-parallel-size N`: N-way tensor parallelism
   - Use with Sequence Parallelism for memory efficiency
   - Recommended for large models where layers don't fit on single GPU

2. **Pipeline Parallelism (PP)**: Splits model depth across GPUs
   - `--pipeline-model-parallel-size N`: N pipeline stages
   - `--virtual-pipeline-model-parallel-size M`: Virtual stages for load balancing
   - Reduces memory per GPU but adds pipeline bubbles

3. **Data Parallelism (DP)**: Replicates model across GPUs
   - Standard DDP or FSDP (ZeRO-1/2/3)
   - `--use-custom-fsdp`: Megatron's optimized FSDP (~15% faster)
   - `--data-parallel-sharding-strategy`: optim, optim_grads, optim_grads_params

4. **Context Parallelism (CP)**: Splits long sequences across GPUs
   - `--context-parallel-size N`: N-way context parallelism
   - For handling very long sequences (8K+)
   - Communication types: p2p, a2a, allgather

5. **Expert Parallelism (EP)**: Distributes MoE experts across GPUs
   - `--expert-model-parallel-size N`: N-way expert parallelism
   - Required for large MoE models (Mixtral, DeepSeek-V3)
   - **Important**: When using EP with TP, Sequence Parallelism must be enabled

6. **Nonuniform Tensor Parallelism (NTP)** *(This Branch)*: Fault tolerance for TP
   - Allows subset of TP ranks to fail while continuing training
   - Configure via `DistributedDataParallelConfig` with tp_base/tp_spares
   - Healthy and unhealthy DP ranks synchronize via parameter resharding

#### Global Rank Calculation

```
global_rank = local_tp_rank +
              dp_rank * tp_size * cp_size +
              pp_rank * tp_size * cp_size * dp_size
```

For NTP, use `tp_base` instead of actual `tp_size` when calculating DP rank.

#### Process Group Initialization

The `initialize_model_parallel()` function in `parallel_state.py` creates all process groups:
- Tensor parallel groups (TP)
- Pipeline parallel groups (PP)
- Data parallel groups (DP)
- Context parallel groups (CP)
- Combined groups (TP+CP, model parallel)

For NTP, pass `ntp_config` dict to enable nonuniform TP mode.

### Data Preprocessing

Data must be preprocessed into binary format:

```bash
python tools/preprocess_data.py \
    --input data.jsonl \
    --output-prefix processed_data \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /path/to/tokenizer.model \
    --workers 8 \
    --append-eod
```

Input format: JSONL with `{"text": "..."}` per line
Output: `.bin` and `.idx` files for efficient loading

### Transformer Layer Structure

Each transformer layer typically contains:
- **Self-Attention**: Multi-head attention with optional TP sharding
  - QKV projections are TP-sharded across `num_attention_heads`
  - Output projection is column-parallel
- **MLP**: Feed-forward network with optional TP sharding
  - First projection (up) is TP-sharded across `ffn_hidden_size`
  - Second projection (down) is row-parallel
- **LayerNorm/RMSNorm**: Normalization layers
- **Residual connections**: With optional LayerNorm

For NTP, call `ntp_init(layer, ddp_config)` after layer creation to set up parameter resharding metadata.

### Communication Patterns

- **TP communication**: All-reduce within TP group for row-parallel layers, identity for column-parallel
- **PP communication**: Point-to-point between adjacent pipeline stages
- **DP communication**: All-reduce gradients across DP group
- **CP communication**: All-gather or point-to-point for sequence chunks
- **NTP communication**: Custom parameter resharding between healthy/unhealthy DP replicas

### Performance Optimizations

Key flags for performance:
- `--fp8-hybrid`: FP8 training (Hopper, Ada, Blackwell GPUs)
- `--attention-backend`: FlashAttention via Transformer Engine (default, recommended)
- `--recompute-activations`: Activation checkpointing for memory savings
- `--overlap-grad-reduce`: Overlap DP gradient communication with backward pass
- `--overlap-param-gather`: Overlap parameter gathering with computation
- `--use-distributed-optimizer`: Shard optimizer state across DP ranks

## Important Conventions

### Parameter Attributes

Tensor-parallel parameters have special attributes:
- `param.tensor_model_parallel` (bool): Whether parameter is TP-sharded
- `param.partition_dim` (int): Which dimension is sharded (0 for rows, 1 for columns)
- `param.send_splits` (list): For NTP, how to split parameter when sending to other ranks
- `param.recv_splits` (list): For NTP, how to split parameter when receiving from other ranks

### Model Configuration

Models use config dataclasses (e.g., `TransformerConfig`) that contain:
- Architecture params: num_layers, hidden_size, num_attention_heads, ffn_hidden_size
- Parallelism settings: tensor_model_parallel_size, pipeline_model_parallel_size
- Training settings: fp16, bf16, params_dtype, pipeline_dtype
- Optimization settings: recompute_granularity, activation_checkpointing

### Checkpointing

Checkpoints are saved with distributed optimizer state sharding:
- Model state is saved per rank or in a consolidated format
- Optimizer state is sharded across DP ranks when using distributed optimizer
- Use `tools/checkpoint/` utilities for conversion between formats

## Working with Nonuniform TP

When modifying or testing NTP code:

1. **Process group reconfiguration**: NTP modifies TP/CP groups after initialization
   - Healthy ranks keep full tp_base size
   - Unhealthy ranks recreate groups with only active ranks
   - Non-active (spare) ranks exit after group creation

2. **Parameter initialization**: Call `ntp_init(layer, ddp_config)` after creating transformer layers
   - Computes send_splits/recv_splits for healthy ranks
   - Unhealthy ranks skip initialization (no resharding needed)

3. **Gradient synchronization**: Modified in `param_and_grad_buffer.py`
   - Healthy ranks reshard parameters before sending to unhealthy ranks
   - Unhealthy ranks synchronize directly without resharding

4. **Testing NTP**: Use `test_ntp()` function in `nonuniform_tp.py` for basic validation

## Related Files

When working on specific features, these files are commonly modified together:

**Distributed Training**:
- `megatron/core/parallel_state.py`: Process group management
- `megatron/core/distributed/*.py`: DDP, FSDP, NTP implementations
- `megatron/core/optimizer/optimizer.py`: Distributed optimizer

**Model Architecture**:
- `megatron/core/models/*.py`: Model definitions
- `megatron/core/transformer/*.py`: Layer building blocks
- `gpt_builders.py`, `mamba_builders.py`: Model factory functions

**Tensor Parallelism**:
- `megatron/core/tensor_parallel/*.py`: TP implementations
- `megatron/core/transformer/dot_product_attention.py`: TP-aware attention

**Pipeline Parallelism**:
- `megatron/core/pipeline_parallel/*.py`: PP schedules and communication
