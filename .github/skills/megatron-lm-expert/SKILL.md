---
name: Megatron-LM - Large-Scale Transformer Training
description: Megatron-LM is NVIDIA's optimized framework for training large transformer models at scale. It provides efficient implementations of tensor, pipeline, data, and sequence parallelism, enabling training of models from 2B to 1T+ parameters across thousands of GPUs with state-of-the-art performance.
---

## Quick Start

```bash
# Install Megatron-Core via pip
pip install megatron-core

# Or clone for full examples
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM

# Install dependencies
pip install -r requirements.txt

# Download training data (example)
wget https://data.together.xyz/redpajama-data-1T/v1.0.0/book/book.jsonl

# Preprocess data
python tools/preprocess_data.py \
    --input book.jsonl \
    --output-prefix my-gpt3 \
    --tokenizer-type GPT2BPETokenizer \
    --vocab-file gpt2-vocab.json \
    --merge-file gpt2-merges.txt \
    --append-eod

# Train a small GPT model (2.7B)
bash examples/pretrain_gpt_distributed.sh
```

## When to Use This Skill

Use Megatron-LM when you need to:
- Train large language models (10B+ parameters) efficiently
- Implement 3D parallelism (tensor + pipeline + data)
- Achieve maximum GPU utilization for transformer training
- Scale training across multiple nodes and data centers
- Train custom architectures with efficient parallelism
- Convert between Megatron and HuggingFace formats
- Implement state-of-the-art training techniques (Flash Attention, RoPE, etc.)
- Train vision transformers and multimodal models at scale

## Prerequisites

**Platform**: Linux (x86_64, aarch64)

**Required Dependencies**:
- NVIDIA GPUs with Compute Capability 7.0+ (Volta, Ampere, Hopper, Blackwell)
- CUDA 11.8+ or 12.0+
- Python 3.8+
- PyTorch 2.0+ (with CUDA support)
- NVIDIA Apex (mixed precision training)
- NCCL 2.12+ (multi-GPU communication)

**Optional Dependencies**:
- Transformer Engine (FP8 training on Hopper/Blackwell)
- Flash Attention 2.x (efficient attention)
- DeepSpeed (alternative distributed backend)
- Weights & Biases (experiment tracking)
- TensorBoard (visualization)
- NeMo (production deployment)

**Hardware Recommendations**:
- **Small models (< 13B)**: 4-8x A100/H100 40/80GB
- **Medium models (13B-70B)**: 16-64x A100/H100 80GB
- **Large models (175B+)**: 128-1024x A100/H100/H200
- **Trillion-scale**: 1000+ GPUs with InfiniBand or NVLink networking

## Compatibility

| Megatron Version | PyTorch | CUDA | GPU Arch | Key Features |
|-----------------|---------|------|----------|--------------|
| 0.11.0 (latest) | 2.0+ | 12.0+ | Ampere, Hopper, Blackwell | Multi-DC, MoE, FP8 |
| 0.9.0 | 2.0+ | 11.8+ | Ampere, Hopper | Flash Attention 2 |
| 0.7.0 | 1.13+ | 11.8+ | Ampere | Context parallelism |
| 0.6.0 | 1.13+ | 11.7+ | Ampere | Sequence parallelism |

**Supported Architectures**:
- GPT (GPT-2, GPT-3, GPT-NeoX)
- BERT (BERT, RoBERTa)
- T5 (T5, UL2)
- LLaMA (LLaMA, LLaMA-2, LLaMA-3)
- Mistral, Mixtral (MoE)
- Mamba (SSM-based)
- Vision: ViT, DINO, Multimodal VLM
- Custom architectures

## Installation

### Method 1: Pip Install (Megatron-Core Only)

```bash
# Install core library
pip install megatron-core

# With optional dependencies
pip install megatron-core[dev,mlm]

# Verify installation
python -c "import megatron; print(megatron.__version__)"
```

### Method 2: From Source (Full Framework)

```bash
# Clone repository
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM

# Install in development mode
pip install -e .

# Install Apex (for mixed precision)
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir \
    --no-build-isolation --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" ./

# Install Transformer Engine (for FP8 on H100/H200)
pip install git+https://github.com/NVIDIA/TransformerEngine.git

# Install Flash Attention 2
pip install flash-attn --no-build-isolation
```

### Method 3: Docker (Recommended for Production)

```bash
# Pull NGC container with Megatron pre-installed
docker pull nvcr.io/nvidia/pytorch:24.09-py3

# Run container
docker run --gpus all \
           --ipc=host \
           --ulimit memlock=-1 \
           --ulimit stack=67108864 \
           -v /data:/data \
           -it nvcr.io/nvidia/pytorch:24.09-py3

# Inside container, clone Megatron-LM
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
pip install -e .
```

### Method 4: NVIDIA NGC Catalog

```bash
# Download from NGC
ngc registry model download-version nvidia/megatron_lm_345m:1.0

# Or use with NeMo framework
pip install nemo_toolkit[nlp]
```

## Configuration

### Core Training Arguments

```bash
# Basic configuration
DISTRIBUTED_ARGS="
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT
"

# Model architecture
MODEL_ARGS="
    --num-layers=32 \
    --hidden-size=4096 \
    --num-attention-heads=32 \
    --seq-length=2048 \
    --max-position-embeddings=2048 \
    --micro-batch-size=4 \
    --global-batch-size=128
"

# Parallelism configuration
PARALLEL_ARGS="
    --tensor-model-parallel-size=4 \
    --pipeline-model-parallel-size=2 \
    --sequence-parallel \
    --use-distributed-optimizer
"

# Training parameters
TRAINING_ARGS="
    --train-iters=100000 \
    --lr=1.5e-4 \
    --min-lr=1.0e-5 \
    --lr-decay-style=cosine \
    --lr-warmup-iters=2000 \
    --weight-decay=0.1 \
    --clip-grad=1.0 \
    --bf16  # or --fp16
"

# Data configuration
DATA_ARGS="
    --data-path=/data/my-gpt3_text_document \
    --split=949,50,1 \
    --tokenizer-type=GPT2BPETokenizer \
    --vocab-file=gpt2-vocab.json \
    --merge-file=gpt2-merges.txt
"

# Checkpointing
CHECKPOINT_ARGS="
    --save=/checkpoints/gpt-model \
    --load=/checkpoints/gpt-model \
    --save-interval=1000 \
    --eval-interval=100 \
    --eval-iters=10
"

# Logging
LOGGING_ARGS="
    --log-interval=10 \
    --tensorboard-dir=/logs \
    --wandb-project=my-gpt-training \
    --wandb-entity=my-org
"
```

### Environment Variables

```bash
# NCCL configuration (critical for performance)
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0                    # Enable InfiniBand
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1       # IB adapters
export NCCL_SOCKET_IFNAME=eth0             # Network interface
export NCCL_P2P_LEVEL=NVL                  # Use NVLink
export NCCL_NET_GDR_LEVEL=5                # Max GPUDirect RDMA
export NCCL_IB_QPS_PER_CONN=4              # QPs per connection
export NCCL_CROSS_NIC=2                    # Cross-NIC communication

# CUDA settings
export CUDA_DEVICE_MAX_CONNECTIONS=1       # Serializes kernel launches
export CUDA_LAUNCH_BLOCKING=0              # Async kernel launch
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Megatron settings
export MEGATRON_NUM_MICROBATCHES=4         # Microbatches per pipeline stage
export NVTE_FLASH_ATTN=1                   # Enable Flash Attention
export NVTE_FUSED_ATTN=1                   # Fused attention kernels

# Performance tuning
export OMP_NUM_THREADS=8                   # OpenMP threads
export TOKENIZERS_PARALLELISM=false        # Disable tokenizer parallelism
```

### Parallelism Strategy Selection

```python
# Rule of thumb for parallelism configuration
# Total GPUs = TP * PP * DP

# Example 1: 8 GPUs (single node)
# Model: 7B params
TP = 2  # Tensor parallel
PP = 1  # Pipeline parallel
DP = 4  # Data parallel (8 / (2*1))

# Example 2: 64 GPUs (8 nodes x 8 GPUs)
# Model: 70B params
TP = 8  # Split each layer across 8 GPUs
PP = 2  # 2 pipeline stages
DP = 4  # 4 data parallel replicas (64 / (8*2))

# Example 3: 256 GPUs (32 nodes x 8 GPUs)
# Model: 175B params
TP = 8
PP = 8
DP = 4  # 256 / (8*8)

# Example 4: 1024 GPUs
# Model: 1T params
TP = 8
PP = 16
DP = 8  # 1024 / (8*16)
```

## Usage Patterns

### Pattern 1: Basic GPT Training

```bash
#!/bin/bash
# train_gpt_basic.sh

GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6000

# Distributed setup
DISTRIBUTED_ARGS="
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT
"

# GPT-3 2.7B configuration
python -m torch.distributed.launch $DISTRIBUTED_ARGS \
    pretrain_gpt.py \
    --num-layers=32 \
    --hidden-size=2560 \
    --num-attention-heads=32 \
    --seq-length=2048 \
    --max-position-embeddings=2048 \
    --micro-batch-size=4 \
    --global-batch-size=32 \
    --tensor-model-parallel-size=2 \
    --pipeline-model-parallel-size=1 \
    --train-iters=100000 \
    --lr=1.5e-4 \
    --min-lr=1.0e-5 \
    --lr-decay-style=cosine \
    --lr-warmup-iters=2000 \
    --weight-decay=0.1 \
    --clip-grad=1.0 \
    --bf16 \
    --data-path=/data/my-gpt3_text_document \
    --split=949,50,1 \
    --tokenizer-type=GPT2BPETokenizer \
    --vocab-file=gpt2-vocab.json \
    --merge-file=gpt2-merges.txt \
    --save=/checkpoints/gpt-2.7b \
    --load=/checkpoints/gpt-2.7b \
    --save-interval=1000 \
    --eval-interval=100 \
    --eval-iters=10 \
    --log-interval=10 \
    --tensorboard-dir=/logs/gpt-2.7b
```

### Pattern 2: Multi-Node Training with SLURM

```bash
#!/bin/bash
#SBATCH --job-name=megatron-gpt
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Get node information
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=6000
export WORLD_SIZE=$((SLURM_NNODES * 8))

echo "MASTER_ADDR: $MASTER_ADDR"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "SLURM_PROCID: $SLURM_PROCID"

# Configure NCCL
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1
export NCCL_IB_DISABLE=0

# GPT-3 70B configuration with 3D parallelism
srun --mpi=pmix python pretrain_gpt.py \
    --num-layers=80 \
    --hidden-size=8192 \
    --num-attention-heads=64 \
    --seq-length=2048 \
    --max-position-embeddings=2048 \
    --micro-batch-size=1 \
    --global-batch-size=128 \
    --tensor-model-parallel-size=8 \
    --pipeline-model-parallel-size=4 \
    --sequence-parallel \
    --use-distributed-optimizer \
    --train-iters=100000 \
    --lr=1.0e-4 \
    --min-lr=1.0e-5 \
    --lr-decay-style=cosine \
    --lr-warmup-iters=2000 \
    --weight-decay=0.1 \
    --clip-grad=1.0 \
    --bf16 \
    --data-path=/scratch/data/pile_text_document \
    --split=949,50,1 \
    --tokenizer-type=GPT2BPETokenizer \
    --vocab-file=/data/vocab/gpt2-vocab.json \
    --merge-file=/data/vocab/gpt2-merges.txt \
    --save=/scratch/checkpoints/gpt-70b \
    --load=/scratch/checkpoints/gpt-70b \
    --save-interval=500 \
    --eval-interval=100 \
    --eval-iters=10 \
    --log-interval=1 \
    --tensorboard-dir=/scratch/logs/gpt-70b \
    --wandb-project=gpt-70b-training \
    --distributed-backend=nccl
```

### Pattern 3: Data Preprocessing

```python
# preprocess_custom_data.py
import json
import argparse
from megatron.data import indexed_dataset

def preprocess_data(input_file, output_prefix, tokenizer):
    """
    Preprocess raw text data for Megatron training

    Input format: JSONL with {"text": "..."}
    Output: Megatron binary format (.bin + .idx)
    """

    from megatron.tokenizer import build_tokenizer

    # Initialize tokenizer
    args = argparse.Namespace(
        tokenizer_type='GPT2BPETokenizer',
        vocab_file='gpt2-vocab.json',
        merge_file='gpt2-merges.txt',
        rank=0
    )
    tokenizer = build_tokenizer(args)

    # Open output files
    builder = indexed_dataset.MMapIndexedDatasetBuilder(
        f"{output_prefix}.bin",
        dtype=indexed_dataset.DType.optimal
    )

    # Process each document
    with open(input_file) as f:
        for i, line in enumerate(f):
            doc = json.loads(line)
            text = doc['text']

            # Tokenize
            tokens = tokenizer.tokenize(text)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)

            # Add to dataset
            builder.add_item(token_ids)

            if i % 10000 == 0:
                print(f"Processed {i} documents")

    # Finalize
    builder.finalize(f"{output_prefix}.idx")
    print(f"Dataset created: {output_prefix}.bin/.idx")

if __name__ == "__main__":
    preprocess_data(
        input_file="data.jsonl",
        output_prefix="my_dataset",
        tokenizer="gpt2"
    )
```

Or use the built-in tool:

```bash
python tools/preprocess_data.py \
    --input=data.jsonl \
    --output-prefix=my_dataset \
    --tokenizer-type=GPT2BPETokenizer \
    --vocab-file=gpt2-vocab.json \
    --merge-file=gpt2-merges.txt \
    --append-eod \
    --workers=32
```

### Pattern 4: Checkpoint Conversion (Megatron ↔ HuggingFace)

```python
# convert_checkpoint.py
from megatron.checkpointing import load_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

def megatron_to_huggingface(
    megatron_checkpoint_path,
    hf_output_path,
    model_type="gpt",
    tp_size=8,
    pp_size=1
):
    """Convert Megatron checkpoint to HuggingFace format"""

    # Load Megatron checkpoint
    print("Loading Megatron checkpoint...")
    # Note: Actual implementation requires proper model initialization
    # and weight mapping between Megatron and HF formats

    # Create HF model
    print("Creating HuggingFace model...")
    config = {
        "vocab_size": 50257,
        "n_positions": 2048,
        "n_ctx": 2048,
        "n_embd": 4096,
        "n_layer": 32,
        "n_head": 32,
    }

    model = AutoModelForCausalLM.from_config(config)

    # Map weights (simplified - actual mapping is complex)
    # Megatron: self_attention.query_key_value.weight
    # HF: c_attn.weight

    # Save HF checkpoint
    print(f"Saving HuggingFace model to {hf_output_path}")
    model.save_pretrained(hf_output_path)

    print("Conversion complete!")

# Or use built-in tools
# Megatron -> HF
bash tools/checkpoint/convert_megatron_to_hf.sh \
    --megatron-path=/checkpoints/megatron-gpt \
    --hf-path=/checkpoints/hf-gpt \
    --tp-size=8 \
    --pp-size=1

# HF -> Megatron
bash tools/checkpoint/convert_hf_to_megatron.sh \
    --hf-path=/checkpoints/hf-gpt \
    --megatron-path=/checkpoints/megatron-gpt \
    --tp-size=8 \
    --pp-size=1
```

### Pattern 5: Custom Model Architecture

```python
# custom_transformer.py
from megatron.core import parallel_state
from megatron.core.transformer import TransformerConfig, TransformerLayer

class CustomGPTModel:
    """Custom GPT model with Megatron-Core"""

    def __init__(self, config):
        self.config = config

        # Define transformer config
        transformer_config = TransformerConfig(
            num_layers=config.num_layers,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            ffn_hidden_size=config.ffn_hidden_size,
            use_flash_attn=True,
            sequence_parallel=True,
            apply_rope_fusion=True
        )

        # Create transformer layers
        self.layers = [
            TransformerLayer(transformer_config, layer_number=i)
            for i in range(config.num_layers)
        ]

    def forward(self, input_ids, attention_mask):
        """Forward pass"""

        # Embedding
        hidden_states = self.embed_tokens(input_ids)

        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        # Output projection
        logits = self.lm_head(hidden_states)

        return logits
```

### Pattern 6: Inference with Trained Model

```python
# inference_megatron.py
import torch
from megatron import get_args, get_tokenizer
from megatron.initialize import initialize_megatron
from megatron.model import GPTModel
from megatron.checkpointing import load_checkpoint

def generate_text(prompt, model, tokenizer, max_length=100):
    """Generate text using trained Megatron model"""

    # Tokenize prompt
    tokens = tokenizer.tokenize(prompt)
    token_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])

    # Move to GPU
    token_ids = token_ids.cuda()

    # Generate
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            logits = model(token_ids)

            # Get next token
            next_token = torch.argmax(logits[:, -1, :], dim=-1)

            # Append to sequence
            token_ids = torch.cat([token_ids, next_token.unsqueeze(0)], dim=1)

            # Stop if EOS
            if next_token == tokenizer.eod:
                break

    # Decode
    output_tokens = token_ids[0].cpu().tolist()
    output_text = tokenizer.detokenize(output_tokens)

    return output_text

# Usage
if __name__ == "__main__":
    # Initialize Megatron
    initialize_megatron(extra_args_provider=None)

    # Load model
    model = GPTModel(...)
    load_checkpoint(model, None, None)

    # Generate
    prompt = "Once upon a time"
    output = generate_text(prompt, model, tokenizer)
    print(output)
```

## Key Features

- **3D Parallelism**: Combines tensor, pipeline, and data parallelism for maximum efficiency
- **State-of-the-Art Performance**: 41-48% Model FLOPs Utilization on H100 clusters
- **Scalability**: Train models from 2B to 1T+ parameters across 1000+ GPUs
- **Flexible Architectures**: Support for GPT, BERT, T5, LLaMA, Mixtral, Mamba, ViT
- **Advanced Optimizations**: Flash Attention, RoPE, FP8 training, activation checkpointing
- **Production Ready**: Checkpoint conversion, fault tolerance, distributed data loading
- **Multi-Data Center**: Train across geographically distributed clusters
- **Open Source**: Apache 2.0 license with active community

## Performance Optimization

### Best Practices

1. **Choose Optimal Parallelism Strategy**

```python
# For model parallelism selection:

# Rule 1: Tensor Parallelism (TP)
# - Use TP when model doesn't fit in single GPU
# - TP size should divide attention heads evenly
# - Best for: 8-64 GPUs per node

# Rule 2: Pipeline Parallelism (PP)
# - Use PP for very large models
# - Minimize pipeline bubbles with micro-batching
# - Best for: Multi-node training

# Rule 3: Data Parallelism (DP)
# - Use remaining GPUs for DP
# - Maximizes throughput
# - Best for: Large batch sizes

# Example for 70B model on 64 GPUs:
TP = 8   # Split attention across 8 GPUs
PP = 2   # 2 pipeline stages
DP = 4   # 4 data replicas (64 / (8*2))
```

2. **Tune Micro-Batch Size**

```bash
# Micro-batch-size: per-GPU batch size
# Global-batch-size: total batch size
# Gradient accumulation steps = global / (micro * DP * num_microbatches)

# Small models: larger micro-batch
--micro-batch-size=8 \
--global-batch-size=256

# Large models: smaller micro-batch (memory constrained)
--micro-batch-size=1 \
--global-batch-size=128
```

3. **Enable All Optimizations**

```bash
# Recommended flags for H100/H200
--bf16 \                              # BF16 precision
--use-flash-attn \                    # Flash Attention 2
--sequence-parallel \                 # Sequence parallelism
--use-distributed-optimizer \         # Distributed optimizer
--overlap-grad-reduce \               # Communication overlap
--overlap-param-gather \              # Parameter gathering overlap
--untie-embeddings-and-output-weights  # Separate embedding weights
```

4. **Configure NCCL for Network**

```bash
# For NVLink (single node)
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1

# For InfiniBand (multi-node)
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1
export NCCL_IB_QPS_PER_CONN=4
export NCCL_CROSS_NIC=2

# For Ethernet
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
```

5. **Use Activation Checkpointing**

```bash
# Trades compute for memory
# Essential for large models

--recompute-granularity=full \    # or 'selective'
--recompute-method=block \        # or 'uniform'
--recompute-num-layers=1
```

6. **Optimize Data Loading**

```bash
# Use multiple workers
--num-workers=8

# Prefetch batches
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use fast storage (NVMe)
--data-path=/nvme/data/dataset
```

### Expected Performance

| Model Size | GPUs | Config (TP/PP/DP) | Batch Size | MFU | Throughput | Hardware |
|-----------|------|-------------------|------------|-----|------------|----------|
| 7B | 8 | 2/1/4 | 256 | 45% | 8K tok/s | 8x H100 |
| 13B | 16 | 4/1/4 | 512 | 46% | 12K tok/s | 16x H100 |
| 70B | 64 | 8/2/4 | 512 | 47% | 10K tok/s | 64x H100 |
| 175B | 256 | 8/8/4 | 1024 | 48% | 8K tok/s | 256x H100 |
| 1T | 1024 | 8/16/8 | 2048 | 42% | 4K tok/s | 1024x H100 |

**MFU** = Model FLOPs Utilization (actual FLOPs / theoretical peak FLOPs)

**Note**: Performance varies based on sequence length, network topology, and specific optimizations.

### Superlinear Scaling

Megatron exhibits superlinear scaling with model size:
- 7B model: ~41% MFU
- 70B model: ~47% MFU
- 175B model: ~48% MFU

This is due to better arithmetic intensity and reduced communication overhead relative to compute.

## Use Cases

1. **Foundation Model Training**: Train GPT, LLaMA, Mistral-style models from scratch
2. **Continued Pretraining**: Continue training on domain-specific data
3. **Research**: Experiment with novel architectures and training techniques
4. **Vision Transformers**: Train ViT, DINO, and multimodal models
5. **Mixture-of-Experts**: Efficient MoE training with expert parallelism
6. **Multi-Task Learning**: Train T5-style models on multiple tasks
7. **Long Context**: Train models with extended context (32K-128K tokens)
8. **Multi-Data Center**: Distributed training across geographic locations

## Examples

### Example 1: Complete Training Pipeline

```bash
#!/bin/bash
# complete_training_pipeline.sh

set -e

WORK_DIR=/workspace/gpt-training
DATA_DIR=$WORK_DIR/data
CHECKPOINT_DIR=$WORK_DIR/checkpoints
LOG_DIR=$WORK_DIR/logs

mkdir -p $DATA_DIR $CHECKPOINT_DIR $LOG_DIR

echo "=== Megatron-LM Complete Training Pipeline ==="

# Step 1: Download and prepare data
echo "Step 1: Preparing data..."
wget -P $DATA_DIR https://data.together.xyz/redpajama-data-1T/v1.0.0/arxiv/arxiv.jsonl

# Step 2: Preprocess data
echo "Step 2: Preprocessing data..."
python tools/preprocess_data.py \
    --input=$DATA_DIR/arxiv.jsonl \
    --output-prefix=$DATA_DIR/arxiv_text_document \
    --tokenizer-type=GPT2BPETokenizer \
    --vocab-file=gpt2-vocab.json \
    --merge-file=gpt2-merges.txt \
    --append-eod \
    --workers=32

# Step 3: Configure environment
echo "Step 3: Configuring environment..."
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Step 4: Launch training
echo "Step 4: Starting training..."

GPUS_PER_NODE=8
NNODES=4
MASTER_ADDR=node01
MASTER_PORT=6000

torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    pretrain_gpt.py \
    --num-layers=32 \
    --hidden-size=4096 \
    --num-attention-heads=32 \
    --seq-length=2048 \
    --max-position-embeddings=2048 \
    --micro-batch-size=2 \
    --global-batch-size=256 \
    --tensor-model-parallel-size=4 \
    --pipeline-model-parallel-size=2 \
    --sequence-parallel \
    --use-distributed-optimizer \
    --train-iters=100000 \
    --lr=1.5e-4 \
    --min-lr=1.0e-5 \
    --lr-decay-style=cosine \
    --lr-warmup-iters=2000 \
    --weight-decay=0.1 \
    --clip-grad=1.0 \
    --bf16 \
    --use-flash-attn \
    --data-path=$DATA_DIR/arxiv_text_document \
    --split=949,50,1 \
    --tokenizer-type=GPT2BPETokenizer \
    --vocab-file=gpt2-vocab.json \
    --merge-file=gpt2-merges.txt \
    --save=$CHECKPOINT_DIR/gpt-13b \
    --load=$CHECKPOINT_DIR/gpt-13b \
    --save-interval=1000 \
    --eval-interval=100 \
    --eval-iters=10 \
    --log-interval=10 \
    --tensorboard-dir=$LOG_DIR \
    --wandb-project=gpt-13b-arxiv \
    2>&1 | tee $LOG_DIR/training.log

echo "Training complete!"
```

### Example 2: Resume from Checkpoint

```python
# resume_training.py
"""
Resume training from checkpoint with modified hyperparameters
"""

import argparse
from megatron import get_args
from megatron.initialize import initialize_megatron
from megatron.training import pretrain
from megatron.model import GPTModel

def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    model = GPTModel(
        config=get_args(),
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    return model

def extra_args_provider(parser):
    """Add custom arguments"""
    group = parser.add_argument_group('custom', 'Custom arguments')
    group.add_argument('--new-lr', type=float, default=None,
                      help='New learning rate after resume')
    return parser

if __name__ == "__main__":
    # Initialize
    initialize_megatron(extra_args_provider=extra_args_provider)
    args = get_args()

    # Override learning rate if specified
    if args.new_lr is not None:
        args.lr = args.new_lr
        print(f"Using new learning rate: {args.lr}")

    # Resume training
    pretrain(
        train_valid_test_dataset_provider=None,
        model_provider=model_provider,
        model_type=ModelType.encoder_or_decoder,
        forward_step_func=None
    )
```

Run:
```bash
python resume_training.py \
    --load=/checkpoints/gpt-13b \
    --save=/checkpoints/gpt-13b-continued \
    --new-lr=5e-5 \
    --train-iters=200000 \
    [... other args ...]
```

### Example 3: Multi-Data Center Training

```bash
#!/bin/bash
# multi_datacenter_training.sh

# Data Center 1 (Primary)
export MASTER_ADDR=dc1-node01.example.com
export MASTER_PORT=6000
export DATACENTER_ID=dc1
export NCCL_CROSS_DC=1

# Configure inter-DC networking
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1
export NCCL_IB_TC=106
export NCCL_IB_QPS_PER_CONN=4

# Data Center 2 (Secondary)
# Run with same MASTER_ADDR, different node rank

torchrun \
    --nproc_per_node=8 \
    --nnodes=16 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    pretrain_gpt.py \
    --tensor-model-parallel-size=8 \
    --pipeline-model-parallel-size=4 \
    --data-parallel-size=32 \
    --micro-batch-size=1 \
    --global-batch-size=512 \
    --datacenter-id=$DATACENTER_ID \
    [... model config ...]
```

### Example 4: Custom Dataset with Packing

```python
# custom_dataset_with_packing.py
"""
Custom dataset implementation with sequence packing
"""

import numpy as np
import torch
from megatron.core.datasets.gpt_dataset import GPTDataset

class PackedGPTDataset(GPTDataset):
    """GPT dataset with sequence packing for efficiency"""

    def __init__(self, name, data_prefix, documents, indexed_dataset,
                 num_samples, seq_length, seed):
        super().__init__(
            name, data_prefix, documents, indexed_dataset,
            num_samples, seq_length, seed
        )
        self.seq_length = seq_length

    def __getitem__(self, idx):
        """Pack multiple documents into single sequence"""

        tokens = []
        total_length = 0

        # Keep adding documents until we reach seq_length
        while total_length < self.seq_length:
            doc_idx = self._get_document_index(idx)
            doc_tokens = self._get_document_tokens(doc_idx)

            remaining = self.seq_length - total_length
            tokens.extend(doc_tokens[:remaining])
            total_length += len(doc_tokens[:remaining])

            if total_length >= self.seq_length:
                break

            idx += 1

        # Pad if necessary
        if len(tokens) < self.seq_length:
            tokens.extend([self.pad_id] * (self.seq_length - len(tokens)))

        # Convert to tensor
        tokens = torch.tensor(tokens, dtype=torch.long)

        # Create labels (shifted by 1)
        labels = tokens[1:].clone()
        labels = torch.cat([labels, torch.tensor([self.pad_id])])

        return {
            'text': tokens,
            'labels': labels
        }

# Usage
def train_valid_test_dataset_provider(train_val_test_num_samples):
    """Build train, validation, and test datasets."""

    train_ds = PackedGPTDataset(
        name='train',
        data_prefix='/data/my_dataset_text_document',
        documents=train_documents,
        indexed_dataset=indexed_ds,
        num_samples=train_val_test_num_samples[0],
        seq_length=args.seq_length,
        seed=args.seed
    )

    return train_ds, valid_ds, test_ds
```

### Example 5: Monitoring and Profiling

```python
# monitor_training.py
"""
Monitor training metrics and profile performance
"""

import torch
import time
from torch.profiler import profile, ProfilerActivity

class TrainingMonitor:
    def __init__(self, log_interval=10):
        self.log_interval = log_interval
        self.step = 0
        self.start_time = time.time()

    def log_metrics(self, loss, lr, grad_norm):
        """Log training metrics"""

        self.step += 1

        if self.step % self.log_interval == 0:
            elapsed = time.time() - self.start_time
            tokens_per_sec = (self.step * global_batch_size * seq_length) / elapsed

            print(f"Step {self.step}:")
            print(f"  Loss: {loss:.4f}")
            print(f"  LR: {lr:.2e}")
            print(f"  Grad Norm: {grad_norm:.4f}")
            print(f"  Tokens/sec: {tokens_per_sec:.0f}")

            # Log to tensorboard
            if writer:
                writer.add_scalar('loss', loss, self.step)
                writer.add_scalar('lr', lr, self.step)
                writer.add_scalar('throughput', tokens_per_sec, self.step)

    def profile_step(self, model, inputs):
        """Profile a training step"""

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True
        ) as prof:

            # Forward pass
            outputs = model(**inputs)
            loss = outputs.loss

            # Backward pass
            loss.backward()

        # Print profiling results
        print(prof.key_averages().table(sort_by="cuda_time_total"))

        # Export trace
        prof.export_chrome_trace("trace.json")

# Usage in training loop
monitor = TrainingMonitor(log_interval=10)

for step, batch in enumerate(train_dataloader):
    # Training step
    loss = train_step(model, batch)

    # Log metrics
    monitor.log_metrics(
        loss=loss.item(),
        lr=scheduler.get_last_lr()[0],
        grad_norm=grad_norm
    )

    # Profile every 100 steps
    if step % 100 == 0:
        monitor.profile_step(model, batch)
```

### Example 6: Fault Tolerance and Checkpointing

```python
# fault_tolerant_training.py
"""
Implement fault-tolerant training with automatic checkpoint recovery
"""

import os
import torch
from megatron.checkpointing import save_checkpoint, load_checkpoint

class FaultTolerantTrainer:
    def __init__(self, model, optimizer, checkpoint_dir):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir
        self.iteration = 0

    def save_checkpoint_if_needed(self, iteration, save_interval=1000):
        """Save checkpoint periodically"""

        if iteration % save_interval == 0:
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"iter_{iteration:07d}"
            )

            print(f"Saving checkpoint to {checkpoint_path}")

            save_checkpoint(
                iteration=iteration,
                model=self.model,
                optimizer=self.optimizer,
                opt_param_scheduler=None
            )

    def recover_from_checkpoint(self):
        """Recover from latest checkpoint"""

        # Find latest checkpoint
        checkpoints = sorted([
            d for d in os.listdir(self.checkpoint_dir)
            if d.startswith('iter_')
        ])

        if not checkpoints:
            print("No checkpoints found, starting from scratch")
            return 0

        latest_checkpoint = checkpoints[-1]
        checkpoint_path = os.path.join(self.checkpoint_dir, latest_checkpoint)

        print(f"Recovering from checkpoint: {checkpoint_path}")

        # Load checkpoint
        iteration = load_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            opt_param_scheduler=None
        )

        print(f"Resumed from iteration {iteration}")
        return iteration

    def train_with_fault_tolerance(self, train_dataloader, num_iterations):
        """Training loop with automatic recovery"""

        # Try to recover from checkpoint
        start_iteration = self.recover_from_checkpoint()

        try:
            for iteration in range(start_iteration, num_iterations):
                # Training step
                batch = next(train_dataloader)
                loss = self.train_step(batch)

                # Save checkpoint periodically
                self.save_checkpoint_if_needed(iteration)

                # Health check
                if torch.isnan(loss) or torch.isinf(loss):
                    raise RuntimeError(f"NaN/Inf loss at iteration {iteration}")

        except Exception as e:
            print(f"Training interrupted: {e}")
            print("Saving emergency checkpoint...")
            self.save_checkpoint_if_needed(iteration, save_interval=1)
            raise

        print("Training completed successfully!")

    def train_step(self, batch):
        """Single training step"""
        # Implementation here
        pass
```

## Troubleshooting

### Issue 1: Out of Memory (OOM)

**Problem**: `CUDA out of memory` during training.

**Solutions**:

```bash
# 1. Reduce micro-batch size
--micro-batch-size=1  # Instead of 4

# 2. Enable activation checkpointing
--recompute-granularity=full \
--recompute-method=block

# 3. Increase tensor parallelism
--tensor-model-parallel-size=8  # Instead of 4

# 4. Use gradient checkpointing
--checkpoint-activations \
--checkpoint-num-layers=1

# 5. Reduce sequence length
--seq-length=1024  # Instead of 2048

# 6. Use FP16 instead of BF16 (if compatible)
--fp16  # Uses less memory than BF16

# 7. Enable CPU offloading (DeepSpeed)
--zero-stage=3 \
--cpu-offload
```

### Issue 2: Slow Training / Low GPU Utilization

**Problem**: GPU utilization < 80%, training slower than expected.

**Solutions**:

```bash
# 1. Check if data loading is bottleneck
nvidia-smi dmon -s u -c 100
# If GPU util drops periodically, increase workers:
--num-workers=8

# 2. Increase micro-batch size
--micro-batch-size=4  # Larger batches

# 3. Reduce pipeline bubbles
# Increase number of micro-batches per pipeline stage
export MEGATRON_NUM_MICROBATCHES=8

# 4. Enable communication overlap
--overlap-grad-reduce \
--overlap-param-gather

# 5. Use faster storage
# Move data to NVMe/local SSD
--data-path=/nvme/data/dataset

# 6. Profile the code
python -m torch.utils.bottleneck pretrain_gpt.py [args]

# 7. Check NCCL performance
export NCCL_DEBUG=INFO
# Look for "Using NVLink" or "Using InfiniBand"
```

### Issue 3: Training Diverges / Loss Becomes NaN

**Problem**: Loss suddenly becomes NaN or increases unexpectedly.

**Solutions**:

```bash
# 1. Reduce learning rate
--lr=1.0e-4  # Instead of 1.5e-4

# 2. Increase warmup steps
--lr-warmup-iters=5000  # Instead of 2000

# 3. Reduce gradient clipping threshold
--clip-grad=0.5  # Instead of 1.0

# 4. Use BF16 instead of FP16 (more stable)
--bf16

# 5. Check for bad data
python tools/verify_dataset.py --data-path=...

# 6. Enable gradient accumulation fusion
--use-distributed-optimizer

# 7. Reduce batch size
--global-batch-size=64  # Smaller batches

# 8. Add gradient checkpointing for stability
--checkpoint-activations
```

### Issue 4: NCCL Timeout / Hangs

**Problem**: Training hangs with NCCL timeout errors.

**Solutions**:

```bash
# 1. Increase NCCL timeout
export NCCL_TIMEOUT=7200000  # 2 hours in ms

# 2. Check network connectivity
ping -c 3 <other-node>

# 3. Verify InfiniBand
ibstat
rdma link show

# 4. Enable NCCL debugging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# 5. Check firewall rules
sudo ufw status
# Allow ports 6000-7000 for distributed training

# 6. Use correct network interface
export NCCL_SOCKET_IFNAME=eth0  # or ib0

# 7. Test NCCL directly
nccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 8

# 8. Reduce parallelism temporarily to isolate issue
--tensor-model-parallel-size=1 \
--pipeline-model-parallel-size=1
```

### Issue 5: Checkpoint Loading Fails

**Problem**: Cannot load checkpoint, mismatched tensor shapes.

**Solutions**:

```python
# 1. Check parallelism settings match
# Load args must match save args:
# --tensor-model-parallel-size=8 (same as save)
# --pipeline-model-parallel-size=2 (same as save)

# 2. Use --no-load-optim to skip optimizer state
--no-load-optim \
--no-load-rng

# 3. Convert checkpoint to different parallelism
python tools/checkpoint/util.py \
    --model-type GPT \
    --load-dir=/checkpoints/tp8-pp2 \
    --save-dir=/checkpoints/tp4-pp4 \
    --target-tp=4 \
    --target-pp=4

# 4. Inspect checkpoint
python tools/checkpoint/inspect_checkpoint.py \
    --checkpoint-dir=/checkpoints/iter_0001000

# 5. Use checkpoint conversion tool
bash tools/checkpoint/convert_checkpoint.sh \
    --input=/checkpoints/old \
    --output=/checkpoints/new \
    --target-tp=8
```

### Issue 6: Unbalanced Pipeline Stages

**Problem**: Some GPUs heavily utilized, others idle (pipeline parallelism).

**Solutions**:

```bash
# 1. Profile pipeline stages
python tools/profile_pipeline.py \
    --model-config=[config] \
    --pp-size=4

# 2. Adjust layer distribution
# Manually specify layers per stage
--pipeline-model-parallel-split-rank=16,32,48

# 3. Increase micro-batches
export MEGATRON_NUM_MICROBATCHES=16

# 4. Use virtual pipeline parallelism
--virtual-pipeline-model-parallel-size=2

# 5. Balance by profiling
# Redistribute layers based on compute time

# 6. Monitor per-GPU utilization
nvidia-smi dmon -s u -c 100
```

### Issue 7: Slow Convergence

**Problem**: Model converges slowly compared to expected learning curve.

**Solutions**:

```bash
# 1. Increase learning rate
--lr=3.0e-4  # Try 2x

# 2. Adjust batch size
--global-batch-size=512  # Larger batches

# 3. Change LR schedule
--lr-decay-style=cosine  # or 'polynomial'
--lr-decay-iters=100000

# 4. Verify data quality
# Check for duplicates, formatting issues

# 5. Add learning rate warmup
--lr-warmup-iters=2000 \
--lr-warmup-init=1.0e-7

# 6. Tune weight decay
--weight-decay=0.01  # Lower value

# 7. Check gradient norms
# Add logging to monitor gradient flow

# 8. Verify tokenization
python tools/verify_tokenization.py --data-path=...
```

## Advanced Topics

### FP8 Training on Hopper/Blackwell

```bash
# Enable FP8 with Transformer Engine
pip install git+https://github.com/NVIDIA/TransformerEngine.git

# Training args
--fp8-format=hybrid \
--fp8-amax-history-len=1024 \
--fp8-amax-compute-algo=max \
--transformer-impl=transformer_engine

# Environment
export NVTE_FP8_DPA_BWD=1
export NVTE_FLASH_ATTN=1
```

### Mixture-of-Experts (MoE)

```bash
# MoE configuration
--num-experts=8 \
--expert-model-parallel-size=4 \
--moe-router-topk=2 \
--moe-router-load-balancing-type=aux_loss \
--moe-aux-loss-coeff=0.01 \
--moe-token-dispatcher-type=alltoall

# Expert parallelism
# EP should divide num_experts evenly
# 8 experts / 4 EP = 2 experts per GPU
```

### Long Context Training

```bash
# Extended context (up to 32K)
--seq-length=32768 \
--max-position-embeddings=32768 \
--position-embedding-type=rope \
--rope-scaling-factor=1.0 \
--use-rotary-position-embeddings

# YaRN RoPE scaling for longer contexts
--rope-scaling-type=yarn \
--rope-scaling-factor=4.0 \
--yarn-alpha=1.0
```

### Multi-Modal Training

```python
# Vision-Language Model
from megatron.model.vision.clip_vit_model import CLIPViTModel
from megatron.model.gpt_model import GPTModel

class VisionLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vision_encoder = CLIPViTModel(config)
        self.language_model = GPTModel(config)
        self.projection = nn.Linear(768, 4096)

    def forward(self, images, text):
        # Encode images
        image_features = self.vision_encoder(images)
        image_embeds = self.projection(image_features)

        # Combine with text
        text_embeds = self.language_model.embed(text)
        combined = torch.cat([image_embeds, text_embeds], dim=1)

        # Generate
        output = self.language_model(combined)
        return output
```

### Custom Learning Rate Schedulers

```python
# custom_lr_scheduler.py
from megatron.training import get_optimizer_param_scheduler

class CustomLRScheduler:
    def __init__(self, optimizer, args):
        self.optimizer = optimizer
        self.args = args
        self.step_count = 0

    def step(self):
        """Update learning rate"""
        self.step_count += 1

        # Custom schedule logic
        if self.step_count < self.args.warmup_steps:
            # Linear warmup
            lr = self.args.lr * (self.step_count / self.args.warmup_steps)
        else:
            # Cosine decay with restarts
            progress = (self.step_count - self.args.warmup_steps)
            total_steps = self.args.train_iters - self.args.warmup_steps
            cycles = progress // (total_steps // self.args.num_restarts)
            cycle_progress = progress % (total_steps // self.args.num_restarts)

            lr = self.args.min_lr + (self.args.lr - self.args.min_lr) * \
                 0.5 * (1 + math.cos(math.pi * cycle_progress / (total_steps // self.args.num_restarts)))

        # Apply to all param groups
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr
```

## Resources

- **Repository**: https://github.com/NVIDIA/Megatron-LM
- **Megatron-Core Docs**: https://docs.nvidia.com/megatron-core/
- **Papers**:
  - Megatron-LM: https://arxiv.org/abs/1909.08053
  - Efficient Large-Scale LM Training: https://arxiv.org/abs/2104.04473
  - Reducing Activation Recomputation: https://arxiv.org/abs/2205.05198
- **NeMo Framework**: https://github.com/NVIDIA/NeMo
- **NGC Catalog**: https://catalog.ngc.nvidia.com/
- **Developer Forums**: https://forums.developer.nvidia.com/
- **Issue Tracker**: https://github.com/NVIDIA/Megatron-LM/issues

## Notes

### Platform Support
- **Linux**: Full support (x86_64, aarch64)
- **Windows**: Not supported
- **Cloud**: AWS, Azure, GCP with GPU instances

### GPU Requirements
- **Minimum**: 4x NVIDIA V100 32GB
- **Recommended**: 8+ NVIDIA A100 80GB or H100 80GB
- **Optimal**: 64-1024x H100/H200 with NVLink/InfiniBand

### Performance Characteristics
- **MFU**: 41-48% on H100 clusters
- **Scaling**: Near-linear up to 1024 GPUs
- **Throughput**: 4K-12K tokens/second (depends on model size)
- **Memory Efficiency**: 3D parallelism enables training of 1T+ param models

### Production Readiness
- Battle-tested at NVIDIA and research institutions
- Used for training GPT, BERT, T5, LLaMA models
- Active development with monthly releases
- Enterprise support available through NVIDIA

### Known Limitations
- Steep learning curve for parallelism configuration
- Requires significant GPU resources for large models
- Checkpoint format not directly compatible with HuggingFace (conversion needed)
- Limited Windows support

### Version Compatibility
- Major version changes may break checkpoint compatibility
- Always check release notes before upgrading
- Test new versions on small models first
- Keep checkpoint conversion tools updated

## Related Technologies

- **PyTorch**: Deep learning framework foundation
- **NCCL**: Multi-GPU communication library
- **Apex**: Mixed precision training utilities
- **Transformer Engine**: FP8 training on Hopper/Blackwell
- **Flash Attention**: Memory-efficient attention implementation
- **DeepSpeed**: Alternative distributed training framework
- **NeMo**: Production deployment framework
- **TensorRT-LLM**: Optimized inference engine
- **Hugging Face Transformers**: Model hub and inference
- **Weights & Biases**: Experiment tracking and visualization
