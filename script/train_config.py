#!/usr/bin/env python3
"""
Corrected configuration for Megatron-LM training scripts.
Based on actual quantization control mechanism in the source code.

IMPORTANT: The --linear-quantization and --attention-quantization parameters
do not exist in Megatron-LM. Quantization is controlled by hardcoded values
in the source code.
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Model configurations
MODELS = {
    "llama31-8b": {
        "tp_size": 2, "cp_size": 1, "pp_size": 4,
        "num_layers": 32, "hidden_size": 4096, "ffn_hidden_size": 14336,
        "num_attention_heads": 32, "num_query_groups": 8, "kv_channels": 128,
        "rotary_base": 1000000, "vocab_size": 128256, "tokenizer_path": "model/llama3"
    },
    "llama32-1b": {
        "tp_size": 4, "cp_size": 1, "pp_size": 1,
        "num_layers": 16, "hidden_size": 2048, "ffn_hidden_size": 8192,
        "num_attention_heads": 32, "num_query_groups": 8, "kv_channels": 128,
        "rotary_base": 500000, "vocab_size": 128256, "tokenizer_path": "model/llama3.2-1b"
    },
    "deepseek2_lite": {
        "tp_size": 2, "cp_size": 1, "pp_size": 2,
        "num_layers": 28, "hidden_size": 3584, "ffn_hidden_size": 12800,
        "num_attention_heads": 28, "num_query_groups": 4, "kv_channels": 128,
        "rotary_base": 1000000, "vocab_size": 128256, "tokenizer_path": "model/deepseek2"
    }
}

# Dataset configurations
DATASETS = {
    "wikipedia": {
        "data_path": "dataset/wikipedia_processed/wikipedia_processed_text_document",
        "tokenizer_type": "HuggingFaceTokenizer"
    },
    "dolma": {
        "data_path": "dataset/dolma_processed/dolma_processed_text_document",
        "tokenizer_type": "HuggingFaceTokenizer"
    }
}

# Quantization configurations
# NOTE: These are based on the actual hardcoded values in the source code
QUANTIZATION = {
    "bf16": {
        "dtype": "bf16", 
        "fp8_format": None, 
        "description": "BF16 precision, no quantization (requires source code modification)"
    },
    "mxfp4": {
        "dtype": "fp8", 
        "fp8_format": "mxfp4", 
        "description": "MXFP4 quantization (requires source code modification)"
    },
    "mxfp8": {
        "dtype": "fp8", 
        "fp8_format": "mxfp8", 
        "description": "MXFP8 quantization (requires source code modification)"
    },
    "hifp8": {
        "dtype": "fp8", 
        "fp8_format": "hifp8", 
        "description": "HIFP8 quantization (current hardcoded default)"
    }
}

# Training configurations
TRAINING = {
    "standard": {
        "micro_batch_size": 1, "global_batch_size": 128, "seq_length": 8192,
        "train_samples": 47340000, "lr": 0.00015, "min_lr": 0.00001, "exit_duration_mins": 235000000
    },
    "fast": {
        "micro_batch_size": 1, "global_batch_size": 32, "seq_length": 2048,
        "train_samples": 1000, "lr": 0.0001, "min_lr": 0.00001, "exit_duration_mins": 60
    }
}

def build_command(model, dataset, quantization, training_config="standard", dry_run=False):
    """Build training command."""
    
    # Get configurations
    model_config = MODELS[model]
    dataset_config = DATASETS[dataset]
    quant_config = QUANTIZATION[quantization]
    train_config = TRAINING[training_config]
    
    # Build paths
    checkpoint_path = f"checkpoints/{model}_{dataset}_{quantization}"
    tensorboard_path = f"tensorboard_logs/{model}_{dataset}_{quantization}"
    
    # Build command
    cmd = [
        "torchrun",
        "--nproc_per_node", "8",
        "--nnodes", "1",
        "--node_rank", "0",
        "--master_addr", "localhost",
        "--master_port", "6000",
        "pretrain_gpt.py",
        
        # Model arguments
        "--use-mcore-models",
        "--num-layers", str(model_config["num_layers"]),
        "--hidden-size", str(model_config["hidden_size"]),
        "--ffn-hidden-size", str(model_config["ffn_hidden_size"]),
        "--num-attention-heads", str(model_config["num_attention_heads"]),
        "--group-query-attention",
        "--num-query-groups", str(model_config["num_query_groups"]),
        "--kv-channels", str(model_config["kv_channels"]),
        "--seq-length", str(train_config["seq_length"]),
        "--max-position-embeddings", str(train_config["seq_length"]),
        "--position-embedding-type", "rope",
        "--rotary-base", str(model_config["rotary_base"]),
        "--rotary-percent", "1.0",
        "--attention-dropout", "0.0",
        "--hidden-dropout", "0.0",
        "--swiglu",
        "--init-method-std", "0.0134",
        "--attention-backend", "fused",
        "--apply-layernorm-1p",
        "--untie-embeddings-and-output-weights",
        "--disable-bias-linear",
        
        # Training arguments
        "--micro-batch-size", str(train_config["micro_batch_size"]),
        "--global-batch-size", str(train_config["global_batch_size"]),
        "--train-samples", str(train_config["train_samples"]),
        "--lr", str(train_config["lr"]),
        "--min-lr", str(train_config["min_lr"]),
        "--lr-decay-style", "cosine",
        "--clip-grad", "1.0",
        "--weight-decay", "0.1",
        "--adam-beta1", "0.9",
        "--adam-beta2", "0.95",
        "--bf16",
        "--grad-reduce-in-bf16",
        "--cross-entropy-loss-fusion",
        "--calculate-per-token-loss",
        "--manual-gc",
        "--empty-unused-memory-level", "1",
        "--exit-duration-in-mins", str(train_config["exit_duration_mins"]),
        "--use-distributed-optimizer",
        "--overlap-grad-reduce",
        "--overlap-param-gather",
        
        # Model parallelism
        "--tensor-model-parallel-size", str(model_config["tp_size"]),
        "--context-parallel-size", str(model_config["cp_size"]),
        "--pipeline-model-parallel-size", str(model_config["pp_size"]),
        "--sequence-parallel",
        
        # Data arguments
        "--data-path", dataset_config["data_path"],
        "--tokenizer-type", dataset_config["tokenizer_type"],
        "--tokenizer-model", model_config["tokenizer_path"],
        "--vocab-size", str(model_config["vocab_size"]),
        "--split", "99,1,0",
        "--no-create-attention-mask-in-dataloader",
        "--num-workers", "1",
        
        # Logging and checkpointing
        "--log-interval", "1",
        "--eval-iters", "32",
        "--eval-interval", "100",
        "--save-interval", "1000",
        "--log-throughput",
        "--ckpt-format", "torch_dist",
        "--distributed-timeout-minutes", "60",
        "--save", checkpoint_path,
        "--load", checkpoint_path,
        "--tensorboard-dir", tensorboard_path
    ]
    
    # Add quantization arguments
    if quant_config["dtype"] == "fp8":
        cmd.extend([
            "--fp8-format", quant_config["fp8_format"],
            "--fp8-amax-history-len", "1024",
            "--fp8-amax-compute-algo", "max"
        ])
    
    return cmd

def main():
    parser = argparse.ArgumentParser(description="Corrected Megatron-LM training script")
    parser.add_argument("--model", choices=list(MODELS.keys()), help="Model name")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), help="Dataset name")
    parser.add_argument("--quantization", choices=list(QUANTIZATION.keys()), help="Quantization type")
    parser.add_argument("--training-config", default="standard", choices=list(TRAINING.keys()), help="Training configuration")
    parser.add_argument("--dry-run", action="store_true", help="Show command without executing")
    parser.add_argument("--list", action="store_true", help="List available configurations")
    
    args = parser.parse_args()
    
    if args.list:
        print("Available Models:", ", ".join(MODELS.keys()))
        print("Available Datasets:", ", ".join(DATASETS.keys()))
        print("Available Quantization:")
        for quant, config in QUANTIZATION.items():
            print(f"  - {quant}: {config['description']}")
        print("Available Training Configs:", ", ".join(TRAINING.keys()))
        print("\nIMPORTANT NOTE:")
        print("Quantization is controlled by hardcoded values in the source code.")
        print("To change quantization type, you need to modify:")
        print("  - megatron/core/tensor_parallel/layers.py (line 783)")
        print("  - megatron/core/transformer/dot_product_attention.py (line 166)")
        return 0
    
    # Check required arguments for training
    if not args.model or not args.dataset or not args.quantization:
        print("Error: --model, --dataset, and --quantization are required for training")
        return 1
    
    try:
        cmd = build_command(args.model, args.dataset, args.quantization, args.training_config, args.dry_run)
        
        if args.dry_run:
            print("Training command:")
            print(" ".join(cmd))
            print(f"\nNOTE: Quantization type '{args.quantization}' requires source code modification.")
            print("Current hardcoded values:")
            print("  - Linear layers: hifp8 (in layers.py line 783)")
            print("  - Attention layers: hifp8 (in dot_product_attention.py line 166)")
        else:
            # Change to Megatron-LM root directory
            script_dir = Path(__file__).parent
            megatron_root = script_dir.parent
            subprocess.run(cmd, cwd=megatron_root, check=True)
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
