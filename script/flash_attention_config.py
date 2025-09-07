#!/usr/bin/env python3
"""
Corrected configuration for Flash Attention QK and PV quantization.
Based on actual quantization control mechanism in the source code.

IMPORTANT: The --attention-quantization parameter does not exist in Megatron-LM.
QK and PV quantization are controlled by hardcoded values in the source code.
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

# Flash Attention quantization configurations
# NOTE: These are based on the actual hardcoded values in the source code
FA_QUANTIZATION = {
    "bf16": {
        "dtype": "bf16", 
        "fp8_format": None, 
        "description": "BF16 precision, no quantization (requires source code modification)"
    },
    "qk_mxfp4": {
        "dtype": "fp8", 
        "fp8_format": "mxfp4", 
        "description": "QK computation with MXFP4 quantization (requires source code modification)"
    },
    "qk_mxfp8": {
        "dtype": "fp8", 
        "fp8_format": "mxfp8", 
        "description": "QK computation with MXFP8 quantization (requires source code modification)"
    },
    "qk_hifp8": {
        "dtype": "fp8", 
        "fp8_format": "hifp8", 
        "description": "QK computation with HIFP8 quantization (requires source code modification)"
    },
    "pv_mxfp4": {
        "dtype": "fp8", 
        "fp8_format": "mxfp4", 
        "description": "PV computation with MXFP4 quantization (requires source code modification)"
    },
    "pv_mxfp8": {
        "dtype": "fp8", 
        "fp8_format": "mxfp8", 
        "description": "PV computation with MXFP8 quantization (requires source code modification)"
    },
    "pv_hifp8": {
        "dtype": "fp8", 
        "fp8_format": "hifp8", 
        "description": "PV computation with HIFP8 quantization (requires source code modification)"
    },
    "qk_pv_mxfp4": {
        "dtype": "fp8", 
        "fp8_format": "mxfp4", 
        "description": "Both QK and PV computation with MXFP4 quantization (requires source code modification)"
    },
    "qk_pv_mxfp8": {
        "dtype": "fp8", 
        "fp8_format": "mxfp8", 
        "description": "Both QK and PV computation with MXFP8 quantization (requires source code modification)"
    },
    "qk_pv_hifp8": {
        "dtype": "fp8", 
        "fp8_format": "hifp8", 
        "description": "Both QK and PV computation with HIFP8 quantization (current hardcoded default)"
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

def build_command(model, dataset, fa_quantization, training_config="standard", dry_run=False):
    """Build training command."""
    
    # Get configurations
    model_config = MODELS[model]
    dataset_config = DATASETS[dataset]
    quant_config = FA_QUANTIZATION[fa_quantization]
    train_config = TRAINING[training_config]
    
    # Build paths
    checkpoint_path = f"checkpoints/{model}_{dataset}_{fa_quantization}"
    tensorboard_path = f"tensorboard_logs/{model}_{dataset}_{fa_quantization}"
    
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
    parser = argparse.ArgumentParser(description="Corrected Flash Attention quantization training script")
    parser.add_argument("--model", choices=list(MODELS.keys()), help="Model name")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), help="Dataset name")
    parser.add_argument("--fa-quantization", choices=list(FA_QUANTIZATION.keys()), help="Flash Attention quantization type")
    parser.add_argument("--training-config", default="standard", choices=list(TRAINING.keys()), help="Training configuration")
    parser.add_argument("--dry-run", action="store_true", help="Show command without executing")
    parser.add_argument("--list", action="store_true", help="List available configurations")
    
    args = parser.parse_args()
    
    if args.list:
        print("Available Models:", ", ".join(MODELS.keys()))
        print("Available Datasets:", ", ".join(DATASETS.keys()))
        print("Available Flash Attention Quantization:")
        for quant, config in FA_QUANTIZATION.items():
            print(f"  - {quant}: {config['description']}")
        print("Available Training Configs:", ", ".join(TRAINING.keys()))
        print("\nIMPORTANT NOTE:")
        print("Flash Attention quantization is controlled by hardcoded values in the source code.")
        print("To change quantization type, you need to modify:")
        print("  - megatron/core/transformer/dot_product_attention.py (line 166 for QK, line 238 for PV)")
        print("\nCurrent hardcoded values:")
        print("  - QK computation (baddbmm): hifp8")
        print("  - PV computation (bmm): hifp8")
        return 0
    
    # Check required arguments for training
    if not args.model or not args.dataset or not args.fa_quantization:
        print("Error: --model, --dataset, and --fa-quantization are required for training")
        return 1
    
    try:
        cmd = build_command(args.model, args.dataset, args.fa_quantization, args.training_config, args.dry_run)
        
        if args.dry_run:
            print("Training command:")
            print(" ".join(cmd))
            print(f"\nNOTE: Flash Attention quantization type '{args.fa_quantization}' requires source code modification.")
            print("Current hardcoded values:")
            print("  - QK computation (baddbmm): hifp8 (line 166)")
            print("  - PV computation (bmm): hifp8 (line 238)")
            print("\nTo implement independent QK and PV quantization control:")
            print("1. Modify line 166 in dot_product_attention.py for QK quantization")
            print("2. Modify line 238 in dot_product_attention.py for PV quantization")
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
