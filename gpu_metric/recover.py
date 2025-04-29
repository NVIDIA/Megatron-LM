import os
import sys
import json
import time
import subprocess
import logging
import threading
import socket
import torch
import argparse
import signal
import re
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("megatron_recovery.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("megatron_recovery")

class MegatronConfig:
    """Represents Megatron training configuration"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize with either a config file or default values
        
        Args:
            config_file: Path to Megatron configuration file (JSON)
        """
        # Default configuration
        self.config = {
            "tensor_model_parallel_size": 2,
            "pipeline_model_parallel_size": 2,
            "data_parallel_size": 2,
            "world_size": 8,  # Total number of GPUs
            "checkpoint_dir": "./checkpoints",
            "megatron_path": "./Megatron-LM",
            "model_type": "gpt",
            "hidden_size": 2048,
            "num_layers": 24,
            "num_attention_heads": 16,
            "micro_batch_size": 4,
            "global_batch_size": 32,
            "max_position_embeddings": 2048,
            "optimizer": "Adam",
            "learning_rate": 1e-4,
            "min_lr": 1e-5,
            "lr_decay_style": "cosine",
            "lr_warmup_fraction": 0.01,
            "weight_decay": 0.1,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "clip_grad": 1.0,
            "fp16": True,
            "initial_loss_scale": 4294967296,
            "train_iters": 500000,
            "exit_interval": 10000,
            "distributed_backend": "nccl",
            "seed": 42
        }
        
        # Load from file if provided
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                # Update with loaded values
                self.config.update(loaded_config)
                logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
        
        # Calculate derived values
        self._calculate_derived_values()
    
    def _calculate_derived_values(self):
        """Calculate derived configuration values"""
        # Total parallelism degree
        self.config["total_parallelism"] = (
            self.config["tensor_model_parallel_size"] * 
            self.config["pipeline_model_parallel_size"] * 
            self.config["data_parallel_size"]
        )
        
        # Ensure world_size matches parallelism
        if self.config["world_size"] != self.config["total_parallelism"]:
            logger.warning(
                f"World size ({self.config['world_size']}) doesn't match total parallelism "
                f"({self.config['total_parallelism']}). Updating world_size."
            )
            self.config["world_size"] = self.config["total_parallelism"]
    
    def reconfigure_for_gpu_count(self, available_gpus: int) -> bool:
        """
        Reconfigure parallelism settings for the available GPU count
        
        Args:
            available_gpus: Number of available GPUs
            
        Returns:
            bool: True if reconfiguration was successful, False otherwise
        """
        original_config = self.config.copy()
        
        # Can't run with fewer than minimum required GPUs
        if available_gpus < 1:
            logger.error("Cannot reconfigure for 0 GPUs")
            return False
        
        # Try to maintain as much parallelism as possible
        if self._find_parallelism_config(available_gpus):
            logger.info(f"Reconfigured for {available_gpus} GPUs: "
                      f"TP={self.config['tensor_model_parallel_size']}, "
                      f"PP={self.config['pipeline_model_parallel_size']}, "
                      f"DP={self.config['data_parallel_size']}")
            return True
        else:
            # Restore original config if reconfiguration failed
            self.config = original_config
            logger.error(f"Failed to reconfigure for {available_gpus} GPUs")
            return False
    
    def _find_parallelism_config(self, available_gpus: int) -> bool:
        """
        Find a valid parallelism configuration for the given GPU count
        
        Args:
            available_gpus: Number of available GPUs
            
        Returns:
            bool: True if a valid configuration was found
        """
        # Try to preserve tensor parallelism first, then pipeline
        tp_size = self.config["tensor_model_parallel_size"]
        pp_size = self.config["pipeline_model_parallel_size"]
        
        # Special case: if we have too few GPUs for current TP + PP
        if available_gpus < tp_size * pp_size:
            # For very small GPU counts, try minimal viable configurations
            viable_configs = []
            
            # Try configs where TP and PP are powers of 2
            for tp in [1, 2, 4, 8]:
                for pp in [1, 2, 4, 8]:
                    if tp * pp <= available_gpus and available_gpus % (tp * pp) == 0:
                        dp = available_gpus // (tp * pp)
                        viable_configs.append((tp, pp, dp))
            
            # Sort by preference: 
            # 1. Maximize TP (better performance for large models)
            # 2. Then maximize PP
            # 3. Then maximize DP
            if viable_configs:
                viable_configs.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
                tp_size, pp_size, dp_size = viable_configs[0]
            else:
                # If no clean configuration found, fall back to minimal
                tp_size = min(tp_size, available_gpus)
                pp_size = 1
                dp_size = available_gpus // tp_size
        else:
            # We have enough GPUs for current TP and PP
            # Just adjust DP to fill available GPUs
            dp_size = available_gpus // (tp_size * pp_size)
        
        # Update configuration with new parallelism settings
        self.config["tensor_model_parallel_size"] = tp_size
        self.config["pipeline_model_parallel_size"] = pp_size
        self.config["data_parallel_size"] = dp_size
        self.config["world_size"] = available_gpus
        
        # Update derived values
        self._calculate_derived_values()
        
        return True
    
    def save_config(self, filepath: str):
        """Save current configuration to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Saved configuration to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def get_launch_command(self, script_path: str, checkpoint_path: Optional[str] = None) -> List[str]:
        """
        Generate the command to launch Megatron training
        
        Args:
            script_path: Path to the main Megatron training script
            checkpoint_path: Path to checkpoint directory for resuming
            
        Returns:
            List of command line arguments
        """
        cmd = [
            "python", script_path,
            "--tensor-model-parallel-size", str(self.config["tensor_model_parallel_size"]),
            "--pipeline-model-parallel-size", str(self.config["pipeline_model_parallel_size"]),
            "--num-layers", str(self.config["num_layers"]),
            "--hidden-size", str(self.config["hidden_size"]),
            "--num-attention-heads", str(self.config["num_attention_heads"]),
            "--micro-batch-size", str(self.config["micro_batch_size"]),
            "--global-batch-size", str(self.config["global_batch_size"]),
            "--max-position-embeddings", str(self.config["max_position_embeddings"]),
            "--train-iters", str(self.config["train_iters"]),
            "--lr", str(self.config["learning_rate"]),
            "--min-lr", str(self.config["min_lr"]),
            "--lr-decay-style", self.config["lr_decay_style"],
            "--lr-warmup-fraction", str(self.config["lr_warmup_fraction"]),
            "--seed", str(self.config["seed"]),
            "--optimizer", self.config["optimizer"],
            "--weight-decay", str(self.config["weight_decay"]),
            "--clip-grad", str(self.config["clip_grad"]),
            "--adam-beta1", str(self.config["adam_beta1"]),
            "--adam-beta2", str(self.config["adam_beta2"]),
            "--distributed-backend", self.config["distributed_backend"],
            "--exit-interval", str(self.config["exit_interval"])
        ]
        
        # Add FP16 args if enabled
        if self.config["fp16"]:
            cmd.extend([
                "--fp16",
                "--initial-loss-scale", str(self.config["initial_loss_scale"])
            ])
        
        # Add checkpoint path for resuming if provided
        if checkpoint_path:
            cmd.extend([
                "--load", checkpoint_path
            ])
        
        return cmd