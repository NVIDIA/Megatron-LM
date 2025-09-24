"""
Adaptive Quantization Training Manager

This module implements time-resume adaptive quantization training that dynamically
switches between quantized (fp8/fp4) and high-precision (bf16) training based on loss.
"""

import os
import threading
import time
from collections import deque
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.training.checkpointing import save_checkpoint, load_checkpoint


class AdaptiveQuantizationManager:
    """
    Manages adaptive quantization training with time-resume capability.
    
    Features:
    - Dynamic switching between quantized and high-precision training
    - Asynchronous checkpoint saving
    - Loss-based threshold triggering
    - Window-based training management
    """
    
    def __init__(self, args, model, optimizer, opt_param_scheduler, iteration, num_floating_point_operations_so_far):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.opt_param_scheduler = opt_param_scheduler
        self.iteration = iteration
        self.num_floating_point_operations_so_far = num_floating_point_operations_so_far
        
        # Time-resume parameters
        self.enabled = getattr(args, 'time_resume', False)
        if not self.enabled:
            return
            
        self.loss_threshold = getattr(args, 'quant_loss_threshold', 0.1)
        self.window_size = getattr(args, 'quant_window_size', 5)
        self.checkpoint_interval = getattr(args, 'quant_checkpoint_interval', 1)
        self.fallback_strategy = getattr(args, 'quant_fallback_strategy', 'bf16')
        self.recovery_buffer_size = getattr(args, 'quant_recovery_buffer', 2)
        
        # State management
        self.current_precision = 'quantized'  # 'quantized' or 'bf16'
        self.window_iterations = 0
        self.window_start_iteration = iteration
        self.last_checkpoint_iteration = iteration
        self.checkpoint_queue = deque(maxlen=self.recovery_buffer_size)
        self.loss_history = deque(maxlen=10)  # Keep last 10 losses
        
        # Asynchronous checkpoint saving
        self.checkpoint_thread = None
        self.checkpoint_lock = threading.Lock()
        self.pending_checkpoints = []
        
        # Quantization type management
        self.quant_types = ['mxfp4', 'mxfp8', 'hifp8']
        self.current_quant_type = 'mxfp4'
        
        if parallel_state.get_tensor_model_parallel_rank() == 0:
            print(f"[AdaptiveQuantization] Initialized with window_size={self.window_size}, "
                  f"threshold={self.loss_threshold}, fallback={self.fallback_strategy}")
    
    def should_save_checkpoint(self, iteration: int) -> bool:
        """Check if we should save a checkpoint at this iteration."""
        if not self.enabled:
            return False
            
        return (iteration - self.last_checkpoint_iteration) >= self.checkpoint_interval
    
    def should_switch_precision(self, current_loss: float) -> Tuple[bool, str]:
        """
        Determine if we should switch training precision based on loss.
        
        Returns:
            (should_switch, new_precision)
        """
        if not self.enabled:
            return False, self.current_precision
            
        self.loss_history.append(current_loss)
        
        if len(self.loss_history) < 3:
            return False, self.current_precision
        
        # Calculate recent loss trend
        recent_losses = list(self.loss_history)[-3:]
        avg_recent_loss = sum(recent_losses) / len(recent_losses)
        
        # Switch to BF16 if loss exceeds threshold
        if self.current_precision == 'quantized' and avg_recent_loss > self.loss_threshold:
            if parallel_state.get_tensor_model_parallel_rank() == 0:
                print(f"[AdaptiveQuantization] Loss {avg_recent_loss:.4f} exceeds threshold {self.loss_threshold:.4f}, "
                      f"switching to {self.fallback_strategy}")
            return True, self.fallback_strategy
        
        # Switch back to quantized if loss is stable and low
        elif self.current_precision == self.fallback_strategy and avg_recent_loss < self.loss_threshold * 0.8:
            if parallel_state.get_tensor_model_parallel_rank() == 0:
                print(f"[AdaptiveQuantization] Loss {avg_recent_loss:.4f} is stable, "
                      f"switching back to {self.current_quant_type}")
            return True, 'quantized'
        
        return False, self.current_precision
    
    def save_checkpoint_async(self, iteration: int, tag: str = None):
        """Save checkpoint asynchronously to avoid blocking training."""
        if not self.enabled:
            return
            
        if self.checkpoint_thread and self.checkpoint_thread.is_alive():
            # Wait for previous checkpoint to complete
            self.checkpoint_thread.join()
        
        checkpoint_info = {
            'iteration': iteration,
            'tag': tag or f"window_{iteration // self.window_size}",
            'precision': self.current_precision,
            'quant_type': self.current_quant_type,
            'timestamp': time.time()
        }
        
        self.checkpoint_thread = threading.Thread(
            target=self._save_checkpoint_worker,
            args=(checkpoint_info,)
        )
        self.checkpoint_thread.start()
        
        self.last_checkpoint_iteration = iteration
        self.checkpoint_queue.append(checkpoint_info)
    
    def _save_checkpoint_worker(self, checkpoint_info: Dict[str, Any]):
        """Worker function for asynchronous checkpoint saving."""
        try:
            with self.checkpoint_lock:
                # Save checkpoint with timestamp
                checkpoint_name = f"{checkpoint_info['tag']}_iter{checkpoint_info['iteration']}_{checkpoint_info['precision']}"
                
                # Temporarily modify args.save to include checkpoint name
                original_save = getattr(self.args, 'save', None)
                
                # Create a unique checkpoint directory for this save
                if original_save:
                    checkpoint_dir = f"{original_save}_{checkpoint_info['tag']}_iter{checkpoint_info['iteration']}_{checkpoint_info['precision']}"
                    self.args.save = checkpoint_dir
                
                save_checkpoint(
                    iteration=checkpoint_info['iteration'],
                    model=self.model,
                    optimizer=self.optimizer,
                    opt_param_scheduler=self.opt_param_scheduler,
                    num_floating_point_operations_so_far=self.num_floating_point_operations_so_far
                )
                
                # Restore original save path
                self.args.save = original_save
                
                if parallel_state.get_tensor_model_parallel_rank() == 0:
                    print(f"[AdaptiveQuantization] Checkpoint saved: {checkpoint_name}")
                    
        except Exception as e:
            if parallel_state.get_tensor_model_parallel_rank() == 0:
                print(f"[AdaptiveQuantization] Error saving checkpoint: {e}")
    
    def load_recovery_checkpoint(self, target_iteration: int = None) -> bool:
        """
        Load the most recent checkpoint for recovery.
        
        Args:
            target_iteration: Specific iteration to load, or None for most recent
            
        Returns:
            True if checkpoint was loaded successfully
        """
        if not self.enabled or not self.checkpoint_queue:
            return False
        
        # Find the best checkpoint to load
        if target_iteration is not None:
            # Find checkpoint closest to target iteration
            best_checkpoint = None
            min_diff = float('inf')
            for checkpoint in self.checkpoint_queue:
                diff = abs(checkpoint['iteration'] - target_iteration)
                if diff < min_diff:
                    min_diff = diff
                    best_checkpoint = checkpoint
        else:
            # Load most recent checkpoint
            best_checkpoint = self.checkpoint_queue[-1]
        
        if best_checkpoint is None:
            return False
        
        try:
            checkpoint_name = f"{best_checkpoint['tag']}_iter{best_checkpoint['iteration']}_{best_checkpoint['precision']}"
            
            # Temporarily modify args.load to point to the checkpoint directory
            original_load = getattr(self.args, 'load', None)
            
            # Set the load path to the specific checkpoint directory
            if hasattr(self.args, 'save') and self.args.save:
                # Construct the checkpoint directory path
                checkpoint_dir = f"{self.args.save}_{checkpoint_name}"
                self.args.load = checkpoint_dir
            
            # Load checkpoint
            iteration, num_floating_point_operations_so_far = load_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                opt_param_scheduler=self.opt_param_scheduler,
                load_arg='load'
            )
            
            # Restore original load path
            self.args.load = original_load
            
            # Update state
            self.iteration = iteration
            self.num_floating_point_operations_so_far = num_floating_point_operations_so_far
            self.current_precision = best_checkpoint['precision']
            self.current_quant_type = best_checkpoint.get('quant_type', 'mxfp4')
            
            if parallel_state.get_tensor_model_parallel_rank() == 0:
                print(f"[AdaptiveQuantization] Loaded checkpoint: {checkpoint_name}, "
                      f"iteration={iteration}, precision={self.current_precision}")
            
            return True
            
        except Exception as e:
            if parallel_state.get_tensor_model_parallel_rank() == 0:
                print(f"[AdaptiveQuantization] Error loading checkpoint: {e}")
            return False
    
    def update_window_state(self, iteration: int):
        """Update window state and handle window transitions."""
        if not self.enabled:
            return
            
        self.window_iterations += 1
        
        # Check if we've completed a window
        if self.window_iterations >= self.window_size:
            # Save window checkpoint
            self.save_checkpoint_async(iteration, f"window_end_{iteration // self.window_size}")
            
            # Reset window state
            self.window_iterations = 0
            self.window_start_iteration = iteration
            
            if parallel_state.get_tensor_model_parallel_rank() == 0:
                print(f"[AdaptiveQuantization] Completed window, saved checkpoint at iteration {iteration}")
    
    def get_current_quantization_type(self) -> str:
        """Get the current quantization type for training."""
        if not self.enabled:
            return 'hifp8'  # Default
            
        if self.current_precision == 'quantized':
            return self.current_quant_type
        else:
            return 'bf16'
    
    def set_quantization_type(self, quant_type: str):
        """Set the quantization type for quantized training."""
        if not self.enabled:
            return
            
        if quant_type in self.quant_types + ['bf16']:
            self.current_quant_type = quant_type
            if parallel_state.get_tensor_model_parallel_rank() == 0:
                print(f"[AdaptiveQuantization] Set quantization type to {quant_type}")
    
    def get_training_state(self) -> Dict[str, Any]:
        """Get current training state for logging."""
        if not self.enabled:
            return {}
            
        return {
            'precision': self.current_precision,
            'quant_type': self.current_quant_type,
            'window_iterations': self.window_iterations,
            'window_start': self.window_start_iteration,
            'recent_losses': list(self.loss_history)[-5:] if self.loss_history else [],
            'checkpoints_available': len(self.checkpoint_queue)
        }
    
    def finalize(self):
        """Clean up resources and save final checkpoint."""
        if not self.enabled:
            return
            
        # Wait for any pending checkpoint saves
        if self.checkpoint_thread and self.checkpoint_thread.is_alive():
            self.checkpoint_thread.join()
        
        if parallel_state.get_tensor_model_parallel_rank() == 0:
            print("[AdaptiveQuantization] Finalized adaptive quantization training")


def get_adaptive_quantization_manager(args, model, optimizer, opt_param_scheduler, 
                                    iteration, num_floating_point_operations_so_far):
    """Factory function to create adaptive quantization manager."""
    return AdaptiveQuantizationManager(
        args, model, optimizer, opt_param_scheduler, 
        iteration, num_floating_point_operations_so_far
    )
