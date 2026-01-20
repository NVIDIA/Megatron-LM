#!/usr/bin/env python3
"""Pretrain GPT-OSS 20B from scratch (or from converted checkpoint)."""

from megatron.bridge.recipes.gpt_oss import gpt_oss_20b_pretrain_config
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain

if __name__ == "__main__":
    cfg = gpt_oss_20b_pretrain_config(
        seq_length=1024,
        mock=True,  # Use mock data
    )
    
    cfg.train.train_iters = 100
    cfg.scheduler.lr_decay_iters = 100
    cfg.model.vocab_size = 8192
    cfg.tokenizer.vocab_size = cfg.model.vocab_size
    
    # Optional: Load converted checkpoint
    # cfg.checkpoint.pretrained_checkpoint = "./megatron_checkpoints/gpt_oss_20b"
    
    pretrain(cfg, forward_step)