# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Llama 3 pretraining using MLM config types and MLM's training loop.

The architecture and training settings follow Bridge's Llama 3 8B BF16 recipe
(eea53e431a8b0b4e9c4f47a76784c6f3200c9a1f). No Bridge package is required.
This recipe keeps Llama 3's 128256-token architecture vocabulary. Mock data and
a NullTokenizer support from-scratch training without downloading weights.

Replace cfg.data and cfg.tokenizer together to train on tokenized real data.
"""

import torch
import torch.nn.functional as F

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer import TransformerConfig
from megatron.training.config import (
    CheckpointConfig,
    DataConfig,
    DataloaderConfig,
    LoggerConfig,
    PretrainConfigContainer,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
    ValidationConfig,
)
from megatron.training.models.gpt import GPTModelConfig


def llama3_8b_pretrain_config() -> PretrainConfigContainer:
    """Return the Llama 3 8B mock pretraining recipe as native MLM configuration."""
    transformer = TransformerConfig(
        num_layers=32,
        hidden_size=4096,
        ffn_hidden_size=14336,
        num_attention_heads=32,
        num_query_groups=8,
        normalization="RMSNorm",
        layernorm_epsilon=1e-5,
        activation_func=F.silu,
        gated_linear_unit=True,
        add_bias_linear=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        tensor_model_parallel_size=2,
        sequence_parallel=True,
        max_seq_length_per_sample=8192,
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl="te",
    )
    return PretrainConfigContainer(
        model=GPTModelConfig(
            transformer=transformer,
            vocab_size=128256,
            seq_length=8192,
            position_embedding_type="rope",
            rotary_base=500000,
            share_embeddings_and_output_weights=False,
        ),
        train=TrainingConfig(
            train_iters=1168251,
            global_batch_size=512,
            micro_batch_size=1,
            manual_gc=True,
            manual_gc_interval=100,
        ),
        validation=ValidationConfig(eval_interval=2000, eval_iters=32),
        optimizer=OptimizerConfig(
            optimizer="adam",
            lr=3e-4,
            min_lr=3e-5,
            adam_beta1=0.9,
            adam_beta2=0.95,
            # Bridge's constant scheduler applies 0.033, overriding its optimizer default.
            weight_decay=0.033,
            clip_grad=1.0,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_distributed_optimizer=True,
            overlap_param_gather=True,
        ),
        scheduler=SchedulerConfig(lr_decay_style="cosine", lr_warmup_iters=2000),
        ddp=DistributedDataParallelConfig(
            use_distributed_optimizer=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            grad_reduce_in_fp32=True,
            average_in_collective=True,
            check_for_nan_in_grad=True,
        ),
        data=DataConfig(
            dataset=GPTDatasetConfig(
                random_seed=1234,
                sequence_length=8192,
                split="9999,8,2",
                reset_position_ids=False,
                reset_attention_mask=False,
                eod_mask_loss=False,
                create_attention_mask=False,
            ),
            dataset_type=MockGPTDataset,
            dataloader=DataloaderConfig(num_workers=8),
        ),
        tokenizer=TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=128256),
        logger=LoggerConfig(log_interval=10, log_throughput=True),
        checkpoint=CheckpointConfig(save_interval=500, ckpt_format="torch_dist"),
    )
