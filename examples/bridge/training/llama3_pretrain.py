# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Llama 3 pretraining using only MLM config types and MLM's training loop.

The architecture and training settings follow Bridge's Llama 3 8B BF16 recipe
(eea53e431a8b0b4e9c4f47a76784c6f3200c9a1f). No Bridge package is required.
Like that recipe, this example uses mock data and a 32K NullTokenizer for
from-scratch training; it does not load pretrained Llama weights.

Import ``llama3_pretrain_config`` to adjust run length, batch size, or topology.
"""

import torch
import torch.nn.functional as F

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer import TransformerConfig
from megatron.training.api import pretrain
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
from pretrain_gpt import forward_step


def llama3_pretrain_config() -> PretrainConfigContainer:
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
        context_parallel_size=2,
        max_seq_length_per_sample=8192,
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl="te",
    )
    return PretrainConfigContainer(
        model=GPTModelConfig(
            transformer=transformer,
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
            weight_decay=0.1,
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
        tokenizer=TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=32000),
        logger=LoggerConfig(log_interval=10),
        checkpoint=CheckpointConfig(save_interval=500, ckpt_format="torch_dist"),
    )


if __name__ == "__main__":
    pretrain(llama3_pretrain_config(), forward_step)
