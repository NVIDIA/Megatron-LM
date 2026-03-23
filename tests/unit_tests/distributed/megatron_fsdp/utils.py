# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
import sys
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from gpt_builders import gpt_builder
from megatron.core.distributed import finalize_model_grads
from megatron.core.enums import ModelType
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.utils import get_attr_wrapped_model
from megatron.training.arguments import parse_args, validate_args
from megatron.training.global_vars import destroy_global_vars, set_global_variables
from megatron.training.training import setup_model_and_optimizer
from megatron.training.utils import is_first_or_last_pipeline_stage
from model_provider import model_provider


def pretrain_forward_backward(
    *, model, data_iterator, sequence_length=128, micro_batch_size=2, num_micro_batches=1
):
    forward_backward_func = get_forward_backward_func()
    output = forward_backward_func(
        forward_step_func=_forward_step_func,
        data_iterator=data_iterator,
        model=model,
        num_microbatches=num_micro_batches,
        seq_length=sequence_length,
        micro_batch_size=micro_batch_size,
        forward_only=False,
    )
    return output


def make_gpt_mock_data_iterator(
    dp_group, num_samples=1000, vocab_size=50257, sequence_length=128, batch_size=8, seed=42
):
    dataset = GPTMockDataset(
        num_samples=num_samples, sequence_length=sequence_length, vocab_size=vocab_size, seed=seed
    )
    sampler = DistributedSampler(dataset, num_replicas=dp_group.size(), rank=dp_group.rank())
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    for batch in dataloader:
        batch["position_ids"] = torch.arange(sequence_length, dtype=torch.int64)
        yield batch


def make_moe_args_model_and_optimizer(ut_filename, **overrides):
    sys.argv = [ut_filename]
    base_args = dict(
        num_layers=4,
        mtp_num_layers=1,
        hidden_size=128,
        num_attention_heads=2,
        max_position_embeddings=128,
        bf16=False,
        add_bias_linear=False,
        swiglu=True,
        position_embedding_type="rope",
        rotary_percent=1.0,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        num_experts=4,
        moe_shared_expert_intermediate_size=256,
        moe_layer_freq=[0, 0, 1, 1],
        moe_permute_fusion=True,
        moe_router_fusion=True,
        moe_router_topk=2,
        moe_router_dtype="fp32",
        create_attention_mask_in_dataloader=True,
        lr=3e-5,
        min_lr=3e-5,
        use_distributed_optimizer=True,
        finalize_model_grads_func=finalize_model_grads,
    )

    base_args.update(overrides)
    args = parse_args()
    for key, value in base_args.items():
        setattr(args, key, value)

    validate_args(args)

    destroy_global_vars()
    destroy_num_microbatches_calculator()
    set_global_variables(args, build_tokenizer=False)

    model, optimizer, _ = setup_model_and_optimizer(
        model_provider_func=partial(model_provider, gpt_builder),
        model_type=ModelType.encoder_or_decoder,
    )
    return model, optimizer


def set_manual_seed(seed=42):
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)


class GPTMockDataset(Dataset):
    """
    Mock dataset for torchtitan GPT training tests
    Generates synthetic tokenized sequences on-the-fly
    """

    def __init__(
        self,
        num_samples=10000,
        micro_batch_size=1,
        sequence_length=2048,
        vocab_size=128256,
        seed=42,
    ):
        """
        Initialize mock dataset

        Args:
            num_samples: Total number of samples
            sequence_length: Length of each sequence
            vocab_size: Size of vocabulary
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.micro_batch_size = micro_batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.seed = seed

        # Set numpy seed for deterministic generation
        np.random.seed(seed)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Generate a single training sample

        Returns:
            dict with 'tokens' and 'labels'
        """
        # Use idx as seed for reproducible but varied samples
        rng = np.random.RandomState(self.seed + idx)

        # Generate random token sequence
        tokens = rng.randint(0, self.vocab_size, size=self.sequence_length, dtype=np.int64)

        # Labels are tokens shifted by 1 (next token prediction)
        labels = 1 + tokens

        return {
            'tokens': torch.from_numpy(tokens.copy()),
            'labels': torch.from_numpy(labels.copy()),
            "attention_mask": torch.ones(
                (1, self.sequence_length, self.sequence_length), dtype=bool
            ),
            "loss_mask": torch.ones(self.sequence_length),
        }


def _forward_step_func(data_iterator, model, device="cuda"):

    def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):

        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        # If you have data parallel reduce loss across data parallel groups.
        # If pipeline parallel, loss computation is done only in last stage.

        return loss, {'lm loss': loss}

    vp_stage = get_attr_wrapped_model(model, "vp_stage")

    if not is_first_or_last_pipeline_stage(vp_stage):
        tokens, labels, loss_mask, attention_mask, position_ids = None, None, None, None, None
    else:
        data = next(data_iterator)
        tokens = data["tokens"].to(device, non_blocking=True)
        labels = data["labels"].to(device, non_blocking=True)
        loss_mask = data["loss_mask"].to(device, non_blocking=True)
        attention_mask = (
            None
            if "attention_mask" not in data
            else data["attention_mask"].to(device, non_blocking=True)
        )
        position_ids = data["position_ids"].to(device, non_blocking=True)

    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(loss_func, loss_mask)
