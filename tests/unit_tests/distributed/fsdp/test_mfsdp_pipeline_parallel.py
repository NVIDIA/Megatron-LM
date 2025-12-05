# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
from contextlib import contextmanager
from functools import partial

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import megatron.core.parallel_state as ps
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.utils import get_attr_wrapped_model
from megatron.training.utils import is_first_or_last_pipeline_stage
from tests.unit_tests.a2a_overlap.utils import deterministic_mode, get_test_config
from tests.unit_tests.test_utilities import Utils


class MockDataset(Dataset):
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


def get_gpt_data_iterator(
    dp_group, num_samples=1000, vocab_size=50257, sequence_length=128, batch_size=8, seed=42
):
    dataset = MockDataset(
        num_samples=num_samples, sequence_length=sequence_length, vocab_size=vocab_size, seed=seed
    )
    sampler = DistributedSampler(dataset, num_replicas=dp_group.size(), rank=dp_group.rank())
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    for batch in dataloader:
        # Move tensors to GPU
        batch["position_ids"] = torch.arange(sequence_length, dtype=torch.int64, device="cuda")
        batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
        yield batch


def get_batch(data_iterator, vp_stage=None):
    """Generate a batch."""
    rank = torch.distributed.get_rank()
    # TODO: this is pretty hacky, find a better way
    if not is_first_or_last_pipeline_stage(vp_stage):
        return None, None, None, None, None

    data = next(data_iterator)
    batch = {
        'tokens': data["tokens"].cuda(non_blocking=True),
        'labels': data["labels"].cuda(non_blocking=True),
        'loss_mask': data["loss_mask"].cuda(non_blocking=True),
        'attention_mask': (
            None if "attention_mask" not in data else data["attention_mask"].cuda(non_blocking=True)
        ),
        'position_ids': data["position_ids"].cuda(non_blocking=True),
    }

    return batch.values()


def _forward_step_func(data_iterator, model, device="cuda"):

    def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):

        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        # If you have data parallel reduce loss across data parallel groups.
        # If pipeline parallel, loss computation is done only in last stage.

        return loss, {'lm loss': loss}

    vp_stage = get_attr_wrapped_model(model, "vp_stage")
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator, vp_stage)
    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def build_model(config, max_seq_len=1024, vocab_size=100):
    # build layer spec
    transformer_layer_spec = get_gpt_decoder_block_spec(config=config, use_transformer_engine=True)
    mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec.layer_specs[-1], True)

    # build model
    gpt_model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        mtp_block_spec=mtp_block_spec,
        vocab_size=vocab_size,
        pre_process=ps.is_pipeline_first_stage(),
        post_process=ps.is_pipeline_last_stage(),
        max_sequence_length=max_seq_len,
    )
    return gpt_model


# Define a reusable context manager
@contextmanager
def megatron_model_parallel(tp, pp, ep):
    Utils.destroy_model_parallel()
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        expert_model_parallel_size=ep,
    )
    yield
    Utils.destroy_model_parallel()


class Test_MFSDP_PipelineParallel:

    @pytest.mark.parametrize("mtp_layers", [1, 0])
    @pytest.mark.parametrize(
        "mfsdp_sharding_strategy", ["optim", "optim_grads", "optim_grads_params"]
    )
    @pytest.mark.parametrize(
        "tp_pp_ep",
        [{"tp": 2, "pp": 2, "ep": 1}, {"tp": 1, "pp": 2, "ep": 1}, {"tp": 2, "pp": 1, "ep": 1}],
    )
    def test_gpt_model(self, mtp_layers, mfsdp_sharding_strategy, tp_pp_ep):
        with megatron_model_parallel(**tp_pp_ep) as _, deterministic_mode() as _:
            # create TransformerConfig
            extra_kwargs = {}
            if mtp_layers > 0:
                extra_kwargs["mtp_num_layers"] = mtp_layers
                extra_kwargs["mtp_loss_scaling_factor"] = 1.1

            # build config
            config = get_test_config(num_layers=4, extra_kwargs=extra_kwargs, num_moe_experts=None)
            config.pipeline_model_parallel_size = ps.get_pipeline_model_parallel_world_size()
            config.tensor_model_parallel_size = ps.get_tensor_model_parallel_world_size()
            config.expert_model_parallel_size = ps.get_expert_model_parallel_world_size()
            config.deallocate_pipeline_outputs = True

            # build model
            max_seq_len = 128
            vocab_size = 1000
            gpt_model = build_model(config, max_seq_len, vocab_size).cuda()
            gpt_model = FullyShardedDataParallel(
                module=gpt_model,
                config=config,
                ddp_config=DistributedDataParallelConfig(
                    use_megatron_fsdp=True,
                    data_parallel_sharding_strategy=mfsdp_sharding_strategy,
                    suggested_communication_unit_size=100,
                ),
            )

            # build optimizer
            optim = get_megatron_optimizer(
                config=OptimizerConfig(
                    optimizer='adam', lr=0.01, bf16=True, use_distributed_optimizer=True
                ),
                model_chunks=[gpt_model],
            )

            # training loop
            micro_batch_size = 2
            data_iterator = get_gpt_data_iterator(
                ps.get_data_parallel_group(),
                vocab_size=vocab_size,
                sequence_length=max_seq_len,
                batch_size=micro_batch_size,
            )
            forward_backward_func = get_forward_backward_func()
            for _ in range(3):  # run 3 iterations
                optim.zero_grad()
                forward_backward_func(
                    forward_step_func=_forward_step_func,
                    data_iterator=data_iterator,
                    model=gpt_model,
                    num_microbatches=2,
                    seq_length=max_seq_len,
                    micro_batch_size=micro_batch_size,
                    forward_only=False,
                )
                optim.step()
