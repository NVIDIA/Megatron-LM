# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
import contextlib
from contextlib import ExitStack

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import megatron.core.parallel_state as ps
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from tests.unit_tests.a2a_overlap.utils import (
    deterministic_mode,
    get_test_config,
    get_valid_fp8_flags,
    get_valid_token_dispatcher_types,
)
from tests.unit_tests.test_utilities import Utils


class MockDataset(Dataset):
    """
    Mock dataset for torchtitan GPT training tests
    Generates synthetic tokenized sequences on-the-fly
    """

    def __init__(
        self,
        num_samples=10000,
        micro_batch_size=4,
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
        labels = rng.randint(0, self.vocab_size, size=self.sequence_length, dtype=np.int64)

        return {
            'input_ids': torch.from_numpy(tokens.copy()),
            'labels': torch.from_numpy(labels.copy()),
            "attention_mask": torch.ones(
                (1, self.sequence_length, self.sequence_length), dtype=bool
            ),
        }


def build_model(config):
    max_seq_len = 300

    # build layer spec
    transformer_layer_spec = get_gpt_decoder_block_spec(config=config, use_transformer_engine=True)
    mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec.layer_specs[-1], True)

    # build model
    gpt_model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        mtp_block_spec=mtp_block_spec,
        vocab_size=100,
        pre_process=True,
        post_process=True,
        max_sequence_length=max_seq_len,
    )
    return gpt_model


# Define a reusable context manager
@contextlib.contextmanager
def init_model_parallel(tp=1, pp=1, ep=1):
    try:
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            expert_model_parallel_size=ep,
        )
        yield
    finally:
        Utils.destroy_model_parallel()


def init_gpt_dataloader(
    dp_group, micro_batch_size=1, vocab_size=50257, sequence_length=128, batch_size=8
):
    dataset = MockDataset(
        num_samples=1000,
        micro_batch_size=micro_batch_size,
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        seed=42,
    )
    sampler = DistributedSampler(dataset, num_replicas=dp_group.size(), rank=dp_group.rank())
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return dataloader


class TestFusedLinearCrossEntropy:

    @pytest.mark.parametrize("fp8_flag", get_valid_fp8_flags())
    @pytest.mark.parametrize("mtp_layers", [0, 1])
    @pytest.mark.parametrize("dispatcher_type", get_valid_token_dispatcher_types())
    @pytest.mark.parametrize("layer_num", [2])
    def test_gpt_model(self, mtp_layers, dispatcher_type, fp8_flag, layer_num):
        with ExitStack() as stack:
            gpu_count = torch.cuda.device_count()
            tp = min(2, gpu_count)
            ep = gpu_count // tp
            stack.enter_context(init_model_parallel(tp=tp, ep=ep))
            stack.enter_context(deterministic_mode())

            # create TransformerConfig
            extra_kwargs = {
                "moe_token_dispatcher_type": dispatcher_type,
                "sequence_parallel": tp > 1,
                "tensor_model_parallel_size": tp,
            }
            if dispatcher_type == "flex":
                extra_kwargs["moe_enable_deepep"] = True
                extra_kwargs["moe_router_dtype"] = "fp32"
            if fp8_flag is not None:
                extra_kwargs["fp8"] = fp8_flag[0]
                extra_kwargs["fp8_recipe"] = fp8_flag[1]
            if mtp_layers > 0:
                extra_kwargs["mtp_num_layers"] = mtp_layers
                extra_kwargs["mtp_loss_scaling_factor"] = 1.1

            # build config
            config = get_test_config(num_layers=layer_num, extra_kwargs=extra_kwargs)
            config.expert_model_parallel_size = ep

            # build model
            gpt_model = build_model(config)
            gpt_model.cuda()

            dataloader = init_gpt_dataloader(
                ps.get_data_parallel_group(),
                vocab_size=gpt_model.vocab_size,
                micro_batch_size=1,
                sequence_length=gpt_model.max_sequence_length,
                batch_size=4,
            )
            # for batch in dataloder:
            for batch in dataloader:
                batch["position_ids"] = torch.arange(
                    gpt_model.max_sequence_length, dtype=torch.int64
                )
                batch = {k: v.cuda() for k, v in batch.items()}
                gpt_model.zero_grad()
                output = gpt_model(**batch)
                loss = output.sum()
                loss.backward()
