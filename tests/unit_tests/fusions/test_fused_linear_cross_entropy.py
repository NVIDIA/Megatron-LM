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

from megatron.core.fusions.fused_linear_cross_entropy import linear_cross_entropy

import os

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


@pytest.mark.skipif(
    "WORLD_SIZE" not in os.environ or os.environ["WORLD_SIZE"] < "2",
    reason="Requires torchrun with multiple GPUs"
)
class TestFusedLinearCrossEntropyOnGptModel:
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


@pytest.mark.skipif(
    "WORLD_SIZE" in os.environ and os.environ["WORLD_SIZE"] != "1",
    reason="Requires single GPU"
)
class TestFusedLinearCrossEntropyDataParallel:
    def cleanup(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        import gc

        gc.collect()
        torch.cuda.synchronize()

    @staticmethod
    def torch_linear_cross_entropy(
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        reduction: str,
        ignore_index: int
    ):
        # NOTE: need to convert to fp32 to fp32 accumulation,
        # thus assure accuracy
        logits = hidden.to(torch.float32) @ weight.T.to(torch.float32)
        logprobs = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            labels.view(-1),
            reduction=reduction,
            ignore_index=ignore_index,
        )
        return logprobs.to(torch.float32)

    @staticmethod
    def get_problems():
        return [
            (80, 125, 64),
            (80, 152064, 64),
            (1024, 152064, 4096),
            (4096, 152063, 8192),
            ((1, 4096), 152064, 8192),
            ((2, 4096), 152064, 8192),
        ]

    @staticmethod
    def get_ignore_index():
        return [-100, 4]

    def test_kernel_launch(self):
        """
        Check if the compiled kernel can be
        launched with different problem sizes
        """
        self.cleanup()

        num_tokens = [15, 26, 128, 513, 2048, 8192]
        vocab_size = 152064
        dim = 4096
        dtype = torch.bfloat16
        reduction = "mean"
        ignore_index = -100

        weight = torch.randn(vocab_size, dim, dtype=dtype, device="cuda").requires_grad_()
        for num_token in num_tokens:
            hidden = torch.randn(num_token, dim, dtype=dtype, device="cuda").requires_grad_()
            labels = torch.randint(0, vocab_size, (num_token,), dtype=torch.long, device="cuda")
            
            logprobs = linear_cross_entropy(hidden, weight, labels, reduction=reduction, ignore_index=ignore_index)
            assert not torch.isnan(logprobs).any()

            gLogprobs = torch.randn_like(logprobs)
            (d_hidden, d_weight) = torch.autograd.grad(
                (logprobs,),
                (hidden, weight),
                (gLogprobs,),
                retain_graph=False
            )
            assert not torch.isnan(d_hidden).any()
            assert not torch.isnan(d_weight).any()


    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("problem", get_problems())
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    @pytest.mark.parametrize("ignore_index", get_ignore_index())
    def test_correctness(
        self,
        dtype,
        problem,
        reduction,
        ignore_index
    ):
        num_tokens, vocabsize, dim = problem
        hidden_shape = (num_tokens, dim) if isinstance(num_tokens, int) else (*num_tokens, dim)
        labels_shape = (num_tokens,) if isinstance(num_tokens, int) else num_tokens
        
        hidden = (
            torch.empty(hidden_shape, dtype=dtype, device="cuda")
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        weight = (
            torch.empty((vocabsize, dim), dtype=dtype, device="cuda")
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        labels = torch.randint(0, vocabsize, labels_shape, dtype=torch.long, device="cuda")
        if ignore_index >=0 and ignore_index < vocabsize:
            pad_labels = torch.nn.functional.pad(labels, (0, 1), value=ignore_index)
            labels = pad_labels[..., 1:].contiguous()

        # forward
        torch_logprobs = self.torch_linear_cross_entropy(hidden, weight, labels, 
            reduction=reduction, ignore_index=ignore_index)

        custom_logprobs = linear_cross_entropy(hidden, weight, labels, 
            reduction=reduction, ignore_index=ignore_index)

        torch.testing.assert_close(
            torch_logprobs,
            custom_logprobs
        )

        # backward
        g_logprobs = (
            torch.empty_like(torch_logprobs)
            .uniform_(-0.1, 0.1)
        )

        (d_torch_hidden, d_torch_weight) = torch.autograd.grad(
            (torch_logprobs,),
            (hidden, weight),
            (g_logprobs,),
            retain_graph=False
        )

        (d_custom_hidden, d_custom_weight) = torch.autograd.grad(
            (custom_logprobs,),
            (hidden, weight),
            (g_logprobs,),
            retain_graph=False)

        torch.testing.assert_close(
            d_torch_hidden,
            d_custom_hidden,
            atol=1e-3,
            rtol=1e-3
        )
        torch.testing.assert_close(
            d_torch_weight,
            d_custom_weight,
            atol=1e-3,
            rtol=1e-3
        )

    @pytest.mark.parametrize("problem", [((1, 4096), 129280, 7168)])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("reduction", ["mean"])
    @pytest.mark.parametrize("ignore_index", [-100])
    def test_performance(
        self,
        problem,
        dtype,
        reduction,
        ignore_index
    ):
        num_tokens, vocabsize, dim = problem
        hidden_shape = (num_tokens, dim) if isinstance(num_tokens, int) else (*num_tokens, dim)
        labels_shape = (num_tokens,) if isinstance(num_tokens, int) else num_tokens

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch_fwd_latency = list()
        torch_bwd_latency = list()
        custom_fwd_latency = list()
        custom_bwd_latency = list()

        iterations = 5
        for i in range(iterations):
            hidden = (
                torch.empty(hidden_shape, dtype=dtype, device="cuda")
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )
            weight = (
                torch.empty((vocabsize, dim), dtype=dtype, device="cuda")
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )
            labels = torch.randint(0, vocabsize, labels_shape, dtype=torch.long, device="cuda")
            if ignore_index >=0 and ignore_index < vocabsize:
                pad_labels = torch.nn.functional.pad(labels, (0, 1), value=ignore_index)
                labels = pad_labels[..., 1:].contiguous()

            # -------- forward -------- #
            start_event.record()
            torch_logprobs = self.torch_linear_cross_entropy(
                hidden, weight, labels,
                reduction=reduction, 
                ignore_index=ignore_index
            )
            end_event.record()
            torch.cuda.synchronize()
            torch_fwd_latency.append(
                start_event.elapsed_time(end_event)
            )

            start_event.record()
            custom_logprobs = linear_cross_entropy(
                hidden, weight, labels,
                reduction=reduction, 
                ignore_index=ignore_index
            )
            end_event.record()
            torch.cuda.synchronize()
            custom_fwd_latency.append(
                start_event.elapsed_time(end_event)
            )

            # -------- backward -------- #
            g_logprobs = (
                torch.empty_like(torch_logprobs)
                .uniform_(-0.1, 0.1)
            )

            start_event.record()
            (d_torch_hidden, d_torch_weight) = torch.autograd.grad(
                (torch_logprobs,),
                (hidden, weight),
                (g_logprobs,),
                retain_graph=False
            )
            end_event.record()
            torch.cuda.synchronize()
            torch_bwd_latency.append(
                start_event.elapsed_time(end_event)
            )

            start_event.record()
            (d_custom_hidden, d_custom_weight) = torch.autograd.grad(
                (custom_logprobs,),
                (hidden, weight),
                (g_logprobs,),
                retain_graph=False
            )
            end_event.record()
            torch.cuda.synchronize()
            custom_bwd_latency.append(
                start_event.elapsed_time(end_event)
            )

        # --- remove first latency due to warmup --- #
        torch_fwd_latency = torch_fwd_latency[1:]
        torch_bwd_latency = torch_bwd_latency[1:]
        custom_fwd_latency = custom_fwd_latency[1:]
        custom_bwd_latency = custom_bwd_latency[1:]

        print()
        print(f"[INFO]: On problem {problem}, dtype {dtype}, reduction {reduction}:")
        print(f"[INFO]: Torch forward latency: {sum(torch_fwd_latency) / len(torch_fwd_latency):.2f} ms")
        print(f"[INFO]: Custom forward latency: {sum(custom_fwd_latency) / len(custom_fwd_latency):.2f} ms")
        print(f"[INFO]: Torch backward latency: {sum(torch_bwd_latency) / len(torch_bwd_latency):.2f} ms")
        print(f"[INFO]: Custom backward latency: {sum(custom_bwd_latency) / len(custom_bwd_latency):.2f} ms")

    @pytest.mark.parametrize("problem", [((1, 4096), 129280, 7168)])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("reduction", ["mean"])
    @pytest.mark.parametrize("ignore_index", [-100])
    def test_storage(
        self,
        problem,
        dtype,
        reduction,
        ignore_index
    ):
        num_tokens, vocabsize, dim = problem
        hidden_shape = (num_tokens, dim) if isinstance(num_tokens, int) else (*num_tokens, dim)
        labels_shape = (num_tokens,) if isinstance(num_tokens, int) else num_tokens
        print()
        print(f"[INFO]: On problem {problem}, dtype {dtype}, reduction {reduction}:")

        def torch_storage():
            hidden = (
                torch.empty(hidden_shape, dtype=dtype, device="cuda")
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )
            weight = (
                torch.empty((vocabsize, dim), dtype=dtype, device="cuda")
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )
            labels = torch.randint(0, vocabsize, labels_shape, dtype=torch.long, device="cuda")
            if ignore_index >=0 and ignore_index < vocabsize:
                pad_labels = torch.nn.functional.pad(labels, (0, 1), value=ignore_index)
                labels = pad_labels[..., 1:].contiguous()

            torch.cuda.reset_peak_memory_stats()
            torch_logprobs = self.torch_linear_cross_entropy(
                hidden, weight, labels,
                reduction=reduction, 
                ignore_index=ignore_index
            )
            torch.cuda.synchronize()
            torch_max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"[INFO]: Torch Forward pass peak memory: {torch_max_memory:.2f} MB")

            torch.cuda.reset_peak_memory_stats()
            g_logprobs = (
                torch.empty_like(torch_logprobs)
                .uniform_(-0.1, 0.1)
            )
            (d_torch_hidden, d_torch_weight) = torch.autograd.grad(
                (torch_logprobs,),
                (hidden, weight),
                (g_logprobs,),
                retain_graph=False
            )
            torch.cuda.synchronize()
            torch_backward_max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"[INFO]: Torch Backward pass peak memory: {torch_backward_max_memory:.2f} MB")

        def custom_storage():
            hidden = (
                torch.empty(hidden_shape, dtype=dtype, device="cuda")
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )
            weight = (
                torch.empty((vocabsize, dim), dtype=dtype, device="cuda")
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )
            labels = torch.randint(0, vocabsize, labels_shape, dtype=torch.long, device="cuda")
            if ignore_index >=0 and ignore_index < vocabsize:
                pad_labels = torch.nn.functional.pad(labels, (0, 1), value=ignore_index)
                labels = pad_labels[..., 1:].contiguous()

            torch.cuda.reset_peak_memory_stats()
            custom_logprobs = linear_cross_entropy(
                hidden, weight, labels,
                reduction=reduction, 
                ignore_index=ignore_index
            )
            torch.cuda.synchronize()
            custom_max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"[INFO]: Custom Forward pass peak memory: {custom_max_memory:.2f} MB")

            torch.cuda.reset_peak_memory_stats()
            g_logprobs = (
                torch.empty_like(custom_logprobs)
                .uniform_(-0.1, 0.1)
            )
            (d_custom_hidden, d_custom_weight) = torch.autograd.grad(
                (custom_logprobs,),
                (hidden, weight),
                (g_logprobs,),
                retain_graph=False
            )
            torch.cuda.synchronize()
            custom_backward_max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"[INFO]: Custom Backward pass peak memory: {custom_backward_max_memory:.2f} MB")

        
        self.cleanup()
        torch_storage()
        self.cleanup()
        custom_storage()