# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import itertools
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.cuda_graphs import (
    CudaGraphManager,
    _CudagraphGlobalRecord,
    create_cudagraphs,
    delete_cuda_graphs,
)
from megatron.core.transformer.module import Float16Module
from megatron.rl import rl_utils
from megatron.rl.agent.api import TokenRollout
from megatron.rl.sequence_packing_utils import get_default_packed_seq_params
from megatron.training.arguments import parse_args, validate_args
from megatron.training.global_vars import destroy_global_vars, set_global_variables
from tests.unit_tests.test_utilities import Utils

BATCH = 2
SEQ = 4
VOCAB = 754


class MockModel(LanguageModule):
    def __init__(self, batch=BATCH, seq=SEQ, vocab=VOCAB):
        self.batch = batch
        self.seq = seq
        self.vocab = vocab
        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.config = TransformerConfig(
            num_attention_heads=8, num_layers=8, pipeline_dtype=torch.bfloat16
        )
        self.model_type = ModelType.encoder_or_decoder

    def __call__(self, x, position_ids, attention_mask, **kwargs):
        del position_ids
        del attention_mask
        batch, seq = x.shape
        mock_model_outputs = torch.ones((batch, seq, self.vocab), device=x.device)
        return mock_model_outputs

    def load_state_dict(self, params):
        del params

    def train(self, mode=True):
        del mode

    def state_dict(self):
        return {}

    def set_input_tensor(self, input_tensor):
        pass


class MockTokenizer:
    def __init__(self):
        self.pad = 42
        self.eod = 43
        self.vocab_size = VOCAB
        self.bos = None
        self.library = None

    def detokenize(self, tokens):
        return [str(tok) for tok in tokens]


@pytest.fixture
def initialize_model_parallel(request, monkeypatch):
    """Fixture to initialize and destroy model parallel.

    Parameters are passed via request.param as a tuple: (tp, pp)
    Skips if world_size < tp * pp.
    """
    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("LOG_TO_WANDB", "false")

    initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)

    tp, pp = request.param
    world_size = Utils.world_size
    Utils.initialize_model_parallel(tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp)
    model_parallel_cuda_manual_seed(123)
    dp = world_size // (tp * pp)
    yield world_size, dp, tp, pp
    Utils.destroy_model_parallel()
    destroy_global_vars()
    destroy_num_microbatches_calculator()


@pytest.fixture(autouse=True)
def cleanup_global_state():
    """Ensure global state is correctly cleaned up after every test."""
    yield
    destroy_global_vars()
    destroy_num_microbatches_calculator()


class TestRLUtils:
    """Test class for RL utilities."""

    def create_test_args(self, **kwargs):
        destroy_global_vars()
        destroy_num_microbatches_calculator()

        args = parse_args(ignore_unknown_args=True)
        args.num_layers = 8
        args.num_attention_heads = 8
        args.vocab_size = VOCAB
        args.hidden_size = 128
        args.max_position_embeddings = 256
        args.seq_length = 256
        args.wandb_project = None

        args.micro_batch_size = 1

        for key, value in kwargs.items():
            setattr(args, key, value)

        args = validate_args(args)
        set_global_variables(args, False)
        return args

    @pytest.mark.parametrize(
        "initialize_model_parallel",
        [
            pytest.param((tp, pp), id=f"tp{tp}-pp{pp}")
            for tp, pp in itertools.product([1, 2, 4, 8], [1, 2, 4, 8])
            if tp * pp <= Utils.world_size
        ],
        indirect=["initialize_model_parallel"],
    )
    @pytest.mark.parametrize("use_sequence_packing", [False])
    def test_get_logprobs(self, initialize_model_parallel, use_sequence_packing):
        """Test that getting logprobs at least does not crash."""
        self.create_test_args(rl_use_sequence_packing=use_sequence_packing)

        model = MockModel()
        tokens = torch.ones((BATCH, SEQ), dtype=torch.long)
        logprobs = rl_utils.get_logprobs(
            model, tokens, position_ids=None, sequence_packing=use_sequence_packing
        )
        if is_pp_last_stage(model.pg_collection.pp):
            # We chop off 1 element from the sequence dimension.
            assert logprobs.shape == (BATCH, SEQ - 1)
            # As we return ones as logits, all logprobs should be the same.
            assert torch.all(logprobs == logprobs[0, 0]).item()
        else:
            assert logprobs.shape == (BATCH, SEQ, VOCAB)

    def test_grpo_loss_calculation_all_pi_eq(self):
        # All policies are equal: clamping is inactive, ratios are ones.
        current_logprobs = torch.ones(BATCH, SEQ)
        old_logprobs = torch.ones(BATCH, SEQ)
        ref_logprobs = torch.ones(BATCH, SEQ)
        advantages = torch.zeros(BATCH)
        loss, kl_term, ratios, entropy_term, _, _ = rl_utils.calculate_grpo_loss(
            current_logprobs=current_logprobs,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            advantages=advantages,
            clamp_eps_lower=0.1,
            clamp_eps_upper=0.1,
            kl_beta=0.1,
            entropy_weight=0.0,
        )
        torch.testing.assert_close(loss, torch.zeros_like(loss))
        torch.testing.assert_close(kl_term, torch.zeros_like(kl_term))
        torch.testing.assert_close(ratios, torch.ones_like(ratios))
        torch.testing.assert_close(entropy_term, -torch.ones_like(ratios) * torch.e)

    def test_grpo_loss_calculation_2x_ratios(self):
        # All policies are equal: clamping is inactive, ratios are ones.
        current_logprobs = torch.ones(BATCH, SEQ)
        old_logprobs = torch.ones(BATCH, SEQ) - torch.log(torch.tensor([2.0]))
        ref_logprobs = torch.ones(BATCH, SEQ)
        advantages = torch.ones(BATCH)
        loss, kl_term, ratios, _, _, _ = rl_utils.calculate_grpo_loss(
            current_logprobs=current_logprobs,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            advantages=advantages,
            clamp_eps_lower=2.1,
            clamp_eps_upper=2.1,
            kl_beta=0.0,
            entropy_weight=0.0,
        )
        # Clamping does not affect us, as 2.1 [eps] > 2 [ratio].
        # kl_beta = 0 -> we only have the non-kl term of the loss active.
        torch.testing.assert_close(loss, -torch.ones_like(loss) * 2)
        # pi and pi_{ref} are the same here.
        torch.testing.assert_close(kl_term, torch.zeros_like(kl_term))
        # Current probs are 2x more probable than old pi.
        torch.testing.assert_close(ratios, torch.ones_like(ratios) * 2)

    def test_entropy_calculation(self):
        # All policies are equal: clamping is inactive, ratios are ones.
        current_logprobs = torch.ones(BATCH, SEQ)
        old_logprobs = torch.ones(BATCH, SEQ)
        ref_logprobs = torch.ones(BATCH, SEQ)
        advantages = torch.zeros(BATCH)
        loss, _, ratios, entropy_term, _, _ = rl_utils.calculate_grpo_loss(
            current_logprobs=current_logprobs,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            advantages=advantages,
            clamp_eps_lower=0.1,
            clamp_eps_upper=0.1,
            kl_beta=0.0,
            entropy_weight=1.0,
        )
        torch.testing.assert_close(loss, torch.ones_like(ratios) * torch.e)
        torch.testing.assert_close(entropy_term, -torch.ones_like(ratios) * torch.e)

    def test_grpo_loss_truncation(self):
        # All ratios are 2
        _, _, _, _, truncated_from_above, truncated_from_below = rl_utils.calculate_grpo_loss(
            current_logprobs=torch.ones(BATCH, SEQ),
            old_logprobs=0.5 * torch.ones(BATCH, SEQ),
            ref_logprobs=torch.ones(BATCH, SEQ),
            advantages=torch.zeros(BATCH),
            clamp_eps_lower=0.1,
            clamp_eps_upper=0.1,
            kl_beta=0.1,
            entropy_weight=0.0,
        )
        assert truncated_from_above.float().mean() == 1
        assert truncated_from_below.float().sum() == 0

        # All ratios are 0.01
        _, _, _, _, truncated_from_above, truncated_from_below = rl_utils.calculate_grpo_loss(
            current_logprobs=0.01 * torch.ones(BATCH, SEQ),
            old_logprobs=torch.ones(BATCH, SEQ),
            ref_logprobs=torch.ones(BATCH, SEQ),
            advantages=torch.zeros(BATCH),
            clamp_eps_lower=0.1,
            clamp_eps_upper=0.1,
            kl_beta=0.1,
            entropy_weight=0.0,
        )
        assert truncated_from_above.float().sum() == 0
        assert truncated_from_below.float().mean() == 1

        # Mixed ratios: [[2., 0.5], [20., 1.]]
        current_logprobs = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        old_logprobs = torch.tensor([[0.5, 2.0], [0.05, 1.0]])
        _, _, _, _, truncated_from_above, truncated_from_below = rl_utils.calculate_grpo_loss(
            current_logprobs=current_logprobs,
            old_logprobs=old_logprobs,
            ref_logprobs=old_logprobs,
            advantages=torch.zeros(BATCH),
            clamp_eps_lower=0.1,
            clamp_eps_upper=0.1,
            kl_beta=0.1,
            entropy_weight=0.0,
        )
        torch.testing.assert_close(
            truncated_from_above, torch.tensor([[True, False], [True, False]])
        )
        torch.testing.assert_close(
            truncated_from_below, torch.tensor([[False, True], [False, False]])
        )

    @pytest.mark.parametrize(
        "initialize_model_parallel",
        [
            pytest.param((tp, pp), id=f"tp{tp}-pp{pp}")
            for tp, pp in itertools.product([1, 2, 4, 8], [1, 2, 4, 8])
            if tp * pp <= Utils.world_size
        ],
        indirect=["initialize_model_parallel"],
    )
    def test_prepare_data_for_update(self, initialize_model_parallel):
        """Test that getting logprobs at least does not crash."""
        world_size, dp, tp, pp = initialize_model_parallel
        # Here I assume that we will be consuming all data in one step.
        group_size = 2
        self.create_test_args(
            micro_batch_size=2,
            seq_length=4,
            curr_iteration=1,
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            global_batch_size=dp * 2,
            grpo_prompts_per_step=dp,
            grpo_group_size=group_size,
        )

        model = MockModel()
        tokenizer = MockTokenizer()

        r1 = TokenRollout(
            trajectory=[[1, 2, 3]],
            reward=3.14,
            generation_mask=[[False, True, True]],
            logprobs=[[0.1, 0.2, 0.3]],
            env_id='MEGAENV',
            problem_id="2",
        )
        r2 = TokenRollout(
            trajectory=[[1, 2, 3, 4]],
            reward=0.14,
            generation_mask=[[False, True, True, True]],
            logprobs=[[0.1, 0.2, 0.3, -1.2]],
            env_id='MEGAENV',
            problem_id="2",
        )

        rollouts = [[r1, r2] for _ in range(dp)]
        try:
            rl_utils.prepare_data_for_update(
                [model], {}, rollouts, tokenizer, sequence_packing=False, is_correction=False
            )
        except AssertionError as e:
            # We expect trajectories to come padded there.
            assert str(e).startswith('Rollout is not the correct length')

        r1 = TokenRollout(
            trajectory=torch.tensor([[1, 2, 3, tokenizer.eod]], dtype=torch.float).cuda(),
            reward=3.14,
            generation_mask=torch.tensor([[False, True, True, True]], dtype=torch.float).cuda(),
            logprobs=torch.tensor([[-0.2, -0.3, -3.2]]).cuda(),
            env_id='MEGAENV',
            problem_id="2",
        )
        r2 = TokenRollout(
            trajectory=torch.tensor([[1, 2, 234, tokenizer.eod]], dtype=torch.float).cuda(),
            reward=0.14,
            generation_mask=torch.tensor([[False, True, True, True]], dtype=torch.float).cuda(),
            logprobs=torch.tensor([[-0.2, -0.3, -1.2]]),
            env_id='MEGAENV',
            problem_id="2",
        )
        rollouts = [[r1, r2] for _ in range(dp)]
        data_iter = rl_utils.prepare_data_for_update(
            [model], {}, rollouts, tokenizer, sequence_packing=False, is_correction=False
        )

        _, _, old_logprobs, _, _, _, _ = next(data_iter)
        # All logits are ones in the MockModel.
        # All probabilities should be uniform.
        torch.testing.assert_close(old_logprobs.exp(), torch.ones_like(old_logprobs) / VOCAB)

    @pytest.mark.parametrize("use_sequence_packing", [True, False])
    @pytest.mark.parametrize("num_turns", [1, 2])
    def test_prepare_trajectories(self, use_sequence_packing, num_turns):
        """Test that rollouts are properly prepared for training."""
        seq_length = 8
        self.create_test_args(
            rl_use_sequence_packing=use_sequence_packing,
            rl_sequence_packing_bin_size=20,
            rl_skip_bos_token=False,
            micro_batch_size=1,
            seq_length=seq_length,
        )
        tokenizer = MockTokenizer()

        # Create rollouts of varying lengths
        r1 = TokenRollout(
            trajectory=[[1, 2, 3, tokenizer.eod]] * num_turns,
            reward=3.14,
            generation_mask=[[False, True, True, True]] * num_turns,
            logprobs=[[0.1, 0.2, 0.3, 0.35]] * num_turns,
            env_id='MEGAENV',
            problem_id="1",
        )
        r2 = TokenRollout(
            trajectory=[[4, 5, 6, 7, tokenizer.eod]] * num_turns,
            reward=0.14,
            generation_mask=[[False, True, True, True, True]] * num_turns,
            logprobs=[[0.4, 0.5, 0.6, 0.7, 0.75]] * num_turns,
            env_id='MEGAENV',
            problem_id="2",
        )
        r3 = TokenRollout(
            trajectory=[[8, 9, tokenizer.eod]] * num_turns,
            reward=2.71,
            generation_mask=[[False, True, True]] * num_turns,
            logprobs=[[0.8, 0.9, 0.95]] * num_turns,
            env_id='MEGAENV',
            problem_id="3",
        )

        rollouts = [r1, r2, r3]

        trajs, genmask, inference_logprobs = rl_utils.prepare_trajectories(
            rollouts,
            tokenizer,
            seq_length,
            sequence_packing=use_sequence_packing,
            skip_bos_token=False,
        )

        expected_trajs = torch.tensor(
            [
                [1, 2, 3, tokenizer.eod] + [tokenizer.pad] * 4,
                [4, 5, 6, 7, tokenizer.eod] + [tokenizer.pad] * 3,
                [8, 9, tokenizer.eod] + [tokenizer.pad] * 5,
            ],
            dtype=torch.long,
            device=trajs.device,
        ).repeat_interleave(num_turns, dim=0)
        assert torch.equal(trajs, expected_trajs)

        expected_genmask = torch.tensor(
            [
                [False, True, True, True] + [False] * 4,
                [False, True, True, True, True] + [False] * 3,
                [False, True, True] + [False] * 5,
            ],
            dtype=torch.bool,
            device=genmask.device,
        ).repeat_interleave(num_turns, dim=0)
        assert torch.equal(genmask, expected_genmask)

        if use_sequence_packing:
            expected_logprobs = torch.tensor(
                [
                    [0.1, 0.2, 0.3, 0.35] + [0.0] * 4,
                    [0.4, 0.5, 0.6, 0.7, 0.75] + [0.0] * 3,
                    [0.8, 0.9, 0.95] + [0.0] * 5,
                ],
                dtype=torch.float32,
                device=inference_logprobs.device,
            ).repeat_interleave(num_turns, dim=0)
            torch.testing.assert_close(inference_logprobs, expected_logprobs, rtol=0, atol=0)
        else:
            expected_logprobs = [
                [0.1, 0.2, 0.3, 0.35],
                [0.4, 0.5, 0.6, 0.7, 0.75],
                [0.8, 0.9, 0.95],
            ]
            expected_logprobs = [el for el in expected_logprobs for _ in range(num_turns)]
            assert len(inference_logprobs) == len(expected_logprobs)
            for got, exp in zip(inference_logprobs, expected_logprobs):
                got_t = got if torch.is_tensor(got) else torch.tensor(got, dtype=torch.float32)
                exp_t = torch.tensor(exp, dtype=torch.float32, device=got_t.device)
                torch.testing.assert_close(got_t, exp_t, rtol=0, atol=0)

    def test_single_turn_advantage_calculation(self):
        rewards = [[-1, 1], [4, 4]]
        num_turns = [[1, 1], [1, 1]]
        advs = rl_utils.calculate_grpo_advantages(rewards, num_turns)
        torch.testing.assert_close(
            torch.tensor(advs), torch.tensor([-1, 1.0, 0.0, 0.0]), atol=1e-4, rtol=1e-5
        )

    def test_multi_turn_advantage_calculation(self):
        rewards = [[-1, 1], [4, 4]]
        num_turns = [[2, 1], [1, 3]]
        advs = rl_utils.calculate_grpo_advantages(rewards, num_turns)
        torch.testing.assert_close(
            torch.tensor(advs),
            torch.tensor([-1, -1, 1.0, 0.0, 0.0, 0.0, 0.0]),
            atol=1e-4,
            rtol=1e-5,
        )

    def test_pad_list_of_nones(self):
        with pytest.raises(ValueError) as e_info:
            rl_utils._pad_nonnull_with_zeros([None] * 3, 42)
        assert "At least one" in str(e_info)

    def test_pad_with_wrong_params(self):
        with pytest.raises(ValueError) as e_info:
            rl_utils._pad_nonnull_with_zeros([torch.zeros(5)], 4)
        assert "larger length" in str(e_info)

    def test_pad_full_size(self):
        padded = rl_utils._pad_nonnull_with_zeros([torch.zeros(5), torch.zeros(5)], 5)
        assert padded.shape == (2, 5)

    def test_pad_some_nones(self):
        padded = rl_utils._pad_nonnull_with_zeros([None, torch.zeros(5)], 5)
        assert padded.shape == (2, 5)
        assert (padded[0] == 0).all()

    def test_pad_normal(self):
        padded = rl_utils._pad_nonnull_with_zeros(
            [torch.zeros(2), torch.zeros(3), torch.zeros(4)], 5
        )
        assert padded.shape == (3, 5)

    @pytest.mark.parametrize(
        "initialize_model_parallel",
        [
            pytest.param((tp, pp), id=f"tp{tp}-pp{pp}")
            for tp, pp in itertools.product([1, 2], [1, 2])
            if tp * pp <= Utils.world_size
        ],
        indirect=["initialize_model_parallel"],
    )
    def test_grad_buffer_offload(self, initialize_model_parallel):
        """Test that grad buffer offload/restore correctly frees and restores GPU memory."""
        world_size, dp, tp, pp = initialize_model_parallel
        self.create_test_args(tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp)

        model_parallel_cuda_manual_seed(123)

        # Create a realistic GPTModel as used in RL training
        transformer_config = TransformerConfig(
            num_layers=2, hidden_size=64, num_attention_heads=4, use_cpu_initialization=True
        )
        gpt_model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=256,
            max_sequence_length=32,
        ).cuda()

        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=True,
            use_distributed_optimizer=True,
            overlap_grad_reduce=False,
            bucket_size=None,  # Single bucket for simplicity
        )

        ddp_model = DistributedDataParallel(
            transformer_config, ddp_config=ddp_config, module=gpt_model
        )

        all_buffers = ddp_model.buffers + ddp_model.expert_parallel_buffers

        # Verify initial storage is allocated
        initial_sizes = [buf.grad_data.storage().size() for buf in all_buffers]
        assert all(size > 0 for size in initial_sizes), "Expected non-zero initial storage"

        # Offload grad buffers to CPU
        ddp_model.offload_grad_buffers()

        # Verify storage is released
        for buf in all_buffers:
            assert buf.grad_data.storage().size() == 0, "Expected zero storage after offload"

        # Restore grad buffers to GPU
        ddp_model.restore_grad_buffers()

        # Verify storage is restored
        restored_sizes = [buf.grad_data.storage().size() for buf in all_buffers]
        assert (
            initial_sizes == restored_sizes
        ), f"Expected restored sizes {restored_sizes} to match initial {initial_sizes}"

    @pytest.mark.parametrize(
        "initialize_model_parallel",
        [
            pytest.param((tp, pp), id=f"tp{tp}-pp{pp}")
            for tp, pp in itertools.product([1, 2], [1, 2])
            if tp * pp <= Utils.world_size
        ],
        indirect=["initialize_model_parallel"],
    )
    def test_optimizer_offload(self, initialize_model_parallel):
        """Test that optimizer offload_to_cpu/restore_from_cpu correctly moves state to/from CPU."""
        world_size, dp, tp, pp = initialize_model_parallel
        self.create_test_args(tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp)
        model_parallel_cuda_manual_seed(123)

        # Create a realistic GPTModel as used in RL training
        transformer_config = TransformerConfig(
            num_layers=2, hidden_size=64, num_attention_heads=4, use_cpu_initialization=True
        )
        gpt_model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=256,
            max_sequence_length=32,
        ).cuda()

        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=True,
            use_distributed_optimizer=True,
            overlap_grad_reduce=False,
            bucket_size=None,  # Single bucket for simplicity
        )

        ddp_model = DistributedDataParallel(
            transformer_config, ddp_config=ddp_config, module=gpt_model
        )

        # Create optimizer
        optimizer_config = OptimizerConfig(
            optimizer='adam', bf16=True, use_distributed_optimizer=True
        )
        optimizer = get_megatron_optimizer(optimizer_config, [ddp_model])

        # Manually initialize optimizer state (simulating what happens after first step)
        # This avoids needing to run a full forward/backward/step cycle
        for opt in optimizer.chained_optimizers:
            if hasattr(opt, 'optimizer') and opt.optimizer is not None:
                for group in opt.optimizer.param_groups:
                    for p in group['params']:
                        if len(opt.optimizer.state[p]) == 0:
                            # Initialize Adam state (exp_avg and exp_avg_sq) on GPU
                            opt.optimizer.state[p]['exp_avg'] = torch.rand_like(p.data)
                            opt.optimizer.state[p]['exp_avg_sq'] = torch.rand_like(p.data)
                            opt.optimizer.state[p]['step'] = torch.tensor(1)

        # Helper to check if optimizer state tensors are on GPU or CPU
        def get_optimizer_state_devices():
            devices = set()
            for opt in optimizer.chained_optimizers:
                if hasattr(opt, 'optimizer') and opt.optimizer is not None:
                    for state_dict in opt.optimizer.state.values():
                        for v in state_dict.values():
                            if isinstance(v, torch.Tensor):
                                devices.add(str(v.device))
            return devices

        # Verify optimizer state is initially on GPU
        initial_devices = get_optimizer_state_devices()
        assert any(
            'cuda' in d for d in initial_devices
        ), f"Expected optimizer state on GPU initially, got devices: {initial_devices}"

        # Record GPU memory before offload
        torch.cuda.synchronize()
        memory_before_offload = torch.cuda.memory_allocated()

        # Offload optimizer state to CPU
        optimizer.offload_to_cpu()

        # Verify GPU memory decreased (optimizer state should be freed)
        torch.cuda.synchronize()
        memory_after_offload = torch.cuda.memory_allocated()
        assert memory_after_offload < memory_before_offload, (
            f"Expected GPU memory to decrease after offload. "
            f"Before: {memory_before_offload}, After: {memory_after_offload}"
        )

        # Verify optimizer state is now on CPU
        offloaded_devices = get_optimizer_state_devices()
        assert all(
            'cpu' in d for d in offloaded_devices
        ), f"Expected all optimizer state on CPU after offload, got devices: {offloaded_devices}"

        # Restore optimizer state to GPU
        optimizer.restore_from_cpu()

        # Verify optimizer state is back on GPU
        restored_devices = get_optimizer_state_devices()
        assert any(
            'cuda' in d for d in restored_devices
        ), f"Expected optimizer state on GPU after restore, got devices: {restored_devices}"

        # Verify GPU memory increased after restore (optimizer state reallocated)
        torch.cuda.synchronize()
        memory_after_restore = torch.cuda.memory_allocated()
        assert memory_after_restore > memory_after_offload, (
            f"Expected GPU memory to increase after restore. "
            f"After offload: {memory_after_offload}, After restore: {memory_after_restore}"
        )

    @pytest.mark.parametrize(
        "initialize_model_parallel",
        [
            pytest.param((tp, pp), id=f"tp{tp}-pp{pp}")
            for tp, pp in itertools.product([1, 2, 4], [1, 2])
            if tp * pp <= Utils.world_size
        ],
        indirect=["initialize_model_parallel"],
    )
    def test_gpt_logprobs(self, initialize_model_parallel):
        """Test get logprobs on an actual model, not on a mocked one.

        This can be useful for quick benchmarking/analyzing regressions too.
        """

        world_size, dp, tp, pp = initialize_model_parallel
        micro_batch_size = 2
        self.create_test_args(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            bf16=True,
            micro_batch_size=micro_batch_size,
        )
        model_parallel_cuda_manual_seed(123)

        transformer_config = TransformerConfig(
            num_layers=10,
            hidden_size=128,
            num_attention_heads=16,
            use_cpu_initialization=True,
            embedding_init_method_std=1.0,
            bf16=True,
            pipeline_dtype=torch.bfloat16,  # Without this, pp!=1 runs will fail.
        )
        vocab_size = 10_000
        pp_group = ProcessGroupCollection.use_mpu_process_groups().pp
        gpt_model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=vocab_size,
            max_sequence_length=4192,
            pre_process=is_pp_first_stage(pp_group),
            post_process=is_pp_last_stage(pp_group),
        ).cuda()
        sequence_length = gpt_model.max_sequence_length

        gpt_model = Float16Module(gpt_model.config, gpt_model)

        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()

        with torch.no_grad():
            logprobs = rl_utils.compute_logprobs_batch(
                model=gpt_model,
                data_loader=[(input_ids, position_ids)],
                forward_backward_func=get_forward_backward_func(pp_size=pp),
                packing_context=None,
                trajs_batch_size=micro_batch_size,
                seq_length=sequence_length,
                logprobs_batch_size=micro_batch_size,
                decoder_seq_length=sequence_length,
                dtype=torch.bfloat16,
                pp_group=gpt_model.pg_collection.pp,
                is_correction=False,
                collect_non_loss_data=True,
            )
        if is_pp_last_stage(pp_group):
            assert logprobs.shape == (micro_batch_size, sequence_length - 1)

    @pytest.mark.parametrize(
        "initialize_model_parallel",
        [pytest.param((1, 1), id="tp1-pp1")],
        indirect=["initialize_model_parallel"],
    )
    def test_get_logprobs_cuda_graphs(self, initialize_model_parallel):
        """Test that get_logprobs reuses CUDA graphs created during training forward pass.

        This test verifies that rl_utils.get_logprobs can reuse CUDA graphs by:
        1. Running a training-style forward pass on some model to record CUDA graph runners.
        2. Creating the CUDA graphs.
        3. Running `get_logprobs` to verify it reuses the same forward graph from training.
        """

        num_layers = 2

        world_size, dp, tp, pp = initialize_model_parallel
        self.create_test_args(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            rl_training_cuda_graphs=True,
            cuda_graph_impl="local",
            bf16=True,
            rl_sequence_packing_max_sequences_per_bin=4,
        )

        # Create a model with training CUDA graphs enabled
        transformer_config = TransformerConfig(
            num_layers=num_layers,
            hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=True,
            cuda_graph_impl="local",
            bf16=True,
        )
        model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=256,
            max_sequence_length=32,
        ).cuda()

        # Wrap in Float16Module so it accepts fp32_output argument from get_logprobs
        wrapped_model = Float16Module(transformer_config, model)

        # Create test inputs (batch_size=1 required for thd format with sequence packing)
        batch_size = 1
        seq_length = 16
        tokens = torch.randint(0, 256, (batch_size, seq_length), dtype=torch.long).cuda()
        position_ids = torch.arange(seq_length).unsqueeze(0).expand(batch_size, -1).cuda()

        # Create packed_seq_params for dummy data
        packed_seq_params = get_default_packed_seq_params(
            seq_length=seq_length, max_sequences_per_bin=4, device=tokens.device
        )

        # Run a single training forward pass to record cudagraphs
        output = wrapped_model(
            tokens,
            position_ids,
            attention_mask=None,
            packed_seq_params=packed_seq_params,
            runtime_gather_output=True,
            fp32_output=False,
        )

        # Run backward to reset runner status from BWD_READY back to FWD_READY
        # This is needed because get_logprobs runs in no_grad mode and expects FWD_READY
        loss = output.sum()
        loss.backward()

        # Collect all CudaGraphManager instances and their runners
        cudagraph_managers = []
        for module in wrapped_model.modules():
            if hasattr(module, 'cudagraph_manager') and module.cudagraph_manager is not None:
                cudagraph_managers.append(module.cudagraph_manager)

        # Record runner count before creating graphs
        runners_before = {id(mgr): len(mgr.cudagraph_runners) for mgr in cudagraph_managers}

        create_cudagraphs()

        # Verify that each runner has a fwd_graph created
        for mgr in cudagraph_managers:
            for runner in mgr.cudagraph_runners:
                assert runner.fwd_graph is not None, (
                    f"Expected runner to have fwd_graph created after create_cudagraphs(), "
                    f"but fwd_graph is None"
                )

        # Now test `get_logprobs`; this should reuse the existing CUDA graphs
        # We do not pass packed_seq_params; it should be created within `get_logprobs`
        logprobs = rl_utils.get_logprobs(
            wrapped_model, tokens, position_ids=position_ids, sequence_packing=True
        )

        # Verify that no new runners were created and graph was reused
        runners_after = {id(mgr): len(mgr.cudagraph_runners) for mgr in cudagraph_managers}
        for mgr_id, count_before in runners_before.items():
            count_after = runners_after[mgr_id]
            assert count_after == count_before, (
                f"Expected runner count to remain {count_before} after `get_logprobs`, "
                f"but got {count_after}. `get_logprobs` should not create new runners."
            )

        # Verify outputs are valid
        assert output is not None, "Training forward pass should return valid output"
        assert logprobs is not None, "get_logprobs should return valid output"

        # Destroy all captured graphs deterministically
        for l in model.decoder.layers:
            for runner in getattr(l.cudagraph_manager, "cudagraph_runners", []):
                # Safely delete both graphs if present
                if hasattr(runner, "fwd_graph"):
                    del runner.fwd_graph
                if hasattr(runner, "bwd_graph"):
                    del runner.bwd_graph

        # Ensure all pending work is complete and graph destruction runs now
        torch.cuda.synchronize()

        _CudagraphGlobalRecord.cudagraph_created = False
        _CudagraphGlobalRecord.cudagraph_record = []
        CudaGraphManager.global_mempool = None
        CudaGraphManager.fwd_mempools = None
        CudaGraphManager.bwd_mempools = None

    @pytest.mark.parametrize(
        "initialize_model_parallel",
        [pytest.param((1, 1), id="tp1-pp1")],
        indirect=["initialize_model_parallel"],
    )
    def test_prep_wandb_metrics(self, initialize_model_parallel):
        # This tests the computation and makes us fail noisily if
        # inputs assumptions are changed, e.g. we expect rewards to come in groups (list[list[int]]).
        traj_lens = [[3, 3], [1, 2]]
        turn_lens = [[1, 2, 1, 1, 1], [1, 2]]
        rewards = [[1, 1], [-1, 2]]
        num_turns = [[42, 2], [10, 8]]
        advantages = [0, 1]
        metrics = rl_utils.prep_wandb_metrics(
            MagicMock(), traj_lens, turn_lens, rewards, num_turns, advantages
        )
        assert metrics["mean_reward"] == 0.75
        assert metrics["mean_advantage"] == 0.5
        assert metrics["nonzero_groups_ratio"] == 0.5
        assert metrics["max_traj_length"] == 3
        assert metrics["min_traj_length"] == 1
        assert metrics["mean_traj_length"] == 2.25
        assert metrics["mean_traj_length_std"] == 0.25
        assert metrics["max_turn_length"] == 2
        assert metrics["min_turn_length"] == 1
        assert metrics["mean_turn_length"] == 1.35
        assert np.isclose(metrics["mean_turn_length_std"], 0.45)
        assert metrics["mean_num_turns"] == 15.5
        assert metrics["max_num_turns"] == 42
        assert metrics["min_num_turns"] == 2
