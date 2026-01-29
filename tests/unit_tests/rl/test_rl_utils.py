# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import itertools
import os
from types import SimpleNamespace

import pytest
import torch

from megatron.core.enums import ModelType
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.core.pipeline_parallel.utils import is_pp_last_stage
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from megatron.rl import rl_utils
from megatron.rl.agent.api import TokenRollout
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

    tp, pp = request.param
    world_size = Utils.world_size
    Utils.initialize_model_parallel(tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp)
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
