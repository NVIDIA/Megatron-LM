# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
from unittest.mock import patch

import pytest
import torch

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import Float16Module
from megatron.rl import rl_utils, sequence_packing_utils
from megatron.rl.agent.api import TokenRollout
from megatron.training import arguments, global_vars
from tests.unit_tests.test_utilities import Utils

BATCH = 2
SEQ = 4
VOCAB = 754


class MockModel(LanguageModule):
    def __init__(self, batch=BATCH, seq=SEQ, vocab=VOCAB):
        self.batch = batch
        self.seq = seq
        self.vocab = vocab
        self.config = TransformerConfig(num_attention_heads=1, num_layers=1)
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


@pytest.fixture(scope='module', autouse=True)
def mock_pipeline_stuff():
    with patch('megatron.rl.rl_utils.is_pipeline_last_stage', return_value=True):
        yield


def test_get_logprobs():
    """Test that getting logprobs at least does not crash."""
    # We use args inside of get_logprobs, we need to initialize them.
    args = arguments.parse_args(ignore_unknown_args=True)
    global_vars.set_args(args)

    tokens = torch.ones((BATCH, SEQ), dtype=torch.long)
    logprobs = rl_utils.get_logprobs(MockModel(), tokens, position_ids=None)
    # We chop off 1 element from the sequence dimension.
    assert logprobs.shape == (BATCH, SEQ - 1)
    # As we return ones as logits, all logprobs should be the same.
    assert torch.all(logprobs == logprobs[0, 0]).item()


def test_get_logprobs_with_sequence_packing():
    """Test that getting logprobs at least does not crash."""
    # We use args inside of get_logprobs, we need to initialize them.
    args = arguments.parse_args(ignore_unknown_args=True)
    setattr(args, 'rl_use_sequence_packing', True)
    global_vars.set_args(args)

    tokens = torch.ones((BATCH, SEQ), dtype=torch.long)
    logprobs = rl_utils.get_logprobs(MockModel(), tokens, position_ids=None)
    # We chop off 1 element from the sequence dimension.
    assert logprobs.shape == (BATCH, SEQ - 1)
    # As we return ones as logits, all logprobs should be the same.
    assert torch.all(logprobs == logprobs[0, 0]).item()


@patch('torch.distributed.get_rank', return_value=0)
def test_prepare_trajectories(mock_rank):
    # Make sure sequence packing is disabled for this test
    import megatron.training.global_vars as global_vars

    old_args = global_vars.get_args() if global_vars.get_args() is not None else None

    # Create minimal args without sequence packing
    args = type('Args', (), {})()
    args.rl_use_sequence_packing = False
    args.rl_inference_logprobs_is_correction = True
    global_vars.set_args(args)

    tokenizer = MockTokenizer()
    r1 = TokenRollout(
        trajectory=[1, 2, tokenizer.eod],
        reward=3.14,
        generation_mask=[False, True, True],
        logprobs=[0.1, 0.2, 0.3],
        env_id='MEGAENV',
        problem_id="2",
    )
    r2 = TokenRollout(
        trajectory=[1, 2, tokenizer.eod],
        reward=0.14,
        generation_mask=[False, True, True],
        logprobs=[0.1, 0.2, 0.3],
        env_id='MEGAENV',
        problem_id="2",
    )
    rollouts = [[r1, r2]]
    seq_len = 7

    trajs, genmask, inference_logprobs = rl_utils.prepare_trajectories(rollouts, tokenizer, seq_len)

    # Check that inference logprobs are being returned.
    torch.testing.assert_close(inference_logprobs[0], torch.tensor([0.1, 0.2, 0.3]))
    torch.testing.assert_close(inference_logprobs[1], torch.tensor([0.1, 0.2, 0.3]))

    expected_mask = torch.tensor(
        [
            [False, True, True, False, False, False, False],
            [False, True, True, False, False, False, False],
        ]
    )
    torch.testing.assert_close(genmask, expected_mask)

    expected_trajs = torch.tensor([[1, 2, 43, 42, 42, 42, 42], [1, 2, 43, 42, 42, 42, 42]])
    torch.testing.assert_close(trajs, expected_trajs)


@patch('torch.distributed.get_rank', return_value=0)
def test_prepare_trajectories_with_packing(mock_rank):
    """Test that rollouts data is properly prepared with sequence packing enabled."""
    # Initialize args for sequence packing
    args = arguments.parse_args(ignore_unknown_args=True)
    setattr(args, 'micro_batch_size', 1)
    setattr(args, 'global_batch_size', 1)
    setattr(args, 'rl_use_sequence_packing', True)
    global_vars.set_args(args)

    tokenizer = MockTokenizer()
    r1 = TokenRollout(
        trajectory=[1, 2, tokenizer.eod],
        reward=3.14,
        generation_mask=[False, True, True],
        logprobs=[0.1, 0.2, 0.3],
        env_id='MEGAENV',
        problem_id="2",
    )
    r2 = TokenRollout(
        trajectory=[1, 2, 3, tokenizer.eod],
        reward=0.14,
        generation_mask=[False, True, True, True],
        logprobs=[0.1, 0.2, 0.3, -1.2],
        env_id='MEGAENV',
        problem_id="2",
    )
    rollouts = [[r1, r2]]
    seq_len = 7

    trajs, genmask, inference_logprobs = rl_utils.prepare_trajectories(rollouts, tokenizer, seq_len)

    # With sequence packing, inference logprobs should be padded to same length
    assert isinstance(inference_logprobs, torch.Tensor)
    assert inference_logprobs.shape == (2, 7)  # 2 sequences, each padded to seq_len

    # Check values (padded with zeros)
    torch.testing.assert_close(
        inference_logprobs[0], torch.tensor([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0])
    )
    torch.testing.assert_close(
        inference_logprobs[1], torch.tensor([0.1, 0.2, 0.3, -1.2, 0.0, 0.0, 0.0])
    )

    expected_mask = torch.tensor(
        [
            [False, True, True, False, False, False, False],
            [False, True, True, True, False, False, False],
        ]
    )
    torch.testing.assert_close(genmask, expected_mask)

    expected_trajs = torch.tensor([[1, 2, 43, 42, 42, 42, 42], [1, 2, 3, 43, 42, 42, 42]])
    torch.testing.assert_close(trajs, expected_trajs)


def test_grpo_loss_calculation_all_pi_eq():
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


def test_grpo_loss_calculation_2x_ratios():
    # All policies are equal: clamping is inactive, ratios are ones.
    current_logprobs = torch.ones(BATCH, SEQ)
    old_logprobs = torch.ones(BATCH, SEQ) - torch.log(torch.Tensor([2]))
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


def test_entropy_calculation():
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


def test_grpo_loss_truncation():

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
    # ratios: [[2., 0.5],[20., 1.]]
    torch.testing.assert_close(truncated_from_above, torch.tensor([[True, False], [True, False]]))
    torch.testing.assert_close(truncated_from_below, torch.tensor([[False, True], [False, False]]))


@pytest.mark.skipif(True, reason="broken")
def test_prepare_data_for_update():
    """Test that getting logprobs at least does not crash."""
    Utils.initialize_model_parallel()

    args = arguments.parse_args(ignore_unknown_args=True)
    setattr(args, 'data_parallel_size', 1)
    setattr(args, 'micro_batch_size', 2)
    setattr(args, 'global_batch_size', 2)
    setattr(args, 'seq_length', 4)
    setattr(args, 'curr_iteration', 1)
    global_vars.unset_global_variables()
    global_vars.set_global_variables(args, build_tokenizer=False)

    model = MockModel()
    tokenizer = MockTokenizer()

    try:
        r1 = TokenRollout(
            trajectory=[1, 2, 3],
            reward=3.14,
            generation_mask=[False, True, True],
            logprobs=[0.1, 0.2, 0.3],
            env_id='MEGAENV',
            problem_id="2",
        )
        r2 = TokenRollout(
            trajectory=[1, 2, 3, 4],
            reward=0.14,
            generation_mask=[False, True, True, True],
            logprobs=[0.1, 0.2, 0.3, -1.2],
            env_id='MEGAENV',
            problem_id="2",
        )
        rollouts = [[r1, r2]]
        try:
            data_iter = rl_utils.prepare_data_for_update([model], {}, rollouts, tokenizer)
        except AssertionError as e:
            # We expect trajectories to come padded there.
            assert str(e).startswith('Rollout is not the correct length')

        r1 = TokenRollout(
            trajectory=torch.Tensor([1, 2, 3, tokenizer.eod]).cuda(),
            reward=3.14,
            generation_mask=torch.Tensor([False, True, True, True]).cuda(),
            logprobs=torch.Tensor([-0.2, -0.3, -3.2]).cuda(),
            env_id='MEGAENV',
            problem_id="2",
        )
        r2 = TokenRollout(
            trajectory=torch.Tensor([1, 2, 234, tokenizer.eod]).cuda(),
            reward=0.14,
            generation_mask=torch.Tensor([False, True, True, True]).cuda(),
            logprobs=torch.Tensor([-0.2, -0.3, -1.2]),
            env_id='MEGAENV',
            problem_id="2",
        )
        rollouts = [[r1, r2]]
        data_iter = rl_utils.prepare_data_for_update([model], {}, rollouts, tokenizer)

        _, _, old_logprobs, _, _, _, _ = next(data_iter)
        # All logits are ones in the MockModel.
        # All probabilities should be uniform.
        torch.testing.assert_close(old_logprobs.exp(), torch.ones_like(old_logprobs) / VOCAB)
    finally:
        Utils.destroy_model_parallel()


@patch('torch.distributed.get_rank', return_value=0)
def test_prepare_trajectories_with_sequence_packing(mock_rank):
    """Test prepare_trajectories with sequence packing enabled."""
    # Set up args with sequence packing
    args = arguments.parse_args(ignore_unknown_args=True)
    setattr(args, 'rl_use_sequence_packing', True)
    setattr(args, 'rl_sequence_packing_bin_size', 16)
    setattr(args, 'data_parallel_size', 1)
    setattr(args, 'micro_batch_size', 2)
    setattr(args, 'global_batch_size', 2)
    setattr(args, 'seq_length', 16)
    setattr(args, 'curr_iteration', 1)
    global_vars.unset_global_variables()
    global_vars.set_global_variables(args, build_tokenizer=False)

    tokenizer = MockTokenizer()

    # Create rollouts of varying lengths
    r1 = TokenRollout(
        trajectory=[1, 2, tokenizer.eod],
        reward=3.14,
        generation_mask=[False, True, True],
        logprobs=[0.1, 0.2, 0.3],
        env_id='MEGAENV',
        problem_id="1",
    )
    r2 = TokenRollout(
        trajectory=[4, 5, 6, 7, tokenizer.eod],
        reward=0.14,
        generation_mask=[False, True, True, True, True],
        logprobs=[0.4, 0.5, 0.6, 0.7, 0.8],
        env_id='MEGAENV',
        problem_id="2",
    )
    r3 = TokenRollout(
        trajectory=[8, 9, tokenizer.eod],
        reward=2.71,
        generation_mask=[False, True, True],
        logprobs=[0.9, 1.0, 1.1],
        env_id='MEGAENV',
        problem_id="3",
    )

    rollouts = [[r1, r2, r3]]
    seq_len = 16

    # Call prepare_trajectories with sequence packing
    trajs, genmask, inference_logprobs = rl_utils.prepare_trajectories(rollouts, tokenizer, seq_len)

    # With sequence packing enabled but called from prepare_trajectories,
    # it might still return individual sequences (not packed into bins yet)
    # because the actual packing happens later in prepare_data_for_update
    assert trajs.shape[0] == 3  # Three sequences
    assert trajs.shape[1] == seq_len

    # Verify that each sequence is properly padded
    # Sequence 1: [1, 2, eod, pad] + padding
    assert trajs[0, 0] == 1
    assert trajs[0, 1] == 2
    assert trajs[0, 2] == tokenizer.eod
    assert trajs[0, 3] == tokenizer.pad

    # Sequence 2: [4, 5, 6, 7, eod, pad] + padding
    assert trajs[1, 0] == 4
    assert trajs[1, 1] == 5
    assert trajs[1, 4] == tokenizer.eod
    assert trajs[1, 5] == tokenizer.pad
