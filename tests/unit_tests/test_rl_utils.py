# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from unittest.mock import patch

import torch

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import Float16Module
from megatron.rl import rl_utils
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


class MockTokenizer:
    def __init__(self):
        self.pad = 42
        self.eod = 43
        self.vocab_size = VOCAB
        self.bos = None

    def detokenize(self, tokens):
        return [str(tok) for tok in tokens]


def test_get_logprobs():
    """Test that getting logprobs at least does not crash."""
    # We use args inside of get_logprobs, we need to initialize them.
    args = arguments.parse_args(ignore_unknown_args=True)
    global_vars.set_args(args)

    tokens = torch.ones((BATCH, SEQ), dtype=torch.long)
    logprobs = rl_utils.get_logprobs(MockModel(), tokens, position_ids=None, attention_mask=None)
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
    logprobs = rl_utils.get_logprobs(MockModel(), tokens, position_ids=None, attention_mask=None)
    # We chop off 1 element from the sequence dimension.
    assert logprobs.shape == (BATCH, SEQ - 1)
    # As we return ones as logits, all logprobs should be the same.
    assert torch.all(logprobs == logprobs[0, 0]).item()


def test_prepare_trajectories():
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


def test_prepare_trajectories_with_packing():
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
    torch.testing.assert_close(entropy_term, torch.ones_like(ratios) * torch.e)


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
    torch.testing.assert_close(entropy_term, torch.ones_like(ratios) * torch.e)


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


@patch('megatron.rl.rl_utils.mpu')
def test_prepare_data_for_update(mock_mpu):
    """Test that getting logprobs at least does not crash."""
    mock_mpu.get_expert_data_parallel_world_size.return_value = 0
    # We use args inside of get_logprobs, we need to initialize them.

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


def test_sequence_packing_basic():
    """Test basic sequence packing functionality."""
    # Initialize args as required by SequencePacker
    args = arguments.parse_args(ignore_unknown_args=True)
    setattr(args, 'seq_length', 16)
    global_vars.set_args(args)

    tokenizer = MockTokenizer()
    bin_size = 16
    packer = rl_utils.SequencePacker(bin_size=bin_size, pad_token=tokenizer.pad)

    # Create test sequences of varying lengths, all padded to same length
    max_len = 5
    sequences = [
        torch.cat(
            [
                torch.tensor([1, 2, 3, tokenizer.eod]),
                torch.full((1,), tokenizer.pad, dtype=torch.long),
            ]
        ),  # length 4 -> 5
        torch.cat(
            [torch.tensor([4, 5, tokenizer.eod]), torch.full((2,), tokenizer.pad, dtype=torch.long)]
        ),  # length 3 -> 5
        torch.tensor([6, 7, 8, 9, tokenizer.eod]),  # length 5
        torch.cat(
            [torch.tensor([10, tokenizer.eod]), torch.full((3,), tokenizer.pad, dtype=torch.long)]
        ),  # length 2 -> 5
    ]

    generation_masks = torch.tensor(
        [
            [False, True, True, True, False],  # Matches padded length
            [False, True, True, False, False],
            [False, True, True, True, True],
            [False, True, False, False, False],
        ]
    )

    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])

    # Pack sequences
    packed_trajs, packed_position_ids, packed_attention_mask, packed_loss_mask, packing_info = (
        packer.pack_sequences(sequences, generation_masks)
    )

    # Verify packed data structure
    assert packed_trajs is not None
    assert packed_position_ids is not None
    assert packed_attention_mask is not None
    assert packed_loss_mask is not None
    assert packing_info is not None

    # Check that sequences fit in bins properly
    # The packer trims sequences to their actual length (removing padding)
    # Actual lengths: 4, 3, 5, 2 = 14 total tokens
    # With bin_size=16, this should fit in 1 bin
    assert packed_trajs.shape[0] >= 1  # At least one bin
    assert packed_trajs.shape[1] == bin_size

    # Verify position_ids are correct
    for bin_idx in range(packed_trajs.shape[0]):
        # Check that position_ids reset for each sequence in the bin
        for i in range(packed_trajs.shape[1]):
            if i == 0 or packed_trajs[bin_idx, i - 1] == tokenizer.eod:
                # Start of a new sequence
                if packed_trajs[bin_idx, i] != tokenizer.pad:
                    assert packed_position_ids[bin_idx, i] == 0


def test_sequence_packing_with_generation_masks():
    """Test sequence packing with generation masks."""
    # Initialize args as required by SequencePacker
    args = arguments.parse_args(ignore_unknown_args=True)
    setattr(args, 'seq_length', 20)
    global_vars.set_args(args)

    tokenizer = MockTokenizer()
    bin_size = 20
    packer = rl_utils.SequencePacker(bin_size=bin_size, pad_token=tokenizer.pad)

    # Create test data with generation masks
    sequences = [torch.tensor([1, 2, 3, tokenizer.eod]), torch.tensor([4, 5, 6, 7, tokenizer.eod])]

    # Pad sequences to same length for stacking
    max_len = max(len(s) for s in sequences)
    padded_sequences = []
    for seq in sequences:
        padded = torch.cat([seq, torch.full((max_len - len(seq),), tokenizer.pad, dtype=seq.dtype)])
        padded_sequences.append(padded)

    generation_masks = torch.tensor(
        [
            [False, True, True, True, False],  # Padded to match max_len
            [False, True, True, True, True],
        ]
    )

    # Pack sequences
    packed_trajs, packed_position_ids, packed_attention_mask, packed_loss_mask, packing_info = (
        packer.pack_sequences(padded_sequences, generation_masks)
    )

    # Verify packed tensors
    assert packed_trajs.shape[0] == 1  # One bin
    assert packed_trajs.shape[1] == bin_size

    # Check that loss mask is set correctly for generation tokens
    # The loss mask should be 1 for generation tokens and 0 for padding/prompt


def test_sequence_packing_empty_bins():
    """Test that empty bins are created correctly."""
    # Initialize args if needed
    args = arguments.parse_args(ignore_unknown_args=True)
    setattr(args, 'seq_length', 8)
    global_vars.set_args(args)

    tokenizer = MockTokenizer()
    bin_size = 8
    num_empty_bins = 3

    # Create a simple packed data structure
    packed_trajs = torch.tensor(
        [[1, 2, 3, tokenizer.eod, tokenizer.pad, tokenizer.pad, tokenizer.pad, tokenizer.pad]]
    )
    packed_position_ids = torch.tensor([[0, 1, 2, 3, 0, 0, 0, 0]])
    packed_loss_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.float)
    packed_attention_mask = torch.ones(1, bin_size, bin_size)  # Simple full attention mask

    # Create empty bins
    empty_trajs, empty_position_ids, empty_loss_mask, empty_attention_mask, empty_packing_info = (
        rl_utils.create_empty_bins(
            num_empty_bins=num_empty_bins,
            bin_size=bin_size,
            packed_trajs=packed_trajs,
            packed_position_ids=packed_position_ids,
            packed_loss_mask=packed_loss_mask,
            packed_attention_mask=packed_attention_mask,
            tokenizer=tokenizer,
        )
    )

    # Verify shapes
    assert empty_trajs.shape[0] == num_empty_bins
    assert empty_trajs.shape[1] == bin_size

    # Check that empty bins are filled with padding
    for i in range(num_empty_bins):
        assert torch.all(empty_trajs[i] == tokenizer.pad)
        assert torch.all(empty_position_ids[i] == 0)
        assert torch.all(empty_loss_mask[i] == 0)

    # Verify packing info for empty bins
    assert len(empty_packing_info) == num_empty_bins
    for info in empty_packing_info:
        assert len(info['bin_seq_indices']) == 0  # No sequences in empty bins
        assert len(info['seq_starts']) == 0  # No sequence starts


def test_prepare_trajectories_with_sequence_packing():
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


def test_sequence_packing_integration():
    """Simple integration test for sequence packing - just verifies the packing works."""
    # Initialize minimal args needed for SequencePacker
    args = arguments.parse_args(ignore_unknown_args=True)
    setattr(args, 'seq_length', 16)
    global_vars.set_args(args)

    tokenizer = MockTokenizer()
    bin_size = 16

    # Test that we can pack sequences and get expected outputs
    packer = rl_utils.SequencePacker(bin_size=bin_size, pad_token=tokenizer.pad)

    # Create test data - need to pad to same length for stacking
    max_len = 5
    sequences = [
        torch.cat(
            [
                torch.tensor([1, 2, 3, tokenizer.eod]),
                torch.full((1,), tokenizer.pad, dtype=torch.long),
            ]
        ),  # length 4 -> 5
        torch.cat(
            [torch.tensor([4, 5, tokenizer.eod]), torch.full((2,), tokenizer.pad, dtype=torch.long)]
        ),  # length 3 -> 5
        torch.tensor([6, 7, 8, 9, tokenizer.eod]),  # length 5
    ]
    generation_masks = [
        torch.tensor([False, True, True, True, False]),
        torch.tensor([False, True, True, False, False]),
        torch.tensor([False, True, True, True, True]),
    ]

    # Pack sequences
    packed_trajs, packed_position_ids, packed_attention_mask, packed_loss_mask, packing_info = (
        packer.pack_sequences(sequences, generation_masks)
    )

    # Basic assertions
    assert packed_trajs is not None
    assert packed_trajs.shape[1] == bin_size  # Each bin should be bin_size
    assert packed_position_ids.shape == packed_trajs.shape
    assert packed_loss_mask.shape == packed_trajs.shape

    # Verify the sequences are packed correctly
    # Total length: 4 + 3 + 5 = 12, should fit in 1 bin
    assert packed_trajs.shape[0] == 1

    # The packer sorts sequences by length (descending), so order is: seq3 (len 5), seq1 (len 4), seq2 (len 3)
    expected_start = torch.tensor(
        [6, 7, 8, 9, tokenizer.eod, 1, 2, 3, tokenizer.eod, 4, 5, tokenizer.eod]
    )
    assert torch.all(packed_trajs[0, :12] == expected_start)

    # Rest should be padding
    assert torch.all(packed_trajs[0, 12:] == tokenizer.pad)
