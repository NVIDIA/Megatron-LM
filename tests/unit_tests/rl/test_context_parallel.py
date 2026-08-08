# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core import mpu
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from megatron.rl import rl_utils
from megatron.rl.sequence_packing_utils import SequencePacker, create_packed_seq_params_for_bin
from megatron.training.arguments import parse_args, validate_args
from megatron.training.global_vars import destroy_global_vars, set_global_variables
from tests.unit_tests.test_utilities import Utils

VOCAB = 128
MAX_SEQUENCES_PER_BIN = 4

pytestmark = pytest.mark.skipif(
    rl_utils.tex is None, reason="CP logprobs require Transformer Engine THD indices"
)

CP_SIZES = [pytest.param(cp, id=f"cp{cp}") for cp in (2, 4, 8) if cp <= Utils.world_size]


class PositionWiseModel:
    """Deterministic model whose logits at each index depend only on that index's
    token and position id (tables are identical on every rank)."""

    def __init__(self, max_position: int, device: torch.device):
        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.config = TransformerConfig(
            num_attention_heads=8, num_layers=8, pipeline_dtype=torch.bfloat16
        )
        generator = torch.Generator().manual_seed(1234)
        self.token_logits = torch.randn(VOCAB, VOCAB, generator=generator).to(device)
        self.position_logits = torch.randn(max_position, VOCAB, generator=generator).to(device)

    def __call__(self, tokens, position_ids, attention_mask, **kwargs):
        return self.token_logits[tokens] + self.position_logits[position_ids]


def _set_minimal_args():
    destroy_global_vars()
    destroy_num_microbatches_calculator()
    args = parse_args(ignore_unknown_args=True)
    args.num_layers = 8
    args.num_attention_heads = 8
    args.vocab_size = VOCAB
    args.hidden_size = 128
    args.max_position_embeddings = 256
    args.seq_length = 256
    args.micro_batch_size = 1
    args.wandb_project = None
    args = validate_args(args)
    set_global_variables(args, False)


@pytest.fixture
def initialize_context_parallel(request, monkeypatch):
    """Initialize/destroy model parallel with the requested CP size (tp = pp = 1)."""
    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("LOG_TO_WANDB", "false")
    Utils.initialize_model_parallel(context_parallel_size=request.param)
    _set_minimal_args()
    yield request.param
    Utils.destroy_model_parallel()
    destroy_global_vars()
    destroy_num_microbatches_calculator()


def _pack_bin(lengths, seq_length_multiple, bin_size, device):
    """Pack variable-length sequences into one bin and build its PackedSeqParams."""
    generator = torch.Generator().manual_seed(7)
    seqs = torch.zeros(len(lengths), max(lengths), dtype=torch.long)  # 0 is the pad token
    for i, length in enumerate(lengths):
        seqs[i, :length] = torch.randint(1, VOCAB, (length,), generator=generator)
    packer = SequencePacker(
        bin_size=bin_size,
        pad_token=0,
        max_sequences_per_bin=MAX_SEQUENCES_PER_BIN,
        seq_length_multiple=seq_length_multiple,
    )
    packed_trajs, packed_position_ids, _, packing_info = packer.pack_sequences(seqs)
    assert packed_trajs.shape[0] == 1, "test data must fit a single bin"
    packed_seq_params = create_packed_seq_params_for_bin(
        packing_info=packing_info,
        bin_idx=0,
        bin_size=bin_size,
        max_sequences_per_bin=MAX_SEQUENCES_PER_BIN,
        device=device,
        seq_length_multiple=seq_length_multiple,
    )
    return packed_trajs[:1].to(device), packed_position_ids[:1].to(device), packed_seq_params


class TestContextParallelLogprobs:

    @pytest.mark.parametrize("initialize_context_parallel", CP_SIZES, indirect=True)
    @pytest.mark.parametrize(
        "packed", [pytest.param(False, id="single_sequence"), pytest.param(True, id="packed_bin")]
    )
    @pytest.mark.parametrize(
        "no_grad", [pytest.param(True, id="no_grad"), pytest.param(False, id="grad")]
    )
    def test_cp_logprobs_match_single_rank_reference(
        self, initialize_context_parallel, packed, no_grad
    ):
        """get_logprobs under CP must return the exact cp_size == 1 result on every rank."""
        cp_size = initialize_context_parallel
        slot = 2 * cp_size  # TE splits every padded slot into 2*cp_size chunks
        device = torch.device("cuda")
        if packed:
            # Footprints of 3/2/1 slots: pad gaps after each sequence and a
            # trailing ghost slot of 2 more slots at the end of the bin.
            bin_size, lengths = 8 * slot, [2 * slot + 1, slot + 3, slot - 2]
        else:
            bin_size, lengths = 4 * slot, [4 * slot]
        tokens, position_ids, packed_seq_params = _pack_bin(lengths, slot, bin_size, device)
        # The packer must reserve every slot (incl. ghosts) on the 2*cp_size grid.
        boundaries = packed_seq_params.cu_seqlens_q_padded
        assert boundaries[-1].item() == bin_size
        assert torch.all((boundaries[1:] - boundaries[:-1]) % slot == 0)

        model = PositionWiseModel(max_position=bin_size, device=device)
        if not no_grad:
            model.token_logits.requires_grad_()

        logprobs = rl_utils.get_logprobs(
            model,
            tokens,
            position_ids,
            no_grad=no_grad,
            sequence_packing=True,
            packed_seq_params=packed_seq_params,
        )

        with torch.no_grad():
            full_logits = model(tokens, position_ids, None)
            reference = rl_utils.selective_log_softmax(full_logits[:, :-1, :], tokens[:, 1:])
        # The tables are identical on every rank, so matching the local reference
        # on each rank also means all CP ranks returned the same logprobs.
        torch.testing.assert_close(logprobs, reference)
        # The CP scatter must annotate a copy, not the caller's params.
        assert packed_seq_params.cp_group is None
        assert packed_seq_params.local_cp_size is None

        # A second call must hit the memoized per-bin partition and return the same reference.
        cp_scatter = packed_seq_params._rl_cp_scatter
        logprobs_again = rl_utils.get_logprobs(
            model,
            tokens,
            position_ids,
            no_grad=True,
            sequence_packing=True,
            packed_seq_params=packed_seq_params,
        )
        assert packed_seq_params._rl_cp_scatter is cp_scatter
        torch.testing.assert_close(logprobs_again, reference)

        if not no_grad:
            # Gradients must flow back through the differentiable logprob all-gather.
            logprobs.sum().backward()
            grad = model.token_logits.grad
            assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0

    @pytest.mark.parametrize("initialize_context_parallel", CP_SIZES[:1], indirect=True)
    def test_scatter_rejects_indivisible_seq_len(self, initialize_context_parallel):
        """Unaligned inputs (bypassing the packer) must be rejected, not mis-partitioned."""
        cp_size = initialize_context_parallel
        seq_len = 2 * cp_size + 1
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
        packed_seq_params = PackedSeqParams(
            qkv_format='thd',
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=seq_len,
            max_seqlen_kv=seq_len,
            total_tokens=seq_len,
        )
        tokens = torch.zeros(1, seq_len, dtype=torch.long, device="cuda")
        with pytest.raises(AssertionError, match="divisible"):
            rl_utils._scatter_for_context_parallel(
                tokens, tokens.clone(), packed_seq_params, mpu.get_context_parallel_group()
            )
