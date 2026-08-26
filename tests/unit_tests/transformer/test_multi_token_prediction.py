# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import itertools
import os
import sys
import types

import pytest
import torch
from torch import Tensor

from megatron.core.enums import ModelType
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_data_parallel_group,
    get_tensor_and_context_parallel_group,
    get_tensor_model_parallel_group,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import checkpoint as tensor_parallel_checkpoint
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.hyper_connection import learned_output_contract
from megatron.core.transformer.multi_token_prediction import (
    ContiguousPackedCPRollContext,
    ContiguousPackedCPRollHalos,
    ContiguousPackedSeqRollPlan,
    MTPLossLoggingHelper,
    MultiTokenPredictionBlock,
    _initialize_hidden_state_mixing_rng_tracker,
    _mix_hidden_state_history,
    _mtp_logits_are_vocab_sharded,
    prepare_mtp_sequence_roll_context,
    process_mtp_loss,
    roll_tensor,
)
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import (
    get_batch_on_this_cp_rank,
    get_pg_rank,
    get_pg_size,
    is_te_min_version,
    unwrap_model,
)
from megatron.training.argument_utils import gpt_config_from_args, hybrid_config_from_args
from megatron.training.arguments import core_transformer_config_from_args, parse_args, validate_args
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.training.global_vars import (
    destroy_global_vars,
    get_args,
    set_args,
    set_global_variables,
)
from megatron.training.training import get_model, setup_model_and_optimizer
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import TEColumnParallelGroupedLinear
else:
    TEColumnParallelGroupedLinear = None

_SEED = 42


def _randint_stub(expected_size, make_result):
    """Build a ``torch.randint`` replacement that pins the shape HSM asks for.

    Hidden-state mixing draws one index per (owner, position, batch), so the requested shape
    is where its sequence-parallel and context-parallel owner arithmetic shows up. A
    stub that ignored ``size`` would keep returning a valid-looking selection even if
    that arithmetic changed, so the shape is asserted rather than assumed.
    """

    def fake_randint(high, size, device=None):
        requested, expected = tuple(size), tuple(expected_size)
        assert requested == expected, f"HSM requested index shape {requested}, expected {expected}"
        return make_result(high, size, device)

    return fake_randint


class TestMultiTokenPredictionLayer:
    def setup_method(self, method):
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
        MTPLossLoggingHelper.tracker = {}

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        MTPLossLoggingHelper.tracker = {}

    def _create_config_and_mtp_block_spec(self, tp, cp, use_te=False):
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)
        config = TransformerConfig(
            mtp_num_layers=2,
            num_layers=4,
            hidden_size=64,
            num_attention_heads=8,
            use_cpu_initialization=True,
            tensor_model_parallel_size=tp,
            sequence_parallel=True if tp > 1 else False,
            context_parallel_size=cp,  # Enable CP for MTP testing
        )
        if use_te:
            transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec()
        else:
            transformer_layer_spec = get_gpt_layer_local_spec()
        mtp_block_spec = get_gpt_mtp_block_spec(
            config=config, spec=transformer_layer_spec, use_transformer_engine=use_te
        )
        return config, mtp_block_spec

    def test_hsm_mix_preserves_dtype_and_gradients(self, monkeypatch):
        """HSM selects per-element history entries without changing dtype or gradients."""
        first = torch.full((2, 1, 3), 1.0, dtype=torch.bfloat16, requires_grad=True)
        second = torch.full((2, 1, 3), 2.0, dtype=torch.bfloat16, requires_grad=True)
        selection = torch.tensor([[[[0]], [[1]]]], dtype=torch.long)

        monkeypatch.setattr(
            torch, "randint", _randint_stub((1, 2, 1, 1), lambda high, size, device: selection)
        )
        mixed = _mix_hidden_state_history(first.unsqueeze(0), second)

        assert mixed.dtype == torch.bfloat16
        torch.testing.assert_close(
            mixed, torch.tensor([[[1, 1, 1]], [[2, 2, 2]]], dtype=torch.bfloat16)
        )
        mixed.sum().backward()
        torch.testing.assert_close(
            first.grad, torch.tensor([[[1, 1, 1]], [[0, 0, 0]]], dtype=torch.bfloat16)
        )
        torch.testing.assert_close(
            second.grad, torch.tensor([[[0, 0, 0]], [[1, 1, 1]]], dtype=torch.bfloat16)
        )

    def test_hsm_mix_uses_newest_state_for_zero_roll_sentinel(self, monkeypatch):
        """A sampled zero produced by an invalid roll selects the newest hidden state."""
        first = torch.tensor([1.0, 0.0, 3.0]).view(3, 1, 1).requires_grad_()
        newest = torch.tensor([11.0, 12.0, 13.0]).view(3, 1, 1).requires_grad_()
        selection = torch.zeros((1, 3, 1, 1), dtype=torch.long)
        monkeypatch.setattr(
            torch, "randint", _randint_stub((1, 3, 1, 1), lambda high, size, device: selection)
        )

        mixed = _mix_hidden_state_history(first.unsqueeze(0), newest)

        torch.testing.assert_close(mixed, torch.tensor([1.0, 12.0, 3.0]).view(3, 1, 1))
        mixed.sum().backward()
        torch.testing.assert_close(first.grad, torch.tensor([1.0, 0.0, 1.0]).view(3, 1, 1))
        torch.testing.assert_close(newest.grad, torch.tensor([0.0, 1.0, 0.0]).view(3, 1, 1))

    def test_hsm_rng_matches_across_replicated_tensor_parallel_ranks(self):
        """Replicated tokens select the same HSM history on every TP rank."""
        tp = 2
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp)
        model_parallel_cuda_manual_seed(_SEED, force_reset_rng=True)
        tp_group = get_tensor_model_parallel_group()

        history = [
            torch.full((32, 2, 4), value, dtype=torch.float32, device="cuda")
            for value in (1.0, 2.0, 3.0)
        ]
        mixed = _mix_hidden_state_history(torch.stack(history[:-1]), history[-1])
        gathered = [torch.empty_like(mixed) for _ in range(tp)]
        torch.distributed.all_gather(gathered, mixed, group=tp_group)

        for peer_mixed in gathered[1:]:
            torch.testing.assert_close(peer_mixed, gathered[0])
        assert torch.unique(gathered[0][..., 0]).numel() > 1

    def test_hsm_rng_differs_across_sequence_parallel_owners(self):
        """Disjoint TP+CP token shards receive different tracked RNG slices."""
        tp = 2
        cp = 2
        if int(os.environ.get("WORLD_SIZE", "1")) < tp * cp:
            pytest.skip(f"TP={tp}, CP={cp} requires at least {tp * cp} ranks")
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)
        model_parallel_cuda_manual_seed(_SEED, force_reset_rng=True)
        tp_group = get_tensor_model_parallel_group()
        cp_group = get_context_parallel_group()
        tp_cp_group = get_tensor_and_context_parallel_group()

        history = [
            torch.full((64, 2, 4), value, dtype=torch.float32, device="cuda")
            for value in range(1, 8)
        ]
        mixed = _mix_hidden_state_history(
            torch.stack(history[:-1]),
            history[-1],
            sequence_parallel=True,
            tp_group=tp_group,
            cp_group=cp_group,
        )
        selections = mixed[..., 0].to(torch.long)
        gathered = [torch.empty_like(selections) for _ in range(tp * cp)]
        torch.distributed.all_gather(gathered, selections, group=tp_cp_group)

        assert len({tuple(mask.cpu().flatten().tolist()) for mask in gathered}) == tp * cp
        assert all(torch.unique(mask).numel() > 1 for mask in gathered)

    def test_hsm_rng_differs_across_data_parallel_replicas(self, monkeypatch):
        """DP replicas draw disjoint slices while each replica keeps TP/CP consistency."""
        tp, cp = 2, 2
        if int(os.environ.get("WORLD_SIZE", "1")) < 2 * tp * cp:
            pytest.skip("This test requires two TP=2, CP=2 data-parallel replicas")
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)
        model_parallel_cuda_manual_seed(_SEED, force_reset_rng=True)
        tp_group = get_tensor_model_parallel_group()
        cp_group = get_context_parallel_group()
        dp_group = get_data_parallel_group()
        rng_tracker_name = _initialize_hidden_state_mixing_rng_tracker(dp_group)
        assert rng_tracker_name is not None
        history = [
            torch.full((32, 2, 4), value, dtype=torch.float32, device="cuda")
            for value in range(1, 8)
        ]

        real_randint = torch.randint
        monkeypatch.setattr(
            torch,
            "randint",
            _randint_stub(
                (1, tp * cp * 32, 2, 1),
                lambda high, size, device: real_randint(high, size, device=device),
            ),
        )
        mixed = _mix_hidden_state_history(
            torch.stack(history[:-1]),
            history[-1],
            sequence_parallel=True,
            tp_group=tp_group,
            cp_group=cp_group,
            rng_tracker_name=rng_tracker_name,
        )
        gathered = [torch.empty_like(mixed) for _ in range(get_pg_size(dp_group))]
        torch.distributed.all_gather(gathered, mixed, group=dp_group)

        assert not torch.equal(gathered[0], gathered[1])

    def test_hsm_rng_replays_during_activation_checkpointing(self):
        """MCore checkpoint recomputation reuses the original HSM selection."""
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)
        model_parallel_cuda_manual_seed(_SEED, force_reset_rng=True)
        tp_group = get_tensor_model_parallel_group()
        history = [
            torch.full((32, 2, 4), value, dtype=torch.float32, device="cuda", requires_grad=True)
            for value in (1.0, 2.0, 3.0)
        ]
        selections = []

        def mix_with_recording(*states):
            mixed = _mix_hidden_state_history(
                torch.stack(states[:-1]), states[-1], sequence_parallel=True, tp_group=tp_group
            )
            selections.append(mixed.detach().clone())
            return mixed

        mixed = tensor_parallel_checkpoint(mix_with_recording, False, *history)
        mixed.sum().backward()

        assert len(selections) == 2
        torch.testing.assert_close(selections[1], selections[0])
        assert all(state.grad is not None for state in history)

    def test_mtp_hsm_defaults_off(self):
        """HSM remains opt-in."""
        config = TransformerConfig(num_layers=2, hidden_size=4, num_attention_heads=1)
        assert config.mtp_hsm is False

    @pytest.mark.parametrize("mtp_num_layers", [None, 0, 1])
    def test_mtp_hsm_requires_multiple_layers(self, mtp_num_layers):
        """TransformerConfig rejects HSM when there is no history to mix."""
        with pytest.raises(ValueError, match="mtp_hsm=True requires mtp_num_layers >= 2"):
            TransformerConfig(
                num_layers=2,
                hidden_size=4,
                num_attention_heads=1,
                mtp_num_layers=mtp_num_layers,
                mtp_hsm=True,
            )

    @pytest.mark.parametrize(
        ("training", "expected_second_input"), [(True, [1.0, 2.0]), (False, [2.0, 2.0])]
    )
    def test_hsm_runs_only_during_training(self, monkeypatch, training, expected_second_input):
        """The block applies HSM in training and leaves evaluation deterministic."""
        config = TransformerConfig(
            num_layers=2, hidden_size=4, num_attention_heads=1, mtp_num_layers=2, mtp_hsm=True
        )
        seen_inputs = []

        class _IncrementLayer:
            def __call__(self, hidden_states, input_ids, position_ids, padding_mask, **kwargs):
                seen_inputs.append(hidden_states.clone())
                return hidden_states + 1, input_ids, position_ids, padding_mask

        block = types.SimpleNamespace(
            config=config,
            training=training,
            vp_stage=None,
            mtp_use_repeated_layer=False,
            cp_group=None,
            tp_group=None,
            hidden_state_mixing_rng_tracker_name=None,
            sequence_parallel=False,
            layers=[_IncrementLayer(), _IncrementLayer()],
        )
        monkeypatch.setattr(
            torch,
            "randint",
            _randint_stub(
                (1, 2, 1, 1),
                lambda high, size, device: torch.zeros(size, dtype=torch.long, device=device),
            ),
        )

        hidden_states = torch.ones(2, 1, config.hidden_size)
        MultiTokenPredictionBlock.forward(
            block,
            input_ids=torch.zeros(1, 2, dtype=torch.long),
            position_ids=torch.zeros(1, 2, dtype=torch.long),
            hidden_states=hidden_states,
            attention_mask=torch.ones(1, 1, 2, 2, dtype=torch.bool),
        )

        torch.testing.assert_close(seen_inputs[0], hidden_states)
        torch.testing.assert_close(
            seen_inputs[1],
            torch.tensor(expected_second_input).view(2, 1, 1).expand_as(hidden_states),
        )

    @pytest.mark.parametrize("packed", [False, True])
    def test_hsm_aligns_history_with_target_tokens(self, monkeypatch, packed):
        """Each reuse of an older HSM state advances it by one target token."""
        config = TransformerConfig(
            num_layers=3, hidden_size=1, num_attention_heads=1, mtp_num_layers=3, mtp_hsm=True
        )
        seen_inputs = []

        class _IncrementLayer:
            def __call__(self, hidden_states, input_ids, position_ids, padding_mask, **kwargs):
                seen_inputs.append(hidden_states.clone())
                return hidden_states + 10, input_ids, position_ids, padding_mask

        block = types.SimpleNamespace(
            config=config,
            training=True,
            vp_stage=None,
            mtp_use_repeated_layer=False,
            cp_group=None,
            tp_group=None,
            hidden_state_mixing_rng_tracker_name=None,
            sequence_parallel=False,
            layers=[_IncrementLayer(), _IncrementLayer(), _IncrementLayer()],
        )
        monkeypatch.setattr(
            torch,
            "randint",
            _randint_stub(
                (1, 5, 1, 1),
                lambda high, size, device: torch.zeros(size, dtype=torch.long, device=device),
            ),
        )

        packed_seq_params = None
        if packed:
            cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)
            packed_seq_params = PackedSeqParams(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                max_seqlen_q=3,
                max_seqlen_kv=3,
                qkv_format='thd',
            )

        hidden_states = torch.arange(5, dtype=torch.float32).view(5, 1, 1)
        MultiTokenPredictionBlock.forward(
            block,
            input_ids=torch.zeros(1, 5, dtype=torch.long),
            position_ids=torch.zeros(1, 5, dtype=torch.long),
            hidden_states=hidden_states,
            attention_mask=torch.ones(1, 1, 5, 5, dtype=torch.bool),
            packed_seq_params=packed_seq_params,
        )

        expected_once = [1, 2, 12, 4, 14] if packed else [1, 2, 3, 4, 14]
        expected_twice = [2, 12, 22, 14, 24] if packed else [2, 3, 4, 14, 24]
        torch.testing.assert_close(seen_inputs[0], hidden_states)
        torch.testing.assert_close(
            seen_inputs[1], torch.tensor(expected_once, dtype=torch.float32).view(5, 1, 1)
        )
        torch.testing.assert_close(
            seen_inputs[2], torch.tensor(expected_twice, dtype=torch.float32).view(5, 1, 1)
        )

    def test_mtp_detach_heads_config(self):
        """Test that mtp_detach_heads config defaults to False."""
        config = TransformerConfig(
            num_layers=4, hidden_size=64, num_attention_heads=8, use_cpu_initialization=True
        )
        assert config.mtp_detach_heads is False

    def test_constructor_with_detach_heads(self):
        """Test construction of MTP module with mtp_detach_heads=True."""
        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
        config = TransformerConfig(
            mtp_num_layers=2,
            num_layers=4,
            hidden_size=64,
            num_attention_heads=8,
            use_cpu_initialization=True,
            mtp_detach_heads=True,
        )
        transformer_layer_spec = get_gpt_layer_local_spec()
        mtp_block_spec = get_gpt_mtp_block_spec(
            config=config, spec=transformer_layer_spec, use_transformer_engine=False
        )
        mtp = MultiTokenPredictionBlock(config=config, spec=mtp_block_spec)

        assert isinstance(mtp, MultiTokenPredictionBlock)
        assert mtp.config.mtp_detach_heads is True

        # Verify all parameters are tagged for separate MTP grad-norm handling.
        for name, param in mtp.named_parameters():
            assert (
                getattr(param, 'grad_norm_group', None) == 'mtp'
            ), f"Parameter {name} missing grad_norm_group attribute"

    @pytest.mark.parametrize(('tp'), [(1), (2), (4)])
    def test_constructor_local(self, tp):
        """Test basic construction of MTP module."""

        torch.manual_seed(_SEED)
        config, mtp_block_spec = self._create_config_and_mtp_block_spec(tp, cp=1)
        mtp = MultiTokenPredictionBlock(config=config, spec=mtp_block_spec)

        assert isinstance(mtp, MultiTokenPredictionBlock)
        assert len(mtp.layers) == config.mtp_num_layers
        for i in range(config.mtp_num_layers):
            assert mtp.layers[i].layer_number == i + 1
            assert mtp.layers[i].enorm.weight.shape[0] == config.hidden_size
            assert mtp.layers[i].hnorm.weight.shape[0] == config.hidden_size
            assert mtp.layers[i].eh_proj.weight.shape[0] == config.hidden_size / tp
            assert mtp.layers[i].eh_proj.weight.shape[1] == config.hidden_size * 2
            assert mtp.layers[i].mtp_model_layer is not None
        num_weights = sum([p.numel() for p in mtp.parameters()])
        if tp == 1:
            assert num_weights == 58560 * config.mtp_num_layers
        elif tp == 2:
            assert num_weights == 29664 * config.mtp_num_layers
        elif tp == 4:
            assert num_weights == 15216 * config.mtp_num_layers

    @pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
    @pytest.mark.parametrize(('tp', 'cp'), [(1, 1), (2, 1), (2, 2)])
    def test_constructor_ues_te(self, tp, cp):
        """Test basic construction of MTP module."""
        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)
        config, mtp_block_spec = self._create_config_and_mtp_block_spec(tp, cp, use_te=True)
        mtp = MultiTokenPredictionBlock(config=config, spec=mtp_block_spec)

        assert isinstance(mtp, MultiTokenPredictionBlock)
        assert len(mtp.layers) == config.mtp_num_layers
        for i in range(config.mtp_num_layers):
            assert mtp.layers[i].layer_number == i + 1
            assert mtp.layers[i].enorm.weight.shape[0] == config.hidden_size
            assert mtp.layers[i].hnorm.weight.shape[0] == config.hidden_size
            assert mtp.layers[i].eh_proj.weight.shape[0] == config.hidden_size / tp
            assert mtp.layers[i].eh_proj.weight.shape[1] == config.hidden_size * 2
            assert mtp.layers[i].mtp_model_layer is not None
        num_weights = sum([p.numel() for p in mtp.parameters()])
        if tp == 1:
            assert num_weights == 58560 * config.mtp_num_layers
        elif tp == 2:
            assert num_weights == 29664 * config.mtp_num_layers
        elif tp == 4:
            assert num_weights == 15216 * config.mtp_num_layers

    def test_get_embeddings_rolls_padding_mask(self):
        """Test that _get_embeddings rolls padding_mask alongside input ids.

        padding_mask uses the router convention: True = padded, False = valid.
        Boundary positions are filled with True (padded) via the field's fill value.
        """
        torch.manual_seed(_SEED)
        config, mtp_block_spec = self._create_config_and_mtp_block_spec(tp=1, cp=1)
        mtp = MultiTokenPredictionBlock(config=config, spec=mtp_block_spec)
        mtp_layer = mtp.layers[0]

        seq_len = 6
        batch_size = 2
        input_ids = torch.tensor([[1, 2, 3, 4, 0, 0], [5, 6, 7, 0, 0, 0]], dtype=torch.int64)
        position_ids = torch.arange(seq_len, dtype=torch.int64).repeat(batch_size, 1)
        padding_mask = torch.tensor(
            [[False, False, False, False, True, True], [False, False, False, True, True, True]]
        )
        hidden_states = torch.randn(seq_len, batch_size, config.hidden_size)

        def fake_embedding(input_ids, position_ids):
            return torch.zeros(seq_len, batch_size, config.hidden_size, dtype=hidden_states.dtype)

        rolled_input_ids, rolled_position_ids, rolled_padding_mask, _, _ = (
            mtp_layer._get_embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                padding_mask=padding_mask,
                embedding=fake_embedding,
                hidden_states=hidden_states,
                packed_seq_params=None,
            )
        )

        expected_input_ids, expected_position_ids, expected_padding_mask = roll_tensor(
            [input_ids, position_ids, padding_mask], shifts=-1, dims=-1, fill_values=[0, 0, True]
        )

        assert torch.equal(rolled_input_ids, expected_input_ids)
        assert torch.equal(rolled_position_ids, expected_position_ids)
        assert torch.equal(rolled_padding_mask, expected_padding_mask)

    def test_forward_propagates_rolled_padding_mask(self, monkeypatch):
        """Test forward passes rolled padding_mask to transformer path.

        padding_mask uses the router convention: True = padded, False = valid.
        Boundary positions are filled with True (padded) via the field's fill value.
        """
        torch.manual_seed(_SEED)
        config, mtp_block_spec = self._create_config_and_mtp_block_spec(tp=1, cp=1)
        mtp = MultiTokenPredictionBlock(config=config, spec=mtp_block_spec)
        mtp_layer = mtp.layers[0]

        seq_len = 4
        batch_size = 2
        input_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.int64)
        position_ids = torch.arange(seq_len, dtype=torch.int64).repeat(batch_size, 1)
        padding_mask = torch.tensor([[False, False, False, True], [False, False, True, True]])
        hidden_states = torch.randn(seq_len, batch_size, config.hidden_size)
        attention_mask = torch.ones((batch_size, 1, seq_len, seq_len), dtype=torch.bool)
        seen = {}

        def fake_embedding(input_ids, position_ids):
            return torch.zeros(seq_len, batch_size, config.hidden_size, dtype=hidden_states.dtype)

        def fake_proj_and_transformer_layer(
            self,
            hidden_states,
            decoder_input,
            input_ids=None,
            attention_mask=None,
            padding_mask=None,
            context=None,
            context_mask=None,
            rotary_pos_emb=None,
            rotary_pos_cos=None,
            rotary_pos_sin=None,
            attention_bias=None,
            inference_params=None,
            packed_seq_params=None,
            sequence_len_offset=None,
        ):
            seen["padding_mask"] = padding_mask
            return hidden_states

        monkeypatch.setattr(
            mtp_layer,
            "_proj_and_transformer_layer",
            types.MethodType(fake_proj_and_transformer_layer, mtp_layer),
        )

        _, _, _, returned_padding_mask = mtp_layer.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            embedding=fake_embedding,
        )

        expected_padding_mask = roll_tensor([padding_mask], shifts=-1, dims=-1, fill_values=[True])[
            0
        ]
        assert torch.equal(seen["padding_mask"], expected_padding_mask)
        assert torch.equal(returned_padding_mask, expected_padding_mask)

    def test_get_embeddings_skips_position_roll_without_absolute_embedding(self):
        """RoPE embeddings ignore position_ids, so MTP only rolls the token IDs."""
        torch.manual_seed(_SEED)
        config, mtp_block_spec = self._create_config_and_mtp_block_spec(tp=1, cp=1)
        mtp = MultiTokenPredictionBlock(config=config, spec=mtp_block_spec)
        mtp_layer = mtp.layers[0]

        seq_len = 6
        batch_size = 2
        input_ids = torch.arange(batch_size * seq_len, dtype=torch.int64).view(batch_size, seq_len)
        position_ids = torch.arange(seq_len, dtype=torch.int64).repeat(batch_size, 1)
        hidden_states = torch.randn(seq_len, batch_size, config.hidden_size)

        class FakeRotaryEmbedding:
            add_position_embedding = False

            def __init__(self):
                self.seen_position_ids = None

            def __call__(self, input_ids, position_ids):
                self.seen_position_ids = position_ids
                return torch.zeros(
                    seq_len, batch_size, config.hidden_size, dtype=hidden_states.dtype
                )

        embedding = FakeRotaryEmbedding()
        rolled_input_ids, returned_position_ids, _, _, _ = mtp_layer._get_embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            embedding=embedding,
            hidden_states=hidden_states,
        )

        expected_input_ids = roll_tensor([input_ids], shifts=-1, dims=-1)[0]
        assert torch.equal(rolled_input_ids, expected_input_ids)
        assert returned_position_ids is position_ids
        assert embedding.seen_position_ids is position_ids

    def test_get_embeddings_detaches_decoder_input(self):
        """With mtp_detach_heads=True, _get_embeddings detaches decoder_input (severing
        gradient flow to the shared embedding) while still returning a hidden_states
        tensor that requires grad so MTP layer params and activation checkpointing work."""
        torch.manual_seed(_SEED)
        config, mtp_block_spec = self._create_config_and_mtp_block_spec(tp=1, cp=1)
        config.mtp_detach_heads = True
        mtp = MultiTokenPredictionBlock(config=config, spec=mtp_block_spec)
        mtp_layer = mtp.layers[0]

        seq_len = 4
        batch_size = 2
        input_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.int64)
        position_ids = torch.arange(seq_len, dtype=torch.int64).repeat(batch_size, 1)
        # hidden_states arrives without requires_grad (it is detached upstream by the block).
        hidden_states = torch.randn(seq_len, batch_size, config.hidden_size)
        emb_weight = torch.nn.Parameter(torch.randn(seq_len, batch_size, config.hidden_size))

        def fake_embedding(input_ids, position_ids):
            return emb_weight.clone()

        _, _, _, decoder_input, returned_hidden_states = mtp_layer._get_embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            embedding=fake_embedding,
            hidden_states=hidden_states,
            packed_seq_params=None,
        )

        # decoder_input is detached from the embedding graph.
        assert decoder_input.requires_grad is False
        assert decoder_input.grad_fn is None
        # hidden_states is still marked requires_grad so checkpointing and the MTP
        # layer parameters keep a differentiable path.
        assert returned_hidden_states.requires_grad is True

    @pytest.mark.parametrize(
        ("embedding_scatters", "expect_scatter"), [(False, True), (True, False), (None, False)]
    )
    def test_get_embeddings_scatters_when_embedding_does_not(
        self, monkeypatch, embedding_scatters, expect_scatter
    ):
        """Multimodal language models build LanguageModelEmbedding with
        scatter_to_sequence_parallel=False so media features can be inserted into a
        full-length embedding, and their outer forward performs the sequence-parallel
        scatter afterwards. _get_embeddings calls the embedding directly, so it has to
        apply the same scatter; otherwise decoder_input stays full length while the
        backbone hidden_states arrive sequence-parallel sharded and _concat_embeddings
        fails to concatenate them.

        embedding_scatters=None covers a plain callable with no such attribute, which
        must keep the previous behaviour of not scattering.
        """
        torch.manual_seed(_SEED)
        config, mtp_block_spec = self._create_config_and_mtp_block_spec(tp=1, cp=1)
        config.sequence_parallel = True
        mtp = MultiTokenPredictionBlock(config=config, spec=mtp_block_spec)
        mtp_layer = mtp.layers[0]

        seq_len = 4
        batch_size = 2
        input_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.int64)
        position_ids = torch.arange(seq_len, dtype=torch.int64).repeat(batch_size, 1)
        hidden_states = torch.randn(seq_len, batch_size, config.hidden_size)
        embeddings = torch.randn(seq_len, batch_size, config.hidden_size)

        def embedding(input_ids, position_ids):
            return embeddings.clone()

        if embedding_scatters is not None:
            embedding.scatter_to_sequence_parallel = embedding_scatters

        scattered = []

        def fake_scatter(tensor, group=None):
            scattered.append(tensor)
            return tensor

        monkeypatch.setattr(
            "megatron.core.transformer.multi_token_prediction."
            "scatter_to_sequence_parallel_region",
            fake_scatter,
        )

        mtp_layer._get_embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            embedding=embedding,
            hidden_states=hidden_states,
            packed_seq_params=None,
        )

        assert (len(scattered) == 1) is expect_scatter

    @pytest.mark.parametrize("detach_heads", [False, True])
    def test_forward_detach_heads_gradient_flow(self, monkeypatch, detach_heads):
        """Block-level check of mtp_detach_heads: with the flag on, MTP gradients must
        not reach the main-model hidden_states or the shared embedding, while the MTP
        layer parameters still receive gradients."""
        torch.manual_seed(_SEED)
        config, mtp_block_spec = self._create_config_and_mtp_block_spec(tp=1, cp=1)
        config.mtp_detach_heads = detach_heads
        # Runs on GPU because _concat_embeddings exercises the (fused) norm and
        # projection kernels; the rest of the MTP transformer layer is stubbed out.
        mtp = MultiTokenPredictionBlock(config=config, spec=mtp_block_spec).cuda()

        # Replace each MTP transformer layer with an identity so the test isolates
        # gradient flow to the detach logic (not the attention kernels). Must be an
        # nn.Module since it is assigned as a child module of the layer.
        class _IdentityMTPLayer(torch.nn.Module):
            def forward(self, hidden_states, **kwargs):
                return hidden_states, None

        for layer in mtp.layers:
            monkeypatch.setattr(layer, "mtp_model_layer", _IdentityMTPLayer())

        seq_len = 4
        batch_size = 2
        input_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.int64).cuda()
        position_ids = torch.arange(seq_len, dtype=torch.int64).repeat(batch_size, 1).cuda()
        attention_mask = torch.ones((batch_size, 1, seq_len, seq_len), dtype=torch.bool).cuda()
        hidden_states = torch.randn(
            seq_len, batch_size, config.hidden_size, device="cuda", requires_grad=True
        )
        emb_weight = torch.nn.Parameter(
            torch.randn(seq_len, batch_size, config.hidden_size, device="cuda")
        )

        def fake_embedding(input_ids, position_ids):
            return emb_weight.clone()

        output = mtp.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            embedding=fake_embedding,
        )

        # forward concatenates [main_hidden_states, mtp_out_0, mtp_out_1] along dim 0;
        # back-propagate only from the MTP outputs to mimic the MTP loss path.
        mtp_outputs = output[seq_len:]
        mtp_outputs.sum().backward()

        # MTP layer parameters always receive gradients.
        for layer in mtp.layers:
            assert layer.enorm.weight.grad is not None
            assert layer.hnorm.weight.grad is not None
            assert layer.eh_proj.weight.grad is not None

        if detach_heads:
            # Gradients must not reach the main model or the shared embedding.
            # The returned block output still includes the original hidden-state
            # chunk, so autograd may allocate a zero grad for it through cat().
            if hidden_states.grad is not None:
                torch.testing.assert_close(hidden_states.grad, torch.zeros_like(hidden_states))
            assert emb_weight.grad is None
        else:
            assert hidden_states.grad is not None
            assert emb_weight.grad is not None

    @pytest.mark.parametrize("detach_heads", [False, True])
    @pytest.mark.parametrize("provide_output_weight", [False, True])
    @pytest.mark.parametrize("fuse_linear_cross_entropy", [False, True])
    def test_process_mtp_loss_detaches_output_weight(
        self, detach_heads, provide_output_weight, fuse_linear_cross_entropy
    ):
        """process_mtp_loss must detach the output-head weight when mtp_detach_heads=True
        so the MTP loss does not update the (shared) output projection weight."""
        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
        config = TransformerConfig(
            mtp_num_layers=2,
            num_layers=4,
            hidden_size=64,
            num_attention_heads=8,
            use_cpu_initialization=True,
            mtp_detach_heads=detach_heads,
        )
        config.cross_entropy_loss_fusion = fuse_linear_cross_entropy
        config.cross_entropy_fusion_impl = "linear" if fuse_linear_cross_entropy else "native"

        seq_len = 4
        batch_size = 2
        vocab_size = 16
        # hidden_states is the concatenation [main, mtp_0, mtp_1] along the sequence dim;
        # requires_grad so the returned tensor stays in the autograd graph for backward.
        hidden_states = torch.randn(
            (1 + config.mtp_num_layers) * seq_len,
            batch_size,
            config.hidden_size,
            requires_grad=True,
        )
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        loss_mask = torch.ones(batch_size, seq_len)
        explicit_output_weight = torch.nn.Parameter(torch.randn(vocab_size, config.hidden_size))

        class _OutputLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(vocab_size, config.hidden_size))

            def forward(
                self,
                hidden,
                weight=None,
                runtime_gather_output=None,
                output_cross_entropy_loss=False,
                labels=None,
            ):
                del runtime_gather_output, labels
                weight = self.weight if weight is None else weight
                # hidden: [s, b, h] -> logits: [s, b, vocab]
                logits = torch.matmul(hidden, weight.t())
                if output_cross_entropy_loss:
                    return logits.sum(dim=-1).transpose(0, 1)
                return logits, None

        output_layer = _OutputLayer()
        output_weight = explicit_output_weight if provide_output_weight else None
        weight_to_check = explicit_output_weight if provide_output_weight else output_layer.weight

        def compute_language_model_loss(labels, logits):
            # per-token loss of shape [b, s] that depends on logits (hence output_weight).
            return logits.sum(dim=-1).transpose(0, 1)

        result = process_mtp_loss(
            hidden_states=hidden_states,
            labels=labels,
            loss_mask=loss_mask,
            output_layer=output_layer,
            output_weight=output_weight,
            runtime_gather_output=None,
            is_training=False,
            compute_language_model_loss=compute_language_model_loss,
            config=config,
        )
        result.sum().backward()

        if detach_heads:
            assert weight_to_check.grad is None
        else:
            assert weight_to_check.grad is not None

    def test_process_mtp_loss_all_masked_positions_have_zero_gradient(self):
        """An all-zero mask must make MTP a finite no-op, including after mask rolling."""
        config = TransformerConfig(
            mtp_num_layers=1,
            num_layers=2,
            hidden_size=8,
            num_attention_heads=2,
            use_cpu_initialization=True,
        )
        seq_len = 4
        hidden_states = torch.randn(
            (1 + config.mtp_num_layers) * seq_len, 1, config.hidden_size, requires_grad=True
        )
        output_weight = torch.nn.Parameter(torch.randn(16, config.hidden_size))

        def output_layer(hidden, weight=None, runtime_gather_output=None):
            return torch.matmul(hidden, weight.t()), None

        def compute_language_model_loss(labels, logits):
            return logits.square().sum(dim=-1).transpose(0, 1)

        result = process_mtp_loss(
            hidden_states=hidden_states,
            labels=torch.arange(seq_len).unsqueeze(0),
            loss_mask=torch.zeros(1, seq_len),
            output_layer=output_layer,
            output_weight=output_weight,
            runtime_gather_output=None,
            is_training=False,
            compute_language_model_loss=compute_language_model_loss,
            config=config,
        )
        result.sum().backward()

        assert torch.isfinite(result).all()
        assert torch.count_nonzero(hidden_states.grad[seq_len:]) == 0
        assert output_weight.grad is not None
        assert torch.count_nonzero(output_weight.grad) == 0


class TestMTPHiddenStateRollUnderParallelism:
    """Token-level checks on the HSM hidden-state roll for every sequence-sharding mode.

    Hidden State Mixing keeps a history of hidden states and, at each MTP depth,
    rolls the older entries one position left so that every candidate lines up
    with the same target token before one is sampled per position. The property
    under test is that alignment: after rolling and sampling, the state sitting
    in a given slot must still be in charge of predicting the same token it
    would have been in charge of without HSM.

    Instrumentation
    ---------------
    A hidden state's value encodes the index of the newest token it has consumed:
    the backbone state owning global position ``p`` starts at ``p + 1``, and each
    stubbed MTP layer adds ``1``. A distinct offset per (batch, channel) pair is
    added so a transposed permute in the flatten/roll/unflatten round trip cannot
    slip through. ``0`` is therefore never a legitimate value, which keeps it
    usable as the "no local continuation" sentinel that ``roll_tensor`` writes
    and hidden-state mixing replaces with the newest entry.

    Under that encoding the alignment property collapses to one closed form: at
    depth ``d``, *every* history entry must read ``base(p) + d`` in the slot
    owning global position ``p``, no matter how many times it was rolled. An
    entry created at depth ``j`` holds ``base(p) + j`` and is rolled ``d - j``
    times, giving ``base(p + d - j) + j == base(p) + d``. A misaligned roll shows
    up as a numeric mismatch, and a correct implementation makes the mixed result
    independent of which index was sampled.

    Nothing here needs weights, data, or a real model: a ``SimpleNamespace``
    supplies the attributes ``MultiTokenPredictionBlock.forward`` reads, and the
    sharding is reproduced with the same helpers production uses. Real process
    groups and CUDA are still required, because ``roll_tensor``'s CP path *is*
    point-to-point communication.
    """

    SEQ_LENGTH = 32
    HIDDEN_SIZE = 8
    MTP_NUM_LAYERS = 3
    PACKED_DOC_LENGTHS = (8, 24)
    # Spacing between the per-(batch, channel) offsets. Must exceed SEQ_LENGTH so that
    # a value still decodes to exactly one (position, batch, channel) triple, which is
    # what lets a cross-channel mix-up be told apart from a plain positional shift.
    CHANNEL_STRIDE = 1000.0

    def setup_method(self, method):
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        destroy_global_vars()
        destroy_num_microbatches_calculator()

    @classmethod
    def _base_values(cls, global_positions, batch_size):
        """Build hidden states whose values encode the global positions they own."""
        channel_offset = cls.CHANNEL_STRIDE * torch.arange(
            batch_size * cls.HIDDEN_SIZE, device=global_positions.device, dtype=torch.float32
        ).view(batch_size, cls.HIDDEN_SIZE)
        return (global_positions + 1).view(-1, 1, 1).float() + channel_offset

    @classmethod
    def _local_global_positions(
        cls, cp_group, tp_group, sequence_parallel, cu_seqlens, cu_seqlens_padded=None
    ):
        """Return the global token positions this rank owns, in local order.

        Reproduces the production sharding: ``get_batch_on_this_cp_rank`` applies the
        context-parallel layout (per-sequence zigzag, or per-document zigzag when the
        batch is packed), then the sequence-parallel scatter keeps a contiguous 1/tp
        slice of what remains (``tensor_parallel.mappings._split_along_first_dim``).
        """
        positions = torch.arange(cls.SEQ_LENGTH, device="cuda").view(1, cls.SEQ_LENGTH)
        batch = {
            "tokens": positions,
            "cu_seqlens": None if cu_seqlens is None else cu_seqlens.view(1, -1),
            "cu_seqlens_padded": (
                None if cu_seqlens_padded is None else cu_seqlens_padded.view(1, -1)
            ),
        }
        batch = get_batch_on_this_cp_rank(batch, is_hybrid_cp=False, cp_group=cp_group)
        positions = batch["tokens"][0]
        if sequence_parallel:
            positions = positions.chunk(get_pg_size(tp_group))[get_pg_rank(tp_group)]
        return positions.contiguous()

    @classmethod
    def _document_end(cls, global_positions, cu_seqlens, cu_seqlens_padded=None):
        """Exclusive end of the document owning each global position."""
        if cu_seqlens is None:
            return torch.full_like(global_positions, cls.SEQ_LENGTH)
        if cu_seqlens_padded is not None:
            padded_ends = cu_seqlens_padded[1:].to(
                device=global_positions.device, dtype=global_positions.dtype
            )
            valid_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(
                device=global_positions.device, dtype=global_positions.dtype
            )
            valid_ends = (
                cu_seqlens_padded[:-1].to(
                    device=global_positions.device, dtype=global_positions.dtype
                )
                + valid_lengths
            )
            document_indices = torch.searchsorted(padded_ends, global_positions, right=True)
            return valid_ends[document_indices]
        ends = cu_seqlens[1:].to(device=global_positions.device, dtype=global_positions.dtype)
        return ends[torch.searchsorted(ends, global_positions, right=True)]

    @classmethod
    def _sentinel_mask(
        cls, global_positions, rolls, cu_seqlens, sequence_parallel, cu_seqlens_padded=None
    ):
        """Slots where the roll legitimately has no continuation to pull in.

        Always covers slots whose continuation lies past the end of their document
        (the end of the sequence when the batch is not packed), which is a semantic
        rule: MTP must not predict across a document boundary however the tokens are
        laid out.

        Under sequence parallelism ``roll_tensor`` exchanges nothing across shards, so
        a second, layout-driven rule applies: a slot is a sentinel whenever its
        continuation is not sitting in the next local slot. That covers the end of the
        shard, and with packed input it also covers the seam inside each document,
        where per-document CP zigzag places two globally distant chunks side by side.
        Hidden-state mixing substitutes the newest entry in all of those slots, so they lose
        their older candidates rather than pointing at the wrong token.
        """
        mask = global_positions + rolls >= cls._document_end(
            global_positions, cu_seqlens, cu_seqlens_padded
        )
        if sequence_parallel or cu_seqlens_padded is not None:
            local_count = len(global_positions)
            target = torch.arange(local_count, device=global_positions.device) + rolls
            in_shard = target < local_count
            continues_locally = torch.zeros_like(mask)
            continues_locally[in_shard] = (
                global_positions[target[in_shard]] == global_positions[in_shard] + rolls
            )
            mask = mask | ~continues_locally
        return mask

    @staticmethod
    def _mismatch_report(label, actual, expected, tolerated, global_positions):
        """Describe misaligned elements so a failure doubles as an investigation log.

        Compared exactly rather than with ``isclose``: every encoded value is a whole
        number well under 2**24 and the stubbed layers only ever add one, so no rounding
        error can arise. Exactness also keeps the check sharp on a long sequence, where
        ``isclose``'s relative tolerance grows past one token and would start accepting
        an off-by-one shift.
        """
        wrong = (actual != expected) & ~tolerated
        if not wrong.any():
            return ""
        lines = [f"{label}: {int(wrong.sum())} misaligned element(s)"]
        for slot, batch_index, channel in wrong.nonzero()[:8].tolist():
            lines.append(
                f"  slot {slot} (global token {int(global_positions[slot])}), "
                f"batch {batch_index}, channel {channel}: "
                f"expected {float(expected[slot, batch_index, channel])}, "
                f"got {float(actual[slot, batch_index, channel])}"
            )
        return "\n".join(lines)

    def _run_hsm_block(
        self,
        monkeypatch,
        config,
        hidden_states,
        cp_group,
        tp_group,
        sequence_parallel,
        packed_seq_params=None,
        force_oldest_selection=False,
    ):
        """Drive the real HSM path in ``MultiTokenPredictionBlock.forward`` with stub layers.

        Returns the rolled history handed to hidden-state mixing at each depth, the mixed
        result of each depth, the input each layer saw, and the block output.
        """
        layer_inputs = []

        class _TokenCountingLayer:
            """Stands in for an MTP layer: advances the state by exactly one token."""

            def __call__(self, hidden_states, input_ids, position_ids, padding_mask=None, **kwargs):
                layer_inputs.append(hidden_states.detach().clone())
                return hidden_states + 1, input_ids, position_ids, padding_mask

        rolled_history = []
        mixed_states = []
        real_mix_hidden_state_history = mtp_module._mix_hidden_state_history

        def recording_mix_hidden_state_history(older_hidden_states, newest_hidden_state, **kwargs):
            rolled_history.append(
                [state.detach().clone() for state in older_hidden_states.unbind(0)]
                + [newest_hidden_state.detach().clone()]
            )
            output = real_mix_hidden_state_history(
                older_hidden_states, newest_hidden_state, **kwargs
            )
            mixed_states.append(output.detach().clone())
            return output

        monkeypatch.setattr(
            mtp_module, "_mix_hidden_state_history", recording_mix_hidden_state_history
        )

        local_seq_length, batch_size = hidden_states.shape[:2]
        # One draw per (sequence owner, local position, batch): the owner count is the
        # sharding arithmetic under test, so it is pinned rather than assumed. Asserted
        # on both selection modes, which is what stops the real-sampler runs from being
        # a bare smoke test of a property that holds for any index.
        owner_count = (get_pg_size(tp_group) if sequence_parallel else 1) * get_pg_size(cp_group)
        expected_index_shape = (1, owner_count * local_seq_length, batch_size, 1)
        real_randint = torch.randint

        def instrumented_randint(high, size, device):
            if force_oldest_selection:
                # Always sample the oldest entry so the zero-sentinel fallback is
                # exercised rather than hidden behind a lucky draw of the newest entry.
                return torch.zeros(size, dtype=torch.long, device=device)
            return real_randint(high, size, device=device)

        monkeypatch.setattr(
            torch, "randint", _randint_stub(expected_index_shape, instrumented_randint)
        )

        block = types.SimpleNamespace(
            config=config,
            training=True,
            vp_stage=None,
            mtp_use_repeated_layer=False,
            cp_group=cp_group,
            tp_group=tp_group,
            hidden_state_mixing_rng_tracker_name=None,
            sequence_parallel=sequence_parallel,
            layers=[_TokenCountingLayer() for _ in range(config.mtp_num_layers)],
        )

        token_ids = torch.zeros(batch_size, local_seq_length, dtype=torch.int64, device="cuda")
        output = MultiTokenPredictionBlock.forward(
            block,
            input_ids=token_ids,
            position_ids=token_ids.clone(),
            hidden_states=hidden_states,
            attention_mask=None,
            packed_seq_params=packed_seq_params,
        )
        return rolled_history, mixed_states, layer_inputs, output

    def _assert_history_stays_aligned(
        self,
        rolled_history,
        mixed_states,
        layer_inputs,
        output,
        hidden_states,
        global_positions,
        batch_size,
        cu_seqlens,
        sequence_parallel,
        num_depths,
        cu_seqlens_padded=None,
    ):
        """Assert every candidate, every mix, and every layer input targets the same token."""
        assert len(rolled_history) == num_depths - 1, "HSM should mix at every depth but the first"

        for depth, (history, mixed) in enumerate(zip(rolled_history, mixed_states), start=1):
            expected = self._base_values(global_positions, batch_size) + depth
            assert len(history) == depth + 1, f"depth {depth} should offer {depth + 1} candidates"

            for entry_index, entry in enumerate(history):
                rolls = depth - entry_index
                sentinel = self._sentinel_mask(
                    global_positions, rolls, cu_seqlens, sequence_parallel, cu_seqlens_padded
                )
                tolerated = sentinel.view(-1, 1, 1).expand_as(entry) & (entry == 0)
                report = self._mismatch_report(
                    f"depth {depth}, history entry {entry_index} (rolled {rolls}x)",
                    entry,
                    expected,
                    tolerated,
                    global_positions,
                )
                assert not report, report

            # No tolerance here: the sentinel positions must have taken the newest
            # entry, so the sampled index cannot change which token is predicted.
            report = self._mismatch_report(
                f"depth {depth}, mixed hidden states",
                mixed,
                expected,
                torch.zeros_like(mixed, dtype=torch.bool),
                global_positions,
            )
            assert not report, report

        torch.testing.assert_close(layer_inputs[0], hidden_states)
        for depth in range(1, num_depths):
            torch.testing.assert_close(layer_inputs[depth], mixed_states[depth - 1])

        # End to end: MTP depth k's output must sit exactly k + 1 tokens ahead of the
        # backbone state in the same slot.
        chunks = torch.chunk(output, 1 + num_depths, dim=0)
        torch.testing.assert_close(chunks[0], hidden_states)
        for depth in range(num_depths):
            torch.testing.assert_close(chunks[depth + 1], hidden_states + depth + 1)

    @pytest.mark.parametrize("force_oldest_selection", [False, True])
    @pytest.mark.parametrize(
        ("tp", "cp", "sequence_parallel"),
        [
            (1, 1, False),
            (2, 1, False),  # TP without SP: the sequence is replicated across TP ranks
            (1, 2, False),
            (1, 4, False),
            (2, 2, False),
            (2, 1, True),
            (4, 1, True),
            # CP combined with sequence parallelism: roll_tensor's CP branch used to split
            # the local tensor in two assuming it held this rank's pair of zigzag chunks,
            # which SP had already sharded further, so the neighbour exchange swapped in
            # unrelated tokens. Withholding cp_group under SP fixed these.
            (2, 2, True),
            (4, 2, True),
            (2, 4, True),  # Second CP width, so the property is not pinned to CP=2. Needs 8 ranks.
        ],
    )
    def test_hsm_roll_keeps_every_candidate_on_the_same_target(
        self, monkeypatch, tp, cp, sequence_parallel, force_oldest_selection
    ):
        """Rolled HSM candidates must all predict the same token as the newest one."""
        if int(os.environ.get("WORLD_SIZE", "1")) < tp * cp:
            pytest.skip(f"TP={tp} x CP={cp} requires at least {tp * cp} ranks")
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)
        model_parallel_cuda_manual_seed(_SEED, force_reset_rng=True)

        batch_size = 2
        config = TransformerConfig(
            num_layers=2,
            hidden_size=self.HIDDEN_SIZE,
            num_attention_heads=self.HIDDEN_SIZE,
            mtp_num_layers=self.MTP_NUM_LAYERS,
            mtp_hsm=True,
            tensor_model_parallel_size=tp,
            context_parallel_size=cp,
            sequence_parallel=sequence_parallel,
        )
        tp_group = get_tensor_model_parallel_group()
        cp_group = get_context_parallel_group()

        global_positions = self._local_global_positions(
            cp_group, tp_group, sequence_parallel, cu_seqlens=None
        )
        assert len(global_positions) == self.SEQ_LENGTH // (cp * (tp if sequence_parallel else 1))
        hidden_states = self._base_values(global_positions, batch_size)

        rolled_history, mixed_states, layer_inputs, output = self._run_hsm_block(
            monkeypatch,
            config=config,
            hidden_states=hidden_states,
            cp_group=cp_group,
            tp_group=tp_group,
            sequence_parallel=sequence_parallel,
            force_oldest_selection=force_oldest_selection,
        )

        self._assert_history_stays_aligned(
            rolled_history,
            mixed_states,
            layer_inputs,
            output,
            hidden_states,
            global_positions,
            batch_size,
            cu_seqlens=None,
            sequence_parallel=sequence_parallel,
            num_depths=self.MTP_NUM_LAYERS,
        )

    @pytest.mark.skipif(not HAVE_TE, reason="per-document CP partitioning needs transformer_engine")
    @pytest.mark.parametrize("force_oldest_selection", [False, True])
    @pytest.mark.parametrize(
        ("tp", "cp", "sequence_parallel"),
        [
            (1, 1, False),
            (2, 1, False),
            (1, 2, False),
            # Second CP width without sequence parallelism, so the per-document CP
            # branch of the roll -- the one that exchanges a boundary token per
            # document per rank -- is not only ever exercised at CP=2.
            (1, 4, False),
            # Packed input under sequence parallelism needs the document boundaries
            # translated into the shard's frame, seams included. These were expected
            # failures until that landed.
            (2, 1, True),
            # Four shards against a document that ends exactly on a shard edge, which
            # is where the translation clamps a document down to zero length.
            (4, 1, True),
            (2, 2, True),
            (2, 4, True),  # Needs 8 ranks.
        ],
    )
    @pytest.mark.parametrize("padded_boundaries", [False, True])
    def test_hsm_roll_respects_packed_document_boundaries(
        self, monkeypatch, tp, cp, sequence_parallel, force_oldest_selection, padded_boundaries
    ):
        """With packed input the roll must stop at document ends, never cross them.

        ``padded_boundaries`` mirrors the shape ``pretrain_gpt.py`` and
        ``pretrain_hybrid.py`` build whenever ``CP > 1``: one tensor passed as both
        ``cu_seqlens_q`` and ``cu_seqlens_q_padded``. Without this case the suite only
        ever exercised the unpadded shape, and a shard translation that bails out on a
        set ``cu_seqlens_q_padded`` would look correct here while doing nothing in a
        real run.
        """
        if int(os.environ.get("WORLD_SIZE", "1")) < tp * cp:
            pytest.skip(f"TP={tp} x CP={cp} requires at least {tp * cp} ranks")
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)
        model_parallel_cuda_manual_seed(_SEED, force_reset_rng=True)

        batch_size = 1
        config = TransformerConfig(
            num_layers=2,
            hidden_size=self.HIDDEN_SIZE,
            num_attention_heads=self.HIDDEN_SIZE,
            mtp_num_layers=self.MTP_NUM_LAYERS,
            mtp_hsm=True,
            tensor_model_parallel_size=tp,
            context_parallel_size=cp,
            sequence_parallel=sequence_parallel,
        )
        tp_group = get_tensor_model_parallel_group()
        cp_group = get_context_parallel_group()

        # cu_seqlens stays in global (unsharded) coordinates, as production passes it.
        cu_seqlens = torch.tensor(
            [0, *itertools.accumulate(self.PACKED_DOC_LENGTHS)], dtype=torch.int32, device="cuda"
        )
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens if padded_boundaries else None,
            cu_seqlens_kv_padded=cu_seqlens if padded_boundaries else None,
            max_seqlen_q=max(self.PACKED_DOC_LENGTHS),
            max_seqlen_kv=max(self.PACKED_DOC_LENGTHS),
            # Set alongside the padded boundaries, as the training scripts do, so the
            # seq_idx rebuild in __post_init__ is part of what this covers.
            total_tokens=self.SEQ_LENGTH if padded_boundaries else None,
            qkv_format='thd',
        )

        global_positions = self._local_global_positions(
            cp_group, tp_group, sequence_parallel, cu_seqlens=cu_seqlens
        )
        assert len(global_positions) == self.SEQ_LENGTH // (cp * (tp if sequence_parallel else 1))
        hidden_states = self._base_values(global_positions, batch_size)

        rolled_history, mixed_states, layer_inputs, output = self._run_hsm_block(
            monkeypatch,
            config=config,
            hidden_states=hidden_states,
            cp_group=cp_group,
            tp_group=tp_group,
            sequence_parallel=sequence_parallel,
            packed_seq_params=packed_seq_params,
            force_oldest_selection=force_oldest_selection,
        )

        self._assert_history_stays_aligned(
            rolled_history,
            mixed_states,
            layer_inputs,
            output,
            hidden_states,
            global_positions,
            batch_size,
            cu_seqlens=cu_seqlens,
            sequence_parallel=sequence_parallel,
            num_depths=self.MTP_NUM_LAYERS,
        )

    def test_shard_boundary_translation_handles_production_packed_params(self):
        """Pin the coordinate change the packed roll depends on, boundary by boundary.

        Global ``cu_seqlens`` has to become this shard's boundaries: divided by
        ``cp_size``, shifted by where the shard starts, and split again at each
        document's zigzag seam, where two globally distant chunks sit side by side.
        The higher-level tests can only show the result of getting this wrong; this
        shows the numbers.

        Also pins which packed-parameter shapes are recognised. One tensor used as both
        ``cu_seqlens_q`` and ``cu_seqlens_q_padded`` is what the training scripts build
        when ``CP > 1`` and must be translated; genuinely different padded boundaries
        are not handled and must leave the caller alone rather than be mistranslated.
        """
        tp, cp = 2, 2
        if int(os.environ.get("WORLD_SIZE", "1")) < tp * cp:
            pytest.skip(f"TP={tp} x CP={cp} requires at least {tp * cp} ranks")
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)

        tp_group = get_tensor_model_parallel_group()
        cp_group = get_context_parallel_group()
        local_seq_length = self.SEQ_LENGTH // (cp * tp)
        cu_seqlens = torch.tensor(
            [0, *itertools.accumulate(self.PACKED_DOC_LENGTHS)], dtype=torch.int32, device="cuda"
        )

        # SEQ_LENGTH=32 packed as (8, 24) with cp=2 gives CP-local boundaries [0, 4, 16].
        # CP rank 0 holds chunks 0 and 3 of each document, which are far apart globally,
        # so its seams go in: [0, 2, 4, 10, 16]. CP rank 1 holds chunks 1 and 2, which are
        # neighbours, so its piece is contiguous and keeps [0, 4, 16]. Shard 0 then owns
        # local slots [0, 8) of that list and shard 1 owns [8, 16), each keeping the part
        # inside its window and flattening the rest onto its edges.
        expected = {
            (0, 0): [0, 2, 4, 8, 8],
            (0, 1): [0, 0, 0, 2, 8],
            (1, 0): [0, 4, 8],
            (1, 1): [0, 0, 8],
        }[(get_pg_rank(cp_group), get_pg_rank(tp_group))]

        translated = _packed_seq_params_for_local_hsm_roll(
            PackedSeqParams(cu_seqlens_q=cu_seqlens, cu_seqlens_kv=cu_seqlens, qkv_format='thd'),
            local_seq_length=local_seq_length,
            cp_group=cp_group,
            tp_group=tp_group,
        )
        assert translated is not None
        assert translated.cu_seqlens_q.tolist() == expected
        assert translated.cu_seqlens_kv.tolist() == expected
        # Stale global-scale metadata must not survive into shard-local coordinates.
        assert translated.total_tokens is None
        assert translated.seq_idx is None

        # The production shape: same tensor in both fields, plus total_tokens.
        production = _packed_seq_params_for_local_hsm_roll(
            PackedSeqParams(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                cu_seqlens_q_padded=cu_seqlens,
                cu_seqlens_kv_padded=cu_seqlens,
                total_tokens=self.SEQ_LENGTH,
                qkv_format='thd',
            ),
            local_seq_length=local_seq_length,
            cp_group=cp_group,
            tp_group=tp_group,
        )
        assert production is not None, (
            "Params built the way pretrain_gpt.py and pretrain_hybrid.py build them when "
            "CP > 1 were not recognised, so the shard translation would be skipped in "
            "every real run even though it passes the unpadded tests."
        )
        assert production.cu_seqlens_q.tolist() == expected

        # Genuine padding: two physical 16-token documents with 10 valid tokens each.
        unpadded = torch.tensor([0, 10, 20], dtype=torch.int32, device="cuda")
        padded = torch.tensor([0, 16, 32], dtype=torch.int32, device="cuda")
        really_padded = _packed_seq_params_for_local_hsm_roll(
            PackedSeqParams(
                cu_seqlens_q=unpadded,
                cu_seqlens_kv=unpadded,
                cu_seqlens_q_padded=padded,
                cu_seqlens_kv_padded=padded,
                qkv_format='thd',
            ),
            local_seq_length=local_seq_length,
            cp_group=cp_group,
            tp_group=tp_group,
        )
        assert really_padded is not None
        expected_padded = {
            (0, 0): [0, 4, 8, 8, 8],
            (0, 1): [0, 0, 0, 4, 8],
            (1, 0): [0, 6, 8, 8, 8],
            (1, 1): [0, 0, 0, 6, 8],
        }[(get_pg_rank(cp_group), get_pg_rank(tp_group))]
        assert really_padded.cu_seqlens_q.tolist() == expected_padded
        assert really_padded.cu_seqlens_q_padded is None

        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("force_oldest_selection", [False, True])
    @pytest.mark.parametrize(
        ("tp", "cp", "sequence_parallel"),
        [
            (1, 1, False),
            (2, 1, False),
            (1, 2, False),
            (1, 4, False),
            (2, 2, False),
            (2, 1, True),
            (4, 1, True),
            (2, 2, True),
            (4, 2, True),
            (2, 4, True),
        ],
    )
    def test_hsm_roll_respects_genuinely_padded_boundaries(
        self, monkeypatch, tp, cp, sequence_parallel, force_oldest_selection
    ):
        """Genuine padding stays aligned in every supported TP/CP/SP layout."""
        if int(os.environ.get("WORLD_SIZE", "1")) < tp * cp:
            pytest.skip(f"TP={tp} x CP={cp} requires at least {tp * cp} ranks")
        if not HAVE_TE:
            pytest.skip("per-document CP partitioning needs transformer_engine")
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)
        model_parallel_cuda_manual_seed(_SEED, force_reset_rng=True)
        tp_group = get_tensor_model_parallel_group()
        cp_group = get_context_parallel_group()
        cu_seqlens = torch.tensor([0, 10, 20], dtype=torch.int32, device="cuda")
        padded = torch.tensor([0, 16, 32], dtype=torch.int32, device="cuda")
        params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=padded,
            cu_seqlens_kv_padded=padded,
            qkv_format='thd',
        )
        positions = self._local_global_positions(
            cp_group, tp_group, sequence_parallel, cu_seqlens, cu_seqlens_padded=padded
        )
        batch_size = 2
        hidden_states = self._base_values(positions, batch_size=batch_size)
        config = TransformerConfig(
            num_layers=2,
            hidden_size=self.HIDDEN_SIZE,
            num_attention_heads=self.HIDDEN_SIZE,
            mtp_num_layers=self.MTP_NUM_LAYERS,
            mtp_hsm=True,
            tensor_model_parallel_size=tp,
            context_parallel_size=cp,
            sequence_parallel=sequence_parallel,
        )
        rolled_history, mixed_states, layer_inputs, output = self._run_hsm_block(
            monkeypatch,
            config,
            hidden_states,
            cp_group,
            tp_group,
            sequence_parallel,
            packed_seq_params=params,
            force_oldest_selection=force_oldest_selection,
        )

        self._assert_history_stays_aligned(
            rolled_history,
            mixed_states,
            layer_inputs,
            output,
            hidden_states,
            positions,
            batch_size,
            cu_seqlens,
            sequence_parallel,
            self.MTP_NUM_LAYERS,
            cu_seqlens_padded=padded,
        )

        padding = ((positions >= 10) & (positions < 16)) | (positions >= 26)
        for history, mixed in zip(rolled_history, mixed_states):
            torch.testing.assert_close(mixed[padding], history[-1][padding])

    def test_hsm_genuinely_padded_backward(self, monkeypatch):
        """The genuine-padding fallback remains differentiable through HSM."""
        tp, cp = 2, 2
        if int(os.environ.get("WORLD_SIZE", "1")) < tp * cp:
            pytest.skip(f"TP={tp} x CP={cp} requires at least {tp * cp} ranks")
        if not HAVE_TE:
            pytest.skip("per-document CP partitioning needs transformer_engine")
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)
        model_parallel_cuda_manual_seed(_SEED, force_reset_rng=True)
        tp_group = get_tensor_model_parallel_group()
        cp_group = get_context_parallel_group()
        cu_seqlens = torch.tensor([0, 10, 20], dtype=torch.int32, device="cuda")
        padded = torch.tensor([0, 16, 32], dtype=torch.int32, device="cuda")
        params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=padded,
            cu_seqlens_kv_padded=padded,
            qkv_format='thd',
        )
        positions = self._local_global_positions(
            cp_group, tp_group, True, cu_seqlens, cu_seqlens_padded=padded
        )
        hidden_states = self._base_values(positions, batch_size=1).requires_grad_()
        config = TransformerConfig(
            num_layers=2,
            hidden_size=self.HIDDEN_SIZE,
            num_attention_heads=self.HIDDEN_SIZE,
            mtp_num_layers=self.MTP_NUM_LAYERS,
            mtp_hsm=True,
            tensor_model_parallel_size=tp,
            context_parallel_size=cp,
            sequence_parallel=True,
        )
        _, _, _, output = self._run_hsm_block(
            monkeypatch,
            config,
            hidden_states,
            cp_group,
            tp_group,
            True,
            packed_seq_params=params,
            force_oldest_selection=True,
        )
        torch.chunk(output, 1 + self.MTP_NUM_LAYERS, dim=0)[-1].sum().backward()

        assert hidden_states.grad is not None
        assert torch.isfinite(hidden_states.grad).all()

    @pytest.mark.parametrize("packed", [False, True])
    def test_hsm_roll_stays_aligned_on_a_long_sequence(self, monkeypatch, packed):
        """Repeat the alignment check at a production-sized sequence length.

        The matrices above sweep every parallelism cell at 32 tokens, which is cheap but
        blunt: a context-parallel chunk is only 8 slots wide there, so a roll that
        swapped whole chunks would land a handful of positions from the truth and read
        like an off-by-one. At 8192 the two are thousands of positions apart, and the
        failure message says which one happened. That is the whole reason for the extra
        case, so it runs the one cell where the distinction exists: CP=2 is the narrowest
        sharding whose roll moves tokens between chunks at all.

        Everything else is reused unchanged -- the helpers are shape-agnostic -- and the
        channel stride is lifted above the longer sequence to keep each value decodable
        to exactly one (position, batch, channel) triple.
        """
        cp = 2
        if int(os.environ.get("WORLD_SIZE", "1")) < cp:
            pytest.skip(f"CP={cp} requires at least {cp} ranks")
        if packed and not HAVE_TE:
            pytest.skip("per-document CP partitioning needs transformer_engine")

        seq_length = 8192
        monkeypatch.setattr(type(self), "SEQ_LENGTH", seq_length)
        monkeypatch.setattr(type(self), "CHANNEL_STRIDE", float(2 * seq_length))
        # Same 1:3 document ratio as the short packed case.
        monkeypatch.setattr(
            type(self), "PACKED_DOC_LENGTHS", (seq_length // 4, seq_length - seq_length // 4)
        )

        Utils.initialize_model_parallel(context_parallel_size=cp)
        model_parallel_cuda_manual_seed(_SEED, force_reset_rng=True)

        batch_size = 1 if packed else 2
        config = TransformerConfig(
            num_layers=2,
            hidden_size=self.HIDDEN_SIZE,
            num_attention_heads=self.HIDDEN_SIZE,
            mtp_num_layers=self.MTP_NUM_LAYERS,
            mtp_hsm=True,
            context_parallel_size=cp,
        )
        tp_group = get_tensor_model_parallel_group()
        cp_group = get_context_parallel_group()

        cu_seqlens = None
        packed_seq_params = None
        if packed:
            cu_seqlens = torch.tensor(
                [0, *itertools.accumulate(self.PACKED_DOC_LENGTHS)],
                dtype=torch.int32,
                device="cuda",
            )
            packed_seq_params = PackedSeqParams(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                max_seqlen_q=max(self.PACKED_DOC_LENGTHS),
                max_seqlen_kv=max(self.PACKED_DOC_LENGTHS),
                qkv_format='thd',
            )

        global_positions = self._local_global_positions(
            cp_group, tp_group, sequence_parallel=False, cu_seqlens=cu_seqlens
        )
        assert len(global_positions) == seq_length // cp
        hidden_states = self._base_values(global_positions, batch_size)

        rolled_history, mixed_states, layer_inputs, output = self._run_hsm_block(
            monkeypatch,
            config=config,
            hidden_states=hidden_states,
            cp_group=cp_group,
            tp_group=tp_group,
            sequence_parallel=False,
            packed_seq_params=packed_seq_params,
            # One draw, not two: always taking the oldest entry puts the deepest roll and
            # the zero-sentinel fallback on the path of this single run.
            force_oldest_selection=True,
        )

        self._assert_history_stays_aligned(
            rolled_history,
            mixed_states,
            layer_inputs,
            output,
            hidden_states,
            global_positions,
            batch_size,
            cu_seqlens=cu_seqlens,
            sequence_parallel=False,
            num_depths=self.MTP_NUM_LAYERS,
        )

    def test_sequence_parallel_shard_boundary_falls_back_to_newest_state(self, monkeypatch):
        """Make the accepted sequence-parallel degradation explicit.

        ``roll_tensor`` exchanges boundary tokens across CP ranks but not across
        sequence-parallel shards, so the last slot of every shard loses its
        continuation and is zero-filled. Those slots are exactly the ones
        Hidden-state mixing maps back onto the newest entry, which is why the alignment
        property above still holds. This test pins both halves of that bargain,
        and pins that the loss is confined to the shard boundary.
        """
        tp = 2
        if int(os.environ.get("WORLD_SIZE", "1")) < tp:
            pytest.skip(f"TP={tp} requires at least {tp} ranks")
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp)
        model_parallel_cuda_manual_seed(_SEED, force_reset_rng=True)

        batch_size = 2
        config = TransformerConfig(
            num_layers=2,
            hidden_size=self.HIDDEN_SIZE,
            num_attention_heads=self.HIDDEN_SIZE,
            mtp_num_layers=2,
            mtp_hsm=True,
            tensor_model_parallel_size=tp,
            sequence_parallel=True,
        )
        tp_group = get_tensor_model_parallel_group()
        cp_group = get_context_parallel_group()
        global_positions = self._local_global_positions(
            cp_group, tp_group, sequence_parallel=True, cu_seqlens=None
        )
        hidden_states = self._base_values(global_positions, batch_size)

        rolled_history, mixed_states, _, _ = self._run_hsm_block(
            monkeypatch,
            config=config,
            hidden_states=hidden_states,
            cp_group=cp_group,
            tp_group=tp_group,
            sequence_parallel=True,
            force_oldest_selection=True,
        )

        oldest_after_one_roll, newest = rolled_history[0]
        zeroed = (oldest_after_one_roll == 0).all(dim=-1).all(dim=-1)

        # Exactly one slot per shard is lost: the final one.
        expected_zeroed = torch.zeros_like(zeroed)
        expected_zeroed[-1] = True
        torch.testing.assert_close(zeroed, expected_zeroed)

        # On every rank but the last, that slot has a real continuation elsewhere in
        # the sequence, so the zero is sequence-parallel sharding, not the sequence end.
        last_global_position = int(global_positions[-1])
        if get_pg_rank(tp_group) < tp - 1:
            assert last_global_position + 1 < self.SEQ_LENGTH

        # The fallback restores the target: the mixed state matches the newest entry.
        torch.testing.assert_close(mixed_states[0][-1], newest[-1])
        torch.testing.assert_close(mixed_states[0], hidden_states + 1)


class TestMultiTokenPrediction:
    def setup_method(self, method):
        self.seq_length = 32
        self.micro_batch_size = 2
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        MTPLossLoggingHelper.tracker = {}

    def model_provider(
        self,
        pre_process=True,
        post_process=True,
        layer_spec_fn=get_gpt_layer_with_transformer_engine_spec,
        **config_kwargs,
    ):
        model_parallel_cuda_manual_seed(_SEED)
        args = get_args()
        config = core_transformer_config_from_args(args)
        transformer_layer_spec = layer_spec_fn(
            args.num_experts, args.moe_grouped_gemm, args.qk_layernorm
        )
        mtp_block_spec = get_gpt_mtp_block_spec(
            config=config, spec=transformer_layer_spec, use_transformer_engine=True
        )
        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            mtp_block_spec=mtp_block_spec,
            # create_test_args builds args without a tokenizer, so args.vocab_size stays
            # None; padded_vocab_size is what the training entrypoint passes anyway.
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
        )

        return model

    def create_test_args(
        self,
        tp,
        cp,
        sequence_length,
        micro_batch_size,
        fp8=None,
        full_recompute=False,
        mtp_hsm=False,
    ):
        destroy_global_vars()
        destroy_num_microbatches_calculator()

        sys.argv = ['test_multi_token_predictioin.py']
        args = parse_args()
        args.num_layers = 2
        args.mtp_num_layers = 2
        args.mtp_hsm = mtp_hsm
        args.mtp_loss_scaling_factor = 0.1
        args.padded_vocab_size = 128800
        args.hidden_size = 128
        args.num_attention_heads = 8
        args.max_position_embeddings = 256
        args.micro_batch_size = micro_batch_size
        args.create_attention_mask_in_dataloader = True
        args.seq_length = sequence_length
        args.tensor_model_parallel_size = tp
        args.sequence_parallel = True if tp > 1 else False
        args.context_parallel_size = cp
        args.position_embedding_type = 'rope'
        args.num_experts = 8
        args.train_iters = 1
        args.ckpt_format = 'torch_dist'
        args.moe_router_topk = 2
        args.moe_router_pre_softmax = False
        args.lr = 3e-5
        args.attention_dropout = 0.0
        args.hidden_dropout = 0.0
        args.no_save_optim = True
        args.no_load_optim = True
        args.no_load_rng = True
        args.save_tokenizer_assets = False
        if HAVE_TE:
            # only use grouped gemm if there is TE
            args.moe_grouped_gemm = True
        else:
            args.moe_grouped_gemm = False
        args.bf16 = True
        if fp8 is not None:
            args.fp8 = 'e4m3'
        if full_recompute:
            args.recompute_granularity = 'full'
            args.recompute_method = 'uniform'
            args.recompute_num_layers = 1
        else:
            args.recompute_granularity = None
        args.add_bias_linear = False
        args.swiglu = True

        validate_args(args)
        set_global_variables(args, False)
        return args

    def get_batch(self, seq_length, micro_batch_size):
        data = list(range(seq_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        labels = 1 + torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, seq_length, seq_length), dtype=bool
        ).cuda()
        loss_mask = torch.ones(seq_length).repeat((micro_batch_size, 1)).cuda()
        batch = {
            'tokens': input_ids,
            'labels': labels,
            'loss_mask': loss_mask,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
        }
        return batch

    def get_packed_batch(self, seq_lengths, micro_batch_size):
        """
        Create a packed sequence batch with multiple sequences of varying lengths.

        Args:
            seq_lengths: List of sequence lengths (e.g., [10, 15, 8] for 3 sequences)
            micro_batch_size: Batch size (typically 1 for packed sequences)

        Returns:
            batch: Dictionary containing packed sequences and PackedSeqParams
        """
        total_seq_length = sum(seq_lengths)

        # Create packed input_ids, labels, and position_ids
        input_ids_list = []
        labels_list = []
        position_ids_list = []

        for seq_len in seq_lengths:
            data = list(range(seq_len))
            input_ids_list.extend(data)
            labels_list.extend([x + 1 for x in data])
            position_ids_list.extend(data)

        # Convert to tensors with shape [batch, total_seq_length]
        input_ids = torch.tensor(input_ids_list, dtype=torch.int64).unsqueeze(0).cuda()
        labels = torch.tensor(labels_list, dtype=torch.int64).unsqueeze(0).cuda()
        position_ids = torch.tensor(position_ids_list, dtype=torch.int64).unsqueeze(0).cuda()

        # Create attention mask for packed sequences (all ones for simplicity)
        attention_mask = torch.ones(
            (micro_batch_size, 1, total_seq_length, total_seq_length), dtype=bool
        ).cuda()

        # Create loss mask with shape [batch, total_seq_length]
        loss_mask = torch.ones(micro_batch_size, total_seq_length).cuda()

        # Create cumulative sequence lengths for PackedSeqParams
        cu_seqlens = torch.tensor(
            [0] + [sum(seq_lengths[: i + 1]) for i in range(len(seq_lengths))], dtype=torch.int32
        ).cuda()

        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max(seq_lengths),
            max_seqlen_kv=max(seq_lengths),
            qkv_format='thd',
        )

        batch = {
            'tokens': input_ids,
            'labels': labels,
            'loss_mask': loss_mask,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'packed_seq_params': packed_seq_params,
        }
        return batch

    @pytest.mark.skipif(
        not HAVE_TE or not is_te_min_version("2.1.0"),
        reason="grouped_gemm requires TransformerEngine >= 2.1.0",
    )
    @pytest.mark.parametrize(("tp", "cp"), [(2, 1), (2, 2)])
    def test_sharded_state_dict(self, tp, cp):
        """Test MTP with different tensor parallel sizes."""
        args = self.create_test_args(tp, cp, self.seq_length, self.micro_batch_size)
        set_args(args)
        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)

        model_parallel_cuda_manual_seed(_SEED)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        model_cfg = gpt_config_from_args(args)
        builder_cls = model_cfg.get_builder_cls()
        builder = builder_cls(model_cfg)
        gpt_model = builder.build_distributed_models(
            pg_collection=pg_collection, wrap_with_ddp=False
        )
        sharded_state_dict = gpt_model[0].sharded_state_dict()
        for i in range(args.mtp_num_layers):
            assert f"mtp.layers.{i}.enorm.weight" in sharded_state_dict.keys()
            assert f"mtp.layers.{i}.hnorm.weight" in sharded_state_dict.keys()
            assert f"mtp.layers.{i}.eh_proj.weight" in sharded_state_dict.keys()

    @pytest.mark.flaky_in_dev
    @pytest.mark.skipif(
        not HAVE_TE or not is_te_min_version("2.1.0"),
        reason="grouped_gemm requires TransformerEngine >= 2.1.0",
    )
    @pytest.mark.parametrize(
        ("tp", "cp", "full_recompute"), [(1, 1, False), (1, 4, False), (2, 4, False), (4, 1, True)]
    )
    def test_forward_backward(self, tmp_path_dist_ckpt, tp, cp, full_recompute):
        """Test MTP forward and backward with gptmodel."""
        tp_ref = 1
        cp_ref = 1
        args = self.create_test_args(tp_ref, cp_ref, self.seq_length, self.micro_batch_size)
        set_args(args)
        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_ref, context_parallel_size=cp_ref
        )
        batch = self.get_batch(self.seq_length, self.micro_batch_size)
        tokens, labels, loss_mask, attention_mask, position_ids = batch.values()
        gpt_model_ref, optimizer, opt_param_scheduler = setup_model_and_optimizer(
            ModelType.encoder_or_decoder, self.model_provider
        )
        output_ref = gpt_model_ref[0].forward(
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            loss_mask=loss_mask,
        )
        # Forward accumulates normalized losses. Trigger the DP+CP
        # reduction so tracker["values"] becomes available.
        MTPLossLoggingHelper.reduce_loss_in_tracker()
        tracker = MTPLossLoggingHelper.tracker
        assert "values" in tracker
        mtp_loss_ref = tracker['values'].clone()
        MTPLossLoggingHelper.clean_loss_in_tracker()

        iteration = 123
        num_floating_point_operations_so_far = 456

        def set_ckpt_path(ckpt_path):
            args.save = ckpt_path
            args.load = ckpt_path

        with TempNamedDir(
            tmp_path_dist_ckpt / 'test_mtp_model_reconfiguration_model_A'
        ) as ckpt_dir_A:
            set_ckpt_path(ckpt_dir_A)
            save_checkpoint(
                iteration,
                gpt_model_ref,
                optimizer,
                opt_param_scheduler,
                num_floating_point_operations_so_far,
            )

            expected_ckpt_path = args.save / "iter_0000123" / ".metadata"
            assert os.path.exists(expected_ckpt_path)

            # Test with different TP/CP configuration
            Utils.destroy_model_parallel()
            args = self.create_test_args(
                tp, cp, self.seq_length, self.micro_batch_size, full_recompute=full_recompute
            )
            set_args(args)
            set_ckpt_path(ckpt_dir_A)
            torch.manual_seed(_SEED)
            Utils.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)
            gpt_model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
                ModelType.encoder_or_decoder, self.model_provider
            )
            load_checkpoint(gpt_model, optimizer, opt_param_scheduler, strict=False)
            batch["output_ref"] = output_ref
            # Get batch for current CP rank (handles CP tensor splitting)
            batch = get_batch_on_this_cp_rank(
                batch, is_hybrid_cp=False, cp_group=get_context_parallel_group()
            )
            tokens, labels, loss_mask, attention_mask, position_ids, output_ref = batch.values()
            output = gpt_model[0].forward(
                input_ids=tokens,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=labels,
                loss_mask=loss_mask,
            )
            # Combine normalized loss contributions across DP+CP.
            MTPLossLoggingHelper.reduce_loss_in_tracker()
            tracker = MTPLossLoggingHelper.tracker
            assert "values" in tracker
            mtp_loss = tracker['values'].clone()
            MTPLossLoggingHelper.clean_loss_in_tracker()
            assert torch.allclose(output_ref, output, rtol=1e-03, atol=1e-03)
            assert torch.allclose(mtp_loss, mtp_loss_ref, rtol=1e-02, atol=1e-02)

            # Check output shapes - sequence length is divided by CP size
            assert output.shape[0] == self.micro_batch_size
            assert output.shape[1] == self.seq_length / cp

            # Verify gradients
            loss = output.mean()
            loss.backward()
            # for param in gpt_model[0].parameters():
            for name, param in gpt_model[0].named_parameters():
                assert param.main_grad is not None

    @pytest.mark.skipif(
        not HAVE_TE or not is_te_min_version("2.1.0"),
        reason="grouped_gemm requires TransformerEngine >= 2.1.0",
    )
    @pytest.mark.parametrize("full_recompute", [False, True])
    @pytest.mark.parametrize(("tp", "cp"), [(1, 1), (2, 1), (2, 2)])
    def test_forward_backward_with_hsm(self, tp, cp, full_recompute):
        """Run Hidden State Mixing through a real GPTModel, not a stubbed block.

        ``TestMTPHiddenStateRollUnderParallelism`` pins which token each rolled hidden
        state predicts, but it does so with stub layers and a ``SimpleNamespace``, so a
        whole class of interaction never appears there: real TE layers, bf16 activations,
        the sequence-parallel scatter that actually produces the sharded hidden states,
        and -- with ``full_recompute`` -- an HSM selection drawn inside a region that gets
        recomputed in backward. ``tp=2`` turns sequence parallelism on, so the cell that
        matters most is ``tp=2, cp=2``.

        Values are not compared against a non-HSM reference: mixing changes them by
        design. What must hold is that the shapes are untouched and that every parameter
        still receives a finite gradient.
        """
        if int(os.environ.get("WORLD_SIZE", "1")) < tp * cp:
            pytest.skip(f"TP={tp} x CP={cp} requires at least {tp * cp} ranks")

        args = self.create_test_args(
            tp,
            cp,
            self.seq_length,
            self.micro_batch_size,
            full_recompute=full_recompute,
            mtp_hsm=True,
        )
        set_args(args)
        assert args.mtp_hsm, "validate_args disabled HSM, so this test would prove nothing"
        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)
        model_parallel_cuda_manual_seed(_SEED)

        batch = self.get_batch(self.seq_length, self.micro_batch_size)
        batch = get_batch_on_this_cp_rank(
            batch, is_hybrid_cp=False, cp_group=get_context_parallel_group()
        )
        gpt_model, _, _ = setup_model_and_optimizer(
            ModelType.encoder_or_decoder, self.model_provider
        )
        assert unwrap_model(gpt_model[0]).mtp.config.mtp_hsm
        assert gpt_model[0].training, "HSM only runs in training mode"

        output = gpt_model[0].forward(
            input_ids=batch['tokens'],
            position_ids=batch['position_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
            loss_mask=batch['loss_mask'],
        )

        assert output.shape == (self.micro_batch_size, self.seq_length // cp)
        assert torch.isfinite(output).all()

        tracker = MTPLossLoggingHelper.tracker
        assert "loss_values" in tracker
        mtp_loss = tracker['loss_values'].clone()
        assert mtp_loss.shape[0] == args.mtp_num_layers
        assert torch.isfinite(mtp_loss).all()

        output.mean().backward()
        for name, param in gpt_model[0].named_parameters():
            assert param.main_grad is not None, f"Gradient missing for {name}"
            assert torch.isfinite(param.main_grad).all(), f"Non-finite gradient for {name}"

    @pytest.mark.flaky_in_dev
    @pytest.mark.skipif(
        not HAVE_TE or not is_te_min_version("1.7.0"),
        reason="Only transformer-engine>=1.7.0 supports MoE FP8 training",
    )
    @pytest.mark.parametrize("full_recompute", [False, True])
    def test_fp8_support(self, full_recompute):
        """Test MTP with FP8 training enabled."""
        tp = 1
        cp = 1
        fp8 = 'e4m3'
        args = self.create_test_args(
            tp, cp, self.seq_length, self.micro_batch_size, fp8, full_recompute=full_recompute
        )
        set_args(args)

        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)
        batch = self.get_batch(self.seq_length, self.micro_batch_size)
        tokens, labels, loss_mask, attention_mask, position_ids = batch.values()
        gpt_model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
            ModelType.encoder_or_decoder, self.model_provider
        )

        output = gpt_model[0].forward(
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            loss_mask=loss_mask,
        )

        assert output.dtype == torch.float32  # Output should be converted back to float32

        loss = output.mean()
        loss.backward()

    @pytest.mark.skipif(
        not HAVE_TE or not is_te_min_version("2.1.0"),
        reason="grouped_gemm requires TransformerEngine >= 2.1.0",
    )
    @pytest.mark.parametrize(("tp", "cp"), [(1, 1), (2, 1), (2, 2)])
    def test_packed_sequences(self, tp, cp):
        """Test MTP with packed sequences."""
        # Create args with packed sequences support
        seq_lengths = [16, 24, 12]  # Three sequences of different lengths
        total_seq_length = sum(seq_lengths)

        args = self.create_test_args(tp, cp, total_seq_length, micro_batch_size=1)
        set_args(args)

        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)

        # Get packed batch
        batch = self.get_packed_batch(seq_lengths, micro_batch_size=1)
        tokens = batch['tokens']
        labels = batch['labels']
        loss_mask = batch['loss_mask']
        attention_mask = batch['attention_mask']
        position_ids = batch['position_ids']
        packed_seq_params = batch['packed_seq_params']

        # Create model
        model_parallel_cuda_manual_seed(_SEED)
        cfg_container = Utils.pretrain_config_from_global_args(args, "gpt")
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        gpt_model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
            ModelType.encoder_or_decoder,
            self.model_provider,
            cfg_container=cfg_container,
            pg_collection=pg_collection,
        )

        # Forward pass with packed sequences
        output = gpt_model[0].forward(
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            loss_mask=loss_mask,
            packed_seq_params=packed_seq_params,
        )

        # Verify output shape
        assert output.shape[0] == 1  # batch size
        assert output.shape[1] == total_seq_length

        # Verify MTP loss was computed; reduce local contributions first.
        MTPLossLoggingHelper.reduce_loss_in_tracker()
        tracker = MTPLossLoggingHelper.tracker
        assert "values" in tracker
        mtp_loss = tracker['values'].clone()
        assert mtp_loss.shape[0] == args.mtp_num_layers
        MTPLossLoggingHelper.clean_loss_in_tracker()

        # Backward pass
        loss = output.mean()
        loss.backward()

        # Verify gradients exist
        for name, param in gpt_model[0].named_parameters():
            assert param.main_grad is not None, f"Gradient missing for {name}"

    @pytest.mark.flaky_in_dev
    @pytest.mark.skipif(
        not HAVE_TE or not is_te_min_version("2.1.0"),
        reason="grouped_gemm requires TransformerEngine >= 2.1.0",
    )
    def test_packed_sequences_with_full_recompute(self):
        """MTP + packed sequences + full activation recomputation.

        Regression: MTP._checkpointed_forward used to forward
        ``packed_seq_params`` (a non-tensor PackedSeqParams object) directly
        to ``tensor_parallel.checkpoint``. CheckpointFunction.save_for_backward
        only accepts tensors and ``None``, so this raised
        ``TypeError: save_for_backward can only save variables, but argument
        N is of type PackedSeqParams``. Non-tensor kwargs must be captured
        by closure, not forwarded as args.
        """
        seq_lengths = [16, 24, 12]
        total_seq_length = sum(seq_lengths)

        args = self.create_test_args(
            tp=1, cp=1, sequence_length=total_seq_length, micro_batch_size=1, full_recompute=True
        )
        set_args(args)

        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)

        batch = self.get_packed_batch(seq_lengths, micro_batch_size=1)

        model_parallel_cuda_manual_seed(_SEED)
        cfg_container = Utils.pretrain_config_from_global_args(args, "gpt")
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        gpt_model, _, _ = setup_model_and_optimizer(
            ModelType.encoder_or_decoder,
            self.model_provider,
            cfg_container=cfg_container,
            pg_collection=pg_collection,
        )

        output = gpt_model[0].forward(
            input_ids=batch['tokens'],
            position_ids=batch['position_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
            loss_mask=batch['loss_mask'],
            packed_seq_params=batch['packed_seq_params'],
        )

        # Backward must run end-to-end through the recomputed MTP layer.
        loss = output.mean()
        loss.backward()

        for name, param in gpt_model[0].named_parameters():
            assert param.main_grad is not None, f"Gradient missing for {name}"

    def test_roll_tensor_empty_input(self):
        """An empty field group is a no-op."""
        assert roll_tensor([]) == []

    def test_roll_tensor_return_sum_false(self):
        """Disabling return_sum preserves values and omits the unused reduction."""
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
        tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device="cuda")

        rolled, rolled_sum = roll_tensor(tensor, shifts=-1, dims=-1)
        rolled_without_sum, disabled_sum = roll_tensor(tensor, shifts=-1, dims=-1, return_sum=False)

        assert torch.equal(rolled_without_sum, rolled)
        assert torch.equal(rolled_sum, rolled.sum())
        assert disabled_sum is None
        Utils.destroy_model_parallel()

    def test_roll_tensor_shifts_left_and_zeroes_last(self):
        """Test that roll_tensor(-1) shifts left and zeroes the last position.

        This is the primitive used to derive MTP labels from input_ids when labels
        are not provided (RL training): label[i] = input_id[i+1], last position zeroed.
        The end-to-end derivation is covered by process_mtp_loss (see input_ids path).
        """
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
        # Simulate input_ids [batch=2, seq=5]
        input_ids = torch.tensor(
            [[10, 20, 30, 40, 50], [60, 70, 80, 90, 100]], dtype=torch.int64
        ).cuda()
        rolled = roll_tensor([input_ids], shifts=-1, dims=-1)[0]

        # Expected: each row shifted left by 1, last element zeroed.
        expected = torch.tensor(
            [[20, 30, 40, 50, 0], [70, 80, 90, 100, 0]], dtype=torch.int64
        ).cuda()
        assert torch.equal(rolled, expected)

        Utils.destroy_model_parallel()

    def test_roll_tensor_groups_multiple_fields(self):
        """One dispatcher call rolls fields with different dtypes and fill values."""
        input_ids = torch.tensor([[10, 20, 30]])
        padding_mask = torch.tensor([[False, False, False]])
        rolled_input_ids, rolled_padding_mask = roll_tensor(
            [input_ids, padding_mask], fill_values=[0, True]
        )

        assert torch.equal(rolled_input_ids, torch.tensor([[20, 30, 0]]))
        assert torch.equal(rolled_padding_mask, torch.tensor([[False, False, True]]))

    def test_roll_tensor_packed_cp1_vectorizes_sequence_boundaries(self, monkeypatch):
        """Packed CP1 uses one full-buffer roll per field, including duplicate boundaries."""
        input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        padding_mask = torch.zeros(5, dtype=torch.bool)
        cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)
        cu_seqlens_padded = torch.tensor([0, 3, 5, 5, 5], dtype=torch.int32)
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens_padded,
            cu_seqlens_kv_padded=cu_seqlens_padded,
            max_seqlen_q=3,
            max_seqlen_kv=3,
            qkv_format='thd',
        )

        original_roll = torch.roll
        roll_calls = 0

        def counted_roll(*args, **kwargs):
            nonlocal roll_calls
            roll_calls += 1
            return original_roll(*args, **kwargs)

        monkeypatch.setattr(torch, "roll", counted_roll)
        rolled_input_ids, rolled_padding_mask = roll_tensor(
            [input_ids, padding_mask], packed_seq_params=packed_seq_params, fill_values=[0, True]
        )

        # The number of roll kernels depends on the field count, not the number
        # of packed sequences. Repeated static-padding endpoints are harmless.
        assert roll_calls == 2
        assert torch.equal(rolled_input_ids, torch.tensor([2, 3, 0, 5, 0]))
        assert torch.equal(rolled_padding_mask, torch.tensor([False, False, True, False, True]))
        assert torch.equal(input_ids, torch.tensor([1, 2, 3, 4, 5]))
        assert torch.equal(padding_mask, torch.zeros(5, dtype=torch.bool))

    def test_roll_tensor_packed_cp1_fills_implicit_tail_end(self):
        """A physical tail beyond the last packed boundary cannot wrap to token zero."""
        input_ids = torch.tensor([[10, 20, 30, 40, 0, 0]], dtype=torch.long)
        loss_mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.float32)
        cu_seqlens = torch.tensor([0, 2, 4], dtype=torch.int32)
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=2,
            max_seqlen_kv=2,
            qkv_format="thd",
            total_tokens=6,
        )

        rolled_input_ids, rolled_loss_mask = roll_tensor(
            [input_ids, loss_mask], packed_seq_params=packed_seq_params, fill_values=[0, 0]
        )

        assert torch.equal(rolled_input_ids, torch.tensor([[20, 0, 40, 0, 0, 0]]))
        assert torch.equal(
            rolled_loss_mask, torch.tensor([[1, 0, 1, 0, 0, 0]], dtype=torch.float32)
        )

    def test_process_mtp_loss_skips_when_no_labels_and_no_input_ids(self):
        """When labels and input_ids are both None, MTP loss is skipped (early return)."""
        config = TransformerConfig(
            hidden_size=8, num_layers=2, num_attention_heads=2, mtp_num_layers=1
        )
        hidden_states = torch.ones(2, 1, 4)
        called = {'value': False}

        def output_layer(hidden, weight=None, runtime_gather_output=None):
            return hidden.clone(), None

        def compute_language_model_loss(mtp_labels, mtp_logits):
            called['value'] = True
            return torch.ones_like(mtp_labels, dtype=mtp_logits.dtype)

        out = process_mtp_loss(
            hidden_states=hidden_states,
            labels=None,
            loss_mask=None,
            output_layer=output_layer,
            output_weight=None,
            runtime_gather_output=None,
            is_training=False,
            compute_language_model_loss=compute_language_model_loss,
            config=config,
            cp_group=None,
            packed_seq_params=None,
            input_ids=None,
        )

        # First chunk is returned unchanged and the loss is never computed.
        assert not called['value']
        assert torch.equal(out, torch.chunk(hidden_states, 2, dim=0)[0])

    @pytest.mark.parametrize(
        ("fuse_linear_cross_entropy", "acceptance_consumer"), [(False, False), (True, True)]
    )
    def test_process_mtp_loss_skips_unavailable_acceptance(
        self, monkeypatch, fuse_linear_cross_entropy, acceptance_consumer
    ):
        """No consumer and fused linear CE avoid acceptance work."""
        config = TransformerConfig(
            hidden_size=8, num_layers=2, num_attention_heads=2, mtp_num_layers=1
        )
        config.cross_entropy_loss_fusion = fuse_linear_cross_entropy
        config.cross_entropy_fusion_impl = "linear" if fuse_linear_cross_entropy else "native"

        seq_len = 4
        batch_size = 1
        vocab_size = 16
        hidden_states = torch.randn(
            (1 + config.mtp_num_layers) * seq_len, batch_size, config.hidden_size, device="cuda"
        )
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")
        loss_mask = torch.ones(batch_size, seq_len, device="cuda")

        class OutputLayer:
            gather_output = True

            def __call__(
                self,
                hidden,
                weight=None,
                runtime_gather_output=None,
                output_cross_entropy_loss=False,
                labels=None,
            ):
                del weight, runtime_gather_output
                if output_cross_entropy_loss:
                    return torch.ones_like(labels, dtype=hidden.dtype)
                return (
                    torch.zeros(hidden.size(0), hidden.size(1), vocab_size, device=hidden.device),
                    None,
                )

        def fail_if_acceptance_is_computed(*args, **kwargs):
            raise AssertionError("acceptance computation should have been skipped")

        monkeypatch.setattr(
            "megatron.core.transformer.multi_token_prediction._compute_mtp_acceptance_counts",
            fail_if_acceptance_is_computed,
        )
        monkeypatch.setattr(
            "megatron.core.transformer.multi_token_prediction.parallel_state."
            "get_data_parallel_group",
            lambda **kwargs: None,
        )

        MTPLossLoggingHelper.configure_acceptance_collection(enabled=acceptance_consumer)
        process_mtp_loss(
            hidden_states=hidden_states,
            labels=labels,
            loss_mask=loss_mask,
            output_layer=OutputLayer(),
            output_weight=None,
            runtime_gather_output=None,
            is_training=True,
            compute_language_model_loss=lambda labels, logits: torch.ones_like(
                labels, dtype=logits.dtype
            ),
            config=config,
        )

        assert "loss_sums" in MTPLossLoggingHelper.tracker
        assert "acceptance_counts" not in MTPLossLoggingHelper.tracker

    def test_process_mtp_loss_derives_labels_from_input_ids(self):
        """When labels is None (RL), labels are derived from input_ids by rolling left.

        process_mtp_loss rolls once to build the SFT-format labels (label[i] =
        input_id[i+1]) and once more per MTP layer, so MTP head 0 targets input_id[i+2].
        The loss_mask is rolled in lockstep so the fabricated trailing label is masked.
        """
        config = TransformerConfig(
            hidden_size=8, num_layers=2, num_attention_heads=2, mtp_num_layers=1
        )
        # hidden_states is chunked into (1 + mtp_num_layers) along dim 0.
        hidden_states = torch.ones(2, 1, 5)
        input_ids = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)
        seen = {'labels': None, 'masked_loss': None}

        def output_layer(hidden, weight=None, runtime_gather_output=None):
            return hidden.clone(), None

        def compute_language_model_loss(mtp_labels, mtp_logits):
            seen['labels'] = mtp_labels.clone()
            # Per-position loss of 1.0 so loss_mask * loss exposes the active mask.
            return torch.ones_like(mtp_labels, dtype=torch.float32)

        process_mtp_loss(
            hidden_states=hidden_states,
            labels=None,
            loss_mask=None,
            output_layer=output_layer,
            output_weight=None,
            runtime_gather_output=None,
            is_training=False,
            compute_language_model_loss=compute_language_model_loss,
            config=config,
            cp_group=None,
            packed_seq_params=None,
            input_ids=input_ids,
        )

        # input_ids rolled twice (once to SFT format, once in the MTP layer loop):
        # [10,20,30,40,50] -> [20,30,40,50,0] -> [30,40,50,0,0].
        assert seen['labels'] is not None, "loss should be computed in RL mode"
        assert torch.equal(seen['labels'], torch.tensor([[30, 40, 50, 0, 0]], dtype=torch.long))

    @pytest.mark.parametrize("cp", [1, 2])
    def test_roll_tensor_with_packed_sequences(self, cp):
        """Test roll_tensor function with packed sequences, with and without CP.

        For CP=1: Tests standard packed sequence rolling with verified expected values
        For CP=2: Tests CP-enabled rolling executes without errors
        """
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=cp)
        cp_group = get_context_parallel_group() if cp > 1 else None
        cp_rank = torch.distributed.get_rank(group=cp_group) if cp_group is not None else 0

        if cp == 1:
            # Test case: Simple packed sequences (CP disabled)
            tensor = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32).cuda()
            cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32).cuda()

            packed_seq_params = PackedSeqParams(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                max_seqlen_q=3,
                max_seqlen_kv=3,
                qkv_format='thd',
            )

            # Roll by -1 (shift left)
            rolled = roll_tensor(
                [tensor], shifts=-1, dims=0, cp_group=cp_group, packed_seq_params=packed_seq_params
            )[0]

            # Expected: [2, 3, 0, 5, 0] - boundaries at indices 2 and 4 are zeroed
            expected = torch.tensor([2, 3, 0, 5, 0], dtype=torch.float32).cuda()
            assert torch.equal(rolled, expected), f"Expected {expected}, got {rolled}"
        else:
            # Test case: Packed sequences with CP=2
            # Two sequences:
            #   seq1 = [1, 2, 3, 4, 5, 6, 7, 8]
            #   seq2 = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

            if cp_rank == 0:
                # CP Rank 0: first half of each sequence
                tensor = torch.tensor(
                    [1, 2, 7, 8, 11, 12, 13, 20, 21, 22], dtype=torch.float32
                ).cuda()
                expected = torch.tensor(
                    [2, 3, 8, 0, 12, 13, 14, 21, 22, 0], dtype=torch.float32
                ).cuda()
            else:
                # CP Rank 1: second half of each sequence
                tensor = torch.tensor(
                    [3, 4, 5, 6, 14, 15, 16, 17, 18, 19], dtype=torch.float32
                ).cuda()
                expected = torch.tensor(
                    [4, 5, 6, 7, 15, 16, 17, 18, 19, 20], dtype=torch.float32
                ).cuda()

            cu_seqlens = torch.tensor([0, 8, 20], dtype=torch.int32).cuda()

            packed_seq_params = PackedSeqParams(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                max_seqlen_q=6,  # max(4, 6) - max local seq length per sequence
                max_seqlen_kv=6,
                qkv_format='thd',
            )

            # Roll by -1 (shift left) with CP communication
            rolled = roll_tensor(
                [tensor], shifts=-1, dims=0, cp_group=cp_group, packed_seq_params=packed_seq_params
            )[0]

            # Verify the rolled tensor matches expected values
            assert (
                rolled.shape == expected.shape
            ), f"Shape mismatch: expected {expected.shape}, got {rolled.shape}"
            assert torch.equal(
                rolled, expected
            ), f"CP Rank {cp_rank}: Expected\n{expected}\nbut got\n{rolled}\nDiff:\n{rolled - expected}"

        Utils.destroy_model_parallel()

    def test_roll_tensor_with_packed_sequences_contiguous_cp(self, monkeypatch):
        """Contiguous THD CP rolls across rank boundaries without crossing sequence boundaries."""
        cp = 2
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=cp)
        cp_group = get_context_parallel_group()
        cp_rank = torch.distributed.get_rank(group=cp_group)

        # Full padded layout:
        #   seq1: [1,2,3,4,5,6,7,0]
        #   seq2: [11,12,13,14,15,16,17,18,19,20,21,0]
        # Contiguous CP rank 0 owns global rows [0, 10), rank 1 owns [10, 20).
        if cp_rank == 0:
            tensor = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 0, 11, 12]], dtype=torch.float32).cuda()
            expected = torch.tensor([[2, 3, 4, 5, 6, 7, 0, 0, 12, 13]], dtype=torch.float32).cuda()
            padding_mask = torch.tensor(
                [[False, False, False, False, False, False, False, True, False, False]]
            ).cuda()
            expected_padding_mask = torch.tensor(
                [[False, False, False, False, False, False, True, True, False, False]]
            ).cuda()
        else:
            tensor = torch.tensor(
                [[13, 14, 15, 16, 17, 18, 19, 20, 21, 0]], dtype=torch.float32
            ).cuda()
            expected = torch.tensor(
                [[14, 15, 16, 17, 18, 19, 20, 21, 0, 0]], dtype=torch.float32
            ).cuda()
            padding_mask = torch.tensor(
                [[False, False, False, False, False, False, False, False, False, True]]
            ).cuda()
            expected_padding_mask = torch.tensor(
                [[False, False, False, False, False, False, False, False, True, True]]
            ).cuda()

        cu_seqlens = torch.tensor([0, 7, 18], dtype=torch.int32).cuda()
        cu_seqlens_padded = torch.tensor([0, 8, 20], dtype=torch.int32).cuda()
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens_padded,
            cu_seqlens_kv_padded=cu_seqlens_padded,
            max_seqlen_q=11,
            max_seqlen_kv=11,
            qkv_format='thd',
            cp_partition_mode='contiguous',
        )

        sequence_roll_context = prepare_mtp_sequence_roll_context(
            tensor=tensor, cp_group=cp_group, packed_seq_params=packed_seq_params, dims=-1
        )
        assert isinstance(sequence_roll_context, ContiguousPackedCPRollContext)
        assert isinstance(sequence_roll_context.plan, ContiguousPackedSeqRollPlan)
        batch_isend_irecv = torch.distributed.batch_isend_irecv
        grouped_p2p_calls = 0

        def counted_batch_isend_irecv(p2p_ops):
            nonlocal grouped_p2p_calls
            grouped_p2p_calls += 1
            return batch_isend_irecv(p2p_ops)

        monkeypatch.setattr(torch.distributed, "batch_isend_irecv", counted_batch_isend_irecv)
        rolled, rolled_padding_mask = roll_tensor(
            [tensor, padding_mask],
            shifts=-1,
            dims=-1,
            cp_group=cp_group,
            packed_seq_params=packed_seq_params,
            fill_values=[0, True],
            roll_context=sequence_roll_context,
        )
        assert grouped_p2p_calls == 1

        assert torch.equal(rolled, expected), (
            f"CP Rank {cp_rank}: Expected\n{expected}\nbut got\n{rolled}\nDiff:\n"
            f"{rolled - expected}"
        )
        assert torch.equal(rolled_padding_mask, expected_padding_mask), (
            f"CP Rank {cp_rank}: Expected padding mask\n{expected_padding_mask}\nbut got\n"
            f"{rolled_padding_mask}"
        )

        Utils.destroy_model_parallel()

    def test_contiguous_packed_cp_prefetches_halos_once_before_mtp(self, monkeypatch):
        """One pre-MTP P2P supplies all depths and masks remote packed boundaries."""
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=2)
        cp_group = get_context_parallel_group()
        cp_rank = torch.distributed.get_rank(group=cp_group)

        padding_mask = torch.zeros((1, 4), dtype=torch.bool).cuda()
        if cp_rank == 0:
            tensor = torch.tensor([[1, 2, 3, 4]], dtype=torch.long).cuda()
            expected_halo = torch.tensor([[5, 6, 0]], dtype=torch.long).cuda()
            expected_padding_halo = torch.tensor([[False, False, True]]).cuda()
            expected_by_depth = [
                torch.tensor([[2, 3, 4, 5]], dtype=torch.long).cuda(),
                torch.tensor([[3, 4, 5, 6]], dtype=torch.long).cuda(),
                torch.tensor([[4, 5, 6, 0]], dtype=torch.long).cuda(),
            ]
        else:
            tensor = torch.tensor([[5, 6, 7, 8]], dtype=torch.long).cuda()
            expected_halo = torch.zeros((1, 3), dtype=torch.long).cuda()
            expected_padding_halo = torch.ones((1, 3), dtype=torch.bool).cuda()
            expected_by_depth = [
                torch.tensor([[6, 0, 8, 0]], dtype=torch.long).cuda(),
                torch.zeros((1, 4), dtype=torch.long).cuda(),
                torch.zeros((1, 4), dtype=torch.long).cuda(),
            ]

        # The first physical sequence crosses the CP boundary but ends after two
        # rows on rank 1. Halo offset two must therefore be filled, not token 7.
        cu_seqlens = torch.tensor([0, 6, 8], dtype=torch.int32).cuda()
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens,
            cu_seqlens_kv_padded=cu_seqlens,
            max_seqlen_q=6,
            max_seqlen_kv=6,
            qkv_format="thd",
            cp_partition_mode="contiguous",
        )
        sequence_roll_context = prepare_mtp_sequence_roll_context(
            tensor=tensor, cp_group=cp_group, packed_seq_params=packed_seq_params
        )
        assert isinstance(sequence_roll_context, ContiguousPackedCPRollContext)

        batch_isend_irecv = torch.distributed.batch_isend_irecv
        grouped_p2p_calls = 0

        def counted_batch_isend_irecv(p2p_ops):
            nonlocal grouped_p2p_calls
            grouped_p2p_calls += 1
            return batch_isend_irecv(p2p_ops)

        monkeypatch.setattr(torch.distributed, "batch_isend_irecv", counted_batch_isend_irecv)
        sequence_roll_context = sequence_roll_context.prefetch_halos(
            width=3, input_ids=tensor, padding_mask=padding_mask
        )
        assert grouped_p2p_calls == 1
        assert isinstance(sequence_roll_context, ContiguousPackedCPRollContext)
        assert isinstance(sequence_roll_context.halos, ContiguousPackedCPRollHalos)
        assert sequence_roll_context.halos.width == 3
        assert sequence_roll_context.halos.input_ids._base is None
        assert sequence_roll_context.halos.padding_mask._base is None
        assert torch.equal(sequence_roll_context.halos.input_ids, expected_halo)
        assert torch.equal(sequence_roll_context.halos.padding_mask, expected_padding_halo)

        def unexpected_p2p(_):
            raise AssertionError("Prefetched contiguous CP halos must bypass rolling P2P.")

        monkeypatch.setattr(torch.distributed, "batch_isend_irecv", unexpected_p2p)
        rolled = tensor
        for depth, expected in enumerate(expected_by_depth):
            rolled = roll_tensor(
                [rolled],
                cp_group=cp_group,
                packed_seq_params=packed_seq_params,
                roll_context=sequence_roll_context,
                sequence_fields=["input_ids"],
                roll_depth=depth,
            )[0]
            assert torch.equal(rolled, expected)

        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(("cp_size", "partition_mode"), [(1, "contiguous"), (2, "zigzag")])
    def test_prepare_mtp_sequence_roll_context_skips_other_layouts(self, cp_size, partition_mode):
        """CP1 and zigzag keep their existing roll paths without prepared state."""

        class FakeCPGroup:
            def size(self):
                return cp_size

        cu_seqlens = torch.tensor([0, 4], dtype=torch.int32)
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=4,
            max_seqlen_kv=4,
            qkv_format="thd",
            cp_partition_mode=partition_mode,
        )
        tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

        sequence_roll_context = prepare_mtp_sequence_roll_context(
            tensor=tokens, cp_group=FakeCPGroup(), packed_seq_params=packed_seq_params
        )

        assert sequence_roll_context is None

    @pytest.mark.parametrize("cp", [1, 2])
    def test_roll_tensor_with_packed_sequences_odd_seqlen(self, cp):
        """Test roll_tensor with ODD packed seqlens.

        For CP=1: per-sequence rolling on contiguous packed tensor — odd seqlens are fine
                  with cu_seqlens_q alone (no padding required).
        For CP=2: each per-sequence padded length must be a multiple of 2*cp_size, so odd
                  seqlens require padding. The local THD-CP layout is determined by
                  cu_seqlens_q_padded; the roll function must use the padded boundaries to
                  index local chunks correctly. Without the padded boundaries, real tokens
                  leak across sequence boundaries.
        """
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=cp)
        cp_group = get_context_parallel_group() if cp > 1 else None
        cp_rank = torch.distributed.get_rank(group=cp_group) if cp_group is not None else 0

        if cp == 1:
            # Two odd-length sequences: [3, 5]. Total = 8.
            tensor = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.float32).cuda()
            cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32).cuda()

            packed_seq_params = PackedSeqParams(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                max_seqlen_q=5,
                max_seqlen_kv=5,
                qkv_format='thd',
            )

            rolled = roll_tensor(
                [tensor], shifts=-1, dims=0, cp_group=cp_group, packed_seq_params=packed_seq_params
            )[0]

            # seq1 [1,2,3] -> [2,3,0]; seq2 [4,5,6,7,8] -> [5,6,7,8,0]
            expected = torch.tensor([2, 3, 0, 5, 6, 7, 8, 0], dtype=torch.float32).cuda()
            assert torch.equal(rolled, expected), f"Expected {expected}, got {rolled}"
        else:
            # Two ODD sequences padded up to multiples of 2*cp_size = 4:
            #   seq1: real=[1..7] (len 7), padded with 0 -> [1,2,3,4,5,6,7,0] (len 8)
            #   seq2: real=[11..21] (len 11), padded with 0 ->
            #         [11,12,13,14,15,16,17,18,19,20,21,0] (len 12)
            # Zigzag (4 chunks per padded seq, rank r owns chunks (r, 3-r)):
            #   seq1 chunks: [1,2], [3,4], [5,6], [7,0]
            #     rank 0 -> [1,2, 7,0];  rank 1 -> [3,4, 5,6]
            #   seq2 chunks: [11,12,13], [14,15,16], [17,18,19], [20,21,0]
            #     rank 0 -> [11,12,13, 20,21,0]; rank 1 -> [14,15,16, 17,18,19]
            # Expected after roll(-1) within unpadded region (last real -> 0; pad stays 0):
            #   seq1 rolled real: [2,3,4,5,6,7,0]; padded last -> 0
            #   seq2 rolled real: [12,13,14,15,16,17,18,19,20,21,0]; padded last -> 0
            # Re-zigzag the rolled+padded seqs:
            #   seq1: [2,3], [4,5], [6,7], [0,0]
            #     rank 0 -> [2,3, 0,0];  rank 1 -> [4,5, 6,7]
            #   seq2: [12,13,14], [15,16,17], [18,19,20], [21,0,0]
            #     rank 0 -> [12,13,14, 21,0,0]; rank 1 -> [15,16,17, 18,19,20]
            if cp_rank == 0:
                tensor = torch.tensor(
                    [1, 2, 7, 0, 11, 12, 13, 20, 21, 0], dtype=torch.float32
                ).cuda()
                expected = torch.tensor(
                    [2, 3, 0, 0, 12, 13, 14, 21, 0, 0], dtype=torch.float32
                ).cuda()
            else:
                tensor = torch.tensor(
                    [3, 4, 5, 6, 14, 15, 16, 17, 18, 19], dtype=torch.float32
                ).cuda()
                expected = torch.tensor(
                    [4, 5, 6, 7, 15, 16, 17, 18, 19, 20], dtype=torch.float32
                ).cuda()

            # Unpadded cu_seqlens_q = [0, 7, 18]; padded = [0, 8, 20].
            cu_seqlens = torch.tensor([0, 7, 18], dtype=torch.int32).cuda()
            cu_seqlens_padded = torch.tensor([0, 8, 20], dtype=torch.int32).cuda()

            packed_seq_params = PackedSeqParams(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                cu_seqlens_q_padded=cu_seqlens_padded,
                cu_seqlens_kv_padded=cu_seqlens_padded,
                max_seqlen_q=11,
                max_seqlen_kv=11,
                qkv_format='thd',
            )

            rolled = roll_tensor(
                [tensor], shifts=-1, dims=0, cp_group=cp_group, packed_seq_params=packed_seq_params
            )[0]

            assert (
                rolled.shape == expected.shape
            ), f"Shape mismatch: expected {expected.shape}, got {rolled.shape}"
            assert torch.equal(
                rolled, expected
            ), f"CP Rank {cp_rank}: Expected\n{expected}\nbut got\n{rolled}\nDiff:\n{rolled - expected}"

        Utils.destroy_model_parallel()


class TestMTPLossLoggingHelper:
    def setup_method(self, method):
        self.num_layers = 4
        # Reset the tracker before each test
        MTPLossLoggingHelper.tracker = {}

    def teardown_method(self, method):
        # Clean up the tracker after each test
        MTPLossLoggingHelper.tracker = {}

    def test_save_metrics_to_tracker(self):
        """Test saving metrics to tracker."""
        loss = torch.tensor(1.3)
        correct = torch.tensor(5.0)
        total = torch.tensor(10.0)
        layer_number = 2
        num_layers = self.num_layers

        MTPLossLoggingHelper.save_metrics_to_tracker(
            loss=loss,
            correct=correct,
            total=total,
            layer_number=layer_number,
            num_layers=num_layers,
        )

        tracker = MTPLossLoggingHelper.tracker
        assert "loss_values" in tracker
        assert tracker["loss_values"].shape == (num_layers,)
        assert tracker["loss_values"][layer_number] == loss
        assert tracker["correct_values"][layer_number] == correct
        assert tracker["total_values"][layer_number] == total
        assert tracker["reduce_group"] is None
        assert tracker["avg_group"] is None

    def test_save_loss_to_tracker(self):
        """Test saving a legacy normalized loss to the tracker."""
        loss_sum = torch.tensor(1.3)
        num_tokens = torch.tensor(5.0)
        layer_number = 2
        num_layers = self.num_layers

        MTPLossLoggingHelper.save_loss_to_tracker(
            loss_sum=loss_sum,
            num_tokens=num_tokens,
            layer_number=layer_number,
            num_layers=num_layers,
        )

        assert "loss_sums" in MTPLossLoggingHelper.tracker
        assert MTPLossLoggingHelper.tracker["loss_sums"].shape == (num_layers,)
        assert torch.isclose(
            MTPLossLoggingHelper.tracker["loss_sums"][layer_number], loss_sum / num_tokens
        )
        assert MTPLossLoggingHelper.tracker["reduce_group"] is None
        assert MTPLossLoggingHelper.tracker["avg_group"] is None

    def test_save_loss_to_tracker_per_token_stores_raw_loss_and_tokens(self):
        """Per-token logging keeps raw sums so reduction can weight by tokens."""
        loss_sum = torch.tensor(1.3)
        num_tokens = torch.tensor(5.0)
        layer_number = 2
        num_layers = self.num_layers

        MTPLossLoggingHelper.save_loss_to_tracker(
            loss_sum=loss_sum,
            num_tokens=num_tokens,
            layer_number=layer_number,
            num_layers=num_layers,
            calculate_per_token_loss=True,
        )

        tracker = MTPLossLoggingHelper.tracker
        assert "loss_sums" in tracker
        assert "num_tokens" in tracker
        assert tracker["calculate_per_token_loss"] is True
        assert torch.isclose(tracker["loss_sums"][layer_number], loss_sum)
        assert torch.isclose(tracker["num_tokens"][layer_number], num_tokens)
        assert tracker["reduce_group"] is None
        assert tracker["avg_group"] is None

    def test_acceptance_collection_preserves_all_microbatch_semantics(self):
        """Enabled collection applies to every microbatch; disabled collection applies to none."""
        # Unconfigured standalone callers retain the legacy collect-every-call behavior.
        assert MTPLossLoggingHelper.should_collect_acceptance()

        MTPLossLoggingHelper.configure_acceptance_collection(enabled=True)
        assert MTPLossLoggingHelper.should_collect_acceptance()

        MTPLossLoggingHelper.configure_acceptance_collection(enabled=False)
        assert not MTPLossLoggingHelper.should_collect_acceptance()

    def test_acceptance_counts_use_one_packed_reduction(self, monkeypatch):
        """Correct and total share one SUM collective and retain graph-replay metadata."""
        fake_group = object()
        all_reduce_calls = []

        def record_all_reduce(tensor, group=None, op=None):
            all_reduce_calls.append((tensor, group, op))

        monkeypatch.setattr(torch.distributed, "all_reduce", record_all_reduce)
        MTPLossLoggingHelper.save_loss_to_tracker(
            loss_sum=torch.tensor(1.0, device="cuda"),
            num_tokens=torch.tensor(1.0, device="cuda"),
            correct=torch.tensor(3.0, device="cuda"),
            total=torch.tensor(5.0, device="cuda"),
            layer_number=0,
            num_layers=2,
            avg_group=fake_group,
        )

        MTPLossLoggingHelper.reduce_metrics_in_tracker()
        assert len(all_reduce_calls) == 1
        reduced_tensor, reduced_group, reduced_op = all_reduce_calls[0]
        assert reduced_tensor.shape == (2, 2)
        assert reduced_group is fake_group
        assert reduced_op == torch.distributed.ReduceOp.SUM

        # Full-iteration graph replay updates the captured counter tensor without
        # rerunning Python tracker setup, so per-step cleanup must retain the group.
        MTPLossLoggingHelper.clean_loss_in_tracker()
        assert MTPLossLoggingHelper.tracker["acceptance_avg_group"] is fake_group
        MTPLossLoggingHelper.tracker["acceptance_counts"][:, 0].copy_(
            torch.tensor([4.0, 8.0], device="cuda")
        )
        MTPLossLoggingHelper.reduce_metrics_in_tracker()
        assert len(all_reduce_calls) == 2
        assert all_reduce_calls[1][1] is fake_group
        assert all_reduce_calls[1][2] == torch.distributed.ReduceOp.SUM

    def test_acceptance_reports_each_step_and_cumulates_all_steps(self):
        """The legacy names keep per-step and all-steps, all-microbatch semantics."""

        class DummyWriter:
            def __init__(self):
                self.scalars = {}

            def add_scalar(self, name, value, iteration):
                self.scalars[name] = value

        writer = DummyWriter()
        # Two microbatches from the same step contribute token-weighted counts.
        MTPLossLoggingHelper.save_loss_to_tracker(
            loss_sum=torch.tensor(1.0, device="cuda"),
            num_tokens=torch.tensor(1.0, device="cuda"),
            correct=torch.tensor(1.0, device="cuda"),
            total=torch.tensor(1.0, device="cuda"),
            layer_number=0,
            num_layers=1,
        )
        MTPLossLoggingHelper.save_loss_to_tracker(
            loss_sum=torch.tensor(1.0, device="cuda"),
            num_tokens=torch.tensor(1.0, device="cuda"),
            correct=torch.tensor(1.0, device="cuda"),
            total=torch.tensor(3.0, device="cuda"),
            layer_number=0,
            num_layers=1,
        )
        torch.testing.assert_close(
            MTPLossLoggingHelper.tracker["acceptance_counts"],
            torch.tensor([[2.0], [4.0]], device="cuda"),
        )
        MTPLossLoggingHelper.track_mtp_metrics(
            loss_scale=1.0, iteration=1, writer=writer, total_loss_dict={}
        )

        assert torch.isclose(
            torch.as_tensor(writer.scalars["mtp_1_acceptance_rate"]),
            torch.tensor(50.0, device="cuda"),
        )
        assert torch.isclose(
            torch.as_tensor(writer.scalars["mtp_1_cumulative_acceptance_rate"]),
            torch.tensor(50.0, device="cuda"),
        )
        assert torch.all(MTPLossLoggingHelper.tracker["acceptance_counts"] == 0)

        MTPLossLoggingHelper.save_loss_to_tracker(
            loss_sum=torch.tensor(1.0, device="cuda"),
            num_tokens=torch.tensor(1.0, device="cuda"),
            correct=torch.tensor(1.0, device="cuda"),
            total=torch.tensor(4.0, device="cuda"),
            layer_number=0,
            num_layers=1,
        )
        MTPLossLoggingHelper.track_mtp_metrics(
            loss_scale=1.0, iteration=2, writer=writer, total_loss_dict={}
        )

        assert torch.isclose(
            torch.as_tensor(writer.scalars["mtp_1_acceptance_rate"]),
            torch.tensor(25.0, device="cuda"),
        )
        assert torch.isclose(
            torch.as_tensor(writer.scalars["mtp_1_cumulative_acceptance_rate"]),
            torch.tensor(37.5, device="cuda"),
        )
        assert torch.all(MTPLossLoggingHelper.tracker["acceptance_counts"] == 0)

    def test_mtp_logits_are_vocab_sharded(self):
        """Test detection for vocab-sharded versus gathered MTP logits."""

        class DummyOutputLayer:
            def __init__(self, gather_output):
                self.gather_output = gather_output

        assert _mtp_logits_are_vocab_sharded(DummyOutputLayer(gather_output=True), None) is False
        assert _mtp_logits_are_vocab_sharded(DummyOutputLayer(gather_output=False), None) is True
        assert _mtp_logits_are_vocab_sharded(DummyOutputLayer(gather_output=True), True) is False
        assert _mtp_logits_are_vocab_sharded(DummyOutputLayer(gather_output=True), False) is True

    def test_track_mtp_metrics(self):
        """Test tracking normalized MTP loss and acceptance rate."""
        loss_sum = torch.tensor(2.3)
        num_tokens = torch.tensor(1.0)
        num_layers = self.num_layers
        correct = torch.tensor(7.0)
        total = torch.tensor(10.0)

        for i in range(num_layers):
            MTPLossLoggingHelper.save_loss_to_tracker(
                loss_sum=loss_sum,
                num_tokens=num_tokens,
                correct=correct,
                total=total,
                layer_number=i,
                num_layers=num_layers,
            )

        class DummyWriter:
            def __init__(self):
                self.scalars = {}

            def add_scalar(self, name, value, iteration):
                self.scalars[name] = value

        class DummyWandBWriter:
            def log(self, metrics, iteration):
                pass

        loss_scale = 1.5
        iteration = 2
        writer = DummyWriter()
        wandb_writer = DummyWandBWriter()
        total_loss_dict = {}

        MTPLossLoggingHelper.track_mtp_metrics(
            loss_scale=loss_scale,
            iteration=iteration,
            writer=writer,
            wandb_writer=wandb_writer,
            total_loss_dict=total_loss_dict,
        )

        # track_mtp_metrics reduces the tracker first, so per-layer log value
        # equals (loss_sum / num_tokens) * loss_scale.
        expected_loss = (loss_sum / num_tokens) * loss_scale
        expected_rate = (correct / total) * 100.0
        for i in range(num_layers):
            assert f"mtp_{i + 1} loss" in writer.scalars
            assert torch.isclose(
                torch.as_tensor(writer.scalars[f"mtp_{i + 1} loss"]), expected_loss
            )
            assert torch.isclose(total_loss_dict[f"mtp_{i + 1} loss"], expected_loss)
            assert f"mtp_{i + 1}_acceptance_rate" in writer.scalars
            assert torch.isclose(
                torch.as_tensor(writer.scalars[f"mtp_{i + 1}_acceptance_rate"]), expected_rate
            )
            assert f"mtp_{i + 1}_cumulative_acceptance_rate" in writer.scalars
            assert torch.isclose(
                torch.as_tensor(writer.scalars[f"mtp_{i + 1}_cumulative_acceptance_rate"]),
                expected_rate,
            )

        raw_counter_suffixes = ("_sum", "_tokens", "_correct", "_total")
        assert not any(key.endswith(raw_counter_suffixes) for key in total_loss_dict)

        second_correct = torch.tensor(3.0)
        second_total = torch.tensor(10.0)
        for i in range(num_layers):
            MTPLossLoggingHelper.save_loss_to_tracker(
                loss_sum=loss_sum,
                num_tokens=num_tokens,
                correct=second_correct,
                total=second_total,
                layer_number=i,
                num_layers=num_layers,
            )

        MTPLossLoggingHelper.track_mtp_metrics(
            loss_scale=loss_scale,
            iteration=iteration + 1,
            writer=writer,
            wandb_writer=wandb_writer,
            total_loss_dict=total_loss_dict,
        )

        expected_second_rate = (second_correct / second_total) * 100.0
        expected_cumulative_rate = ((correct + second_correct) / (total + second_total)) * 100.0
        for i in range(num_layers):
            assert torch.isclose(total_loss_dict[f"mtp_{i + 1} loss"], expected_loss * 2)
            assert torch.isclose(
                torch.as_tensor(writer.scalars[f"mtp_{i + 1}_acceptance_rate"]),
                expected_second_rate,
            )
            assert torch.isclose(
                torch.as_tensor(writer.scalars[f"mtp_{i + 1}_cumulative_acceptance_rate"]),
                expected_cumulative_rate,
            )

        # Verify tracker is cleaned
        assert torch.all(MTPLossLoggingHelper.tracker["loss_sums"] == 0)
        assert MTPLossLoggingHelper.tracker["reduce_group"] is None
        assert MTPLossLoggingHelper.tracker["avg_group"] is None

    def test_microbatch_means_are_not_globally_token_weighted(self):
        """MTP logging preserves the pre-#4226 microbatch-normalized semantics."""
        MTPLossLoggingHelper.save_loss_to_tracker(
            loss_sum=torch.tensor(8.0), num_tokens=torch.tensor(2.0), layer_number=0, num_layers=1
        )
        MTPLossLoggingHelper.save_loss_to_tracker(
            loss_sum=torch.tensor(4.0), num_tokens=torch.tensor(4.0), layer_number=0, num_layers=1
        )

        class DummyWriter:
            def __init__(self):
                self.scalars = {}

            def add_scalar(self, name, value, iteration):
                self.scalars[name] = value

        writer = DummyWriter()
        MTPLossLoggingHelper.track_mtp_metrics(
            loss_scale=0.5, iteration=1, writer=writer, total_loss_dict={}
        )

        logged_loss = torch.as_tensor(writer.scalars["mtp_1 loss"])
        microbatch_mean_average = torch.tensor(((8.0 / 2.0) + (4.0 / 4.0)) / 2.0)
        global_token_weighted = torch.tensor((8.0 + 4.0) / (2.0 + 4.0))
        assert torch.isclose(logged_loss, microbatch_mean_average)
        assert not torch.isclose(logged_loss, global_token_weighted)

    def test_per_token_loss_is_globally_token_weighted(self):
        """Per-token MTP logging must match calculate-per-token-loss semantics."""
        MTPLossLoggingHelper.save_loss_to_tracker(
            loss_sum=torch.tensor(8.0),
            num_tokens=torch.tensor(2.0),
            layer_number=0,
            num_layers=1,
            calculate_per_token_loss=True,
        )
        MTPLossLoggingHelper.save_loss_to_tracker(
            loss_sum=torch.tensor(4.0),
            num_tokens=torch.tensor(4.0),
            layer_number=0,
            num_layers=1,
            calculate_per_token_loss=True,
        )

        class DummyWriter:
            def __init__(self):
                self.scalars = {}

            def add_scalar(self, name, value, iteration):
                self.scalars[name] = value

        writer = DummyWriter()
        MTPLossLoggingHelper.track_mtp_metrics(
            loss_scale=1.0, iteration=1, writer=writer, total_loss_dict={}
        )

        logged_loss = torch.as_tensor(writer.scalars["mtp_1 loss"])
        microbatch_mean_average = torch.tensor(((8.0 / 2.0) + (4.0 / 4.0)) / 2.0)
        global_token_weighted = torch.tensor((8.0 + 4.0) / (2.0 + 4.0))
        assert torch.isclose(logged_loss, global_token_weighted)
        assert not torch.isclose(logged_loss, microbatch_mean_average)

    def test_track_mtp_loss_preserves_legacy_normalized_loss_semantics(self):
        """MTP loss logging should not become token-weighted when acceptance counters are added."""
        first_loss = torch.tensor(10.0)
        second_loss = torch.tensor(2.0)
        correct = torch.tensor(0.0)
        total = torch.tensor(1.0)
        loss_scale = torch.tensor(0.5)
        layer_number = 0

        MTPLossLoggingHelper.save_metrics_to_tracker(
            loss=first_loss, correct=correct, total=total, layer_number=layer_number, num_layers=1
        )
        MTPLossLoggingHelper.save_metrics_to_tracker(
            loss=second_loss, correct=correct, total=total, layer_number=layer_number, num_layers=1
        )

        class DummyWriter:
            def __init__(self):
                self.scalars = {}

            def add_scalar(self, name, value, iteration):
                self.scalars[name] = value

        writer = DummyWriter()
        MTPLossLoggingHelper.track_mtp_metrics(
            loss_scale=loss_scale, iteration=1, writer=writer, total_loss_dict={}
        )

        logged_loss = torch.as_tensor(writer.scalars["mtp_1 loss"])
        expected_legacy_loss = (first_loss + second_loss) * loss_scale
        token_weighted_loss = torch.tensor(40.0 / 12.0)
        assert torch.isclose(logged_loss, expected_legacy_loss)
        assert not torch.isclose(logged_loss, token_weighted_loss)


class TestMultiTokenPredictionHybrid:
    """Test Multi-Token Prediction with Mamba hybrid models."""

    def setup_method(self, method):
        self.seq_length = 32
        self.micro_batch_size = 2
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        MTPLossLoggingHelper.tracker = {}

    def model_provider(self, pre_process=True, post_process=True, **config_kwargs):
        """Model provider for Mamba hybrid models with MTP.

        Uses the unified pattern syntax where MTP is configured via hybrid_layer_pattern:
        Format: "<main_pattern>/<mtp_pattern>/<mtp_pattern>/..."
        Example: "M*M*/M*/M*" = main decoder "M*M*", MTP pattern "M*" with 2 depths
        """
        model_parallel_cuda_manual_seed(_SEED)
        args = get_args()
        config = core_transformer_config_from_args(args)

        # MTP is configured via unified pattern in hybrid_layer_pattern
        # HybridModel creates the MTP block internally based on the parsed pattern
        model = HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=args.vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            hybrid_layer_pattern=args.hybrid_layer_pattern,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
        )
        return model

    def create_test_args(
        self, tp, cp, sequence_length, micro_batch_size, fp8=None, full_recompute=False
    ):
        destroy_global_vars()
        destroy_num_microbatches_calculator()

        sys.argv = ['test_multi_token_prediction_hybrid.py']
        args = parse_args()
        args.mtp_num_layers = 2
        args.mtp_loss_scaling_factor = 0.1
        args.padded_vocab_size = 128800
        args.hidden_size = 128
        args.num_attention_heads = 8
        args.num_query_groups = 8
        args.mamba_num_groups = 4
        args.max_position_embeddings = 256
        args.micro_batch_size = micro_batch_size
        args.create_attention_mask_in_dataloader = True
        args.seq_length = sequence_length
        args.tensor_model_parallel_size = tp
        args.sequence_parallel = True if tp > 1 else False
        args.context_parallel_size = cp
        args.position_embedding_type = 'rope'
        args.train_iters = 1
        args.ckpt_format = 'torch_dist'
        args.lr = 3e-5
        args.attention_dropout = 0.0
        args.hidden_dropout = 0.0
        args.no_save_optim = True
        args.no_load_optim = True
        args.no_load_rng = True
        args.bf16 = True
        args.save_tokenizer_assets = False
        # Unified pattern: "main/mtp/mtp" - main decoder "M*M*", MTP pattern "M*" with 2 depths
        args.hybrid_layer_pattern = "M*M*/M*/M*"

        if fp8 is not None:
            args.fp8 = 'e4m3'
        if full_recompute:
            args.recompute_granularity = 'full'
            args.recompute_method = 'uniform'
            args.recompute_num_layers = 1
        else:
            args.recompute_granularity = None
        args.add_bias_linear = False
        args.swiglu = True

        validate_args(args)
        set_global_variables(args, False)
        return args

    def get_batch(self, seq_length, micro_batch_size):
        data = list(range(seq_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        labels = 1 + torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, seq_length, seq_length), dtype=bool
        ).cuda()
        loss_mask = torch.ones(seq_length).repeat((micro_batch_size, 1)).cuda()
        batch = {
            'tokens': input_ids,
            'labels': labels,
            'loss_mask': loss_mask,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
        }
        return batch

    @staticmethod
    def _make_forward_stub():
        hidden_states = torch.arange(4, dtype=torch.float32).reshape(2, 1, 2)
        call_counts = {"mtp": 0, "mtp_loss": 0, "main_loss": 0}

        def decoder(**kwargs):
            decoder_hidden_states = kwargs["hidden_states"]
            return (
                decoder_hidden_states.unwrap()
                if hasattr(decoder_hidden_states, "unwrap")
                else decoder_hidden_states
            )

        def mtp(**kwargs):
            call_counts["mtp"] += 1
            decoder_hidden_states = kwargs["hidden_states"]
            return torch.cat((decoder_hidden_states, decoder_hidden_states + 100.0), dim=0)

        def output_layer(output, **kwargs):
            return output, None

        def compute_language_model_loss(labels, logits):
            call_counts["main_loss"] += 1
            return labels.to(dtype=logits.dtype) + 1000.0

        model = types.SimpleNamespace(
            config=types.SimpleNamespace(
                fine_grained_activation_offloading=False,
                moe_paged_stash=False,
                multi_latent_attention=False,
                mtp_num_layers=1,
                use_mup=False,
                inference_cuda_graph_scope=None,
            ),
            pre_process=False,
            post_process=True,
            position_embedding_type='none',
            decoder=decoder,
            share_embeddings_and_output_weights=False,
            mtp_process=True,
            embedding=None,
            mtp=mtp,
            output_layer=output_layer,
            training=True,
            compute_language_model_loss=compute_language_model_loss,
            pg_collection=types.SimpleNamespace(cp=None),
            tp_group=None,
            _scale_logits=lambda logits: logits,
        )
        return model, hidden_states, call_counts

    @pytest.mark.parametrize(
        (
            "compute_mtp_loss",
            "provide_labels",
            "expected_mtp_calls",
            "expected_mtp_loss_calls",
            "expected_main_loss_calls",
        ),
        [
            pytest.param(True, True, 1, 1, 1, id="enabled-labels-return-loss"),
            pytest.param(True, False, 1, 1, 0, id="enabled-rl-label-derivation"),
            pytest.param(False, True, 0, 0, 1, id="skip-mtp-labels-return-loss"),
            pytest.param(False, False, 0, 0, 0, id="skip-mtp-no-labels-return-logits"),
        ],
    )
    def test_forward_compute_mtp_loss_contract(
        self,
        monkeypatch,
        compute_mtp_loss,
        provide_labels,
        expected_mtp_calls,
        expected_mtp_loss_calls,
        expected_main_loss_calls,
    ):
        """Test that MTP execution and the main output contract are independent."""
        model, hidden_states, call_counts = self._make_forward_stub()
        labels = torch.tensor([[3, 4]]) if provide_labels else None
        input_ids = torch.zeros(1, 2, dtype=torch.long)
        loss_mask = torch.ones(1, 2)

        def process_mtp_loss_spy(**kwargs):
            call_counts["mtp_loss"] += 1
            assert kwargs["labels"] is labels
            assert kwargs["input_ids"] is input_ids
            assert kwargs["loss_mask"] is loss_mask
            return torch.chunk(kwargs["hidden_states"], 1 + kwargs["config"].mtp_num_layers, dim=0)[
                0
            ]

        monkeypatch.setattr(
            "megatron.core.models.hybrid.hybrid_model.process_mtp_loss", process_mtp_loss_spy
        )

        output = HybridModel.forward(
            model,
            input_ids=input_ids,
            position_ids=torch.arange(2).unsqueeze(0),
            attention_mask=None,
            decoder_input=hidden_states,
            labels=labels,
            loss_mask=loss_mask,
            compute_mtp_loss=compute_mtp_loss,
        )

        assert call_counts == {
            "mtp": expected_mtp_calls,
            "mtp_loss": expected_mtp_loss_calls,
            "main_loss": expected_main_loss_calls,
        }
        if provide_labels:
            torch.testing.assert_close(output, labels.to(dtype=hidden_states.dtype) + 1000.0)
        else:
            torch.testing.assert_close(output, hidden_states.transpose(0, 1).contiguous())

    @pytest.mark.parametrize("compute_mtp_loss", [True, False])
    def test_compute_mtp_loss_does_not_control_speculative_decoding(
        self, monkeypatch, compute_mtp_loss
    ):
        """Test the auxiliary MTP switch does not disable speculative-decoding state capture."""
        model, hidden_states, call_counts = self._make_forward_stub()
        inference_context = types.SimpleNamespace(
            is_dynamic_batching=lambda: True,
            num_speculative_tokens=1,
            mtp_decoder_hidden_states=None,
            config=types.SimpleNamespace(materialize_only_last_token_logits=False),
        )

        def unexpected_process_mtp_loss(**kwargs):
            call_counts["mtp_loss"] += 1
            raise AssertionError("MTP loss processing must not run during speculative decoding")

        monkeypatch.setattr(
            "megatron.core.models.hybrid.hybrid_model.process_mtp_loss", unexpected_process_mtp_loss
        )

        with InferenceMode.active():
            output = HybridModel.forward(
                model,
                input_ids=torch.zeros(1, 2, dtype=torch.long),
                position_ids=torch.arange(2).unsqueeze(0),
                attention_mask=None,
                decoder_input=hidden_states,
                inference_context=inference_context,
                runtime_gather_output=True,
                compute_mtp_loss=compute_mtp_loss,
            )

        torch.testing.assert_close(output, hidden_states.transpose(0, 1).contiguous())
        torch.testing.assert_close(inference_context.mtp_decoder_hidden_states, hidden_states)
        assert call_counts == {"mtp": 0, "mtp_loss": 0, "main_loss": 0}

    @pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
    @pytest.mark.parametrize(("tp", "cp"), [(1, 1), (2, 1)])
    def test_sharded_state_dict_mamba(self, tp, cp):
        """Test MTP with Mamba hybrid model - sharded state dict."""
        args = self.create_test_args(tp, cp, self.seq_length, self.micro_batch_size)
        set_args(args)
        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)

        model_parallel_cuda_manual_seed(_SEED)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        model_cfg = hybrid_config_from_args(args)
        builder_cls = model_cfg.get_builder_cls()
        builder = builder_cls(model_cfg)
        mamba_model = builder.build_distributed_models(
            pg_collection=pg_collection, wrap_with_ddp=False
        )
        sharded_state_dict = mamba_model[0].sharded_state_dict()

        # Verify MTP layers are in the state dict
        for i in range(args.mtp_num_layers):
            assert f"mtp.layers.{i}.enorm.weight" in sharded_state_dict.keys()
            assert f"mtp.layers.{i}.hnorm.weight" in sharded_state_dict.keys()
            assert f"mtp.layers.{i}.eh_proj.weight" in sharded_state_dict.keys()

    @pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
    @pytest.mark.parametrize(("tp", "cp"), [(1, 1), (2, 1)])
    def test_forward_backward_mamba(self, tmp_path_dist_ckpt, tp, cp):
        """Test MTP forward and backward with Mamba hybrid model."""
        tp_ref = 1
        cp_ref = 1
        args = self.create_test_args(tp_ref, cp_ref, self.seq_length, self.micro_batch_size)
        set_args(args)
        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_ref, context_parallel_size=cp_ref
        )
        batch = self.get_batch(self.seq_length, self.micro_batch_size)
        tokens, labels, loss_mask, attention_mask, position_ids = batch.values()

        model_parallel_cuda_manual_seed(_SEED)
        cfg_container = Utils.pretrain_config_from_global_args(args, "hybrid")
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        mamba_model_ref, optimizer, opt_param_scheduler = setup_model_and_optimizer(
            ModelType.encoder_or_decoder,
            self.model_provider,
            cfg_container=cfg_container,
            pg_collection=pg_collection,
        )

        output_ref = mamba_model_ref[0].forward(
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            loss_mask=loss_mask,
        )
        # Forward accumulates normalized losses. Reduce them first so
        # tracker["values"] becomes available.
        MTPLossLoggingHelper.reduce_loss_in_tracker()
        tracker = MTPLossLoggingHelper.tracker
        assert "values" in tracker
        mtp_loss_ref = tracker['values'].clone()
        MTPLossLoggingHelper.clean_loss_in_tracker()

        iteration = 123
        num_floating_point_operations_so_far = 456

        def set_ckpt_path(ckpt_path):
            args.save = ckpt_path
            args.load = ckpt_path

        with TempNamedDir(tmp_path_dist_ckpt / 'test_mtp_mamba_model_reconfiguration') as ckpt_dir:
            set_ckpt_path(ckpt_dir)
            save_checkpoint(
                iteration,
                mamba_model_ref,
                optimizer,
                opt_param_scheduler,
                num_floating_point_operations_so_far,
            )

            expected_ckpt_path = args.save / "iter_0000123" / ".metadata"
            assert os.path.exists(expected_ckpt_path)

            Utils.destroy_model_parallel()
            args = self.create_test_args(tp, cp, self.seq_length, self.micro_batch_size)
            set_args(args)
            set_ckpt_path(ckpt_dir)
            torch.manual_seed(_SEED)
            Utils.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)

            model_parallel_cuda_manual_seed(_SEED)
            cfg_container = Utils.pretrain_config_from_global_args(args, "hybrid")
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
            mamba_model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
                ModelType.encoder_or_decoder,
                self.model_provider,
                cfg_container=cfg_container,
                pg_collection=pg_collection,
            )
            load_checkpoint(mamba_model, optimizer, opt_param_scheduler, strict=False)

            batch["output_ref"] = output_ref
            batch = get_batch_on_this_cp_rank(
                batch, is_hybrid_cp=False, cp_group=get_context_parallel_group()
            )
            tokens, labels, loss_mask, attention_mask, position_ids, output_ref = batch.values()
            output = mamba_model[0].forward(
                input_ids=tokens,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=labels,
                loss_mask=loss_mask,
            )
            # Combine normalized loss contributions across DP+CP.
            MTPLossLoggingHelper.reduce_loss_in_tracker()
            tracker = MTPLossLoggingHelper.tracker
            assert "values" in tracker
            mtp_loss = tracker['values'].clone()
            MTPLossLoggingHelper.clean_loss_in_tracker()
            assert torch.allclose(output_ref, output, rtol=1e-03, atol=1e-03)
            assert torch.allclose(mtp_loss, mtp_loss_ref, rtol=1e-02, atol=1e-02)

            assert output.shape[0] == self.micro_batch_size
            assert output.shape[1] == self.seq_length / cp

            loss = output.mean()
            loss.backward()
            for name, param in mamba_model[0].named_parameters():
                assert param.main_grad is not None

    @pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
    def test_attention_mask_validation_mamba(self):
        """Test that attention mask type validation works for Mamba hybrid models."""
        tp = 1
        cp = 1
        args = self.create_test_args(tp, cp, self.seq_length, self.micro_batch_size)
        set_args(args)
        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        model_cfg = hybrid_config_from_args(args)
        builder_cls = model_cfg.get_builder_cls()
        builder = builder_cls(model_cfg)
        try:
            model_parallel_cuda_manual_seed(_SEED)
            mamba_model = builder.build_distributed_models(
                pg_collection=pg_collection, wrap_with_ddp=False
            )
            mamba_model = unwrap_model(mamba_model)
            assert isinstance(mamba_model[0], HybridModel)
            assert mamba_model[0].mtp is not None
        except AssertionError as e:
            if "Multi-Token Prediction (MTP) is not yet supported" in str(e):
                pytest.fail(f"Attention mask validation failed for Mamba hybrid model: {e}")
            else:
                raise


class TestLearnedOutputContract:
    """Tests for learned_output_contract: shape, dtype, gradient, and numerical correctness."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(_SEED)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_shape_and_dtype(self):
        """Output shape is [*, h] from [*, n*h]; dtype matches input after fp32 round-trip."""
        seq_len, batch_size, hidden_size, n_streams = 16, 2, 64, 4
        head_fn = torch.randn(n_streams, n_streams * hidden_size, device='cuda')
        base = torch.zeros(n_streams, device='cuda')
        scale = torch.ones(1, device='cuda')

        for dtype in [torch.bfloat16, torch.float16]:
            hidden_states = torch.randn(
                seq_len, batch_size, n_streams * hidden_size, device='cuda', dtype=dtype
            )
            output = learned_output_contract(
                hidden_states, head_fn, base, scale, n_streams, eps=1e-6
            )
            assert output.shape == (seq_len, batch_size, hidden_size)
            assert output.dtype == dtype

    def test_gradient_and_numerical_correctness(self):
        """Gradients flow to all inputs; output matches reference implementation."""
        torch.manual_seed(_SEED)
        seq_len, batch_size, hidden_size, n_streams = 2, 1, 8, 2
        eps = 1e-6
        hidden_states = torch.randn(
            seq_len,
            batch_size,
            n_streams * hidden_size,
            device='cuda',
            dtype=torch.float32,
            requires_grad=True,
        )
        head_fn = torch.randn(n_streams, n_streams * hidden_size, device='cuda', requires_grad=True)
        base = torch.zeros(n_streams, device='cuda', requires_grad=True)
        scale = torch.ones(1, device='cuda', requires_grad=True)

        output = learned_output_contract(hidden_states, head_fn, base, scale, n_streams, eps)

        # Numerical reference
        hs_fp32 = hidden_states.detach().clone()
        rsqrt_ref = torch.rsqrt(hs_fp32.square().mean(-1, keepdim=True) + eps)
        mixes_ref = torch.nn.functional.linear(hs_fp32, head_fn.detach()) * rsqrt_ref
        pre_ref = torch.sigmoid(mixes_ref * scale.detach() + base.detach()) + 1e-6
        y_ref = torch.sum(
            pre_ref.unsqueeze(-1) * hs_fp32.view(*hs_fp32.shape[:-1], n_streams, -1), dim=-2
        )
        torch.testing.assert_close(output, y_ref, rtol=1e-4, atol=1e-4)

        # Gradient flow
        output.sum().backward()
        for name, tensor in [
            ("hidden_states", hidden_states),
            ("head_fn", head_fn),
            ("base", base),
            ("scale", scale),
        ]:
            assert tensor.grad is not None, f"No gradient for {name}"
            assert not torch.all(tensor.grad == 0), f"Zero gradient for {name}"


class TestMHCMTPIntegration:
    """Integration tests for mHC + MTP: constructor, TransformerBlock output, E2E."""

    def setup_method(self, method):
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        MTPLossLoggingHelper.tracker = {}

    @pytest.mark.parametrize('tp', [1, 2])
    def test_mtp_constructor_with_mhc(self, tp):
        """MTP layers have e_proj/h_proj (not eh_proj) and learned contraction params."""
        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp)
        config = TransformerConfig(
            mtp_num_layers=2,
            num_layers=4,
            hidden_size=64,
            num_attention_heads=8,
            num_residual_streams=4,
            enable_hyper_connections=True,
            use_cpu_initialization=True,
            tensor_model_parallel_size=tp,
            sequence_parallel=True if tp > 1 else False,
        )
        spec = get_gpt_layer_local_spec(enable_hyper_connection=True)
        mtp_block_spec = get_gpt_mtp_block_spec(
            config=config, spec=spec, use_transformer_engine=False
        )
        mtp = MultiTokenPredictionBlock(config=config, spec=mtp_block_spec)

        n, h = config.num_residual_streams, config.hidden_size
        for i in range(config.mtp_num_layers):
            layer = mtp.layers[i]
            assert layer.e_proj is not None and layer.h_proj is not None
            assert layer.eh_proj is None
            assert layer.e_proj.weight.shape == (h // tp, h)
            assert layer.h_proj.weight.shape == (h // tp, h)
            assert layer.hc_head_fn.shape == (n, n * h)
            assert layer.hc_head_base.shape == (n,)
            assert layer.hc_head_scale.shape == (1,)
            if tp > 1:
                assert getattr(layer.hc_head_fn, 'sequence_parallel', False)

    def test_transformer_block_returns_tuple(self):
        """With mHC+MTP the block returns (contracted, multistream); without MTP just a tensor."""
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(_SEED)
        spec = get_gpt_layer_local_spec(enable_hyper_connection=True)

        seq_len, batch_size, h, n = 16, 2, 64, 4

        # With MTP: should return tuple
        config_mtp = TransformerConfig(
            num_layers=2,
            hidden_size=h,
            num_attention_heads=4,
            enable_hyper_connections=True,
            num_residual_streams=n,
            use_cpu_initialization=True,
            mtp_num_layers=2,
        )
        block_mtp = TransformerBlock(config_mtp, spec).cuda()
        hidden_states = torch.randn(seq_len, batch_size, h, device='cuda', requires_grad=True)
        output = block_mtp(hidden_states=hidden_states, attention_mask=None)

        assert isinstance(output, tuple)
        contracted, multistream = output
        assert contracted.shape == (seq_len, batch_size, h)
        assert multistream.shape == (seq_len, batch_size, n * h)

        (contracted.sum() + multistream.sum()).backward()
        assert hidden_states.grad is not None

        # Without MTP: should return single tensor
        config_no_mtp = TransformerConfig(
            num_layers=2,
            hidden_size=h,
            num_attention_heads=4,
            enable_hyper_connections=True,
            num_residual_streams=n,
            use_cpu_initialization=True,
            mtp_num_layers=None,
        )
        block_no_mtp = TransformerBlock(config_no_mtp, spec).cuda()
        hs2 = torch.randn(seq_len, batch_size, h, device='cuda')
        output2 = block_no_mtp(hidden_states=hs2, attention_mask=None)
        assert isinstance(output2, Tensor)
        assert output2.shape == (seq_len, batch_size, h)

    @pytest.mark.skipif(
        not HAVE_TE or not is_te_min_version("1.7.0"), reason="TransformerEngine >= 1.7.0 required"
    )
    @pytest.mark.parametrize('tp', [1, 2])
    def test_e2e_forward_backward(self, tp):
        """GPTModel E2E with mHC + MTP: finite output, MTP loss logged, gradients on HC params."""
        destroy_global_vars()
        destroy_num_microbatches_calculator()

        seq_length, micro_batch_size = 32, 2

        sys.argv = ['test_multi_token_prediction.py']
        args = parse_args()
        args.num_layers = 2
        args.mtp_num_layers = 2
        args.mtp_loss_scaling_factor = 0.1
        args.vocab_size = 128800
        args.hidden_size = 128
        args.num_attention_heads = 8
        args.max_position_embeddings = 256
        args.micro_batch_size = micro_batch_size
        args.create_attention_mask_in_dataloader = True
        args.seq_length = seq_length
        args.tensor_model_parallel_size = tp
        args.sequence_parallel = tp > 1
        args.context_parallel_size = 1
        args.position_embedding_type = 'rope'
        args.num_experts = None
        args.moe_grouped_gemm = False
        args.train_iters = 1
        args.lr = 3e-5
        args.attention_dropout = 0.0
        args.hidden_dropout = 0.0
        args.add_bias_linear = False
        args.swiglu = True
        args.bf16 = True
        args.enable_hyper_connections = True
        args.num_residual_streams = 4
        args.recompute_granularity = None

        validate_args(args)
        set_global_variables(args, False)
        set_args(args)
        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp)

        def model_provider(
            pre_process=True,
            post_process=True,
            layer_spec_fn=get_gpt_layer_with_transformer_engine_spec,
            config=None,
            pg_collection=None,
            vp_stage=None,
            **kwargs,
        ):
            model_parallel_cuda_manual_seed(_SEED)
            a = get_args()
            if config is None:
                config = core_transformer_config_from_args(a)
            layer_spec = layer_spec_fn(
                a.num_experts,
                a.moe_grouped_gemm,
                a.qk_layernorm,
                enable_hyper_connection=config.enable_hyper_connections,
            )
            mtp_spec = get_gpt_mtp_block_spec(
                config=config, spec=layer_spec, use_transformer_engine=True
            )
            return GPTModel(
                config=config,
                transformer_layer_spec=layer_spec,
                mtp_block_spec=mtp_spec,
                vocab_size=a.vocab_size,
                max_sequence_length=a.max_position_embeddings,
                pre_process=pre_process,
                post_process=post_process,
                fp16_lm_cross_entropy=a.fp16_lm_cross_entropy,
                parallel_output=True,
                share_embeddings_and_output_weights=not a.untie_embeddings_and_output_weights,
                position_embedding_type=a.position_embedding_type,
                rotary_percent=a.rotary_percent,
                pg_collection=pg_collection,
                vp_stage=vp_stage,
            )

        gpt_model, _, _ = setup_model_and_optimizer(ModelType.encoder_or_decoder, model_provider)

        data = list(range(seq_length))
        tokens = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        labels = (1 + torch.tensor(data, dtype=torch.int64)).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, seq_length, seq_length), dtype=bool
        ).cuda()
        loss_mask = torch.ones(seq_length).repeat((micro_batch_size, 1)).cuda()

        output = gpt_model[0].forward(
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            loss_mask=loss_mask,
        )
        assert torch.isfinite(output).all(), f"Non-finite output (TP={tp})"

        # Reduce normalized loss contributions before reading.
        MTPLossLoggingHelper.reduce_loss_in_tracker()
        tracker = MTPLossLoggingHelper.tracker
        assert "values" in tracker, f"MTP loss not logged (TP={tp})"
        assert torch.isfinite(tracker['values']).all()
        MTPLossLoggingHelper.clean_loss_in_tracker()

        output.mean().backward()
        hc_param_names = ['hc_head_fn', 'hc_head_base', 'hc_head_scale']
        for name, param in gpt_model[0].named_parameters():
            assert param.main_grad is not None, f"No gradient for {name}"
            if any(n in name for n in hc_param_names):
                assert not torch.all(param.main_grad == 0), f"Zero gradient for {name}"
