# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core import mpu
from megatron.core.transformer import TransformerConfig
from megatron.training.models import BertModelConfig
from megatron.training.vocab_utils import model_vocab_size

_REPO_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
sys.path.insert(0, os.path.join(_REPO_ROOT, 'tools', 'checkpoint'))

from saver_base import MegatronCheckpointSaverBase


def test_target_derived_weight_shard_args_are_not_loaded_from_source_checkpoint():
    source_args = SimpleNamespace(
        tensor_parallel_num_weight_shards=1,
        gtp_weight_remat_size=1,
        expert_tensor_parallel_num_weight_shards=1,
        expert_gtp_weight_remat_size=1,
        num_layers=24,
    )
    target_args = SimpleNamespace(
        tensor_parallel_num_weight_shards=4,
        gtp_weight_remat_size=2,
        expert_tensor_parallel_num_weight_shards=8,
        expert_gtp_weight_remat_size=2,
        num_layers=12,
    )
    saver = MegatronCheckpointSaverBase(args=None, queue=None)
    saver.md = SimpleNamespace(checkpoint_args=source_args)

    result = saver._load_checkpoint_args(target_args)

    assert result.tensor_parallel_num_weight_shards == 4
    assert result.gtp_weight_remat_size == 2
    assert result.expert_tensor_parallel_num_weight_shards == 8
    assert result.expert_gtp_weight_remat_size == 2
    assert result.num_layers == 24


@pytest.mark.parametrize(
    'source_vocab,target_tp,true_vocab,expected_vocab',
    [
        (30592, 8, 30522, 30720),
        (30720, 1, 30522, 30592),
        (30592, 1, 30522, 30592),
        (30592, 8, None, 30592),
    ],
)
def test_receive_lm_updates_vocab_before_model_construction(
    run_config, monkeypatch, source_vocab, target_tp, true_vocab, expected_vocab
):
    run_config.model = BertModelConfig(
        transformer=TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=8,
            tensor_model_parallel_size=target_tp,
        ),
        seq_length=16,
        vocab_size=source_vocab,
    )
    run_config.tokenizer.padded_vocab_size = source_vocab
    saver = MegatronCheckpointSaverBase(
        args=SimpleNamespace(
            target_tensor_parallel_size=target_tp,
            target_pipeline_parallel_size=1,
            target_expert_parallel_size=1,
            checking=False,
        ),
        queue=None,
    )
    saver.margs = SimpleNamespace(
        padded_vocab_size=source_vocab,
        make_vocab_size_divisible_by=128,
        tensor_model_parallel_size=target_tp,
    )
    saver.md = SimpleNamespace(
        position_embedding_type='rope',
        true_vocab_size=true_vocab,
        norm_has_bias=False,
        output_layer=False,
    )
    source_weight = torch.arange(source_vocab, dtype=torch.float32).reshape(-1, 1)
    saver.queue_get = Mock(
        side_effect=[{'word embeddings': source_weight}, {'weight': torch.ones(1)}, 'done']
    )
    saver.models = [[[None for _ in range(target_tp)]]]

    def get_local_model(pp_rank, ep_rank, tp_rank):
        if saver.models[pp_rank][ep_rank][tp_rank] is None:
            assert run_config.tokenizer.padded_vocab_size == expected_vocab
            assert run_config.model.vocab_size == expected_vocab
            assert not run_config.model.should_pad_vocab
            saver.models[pp_rank][ep_rank][tp_rank] = torch.empty(
                model_vocab_size(
                    run_config.model.vocab_size,
                    should_pad_vocab=run_config.model.should_pad_vocab,
                    make_vocab_size_divisible_by=run_config.model.make_vocab_size_divisible_by,
                    tensor_model_parallel_size=run_config.model.transformer.tensor_model_parallel_size,
                )
                // target_tp,
                1,
            )
        return saver.models[pp_rank][ep_rank][tp_rank]

    saver.get_local_model = get_local_model
    schema = Mock()
    schema.has_position_embeddings.return_value = False
    schema.get_num_layers.return_value = 0

    def copy_embeddings(name, model, values):
        if name == 'embeddings':
            model.copy_(values['word'])

    schema.set.side_effect = copy_embeddings
    monkeypatch.setattr(mpu, 'set_pipeline_model_parallel_rank', lambda rank: None)
    saver.receive_lm(schema)

    weights = torch.cat(saver.models[0][0])
    preserved_rows = min(source_vocab, expected_vocab)
    torch.testing.assert_close(weights[:preserved_rows], source_weight[:preserved_rows])
    if expected_vocab > source_vocab:
        torch.testing.assert_close(
            weights[source_vocab:], source_weight[-1:].expand(expected_vocab - source_vocab, -1)
        )
    saved_config = run_config.to_dict()
    assert saved_config['model']['vocab_size'] == expected_vocab
    assert saved_config['tokenizer']['padded_vocab_size'] == expected_vocab
    assert saver.margs.padded_vocab_size == expected_vocab
