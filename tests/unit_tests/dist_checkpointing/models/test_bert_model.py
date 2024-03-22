# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from megatron.core.models.bert.bert_model import BertModel
import pytest

import os
import torch
from torch.distributed._tensor import DeviceMesh

from megatron.core.dist_checkpointing import save, load, load_plain_tensors
from megatron.core import parallel_state as ps
from megatron.core.dist_checkpointing.dict_utils import diff
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.models.bert.bert_layer_specs import bert_layer_local_spec, bert_layer_with_transformer_engine_spec


def initalize_bert_model(seed, layer_spec=bert_layer_with_transformer_engine_spec, **config_kwargs):
    os.environ['NVTE_ALLOW_NONDETERMINISTIC_ALGO'] = '0'
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)

    default_config_kwargs=dict(num_layers=8, hidden_size=16, num_attention_heads=8, use_cpu_initialization=True)
    default_config_kwargs.update(**config_kwargs)
    transformer_config = TransformerConfig(**default_config_kwargs)
    pre_process = ps.is_pipeline_first_stage()
    post_process = ps.is_pipeline_last_stage()
    model = BertModel(config=transformer_config, transformer_layer_spec=layer_spec, vocab_size=128, max_sequence_length=4,
                     pre_process=pre_process, post_process=post_process, num_tokentypes=0)

    with torch.no_grad():
        for p in model.parameters():
            p.random_()
    return model


class TestBertModel:
    @pytest.mark.parametrize('src_layer_spec', [bert_layer_with_transformer_engine_spec, bert_layer_local_spec])
    @pytest.mark.parametrize('dst_layer_spec', [bert_layer_with_transformer_engine_spec, bert_layer_local_spec])
    def test_sharded_state_dict_save_load(self, tmp_path_dist_ckpt,
                                          src_layer_spec, dst_layer_spec):
        Utils.initialize_model_parallel(2,4)
        bert_model = initalize_bert_model(1, src_layer_spec)
        with TempNamedDir(tmp_path_dist_ckpt / 'test_bert_model') as ckpt_dir:
            # Save
            sharded_state_dict = bert_model.sharded_state_dict()
            save(sharded_state_dict, ckpt_dir)

            # Load
            bert_model = initalize_bert_model(2, dst_layer_spec)
            sharded_state_dict = bert_model.sharded_state_dict()
            state_dict = load(sharded_state_dict, ckpt_dir)
            bert_model.load_state_dict(state_dict)
        Utils.destroy_model_parallel()


class TestBERTModelReconfiguration:
    @pytest.mark.parametrize("src_tp_pp,dest_tp_pp,src_layer_spec,dst_layer_spec", [
        ((2, 4), (4, 2), bert_layer_with_transformer_engine_spec, bert_layer_with_transformer_engine_spec),
        ((1, 8), (8, 1), bert_layer_with_transformer_engine_spec, bert_layer_with_transformer_engine_spec),
        ((2, 1), (1, 8), bert_layer_with_transformer_engine_spec, bert_layer_with_transformer_engine_spec),
        ((1, 1), (2, 2), bert_layer_with_transformer_engine_spec, bert_layer_with_transformer_engine_spec),
        ((2, 1), (1, 8), bert_layer_local_spec, bert_layer_local_spec),
        ((1, 1), (2, 4), bert_layer_with_transformer_engine_spec, bert_layer_local_spec),
        ((1, 8), (2, 1), bert_layer_local_spec, bert_layer_with_transformer_engine_spec),
    ])
    def test_parallel_reconfiguration_e2e(self, tmp_path_dist_ckpt, src_tp_pp, dest_tp_pp,
                                          src_layer_spec, dst_layer_spec):
        """ Test model saving and loading with different TP/PP """
        with TempNamedDir(tmp_path_dist_ckpt / 'test_bert_model_reconfiguration_model_A') as ckpt_dir_A, \
             TempNamedDir(tmp_path_dist_ckpt / 'test_bert_model_reconfiguration_model_B') as ckpt_dir_B:
            # Save checkpoint A
            Utils.initialize_model_parallel(*src_tp_pp)
            bert_model_A = initalize_bert_model(1, src_layer_spec)
            save(bert_model_A.sharded_state_dict(), ckpt_dir_A)
            regular_state_dict_A = bert_model_A.state_dict()
            Utils.destroy_model_parallel()

            # Load checkpoint A with different TP/PP and save as checkpoint B
            Utils.initialize_model_parallel(*dest_tp_pp)
            bert_model_B = initalize_bert_model(2, dst_layer_spec)
            state_dict = load(bert_model_B.sharded_state_dict(), ckpt_dir_A)
            bert_model_B.load_state_dict(state_dict)
            save(bert_model_B.sharded_state_dict(), ckpt_dir_B)
            regular_state_dict_B = bert_model_A.state_dict()
            Utils.destroy_model_parallel()

            # Test both checkpoints are equal
            Utils.initialize_model_parallel(1, 1)
            plain_state_dict_A = load_plain_tensors(ckpt_dir_A)
            plain_state_dict_B = load_plain_tensors(ckpt_dir_B)
            diffs = diff(plain_state_dict_A, plain_state_dict_B)
            assert not any(map(bool, diffs)), diffs

            # Test both regular state dicts are equal, turning FP8 states to bytes first
            regular_state_dict_A = {k: v for k, v in regular_state_dict_A.items()
                                    if not k.endswith('_extra_state')}
            regular_state_dict_B = {k: v for k, v in regular_state_dict_B.items()
                                    if not k.endswith('_extra_state')}
            diffs = diff(regular_state_dict_A, regular_state_dict_B)
            assert not any(map(bool, diffs)), diffs
            Utils.destroy_model_parallel()


    def test_state_dict_comparison(self, tmp_path_dist_ckpt):
        Utils.initialize_model_parallel(2, 4)
        with TempNamedDir(tmp_path_dist_ckpt / 'test_state_dict_comparison_A') as ckpt_dir_A, \
             TempNamedDir(tmp_path_dist_ckpt / 'test_state_dict_comparison_B') as ckpt_dir_B:
            bert_model_A = initalize_bert_model(1)
            save(bert_model_A.sharded_state_dict(), ckpt_dir_A)
            bert_model_B = initalize_bert_model(2)
            save(bert_model_B.sharded_state_dict(), ckpt_dir_B)

            state_dict_A = load_plain_tensors(ckpt_dir_A)
            state_dict_A_dup = load_plain_tensors(ckpt_dir_A)
            state_dict_B = load_plain_tensors(ckpt_dir_B)

            # Test that A matches A
            diffs = diff(state_dict_A, state_dict_A_dup)
            assert not any(map(bool, diffs)), diffs

            # Test that A *keys* match B *keys*, but the tensors content is different
            only_left, only_right, mismatch = diff(state_dict_A, state_dict_B)
            assert (not only_left and not only_right), (only_left, only_right)
            assert len(mismatch) == len(state_dict_A), (len(mismatch), (len(state_dict_A)))