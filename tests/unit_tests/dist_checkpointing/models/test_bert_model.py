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
from tests.unit_tests.dist_checkpointing.models.common import \
    common_test_simple_sharded_state_dict_save_load, \
    common_test_parallel_reconfiguration_e2e, common_test_state_dict_comparison, \
    common_test_vocab_size_padding_change
from tests.unit_tests.test_utilities import Utils
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.models.bert.bert_layer_specs import bert_layer_local_spec, bert_layer_with_transformer_engine_spec


def initialize_bert_model(seed, layer_spec_fn=bert_layer_with_transformer_engine_spec, vocab_size=128, **config_kwargs):
    os.environ['NVTE_ALLOW_NONDETERMINISTIC_ALGO'] = '0'
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)

    layer_spec = layer_spec_fn() if callable(layer_spec_fn) else layer_spec_fn

    default_config_kwargs=dict(num_layers=8, hidden_size=16, num_attention_heads=8, use_cpu_initialization=True)
    default_config_kwargs.update(**config_kwargs)
    transformer_config = TransformerConfig(**default_config_kwargs)
    pre_process = ps.is_pipeline_first_stage()
    post_process = ps.is_pipeline_last_stage()
    model = BertModel(config=transformer_config, transformer_layer_spec=layer_spec, vocab_size=vocab_size, max_sequence_length=4,
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
        common_test_simple_sharded_state_dict_save_load(initialize_bert_model, tmp_path_dist_ckpt,
                                                        src_layer_spec, dst_layer_spec)


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
        common_test_parallel_reconfiguration_e2e(initialize_bert_model, tmp_path_dist_ckpt, src_tp_pp,
                                                 dest_tp_pp, src_layer_spec, dst_layer_spec)

    def test_state_dict_comparison(self, tmp_path_dist_ckpt):
        common_test_state_dict_comparison(initialize_bert_model, tmp_path_dist_ckpt)

    @pytest.mark.parametrize("vocab_size_base", [128, 17, 127, 31123])
    @pytest.mark.parametrize("src_tp_pp,dest_tp_pp", [
        ((2, 4), (4, 2)),
        ((1, 8), (8, 1)),
        ((1, 1), (1, 8)),
    ])
    def test_vocab_size_padding_change(self, tmp_path_dist_ckpt, vocab_size_base, src_tp_pp, dest_tp_pp):
        """ Test model loading with different vocab size (caused by TP padding). """
        common_test_vocab_size_padding_change(initialize_bert_model, tmp_path_dist_ckpt, vocab_size_base,
                                              src_tp_pp, dest_tp_pp)
