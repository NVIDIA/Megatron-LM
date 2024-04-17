# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import pytest

import torch

from megatron.core import parallel_state as ps
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from tests.unit_tests.dist_checkpointing.models.common import \
    common_test_simple_sharded_state_dict_save_load, \
    common_test_parallel_reconfiguration_e2e, \
    common_test_state_dict_comparison, common_test_vocab_size_padding_change
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.models.gpt.gpt_layer_specs import \
    get_gpt_layer_with_transformer_engine_spec as gpt_te_spec, get_gpt_layer_local_spec as gpt_local_spec


def initialize_gpt_model(seed, layer_spec_fn=gpt_te_spec, vocab_size=128, **config_kwargs):
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)

    default_config_kwargs=dict(num_layers=8, hidden_size=16, num_attention_heads=8, use_cpu_initialization=True)
    default_config_kwargs.update(**config_kwargs)
    transformer_config = TransformerConfig(**default_config_kwargs)
    pre_process = ps.is_pipeline_first_stage()
    post_process = ps.is_pipeline_last_stage()
    model = GPTModel(config=transformer_config, transformer_layer_spec=layer_spec_fn(), vocab_size=vocab_size, max_sequence_length=4,
                     pre_process=pre_process, post_process=post_process)

    with torch.no_grad():
        for p in model.parameters():
            p.random_()
    return model


class TestGPTModel:
    @pytest.mark.parametrize('src_layer_spec_fn', [gpt_te_spec, gpt_local_spec])
    @pytest.mark.parametrize('dst_layer_spec_fn', [gpt_te_spec, gpt_local_spec])
    def test_sharded_state_dict_save_load(self, tmp_path_dist_ckpt,
                                          src_layer_spec_fn, dst_layer_spec_fn):
        common_test_simple_sharded_state_dict_save_load(initialize_gpt_model, tmp_path_dist_ckpt,
                                                        src_layer_spec_fn, dst_layer_spec_fn)


class TestGPTModelReconfiguration:
    @pytest.mark.parametrize("use_fpsl", [False, True])
    @pytest.mark.parametrize("load_order,store_order", [
        ('tp-dp-pp', 'tp-dp-pp'),
        ('tp-pp-dp', 'tp-pp-dp'),
        ('tp-dp-pp', 'tp-pp-dp'),
    ])
    @pytest.mark.parametrize("src_tp_pp,dest_tp_pp,src_layer_spec_fn,dst_layer_spec_fn", [
        ((2, 4), (4, 2), gpt_te_spec, gpt_te_spec),
        ((1, 8), (8, 1), gpt_te_spec, gpt_te_spec),
        ((2, 1), (1, 8), gpt_te_spec, gpt_te_spec),
        ((1, 1), (2, 2), gpt_te_spec, gpt_te_spec),
        ((2, 1), (1, 8), gpt_local_spec, gpt_local_spec),
        ((1, 1), (2, 4), gpt_te_spec, gpt_local_spec),
        ((1, 8), (2, 1), gpt_local_spec, gpt_te_spec),
    ])
    def test_parallel_reconfiguration_e2e(self, tmp_path_dist_ckpt, src_tp_pp, dest_tp_pp,
                                          src_layer_spec_fn, dst_layer_spec_fn, use_fpsl, load_order, store_order):
        """ Test model saving and loading with different TP/PP """
        common_test_parallel_reconfiguration_e2e(initialize_gpt_model, tmp_path_dist_ckpt, src_tp_pp,
                                                 dest_tp_pp, src_layer_spec_fn, dst_layer_spec_fn, use_fpsl, load_order, store_order)


    def test_state_dict_comparison(self, tmp_path_dist_ckpt):
        common_test_state_dict_comparison(initialize_gpt_model, tmp_path_dist_ckpt)

    @pytest.mark.parametrize("vocab_size_base", [128, 17, 127, 31123])
    @pytest.mark.parametrize("src_tp_pp,dest_tp_pp", [
        ((2, 4), (4, 2)),
        ((1, 8), (8, 1)),
        ((1, 1), (1, 8)),
    ])
    def test_vocab_size_padding_change(self, tmp_path_dist_ckpt, vocab_size_base, src_tp_pp, dest_tp_pp):
        """ Test model loading with different vocab size (caused by TP padding). """
        common_test_vocab_size_padding_change(initialize_gpt_model, tmp_path_dist_ckpt, vocab_size_base,
                                              src_tp_pp, dest_tp_pp)
