# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import types

import pytest
import torch

from megatron.core import parallel_state as ps
from megatron.core.dist_checkpointing import load, load_plain_tensors, save
from megatron.core.dist_checkpointing.validation import StrictHandling
from megatron.core.models.retro import RetroConfig, RetroModel, get_retro_decoder_block_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def initialize_retro_model(seed, decoder_spec_fn, spec_type, num_layers=9, **config_kwargs):
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)

    default_config_kwargs = dict(
        num_layers=num_layers,
        hidden_size=16,
        num_attention_heads=12,
        kv_channels=64,
        ffn_hidden_size=64,
        use_cpu_initialization=True,
        retro_num_neighbors=2,
        retro_chunk_length=4,
        retro_retrieved_length=8,
        retro_split_preprocessing="98,2,0",
    )
    default_config_kwargs.update(**config_kwargs)
    retro_config = RetroConfig(**default_config_kwargs)
    pre_process = ps.is_pipeline_first_stage()
    post_process = ps.is_pipeline_last_stage()

    de_block_spec = decoder_spec_fn(
        retro_config, use_transformer_engine=True if spec_type == "te" else False
    )
    model = RetroModel(
        config=retro_config,
        transformer_layer_spec=de_block_spec,
        pre_process=pre_process,
        post_process=post_process,
        vocab_size=29184,
        max_sequence_length=4,
    )

    with torch.no_grad():
        for p in model.parameters():
            p.random_()
    return model


class TestRetroModel:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize('src_spec_type', ['te', 'local'])
    @pytest.mark.parametrize('dst_spec_type', ['te', 'local'])
    @pytest.mark.parametrize('model_type', ['retro'])
    def test_sharded_state_dict_save_load(
        self, tmp_path_dist_ckpt, src_spec_type, dst_spec_type, model_type
    ):
        decoder_spec_fn = get_retro_decoder_block_spec

        Utils.initialize_model_parallel(1, 1)
        gpt_model = initialize_retro_model(2, decoder_spec_fn, src_spec_type)
        with TempNamedDir(tmp_path_dist_ckpt / 'test_gpt_model') as ckpt_dir:
            # Save
            sharded_state_dict = gpt_model.sharded_state_dict()
            save(sharded_state_dict, ckpt_dir)

            # Load
            gpt_model = initialize_retro_model(2, decoder_spec_fn, dst_spec_type)
            sharded_state_dict = gpt_model.sharded_state_dict()

            state_dict, missing_keys, unexpected_keys = load(
                sharded_state_dict, ckpt_dir, strict=StrictHandling.RETURN_ALL
            )
            # Potential mismatch is because of extra states which is ok
            assert all('_extra_state' in k for k in missing_keys)
            assert all('_extra_state' in k for k in unexpected_keys)
            gpt_model.load_state_dict(state_dict)
            gpt_model.load_state_dict(state_dict)

        Utils.destroy_model_parallel()
