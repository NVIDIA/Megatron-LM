# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core import parallel_state as ps
from megatron.core.dist_checkpointing import load, save
from megatron.core.dist_checkpointing.validation import StrictHandling
from megatron.core.models.retro.decoder_spec import (
    get_retro_decoder_layer_local_spec,
    get_retro_decoder_layer_te_spec,
)
from megatron.core.models.retro.encoder_spec import (
    get_retro_encoder_layer_local_spec,
    get_retro_encoder_layer_te_spec,
)
from megatron.core.models.T5 import T5Model
from megatron.core.models.T5.t5_spec import decoder_model_with_local_spec as t5_decoder_local_spec
from megatron.core.models.T5.t5_spec import (
    decoder_model_with_transformer_engine_default_spec as t5_decoder_te_spec,
)
from megatron.core.models.T5.t5_spec import encoder_model_with_local_spec as t5_encoder_local_spec
from megatron.core.models.T5.t5_spec import (
    encoder_model_with_transformer_engine_default_spec as t5_encoder_te_spec,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.dist_checkpointing.models.common import (
    common_test_parallel_reconfiguration_e2e,
)
from tests.unit_tests.test_utilities import Utils


def initialize_t5_model(seed, encoder_decoder_spec_fn, num_layers=8, **config_kwargs):
    encoder_spec_fn, decoder_spec_fn = encoder_decoder_spec_fn
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)

    add_encoder = None
    add_decoder = None

    encoder_layers_per_pipeline = num_layers
    decoder_layers_per_pipeline = num_layers

    pre_process = ps.is_pipeline_first_stage()
    post_process = ps.is_pipeline_last_stage()

    default_config_kwargs = dict(
        num_layers=num_layers,
        hidden_size=16,
        num_attention_heads=12,
        kv_channels=64,
        ffn_hidden_size=64,
        use_cpu_initialization=True,
        pipeline_dtype=torch.bfloat16,
    )
    default_config_kwargs.update(**config_kwargs)
    transformer_config = TransformerConfig(**default_config_kwargs)

    en_block_spec = TransformerBlockSubmodules([encoder_spec_fn()] * encoder_layers_per_pipeline)
    de_block_spec = TransformerBlockSubmodules([decoder_spec_fn()] * decoder_layers_per_pipeline)
    model = T5Model(
        encoder_config=transformer_config,
        config=transformer_config,
        transformer_encoder_layer_spec=en_block_spec,
        transformer_decoder_layer_spec=de_block_spec,
        vocab_size=29184,
        max_sequence_length=4,
        pre_process=pre_process,
        post_process=post_process,
        add_encoder=add_encoder,
        add_decoder=add_decoder,
    )

    with torch.no_grad():
        for p in model.parameters():
            p.random_()
    return model


class TestT5Model:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize('src_spec_type', ['te', 'local'])
    @pytest.mark.parametrize('dst_spec_type', ['te', 'local'])
    @pytest.mark.parametrize('model_type', ['t5'])
    def test_sharded_state_dict_save_load(
        self, tmp_path_dist_ckpt, src_spec_type, dst_spec_type, model_type
    ):
        enc_dec_spec_fn = {
            'te': {
                't5': (t5_encoder_te_spec, t5_decoder_te_spec),
                'retro': (get_retro_encoder_layer_te_spec, get_retro_decoder_layer_te_spec),
            },
            'local': {
                't5': (t5_encoder_local_spec, t5_decoder_local_spec),
                'retro': (get_retro_encoder_layer_local_spec, get_retro_decoder_layer_local_spec),
            },
        }
        src_encoder_decoder_spec_fn = enc_dec_spec_fn[src_spec_type][model_type]
        dst_encoder_decoder_spec_fn = enc_dec_spec_fn[dst_spec_type][model_type]

        Utils.initialize_model_parallel(1, 1)
        gpt_model = initialize_t5_model(1, src_encoder_decoder_spec_fn)
        with TempNamedDir(tmp_path_dist_ckpt / 'test_gpt_model') as ckpt_dir:
            # Save
            sharded_state_dict = gpt_model.sharded_state_dict()
            save(sharded_state_dict, ckpt_dir)

            # Load
            gpt_model = initialize_t5_model(2, dst_encoder_decoder_spec_fn)
            sharded_state_dict = gpt_model.sharded_state_dict()

            state_dict, missing_keys, unexpected_keys = load(
                sharded_state_dict, ckpt_dir, strict=StrictHandling.RETURN_ALL
            )
            # Potential mismatch is because of extra states which is ok
            assert all('_extra_state' in k for k in missing_keys)
            assert all('_extra_state' in k for k in unexpected_keys)
            gpt_model.load_state_dict(state_dict)

        Utils.destroy_model_parallel()


class TestT5ModelReconfiguration:

    # def teardown_method(self, method):
    #     Utils.destroy_model_parallel()

    @pytest.mark.parametrize('src_spec_type', ['local'])  # ['te', 'local'])
    @pytest.mark.parametrize('dst_spec_type', ['local'])  # ['te', 'local'])
    @pytest.mark.parametrize('model_type', ['t5'])
    @pytest.mark.parametrize(
        ('use_fpsl', 'src_tp_pp_encpp', 'dest_tp_pp_encpp'), [(False, (1, 1, None), (1, 1, None))]
    )
    def test_parallel_reconfiguration_e2e(
        self,
        tmp_path_dist_ckpt,
        src_tp_pp_encpp,
        dest_tp_pp_encpp,
        use_fpsl,
        src_spec_type,
        dst_spec_type,
        model_type,
    ):
        """Test model saving and loading with different TP/PP"""

        *src_tp_pp, src_encpp = src_tp_pp_encpp
        *dest_tp_pp, dst_encpp = dest_tp_pp_encpp

        enc_dec_spec_fn = {
            'te': {
                't5': (t5_encoder_te_spec, t5_decoder_te_spec),
                'retro': (get_retro_encoder_layer_te_spec, get_retro_decoder_layer_te_spec),
            },
            'local': {
                't5': (t5_encoder_local_spec, t5_decoder_local_spec),
                'retro': (get_retro_encoder_layer_local_spec, get_retro_decoder_layer_local_spec),
            },
        }

        common_test_parallel_reconfiguration_e2e(
            initialize_t5_model,
            tmp_path_dist_ckpt,
            src_tp_pp,
            dest_tp_pp,
            enc_dec_spec_fn[src_spec_type][model_type],
            enc_dec_spec_fn[dst_spec_type][model_type],
            use_fpsl,
        )
