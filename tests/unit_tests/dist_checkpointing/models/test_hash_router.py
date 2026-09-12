# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.dist_checkpointing import load, save
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def _build_hash_router(tp_size, vocab_size):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=8,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_router_load_balancing_type='none',
        moe_n_hash_layers=1,
        hash_moe_vocab_size=vocab_size,
        tensor_model_parallel_size=tp_size,
        use_cpu_initialization=True,
        add_bias_linear=False,
    )
    router = TopKRouter(config, pg_collection=get_default_pg_collection())
    router.set_layer_number(1)
    return router


class TestHashRouterReconfiguration:
    def teardown_method(self):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize('src_tp,dest_tp', [(1, 8), (8, 1)])
    def test_tid2eid_checkpoint_loads_across_tp_sizes(self, tmp_path_dist_ckpt, src_tp, dest_tp):
        vocab_size = 100003
        with TempNamedDir(tmp_path_dist_ckpt / 'hash_router_tp_resize') as checkpoint_dir:
            Utils.initialize_model_parallel(src_tp, 1)
            source = _build_hash_router(src_tp, vocab_size)
            expected = source.tid2eid.clone()
            save(source.sharded_state_dict(), checkpoint_dir)
            Utils.destroy_model_parallel()

            Utils.initialize_model_parallel(dest_tp, 1)
            destination = _build_hash_router(dest_tp, vocab_size)
            state_dict = load(destination.sharded_state_dict(), checkpoint_dir)
            destination.load_state_dict(state_dict)

            assert destination.tid2eid.shape == (vocab_size, 2)
            torch.testing.assert_close(destination.tid2eid, expected)
