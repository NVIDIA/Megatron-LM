# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch

from megatron.core.transformer.experimental_attention_variant.csa2_module_spec import (
    csa2_attention_spec,
)
from megatron.core.transformer.spec_utils import build_module
from tests.unit_tests.determinism.kernels.harness import (
    assert_module_replays_bit_exact,
    deterministic_algorithms,
    seeded,
)
from tests.unit_tests.models.test_deepseek_v41 import groups, tiny_config


def test_csa2_module_replay(groups):
    seeded()
    config = tiny_config(
        num_layers=1,
        csa_compress_ratios=[2],
        csa2_kv_source_layers=[0],
        csa2_index_source_layers=[0],
        csa2_candidate_source_layer=None,
        csa2_candidate_topk_blocks=0,
        csa2_candidate_block_size=0,
        dsa_indexer_loss_coeff=0,
    )
    layer = build_module(
        csa2_attention_spec, config=config, layer_number=1, pg_collection=groups
    ).cuda()
    with deterministic_algorithms(True):
        assert_module_replays_bit_exact(
            layer, (torch.randn(9, 2, 32, device="cuda", requires_grad=True), None)
        )
