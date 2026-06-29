# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Real-distributed (8-GPU, no parallel_state) tests for MIMO role-aware data selection."""

import argparse

import pytest
import torch

from examples.mimo.training.data import MockVLMIterator, select_data_iterator
from examples.mimo.training.topology import ModuleGridSpec, create_topology
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from tests.unit_tests.test_utilities import Utils

ENCODER = "images"


def _args(**overrides):
    base = dict(
        seed=1234,
        dataset_provider="mock",
        micro_batch_size=2,
        llm_dp=1,
        encoder_dp=1,
        image_seq_length=4,
        seq_length=8,
        vocab_size=128,
        hidden_size=16,
        image_token_id=100,
        fp32=True,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="requires 8 GPUs")
class TestRoleAwareDataSelection:
    def setup_method(self, method):
        Utils.initialize_distributed()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_encoder_and_language_edges_get_iterator(self):
        # Encoder pp=1 (every rank PP-first) at 0-3; language tp=2,pp=2 at 4-7 (both stages edges).
        topo = create_topology(
            [
                ModuleGridSpec(name=ENCODER, num_ranks=4, tp=2, rank_offset=0),
                ModuleGridSpec(
                    name=MIMO_LANGUAGE_MODULE_KEY, num_ranks=4, tp=2, pp=2, rank_offset=4
                ),
            ]
        )
        try:
            it = select_data_iterator(_args(), topo)
            # Every rank here is either an encoder PP-first or a language PP-edge rank.
            assert isinstance(it, MockVLMIterator)
        finally:
            topo.destroy()

    def test_language_middle_stage_gets_no_iterator(self):
        # Language-only tp=2,pp=4 over the whole world: pp ranks 1 and 2 are non-edge -> None.
        topo = create_topology(
            [ModuleGridSpec(name=MIMO_LANGUAGE_MODULE_KEY, num_ranks=8, tp=2, pp=4, rank_offset=0)]
        )
        try:
            pp_rank = topo.module_pgs[MIMO_LANGUAGE_MODULE_KEY].pp.rank()
            it = select_data_iterator(_args(), topo)
            if pp_rank in (0, 3):
                assert isinstance(it, MockVLMIterator)
            else:
                assert it is None
        finally:
            topo.destroy()
