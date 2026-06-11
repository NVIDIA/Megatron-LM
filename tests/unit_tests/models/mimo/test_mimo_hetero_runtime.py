# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Real-distributed (8-GPU, no parallel_state) tests for MIMO per-rank runtime setup."""

import argparse

import pytest
import torch

from examples.mimo.training.runtime import (
    _resolve_bucket_size,
    configure_module_rng,
    wrap_active_modules_with_ddp,
)
from examples.mimo.training.topology import ModuleGridSpec, create_topology
from megatron.core.distributed import DistributedDataParallel
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
from tests.unit_tests.models.mimo.test_mimo_1f1b_schedule import (
    get_language_model_spec,
    get_vision_submodules_spec,
)
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
        ddp_num_buckets=None,
        ddp_bucket_size=0,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _build_unwrapped_mimo_model(topo, hidden_size=16, num_layers=2, vocab_size=128, seq_len=8):
    """Build a bare (un-DDP-wrapped) MimoModel over a HeteroTopology's per-module PGCs."""
    language_spec = get_language_model_spec(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=4,
        vocab_size=vocab_size,
        seq_len=seq_len,
        pg_collection=topo.module_pgs[MIMO_LANGUAGE_MODULE_KEY],
        bf16=False,
    )
    vision_spec = get_vision_submodules_spec(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=4,
        language_hidden_size=hidden_size,
        pg_collection=topo.module_pgs[ENCODER],
        bf16=False,
    )
    mimo_config = MimoModelConfig(
        language_model_spec=language_spec,
        modality_submodules_spec={ENCODER: vision_spec},
        special_token_ids={ENCODER: 50257},
        module_to_grid_map=topo.grids,
    )
    mimo_model = MimoModel(mimo_config)
    mimo_model.to(torch.device("cuda"))
    return mimo_model


def _eight_gpu_topology():
    """Encoder dp=4 at ranks 0-3; language dp=4 at ranks 4-7 (non-colocated, tiles world)."""
    return create_topology(
        [
            ModuleGridSpec(name=ENCODER, num_ranks=4, rank_offset=0),
            ModuleGridSpec(name=MIMO_LANGUAGE_MODULE_KEY, num_ranks=4, rank_offset=4),
        ]
    )


class TestResolveBucketSize:
    def test_num_buckets_divides_param_count(self):
        module = torch.nn.Linear(8, 16, bias=False)  # 128 params
        args = _args(ddp_num_buckets=4)
        # num_buckets short-circuits before get_pg_size, so the group is unused.
        assert _resolve_bucket_size(args, module, None, overlap_grad_reduce=True) == 128 // 4

    def test_explicit_bucket_size_passes_through(self):
        args = _args(ddp_bucket_size=4096)
        assert (
            _resolve_bucket_size(args, torch.nn.Linear(2, 2), None, overlap_grad_reduce=True)
            == 4096
        )

    def test_overlap_off_returns_none(self):
        # get_model sets bucket_size=None when overlap_grad_reduce is off.
        assert (
            _resolve_bucket_size(_args(), torch.nn.Linear(2, 2), None, overlap_grad_reduce=False)
            is None
        )


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="requires 8 GPUs")
class TestResolveBucketSizeDefault:
    def setup_method(self, method):
        Utils.initialize_distributed()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_default_is_sane_max_over_dp_cp_size(self):
        topo = _eight_gpu_topology()
        try:
            dp_cp = topo.module_pgs[MIMO_LANGUAGE_MODULE_KEY].dp_cp
            dp_size = torch.distributed.get_world_size(group=dp_cp)
            expected = max(40_000_000, 1_000_000 * dp_size)
            got = _resolve_bucket_size(
                _args(), torch.nn.Linear(2, 2), dp_cp, overlap_grad_reduce=True
            )
            assert got == expected
        finally:
            topo.destroy()


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="requires 8 GPUs")
class TestPerRoleRng:
    def setup_method(self, method):
        Utils.initialize_distributed()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_distinct_offsets_give_distinct_rng_states(self):
        # Encoder tp=2,dp=2 at 0-3; language tp=2,pp=2 at 4-7.
        topo = create_topology(
            [
                ModuleGridSpec(name=ENCODER, num_ranks=4, tp=2, rank_offset=0),
                ModuleGridSpec(
                    name=MIMO_LANGUAGE_MODULE_KEY, num_ranks=4, tp=2, pp=2, rank_offset=4
                ),
            ]
        )
        try:
            args = _args()
            # configure_module_rng is only called with a module this rank participates in
            # (its groups are populated): encoder ranks 0-3, language ranks 4-7.
            module = MIMO_LANGUAGE_MODULE_KEY if torch.distributed.get_rank() >= 4 else ENCODER
            my_pgc = topo.module_pgs[module]
            configure_module_rng(args, my_pgc, role_seed_offset=10)
            states_a = get_cuda_rng_tracker().get_states()
            configure_module_rng(args, my_pgc, role_seed_offset=20)
            states_b = get_cuda_rng_tracker().get_states()
            # Different role offsets must reseed every tracked state.
            assert set(states_a) == set(states_b)
            for name in states_a:
                assert not torch.equal(states_a[name], states_b[name])
        finally:
            topo.destroy()


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="requires 8 GPUs")
class TestWrapActiveModulesWithDdp:
    def setup_method(self, method):
        Utils.initialize_distributed()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_active_modules_are_ddp(self):
        topo = _eight_gpu_topology()
        try:
            mimo_model = _build_unwrapped_mimo_model(topo)
            wrap_active_modules_with_ddp(_args(), mimo_model, topo)

            # Non-colocated: each rank owns exactly one active module (language XOR encoder).
            rank = torch.distributed.get_rank()
            if rank < 4:
                active = mimo_model.modality_submodules[ENCODER]
                assert mimo_model.language_model is None
            else:
                active = mimo_model.language_model
                assert ENCODER not in mimo_model.modality_submodules
            assert isinstance(active, DistributedDataParallel)
        finally:
            topo.destroy()
