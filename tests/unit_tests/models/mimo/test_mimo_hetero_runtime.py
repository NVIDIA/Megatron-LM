# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for MIMO per-rank runtime setup (RNG seeding, bucket sizing, DDP wrapping)."""

import argparse

import pytest
import torch

from examples.mimo.training.runtime import configure_module_rng, wrap_active_modules_with_ddp
from examples.mimo.training.topology import ModuleGridSpec, create_topology
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
from megatron.core.transformer.module import Float16Module
from megatron.core.utils import unwrap_model
from megatron.training.training import resolve_ddp_bucket_size
from tests.unit_tests.models.mimo.test_mimo_1f1b_schedule import (
    get_language_model_spec,
    get_vision_submodules_spec,
)
from tests.unit_tests.test_utilities import Utils

ENCODER = "images"


def _args(**overrides):
    base = dict(
        seed=1234, image_token_id=100, fp32=True, ddp_num_buckets=None, ddp_bucket_size=None
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _build_unwrapped_mimo_model(topo, bf16=False):
    """Build a bare (un-DDP-wrapped) MimoModel over a HeteroTopology's per-module PGCs."""
    mimo_config = MimoModelConfig(
        language_model_spec=get_language_model_spec(
            num_layers=2,
            hidden_size=16,
            num_attention_heads=4,
            vocab_size=128,
            seq_len=8,
            pg_collection=topo.module_pgs[MIMO_LANGUAGE_MODULE_KEY],
            bf16=bf16,
        ),
        modality_submodules_spec={
            ENCODER: get_vision_submodules_spec(
                num_layers=2,
                hidden_size=16,
                num_attention_heads=4,
                language_hidden_size=16,
                pg_collection=topo.module_pgs[ENCODER],
                bf16=bf16,
            )
        },
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


@pytest.mark.parametrize(
    "config, overlap, num_params, expected",
    [
        # num_buckets divides the param count.
        (DistributedDataParallelConfig(num_buckets=4), True, 128, 128 // 4),
        # explicit bucket_size passes through.
        (DistributedDataParallelConfig(bucket_size=4096), True, 256, 4096),
        # overlap off -> None, regardless of bucket_size.
        (DistributedDataParallelConfig(bucket_size=4096), False, 256, None),
        # no explicit size with group=None (dp size 1) -> the sane default.
        (DistributedDataParallelConfig(), True, 256, max(40_000_000, 1_000_000)),
    ],
)
def test_resolve_ddp_bucket_size(config, overlap, num_params, expected):
    """The MIMO wrap delegates bucket sizing to this shared get_model helper."""
    assert resolve_ddp_bucket_size(config, None, overlap, num_params) == expected


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="requires 8 GPUs")
class TestRuntimeDistributed:
    def setup_method(self, method):
        Utils.initialize_distributed()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_distinct_offsets_give_distinct_rng_states(self):
        # Encoder tp=2,dp=2 at 0-3; language tp=2,pp=2 at 4-7. Each rank seeds the one
        # module it participates in; distinct role offsets must reseed every tracked state.
        topo = create_topology(
            [
                ModuleGridSpec(name=ENCODER, num_ranks=4, tp=2, rank_offset=0),
                ModuleGridSpec(
                    name=MIMO_LANGUAGE_MODULE_KEY, num_ranks=4, tp=2, pp=2, rank_offset=4
                ),
            ]
        )
        try:
            module = MIMO_LANGUAGE_MODULE_KEY if torch.distributed.get_rank() >= 4 else ENCODER
            pgc = topo.module_pgs[module]
            configure_module_rng(_args(), pgc, role_seed_offset=10)
            states_a = get_cuda_rng_tracker().get_states()
            configure_module_rng(_args(), pgc, role_seed_offset=20)
            states_b = get_cuda_rng_tracker().get_states()
            assert set(states_a) == set(states_b)
            for name in states_a:
                assert not torch.equal(states_a[name], states_b[name])
        finally:
            topo.destroy()

    def test_active_module_is_ddp_over_its_own_grid(self):
        topo = _eight_gpu_topology()
        try:
            mimo_model = _build_unwrapped_mimo_model(topo)
            wrap_active_modules_with_ddp(_args(), mimo_model, topo)
            # Non-colocated: each rank owns exactly one active module (language XOR encoder).
            if torch.distributed.get_rank() < 4:
                active = mimo_model.modality_submodules[ENCODER]
                assert mimo_model.language_model is None
            else:
                active = mimo_model.language_model
                assert ENCODER not in mimo_model.modality_submodules
            assert isinstance(active, DistributedDataParallel)
        finally:
            topo.destroy()

    def test_bf16_wraps_in_float16module_and_freezes_targets(self):
        topo = _eight_gpu_topology()
        try:
            # bf16 -> Float16Module wrap; --freeze-vit freezes the encoder backbone only.
            mimo_model = _build_unwrapped_mimo_model(topo, bf16=True)
            wrap_active_modules_with_ddp(_args(fp32=False, freeze_vit=True), mimo_model, topo)

            if torch.distributed.get_rank() < 4:
                active = mimo_model.modality_submodules[ENCODER]
                # Float16Module sits under DDP, above the bare submodule.
                assert isinstance(active.module, Float16Module)
                submodule = unwrap_model(active)
                # --freeze-vit froze the encoder backbone, not the projector.
                assert all(not p.requires_grad for p in submodule.encoders.parameters())
                assert all(p.requires_grad for p in submodule.input_projections.parameters())
            else:
                assert isinstance(mimo_model.language_model.module, Float16Module)
        finally:
            topo.destroy()
