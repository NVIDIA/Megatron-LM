# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for MIMO per-rank runtime setup (RNG seeding, DDP wrapping)."""

import argparse
from types import SimpleNamespace

import pytest
import torch

from examples.mimo.training.runtime import configure_module_rng, wrap_active_modules_with_ddp
from examples.mimo.training.topology import ModuleGridSpec, create_topology
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
from megatron.core.transformer.module import Float16Module
from megatron.core.utils import unwrap_model
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


def test_builder_seeds_per_role_meta_builds_and_sets_contract(mocker):
    """The non-colocated builder seeds the one active role and sets the model contract."""
    from examples.mimo.training.builder import (
        _LANGUAGE_SEED_OFFSET,
        MimoBuildConfig,
        MimoModelBuilder,
    )

    args = _args(init_model_with_meta_device=True)
    groups = mocker.Mock()
    model = SimpleNamespace(language_model=mocker.Mock(), modality_submodules={})
    topology = mocker.Mock()
    builder = MimoModelBuilder(MimoBuildConfig(_topology=topology))
    mocker.patch("examples.mimo.training.builder.get_args", return_value=args)
    mocker.patch(
        "examples.mimo.training.builder._resolve_role",
        return_value=(MIMO_LANGUAGE_MODULE_KEY, True, groups),
    )
    mocker.patch.object(builder, "build_model", return_value=model)
    wrap = mocker.patch("examples.mimo.training.builder.wrap_active_modules_with_ddp")
    grad_sync = mocker.patch("examples.mimo.training.builder.configure_grad_sync")
    torch_device = mocker.patch(
        "examples.mimo.training.builder.torch.device", return_value=mocker.MagicMock()
    )
    seed = mocker.patch("examples.mimo.training.builder.configure_module_rng")

    assert builder.build_distributed_models(
        mocker.Mock(), ddp_config=DistributedDataParallelConfig(), data_parallel_random_init=True
    ) == [model]

    torch_device.assert_called_once_with("meta")
    seed.assert_called_once_with(args, groups, _LANGUAGE_SEED_OFFSET, True)
    wrap.assert_called_once_with(args, model, topology, True)
    grad_sync.assert_called_once_with(args, model, topology)
    # Load-bearing contract for Increments 2/4: own module PGC and role prefix on the model.
    assert model.pg_collection is groups
    assert model.rng_state_key_prefix == "language."


def test_builder_encoder_role_sets_encoder_contract(mocker):
    """On an encoder-only rank the builder seeds/labels with the encoder role."""
    from examples.mimo.training.builder import (
        _ENCODER_SEED_OFFSET,
        MimoBuildConfig,
        MimoModelBuilder,
    )

    args = _args(init_model_with_meta_device=False)
    encoder_pg = mocker.Mock()
    model = SimpleNamespace(language_model=None, modality_submodules={ENCODER: mocker.Mock()})
    builder = MimoModelBuilder(MimoBuildConfig(_topology=mocker.Mock()))
    mocker.patch("examples.mimo.training.builder.get_args", return_value=args)
    mocker.patch(
        "examples.mimo.training.builder._resolve_role",
        return_value=(ENCODER, False, encoder_pg),
    )
    mocker.patch.object(builder, "build_model", return_value=model)
    mocker.patch("examples.mimo.training.builder.wrap_active_modules_with_ddp")
    mocker.patch("examples.mimo.training.builder.configure_grad_sync")
    seed = mocker.patch("examples.mimo.training.builder.configure_module_rng")

    builder.build_distributed_models(mocker.Mock(), ddp_config=DistributedDataParallelConfig())

    seed.assert_called_once_with(args, encoder_pg, _ENCODER_SEED_OFFSET, False)
    assert model.pg_collection is encoder_pg
    assert model.rng_state_key_prefix == "encoder."


def test_resolve_role_rejects_colocated_or_zero_active_roles(mocker):
    """Colocated (both) or zero active roles are not supported (non-colocated only)."""
    from examples.mimo.training.builder import _resolve_role

    def _topology(active_modules):
        grids = {}
        for name in (MIMO_LANGUAGE_MODULE_KEY, ENCODER):
            grid = mocker.Mock()
            grid.is_current_rank_in_grid.return_value = name in active_modules
            grids[name] = grid
        return SimpleNamespace(grids=grids, module_pgs={})

    with pytest.raises(ValueError, match="exactly one active language or encoder role"):
        _resolve_role(_topology({MIMO_LANGUAGE_MODULE_KEY, ENCODER}))
    with pytest.raises(ValueError, match="exactly one active language or encoder role"):
        _resolve_role(_topology(set()))


def test_builder_applies_outer_hooks_in_order_and_returns_replacement(mocker):
    """Outer MIMO hooks surround preparation (pre -> wrap -> configure -> post) with replacement."""
    from examples.mimo.training.builder import MimoBuildConfig, MimoModelBuilder

    events = []
    original_model = SimpleNamespace()
    pre_replacement = SimpleNamespace()
    post_replacement = SimpleNamespace()

    def pre_hook(model_list):
        assert model_list == [original_model]
        assert original_model.model_type == ModelType.encoder_or_decoder
        events.append("pre")
        return [pre_replacement]

    def post_hook(model_list):
        assert model_list == [pre_replacement]
        events.append("post")
        return [post_replacement]

    groups = mocker.Mock()
    config = MimoBuildConfig(
        _topology=mocker.Mock(),
        pre_wrap_hooks=[pre_hook],
        post_wrap_hooks=[post_hook],
    )
    builder = MimoModelBuilder(config)
    mocker.patch(
        "examples.mimo.training.builder.get_args",
        return_value=_args(init_model_with_meta_device=False),
    )
    mocker.patch(
        "examples.mimo.training.builder._resolve_role",
        return_value=(MIMO_LANGUAGE_MODULE_KEY, True, groups),
    )
    mocker.patch("examples.mimo.training.builder.configure_module_rng")
    mocker.patch.object(builder, "build_model", return_value=original_model)
    mocker.patch(
        "examples.mimo.training.builder.wrap_active_modules_with_ddp",
        side_effect=lambda *_: events.append("wrap"),
    )
    mocker.patch(
        "examples.mimo.training.builder.configure_grad_sync",
        side_effect=lambda *_: events.append("configure"),
    )

    result = builder.build_distributed_models(
        mocker.Mock(), ddp_config=DistributedDataParallelConfig()
    )

    assert events == ["pre", "wrap", "configure", "post"]
    assert result == [post_replacement]


@pytest.mark.parametrize(
    ("hook_stage", "model_count"), [("pre", 0), ("pre", 2), ("post", 0), ("post", 2)]
)
def test_builder_rejects_invalid_outer_hook_cardinality(mocker, hook_stage, model_count):
    """MIMO outer hooks must preserve the builder's single-model contract."""
    from examples.mimo.training.builder import MimoBuildConfig, MimoModelBuilder

    replacement_models = [SimpleNamespace() for _ in range(model_count)]
    hook_kwargs = {"pre_wrap_hooks": [], "post_wrap_hooks": []}
    hook_kwargs[f"{hook_stage}_wrap_hooks"] = [lambda _models: replacement_models]
    builder = MimoModelBuilder(MimoBuildConfig(_topology=mocker.Mock(), **hook_kwargs))
    mocker.patch(
        "examples.mimo.training.builder.get_args",
        return_value=_args(init_model_with_meta_device=False),
    )
    mocker.patch(
        "examples.mimo.training.builder._resolve_role",
        return_value=(MIMO_LANGUAGE_MODULE_KEY, True, mocker.Mock()),
    )
    mocker.patch("examples.mimo.training.builder.configure_module_rng")
    mocker.patch.object(builder, "build_model", return_value=SimpleNamespace())
    mocker.patch("examples.mimo.training.builder.wrap_active_modules_with_ddp")
    mocker.patch("examples.mimo.training.builder.configure_grad_sync")

    with pytest.raises(
        ValueError,
        match=f"MIMO {hook_stage}-wrap hooks must return exactly one outer model; got {model_count}",
    ):
        builder.build_distributed_models(mocker.Mock(), ddp_config=DistributedDataParallelConfig())


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
            configure_module_rng(_args(), pgc, role_seed_offset=10, data_parallel_random_init=True)
            states_a = get_cuda_rng_tracker().get_states()
            configure_module_rng(_args(), pgc, role_seed_offset=20, data_parallel_random_init=True)
            states_b = get_cuda_rng_tracker().get_states()
            assert set(states_a) == set(states_b)
            for name in states_a:
                assert not torch.equal(states_a[name], states_b[name])
        finally:
            topo.destroy()

    def test_active_module_is_ddp_over_its_own_grid(self):
        topo = _eight_gpu_topology()
        try:
            # bf16 = production precision; a bare fp32 modality container has no config for get_model_config.
            mimo_model = _build_unwrapped_mimo_model(topo, bf16=True)
            wrap_active_modules_with_ddp(_args(fp32=False), mimo_model, topo)
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
