# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for ``get_mimo_optimizer`` skipping inner optimizers when a module is
fully frozen across all ranks in its optimizer group.

Covers the fix that uses ``_module_has_any_trainable_parameters`` (an
all-reduce MAX over ``pg_collection.intra_dist_opt``) to decide whether
``get_megatron_optimizer`` should be invoked for a module. Without the
guard, fully frozen modules (e.g. projector-only training with a frozen
vision encoder) produce empty optimizer placeholders that crash later in
setup.

Run with 8 GPUs::

    uv run python -m torch.distributed.run --nproc-per-node=8 \\
        -m pytest tests/unit_tests/models/test_mimo_frozen_modules.py -v -s
"""

import os

import pytest
import torch
import torch.distributed as dist
from packaging import version

from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.optimizer import (
    _module_has_any_trainable_parameters,
    get_mimo_optimizer,
)
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.process_groups_config import ProcessGroupCollection
from tests.unit_tests.models.test_mimo_1f1b_schedule import (
    create_all_embedding_groups,
    create_hypercomm_grid,
    destroy_all_grids,
    get_mimo_model,
)
from tests.unit_tests.test_utilities import Utils

ENCODER_NAME = "images"


def _freeze_module(module):
    """Set ``requires_grad=False`` on every parameter of ``module``.

    ``module`` is the DDP-wrapped submodule returned by ``get_mimo_model``;
    ``parameters()`` walks the inner module so the flag propagates.
    """
    for p in module.parameters():
        p.requires_grad_(False)


def _make_opt_config():
    """Adam OptimizerConfig matching the existing MIMO checkpoint test."""
    return OptimizerConfig(
        optimizer='adam',
        lr=1e-4,
        weight_decay=0.0,
        clip_grad=1.0,
        bf16=True,
        use_distributed_optimizer=False,
    )


def _populate_synthetic_grads(model):
    """Set ``.grad`` on every trainable parameter so ``optimizer.step()`` has
    something to consume. Frozen params are skipped so the optimizer never
    observes a grad it does not own.
    """
    for param in model.parameters():
        if param.requires_grad:
            param.grad = torch.randn_like(param)


@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse('2.3.0'),
    reason="HyperCommGrid device-mesh path requires PyTorch 2.3+",
)
class TestMimoFrozenModules:
    """Validate that fully frozen MIMO modules do not get an inner optimizer."""

    @classmethod
    def setup_class(cls):
        Utils.initialize_distributed()
        cls.world_size = dist.get_world_size()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def teardown_method(self):
        destroy_all_grids()

    def _build_mimo(
        self,
        encoder_grid,
        llm_grid,
        hidden_size=128,
        num_layers=2,
        vocab_size=512,
        seq_len=64,
    ):
        # Clear NVTE backend env vars that conftest's set_env fixture pins to '0';
        # GPTModel asserts they are unset or match the chosen attention backend.
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)

        create_all_embedding_groups([encoder_grid, llm_grid])
        mimo_model, _, _, _, _ = get_mimo_model(
            encoder_name=ENCODER_NAME,
            encoder_grid=encoder_grid,
            llm_grid=llm_grid,
            hidden_size=hidden_size,
            num_layers=num_layers,
            vocab_size=vocab_size,
            seq_len=seq_len,
        )
        return mimo_model

    def test_skip_optimizer_for_fully_frozen_encoder_non_colocated(self):
        """Non-colocated, encoder frozen.

        Encoder grid on ranks 0-3 (TP=2 DP=2), LLM grid on ranks 4-7 (TP=2 DP=2).
        All encoder params frozen everywhere. ``get_mimo_optimizer`` must
        produce a ``None`` inner optimizer for the encoder while keeping the
        LLM inner optimizer intact.
        """
        if self.world_size != 8:
            pytest.skip(f"Requires 8 GPUs, got {self.world_size}")

        encoder_grid = create_hypercomm_grid(offset=0, tp=2, cp=1, pp=1, dp=2)
        llm_grid = create_hypercomm_grid(offset=4, tp=2, cp=1, pp=1, dp=2)

        mimo_model = self._build_mimo(encoder_grid, llm_grid)

        encoder = mimo_model.modality_submodules.get(ENCODER_NAME)
        if encoder is not None:
            _freeze_module(encoder)

        optimizer = get_mimo_optimizer(mimo_model, _make_opt_config())

        encoder_info = optimizer.module_infos[ENCODER_NAME]
        llm_info = optimizer.module_infos[MIMO_LANGUAGE_MODULE_KEY]
        rank = dist.get_rank()

        if rank < 4:
            # Encoder ranks
            assert encoder_info.is_active, "Encoder rank should be active for the encoder module"
            assert encoder_info.optimizer is None, (
                "Encoder inner optimizer must be skipped when fully frozen"
            )
            assert not llm_info.is_active
            assert llm_info.optimizer is None
        else:
            # LLM ranks
            assert not encoder_info.is_active
            assert encoder_info.optimizer is None
            assert llm_info.is_active, "LLM rank should be active for the LLM module"
            assert llm_info.optimizer is not None, (
                "LLM inner optimizer must be built when params are trainable"
            )

        # Smoke step: must not crash under the stub-optimizer plumbing on
        # encoder ranks, nor when no_trainable + trainable groups co-exist.
        _populate_synthetic_grads(mimo_model)
        optimizer.step()

    def test_skip_optimizer_for_fully_frozen_encoder_colocated(self):
        """Colocated, encoder frozen.

        Encoder TP=2 DP=4 and LLM TP=4 DP=2 both span ranks 0-7. Every rank
        owns both modules; freezing the encoder must skip the encoder inner
        optimizer on every rank while leaving the LLM inner optimizer intact.
        """
        if self.world_size != 8:
            pytest.skip(f"Requires 8 GPUs, got {self.world_size}")

        encoder_grid = create_hypercomm_grid(offset=0, tp=2, cp=1, pp=1, dp=4)
        llm_grid = create_hypercomm_grid(offset=0, tp=4, cp=1, pp=1, dp=2)

        mimo_model = self._build_mimo(encoder_grid, llm_grid)

        encoder = mimo_model.modality_submodules.get(ENCODER_NAME)
        if encoder is not None:
            _freeze_module(encoder)

        optimizer = get_mimo_optimizer(mimo_model, _make_opt_config())

        encoder_info = optimizer.module_infos[ENCODER_NAME]
        llm_info = optimizer.module_infos[MIMO_LANGUAGE_MODULE_KEY]

        assert encoder_info.is_active, "All ranks should be active for both modules in colocated"
        assert encoder_info.optimizer is None, (
            "Encoder inner optimizer must be skipped on every rank when fully frozen"
        )
        assert llm_info.is_active
        assert llm_info.optimizer is not None, (
            "LLM inner optimizer must be built when params are trainable"
        )

        _populate_synthetic_grads(mimo_model)
        optimizer.step()

    def test_module_has_any_trainable_parameters_helper(self):
        """Direct unit on ``_module_has_any_trainable_parameters``.

        Reduces over ``ProcessGroupCollection.intra_dist_opt = WORLD`` so the
        per-case truth is consistent across all ranks. Covers all-trainable,
        all-frozen, and the partial case where only one rank holds a
        trainable parameter (MAX reduction must surface True everywhere).
        """
        if self.world_size < 2:
            pytest.skip("Requires >= 2 ranks to exercise the all-reduce")

        pg = ProcessGroupCollection()
        pg.intra_dist_opt = dist.group.WORLD

        linear = torch.nn.Linear(8, 8).cuda()

        # All ranks trainable -> True.
        for p in linear.parameters():
            p.requires_grad_(True)
        assert _module_has_any_trainable_parameters(linear, pg) is True

        # All ranks frozen -> False.
        for p in linear.parameters():
            p.requires_grad_(False)
        assert _module_has_any_trainable_parameters(linear, pg) is False

        # Only rank 0 trainable, rest frozen -> MAX surfaces True everywhere.
        if dist.get_rank() == 0:
            for p in linear.parameters():
                p.requires_grad_(True)
        assert _module_has_any_trainable_parameters(linear, pg) is True
