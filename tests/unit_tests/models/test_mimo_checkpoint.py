# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Integration tests for MIMO distributed checkpoint save/load in non-colocated mode.

Run with 8 GPUs:
    uv run python -m torch.distributed.run --nproc-per-node=8 \
        -m pytest tests/unit_tests/models/test_mimo_checkpoint.py -v -s
"""

import os
import shutil
import tempfile

import pytest
import torch
import torch.distributed as dist
from packaging import version

from megatron.core.dist_checkpointing import load, save
from megatron.core.dist_checkpointing.validation import StrictHandling
from megatron.core.models.mimo.optimizer import get_mimo_optimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from tests.unit_tests.models.test_mimo_1f1b_schedule import (
    create_all_embedding_groups,
    create_hypercomm_grid,
    destroy_all_grids,
    get_mimo_model,
    is_rank_in_grid,
)
from tests.unit_tests.test_utilities import Utils

ENCODER_NAME = "images"


def _get_shared_tmpdir():
    """Create a shared temp directory across all ranks."""
    tmpdir_list = [None]
    if dist.get_rank() == 0:
        tmpdir_list[0] = tempfile.mkdtemp(prefix="mimo_ckpt_test_")
    dist.broadcast_object_list(tmpdir_list, src=0)
    return tmpdir_list[0]


def _cleanup_tmpdir(tmpdir):
    """Clean up temp directory (rank 0 only)."""
    dist.barrier()
    if dist.get_rank() == 0:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _randomize_params(model, seed):
    """Set all model parameters to deterministic random values."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.random_()


def _create_model_and_optimizer(encoder_grid, llm_grid, hidden_size, num_layers, vocab_size, seed):
    """Create MIMO model with DDP + optimizer, do a fake step to populate optimizer state.

    Caller must call create_all_embedding_groups() before this function.
    """
    torch.manual_seed(seed)

    mimo_model, _, _, _, _ = get_mimo_model(
        encoder_name=ENCODER_NAME,
        encoder_grid=encoder_grid,
        llm_grid=llm_grid,
        hidden_size=hidden_size,
        num_layers=num_layers,
        vocab_size=vocab_size,
        seq_len=64,
    )
    _randomize_params(mimo_model, seed)

    # Use Float16Optimizer (not ElementWiseDistributedOptimizer) to exercise the MIMO-specific
    # param_groups/grad_scaler extraction in sharded_state_dict. ElementWiseDistributedOptimizer
    # handles its own checkpointing internally and our code is transparent to it.
    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=1e-4,
        weight_decay=0.01,
        clip_grad=1.0,
        bf16=True,
        use_element_wise_distributed_optimizer=False,
    )
    optimizer = get_mimo_optimizer(mimo_model, opt_config)

    # Fake backward + step to populate optimizer state (Adam m/v)
    for param in mimo_model.parameters():
        param.grad = torch.randn_like(param)
    optimizer.step()

    return mimo_model, optimizer


def run_checkpoint_test(
    encoder_tp,
    encoder_pp,
    encoder_dp,
    encoder_offset,
    llm_tp,
    llm_pp,
    llm_dp,
    llm_offset,
    hidden_size=256,
    num_layers=2,
    vocab_size=1000,
):
    """Save model + optimizer checkpoint, load into fresh instances, verify match."""
    # Clear NVTE env vars that the conftest set_env fixture sets to '0'.
    # GPTModel (LanguageModule) asserts these are unset or match the attention backend.
    os.environ.pop('NVTE_FLASH_ATTN', None)
    os.environ.pop('NVTE_FUSED_ATTN', None)
    os.environ.pop('NVTE_UNFUSED_ATTN', None)

    encoder_grid = create_hypercomm_grid(
        offset=encoder_offset, tp=encoder_tp, cp=1, pp=encoder_pp, dp=encoder_dp
    )
    llm_grid = create_hypercomm_grid(offset=llm_offset, tp=llm_tp, cp=1, pp=llm_pp, dp=llm_dp)
    create_all_embedding_groups([encoder_grid, llm_grid])

    # --- Create model A + optimizer, snapshot state ---
    model_a, optimizer_a = _create_model_and_optimizer(
        encoder_grid, llm_grid, hidden_size, num_layers, vocab_size, seed=1
    )
    params_a = {name: p.clone() for name, p in model_a.named_parameters()}

    ckpt_dir = _get_shared_tmpdir()
    try:
        model_ckpt = os.path.join(ckpt_dir, 'model')
        optim_ckpt = os.path.join(ckpt_dir, 'optimizer')
        if dist.get_rank() == 0:
            os.makedirs(model_ckpt)
            os.makedirs(optim_ckpt)
        dist.barrier()

        # Save model
        save(model_a.sharded_state_dict(), model_ckpt)

        # Save optimizer (needs fresh model sharded_state_dict since save() consumes tensor refs)
        optim_sd_a = optimizer_a.sharded_state_dict(model_a.sharded_state_dict(), is_loading=False)
        save(optim_sd_a, optim_ckpt, validate_access_integrity=False)

        dist.barrier()

        # --- Create model B + optimizer with different weights (reuse same grids) ---
        model_b, optimizer_b = _create_model_and_optimizer(
            encoder_grid, llm_grid, hidden_size, num_layers, vocab_size, seed=2
        )

        # Load model
        model_sd_b = model_b.sharded_state_dict()
        loaded_model_sd, missing, unexpected = load(
            model_sd_b, model_ckpt, strict=StrictHandling.RETURN_ALL
        )
        real_missing = [k for k in missing if '_extra_state' not in k]
        real_unexpected = [k for k in unexpected if '_extra_state' not in k]
        assert not real_missing, f"Missing keys: {real_missing}"
        assert not real_unexpected, f"Unexpected keys: {real_unexpected}"
        model_b.load_state_dict(loaded_model_sd)

        # Load optimizer
        optim_sd_b = optimizer_b.sharded_state_dict(model_b.sharded_state_dict(), is_loading=True)
        loaded_optim_sd = load(optim_sd_b, optim_ckpt, validate_access_integrity=False)
        optimizer_b.load_state_dict(loaded_optim_sd)

        # --- Verify model params match ---
        mismatches = [
            name
            for name, p in model_b.named_parameters()
            if name in params_a and not torch.equal(p, params_a[name])
        ]
        assert not mismatches, f"Model param mismatch after load: {mismatches}"

        # --- Verify optimizer state matches (param_groups + Adam m/v tensors) ---
        for name, info_b in optimizer_b.module_infos.items():
            if not (info_b.is_active and info_b.optimizer):
                continue
            info_a = optimizer_a.module_infos[name]
            sd_a = info_a.optimizer.state_dict()
            sd_b = info_b.optimizer.state_dict()

            # Verify param_groups
            pg_a = sd_a.get('optimizer', {}).get('param_groups', [])
            pg_b = sd_b.get('optimizer', {}).get('param_groups', [])
            assert len(pg_a) == len(pg_b), f"Optimizer {name}: param_groups count mismatch"
            for i, (ga, gb) in enumerate(zip(pg_a, pg_b)):
                assert ga['lr'] == gb['lr'], f"Optimizer {name} group[{i}]: lr mismatch"

            # Verify Adam state tensors (exp_avg, exp_avg_sq)
            state_a = sd_a.get('optimizer', {}).get('state', {})
            state_b = sd_b.get('optimizer', {}).get('state', {})
            for param_id in state_a:
                if param_id not in state_b:
                    continue
                for key in ('exp_avg', 'exp_avg_sq'):
                    if key in state_a[param_id] and key in state_b[param_id]:
                        assert torch.equal(
                            state_a[param_id][key], state_b[param_id][key]
                        ), f"Optimizer {name} param {param_id} {key} mismatch"

    finally:
        _cleanup_tmpdir(ckpt_dir)


@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse('2.3.0'),
    reason="Device mesh requires PyTorch 2.3+",
)
class TestMimoCheckpoint:
    """Distributed checkpoint save/load tests for non-colocated MiMo (8 GPUs)."""

    @classmethod
    def setup_class(cls):
        Utils.initialize_distributed()
        cls.world_size = dist.get_world_size()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def teardown_method(self):
        destroy_all_grids()

    def test_encoder_tp2_llm_tp2_pp3(self):
        if self.world_size != 8:
            pytest.skip(f"Requires 8 GPUs, got {self.world_size}")
        run_checkpoint_test(
            encoder_tp=2,
            encoder_pp=1,
            encoder_dp=1,
            encoder_offset=0,
            llm_tp=2,
            llm_pp=3,
            llm_dp=1,
            llm_offset=2,
            hidden_size=256,
            num_layers=3,
        )

    def test_encoder_tp1_llm_pp7(self):
        if self.world_size != 8:
            pytest.skip(f"Requires 8 GPUs, got {self.world_size}")
        run_checkpoint_test(
            encoder_tp=1,
            encoder_pp=1,
            encoder_dp=1,
            encoder_offset=0,
            llm_tp=1,
            llm_pp=7,
            llm_dp=1,
            llm_offset=1,
            hidden_size=256,
            num_layers=7,
        )

    def test_encoder_tp2_pp2_llm_tp2_pp2(self):
        if self.world_size != 8:
            pytest.skip(f"Requires 8 GPUs, got {self.world_size}")
        run_checkpoint_test(
            encoder_tp=2,
            encoder_pp=2,
            encoder_dp=1,
            encoder_offset=0,
            llm_tp=2,
            llm_pp=2,
            llm_dp=1,
            llm_offset=4,
            hidden_size=256,
            num_layers=2,
        )
