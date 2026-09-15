# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from functools import partial
from pathlib import Path
from unittest import mock

import pytest
import torch

from megatron.core.dist_checkpointing.dict_utils import diff
from megatron.training.arguments import parse_args
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from tests.unit_tests.dist_checkpointing import (
    TempNamedDir,
    init_checkpointing_mock_args,
    initialize_gpt_model,
    setup_model_and_optimizer,
)
from tests.unit_tests.test_utilities import Utils

# Match by suffix (any layer, on every pipeline stage) rather than one exact
# global/local-indexed name: this mirrors a realistic PEFT setup where an adapter
# attaches uniformly to every layer, not just one -- and avoids a PP-local
# ModuleList index (e.g. "layers.0...") mapping onto a *different* global layer per
# pipeline stage, which Megatron-Core represents as one merged ShardedTensor
# spanning all layers for a dense (non-MoE, non-heterogeneous) model.
ADAPTER_PARAM_SUFFIX = "input_layernorm.weight"


def _freeze_all_but_adapter(model_chunks):
    """Unfreeze every parameter whose name ends with ADAPTER_PARAM_SUFFIX, freeze the
    rest. Returns the count of parameters unfrozen on this rank's local model chunks.
    """
    count = 0
    for m in model_chunks:
        for name, p in m.named_parameters():
            if name.endswith(ADAPTER_PARAM_SUFFIX):
                p.requires_grad_(True)
                count += 1
            else:
                p.requires_grad_(False)
    return count


def _initialize_peft_model(*args, use_glu, **kwargs):
    model = initialize_gpt_model(*args, use_glu=use_glu, **kwargs)
    _freeze_all_but_adapter([model])
    return model


def _set_adapter_params(model_chunks, value):
    for m in model_chunks:
        for name, p in m.named_parameters():
            if name.endswith(ADAPTER_PARAM_SUFFIX):
                with torch.no_grad():
                    p.fill_(value)


def _ckpt_size_bytes(ckpt_dir):
    return sum(f.stat().st_size for f in Path(ckpt_dir).rglob("*") if f.is_file())


class TestPeftAdapterOnlyCheckpointing:
    """Issue #7270: --save-trainable-params-only lets a PEFT/LoRA-style fine-tuning
    run (frozen base model, a few trainable adapter parameters) save a checkpoint
    proportional to the adapter instead of duplicating the frozen base weights, load
    it back onto a matching base model, and resume training from it. Exercised at
    TP/PP > 1 and with a SwiGLU-gated MLP (whose linear_fc1 is
    ShardedTensorFactory-wrapped) since both are real correctness hazards this
    feature has to handle: dropping a parameter that Megatron-Core represents as one
    tensor merged across all layers, and a factory-wrapped frozen parameter needing
    load-time request filtering (not just save-time filtering) to avoid crashing in
    apply_factory_merges.
    """

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("tp,pp", [(1, 1), (2, 1), (1, 2)])
    @pytest.mark.parametrize("use_glu", [False, True])
    def test_save_is_smaller_load_is_correct_and_resume_restores_optimizer_state(
        self, tmp_path_dist_ckpt, tp, pp, use_glu
    ):
        Utils.initialize_model_parallel(tp, pp)
        initialize_fn = partial(_initialize_peft_model, use_glu=use_glu)

        with (
            TempNamedDir(tmp_path_dist_ckpt / 'test_peft_ckpt_adapter', sync=True) as ckpt_dir_adapter,
            TempNamedDir(tmp_path_dist_ckpt / 'test_peft_ckpt_full', sync=True) as ckpt_dir_full,
        ):
            mock_args = parse_args(ignore_unknown_args=True)
            init_checkpointing_mock_args(mock_args, ckpt_dir_adapter)
            mock_args.save_tokenizer_assets = False

            with mock.patch('megatron.training.checkpointing.get_args', new=lambda: mock_args):
                # ---- Model A: frozen base + a trainable adapter on every layer ----
                model_a, optimizer_a = setup_model_and_optimizer(
                    seed=1, tp=tp, pp=pp, dist_opt=False, initialize_fn=initialize_fn
                )
                count = _freeze_all_but_adapter(model_a)
                assert count > 0, "no adapter parameters found on this rank"
                _set_adapter_params(model_a, 42.0)

                # Save adapter-only, and (for a size comparison) the full checkpoint
                # from the same model state.
                mock_args.save_trainable_params_only = True
                save_checkpoint(1, model_a, optimizer_a, None, 0)

                mock_args.save_trainable_params_only = False
                mock_args.save = ckpt_dir_full
                save_checkpoint(1, model_a, optimizer_a, None, 0)
                mock_args.save = ckpt_dir_adapter
                mock_args.save_trainable_params_only = True

                torch.distributed.barrier()
                if Utils.rank == 0:
                    adapter_size = _ckpt_size_bytes(ckpt_dir_adapter)
                    full_size = _ckpt_size_bytes(ckpt_dir_full)
                    assert adapter_size < full_size, (
                        f"adapter-only checkpoint ({adapter_size}B) is not smaller "
                        f"than the full checkpoint ({full_size}B)"
                    )

                optimizer_a_state = optimizer_a.state_dict()

                # ---- Model B: same seed (matching base model), but corrupt the
                #      adapter params and the optimizer state before loading, to prove
                #      the load actually restores them rather than trivially matching. ----
                Utils.destroy_model_parallel()
                Utils.initialize_model_parallel(tp, pp)
                model_b, optimizer_b = setup_model_and_optimizer(
                    seed=1, tp=tp, pp=pp, dist_opt=False, initialize_fn=initialize_fn
                )
                _freeze_all_but_adapter(model_b)
                _set_adapter_params(model_b, -1.0)
                # setup_model_and_optimizer seeds momentum state with the same seed
                # for both A and B, so corrupt B's here too -- otherwise a correct
                # post-load match would prove nothing about the load path.
                for group in optimizer_b.optimizer.param_groups:
                    for p in group['params']:
                        for key in ('exp_avg', 'exp_avg_sq'):
                            if key in optimizer_b.optimizer.state[p]:
                                optimizer_b.optimizer.state[p][key].fill_(-99.0)

                frozen_before = {
                    name: p.detach().clone()
                    for m in model_b
                    for name, p in m.named_parameters()
                    if not name.endswith(ADAPTER_PARAM_SUFFIX)
                }

                mock_args.dist_ckpt_strictness = "log_unexpected"
                with mock.patch('megatron.training.checkpointing.check_checkpoint_args'):
                    with mock.patch('megatron.training.checkpointing.update_num_microbatches'):
                        iteration, _ = load_checkpoint(model_b, optimizer_b, None, strict=False)

                assert iteration == 1

                for m in model_b:
                    for name, p in m.named_parameters():
                        if name.endswith(ADAPTER_PARAM_SUFFIX):
                            assert torch.allclose(p.float(), torch.full_like(p.float(), 42.0)), (
                                f"adapter param {name} not restored, got {p.flatten()[:4]}"
                            )
                        else:
                            assert torch.equal(p.detach(), frozen_before[name]), (
                                f"frozen param {name} was modified by the adapter-only load"
                            )

                # Resuming PEFT training must restore the adapter's optimizer state
                # (not just its weight and the iteration count).
                optimizer_b_state = optimizer_b.state_dict()
                diffs = diff(optimizer_a_state, optimizer_b_state)
                assert not any(map(bool, diffs)), (Utils.rank, diffs)
