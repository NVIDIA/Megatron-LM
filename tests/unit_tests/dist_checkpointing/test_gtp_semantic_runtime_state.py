# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Strict checkpoint interoperability for semantic GTP weights and runtime state."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core import dist_checkpointing, parallel_state
from megatron.core.dist_checkpointing.core import CheckpointingException
from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
from megatron.core.dist_checkpointing.optimizer import (
    get_param_id_to_sharded_param_map,
    make_sharded_optimizer_tensor,
)
from megatron.core.optimizer.optimizer import _backfill_gtp_sharded_param_map
from megatron.core.ssm.utils import _split_in_proj_factory, _split_tensor_factory
from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
    GTP_CONFIG,
    reset_gtp_state,
    update_gtp_config,
    wrap_module_params_gtp,
)
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint
from megatron.training.checkpointing import _stage_ignored_runtime_state
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


class TestGTPSemanticRuntimeState:
    @classmethod
    def setup_class(cls):
        # All cases use the same GTP grid. Keep its communicators alive across
        # cases, as in the GTP DCP suite, instead of racing Gloo teardown against
        # the next case's group construction.
        Utils.initialize_distributed()
        if torch.distributed.get_world_size() != 4:
            pytest.skip("Requires four ranks")
        Utils.initialize_model_parallel(1, 1, gtp_remat_size=4)

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def setup_method(self):
        self.old_padding = GTP_CONFIG.pad_for_alignment

    def teardown_method(self):
        update_gtp_config(pad_for_alignment=self.old_padding)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        reset_gtp_state()

    @pytest.mark.parametrize("save_gtp", [False, True])
    @pytest.mark.parametrize("cpu_state", [False, True])
    def test_fused_muon_state_reshards_with_semantic_keys(
        self, tmp_path_dist_ckpt, save_gtp, cpu_state
    ):
        update_gtp_config(pad_for_alignment=16)
        rank = torch.distributed.get_rank()
        group = torch.distributed.group.WORLD
        tp = parallel_state.get_tensor_model_parallel_group()
        dp = parallel_state.get_data_parallel_group(with_context_parallel=True)
        key = "kda.in_proj.weight"
        full = torch.arange(54 * 8, dtype=torch.float32, device="cuda").reshape(54, 8) / 100
        module = torch.nn.Module()
        module.weight = torch.nn.Parameter(full.bfloat16())
        wrap_module_params_gtp(module, ["weight"], group)
        weight = module.weight
        weight._debug_name = "module.module." + key
        sharded = make_tp_sharded_tensor_for_checkpoint(
            weight, key, tp_axis=0, tp_group=tp, dp_cp_group=dp
        )
        factory = _split_in_proj_factory(
            sharded,
            [18, 18, 18],
            ["query", "key", "value"],
            weight=weight,
            tp_group=tp,
            dp_cp_group=dp,
        )
        model_sd = {key: factory}
        mapping = get_param_id_to_sharded_param_map(model_sd, [weight])
        _backfill_gtp_sharded_param_map(mapping, [[weight]], model_sd)
        assert mapping[0] is factory

        local_rows = weight.shape[0]

        def local_tensor(tensor):
            padded = F.pad(tensor, (0, 0, 0, local_rows * 4 - 54))
            return padded[rank * local_rows : (rank + 1) * local_rows].contiguous()

        device = "cpu" if cpu_state else "cuda"
        master = local_tensor(full).to(device)
        momentum = local_tensor(full * 3).to(device)
        gtp_sd = {
            "model": factory,
            "master": make_sharded_optimizer_tensor(factory, master, "optimizer.state.fp32_param"),
            "momentum": make_sharded_optimizer_tensor(
                factory, momentum, "optimizer.state.momentum_buffer"
            ),
        }
        plain_factory = _split_tensor_factory(
            ShardedTensor.from_rank_offsets(key, full.bfloat16(), replica_id=rank),
            [18, 18, 18],
            ["query", "key", "value"],
            0,
        )
        plain_sd = {
            "model": plain_factory,
            "master": make_sharded_optimizer_tensor(
                plain_factory, full.to(device), "optimizer.state.fp32_param"
            ),
            "momentum": make_sharded_optimizer_tensor(
                plain_factory, (full * 3).to(device), "optimizer.state.momentum_buffer"
            ),
        }
        with TempNamedDir(tmp_path_dist_ckpt / "semantic_muon", sync=True) as directory:
            dist_checkpointing.save(gtp_sd if save_gtp else plain_sd, directory)
            loaded = dist_checkpointing.load(
                plain_sd if save_gtp else gtp_sd, directory, strict="raise_all"
            )
        expected = full if save_gtp else local_tensor(full)
        torch.testing.assert_close(loaded["model"].cpu(), expected.bfloat16().cpu(), rtol=0, atol=0)
        torch.testing.assert_close(loaded["master"].cpu(), expected.cpu(), rtol=0, atol=0)
        torch.testing.assert_close(loaded["momentum"].cpu(), (expected * 3).cpu(), rtol=0, atol=0)

    @pytest.mark.parametrize("extra_model_key", [False, True])
    def test_ignored_runtime_objects_keep_model_validation_strict(
        self, tmp_path_dist_ckpt, extra_model_key
    ):
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        values = torch.arange(8, device="cuda")
        saved = {
            "model": ShardedTensor.from_rank_offsets("model.weight", values, replica_id=rank),
            "rng": ShardedObject("rng_state", {"seed": 123}, (1,), (0,), replica_id=rank),
            # Model six saved runtime ranks on the current four-rank test grid.
            "rerun": {
                i: ShardedObject("rerun_state_machine_state", {"old_rank": i}, (6,), (i,))
                for i in range(rank, 6, world_size)
            },
        }
        if extra_model_key:
            saved["extra"] = ShardedTensor.from_rank_offsets(
                "model.unexpected", values, replica_id=rank
            )
        target = {
            "model": ShardedTensor.from_rank_offsets(
                "model.weight", torch.zeros_like(values), replica_id=rank
            )
        }
        with TempNamedDir(tmp_path_dist_ckpt / "ignored_runtime", sync=True) as directory:
            dist_checkpointing.save(saved, directory)
            _stage_ignored_runtime_state(
                target, directory, {"rng_state", "rerun_state_machine_state"}
            )
            if extra_model_key:
                with pytest.raises(CheckpointingException, match="model.unexpected"):
                    dist_checkpointing.load(target, directory, strict="raise_all")
            else:
                loaded = dist_checkpointing.load(target, directory, strict="raise_all")
                torch.testing.assert_close(loaded["model"], values, rtol=0, atol=0)
                assert "rng_state" not in loaded and "rerun_state_machine" not in loaded
