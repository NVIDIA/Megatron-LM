# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree
from typing import Any, Dict, List, Optional
from unittest import mock

import pytest
import torch
import torch.distributed as dist

from megatron.training.arguments import parse_args

nvidia_resiliency_ext = pytest.importorskip(
    "nvidia_resiliency_ext",
    reason="nvidia_resiliency_ext is required for local checkpointing tests",
)

from nvidia_resiliency_ext.checkpointing.local.ckpt_managers.local_manager import (
    LocalCheckpointManager,
)
from nvidia_resiliency_ext.checkpointing.local.replication.group_utils import GroupWrapper
from nvidia_resiliency_ext.checkpointing.local.replication.strategies import (
    CliqueReplicationStrategy,
)

from megatron.training.async_utils import maybe_finalize_async_save
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from tests.unit_tests.dist_checkpointing import (
    TempNamedDir,
    init_basic_mock_args,
    init_checkpointing_mock_args,
    setup_model_and_optimizer,
)
from tests.unit_tests.test_utilities import Utils


def equal_(a, b):
    def bool_generator():
        if isinstance(a, list):
            yield isinstance(b, list)
            yield len(a) == len(b)
            yield all(equal_(aa, bb) for aa, bb in zip(a, b))
        elif isinstance(a, torch.Tensor):
            yield isinstance(b, torch.Tensor)
            yield torch.equal(a, b)
        else:
            yield a == b

    return all(bool_generator())


@pytest.mark.parametrize(('tp,pp'), [(2, 4), (1, 1)])
def test_all_gather_batch(tp, pp):
    Utils.initialize_model_parallel(tp, pp)
    torch.cuda.set_device(dist.get_rank())
    t0 = torch.arange(4, device="cuda").reshape((2, 2))
    t1 = torch.arange(6, device="cuda").reshape((3, 1, 2))
    t2 = torch.arange(12, device="cuda").reshape((2, 3, 2))
    test_ranks = [0, 3, 7]
    test_group = GroupWrapper(dist.new_group(test_ranks))
    rank = dist.get_rank()
    if rank not in test_ranks:
        dist.barrier()
        return
    batch = [[t1, t2], [t0], []]
    pred_batch = test_group.all_gather_batch(batch[test_group.my_group_rank])
    assert equal_(batch, pred_batch)
    dist.barrier()


# TODO: Use mock local checkpointing?
@pytest.mark.parametrize(('tp,pp'), [(2, 4), (1, 1)])
@pytest.mark.parametrize(('async_save'), [True, False])
@pytest.mark.parametrize(('algo'), ['atomic', 'fully_parallel'])
@pytest.mark.parametrize(
    ("repl_groups"), [[[0, 1], [2, 3], [4, 5], [6, 7]], [[2, 6, 7], [3, 1], [5], [0, 4]]]
)
class TestLocalCheckpointingReplication:
    # tp: int
    # pp: int
    # async_save: bool
    # algo: str
    # repl_groups: List[List[int]]
    # # To be filled by post_init
    # checkpointing_context: Optional[Dict[str, LocalCheckpointManager]]
    # repl_groups: Optional[List[dist.ProcessGroup]]
    # local_ckpt_dir: Optional[Path]

    @contextmanager
    def post_init(self, root_tmp_dir, tp, pp, async_save, algo, repl_groups):
        Utils.initialize_model_parallel(tp, pp)

        mock_args = parse_args(ignore_unknown_args=True)
        with mock.patch(
            'megatron.training.checkpointing.get_args', new=lambda: mock_args
        ), mock.patch('megatron.training.async_utils.get_args', new=lambda: mock_args), mock.patch(
            "megatron.training.checkpointing.update_num_microbatches"
        ):
            self.local_ckpt_dir = (
                root_tmp_dir / "subdir"
            )  # Test handling of non-existent directories
            init_basic_mock_args(mock_args, tp, pp)
            init_checkpointing_mock_args(mock_args, None)
            mock_args.non_persistent_ckpt_type = 'local'
            mock_args.non_persistent_local_ckpt_algo = algo
            mock_args.async_save = async_save
            repl_groups_init = [dist.new_group(g) for g in repl_groups]
            my_process_group = GroupWrapper.from_list_of_groups(repl_groups_init)
            repl_strategy = CliqueReplicationStrategy(my_process_group, target_device="cpu")
            self.checkpointing_context = {
                'local_checkpoint_manager': LocalCheckpointManager(
                    self.local_ckpt_dir, repl_strategy=repl_strategy
                )
            }
            self.local_ckpt_dir /= str(dist.get_rank())
            yield
        Utils.destroy_model_parallel()

    def test_repl_save_and_load(self, tmp_dir_per_class, tp, pp, async_save, algo, repl_groups):
        with self.post_init(tmp_dir_per_class, tp, pp, async_save, algo, repl_groups):
            num_floating_point_operations_so_far = 0
            model, optimizer = setup_model_and_optimizer(1, tp, pp)
            opt_param_scheduler = None

            save_checkpoint(
                1,
                model,
                optimizer,
                opt_param_scheduler,
                num_floating_point_operations_so_far,
                checkpointing_context=self.checkpointing_context,
                non_persistent_ckpt=True,
            )
            if async_save:
                maybe_finalize_async_save(True)

            my_group = [group for group in repl_groups if dist.get_rank() in group][0]
            assert {f"iter_0000001_{rank}_local.pt" for rank in my_group} == {
                f.name for f in self.local_ckpt_dir.rglob("*")
            }
        with self.post_init(tmp_dir_per_class, tp, pp, async_save, algo, repl_groups):

            ranks_to_break = [6, 3, 4]
            if dist.get_rank() in ranks_to_break:
                rmtree(self.local_ckpt_dir)
                os.makedirs(self.local_ckpt_dir)

            model, optimizer = setup_model_and_optimizer(2, tp, pp)
            opt_param_scheduler = None

            iteration, _ = load_checkpoint(
                model,
                optimizer,
                opt_param_scheduler,
                checkpointing_context=self.checkpointing_context,
            )
            assert iteration == 1
        # Perform cleanup to ensure no side effects on subsequent tests
        torch.distributed.barrier()
        rmtree(self.local_ckpt_dir)
