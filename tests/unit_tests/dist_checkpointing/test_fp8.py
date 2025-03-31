# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.device_utils import get_current_device
from megatron.core.dist_checkpointing import ShardedTensor, load, save
from megatron.core.dist_checkpointing.serialization import (
    get_default_load_sharded_strategy,
    get_default_save_sharded_strategy,
)
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper,
)
from megatron.core.parallel_state import get_default_process_group
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils

try:
    import transformer_engine  # pylint: disable=W0611

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    # Transformer Engine not found
    HAVE_TE = False

HAVE_TE_FLOAT8TENSOR = False
try:
    from transformer_engine.pytorch.float8_tensor import Float8Tensor

    HAVE_TE_FLOAT8TENSOR = True
except (ImportError, ModuleNotFoundError):
    # Float8Tensor not found
    Float8Tensor = None

@pytest.mark.skipif(not HAVE_TE, reason="Transormer Engine is required for fp8")
class TestFP8:
    @pytest.mark.parametrize('dtype', ['bf16', 'fp16', 'fp8'])
    @pytest.mark.parametrize('src_rank', [0, 6])
    def test_simple_broadcast(self, dtype, src_rank):
        Utils.initialize_model_parallel()

        def get_ten(dtype: str = 'fp8'):
            if dtype == 'fp8':
                return Float8Tensor.to_float8(
                    torch.full((3,), Utils.rank, dtype=torch.bfloat16, device='cuda')
                )
            elif dtype == 'bf16':
                return torch.full((3,), Utils.rank, dtype=torch.bfloat16, device=get_current_device())
            elif dtype == 'fp16':
                return torch.full((3,), Utils.rank, dtype=torch.float16, device=get_current_device())
            else:
                raise NotImplementedError(dtype)

        ten = get_ten(dtype)

        # because of a bug in TE, with the cast broadcast fails
        if isinstance(ten, Float8Tensor):
            ten = ten.dequantize()
        torch.distributed.broadcast(ten, src=src_rank)
        assert torch.all(ten == src_rank)

    @pytest.mark.parametrize(
        ('use_fpsl', 'src_tp_pp', 'dest_tp_pp', 'load_exchange_algo'),
        [
            (True, (2, 4), (2, 4), 'broadcast'),
            (True, (2, 4), (2, 4), 'gather_rounds'),
            (False, (2, 4), (2, 4), None),
        ],
    )
    def test_fp8_save_load(
        self, tmp_path_dist_ckpt, use_fpsl, src_tp_pp, dest_tp_pp, load_exchange_algo
    ):
        Utils.initialize_model_parallel(*src_tp_pp)

        def get_fp8_tensor(fill_val=1):
            return Float8Tensor.to_float8(
                torch.full((3,), fill_val, dtype=torch.bfloat16, device='cuda')
            )

        def get_state_dict(fill_val=1):
            return {
                'a': ShardedTensor.from_rank_offsets(
                    'a', get_fp8_tensor(fill_val), (0, Utils.rank, Utils.world_size), replica_id=0
                ),
                'b': ShardedTensor.from_rank_offsets(
                    'b', get_fp8_tensor(fill_val), replica_id=Utils.rank
                ),
                'c': ShardedTensor.from_rank_offsets(
                    'c', get_fp8_tensor(fill_val), replica_id=Utils.rank
                ),
            }

        with TempNamedDir(tmp_path_dist_ckpt / 'test_fp8_save_load', sync=True) as ckpt_dir:
            save_strategy = get_default_save_sharded_strategy()
            if use_fpsl:
                save_strategy = FullyParallelSaveStrategyWrapper(save_strategy, None, 
                                                                 get_default_process_group(),
                                                                 True)
            save(get_state_dict(4), ckpt_dir, save_strategy)

            Utils.destroy_model_parallel()
            Utils.initialize_model_parallel(*dest_tp_pp)

            if use_fpsl:
                load_strategy = get_default_load_sharded_strategy(ckpt_dir)
                load_strategy = FullyParallelLoadStrategyWrapper(
                    load_strategy, None, False, load_exchange_algo
                )
            else:
                load_strategy = None

            loaded_state_dict = load(get_state_dict(8), ckpt_dir, load_strategy)
            assert torch.all(loaded_state_dict['a'] == 4)
            assert torch.all(loaded_state_dict['b'] == 4)
        Utils.destroy_model_parallel()
