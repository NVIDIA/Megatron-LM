# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import pytest
import torch

from megatron.core import parallel_state
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.distributed.torch_fully_sharded_data_parallel import (
    TorchFullyShardedDataParallel,
)
from megatron.core.num_microbatches_calculator import (
    init_num_microbatches_calculator,
    unset_num_microbatches_calculator,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import ColumnParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.mlp import apply_swiglu_sharded_factory
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import (
    init_method_normal,
    is_torch_min_version,
    make_tp_sharded_tensor_for_checkpoint,
)
from tests.unit_tests.test_utilities import Utils

try:
    from torch.distributed import DeviceMesh
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor.placement_types import Shard

    HAVE_DTENSOR = True
except ImportError:
    HAVE_DTENSOR = False

try:
    import einops

    HAVE_EINOPS = True
except ImportError:
    HAVE_EINOPS = False


class DummyModel(MegatronModule):
    """Setup a few modules to test the FSDP2 constructor."""

    _fsdp_modules = [torch.nn.Linear]

    def __init__(self, config: TransformerConfig):
        """Initialize a dummy model with a few modules."""
        super().__init__(config)
        self.linear = torch.nn.Linear(2, 2)
        self.column_parallel_linear = ColumnParallelLinear(
            input_size=2, output_size=2, config=config, init_method=init_method_normal(0.02)
        )
        self.conv = torch.nn.Conv2d(2, 2, 1)


@pytest.fixture
def init_model_parallel():
    """Init torch distributed."""
    Utils.initialize_model_parallel(1, 1)
    init_num_microbatches_calculator(
        rank=0, global_batch_size=1, micro_batch_size=1, data_parallel_size=1
    )
    model_parallel_cuda_manual_seed(123)
    yield  # Run the actual test.
    Utils.destroy_model_parallel()
    unset_num_microbatches_calculator()


def test_fsdp2_constructor(init_model_parallel):
    """Test the FSDP2 constructor."""
    if not is_torch_min_version("2.4.0"):
        pytest.skip("FSDP2 is not supported on this version of PyTorch.")

    # Create a dummy model and configs.
    config = TransformerConfig(num_layers=1, kv_channels=1, bf16=True)
    ddp_config = DistributedDataParallelConfig()
    model = DummyModel(config)
    model = Float16Module(config, model)
    ddp_config = DistributedDataParallelConfig()

    # Create the sharded model.
    fsdp_model = TorchFullyShardedDataParallel(config, ddp_config, model)

    def _is_fsdp_wrapped_module(instance):
        # FSDP adds a prefix to the class name.
        return instance.__class__.__name__.startswith("FSDP")

    assert isinstance(fsdp_model, TorchFullyShardedDataParallel)
    # We manually added Linear to the list of submodules to wrap.
    assert _is_fsdp_wrapped_module(fsdp_model.module.module.linear)
    # ColumnParallelLinear is in the default list of submodules to wrap.
    assert _is_fsdp_wrapped_module(fsdp_model.module.module.column_parallel_linear)
    # Conv2d is not in the list of submodules to wrap.
    assert not _is_fsdp_wrapped_module(fsdp_model.module.module.conv)


def test_fsdp2_constructor_with_process_group(init_model_parallel):
    """Test the FSDP2 constructor with explicit process group parameter."""
    if not is_torch_min_version("2.4.0"):
        pytest.skip("FSDP2 is not supported on this version of PyTorch.")

    # Create a dummy model and configs.
    config = TransformerConfig(num_layers=1, kv_channels=1, bf16=True)
    ddp_config = DistributedDataParallelConfig()
    model = DummyModel(config)
    model = Float16Module(config, model)

    # Create a custom process group (using the default world for testing)
    custom_process_group = parallel_state.get_data_parallel_group(with_context_parallel=True)

    # Create the sharded model with explicit process group
    fsdp_model = TorchFullyShardedDataParallel(
        config, ddp_config, model, process_group=custom_process_group
    )

    # Verify the process group was set correctly
    assert fsdp_model.process_group is custom_process_group

    # Check that module wrapping still works correctly
    def _is_fsdp_wrapped_module(instance):
        return instance.__class__.__name__.startswith("FSDP")

    assert isinstance(fsdp_model, TorchFullyShardedDataParallel)
    assert _is_fsdp_wrapped_module(fsdp_model.module.module.linear)
    assert _is_fsdp_wrapped_module(fsdp_model.module.module.column_parallel_linear)
    assert not _is_fsdp_wrapped_module(fsdp_model.module.module.conv)


@pytest.mark.skipif(not is_torch_min_version("2.4.0"), reason="FSDP2 requires PyTorch >= 2.4")
@pytest.mark.skipif(not HAVE_EINOPS, reason="einops is not available")
@pytest.mark.skipif(not HAVE_DTENSOR, reason="DTensor is not available")
def test_fsdp2_swiglu_sharded_tensor_factory():
    """
    Test construction of a TP2 DP{N} ShardedTensor for SwiGLU.
    """
    # Initialize distributed with TP2.
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    tp_group = pg_collection.tp
    tp_size = tp_group.size()
    tp_rank = tp_group.rank()
    dp_cp_group = pg_collection.dp_cp
    dp_cp_size = dp_cp_group.size()
    dp_cp_rank = dp_cp_group.rank()

    # Create FSDP2 DTensor with DP only. (Implicitly TP-sharded before FSDP2 init.)
    device_mesh = DeviceMesh.from_group(dp_cp_group, device_type="cuda", mesh_dim_names=["dp_cp"])
    toy_dtensor = DTensor.from_local(
        torch.randn(2, 8), device_mesh=device_mesh, placements=(Shard(dim=0),)
    )
    toy_dtensor.is_torch_fsdp2_param = True

    # Initialize TP-DP ShardedTensor.
    tp_dp_sh_ten = make_tp_sharded_tensor_for_checkpoint(
        toy_dtensor,
        "test_fsdp2_tp_swiglu_weight",
        # SwiGLU FC1 TP & FSDP2 Sharding Dim
        tp_axis=0,
        replica_id=None,
        prepend_offsets=(),
        tp_group=tp_group,
        dp_cp_group=dp_cp_group,
    )
    """
    Before TP2-DP4 Swizzle (TP Rank 1, DP Rank 1):
    (Pdb) tp_dp_sh_ten
    ShardedTensor(
        local_shape=(2, 8),
        global_shape=(16, 8),
        # Canonical Data Offset = 2 * (tp_rank * dp_cp_size + dp_cp_rank)
        # = 2 * (5) = 10
        global_offset=(10, 0),
        axis_fragmentations=(8, 1),
        replica_id=(0, 0, 0),
        prepend_axis_num=0
    )
    """

    # Test SwiGLU factory for FSDP2-TP.
    swiglu_sh_ten_factory = apply_swiglu_sharded_factory(
        tp_dp_sh_ten,
        sharded_offsets=(),  # Vanilla MLP.
        # Fused W/V.
        singleton_local_shards=False,
        tp_group=tp_group,
        dp_group=dp_cp_group,
    )
    sh_ten_shards = swiglu_sh_ten_factory.build()

    """
    After TP2-DP4 Swizzle (TP Rank 1, DP Rank 1):
    (Pdb) sh_ten_shards[0]
    ShardedTensor(
        local_shape=(2, 8),
        global_shape=(16, 8),
        global_offset=(6, 0),   # W/V TP-swizzled, DP-sharded!
        axis_fragmentations=(8, 1),
        replica_id=(0, 0, 0),
        prepend_axis_num=0
    )

    This is a mapping from the checkpoint [W;V] rank offsets:

        W_tp0_dp0 W_tp0_dp1 W_tp1_dp0 W_tp1_dp1 V_tp0_dp2 V_tp0_dp3 V_tp1_dp2 V_tp1_dp3
        0         1         2         3         4         5         6         7
                                      |
                        Data Offset = 3 * 2 = 6 is just the rank offset x local shape.
    
    to the model [ {W_tpx; V_tpx}_dpy ] rank offsets:

        W_tp0_dp0 W_tp0_dp1 V_tp0_dp2 V_tp0_dp3 W_tp1_dp0 W_tp1_dp1 V_tp1_dp2 V_tp1_dp3
    """

    # Validate FSDP2 TP-DP sharding and swizzle.
    assert getattr(tp_dp_sh_ten, "is_torch_fsdp2_param", False)
    assert len(sh_ten_shards) == 1
    shard = sh_ten_shards[0]
    toy_tensor_shape = toy_dtensor.to_local().shape
    assert shard.axis_fragmentations[0] == tp_size * dp_cp_size
    assert shard.global_shape[0] == tp_size * dp_cp_size * toy_tensor_shape[0]
    # Expected global data offsets considering the parallelism ranks and tensor shape.
    expected_global_rank_offsets = {
        # (TP Rank, DP Rank) -> Global Data Rank Location / Offset
        (0, 0): 0,
        (0, 1): 1,
        (0, 2): 4,
        (0, 3): 5,
        (1, 0): 2,
        (1, 1): 3,
        (1, 2): 6,
        (1, 3): 7,
    }
    assert (
        shard.global_offset[0]
        == expected_global_rank_offsets[(tp_rank, dp_cp_rank)] * toy_tensor_shape[0]
    )

    # Destroy distributed.
    Utils.destroy_model_parallel()
    unset_num_microbatches_calculator()
