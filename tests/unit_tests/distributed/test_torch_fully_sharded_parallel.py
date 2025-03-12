# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import pytest
import torch

from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.distributed.torch_fully_sharded_data_parallel import (
    TorchFullyShardedDataParallel,
)
from megatron.core.num_microbatches_calculator import (
    init_num_microbatches_calculator,
    unset_num_microbatches_calculator,
)
from megatron.core.tensor_parallel import ColumnParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import init_method_normal, is_torch_min_version
from tests.unit_tests.test_utilities import Utils


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
    init_num_microbatches_calculator(0, None, 1, 1, 1)
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
