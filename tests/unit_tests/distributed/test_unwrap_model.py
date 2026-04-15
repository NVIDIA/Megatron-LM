# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.distributed.fsdp.src.megatron_fsdp.megatron_fsdp import MegatronFSDP
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import unwrap_model
from tests.unit_tests.test_utilities import Utils

TRANSFORMER_CONFIG = TransformerConfig(
    num_layers=2,
    hidden_size=64,
    num_attention_heads=4,
    use_cpu_initialization=True,
    attention_backend="local",
)


def _build_gpt_model():
    """Build a small GPTModel for testing."""
    return GPTModel(
        config=TRANSFORMER_CONFIG,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=256,
        max_sequence_length=128,
        pre_process=True,
        post_process=True,
    ).cuda()


class TestUnwrapModel:
    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def test_unwrap_bare_model(self):
        """unwrap_model on an unwrapped model should return the model itself."""
        model = _build_gpt_model()
        assert unwrap_model(model) is model

    def test_unwrap_bare_model_list(self):
        """unwrap_model on a list of unwrapped models should return a list."""
        model = _build_gpt_model()
        result = unwrap_model([model])
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] is model

    def test_unwrap_ddp(self):
        """unwrap_model should peel through DDP to reach the underlying GPTModel."""
        model = _build_gpt_model()
        ddp_config = DistributedDataParallelConfig(bucket_size=10000)
        ddp_model = DistributedDataParallel(TRANSFORMER_CONFIG, ddp_config=ddp_config, module=model)

        assert isinstance(ddp_model, DistributedDataParallel)
        unwrapped = unwrap_model(ddp_model)
        assert unwrapped is model
        assert isinstance(unwrapped, GPTModel)

    def test_unwrap_ddp_list(self):
        """unwrap_model on a list with a DDP-wrapped model should unwrap each element."""
        model = _build_gpt_model()
        ddp_config = DistributedDataParallelConfig(bucket_size=10000)
        ddp_model = DistributedDataParallel(TRANSFORMER_CONFIG, ddp_config=ddp_config, module=model)

        result = unwrap_model([ddp_model])
        assert isinstance(result, list)
        assert result[0] is model

    def test_unwrap_megatron_fsdp(self):
        """unwrap_model should peel through FullyShardedDataParallel and MegatronFSDP."""
        model = _build_gpt_model()
        ddp_config = DistributedDataParallelConfig(
            data_parallel_sharding_strategy="optim_grads_params",
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            bucket_size=10000,
            use_megatron_fsdp=True,
        )
        fsdp_model = FullyShardedDataParallel(
            config=TRANSFORMER_CONFIG,
            ddp_config=ddp_config,
            module=model,
            fsdp_unit_modules=[TransformerLayer],
        )

        # Verify the wrapping hierarchy
        assert isinstance(fsdp_model, FullyShardedDataParallel)
        assert isinstance(fsdp_model.module, MegatronFSDP)

        unwrapped = unwrap_model(fsdp_model)
        assert unwrapped is model
        assert isinstance(unwrapped, GPTModel)

    def test_unwrap_megatron_fsdp_list(self):
        """unwrap_model on a list with an FSDP-wrapped model should unwrap each element."""
        model = _build_gpt_model()
        ddp_config = DistributedDataParallelConfig(
            data_parallel_sharding_strategy="optim_grads_params",
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            bucket_size=10000,
            use_megatron_fsdp=True,
        )
        fsdp_model = FullyShardedDataParallel(
            config=TRANSFORMER_CONFIG,
            ddp_config=ddp_config,
            module=model,
            fsdp_unit_modules=[TransformerLayer],
        )

        result = unwrap_model([fsdp_model])
        assert isinstance(result, list)
        assert result[0] is model

    def test_unwrap_with_custom_module_instances(self):
        """unwrap_model with custom module_instances should only peel specified types."""
        model = _build_gpt_model()
        ddp_config = DistributedDataParallelConfig(bucket_size=10000)
        ddp_model = DistributedDataParallel(TRANSFORMER_CONFIG, ddp_config=ddp_config, module=model)

        # Passing an empty tuple should not unwrap anything
        unwrapped = unwrap_model(ddp_model, module_instances=())
        assert unwrapped is ddp_model

        # Passing only DDP should unwrap through DDP
        unwrapped = unwrap_model(ddp_model, module_instances=(DistributedDataParallel,))
        assert unwrapped is model
