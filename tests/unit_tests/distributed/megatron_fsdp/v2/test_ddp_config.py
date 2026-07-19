from megatron.core.distributed import DistributedDataParallelConfig


def test_use_megatron_fsdp_v2_implies_megatron_fsdp():
    ddp_config = DistributedDataParallelConfig(use_megatron_fsdp_v2=True)

    assert ddp_config.use_megatron_fsdp
