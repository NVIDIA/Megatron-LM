# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from types import SimpleNamespace

import pytest
import torch

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig, ParamKey, ParamPredicate
from megatron.core.process_groups_config import ProcessGroupCollection


def test_layer_sharded_muon_tp_mode_requirements():
    """muon_tp_mode='layer_sharded' needs muon, the layer-wise path and no split-QKV."""
    ok = dict(optimizer='muon', use_layer_wise_distributed_optimizer=True, muon_split_qkv=False)
    OptimizerConfig(muon_tp_mode='layer_sharded', **ok).validate()
    # dist_muon is the deprecated alias for muon + the layer-wise path.
    OptimizerConfig(
        muon_tp_mode='layer_sharded', optimizer='dist_muon', muon_split_qkv=False
    ).validate()
    OptimizerConfig(
        muon_tp_mode='layer_sharded',
        optimizer='muon',
        use_distributed_optimizer=True,
        muon_split_qkv=False,
    ).validate()
    with pytest.raises(ValueError, match="requires optimizer='muon'"):
        OptimizerConfig(
            muon_tp_mode='layer_sharded', **{**ok, 'optimizer': 'adaptive_muon'}
        ).validate()
    with pytest.raises(ValueError, match="layer-wise"):
        OptimizerConfig(
            muon_tp_mode='layer_sharded', **{**ok, 'use_layer_wise_distributed_optimizer': False}
        ).validate()
    with pytest.raises(ValueError, match="split-QKV"):
        OptimizerConfig(muon_tp_mode='layer_sharded', **{**ok, 'muon_split_qkv': True}).validate()


@pytest.mark.parametrize("optimizer", ["adam", "adaptive_muon"])
def test_factory_rejects_invalid_optimizer_config(optimizer):
    """Standalone Core callers get validation without a training container."""
    config = OptimizerConfig(optimizer=optimizer, lr=1e-3, muon_tp_mode="layer_sharded")
    with pytest.raises(ValueError, match="requires optimizer='muon'"):
        get_megatron_optimizer(config, [])


@pytest.mark.parametrize("layout", [False, True], ids=["compact", "aligned"])
@pytest.mark.parametrize(
    "settings",
    [
        {"optimizer": "muon", "use_distributed_optimizer": True},
        {"optimizer": "muon", "use_layer_wise_distributed_optimizer": True},
        {"optimizer": "dist_muon"},
        {"optimizer": "dist_adaptive_muon"},
    ],
)
def test_layer_wise_ddp_rejects_multiple_instances(settings, layout):
    """Validate explicit, implicit and legacy layer-wise requests before model setup."""
    config = OptimizerConfig(**settings, use_layer_wise_param_layout=layout)
    ddp = DistributedDataParallelConfig(num_distributed_optimizer_instances=2)
    with pytest.raises(ValueError, match=r"num_distributed_optimizer_instances=1 \(got 2\)"):
        config.validate_run(ddp_config=ddp, use_torch_fsdp2=False, ckpt_format="torch_dist")


@pytest.mark.parametrize(
    "optimizer, instances, mfsdp_v2", [("muon", 1, False), ("adam", 2, False), ("muon", 2, True)]
)
def test_ddp_instance_check_preserves_other_optimizer_modes(optimizer, instances, mfsdp_v2):
    """Only layer-wise DDP owns this instance-count constraint."""
    config = OptimizerConfig(optimizer=optimizer, use_distributed_optimizer=True)
    ddp = DistributedDataParallelConfig(
        num_distributed_optimizer_instances=instances,
        use_megatron_fsdp=mfsdp_v2,
        megatron_fsdp_version=2 if mfsdp_v2 else 1,
    )
    config.validate_run(ddp_config=ddp, use_torch_fsdp2=False, ckpt_format="torch_dist")


def test_direct_layer_wise_factory_checks_ddp_instances():
    """A caller supplying constructed model chunks receives the same actionable error."""
    config = OptimizerConfig(optimizer="dist_muon")
    chunk = SimpleNamespace(
        ddp_config=DistributedDataParallelConfig(num_distributed_optimizer_instances=2)
    )
    with pytest.raises(ValueError, match=r"num_distributed_optimizer_instances=1 \(got 2\)"):
        get_megatron_optimizer(
            config, [chunk], config_overrides={}, pg_collection=ProcessGroupCollection()
        )


def test_paramkey_matches():
    len_1_predicate = ParamPredicate(name="param_len_1", fn=lambda param: len(param.shape) == 1)
    endswith_bias = ParamKey(name="*.bias")
    has_dotbias = ParamKey(name="*.bias*")
    len_1_param = ParamKey(predicate=len_1_predicate)
    has_bias_or_len1_param = ParamKey(name="*.bias", predicate=len_1_predicate)
    has_attr = ParamKey(attr="is_embedding_or_output_parameter")

    assert endswith_bias.matches(torch.nn.Parameter(torch.empty(10, 10)), "interesting.bias")
    assert not endswith_bias.matches(
        torch.nn.Parameter(torch.empty(10, 10)), "something.bias.other"
    )
    assert has_dotbias.matches(torch.nn.Parameter(torch.empty(10)), "random.biasstuff")
    assert not has_dotbias.matches(torch.nn.Parameter(torch.empty(10, 10)), "random_bias_name")
    assert len_1_param.matches(torch.nn.Parameter(torch.empty(10)), "interesting.bias")
    assert not len_1_param.matches(torch.nn.Parameter(torch.empty(10, 10)), "interesting_bias")
    assert has_bias_or_len1_param.matches(
        torch.nn.Parameter(torch.empty(10, 10)), "interesting.bias"
    )
    assert has_bias_or_len1_param.matches(torch.nn.Parameter(torch.empty(10)), "interesting_bias")
    assert not has_bias_or_len1_param.matches(
        torch.nn.Parameter(torch.empty(10, 10)), "random_bias_name"
    )
    p_with_attr = torch.nn.Parameter(torch.empty(10, 10))
    setattr(p_with_attr, "is_embedding_or_output_parameter", True)
    assert has_attr.matches(p_with_attr, "interesting.bias")
    assert not has_attr.matches(torch.nn.Parameter(torch.empty(10, 10)), "interesting.bias")

    # We expect that if the return of the attribute is False, it should not match even if
    #  it has the attribute.
    setattr(p_with_attr, "is_embedding_or_output_parameter", False)
    assert not has_attr.matches(p_with_attr, "interesting.bias")
