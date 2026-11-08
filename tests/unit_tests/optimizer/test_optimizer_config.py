# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import pytest
import torch

from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.optimizer.optimizer_config import (
    OptimizerConfig,
    ParamKey,
    ParamPredicate,
    muon_modes_are_hybrid,
)


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


def test_hybrid_muon_expert_tp_mode_requirements():
    """muon_expert_tp_mode is a muon-only knob; layer_sharded on the expert side needs the
    layer-wise path too; the split-QKV restriction is keyed to the dense side, which alone
    owns QKV weights."""
    ok = dict(optimizer='muon', use_layer_wise_distributed_optimizer=True)
    hybrid = dict(muon_tp_mode='auto', muon_expert_tp_mode='layer_sharded')
    # Dense auto + expert layer_sharded keeps dense split-QKV available.
    OptimizerConfig(**hybrid, muon_split_qkv=True, **ok).validate()
    with pytest.raises(ValueError, match="layer-wise"):
        OptimizerConfig(
            **hybrid, **{**ok, 'use_layer_wise_distributed_optimizer': False}
        ).validate()
    # Reverse hybrid: the dense side is layer_sharded, so split-QKV must be off.
    reverse = dict(muon_tp_mode='layer_sharded', muon_expert_tp_mode='duplicated')
    OptimizerConfig(**reverse, muon_split_qkv=False, **ok).validate()
    with pytest.raises(ValueError, match="split-QKV"):
        OptimizerConfig(**reverse, muon_split_qkv=True, **ok).validate()
    # 'auto' is a dense-only cost model; on expert weights it would run duplicated.
    with pytest.raises(ValueError, match="muon_expert_tp_mode='auto' is not supported"):
        OptimizerConfig(
            optimizer='muon', muon_tp_mode='duplicated', muon_expert_tp_mode='auto'
        ).validate()
    # An explicit expert mode equal to the dense one changes nothing.
    OptimizerConfig(
        optimizer='muon', muon_tp_mode='duplicated', muon_expert_tp_mode='duplicated'
    ).validate()
    # dist_muon is the deprecated alias for muon + the layer-wise path.
    OptimizerConfig(
        optimizer='dist_muon', muon_tp_mode='auto', muon_expert_tp_mode='duplicated'
    ).validate()
    # Every other optimizer applies muon_tp_mode to all weights and would silently ignore the
    # override, so an explicit expert mode is rejected for them, layer_sharded or not.
    for other in ('adaptive_muon', 'adam'):
        with pytest.raises(ValueError, match="only implemented for optimizer='muon'"):
            OptimizerConfig(
                optimizer=other, muon_tp_mode='auto', muon_expert_tp_mode='duplicated'
            ).validate()
        with pytest.raises(ValueError, match="only implemented for optimizer='muon'"):
            OptimizerConfig(optimizer=other, **hybrid).validate()


@pytest.mark.parametrize(
    "tp_mode, expert_tp_mode, hybrid",
    [
        ('auto', None, False),
        ('duplicated', 'duplicated', False),
        # 'auto' already runs every expert weight 'duplicated' (its cost model covers dense
        # GTP weights only), so the explicit pair builds no separate expert optimizer.
        ('auto', 'duplicated', False),
        ('auto', 'distributed', True),
        ('auto', 'layer_sharded', True),
        ('layer_sharded', 'duplicated', True),
    ],
)
def test_muon_modes_are_hybrid(tp_mode, expert_tp_mode, hybrid):
    """The expert optimizer splits off only when the expert weights would run a different
    NS mode than the dense optimizer already gives them."""
    config = OptimizerConfig(
        optimizer='muon',
        use_layer_wise_distributed_optimizer=True,
        muon_tp_mode=tp_mode,
        muon_expert_tp_mode=expert_tp_mode,
        muon_split_qkv=False,
    )
    config.validate()
    assert muon_modes_are_hybrid(config) is hybrid


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
