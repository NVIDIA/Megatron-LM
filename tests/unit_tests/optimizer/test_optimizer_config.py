# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import pytest
import torch

from megatron.core.optimizer.optimizer_config import OptimizerConfig, ParamKey, ParamPredicate


def test_layer_sharded_muon_tp_mode_requirements():
    """muon_tp_mode='layer_sharded' needs muon, the layer-wise path and no split-QKV."""
    ok = dict(optimizer='muon', use_layer_wise_distributed_optimizer=True, muon_split_qkv=False)
    OptimizerConfig(muon_tp_mode='layer_sharded', **ok)
    # dist_muon is the deprecated alias for muon + the layer-wise path.
    OptimizerConfig(muon_tp_mode='layer_sharded', optimizer='dist_muon', muon_split_qkv=False)
    with pytest.raises(ValueError, match="requires optimizer='muon'"):
        OptimizerConfig(muon_tp_mode='layer_sharded', **{**ok, 'optimizer': 'adaptive_muon'})
    with pytest.raises(ValueError, match="layer-wise"):
        OptimizerConfig(
            muon_tp_mode='layer_sharded', **{**ok, 'use_layer_wise_distributed_optimizer': False}
        )
    with pytest.raises(ValueError, match="split-QKV"):
        OptimizerConfig(muon_tp_mode='layer_sharded', **{**ok, 'muon_split_qkv': True})


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


def test_overlap_optimizer_step_with_forward_requirements():
    """The overlapped step needs the distributed optimizer, Adam and no loss scaling."""
    ok = dict(
        optimizer='adam', bf16=True, use_distributed_optimizer=True, overlap_param_gather=True
    )
    OptimizerConfig(overlap_optimizer_step_with_forward=True, **ok)
    for bad in (
        dict(use_distributed_optimizer=False),
        dict(optimizer='sgd'),
        dict(fp16=True, bf16=False),
        dict(loss_scale=1024.0),
        dict(use_precision_aware_optimizer=True),
        dict(optimizer_cpu_offload=True),
        dict(optimizer_cuda_graph=True),
        dict(overlap_param_gather_with_optimizer_step=True),
        dict(log_num_zeros_in_grad=True),
        dict(grad_norm_skip_threshold=10.0),
        dict(overlap_optimizer_step_prefetch=-1),
    ):
        with pytest.raises(ValueError, match="overlap_optimizer_step_with_forward requires"):
            OptimizerConfig(overlap_optimizer_step_with_forward=True, **{**ok, **bad})
    # Off by default: none of the above is checked.
    OptimizerConfig(**{**ok, 'optimizer_cuda_graph': True})


def test_overlap_optimizer_step_flat_adam_requires_overlapped_step():
    ok = dict(
        optimizer='adam', bf16=True, use_distributed_optimizer=True, overlap_param_gather=True
    )
    OptimizerConfig(
        overlap_optimizer_step_with_forward=True, overlap_optimizer_step_flat_adam=True, **ok
    )
    with pytest.raises(ValueError, match="overlap_optimizer_step_flat_adam requires"):
        OptimizerConfig(overlap_optimizer_step_flat_adam=True, **ok)
