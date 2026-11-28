# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import pytest
import torch

from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig, ParamKey, ParamPredicate


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


def test_optimizer_cuda_graph_rejects_cpu_offloaded_state():
    """The optimizer step graph replays fixed device addresses; offloaded state cannot be
    captured."""
    kwargs = dict(
        optimizer='adam',
        bf16=True,
        use_distributed_optimizer=True,
        use_precision_aware_optimizer=True,
        optimizer_cpu_offload=True,
    )
    OptimizerConfig(**kwargs).validate()
    with pytest.raises(AssertionError, match="stay resident on the GPU"):
        OptimizerConfig(optimizer_cuda_graph=True, **kwargs).validate()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("warmup_steps", [0, 1, 2])
def test_optimizer_step_graph_keeps_the_optimizer_state_across_replays(monkeypatch, warmup_steps):
    """An optimizer allocates and zero-initializes its state on its first step. A graph that
    captured that step would run the initialization again at every replay, restarting the
    moments and the step count, so the first step runs eagerly whatever the warm-up count."""
    from megatron.core import full_cuda_graph
    from megatron.core.optimizer.optimizer_cuda_graph import OptimizerCudaGraphWrapper

    for name, value in (("curr_iteration", 0), ("cuda_graph", None), ("result", None)):
        monkeypatch.setattr(OptimizerCudaGraphWrapper, name, value)
    monkeypatch.setattr(full_cuda_graph, "_shared_capture_stream", None)
    monkeypatch.setattr(torch.distributed, "barrier", lambda *args, **kwargs: None)
    torch.manual_seed(0)
    initial = torch.randn(1024, device="cuda")
    grads = [torch.randn(1024, device="cuda") for _ in range(5)]

    def train(graphed):
        param = torch.nn.Parameter(initial.clone())
        param.grad = torch.zeros_like(param)
        optimizer = torch.optim.Adam([param], lr=1e-2, capturable=True)
        step = optimizer.step
        if graphed:
            step = OptimizerCudaGraphWrapper(step, cuda_graph_warmup_steps=warmup_steps)
        for grad in grads:
            param.grad.copy_(grad)
            step()
        torch.cuda.synchronize()
        assert (OptimizerCudaGraphWrapper.cuda_graph is not None) is graphed
        return param.detach().clone(), optimizer.state[param]["step"].item()

    eager_param, eager_steps = train(graphed=False)
    graphed_param, graphed_steps = train(graphed=True)

    assert graphed_steps == eager_steps == len(grads)
    torch.testing.assert_close(graphed_param, eager_param)
