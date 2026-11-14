# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""A captured optimizer step follows the lr schedule.

``OptimizerCudaGraphWrapper`` captures ``optimizer.step()`` once and replays it. Two identical
models get the same gradients every step; one steps eagerly, the other through the wrapper,
under an lr warmup + cosine decay. Their params must stay together after the capture:
capturable Muon's tensor-scaled update matches the eager ``add_(alpha=)`` only up to fp32
rounding, a frozen lr is off by far more from the first replay on. What a captured step cannot
follow is rejected before training: optimizers reading lr as a float, and a changing weight
decay.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.optimizer_cuda_graph import (
    OptimizerCudaGraphWrapper,
    check_weight_decay_constant_for_optimizer_cuda_graph,
)
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

try:
    import emerging_optimizers  # noqa: F401  # pylint: disable=unused-import

    HAVE_EMERGING_OPTIMIZERS = True
except ImportError:
    HAVE_EMERGING_OPTIMIZERS = False

_STEPS = 10
_CAPTURE_STEP = 2


class _Model(nn.Module):
    """Matrices for Muon and biases for its scalar Adam."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(80, 48)
        self.fc2 = nn.Linear(48, 48)
        self.fc3 = nn.Linear(48, 10)


@pytest.fixture
def _model_parallel():
    Utils.initialize_model_parallel()
    yield
    Utils.destroy_model_parallel()
    OptimizerCudaGraphWrapper.cuda_graph = None
    OptimizerCudaGraphWrapper.result = None
    OptimizerCudaGraphWrapper.curr_iteration = 0


def _build(optimizer_kwargs, graph, copy_from=None):
    from megatron.training.training import wrap_model_chunks_with_ddp

    muon = optimizer_kwargs["optimizer"] == "muon"
    torch.manual_seed(1)
    model = _Model().bfloat16().cuda()
    model = wrap_model_chunks_with_ddp(
        [model],
        TransformerConfig(num_attention_heads=1, num_layers=1),
        DistributedDataParallelConfig(),
        use_layer_wise_distributed_optimizer=muon,
    )[0]
    if copy_from is not None:
        model.module.load_state_dict(copy_from.module.state_dict())
    config = OptimizerConfig(
        **optimizer_kwargs,
        bf16=True,
        lr=1e-2,
        min_lr=1e-3,
        weight_decay=0.1,
        clip_grad=1.0,
        use_distributed_optimizer=False,
        use_layer_wise_distributed_optimizer=muon,
        optimizer_cuda_graph=graph,
    )
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    pg_collection.dp_cp = parallel_state.get_data_parallel_group(with_context_parallel=True)
    pg_collection.expt_dp = parallel_state.get_expert_data_parallel_group()
    optimizer = get_megatron_optimizer(
        config, [model], pg_collection=pg_collection, use_gloo_process_groups=False
    )
    scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=0.0,
        max_lr=1e-2,
        min_lr=1e-3,
        lr_warmup_steps=4,
        lr_decay_steps=_STEPS,
        lr_decay_style="cosine",
        start_wd=0.1,
        end_wd=0.1,
        wd_incr_steps=_STEPS,
        wd_incr_style="constant",
    )
    return model, optimizer, scheduler


_MUON = dict(optimizer="muon", muon_split_qkv=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "optimizer_kwargs",
    [
        pytest.param(dict(optimizer="adam"), id="adam"),
        pytest.param(
            dict(**_MUON, muon_tp_mode="duplicated"),
            id="muon_duplicated",
            marks=pytest.mark.skipif(not HAVE_EMERGING_OPTIMIZERS, reason="needs Muon"),
        ),
        pytest.param(
            dict(**_MUON, muon_tp_mode="layer_sharded"),
            id="muon_layer_sharded",
            marks=pytest.mark.skipif(not HAVE_EMERGING_OPTIMIZERS, reason="needs Muon"),
        ),
    ],
)
def test_captured_step_follows_lr_schedule(_model_parallel, optimizer_kwargs):
    reference_model, reference, reference_scheduler = _build(optimizer_kwargs, graph=False)
    model, optimizer, scheduler = _build(optimizer_kwargs, graph=True, copy_from=reference_model)
    step = OptimizerCudaGraphWrapper(optimizer.step, cuda_graph_warmup_steps=_CAPTURE_STEP)

    params = dict(model.module.named_parameters())
    reference_params = dict(reference_model.module.named_parameters())
    generator = torch.Generator().manual_seed(2)
    for iteration in range(_STEPS):
        for name, param in params.items():
            grad = torch.randn(param.shape, generator=generator).cuda()
            # In place: the captured graph reads the grad buffers by address.
            param.main_grad.copy_(grad)
            reference_params[name].main_grad.copy_(grad)
        reference.step()
        step()
        reference_scheduler.step(1)
        scheduler.step(1)
        for name, param in params.items():
            # A few bf16 ulps of these O(0.1) weights; the lr frozen at capture time is off by
            # 8e-3 at the first replay and 8e-2 a few steps later.
            torch.testing.assert_close(
                param,
                reference_params[name],
                atol=2e-3,
                rtol=0,
                msg=lambda msg, it=iteration, n=name: f"step {it} {n}\n\n{msg}",
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(not HAVE_EMERGING_OPTIMIZERS, reason="needs Muon")
def test_capturable_muon_keeps_tensor_lr_across_load(_model_parallel):
    from megatron.core.optimizer.emerging_optimizers import TensorParallelMuon

    params = [nn.Parameter(torch.randn(16, 8, device="cuda"))]
    optimizer = TensorParallelMuon(params, lr=1e-2, weight_decay=0.1, capturable=True)
    lr = optimizer.param_groups[0]["lr"]
    assert isinstance(lr, torch.Tensor) and lr.is_cuda

    state_dict = optimizer.state_dict()
    state_dict["param_groups"][0]["lr"] = 3e-3  # e.g. a checkpoint written without the graph
    optimizer.load_state_dict(state_dict)
    lr = optimizer.param_groups[0]["lr"]
    assert isinstance(lr, torch.Tensor) and lr.is_cuda
    assert lr.item() == pytest.approx(3e-3)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(optimizer="sgd"), "supports optimizer='adam' or 'muon'"),
        (dict(optimizer="adaptive_muon"), "supports optimizer='adam' or 'muon'"),
        (dict(optimizer="muon", muon_scalar_optimizer="lion"), "muon_scalar_optimizer='adam'"),
    ],
)
def test_optimizer_cuda_graph_rejects_float_lr_optimizers(kwargs, message):
    with pytest.raises(ValueError, match=message):
        OptimizerConfig(**kwargs, optimizer_cuda_graph=True).validate()


@pytest.mark.parametrize(
    "style, start_wd, end_wd, group, ok",
    [
        ("constant", 0.1, 0.1, {}, True),
        ("linear", 0.1, 0.1, {}, True),
        ("linear", 0.0, 0.1, {}, False),
        ("cosine", 0.1, 0.1, {"start_wd": 0.0}, False),  # a per-group override
        ("linear", 0.1, 0.1, {"start_wd": 0.05, "end_wd": 0.05}, True),
    ],
)
def test_optimizer_cuda_graph_requires_constant_weight_decay(style, start_wd, end_wd, group, ok):
    scheduler = SimpleNamespace(wd_incr_style=style, start_wd=start_wd, end_wd=end_wd)
    optimizer = SimpleNamespace(param_groups=[{}, group])
    if ok:
        check_weight_decay_constant_for_optimizer_cuda_graph(optimizer, scheduler)
    else:
        with pytest.raises(ValueError, match="requires a constant weight decay"):
            check_weight_decay_constant_for_optimizer_cuda_graph(optimizer, scheduler)
