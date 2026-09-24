# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real Engram forward/backward parity, including lookup and parameter gradients."""

import importlib.metadata
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from packaging.version import Version

from megatron.core.fusions.cudnn_engram import CudnnEngramGate
from megatron.core.models.deepseek_v41.config import DeepSeekV41Config
from megatron.core.models.deepseek_v41.engram import Engram
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.test_utilities import Utils


def require_cudnn_engram():
    """Skip optional dispatch coverage when CI lacks the validated FE dependencies."""
    for package, minimum in (("nvidia-cutlass-dsl", "4.7.0"), ("triton", "3.7.0")):
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pytest.skip(f"cuDNN Engram requires {package}>={minimum}")
        if Version(installed) < Version(minimum):
            pytest.skip(f"cuDNN Engram requires {package}>={minimum}; found {installed}")
    cudnn = pytest.importorskip("cudnn")
    if not all(
        hasattr(cudnn, name) for name in ("EngramGateSavedForward", "EngramGateSavedBackward")
    ):
        pytest.skip("cuDNN Frontend build lacks saved-state Engram APIs")


def test_cudnn_gate_reports_missing_frontend_api(monkeypatch):
    """The default older FE pin must produce an actionable opt-in error."""
    monkeypatch.setitem(sys.modules, "cudnn", ModuleType("cudnn"))
    x = SimpleNamespace(device=SimpleNamespace(index=0), shape=(64, 4, 5120))
    with pytest.raises(RuntimeError, match="saved-state Engram APIs"):
        CudnnEngramGate(1e-20).plans(x, None, None, None, None)


def config(backend="native", **changes):
    """Configure the real gate geometry without allocating a full backbone."""
    values = dict(
        num_layers=1,
        hidden_size=5120,
        num_attention_heads=40,
        params_dtype=torch.bfloat16,
        bf16=True,
        use_cpu_initialization=True,
        mhc_num_residual_streams=4,
        engram_gate_backend=backend,
        layernorm_epsilon=1e-20,
    )
    values.update(changes)
    return DeepSeekV41Config(**values)


def make_module(groups, backend, *, eps=1e-20, source_width=False):
    """Keep real module/lookup execution; unit tests use a smaller embedding width."""
    layout = SimpleNamespace(
        layer_ids=(0,),
        num_embeddings=(1024,),
        max_ngram_size=4 if source_width else 2,
        n_heads=8 if source_width else 2,
        head_dim=256 if source_width else 8,
    )
    return Engram(config(backend, layernorm_epsilon=eps), layout, 0, groups).cuda()


def close(actual, expected, limit=0.03):
    """Check normwise and largest-error gates independently."""
    a, b = actual.detach().double(), expected.detach().double()
    assert torch.isfinite(a).all()
    delta = a - b
    relative = delta.norm() / b.norm().clamp_min(1e-30)
    scaled = delta.abs().max() / b.abs().max().clamp_min(1e-30)
    assert relative <= limit, float(relative)
    assert scaled <= 2 * limit, float(scaled)


@pytest.fixture
def groups():
    Utils.initialize_model_parallel()
    model_parallel_cuda_manual_seed(4176)
    yield ProcessGroupCollection.use_mpu_process_groups()
    Utils.destroy_model_parallel()


def test_backend_config_defaults_and_rejects_invalid_contract():
    assert config().engram_gate_backend == "native"
    assert config("cudnn").engram_gate_backend == "cudnn"
    for changes in (
        dict(engram_gate_backend="frost"),
        dict(hidden_size=256, num_attention_heads=4),
        dict(mhc_num_residual_streams=2),
        dict(params_dtype=torch.float32, bf16=False),
    ):
        with pytest.raises(ValueError, match="cuDNN|engram_gate_backend"):
            config(**(dict(engram_gate_backend="cudnn") | changes))


@pytest.mark.skipif(
    torch.cuda.get_device_capability() != (10, 0), reason="cuDNN gate requires SM100"
)
@pytest.mark.parametrize("batch,eps", [(1, 1e-20), (2, 1e-6)])
def test_actual_engram_two_pending_microbatches(groups, batch, eps):
    require_cudnn_engram()
    native = make_module(groups, "native", eps=eps)
    actual = make_module(groups, "cudnn", eps=eps)
    with torch.no_grad():
        native.q_weight.uniform_(0.6, 1.4)
        native.k_weight.uniform_(0.6, 1.4)
    actual.load_state_dict(native.state_dict())
    length = 64
    data = []
    for index in range(2):
        x = torch.randn(length, batch, 20480, device="cuda", dtype=torch.bfloat16)
        ids = torch.randint(0, 64, (batch, length, 1, 2), device="cuda")
        mask = torch.rand(batch, length, device="cuda") > 0.2
        if index == 1:
            mask.zero_()
        data.append((x, ids, mask, torch.randn_like(x)))
    outputs, input_grads = [], []
    for module in (native, actual):
        xs = [d[0].clone().requires_grad_() for d in data]
        ys = [module(x, d[1], d[2]) for x, d in zip(xs, data)]
        saved_first = ys[0].detach().clone()
        # Reverse completion order and retain one graph for a second backward.
        ys[1].backward(data[1][3])
        ys[0].backward(data[0][3], retain_graph=True)
        ys[0].backward(data[0][3])
        assert torch.equal(saved_first, ys[0])
        assert torch.equal(ys[1], xs[1])
        assert torch.equal(xs[1].grad, data[1][3])
        outputs.append(ys)
        input_grads.append([x.grad for x in xs])
    for ref, got in zip(outputs[0], outputs[1]):
        close(got, ref, 0.01)
    for ref, got in zip(input_grads[0], input_grads[1]):
        close(got, ref)
    for name, parameter in native.named_parameters():
        candidate = actual.get_parameter(name)
        assert parameter.grad is not None and candidate.grad is not None, name
        close(candidate.grad, parameter.grad)
    assert actual._cudnn_gate.forward_calls == 2
    assert actual._cudnn_gate.plan_builds == 1
    assert native.state_dict().keys() == actual.state_dict().keys()


@pytest.mark.skipif(
    torch.cuda.get_device_capability() != (10, 0), reason="cuDNN gate requires SM100"
)
def test_cudnn_gate_rejects_unsupported_token_count():
    gate = CudnnEngramGate(1e-20)
    x = torch.empty(65, 1, 20480, device="cuda", dtype=torch.bfloat16)
    kv = torch.empty(65, 1, 25600, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(4, 5120, device="cuda")
    with pytest.raises(ValueError, match="multiples of 64"):
        gate(x, kv, w, w)


@pytest.mark.skipif(
    torch.cuda.get_device_capability() != (10, 0), reason="cuDNN gate requires SM100"
)
def test_actual_engram_ddp_main_gradient_accumulation(groups):
    """Both providers run through MCore DDP's actual FP32 accumulation hooks."""
    require_cudnn_engram()
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig

    modules = [make_module(groups, backend) for backend in ("native", "cudnn")]
    modules[1].load_state_dict(modules[0].state_dict())
    inputs = [torch.randn(64, 1, 20480, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    hashes = [torch.randint(0, 128, (1, 64, 1, 2), device="cuda") for _ in range(2)]
    mask = torch.ones(1, 64, device="cuda", dtype=torch.bool)
    upstream = torch.randn_like(inputs[0])
    results = []
    for module in modules:
        ddp = DistributedDataParallel(
            module.config,
            DistributedDataParallelConfig(
                overlap_grad_reduce=False, grad_reduce_in_fp32=True, use_distributed_optimizer=False
            ),
            module,
            pg_collection=groups,
        )
        ddp.zero_grad_buffer()
        ys, dxs = [], []
        for x, ids in zip(inputs, hashes):
            value = x.clone().requires_grad_()
            y = ddp(value, ids, mask)
            y.backward(upstream)
            ys.append(y.detach())
            dxs.append(value.grad)
        ddp.finish_grad_sync()
        grads = {name: p.main_grad.clone() for name, p in module.named_parameters()}
        assert all(g.dtype == torch.float32 for g in grads.values())
        results.append((ys, dxs, grads))
    for position in (0, 1):
        for reference, actual in zip(results[0][position], results[1][position]):
            close(actual, reference)
    for name, reference in results[0][2].items():
        close(results[1][2][name], reference)
