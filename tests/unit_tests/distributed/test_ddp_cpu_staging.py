# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import gc

import pytest
import torch

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.fp8_utils import is_float8tensor, is_grouped_tensor
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class StagingModel(torch.nn.Module):
    """BF16 weights with tied storage through the same Parameter and an FP32 parameter."""

    def __init__(self, size: int) -> None:
        super().__init__()
        self.weights = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.full((size,), 0.5, device="cuda", dtype=torch.bfloat16))
                for _ in range(4)
            ]
        )
        self.tied_weight = self.weights[0]
        self.scale = torch.nn.Parameter(torch.ones(1, device="cuda", dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return sum(weight[:16].float().sum() for weight in self.weights) * x * self.scale


class TestDDPCPUStaging:
    """Compare DDP allocation peaks and actual distributed Adam updates."""

    def setup_method(self) -> None:
        Utils.initialize_model_parallel()

    def teardown_method(self) -> None:
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("grad_reduce_in_fp32", [False, True])
    @pytest.mark.parametrize("overlap", [False, True])
    def test_parameter_gradient_and_adam_equivalence(
        self, grad_reduce_in_fp32: bool, overlap: bool
    ) -> None:
        results = []
        for staging in (False, True):
            model = StagingModel(4096)
            parameter_ids = [id(param) for param in model.parameters()]
            ddp_config = DistributedDataParallelConfig(
                use_distributed_optimizer=True,
                ddp_cpu_staging=staging,
                grad_reduce_in_fp32=grad_reduce_in_fp32,
                overlap_grad_reduce=overlap,
                overlap_param_gather=overlap,
                bucket_size=4096,
            )
            ddp = DistributedDataParallel(
                TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
            )
            assert parameter_ids == [id(param) for param in ddp.parameters()]
            assert model.tied_weight is model.weights[0]
            for buffer in ddp.buffers:
                for param in buffer.params:
                    assert param.is_cuda
                    assert (
                        param.untyped_storage().data_ptr()
                        == buffer.param_data.untyped_storage().data_ptr()
                    )
            initial = [param.detach().cpu().clone() for param in ddp.parameters()]
            optimizer = get_megatron_optimizer(
                OptimizerConfig(
                    optimizer="adam",
                    lr=0.1,
                    bf16=True,
                    use_distributed_optimizer=True,
                    use_precision_aware_optimizer=False,
                    clip_grad=0.0,
                    weight_decay=0.0,
                ),
                [ddp],
            )
            step_results = []
            for _ in range(2):
                optimizer.zero_grad()
                ddp.zero_grad_buffer()
                ddp(torch.tensor([Utils.rank + 1.0], device="cuda")).sum().backward()
                ddp.finish_grad_sync()
                grads = [buffer.grad_data.detach().cpu().clone() for buffer in ddp.buffers]
                successful, _, _ = optimizer.step()
                assert successful
                ddp.start_param_sync(force_sync=True)
                torch.cuda.synchronize()
                step_results.append(
                    (grads, [param.detach().cpu().clone() for param in ddp.parameters()])
                )
            assert not torch.equal(initial[0], step_results[-1][1][0])
            results.append((initial, step_results))
            del optimizer, ddp, model
            gc.collect()
        for baseline, staged in zip(results[0][0], results[1][0]):
            torch.testing.assert_close(baseline, staged, rtol=0, atol=0)
        for baseline_step, staged_step in zip(results[0][1], results[1][1]):
            for baseline_values, staged_values in zip(baseline_step, staged_step):
                for baseline, staged in zip(baseline_values, staged_values):
                    torch.testing.assert_close(baseline, staged, rtol=0, atol=0)

    def test_initialization_memory_peak(self) -> None:
        peaks = []
        final_allocations = []
        parameter_bytes = 4 * 8 * 1024 * 1024 * 2
        for staging in (False, True):
            gc.collect()
            torch.cuda.empty_cache()
            model = StagingModel(8 * 1024 * 1024)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            ddp = DistributedDataParallel(
                TransformerConfig(num_attention_heads=1, num_layers=1),
                DistributedDataParallelConfig(
                    use_distributed_optimizer=True,
                    ddp_cpu_staging=staging,
                    grad_reduce_in_fp32=True,
                ),
                model,
            )
            torch.cuda.synchronize()
            peaks.append(torch.cuda.max_memory_allocated())
            final_allocations.append(torch.cuda.memory_allocated())
            del ddp, model
        # Small FP32 buffers and CUDA bookkeeping can add allocator-sized blocks.
        assert peaks[0] - peaks[1] >= parameter_bytes - 4096
        assert abs(final_allocations[0] - final_allocations[1]) <= 4096

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_without_distributed_optimizer(self, dtype: torch.dtype) -> None:
        model = torch.nn.Linear(16, 16, device="cuda", dtype=dtype)
        pointers = [param.data_ptr() for param in model.parameters()]
        ddp = DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1),
            DistributedDataParallelConfig(ddp_cpu_staging=True),
            model,
        )
        assert pointers == [param.data_ptr() for param in ddp.parameters()]
        assert all(param.is_cuda for param in ddp.parameters())

    @pytest.mark.parametrize("kind", ["fp8", "grouped"])
    def test_specialized_parameter_storage(
        self, kind: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if kind == "fp8":
            import transformer_engine_torch as tex
            from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer

            quantizer = Float8Quantizer(
                torch.ones(1, device="cuda"),
                torch.zeros(1, device="cuda"),
                tex.DType.kFloat8E4M3,
                rowwise=True,
                columnwise=False,
            )
            parameter = torch.nn.Parameter(
                quantizer(torch.ones(16, 16, device="cuda", dtype=torch.bfloat16))
            )
            assert is_float8tensor(parameter)
            model = torch.nn.Module()
            model.register_parameter("weight", parameter)
            expected = parameter.detach().float().cpu()
        else:
            from transformer_engine.pytorch.module import GroupedLinear

            monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "1")
            model = GroupedLinear(
                2,
                16,
                16,
                bias=False,
                params_dtype=torch.bfloat16,
                single_grouped_weight=True,
                use_grouped_tensor=True,
            )
            parameter = model.weight
            assert is_grouped_tensor(parameter)
            expected = parameter.rowwise_data.detach().cpu().clone()

        ddp = DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1),
            DistributedDataParallelConfig(use_distributed_optimizer=True, ddp_cpu_staging=True),
            model,
        )
        assert any(param is parameter for param in ddp.parameters())
        actual = (
            parameter.detach().float().cpu()
            if kind == "fp8"
            else parameter.rowwise_data.detach().cpu()
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
