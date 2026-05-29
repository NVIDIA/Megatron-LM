# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Tests for Muon optimizer CPU offloading in LayerWiseDistributedOptimizer.

This module verifies that the CPU offloading mechanism correctly:
- Moves fp32 master weights and optimizer state (momentum) to CPU pinned memory.
- Preserves tensor values exactly through offload/reload round-trips.
- Produces numerically identical results to the non-offloaded code path.

These tests require multi-GPU execution (via torchrun or pytest with distributed
launcher) since LayerWiseDistributedOptimizer shards parameters across DP ranks.
"""

from typing import Generator

import pytest
import torch
from tests.unit_tests.test_utilities import Utils

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig


def _create_model(seed: int, tp: int, pp: int, bf16_params: bool = True) -> GPTModel:
    """Create a small GPT model for testing.

    Args:
        seed: Random seed for reproducibility.
        tp: Tensor parallel size (already initialized via Utils).
        pp: Pipeline parallel size (already initialized via Utils).
        bf16_params: If True, model params are bf16 (exercises float16_groups path).
            If False, params are fp32 (exercises fp32_from_fp32_groups path).

    Returns:
        A GPTModel instance on the current CUDA device.
    """
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    config = TransformerConfig(
        num_layers=6,
        hidden_size=16,
        num_attention_heads=8,
        use_cpu_initialization=not bf16_params,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
    )
    layer_spec = get_gpt_layer_with_transformer_engine_spec()
    model = GPTModel(
        config=config,
        transformer_layer_spec=layer_spec,
        vocab_size=128,
        max_sequence_length=4,
        pre_process=parallel_state.is_pipeline_first_stage(),
        post_process=parallel_state.is_pipeline_last_stage(),
    )
    if bf16_params:
        model.bfloat16().cuda(torch.cuda.current_device())
    else:
        model.cuda(torch.cuda.current_device())
    return model


def _create_optimizer(model: GPTModel, cpu_offload: bool = True) -> LayerWiseDistributedOptimizer:
    """Create a Muon LayerWise optimizer with optional CPU offloading.

    Args:
        model: The GPT model whose parameters will be optimized.
        cpu_offload: Whether to enable CPU offloading of optimizer states.

    Returns:
        A LayerWiseDistributedOptimizer wrapping Muon + Adam fallback.
    """
    config = OptimizerConfig(
        bf16=True,
        params_dtype=torch.bfloat16,
        use_distributed_optimizer=False,
        use_layer_wise_distributed_optimizer=True,
        optimizer='muon',
        lr=0.0,
        optimizer_cpu_offload=cpu_offload,
    )
    optimizer = get_megatron_optimizer(config, [model])

    if isinstance(optimizer, LayerWiseDistributedOptimizer):
        for opt in optimizer.chained_optimizers:
            init_fn = getattr(opt, 'init_state_fn', None)
            if init_fn is None:
                continue
            if hasattr(opt, 'optimizer'):
                init_fn(opt.optimizer)
            else:
                init_fn(opt)
        if cpu_offload:
            optimizer.offload_optimizer_states()
    return optimizer


def _iter_fp16_opts(
    optimizer: LayerWiseDistributedOptimizer,
) -> Generator[Float16OptimizerWithFloat16Params, None, None]:
    """Yield Float16OptimizerWithFloat16Params sub-optimizers, skipping stubs."""
    for opt in optimizer.chained_optimizers:
        if getattr(opt, 'is_stub_optimizer', False):
            continue
        if isinstance(opt, Float16OptimizerWithFloat16Params):
            yield opt


class TestAdamOffloadConfig:
    """Verify Adam fallback offloading behavior in the legacy LayerWise path.

    In the legacy LayerWise path (use_distributed_optimizer=False), Adam params
    feed INTO LayerWise as a child — LayerWise handles offloading, so Adam's own
    HybridDeviceOptimizer must be disabled (optimizer_cpu_offload=False on the
    fallback config).  This prevents creating a HybridDeviceOptimizer for Adam
    and avoids double-offloading.

    In the separate DistributedOptimizer path (use_distributed_optimizer=True),
    Adam's DistOpt is a sibling of LayerWise — it manages its own offloading.
    That path requires full DDP setup and is validated via integration tests.
    """

    def setup_method(self, method) -> None:
        pass

    def teardown_method(self, method) -> None:
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize('tp,pp', [(2, 2)])
    def test_legacy_path_adam_not_hybrid_device_optimizer(self, tp: int, pp: int) -> None:
        """In legacy path, Adam child is NOT a HybridDeviceOptimizer."""
        if tp * pp > torch.cuda.device_count():
            pytest.skip("Not enough GPUs")

        from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer

        Utils.initialize_model_parallel(tp, pp)
        model = _create_model(seed=1, tp=tp, pp=pp, bf16_params=True)

        config = OptimizerConfig(
            bf16=True,
            params_dtype=torch.bfloat16,
            use_distributed_optimizer=False,
            use_layer_wise_distributed_optimizer=True,
            optimizer='muon',
            lr=0.0,
            optimizer_cpu_offload=True,
        )
        optimizer = get_megatron_optimizer(config, [model])
        assert isinstance(optimizer, LayerWiseDistributedOptimizer)

        # In the legacy path, Adam params go inside LayerWise. The fallback config
        # has optimizer_cpu_offload=False, so no HybridDeviceOptimizer is created.
        # LayerWise handles offloading for all children uniformly.
        for opt in optimizer.chained_optimizers:
            if isinstance(opt, Float16OptimizerWithFloat16Params):
                inner_opt = opt.optimizer
            else:
                inner_opt = opt
            assert not isinstance(inner_opt, HybridDeviceOptimizer), (
                f"In legacy LayerWise path, Adam should NOT use "
                f"HybridDeviceOptimizer (LayerWise manages offloading). "
                f"Got {type(inner_opt).__name__}"
            )

    @pytest.mark.parametrize('tp,pp', [(2, 2)])
    def test_legacy_path_layerwise_manages_adam_offload(self, tp: int, pp: int) -> None:
        """LayerWise offload/reload cycle covers Adam params in legacy path."""
        if tp * pp > torch.cuda.device_count():
            pytest.skip("Not enough GPUs")

        Utils.initialize_model_parallel(tp, pp)
        model = _create_model(seed=1, tp=tp, pp=pp, bf16_params=True)

        config = OptimizerConfig(
            bf16=True,
            params_dtype=torch.bfloat16,
            use_distributed_optimizer=False,
            use_layer_wise_distributed_optimizer=True,
            optimizer='muon',
            lr=0.0,
            optimizer_cpu_offload=True,
        )
        optimizer = get_megatron_optimizer(config, [model])
        assert isinstance(optimizer, LayerWiseDistributedOptimizer)
        assert optimizer._cpu_offload

        # After init, LayerWise offloads ALL children's fp32 master weights —
        # including Adam-managed params (biases, layernorms, embeddings).
        # Init state so optimizer.state has tensors.
        for opt in optimizer.chained_optimizers:
            init_fn = getattr(opt, 'init_state_fn', None)
            if init_fn is None:
                continue
            if hasattr(opt, 'optimizer'):
                init_fn(opt.optimizer)
            else:
                init_fn(opt)
        optimizer.offload_optimizer_states()

        # Verify ALL fp32_from_float16_groups params across all children are on CPU.
        found_any = False
        for opt in _iter_fp16_opts(optimizer):
            for group in opt.fp32_from_float16_groups:
                for param in group:
                    found_any = True
                    assert not param.data.is_cuda, (
                        "LayerWise should offload ALL children's master weights"
                    )
        assert found_any, "Expected at least some fp32 master weights to verify"

        # Reload and verify they're back on GPU.
        optimizer.reload_optimizer_states()
        for opt in _iter_fp16_opts(optimizer):
            for group in opt.fp32_from_float16_groups:
                for param in group:
                    assert param.data.is_cuda, "After reload, all should be on GPU"

        optimizer.offload_optimizer_states()


class TestMuonCPUOffload:
    """Tests for Muon CPU offloading in LayerWiseDistributedOptimizer.

    Verifies the correctness of the CPU offload mechanism that moves fp32 master
    weights and momentum buffers between GPU and CPU pinned memory each step.
    Tests cover state placement, round-trip fidelity, step execution, and
    bit-exact numerical equivalence against the non-offloaded baseline.
    """

    def setup_method(self, method) -> None:
        pass

    def teardown_method(self, method) -> None:
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize('tp,pp', [(2, 2), (1, 4), (4, 1)])
    def test_states_on_cpu(self, tp: int, pp: int) -> None:
        print(f"test_states_on_cpu: tp={tp}, pp={pp}")
        """After init with cpu_offload=True, fp32 master weights and state are on CPU."""
        if tp * pp > torch.cuda.device_count():
            pytest.skip("Not enough GPUs")

        Utils.initialize_model_parallel(tp, pp)
        model = _create_model(seed=2, tp=tp, pp=pp)
        optimizer = _create_optimizer(model, cpu_offload=True)

        assert isinstance(optimizer, LayerWiseDistributedOptimizer)
        assert optimizer._cpu_offload

        for opt in _iter_fp16_opts(optimizer):
            for group in opt.fp32_from_float16_groups:
                for param in group:
                    assert (
                        not param.data.is_cuda
                    ), f"fp32 master weight should be on CPU, got {param.data.device}"
            for state_vals in opt.optimizer.state.values():
                for key, val in state_vals.items():
                    if isinstance(val, torch.Tensor):
                        assert (
                            not val.is_cuda
                        ), f"optimizer state '{key}' should be on CPU, got {val.device}"

    @pytest.mark.parametrize('tp,pp', [(2, 2), (1, 4), (4, 1)])
    def test_roundtrip_correctness(self, tp: int, pp: int) -> None:
        """Offload -> reload preserves fp32 master weight values exactly."""
        print(f"test_roundtrip_correctness: tp={tp}, pp={pp}")
        if tp * pp > torch.cuda.device_count():
            pytest.skip("Not enough GPUs")

        Utils.initialize_model_parallel(tp, pp)
        model = _create_model(seed=2, tp=tp, pp=pp)
        optimizer = _create_optimizer(model, cpu_offload=True)

        assert isinstance(optimizer, LayerWiseDistributedOptimizer)

        snapshots = {}
        for opt in _iter_fp16_opts(optimizer):
            for gidx, group in enumerate(opt.fp32_from_float16_groups):
                for pidx, param in enumerate(group):
                    snapshots[(id(opt), gidx, pidx)] = param.data.clone()

        optimizer.reload_optimizer_states()

        for opt in _iter_fp16_opts(optimizer):
            for gidx, group in enumerate(opt.fp32_from_float16_groups):
                for pidx, param in enumerate(group):
                    assert param.data.is_cuda, "After reload, param should be on GPU"
                    expected = snapshots[(id(opt), gidx, pidx)].to(param.data.device)
                    assert torch.equal(
                        param.data, expected
                    ), "Master weight mismatch after offload->reload roundtrip"

        optimizer.offload_optimizer_states()

        for opt in _iter_fp16_opts(optimizer):
            for gidx, group in enumerate(opt.fp32_from_float16_groups):
                for pidx, param in enumerate(group):
                    assert not param.data.is_cuda, "After offload, param should be on CPU"

    @pytest.mark.parametrize('tp,pp', [(2, 2), (1, 4), (4, 1)])
    @pytest.mark.parametrize('bf16_params', [True, False])
    def test_step_runs(self, tp: int, pp: int, bf16_params: bool) -> None:
        """A full optimizer.step() succeeds with CPU offloading.

        Verifies the full prepare_grads → step_with_ready_grads cycle works
        when states start on CPU.  Sets main_grad on the params tracked by
        Float16OptimizerWithFloat16Params.float16_groups to exercise the
        _copy_model_grads_to_main_grads path that assigns CUDA grads to
        the fp32 master params (which are on CPU before reload).

        When bf16_params=True, model params are bf16 and the float16_groups →
        fp32_from_float16_groups path is exercised (the device mismatch path).
        When bf16_params=False, params are fp32 and fp32_from_fp32_groups is used.
        """
        if tp * pp > torch.cuda.device_count():
            pytest.skip("Not enough GPUs")

        Utils.initialize_model_parallel(tp, pp)
        model = _create_model(seed=2, tp=tp, pp=pp, bf16_params=bf16_params)
        optimizer = _create_optimizer(model, cpu_offload=True)

        assert isinstance(optimizer, LayerWiseDistributedOptimizer)

        # Set main_grad on all params tracked by the sub-optimizers so that
        # _copy_model_grads_to_main_grads is exercised during prepare_grads().
        for opt in optimizer.chained_optimizers:
            if getattr(opt, 'is_stub_optimizer', False):
                continue
            if isinstance(opt, Float16OptimizerWithFloat16Params):
                for group in opt.float16_groups:
                    for model_param in group:
                        model_param.main_grad = torch.randn_like(model_param.data)
                for group in opt.fp32_from_fp32_groups:
                    for model_param in group:
                        model_param.main_grad = torch.randn_like(model_param.data)

        update_successful, grad_norm, num_zeros = optimizer.step()
        assert isinstance(update_successful, bool)

        for opt in _iter_fp16_opts(optimizer):
            for group in opt.fp32_from_float16_groups:
                for param in group:
                    assert (
                        not param.data.is_cuda
                    ), "After step, fp32 master weights should be back on CPU"

    @pytest.mark.parametrize('tp,pp', [(2, 2), (4, 1)])
    @pytest.mark.parametrize('n_steps', [3, 5])
    @pytest.mark.parametrize('bf16_params', [True, False])
    def test_numerical_equivalence(
        self, tp: int, pp: int, n_steps: int, bf16_params: bool
    ) -> None:
        """Offloaded and non-offloaded optimizers produce bit-identical results.

        Runs both an offloaded and a non-offloaded optimizer for ``n_steps``
        with identical random gradients, then verifies that fp32 master weights
        and optimizer state tensors match exactly.
        """
        if tp * pp > torch.cuda.device_count():
            pytest.skip("Not enough GPUs")

        Utils.initialize_model_parallel(tp, pp)

        model_off = _create_model(seed=42, tp=tp, pp=pp, bf16_params=bf16_params)
        model_ref = _create_model(seed=42, tp=tp, pp=pp, bf16_params=bf16_params)

        opt_off = _create_optimizer(model_off, cpu_offload=True)
        opt_ref = _create_optimizer(model_ref, cpu_offload=False)

        assert isinstance(opt_off, LayerWiseDistributedOptimizer)
        assert isinstance(opt_ref, LayerWiseDistributedOptimizer)

        rank = torch.distributed.get_rank()

        for step_i in range(n_steps):
            torch.manual_seed(1000 + step_i + rank)

            # Set main_grad on all tracked params (same path as DDP grad buffers).
            for fp16_opt_off, fp16_opt_ref in zip(
                _iter_fp16_opts(opt_off), _iter_fp16_opts(opt_ref)
            ):
                for grp_off, grp_ref in zip(
                    fp16_opt_off.float16_groups, fp16_opt_ref.float16_groups
                ):
                    for p_off, p_ref in zip(grp_off, grp_ref):
                        g = torch.randn_like(p_off.data)
                        p_off.main_grad = g.clone()
                        p_ref.main_grad = g.clone()
                for grp_off, grp_ref in zip(
                    fp16_opt_off.fp32_from_fp32_groups, fp16_opt_ref.fp32_from_fp32_groups
                ):
                    for p_off, p_ref in zip(grp_off, grp_ref):
                        g = torch.randn_like(p_off.data)
                        p_off.main_grad = g.clone()
                        p_ref.main_grad = g.clone()

            opt_off.step()
            opt_ref.step()

        opt_off.reload_optimizer_states()

        for opt_o, opt_r in zip(_iter_fp16_opts(opt_off), _iter_fp16_opts(opt_ref)):
            for grp_o, grp_r in zip(opt_o.fp32_from_float16_groups, opt_r.fp32_from_float16_groups):
                for pidx, (p_o, p_r) in enumerate(zip(grp_o, grp_r)):
                    p_o_gpu = p_o.data.to('cuda') if not p_o.data.is_cuda else p_o.data
                    assert torch.equal(p_o_gpu, p_r.data), (
                        f"fp32 master weight mismatch at param {pidx} after {n_steps} steps, "
                        f"max diff = {(p_o_gpu - p_r.data).abs().max().item()}"
                    )

            for (key_o, state_o), (key_r, state_r) in zip(
                opt_o.optimizer.state.items(), opt_r.optimizer.state.items()
            ):
                common_keys = set(state_o.keys()) & set(state_r.keys())
                for skey in common_keys:
                    v_o, v_r = state_o[skey], state_r[skey]
                    if not isinstance(v_o, torch.Tensor):
                        continue
                    v_o_gpu = v_o.to('cuda') if not v_o.is_cuda else v_o
                    assert torch.equal(v_o_gpu, v_r), (
                        f"optimizer state '{skey}' mismatch after {n_steps} steps, "
                        f"max diff = {(v_o_gpu - v_r).abs().max().item()}"
                    )

        opt_off.offload_optimizer_states()

    @pytest.mark.parametrize('tp,pp', [(2, 2), (4, 1)])
    @pytest.mark.parametrize('bf16_params', [True, False])
    def test_prepare_grads_reloads_before_grad_copy(
        self, tp: int, pp: int, bf16_params: bool
    ) -> None:
        """prepare_grads() must reload states to GPU before gradient assignment.

        This directly tests the fix for the reviewer-identified bug where
        _copy_model_grads_to_main_grads assigns a CUDA grad tensor to the
        fp32 main_param — which requires main_param.data to be on GPU (since
        nn.Parameter forbids cross-device .data/.grad).

        With bf16_params=True, the float16_groups → fp32_from_float16_groups
        path is exercised — this is the path that would RuntimeError without
        the prepare_grads() fix.
        """
        if tp * pp > torch.cuda.device_count():
            pytest.skip("Not enough GPUs")

        Utils.initialize_model_parallel(tp, pp)
        model = _create_model(seed=7, tp=tp, pp=pp, bf16_params=bf16_params)
        optimizer = _create_optimizer(model, cpu_offload=True)

        assert isinstance(optimizer, LayerWiseDistributedOptimizer)

        # Verify states start on CPU after construction.
        for opt in _iter_fp16_opts(optimizer):
            for group in opt.fp32_from_float16_groups:
                for param in group:
                    assert not param.data.is_cuda

        # Set main_grad (CUDA tensors) on all tracked params — this is what DDP does.
        for opt in _iter_fp16_opts(optimizer):
            for group in opt.float16_groups:
                for model_param in group:
                    model_param.main_grad = torch.randn_like(model_param.data)
            for group in opt.fp32_from_fp32_groups:
                for model_param in group:
                    model_param.main_grad = torch.randn_like(model_param.data)

        # Call prepare_grads() — should reload states and then copy grads.
        # Without the fix, this raises RuntimeError (cross-device grad assignment).
        result = optimizer.prepare_grads()
        assert isinstance(result, bool)

        # After prepare_grads, fp32 master params should be on GPU with grads set.
        for opt in _iter_fp16_opts(optimizer):
            for group in opt.fp32_from_float16_groups:
                for param in group:
                    assert param.data.is_cuda, (
                        "After prepare_grads, fp32 master weight must be on GPU"
                    )

        # Now step_with_ready_grads should work and offload back.
        optimizer.step_with_ready_grads()

        for opt in _iter_fp16_opts(optimizer):
            for group in opt.fp32_from_float16_groups:
                for param in group:
                    assert not param.data.is_cuda, (
                        "After step_with_ready_grads, fp32 master weight must be on CPU"
                    )
