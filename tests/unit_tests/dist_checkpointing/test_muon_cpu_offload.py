# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _create_model(seed, tp, pp):
    """Create a small GPT model for testing."""
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    config = TransformerConfig(
        num_layers=6,
        hidden_size=16,
        num_attention_heads=8,
        use_cpu_initialization=True,
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
    model.cuda(torch.cuda.current_device())
    return model


def _create_optimizer(model, cpu_offload=True):
    """Create a Muon LayerWise optimizer with optional CPU offloading."""
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


def _iter_fp16_opts(optimizer):
    """Yield Float16OptimizerWithFloat16Params sub-optimizers."""
    for opt in optimizer.chained_optimizers:
        if getattr(opt, 'is_stub_optimizer', False):
            continue
        if isinstance(opt, Float16OptimizerWithFloat16Params):
            yield opt


class TestMuonCPUOffload:
    """Tests for Muon CPU offloading in LayerWiseDistributedOptimizer."""

    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize('tp,pp', [(2, 2), (1, 4), (4, 1)])
    def test_states_on_cpu(self, tp, pp):
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
                    assert not param.data.is_cuda, (
                        f"fp32 master weight should be on CPU, got {param.data.device}"
                    )
            for state_vals in opt.optimizer.state.values():
                for key, val in state_vals.items():
                    if isinstance(val, torch.Tensor):
                        assert not val.is_cuda, (
                            f"optimizer state '{key}' should be on CPU, got {val.device}"
                        )

    @pytest.mark.parametrize('tp,pp', [(2, 2), (1, 4), (4, 1)])
    def test_roundtrip_correctness(self, tp, pp):
        """Offload -> reload preserves fp32 master weight values exactly."""
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
                    assert torch.equal(param.data, expected), (
                        "Master weight mismatch after offload->reload roundtrip"
                    )

        optimizer.offload_optimizer_states()

        for opt in _iter_fp16_opts(optimizer):
            for gidx, group in enumerate(opt.fp32_from_float16_groups):
                for pidx, param in enumerate(group):
                    assert not param.data.is_cuda, "After offload, param should be on CPU"

    @pytest.mark.parametrize('tp,pp', [(2, 2), (1, 4), (4, 1)])
    def test_step_runs(self, tp, pp):
        """A full optimizer.step() succeeds with CPU offloading."""
        if tp * pp > torch.cuda.device_count():
            pytest.skip("Not enough GPUs")

        Utils.initialize_model_parallel(tp, pp)
        model = _create_model(seed=2, tp=tp, pp=pp)
        optimizer = _create_optimizer(model, cpu_offload=True)

        assert isinstance(optimizer, LayerWiseDistributedOptimizer)

        for param in model.parameters():
            if param.requires_grad:
                g = torch.randn_like(param.data)
                param.grad = g
                param.main_grad = g

        update_successful, grad_norm, num_zeros = optimizer.step()
        assert isinstance(update_successful, bool)

        for opt in _iter_fp16_opts(optimizer):
            for group in opt.fp32_from_float16_groups:
                for param in group:
                    assert not param.data.is_cuda, (
                        "After step, fp32 master weights should be back on CPU"
                    )

    @pytest.mark.parametrize('tp,pp', [(2, 2), (4, 1)])
    @pytest.mark.parametrize('n_steps', [3, 5])
    def test_numerical_equivalence(self, tp, pp, n_steps):
        """Offloaded and non-offloaded optimizers produce identical results."""
        if tp * pp > torch.cuda.device_count():
            pytest.skip("Not enough GPUs")

        Utils.initialize_model_parallel(tp, pp)

        model_off = _create_model(seed=42, tp=tp, pp=pp)
        model_ref = _create_model(seed=42, tp=tp, pp=pp)

        opt_off = _create_optimizer(model_off, cpu_offload=True)
        opt_ref = _create_optimizer(model_ref, cpu_offload=False)

        assert isinstance(opt_off, LayerWiseDistributedOptimizer)
        assert isinstance(opt_ref, LayerWiseDistributedOptimizer)

        rank = torch.distributed.get_rank()

        for step_i in range(n_steps):
            torch.manual_seed(1000 + step_i + rank)

            for p_off, p_ref in zip(model_off.parameters(), model_ref.parameters()):
                if not p_off.requires_grad:
                    continue
                g = torch.randn_like(p_off.data)
                p_off.grad = g.clone()
                p_off.main_grad = p_off.grad
                p_ref.grad = g.clone()
                p_ref.main_grad = p_ref.grad

            opt_off.step()
            opt_ref.step()

        opt_off.reload_optimizer_states()

        for opt_o, opt_r in zip(_iter_fp16_opts(opt_off), _iter_fp16_opts(opt_ref)):
            for grp_o, grp_r in zip(
                opt_o.fp32_from_float16_groups, opt_r.fp32_from_float16_groups
            ):
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
