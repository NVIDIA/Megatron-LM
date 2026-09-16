# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real TE capture/replay regressions for the existing main mHC model entry points.

Fused cases count the actual accelerated forward/backward launchers during CUDA
capture, independently of eager warmup and reference execution. SM10x Blackwell
also requires the cuTile projection and aggregation-backward paths; Hopper can
use the supported Triton/native combination selected by the same auto policy.
"""

from collections import Counter
from functools import wraps

import pytest
import torch

from megatron.core.fusions import fused_mhc_kernels
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.hybrid.layers.hybrid_hyper_connection import HyperConnectionHybridLayer
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    init_num_microbatches_calculator,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.cuda_graphs import (
    TECudaGraphHelper,
    _get_mtp_te_layers,
    _layer_is_graphable,
    set_current_microbatch,
)
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.module import convert_module_to_dtype_except_fp32_marked
from megatron.core.transformer.transformer_layer import (
    HyperConnectionTransformerLayer,
    TransformerLayer,
)
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils


def _record_mhc_backend_calls(monkeypatch):
    """Observe real kernel dispatch without changing the selected implementation."""
    calls = {'eager': Counter(), 'capture': Counter()}

    def record(name, implementation):
        @wraps(implementation)
        def counted(*args, **kwargs):
            phase = 'capture' if torch.cuda.is_current_stream_capturing() else 'eager'
            result = implementation(*args, **kwargs)
            calls[phase][name] += 1
            return result

        return counted

    # These Python launchers are looked up directly by the autograd functions.
    for name in (
        '_triton_sinkhorn_fwd',
        '_triton_sinkhorn_bwd',
        '_cutile_sinkhorn_fwd',
        '_cutile_sinkhorn_bwd',
        '_cutile_h_aggregate_fwd',
        '_cutile_h_aggregate_bwd',
        '_cutile_h_post_bda_fwd',
        '_cutile_h_post_bda_bwd',
        '_cutile_proj_rms_compute_h_fwd',
        '_cutile_fused_compute_h_proj_rms_bwd',
    ):
        implementation = getattr(fused_mhc_kernels, name, None)
        if implementation is not None:
            monkeypatch.setattr(fused_mhc_kernels, name, record(name, implementation))

    # The other Triton launchers are saved in the implementation table at import.
    for name in ('h_aggregate_fwd', 'h_post_bda_fwd', 'h_post_bda_bwd'):
        implementation = fused_mhc_kernels._TRITON_IMPLS.get(name)
        if implementation is not None:
            monkeypatch.setitem(
                fused_mhc_kernels._TRITON_IMPLS, name, record(f'_triton_{name}', implementation)
            )
    return calls


def _expected_mhc_backend_calls(fused):
    """Require acceleration for fused cases, using the existing auto policy."""
    if not fused:
        return set()
    triton = fused_mhc_kernels.is_triton_available()
    cutile = fused_mhc_kernels.is_cutile_available()
    assert triton or cutile, 'The fused mHC case requires an accelerated backend.'
    if torch.cuda.get_device_capability()[0] == 10:
        assert cutile, 'The SM10x Blackwell fused mHC lane requires cuTile support.'

    backend = 'triton' if triton else 'cutile'
    expected = {
        f'_{backend}_{operation}'
        for operation in (
            'sinkhorn_fwd',
            'sinkhorn_bwd',
            'h_aggregate_fwd',
            'h_post_bda_fwd',
            'h_post_bda_bwd',
        )
    }
    if cutile:
        expected.update(
            {
                '_cutile_h_aggregate_bwd',
                '_cutile_proj_rms_compute_h_fwd',
                '_cutile_fused_compute_h_proj_rms_bwd',
            }
        )
    return expected


def _config(kind, scopes, fused=False, tp=1, cp=1, ep=1):
    moe = 'moe' in kind
    return TransformerConfig(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        ffn_hidden_size=256,
        activation_func=torch.nn.functional.silu if 'gdn' in kind else torch.nn.functional.gelu,
        gated_linear_unit='gdn' in kind,
        use_cpu_initialization=True,
        params_dtype=torch.bfloat16,
        bf16=True,
        use_te_rng_tracker=True,
        gradient_accumulation_fusion=False,
        enable_mhc_connections=True,
        mhc_num_residual_streams=2,
        mhc_sinkhorn_iterations=3,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        cuda_graph_impl='transformer_engine' if scopes is not None else 'none',
        cuda_graph_modules=scopes,
        cuda_graph_warmup_steps=3,
        use_fused_mhc=fused,
        num_moe_experts=4 if moe else None,
        moe_grouped_gemm=moe,
        moe_router_topk=2,
        moe_router_load_balancing_type='none',
        moe_token_dispatcher_type='alltoall',
        mtp_num_layers=2 if 'mtp' in kind else None,
        tensor_model_parallel_size=tp,
        context_parallel_size=cp,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=tp,
        add_bias_linear=not moe or tp == 1,
        sequence_parallel=tp > 1,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        mamba_state_dim=16,
        mamba_head_dim=32,
        mamba_num_groups=1,
    )


def _model(kind, config, pg):
    kwargs = dict(config=config, vocab_size=128, max_sequence_length=64, pg_collection=pg)
    if kind.startswith('gpt'):
        spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=config.num_moe_experts, moe_grouped_gemm=config.moe_grouped_gemm
        )
        spec.module = HyperConnectionTransformerLayer
        spec.submodules.self_attention_hyper_connection = HyperConnectionModule
        spec.submodules.mlp_hyper_connection = HyperConnectionModule
        model = GPTModel(transformer_layer_spec=spec, position_embedding_type='rope', **kwargs)
    else:
        pattern = '*E' if 'moe' in kind else '*-'
        if 'gdn' in kind:
            pattern = 'G-'
        if 'mamba' in kind:
            pattern = 'M-'
        if 'mtp' in kind:
            pattern += ('/' + pattern) * config.mtp_num_layers
        model = HybridModel(
            hybrid_stack_spec=hybrid_stack_spec,
            hybrid_layer_pattern=pattern,
            position_embedding_type='rope',
            **kwargs,
        )
    model.cuda().train()
    convert_module_to_dtype_except_fp32_marked(model, torch.bfloat16)
    # TECudaGraphHelper's reset protocol normally comes from MCore DDP. These
    # tests compare every local gradient directly, without reducing it through DDP.
    model.zero_grad_buffer = lambda: model.zero_grad(set_to_none=True)
    return model


@pytest.mark.internal
@pytest.mark.launch_on_gb200
@pytest.mark.skipif(not is_te_min_version('2.10.0'), reason='TE graph tests require TE >= 2.10')
class TestMHCTEGraphs:
    def setup_method(self):
        self.helper = None
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)

    def teardown_method(self):
        if self.helper is not None and self.helper.graphs_created():
            self.helper.delete_cuda_graphs()
        destroy_num_microbatches_calculator()
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        'kind,scopes',
        [
            ('hybrid', [CudaGraphModule.attn, CudaGraphModule.mlp]),
            ('hybrid_gdn', [CudaGraphModule.attn, CudaGraphModule.mlp]),
            ('hybrid_mamba', [CudaGraphModule.mamba, CudaGraphModule.mlp]),
            ('hybrid_mtp', [CudaGraphModule.attn, CudaGraphModule.mlp]),
            ('hybrid_moe', [CudaGraphModule.attn, CudaGraphModule.moe_router]),
            ('hybrid_moe_mtp', [CudaGraphModule.attn, CudaGraphModule.moe_router]),
            (
                'hybrid_moe',
                [CudaGraphModule.attn, CudaGraphModule.moe_router, CudaGraphModule.moe_preprocess],
            ),
            ('gpt', [CudaGraphModule.attn]),
            ('gpt', [CudaGraphModule.mlp]),
            ('gpt_moe', [CudaGraphModule.attn, CudaGraphModule.moe_router]),
            (
                'gpt_moe',
                [CudaGraphModule.attn, CudaGraphModule.moe_router, CudaGraphModule.moe_preprocess],
            ),
        ],
    )
    @pytest.mark.parametrize('fused', [False, True])
    def test_capture_replay_all_gradients(self, kind, scopes, fused, monkeypatch):
        self._run(kind, scopes, fused, monkeypatch)

    def test_combined_hybrid_attention_moe_graph_is_rejected(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123, te_rng_tracker=True, force_reset_rng=True)
        config = _config('hybrid_moe', [CudaGraphModule.attn, CudaGraphModule.moe_router])
        spec = get_gpt_layer_with_transformer_engine_spec(num_experts=4, moe_grouped_gemm=True)
        inner = TransformerLayer(config, spec.submodules)
        with pytest.raises(NotImplementedError, match='MoE-only inner layer'):
            HyperConnectionHybridLayer(config, inner)

    @pytest.mark.parametrize('kind', ['hybrid_moe', 'gpt_moe'])
    @pytest.mark.parametrize('input_name', ['padding_mask', 'packed_seq_params'])
    @pytest.mark.parametrize('method', ['_te_cuda_graph_capture', '_te_cuda_graph_replay'])
    def test_te_training_rejects_inputs_without_static_samples(self, kind, input_name, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123, te_rng_tracker=True, force_reset_rng=True)
        config = _config(kind, [CudaGraphModule.attn, CudaGraphModule.moe_router])
        model = _model(kind, config, ProcessGroupCollection.use_mpu_process_groups())
        layer = model.decoder.layers[-1]
        unsupported = (
            torch.zeros(2, 8, dtype=torch.bool, device='cuda')
            if input_name == 'padding_mask'
            else object()
        )
        with pytest.raises(NotImplementedError, match=input_name):
            getattr(layer, method)(
                hidden_states=torch.ones(8, 2, 256, device='cuda', dtype=torch.bfloat16),
                **{input_name: unsupported},
            )

    @pytest.mark.parametrize('kind', ['hybrid_moe_mtp', 'gpt_moe'])
    def test_parallel_capture_replay(self, kind, monkeypatch):
        self._run(
            kind,
            [CudaGraphModule.attn, CudaGraphModule.moe_router],
            False,
            monkeypatch,
            tp=2,
            cp=2,
            ep=2,
        )

    def _run(self, kind, scopes, fused, monkeypatch, tp=1, cp=1, ep=1):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=1,
            context_parallel_size=cp,
            expert_model_parallel_size=ep,
            expert_tensor_parallel_size=tp,
        )
        model_parallel_cuda_manual_seed(123, te_rng_tracker=True, force_reset_rng=True)
        expected_backend_calls = _expected_mhc_backend_calls(fused)
        backend_calls = _record_mhc_backend_calls(monkeypatch)
        init_num_microbatches_calculator(
            rank=0,
            global_batch_size=2,
            micro_batch_size=2,
            data_parallel_size=1,
            decrease_batch_size_if_needed=False,
        )
        pg = ProcessGroupCollection.use_mpu_process_groups()
        config = _config(kind, scopes, fused, tp, cp, ep)
        graphed = _model(kind, config, pg)
        reference = _model(kind, _config(kind, None, fused, tp, cp, ep), pg)
        reference.load_state_dict(graphed.state_dict(), strict=True)
        optimizer = torch.optim.SGD(graphed.parameters(), lr=0.01)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        initial_keys = set(graphed.state_dict())

        local_seq = 32 // cp
        ids = torch.arange(local_seq, device='cuda').expand(2, -1).contiguous()
        positions = ids.clone()
        inputs = [
            torch.randn(local_seq // tp, 2, 128, device='cuda', dtype=torch.bfloat16)
            for _ in range(4)
        ]

        def step(model, value):
            model.zero_grad(set_to_none=True)

            def own_gradient(parameter):
                # TE reuses gradient buffers between captured backward graphs.
                # Native DDP immediately copies into owned main_grad storage;
                # this standalone optimizer must likewise own each .grad before
                # the next backward graph can reuse its returned buffer.
                parameter.grad = parameter.grad.clone()

            handles = [
                parameter.register_post_accumulate_grad_hook(own_gradient)
                for parameter in model.parameters()
                if parameter.requires_grad
            ]
            try:
                hidden = value.detach().clone().requires_grad_()
                output = model(
                    input_ids=ids, position_ids=positions, attention_mask=None, decoder_input=hidden
                )
                output.float().square().mean().backward()
            finally:
                for handle in handles:
                    handle.remove()
            grads = {
                name: None if p.grad is None else p.grad.detach().clone()
                for name, p in model.named_parameters()
            }
            return output.detach().clone(), hidden.grad.detach().clone(), grads

        # Warm up off the default stream, as TE does for its own warmup. Retained
        # AccumulateGrad nodes must not pull the default stream into capture.
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                step(graphed, inputs[0])
        torch.cuda.current_stream().wait_stream(warmup_stream)
        graphed.zero_grad(set_to_none=True)
        self.helper = TECudaGraphHelper([graphed], config, 32, 2, pg_collection=pg)
        expected = list(graphed.decoder.layers)
        for mtp in getattr(getattr(graphed, 'mtp', None), 'layers', []):
            expected.extend(_get_mtp_te_layers(mtp.mtp_model_layer))
        expected = [layer for layer in expected if _layer_is_graphable(layer, config)]
        assert self.helper.flattened_callables == expected and expected
        if 'mtp' in kind:
            assert sum(self.helper.flattened_callables_is_mtp) == 4
        self.helper.create_cudagraphs()
        assert self.helper.graphs_created()
        # Python dispatch runs when recording a CUDA graph, not on its replay.
        # Counting capture separately prevents eager warmup from proving this.
        assert set(backend_calls['capture']) == expected_backend_calls, backend_calls
        print(f'mHC fused={fused}; captured backend calls: {dict(backend_calls["capture"])}')
        assert all(layer.cuda_graphs for layer in expected)
        assert set(graphed.state_dict()) == initial_keys

        hooks = Counter()
        graphed._make_forward_pre_hook = lambda: lambda module: hooks.update([id(module)])
        self.helper.cuda_graph_set_manual_hooks()
        covered_modules = {
            id(module)
            for layer in expected
            for submodule in layer._get_submodules_under_cudagraphs()
            for module in submodule.modules()
            if next(module.parameters(recurse=False), None) is not None
        }
        for layer in expected:
            if isinstance(layer, HyperConnectionHybridLayer) and layer._is_partial_moe_graph():
                assert (
                    not {id(module) for module in layer.inner_layer.mlp.experts.modules()}
                    & covered_modules
                )

        replay_calls = Counter()
        for layer in expected:
            graphs = layer.cuda_graphs

            def counted(*args, _layer=layer, _graph=graphs[0], **kwargs):
                replay_calls[id(_layer)] += 1
                return _graph(*args, **kwargs)

            counted.reset = graphs[0].reset
            layer.cuda_graphs = [counted]

        for index, value in enumerate(inputs):
            set_current_microbatch(graphed, index)
            assert all(layer.current_microbatch == index for layer in expected)
            expected_output, expected_input_grad, expected_grads = step(reference, value)
            actual_output, actual_input_grad, actual_grads = step(graphed, value)
            torch.testing.assert_close(actual_output, expected_output, rtol=2e-2, atol=2e-3)
            torch.testing.assert_close(actual_input_grad, expected_input_grad, rtol=2e-2, atol=2e-3)
            assert actual_grads.keys() == expected_grads.keys()
            for name, expected_grad in expected_grads.items():
                actual_grad = actual_grads[name]
                assert (actual_grad is None) == (expected_grad is None), name
                if expected_grad is not None:
                    torch.testing.assert_close(
                        actual_grad,
                        expected_grad,
                        rtol=3e-2,
                        atol=2e-3,
                        msg=lambda message: f'replay={index} gradient {name}: {message}',
                    )
            mhc_grads = [
                grad
                for name, grad in actual_grads.items()
                if 'hyper_connection' in name or 'mhc' in name
            ]
            assert mhc_grads and all(grad is not None for grad in mhc_grads)
            assert any(torch.count_nonzero(grad) for grad in mhc_grads)
            reference_optimizer.step()
            optimizer.step()
            for (name, actual), (reference_name, reference_param) in zip(
                graphed.named_parameters(), reference.named_parameters()
            ):
                assert name == reference_name
                torch.testing.assert_close(
                    actual,
                    reference_param,
                    rtol=3e-2,
                    atol=2e-3,
                    msg=lambda message: f'replay={index} parameter {name}: {message}',
                )
            assert all(replay_calls[id(layer)] == index + 1 for layer in expected)
            assert set(hooks) == covered_modules
            assert all(count == index + 1 for count in hooks.values())
        if not fused:
            assert not any(backend_calls.values()), backend_calls


@pytest.mark.launch_on_gb200
@pytest.mark.parametrize('impl', ['transformer_engine', 'full_iteration'])
@pytest.mark.parametrize(
    'extra,error',
    [
        ({'recompute_granularity': 'selective', 'recompute_modules': ['mhc']}, 'selective mHC'),
        (
            {'fine_grained_activation_offloading': True, 'offload_modules': ['core_attn']},
            'offloading',
        ),
        ({'overlap_moe_expert_parallel_comm': True}, 'EP A2A overlap'),
    ],
)
def test_mhc_graph_combination_guards(impl, extra, error):
    with pytest.raises((NotImplementedError, AssertionError), match=error):
        TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            enable_mhc_connections=True,
            cuda_graph_impl=impl,
            **extra,
        )


@pytest.mark.launch_on_gb200
def test_mhc_te_repeated_mtp_is_rejected():
    with pytest.raises(NotImplementedError, match='independent MTP layers'):
        TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            enable_mhc_connections=True,
            cuda_graph_impl='transformer_engine',
            mtp_num_layers=2,
            mtp_use_repeated_layer=True,
        )
