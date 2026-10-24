# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import BaseTransformerLayer, TransformerLayer
from megatron.core.transformer.utils import validate_extra_layer_kwargs
from tests.unit_tests.test_utilities import Utils


class _AdaLNLayer(torch.nn.Module, BaseTransformerLayer):
    """Minimal DiT/AdaLN-style layer that consumes a conditioning embedding."""

    def __init__(self, config, layer_number, **kwargs):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.modulation = torch.nn.Linear(config.hidden_size, 2 * config.hidden_size)
        self.received_condition = []

    def forward(self, hidden_states, condition_emb=None, **kwargs):
        self.received_condition.append(condition_emb)
        if condition_emb is None:
            return hidden_states + 1, None
        # condition_emb: [b, h] broadcasts over hidden_states: [s, b, h].
        scale, shift = self.modulation(condition_emb).chunk(2, dim=-1)
        return hidden_states * (1 + scale) + shift, None


class _AdaLNTransformerLayer(TransformerLayer):
    """TransformerLayer subclass that modulates its output with a conditioning embedding."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.modulation = torch.nn.Linear(config.hidden_size, 2 * config.hidden_size)

    def forward(self, *args, condition_emb=None, **kwargs):
        output, context = super().forward(*args, **kwargs)
        if condition_emb is not None:
            scale, shift = self.modulation(condition_emb).chunk(2, dim=-1)
            output = output * (1 + scale) + shift
        return output, context


class _InnerLayerWrapper(torch.nn.Module, BaseTransformerLayer):
    """Wraps a TransformerLayer under ``inner_layer`` with a fixed forward signature,
    like HybridStack's mHC wrapper. It does not accept extra layer kwargs."""

    def __init__(self, config, layer_number, **kwargs):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.inner_layer = build_module(
            get_gpt_layer_with_transformer_engine_spec(),
            config=config,
            layer_number=layer_number,
            **kwargs,
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        rotary_pos_emb=None,
        inference_context=None,
        packed_seq_params=None,
        padding_mask=None,
    ):
        return self.inner_layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
            padding_mask=padding_mask,
        )


class TestTransformerBlockExtraLayerKwargs:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.config = TransformerConfig(
            num_layers=4, hidden_size=16, num_attention_heads=4, use_cpu_initialization=True
        )
        self.block = TransformerBlock(
            self.config, ModuleSpec(module=_AdaLNLayer), post_process=False
        ).cuda()
        self.seq_len, self.batch = 8, 2

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _inputs(self):
        hidden_states = torch.randn(
            self.seq_len, self.batch, self.config.hidden_size, device="cuda", requires_grad=True
        )
        condition_emb = torch.randn(
            self.batch, self.config.hidden_size, device="cuda", requires_grad=True
        )
        return hidden_states, condition_emb

    def _set_recompute(self, method):
        self.config.recompute_granularity = None if method is None else 'full'
        self.config.recompute_method = method
        self.config.recompute_num_layers = 2 if method == 'uniform' else self.config.num_layers

    def test_none_preserves_behavior(self):
        hidden_states, _ = self._inputs()
        output = self.block(hidden_states=hidden_states, attention_mask=None)
        torch.testing.assert_close(output, hidden_states + self.config.num_layers)
        for layer in self.block.layers:
            assert layer.received_condition == [None]

    def test_condition_reaches_every_layer(self):
        hidden_states, condition_emb = self._inputs()
        self.block(
            hidden_states=hidden_states,
            attention_mask=None,
            extra_layer_kwargs={"condition_emb": condition_emb},
        )
        for layer in self.block.layers:
            assert len(layer.received_condition) == 1
            assert layer.received_condition[0] is condition_emb

    @pytest.mark.parametrize("recompute_method", ["uniform", "block"])
    def test_full_recompute_matches_outputs_and_gradients(self, recompute_method):
        hidden_states, condition_emb = self._inputs()

        def run(method):
            self._set_recompute(method)
            self.block.zero_grad(set_to_none=True)
            h = hidden_states.detach().clone().requires_grad_(True)
            c = condition_emb.detach().clone().requires_grad_(True)
            out = self.block(
                hidden_states=h, attention_mask=None, extra_layer_kwargs={"condition_emb": c}
            )
            out.sum().backward()
            param_grads = [p.grad.clone() for p in self.block.parameters()]
            return out.detach(), h.grad, c.grad, param_grads

        ref_out, ref_h_grad, ref_c_grad, ref_param_grads = run(None)
        out, h_grad, c_grad, param_grads = run(recompute_method)

        assert c_grad is not None, "condition_emb lost its gradient under full recompute"
        torch.testing.assert_close(out, ref_out)
        torch.testing.assert_close(h_grad, ref_h_grad)
        torch.testing.assert_close(c_grad, ref_c_grad)
        for grad, ref_grad in zip(param_grads, ref_param_grads):
            torch.testing.assert_close(grad, ref_grad)

    def test_colliding_key_rejected(self):
        hidden_states, _ = self._inputs()
        with pytest.raises(TypeError, match="attention_mask"):
            self.block(
                hidden_states=hidden_states,
                attention_mask=None,
                extra_layer_kwargs={"attention_mask": None},
            )


class TestTransformerLayerSubclassExtraLayerKwargs:
    """Covers the TransformerLayer branch of checkpointed_forward with a real layer."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=True,
            hidden_dropout=0.0,
            attention_dropout=0.0,
        )
        spec = get_gpt_layer_with_transformer_engine_spec()
        spec.module = _AdaLNTransformerLayer
        self.block = TransformerBlock(self.config, spec, post_process=False).cuda()
        self.seq_len, self.batch = 8, 2

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("recompute_method", ["uniform", "block"])
    def test_full_recompute_matches_outputs_and_gradients(self, recompute_method):
        assert all(isinstance(layer, TransformerLayer) for layer in self.block.layers)
        hidden_states = torch.randn(
            self.seq_len, self.batch, self.config.hidden_size, device="cuda"
        )
        condition_emb = torch.randn(self.batch, self.config.hidden_size, device="cuda")
        attention_mask = torch.ones(
            (1, 1, self.seq_len, self.seq_len), dtype=torch.bool, device="cuda"
        )

        def run(method):
            self.config.recompute_granularity = None if method is None else 'full'
            self.config.recompute_method = method
            self.config.recompute_num_layers = 1 if method == 'uniform' else 2
            self.block.zero_grad(set_to_none=True)
            h = hidden_states.clone().requires_grad_(True)
            c = condition_emb.clone().requires_grad_(True)
            out = self.block(
                hidden_states=h,
                attention_mask=attention_mask,
                extra_layer_kwargs={"condition_emb": c},
            )
            out.sum().backward()
            param_grads = [p.grad.clone() for p in self.block.parameters()]
            return out.detach(), h.grad, c.grad, param_grads

        ref_out, ref_h_grad, ref_c_grad, ref_param_grads = run(None)
        out, h_grad, c_grad, param_grads = run(recompute_method)

        assert c_grad is not None, "condition_emb lost its gradient under full recompute"
        assert ref_c_grad.abs().sum() > 0
        torch.testing.assert_close(out, ref_out)
        torch.testing.assert_close(h_grad, ref_h_grad)
        torch.testing.assert_close(c_grad, ref_c_grad)
        for grad, ref_grad in zip(param_grads, ref_param_grads):
            torch.testing.assert_close(grad, ref_grad)


class TestUnsupportedExtraLayerKwargs:
    """Layers that do not accept an extra kwarg must fail loudly, not drop it."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.config = TransformerConfig(
            num_layers=2, hidden_size=64, num_attention_heads=4, use_cpu_initialization=True
        )
        self.seq_len, self.batch = 8, 2

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _run_block(self, block, recompute_method):
        self.config.recompute_granularity = None if recompute_method is None else 'full'
        self.config.recompute_method = recompute_method
        self.config.recompute_num_layers = 1
        hidden_states = torch.randn(
            self.seq_len, self.batch, self.config.hidden_size, device="cuda", requires_grad=True
        )
        attention_mask = torch.ones(
            (1, 1, self.seq_len, self.seq_len), dtype=torch.bool, device="cuda"
        )
        condition_emb = torch.randn(self.batch, self.config.hidden_size, device="cuda")
        block(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            extra_layer_kwargs={"condition_emb": condition_emb},
        )

    @pytest.mark.parametrize("recompute_method", [None, "uniform"])
    def test_builtin_transformer_layer_rejects_unknown_kwarg(self, recompute_method):
        block = TransformerBlock(
            self.config, get_gpt_layer_with_transformer_engine_spec(), post_process=False
        ).cuda()
        with pytest.raises(TypeError, match="condition_emb"):
            self._run_block(block, recompute_method)

    def test_inner_layer_wrapper_rejects_unknown_kwarg_under_recompute(self):
        block = TransformerBlock(
            self.config, ModuleSpec(module=_InnerLayerWrapper), post_process=False
        ).cuda()
        assert all(isinstance(layer.inner_layer, TransformerLayer) for layer in block.layers)
        assert not any(isinstance(layer, TransformerLayer) for layer in block.layers)
        with pytest.raises(TypeError, match="condition_emb"):
            self._run_block(block, "uniform")


class TestValidateExtraLayerKwargs:

    def test_none_returns_empty_dict(self):
        assert validate_extra_layer_kwargs(None) == {}

    def test_accepts_tensor_and_none_values(self):
        kwargs = {"condition_emb": torch.zeros(1), "optional_emb": None}
        assert validate_extra_layer_kwargs(kwargs) is kwargs

    @pytest.mark.parametrize(
        "kwargs, error",
        [
            ([("condition_emb", None)], TypeError),
            ({1: torch.zeros(1)}, TypeError),
            ({"condition_emb": 1.0}, TypeError),
            ({"condition_emb": [torch.zeros(1)]}, TypeError),
        ],
    )
    def test_rejects_invalid_kwargs(self, kwargs, error):
        with pytest.raises(error):
            validate_extra_layer_kwargs(kwargs)
