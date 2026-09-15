# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import (
    Float16Module,
    MegatronModule,
    is_first_microbatch_tracked,
    mark_keep_in_fp32,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

# Seed for the GB200 unit-test lane: launch this module on GB200 hardware
# (4 GPUs/node) in CI. Extend coverage by adding this marker to other tests.
pytestmark = pytest.mark.launch_on_gb200

DEVICE_CAPABILITY = None
if torch.cuda.is_available():
    DEVICE_CAPABILITY = torch.cuda.get_device_capability()


class DummyModule(MegatronModule):
    # def __init__(self, config: TransformerConfig, share_embeddings_and_output_weights=True):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)

        self.linear = torch.nn.modules.Linear(in_features=2, out_features=1)

    def forward(self, x):
        return self.linear(x)


class TestMegatronModule:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=2, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True
        )
        self.megatron_module = DummyModule(config=transformer_config).cuda()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_megatron_module(self):
        megatron_module = self.megatron_module
        assert megatron_module
        assert megatron_module.config.hidden_size == 12
        assert megatron_module.config.ffn_hidden_size == 48
        assert megatron_module.linear.weight.dtype == torch.float32

        x = torch.ones((2, 2)).cuda()
        assert megatron_module(x).dtype == torch.float32

        # TODO: test bad configs actually fail
        # failed_module = megatron_module
        # failed_module.fp16 = True
        # failed_module.bf16 = True


class _FirstMicrobatchModule(torch.nn.Module):
    """Stand-in for a TE module that exposes the is_first_microbatch flag."""

    def __init__(self):
        super().__init__()
        self.is_first_microbatch = False


class DummyQuantModule(MegatronModule):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.child = _FirstMicrobatchModule()

    def forward(self, x):
        return x


class TestSetIsFirstMicrobatch:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _build_module(self, **overrides):
        config = TransformerConfig(
            num_layers=2, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True
        )
        for key, value in overrides.items():
            setattr(config, key, value)
        return DummyQuantModule(config=config)

    def test_quant_recipe_sets_flag(self):
        # quant_recipe alone must enable the flag, even with fp8/fp4/kitchen off.
        module = self._build_module(quant_recipe=object())
        assert module.config.fp8 is None
        assert module.config.fp4 is None
        assert getattr(module.config, 'use_kitchen', False) is False
        assert module.config.quant_recipe is not None
        assert module.child.is_first_microbatch is False

        module.set_is_first_microbatch()
        assert module.child.is_first_microbatch is True

    def test_no_quant_leaves_flag_untouched(self):
        # With no quantization mode configured the flag must not be touched.
        module = self._build_module()
        assert module.config.fp8 is None
        assert module.config.fp4 is None
        assert getattr(module.config, 'use_kitchen', False) is False
        assert module.config.quant_recipe is None
        assert module.child.is_first_microbatch is False

        module.set_is_first_microbatch()
        assert module.child.is_first_microbatch is False


class _QuantizedExecutionModule(torch.nn.Module):
    """Stand-in for a TE module answering whether it will execute quantized.

    ``executes_quantized=None`` mirrors a module with no per-module recipe, which defers to
    the ambient autocast.
    """

    def __init__(self, executes_quantized=None):
        super().__init__()
        self._executes_quantized = executes_quantized

    def will_execute_quantized(self, is_context_quantized: bool) -> bool:
        if self._executes_quantized is None:
            return is_context_quantized
        return self._executes_quantized


class TestIsFirstMicrobatchTracked:
    """quant_recipe decides quantization per module, so the predicate must ask the module."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _config(self, **overrides):
        config = TransformerConfig(
            num_layers=2, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True
        )
        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    def test_unquantized_config_is_untracked(self):
        assert is_first_microbatch_tracked(self._config()) is False

    @pytest.mark.parametrize(
        ('field', 'value'), [('fp8', 'hybrid'), ('fp4', 'e2m1'), ('quant_recipe', object())]
    )
    def test_model_wide_answer_without_module(self, field, value):
        # set_is_first_microbatch has no module in hand and must keep its old answer.
        assert is_first_microbatch_tracked(self._config(**{field: value})) is True

    def test_kitchen_is_tracked_even_with_module(self):
        # Kitchen sits outside TE's autocast state, so there is no per-module answer.
        config = self._config(use_kitchen=True)
        module = _QuantizedExecutionModule(executes_quantized=False)
        assert is_first_microbatch_tracked(config, module, False) is True

    def test_recipe_quantizing_unquantized_layer_is_tracked(self):
        config = self._config(quant_recipe=object())
        module = _QuantizedExecutionModule(executes_quantized=True)
        assert is_first_microbatch_tracked(config, module, False) is True

    def test_recipe_forcing_high_precision_under_fp8_is_untracked(self):
        # The recipe overrides the fp8 autocast for this layer, so it has no quantized weight
        # cache and the flag would only change its wgrad accumulation.
        config = self._config(fp8='hybrid', quant_recipe=object())
        module = _QuantizedExecutionModule(executes_quantized=False)
        assert is_first_microbatch_tracked(config, module, True) is False

    def test_module_without_recipe_follows_autocast(self):
        config = self._config(fp8='hybrid')
        module = _QuantizedExecutionModule()
        assert is_first_microbatch_tracked(config, module, True) is True
        assert is_first_microbatch_tracked(config, module, False) is False

    def test_module_lacking_hook_falls_back_to_config(self):
        config = self._config(fp8='hybrid')
        assert is_first_microbatch_tracked(config, torch.nn.Linear(2, 2), False) is True


class TestFloat16Module:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = TransformerConfig(
            num_layers=2, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True
        )
        self.megatron_module = DummyModule(config=self.transformer_config).cuda()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_fp16_module(self):
        transformer_config = self.transformer_config
        megatron_module = self.megatron_module
        transformer_config.fp16 = True
        fp16_module = Float16Module(config=transformer_config, module=megatron_module)

        assert fp16_module
        assert fp16_module.config.hidden_size == 12
        assert fp16_module.config.ffn_hidden_size == 48
        assert fp16_module.module.linear.weight.dtype == torch.float16

        x = torch.ones((2, 2)).cuda()
        # inputs are converted to fp16 then outputs are converted to fp32
        assert fp16_module(x).dtype == torch.float32

    pytest.mark.skipif(
        not DEVICE_CAPABILITY or DEVICE_CAPABILITY[0] < 8,
        reason='bfloat16 is not supported on this device',
    )

    def test_bf16_module(self):
        transformer_config = self.transformer_config
        megatron_module = self.megatron_module
        transformer_config.bf16 = True
        bf16_module = Float16Module(config=transformer_config, module=megatron_module)

        assert bf16_module
        assert bf16_module.config.hidden_size == 12
        assert bf16_module.config.ffn_hidden_size == 48
        assert bf16_module.module.linear.weight.dtype == torch.bfloat16

        x = torch.ones((2, 2)).cuda()
        # inputs are converted to bf16 then outputs are converted to fp32
        assert bf16_module(x).dtype == torch.float32

    @pytest.mark.parametrize(
        ('precision', 'dtype'), [('fp16', torch.float16), ('bf16', torch.bfloat16)]
    )
    def test_keep_in_fp32_params(self, precision, dtype):
        transformer_config = self.transformer_config
        megatron_module = self.megatron_module
        megatron_module.fp32_param = mark_keep_in_fp32(
            torch.nn.Parameter(torch.zeros(4, dtype=torch.float32, device='cuda'))
        )
        setattr(transformer_config, precision, True)
        float16_module = Float16Module(config=transformer_config, module=megatron_module)

        assert float16_module.module.linear.weight.dtype == dtype
        assert float16_module.module.fp32_param.dtype == torch.float32
