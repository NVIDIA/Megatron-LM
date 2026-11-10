# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for TELMHeadColumnParallelLinear and is_mxfp8_output_proj_active."""

import functools
from types import SimpleNamespace

import pytest
import torch

from megatron.core import tensor_parallel
from megatron.core.extensions.transformer_engine import HAVE_TE, TELMHeadColumnParallelLinear
from megatron.core.fp8_utils import is_mxfp8_output_proj_active
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel, _cross_entropy_mxfp8_grad_format
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

_IS_BLACKWELL = torch.cuda.is_available() and (torch.cuda.get_device_properties(0).major >= 10)


def _fake_config(*, fp8_output_proj=True, fp8=True, fp8_recipe="mxfp8"):
    """SimpleNamespace stub of TransformerConfig used by is_mxfp8_output_proj_active."""
    return SimpleNamespace(fp8_output_proj=fp8_output_proj, fp8=fp8, fp8_recipe=fp8_recipe)


class TestIsMxfp8OutputProjActive:
    """Pure-Python tests for the gating helper."""

    def test_returns_true_for_complete_mxfp8_config(self):
        if not HAVE_TE:
            pytest.skip("TE not installed; helper always returns False")
        assert is_mxfp8_output_proj_active(_fake_config()) is True

    def test_returns_false_when_flag_off(self):
        assert is_mxfp8_output_proj_active(_fake_config(fp8_output_proj=False)) is False

    def test_returns_false_when_fp8_off(self):
        assert is_mxfp8_output_proj_active(_fake_config(fp8=False)) is False

    def test_returns_false_for_non_mxfp8_recipe(self):
        assert is_mxfp8_output_proj_active(_fake_config(fp8_recipe="tensorwise")) is False
        assert is_mxfp8_output_proj_active(_fake_config(fp8_recipe="delayed")) is False
        assert is_mxfp8_output_proj_active(_fake_config(fp8_recipe="blockwise")) is False

    def test_accepts_enum_style_recipe(self):
        if not HAVE_TE:
            pytest.skip("TE not installed; helper always returns False")
        enum_like = SimpleNamespace(value="mxfp8")
        assert is_mxfp8_output_proj_active(_fake_config(fp8_recipe=enum_like)) is True

    def test_returns_false_when_attributes_missing(self):
        assert is_mxfp8_output_proj_active(SimpleNamespace()) is False


@pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine not installed")
class TestTELMHeadColumnParallelLinearRejections:
    """Constructor validation checks.

    Each invalid kwarg triggers a raise before super().__init__() runs, so these
    tests do not require GPU initialization of the TE parent.
    """

    def _kwargs(self, **overrides):
        kwargs = dict(
            input_size=8, output_size=16, config=_fake_config(), init_method=lambda w: None
        )
        kwargs.update(overrides)
        return kwargs

    def test_rejects_inactive_config(self):
        with pytest.raises(RuntimeError, match="fp8_output_proj=True"):
            TELMHeadColumnParallelLinear(**self._kwargs(config=_fake_config(fp8=False)))

    def test_rejects_keep_master_weight_for_test(self):
        with pytest.raises(ValueError, match="keep_master_weight_for_test"):
            TELMHeadColumnParallelLinear(**self._kwargs(keep_master_weight_for_test=True))

    def test_rejects_skip_weight_param_allocation(self):
        with pytest.raises(ValueError, match="skip_weight_param_allocation"):
            TELMHeadColumnParallelLinear(**self._kwargs(skip_weight_param_allocation=True))

    def test_rejects_embedding_activation_buffer(self):
        with pytest.raises(ValueError, match="defer_embedding_wgrad_compute"):
            TELMHeadColumnParallelLinear(**self._kwargs(embedding_activation_buffer=[]))

    def test_rejects_grad_output_buffer(self):
        with pytest.raises(ValueError, match="defer_embedding_wgrad_compute"):
            TELMHeadColumnParallelLinear(**self._kwargs(grad_output_buffer=[]))

    def test_rejects_disable_grad_reduce(self):
        with pytest.raises(ValueError, match="disable_grad_reduce"):
            TELMHeadColumnParallelLinear(**self._kwargs(disable_grad_reduce=True))

    def test_rejects_output_dtype(self):
        with pytest.raises(ValueError, match="output_dtype"):
            TELMHeadColumnParallelLinear(**self._kwargs(output_dtype=torch.float32))


class TestGPTModelOutputLayerSelection:
    """Verify GPTModel picks the right output-layer class based on config."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_default_uses_column_parallel_linear(self):
        """Without fp8_output_proj, the LM head is the plain ColumnParallelLinear."""
        config = TransformerConfig(
            num_layers=2, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True
        )
        model = GPTModel(
            config=config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=100,
            max_sequence_length=4,
        )
        assert isinstance(model.output_layer, tensor_parallel.ColumnParallelLinear)
        assert not isinstance(model.output_layer, TELMHeadColumnParallelLinear)

    @pytest.mark.internal
    def test_logit_dtype_is_forwarded_to_output_layer(self):
        config = TransformerConfig(
            num_layers=2, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True
        )
        model = GPTModel(
            config=config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=100,
            max_sequence_length=4,
            logit_dtype=torch.float32,
        )
        assert model.output_layer.output_dtype == torch.float32

    @pytest.mark.internal
    @pytest.mark.skipif(
        not _IS_BLACKWELL, reason="MXFP8 output projection requires Blackwell (SM >= 10)"
    )
    def test_mxfp8_active_uses_te_lm_head(self):
        """With fp8_output_proj=True under mxfp8, the LM head is TELMHeadColumnParallelLinear."""
        config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            use_cpu_initialization=False,
            params_dtype=torch.bfloat16,
            bf16=True,
            fp8="hybrid",
            fp8_recipe="mxfp8",
            fp8_output_proj=True,
        )
        model = GPTModel(
            config=config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=128,
            max_sequence_length=8,
        )
        assert isinstance(model.output_layer, TELMHeadColumnParallelLinear)


def _te_ce_supports_grad_quantizer() -> bool:
    if not HAVE_TE:
        return False
    from megatron.core.extensions import transformer_engine as te_ext

    return getattr(te_ext, "_TE_FUSED_PARALLEL_CE_GRAD_QUANTIZER", False)


def _ce_mxfp8_grad_config(**overrides):
    """Small MXFP8 GPT config with the MXFP8 LM head and TE's fused cross entropy."""
    kwargs = dict(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        use_cpu_initialization=False,
        params_dtype=torch.bfloat16,
        bf16=True,
        fp8="e4m3",
        fp8_recipe="mxfp8",
        fp8_output_proj=True,
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl="te",
        cross_entropy_mxfp8_grad=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def _hybrid_recipe_attrs():
    """fp8_recipe_attrs that switch the recipe to HYBRID (E5M2 gradients) after fp8='e4m3'."""
    from transformer_engine.common.recipe import Format

    return {"fp8_format": Format.HYBRID}


# (fp8, fp8_recipe_attrs factory, gradient format the LM head's recipe expects)
_GRAD_FORMAT_CASES = [
    pytest.param("e4m3", None, "e4m3", id="e4m3"),
    pytest.param("hybrid", None, "e5m2", id="hybrid"),
    pytest.param("e4m3", _hybrid_recipe_attrs, "e5m2", id="e4m3_with_hybrid_recipe_attrs"),
]


@pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine not installed")
class TestCrossEntropyMxfp8GradFormat:
    """The gradient format comes from the LM head's effective recipe."""

    @pytest.mark.parametrize("fp8, recipe_attrs, expected_format", _GRAD_FORMAT_CASES)
    def test_format_from_effective_recipe(self, fp8, recipe_attrs, expected_format):
        config = _ce_mxfp8_grad_config(
            fp8=fp8, fp8_recipe_attrs=recipe_attrs() if recipe_attrs else None
        )
        assert _cross_entropy_mxfp8_grad_format(config) == expected_format

    @pytest.mark.parametrize("backward_override", ["high_precision", "dequantized"])
    def test_rejects_high_precision_backward(self, backward_override):
        config = _ce_mxfp8_grad_config(fp8_recipe_attrs={"backward_override": backward_override})
        with pytest.raises(ValueError, match="backward_override"):
            _cross_entropy_mxfp8_grad_format(config)

    def test_rejects_unsupported_format(self):
        from transformer_engine.common.recipe import Format

        config = _ce_mxfp8_grad_config(fp8_recipe_attrs={"fp8_format": Format.E5M2})
        with pytest.raises(ValueError, match="E4M3 and HYBRID"):
            _cross_entropy_mxfp8_grad_format(config)


class TestCrossEntropyMxfp8GradConfig:
    """cross_entropy_mxfp8_grad validation."""

    def test_requires_mxfp8_output_proj(self):
        with pytest.raises(ValueError, match="requires fp8_output_proj"):
            _ce_mxfp8_grad_config(fp8_output_proj=False)

    @pytest.mark.parametrize(
        "overrides",
        [dict(cross_entropy_loss_fusion=False), dict(cross_entropy_fusion_impl="native")],
        ids=["unfused", "native"],
    )
    def test_requires_te_fused_cross_entropy(self, overrides):
        with pytest.raises(ValueError, match="cross_entropy_fusion_impl='te'"):
            _ce_mxfp8_grad_config(**overrides)

    def test_rejects_mup(self):
        with pytest.raises(ValueError, match="use_mup"):
            _ce_mxfp8_grad_config(use_mup=True)


@pytest.mark.internal
@pytest.mark.launch_on_gb200
@pytest.mark.skipif(
    not _IS_BLACKWELL, reason="MXFP8 output projection requires Blackwell (SM >= 10)"
)
@pytest.mark.skipif(
    not _te_ce_supports_grad_quantizer(),
    reason="Transformer Engine's parallel_cross_entropy does not accept grad_quantizer",
)
class TestCrossEntropyMxfp8Grad:
    """The LM head gets the MXFP8 gradient from the loss and computes the same gradients."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @staticmethod
    def _build(fp8, seq_len, cross_entropy_mxfp8_grad, mtp_num_layers=None, recipe_attrs=None):
        model_parallel_cuda_manual_seed(123)
        torch.manual_seed(123)
        config = _ce_mxfp8_grad_config(
            fp8=fp8,
            cross_entropy_mxfp8_grad=cross_entropy_mxfp8_grad,
            mtp_num_layers=mtp_num_layers,
            fp8_recipe_attrs=recipe_attrs() if recipe_attrs else None,
        )
        layer_spec = get_gpt_layer_with_transformer_engine_spec()
        mtp_block_spec = (
            get_gpt_mtp_block_spec(config, layer_spec, use_transformer_engine=True)
            if mtp_num_layers
            else None
        )
        return GPTModel(
            config=config,
            transformer_layer_spec=layer_spec,
            vocab_size=256,
            max_sequence_length=seq_len,
            mtp_block_spec=mtp_block_spec,
        ).cuda()

    @staticmethod
    def _run(model, input_ids, labels):
        """Loss, LM-head weight gradient, LM-head input gradient, type of the logits gradient."""
        seen = {}

        seen["logits_grad_types"] = []

        def hook(module, args, output):
            args[0].register_hook(lambda grad: seen.__setitem__("dgrad", grad.detach().clone()))
            output[0].register_hook(lambda grad: seen["logits_grad_types"].append(type(grad)))

        handle = model.output_layer.register_forward_hook(hook)
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device="cuda").repeat(input_ids.shape[0], 1)
        attention_mask = torch.ones(
            (input_ids.shape[0], 1, seq_len, seq_len), dtype=bool, device="cuda"
        )
        loss = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss.sum().backward()
        handle.remove()
        return (
            loss.detach(),
            model.output_layer.weight.grad.detach().clone(),
            seen["dgrad"],
            seen["logits_grad_types"],
        )

    @pytest.mark.parametrize("fp8, recipe_attrs, expected_format", _GRAD_FORMAT_CASES)
    @pytest.mark.parametrize(
        "seq_len", [64, 48], ids=["written_by_loss_kernel", "quantized_after_loss_kernel"]
    )
    def test_matches_lm_head_quantization(self, fp8, recipe_attrs, expected_format, seq_len):
        from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor

        ref_model = self._build(
            fp8, seq_len, cross_entropy_mxfp8_grad=False, recipe_attrs=recipe_attrs
        )
        model = self._build(fp8, seq_len, cross_entropy_mxfp8_grad=True, recipe_attrs=recipe_attrs)
        model.load_state_dict(ref_model.state_dict())
        assert model.vocab_parallel_cross_entropy.keywords["grad_fp8_format"] == expected_format

        input_ids = torch.randint(0, 256, (2, seq_len), device="cuda")
        labels = torch.randint(0, 256, (2, seq_len), device="cuda")
        ref = self._run(ref_model, input_ids, labels)
        out = self._run(model, input_ids, labels)

        assert ref[3] == [torch.Tensor]
        assert len(out[3]) == 1 and issubclass(out[3][0], MXFP8Tensor)
        for name, a, b in zip(("loss", "weight grad", "input grad"), out[:3], ref[:3]):
            assert torch.equal(a, b), name

    def test_mtp_heads_get_mxfp8_grad(self):
        """The MTP heads share the LM head and its loss, so they get the MXFP8 gradient too."""
        from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor

        seq_len = 64
        ref_model = self._build("e4m3", seq_len, cross_entropy_mxfp8_grad=False, mtp_num_layers=1)
        model = self._build("e4m3", seq_len, cross_entropy_mxfp8_grad=True, mtp_num_layers=1)
        model.load_state_dict(ref_model.state_dict())

        input_ids = torch.randint(0, 256, (2, seq_len), device="cuda")
        labels = torch.randint(0, 256, (2, seq_len), device="cuda")
        ref = self._run(ref_model, input_ids, labels)
        out = self._run(model, input_ids, labels)

        # One logits gradient for the main head and one for the MTP head.
        assert ref[3] == [torch.Tensor, torch.Tensor]
        assert len(out[3]) == 2 and all(issubclass(t, MXFP8Tensor) for t in out[3])
        for name, a, b in zip(("loss", "weight grad"), out[:2], ref[:2]):
            assert torch.equal(a, b), name


class TestCrossEntropyMxfp8GradBinding:
    """GPTModel binds the MXFP8 gradient format to its cross entropy only with the MXFP8 head."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @staticmethod
    def _build(**config_overrides):
        return GPTModel(
            config=_ce_mxfp8_grad_config(**config_overrides),
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=256,
            max_sequence_length=64,
        )

    @pytest.mark.internal
    @pytest.mark.launch_on_gb200
    @pytest.mark.skipif(
        not _IS_BLACKWELL, reason="MXFP8 output projection requires Blackwell (SM >= 10)"
    )
    @pytest.mark.skipif(
        not _te_ce_supports_grad_quantizer(),
        reason="Transformer Engine's parallel_cross_entropy does not accept grad_quantizer",
    )
    @pytest.mark.parametrize("fp8, recipe_attrs, expected_format", _GRAD_FORMAT_CASES)
    def test_binds_format_with_mxfp8_head(self, fp8, recipe_attrs, expected_format):
        model = self._build(fp8=fp8, fp8_recipe_attrs=recipe_attrs() if recipe_attrs else None)
        assert isinstance(model.output_layer, TELMHeadColumnParallelLinear)
        assert isinstance(model.vocab_parallel_cross_entropy, functools.partial)
        assert model.vocab_parallel_cross_entropy.keywords["grad_fp8_format"] == expected_format

    @pytest.mark.internal
    @pytest.mark.launch_on_gb200
    @pytest.mark.skipif(
        not _IS_BLACKWELL, reason="MXFP8 output projection requires Blackwell (SM >= 10)"
    )
    def test_rejects_high_precision_backward(self):
        with pytest.raises(ValueError, match="backward_override='high_precision'"):
            self._build(fp8_recipe_attrs={"backward_override": "high_precision"})

    @pytest.mark.internal
    @pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine not installed")
    def test_no_binding_without_mxfp8_head(self, monkeypatch):
        """With a BF16 head, the cross entropy keeps returning a BF16 gradient."""
        from megatron.core.models.gpt import gpt_model

        monkeypatch.setattr(gpt_model, "is_mxfp8_output_proj_active", lambda config: False)
        model = self._build()
        assert not isinstance(model.output_layer, TELMHeadColumnParallelLinear)
        # The selected cross entropy may itself be a partial; it must not carry the format.
        assert "grad_fp8_format" not in getattr(model.vocab_parallel_cross_entropy, "keywords", {})

    @pytest.mark.internal
    @pytest.mark.launch_on_gb200
    @pytest.mark.skipif(
        not _IS_BLACKWELL, reason="MXFP8 output projection requires Blackwell (SM >= 10)"
    )
    def test_rejects_cross_entropy_without_mxfp8_grad(self, monkeypatch):
        from megatron.core.models.common.language_module import language_module

        def cross_entropy(logits, labels, tp_group):
            raise AssertionError("not called")

        monkeypatch.setattr(language_module, "backend_slot", lambda **kwargs: cross_entropy)
        with pytest.raises(ValueError, match="without an MXFP8 gradient"):
            self._build()


class TestCrossEntropyMxfp8GradRejectedByOtherModels:
    """Models that build a BF16 LM head reject cross_entropy_mxfp8_grad."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @staticmethod
    def _hybrid(config):
        from megatron.core.models.hybrid.hybrid_model import HybridModel

        return HybridModel(
            config=config, hybrid_stack_spec=None, vocab_size=256, max_sequence_length=64
        )

    @staticmethod
    def _bert(config):
        from megatron.core.models.bert.bert_model import BertModel

        return BertModel(
            config=config,
            num_tokentypes=0,
            transformer_layer_spec=None,
            vocab_size=256,
            max_sequence_length=64,
        )

    @staticmethod
    def _t5(config):
        from megatron.core.models.T5.t5_model import T5Model

        return T5Model(
            config=config,
            encoder_config=config,
            transformer_encoder_layer_spec=None,
            transformer_decoder_layer_spec=None,
            vocab_size=256,
            max_sequence_length=64,
        )

    @pytest.mark.parametrize("build", ["_hybrid", "_bert", "_t5"])
    def test_rejects_flag(self, build):
        with pytest.raises(ValueError, match="requires GPTModel's MXFP8 LM head"):
            getattr(self, build)(_ce_mxfp8_grad_config())
