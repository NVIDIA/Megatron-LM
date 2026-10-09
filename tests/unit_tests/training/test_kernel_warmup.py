# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from argparse import Namespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.activations import squared_relu
from megatron.core.fusions import fused_cross_entropy
from megatron.core.fusions.fused_bias_geglu import quick_gelu
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training import kernel_warmup
from megatron.training.kernel_warmup import _warmup_training_kernels
from megatron.training.models import GPTModelConfig, HybridModelConfig
from tests.unit_tests.test_utilities import Utils


def configuration(tp_group, dtype=torch.float32):
    config = TransformerConfig(
        num_layers=1,
        num_attention_heads=tp_group.size(),
        tensor_model_parallel_size=tp_group.size(),
        context_parallel_size=1,
        params_dtype=dtype,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl="native",
        activation_func=squared_relu,
        use_te_activation_func=False,
        use_fused_weighted_squared_relu=True,
        activation_func_tanh_clamp_scale=16.0,
        activation_func_tanh_clamp_scale_linear=None,
        bias_activation_fusion=False,
        add_bias_linear=False,
        ffn_hidden_size=64 * tp_group.size(),
        moe_shared_expert_intermediate_size=32 * tp_group.size(),
        hidden_size=32,
        bias_dropout_fusion=True,
        sequence_parallel=False,
        fp32_residual_connection=False,
        hidden_dropout=0.1,
        activation_func_clamp_value=None,
    )
    return config, dict(
        seq_length=8, micro_batch_size=2, padded_vocab_size=128 * tp_group.size(), logit_dtype=dtype
    )


@pytest.fixture(scope="module")
def tp_group():
    Utils.initialize_distributed()
    group = torch.distributed.group.WORLD
    yield group
    Utils.destroy_model_parallel()


@pytest.fixture(autouse=True)
def reset_compilation_cache(monkeypatch):
    # Each case represents a separate training configuration. Accumulating all
    # dtype/layout variants can exceed Dynamo's limit and silently test eager code.
    torch._dynamo.reset()
    monkeypatch.setattr(torch._dynamo.config, "suppress_errors", False)


@pytest.mark.parametrize(
    "dtype,logit_dtype",
    [
        (torch.float32, None),
        (torch.bfloat16, None),
        (torch.float16, None),
        (torch.bfloat16, torch.float32),
        (torch.float16, torch.float32),
    ],
)
@pytest.mark.parametrize("micro_batch_size", [1, 2])
def test_warmup_preserves_rng_and_live_storage(
    tp_group, dtype, logit_dtype, micro_batch_size, monkeypatch
):
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    live = torch.full((1024,), 7.0, pin_memory=True)
    address = live.data_ptr()
    calls = []
    host_memory = getattr(getattr(torch, "accelerator", None), "memory", None)
    original = getattr(host_memory, "empty_host_cache", None)

    def trim():
        calls.append(True)
        original()

    if original is not None:
        monkeypatch.setattr(host_memory, "empty_host_cache", trim)
    layouts = []
    label_layouts = []
    calculate_gradients = fused_cross_entropy.calculate_gradients
    calculate_predicted_logits = fused_cross_entropy.calculate_predicted_logits

    def observe_labels(logits, labels, *args):
        label_layouts.append((labels.shape, labels.stride()))
        return calculate_predicted_logits(logits, labels, *args)

    def observe_gradient(softmax, grad_output, target_mask, masked_target_1d, logits_dtype):
        layouts.append((grad_output.shape, grad_output.stride()))
        return calculate_gradients(
            softmax, grad_output, target_mask, masked_target_1d, logits_dtype
        )

    monkeypatch.setattr(fused_cross_entropy, "calculate_gradients", observe_gradient)
    monkeypatch.setattr(fused_cross_entropy, "calculate_predicted_logits", observe_labels)
    config, inputs = configuration(tp_group, dtype)
    inputs.update(seq_length=16, micro_batch_size=micro_batch_size, logit_dtype=logit_dtype)
    config.context_parallel_size = 2
    _warmup_training_kernels(config, tp_group, **inputs)
    warmup_layouts = layouts.copy()
    layouts.clear()
    warmup_label_layouts = label_layouts.copy()
    label_layouts.clear()
    assert calls == ([True] if original is not None else [])
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    assert live.data_ptr() == address and torch.all(live == 7.0)

    # Reuse the warmed shape with different values and compare distributed CE
    # forward/backward against an unsharded PyTorch reference.
    vocab = 128 * tp_group.size()
    logits_dtype = logit_dtype or dtype
    full = torch.linspace(-1, 1, vocab, device="cuda", dtype=logits_dtype).expand(
        8, micro_batch_size, -1
    )
    local = full[..., tp_group.rank() * 128 : (tp_group.rank() + 1) * 128]
    logits = local.clone().requires_grad_(True)
    labels = torch.arange(8 * micro_batch_size, device="cuda").view(micro_batch_size, 8) % vocab
    # Exercise the actual model loss wrapper, rather than assuming the layout
    # of the gradient that it supplies to the fused CE backward.
    model = Namespace(
        tp_group=tp_group,
        vocab_parallel_cross_entropy=fused_cross_entropy.fused_vocab_parallel_cross_entropy,
    )
    loss = LanguageModule.compute_language_model_loss(model, labels, logits)
    loss_gradient = torch.ones(loss.shape, device="cuda", dtype=loss.dtype)
    grad = torch.autograd.grad(loss, logits, grad_outputs=loss_gradient)[0]
    assert warmup_layouts == layouts * 2
    assert warmup_label_layouts == label_layouts * 2
    assert layouts == [(torch.Size([8, micro_batch_size]), (1, 8))]
    reference = full.float().contiguous().requires_grad_(True)
    expected = (
        torch.nn.functional.cross_entropy(
            reference.flatten(0, 1), labels.T.contiguous().flatten(), reduction="none"
        )
        .view(8, micro_batch_size)
        .T.contiguous()
    )
    expected_grad = torch.autograd.grad(expected, reference, torch.ones_like(expected))[0]
    expected_grad = expected_grad[..., tp_group.rank() * 128 : (tp_group.rank() + 1) * 128]
    torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(grad, expected_grad.to(logits_dtype), rtol=1e-5, atol=1e-5)


def test_warmup_rejects_invalid_shape(tp_group):
    config, inputs = configuration(tp_group)
    inputs["seq_length"] = 0
    with pytest.raises(ValueError, match="positive"):
        _warmup_training_kernels(config, tp_group, **inputs)
    inputs["seq_length"] = 8
    inputs["padded_vocab_size"] = None
    with pytest.raises(ValueError, match="vocabulary must be resolved"):
        _warmup_training_kernels(config, tp_group, **inputs)
    if tp_group.size() > 1:
        inputs["padded_vocab_size"] = 129
        with pytest.raises(ValueError, match="divisible"):
            _warmup_training_kernels(config, tp_group, **inputs)


@pytest.mark.parametrize("activation", ["gelu", "swiglu", "situ_glu"])
def test_other_configurations_and_later_compilation(tp_group, activation, monkeypatch):
    config, inputs = configuration(tp_group, torch.bfloat16)
    config.cross_entropy_loss_fusion = False
    config.activation_func = F.gelu if activation == "gelu" else F.silu
    config.activation_func_tanh_clamp_scale = None
    config.gated_linear_unit = activation in {"swiglu", "situ_glu"}
    config.bias_activation_fusion = True
    if activation == "situ_glu":
        config.activation_func_tanh_clamp_scale = 16.0
        config.activation_func_tanh_clamp_scale_linear = 8.0
    config.add_bias_linear = True
    config.sequence_parallel = True
    config.fp32_residual_connection = True

    def unused_ce(*unused):
        pytest.fail("Disabled CE must not be warmed up")

    monkeypatch.setattr(kernel_warmup, "_warmup_cross_entropy", unused_ce)
    _warmup_training_kernels(config, tp_group, **inputs)

    # A new specialization must still compile after startup workers shut down.
    @torch.compile
    def later(x):
        return torch.cos(x) * 3.25 + 1

    x = torch.linspace(-1, 1, 97, device="cuda")
    torch.testing.assert_close(later(x), torch.cos(x) * 3.25 + 1)


@pytest.mark.parametrize("activation", ["te", "quick_geglu", "geglu", "unfused"])
def test_does_not_warmup_unused_activation(tp_group, activation, monkeypatch):
    config, inputs = configuration(tp_group)
    config.cross_entropy_loss_fusion = False
    config.bias_dropout_fusion = False
    config.activation_func = F.gelu
    config.activation_func_tanh_clamp_scale = None
    config.use_te_activation_func = activation == "te"
    config.bias_activation_fusion = activation not in {"te", "unfused"}
    config.gated_linear_unit = activation in {"quick_geglu", "geglu"}
    config.add_bias_linear = True
    if activation == "quick_geglu":
        config.activation_func = quick_gelu

    def unused_activation(*unused):
        pytest.fail("Must not warm up a native activation that the model will not use")

    monkeypatch.setattr(kernel_warmup, "_warmup_activation", unused_activation)
    _warmup_training_kernels(config, tp_group, **inputs)


@pytest.mark.parametrize("add_bias", [False, True])
@pytest.mark.parametrize("fp8_input_store", [False, True])
def test_config_activation_storage_reuses_warmed_backward(tp_group, add_bias, fp8_input_store):
    config, inputs = configuration(tp_group, torch.bfloat16)
    config.cross_entropy_loss_fusion = False
    config.bias_dropout_fusion = False
    config.activation_func = F.silu
    config.gated_linear_unit = True
    config.bias_activation_fusion = True
    config.activation_func_tanh_clamp_scale = None
    config.activation_func_fp8_input_store = fp8_input_store
    config.add_bias_linear = add_bias
    config.moe_shared_expert_intermediate_size = None
    # This setting is available only on TransformerConfig, not the generated CLI.
    _warmup_training_kernels(config, tp_group, **inputs)
    graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]

    x = torch.linspace(-1, 1, 8 * 2 * 128, device="cuda", dtype=torch.bfloat16)
    x = x.reshape(8, 2, 128).requires_grad_()
    bias = torch.zeros(128, device="cuda", dtype=x.dtype, requires_grad=True) if add_bias else None
    saved_dtypes = []

    def pack(value):
        saved_dtypes.append(value.dtype)
        return value

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        output = bias_swiglu_impl(x, bias, config.activation_func_fp8_input_store)
        gradients = torch.autograd.grad(
            output, (x, bias) if add_bias else (x,), torch.ones_like(output)
        )
    assert (torch.float8_e4m3fn in saved_dtypes) == fp8_input_store
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    # With a False warmup and True model setting the saved input has different
    # requires_grad metadata, which triggers a new Dynamo backward specialization.
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == graphs


@pytest.fixture
def pretrain_until_warmup(monkeypatch):
    from megatron.training import training

    class WarmupReached(Exception):
        pass

    monkeypatch.setattr(training, "_STARTUP_TIMESTAMPS", {})
    monkeypatch.setattr(training.ft_integration, "setup", lambda: None)
    monkeypatch.setattr(training, "initialize_megatron", lambda **kwargs: None)
    monkeypatch.setattr(training, "set_run_config", lambda cfg: None)
    monkeypatch.setattr(training, "get_timers", lambda: None)
    monkeypatch.setattr(training, "set_jit_fusion_options", lambda **kwargs: None)

    def invoke(args, model_config, tp_group):
        monkeypatch.setattr(training, "get_args", lambda: args)
        calls = []

        def observe(config, group, **inputs):
            calls.append((config, group, inputs))
            raise WarmupReached

        monkeypatch.setattr(kernel_warmup, "_warmup_training_kernels", observe)
        cfg = Namespace(
            model=model_config,
            train=Namespace(micro_batch_size=9),
            logger=Namespace(log_progress=False),
        )
        with pytest.raises(WarmupReached):
            training.pretrain(cfg, None, None, None, pg_collection=Namespace(tp=tp_group))
        return calls[0]

    return invoke


@pytest.mark.parametrize("model_cls", [GPTModelConfig, HybridModelConfig])
@pytest.mark.parametrize("pad_vocab", [False, True])
def test_pretrain_passes_model_config_and_runtime_batch(
    tp_group, model_cls, pad_vocab, pretrain_until_warmup
):
    config, _ = configuration(tp_group, torch.bfloat16)
    config.context_parallel_size = 2
    config.activation_func_fp8_input_store = True
    model_config = model_cls(
        transformer=config,
        seq_length=4096,
        vocab_size=129 if pad_vocab else 128 * tp_group.size(),
        should_pad_vocab=pad_vocab,
        logit_dtype=torch.float32,
    )
    # Batch dimensions follow the runtime args; model settings follow its config.
    args = Namespace(
        seq_length=16,
        micro_batch_size=1,
        fine_grained_activation_offloading=False,
        tensor_model_parallel_size=tp_group.size(),
        context_parallel_size=1,
        padded_vocab_size=8,
        logit_dtype=torch.float16,
        params_dtype=torch.float16,
        cross_entropy_loss_fusion=False,
    )
    passed_config, passed_group, inputs = pretrain_until_warmup(args, model_config, tp_group)
    multiple = 128 * tp_group.size()
    padded_vocab = ((129 + multiple - 1) // multiple) * multiple if pad_vocab else multiple
    assert passed_config is config
    assert passed_group is tp_group
    assert inputs == dict(
        seq_length=16, micro_batch_size=1, padded_vocab_size=padded_vocab, logit_dtype=torch.float32
    )


def test_pretrain_legacy_config_fallback(tp_group, pretrain_until_warmup, monkeypatch):
    from megatron.training import argument_utils

    config, inputs = configuration(tp_group)
    args = Namespace(
        **inputs,
        fine_grained_activation_offloading=False,
        tensor_model_parallel_size=tp_group.size(),
    )
    calls = []

    def from_args(value):
        calls.append(value)
        return config

    monkeypatch.setattr(argument_utils, "core_transformer_config_from_args", from_args)
    passed_config, passed_group, passed_inputs = pretrain_until_warmup(args, None, tp_group)
    assert calls == [args]
    assert passed_config is config
    assert passed_group is tp_group
    assert passed_inputs == inputs


@pytest.mark.parametrize("micro_batch_size", [None, 0])
def test_rejects_invalid_microbatch(tp_group, micro_batch_size):
    config, inputs = configuration(tp_group)
    inputs["micro_batch_size"] = micro_batch_size
    with pytest.raises(ValueError, match="positive"):
        _warmup_training_kernels(config, tp_group, **inputs)
