# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Warm up configured training kernels before model and offload storage is live."""

import gc
import logging
import sys
from argparse import Namespace

import torch
import torch.nn.functional as F

from megatron.core.activations import squared_relu
from megatron.core.transformer.transformer_config import TransformerConfig

logger = logging.getLogger(__name__)


def _warmup_cross_entropy(
    config: TransformerConfig,
    tp_group: torch.distributed.ProcessGroup,
    sequence: int,
    micro_batch_size: int,
    vocab: int,
    logit_dtype: torch.dtype | None,
) -> None:
    from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy

    if vocab <= 0:
        raise ValueError("CE vocabulary must be positive")
    if vocab % tp_group.size():
        raise ValueError("CE vocabulary must be divisible by tensor parallel size")
    for _ in range(2):
        logits = torch.zeros(
            (sequence, micro_batch_size, vocab // tp_group.size()),
            device="cuda",
            dtype=logit_dtype or config.params_dtype,
            requires_grad=True,
        )
        # Match the model's batch-major labels, including singleton-dimension
        # strides: contiguous() need not copy a transposed microbatch of one.
        labels = torch.zeros((micro_batch_size, sequence), device="cuda", dtype=torch.long)
        labels = labels.transpose(0, 1).contiguous()
        loss = fused_vocab_parallel_cross_entropy(logits, labels, tp_group)
        # Match LanguageModule.compute_language_model_loss, including the [b, s]
        # loss layout: its backward transpose gives CE a [1, s] gradient stride.
        loss = loss.transpose(0, 1).contiguous()
        # The loss reduction supplies a contiguous batch-major gradient.
        # ones_like(loss) can preserve a different stride when microbatch == 1.
        loss_gradient = torch.ones(loss.shape, device="cuda", dtype=loss.dtype)
        gradient = torch.autograd.grad(loss, logits, grad_outputs=loss_gradient)[0]
        del gradient, loss_gradient, loss, labels, logits


def _warmup_activation(
    config: TransformerConfig,
    tp_size: int,
    width: int,
    sequence: int,
    micro_batch_size: int,
    *,
    flatten: bool = False,
) -> None:
    from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
    from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl
    from megatron.core.fusions.fused_weighted_squared_relu import weighted_squared_relu_impl

    leading_shape = (sequence, micro_batch_size)
    if flatten:
        leading_shape = (sequence * micro_batch_size,)
    for _ in range(2):
        x = torch.zeros(
            (*leading_shape, width // tp_size * (2 if config.gated_linear_unit else 1)),
            device="cuda",
            dtype=config.params_dtype,
            requires_grad=True,
        )
        bias = (
            torch.zeros(x.shape[-1], device="cuda", dtype=x.dtype, requires_grad=True)
            if config.add_bias_linear and config.activation_func != squared_relu
            else None
        )
        if config.activation_func == squared_relu:
            output = weighted_squared_relu_impl(x, None, config.activation_func_tanh_clamp_scale)
        elif config.activation_func == F.silu:
            output = bias_swiglu_impl(
                x,
                bias,
                config.activation_func_fp8_input_store,
                clamp_value=config.activation_func_clamp_value,
                gate_clamp_scale=config.activation_func_tanh_clamp_scale,
                linear_clamp_scale=config.activation_func_tanh_clamp_scale_linear,
            )
        else:
            output = bias_gelu_impl(x, bias)
        inputs = (x, bias) if bias is not None else (x,)
        gradients = torch.autograd.grad(output, inputs, torch.ones_like(output))
        del gradients, output, inputs, bias, x


def _warmup_residual(
    config: TransformerConfig, tp_size: int, sequence: int, micro_batch_size: int
) -> None:
    from megatron.core.fusions.fused_bias_dropout import bias_dropout_add_fused_train

    if config.sequence_parallel:
        sequence //= tp_size
    for grad_enabled in (False, True):
        with torch.set_grad_enabled(grad_enabled):
            for _ in range(2):
                x = torch.zeros(
                    (sequence, micro_batch_size, config.hidden_size),
                    device="cuda",
                    dtype=config.params_dtype,
                    requires_grad=True,
                )
                residual = torch.zeros_like(
                    x,
                    dtype=torch.float32 if config.fp32_residual_connection else x.dtype,
                    requires_grad=True,
                )
                bias = (
                    torch.zeros_like(x[0, 0], requires_grad=True)
                    if config.add_bias_linear
                    else None
                )
                output = bias_dropout_add_fused_train((x, bias), residual, config.hidden_dropout)
                if grad_enabled:
                    inputs = (x, residual, bias) if bias is not None else (x, residual)
                    gradients = torch.autograd.grad(output, inputs, torch.ones_like(output))
                    del gradients, inputs
                del output, residual, bias, x


def _shutdown_compile_workers() -> None:
    # Do not import Inductor solely for cleanup. This version-dependent helper
    # resets the pool, preserving compiled kernels and allowing later compilation.
    module = sys.modules.get("torch._inductor.async_compile")
    if module is not None:
        shutdown = getattr(module, "shutdown_compile_workers", None)
        if shutdown is not None:
            shutdown()
        else:
            logger.warning("This PyTorch build lacks compiler-worker shutdown")


def warmup_training_kernels(
    model_config,
    tp_group: torch.distributed.ProcessGroup,
    *,
    seq_length: int,
    micro_batch_size: int,
    legacy_args: Namespace,
) -> None:
    """Resolve model-specific inputs, then warm up at the explicit training batch shape.

    GPT/Hybrid settings come from model_config. legacy_args is used only for
    providers such as MiMo that do not expose a single TransformerConfig here.
    """
    from megatron.training.argument_utils import core_transformer_config_from_args
    from megatron.training.models import GPTModelConfig, HybridModelConfig
    from megatron.training.vocab_utils import calculate_padded_vocab_size

    if isinstance(model_config, (GPTModelConfig, HybridModelConfig)):
        config = model_config.transformer
        logit_dtype = model_config.logit_dtype
        vocab_size = model_config.vocab_size
        if (
            config.cross_entropy_loss_fusion
            and config.cross_entropy_fusion_impl == "native"
            and vocab_size is not None
            and model_config.should_pad_vocab
        ):
            vocab_size = calculate_padded_vocab_size(
                vocab_size,
                model_config.make_vocab_size_divisible_by,
                config.tensor_model_parallel_size,
                logging_enabled=False,
            )
    else:
        config = core_transformer_config_from_args(legacy_args)
        logit_dtype = legacy_args.logit_dtype
        vocab_size = legacy_args.padded_vocab_size

    _warmup_training_kernels(
        config,
        tp_group,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        padded_vocab_size=vocab_size,
        logit_dtype=logit_dtype,
    )


def _warmup_training_kernels(
    config: TransformerConfig,
    tp_group: torch.distributed.ProcessGroup,
    *,
    seq_length: int,
    micro_batch_size: int,
    padded_vocab_size: int | None,
    logit_dtype: torch.dtype | None = None,
) -> None:
    """Compile/autotune supported kernels, then release temporary startup resources.

    Call after set_jit_fusion_options() on every tensor-parallel rank, before
    model initialization.
    Warmup executes actual forward/backward functions at configured static
    shapes; no training data or parameters are needed. RNG state and compiler
    parallelism are preserved. Dynamic expert shapes and library-specific kernels
    can still compile later; shutting down workers does not disable compilation.

    Args:
        config: The TransformerConfig used to build the model.
        tp_group: The model's tensor-parallel process group.
        seq_length: Actual training sequence length before context parallel partitioning.
        micro_batch_size: Microbatch size used by the training schedule and dataloader.
        padded_vocab_size: Global padded vocabulary size before tensor parallel partitioning.
            May be None when native fused cross-entropy is disabled.
        logit_dtype: Model output-projection dtype; None uses config.params_dtype for warmup.
    """
    sequence = seq_length // config.context_parallel_size
    if sequence <= 0 or micro_batch_size is None or micro_batch_size <= 0:
        raise ValueError("Kernel warmup dimensions must be positive")
    tp_size = tp_group.size()
    # New warmups (notably dropout) must not advance training RNG state. The
    # existing JIT warmup runs separately and retains its original RNG behavior.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.enable_grad():
        if config.cross_entropy_loss_fusion and config.cross_entropy_fusion_impl == "native":
            if padded_vocab_size is None:
                raise ValueError("Model vocabulary must be resolved before kernel warmup")
            _warmup_cross_entropy(
                config, tp_group, sequence, micro_batch_size, padded_vocab_size, logit_dtype
            )

        fused_squared_relu = (
            config.activation_func == squared_relu
            and not config.gated_linear_unit
            and config.use_fused_weighted_squared_relu
            and config.activation_func_tanh_clamp_scale is not None
        )
        swiglu = (
            config.activation_func == F.silu
            and config.gated_linear_unit
            and config.bias_activation_fusion
        )
        gelu = (
            config.activation_func == F.gelu
            and not config.gated_linear_unit
            and config.bias_activation_fusion
            and config.add_bias_linear
        )
        if not config.use_te_activation_func and (fused_squared_relu or swiglu or gelu):
            # Shared and dense MLPs have fixed shapes. Routed-expert token counts
            # are data-dependent and are intentionally left to runtime compilation.
            widths = {config.ffn_hidden_size}
            if config.moe_shared_expert_intermediate_size:
                widths.add(config.moe_shared_expert_intermediate_size)
            for width in sorted(widths):
                _warmup_activation(config, tp_size, width, sequence, micro_batch_size)
                # MLP paths also use flattened [tokens, width] activations;
                # Dynamo specializes on tensor rank even with identical numel.
                _warmup_activation(config, tp_size, width, sequence, micro_batch_size, flatten=True)
        if config.bias_dropout_fusion:
            _warmup_residual(config, tp_size, sequence, micro_batch_size)

    # Each helper has returned after invoking its kernels and waiting for their
    # compilation. Finish GPU work before releasing unused pinned blocks.
    torch.cuda.synchronize()
    _shutdown_compile_workers()
    gc.collect()
    host_memory = getattr(getattr(torch, "accelerator", None), "memory", None)
    empty_host_cache = getattr(host_memory, "empty_host_cache", None)
    if empty_host_cache is not None:
        empty_host_cache()
    else:
        logger.warning("This PyTorch build lacks empty_host_cache; pinned cache may remain")
    torch.cuda.empty_cache()
