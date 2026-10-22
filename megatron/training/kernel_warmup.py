# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Warm up configured training kernels before model and offload storage is live."""

import gc
import logging
import sys
from argparse import Namespace

import torch

logger = logging.getLogger(__name__)


def _warmup_cross_entropy(args: Namespace, tp_group: torch.distributed.ProcessGroup) -> None:
    from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy

    sequence = args.seq_length // args.context_parallel_size
    vocab = args.padded_vocab_size
    if vocab <= 0:
        raise ValueError("CE vocabulary must be positive")
    if vocab % tp_group.size():
        raise ValueError("CE vocabulary must be divisible by tensor parallel size")
    for _ in range(2):
        logits = torch.zeros(
            (sequence, args.micro_batch_size, vocab // tp_group.size()),
            device="cuda",
            dtype=args.logit_dtype or args.params_dtype,
            requires_grad=True,
        )
        # Match the model's batch-major labels, including singleton-dimension
        # strides: contiguous() need not copy a transposed microbatch of one.
        labels = torch.zeros((args.micro_batch_size, sequence), device="cuda", dtype=torch.long)
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


def _warmup_activation(args: Namespace, tp_size: int, width: int, *, flatten: bool = False) -> None:
    from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
    from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl
    from megatron.core.fusions.fused_weighted_squared_relu import weighted_squared_relu_impl

    sequence = args.seq_length // args.context_parallel_size
    leading_shape = (sequence, args.micro_batch_size)
    if flatten:
        leading_shape = (sequence * args.micro_batch_size,)
    for _ in range(2):
        x = torch.zeros(
            (*leading_shape, width // tp_size * (2 if args.swiglu else 1)),
            device="cuda",
            dtype=args.params_dtype,
            requires_grad=True,
        )
        bias = (
            torch.zeros(x.shape[-1], device="cuda", dtype=x.dtype, requires_grad=True)
            if args.add_bias_linear and not args.squared_relu
            else None
        )
        if args.squared_relu:
            output = weighted_squared_relu_impl(x, None, args.activation_func_tanh_clamp_scale)
        elif args.swiglu:
            output = bias_swiglu_impl(
                x,
                bias,
                getattr(args, "activation_func_fp8_input_store", False),
                clamp_value=args.activation_func_clamp_value,
                gate_clamp_scale=args.activation_func_tanh_clamp_scale,
                linear_clamp_scale=args.activation_func_tanh_clamp_scale_linear,
            )
        else:
            output = bias_gelu_impl(x, bias)
        inputs = (x, bias) if bias is not None else (x,)
        gradients = torch.autograd.grad(output, inputs, torch.ones_like(output))
        del gradients, output, inputs, bias, x


def _warmup_residual(args: Namespace, tp_size: int) -> None:
    from megatron.core.fusions.fused_bias_dropout import bias_dropout_add_fused_train

    sequence = args.seq_length // args.context_parallel_size
    if args.sequence_parallel:
        sequence //= tp_size
    for grad_enabled in (False, True):
        with torch.set_grad_enabled(grad_enabled):
            for _ in range(2):
                x = torch.zeros(
                    (sequence, args.micro_batch_size, args.hidden_size),
                    device="cuda",
                    dtype=args.params_dtype,
                    requires_grad=True,
                )
                residual = torch.zeros_like(
                    x,
                    dtype=torch.float32 if args.fp32_residual_connection else x.dtype,
                    requires_grad=True,
                )
                bias = (
                    torch.zeros_like(x[0, 0], requires_grad=True) if args.add_bias_linear else None
                )
                output = bias_dropout_add_fused_train((x, bias), residual, args.hidden_dropout)
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


def warmup_training_kernels(args: Namespace, tp_group: torch.distributed.ProcessGroup) -> None:
    """Compile/autotune supported kernels, then release temporary startup resources.

    Call after set_jit_fusion_options() on every tensor-parallel rank, before
    model initialization.
    Warmup executes actual forward/backward functions at configured static
    shapes; no training data or parameters are needed. RNG state and compiler
    parallelism are preserved. Dynamic expert shapes and library-specific kernels
    can still compile later; shutting down workers does not disable compilation.
    """
    if args.seq_length // args.context_parallel_size <= 0 or args.micro_batch_size <= 0:
        raise ValueError("Kernel warmup dimensions must be positive")
    tp_size = tp_group.size()
    # New warmups (notably dropout) must not advance training RNG state. The
    # existing JIT warmup runs separately and retains its original RNG behavior.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.enable_grad():
        if args.cross_entropy_loss_fusion and args.cross_entropy_fusion_impl == "native":
            _warmup_cross_entropy(args, tp_group)

        squared_relu = (
            args.squared_relu
            and args.use_fused_weighted_squared_relu
            and args.activation_func_tanh_clamp_scale is not None
        )
        swiglu = args.swiglu and args.bias_swiglu_fusion
        gelu = (
            not args.squared_relu
            and not args.swiglu
            and not args.quick_geglu
            and args.bias_gelu_fusion
            and args.add_bias_linear
        )
        if not args.use_te_activation_func and (squared_relu or swiglu or gelu):
            # Shared and dense MLPs have fixed shapes. Routed-expert token counts
            # are data-dependent and are intentionally left to runtime compilation.
            widths = {args.ffn_hidden_size}
            if args.moe_shared_expert_intermediate_size:
                widths.add(args.moe_shared_expert_intermediate_size)
            for width in sorted(widths):
                _warmup_activation(args, tp_size, width)
                # MLP paths also use flattened [tokens, width] activations;
                # Dynamo specializes on tensor rank even with identical numel.
                _warmup_activation(args, tp_size, width, flatten=True)
        if args.bias_dropout_fusion:
            _warmup_residual(args, tp_size)

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
