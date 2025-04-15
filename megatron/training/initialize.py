# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Megatron initialization."""
import logging
import os
import random
import time
import warnings
from datetime import timedelta

import numpy as np
import torch

from megatron.core import mpu, tensor_parallel
from megatron.core.fusions.fused_bias_dropout import bias_dropout_add_fused_train
from megatron.core.fusions.fused_bias_gelu import bias_gelu
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu
from megatron.core.parallel_state import create_group
from megatron.core.rerun_state_machine import (
    RerunDiagnostic,
    RerunErrorInjector,
    RerunMode,
    initialize_rerun_state_machine,
)
from megatron.core.utils import get_te_version, is_te_min_version, is_torch_min_version
from megatron.legacy import fused_kernels
from megatron.training import get_adlr_autoresume, get_args, get_tensorboard_writer
from megatron.training.arguments import parse_args, validate_args
from megatron.training.async_utils import init_persistent_async_worker
from megatron.training.checkpointing import load_args_from_checkpoint
from megatron.training.global_vars import set_global_variables
from megatron.training.yaml_arguments import validate_yaml

logger = logging.getLogger(__name__)


def initialize_megatron(
    extra_args_provider=None,
    args_defaults={},
    ignore_unknown_args=False,
    allow_no_cuda=False,
    skip_mpu_initialization=False,
    get_embedding_ranks=None,
    get_position_embedding_ranks=None,
    parsed_args=None,
):
    """Set global variables, initialize distributed, and
    set autoresume and random seeds.
    `allow_no_cuda` should not be set unless using megatron for cpu only
    data processing. In general this arg should not be set unless you know
    what you are doing.
    Returns a function to finalize distributed env initialization
    (optionally, only when args.lazy_mpu_init == True)
    """
    if not allow_no_cuda:
        # Make sure cuda is available.
        assert torch.cuda.is_available(), "Megatron requires CUDA."

    # Parse arguments
    if parsed_args is None:
        args = parse_args(extra_args_provider, ignore_unknown_args)
    else:
        args = parsed_args

    # Prep for checkpoint conversion.
    if args.ckpt_convert_format is not None:
        assert args.ckpt_convert_save is not None
        assert args.load is not None
        args.exit_on_missing_checkpoint = True

    if args.use_checkpoint_args or args_defaults.get("use_checkpoint_args", False):
        assert args.load is not None, "--use-checkpoint-args requires --load argument"
        assert args.non_persistent_ckpt_type != "local", (
            "--use-checkpoint-args is not supported with --non_persistent_ckpt_type=local. "
            "Two-stage checkpoint loading is not implemented, and all arguments must be defined "
            "before initializing LocalCheckpointManager."
        )
        load_args_from_checkpoint(args)

    if args.async_save and args.use_persistent_ckpt_worker:
        init_persistent_async_worker()

    if args.yaml_cfg is not None:
        args = validate_yaml(args, args_defaults)
    else:
        validate_args(args, args_defaults)

    # set global args, build tokenizer, and set adlr-autoresume,
    # tensorboard-writer, and timers.
    set_global_variables(args)

    # set logging level
    setup_logging()

    # init rerun state
    def state_save_func():
        return {'rng_tracker_states': tensor_parallel.get_cuda_rng_tracker().get_states()}

    def state_restore_func(state_dict):
        if state_dict['rng_tracker_states']:
            tensor_parallel.get_cuda_rng_tracker().set_states(state_dict['rng_tracker_states'])

    args = get_args()
    initialize_rerun_state_machine(
        state_save_func=state_save_func,
        state_restore_func=state_restore_func,
        mode=RerunMode(args.rerun_mode),
        error_injector=RerunErrorInjector(
            error_injection_rate=args.error_injection_rate,
            error_injection_type=RerunDiagnostic(args.error_injection_type),
        ),
        result_rejected_tracker_filename=args.result_rejected_tracker_filename,
    )

    # torch.distributed initialization
    def finish_mpu_init():
        args = get_args()
        # Pytorch distributed.
        _initialize_distributed(get_embedding_ranks, get_position_embedding_ranks)

        # Random seeds for reproducibility.
        if args.rank == 0:
            print("> setting random seeds to {} ...".format(args.seed))
        _set_random_seed(
            args.seed,
            args.data_parallel_random_init,
            args.te_rng_tracker,
            args.inference_rng_tracker,
            use_cudagraphable_rng=args.enable_cuda_graph,
        )

        # Setup MoE aux loss scale value.
        if args.num_experts is not None:
            from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler

            MoEAuxLossAutoScaler.set_loss_scale(torch.ones(1, device=torch.cuda.current_device()))

    if skip_mpu_initialization:
        return None

    args = get_args()
    if args.lazy_mpu_init:
        # TODO is this still a necessary option?
        args.use_cpu_initialization = True
        # delayed initialization of DDP-related stuff
        # We only set basic DDP globals
        mpu.set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
        # and return function for external DDP manager
        # to call when it has DDP initialized
        mpu.set_tensor_model_parallel_rank(args.rank)
        return finish_mpu_init
    else:
        # Megatron's MPU is the master. Complete initialization right away.
        finish_mpu_init()

        # Autoresume.
        _init_autoresume()

        # Compile dependencies.
        _compile_dependencies()

        if args.tp_comm_overlap:
            # TODO: Should this be activated with just decoder-tp-comm-overlap too?
            _initialize_tp_communicators()

        # No continuation function
        return None


def _compile_dependencies():

    args = get_args()

    # =========================
    # Compile dataset C++ code.
    # =========================
    # TODO: move this to ninja
    if torch.distributed.get_rank() == 0:
        start_time = time.time()
        print("> compiling dataset index builder ...")
        from megatron.core.datasets.utils import compile_helpers

        compile_helpers()
        print(
            ">>> done with dataset index builder. Compilation time: {:.3f} "
            "seconds".format(time.time() - start_time),
            flush=True,
        )

    # ==================
    # Load fused kernels
    # ==================

    # Custom kernel constraints check.
    seq_len = args.seq_length
    attn_batch_size = (
        args.num_attention_heads / args.tensor_model_parallel_size
    ) * args.micro_batch_size
    # Constraints on sequence length and attn_batch_size to enable warp based
    # optimization and upper triangular optimization (for causal mask)
    custom_kernel_constraint = (
        seq_len > 16 and seq_len <= 16384 and seq_len % 4 == 0 and attn_batch_size % 4 == 0
    )
    # Print a warning.
    if not ((args.fp16 or args.bf16) and custom_kernel_constraint and args.masked_softmax_fusion):
        if args.rank == 0:
            print(
                "WARNING: constraints for invoking optimized"
                " fused softmax kernel are not met. We default"
                " back to unfused kernel invocations.",
                flush=True,
            )

    # Always build on rank zero first.
    if torch.distributed.get_rank() == 0:
        start_time = time.time()
        print("> compiling and loading fused kernels ...", flush=True)
        fused_kernels.load(args)
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()
        fused_kernels.load(args)
    # Simple barrier to make sure all ranks have passed the
    # compilation phase successfully before moving on to the
    # rest of the program. We think this might ensure that
    # the lock is released.
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(
            ">>> done with compiling and loading fused kernels. "
            "Compilation time: {:.3f} seconds".format(time.time() - start_time),
            flush=True,
        )


def _initialize_tp_communicators():
    """initializing the communicators with user buffers for high-performance tensor-model-parallel
    communication overlap"""

    try:
        import transformer_engine
        import yaml
        from transformer_engine.pytorch import module as te_module

    except ImportError:
        raise RuntimeError(
            "Tensor Parallel Communication/GEMM Overlap optimization needs 'yaml' and "
            "'transformer_engine' packages"
        )

    args = get_args()

    if args.tp_comm_overlap_cfg is not None:
        with open(args.tp_comm_overlap_cfg, "r") as stream:
            ub_cfgs = yaml.safe_load(stream)
    else:
        ub_cfgs = {}

    if getattr(args, 'decoder_tp_comm_overlap', False):
        input_shape = [
            (args.decoder_seq_length * args.micro_batch_size) // args.context_parallel_size,
            args.hidden_size,
        ]
    else:
        input_shape = [
            (args.seq_length * args.micro_batch_size) // args.context_parallel_size,
            args.hidden_size,
        ]

    if is_te_min_version("1.9.0"):
        # The process group with the target bootstrap backend is created in Transformer Engine.
        te_module.base.initialize_ub(
            shape=input_shape,
            tp_size=args.tensor_model_parallel_size,
            use_fp8=(args.fp8 is not None),
            ub_cfgs=ub_cfgs,
            bootstrap_backend=args.tp_comm_bootstrap_backend,
        )
    else:
        if args.tp_comm_bootstrap_backend != 'mpi':
            warnings.warn(
                f"Transformer Engine v{get_te_version()} supports only MPI bootstrap backend."
            )
        # Create a MPI process group to help with TP communication overlap bootstrap.
        create_group(backend='mpi', group_desc='TP_BOOTSTRAP_GROUP_MPI')

        te_module.base.initialize_ub(
            shape=input_shape,
            tp_size=args.tensor_model_parallel_size,
            use_fp8=(args.fp8 is not None),
            ub_cfgs=ub_cfgs,
        )


def _initialize_distributed(get_embedding_ranks, get_position_embedding_ranks):
    """Initialize torch.distributed and core model parallel."""
    args = get_args()

    device_count = torch.cuda.device_count()
    if torch.distributed.is_initialized():

        if args.rank == 0:
            print(
                "torch distributed is already initialized, " "skipping initialization ...",
                flush=True,
            )
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()

    else:

        if args.rank == 0:
            print("> initializing torch distributed ...", flush=True)
        # Manually set the device ids.
        if device_count > 0:
            torch.cuda.set_device(args.local_rank)
            device_id = torch.device(f'cuda:{args.local_rank}')
        else:
            device_id = None

        # Set to non-default stream for cudagraph capturing.
        if args.external_cuda_graph:
            torch.cuda.set_stream(torch.cuda.Stream())

        # Call the init process
        init_process_group_kwargs = {
            'backend': args.distributed_backend,
            'world_size': args.world_size,
            'rank': args.rank,
            'timeout': timedelta(minutes=args.distributed_timeout_minutes),
        }

        torch.distributed.init_process_group(**init_process_group_kwargs)

    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    if device_count > 0:
        if mpu.model_parallel_is_initialized():
            print("model parallel is already initialized")
        else:
            mpu.initialize_model_parallel(
                args.tensor_model_parallel_size,
                args.pipeline_model_parallel_size,
                args.virtual_pipeline_model_parallel_size,
                args.pipeline_model_parallel_split_rank,
                pipeline_model_parallel_comm_backend=args.pipeline_model_parallel_comm_backend,
                context_parallel_size=args.context_parallel_size,
                hierarchical_context_parallel_sizes=args.hierarchical_context_parallel_sizes,
                expert_model_parallel_size=args.expert_model_parallel_size,
                num_distributed_optimizer_instances=args.num_distributed_optimizer_instances,
                expert_tensor_parallel_size=args.expert_tensor_parallel_size,
                distributed_timeout_minutes=args.distributed_timeout_minutes,
                nccl_communicator_config_path=args.nccl_communicator_config_path,
                order='tp-cp-ep-dp-pp' if not args.use_tp_pp_dp_mapping else 'tp-cp-ep-pp-dp',
                encoder_tensor_model_parallel_size=args.encoder_tensor_model_parallel_size,
                encoder_pipeline_model_parallel_size=args.encoder_pipeline_model_parallel_size,
                get_embedding_ranks=get_embedding_ranks,
                get_position_embedding_ranks=get_position_embedding_ranks,
                create_gloo_process_groups=args.enable_gloo_process_groups,
            )
            if args.rank == 0:
                print(
                    f"> initialized tensor model parallel with size "
                    f"{mpu.get_tensor_model_parallel_world_size()}"
                )
                print(
                    f"> initialized pipeline model parallel with size "
                    f"{mpu.get_pipeline_model_parallel_world_size()}"
                )


def _init_autoresume():
    """Set autoresume start time."""
    autoresume = get_adlr_autoresume()
    if autoresume:
        torch.distributed.barrier()
        autoresume.init()
        torch.distributed.barrier()


def _set_random_seed(
    seed_: int,
    data_parallel_random_init: bool = False,
    te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
):
    """Set random seed for reproducability."""
    if seed_ is not None and seed_ > 0:
        # Ensure that different pipeline MP stages get different seeds.
        seed = seed_ + (100 * mpu.get_pipeline_model_parallel_rank())
        # Ensure different data parallel ranks get different seeds
        if data_parallel_random_init:
            seed = seed + (10 * mpu.get_data_parallel_rank())
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.device_count() > 0:
            tensor_parallel.model_parallel_cuda_manual_seed(
                seed, te_rng_tracker, inference_rng_tracker, use_cudagraphable_rng
            )
    else:
        raise ValueError("Seed ({}) should be a positive integer.".format(seed_))


def write_args_to_tensorboard():
    """Write arguments to tensorboard."""
    args = get_args()
    writer = get_tensorboard_writer()
    if writer:
        for arg in vars(args):
            writer.add_text(arg, str(getattr(args, arg)), global_step=args.iteration)


def set_jit_fusion_options():
    """Set PyTorch JIT layer fusion options."""
    # flags required to enable jit fusion kernels
    if is_torch_min_version("2.2.0a0"):
        pass  # we're using torch.compile for jit fusion
    elif is_torch_min_version("1.10.0a0"):
        # nvfuser
        torch._C._jit_set_profiling_executor(True)
        torch._C._jit_set_profiling_mode(True)
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_nvfuser_enabled(True)
        torch._C._debug_set_autodiff_subgraph_inlining(False)
    else:
        # legacy pytorch fuser
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)

    _warmup_jit_function()


def _warmup_jit_function():
    """Compilie JIT functions before the main training steps"""
    args = get_args()
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Warmup fused bias+gelu
    bias = torch.rand(
        args.ffn_hidden_size // args.tensor_model_parallel_size, dtype=dtype, device="cuda"
    )
    input = torch.rand(
        (
            args.seq_length // args.context_parallel_size,
            args.micro_batch_size,
            args.ffn_hidden_size // args.tensor_model_parallel_size,
        ),
        dtype=dtype,
        device="cuda",
    )
    # Warmup JIT fusions with the input grad_enable state of both forward
    # prop and recomputation
    for bias_grad, input_grad in zip([True, True], [False, True]):
        bias.requires_grad, input.requires_grad = bias_grad, input_grad
        for _ in range(5):
            if args.swiglu:
                output = bias_swiglu(input, bias)
            else:
                output = bias_gelu(bias, input)
    del bias, input, output

    # Warmup fused bias+dropout+add
    if args.sequence_parallel:
        seq_length = args.seq_length // mpu.get_tensor_model_parallel_world_size()
    else:
        seq_length = args.seq_length
    input = torch.rand(
        (seq_length // args.context_parallel_size, args.micro_batch_size, args.hidden_size),
        dtype=dtype,
        device="cuda",
    )
    residual = torch.rand(
        (seq_length // args.context_parallel_size, args.micro_batch_size, args.hidden_size),
        dtype=dtype,
        device="cuda",
    )
    bias = torch.rand((args.hidden_size), dtype=dtype, device="cuda").expand_as(residual)
    dropout_rate = 0.1
    # Warmup JIT fusions with the input grad_enable state of both forward
    # prop and recomputation
    for input_grad, bias_grad, residual_grad in zip([False, True], [True, True], [True, True]):
        input.requires_grad = input_grad
        bias.requires_grad = bias_grad
        residual.requires_grad = residual_grad
        for _ in range(5):
            output = bias_dropout_add_fused_train([input, bias], residual, dropout_rate)
    del bias, input, residual, output
    torch.cuda.empty_cache()


def setup_logging() -> None:
    """Sets the default logging level based on cmdline args and env vars.

    Precedence:
    1. Command line argument `--logging-level`
    2. Env var `MEGATRON_LOGGING_LEVEL`
    3. Default logging level (INFO)

    Returns: None
    """
    args = get_args()
    logging_level = None
    env_logging_level = os.getenv('MEGATRON_LOGGING_LEVEL', None)
    if env_logging_level is not None:
        logging_level = int(env_logging_level)
    if args.logging_level is not None:
        logging_level = args.logging_level

    if logging_level is not None:
        logger.info(f'Setting logging level to {logging_level}')
        logging.getLogger().setLevel(logging_level)
