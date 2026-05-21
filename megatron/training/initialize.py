# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Megatron initialization."""
import logging
import os
import random
import time
import warnings
from datetime import timedelta
from typing import Callable

import numpy as np
from megatron.core._rank_utils import safe_get_rank, safe_get_world_size
from megatron.training.utils.common_utils import get_local_rank_preinit
import torch
import torch.nn.functional as F

from megatron.core import mpu, tensor_parallel
from megatron.core.fusions.fused_bias_dropout import bias_dropout_add_fused_train
from megatron.core.fusions.fused_bias_gelu import bias_gelu
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu
from megatron.core.parallel_state import create_group
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.rerun_state_machine import (
    RerunDiagnostic,
    RerunErrorInjector,
    RerunMode,
    initialize_rerun_state_machine,
)
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
    enable_batch_invariant_mode,
)
from megatron.core.utils import get_te_version, is_te_min_version, is_torch_min_version, get_pg_rank
from megatron.training import (
    get_adlr_autoresume,
    get_args,
    get_tensorboard_writer,
    inprocess_restart,
)
from megatron.training.async_utils import init_persistent_async_worker
from megatron.training.utils import is_rank0, print_rank_0, warn_rank_0
from megatron.training.config import DistributedInitConfig, RNGConfig, RerunStateMachineConfig
from megatron.training.models import HybridModelConfig, GPTModelConfig

logger = logging.getLogger(__name__)


def initialize_megatron(
    allow_no_cuda=False,
    skip_mpu_initialization=False,
    get_embedding_ranks=None,
    get_position_embedding_ranks=None,
    store=None,
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

    args = get_args()

    # set logging level
    setup_logging()

    if args.async_save and args.use_persistent_ckpt_worker:
        init_persistent_async_worker(args.rank, 'forkserver')

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

    if args.batch_invariant_mode:
        print_rank_0("Enabling batch invariant mode globally")
        enable_batch_invariant_mode()

    # torch.distributed initialization
    def finish_mpu_init():
        args = get_args()
        # Pytorch distributed.
        _initialize_distributed(get_embedding_ranks, get_position_embedding_ranks, store)

        # Random seeds for reproducibility.
        print_rank_0("> setting random seeds to {} ...".format(args.seed))
        _set_random_seed(
            args.seed,
            args.data_parallel_random_init,
            args.te_rng_tracker,
            args.inference_rng_tracker,
            use_cudagraphable_rng=args.cuda_graph_impl != "none",
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


def torch_dist_init(
    model_config: GPTModelConfig | HybridModelConfig,
    dist_config: DistributedInitConfig,
    rng_config: RNGConfig,
    micro_batch_size: int,
    num_distributed_optimizer_instances: int,
    get_embedding_ranks: Callable[[list[int], int | None], list[int]] | None,
    get_position_embedding_ranks: Callable[[list[int], int | None], list[int]] | None,
    skip_mpu_initialization: bool,
    restart_store: torch.distributed.Store | None = None,
    use_inprocess_restart: bool = False,
) -> Callable[[], ProcessGroupCollection] | ProcessGroupCollection | None:
    """Initialize torch.distributed and dependent components.

    Handles the core distributed setup, including process group initialization,
    MPU (Model Parallel Unit) setup, random seed setting, and optional
    compilation/warmup steps.

    Args:
        model_config: Configuration for the specific model (GPTConfig or T5Config).
        dist_config: Configuration for distributed initialization settings.
        rng_config: Configuration for random number generation.
        micro_batch_size: The micro batch size for JIT warmup.
        num_distributed_optimizer_instances: Number of parallel optimizer instances.
        get_embedding_ranks: Optional function to determine embedding layer ranks.
        get_position_embedding_ranks: Optional function to determine position embedding ranks.
        skip_mpu_initialization: If True, returns a function to finish MPU setup later.

    Returns:
        An optional callable to finish MPU initialization if skip_mpu_initialization
        or lazy_mpu_init is True, otherwise None.
    """

    def finish_mpu_init() -> ProcessGroupCollection:
        # Pytorch distributed.
        pg_collection = _initialize_distributed(
            model_config=model_config.transformer,
            dist_config=dist_config,
            num_distributed_optimizer_instances=num_distributed_optimizer_instances,
            get_embedding_ranks=get_embedding_ranks,
            get_position_embedding_ranks=get_position_embedding_ranks,
            restart_store=restart_store,
            use_inprocess_restart=use_inprocess_restart,
        )

        # Random seeds for reproducibility.
        print_rank_0("> setting random seeds to {} ...".format(rng_config.seed))
        _set_random_seed(
            rng_config.seed,
            pg_collection,
            rng_config.data_parallel_random_init,
            rng_config.te_rng_tracker,
            rng_config.inference_rng_tracker,
            use_cudagraphable_rng=(model_config.cuda_graph_impl != "none"),
        )

        if model_config.num_moe_experts is not None:
            from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler

            MoEAuxLossAutoScaler.set_loss_scale(torch.ones(1, device=torch.cuda.current_device()))

        return pg_collection

    if skip_mpu_initialization:
        return None

    if dist_config.lazy_mpu_init:
        # delayed initialization of DDP-related stuff
        # We only set basic DDP globals
        mpu.set_tensor_model_parallel_world_size(model_config.tensor_model_parallel_size)
        # and return function for external DDP manager
        # to call when it has DDP initialized
        mpu.set_tensor_model_parallel_rank(safe_get_rank())
        return finish_mpu_init

    # Megatron's MPU is the master. Complete initialization right away.
    pg_collection = finish_mpu_init()

    # Autoresume.
    _init_autoresume()

    # Compile dependencies.
    _compile_dependencies()

    if model_config.tp_comm_overlap:
        _initialize_tp_communicators(model_config, micro_batch_size)

    return pg_collection


def init_rerun_state(rerun_state_machine_config: RerunStateMachineConfig) -> None:
    """Initialize the rerun state machine for result validation or stats.

    Sets up state saving and restoration functions, particularly for RNG trackers.

    Args:
        rerun_state_machine_config: Configuration for the rerun state machine.
    """
    from megatron.core.rerun_state_machine import (
        RerunDiagnostic,
        RerunErrorInjector,
        RerunMode,
        get_rerun_state_machine,
        initialize_rerun_state_machine,
    )

    def state_save_func():
        return {"rng_tracker_states": tensor_parallel.get_cuda_rng_tracker().get_states()}

    def state_restore_func(state_dict):
        if state_dict["rng_tracker_states"]:
            tensor_parallel.get_cuda_rng_tracker().set_states(state_dict["rng_tracker_states"])

    initialize_rerun_state_machine(
        state_save_func=state_save_func,
        state_restore_func=state_restore_func,
        mode=RerunMode(rerun_state_machine_config.rerun_mode),
        error_injector=RerunErrorInjector(
            error_injection_rate=rerun_state_machine_config.error_injection_rate,
            error_injection_type=RerunDiagnostic(rerun_state_machine_config.error_injection_type),
        ),
        result_rejected_tracker_filename=rerun_state_machine_config.result_rejected_tracker_filename,
    )


def _compile_dependencies():

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

    torch.distributed.barrier()

def _initialize_tp_communicators(
    model_config: GPTModelConfig | HybridModelConfig, micro_batch_size: int, ub_shape: list[int] | None=None,
) -> None:
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

    tp_comm_overlap_cfg = getattr(model_config, "tp_comm_overlap_cfg", None)
    if tp_comm_overlap_cfg is not None:
        if isinstance(tp_comm_overlap_cfg, str):
            with open(tp_comm_overlap_cfg, "r") as stream:
                ub_cfgs = yaml.safe_load(stream)
        else:
            ub_cfgs = tp_comm_overlap_cfg
    else:
        ub_cfgs = {}

    # pretrain_vlm.py can explicitly call _initialize_tp_communicators() with ub_shape set as below
    # if getattr(args, 'decoder_tp_comm_overlap', False):
    #     ub_shape = [
    #         (args.decoder_seq_length * args.micro_batch_size) // args.context_parallel_size,
    #         args.hidden_size,
    #     ]
    if ub_shape is not None:
        input_shape = ub_shape
    else:
        input_shape = [
            (model_config.seq_length * micro_batch_size) // model_config.context_parallel_size,
            model_config.hidden_size,
        ]

    if is_te_min_version("2.7.0"):
        UserBufferQuantizationMode = te_module.base.UserBufferQuantizationMode
        quantization_modes = [
            UserBufferQuantizationMode.FP8 if model_config.fp8 else UserBufferQuantizationMode.NONE
        ]
        if (
            model_config.fp8 is not None
            and model_config.first_last_layers_bf16
            and (model_config.num_layers_at_start_in_bf16 > 0 or model_config.num_layers_at_end_in_bf16 > 0)
        ):
            quantization_modes.append(UserBufferQuantizationMode.NONE)
        # The process group with the target bootstrap backend is created in Transformer Engine.
        te_module.base.initialize_ub(
            shape=input_shape,
            tp_size=model_config.tensor_model_parallel_size,
            quantization_modes=quantization_modes,
            ub_cfgs=ub_cfgs,
            bootstrap_backend=model_config.tp_comm_bootstrap_backend,
        )
    elif is_te_min_version("1.9.0"):
        # The process group with the target bootstrap backend is created in Transformer Engine.
        te_module.base.initialize_ub(
            shape=input_shape,
            tp_size=model_config.tensor_model_parallel_size,
            use_fp8=(model_config.fp8 is not None),
            ub_cfgs=ub_cfgs,
            bootstrap_backend=model_config.tp_comm_bootstrap_backend,
        )
    else:
        if model_config.tp_comm_bootstrap_backend != 'mpi':
            warnings.warn(
                f"Transformer Engine v{get_te_version()} supports only MPI bootstrap backend."
            )
        # Create a MPI process group to help with TP communication overlap bootstrap.
        create_group(backend='mpi', group_desc='TP_BOOTSTRAP_GROUP_MPI')

        te_module.base.initialize_ub(
            shape=input_shape,
            tp_size=model_config.tensor_model_parallel_size,
            use_fp8=(model_config.fp8 is not None),
            ub_cfgs=ub_cfgs,
        )


def _setup_flight_recorder_env(dist_config: DistributedInitConfig) -> None:
    """Set flight recorder env vars based on config or pre-existing environment.

    Priority: pre-existing env var > config value. If no dump path is provided
    (either via config or env), no env vars are set.
    """
    _fr_path = (
        os.environ.get("TORCH_FR_DUMP_TEMP_FILE")
        or os.environ.get("TORCH_NCCL_DEBUG_INFO_TEMP_FILE")
        or dist_config.flight_recorder_dump_path
    )
    if _fr_path is None:
        return

    _fr_dump_prefix = _fr_path
    if os.path.isdir(_fr_path):
        _fr_dump_prefix = os.path.join(_fr_path, '_dump_')
        warnings.warn(
            "Flight recorder: using directory "
            f"'{_fr_path}' for dump path, appending per-rank prefix "
            f"'{_fr_dump_prefix}'.",
            stacklevel=2,
        )
    _fr_env_defaults = {
        "TORCH_FR_DUMP_TEMP_FILE": _fr_dump_prefix,
        "TORCH_NCCL_DEBUG_INFO_TEMP_FILE": _fr_dump_prefix,
        "TORCH_NCCL_TRACE_BUFFER_SIZE": str(dist_config.flight_recorder_trace_buffer_size),
        "TORCH_NCCL_DUMP_ON_TIMEOUT": str(int(dist_config.flight_recorder_dump_on_timeout)),
        "TORCH_INCLUDE_STACK_TRACE": str(int(dist_config.flight_recorder_include_stack_trace)),
        "TORCH_INCLUDE_ONLY_ACTIVE": str(int(dist_config.flight_recorder_include_only_active)),
        "TORCH_NCCL_EXTRA_DUMP_ON_EXEC": str(int(dist_config.flight_recorder_extra_dump_on_exec)),
    }
    for _var, _default in _fr_env_defaults.items():
        if _var in os.environ:
            warnings.warn(
                f"Flight recorder: env var {_var} is already set to "
                f"'{os.environ[_var]}'; ignoring config value '{_default}'.",
                stacklevel=2,
            )
        else:
            os.environ[_var] = _default
    print_rank_0(
            "Flight recorder env vars:\n" + "\n".join(f"  {k}={os.environ[k]}" for k in _fr_env_defaults),
        )


def _create_pg_collection(
    model_config: TransformerConfig,
    num_distributed_optimizer_instances: int,
    get_embedding_ranks: Callable[[list[int], int | None], list[int]] | None = None,
    get_position_embedding_ranks: Callable[[list[int], int | None], list[int]] | None = None,
    world_size: int | None = None,
    rank_offset: int | None = None,
    save_grid: bool = False,
) -> ProcessGroupCollection:
    """Create all process groups via HyperCommGrid and return a ProcessGroupCollection."""
    hcp_sizes = getattr(model_config, "hierarchical_context_parallel_sizes", None)
    if hcp_sizes is not None:
        raise NotImplementedError(
            "Decentralized process groups (use_decentralized_pg=True) do not support "
            "hierarchical_context_parallel_sizes. Use cp_comm_type='a2a' or 'p2p' instead, "
            "or set use_decentralized_pg=False to use the MPU path which supports 'a2a+p2p'."
        )
    if world_size is None:
        world_size = torch.distributed.get_world_size()
    if rank_offset is None:
        rank_offset = 0
    tp_size = int(model_config.tensor_model_parallel_size)
    pp_size = int(model_config.pipeline_model_parallel_size)
    cp_size = int(model_config.context_parallel_size) if getattr(model_config, "context_parallel_size", 1) else 1
    model_size = tp_size * pp_size * cp_size
    if world_size % model_size != 0:
        raise RuntimeError(f"world_size ({world_size}) is not divisible by {model_size}")
    dp_size = world_size // model_size

    grid = HyperCommGrid(
        shape=[tp_size, cp_size, dp_size, pp_size],
        dim_names=["tp", "cp", "dp", "pp"],
        rank_offset=rank_offset,
        backend="nccl",
    )
    # Core groups
    tp_pg = grid.create_pg(["tp"])
    cp_pg = grid.create_pg(["cp"])
    pp_pg = grid.create_pg(["pp"])
    dp_pg = grid.create_pg(["dp"])
    mp_pg = grid.create_pg(["tp", "pp"])
    tp_cp_pg = grid.create_pg(["tp", "cp"])
    tp_dp_cp_pg = grid.create_pg(["tp", "dp", "cp"])
    dp_cp_pg = grid.create_pg(["dp", "cp"])

    # Expert/MoE related groups (refer to original parallel_state.initialize_model_parallel)
    expert_tp_size = (
        int(model_config.expert_tensor_parallel_size)
        if getattr(model_config, "expert_tensor_parallel_size", None)
        else tp_size
    )
    ep_size = (
        int(model_config.expert_model_parallel_size) if getattr(model_config, "expert_model_parallel_size", 1) else 1
    )
    # Expert data-parallel size folds CP into DP (as in original expert rank generator)
    expt_model_block = expert_tp_size * ep_size * pp_size
    if world_size % expt_model_block != 0:
        raise RuntimeError(
            f"world_size ({world_size}) is not divisible by expert_tensor_model_pipeline size ({expt_model_block})"
        )
    expt_dp_size = world_size // expt_model_block
    use_optimizer_instance_groups = num_distributed_optimizer_instances > 1
    inner_dp_dim: str | None = None
    outer_dp_dim: str | None = None
    if use_optimizer_instance_groups:
        assert expt_dp_size % num_distributed_optimizer_instances == 0, (
            "Expert DP size must be divisible by the number of optimizer instances."
        )
        inner_expt_dp_size = expt_dp_size // num_distributed_optimizer_instances
        expert_grid = HyperCommGrid(
            shape=[expert_tp_size, ep_size, inner_expt_dp_size, num_distributed_optimizer_instances, pp_size],
            dim_names=["tp", "ep", "inner_dp", "outer_dp", "pp"],
            rank_offset=rank_offset,
            backend="nccl",
        )
        dp_group_dims: list[str] = ["inner_dp", "outer_dp"]
        inner_dp_dim = "inner_dp"
        outer_dp_dim = "outer_dp"
    else:
        expert_grid = HyperCommGrid(
            shape=[expert_tp_size, ep_size, expt_dp_size, pp_size],
            dim_names=["tp", "ep", "dp", "pp"],
            rank_offset=rank_offset,
            backend="nccl",
        )
        dp_group_dims = ["dp"]
    ep_pg = expert_grid.create_pg(["ep"])
    expt_tp_pg = expert_grid.create_pg(["tp"])
    tp_ep_pg = expert_grid.create_pg(["tp", "ep"])
    tp_ep_pp_pg = expert_grid.create_pg(["tp", "ep", "pp"])
    expt_dp_pg = expert_grid.create_pg(dp_group_dims)

    # Embedding and position-embedding groups
    embd_pg = None
    pos_embd_pg = None
    # Enumerate ranks per PP group
    pp_rank_lists = grid._gen_rank_enum(["pp"])
    # Determine embedding ranks for each pp group
    embedding_rank_lists: list[list[int]] = []
    pos_embedding_rank_lists: list[list[int]] = []
    for ranks in pp_rank_lists:
        if not ranks:
            continue
        if get_embedding_ranks is not None:
            # Use custom callback to determine embedding ranks
            embedding_rank_lists.append(get_embedding_ranks(ranks, pp_size))
        else:
            # Default: embedding_ranks are first and last pp stage (or only one if pp_size==1)
            embedding_rank_lists.append([ranks[0]] if len(ranks) == 1 else [ranks[0], ranks[-1]])
        if get_position_embedding_ranks is not None:
            # Use custom callback to determine position embedding ranks
            pos_embedding_rank_lists.append(get_position_embedding_ranks(ranks, pp_size))
        else:
            # Default: position embedding ranks are first pp stage only
            pos_embedding_rank_lists.append([ranks[0]])
    if embedding_rank_lists:
        embd_pg, _ = torch.distributed.new_subgroups_by_enumeration(embedding_rank_lists, backend="nccl")
    if pos_embedding_rank_lists:
        pos_embd_pg, _ = torch.distributed.new_subgroups_by_enumeration(pos_embedding_rank_lists, backend="nccl")

    # Build Partial-Distributed-Optimizer groups for Expert DP when multiple instances are used.
    intra_expt_dp_pg = None
    inter_dist_opt_pg = None
    intra_dist_opt_pg = None
    if inner_dp_dim is not None and outer_dp_dim is not None:
        intra_expt_dp_pg = expert_grid.create_pg([inner_dp_dim])
        inter_dist_opt_pg = expert_grid.create_pg([outer_dp_dim])
        # Match distributed optimizer instance grouping from parallel_state:
        # combine tp-ep-pp ranks across the intra-partial DP slice.
        intra_dist_opt_pg = expert_grid.create_pg(["tp", "ep", inner_dp_dim, "pp"])

    # Build ProcessGroupCollection with available groups.
    pg_collection = ProcessGroupCollection(
        tp=tp_pg,
        pp=pp_pg,
        mp=mp_pg,
        embd=embd_pg,
        pos_embd=pos_embd_pg,
        cp=cp_pg,
        tp_cp=tp_cp_pg,
        hcp=None,
        ep=ep_pg,
        expt_tp=expt_tp_pg,
        tp_ep=tp_ep_pg,
        tp_ep_pp=tp_ep_pp_pg,
        tp_dp_cp=tp_dp_cp_pg,
        dp=dp_pg,
        dp_cp=dp_cp_pg,
        expt_dp=expt_dp_pg,
        intra_dp_cp=dp_cp_pg,
        intra_expt_dp=intra_expt_dp_pg if intra_expt_dp_pg is not None else expt_dp_pg,
        inter_dist_opt=inter_dist_opt_pg,
        intra_dist_opt=intra_dist_opt_pg,
    )
    if save_grid:
        model_config.grid = grid
    return pg_collection


def _initialize_distributed(
    model_config: TransformerConfig,
    dist_config: DistributedInitConfig,
    num_distributed_optimizer_instances: int,
    get_embedding_ranks: Callable[[list[int], int | None], list[int]] | None,
    get_position_embedding_ranks: Callable[[list[int], int | None], list[int]] | None,
    restart_store: torch.distributed.Store | None = None,
    use_inprocess_restart: bool = False,
) -> ProcessGroupCollection:
    """Initialize torch.distributed and core model parallel."""

    device_count = torch.cuda.device_count()
    if torch.distributed.is_initialized():

        print_rank_0("torch distributed is already initialized, skipping initialization ...")

    else:

        print_rank_0("> initializing torch distributed ...")
        # Manually set the device ids.
        if device_count > 0:
            if dist_config.external_gpu_device_mapping:
                torch.cuda.set_device(0)
                device_id = torch.device(f'cuda:0')
            else:
                local_rank = get_local_rank_preinit()
                torch.cuda.set_device(local_rank)
                device_id = torch.device(f'cuda:{local_rank}')
        else:
            device_id = None

        # Set to non-default stream for cudagraph capturing.
        if model_config.cuda_graph_impl == "transformer_engine":
            torch.cuda.set_stream(torch.cuda.Stream())

        # Ensure MASTER_ADDR and MASTER_PORT are set for distributed initialization
        # These may come from torchrun, SLURM, or defaults
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = get_master_addr_safe()
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(get_master_port_safe())

        _setup_flight_recorder_env(dist_config)

        # Call the init process
        init_process_group_kwargs = {
            "backend": dist_config.distributed_backend,
            "world_size": safe_get_world_size(),
            "rank": safe_get_rank(),
            "store": restart_store,
            "timeout": timedelta(minutes=dist_config.distributed_timeout_minutes),
        }
        if dist_config.fake_process_group:
            assert is_torch_min_version(
                "2.3.0"
            ), "Fake process group is only supported with PyTorch 2.3.0 and above."
            from torch.testing._internal.distributed.fake_pg import FakeStore

            store = FakeStore()
            init_process_group_kwargs['backend'] = 'fake'
            init_process_group_kwargs['store'] = store

        torch.distributed.init_process_group(**init_process_group_kwargs)

        # Force NCCL backend initialization if using in-process restart
        if use_inprocess_restart:
            inprocess_restart.force_nccl_backend_init(device_id)

        if dist_config.external_gpu_device_mapping:
            torch.distributed.barrier(device_ids=[0])
        else:
            torch.distributed.barrier(device_ids=[get_local_rank_preinit()])

    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.

    if device_count == 0:
        if dist_config.use_decentralized_pg or dist_config.distributed_backend == "nccl":
            raise RuntimeError("Cannot initialize parallel groups with no CUDA devices available (device_count=0)")

    if dist_config.use_decentralized_pg:
        # Use HyperCommGrid to create local parallel groups passed through functions
        # instead of relying on mcore's global parallel state (mpu) variables.
        mpu._set_global_memory_buffer()
        pg_collection = _create_pg_collection(
            model_config,
            num_distributed_optimizer_instances,
            get_embedding_ranks=get_embedding_ranks,
            get_position_embedding_ranks=get_position_embedding_ranks,
        )
        if safe_get_rank() == 0:
            tp = int(model_config.tensor_model_parallel_size)
            pp = int(model_config.pipeline_model_parallel_size)
            cp = int(model_config.context_parallel_size) if getattr(model_config, "context_parallel_size", 1) else 1
            dp = torch.distributed.get_world_size() // (tp * pp * cp)
            print(f"> initialized HyperCommGrid with tp={tp}, pp={pp}, cp={cp}, dp={dp}")
        return pg_collection
    else:
        # Use the original mcore parallel_state.initialize_model_parallel approach
        if mpu.model_parallel_is_initialized():
            print("model parallel is already initialized")
        else:
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=model_config.tensor_model_parallel_size,
                pipeline_model_parallel_size=model_config.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=model_config.virtual_pipeline_model_parallel_size,
                pipeline_model_parallel_comm_backend=model_config.pipeline_model_parallel_comm_backend,
                context_parallel_size=model_config.context_parallel_size,
                hierarchical_context_parallel_sizes=model_config.hierarchical_context_parallel_sizes,
                hybrid_context_parallel=model_config.hybrid_context_parallel,
                expert_model_parallel_size=model_config.expert_model_parallel_size,
                num_distributed_optimizer_instances=num_distributed_optimizer_instances,
                expert_tensor_parallel_size=model_config.expert_tensor_parallel_size,
                distributed_timeout_minutes=dist_config.distributed_timeout_minutes,
                nccl_communicator_config_path=dist_config.nccl_communicator_config_path,
                order="tp-cp-ep-dp-pp" if not dist_config.use_tp_pp_dp_mapping else "tp-cp-ep-pp-dp",
                get_embedding_ranks=get_embedding_ranks,
                get_position_embedding_ranks=get_position_embedding_ranks,
                create_gloo_process_groups=dist_config.use_gloo_process_groups,
                use_sharp=dist_config.use_sharp,
                high_priority_stream_groups=dist_config.high_priority_stream_groups,
                sharp_enabled_group=dist_config.sharp_enabled_group,
            )
            print_rank_0(
                f"> initialized tensor model parallel with size "
                f"{mpu.get_tensor_model_parallel_world_size()}"
            )
            print_rank_0(
                f"> initialized pipeline model parallel with size "
                f"{mpu.get_pipeline_model_parallel_world_size()}"
            )

        # Return a ProcessGroupCollection using mpu process groups
        return ProcessGroupCollection.use_mpu_process_groups()


def _init_autoresume():
    """Set autoresume start time."""
    autoresume = get_adlr_autoresume()
    if autoresume:
        torch.distributed.barrier()
        autoresume.init()
        torch.distributed.barrier()


def _set_random_seed(
    seed_: int,
    pg_collection: ProcessGroupCollection,
    data_parallel_random_init: bool = False,
    te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
):
    """Set random seed for reproducibility."""
    assert seed_ is not None and seed_ > 0, f"Seed ({seed_}) should be a positive integer."

    current_rank = torch.distributed.get_rank()

    # Ensure that different pipeline MP stages get different seeds.
    pp_rank = torch.distributed.get_group_rank(pg_collection.pp, current_rank)
    seed = seed_ + (100 * pp_rank)
    # Ensure different data parallel ranks get different seeds
    if data_parallel_random_init:
        dp_rank = torch.distributed.get_group_rank(pg_collection.dp, current_rank)
        seed = seed + (10 * dp_rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        # Derive TP/EP/ETP ranks from provided process groups using helper utils
        tp_rank = get_pg_rank(pg_collection.tp)
        ep_rank = get_pg_rank(pg_collection.ep)
        etp_rank = get_pg_rank(pg_collection.expt_tp)

        tensor_parallel.model_parallel_cuda_manual_seed(
            seed,
            te_rng_tracker,
            inference_rng_tracker,
            use_cudagraphable_rng,
            tp_rank=tp_rank,
            ep_rank=ep_rank,
            etp_rank=etp_rank,
        )


def write_args_to_tensorboard():
    """Write arguments to tensorboard."""
    args = get_args()
    writer = get_tensorboard_writer()
    if writer:
        for arg in vars(args):
            writer.add_text(arg, str(getattr(args, arg)), global_step=args.iteration)


def set_jit_fusion_options(
    model_config: TransformerConfig, micro_batch_size: int
) -> None:
    """Set PyTorch JIT layer fusion options and warmup JIT functions.

    Configures the JIT fuser (nvFuser or legacy) based on the PyTorch version
    and warms up common fused kernels like bias_gelu and bias_dropout_add.

    Args:
        model_config: Transformer Config
        micro_batch_size: The micro batch size used for warmup tensor shapes.
    """
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

    _warmup_jit_function(model_config, micro_batch_size)


def _warmup_jit_function(model_config: TransformerConfig, micro_batch_size: int) -> None:
    """Compilie JIT functions before the main training steps"""
    if model_config.bf16:
        dtype = torch.bfloat16
    elif model_config.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Warmup fused bias+gelu
    bias = torch.rand(
        model_config.ffn_hidden_size // model_config.tensor_model_parallel_size,
        dtype=dtype,
        device="cuda",
    )
    input = torch.rand(
        (
            model_config.seq_length // model_config.context_parallel_size,
            micro_batch_size,
            model_config.ffn_hidden_size // model_config.tensor_model_parallel_size,
        ),
        dtype=dtype,
        device="cuda",
    )
    # Warmup JIT fusions with the input grad_enable state of both forward
    # prop and recomputation
    for bias_grad, input_grad in zip([True, True], [False, True]):
        bias.requires_grad, input.requires_grad = bias_grad, input_grad
        for _ in range(5):
            if model_config.activation_func == F.silu:
                output = bias_swiglu(input, bias)
            else:
                output = bias_gelu(bias, input)
    del bias, input, output

    # Warmup fused bias+dropout+add
    if model_config.sequence_parallel:
        seq_length = model_config.seq_length // model_config.tensor_model_parallel_size
    else:
        seq_length = model_config.seq_length
    input = torch.rand(
        (
            seq_length // model_config.context_parallel_size,
            micro_batch_size,
            model_config.hidden_size,
        ),
        dtype=dtype,
        device="cuda",
    )
    residual = torch.rand(
        (
            seq_length // model_config.context_parallel_size,
            micro_batch_size,
            model_config.hidden_size,
        ),
        dtype=dtype,
        device="cuda",
    )
    bias = torch.rand((model_config.hidden_size), dtype=dtype, device="cuda").expand_as(residual)
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


def destroy_global_state() -> None:
    """Destroy Megatron global states.

    Cleans up resources used by microbatch calculator, global memory buffer,
    model parallel groups, and the rerun state machine.
    """
    from megatron.core.rerun_state_machine import destroy_rerun_state_machine
    from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator

    destroy_num_microbatches_calculator()
    mpu.destroy_global_memory_buffer()
    mpu.destroy_model_parallel()
    destroy_rerun_state_machine()


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
        if is_rank0():
            logger.info(f'Setting logging level to {logging_level}')
        logging.getLogger().setLevel(logging_level)
