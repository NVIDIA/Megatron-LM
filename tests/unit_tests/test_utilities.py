# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import contextlib
import os
import sys
import types
from argparse import Namespace
from datetime import timedelta
from typing import Literal

import torch
from torch._C._distributed_c10d import PrefixStore
from torch.distributed import rendezvous

import megatron.core.parallel_state as ps
from megatron.core import process_groups_config
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.inference import utils as inference_utils
from megatron.core.inference.utils import InferenceMode
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import random as tp_random
from megatron.core.transformer import cuda_graphs, multi_token_prediction
from megatron.training.argument_utils import (
    gpt_config_from_args,
    hybrid_config_from_args,
    pretrain_cfg_container_from_args,
)
from tools import check_process_group_usage

try:
    from transformer_engine.pytorch import distributed as te_distributed
except ImportError:
    te_distributed = None

_NVTE_ATTN_ENV_VARS = (
    'NVTE_FLASH_ATTN',
    'NVTE_FUSED_ATTN',
    'NVTE_UNFUSED_ATTN',
    'NVTE_FLASH_ATTN_V2',
    'NVTE_FLASH_ATTN_V3',
    'NVTE_FLASH_ATTN_V4',
)


class TestModel(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int,
        bias: bool,
        shared_embedding: bool = False,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(input_dim, output_dim, bias) for _ in range(num_layers)]
        )
        if shared_embedding:
            self.layers[-1].weight.shared_embedding = True


def clear_nvte_env_vars():
    """Clear NVTE attention backend environment variables."""
    for name in _NVTE_ATTN_ENV_VARS:
        os.environ.pop(name, None)


def reset_cuda_graph_global_state():
    """Reset the process-global CUDA-graph state a passing test can leave behind."""
    record = cuda_graphs._CudagraphGlobalRecord
    # TestLLaVACudaGraph is an example of where the pool leaks.
    # TestPackedSeqCudagraphs is an example that is affected by a leaked pool.

    # TestMHCWithCudaGraph is an example of where training records leak.
    # TestLocalCudagraphPipelineOutput is an example that is affected by leaked training records.
    if (
        record.cudagraph_record
        or record.cudagraph_inference_record
        or record.cudagraph_created
        or record._saved_tensors_observer is not None
        or cuda_graphs.CudaGraphManager.global_mempool is not None
    ):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        cuda_graphs.delete_cuda_graphs()


def _snapshot_torch_settings():
    return {
        "deterministic": (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
        ),
        "fill_uninitialized_memory": torch.utils.deterministic.fill_uninitialized_memory,
        "cudnn": (
            torch.backends.cudnn.deterministic,
            torch.backends.cudnn.benchmark,
            torch.backends.cudnn.allow_tf32,
        ),
        "matmul": (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        ),
    }


def _restore_torch_settings(settings):
    mode, warn_only = settings["deterministic"]
    torch.use_deterministic_algorithms(mode, warn_only=warn_only)
    torch.utils.deterministic.fill_uninitialized_memory = settings["fill_uninitialized_memory"]
    (
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.allow_tf32,
    ) = settings["cudnn"]
    (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
    ) = settings["matmul"]


def snapshot_process_state():
    """Capture process-global settings a test may change; restoring puts back the values."""
    state = {
        # Set by every engine, cleared only by suspend(); breaks TestInferenceTopKRouter.
        "inference_mode": (InferenceMode._is_active, InferenceMode._use_bounded_mxfp8_rows),
        # The first init fixes the tracker type;
        # TestPartialCudaGraph forces TE, TestMTPCudaGraphInference the no-op one.
        "rng_tracker": (
            tp_random._CUDA_RNG_STATE_TRACKER,
            tp_random._CUDA_RNG_STATE_TRACKER_INITIALIZED,
        ),
        # test_thd_format (deterministic), TestGPTModelBatchInvariant (tf32),
        # test_guard_agrees_with_config_resolution (fill_uninitialized_memory).
        "torch": _snapshot_torch_settings(),
    }
    if te_distributed is not None:
        # TestParallelAttention fills it with byte tensors;
        # test_forward_backward_func_with_full_cuda_graph then expects generators.
        state["te_rng_states"] = te_distributed._ALL_ACTIVE_RNG_STATES
    return state


def restore_process_state(state):
    """Return every item captured by `snapshot_process_state` to its captured value."""
    InferenceMode._is_active, InferenceMode._use_bounded_mxfp8_rows = state["inference_mode"]
    tp_random._CUDA_RNG_STATE_TRACKER, tp_random._CUDA_RNG_STATE_TRACKER_INITIALIZED = state[
        "rng_tracker"
    ]
    _restore_torch_settings(state["torch"])
    if "te_rng_states" in state:
        te_distributed._ALL_ACTIVE_RNG_STATES = state["te_rng_states"]


def reset_transient_process_state():
    """Drop the lazily built caches no later test may inherit."""
    reset_cuda_graph_global_state()
    # Sized at the first num_layers seen; TestMTPLossLoggingHelper reads it back.
    multi_token_prediction.MTPLossLoggingHelper.tracker.clear()
    # Built from the first model (TestMTPCudaGraphExpertParallel resets it by hand).
    inference_utils.moe_layer_cache = None
    inference_utils._moe_metadata_sync_initialized = False


def is_nccl_ep_available():
    """NCCL EP built into TE, with the ``ep_bootstrap`` signature mcore actually calls.

    A bare import probe of ``transformer_engine.pytorch.ep`` is not enough: TE releases ship the
    module with an older API (v2.17/v2.18 predate the ``num_topk`` / ``drop_on_overflow`` kwargs
    and require ``recv_capacity_per_rank``), so the import succeeds but
    ``ensure_nccl_ep_bootstrapped`` raises TypeError on the first bootstrap of every ncclEP path.
    Probe the signature instead: ``num_topk`` and ``drop_on_overflow`` must be accepted (mcore
    always passes them) and ``recv_capacity_per_rank`` must be optional (``None`` selects eager
    mode, which the over-budget replay depends on).
    """
    from megatron.core.transformer.moe.fused_a2a import HAVE_TE_EP

    if not HAVE_TE_EP:
        return False

    import inspect

    from transformer_engine.pytorch.ep import ep_bootstrap

    params = inspect.signature(ep_bootstrap).parameters
    recv_capacity = params.get("recv_capacity_per_rank")
    return (
        recv_capacity is not None
        and recv_capacity.default is None
        and "num_topk" in params
        and "drop_on_overflow" in params
    )


def is_nccl_ep_zero_copy_available():
    """Zero-copy needs the TE symm-mem APIs (symm_mem_alloc/is_symm_backed) on top of NCCL EP."""
    if not is_nccl_ep_available():
        return False
    try:
        from transformer_engine.pytorch.ep import is_symm_backed, symm_mem_alloc  # noqa: F401
    except ImportError:
        return False
    return True


def is_op_fuser_available():
    """The static-shape/zero-copy path runs the TE op-fuser grouped GEMM (needs TE>=2.14 ops)."""
    from megatron.core.utils import is_te_min_version

    try:
        from transformer_engine.pytorch.ops import GroupedLinear, ScaledSwiGLU  # noqa: F401
    except ImportError:
        return False
    return is_te_min_version("2.14.0")


def is_nccl_ep_fp8_dispatch_available():
    """MXFP8 wire dtypes need a TE build whose EpBuffer takes the quant recipes AND that returns
    the plain-tensor MXFP8 carrier (mxfp8_carrier_to_grouped, TE PR #3355 -- older quant-recipe
    builds return a GroupedTensor payload the op-fuser attrs cannot rebuild), plus MXFP8 hardware
    support (Blackwell) for the quantize kernels and the grouped GEMM."""
    if not is_nccl_ep_available():
        return False
    import inspect

    try:
        import transformer_engine.pytorch.ep as te_ep
        from transformer_engine.pytorch.fp8 import check_mxfp8_support
    except ImportError:
        return False
    if "dispatch_fwd_quant_recipe" not in inspect.signature(te_ep.EpBuffer).parameters:
        return False
    if not hasattr(te_ep, "mxfp8_carrier_to_grouped"):
        return False
    return check_mxfp8_support()[0]


class Utils:

    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    # Global rank, which LOCAL_RANK only matches when the launch is confined to one
    # node. Reading LOCAL_RANK here makes every node claim ranks 0..local_size-1 of
    # the rendezvous, so multi-node runs hang on duplicate check-ins.
    rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0')))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    inited = False
    store = None

    @staticmethod
    def initialize_distributed():
        clear_nvte_env_vars()
        if torch.cuda.is_available():
            # Also when another test already created the default group without binding one.
            torch.cuda.set_device(Utils.local_rank % torch.cuda.device_count())

        if not torch.distributed.is_initialized() and Utils.rank >= 0:
            print(
                f'Initializing torch.distributed with rank: {Utils.rank}, '
                f'world_size: {Utils.world_size}'
            )
            init_method = 'tcp://'
            master_ip = os.getenv('MASTER_ADDR', 'localhost')
            master_port = os.getenv('MASTER_PORT', '29500')
            init_method += master_ip + ':' + master_port
            rendezvous_iterator = rendezvous(
                init_method, Utils.rank, Utils.world_size, timeout=timedelta(minutes=1)
            )
            store, rank, world_size = next(rendezvous_iterator)
            store.set_timeout(timedelta(minutes=1))

            # Use a PrefixStore to avoid accidental overrides of keys used by
            # different systems (e.g. RPC) in case the store is multi-tenant.
            store = PrefixStore("default_pg", store)
            Utils.store = store

            torch.distributed.init_process_group(
                backend='nccl', world_size=Utils.world_size, rank=Utils.rank, store=store
            )

            torch.distributed.barrier()
        Utils.inited = True

    @staticmethod
    def set_world_size(world_size=None, rank=None):
        Utils.world_size = torch.cuda.device_count() if world_size is None else world_size
        if (
            torch.distributed.is_initialized()
            and Utils.world_size != torch.distributed.get_world_size()
        ):
            torch.distributed.destroy_process_group()

        if rank is None:
            Utils.rank = int(os.environ.get('RANK', os.environ['LOCAL_RANK']))
            if Utils.rank >= Utils.world_size:
                Utils.rank = -1
        else:
            Utils.rank = rank

    @staticmethod
    def destroy_model_parallel():
        clear_nvte_env_vars()
        if not Utils.inited:
            return

        try:
            # Flush pending CUDA work before the barrier so slow ranks don't
            # time out while fast ranks tear down process groups.
            torch.cuda.synchronize()
            torch.distributed.barrier()
        except Exception:
            Utils.inited = False
            return
        ps.destroy_model_parallel()
        Utils.inited = False
        torch.cuda.memory.empty_cache()

    @staticmethod
    def initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        **kwargs,
    ):
        # Need to unset these variables to make sure previous
        # tests setting them doesn't interfere current test.
        clear_nvte_env_vars()

        ps.destroy_model_parallel()
        Utils.initialize_distributed()
        ps.initialize_model_parallel(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size,
            **kwargs,
        )
        Utils.inited = True

    @staticmethod
    def pretrain_config_from_global_args(args: Namespace, model_class: Literal["gpt", "hybrid"]):
        if model_class == "gpt":
            model_cfg = gpt_config_from_args(args)
        elif model_class == "hybrid":
            model_cfg = hybrid_config_from_args(args)
        else:
            raise ValueError(
                f"MCore model type {model_class} not supported. Choose one of 'gpt' or 'hybrid'."
            )

        return pretrain_cfg_container_from_args(args, model_cfg)

    @staticmethod
    def fake_initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        expert_model_parallel_size=1,
    ):
        """Used for layer-wise UT as a proxy for NeMo-style intialization."""
        ps.set_tensor_model_parallel_world_size(tensor_model_parallel_size)
        ps.set_tensor_model_parallel_rank(0)

        ps.set_expert_model_parallel_world_size(expert_model_parallel_size)
        ps.set_expert_model_parallel_rank(0)
        if virtual_pipeline_model_parallel_size is not None:
            ps.set_virtual_pipeline_model_parallel_world_size(virtual_pipeline_model_parallel_size)
        ps.set_virtual_pipeline_model_parallel_rank(0)

        ps.set_pipeline_model_parallel_world_size(pipeline_model_parallel_size)


# Pipeline-stage predicates read the global grid, but the process-group checker does not count them.
_STAGE_PREDICATES = ("is_pipeline_first_stage", "is_pipeline_last_stage")


def _rebind_in_megatron_modules(replacements):
    """In every imported megatron module, replace each value whose id is a key of `replacements`."""
    modules = [
        module
        for name, module in list(sys.modules.items())
        if name.split(".")[0] == "megatron" and isinstance(module, types.ModuleType)
    ]
    for module in modules:
        for attr, value in list(vars(module).items()):
            if id(value) in replacements:
                setattr(module, attr, replacements[id(value)])


@contextlib.contextmanager
def forbid_global_process_groups(*, allow=(), forbid_shim=True, forbid_stage_predicates=True):
    """Make every read of the global parallel grid raise AssertionError inside the block.

    Covers each parallel_state accessor that tools/check_process_group_usage.py counts, both on
    parallel_state and wherever an imported megatron module bound it by name (for example
    `from megatron.core.parallel_state import get_tensor_model_parallel_group`). Calls made by
    parallel_state itself still work, so initialize_model_parallel, destroy_model_parallel,
    is_initialized and get_global_memory_buffer stay usable. Everything is restored on exit.

    A `group=None` passed to torch.distributed means WORLD and reads no accessor; to catch it,
    give the component a grid whose layout differs from the global one.

    Args:
        allow: Names of accessors or stage predicates that may still be called.
        forbid_shim: Also forbid `ProcessGroupCollection.use_mpu_process_groups()`. When False,
            code in process_groups_config.py (the shim and its fallbacks) may read the grid.
        forbid_stage_predicates: Also forbid `is_pipeline_first_stage` and
            `is_pipeline_last_stage`.
    """
    accessors = [
        name for name in vars(ps) if check_process_group_usage._is_deprecated_accessor(name)
    ]
    unknown = set(allow) - set(accessors) - set(_STAGE_PREDICATES)
    if unknown:
        raise ValueError(f"not a parallel_state accessor or stage predicate: {sorted(unknown)}")
    names = accessors + list(_STAGE_PREDICATES if forbid_stage_predicates else ())
    originals = {name: vars(ps)[name] for name in names if name not in allow}
    # Code in these modules may read the grid, like the files the checker exempts.
    exempt_modules = {id(vars(ps))}
    if not forbid_shim:
        exempt_modules.add(id(vars(process_groups_config)))
    active = True

    def forbidden(label, original):
        def read_global_grid(*args, **kwargs):
            caller = sys._getframe(1)
            # The stub of a nested block delegates to this one; judge the call by its caller.
            while caller.f_code is read_global_grid.__code__:
                caller = caller.f_back
            # A reference taken inside the block keeps working after it.
            if active and id(caller.f_globals) not in exempt_modules:
                raise AssertionError(f"{label} reads the global parallel grid")
            return original(*args, **kwargs)

        return read_global_grid

    stubs = {name: forbidden(f"parallel_state.{name}()", fn) for name, fn in originals.items()}
    _rebind_in_megatron_modules({id(originals[name]): stubs[name] for name in originals})
    shim = vars(ProcessGroupCollection)["use_mpu_process_groups"]
    if forbid_shim:
        ProcessGroupCollection.use_mpu_process_groups = classmethod(
            forbidden("ProcessGroupCollection.use_mpu_process_groups()", shim.__func__)
        )
    try:
        yield
    finally:
        active = False
        ProcessGroupCollection.use_mpu_process_groups = shim
        # Also restores the names bound by modules first imported inside the block.
        _rebind_in_megatron_modules({id(stubs[name]): originals[name] for name in originals})


def new_group_with_same_ranks(group):
    """Return a new process group over the ranks of `group`.

    The result is a different communicator with the same ranks, so a test can tell whether a
    component used the group it was given or read the global one. Creating groups is collective:
    call this on every rank, each with its own group of one kind (its TP group, for example), or
    with None on a rank outside every such group, which then gets None back. Destroy the result
    with torch.distributed.destroy_process_group.
    """
    ranks = None if group is None else torch.distributed.get_process_group_ranks(group)
    ranks_of_every_rank = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(ranks_of_every_rank, ranks)
    # Every rank creates every group, in the same order.
    rank_lists = sorted({tuple(r) for r in ranks_of_every_rank if r is not None})
    new_group, _ = torch.distributed.new_subgroups_by_enumeration([list(r) for r in rank_lists])
    return new_group


def build_test_pg_collection(*, tp=1, pp=1, cp=1, ep=1, expt_tp=None, order="tp-cp-ep-dp-pp"):
    """Build the ProcessGroupCollection of a standard grid without the parallel_state globals.

    The groups have the same ranks as the ones initialize_model_parallel creates for the same
    sizes and order, with GTP-remat off and one distributed-optimizer instance. In that
    configuration the GTP-remat and partial data-parallel fields span the same ranks as the plain
    groups, so they reuse them: dp_gtp_remat, dp_cp_gtp_remat, intra_dp_cp, expt_dp_gtp_remat,
    intra_expt_dp and tp_ep_pp_with_egtp_remat. The fields it does not build are set to None
    explicitly, so no resolver falls back to the global grid for an absent field: hcp, gtp_remat,
    expt_gtp_remat, inter_dist_opt, dp_cp_ag and expt_dp_ag.

    Creating groups is collective: call this on every rank, and release the groups with
    destroy_test_pg_collection.

    Args:
        tp, pp, cp, ep: Tensor, pipeline, context and expert parallel sizes.
        expt_tp: Expert tensor parallel size; defaults to `tp`.
        order: Rank order, as for initialize_model_parallel.
    """
    expt_tp = tp if expt_tp is None else expt_tp
    world_size = torch.distributed.get_world_size()
    if world_size % (tp * cp * pp) or world_size % (expt_tp * ep * pp):
        raise ValueError(
            f"world size {world_size} must be divisible by tp*cp*pp ({tp * cp * pp}) "
            f"and by expt_tp*ep*pp ({expt_tp * ep * pp})"
        )
    dims = order.split("-")
    if sorted(dims) != sorted(["tp", "cp", "ep", "dp", "pp"]):
        raise ValueError(f"order must arrange tp, cp, ep, dp and pp, got {order!r}")
    sizes = {
        "tp": tp,
        "cp": cp,
        "dp": world_size // (tp * cp * pp),
        "pp": pp,
        "expt_tp": expt_tp,
        "ep": ep,
        "expt_dp": world_size // (expt_tp * ep * pp),
    }
    # Dense layers have no ep axis and expert layers no cp axis, as in initialize_model_parallel.
    dense_dims = [dim for dim in dims if dim != "ep"]
    expert_dims = [{"tp": "expt_tp", "dp": "expt_dp"}.get(dim, dim) for dim in dims if dim != "cp"]
    grid = HyperCommGrid([sizes[dim] for dim in dense_dims], dense_dims)
    # The expert layers factor the same ranks differently but share the pipeline groups.
    grid.register_view(
        "expert", [sizes[dim] for dim in expert_dims], expert_dims, shared_dims=["pp"]
    )

    dp = grid.create_pg("dp")
    dp_cp = grid.create_pg(["cp", "dp"])
    expt_dp = grid.create_pg("expt_dp", view="expert")
    tp_ep_pp = grid.create_pg(["expt_tp", "ep", "pp"], view="expert")
    # Every rank creates every embedding group, in the same order; other ranks get None.
    pp_rank_lists = grid.get_rank_enum("pp")
    embd, _ = torch.distributed.new_subgroups_by_enumeration(
        [ps.default_embedding_ranks(ranks) for ranks in pp_rank_lists]
    )
    pos_embd, _ = torch.distributed.new_subgroups_by_enumeration(
        [ps.default_position_embedding_ranks(ranks) for ranks in pp_rank_lists]
    )
    return ProcessGroupCollection(
        tp=grid.create_pg("tp"),
        cp=grid.create_pg("cp"),
        pp=grid.create_pg("pp"),
        mp=grid.create_pg(["tp", "pp"]),
        tp_cp=grid.create_pg(["tp", "cp"]),
        tp_dp_cp=grid.create_pg(["tp", "cp", "dp"]),
        dp=dp,
        dp_gtp_remat=dp,
        dp_cp=dp_cp,
        dp_cp_gtp_remat=dp_cp,
        intra_dp_cp=dp_cp,
        embd=embd,
        pos_embd=pos_embd,
        ep=grid.create_pg("ep", view="expert"),
        expt_tp=grid.create_pg("expt_tp", view="expert"),
        tp_ep=grid.create_pg(["expt_tp", "ep"], view="expert"),
        tp_ep_pp=tp_ep_pp,
        tp_ep_pp_with_egtp_remat=tp_ep_pp,
        expt_dp=expt_dp,
        expt_dp_gtp_remat=expt_dp,
        intra_expt_dp=expt_dp,
        intra_dist_opt=grid.create_pg(dense_dims),
        hcp=None,
        gtp_remat=None,
        expt_gtp_remat=None,
        inter_dist_opt=None,
        dp_cp_ag=None,
        expt_dp_ag=None,
    )


def destroy_test_pg_collection(pg_collection):
    """Destroy this rank's groups of a build_test_pg_collection result; call on every rank."""
    # Every rank finishes its collectives before any communicator goes away.
    torch.cuda.synchronize()
    torch.distributed.barrier()
    groups = {id(group): group for group in vars(pg_collection).values() if group is not None}
    for group in groups.values():
        torch.distributed.destroy_process_group(group)
