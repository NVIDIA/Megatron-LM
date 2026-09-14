# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import os
from argparse import Namespace
from datetime import timedelta
from typing import Literal

import torch
from torch._C._distributed_c10d import PrefixStore
from torch.distributed import rendezvous

import megatron.core.parallel_state as ps
from megatron.training.argument_utils import (
    gpt_config_from_args,
    hybrid_config_from_args,
    pretrain_cfg_container_from_args,
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
    """Clear NVTE env vars set by conftest set_env fixture."""
    os.environ.pop('NVTE_FLASH_ATTN', None)
    os.environ.pop('NVTE_FUSED_ATTN', None)
    os.environ.pop('NVTE_UNFUSED_ATTN', None)


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

        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)

        if not torch.distributed.is_initialized() and Utils.rank >= 0:
            print(
                f'Initializing torch.distributed with rank: {Utils.rank}, '
                f'world_size: {Utils.world_size}'
            )
            torch.cuda.set_device(Utils.local_rank % torch.cuda.device_count())
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
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)
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
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)

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
