# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Experimental BF16 Torch FSDP2 adapter for the separate-rank MIMO recipe.

This is an experiment adapter, not a general MCore backend. It uses native
PyTorch fully_shard for every parameter collective. TP/PP, checkpoint resume,
CPU offload, FP8 and precision-aware optimizers are deliberately unsupported.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.utils._pytree import tree_map_only

try:
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.fsdp._fully_shard._fsdp_common import FSDPMeshInfo, ShardPlacementResult
except ImportError:
    MixedPrecisionPolicy = fully_shard = FSDPMeshInfo = ShardPlacementResult = None

from megatron.core.distributed.data_parallel_base import _BaseDataParallel
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.distributed.torch_fully_sharded_data_parallel import (
    TorchFullyShardedDataParallel,
)
from megatron.core.optimizer.optimizer import MegatronOptimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig


def _backward_boundary(value: torch.Tensor) -> torch.Tensor:
    if not value.requires_grad:
        return value
    result = value.clone()
    # MIMO's bridge communicator reads uneven split metadata from the tensor.
    for name, attribute in vars(value).items():
        setattr(result, name, attribute)
    return result


def plan_units(module: torch.nn.Module) -> tuple[list[torch.nn.Module], list[torch.nn.Module]]:
    """Select called execution regions; never shard children of a shortcut pair.

    The hybrid decoder and MTP HybridStacks expose their actual execution regions
    through ``layers``. Embedding/output/tied MTP weights stay owned by the root
    MIMO stage, whose forward encloses their early/direct uses.
    """
    from megatron.core.models.hybrid.hybrid_block import HybridStack
    from megatron.core.transformer.moe.moe_layer import MoELayer

    dense = []
    for child in module.modules():
        if isinstance(child, HybridStack):
            dense.extend(child.layers)
    experts = [child.experts for child in module.modules() if isinstance(child, MoELayer)]
    return list(dict.fromkeys(dense)), list(dict.fromkeys(experts))


class TorchMimoFSDP(TorchFullyShardedDataParallel):
    """Shard a full local MIMO stage, using separate dense and expert meshes."""

    def _dtype_placement(self, mesh, parameters):
        """Use native per-PG groups without splitting a called/replay region.

        Torch requires homogeneous original dtypes per FSDPParamGroup. Its
        per-parameter mesh API groups by process-group identity, so each extra
        dtype gets a distinct PG with exactly the original ranks/order. All
        gathering, resharding and gradient reduction remain native FSDP.
        """
        dtypes = {p.dtype for p in parameters}
        if len(dtypes) <= 1:
            return None
        if not dtypes <= {torch.bfloat16, torch.float32}:
            raise ValueError(f"Unsupported original parameter dtypes: {dtypes}")
        base_pg = mesh.get_group()
        key = (base_pg, torch.float32)
        if key not in self._dtype_mesh_infos:
            ranks = torch.distributed.get_process_group_ranks(base_pg)
            group = torch.distributed.new_group(ranks, use_local_synchronization=True)
            if torch.distributed.get_process_group_ranks(group) != ranks:
                raise RuntimeError("Dtype group changed the original parameter shard rank order")
            dtype_mesh = DeviceMesh.from_group(group, "cuda")
            self._dtype_mesh_infos[key] = FSDPMeshInfo(dtype_mesh, shard_mesh_dim=0)
        fp32_info = self._dtype_mesh_infos[key]

        def placement(parameter):
            return (
                ShardPlacementResult(None, fp32_info) if parameter.dtype == torch.float32 else None
            )

        return placement

    def __init__(
        self,
        config: TransformerConfig,
        ddp_config: DistributedDataParallelConfig,
        module: torch.nn.Module,
        pg_collection: ProcessGroupCollection,
        *,
        units: tuple[Sequence[torch.nn.Module], Sequence[torch.nn.Module]] | None = None,
    ) -> None:
        if fully_shard is None or ShardPlacementResult is None:
            raise RuntimeError("Torch MIMO requires FSDP2 per-parameter mesh placement support.")
        if config.tensor_model_parallel_size != 1 or config.pipeline_model_parallel_size != 1:
            raise ValueError("The Torch MIMO comparison requires TP=PP=1.")
        if config.gradient_accumulation_fusion or config.fp16 or getattr(config, "fp8", None):
            raise ValueError("The Torch MIMO comparison requires BF16 without fused accumulation.")
        if ddp_config.use_megatron_fsdp or ddp_config.use_distributed_optimizer:
            raise ValueError("Torch FSDP2 cannot share parameter ownership with MFSDP/distopt.")
        _BaseDataParallel.__init__(self, config=config, module=module)
        self.ddp_config = ddp_config
        self.pg_collection = pg_collection
        self.dense_mesh = DeviceMesh.from_group(pg_collection.dp_cp, "cuda")
        self.expert_mesh = DeviceMesh.from_group(pg_collection.expt_dp, "cuda")
        self._dtype_mesh_infos: dict[
            tuple[torch.distributed.ProcessGroup, torch.dtype], FSDPMeshInfo
        ] = {}
        self.units = []
        self.ownership = []
        self._forward_boundary_hooks = []
        dense, experts = plan_units(module) if units is None else units
        names = {id(child): name for name, child in module.named_modules()}
        # Capture attributes before fully_shard replaces Parameters with DTensors.
        attrs = {name: dict(vars(p)) for name, p in module.named_parameters()}
        expert_params = {p for child in experts for p in child.parameters()}
        expected = {p for p in module.parameters() if not getattr(p, "allreduce", True)}
        if expert_params != expected:
            raise ValueError(
                "Every expert parameter must belong to a called MoELayer.experts unit."
            )
        if any(not p.is_cuda for p in module.parameters()):
            raise ValueError("Load pretrained weights onto CUDA before Torch FSDP2 wrapping.")
        for child in experts:
            if any(getattr(p, "allreduce", True) for p in child.parameters()):
                raise ValueError("Expert units must contain only expert parameters.")
        claimed: set[int] = set()
        # Native FSDP excludes parameters already owned by a nested FSDP module.
        # Experts must be registered first so dense block gathers cannot own them.
        for child, mesh, kind in [
            *((c, self.expert_mesh, "expert") for c in experts),
            *((c, self.dense_mesh, "dense") for c in dense),
            (module, self.dense_mesh, "root"),
        ]:
            owned = [p for p in child.parameters() if id(p) not in claimed]
            if kind != "root" and not owned:
                continue
            claimed.update(id(p) for p in owned)
            placement = self._dtype_placement(mesh, owned)

            # Residual replay registers hooks inside the unit's forward. Give
            # FSDP a later autograd boundary so its pre-backward gather runs
            # before those hooks replay parameter reads. Register before
            # fully_shard so its post-forward hook sees these new outputs.
            def clone_grad_outputs(_module, _args, result):
                return tree_map_only(torch.Tensor, _backward_boundary, result)

            self._forward_boundary_hooks.append(child.register_forward_hook(clone_grad_outputs))
            fully_shard(
                child,
                mesh=mesh,
                reshard_after_forward=True,
                shard_placement_fn=placement,
                mp_policy=MixedPrecisionPolicy(
                    reduce_dtype=torch.float32 if ddp_config.grad_reduce_in_fp32 else None
                ),
            )
            claimed.update(id(p) for p in child.parameters())
            groups = fully_shard.state(child)._fsdp_param_groups
            group_dtypes = []
            for group in groups:
                dtypes = {p.sharded_param.dtype for p in group.fsdp_params}
                if len(dtypes) != 1:
                    raise RuntimeError(
                        f"Mixed original dtypes remain in native FSDP group: {dtypes}"
                    )
                group_dtypes.append(str(next(iter(dtypes))))
            # Expert dispatch already sums over EP source ranks. Both meshes
            # divide by dense DP×CP for mean loss, or by 1 for token-summed loss.
            child.set_gradient_divide_factor(
                1.0 if config.calculate_per_token_loss else float(self.dense_mesh.size())
            )
            # Avoid inferred prefetch through bypassed shortcut children/replay.
            child.set_modules_to_backward_prefetch([])
            self.units.append(child)
            self.ownership.append(
                {
                    "name": names[id(child)],
                    "kind": kind,
                    "mesh": mesh.size(),
                    "native_group_dtypes": group_dtypes,
                }
            )
        for name, p in module.named_parameters():
            for attr, value in attrs[name].items():
                setattr(p, attr, value)
            p.is_torch_fsdp2_param = True
            if not isinstance(p, DTensor):
                raise RuntimeError(f"Unsharded parameter after wrapping: {name}")
        output = os.environ.get("TORCH_MIMO_FSDP_AUDIT_DIR")
        if output:
            path = Path(output)
            path.mkdir(parents=True, exist_ok=True)
            receipt = {
                "rank": torch.distributed.get_rank(),
                "backend": "torch.distributed.fsdp.fully_shard",
                "torch": torch.__version__,
                "dense_ranks": torch.distributed.get_process_group_ranks(pg_collection.dp_cp),
                "expert_ranks": torch.distributed.get_process_group_ranks(pg_collection.expt_dp),
                "fp32_reductions": ddp_config.grad_reduce_in_fp32,
                "calculate_per_token_loss": config.calculate_per_token_loss,
                "units": self.ownership,
                "parameters": [
                    {
                        "name": name,
                        "shape": list(p.shape),
                        "local_shape": list(p.to_local().shape),
                        "dtype": str(p.dtype),
                        "expert": not getattr(p, "allreduce", True),
                    }
                    for name, p in module.named_parameters()
                ],
            }
            with (path / f"rank-{receipt['rank']}.json").open("x") as handle:
                json.dump(receipt, handle, indent=2)

    @contextmanager
    def no_sync(self):
        """Reduce each microbatch to keep only sharded accumulated gradients.

        This matches MCore's existing Torch FSDP2 wrapper's no_sync behavior.
        Disabling reduction would retain full dense gradients over CP=64.
        """
        yield

    @torch.no_grad()
    def scale_gradients(self, scaling_factor):
        for p in self.module.parameters():
            if p.grad is not None:
                p.grad.to_local().mul_(scaling_factor)

    def zero_grad_buffer(self):
        for p in self.module.parameters():
            p.grad = None

    def finish_grad_sync(self, force_all_reduce=False):
        # FSDP's root post-backward callback completes the shard reductions.
        pass


class TorchMimoOptimizer(MegatronOptimizer):
    """MIMO optimizer contract with TE Adam on local FP32 master shards.

    DTensor optimizer dispatch cannot mix dense and expert meshes in fused
    multi-tensor calls. Local tensors also allow the global norm to sum each
    shard exactly once over the language grid, without mixing vision ranks.
    """

    def __init__(
        self,
        model: TorchMimoFSDP,
        config: OptimizerConfig,
        stats_group: torch.distributed.ProcessGroup,
        param_group_process_group: torch.distributed.ProcessGroup,
    ) -> None:
        from transformer_engine.pytorch.optimizers import FusedAdam

        from megatron.core.optimizer import _get_param_groups, get_standard_config_overrides

        if (
            config.optimizer != "adam"
            or config.use_precision_aware_optimizer
            or config.use_distributed_optimizer
            or config.use_layer_wise_distributed_optimizer
            or config.optimizer_cpu_offload
            or config.fp16
            or config.loss_scale
        ):
            raise ValueError(
                "Torch MIMO comparison supports resident BF16 Adam with FP32 masters only."
            )
        groups = _get_param_groups(
            [model], config, get_standard_config_overrides(config), param_group_process_group
        )
        self.model = model
        self.model_to_master = []
        for group in groups:
            group["weight_decay"] = config.weight_decay * group.get("wd_mult", 1.0)
            masters = []
            for p in group["params"]:
                local = p.to_local()
                # Older TE versions do not reliably update groups ending in an
                # empty shard. Empty shards have no optimizer state to maintain.
                if local.numel() == 0:
                    continue
                master = torch.nn.Parameter(local.detach().float().clone())
                self.model_to_master.append((p, master))
                masters.append(master)
            group["params"] = masters
        optimizer = FusedAdam(
            groups,
            lr=config.lr,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
        )
        super().__init__(optimizer, config)
        self.is_stub_optimizer = False
        self.grad_stats_parallel_group = stats_group
        self._loss_scale = torch.ones(1, device="cuda")

    @torch.no_grad()
    def prepare_grads(self):
        # Reduction dtype does not control stored sharded-gradient precision.
        # Native FSDP2 casts to the original parameter dtype before accumulating;
        # converting here cannot recover precision already lost in BF16 gradients.
        for p, master in self.model_to_master:
            master.grad = None if p.grad is None else p.grad.to_local().float()
        return False

    @torch.no_grad()
    def get_grad_norm(self):
        norm_sq = torch.zeros((), device="cuda", dtype=torch.float32)
        for _, master in self.model_to_master:
            if master.grad is not None:
                norm_sq += master.grad.square().sum()
        torch.distributed.all_reduce(norm_sq, group=self.grad_stats_parallel_group)
        return norm_sq.sqrt()

    @torch.no_grad()
    def count_zeros(self):
        count = torch.zeros((), device="cuda", dtype=torch.int64)
        for _, master in self.model_to_master:
            if master.grad is not None:
                count += (master.grad == 0).sum()
        torch.distributed.all_reduce(count, group=self.grad_stats_parallel_group)
        return count.item()

    @torch.no_grad()
    def step_with_ready_grads(self):
        self.optimizer.step()
        for p, master in self.model_to_master:
            p.to_local().copy_(master)
        return True

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
        self.model.zero_grad_buffer()

    def get_loss_scale(self):
        return self._loss_scale

    @torch.no_grad()
    def reload_model_params(self, state_dict=None):
        if state_dict is not None:
            raise NotImplementedError("Torch MIMO comparison does not support checkpoint resume.")
        for p, master in self.model_to_master:
            master.copy_(p.to_local())

    def state_dict(self):
        raise NotImplementedError("Torch MIMO comparison does not support checkpoint save.")

    def sharded_state_dict(self, *args, **kwargs):
        raise NotImplementedError("Torch MIMO comparison does not support checkpoint save.")

    @torch.no_grad()
    def step(self):
        self.prepare_grads()
        norm = self.get_grad_norm()
        if self.config.clip_grad > 0:
            from megatron.core.optimizer.clip_grads import clip_grad_by_total_norm_fp32

            clip_grad_by_total_norm_fp32(self.get_parameters(), self.config.clip_grad, norm)
        return self.step_with_ready_grads(), norm, None

    def load_state_dict(self, state_dict):
        raise NotImplementedError("Torch MIMO comparison does not support checkpoint resume.")
