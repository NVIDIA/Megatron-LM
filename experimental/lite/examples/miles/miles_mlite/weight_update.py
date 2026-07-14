# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Raw HF weight update bridge for miles rollout engines."""

from __future__ import annotations

import logging
import importlib
from dataclasses import replace
from collections.abc import Sequence

import ray
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class RawHFWeightUpdater:
    """Send MLite-exported HF-format weights through miles update APIs."""

    def __init__(self, args, runtime, handle) -> None:
        self.args = args
        self.runtime = runtime
        self.handle = handle
        self.weight_version = 0
        self._ipc_gather_group = None
        self._ipc_gather_src = None
        self._ipc_engine = None
        self._model_update_groups = None
        self._distributed_group_name = None
        self.rollout_engines = []
        self.distributed_rollout_engines = []
        self.use_distribute = False

    @property
    def _ps(self):
        return self.handle._parallel_state

    @property
    def _is_distributed_src_rank(self) -> bool:
        ps = self._ps
        return (
            int(getattr(ps, "dp_rank", 0) or 0) == 0
            and int(getattr(ps, "cp_rank", 0) or 0) == 0
            and int(getattr(ps, "tp_rank", 0) or 0) == 0
        )

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence,
        rollout_engine_lock,
        *,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        broadcast = importlib.import_module(
            "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast"
        )
        connect_rollout_engines_from_distributed = broadcast.connect_rollout_engines_from_distributed
        disconnect_rollout_engines_from_distributed = broadcast.disconnect_rollout_engines_from_distributed

        self.rollout_engines = list(rollout_engines)
        self.rollout_engine_lock = rollout_engine_lock

        if engine_gpu_counts is None:
            engine_gpu_counts = [self.args.rollout_num_gpus_per_engine] * len(rollout_engines)
        if engine_gpu_offsets is None:
            engine_gpu_offsets = []
            offset = 0
            for count in engine_gpu_counts:
                engine_gpu_offsets.append(offset)
                offset += count

        if not getattr(self.args, "colocate", False):
            colocate_engine_nums = 0
        else:
            total_actor_gpus = self.args.actor_num_nodes * self.args.actor_num_gpus_per_node
            colocate_engine_nums = 0
            for gpu_offset, gpu_count in zip(engine_gpu_offsets, engine_gpu_counts, strict=True):
                if gpu_offset + gpu_count > total_actor_gpus:
                    break
                colocate_engine_nums += 1

        self.use_distribute = len(rollout_engines) > colocate_engine_nums
        self.distributed_rollout_engines = list(rollout_engines[colocate_engine_nums:])
        self.rollout_engines = list(rollout_engines[:colocate_engine_nums])

        if self.use_distribute and self._is_distributed_src_rank:
            self._distributed_group_name = f"mlite-pp_{getattr(self._ps, 'pp_rank', 0)}"
            if self._model_update_groups is not None:
                disconnect_rollout_engines_from_distributed(
                    self.args,
                    self._distributed_group_name,
                    self._model_update_groups,
                    self.distributed_rollout_engines,
                )
            self._model_update_groups = connect_rollout_engines_from_distributed(
                self.args,
                self._distributed_group_name,
                self.distributed_rollout_engines,
                engine_gpu_counts=engine_gpu_counts[colocate_engine_nums:],
            )

        self._connect_colocated_groups(engine_gpu_counts[:colocate_engine_nums], engine_gpu_offsets[:colocate_engine_nums])

    @property
    def _active_rollout_engines(self):
        return [*self.rollout_engines, *self.distributed_rollout_engines]

    def _connect_colocated_groups(self, engine_gpu_counts, engine_gpu_offsets) -> None:
        rank = dist.get_rank()
        self._ipc_gather_group = None
        self._ipc_gather_src = None
        self._ipc_engine = None

        for engine_idx, (offset, count) in enumerate(zip(engine_gpu_offsets, engine_gpu_counts, strict=True)):
            group_ranks = list(range(offset, offset + count))
            new_group = dist.new_group(ranks=group_ranks, backend="gloo")
            if rank in group_ranks:
                self._ipc_gather_group = new_group
                self._ipc_gather_src = offset
                self._ipc_engine = self.rollout_engines[engine_idx]

    def update_weights(self) -> None:
        common = importlib.import_module("miles.backends.megatron_utils.update_weight.common")
        broadcast = importlib.import_module(
            "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast"
        )
        distributed_utils = importlib.import_module("miles.utils.distributed_utils")
        sglang_utils = importlib.import_module("miles.backends.megatron_utils.sglang")
        _check_weight_sync_results = common._check_weight_sync_results
        post_process_weights = common.post_process_weights
        update_weights_from_distributed = broadcast.update_weights_from_distributed
        get_gloo_group = distributed_utils.get_gloo_group
        monkey_patch_torch_reductions = sglang_utils.monkey_patch_torch_reductions

        self.weight_version += 1
        monkey_patch_torch_reductions()
        rank = dist.get_rank()
        if rank == 0:
            mode = self.args.pause_generation_mode
            ray.get([engine.pause_generation.remote(mode=mode) for engine in self._active_rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self._active_rollout_engines])
        dist.barrier(group=get_gloo_group())

        refs = []
        live_tensors = []
        for chunk in self._export_weight_chunks():
            colocated_refs, long_lived = _send_to_colocated_engine_direct(
                hf_named_tensors=chunk,
                ipc_engine=self._ipc_engine,
                ipc_gather_src=self._ipc_gather_src,
                ipc_gather_group=self._ipc_gather_group,
                weight_version=self.weight_version,
            )
            refs.extend(colocated_refs)
            if long_lived is not None:
                live_tensors.append(long_lived)
            if self.use_distribute and self._is_distributed_src_rank:
                refs.extend(
                    update_weights_from_distributed(
                        self._distributed_group_name,
                        self._model_update_groups,
                        self.weight_version,
                        self.distributed_rollout_engines,
                        chunk,
                    )
                )
        if refs:
            _check_weight_sync_results(ray.get(refs), is_lora=False)
        del live_tensors

        dist.barrier(group=get_gloo_group())
        if rank == 0:
            post_process_weights(
                rollout_engines=self._active_rollout_engines,
                restore_weights_before_load=False,
                post_process_quantization=True,
            )
            ray.get([engine.continue_generation.remote() for engine in self._active_rollout_engines])
        dist.barrier(group=get_gloo_group())

    def _export_weight_chunks(self):
        chunk = []
        chunk_bytes = 0
        limit = int(getattr(self.args, "update_weight_buffer_size", 2**30))
        export_kwargs = {}
        if getattr(self.args, "mlite_export_dtype", None):
            export_kwargs["export_dtype"] = self.args.mlite_export_dtype
        for name, tensor in self._iter_local_hf_weights(**export_kwargs):
            if not self._needs_weight_payload:
                del tensor
                continue
            tensor = tensor.detach()
            if not tensor.is_cuda:
                tensor = tensor.to(device=torch.cuda.current_device(), non_blocking=True)
            item_bytes = tensor.numel() * tensor.element_size()
            if chunk and chunk_bytes + item_bytes > limit:
                torch.cuda.synchronize()
                yield chunk
                chunk = []
                chunk_bytes = 0
            chunk.append((name, tensor))
            chunk_bytes += item_bytes
        if chunk:
            torch.cuda.synchronize()
            yield chunk

    @property
    def _needs_weight_payload(self) -> bool:
        return self._is_distributed_src_rank or self._ipc_gather_group is not None

    def _iter_local_hf_weights(self, **export_kwargs):
        """Export this PP stage only; each PP source broadcasts its own group."""

        ps = self._ps
        if int(getattr(ps, "pp_size", 1) or 1) <= 1:
            yield from self.runtime.export_weights(self.handle, **export_kwargs)
            return

        proto = self.handle._extras.get("protocol")
        model_cfg = self.handle._extras.get("model_cfg")
        model_chunks = self.handle._extras.get("model_chunks", [self.handle._model])
        if proto and hasattr(proto, "export_hf_weights"):
            local_pp_ps = replace(
                ps,
                pp_group=None,
                pp_cpu_group=None,
                pp_global_ranks=None,
                pp_size=1,
                pp_rank=0,
                pp_is_first=True,
                pp_is_last=True,
                pp_next_rank=-1,
                pp_prev_rank=-1,
            )
            yield from proto.export_hf_weights(model_chunks, model_cfg, local_pp_ps, **export_kwargs)
            return

        for chunk in model_chunks:
            yield from chunk.named_parameters()


def _send_to_colocated_engine_direct(
    hf_named_tensors,
    *,
    ipc_engine,
    ipc_gather_src,
    ipc_gather_group,
    weight_version=None,
):
    if ipc_gather_group is None:
        return [], None

    sglang_utils = importlib.import_module("miles.backends.megatron_utils.sglang")
    model_runner = importlib.import_module("sglang.srt.model_executor.model_runner")
    MultiprocessingSerializer = sglang_utils.MultiprocessingSerializer
    LocalSerializedTensor = model_runner.LocalSerializedTensor

    is_gather_src = dist.get_rank() == ipc_gather_src
    serialized_tensors = [
        (name, MultiprocessingSerializer.serialize(tensor)) for name, tensor in hf_named_tensors
    ]
    serialized_named_tensors = [None] * dist.get_world_size(ipc_gather_group) if is_gather_src else None
    dist.gather_object(
        serialized_tensors,
        object_gather_list=serialized_named_tensors,
        dst=ipc_gather_src,
        group=ipc_gather_group,
    )

    refs = []
    if is_gather_src:
        named_tensors = []
        for tensor_group in zip(*serialized_named_tensors, strict=True):
            names = {name for name, _ in tensor_group}
            if len(names) != 1:
                raise RuntimeError(f"Mismatched TP tensor names during weight sync: {sorted(names)}")
            name = tensor_group[0][0]
            named_tensors.append(
                (name, LocalSerializedTensor(values=[rank_part[1] for rank_part in tensor_group]))
            )
        payload = [
            MultiprocessingSerializer.serialize(named_tensors, output_str=True)
            for _ in range(len(serialized_named_tensors))
        ]
        refs.append(
            ipc_engine.update_weights_from_tensor.remote(
                serialized_named_tensors=payload,
                weight_version=str(weight_version),
            )
        )

    return refs, [tensor for _, tensor in hf_named_tensors]
