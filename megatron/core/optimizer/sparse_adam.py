# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Touched-row Adam for Engram row-sharded embedding tables."""

import math
import pickle
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import torch

from megatron.core.dist_checkpointing.mapping import LocalNonpersistentObject, ShardedObject
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.engram.memory import row_shard_owner

from .optimizer import MegatronOptimizer
from .optimizer_config import OptimizerConfig

_CHECKPOINT_BUCKETS = 256


def _get_row_table_metadata(param):
    """Return row-parallel Engram table metadata attached to ``param``."""
    metadata = getattr(param, 'engram_table_metadata', None)
    if metadata is None or not metadata.row_parallel:
        raise ValueError("RowSparseAdam requires Engram row-table parameter metadata")
    return metadata


def _detach_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """Return one independent CPU copy of ``tensor``."""
    detached = tensor.detach()
    if detached.device.type == 'cpu':
        return detached.clone()
    return detached.to(device='cpu', copy=True)


def _copy_bytes_to_uint8(dest: torch.Tensor, payload: bytes) -> None:
    """Write pickle bytes into a preallocated CPU ``uint8`` slice."""
    if dest.device.type != 'cpu' or dest.dtype != torch.uint8:
        raise ValueError("expected a CPU uint8 destination")
    if dest.numel() != len(payload):
        raise ValueError(f"destination has {dest.numel()} bytes, payload has {len(payload)}")
    dest.numpy()[:] = np.frombuffer(payload, dtype=np.uint8)


def _loads_uint8(chunk: torch.Tensor):
    """Unpickle a CPU ``uint8`` view without an extra ``tobytes()`` copy."""
    contiguous = chunk.detach().contiguous()
    return pickle.loads(memoryview(contiguous.numpy()))


def _all_to_all_objects(send_objects, group):
    """Exchange one picklable object per peer without global replication."""
    world_size = torch.distributed.get_world_size(group=group)
    if len(send_objects) != world_size:
        raise ValueError(f"Expected {world_size} all-to-all objects, got {len(send_objects)}")
    if world_size == 1:
        return send_objects

    backend = str(torch.distributed.get_backend(group)).lower()
    use_cuda = 'nccl' in backend
    comm_device = (
        torch.device('cuda', torch.cuda.current_device()) if use_cuda else torch.device('cpu')
    )

    # Length pass keeps only sizes so the full pickled batch never stays resident.
    send_splits = []
    for value in send_objects:
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        send_splits.append(len(payload))
        del payload

    send_lengths = torch.tensor(send_splits, dtype=torch.int64, device=comm_device)
    recv_lengths = torch.empty_like(send_lengths)
    torch.distributed.all_to_all_single(recv_lengths, send_lengths, group=group)
    recv_splits = recv_lengths.cpu().tolist()
    del send_lengths, recv_lengths

    send_cpu = torch.empty(sum(send_splits), dtype=torch.uint8)
    offset = 0
    for index, value in enumerate(send_objects):
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        if len(payload) != send_splits[index]:
            raise RuntimeError("pickle size changed between length and payload passes")
        _copy_bytes_to_uint8(send_cpu[offset : offset + send_splits[index]], payload)
        offset += send_splits[index]
        del payload
        send_objects[index] = None

    if use_cuda:
        send_payload = send_cpu.to(comm_device)
        del send_cpu
    else:
        send_payload = send_cpu

    recv_payload = torch.empty(sum(recv_splits), dtype=torch.uint8, device=comm_device)
    torch.distributed.all_to_all_single(
        recv_payload,
        send_payload,
        output_split_sizes=recv_splits,
        input_split_sizes=send_splits,
        group=group,
    )
    del send_payload

    if recv_payload.device.type != 'cpu':
        recv_cpu = recv_payload.cpu()
        del recv_payload
    else:
        recv_cpu = recv_payload

    result = []
    offset = 0
    for length in recv_splits:
        result.append(_loads_uint8(recv_cpu[offset : offset + length]))
        offset += length
    del recv_cpu
    return result


class RowSparseAdam(torch.optim.Optimizer):
    """Adam with append-only touched-row state and amortized storage growth."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | list[dict],
        lr: float,
        betas: tuple[float, float],
        eps: float,
    ) -> None:
        """Initialize Adam hyperparameters for touched-row table parameters."""
        if lr < 0 or eps < 0 or any(not 0 <= beta < 1 for beta in betas):
            raise ValueError("Adam requires non-negative lr/eps and betas in [0, 1)")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=0.0))

    @staticmethod
    def _capacity_for(required: int, current: int = 0) -> int:
        """Return the next power-of-two storage size that fits ``required`` rows."""
        capacity = max(16, current)
        while capacity < required:
            capacity *= 2
        return capacity

    @staticmethod
    def _refresh_active_views(state):
        """Point public state tensors at the live prefix of each storage buffer."""
        active = state['num_active']
        state['rows'] = state['_rows_storage'][:active]
        state['exp_avg'] = state['_exp_avg_storage'][:active]
        state['exp_avg_sq'] = state['_exp_avg_sq_storage'][:active]
        state['step'] = state['_step_storage'][:active]
        if '_master_param_storage' in state:
            state['master_param'] = state['_master_param_storage'][:active]

    def install_compact_state(
        self, param: torch.nn.Parameter, compact_state: dict[str, Any]
    ) -> None:
        """Install the compact touched-row checkpoint representation."""
        active = int(compact_state.get('rows', torch.empty(0)).numel())
        capacity = self._capacity_for(active)
        width = param.shape[1]
        state = self.state[param]
        state.clear()
        state['num_active'] = active
        state['_capacity'] = capacity
        state['_rows_storage'] = torch.empty(capacity, dtype=torch.long, device=param.device)
        state['_exp_avg_storage'] = torch.zeros(
            (capacity, width), dtype=torch.float32, device=param.device
        )
        state['_exp_avg_sq_storage'] = torch.zeros_like(state['_exp_avg_storage'])
        state['_step_storage'] = torch.zeros(capacity, dtype=torch.int64, device=param.device)
        if param.dtype == torch.bfloat16:
            state['_master_param_storage'] = torch.empty(
                (capacity, width), dtype=torch.float32, device=param.device
            )
        if active:
            state['_rows_storage'][:active].copy_(compact_state['rows'].to(param.device))
            state['_exp_avg_storage'][:active].copy_(compact_state['exp_avg'].to(param.device))
            state['_exp_avg_sq_storage'][:active].copy_(
                compact_state['exp_avg_sq'].to(param.device)
            )
            state['_step_storage'][:active].copy_(compact_state['step'].to(param.device))
            if param.dtype == torch.bfloat16:
                master = compact_state.get('master_param')
                if master is None:
                    master = param.data[state['_rows_storage'][:active]].float()
                state['_master_param_storage'][:active].copy_(master.to(param.device))
        self._refresh_active_views(state)
        self._ensure_row_slot_map(param, state)

    def _ensure_row_slot_map(self, param, state):
        """GPU table-row -> optimizer-slot map. Unseen rows are -1."""
        if '_row_slot_map' in state and state['_row_slot_map'].numel() == param.shape[0]:
            return state['_row_slot_map']
        slot_map = torch.full((param.shape[0],), -1, dtype=torch.long, device=param.device)
        active = int(state['num_active'])
        if active:
            slot_map[state['_rows_storage'][:active]] = torch.arange(
                active, device=param.device, dtype=torch.long
            )
        state['_row_slot_map'] = slot_map
        return slot_map

    def _ensure_runtime_state(self, param):
        """Materialize compact storage and the GPU row-to-slot map if missing."""
        state = self.state[param]
        if '_rows_storage' not in state:
            self.install_compact_state(param, dict(state))
            state = self.state[param]
        self._ensure_row_slot_map(param, state)
        return state

    def _ensure_capacity(self, param, required: int):
        """Grow amortized moment storage so at least ``required`` rows fit."""
        state = self._ensure_runtime_state(param)
        current = state['_capacity']
        if required <= current:
            return state
        active = state['num_active']
        new_capacity = self._capacity_for(required, current)
        state['_capacity'] = new_capacity
        width = param.shape[1]
        for public_key, storage_key, dtype in (
            ('rows', '_rows_storage', torch.long),
            ('exp_avg', '_exp_avg_storage', torch.float32),
            ('exp_avg_sq', '_exp_avg_sq_storage', torch.float32),
            ('step', '_step_storage', torch.int64),
        ):
            shape = (new_capacity,) if public_key in ('rows', 'step') else (new_capacity, width)
            storage = torch.zeros(shape, dtype=dtype, device=param.device)
            storage[:active].copy_(state[public_key])
            state[storage_key] = storage
        if param.dtype == torch.bfloat16:
            master = torch.empty((new_capacity, width), dtype=torch.float32, device=param.device)
            master[:active].copy_(state['master_param'])
            state['_master_param_storage'] = master
        self._refresh_active_views(state)
        return state

    def compact_state(self, param: torch.nn.Parameter) -> dict[str, torch.Tensor]:
        """Return a clone of only the active touched-row state."""
        state = self.state.get(param, {})
        if not state:
            return {}
        if '_rows_storage' not in state:
            keys = ('rows', 'exp_avg', 'exp_avg_sq', 'step', 'master_param')
            return {key: state[key].clone() for key in keys if key in state}
        active = state['num_active']
        keys = ('rows', 'exp_avg', 'exp_avg_sq', 'step', 'master_param')
        return {key: state[key][:active].clone() for key in keys if key in state}

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        """Apply Adam to the coalesced sparse rows touched on this step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for param in group['params']:
                if param.grad is None:
                    continue
                if param.ndim != 2:
                    raise RuntimeError("RowSparseAdam requires two-dimensional table parameters")
                if not param.grad.is_sparse:
                    raise RuntimeError("Engram row sparse Adam requires sparse COO gradients")
                grad = param.grad.coalesce()
                rows = grad.indices()[0]
                values = grad.values().float()
                state = self._ensure_runtime_state(param)
                slot_map = state['_row_slot_map']
                touched_slots = slot_map[rows]
                new_mask = touched_slots < 0
                n_new = int(new_mask.sum().item())
                if n_new:
                    new_rows = rows[new_mask]
                    start = state['num_active']
                    state = self._ensure_capacity(param, start + n_new)
                    slot_map = state['_row_slot_map']
                    new_slots = torch.arange(
                        start, start + n_new, device=param.device, dtype=torch.long
                    )
                    state['_rows_storage'][start : start + n_new] = new_rows
                    if param.dtype == torch.bfloat16:
                        state['_master_param_storage'][start : start + n_new].copy_(
                            param.data[new_rows].float()
                        )
                    slot_map[new_rows] = new_slots
                    state['num_active'] = start + n_new
                    self._refresh_active_views(state)
                    touched_slots = slot_map[rows]
                exp_avg = state['_exp_avg_storage']
                exp_avg_sq = state['_exp_avg_sq_storage']
                steps = state['_step_storage']
                exp_avg[touched_slots] = exp_avg[touched_slots] * beta1 + values * (1.0 - beta1)
                exp_avg_sq[touched_slots] = exp_avg_sq[touched_slots] * beta2 + values * values * (
                    1.0 - beta2
                )
                steps[touched_slots] += 1

                if param.dtype == torch.bfloat16:
                    master = state['_master_param_storage']
                    current = master[touched_slots]
                else:
                    master = None
                    current = param.data[rows].float()

                row_steps = steps[touched_slots].float()
                bias_correction1 = 1.0 - torch.pow(beta1, row_steps)
                bias_correction2 = 1.0 - torch.pow(beta2, row_steps)
                denom = exp_avg_sq[touched_slots].sqrt()
                denom.div_(bias_correction2.sqrt().unsqueeze(1)).add_(group['eps'])
                update = exp_avg[touched_slots] / bias_correction1.unsqueeze(1)
                current.addcdiv_(update, denom, value=-group['lr'])
                param.data.index_copy_(0, rows, current.to(param.dtype))
                if master is not None:
                    master[touched_slots] = current
                self._refresh_active_views(state)
        return loss

    def state_dict(self) -> dict[str, Any]:
        """Return optimizer state with only the compact touched-row tensors."""
        result = super().state_dict()
        id_to_param: dict[int, torch.nn.Parameter] = {}
        for group, saved_group in zip(self.param_groups, result['param_groups']):
            id_to_param.update(zip(saved_group['params'], group['params']))
        result['state'] = {
            param_id: self.compact_state(id_to_param[param_id]) for param_id in result['state']
        }
        return result

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore compact state and reinstall FP32 moments after torch's cast."""
        super().load_state_dict(state_dict)
        # torch.optim.Optimizer casts floating-point state to the parameter dtype
        # while loading. Touched-row moments and BF16 master weights are always FP32,
        # so reinstall them from the original compact payload instead of the cast copy.
        saved_states = state_dict['state']
        param_to_saved_state = {}
        for group, saved_group in zip(self.param_groups, state_dict['param_groups']):
            for param, param_id in zip(group['params'], saved_group['params']):
                if param_id in saved_states:
                    param_to_saved_state[param] = saved_states[param_id]
        self.state.clear()
        for param, compact_state in param_to_saved_state.items():
            self.install_compact_state(param, dict(compact_state))


def _local_rows_to_coordinates(param, local_rows: torch.Tensor) -> torch.Tensor:
    """Map local row ids to (layer, table, global-row) checkpoint coordinates."""
    metadata = _get_row_table_metadata(param)
    coords = torch.empty((local_rows.numel(), 3), dtype=torch.int64, device=local_rows.device)
    coords[:, 0] = int(metadata.layer_id)
    for table_index, ((start, end), local_offset) in enumerate(
        zip(metadata.local_bounds, metadata.local_offsets)
    ):
        count = end - start
        mask = (local_rows >= local_offset) & (local_rows < local_offset + count)
        coords[mask, 1] = table_index
        coords[mask, 2] = start + local_rows[mask] - local_offset
    return coords


def _coordinate_bucket(coords: torch.Tensor) -> torch.Tensor:
    """Hash checkpoint coordinates into a fixed number of DCP buckets."""
    return torch.remainder(
        coords[:, 0] * 1_000_003 + coords[:, 1] * 9_176 + coords[:, 2], _CHECKPOINT_BUCKETS
    )


class RowSparseAdamOptimizer(MegatronOptimizer):
    """Megatron optimizer adapter with table-aware norm and DCP state handling."""

    uses_custom_grad_norm = True

    def __init__(
        self,
        param_groups: list[dict],
        config: OptimizerConfig,
        pg_collection: ProcessGroupCollection | None = None,
    ) -> None:
        """Wrap ``RowSparseAdam`` and cache the table-group communication handles."""
        optimizer = RowSparseAdam(
            param_groups,
            lr=config.lr,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
        )
        super().__init__(optimizer, config)
        self.is_stub_optimizer = False
        parameters = self.get_parameters()
        metadata = _get_row_table_metadata(parameters[0]) if parameters else None
        if pg_collection is None and metadata is not None:
            pg_collection = ProcessGroupCollection(
                tp=metadata.tp_group,
                mp=metadata.model_parallel_group,
                tp_dp_cp=metadata.table_group,
            )
        if pg_collection is None:
            # Bootstrap for standalone optimizers, including table-free PP stages.
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(["tp", "mp", "tp_dp_cp"])
        self.table_group = pg_collection.tp_dp_cp
        self.tp_group = pg_collection.tp
        self.grad_stats_parallel_group = pg_collection.mp
        for parameter in parameters:
            parameter_metadata = _get_row_table_metadata(parameter)
            if parameter_metadata.table_group is not self.table_group:
                raise ValueError("Sparse optimizer parameters must share one table process group")
        device = parameters[0].device if parameters else torch.cuda.current_device()
        self._scale = torch.ones(1, dtype=torch.float32, device=device)

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear sparse table gradients on the inner optimizer."""
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def get_loss_scale(self) -> torch.Tensor:
        """Return the constant FP32 scale used by BF16 training."""
        return self._scale

    def reload_model_params(self, state_dict: dict | None = None) -> None:
        """Refresh active BF16 master rows after model-only checkpoint loading."""
        del state_dict
        for param in self.get_parameters():
            state = self.optimizer.state.get(param)
            if state and param.dtype == torch.bfloat16:
                state['master_param'].copy_(param.data[state['rows']].float())

    @torch.no_grad()
    def prepare_grads(self) -> bool:
        """Coalesce sparse grads and propagate nonfinite values to every optimizer rank."""
        found_nonfinite = torch.zeros_like(self._scale)
        for param in self.get_parameters():
            if param.grad is not None:
                if not param.grad.is_sparse:
                    raise RuntimeError("row_a2a table gradient entered a dense buffer")
                param.grad = param.grad.coalesce()
                found_nonfinite.add_((~torch.isfinite(param.grad.values())).any())
        # Tables bypass DDP's dense gradient validation. All row owners and PP
        # stages, including table-free stages, must make the same skip decision.
        for group in (self.table_group, self.get_grad_stats_parallel_group()):
            torch.distributed.all_reduce(
                found_nonfinite, op=torch.distributed.ReduceOp.MAX, group=group
            )
        return bool(found_nonfinite.item())

    def get_main_grads_for_grad_norm(self) -> list[torch.Tensor]:
        """Exclude sparse table grads from the dense multi-tensor norm path."""
        return []

    def get_grads_for_grad_norm(self, grad_norm_group: str | None = None) -> list[torch.Tensor]:
        """Sparse COO table grads are not valid inputs to dense multi-tensor norms."""
        del grad_norm_group
        return []

    @torch.no_grad()
    def get_grad_norm(self) -> float:
        """Return the globally reduced L2 norm of the sparse table gradients."""
        norm_sq = self.get_extra_grad_norm_squared()
        torch.distributed.all_reduce(norm_sq, group=self.get_grad_stats_parallel_group())
        return math.sqrt(norm_sq.item())

    @torch.no_grad()
    def get_extra_grad_norm_squared(self) -> torch.Tensor:
        """Sum squared COO values and reduce them across the table group."""
        total = torch.zeros_like(self._scale)
        for param in self.get_parameters():
            if param.grad is not None:
                values = param.grad.coalesce().values().float()
                total.add_(torch.sum(values * values))
        torch.distributed.all_reduce(
            total, op=torch.distributed.ReduceOp.SUM, group=self.table_group
        )
        if torch.distributed.get_rank(group=self.tp_group) != 0:
            total.zero_()
        return total

    @torch.no_grad()
    def clip_grad_by_total_norm(self, max_norm: float, total_norm: float) -> None:
        """Scale sparse COO values by the chained optimizer's total-norm coefficient."""
        coefficient = max_norm / (total_norm + 1.0e-6)
        if coefficient >= 1.0:
            return
        for param in self.get_parameters():
            if param.grad is not None:
                grad = param.grad.coalesce()
                grad.values().mul_(coefficient)
                param.grad = grad

    @torch.no_grad()
    def count_zeros(self) -> int:
        """Count zero values in touched-row gradients without densifying them."""
        total = torch.zeros(1, dtype=torch.int64, device=self._scale.device)
        for param in self.get_parameters():
            if param.grad is not None:
                values = param.grad.coalesce().values()
                total.add_(values.numel() - torch.count_nonzero(values))
        torch.distributed.all_reduce(
            total, op=torch.distributed.ReduceOp.SUM, group=self.table_group
        )
        if torch.distributed.get_rank(group=self.tp_group) != 0:
            total.zero_()
        torch.distributed.all_reduce(total, group=self.get_grad_stats_parallel_group())
        return total.item()

    @torch.no_grad()
    def step_with_ready_grads(self) -> bool:
        """Apply the inner touched-row Adam update."""
        self.optimizer.step()
        return True

    @torch.no_grad()
    def step(self) -> tuple[bool, float | None, int | None]:
        """Clip sparse grads by their own norm, then step when used standalone."""
        if self.prepare_grads():
            return False, None, None
        norm_sq = self.get_extra_grad_norm_squared()
        torch.distributed.all_reduce(norm_sq, group=self.get_grad_stats_parallel_group())
        norm = math.sqrt(norm_sq.item())
        if self.config.clip_grad > 0:
            self.clip_grad_by_total_norm(self.config.clip_grad, norm)
        num_zeros = self.count_zeros() if self.config.log_num_zeros_in_grad else None
        self.step_with_ready_grads()
        return True, norm, num_zeros

    def state_dict(self) -> dict[str, Any]:
        """Serialize compact per-parameter state and optimizer group metadata."""
        states = []
        for param in self.get_parameters():
            states.append(self.optimizer.compact_state(param))
        groups = [
            {key: value for key, value in group.items() if key != 'params'}
            for group in self.optimizer.param_groups
        ]
        return {'state': states, 'param_groups': groups}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load either DCP bucket payloads or a compact per-parameter state dict."""
        if 'buckets' in state_dict:
            self._load_bucket_state(state_dict['buckets'])
            for group in self.optimizer.param_groups:
                saved = [
                    state_dict['layer_param_groups'][int(_get_row_table_metadata(param).layer_id)]
                    for param in group['params']
                ]
                if saved:
                    if any(item != saved[0] for item in saved[1:]):
                        raise ValueError(
                            "Restored layers disagree on sparse optimizer hyperparameters"
                        )
                    group.update(saved[0])
            return
        for param, saved in zip(self.get_parameters(), state_dict['state']):
            compact_state = {key: value.to(param.device) for key, value in saved.items()}
            self.optimizer.install_compact_state(param, compact_state)
        for group, saved_group in zip(self.optimizer.param_groups, state_dict['param_groups']):
            group.update(saved_group)

    def _local_checkpoint_records(self) -> list[dict]:
        """Collect CPU copies of local touched-row moments for DCP routing."""
        records = []
        for param in self.get_parameters():
            state = self.optimizer.state.get(param)
            if not state or state['rows'].numel() == 0:
                continue
            record = {
                'coordinates': _detach_to_cpu(_local_rows_to_coordinates(param, state['rows'])),
                'exp_avg': _detach_to_cpu(state['exp_avg']),
                'exp_avg_sq': _detach_to_cpu(state['exp_avg_sq']),
                'step': _detach_to_cpu(state['step']),
            }
            if 'master_param' in state:
                record['master_param'] = _detach_to_cpu(state['master_param'])
            records.append(record)
        return records

    def _route_checkpoint_records(self):
        """Route each fixed checkpoint bucket to exactly one table-group rank."""
        group = self.table_group
        world_size = torch.distributed.get_world_size(group=group)
        send_objects = [[] for _ in range(world_size)]
        local_records = self._local_checkpoint_records()
        sent_rows = 0
        coordinates = bucket_ids = mask = piece = None
        for record in local_records:
            coordinates = record['coordinates']
            bucket_ids = _coordinate_bucket(coordinates)
            layer = int(coordinates[0, 0])
            for bucket in torch.unique(bucket_ids).tolist():
                mask = bucket_ids == bucket
                piece = {key: value[mask] for key, value in record.items()}
                send_objects[bucket % world_size].append((layer, bucket, piece))
                sent_rows += int(mask.sum())
            record.clear()
        del local_records, coordinates, bucket_ids, mask, piece

        received = _all_to_all_objects(send_objects, group)
        del send_objects
        assigned = {}
        received_rows = 0
        for source_pieces in received:
            for layer, bucket, piece in source_pieces:
                assigned.setdefault((layer, bucket), []).append(piece)
                received_rows += piece['coordinates'].shape[0]
        del received
        self._checkpoint_routing_stats = {
            'save_sent_rows': sent_rows,
            'save_received_rows': received_rows,
            'save_received_buckets': len(assigned),
        }
        return assigned

    def sharded_state_dict(
        self,
        model_sharded_state_dict: dict,
        is_loading: bool = False,
        metadata: dict | None = None,
        **kwargs: Any,
    ) -> dict:
        """Build fixed logical-row buckets, ignoring DistOpt-only forwarded kwargs."""
        del model_sharded_state_dict, metadata, kwargs
        group = self.table_group
        rank = torch.distributed.get_rank(group=group)
        world_size = torch.distributed.get_world_size(group=group)
        layers = sorted(
            {int(_get_row_table_metadata(param).layer_id) for param in self.get_parameters()}
        )
        assigned = {} if is_loading else self._route_checkpoint_records()
        buckets: dict[int, dict[int, ShardedObject]] = {layer: {} for layer in layers}
        for layer in layers:
            for bucket in range(rank, _CHECKPOINT_BUCKETS, world_size):
                data = self._merge_record_pieces(assigned.pop((layer, bucket), []))
                buckets[layer][bucket] = ShardedObject(
                    f'optimizer.engram_sparse.layer_{layer}.buckets',
                    data,
                    (_CHECKPOINT_BUCKETS,),
                    (bucket,),
                    replica_id=0,
                )
        del assigned
        layer_param_groups = {}
        for param_group in self.optimizer.param_groups:
            group_metadata = {key: value for key, value in param_group.items() if key != 'params'}
            for param in param_group['params']:
                layer_id = int(_get_row_table_metadata(param).layer_id)
                layer_param_groups[layer_id] = ShardedObject(
                    f'optimizer.engram_sparse.layer_{layer_id}.param_group',
                    group_metadata,
                    (1,),
                    (0,),
                    replica_id=rank,
                )
        # Native extraction drops empty containers. Preserve the chain entry on
        # pipeline stages without a consumer, without persisting rank-local state.
        return {
            'buckets': buckets if buckets else LocalNonpersistentObject(buckets),
            'layer_param_groups': (
                layer_param_groups
                if layer_param_groups
                else LocalNonpersistentObject(layer_param_groups)
            ),
        }

    @staticmethod
    def _merge_record_pieces(pieces: list[dict]) -> dict:
        """Concatenate and sort one checkpoint bucket's record pieces."""
        if not pieces:
            return {
                'coordinates': torch.empty((0, 3), dtype=torch.int64),
                'exp_avg': torch.empty((0, 0), dtype=torch.float32),
                'exp_avg_sq': torch.empty((0, 0), dtype=torch.float32),
                'step': torch.empty(0, dtype=torch.int64),
            }
        result = {key: torch.cat([piece[key] for piece in pieces]) for key in pieces[0]}
        pieces.clear()
        coords = result['coordinates']
        order = torch.argsort(coords[:, 0] * 2**48 + coords[:, 1] * 2**40 + coords[:, 2])
        ordered = {key: value[order] for key, value in result.items()}
        del result
        return ordered

    def _load_bucket_state(self, buckets):
        """Route DCP buckets back to owning ranks and install compact state."""
        local_data = []
        for layer_buckets in buckets.values():
            local_data.extend(layer_buckets.values())
        group = self.table_group
        world_size = torch.distributed.get_world_size(group=group)
        by_layer = {
            int(_get_row_table_metadata(param).layer_id): param for param in self.get_parameters()
        }
        send_objects = [[] for _ in range(world_size)]
        sent_rows = 0

        for bucket in local_data:
            coordinates = bucket['coordinates']
            if coordinates.numel() == 0:
                continue
            destinations = [
                row_shard_owner(
                    _get_row_table_metadata(by_layer[int(coord[0])]).table_sizes[int(coord[1])],
                    int(coord[2]),
                    world_size,
                )
                for coord in coordinates
            ]
            for destination in set(destinations):
                indices = torch.tensor(
                    [index for index, owner in enumerate(destinations) if owner == destination],
                    dtype=torch.long,
                )
                send_objects[destination].append(
                    {key: value[indices] for key, value in bucket.items()}
                )
                sent_rows += indices.numel()
            bucket.clear()

        received = _all_to_all_objects(send_objects, group)
        del send_objects, local_data
        owner_records = [record for source_records in received for record in source_records]
        del received
        self._checkpoint_routing_stats = {
            **getattr(self, '_checkpoint_routing_stats', {}),
            'load_sent_rows': sent_rows,
            'load_received_rows': sum(record['coordinates'].shape[0] for record in owner_records),
        }
        for layer, param in by_layer.items():
            metadata = _get_row_table_metadata(param)
            pieces = []
            for record in owner_records:
                coords = record['coordinates']
                if coords.numel() == 0:
                    continue
                layer_mask = coords[:, 0] == layer
                if layer_mask.any():
                    pieces.append({key: value[layer_mask] for key, value in record.items()})
            merged = self._merge_record_pieces(pieces)
            local_rows = []
            keep_indices = []
            for index, coord in enumerate(merged['coordinates']):
                table = int(coord[1])
                row = int(coord[2])
                start, end = metadata.local_bounds[table]
                if start <= row < end:
                    local_rows.append(metadata.local_offsets[table] + row - start)
                    keep_indices.append(index)
            indices = torch.tensor(keep_indices, dtype=torch.long)
            rows = torch.tensor(local_rows, dtype=torch.long, device=param.device)
            order = torch.argsort(rows)
            state = {'rows': rows[order]}
            for key in ('exp_avg', 'exp_avg_sq', 'step', 'master_param'):
                if key in merged:
                    state[key] = merged[key][indices].to(param.device)[order]
            self.optimizer.install_compact_state(param, state)
