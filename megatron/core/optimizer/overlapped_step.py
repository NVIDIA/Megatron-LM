# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Distributed optimizer step issued bucket group by bucket group, overlapped with the forward."""

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from .. import tensor_parallel
from ..distributed.param_and_grad_buffer import shard_buffer
from ..full_cuda_graph import get_shared_capture_stream
from ..transformer.module import param_is_not_shared
from ..utils import get_attr_wrapped_model
from .clip_grads import l2_norm_impl, multi_tensor_applier
from .optimizer import _get_param_grad_norm_group

try:
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.optimizers import FusedAdam
except ImportError:
    tex = None
    FusedAdam = None

if TYPE_CHECKING:
    from .distrib_optimizer import DistributedOptimizer

logger = logging.getLogger(__name__)

# Hyperparameters of a param group that are arguments of the Adam kernel, not device tensors.
HParams = Tuple[float, float, float, float, int]


class _ParamGroupScalars:
    """Device scalars read by the Adam kernels of one param group, and its kernel arguments."""

    def __init__(self, hparams: HParams):
        self.lr = torch.zeros(1, dtype=torch.float32, device='cuda')
        self.step = torch.zeros(1, dtype=torch.int32, device='cuda')
        self.hparams = hparams


class _FlatShard:
    """One bucket shard's buffers as the flat Adam kernel takes them."""

    def __init__(self, grad, param, exp_avg, exp_avg_sq, master_param, seg_starts, seg_groups):
        self.grad = grad
        self.param = param
        self.exp_avg = exp_avg
        self.exp_avg_sq = exp_avg_sq
        self.master_param = master_param
        self.seg_starts = seg_starts
        self.seg_groups = seg_groups


class _BucketGroupWork:
    """The kernels one bucket group's update and grad-norm partial run on."""

    def __init__(self, index: int):
        self.index = index
        # Reduce-scattered grad shards that count towards the gradient norm.
        self.norm_grads: List[torch.Tensor] = []
        # (param group index, Adam kernel) -> aligned tensor lists for multi_tensor_applier.
        self.adam_calls: Dict[Tuple[int, object], List[List[torch.Tensor]]] = {}
        # Bucket shards updated by one flat Adam launch each.
        self.flat_shards: List[_FlatShard] = []


def _hparams(group: dict) -> HParams:
    return (
        float(group['betas'][0]),
        float(group['betas'][1]),
        float(group['eps']),
        float(group['weight_decay']),
        1 if group.get('bias_correction', True) else 0,
    )


class OverlappedOptimizerStep:
    """Issues a DistributedOptimizer's step bucket group by bucket group on a side stream.

    Enabled by ``OptimizerConfig.overlap_optimizer_step_with_forward``. The step is split along
    the DDP bucket groups and hooked into their communication:

    * ``grad_reduce_finished`` (``DistributedDataParallel.finish_grad_sync``, end of backward):
      the squared L2 norm of the bucket group's reduce-scattered grad shard is taken on the side
      stream while the remaining reduce-scatters drain.
    * ``optimizer.step()``: the partials are all-reduced, the gradient clipping coefficient is
      written into the ``inv_scale`` tensor the Adam kernels read, the learning rate and step
      tensors are filled and the update is scheduled (``noop`` flag off).
    * ``dispatch_param_gather`` (the DDP forward pre-hooks of the next iteration): the capturable
      TransformerEngine Adam of the bucket group runs on the side stream, its grad buffer is
      zeroed and its param all-gather started, in the order the forward first needs the bucket
      groups and ``prefetch`` bucket groups ahead of the one being waited for.

    Everything the forward and backward issue is a fixed kernel sequence on static tensors, so a
    full-iteration CUDA graph captures it with the model; ``step()`` itself is one all-reduce of a
    float and a few scalar fills. Hyperparameters other than the learning rate are arguments of
    the launched kernels and must not change once a graph has been captured.
    """

    def __init__(self, optimizer: 'DistributedOptimizer'):
        self.optimizer = optimizer
        self.config = optimizer.config
        self.prefetch = self.config.overlap_optimizer_step_prefetch
        self._check_supported()

        # Same priority as the compute stream: the update yields to the forward through its
        # dispatch points; a lower priority starves it behind the forward's kernels and exposes
        # the all-gathers it precedes. Under a full-iteration CUDA graph the compute stream is
        # the shared capture stream, whose priority may be raised (overlap_wgrad_compute).
        model_config = optimizer.model_chunks[0].config
        self.stream = torch.cuda.Stream(
            priority=(
                get_shared_capture_stream().priority
                if model_config.cuda_graph_impl == "full_iteration"
                else 0
            )
        )
        self.noop = torch.ones(1, dtype=torch.int32, device='cuda')
        self.inv_scale = torch.ones(1, dtype=torch.float32, device='cuda')
        self._dummy_overflow_buf = torch.zeros(1, dtype=torch.int32, device='cuda')
        self._groups = [_ParamGroupScalars(_hparams(g)) for g in optimizer.optimizer.param_groups]
        self._adam_w_mode = int(optimizer.optimizer.adam_w_mode)

        # Bucket groups in DDP order; the forward's first-use order is recorded in the first
        # iteration and drives the dispatch from the second one on.
        owned = set()
        for group in optimizer.model_float16_groups + optimizer.model_fp32_groups:
            for model_param in group:
                for model_chunk in optimizer.model_chunks:
                    bucket_group = model_chunk.param_to_bucket_group.get(model_param)
                    if bucket_group is not None:
                        owned.add(bucket_group)
        self.bucket_groups = [
            bucket_group
            for model_chunk in optimizer.model_chunks
            for bucket_group in model_chunk.bucket_groups
            + model_chunk.expert_parallel_bucket_groups
            if bucket_group in owned
        ]
        for bucket_group in self.bucket_groups:
            bucket_group.overlapped_step = self
        self._partials = torch.zeros(len(self.bucket_groups), dtype=torch.float32, device='cuda')
        self._work: Optional[Dict[object, _BucketGroupWork]] = None
        self._order: List[object] = []
        self._position: Dict[object, int] = {}
        self._order_known = False
        self._cursor = 0
        self._dispatched = set()
        self._norm_done = set()
        self._update_scheduled = False
        self.grads_pending_update = False
        self._captured_hparams: Optional[List[HParams]] = None

    def _check_supported(self):
        optimizer = self.optimizer
        ddp_config = optimizer.ddp_config
        model_config = optimizer.model_chunks[0].config
        unmet = []
        if FusedAdam is None or not isinstance(optimizer.optimizer, FusedAdam):
            unmet.append("TransformerEngine's FusedAdam as the inner optimizer")
        elif optimizer.optimizer.capturable or optimizer.optimizer.master_weights:
            unmet.append("FusedAdam with capturable=False and master_weights=False")
        if self.config.overlap_optimizer_step_flat_adam and not hasattr(
            tex, 'flat_adam_capturable_master'
        ):
            unmet.append("TransformerEngine with flat_adam_capturable_master")
        if optimizer.grad_scaler is not None:
            unmet.append("no gradient scaler")
        ddp_requirements = {
            "overlap_grad_reduce": ddp_config.overlap_grad_reduce,
            "overlap_param_gather": ddp_config.overlap_param_gather,
            "not align_param_gather": not ddp_config.align_param_gather,
            "one distributed optimizer instance": (
                ddp_config.num_distributed_optimizer_instances == 1
            ),
            "not fp8_param_gather / fp4_param_gather": not (
                ddp_config.fp8_param_gather or ddp_config.fp4_param_gather
            ),
            "not use_megatron_fsdp": not ddp_config.use_megatron_fsdp,
        }
        unmet += [name for name, ok in ddp_requirements.items() if not ok]
        # The gradient norm of a bucket group is taken when its reduce-scatter lands, so nothing
        # may modify the reduced gradients afterwards (see finalize_model_grads).
        if model_config.calculate_per_token_loss:
            unmet.append("not calculate_per_token_loss")
        if model_config.tensor_model_parallel_size > 1 and (
            model_config.sequence_parallel or model_config.qk_layernorm
        ):
            unmet.append(
                "no gradient all-reduce across tensor-parallel ranks "
                "(sequence_parallel or qk_layernorm with TP > 1)"
            )
        if model_config.pipeline_model_parallel_size > 1:
            try:
                shares_embeddings = get_attr_wrapped_model(
                    optimizer.model_chunks[0], 'share_embeddings_and_output_weights'
                )
            except RuntimeError:
                shares_embeddings = False
            if (
                shares_embeddings
                or getattr(model_config, 'mtp_num_layers', None)
                or getattr(model_config, 'position_embedding_type', None) == 'learned_absolute'
            ):
                unmet.append("no embedding gradient all-reduce across pipeline stages")
        if model_config.gtp_weight_remat_size > 1 or model_config.expert_gtp_weight_remat_size > 1:
            unmet.append("no GTP weight rematerialization")
        for model_chunk in optimizer.model_chunks:
            for buffer in model_chunk.buffers + model_chunk.expert_parallel_buffers:
                if buffer.extra_main_grads:
                    unmet.append("no fp32 local gradient accumulation (extra main grads)")
                    break
        for group in optimizer.model_float16_groups + optimizer.model_fp32_groups:
            for model_param in group:
                if _get_param_grad_norm_group(model_param) is not None:
                    unmet.append("no separately clipped gradient-norm groups")
                if getattr(model_param, 'average_gradients_across_tp_domain', False):
                    unmet.append("no gradients averaged across the tensor-parallel domain")
        for group in optimizer.shard_float16_groups:
            if any(shard is None for shard in group):
                unmet.append("no quantized model parameters")
                break
        if unmet:
            raise ValueError(
                "overlap_optimizer_step_with_forward requires "
                + ", ".join(dict.fromkeys(unmet))
                + "."
            )

    def _ensure_built(self):
        if self._work is not None:
            return
        from transformer_engine.pytorch.optimizers import fused_adam

        self._max_tensors = getattr(fused_adam, '_MAX_TENSORS_PER_MULTI_TENSOR_ADAM_CALL', 1024)
        optimizer = self.optimizer
        inner = optimizer.optimizer
        if optimizer.init_state_fn is not None:
            optimizer.init_state_fn(inner, optimizer.config)
        self._work = {
            bucket_group: _BucketGroupWork(index)
            for index, bucket_group in enumerate(self.bucket_groups)
        }
        # bucket -> params of the flat Adam launch over its shard
        flat_pieces: Dict[object, List[Tuple[int, torch.nn.Parameter, List[torch.Tensor]]]] = {}
        for group_index in range(len(inner.param_groups)):
            for model_param, shard_param, main_param in zip(
                optimizer.model_float16_groups[group_index],
                optimizer.shard_float16_groups[group_index],
                optimizer.shard_fp32_from_float16_groups[group_index],
            ):
                state = inner.state[main_param]
                tensors = [shard_param, state['exp_avg'], state['exp_avg_sq'], main_param]
                if self.config.overlap_optimizer_step_flat_adam:
                    bucket = self._bucket_of(model_param)
                    if bucket is not None:
                        flat_pieces.setdefault(bucket, []).append(
                            (group_index, model_param, tensors)
                        )
                        continue
                self._add_param(
                    group_index, model_param, tex.multi_tensor_adam_capturable_master, tensors
                )
            for model_param, shard_param in zip(
                optimizer.model_fp32_groups[group_index], optimizer.shard_fp32_groups[group_index]
            ):
                state = inner.state[shard_param]
                self._add_param(
                    group_index,
                    model_param,
                    tex.multi_tensor_adam_capturable,
                    [shard_param, state['exp_avg'], state['exp_avg_sq']],
                )
        for bucket, pieces in flat_pieces.items():
            self._add_flat_shard(bucket, pieces)

    def _grad_shard(self, model_param):
        """This rank's view of the model param's reduce-scattered gradient, and its bucket group."""
        param_range = self.optimizer._get_model_param_range_map(model_param)["param"]
        grad_shard = model_param.main_grad.view(-1)[param_range.start : param_range.end]
        bucket_group = next(
            model_chunk.param_to_bucket_group[model_param]
            for model_chunk in self.optimizer.model_chunks
            if model_param in model_chunk.param_to_bucket_group
        )
        return grad_shard, bucket_group

    def _bucket_of(self, model_param):
        """The bucket holding the model param's gradient, if this rank owns part of it."""
        grad_shard, bucket_group = self._grad_shard(model_param)
        if grad_shard.numel() == 0:
            return None
        return next(
            bucket
            for bucket in bucket_group.buckets
            if bucket.grad_data.data_ptr()
            <= grad_shard.data_ptr()
            < bucket.grad_data.data_ptr() + bucket.grad_data.nbytes
        )

    def _add_param(self, group_index, model_param, kernel, tensors):
        """Register a model param's shard: its grad shard view feeds the kernel and the norm."""
        grad_shard, bucket_group = self._grad_shard(model_param)
        if grad_shard.numel() == 0:
            return
        work = self._work[bucket_group]
        lists = work.adam_calls.setdefault(
            (group_index, kernel), [[] for _ in range(1 + len(tensors))]
        )
        for tensor_list, tensor in zip(lists, [grad_shard] + tensors):
            tensor_list.append(tensor)
        self._add_norm_grad(work, model_param, grad_shard)

    def _add_norm_grad(self, work, model_param, grad_shard):
        if self._counts_towards_grad_norm(model_param):
            work.norm_grads.append(grad_shard)

    def _add_flat_shard(self, bucket, pieces):
        """Register one flat Adam launch over this rank's shard of the bucket: the FP32 main
        params and moments of the shard's params move into one contiguous buffer each, laid out
        like the shard (the tensors the optimizer holds become views into them). A layout the
        kernel cannot take keeps the multi-tensor launches for the bucket's params."""
        from transformer_engine.pytorch.optimizers import flat_adam_segments

        bucket_group = next(
            bucket_group for bucket_group in self.bucket_groups if bucket in bucket_group.buckets
        )
        size = bucket_group.intra_distributed_optimizer_instance_size
        rank = bucket_group.intra_distributed_optimizer_instance_rank
        grad = shard_buffer(bucket.grad_data, size)[rank]
        param = shard_buffer(bucket.param_data, size)[rank]
        runs = []  # (offset, numel, param group, main param, exp_avg, exp_avg_sq)
        unmet = None
        if grad.data_ptr() % 16 or param.data_ptr() % 16:
            unmet = "the bucket shard is not 16-byte aligned"
        for group_index, model_param, (shard_param, exp_avg, exp_avg_sq, main_param) in pieces:
            if unmet is not None:
                break
            grad_shard, _ = self._grad_shard(model_param)
            numel = grad_shard.numel()
            offset = (grad_shard.data_ptr() - grad.data_ptr()) // grad.element_size()
            if not 0 <= offset <= grad.numel() - numel:
                unmet = "a gradient shard lies outside the bucket shard"
            elif shard_param.data_ptr() != param.data_ptr() + offset * param.element_size():
                unmet = "a param shard is not at its gradient's offset of the bucket"
            elif not all(
                tensor.dtype == torch.float32
                and tensor.is_contiguous()
                and tensor.numel() == numel
                and tensor.dim() == 1
                for tensor in (main_param, exp_avg, exp_avg_sq)
            ):
                unmet = "a main param or moment is not a contiguous FP32 vector of the shard's size"
            else:
                runs.append((offset, numel, group_index, main_param, exp_avg, exp_avg_sq))
        if unmet is None:
            try:
                seg_starts, seg_groups = flat_adam_segments(
                    [(offset, numel, group) for offset, numel, group, *_ in runs],
                    grad.numel(),
                    grad.device,
                )
            except ValueError as e:
                unmet = str(e)
        if unmet is not None:
            logger.warning(
                "overlap_optimizer_step_flat_adam: a bucket of %d elements keeps the multi-tensor "
                "Adam launches: %s.",
                bucket.grad_data.numel(),
                unmet,
            )
            for group_index, model_param, tensors in pieces:
                self._add_param(
                    group_index, model_param, tex.multi_tensor_adam_capturable_master, tensors
                )
            return
        with torch.no_grad():
            master_param, exp_avg, exp_avg_sq = (
                torch.zeros_like(grad, dtype=torch.float32) for _ in range(3)
            )
            for offset, numel, _, *tensors in runs:
                for buffer, tensor in zip((master_param, exp_avg, exp_avg_sq), tensors):
                    view = buffer[offset : offset + numel]
                    view.copy_(tensor)
                    tensor.set_(view)
        work = self._work[bucket_group]
        work.flat_shards.append(
            _FlatShard(grad, param, exp_avg, exp_avg_sq, master_param, seg_starts, seg_groups)
        )
        for _, model_param, _ in pieces:
            self._add_norm_grad(work, model_param, self._grad_shard(model_param)[0])

    def _counts_towards_grad_norm(self, param) -> bool:
        """Same selection as MegatronOptimizer._filter_grads_for_norm."""
        optimizer = self.optimizer
        return (
            param_is_not_shared(param)
            and tensor_parallel.param_is_not_tensor_parallel_duplicate(
                param,
                tp_group=getattr(optimizer, 'tp_group', None),
                expert_tp_group=getattr(optimizer, 'expert_tp_group', None),
            )
            and tensor_parallel.param_is_not_gtp_duplicate(
                param,
                gtp_group=getattr(optimizer, 'gtp_group', None),
                expert_gtp_group=getattr(optimizer, 'expert_gtp_group', None),
            )
        )

    def invalidate(self):
        """Rebuild the kernel argument lists before the next use (optimizer state was replaced)."""
        self._work = None

    # ------------------------------------------------------------------ per iteration, host side

    def begin_iteration(self):
        """Start of a training iteration (optimizer.zero_grad()): the forward that follows applies
        the scheduled update, and the dispatch bookkeeping restarts."""
        self._dispatched.clear()
        self._norm_done.clear()
        self._cursor = 0
        self._update_scheduled = False
        if not self._order_known and self._order:
            # First iteration done: bucket groups the forward never waited for go last.
            for bucket_group in self.bucket_groups:
                if bucket_group not in self._position:
                    self._position[bucket_group] = len(self._order)
                    self._order.append(bucket_group)
            self._order_known = True

    def local_grad_norm_sq(self) -> torch.Tensor:
        """Sum of this rank's squared grad-norm partials over all bucket groups, shape [1]."""
        return self._partials.sum(dim=0, keepdim=True)

    def schedule_update(self, grad_norm: torch.Tensor):
        """optimizer.step(): make the next forward's Adam launches apply this iteration's update.

        ``grad_norm`` is the total gradient norm (a device tensor); the clipping coefficient is
        folded into the gradient scale the kernels apply.
        """
        param_groups = self.optimizer.optimizer.param_groups
        hparams = [_hparams(group) for group in param_groups]
        if self._captured_hparams is not None and hparams != self._captured_hparams:
            raise RuntimeError(
                "Optimizer hyperparameters other than the learning rate changed after the "
                "optimizer step was captured into a CUDA graph; the captured Adam launches use "
                f"{self._captured_hparams}, the param groups now have {hparams}."
            )
        for scalars, group, group_hparams in zip(self._groups, param_groups, hparams):
            scalars.hparams = group_hparams
            group['step'] = group.get('step', 0) + 1
            scalars.step.fill_(group['step'])
            if isinstance(group['lr'], torch.Tensor):
                scalars.lr.copy_(group['lr'])
            else:
                scalars.lr.fill_(group['lr'])
        if self.config.clip_grad > 0.0:
            # Same expression as clip_grad_by_total_norm_fp32.
            torch.clamp(self.config.clip_grad / (grad_norm + 1.0e-6), max=1.0, out=self.inv_scale)
        self.noop.fill_(0)
        self._update_scheduled = True
        # Read by DistributedDataParallel.zero_grad_buffer: the grads stay until the update
        # consumed them (each bucket is zeroed right after its Adam), or until the update is
        # applied or discarded here.
        self.grads_pending_update = True

    def apply_scheduled_update(self):
        """Apply a scheduled update now, on the current stream (before a forced param sync)."""
        if not self._update_scheduled:
            return
        self._ensure_built()
        for bucket_group in self.bucket_groups:
            self._launch_update(bucket_group)
        self.noop.fill_(1)
        self._update_scheduled = False
        self.grads_pending_update = False

    def discard_scheduled_update(self):
        """Drop a scheduled update; parameters and optimizer state stay as they are."""
        if not self._update_scheduled:
            return
        for group in self.optimizer.optimizer.param_groups:
            group['step'] -= 1
        self.noop.fill_(1)
        self._update_scheduled = False
        self.grads_pending_update = False

    # ------------------------------------------------------------------ captured, from DDP

    def grad_reduce_finished(self, bucket_group):
        """The bucket group's reduce-scatter is done: take its grad-norm partial on the side
        stream (overlapping the waits for the remaining bucket groups), joined back."""
        if bucket_group in self._norm_done:
            return
        self._norm_done.add(bucket_group)
        self._ensure_built()
        work = self._work[bucket_group]
        if not work.norm_grads:
            return
        current_stream = torch.cuda.current_stream()
        self.stream.wait_stream(current_stream)
        with torch.cuda.stream(self.stream):
            norm, _ = multi_tensor_applier(
                l2_norm_impl, self._dummy_overflow_buf, [list(work.norm_grads)], False
            )
            torch.mul(norm, norm, out=self._partials[work.index : work.index + 1])
        current_stream.wait_stream(self.stream)

    def dispatch_param_gather(self, bucket_group):
        """The forward needs the bucket group's params: issue its update and all-gather if not
        done yet, plus those of the next ``prefetch`` bucket groups in first-use order."""
        self._ensure_built()
        if not self._order_known:
            if bucket_group not in self._position:
                self._position[bucket_group] = len(self._order)
                self._order.append(bucket_group)
            if bucket_group not in self._dispatched:
                self._dispatch(bucket_group)
            return
        last = min(self._position[bucket_group] + 1 + self.prefetch, len(self._order))
        while self._cursor < last:
            next_bucket_group = self._order[self._cursor]
            self._cursor += 1
            if next_bucket_group not in self._dispatched:
                self._dispatch(next_bucket_group)

    def _dispatch(self, bucket_group):
        self._dispatched.add(bucket_group)
        if self._captured_hparams is None and torch.cuda.is_current_stream_capturing():
            self._captured_hparams = [scalars.hparams for scalars in self._groups]
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self._launch_update(bucket_group)
            bucket_group.start_param_sync()

    def _launch_update(self, bucket_group):
        """Adam of every shard in the bucket group (a no-op while ``noop`` is set), then zero its
        grads. Runs on the current stream."""
        work = self._work[bucket_group]
        if work.flat_shards:
            beta1, beta2, eps, weight_decay, bias_correction = (
                list(values) for values in zip(*(scalars.hparams for scalars in self._groups))
            )
            for shard in work.flat_shards:
                tex.flat_adam_capturable_master(
                    shard.grad,
                    shard.param,
                    shard.exp_avg,
                    shard.exp_avg_sq,
                    shard.master_param,
                    self.noop,
                    shard.seg_starts,
                    shard.seg_groups,
                    [scalars.lr for scalars in self._groups],
                    beta1,
                    beta2,
                    eps,
                    [scalars.step for scalars in self._groups],
                    self._adam_w_mode,
                    bias_correction,
                    weight_decay,
                    self.inv_scale,
                    False,
                )
        for (group_index, kernel), lists in work.adam_calls.items():
            scalars = self._groups[group_index]
            beta1, beta2, eps, weight_decay, bias_correction = scalars.hparams
            for start in range(0, len(lists[0]), self._max_tensors):
                multi_tensor_applier(
                    kernel,
                    self.noop,
                    [tensors[start : start + self._max_tensors] for tensors in lists],
                    scalars.lr,
                    beta1,
                    beta2,
                    eps,
                    scalars.step,
                    self._adam_w_mode,
                    bias_correction,
                    weight_decay,
                    self.inv_scale,
                )
        bucket_group.zero_grads()
