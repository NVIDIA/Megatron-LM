# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Full iteration CUDA graph for training."""

import gc
import logging

import torch

from megatron.core.tensor_parallel.random import (
    convert_cuda_rng_state,
    get_all_rng_states,
    get_cuda_rng_tracker,
    is_graph_safe_cuda_rng_tracker,
)

logger = logging.getLogger(__name__)

# Optional pure-Python metadata attached to an input batch whose values are baked
# into the captured graph (for example, packed-CP route split sizes).  Tensor
# payloads may change between replays, but this metadata must remain identical for
# each static-buffer slot.
FULL_CUDA_GRAPH_STATIC_METADATA_KEY = 'full_cuda_graph_static_metadata'

# Process-wide handle so full-iter and optimizer graph captures share one pool and one
# non-default stream (per-stream alloc segments can inflate memory_reserved; see
# tools/debug_cuda_graph_pool_memory*.py).
_shared_graph_pool = None
_shared_capture_stream = None


def get_shared_capture_stream():
    """Return one `torch.cuda.Stream` for all full-iter and optimizer graph captures.

    Call after the target CUDA device is selected.
    """
    global _shared_capture_stream
    if _shared_capture_stream is None:
        _shared_capture_stream = torch.cuda.Stream()
    return _shared_capture_stream


def get_shared_graph_pool():
    """Return a process-wide handle so all call sites share one graph memory pool.

    `torch.cuda.graph_pool_handle()` returns a new pool each time; this lazy singleton
    ensures e.g. full-iteration and optimizer captures reuse the same pool.
    """
    global _shared_graph_pool
    if _shared_graph_pool is None:
        _shared_graph_pool = torch.cuda.graph_pool_handle()
    return _shared_graph_pool


def get_graph_pool(use_single_mempool):
    """Return graph pool handle for full-iter/optimizer graph capture.

    When `use_single_mempool` is True, train/eval and optimizer captures reuse one
    process-wide pool. Otherwise, each capture call gets a new pool handle.
    """
    if use_single_mempool:
        return get_shared_graph_pool()
    return torch.cuda.graph_pool_handle()


# The below functions traverse through nested data structures (tuples, lists, dicts)
# present in src and creates a deep copy where all PyTorch tensors are cloned,
# detached from the computation graph, and moved to CUDA device. Non-tensor objects
# are returned as-is.


def copy_tensors_in_struct(src):
    """Copy src to new tensors."""
    if isinstance(src, tuple):
        return tuple(copy_tensors_in_struct(i) for i in src)
    elif isinstance(src, list):
        return list(copy_tensors_in_struct(i) for i in src)
    elif isinstance(src, dict):
        return {k: copy_tensors_in_struct(src[k]) for k in src}
    elif isinstance(src, torch.Tensor):
        return src.clone().detach().cuda()
    else:
        return src


def clone_tensors_in_struct(tgt, src):
    """Copy src to pre-existing tensors in tgt."""
    if isinstance(src, tuple):
        if not isinstance(tgt, tuple) or len(tgt) != len(src):
            return copy_tensors_in_struct(src)
        return tuple(clone_tensors_in_struct(t, s) for t, s in zip(tgt, src))
    elif isinstance(src, list):
        if not isinstance(tgt, list) or len(tgt) != len(src):
            return copy_tensors_in_struct(src)
        for i in range(len(src)):
            if isinstance(src[i], (tuple, list, dict, torch.Tensor)):
                tgt[i] = clone_tensors_in_struct(tgt[i], src[i])
            else:
                tgt[i] = src[i]
        return tgt
    elif isinstance(src, dict):
        if not isinstance(tgt, dict):
            return copy_tensors_in_struct(src)
        for k in src:
            # This metadata was compared before any static-buffer update and is
            # immutable for the lifetime of the graph.  Keep the captured copy
            # instead of descending into pure-Python tuples such as CP split
            # sizes and packed boundaries.
            if k == FULL_CUDA_GRAPH_STATIC_METADATA_KEY:
                continue
            if isinstance(src[k], (tuple, list, dict, torch.Tensor)):
                clone_tensors_in_struct(tgt[k], src[k])
            else:
                tgt[k] = src[k]
    elif isinstance(src, torch.Tensor):
        tgt.copy_(src, non_blocking=True)
    else:
        raise Exception(f"Expect top-level as container type but got: {type(src)}")


# Class to copy dataloader output to static CUDA tensors for CUDA graph input. This
# maintains separate static buffers for training and validation CUDA graphs.
class StaticBufferLoader:
    """Load data to static buffers."""

    static_buffers: dict = {'training': [], 'validation': []}

    def __init__(self):
        self.stream = torch.cuda.Stream()
        self.defer_static_metadata_mismatch = False
        self.static_metadata_mismatch = None
        self.batch_stage = None
        self.batch_start_length = None
        self.batch_dynamic_cp_metadata = []

    def begin_batch(self, stage, *, defer_static_metadata_mismatch=False):
        """Reset per-call metadata state before loading all static-buffer slots."""
        assert stage in ('training', 'validation')
        self.defer_static_metadata_mismatch = defer_static_metadata_mismatch
        self.static_metadata_mismatch = None
        self.batch_stage = stage
        self.batch_start_length = len(StaticBufferLoader.static_buffers[stage])
        self.batch_dynamic_cp_metadata = []

    def rollback_batch(self):
        """Discard only slots appended since ``begin_batch``.

        Existing slots may already be owned by a captured CUDA graph and must
        never be replaced or truncated after a rejected replay batch.
        """
        if self.batch_stage is None:
            return
        assert self.batch_start_length is not None
        del StaticBufferLoader.static_buffers[self.batch_stage][self.batch_start_length :]

    def end_batch(self):
        """Clear per-call bookkeeping after consensus succeeds or fails."""
        self.defer_static_metadata_mismatch = False
        self.static_metadata_mismatch = None
        self.batch_stage = None
        self.batch_start_length = None
        self.batch_dynamic_cp_metadata = []

    @classmethod
    def reset(cls, stage=None):
        """Drop all static buffers (e.g. between models or tests).

        Only call after the CUDA graphs referencing these buffers have been
        destroyed via ``FullCudaGraphWrapper.reset_cuda_graph``.
        """
        for reset_stage in ('training', 'validation'):
            if stage is None or stage == reset_stage:
                cls.static_buffers[reset_stage] = []

    def __call__(self, inputs, stage, microbatch):
        assert stage in ['training', 'validation']
        assert microbatch <= len(StaticBufferLoader.static_buffers[stage])
        if isinstance(inputs, tuple) and isinstance(inputs[0], dict):
            inputs = inputs[0]

        assert isinstance(inputs, dict)
        incoming_static_metadata = inputs.get(FULL_CUDA_GRAPH_STATIC_METADATA_KEY)
        dynamic_cp_metadata = (
            incoming_static_metadata.get('thd_dynamic_cp')
            if isinstance(incoming_static_metadata, dict)
            else None
        )
        if self.batch_stage is not None:
            assert stage == self.batch_stage
            self.batch_dynamic_cp_metadata.append(dynamic_cp_metadata)
        validation_error = (
            dynamic_cp_metadata.get('validation_error')
            if isinstance(dynamic_cp_metadata, dict)
            else None
        )
        if validation_error is not None:
            message = (
                "Full-iteration CUDA graph static input metadata is invalid for "
                f"{stage} microbatch slot {microbatch}: {validation_error}"
            )
            if not self.defer_static_metadata_mismatch:
                raise RuntimeError(message)
            if self.static_metadata_mismatch is None:
                self.static_metadata_mismatch = message

        if microbatch == len(StaticBufferLoader.static_buffers[stage]):
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                StaticBufferLoader.static_buffers[stage].append(copy_tensors_in_struct(inputs))
        else:
            captured_static_metadata = StaticBufferLoader.static_buffers[stage][microbatch].get(
                FULL_CUDA_GRAPH_STATIC_METADATA_KEY
            )
            if incoming_static_metadata != captured_static_metadata:
                message = (
                    "Full-iteration CUDA graph static input metadata changed for "
                    f"{stage} microbatch slot {microbatch}. The captured graph contains "
                    "Python-derived layout values (such as packed-CP route/split metadata), "
                    "so replay with a different layout would be incorrect. Keep the packed "
                    "geometry fixed for each microbatch slot or reset and recapture the graph."
                )
                if not self.defer_static_metadata_mismatch:
                    raise RuntimeError(message)
                if self.static_metadata_mismatch is None:
                    self.static_metadata_mismatch = message
                # Do not mutate the captured slot. The wrapper performs one
                # world-wide consensus after every rank has inspected all slots,
                # then every rank fails together before capture or replay.
                return StaticBufferLoader.static_buffers[stage][microbatch].copy()

            for k in inputs.keys():
                if k not in StaticBufferLoader.static_buffers[stage][microbatch]:
                    if isinstance(inputs[k], torch.Tensor):
                        StaticBufferLoader.static_buffers[stage][microbatch][k] = torch.empty_like(
                            inputs[k], device="cuda"
                        )
                    else:
                        StaticBufferLoader.static_buffers[stage][microbatch][k] = inputs[k]

            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                clone_tensors_in_struct(
                    StaticBufferLoader.static_buffers[stage][microbatch], inputs
                )
        torch.cuda.current_stream().wait_stream(self.stream)
        # Shallow-copy so callers may replace or remove top-level entries to tailor the
        # batch to their pipeline stage without mutating the cached static buffer. Nested
        # containers and the tensors themselves are still shared with the buffer.
        return StaticBufferLoader.static_buffers[stage][microbatch].copy()


class FullCudaGraphWrapper:
    """Wrapper class to enable FullIterationCUDAgraph."""

    curr_iteration = {'training': 0, 'validation': 0}
    cuda_graph = {'training': None, 'validation': None}
    result = {'training': None, 'validation': None}
    capture_signature = {'training': None, 'validation': None}

    @staticmethod
    def _get_graphable_rng_states():
        """Validate and return the graph-safe generators used during capture."""
        tracker = get_cuda_rng_tracker()
        if not is_graph_safe_cuda_rng_tracker(tracker):
            raise RuntimeError(
                "Full-iteration CUDA graph capture requires a graph-safe CUDA RNG tracker. "
                "Initialize the native tracker with use_cudagraphable_rng=True."
            )

        tracker_states = tracker.get_states()
        invalid_states = {
            name: type(state).__name__
            for name, state in tracker_states.items()
            if not isinstance(state, (torch.Tensor, torch.Generator))
        }
        if invalid_states:
            raise RuntimeError(
                "Full-iteration CUDA graph capture requires tensor or generator RNG states; "
                f"tracker returned unsupported states: {invalid_states}."
            )

        if any(isinstance(state, torch.Tensor) for state in tracker_states.values()):
            tracker_states = {
                name: convert_cuda_rng_state(state, to_graphable=True)
                for name, state in tracker_states.items()
            }
        # Besides updating the tracker, this synchronizes TE's process-global
        # state registry and discards any states left by an older tracker.
        tracker.set_states(tracker_states)
        graphable_states = get_all_rng_states()

        invalid_states = {
            name: type(state).__name__
            for name, state in graphable_states.items()
            if not isinstance(state, torch.Generator)
        }
        if invalid_states:
            raise RuntimeError(
                "Full-iteration CUDA graph capture requires graphable RNG generators; "
                f"tracker returned non-generator states: {invalid_states}."
            )

        detached_states = [
            name
            for name, state in graphable_states.items()
            if tracker.get_states().get(name) is not state
        ]
        missing_states = set(tracker.get_states()) ^ set(graphable_states)
        if detached_states or missing_states:
            raise RuntimeError(
                "Full-iteration CUDA graph capture requires the registered generators to be "
                "owned by the active RNG tracker; "
                f"detached states: {detached_states}, mismatched names: {sorted(missing_states)}."
            )
        return graphable_states

    def __init__(
        self,
        forward_backward_func,
        cuda_graph_warmup_steps=1,
        use_single_mempool=False,
        batch_preparation_fn=None,
        require_global_static_metadata_consensus=False,
    ):
        """
        Args:
            forward_backward_func: The pipeline-parallel forward-backward function to wrap.
            cuda_graph_warmup_steps: Number of eager iterations to run before capture.
            use_single_mempool: Share one memory pool across full-iter/optimizer captures.
            batch_preparation_fn: Optional ``fn(data_iterator, vp_stage) -> dict`` hook that
                canonicalizes one microbatch to graph-static shapes outside the captured
                region (e.g. THD packed batches). It is called on every rank for every
                (model chunk, microbatch) pair in the same order — even on ranks whose
                data_iterator is None — so it may issue collectives such as TP broadcasts.
            require_global_static_metadata_consensus: Defer local static-metadata mismatches
                until all ranks perform one consensus outside capture. This guarantees that
                a rank-local packed-layout change cannot leave peer ranks replaying a graph
                whose collectives wait for the failed rank.
        """
        self.forward_backward_func = forward_backward_func
        self.static_loader = StaticBufferLoader()
        self.cuda_graph_warmup_steps = cuda_graph_warmup_steps
        self.use_single_mempool = use_single_mempool
        self.batch_preparation_fn = batch_preparation_fn
        self.require_global_static_metadata_consensus = require_global_static_metadata_consensus

    @staticmethod
    def _encode_new_dynamic_cp_group_state(
        dynamic_cp_metadata, mismatch_found, appended_count, world_size
    ):
        """Encode all per-slot DCP groups into one fixed-shape WORLD payload."""
        encoded = [int(mismatch_found), len(dynamic_cp_metadata), appended_count]
        for metadata in dynamic_cp_metadata:
            if metadata is None:
                encoded.extend([0, 0, -1])
                encoded.extend([-1] * world_size)
                continue

            members = [int(rank) for rank in metadata.get('cp_group_global_ranks', ())]
            # Preserve the reported group size separately.  If it exceeds the
            # WORLD-sized membership field, the validator below rejects it on
            # every rank rather than changing the collective shape.
            group_size = int(metadata.get('local_cp_size', -1))
            group_rank = int(metadata.get('cp_group_rank', -1))
            encoded.extend([1, group_size, group_rank])
            encoded.extend((members + [-1] * world_size)[:world_size])
        return encoded

    @staticmethod
    def _validate_new_dynamic_cp_group_state(
        gathered_state, slot_count, appended_count, world_size
    ):
        """Validate that every DCP group is an ordered membership equivalence class."""
        rows = gathered_state.to(device='cpu').tolist()
        if any(row[1] != slot_count for row in rows):
            return "ranks prepared different numbers of static DCP microbatch slots"
        if any(row[2] != appended_count for row in rows):
            return "ranks appended different numbers of static DCP microbatch slots"

        block_width = world_size + 3
        for slot in range(slot_count):
            start = 3 + slot * block_width
            present = [bool(row[start]) for row in rows]
            if not any(present):
                continue
            if not all(present):
                return f"slot {slot} has dynamic-CP metadata on only a subset of ranks"

            for reporter_rank, row in enumerate(rows):
                group_size = row[start + 1]
                group_rank = row[start + 2]
                members = tuple(row[start + 3 : start + 3 + world_size])
                if not 1 <= group_size <= world_size:
                    return (
                        f"slot {slot} rank {reporter_rank} reported invalid dynamic-CP "
                        f"group size {group_size} for WORLD size {world_size}"
                    )
                members = members[:group_size]
                if len(set(members)) != group_size or any(
                    member < 0 or member >= world_size for member in members
                ):
                    return (
                        f"slot {slot} rank {reporter_rank} reported invalid dynamic-CP "
                        f"members {members}"
                    )
                if not 0 <= group_rank < group_size or members[group_rank] != reporter_rank:
                    return (
                        f"slot {slot} rank {reporter_rank} reported group rank {group_rank} "
                        f"for ordered members {members}"
                    )

                for expected_group_rank, member in enumerate(members):
                    peer_row = rows[member]
                    peer_start = start
                    peer_size = peer_row[peer_start + 1]
                    peer_group_rank = peer_row[peer_start + 2]
                    peer_members = tuple(peer_row[peer_start + 3 : peer_start + 3 + group_size])
                    if (
                        not bool(peer_row[peer_start])
                        or peer_size != group_size
                        or peer_group_rank != expected_group_rank
                        or peer_members != members
                    ):
                        return (
                            f"slot {slot} dynamic-CP membership is not coherent: rank "
                            f"{reporter_rank} reports {members}, but member {member} reports "
                            f"members={peer_members}, group_rank={peer_group_rank}"
                        )
        return None

    def _check_new_dynamic_cp_group_consensus(self, mismatch_found, appended_count):
        """Run the exact DCP membership check whenever new slots are created."""
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        local_encoded = self._encode_new_dynamic_cp_group_state(
            self.static_loader.batch_dynamic_cp_metadata, mismatch_found, appended_count, world_size
        )
        local_state = torch.tensor(
            local_encoded, dtype=torch.int64, device=torch.cuda.current_device()
        )
        if world_size > 1:
            gathered = torch.empty(
                world_size * local_state.numel(), dtype=local_state.dtype, device=local_state.device
            )
            torch.distributed.all_gather_into_tensor(gathered, local_state)
            gathered = gathered.view(world_size, local_state.numel())
        else:
            gathered = local_state.unsqueeze(0)

        mismatch_found = bool(torch.any(gathered[:, 0] != 0).item())
        group_error = self._validate_new_dynamic_cp_group_state(
            gathered, len(self.static_loader.batch_dynamic_cp_metadata), appended_count, world_size
        )
        return mismatch_found or group_error is not None, group_error

    def _check_global_static_metadata_consensus(self, stage):
        """Make every rank fail together when any rank's static layout changes."""
        assert self.static_loader.batch_stage == stage
        assert self.static_loader.batch_start_length is not None
        local_mismatch = self.static_loader.static_metadata_mismatch
        mismatch_found = local_mismatch is not None
        group_error = None
        appended_count = (
            len(StaticBufferLoader.static_buffers[stage]) - self.static_loader.batch_start_length
        )
        try:
            if self.require_global_static_metadata_consensus:
                if appended_count > 0:
                    # Any batch that creates static slots validates all ordered
                    # DCP memberships in the same fixed-shape WORLD all-gather
                    # that propagates rank-local metadata failures.
                    mismatch_found, group_error = self._check_new_dynamic_cp_group_consensus(
                        mismatch_found, appended_count
                    )
                elif torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                    mismatch_flag = torch.tensor(
                        [int(mismatch_found)], dtype=torch.int32, device=torch.cuda.current_device()
                    )
                    torch.distributed.all_reduce(mismatch_flag, op=torch.distributed.ReduceOp.MAX)
                    mismatch_found = bool(mismatch_flag.item())

            if mismatch_found:
                detail = (
                    local_mismatch
                    or group_error
                    or ("Full-iteration CUDA graph static input metadata changed on another rank.")
                )
                raise RuntimeError(
                    f"{detail} All ranks rejected the {stage} batch before capture or replay."
                )
        except Exception:
            # Collective/runtime failures must not leave newly allocated slots
            # behind either.  Existing graph-owned slots are preserved by the
            # begin-batch boundary.
            self.static_loader.rollback_batch()
            raise
        finally:
            self.static_loader.end_batch()

    def _data_read_with_batch_preparation(self, data_iterator, model, stage, num_microbatches):
        """Canonicalize each microbatch outside the graph, then load static buffers.

        Every rank receives an iterator of static batches (the preparation
        function broadcasts data to ranks without a data_iterator), and each
        (model chunk, microbatch) pair gets its own static buffer slot.
        """
        num_chunks = len(model) if isinstance(model, list) else 1
        if isinstance(data_iterator, list):
            assert len(data_iterator) == num_chunks
            iterators = data_iterator
        else:
            iterators = [data_iterator] * num_chunks
        use_vp_stage = isinstance(model, list) and len(model) > 1
        data_list = []
        for i in range(num_chunks):
            chunk_batches = []
            for b in range(num_microbatches):
                batch = self.batch_preparation_fn(iterators[i], i if use_vp_stage else None)
                chunk_batches.append(self.static_loader(batch, stage, i * num_microbatches + b))
            data_list.append(iter(chunk_batches))
        return data_list

    def data_read(self, data_iterator, model, training, num_microbatches):
        """Read all microbatch inputs from Dataloader and copy to static buffers."""
        if self.batch_preparation_fn is not None:
            return self._data_read_with_batch_preparation(
                data_iterator, model, 'training' if training else 'validation', num_microbatches
            )
        if not isinstance(model, list) or len(model) == 1:
            assert not isinstance(data_iterator, list) or len(data_iterator) == 1
            iterator0 = data_iterator if not isinstance(data_iterator, list) else data_iterator[0]
            data_list = []
            if iterator0 is not None:
                for b in range(num_microbatches):
                    data_list.append(
                        self.static_loader(
                            next(iterator0), 'training' if training else 'validation', b
                        )
                    )
                data_list = [iter(data_list)]
            else:
                data_list.append(None)
        else:
            assert isinstance(data_iterator, list) and len(data_iterator) == len(model)
            data_list = []
            for i in range(len(model)):
                if data_iterator[i] is not None:
                    data_list_i = []
                    for b in range(num_microbatches):
                        data_list_i.append(
                            self.static_loader(
                                next(data_iterator[i]), 'training' if training else 'validation', b
                            )
                        )
                    data_list.append(iter(data_list_i))
                else:
                    data_list.append(None)
        return data_list

    def __call__(self, *args, **kwargs):
        assert len(args) == 0, 'forward_backward_func does not accept positional args'
        assert all(
            [
                kwarg in kwargs
                for kwarg in [
                    'model',
                    'data_iterator',
                    'num_microbatches',
                    'seq_length',
                    'forward_only',
                ]
            ]
        )
        model = kwargs['model']
        num_microbatches = kwargs['num_microbatches']

        training = not kwargs['forward_only']
        data_iterator = kwargs['data_iterator']
        training_str = 'training' if training else 'validation'

        # A captured graph bakes in the schedule topology; replaying it with a
        # different signature would silently reuse stale shapes and buffers.
        signature = {
            'num_microbatches': num_microbatches,
            'num_model_chunks': len(model) if isinstance(model, list) else 1,
            'seq_length': kwargs.get('seq_length'),
            'micro_batch_size': kwargs.get('micro_batch_size'),
            'decoder_seq_length': kwargs.get('decoder_seq_length'),
        }
        self._check_capture_signature(training_str, signature)

        self.static_loader.begin_batch(
            training_str,
            defer_static_metadata_mismatch=self.require_global_static_metadata_consensus,
        )
        data_list = self.data_read(data_iterator, model, training, num_microbatches)
        self._check_global_static_metadata_consensus(training_str)
        kwargs['data_iterator'] = data_list

        curr_iteration = self.curr_iter(training_str)
        if curr_iteration == self.cuda_graph_warmup_steps:
            from megatron.core.transformer.cuda_graphs import (
                _prepare_dsa_metric_tracker_for_capture,
                _restore_metric_tracker,
                _snapshot_metric_tracker,
            )
            from megatron.core.transformer.experimental_attention_variant.dsa import (
                DSAIndexerLossLoggingHelper,
            )

            pg_collection = kwargs.get('pg_collection')
            prepare_dsa_tracker = True
            if pg_collection is not None and hasattr(
                pg_collection, "get_language_model_collection"
            ):
                prepare_dsa_tracker = pg_collection.has_language_model()
                pp_group = (
                    pg_collection.get_language_model_collection().pp
                    if prepare_dsa_tracker
                    else None
                )
            else:
                pp_group = getattr(pg_collection, "pp", None)
            if prepare_dsa_tracker:
                _prepare_dsa_metric_tracker_for_capture(model, pp_group)
            dsa_metric_tracker = DSAIndexerLossLoggingHelper.tracker
            dsa_metric_snapshot = _snapshot_metric_tracker(dsa_metric_tracker)
            logger.info(f'Capture CUDA graph for {training_str}!!!')
            if hasattr(torch.autograd.graph, 'set_override_stale_capture_stream'):
                torch.autograd.graph.set_override_stale_capture_stream(True)
            else:
                logger.warning(
                    'torch.autograd.graph.set_override_stale_capture_stream is not '
                    'available in this PyTorch version; CUDA graph capture may fail '
                    'if autograd nodes hold stale references to non-capturing streams. '
                    'Upgrade to a PyTorch build that includes pytorch/pytorch#180090.'
                )
            torch.distributed.barrier()
            # Release cached blocks reserved during the eager warmup iterations
            # before the capture allocates its private pool: the two pools
            # coexist for the lifetime of the graph, and warmup fragmentation
            # (reserved-but-unallocated blocks) otherwise counts against the
            # capture's headroom.
            gc.collect()
            torch.cuda.empty_cache()
            assert FullCudaGraphWrapper.cuda_graph[training_str] is None
            graphable_rng_states = self._get_graphable_rng_states()
            FullCudaGraphWrapper.capture_signature[training_str] = signature
            FullCudaGraphWrapper.cuda_graph[training_str] = torch.cuda.CUDAGraph()
            for state in graphable_rng_states.values():
                FullCudaGraphWrapper.cuda_graph[training_str].register_generator_state(state)
            torch.cuda.synchronize()
            capture_stream = get_shared_capture_stream()
            with torch.cuda.graph(
                FullCudaGraphWrapper.cuda_graph[training_str],
                stream=capture_stream,
                pool=get_graph_pool(self.use_single_mempool),
                capture_error_mode="thread_local",
            ):
                FullCudaGraphWrapper.result[training_str] = self.forward_backward_func(
                    *args, **kwargs
                )
            torch.cuda.synchronize()
            torch.distributed.barrier()
            captured_reduction_metadata = {
                key: dsa_metric_tracker[key]
                for key in ("reduce_group", "avg_group")
                if key in dsa_metric_tracker
            }
            # Recording executes the graph body once. Restore the pre-capture metric values
            # before the replay below so the capture iteration is accounted exactly once while
            # retaining the storage referenced by the graph. With zero eager warmup, reduction
            # groups are first discovered by Python during capture and must survive because
            # replay only executes the recorded GPU work.
            _restore_metric_tracker(dsa_metric_tracker, dsa_metric_snapshot)
            for key, value in captured_reduction_metadata.items():
                if dsa_metric_snapshot[1].get(key) is None:
                    dsa_metric_tracker[key] = value
            logger.info(f'CUDA graph capture done for {training_str}!!!')
        if FullCudaGraphWrapper.cuda_graph[training_str] is None:
            FullCudaGraphWrapper.result[training_str] = self.forward_backward_func(*args, **kwargs)
        else:
            FullCudaGraphWrapper.cuda_graph[training_str].replay()
        self.next_iter(training_str)
        return FullCudaGraphWrapper.result[training_str]

    def _check_capture_signature(self, stage, signature):
        """Refuse inconsistent or changed signatures before any rank reads data.

        Static-certified dynamic CP may run different effective CP groups, but
        full-iteration capture still requires one world-wide schedule signature.
        Check that invariant collectively so one rank cannot fail locally while
        its peers enter batch-preparation collectives or graph replay.
        """
        captured = FullCudaGraphWrapper.capture_signature[stage]
        mismatches = (
            {}
            if captured is None
            else {
                key: (captured[key], signature[key])
                for key in captured
                if captured[key] != signature[key]
            }
        )

        if not self.require_global_static_metadata_consensus or not (
            torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
        ):
            if not mismatches:
                return
            details = ', '.join(
                f"{key}: captured={old} vs current={new}" for key, (old, new) in mismatches.items()
            )
            raise RuntimeError(
                f"Full-iteration CUDA graph signature mismatch for {stage} ({details}). "
                "The captured graph bakes in the schedule topology (e.g. a fixed "
                "num_microbatches), so these values must stay constant after capture. "
                "Keep the schedule fixed or reset the graph via reset_cuda_graph()."
            )

        signature_keys = (
            'num_microbatches',
            'num_model_chunks',
            'seq_length',
            'micro_batch_size',
            'decoder_seq_length',
        )
        encoded_signature = [0 if stage == 'training' else 1]
        encoded_signature.extend(
            -1 if signature[key] is None else int(signature[key]) for key in signature_keys
        )
        # The buffer count decides whether the post-data path performs the
        # one-time topology all-gather or the steady-state mismatch all-reduce.
        # Certify it before data reads so all ranks select the same collective.
        encoded_signature.append(len(StaticBufferLoader.static_buffers[stage]))
        # Include graph-presence as well as the local captured-signature check.
        # Otherwise one rank that lost/reset its graph could enter warmup while
        # its peers replay, even if their current call signatures still match.
        local_state = torch.tensor(
            encoded_signature + [int(captured is not None), int(bool(mismatches))],
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )
        world_size = torch.distributed.get_world_size()
        gathered = torch.empty(
            world_size * local_state.numel(), dtype=local_state.dtype, device=local_state.device
        )
        torch.distributed.all_gather_into_tensor(gathered, local_state)
        gathered = gathered.view(world_size, local_state.numel())

        current_signature_differs = torch.any(gathered[:, :-2] != gathered[0, :-2])
        graph_presence_differs = torch.any(gathered[:, -2] != gathered[0, -2])
        captured_signature_mismatch = torch.any(gathered[:, -1] != 0)
        if not bool(
            (
                current_signature_differs | graph_presence_differs | captured_signature_mismatch
            ).item()
        ):
            return

        details = (
            ', '.join(
                f"{key}: captured={old} vs current={new}" for key, (old, new) in mismatches.items()
            )
            if mismatches
            else "the current signature, graph presence, or captured signature differs across ranks"
        )
        raise RuntimeError(
            f"Full-iteration CUDA graph signature mismatch for {stage} ({details}). "
            "All ranks rejected the call before reading data, capture, or replay. "
            "Keep the schedule fixed across ranks or reset the graph via reset_cuda_graph()."
        )

    def curr_iter(self, stage):
        """Return current training/validation iteration."""
        return FullCudaGraphWrapper.curr_iteration[stage]

    def next_iter(self, stage):
        """Increment current training/validation iteration."""
        FullCudaGraphWrapper.curr_iteration[stage] += 1

    @classmethod
    def reset_cuda_graph(cls, stage=None):
        """Destroy captured CUDA graph(s) and reset the class-level state.

        Must be called before tearing down the process groups whose collectives
        were captured (e.g. PP P2P): a live graph keeps references to NCCL
        resources and destroying the communicators first can hang shutdown.
        """
        for reset_stage in ('training', 'validation'):
            if stage is not None and stage != reset_stage:
                continue
            if cls.cuda_graph[reset_stage] is not None:
                del cls.cuda_graph[reset_stage]
                cls.cuda_graph[reset_stage] = None
            cls.result[reset_stage] = None
            cls.curr_iteration[reset_stage] = 0
            cls.capture_signature[reset_stage] = None
            StaticBufferLoader.reset(stage=reset_stage)
        gc.collect()
