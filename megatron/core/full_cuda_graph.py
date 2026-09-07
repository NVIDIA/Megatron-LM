# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Full iteration CUDA graph for training."""

import gc
import logging
from typing import Any, Dict, Iterable, Iterator, List

import torch

from megatron.core.tensor_parallel.random import get_all_rng_states

logger = logging.getLogger(__name__)

# Process-wide handle so full-iter and optimizer graph captures share one pool and one
# non-default stream (per-stream alloc segments can inflate memory_reserved; see
# tools/debug_cuda_graph_pool_memory*.py).
_shared_graph_pool = None
_shared_capture_stream = None

_FULL_CUDA_GRAPH_STAGES = ('training', 'validation')
_PREPARED_CONSTANT_TYPES = (type(None), bool, int, float, str)


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


class FullCudaGraphPreparedIterator(Iterator[Dict[str, Any]]):
    """Marker iterator over fixed-address batches prepared outside a full graph.

    ``pretrain_gpt.get_batch`` uses this concrete type to distinguish finalized
    tensor-only batches from ordinary dataloader output without peeking at (and
    consuming) a raw iterator. Each yielded dictionary is a top-level shallow
    copy: pipeline-stage code may replace an entry without changing the static
    owner dictionary captured by the full-iteration graph.
    """

    def __init__(self, batches: Iterable[Dict[str, Any]]):
        self._batches = tuple(batches)
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._index >= len(self._batches):
            raise StopIteration
        batch = self._batches[self._index]
        self._index += 1
        return batch.copy()


def _prepared_batch_schema(batch, *, stage, model_chunk_index, microbatch_index):
    """Return the exact flat schema for one finalized full-graph batch."""
    slot = f"stage={stage}, model_chunk={model_chunk_index}, " f"microbatch={microbatch_index}"
    if not isinstance(batch, dict):
        raise TypeError(f"full-iteration prepared batch must be a dict ({slot})")

    schema = []
    tensor_devices = set()
    invalid_keys = [key for key in batch if not isinstance(key, str)]
    if invalid_keys:
        raise TypeError(
            "full-iteration prepared batch keys must be strings "
            f"({slot}, got keys {invalid_keys!r})"
        )
    for key in sorted(batch):
        value = batch[key]
        if isinstance(value, torch.Tensor):
            tensor_devices.add(value.device)
            value_schema = ('tensor', tuple(value.shape), value.dtype, value.device)
        elif type(value) in _PREPARED_CONSTANT_TYPES:
            value_schema = ('constant', type(value), value)
        else:
            raise TypeError(
                "full-iteration prepared batches must be flat dictionaries of tensors "
                "and immutable constants "
                f"({slot}, key={key!r}, got {type(value).__name__})"
            )
        schema.append((key, value_schema))

    if len(tensor_devices) > 1:
        raise ValueError(
            "all tensors in one full-iteration prepared batch must use the same device "
            f"({slot}, got {sorted(map(str, tensor_devices))})"
        )
    return tuple(schema)


def _allocate_prepared_batch_owners(batch):
    """Allocate fixed-address owners without copying any source values."""
    return {
        key: (
            torch.empty_like(value, requires_grad=False)
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in batch.items()
    }


def _prepared_batch_owner_ptrs(batch):
    """Return tensor owner pointers keyed by their exact top-level names."""
    return {
        key: value.data_ptr() for key, value in batch.items() if isinstance(value, torch.Tensor)
    }


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
    prepared_static_buffers: dict = {'training': None, 'validation': None}
    prepared_schemas: dict = {'training': None, 'validation': None}
    prepared_owner_ptrs: dict = {'training': None, 'validation': None}

    def __init__(self):
        # Constructing the wrapper must remain CPU-testable. Streams/events are
        # created lazily on the selected CUDA device when staging first occurs.
        self.stream = None
        self.prepared_ready_event = None

    def _get_copy_stream(self):
        if self.stream is None:
            self.stream = torch.cuda.Stream()
        return self.stream

    def _get_prepared_ready_event(self):
        if self.prepared_ready_event is None:
            self.prepared_ready_event = torch.cuda.Event()
        return self.prepared_ready_event

    def __call__(self, inputs, stage, microbatch):
        assert stage in ['training', 'validation']
        assert microbatch <= len(StaticBufferLoader.static_buffers[stage])
        if isinstance(inputs, tuple) and isinstance(inputs[0], dict):
            inputs = inputs[0]

        assert isinstance(inputs, dict)
        stream = self._get_copy_stream()
        if microbatch == len(StaticBufferLoader.static_buffers[stage]):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                StaticBufferLoader.static_buffers[stage].append(copy_tensors_in_struct(inputs))
        else:

            for k in inputs.keys():
                if k not in StaticBufferLoader.static_buffers[stage][microbatch]:
                    if isinstance(inputs[k], torch.Tensor):
                        StaticBufferLoader.static_buffers[stage][microbatch][k] = torch.empty_like(
                            inputs[k], device="cuda"
                        )
                    else:
                        StaticBufferLoader.static_buffers[stage][microbatch][k] = inputs[k]

            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                clone_tensors_in_struct(
                    StaticBufferLoader.static_buffers[stage][microbatch], inputs
                )
        torch.cuda.current_stream().wait_stream(stream)
        # Shallow-copy so callers may replace or remove top-level entries to tailor the
        # batch to their pipeline stage without mutating the cached static buffer. Nested
        # containers and the tensors themselves are still shared with the buffer.
        return StaticBufferLoader.static_buffers[stage][microbatch].copy()

    @classmethod
    def reset_prepared_state(cls, stage=None):
        """Drop strict prepared owners and their pinned schemas."""
        stages = _FULL_CUDA_GRAPH_STAGES if stage is None else (stage,)
        for stage_name in stages:
            if stage_name not in _FULL_CUDA_GRAPH_STAGES:
                raise ValueError(f"invalid full-iteration CUDA graph stage: {stage_name!r}")
            cls.prepared_static_buffers[stage_name] = None
            cls.prepared_schemas[stage_name] = None
            cls.prepared_owner_ptrs[stage_name] = None

    def _copy_prepared_iteration(self, owners, prepared_batches):
        """Commit every slot in one ordered copy-stream transaction."""
        tensor_pairs = []
        tensor_devices = set()
        for owner_chunk, source_chunk in zip(owners, prepared_batches):
            for owner_batch, source_batch in zip(owner_chunk, source_chunk):
                for key, source in source_batch.items():
                    if isinstance(source, torch.Tensor):
                        owner = owner_batch[key]
                        tensor_pairs.append((owner, source))
                        tensor_devices.add(source.device)

        if len(tensor_devices) > 1:
            raise ValueError(
                "one full-iteration prepared staging transaction cannot span devices: "
                f"{sorted(map(str, tensor_devices))}"
            )
        if not tensor_pairs:
            return

        device = next(iter(tensor_devices))
        if device.type != 'cuda':
            # Pure synchronous path for focused CPU tests. Full-iteration CUDA
            # graph callers naturally provide CUDA tensors.
            with torch.no_grad():
                for owner, source in tensor_pairs:
                    owner.copy_(source)
            return

        current_device = torch.device('cuda', torch.cuda.current_device())
        if device != current_device:
            raise ValueError(
                "full-iteration prepared tensors must be on the current CUDA device: "
                f"expected {current_device}, got {device}"
            )

        current_stream = torch.cuda.current_stream()
        copy_stream = self._get_copy_stream()
        ready_event = self._get_prepared_ready_event()
        # Exactly one dependency into the copy stream, one transaction covering
        # every slot/tensor, one completion event, and one replay-stream wait.
        copy_stream.wait_stream(current_stream)
        with torch.cuda.stream(copy_stream), torch.no_grad():
            for owner, source in tensor_pairs:
                owner.copy_(source, non_blocking=True)
                # The eager prepared batch is temporary and may lose its last
                # Python reference as soon as staging returns. Tell the CUDA
                # caching allocator that its storage remains live until this
                # private stream has consumed the nonblocking copy.
                source.record_stream(copy_stream)
            ready_event.record(copy_stream)
        current_stream.wait_event(ready_event)

    def stage_prepared_iteration(self, prepared_batches, stage):
        """Validate then atomically stage a complete prepared iteration.

        ``prepared_batches`` is chunk-major, then microbatch-minor. No static
        owner is allocated or mutated until every slot schema is known to be
        valid, so a late callback/schema failure cannot partially update the
        graph inputs for an iteration that will never replay.
        """
        if stage not in _FULL_CUDA_GRAPH_STAGES:
            raise ValueError(f"invalid full-iteration CUDA graph stage: {stage!r}")
        if not isinstance(prepared_batches, list) or not prepared_batches:
            raise ValueError("full-iteration prepared batches require at least one model chunk")

        actual_schemas = []
        for model_chunk_index, chunk in enumerate(prepared_batches):
            if not isinstance(chunk, list):
                raise TypeError("full-iteration prepared batch chunks must be lists")
            chunk_schemas = []
            for microbatch_index, batch in enumerate(chunk):
                chunk_schemas.append(
                    _prepared_batch_schema(
                        batch,
                        stage=stage,
                        model_chunk_index=model_chunk_index,
                        microbatch_index=microbatch_index,
                    )
                )
            actual_schemas.append(tuple(chunk_schemas))
        actual_schemas = tuple(actual_schemas)

        expected_schemas = StaticBufferLoader.prepared_schemas[stage]
        existing_owners = StaticBufferLoader.prepared_static_buffers[stage]
        expected_ptrs = StaticBufferLoader.prepared_owner_ptrs[stage]
        if expected_schemas is None:
            if existing_owners is not None or expected_ptrs is not None:
                raise RuntimeError("incomplete full-iteration prepared static-buffer state")
            # Allocate into a private tree only after *all* schemas validate.
            candidate_owners = [
                [_allocate_prepared_batch_owners(batch) for batch in chunk]
                for chunk in prepared_batches
            ]
            candidate_ptrs = tuple(
                tuple(_prepared_batch_owner_ptrs(batch) for batch in chunk)
                for chunk in candidate_owners
            )
            owners = candidate_owners
            owner_ptrs = candidate_ptrs
        else:
            if actual_schemas != expected_schemas:
                raise RuntimeError(
                    "full-iteration prepared batch schema changed before replay: "
                    f"expected={expected_schemas!r}, actual={actual_schemas!r}"
                )
            if existing_owners is None or expected_ptrs is None:
                raise RuntimeError("incomplete full-iteration prepared static-buffer state")
            owners = existing_owners
            owner_ptrs = tuple(
                tuple(_prepared_batch_owner_ptrs(batch) for batch in chunk) for chunk in owners
            )
            if owner_ptrs != expected_ptrs:
                raise RuntimeError(
                    "full-iteration prepared static owner address changed before replay: "
                    f"expected={expected_ptrs!r}, actual={owner_ptrs!r}"
                )

        # This is the first and only owner mutation point after every slot has
        # passed validation.
        self._copy_prepared_iteration(owners, prepared_batches)

        actual_ptrs_after_copy = tuple(
            tuple(_prepared_batch_owner_ptrs(batch) for batch in chunk) for chunk in owners
        )
        if actual_ptrs_after_copy != owner_ptrs:
            raise RuntimeError("full-iteration prepared static owner address changed during copy")

        if expected_schemas is None:
            StaticBufferLoader.prepared_schemas[stage] = actual_schemas
            StaticBufferLoader.prepared_static_buffers[stage] = owners
            StaticBufferLoader.prepared_owner_ptrs[stage] = owner_ptrs

        return owners


class FullCudaGraphWrapper:
    """Wrapper class to enable FullIterationCUDAgraph."""

    curr_iteration = {'training': 0, 'validation': 0}
    cuda_graph = {'training': None, 'validation': None}
    result = {'training': None, 'validation': None}
    prepared_run_signatures = {'training': None, 'validation': None}

    def __init__(
        self,
        forward_backward_func,
        cuda_graph_warmup_steps=1,
        use_single_mempool=False,
        batch_prepare_func=None,
    ):
        self.forward_backward_func = forward_backward_func
        self.static_loader = StaticBufferLoader()
        self.cuda_graph_warmup_steps = cuda_graph_warmup_steps
        self.use_single_mempool = use_single_mempool
        self.batch_prepare_func = batch_prepare_func

    @staticmethod
    def _identity(value):
        if value is None:
            return None
        return (type(value).__module__, type(value).__qualname__, id(value))

    @staticmethod
    def _signature_constant(value):
        if type(value) in _PREPARED_CONSTANT_TYPES:
            return (type(value), value)
        return FullCudaGraphWrapper._identity(value)

    def _prepared_run_signature(self, stage, kwargs):
        """Build the graph-shaping signature without consuming input data."""
        model = kwargs['model']
        data_iterator = kwargs['data_iterator']
        model_chunks, _ = self._normalize_chunks(data_iterator, model)
        if isinstance(data_iterator, list):
            iterator_topology = (
                'list',
                len(data_iterator),
                tuple(iterator is None for iterator in data_iterator),
            )
        else:
            iterator_topology = ('single', data_iterator is None)

        return {
            'stage': self._signature_constant(stage),
            'forward_only': self._signature_constant(kwargs['forward_only']),
            'num_microbatches': self._signature_constant(kwargs['num_microbatches']),
            'seq_length': self._signature_constant(kwargs['seq_length']),
            'micro_batch_size': self._signature_constant(kwargs.get('micro_batch_size')),
            'decoder_seq_length': self._signature_constant(kwargs.get('decoder_seq_length')),
            'model_container': 'list' if isinstance(model, list) else 'single',
            'model_chunks': tuple(self._identity(chunk) for chunk in model_chunks),
            'data_iterator_topology': iterator_topology,
            'forward_step_func': self._identity(kwargs.get('forward_step_func')),
            'batch_prepare_func': self._identity(self.batch_prepare_func),
            'collect_non_loss_data': self._signature_constant(
                kwargs.get('collect_non_loss_data', False)
            ),
            'first_val_step': self._signature_constant(kwargs.get('first_val_step')),
            'adjust_tensor_shapes_fn': self._identity(kwargs.get('adjust_tensor_shapes_fn')),
            'p2p_communicator': self._identity(kwargs.get('p2p_communicator')),
            'pg_collection': self._identity(kwargs.get('pg_collection')),
            'force_all_reduce': self._signature_constant(kwargs.get('force_all_reduce', False)),
        }

    @classmethod
    def _pin_prepared_run_signature(cls, stage, signature):
        """Pin once and reject drift before any iterator can be consumed."""
        expected = cls.prepared_run_signatures[stage]
        if expected is None:
            cls.prepared_run_signatures[stage] = signature.copy()
            return
        if expected == signature:
            return
        changed = {
            key: (expected.get(key), signature.get(key))
            for key in sorted(set(expected) | set(signature))
            if expected.get(key) != signature.get(key)
        }
        raise RuntimeError(
            "full-iteration prepared run signature changed before input consumption: "
            f"stage={stage}, changed={changed!r}"
        )

    @staticmethod
    def _normalize_chunks(data_iterator, model):
        """Return matching model chunks and raw iterators without consuming them."""
        if isinstance(model, list):
            model_chunks = model
        else:
            model_chunks = [model]
        if not model_chunks:
            raise ValueError("full-iteration CUDA graphs require at least one model chunk")

        if isinstance(data_iterator, list):
            if len(data_iterator) != len(model_chunks):
                raise ValueError(
                    "full-iteration data_iterator/model chunk counts must match: "
                    f"iterators={len(data_iterator)}, model_chunks={len(model_chunks)}"
                )
            raw_iterators = data_iterator
        else:
            if len(model_chunks) != 1:
                raise ValueError(
                    "full-iteration model chunking requires one raw iterator entry per chunk"
                )
            raw_iterators = [data_iterator]
        return model_chunks, raw_iterators

    def _prepared_data_read(self, data_iterator, model, training, num_microbatches):
        """Prepare every rank/slot first, then stage the complete iteration."""
        stage = 'training' if training else 'validation'
        model_chunks, raw_iterators = self._normalize_chunks(data_iterator, model)
        prepared_batches: List[List[Dict[str, Any]]] = []
        # Do not guard on raw_iterator being None. Non-source TP/CP ranks must
        # execute the callback in exactly the same collective order.
        for model_chunk_index, (model_chunk, raw_iterator) in enumerate(
            zip(model_chunks, raw_iterators)
        ):
            chunk_batches = []
            for microbatch_index in range(num_microbatches):
                batch = self.batch_prepare_func(
                    data_iterator=raw_iterator,
                    model=model_chunk,
                    stage=stage,
                    microbatch_index=microbatch_index,
                    model_chunk_index=model_chunk_index,
                )
                if not isinstance(batch, dict):
                    raise TypeError(
                        "full-iteration batch_prepare_func must return a flat dict "
                        f"(stage={stage}, model_chunk={model_chunk_index}, "
                        f"microbatch={microbatch_index}, got {type(batch).__name__})"
                    )
                chunk_batches.append(batch)
            prepared_batches.append(chunk_batches)

        static_chunks = self.static_loader.stage_prepared_iteration(prepared_batches, stage)
        return [FullCudaGraphPreparedIterator(chunk) for chunk in static_chunks]

    def data_read(self, data_iterator, model, training, num_microbatches):
        """Read all microbatch inputs from Dataloader and copy to static buffers."""
        if self.batch_prepare_func is not None:
            return self._prepared_data_read(data_iterator, model, training, num_microbatches)

        # Legacy behavior below intentionally remains permissive and unchanged.
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
        if self.batch_prepare_func is not None:
            if (
                not isinstance(num_microbatches, int)
                or isinstance(num_microbatches, bool)
                or num_microbatches <= 0
            ):
                raise ValueError(
                    "full-iteration prepared batches require a fixed positive integer "
                    f"num_microbatches, got {num_microbatches!r}"
                )
            signature = self._prepared_run_signature(training_str, kwargs)
            self._pin_prepared_run_signature(training_str, signature)
        data_list = self.data_read(data_iterator, model, training, num_microbatches)
        kwargs['data_iterator'] = data_list

        curr_iteration = self.curr_iter(training_str)
        if curr_iteration == self.cuda_graph_warmup_steps:
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
            FullCudaGraphWrapper.cuda_graph[training_str] = torch.cuda.CUDAGraph()
            for _, state in get_all_rng_states().items():
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
            logger.info(f'CUDA graph capture done for {training_str}!!!')
        if FullCudaGraphWrapper.cuda_graph[training_str] is None:
            FullCudaGraphWrapper.result[training_str] = self.forward_backward_func(*args, **kwargs)
        else:
            FullCudaGraphWrapper.cuda_graph[training_str].replay()
        self.next_iter(training_str)
        return FullCudaGraphWrapper.result[training_str]

    def curr_iter(self, stage):
        """Return current training/validation iteration."""
        return FullCudaGraphWrapper.curr_iteration[stage]

    def next_iter(self, stage):
        """Increment current training/validation iteration."""
        FullCudaGraphWrapper.curr_iteration[stage] += 1

    def reset_cuda_graph(self, stage=None):
        """Reset CUDA graph."""
        if stage is None or stage == 'training':
            if FullCudaGraphWrapper.cuda_graph['training'] is not None:
                del FullCudaGraphWrapper.cuda_graph['training']
                FullCudaGraphWrapper.cuda_graph['training'] = None
            FullCudaGraphWrapper.result['training'] = None
            FullCudaGraphWrapper.curr_iteration['training'] = 0
            FullCudaGraphWrapper.prepared_run_signatures['training'] = None
            StaticBufferLoader.reset_prepared_state('training')
        if stage is None or stage == 'validation':
            if FullCudaGraphWrapper.cuda_graph['validation'] is not None:
                del FullCudaGraphWrapper.cuda_graph['validation']
                FullCudaGraphWrapper.cuda_graph['validation'] = None
            FullCudaGraphWrapper.result['validation'] = None
            FullCudaGraphWrapper.curr_iteration['validation'] = 0
            FullCudaGraphWrapper.prepared_run_signatures['validation'] = None
            StaticBufferLoader.reset_prepared_state('validation')
        gc.collect()
