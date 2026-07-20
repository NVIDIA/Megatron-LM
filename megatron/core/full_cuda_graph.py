# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Full iteration CUDA graph for training."""

import gc
import logging

import torch

from megatron.core.tensor_parallel.random import get_all_rng_states

logger = logging.getLogger(__name__)

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

    @classmethod
    def reset(cls):
        """Drop all static buffers (e.g. between models or tests).

        Only call after the CUDA graphs referencing these buffers have been
        destroyed via ``FullCudaGraphWrapper.reset_cuda_graph``.
        """
        cls.static_buffers = {'training': [], 'validation': []}

    def __call__(self, inputs, stage, microbatch):
        assert stage in ['training', 'validation']
        assert microbatch <= len(StaticBufferLoader.static_buffers[stage])
        if isinstance(inputs, tuple) and isinstance(inputs[0], dict):
            inputs = inputs[0]

        assert isinstance(inputs, dict)
        if microbatch == len(StaticBufferLoader.static_buffers[stage]):
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
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

            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                clone_tensors_in_struct(
                    StaticBufferLoader.static_buffers[stage][microbatch], inputs
                )
        torch.cuda.current_stream().wait_stream(self.stream)
        return StaticBufferLoader.static_buffers[stage][microbatch]


class FullCudaGraphWrapper:
    """Wrapper class to enable FullIterationCUDAgraph."""

    curr_iteration = {'training': 0, 'validation': 0}
    cuda_graph = {'training': None, 'validation': None}
    result = {'training': None, 'validation': None}
    capture_signature = {'training': None, 'validation': None}

    def __init__(
        self,
        forward_backward_func,
        cuda_graph_warmup_steps=1,
        use_single_mempool=False,
        batch_preparation_fn=None,
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
        """
        self.forward_backward_func = forward_backward_func
        self.static_loader = StaticBufferLoader()
        self.cuda_graph_warmup_steps = cuda_graph_warmup_steps
        self.use_single_mempool = use_single_mempool
        self.batch_preparation_fn = batch_preparation_fn

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

        data_list = self.data_read(data_iterator, model, training, num_microbatches)
        kwargs['data_iterator'] = data_list

        curr_iteration = self.curr_iter(training_str)
        if curr_iteration == self.cuda_graph_warmup_steps:
            logger.info(f'Capture CUDA graph for {training_str}!!!')
            torch.distributed.barrier()
            assert FullCudaGraphWrapper.cuda_graph[training_str] is None
            FullCudaGraphWrapper.capture_signature[training_str] = signature
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

    def _check_capture_signature(self, stage, signature):
        """Refuse to replay a captured graph whose call signature changed."""
        captured = FullCudaGraphWrapper.capture_signature[stage]
        if captured is None:
            # No graph captured for this stage yet; nothing to enforce.
            return
        mismatches = {
            key: (captured[key], signature[key])
            for key in captured
            if captured[key] != signature[key]
        }
        if mismatches:
            details = ', '.join(
                f"{key}: captured={old} vs current={new}" for key, (old, new) in mismatches.items()
            )
            raise RuntimeError(
                f"Full-iteration CUDA graph signature mismatch for {stage} ({details}). "
                "The captured graph bakes in the schedule topology (e.g. a fixed "
                "num_microbatches), so these values must stay constant after capture. "
                "Keep the schedule fixed or reset the graph via reset_cuda_graph()."
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
        gc.collect()
