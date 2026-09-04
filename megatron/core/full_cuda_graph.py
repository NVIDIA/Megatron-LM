# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Full iteration CUDA graph for training."""

import gc
import logging

import torch

from megatron.core.tensor_parallel.random import get_all_rng_states
from megatron.core.transformer.experimental_attention_variant import dsa_logging

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
        # Shallow-copy so callers may replace or remove top-level entries to tailor the
        # batch to their pipeline stage without mutating the cached static buffer. Nested
        # containers and the tensors themselves are still shared with the buffer.
        return StaticBufferLoader.static_buffers[stage][microbatch].copy()


class FullCudaGraphWrapper:
    """Wrapper class to enable FullIterationCUDAgraph."""

    curr_iteration = {'training': 0, 'validation': 0}
    cuda_graph = {'training': None, 'validation': None}
    result = {'training': None, 'validation': None}
    _dsa_metric_tracker_prepared = False

    def __init__(self, forward_backward_func, cuda_graph_warmup_steps=1, use_single_mempool=False):
        self.forward_backward_func = forward_backward_func
        self.static_loader = StaticBufferLoader()
        self.cuda_graph_warmup_steps = cuda_graph_warmup_steps
        self.use_single_mempool = use_single_mempool

    def data_read(self, data_iterator, model, training, num_microbatches):
        """Read all microbatch inputs from Dataloader and copy to static buffers."""
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
        data_list = self.data_read(data_iterator, model, training, num_microbatches)
        kwargs['data_iterator'] = data_list

        training_str = 'training' if training else 'validation'
        curr_iteration = self.curr_iter(training_str)
        if curr_iteration == self.cuda_graph_warmup_steps:
            pg_collection = kwargs.get('pg_collection')
            prepare_dsa_tracker, metric_pg_collection = (
                dsa_logging.resolve_dsa_metric_pg_collection(pg_collection)
            )
            dsa_metric_tracker = None
            dsa_metric_snapshot = None
            prepared_dsa_tracker_size = (
                dsa_logging.prepare_dsa_metric_tracker_for_capture(
                    model, getattr(metric_pg_collection, "pp", None)
                )
                if prepare_dsa_tracker
                else 0
            )
            if prepared_dsa_tracker_size > 0:
                FullCudaGraphWrapper._dsa_metric_tracker_prepared = True
                dsa_metric_tracker = dsa_logging.DSAIndexerLossLoggingHelper.tracker
                dsa_metric_snapshot = dsa_logging.snapshot_dsa_metric_tracker_for_capture(
                    dsa_metric_tracker
                )
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
            if dsa_metric_snapshot is not None:
                captured_reduction_metadata = {
                    key: dsa_metric_tracker[key]
                    for key in ("reduce_group", "avg_group")
                    if key in dsa_metric_tracker
                }
                # Recording executes the graph body once. Restore the pre-capture metric values
                # before the replay below so the capture iteration is accounted exactly once
                # while retaining the storage referenced by the graph. With zero eager warmup,
                # reduction groups are first discovered by Python during capture and must survive
                # because replay only executes the recorded GPU work.
                dsa_logging.restore_dsa_metric_tracker_after_capture(
                    dsa_metric_tracker, dsa_metric_snapshot
                )
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
        if stage is None or stage == 'validation':
            if FullCudaGraphWrapper.cuda_graph['validation'] is not None:
                del FullCudaGraphWrapper.cuda_graph['validation']
                FullCudaGraphWrapper.cuda_graph['validation'] = None
            FullCudaGraphWrapper.result['validation'] = None
            FullCudaGraphWrapper.curr_iteration['validation'] = 0
        if FullCudaGraphWrapper._dsa_metric_tracker_prepared and all(
            graph is None for graph in FullCudaGraphWrapper.cuda_graph.values()
        ):
            dsa_logging.clear_dsa_metric_tracker_capture_state()
            FullCudaGraphWrapper._dsa_metric_tracker_prepared = False
        gc.collect()
