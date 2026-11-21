# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Full iteration CUDA graph for training."""

import gc
import logging
from collections.abc import Callable, Iterable, Sequence
from typing import Optional

import torch

from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.tensor_parallel.gtp_symmetric_memory import symmetric_wgrad_pool
from megatron.core.tensor_parallel.random import (
    cudagraph_needs_generator_registration,
    get_all_rng_states,
    prime_cuda_rng_states_for_graph_capture,
)
from megatron.core.transformer.cuda_graphs import capture_constants

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


class FullIterationGradCopy:
    """Reuse forward/backward graph scratch for later optimizer-gradient conversion.

    Replay must stay ordered: forward/backward, gradient conversion, then the
    eager optimizer update. Retained gradient handles are rebound after zero_grad;
    their values may be overwritten by the next forward/backward replay.
    """

    def __init__(self) -> None:
        self.graph = None
        self.forward_backward_graph = None
        self.outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.source_signature = None

    @torch.no_grad()
    def copy(
        self,
        copy_fn: Callable[[], None],
        group_pairs: Iterable[tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]]],
        source_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        """Replay ``copy_fn`` as a CUDA graph in the training graph's pool and warmup stream.

        Args:
            copy_fn: The optimizer's ordinary gradient-conversion operation.
            group_pairs: Matching (model parameter group, optimizer parameter group)
                sequences, as the optimizers already keep them.
            source_fn: Maps a model parameter to the gradient storage ``copy_fn`` reads;
                defaults to its ``main_grad``. The distributed optimizer reads a shard of it.
        """
        source_fn = source_fn or (lambda model_param: model_param.main_grad)
        pairs = [
            (source_fn(model_param), main_param)
            for model_group, main_group in group_pairs
            for model_param, main_param in zip(model_group, main_group)
        ]
        for source, main_param in pairs:
            assert source.numel() == main_param.numel()
        if not pairs:
            copy_fn()
            return
        assert not torch.cuda.is_current_stream_capturing()
        fb_graph = FullCudaGraphWrapper.cuda_graph.get("training")
        if self.forward_backward_graph is not fb_graph:
            # reset_cuda_graph() can discard the training graph without rebuilding
            # the optimizer. Conversion state belongs to that graph's pool and must
            # be rebuilt if training is warmed up or captured again.
            self.graph = None
            self.outputs = []
            self.source_signature = None
            self.forward_backward_graph = fb_graph
        stream = get_shared_capture_stream()
        if fb_graph is None:
            caller = torch.cuda.current_stream()
            stream.wait_stream(caller)
            with torch.cuda.stream(stream):
                copy_fn()
            caller.wait_stream(stream)
            # Adam reads on caller after conversion allocated on the F/B stream.
            for _, main_param in pairs:
                main_param.grad.record_stream(caller)
            return

        signature = [
            (
                source.data_ptr(),
                tuple(source.shape),
                source.stride(),
                source.dtype,
                source.device,
                id(main_param),
            )
            for source, main_param in pairs
        ]
        if self.graph is None:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                graph, pool=fb_graph.pool(), stream=stream, capture_error_mode="thread_local"
            ):
                copy_fn()
            self.outputs = [(main_param, main_param.grad) for _, main_param in pairs]
            self.source_signature = signature
            self.graph = graph
        if signature != self.source_signature:
            raise RuntimeError("Full-CG optimizer gradient storage changed after capture")
        for main_param, grad in self.outputs:
            main_param.grad = grad
        self.graph.replay()


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
        raise Exception(f"Unsupported copy for tuple yet: {type(src)}")
    elif isinstance(src, list):
        for i in range(len(src)):
            if isinstance(src[i], (tuple, list, dict, torch.Tensor)):
                clone_tensors_in_struct(tgt[i], src[i])
            else:
                tgt[i] = src[i]
    elif isinstance(src, dict):
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

    # State of the batched staging path (load_microbatches). Like static_buffers, it is kept on
    # the class so that all wrapper instances of a stage share it. All are keyed by stage.
    # {key: (CUDA tensor [num_microbatches, ...], its row views)}; the row views are the static
    # buffers of that key.
    stacked_buffers: dict = {'training': {}, 'validation': {}}
    # {key: pinned host tensor} that host inputs are stacked into before one H2D copy.
    pinned_buffers: dict = {'training': {}, 'validation': {}}
    # Event after the last copy out of the pinned host tensors.
    staging_done_event: dict = {'training': None, 'validation': None}
    # Event after the last graph replay, which is the last reader of the static buffers.
    consumed_event: dict = {'training': None, 'validation': None}

    def __init__(self):
        self.stream = torch.cuda.Stream()

    @staticmethod
    def unwrap_inputs(inputs):
        """Return the input dict of one microbatch, as __call__ accepts it."""
        if isinstance(inputs, tuple) and isinstance(inputs[0], dict):
            inputs = inputs[0]
        assert isinstance(inputs, dict)
        return inputs

    @staticmethod
    def _has_cuda_tensor(src):
        if isinstance(src, torch.Tensor):
            return src.is_cuda
        if isinstance(src, (tuple, list)):
            return any(StaticBufferLoader._has_cuda_tensor(i) for i in src)
        if isinstance(src, dict):
            return any(StaticBufferLoader._has_cuda_tensor(i) for i in src.values())
        return False

    @staticmethod
    def _stackable_keys(batches):
        """Return the top-level keys that hold, in every microbatch, a plain contiguous tensor
        with the same shape, dtype and device (host or the current CUDA device)."""
        keys = []
        for key, first in batches[0].items():
            if (
                type(first) is not torch.Tensor
                or first.layout != torch.strided
                or first.requires_grad
                or not (
                    first.device.type == "cpu"
                    or (first.is_cuda and first.device.index == torch.cuda.current_device())
                )
            ):
                continue
            if all(
                type(inputs.get(key)) is torch.Tensor
                and inputs[key].shape == first.shape
                and inputs[key].dtype == first.dtype
                and inputs[key].device == first.device
                and inputs[key].is_contiguous()
                and not inputs[key].requires_grad
                for inputs in batches
            ):
                keys.append(key)
        return keys

    def load_microbatches(self, batches, stage):
        """Copy the inputs of all microbatches of one step into the static buffers.

        The static buffers end up with the same bytes as calling ``self(batches[b], stage, b)``
        for every microbatch b in order. The differences are in how the copies are issued:

        - All copies run under one switch to the staging stream, with one wait in each direction.
        - Each top-level tensor key that has the same shape, dtype and device in every microbatch
          is backed by one contiguous [num_microbatches, ...] CUDA tensor. The static buffer of
          microbatch b is row b of that tensor, so one copy per key fills all microbatches. Host
          inputs are first stacked into a pinned host tensor that is reused across steps.
        - If a graph replay was the last reader of the static buffers, all inputs are host
          tensors and no buffer is allocated, the staging stream waits only for that replay
          instead of all work queued on the current stream, so the copies can overlap that work.

        Args:
            batches: One input dict (or a tuple whose first item is the dict) per microbatch.
            stage: 'training' or 'validation'.

        Returns:
            The static buffer dicts of the microbatches, in microbatch order.
        """
        assert stage in ['training', 'validation']
        batches = [self.unwrap_inputs(inputs) for inputs in batches]
        num_microbatches = len(batches)
        buffers = StaticBufferLoader.static_buffers[stage]
        consumed_event = StaticBufferLoader.consumed_event[stage]
        StaticBufferLoader.consumed_event[stage] = None
        if num_microbatches == 0:
            return []
        if 0 < len(buffers) < num_microbatches:
            # Buffers that the per-microbatch path created for fewer microbatches.
            return [self(batches[b], stage, b) for b in range(num_microbatches)]

        first_call = len(buffers) == 0
        allocated = first_call
        if first_call:
            stacked = {}
            for key in self._stackable_keys(batches):
                first = batches[0][key]
                stacked_tensor = torch.empty(
                    (num_microbatches, *first.shape), dtype=first.dtype, device="cuda"
                )
                stacked[key] = (stacked_tensor, stacked_tensor.unbind(0))
            StaticBufferLoader.stacked_buffers[stage] = stacked
            StaticBufferLoader.pinned_buffers[stage] = {}
            StaticBufferLoader.staging_done_event[stage] = None
            stack_keys = list(stacked)
        else:
            stacked = StaticBufferLoader.stacked_buffers[stage]
            stack_keys = []
            for key in self._stackable_keys(batches):
                if key not in stacked:
                    continue
                stacked_tensor, rows = stacked[key]
                first = batches[0][key]
                # Use the stacked copy only while the static buffers are still its row views.
                if (
                    len(rows) >= num_microbatches
                    and stacked_tensor.shape[1:] == first.shape
                    and stacked_tensor.dtype == first.dtype
                    and all(buffers[b].get(key) is rows[b] for b in range(num_microbatches))
                ):
                    stack_keys.append(key)
            # Same as the per-microbatch path: add the keys that are new in this step.
            for inputs, static in zip(batches, buffers):
                for key, value in inputs.items():
                    if key not in static:
                        if isinstance(value, torch.Tensor):
                            static[key] = torch.empty_like(value, device="cuda")
                            allocated = True
                        else:
                            static[key] = value

        host_keys = [key for key in stack_keys if batches[0][key].device.type == "cpu"]
        pinned = StaticBufferLoader.pinned_buffers[stage]
        if host_keys:
            staging_done = StaticBufferLoader.staging_done_event[stage]
            if staging_done is not None:
                # The pinned host tensors are free once the previous copies out of them are done.
                staging_done.synchronize()
            for key in host_keys:
                stacked_tensor = stacked[key][0]
                if key not in pinned or pinned[key].shape != stacked_tensor.shape:
                    pinned[key] = torch.empty(
                        stacked_tensor.shape, dtype=stacked_tensor.dtype, pin_memory=True
                    )
                torch.stack([inputs[key] for inputs in batches], out=pinned[key][:num_microbatches])

        # The early wait orders the copies only after the readers of the static buffers. It is
        # not safe for inputs made on the current stream, nor for buffers just allocated on it.
        if (
            consumed_event is not None
            and not allocated
            and not any(self._has_cuda_tensor(inputs) for inputs in batches)
        ):
            self.stream.wait_event(consumed_event)
        else:
            self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            stack_key_set = set(stack_keys)
            if first_call:
                for b, inputs in enumerate(batches):
                    buffers.append(
                        {
                            key: (
                                stacked[key][1][b]
                                if key in stack_key_set
                                else copy_tensors_in_struct(value)
                            )
                            for key, value in inputs.items()
                        }
                    )
            else:
                for inputs, static in zip(batches, buffers):
                    clone_tensors_in_struct(
                        static,
                        {key: value for key, value in inputs.items() if key not in stack_key_set},
                    )
            for key in stack_keys:
                target = stacked[key][0][:num_microbatches]
                if key in host_keys:
                    target.copy_(pinned[key][:num_microbatches], non_blocking=True)
                else:
                    torch.stack([inputs[key] for inputs in batches], out=target)
            if host_keys:
                staging_done = torch.cuda.Event()
                staging_done.record(self.stream)
                StaticBufferLoader.staging_done_event[stage] = staging_done
        torch.cuda.current_stream().wait_stream(self.stream)
        return buffers[:num_microbatches]

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

    def __init__(
        self,
        forward_backward_func,
        cuda_graph_warmup_steps=1,
        use_single_mempool=False,
        batched_input_staging=False,
    ) -> None:
        self.forward_backward_func = forward_backward_func
        self.static_loader = StaticBufferLoader()
        self.cuda_graph_warmup_steps = cuda_graph_warmup_steps
        self.use_single_mempool = use_single_mempool
        self.batched_input_staging = batched_input_staging

    @staticmethod
    def _merge_chunk_inputs(chunk_inputs):
        """Merge the inputs that the model chunks read for each microbatch.

        All chunks stage microbatch b into the same static buffer b, in chunk order, so for each
        top-level key the last chunk that has the key decides the final contents. Returns one
        merged dict per microbatch, or None if a value is a nested container or the chunks
        disagree on the type, shape, dtype or strides of a key; then the merged dicts might not
        give the same static buffers as staging every chunk in order.
        """
        if len(chunk_inputs) == 1:
            return chunk_inputs[0]
        merged = []
        for per_chunk in zip(*chunk_inputs):
            result = {}
            for inputs in per_chunk:
                for key, value in inputs.items():
                    if isinstance(value, (tuple, list, dict)):
                        return None
                    if key in result:
                        previous = result[key]
                        if isinstance(previous, torch.Tensor) != isinstance(value, torch.Tensor):
                            return None
                        if isinstance(value, torch.Tensor) and (
                            previous.shape != value.shape
                            or previous.dtype != value.dtype
                            or previous.stride() != value.stride()
                        ):
                            return None
                    result[key] = value
            merged.append(result)
        return merged

    def _batched_data_read(self, data_iterator, model, training, num_microbatches):
        """Same as data_read, but stage each microbatch once with batched copies.

        The iterators are advanced exactly as in data_read, and the static buffers hold the same
        bytes after the call.
        """
        stage = 'training' if training else 'validation'
        if not isinstance(model, list) or len(model) == 1:
            assert not isinstance(data_iterator, list) or len(data_iterator) == 1
            iterators = [data_iterator if not isinstance(data_iterator, list) else data_iterator[0]]
        else:
            assert isinstance(data_iterator, list) and len(data_iterator) == len(model)
            iterators = data_iterator
        # Read in the order of data_read: all microbatches of chunk 0, then of chunk 1, and so on.
        chunk_inputs = [
            (
                None
                if iterator is None
                else [
                    StaticBufferLoader.unwrap_inputs(next(iterator))
                    for _ in range(num_microbatches)
                ]
            )
            for iterator in iterators
        ]
        present = [inputs for inputs in chunk_inputs if inputs is not None]
        buffers = []
        if present:
            merged = self._merge_chunk_inputs(present)
            if merged is not None:
                buffers = self.static_loader.load_microbatches(merged, stage)
            else:
                for inputs in present:
                    buffers = [self.static_loader(x, stage, b) for b, x in enumerate(inputs)]
        return [None if inputs is None else iter(list(buffers)) for inputs in chunk_inputs]

    def data_read(self, data_iterator, model, training, num_microbatches):
        """Read all microbatch inputs from Dataloader and copy to static buffers."""
        if self.batched_input_staging:
            return self._batched_data_read(data_iterator, model, training, num_microbatches)
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
            assert FullCudaGraphWrapper.cuda_graph[training_str] is None
            FullCudaGraphWrapper.cuda_graph[training_str] = torch.cuda.CUDAGraph()
            if cudagraph_needs_generator_registration():
                for _, state in get_all_rng_states().items():
                    FullCudaGraphWrapper.cuda_graph[training_str].register_generator_state(state)
            torch.cuda.synchronize()
            capture_stream = get_shared_capture_stream()
            symmetric_wgrad_pool.reset_completed_reuse_events()
            rerun_state_machine = get_rerun_state_machine()
            with (
                capture_constants(FullCudaGraphWrapper.cuda_graph[training_str]),
                torch.cuda.graph(
                    FullCudaGraphWrapper.cuda_graph[training_str],
                    stream=capture_stream,
                    pool=get_graph_pool(self.use_single_mempool),
                    capture_error_mode="thread_local",
                ),
                rerun_state_machine.capture_deferred_validations(training_str),
            ):
                # Must precede all other captured work; see the helper's docstring.
                prime_cuda_rng_states_for_graph_capture()
                FullCudaGraphWrapper.result[training_str] = self.forward_backward_func(
                    *args, **kwargs
                )
            torch.cuda.synchronize()
            torch.distributed.barrier()
            logger.info(f'CUDA graph capture done for {training_str}!!!')
        if FullCudaGraphWrapper.cuda_graph[training_str] is None:
            # Keep autograd nodes retained during warmup on the eventual capture stream.
            # Leaves receiving None gradients can bypass PyTorch's stale-stream override.
            caller_stream = torch.cuda.current_stream()
            warmup_stream = get_shared_capture_stream()
            warmup_stream.wait_stream(caller_stream)
            with torch.cuda.stream(warmup_stream):
                FullCudaGraphWrapper.result[training_str] = self.forward_backward_func(
                    *args, **kwargs
                )
            caller_stream.wait_stream(warmup_stream)
        else:
            FullCudaGraphWrapper.cuda_graph[training_str].replay()
            if self.batched_input_staging:
                # The replay is the last reader of the static buffers until the next step.
                consumed_event = torch.cuda.Event()
                consumed_event.record()
                StaticBufferLoader.consumed_event[training_str] = consumed_event
        self.next_iter(training_str)
        # Validate before returning to the training loop's rerun/update decision.
        get_rerun_state_machine().run_deferred_validations(training_str)
        return FullCudaGraphWrapper.result[training_str]

    def curr_iter(self, stage):
        """Return current training/validation iteration."""
        return FullCudaGraphWrapper.curr_iteration[stage]

    def next_iter(self, stage):
        """Increment current training/validation iteration."""
        FullCudaGraphWrapper.curr_iteration[stage] += 1

    def reset_cuda_graph(self, stage=None):
        """Reset CUDA graph."""
        captured = any(graph is not None for graph in FullCudaGraphWrapper.cuda_graph.values())
        if captured:
            torch.cuda.synchronize()
        if stage is None or stage == 'training':
            if FullCudaGraphWrapper.cuda_graph['training'] is not None:
                del FullCudaGraphWrapper.cuda_graph['training']
                FullCudaGraphWrapper.cuda_graph['training'] = None
            FullCudaGraphWrapper.result['training'] = None
            FullCudaGraphWrapper.curr_iteration['training'] = 0
            StaticBufferLoader.consumed_event['training'] = None
            get_rerun_state_machine().clear_deferred_validations("training")
        if stage is None or stage == 'validation':
            if FullCudaGraphWrapper.cuda_graph['validation'] is not None:
                del FullCudaGraphWrapper.cuda_graph['validation']
                FullCudaGraphWrapper.cuda_graph['validation'] = None
            FullCudaGraphWrapper.result['validation'] = None
            FullCudaGraphWrapper.curr_iteration['validation'] = 0
            StaticBufferLoader.consumed_event['validation'] = None
            get_rerun_state_machine().clear_deferred_validations("validation")
        if captured:
            symmetric_wgrad_pool.reset_completed_reuse_events()
        gc.collect()
