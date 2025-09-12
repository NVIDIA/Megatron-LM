from collections import deque, defaultdict
import torch
from megatron.core import parallel_state
from typing import Any
from transformer_engine.pytorch.float8_tensor import Float8Tensor
from transformer_engine.pytorch.cpu_offload import AsyncDoubleBufferGroupOffloadHandler

# cpu offload for pipeline

class PipelineOffloadManager:
    OFFLOAD_MGR = None
    @classmethod
    def get_instance(cls):
        if cls.OFFLOAD_MGR is None:
            cls.OFFLOAD_MGR = PipelineOffloadManager()
        return  cls.OFFLOAD_MGR

    def __init__(self):
        self._queue = deque()
        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is None:
            self._vpp = 1
        else:
            self._vpp = parallel_state.get_virtual_pipeline_model_parallel_world_size()

        # cache vpp - 1 stages
        self._stages = [[] for _ in range(self._vpp)]
        # allocate streams and events for synchronization
        self._d2h_stream = torch.cuda.Stream()
        self._h2d_stream = torch.cuda.Stream()
        self.reset()

    @property
    def d2h_stream(self):
        return self._d2h_stream

    @property
    def h2d_stream(self):
        return self._h2d_stream

    def reset(self):
        self._inside_context = False
        self._cur_forward_chunk = None
        self._cur_backward_chunk = None
        self._first_last_vpp_rank = True

    def flush(self):
        # put into the queue in the backward order
        if len(self._stages[0]) == len(self._stages[-1]):
            lens = [len(e) for e in self._stages]
            assert min(lens) == max(lens)
            self._stages[-1] = []
            for chunks in reversed(self._stages):
                for chunk in chunks:
                    self.push(chunk)
            for i in range(self._vpp):
                self._stages[i] = []

    def push(self, handler):
        self._queue.append(handler)

    def pop(self):
        assert self.size()
        self._cur_backward_chunk = self._queue.popleft()

    def front(self):
        if not len(self._queue):
            return None
        f = self._queue.popleft()
        self._queue.appendleft(f)
        return f

    def size(self):
        return len(self._queue)

    def reset_chunk_handler(self, num_layer, vp_stage, offload=True, first_layer_index=0):
        if vp_stage is None:
            cur_vpp_rank = 0
        else:
            cur_vpp_rank = vp_stage

        first_last_vpp_rank = self._first_last_vpp_rank
        # rewind
        if cur_vpp_rank == self._vpp - 1:
            self.flush()
        first_last_vpp_rank = first_last_vpp_rank and (cur_vpp_rank == self._vpp - 1)
        cur_chunk = ChunkOffloadHandler(num_layer, first_last_vpp_rank, offload, first_layer_index)
        # save for latter push
        self._stages[cur_vpp_rank].append(cur_chunk)
        if cur_vpp_rank == self._vpp - 1:
            self._first_last_vpp_rank = False
            self.push(cur_chunk)
            self.flush()
        self._cur_forward_chunk = cur_chunk
        cur_chunk.vpp_rank = cur_vpp_rank

    def cur_forward_chunk(self):
        return self._cur_forward_chunk

    def cur_backward_chunk(self):
        return self._cur_backward_chunk

    def __enter__(self):
        self.OFFLOAD_MGR
        self.inside_context = True

        torch._C._autograd._push_saved_tensors_default_hooks(
            self.on_save_for_backward, self.on_get_saved_tensor
        )

    def __exit__(self, *args: Any):
        self.inside_context = False
        torch._C._autograd._pop_saved_tensors_default_hooks()

    def on_save_for_backward(self, tensor: torch.Tensor) -> Any:
        assert self.inside_context
        if self.cur_forward_chunk().is_registered_tensor(tensor.data_ptr()):
            tensor.offloading_activation = True
        return self.cur_forward_chunk().tensor_push(tensor)

    def on_get_saved_tensor(self, saved_state: Any) -> torch.Tensor:
        return self.cur_backward_chunk().tensor_pop(saved_state)


class ChunkOffloadHandler(AsyncDoubleBufferGroupOffloadHandler):
    @staticmethod
    def offload(src_tensor, pin_memory=True):
        """Offload."""
        fp8_offload = isinstance(src_tensor, Float8Tensor)

        cpu_backup = torch.empty(
            src_tensor.size(),
            dtype=torch.uint8 if fp8_offload else src_tensor.dtype,
            layout=src_tensor.layout,
            device="cpu",
            pin_memory=pin_memory,
        )

        if fp8_offload:
            cpu_backup = Float8Tensor.make_like(src_tensor, data=cpu_backup)

        cpu_backup.copy_(src_tensor, non_blocking=pin_memory)
        state = (src_tensor.device, cpu_backup)
        return state

    @staticmethod
    def reload(state, non_blocking=None):
        """Reload."""
        dev, cpu_backup = state
        if non_blocking is None:
            non_blocking = cpu_backup.is_pinned()
        return cpu_backup.to(dev, non_blocking=non_blocking)

    def __init__(self, num_layer, is_first_last_vpp_chunk, offload=True, first_layer_index=0):
        self._num_layers = num_layer
        # Data Structure to maintain reference to activation tensors
        self._tensor_tag_to_state = {}
        # Tracking the number of layers offloaded
        self._offloaded_group_count = 0
        self._is_first_last_vpp_chunk = is_first_last_vpp_chunk

        self._layer_index = first_layer_index
        self.first_layer_index = first_layer_index
        self._tensor_count_current_layer = 0
        self.multi_input_offload_count = False
        self.offload_count_per_layer = defaultdict(int)

        self.tensor_need_offloading_checker = None
        self.torch_tensor_count = 0
        self.d2h_stream = PipelineOffloadManager.get_instance().d2h_stream
        self.h2d_stream = PipelineOffloadManager.get_instance().h2d_stream
        self.do_offload = offload

        self._offload_tensor_ptrs = deque()

    def is_first_last_layer(self):
        return self._is_first_last_vpp_chunk and self.is_last_layer()

    def is_last_layer(self):
        return  (self._layer_index == self._num_layers - 1)

    def tensor_push(self, tensor):
        torch_stray_tensor = isinstance(
            tensor,
            (
                torch._subclasses.fake_tensor.FakeTensor,
                torch._subclasses.functional_tensor.FunctionalTensor,
            ),
        )

        if not torch_stray_tensor:# True
            # obtain a unique tensor tag
            tensor_tag = (self._layer_index, self._tensor_count_current_layer)
            self._tensor_count_current_layer += 1
            assert tensor_tag not in self._tensor_tag_to_state
            self._tensor_tag_to_state[tensor_tag] = tensor
        else:
            tensor_tag = (-1, self.torch_tensor_count)
            self.torch_tensor_count += 1
            self._tensor_tag_to_state[tensor_tag] = tensor
        return tensor_tag

    def tensor_pop(self, tensor_tag):
        assert tensor_tag in self._tensor_tag_to_state, f"{tensor_tag}, {self._tensor_tag_to_state.keys()}"
        tensor = self._tensor_tag_to_state.pop(tensor_tag)
        assert not isinstance(tensor, tuple)
        return tensor

    def set_offloading_checker(self, check_func):
        self.tensor_need_offloading_checker = check_func

    def bulk_offload_group(self, group_to_offload):
        """Bulk offload group."""
        if not self.do_offload:
            return
        assert not self.is_first_last_layer()
        with torch.cuda.stream(self.d2h_stream):
            for tensor_tag, state in self._tensor_tag_to_state.items():
                group_id, _ = tensor_tag
                if group_id == group_to_offload:
                    assert not isinstance(state, tuple)
                    tensor_on_device = state
                    # if offload, return the reference to cpu copy
                    if self.tensor_need_offloading_checker is not None and self.tensor_need_offloading_checker(tensor_on_device):
                        state = self.offload(tensor_on_device)
                        tensor_on_device.record_stream(self.d2h_stream)
                        self.offload_count_per_layer[group_to_offload] += 1
                        self._tensor_tag_to_state[tensor_tag] = state
        self._offloaded_group_count = group_to_offload + 1

    def bulk_reload_group(self, group_to_reload):
        """Bulk reload group."""
        if not self.do_offload:
            return
        with torch.cuda.stream(self.h2d_stream):
            # move back tensors
            for tensor_label, state in self._tensor_tag_to_state.items():
                group_id, _ = tensor_label
                if group_id == group_to_reload:
                    if isinstance(state, tuple):
                        recovered_tensor = self.reload(state)
                        self._tensor_tag_to_state[tensor_label] = recovered_tensor
                        self.offload_count_per_layer[group_to_reload] -= 1
                        if self.offload_count_per_layer[group_to_reload] > 0 and self.multi_input_offload_count:
                            break
        if self.offload_count_per_layer[group_to_reload] == 0:
            self._offloaded_group_count = group_to_reload

    def pre_reload_last_layer(self):
        if not self.do_offload:
            return
        assert not self._is_first_last_vpp_chunk
        if self._num_layers == self._offloaded_group_count:
            self.bulk_reload_group(self._num_layers - 1)
        # assert self._num_layers  - 1 == self._offloaded_group_count

    def should_bulk_offload(self):
        if not self.do_offload:
            return False
        # first backward chunk
        if self.is_first_last_layer():
            return False

        # if next backward chunk is this chunk (for last pp stage)
        next_backward_chunk = PipelineOffloadManager.get_instance().get_instance().front()
        if next_backward_chunk is not None and next_backward_chunk is self:
            if self.is_last_layer():
                return False

        return True

    def forward_sync(self):
        self.d2h_stream.wait_stream(torch.cuda.current_stream())
        torch.cuda.current_stream().wait_stream(self.d2h_stream)

    def bulk_offload(self, release_tensors):
        if self.should_bulk_offload():
            self.bulk_offload_group(self._layer_index)
            if len(release_tensors) > 0:
                cur_stream = torch.cuda.current_stream()
                for release_tensor in release_tensors:
                    release_tensor.record_stream(cur_stream)
                    release_tensor.untyped_storage().resize_(0)

    def on_group_commit_forward(self, release_tensors):
        # wait each other
        self.forward_sync()
        self.bulk_offload(release_tensors)
        self._layer_index = self._layer_index + 1
        self._tensor_count_current_layer = 0

    def bulk_reload(self):
        if self.do_offload:
            assert self._layer_index == self._offloaded_group_count
        if self._layer_index > self.first_layer_index:
            # load next layer
            self.bulk_reload_group(self._layer_index - 1)
        else:
            next_backward_chunk = PipelineOffloadManager.get_instance().front()
            if next_backward_chunk is not None:
                next_backward_chunk.pre_reload_last_layer()

    def backward_sync(self):
        self.h2d_stream.wait_stream(torch.cuda.current_stream())
        torch.cuda.current_stream().wait_stream(self.h2d_stream)

    def on_group_commit_backward(self):
        cur_backward_chunk = PipelineOffloadManager.get_instance().cur_backward_chunk()
        if not cur_backward_chunk is self:
            PipelineOffloadManager.get_instance().pop()
        cur_backward_chunk = PipelineOffloadManager.get_instance().cur_backward_chunk()
        assert cur_backward_chunk is self
        self._layer_index = self._layer_index - 1
        self.backward_sync()

    def on_group_start_forward(self):
        pass

    def on_group_start_backward(self):
        self.h2d_stream.wait_stream(torch.cuda.current_stream())
        self.bulk_reload()

    def register_offload_tensor(self, tensors):
        self.multi_input_offload_count = True
        if isinstance(tensors, list):
            for tensor in tensors:
                self._offload_tensor_ptrs.append(tensor.data_ptr())
        else:
            self._offload_tensor_ptrs.append(tensors.data_ptr())

    def is_registered_tensor(self, tensor_ptr: int) -> bool:
        if len(self._offload_tensor_ptrs) == 0:
            return False
        is_registered = tensor_ptr == self._offload_tensor_ptrs[0]
        if is_registered:
            self._offload_tensor_ptrs.popleft()
        return is_registered


class GroupCommitFunction(torch.autograd.Function):
    """this is a dummy op with output identical to input.
    However, it is necessary for marking a timepoint for offload handler to
    accomplish all synchronizations. Implementing it as a function is necessary
    because we need to actions in both forward and backward.
    """

    @staticmethod
    def forward(ctx, *args):
        # pylint: disable=missing-function-docstring

        release_tensors = args[-1]
        cpu_offload_handler = args[-2]
        tensor = args[:-2]
        cpu_offload_handler.on_group_commit_forward(release_tensors)
        ctx.cpu_offload_handler = cpu_offload_handler

        # return the identical tensor
        return tensor

    @staticmethod
    def backward(ctx, *grad_output):
        # pylint: disable=missing-function-docstring

        cpu_offload_handler = ctx.cpu_offload_handler
        cpu_offload_handler.on_group_commit_backward()
        return grad_output + (None, None)


def group_prefetch_offload_commit(*tensor, release_tensors=[]):
    cur_forward_chunk = PipelineOffloadManager.get_instance().cur_forward_chunk()
    return GroupCommitFunction.apply(*tensor, cur_forward_chunk, release_tensors)


class GroupStartFunction(torch.autograd.Function):
    """this is a dummy op with output identical to input.
    However, it is necessary for marking a timepoint for offload handler to
    accomplish all synchronizations. Implementing it as a function is necessary
    because we need to actions in both forward and backward.
    """

    @staticmethod
    def forward(ctx, tensor, cpu_offload_handler):
        # pylint: disable=missing-function-docstring
        ctx.cpu_offload_handler = cpu_offload_handler

        # return the identical tensor
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        # pylint: disable=missing-function-docstring
        cpu_offload_handler = ctx.cpu_offload_handler
        cpu_offload_handler.on_group_start_backward()
        return grad_output, None


def group_prefetch_offload_start(tensor):
    cur_forward_chunk = PipelineOffloadManager.get_instance().cur_forward_chunk()
    return GroupStartFunction.apply(tensor, cur_forward_chunk)
