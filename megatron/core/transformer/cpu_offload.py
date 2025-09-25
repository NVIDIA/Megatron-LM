from collections import deque, defaultdict
import torch
from typing import Any
from transformer_engine.pytorch.float8_tensor import Float8Tensor
from transformer_engine.pytorch.cpu_offload import AsyncDoubleBufferGroupOffloadHandler

# cpu offload for pipeline
DEBUG = False
DEBUG_RANK = 0
MIN_OFFLOADED_TENSOR_SIZE = 1024 * 1024

def print_rank(message):
    assert torch.distributed.is_initialized()
    if DEBUG and torch.distributed.get_rank() == DEBUG_RANK:
        print(message, flush=True)

def set_ideal_affinity_for_current_gpu():
    import cuda.cuda
    import cuda.cudart
    import pynvml
    import uuid
    err, device_id = cuda.cudart.cudaGetDevice()
    assert err == cuda.cudart.cudaError_t.cudaSuccess
    err, device_uuid = cuda.cuda.cuDeviceGetUuid(device_id)
    assert err == cuda.cuda.CUresult.CUDA_SUCCESS
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByUUID("GPU-" + str(uuid.UUID(bytes=device_uuid.bytes)))
    pynvml.nvmlDeviceSetCpuAffinity(handle)

class PipelineOffloadManager:
    OFFLOAD_MGR = None
    @classmethod
    def get_instance(cls):
        if cls.OFFLOAD_MGR is None:
            cls.OFFLOAD_MGR = PipelineOffloadManager()
        return  cls.OFFLOAD_MGR

    def __init__(self):
        from megatron.core import parallel_state
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
        set_ideal_affinity_for_current_gpu()
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
        print_rank("pushing handler")
        self._queue.append(handler)

    def pop(self):
        assert self.size()
        while self._queue:
            self._cur_backward_chunk = self._queue.popleft()
            if not isinstance(self._cur_backward_chunk, NullChunkOffloadHandler):
                break
        print_rank(f"popping handler {self._cur_backward_chunk}")

    def front(self):
        if not len(self._queue):
            return None
        for chunk_handler in self._queue:
            if not isinstance(chunk_handler, NullChunkOffloadHandler):
                return chunk_handler
        return None

    def size(self):
        return len(self._queue)

    def reset_chunk_handler(self, num_layer, vp_stage, offload=True, num_dense_layer=0):
        if vp_stage is None:
            cur_vpp_rank = 0
        else:
            cur_vpp_rank = vp_stage

        first_last_vpp_rank = self._first_last_vpp_rank
        # rewind
        if cur_vpp_rank == self._vpp - 1:
            self.flush()
        first_last_vpp_rank = first_last_vpp_rank and (cur_vpp_rank == self._vpp - 1)
        # If the model chunk contains only the dense layers, initialize a null chunk handler.
        if num_layer <= num_dense_layer:
            cur_chunk = NullChunkOffloadHandler(num_layer, first_last_vpp_rank, offload, num_dense_layer)
        else:
            cur_chunk = ChunkOffloadHandler(num_layer, first_last_vpp_rank, offload, num_dense_layer)
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
        print_rank("__enter__")
        self.OFFLOAD_MGR
        self.inside_context = True

        if not isinstance(self.cur_forward_chunk(), NullChunkOffloadHandler):
            torch._C._autograd._push_saved_tensors_default_hooks(
                self.on_save_for_backward, self.on_get_saved_tensor
            )

    def __exit__(self, *args: Any):
        print_rank("__exit__")
        self.inside_context = False
        if not isinstance(self.cur_forward_chunk(), NullChunkOffloadHandler):
            torch._C._autograd._pop_saved_tensors_default_hooks()

    def on_save_for_backward(self, tensor: torch.Tensor) -> Any:
        print_rank(f"on_save_for_backward {tensor.shape}")
        assert self.inside_context
        return self.cur_forward_chunk().tensor_push(tensor)

    def on_get_saved_tensor(self, saved_state: Any) -> torch.Tensor:
        print_rank(f"on_get_saved_tensor {saved_state}")
        return self.cur_backward_chunk().tensor_pop(saved_state)


class ChunkOffloadHandler(AsyncDoubleBufferGroupOffloadHandler):
    @staticmethod
    def offload(src_tensor, pin_memory=True):
        """Offload."""
        print_rank("offload")
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

        if not src_tensor.is_contiguous():
            src_tensor = src_tensor.contiguous()

        cpu_backup.copy_(src_tensor, non_blocking=pin_memory)
        state = (src_tensor.device, cpu_backup)
        return state

    @staticmethod
    def reload(state, non_blocking=None):
        """Reload."""
        print_rank("reload")
        dev, cpu_backup = state
        if non_blocking is None:
            non_blocking = cpu_backup.is_pinned()
        return cpu_backup.to(dev, non_blocking=non_blocking)

    def __init__(self, num_layer, is_first_last_vpp_chunk, offload=True, num_dense_layer=0):
        self._num_layers = num_layer
        # Data Structure to maintain reference to activation tensors
        self._tensor_tag_to_state = {}
        # Tracking the number of layers offloaded
        # self._offloaded_group_count = 0
        self._is_first_last_vpp_chunk = is_first_last_vpp_chunk

        self._offloaded_group_index = 0
        self._groups_to_offload = []
        self._groups_to_reload = []
        self.num_dense_layer = num_dense_layer
        self._layer_index = 0
        self._tensor_count_current_group = 0
        self.multi_input_offload_count = False
        # self.offload_count_per_layer = defaultdict(int)

        self.torch_tensor_count = 0
        self.d2h_stream = PipelineOffloadManager.get_instance().d2h_stream
        self.h2d_stream = PipelineOffloadManager.get_instance().h2d_stream
        self._offload_events = []
        self.do_offload = offload
        self.is_last_layer = False


    def is_first_last_layer(self):
        """Do not offload the last layer of the last pp stage."""
        print_rank(f"is_first_last_layer {self._is_first_last_vpp_chunk} {self.is_last_layer}")
        return self._is_first_last_vpp_chunk and self.is_last_layer

    def tensor_push(self, tensor):
        print_rank("tensor_push")
        torch_stray_tensor = isinstance(
            tensor,
            (
                torch._subclasses.fake_tensor.FakeTensor,
                torch._subclasses.functional_tensor.FunctionalTensor,
            ),
        )

        if not torch_stray_tensor:# True
            # obtain a unique tensor tag
            tensor_tag = (self._offloaded_group_index, self._tensor_count_current_group)
            self._tensor_count_current_group += 1
            assert tensor_tag not in self._tensor_tag_to_state
            self._tensor_tag_to_state[tensor_tag] = tensor
        else:
            tensor_tag = (-1, self.torch_tensor_count)
            self.torch_tensor_count += 1
            self._tensor_tag_to_state[tensor_tag] = tensor
        print_rank(f"tensor_push {tensor_tag}")
        return tensor_tag

    def tensor_pop(self, tensor_tag):
        print_rank(f"tensor_pop {tensor_tag}")
        assert tensor_tag in self._tensor_tag_to_state, f"{tensor_tag}, {self._tensor_tag_to_state.keys()}"
        tensor = self._tensor_tag_to_state.pop(tensor_tag)
        assert not isinstance(tensor, tuple)
        print_rank(f"tensor_pop {tensor.shape}")
        return tensor

    def tensor_need_offloading_checker(self, tensor):
        """Check if the tensor needs to be offloaded."""
        if tensor.numel() < MIN_OFFLOADED_TENSOR_SIZE:
            return False
        if hasattr(tensor, "offloading_activation") and not tensor.offloading_activation:
            return False
        return True

    def bulk_offload_group(self, group_to_offload):
        """offload a group of tensors recorded in tensor_push().
        """
        print_rank("bulk_offload_group")
        if not self.do_offload:
            return
        assert not self.is_first_last_layer()
        group_id_to_offload, name = group_to_offload
        torch.cuda.nvtx.range_push(name)
        with torch.cuda.stream(self.d2h_stream):
            for tensor_tag, state in self._tensor_tag_to_state.items():
                group_id, _ = tensor_tag
                if group_id == group_id_to_offload:
                    print_rank(f"tensor_tag {tensor_tag}")
                    print_rank(f"group_to_offload {group_to_offload}")
                    assert not isinstance(state, tuple)
                    tensor_on_device = state
                    # if offload, return the reference to cpu copy
                    # print_rank(f"tensor_need_offloading_checker {self.tensor_need_offloading_checker(tensor_on_device)}")
                    # print_rank(f"tensor_on_device {tensor_on_device.shape}")
                    if self.tensor_need_offloading_checker(tensor_on_device):
                        state = self.offload(tensor_on_device)
                        tensor_on_device.record_stream(self.d2h_stream)
                        # self.offload_count_per_layer[group_to_offload] += 1
                        self._tensor_tag_to_state[tensor_tag] = state
        # self._offloaded_group_count = group_to_offload + 1
        print_rank("exit bulk_offload_group")
        torch.cuda.nvtx.range_pop()

    def bulk_reload_group(self, group_to_reload):
        """Bulk reload group."""
        print_rank("bulk_reload_group")
        if not self.do_offload:
            return
        found_reload_group = False
        group_id_to_reload, name = group_to_reload
        torch.cuda.nvtx.range_push(name)
        with torch.cuda.stream(self.h2d_stream):
            # move back tensors
            for tensor_label, state in self._tensor_tag_to_state.items():
                group_id, _ = tensor_label
                if group_id == group_id_to_reload:
                    found_reload_group = True
                    event = self._offload_events[-1]
                    if isinstance(state, tuple):
                        # make sure the tensor is already offloaded to cpu before reloading it.
                        torch.cuda.current_stream().wait_event(event)
                        recovered_tensor = self.reload(state)
                        print_rank(f"recovered_tensor {recovered_tensor.shape}")
                        self._tensor_tag_to_state[tensor_label] = recovered_tensor
        torch.cuda.nvtx.range_pop()
        if found_reload_group:
            self._offload_events.pop()
        return found_reload_group

    def pre_reload_last_layer(self):
        """Pre-reload the last layer of the next model chunk."""
        print_rank("pre_reload_last_layer")
        if not self.do_offload:
            return
        assert not self._is_first_last_vpp_chunk
        print_rank(f"len(self._groups_to_reload) {len(self._groups_to_reload)}")
        print_rank(f"len(self._offload_events) {len(self._offload_events)}")
        if len(self._groups_to_reload) > 0:
            if self.bulk_reload_group(self._groups_to_reload[-1]):
                self._groups_to_reload.pop()

    def should_bulk_offload(self):
        """Check if the chunk should be offloaded."""
        if not self.do_offload:
            return False
        if self._layer_index < self.num_dense_layer:
            return False
        # first backward chunk
        if self.is_first_last_layer():
            return False

        # if next backward chunk is this chunk (for last pp stage)
        next_backward_chunk = PipelineOffloadManager.get_instance().get_instance().front()
        if next_backward_chunk is not None and next_backward_chunk is self:
            if self.is_last_layer:
                return False

        return True

    def bulk_offload(self, release_tensors):
        print_rank("bulk_offload")
        if self.should_bulk_offload():
            group_to_offload = self._groups_to_offload.pop()
            self._groups_to_reload.append(group_to_offload)
            self.bulk_offload_group(group_to_offload)
            event = torch.cuda.Event()
            event.record(self.d2h_stream)
            # TODO: check if we really need it.
            self._offload_events.append(event)
            if len(release_tensors) > 0:
                cur_stream = torch.cuda.current_stream()
                for release_tensor in release_tensors:
                    release_tensor.record_stream(cur_stream)
                    release_tensor.untyped_storage().resize_(0)
        print_rank("exit bulk_offload")

    def on_group_commit_forward(self, release_tensors):
        """Offload a group of tensors."""
        print_rank("on_group_commit_forward")
        # wait for the compute stream for offloading
        self.d2h_stream.wait_stream(torch.cuda.current_stream())
        self.bulk_offload(release_tensors)
        print_rank("exit on_group_commit_forward")

    def bulk_reload(self):
        print_rank("bulk_reload")
        if len(self._groups_to_reload) > 0 and self._layer_index > self.num_dense_layer:
            # load next layer
            if self.bulk_reload_group(self._groups_to_reload[-1]):
                self._groups_to_reload.pop()
        else:
            # load the last layer of one backward chunk in advance
            next_backward_chunk = PipelineOffloadManager.get_instance().front()
            if next_backward_chunk is not None:
                next_backward_chunk.pre_reload_last_layer()

    def on_group_commit_backward(self):
        """Prepare for reloadingthe next group of tensors."""
        print_rank("on_group_commit_backward")
        cur_backward_chunk = PipelineOffloadManager.get_instance().cur_backward_chunk()
        if not cur_backward_chunk is self:
            PipelineOffloadManager.get_instance().pop()
        cur_backward_chunk = PipelineOffloadManager.get_instance().cur_backward_chunk()
        assert cur_backward_chunk is self
        # self.h2d_stream.wait_stream(torch.cuda.current_stream())
        self._offloaded_group_index = self._offloaded_group_index - 1

    def on_group_start_forward(self, name):
        """Prepare for offloading the next group of tensors."""
        print_rank(f"on_group_start_forward {self._layer_index} {self._num_layers}")
        if self._layer_index == self._num_layers:
            self.is_last_layer = True
        else:
            self.is_last_layer = False
        self._offloaded_group_index = self._offloaded_group_index + 1
        self._tensor_count_current_group = 0
        self._groups_to_offload.append((self._offloaded_group_index, name))

    def on_group_start_backward(self):
        """Reload the next group of tensors."""
        print_rank("on_group_start_backward")
        self.h2d_stream.wait_stream(torch.cuda.current_stream())
        self.bulk_reload()
    
    def on_layer_start_forward(self):
        """Increment the layer index."""
        print_rank("on_layer_start_forward")
        self._layer_index = self._layer_index + 1
    
    def on_layer_start_backward(self):
        """When the bprop of one layer finishes, make sure the reloading jobs on h2d stream are done.
        """
        print_rank("on_layer_start_backward")
        self._layer_index = self._layer_index - 1
        torch.cuda.current_stream().wait_stream(self.h2d_stream)

class NullChunkOffloadHandler(ChunkOffloadHandler):
    pass

class GroupCommitFunction(torch.autograd.Function):
    """this is a dummy op with output identical to input.
    However, it is necessary for marking a timepoint for offload handler to
    accomplish all synchronizations. Implementing it as a function is necessary
    because we need to actions in both forward and backward.
    """

    @staticmethod
    def forward(ctx, *args):
        # pylint: disable=missing-function-docstring
        print_rank("GroupCommitFunction forward")

        release_tensors = args[-1]
        cpu_offload_handler = args[-2]
        tensor = args[:-2]
        if not isinstance(cpu_offload_handler, NullChunkOffloadHandler):
            cpu_offload_handler.on_group_commit_forward(release_tensors)
        ctx.cpu_offload_handler = cpu_offload_handler

        # return the identical tensor
        return tensor

    @staticmethod
    def backward(ctx, *grad_output):
        # pylint: disable=missing-function-docstring
        print_rank("GroupCommitFunction backward")

        cpu_offload_handler = ctx.cpu_offload_handler
        if not isinstance(cpu_offload_handler, NullChunkOffloadHandler):
            cpu_offload_handler.on_group_commit_backward()
        return grad_output + (None, None)


def group_prefetch_offload_commit(*tensor, release_tensors=[]):
    """Specify the tensors to be released after offloading.
    release_tensors is a list of tensors to be released after offloading.
    The tensors will be untyped_storage().resize_(0) after offloading.
    Note: specify the tensors only when they are not automatically released by torch gc.
    """
    cur_forward_chunk = PipelineOffloadManager.get_instance().cur_forward_chunk()
    return GroupCommitFunction.apply(*tensor, cur_forward_chunk, release_tensors)


class GroupStartFunction(torch.autograd.Function):
    """this is a dummy op with output identical to input.
    However, it is necessary for marking a timepoint for offload handler to
    accomplish all synchronizations. Implementing it as a function is necessary
    because we need to actions in both forward and backward.
    """

    @staticmethod
    def forward(ctx, tensor, cpu_offload_handler, name):
        # pylint: disable=missing-function-docstring
        ctx.cpu_offload_handler = cpu_offload_handler
        print_rank("GroupStartFunction forward")

        if not isinstance(cpu_offload_handler, NullChunkOffloadHandler):
            cpu_offload_handler.on_group_start_forward("activation offloading " + name)
        # return the identical tensor
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        print_rank("GroupStartFunction backward")
        # pylint: disable=missing-function-docstring
        cpu_offload_handler = ctx.cpu_offload_handler
        if not isinstance(cpu_offload_handler, NullChunkOffloadHandler):
            cpu_offload_handler.on_group_start_backward()
        return grad_output, None, None


def group_prefetch_offload_start(tensor, name=None):
    cur_forward_chunk = PipelineOffloadManager.get_instance().cur_forward_chunk()
    return GroupStartFunction.apply(tensor, cur_forward_chunk, name)

class MarkLayerStartFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, cpu_offload_handler):
        ctx.cpu_offload_handler = cpu_offload_handler
        print_rank("MarkLayerStartFunction forward")
        if not isinstance(cpu_offload_handler, NullChunkOffloadHandler):
            cpu_offload_handler.on_layer_start_forward()
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        print_rank("MarkLayerStartFunction backward")
        cpu_offload_handler = ctx.cpu_offload_handler
        if not isinstance(cpu_offload_handler, NullChunkOffloadHandler):
            cpu_offload_handler.on_layer_start_backward()
        return grad_output, None, None

def mark_layer_start(tensor):
    cur_forward_chunk = PipelineOffloadManager.get_instance().cur_forward_chunk()
    return MarkLayerStartFunction.apply(tensor, cur_forward_chunk)