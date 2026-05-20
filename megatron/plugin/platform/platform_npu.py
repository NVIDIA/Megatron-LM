# Adopted from DeepSpeed Accelerator, https://github.com/deepspeedai/DeepSpeed/

import os
import sys
import functools
import torch

from .platform_base import PlatformBase

try:
    import torch_npu
except ImportError:
    pass


class PlatformNPU(PlatformBase):

    def __init__(self):
        self._name = 'npu'

    def is_available(self):
        try:
            import torch
            # Determine if we are on a NPU device.
            if torch_npu.npu.device_count() > 0 and torch_npu.npu.is_available():  #ignore-npu
                return True
            else:
                return False
        except Exception as e:
            return False

    def get_device_properties(self, device_index=None):
        return torch_npu.npu.get_device_properties(device_index)

    def get_device_capability(self, device_index=None):
        return torch_npu.npu.get_device_capability(device_index)

    def is_synchronized_device(self):
        return False

    def use_host_timers(self):
        return self.is_synchronized_device()

    def resolves_data_dependency(self):
        return self.is_synchronized_device()

    def handles_memory_backpressure(self):
        return self.is_synchronized_device()

    # Device APIs
    def device_name(self, device_index=None):
        if device_index is None:
            return 'npu'
        return 'npu:{}'.format(device_index)

    def device(self, device_index=None):
        return torch.device('npu', device_index)

    def set_device(self, device_index):
        torch_npu.npu.set_device(device_index)

    def current_device(self):
        return torch_npu.npu.current_device()

    def current_device_name(self):
        return 'npu:{}'.format(torch_npu.npu.current_device())

    def device_count(self):
        return torch_npu.npu.device_count()

    def synchronize(self, device_index=None):
        return torch_npu.npu.synchronize(device_index)

    # RNG APIs
    def random(self):
        return torch.random

    def set_rng_state(self, new_state, device_index=None):
        if device_index is None:
            return torch_npu.npu.set_rng_state(new_state)

        return torch_npu.npu.set_rng_state(new_state, device_index)

    def get_rng_state(self, device=None):
        if device is None:
            return torch_npu.npu.get_rng_state()

        return torch_npu.npu.get_rng_state(device)

    def manual_seed(self, seed):
        return torch_npu.npu.manual_seed(seed)

    def manual_seed_all(self, seed):
        return torch_npu.npu.manual_seed_all(seed)

    def initial_seed(self):
        return torch_npu.npu.initial_seed()

    @property
    def default_generators(self):
        return torch.npu.default_generators

    # Streams/Events
    @property
    def Stream(self):
        return torch_npu.npu.Stream

    def stream(self, stream):
        return torch_npu.npu.stream(stream)
    
    def set_stream(self, stream):
        return torch_npu.npu.set_stream(stream)

    def current_stream(self, device_index=None):
        return torch_npu.npu.current_stream(device_index)

    def default_stream(self, device_index=None):
        return torch_npu.npu.default_stream(device_index)

    @property
    def MemPool(self):
        return torch.npu.MemPool

    def use_mem_pool(self, pool):
        return torch.npu.use_mem_pool(pool)

    @property
    def Event(self):
        return torch_npu.npu.Event

    # Memory management
    def empty_cache(self):
        return torch_npu.npu.empty_cache()

    def memory_allocated(self, device_index=None):
        return torch_npu.npu.memory_allocated(device_index)

    def max_memory_allocated(self, device_index=None):
        return torch_npu.npu.max_memory_allocated(device_index)

    def reset_max_memory_allocated(self, device_index=None):
        return torch_npu.npu.reset_max_memory_allocated(device_index)

    def memory_cached(self, device_index=None):
        return torch_npu.npu.memory_cached(device_index)

    def max_memory_cached(self, device_index=None):
        return torch_npu.npu.max_memory_cached(device_index)

    def reset_max_memory_cached(self, device_index=None):
        return torch_npu.npu.reset_max_memory_cached(device_index)

    def memory_stats(self, device_index=None):
        if hasattr(torch_npu.npu, 'memory_stats'):
            return torch_npu.npu.memory_stats(device_index)

    def reset_peak_memory_stats(self, device_index=None):
        if hasattr(torch_npu.npu, 'reset_peak_memory_stats'):
            return torch_npu.npu.reset_peak_memory_stats(device_index)

    def memory_reserved(self, device_index=None):
        if hasattr(torch_npu.npu, 'memory_reserved'):
            return torch_npu.npu.memory_reserved(device_index)

    def max_memory_reserved(self, device_index=None):
        if hasattr(torch_npu.npu, 'max_memory_reserved'):
            return torch_npu.npu.max_memory_reserved(device_index)

    def total_memory(self, device_index=None):
        return torch_npu.npu.get_device_properties(device_index).total_memory

    def available_memory(self, device_index=None):
        return self.total_memory(device_index) - self.memory_allocated(device_index)

    # Data types
    def is_bf16_supported(self):
        if not torch_npu.npu.is_available():
            return False
        return True

    def is_fp16_supported(self):
        if not torch_npu.npu.is_available():
            return False
        return True

    def supported_dtypes(self):
        supported_dtypes = [torch.float]
        if self.is_fp16_supported():
            supported_dtypes.append(torch.half)
        if self.is_bf16_supported():
            supported_dtypes.append(torch.bfloat16)
        return supported_dtypes

    # Misc
    def amp(self):
        if hasattr(torch_npu.npu, 'amp'):
            return torch_npu.npu.amp
        return None

    def range(self, msg):
        if hasattr(torch_npu.npu.mstx, 'mstx_range'):
            return torch_npu.npu.mstx.mstx_range(msg)

    def range_push(self, msg):
        if hasattr(torch_npu.npu.mstx, 'range_start'):
            return torch_npu.npu.mstx.range_start(msg)

    def range_pop(self):
        if hasattr(torch_npu.npu.mstx, 'range_end'):
            return torch_npu.npu.mstx.range_end()

    def lazy_call(self, callback):
        pass

    def is_triton_supported(self):
        pass

    # Graph operations
    def create_graph(self):
        return torch.npu.NPUGraph()

    def capture_to_graph(self, graph, pool=None, stream=None):
        return torch.npu.graph(graph, pool, stream)

    def replay_graph(self, graph):
        graph.replay()
        return

    # Tensor operations

    @property
    def BFloat16Tensor(self):
        return torch.npu.BFloat16Tensor
        # return functools.partial(torch.tensor, dtype=torch.bfloat16, device='npu')

    @property
    def ByteTensor(self):
        return torch.npu.ByteTensor
        # return functools.partial(torch.tensor, dtype=torch.uint8, device='npu')

    @property
    def DoubleTensor(self):
        return torch.npu.DoubleTensor
        # return functools.partial(torch.tensor, dtype=torch.double, device='npu')

    @property
    def FloatTensor(self):
        return torch.npu.FloatTensor
        # return functools.partial(torch.tensor, dtype=torch.float, device='npu')

    @property
    def HalfTensor(self):
        return torch.npu.HalfTensor
        # return functools.partial(torch.tensor, dtype=torch.half, device='npu')

    @property
    def IntTensor(self):
        return torch.npu.IntTensor
        # return functools.partial(torch.tensor, dtype=torch.int, device='npu')

    @property
    def LongTensor(self):
        return torch.npu.LongTensor
        # return functools.partial(torch.tensor, dtype=torch.long, device='npu')

    def pin_memory(self, tensor, align_bytes=1):
        return tensor.pin_memory()

    def is_pinned(self, tensor):
        return tensor.is_pinned()

    def on_accelerator(self, tensor):
        device_str = str(tensor.device)
        if device_str.startswith('npu:'):
            return True
        else:
            return False

    def build_extension(self):
        from torch.utils.cpp_extension import BuildExtension
        return BuildExtension

    def visible_devices_envs(self):
        return ['ASCEND_RT_VISIBLE_DEVICES']

    def set_visible_devices_envs(self, current_env, local_accelerator_ids):
        for env in self.visible_devices_envs():
            current_env[env] = ",".join(map(str, local_accelerator_ids))

    def get_compile_backend(self):
        pass

    def set_compile_backend(self, backend):
        pass

    def temperature(self):
        pass

    def power_draw(self):
        pass

    def utilization(self):
        pass

    def clock_rate(self):
        pass
