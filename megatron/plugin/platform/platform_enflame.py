# Adopted from DeepSpeed Accelerator, https://github.com/deepspeedai/DeepSpeed/

import os
import sys

from .platform_base import PlatformBase

try:
    import torch
    import torch_gcu
except ImportError:
    pass


class PlatformENFLAME(PlatformBase):

    def __init__(self):
        self._name = 'enflame'

    def is_available(self):
        try:
            import torch
            # Determine if we are on a ENFLAME device
            if torch.gcu.device_count() > 0 and torch.gcu.is_available():
                return True
            else:
                return False
        except Exception as e:
            return False

    def get_device_properties(self, device_index=None):
        return torch.gcu.get_device_properties(device_index)

    def get_device_capability(self, device_index=None):
        return torch.gcu.get_device_capability(device_index)

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
            return 'gcu'
        return 'gcu:{}'.format(device_index)

    def device(self, device_index=None):
        return torch.device('gcu', device_index)

    def set_device(self, device_index):
        torch.gcu.set_device(device_index)

    def current_device(self):
        return torch.gcu.current_device()

    def current_device_name(self):
        return 'gcu:{}'.format(torch.gcu.current_device())

    def device_count(self):
        return torch.gcu.device_count()

    def synchronize(self, device_index=None):
        return torch.gcu.synchronize(device_index)

    # RNG APIs
    def random(self):
        return torch.random

    def set_rng_state(self, new_state, device_index=None):
        if device_index is None:
            return torch.gcu.set_rng_state(new_state)

        return torch.gcu.set_rng_state(new_state, device_index)

    def get_rng_state(self, device=None):
        if device is None:
            return torch.gcu.get_rng_state()

        return torch.gcu.get_rng_state(device)

    def manual_seed(self, seed):
        return torch.gcu.manual_seed(seed)

    def manual_seed_all(self, seed):
        return torch.gcu.manual_seed_all(seed)

    def initial_seed(self):
        return torch.gcu.initial_seed()

    @property
    def default_generators(self):
        return torch.gcu.default_generators

    # Streams/Events
    @property
    def Stream(self):
        return torch.gcu.Stream

    def stream(self, stream):
        return torch.gcu.stream(stream)
    
    def set_stream(self, stream):
        return torch.gcu.set_stream(stream)

    def current_stream(self, device_index=None):
        return torch.gcu.current_stream(device_index)

    def default_stream(self, device_index=None):
        return torch.gcu.default_stream(device_index)

    @property
    def MemPool(self):
        return torch.gcu.MemPool

    def use_mem_pool(self, pool):
        return torch.gcu.use_mem_pool(pool)

    @property
    def Event(self):
        return torch.gcu.Event

    # Memory management
    def empty_cache(self):
        return torch.gcu.empty_cache()

    def memory_allocated(self, device_index=None):
        return torch.gcu.memory_allocated(device_index)

    def max_memory_allocated(self, device_index=None):
        return torch.gcu.max_memory_allocated(device_index)

    def reset_max_memory_allocated(self, device_index=None):
        return torch.gcu.reset_max_memory_allocated(device_index)

    def memory_cached(self, device_index=None):
        return torch.gcu.memory_cached(device_index)

    def max_memory_cached(self, device_index=None):
        return torch.gcu.max_memory_cached(device_index)

    def reset_max_memory_cached(self, device_index=None):
        return torch.gcu.reset_max_memory_cached(device_index)

    def memory_stats(self, device_index=None):
        if hasattr(torch.gcu, 'memory_stats'):
            return torch.gcu.memory_stats(device_index)

    def reset_peak_memory_stats(self, device_index=None):
        if hasattr(torch.gcu, 'reset_peak_memory_stats'):
            return torch.gcu.reset_peak_memory_stats(device_index)

    def memory_reserved(self, device_index=None):
        if hasattr(torch.gcu, 'memory_reserved'):
            return torch.gcu.memory_reserved(device_index)

    def max_memory_reserved(self, device_index=None):
        if hasattr(torch.gcu, 'max_memory_reserved'):
            return torch.gcu.max_memory_reserved(device_index)

    def total_memory(self, device_index=None):
        return torch.gcu.get_device_properties(device_index).total_memory

    def available_memory(self, device_index=None):
        return self.total_memory(device_index) - self.memory_allocated(device_index)

    # Data types
    def is_bf16_supported(self):
        if not torch.gcu.is_available():
            return False
        return torch.gcu.is_bf16_supported()

    def is_fp16_supported(self):
        if not torch.gcu.is_available():
            return False
        # Some backends expose an explicit check; otherwise assume fp16 is supported.
        if hasattr(torch.gcu, "is_fp16_supported"):
            return torch.gcu.is_fp16_supported()
        return False

    def supported_dtypes(self):
        supported_dtypes = [torch.float]
        if self.is_fp16_supported():
            supported_dtypes.append(torch.half)
        if self.is_bf16_supported():
            supported_dtypes.append(torch.bfloat16)
        return supported_dtypes

    # Misc
    def amp(self):
        if hasattr(torch.gcu, 'amp'):
            return torch.gcu.amp
        return None

    def range(self, msg):
        if hasattr(torch.cuda.nvtx, 'range'):
            return torch.cuda.nvtx.range(msg)

    def range_push(self, msg):
        if hasattr(torch.cuda.nvtx, 'range_push'):
            return torch.cuda.nvtx.range_push(msg)

    def range_pop(self):
        if hasattr(torch.cuda.nvtx, 'range_pop'):
            return torch.cuda.nvtx.range_pop()

    def lazy_call(self, callback):
        pass

    def is_triton_supported(self):
        pass

    # Graph operations
    def create_graph(self):
        return torch.gcu.TopsGraph()

    def capture_to_graph(self, graph, pool=None, stream=None):
        return torch.gcu.graph(graph, pool, stream)

    def replay_graph(self, graph):
        graph.replay()
        return

    # Tensor operations
    @property
    def BFloat16Tensor(self):
        return torch.gcu.BFloat16Tensor

    @property
    def ByteTensor(self):
        return torch.gcu.ByteTensor

    @property
    def DoubleTensor(self):
        return torch.gcu.DoubleTensor

    @property
    def FloatTensor(self):
        return torch.gcu.FloatTensor

    @property
    def HalfTensor(self):
        return torch.gcu.HalfTensor

    @property
    def IntTensor(self):
        return torch.gcu.IntTensor

    @property
    def LongTensor(self):
        return torch.gcu.LongTensor

    def pin_memory(self, tensor, align_bytes=1):
        return tensor.pin_memory()

    def is_pinned(self, tensor):
        return tensor.is_pinned()

    def on_accelerator(self, tensor):
        device_str = str(tensor.device)
        if device_str.startswith('gcu:'):
            return True
        else:
            return False

    def build_extension(self):
        from torch.utils.cpp_extension import BuildExtension
        return BuildExtension

    def visible_devices_envs(self):
        return ['TOPS_VISIBLE_DEVICES']

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