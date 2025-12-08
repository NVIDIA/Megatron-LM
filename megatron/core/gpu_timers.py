import torch
from typing import Optional
from contextlib import contextmanager
from collections import defaultdict

class GPUTimer:

    def __init__(self, use_gpu_timer):
        self._starts = defaultdict(list)  # name -> [Event, ...]
        self._ends   = defaultdict(list)  # name -> [Event, ...]
        self._times  = defaultdict(list)  # name -> [elapsed_ms, ...]
        self.inactive = not use_gpu_timer

    def activate(self):
        self.inactive = False

    def inactivate(self):
        self.inactive = True

    def start(self, name: str = "default"):
        if self.inactive:
            return
        evt = torch.cuda.Event(enable_timing=True)
        evt.record()
        self._starts[name].append(evt)

    def stop(self, name: str = "default"):
        if self.inactive:
            return
        if not self._starts[name]:
            raise ValueError(f"No start event recorded for '{name}'")
        end_evt = torch.cuda.Event(enable_timing=True)
        end_evt.record()
        self._ends[name].append(end_evt)

    def compute(self, name: str = None):
        if self.inactive:
            return
        keys = [name] if name is not None else list(self._starts.keys())
        for key in keys:
            starts = self._starts[key]
            ends   = self._ends[key]
            n = min(len(starts), len(ends))
            for i in range(n):
                if i < len(self._times[key]):
                    continue
                s_evt = starts[i]
                e_evt = ends[i]

                e_evt.synchronize()
                elapsed_ms = s_evt.elapsed_time(e_evt)
                self._times[key].append(elapsed_ms)

    def elapsed(self, name: str = "default"):
        if self.inactive:
            return
        if name not in self._times or not self._times[name]:
            raise ValueError(f"No computed timings for '{name}'")
        return list(self._times[name])

    def reset(self, name: str = None):
        if self.inactive:
            return
        if name is None:
            self._starts.clear()
            self._ends.clear()
            self._times.clear()
        else:
            self._starts.pop(name, None)
            self._ends.pop(name, None)
            self._times.pop(name, None)

    def summary(self):
        if self.inactive:
            return
        result = {key: list(vals) for key, vals in self._times.items()}
        for name, times in result.items():
            print(f"DP rank {torch.distributed.get_rank()} {name}: {times} ms")
        return result

    @contextmanager
    def time(self, name: str = "default", auto_compute: bool = True):
        self.start(name)
        yield
        self.stop(name)
        if auto_compute:
            self.compute(name)
