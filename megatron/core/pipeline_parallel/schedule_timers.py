import torch

from megatron.core.timers import Timer


class AsyncTimer:
    def __init__(self, name):
        self.start_events = []
        self.end_events = []
        self._started = False
        self.name = name

    def start(self):
        assert not self._started, 'timer has already been started'
        self.start_events.append(torch.cuda.Event(enable_timing=True))
        self.start_events[-1].record()
        self._started = True

    def stop(self):
        assert self._started, 'timer is not started'
        self.end_events.append(torch.cuda.Event(enable_timing=True))
        self.end_events[-1].record()
        self._started = False
    
    def reset(self):
        self.start_events = []
        self.end_events = []

    def elapsed(self, reset=True):
        assert not self._started, 'timer is still elapsing'
        # Get the elapsed time.
        torch.cuda.synchronize()
        total = sum(s.elapsed_time(e) for s, e in zip(self.start_events, self.end_events))
        
        # Reset the elapsed time
        if reset:
            self.reset()
        return total / 1000


class SyncTimers:
    comm_time = 0
    _joint_conclusion = None
    timer_cls = None

    chunks = []

    def __init__(self):
        self.f = Timer('f')
        self.b = Timer('b')
        self.s = Timer('s')
        self.input_f = Timer('input_f')
        self.input_b = Timer('input_b')
        self.f_cnt = 0
        self.b_cnt = 0
        self.s_cnt = 0
        self.input_f_cnt = 0
        self.input_b_cnt = 0
        self.f_mem = 0
        self.b_mem = 0

    def conclusion(self):
        assert self.f_cnt > 0
        assert self.b_cnt > 0
        avg_f = int(self.f.elapsed(reset=False) / self.f_cnt * 1000000)
        avg_f_mem = self.f_mem / self.f_cnt // 1000000
        avg_b = int(self.b.elapsed(reset=False) / self.b_cnt * 1000000)
        avg_b_mem = self.b_mem / self.b_cnt // 1000000
        if self.s_cnt > 0:
            avg_s = int(self.s.elapsed(reset=False) / self.s_cnt * 1000000)
        else:
            avg_s = 0
        if self.input_f_cnt > 0:
            avg_input_f = int(self.input_f.elapsed(reset=False) / self.input_f_cnt * 1000000)
        else:
            avg_input_f = 0
        if self.input_b_cnt > 0:
            avg_input_b = int(self.input_b.elapsed(reset=False) / self.input_b_cnt * 1000000)
        else:
            avg_input_b = 0
        return (avg_f, avg_b, avg_s, avg_input_f, avg_input_b, int(self.comm_time * 1000000), avg_f_mem, avg_b_mem)

    @classmethod
    def for_chunk(cls, chunk):
        while len(cls.chunks) <= chunk:
            cls.chunks.append(cls())
        return cls.chunks[chunk]

    @classmethod
    def joint_conclusion(cls, global_reduce=True):
        if not global_reduce:
            ret = [x.conclusion() for x in cls.chunks]
            ret = list(zip(ret))
            return ret

        if cls._joint_conclusion is not None:
            return cls._joint_conclusion
        
        ret = [x.conclusion() for x in cls.chunks]
        ret = list(zip(ret))
        ret_tensor = torch.tensor(
            ret,
            dtype=torch.float32,
            device=torch.cuda.current_device(),
        )
        torch.distributed.all_reduce(
            ret_tensor,
            op=torch.distributed.ReduceOp.AVG,
        )
        cls._joint_conclusion = ret_tensor.tolist()
        return cls._joint_conclusion


class AsyncTimers:
    comm_time = 0
    _joint_conclusion = None
    timer_cls = None

    chunks = []

    def __init__(self):
        self.f = AsyncTimer('f')
        self.b = AsyncTimer('b')
        self.s = AsyncTimer('s')
        self.input_f = AsyncTimer('input_f')
        self.input_b = AsyncTimer('input_b')
        self.f_cnt = 0
        self.b_cnt = 0
        self.s_cnt = 0
        self.input_f_cnt = 0
        self.input_b_cnt = 0
        self.f_mem = 0
        self.b_mem = 0

    def conclusion(self):
        assert self.f_cnt > 0
        assert self.b_cnt > 0
        avg_f = int(self.f.elapsed(reset=False) / self.f_cnt * 1000000)
        avg_f_mem = self.f_mem / self.f_cnt // 1000000
        avg_b = int(self.b.elapsed(reset=False) / self.b_cnt * 1000000)
        avg_b_mem = self.b_mem / self.b_cnt // 1000000
        if self.s_cnt > 0:
            avg_s = int(self.s.elapsed(reset=False) / self.s_cnt * 1000000)
        else:
            avg_s = 0
        if self.input_f_cnt > 0:
            avg_input_f = int(self.input_f.elapsed(reset=False) / self.input_f_cnt * 1000000)
        else:
            avg_input_f = 0
        if self.input_b_cnt > 0:
            avg_input_b = int(self.input_b.elapsed(reset=False) / self.input_b_cnt * 1000000)
        else:
            avg_input_b = 0
        return (avg_f, avg_b, avg_s, avg_input_f, avg_input_b, int(self.comm_time * 1000000), avg_f_mem, avg_b_mem)

    @classmethod
    def for_chunk(cls, chunk):
        while len(cls.chunks) <= chunk:
            cls.chunks.append(cls())
        return cls.chunks[chunk]

    @classmethod
    def joint_conclusion(cls, global_reduce=True):
        if not global_reduce:
            ret = [x.conclusion() for x in cls.chunks]
            ret = list(zip(ret))
            return ret

        if cls._joint_conclusion is not None:
            return cls._joint_conclusion
        
        ret = [x.conclusion() for x in cls.chunks]
        ret = list(zip(ret))
        ret_tensor = torch.tensor(
            ret,
            dtype=torch.float32,
            device=torch.cuda.current_device(),
        )
        torch.distributed.all_reduce(
            ret_tensor,
            op=torch.distributed.ReduceOp.AVG,
        )
        cls._joint_conclusion = ret_tensor.tolist()
        return cls._joint_conclusion


class ScheduleTimers:
    sync_timer = True
    iter_counter = 0
    comm_time = 0

    @classmethod
    def for_chunk(cls, chunk):
        if cls.sync_timer:
            return SyncTimers.for_chunk(chunk)
        else:
            return AsyncTimers.for_chunk(chunk)

    @classmethod
    def joint_conclusion(cls, sync_timer=True, global_reduce=True):
        SyncTimers.comm_time = ScheduleTimers.comm_time
        AsyncTimers.comm_time = ScheduleTimers.comm_time
        if sync_timer:
            return SyncTimers.joint_conclusion(global_reduce)
        else:
            return AsyncTimers.joint_conclusion(global_reduce)
