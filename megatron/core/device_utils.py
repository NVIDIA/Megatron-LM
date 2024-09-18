import os
from typing import Union
import torch

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    import torch_xla.distributed.xla_backend as xb
except ImportError:
    xm = None
    xr = None
    xb = None


def get_xla_model():
    return xm


def get_xla_runtime():
    return xr


def get_current_device() -> torch.device:
    global __current_device

    try:
        return __current_device
    except NameError:
        if xm is not None:
            __current_device = xm.xla_device()
        elif torch.cuda.is_available():
            local_rank = int(os.getenv("LOCAL_RANK", 0))
            __current_device = torch.device(f'cuda:{local_rank}')
            torch.cuda.set_device(__current_device)
        else:
            device = os.getenv("DEFAULT_DEVICE", "cpu")
            __current_device = torch.device(device)

    return __current_device


def get_current_device_type() -> str:
    global __current_device_type

    try:
        return __current_device_type
    except NameError:
        if xm is not None:
            __current_device_type = "xla"
        elif torch.cuda.is_available():
            __current_device_type = "cuda"
        else:
            __current_device_type = os.getenv("DEFAULT_DEVICE_TYPE", "cpu")

    return __current_device_type


def get_local_device_count() -> int:
    device_count = 1

    if xr is not None:
        device_count = xr.global_device_count()
    elif torch.cuda.is_available():
        device_count = torch.cuda.device_count()
    
    return device_count


def get_distributed_backend(backend=None) -> str:
    if xm is not None:
        backend = "xla"
    elif torch.cuda.is_available():
        backend = backend if backend is not None else "nccl"
    else:
        backend = backend if backend is not None else "gloo"

    return backend


def get_distributed_init_method() -> str:
    if xm is not None:
        init_method = 'xla://'
    else:
        init_method =  "env://"

    return init_method


def get_current_rng_state() -> Union[torch.Tensor, int]:
    if torch.cuda.is_available():
        rng_state = torch.cuda.get_rng_state(device=get_current_device())
    elif xm:
        rng_state = xm.get_rng_state(device=get_current_device())
    else:
        rng_state = torch.get_rng_state()

    return rng_state


def set_manual_seed(seed: int):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    elif xm is not None:
        xm.set_rng_state(seed, device=get_current_device())
    else:
        torch.manual_seed(seed)


def set_current_rng_state(new_state):
    if torch.cuda.is_available():
        new_state = new_state.type(torch.ByteTensor)
        torch.cuda.set_rng_state(new_state, device=get_current_device())
    elif xm is not None:
        new_state = int(new_state)
        xm.set_rng_state(new_state, device=get_current_device())
    else:
        new_state = new_state.type(torch.ByteTensor)
        torch.set_rng_state(new_state)

if xb:
    def make_send_channel_id_impl(self, dst_rank, tag):
        return int(dst_rank)*2

    def make_recv_channel_id_impl(self, src_rank, tag):
        return int(src_rank)*3
    
    xb.ProcessGroupXla.make_send_channel_id = make_send_channel_id_impl
    xb.ProcessGroupXla.make_recv_channel_id = make_recv_channel_id_impl