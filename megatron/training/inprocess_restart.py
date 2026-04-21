# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import os
import socket
from datetime import timedelta

try:
    import nvidia_resiliency_ext.inprocess as inprocess
except ImportError:
    inprocess = None

import warnings

import torch

from megatron.core import rerun_state_machine
from megatron.training import get_args

from . import arguments


def destroy_state():
    from . import training
    training.destroy_global_state()
    rerun_state_machine.destroy_rerun_state_machine()


def inprocess_restart(train, args):
    if inprocess is None:
        warnings.warn('In-process restart is not available')
        return train

    if 'TORCH_CPP_LOG_LEVEL' not in os.environ or os.environ['TORCH_CPP_LOG_LEVEL'] not in (
        'error',
        'fatal',
    ):
        warnings.warn(
            'Set TORCH_CPP_LOG_LEVEL=error to suppress c10d waitForInput timeout warning messages'
        )

    # Layers represents a configuration for a layer of branches at a certain
    # depth in a topology tree constructed by inprocess.rank_assignment.Tree.
    # First layer contains all ranks and it's the root of the topology tree,
    # the second optional layer groups ranks by nodes.
    layers = [
        inprocess.rank_assignment.Layer(
            min_ranks=args.inprocess_active_world_size,
            max_ranks=args.inprocess_active_world_size,
            flag=inprocess.rank_assignment.LayerFlag.RESERVE,
        )
    ]
    if args.inprocess_granularity == 'node':
        device_count = torch.cuda.device_count()

        layers.append(
            inprocess.rank_assignment.Layer(
                min_ranks=device_count,
                max_ranks=device_count,
                key_or_fn=lambda _: socket.gethostname(),
                flag=inprocess.rank_assignment.LayerFlag.RESERVE,
            )
        )

    finalize = [
        inprocess.finalize.ThreadedFinalize(timeout=timedelta(seconds=10), fn=destroy_state)
    ]

    if args.inprocess_empty_cuda_cache:
        finalize.append(
            inprocess.finalize.ThreadedFinalize(
                timeout=timedelta(seconds=10), fn=torch.cuda.empty_cache
            )
        )

    initialize = inprocess.Compose(
        inprocess.initialize.RetryController(min_world_size=args.inprocess_active_world_size),
        inprocess.nested_restarter.NestedRestarterHandlingCompleted(),
    )
    abort = inprocess.Compose(
        inprocess.abort.AbortTransformerEngine(),
        inprocess.abort.AbortTorchDistributed(),
        inprocess.nested_restarter.NestedRestarterHandlingStarting(),
    )
    completion = inprocess.nested_restarter.NestedRestarterFinalized()
    terminate = inprocess.nested_restarter.NestedRestarterAborted()

    train = inprocess.Wrapper(
        store_kwargs={
            'timeout': timedelta(seconds=300),
            'port': int(os.environ['MASTER_PORT']) + 2,
        },
        initialize=initialize,
        abort=abort,
        completion=completion,
        terminate=terminate,
        health_check=inprocess.health_check.CudaHealthCheck(timeout=timedelta(seconds=10)),
        rank_assignment=inprocess.rank_assignment.Tree(layers=layers),
        finalize=inprocess.Compose(*finalize),
        heartbeat_interval=timedelta(seconds=args.inprocess_heartbeat_interval),
        heartbeat_timeout=timedelta(seconds=args.inprocess_heartbeat_timeout),
        barrier_timeout=timedelta(seconds=args.inprocess_barrier_timeout),
        completion_timeout=timedelta(seconds=args.inprocess_completion_timeout),
        monitor_process_interval=timedelta(seconds=args.inprocess_monitor_process_interval),
        monitor_thread_interval=timedelta(seconds=args.inprocess_monitor_thread_interval),
        last_call_wait=timedelta(seconds=args.inprocess_last_call_wait),
        soft_timeout=timedelta(seconds=args.inprocess_soft_timeout),
        hard_timeout=timedelta(seconds=args.inprocess_hard_timeout),
        termination_grace_time=timedelta(seconds=args.inprocess_termination_grace_time),
        enabled=True,
    )(train)

    return train


def maybe_wrap_for_inprocess_restart(pretrain):

    args = arguments.parse_args(ignore_unknown_args=True)

    if args.inprocess_restart:
        pretrain = inprocess_restart(pretrain, args)

        store = torch.distributed.TCPStore(
            host_name=os.environ['MASTER_ADDR'],
            port=int(os.environ['MASTER_PORT'])+1,
            world_size=int(os.getenv('WORLD_SIZE', '1')),
            is_master=(int(os.getenv('RANK', '0')) == 0),
            timeout=timedelta(seconds=300),
            wait_for_workers=True,
            use_libuv=True,
        )
    else:
        store = None

    return pretrain, store


def maybe_force_nccl_backend_init(device_id):

    args = get_args()

    # Inprocess uses destroy_process_group to terminate NCCL backend, which
    # does not terminate NCCL kernels if NCCL backend wasn't fully initialized
    # before additional distributed subgroups are created. This forces initialization
    # of the NCCL backend.
    if args.inprocess_restart:
        tensor = torch.ones(128, device=device_id)
        torch.distributed.all_reduce(tensor)
        torch.cuda.synchronize()
