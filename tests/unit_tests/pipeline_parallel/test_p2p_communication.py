# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import torch

import megatron.core.pipeline_parallel.p2p_communication as p2p_communication


def test_communicate_shapes_waits_without_device_synchronize(mocker):
    communicator = p2p_communication.P2PCommunicator.__new__(p2p_communication.P2PCommunicator)
    communicator.config = SimpleNamespace(use_ring_exchange_p2p=False)
    communicator.pp_group = object()
    communicator.prev_rank = 1
    communicator.next_rank = 3

    mocker.patch.object(
        p2p_communication.torch.cuda, "current_device", return_value=torch.device("cpu")
    )
    synchronize = mocker.patch.object(p2p_communication.torch.cuda, "synchronize")
    mocker.patch.object(
        p2p_communication.torch.distributed,
        "P2POp",
        side_effect=lambda op, tensor, peer, group: SimpleNamespace(
            op=op, tensor=tensor, peer=peer, group=group
        ),
    )

    requests = [mocker.Mock() for _ in range(4)]

    def batch_isend_irecv(ops):
        for op in ops:
            if op.op is torch.distributed.irecv:
                shape = [7, 2, 16] if op.peer == communicator.prev_rank else [11, 2, 16]
                op.tensor.copy_(torch.tensor(shape))
        return requests

    batch_p2p = mocker.patch.object(
        p2p_communication.torch.distributed,
        "batch_isend_irecv",
        side_effect=batch_isend_irecv,
    )

    recv_prev_shape, recv_next_shape = communicator._communicate_shapes(
        tensor_send_next=torch.empty((13, 2, 16)),
        tensor_send_prev=torch.empty((5, 2, 16)),
        recv_prev=True,
        recv_next=True,
    )

    assert recv_prev_shape == [7, 2, 16]
    assert recv_next_shape == [11, 2, 16]
    batch_p2p.assert_called_once()
    for request in requests:
        request.wait.assert_called_once_with()
    synchronize.assert_not_called()
