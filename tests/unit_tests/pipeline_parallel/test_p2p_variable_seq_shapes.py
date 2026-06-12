# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""Regression tests for the variable-sequence-length P2P shape exchange.

These tests target the silent shape-swap bug (MCORE-382 / PR#1451 lineage) in
``P2PCommunicator._communicate_shapes``. When the pipeline-parallel group has
size 2, ``next_rank == prev_rank`` (a single physical peer). The shape exchange
always enqueues ops in the fixed order ``[send_prev, recv_prev, send_next,
recv_next]`` into one tagless ``batch_isend_irecv`` collective and never
consults ``config.batch_p2p_comm``. Because NCCL P2P is tagless, same-peer ops
pair FIFO by enqueue order, so the ``recv_prev`` and ``recv_next`` shape
tensors get crossed when both a send-next and a send-prev target the same peer.
The data transfer (when it goes through the batched ``batch_isend_irecv`` path)
uses the identical fixed order, so the byte sizes still match and the swap is
silent at the comm layer -- the returned tensors simply carry the wrong
(swapped) shapes.

Test design notes (why the structure is what it is):

  * ``test_variable_seq_shape_swap_pp2`` exercises the full end-to-end
    ``send_forward_backward_recv_forward_backward`` path with
    ``batch_p2p_comm=True``. Both the shape exchange AND the data transfer use
    ``batch_isend_irecv``, so the FIFO byte sizes match the (swapped) recv
    buffers and the call completes silently -- producing a clean shape
    assertion failure on buggy main rather than an NCCL count-mismatch.

  * ``test_communicate_shapes_swap_pp2`` is parametrized over
    ``batch_p2p_comm`` and calls ``_communicate_shapes`` DIRECTLY. This proves
    Claim C: the shape exchange ignores ``batch_p2p_comm`` entirely (it only
    branches on ``use_ring_exchange_p2p``), so the swap occurs for BOTH flag
    values. Calling the shape helper directly is deliberate: the non-batched
    (``batch_p2p_comm=False``) DATA path uses point-to-point ``isend``/``irecv``
    matched by explicit src/dst, and after a swapped shape exchange the recv
    buffers would be sized wrong, risking an NCCL count mismatch / hang rather
    than a clean assertion. The shape exchange itself is symmetric and balanced
    (2 sends + 2 recvs per rank, all to the single peer), so it cannot hang.
"""

import pytest
import torch
from packaging import version

from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from tests.unit_tests.test_utilities import Utils

# Distinct sequence lengths so a swap is observable. The activation (sent to
# the next stage) and the gradient (sent to the previous stage) must differ.
HIDDEN_SIZE = 16
ACT_SEQ_LEN = 10  # output_tensor / activation -> sent to next stage
GRAD_SEQ_LEN = 20  # input_tensor_grad / gradient -> sent to prev stage


@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse('2.3.0'),
    reason="HyperCommGrid device mesh feature requires PyTorch 2.3 or later",
)
@pytest.mark.internal
def test_variable_seq_shape_swap_pp2():
    """Bidirectional shape exchange at PP-group size 2 crosses the shapes (end-to-end).

    Reproduces Claim A through the full
    ``send_forward_backward_recv_forward_backward`` entry point with
    ``batch_p2p_comm=True`` (the default). Both the shape exchange and the data
    transfer go through ``batch_isend_irecv``, so the swapped recv-buffer byte
    sizes still match the FIFO pairing and the call completes silently --
    yielding a clean shape assertion failure on buggy main (not an NCCL error).

    2-rank ring reasoning (PP group indices {0, 1}):
      * For rank ``r``: ``next = (r+1) % 2``, ``prev = (r-1) % 2``; with size 2
        this gives ``next == prev`` (one physical peer).
      * Every rank sends its activation (seq=ACT_SEQ_LEN) to ``next`` and its
        gradient (seq=GRAD_SEQ_LEN) to ``prev``.
      * The activation a rank RECEIVES comes from its ``prev`` stage; the peer
        that has this rank as its ``next`` sends its activation here. So the
        received ``input_tensor`` (activation from prev) must carry the
        activation seq len -> shape[0] == ACT_SEQ_LEN (10).
      * The gradient a rank RECEIVES comes from its ``next`` stage; that peer
        sends its gradient to its ``prev`` (this rank). So the received
        ``output_tensor_grad`` (grad from next) must carry the gradient seq
        len -> shape[0] == GRAD_SEQ_LEN (20).

    On current main the fixed-order same-peer exchange swaps the two shape
    tensors, so ``input_tensor.shape[0] == 20`` and
    ``output_tensor_grad.shape[0] == 10`` -- the assertions below fail.
    """
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=2)

    # 8 GPUs: PP=2 (the size-2 group that triggers the same-peer collision),
    # DP=4. The pp group therefore has exactly two ranks.
    grid = HyperCommGrid([1, 1, 2, 4], ["tp", "cp", "pp", "dp"])
    pp_group = grid.create_pg("pp")
    assert pp_group.size() == 2, "This test requires a size-2 pipeline-parallel group."

    config = ModelParallelConfig(
        pipeline_model_parallel_size=2, pipeline_dtype=torch.float, variable_seq_lengths=True
    )
    config.hidden_size = HIDDEN_SIZE
    # batch_p2p_comm=True (the default): the data transfer uses batch_isend_irecv,
    # so swapped recv buffers still byte-match the FIFO pairing and the call
    # completes silently instead of raising an NCCL count mismatch.
    config.batch_p2p_comm = True

    comm = P2PCommunicator(pp_group=pp_group, config=config)

    # Deterministic, identical construction on every rank: the activation has a
    # distinct seq length from the gradient so a swap is observable.
    output_tensor = torch.ones(
        (ACT_SEQ_LEN, 1, HIDDEN_SIZE), device=torch.cuda.current_device(), dtype=torch.float
    )
    input_tensor_grad = torch.ones(
        (GRAD_SEQ_LEN, 1, HIDDEN_SIZE), device=torch.cuda.current_device(), dtype=torch.float
    )

    # tensor_shape is ignored when variable_seq_lengths=True, but must be valid.
    input_tensor, output_tensor_grad = comm.send_forward_backward_recv_forward_backward(
        output_tensor=output_tensor,
        input_tensor_grad=input_tensor_grad,
        recv_prev=True,
        recv_next=True,
        tensor_shape=(ACT_SEQ_LEN, 1, HIDDEN_SIZE),
    )

    assert input_tensor is not None and output_tensor_grad is not None

    assert input_tensor.shape[0] == ACT_SEQ_LEN, (
        f"SHAPE SWAP DETECTED: input_tensor is the activation received from the "
        f"previous stage and must have seq len {ACT_SEQ_LEN}, but got "
        f"{input_tensor.shape[0]}. A value of {GRAD_SEQ_LEN} means the "
        f"recv_prev/recv_next shape tensors were crossed in _communicate_shapes "
        f"(same-peer FIFO pairing at PP-group size 2)."
    )
    assert output_tensor_grad.shape[0] == GRAD_SEQ_LEN, (
        f"SHAPE SWAP DETECTED: output_tensor_grad is the gradient received from the "
        f"next stage and must have seq len {GRAD_SEQ_LEN}, but got "
        f"{output_tensor_grad.shape[0]}. A value of {ACT_SEQ_LEN} means the "
        f"recv_prev/recv_next shape tensors were crossed in _communicate_shapes "
        f"(same-peer FIFO pairing at PP-group size 2)."
    )

    Utils.destroy_model_parallel()


@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse('2.3.0'),
    reason="HyperCommGrid device mesh feature requires PyTorch 2.3 or later",
)
@pytest.mark.internal
@pytest.mark.parametrize("batch_p2p_comm", [True, False])
def test_communicate_shapes_swap_pp2(batch_p2p_comm):
    """``_communicate_shapes`` crosses the shapes at PP size 2 for BOTH flag values.

    Reproduces Claim A and Claim C directly at the shape-exchange layer. We call
    ``_communicate_shapes`` itself (not the full ``_communicate`` data path) for
    two reasons:

      1. Claim C: ``_communicate_shapes`` only branches on
         ``use_ring_exchange_p2p`` and never reads ``config.batch_p2p_comm``, so
         the documented "set batch_p2p_comm=False" workaround cannot reach it.
         Parametrizing the flag and asserting the swap happens for BOTH values
         proves the flag is irrelevant to the shape exchange.

      2. Safety on the cluster: the shape exchange is symmetric and balanced
         (each rank posts 2 sends + 2 recvs, all to the single peer), so it
         completes regardless of the flag and cannot hang. The non-batched DATA
         path, by contrast, would mis-size its recv buffers after a swapped
         shape exchange and risk an NCCL count mismatch -- which is why the
         end-to-end test above stays on the batched path.

    Each rank passes ``tensor_send_next`` = activation (seq=ACT_SEQ_LEN) and
    ``tensor_send_prev`` = gradient (seq=GRAD_SEQ_LEN), with recv_prev=recv_next=True.

    Correct semantics (independently of the bug):
      * ``recv_prev_shape`` is the shape received from ``prev``; the prev peer
        has this rank as its ``next`` and sends its activation -> seq=ACT_SEQ_LEN.
      * ``recv_next_shape`` is the shape received from ``next``; the next peer
        sends its gradient backward to its ``prev`` (this rank) -> seq=GRAD_SEQ_LEN.

    On buggy main both ranks enqueue [send_prev(grad), recv_prev, send_next(act),
    recv_next] to the single peer; FIFO pairing crosses them, giving
    recv_prev_shape[0] == GRAD_SEQ_LEN and recv_next_shape[0] == ACT_SEQ_LEN.
    """
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=2)

    grid = HyperCommGrid([1, 1, 2, 4], ["tp", "cp", "pp", "dp"])
    pp_group = grid.create_pg("pp")
    assert pp_group.size() == 2, "This test requires a size-2 pipeline-parallel group."

    config = ModelParallelConfig(
        pipeline_model_parallel_size=2, pipeline_dtype=torch.float, variable_seq_lengths=True
    )
    config.hidden_size = HIDDEN_SIZE
    # Parametrized: this flag MUST NOT change the outcome, because
    # _communicate_shapes never reads it (Claim C). The swap happens regardless.
    config.batch_p2p_comm = batch_p2p_comm

    comm = P2PCommunicator(pp_group=pp_group, config=config)

    output_tensor = torch.ones(
        (ACT_SEQ_LEN, 1, HIDDEN_SIZE), device=torch.cuda.current_device(), dtype=torch.float
    )
    input_tensor_grad = torch.ones(
        (GRAD_SEQ_LEN, 1, HIDDEN_SIZE), device=torch.cuda.current_device(), dtype=torch.float
    )

    recv_prev_shape, recv_next_shape = comm._communicate_shapes(
        tensor_send_next=output_tensor,
        tensor_send_prev=input_tensor_grad,
        recv_prev=True,
        recv_next=True,
    )

    assert recv_prev_shape[0] == ACT_SEQ_LEN, (
        f"SHAPE SWAP DETECTED (batch_p2p_comm={batch_p2p_comm}): recv_prev_shape is the "
        f"activation shape received from the previous stage and must have seq len "
        f"{ACT_SEQ_LEN}, but got {recv_prev_shape[0]}. A value of {GRAD_SEQ_LEN} means "
        f"the recv_prev/recv_next shape tensors were crossed by the same-peer FIFO "
        f"pairing at PP-group size 2 -- and the batch_p2p_comm flag did not prevent it "
        f"(Claim C: _communicate_shapes ignores that flag)."
    )
    assert recv_next_shape[0] == GRAD_SEQ_LEN, (
        f"SHAPE SWAP DETECTED (batch_p2p_comm={batch_p2p_comm}): recv_next_shape is the "
        f"gradient shape received from the next stage and must have seq len "
        f"{GRAD_SEQ_LEN}, but got {recv_next_shape[0]}. A value of {ACT_SEQ_LEN} means "
        f"the recv_prev/recv_next shape tensors were crossed by the same-peer FIFO "
        f"pairing at PP-group size 2 -- and the batch_p2p_comm flag did not prevent it "
        f"(Claim C: _communicate_shapes ignores that flag)."
    )

    Utils.destroy_model_parallel()


@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse('2.3.0'),
    reason="HyperCommGrid device mesh feature requires PyTorch 2.3 or later",
)
@pytest.mark.internal
def test_variable_seq_shape_ring_pp4_control():
    """Positive control: ring shape exchange at PP-group size 4 is correct (no swap).

    At PP-group size >= 4, ``prev_rank != next_rank``. ``send_forward_recv_forward``
    posts ``send_next`` and ``recv_prev`` to DIFFERENT peers, so there is no
    same-peer FIFO collision and no shape swap. This must PASS on current main,
    proving the harness is sound and the bug is specific to the size-2
    bidirectional same-peer case.

    IMPORTANT (why a ring and not send_forward_recv_backward): every rank issues
    the SAME call simultaneously. ``send_forward_recv_forward`` = ``send_next`` +
    ``recv_prev``, which forms a balanced ring rotation (rank r sends to r+1 and
    receives from r-1), so every send has a matching recv and the collective
    cannot hang. By contrast a ring of ``send_forward_recv_backward``
    (``send_next`` + ``recv_next``) is NOT balanced -- rank r's send to r+1 has
    no matching recv on r+1 -- and would deadlock.

    Reasoning (ring of 4, every rank sends the same fixed activation seq len):
      * ``input_tensor`` is received from ``prev``; the prev stage sends its
        activation (seq=ACT_SEQ_LEN) forward to its ``next`` (this rank), so
        ``input_tensor.shape[0] == ACT_SEQ_LEN``.
    """
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=4)

    # 8 GPUs: PP=4 (prev != next, no same-peer collision), DP=2.
    grid = HyperCommGrid([1, 1, 4, 2], ["tp", "cp", "pp", "dp"])
    pp_group = grid.create_pg("pp")
    assert pp_group.size() == 4, "This control requires a size-4 pipeline-parallel group."

    config = ModelParallelConfig(
        pipeline_model_parallel_size=4, pipeline_dtype=torch.float, variable_seq_lengths=True
    )
    config.hidden_size = HIDDEN_SIZE

    comm = P2PCommunicator(pp_group=pp_group, config=config)

    output_tensor = torch.ones(
        (ACT_SEQ_LEN, 1, HIDDEN_SIZE), device=torch.cuda.current_device(), dtype=torch.float
    )

    # Balanced ring: send activation to next, recv activation from prev.
    # (send_next + recv_prev) -- exercises _communicate_shapes at size 4 with the
    # send and recv targeting DIFFERENT peers, so there is no swap and no hang.
    input_tensor = comm.send_forward_recv_forward(
        output_tensor=output_tensor, recv_prev=True, tensor_shape=(ACT_SEQ_LEN, 1, HIDDEN_SIZE)
    )

    assert input_tensor is not None
    assert input_tensor.shape[0] == ACT_SEQ_LEN, (
        f"input_tensor (activation from prev stage) must have seq len {ACT_SEQ_LEN}, "
        f"got {input_tensor.shape[0]}. At PP size 4 there is no same-peer collision, "
        f"so the shape exchange must be correct."
    )

    Utils.destroy_model_parallel()
