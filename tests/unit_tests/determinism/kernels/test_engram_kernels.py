# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay of the Engram kernels.

* ``fused_dense.py``: the Triton gate / conv-norm / causal depthwise convolution / SiLU /
  residual chain. Its parameter gradients are per-program partials reduced with ``torch.sum``.
* ``distributed_embedding.py``: the hashed n-gram table lookup, whose backward accumulates
  every repeated row. Both table branches are replayed: the sorted, segmented accumulation
  that ``deterministic_mode`` selects (forward and backward) and ``F.embedding`` (forward
  only, its backward is not reproducible on CUDA). The lookup runs with one EP
  rank, where it is local; the request routing of larger EP groups only counts requests in
  integers (``scatter_add_`` / ``bincount``) and is not replayed here.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.engram import fused_dense
from megatron.core.models.engram.distributed_embedding import EPShardedMultiTableEmbedding
from tests.unit_tests.determinism.kernels.harness import (
    CONTENTION_TOKENS,
    assert_module_replays_bit_exact,
    assert_replays_bit_exact,
    seeded,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
needs_triton = pytest.mark.skipif(not fused_dense.HAVE_TRITON, reason="needs Triton")

STREAMS, HIDDEN, KERNEL_SIZE, DILATION, EPS = 4, 1024, 4, 3, 1e-6


def _fused_dense_inputs():
    seeded()
    width = STREAMS * HIDDEN

    def rand(*shape, scale=1.0):
        return (torch.randn(*shape, device="cuda") * scale).bfloat16().requires_grad_()

    inputs = dict(
        key=rand(CONTENTION_TOKENS, 1, width),
        hidden=rand(CONTENTION_TOKENS, 1, width),
        value=rand(CONTENTION_TOKENS, 1, HIDDEN),
        key_gamma=rand(width, scale=0.2),
        query_gamma=rand(width, scale=0.2),
        conv_gamma=rand(width, scale=0.2),
        conv_weight=rand(width, 1, KERNEL_SIZE, scale=0.3),
    )
    return inputs, {"out": torch.randn_like(inputs["key"])}


@needs_triton
@pytest.mark.parametrize("zero_centered", [True, False], ids=["zero_centered", "plain_gamma"])
def test_fused_engram_dense_replays_fwd_bwd(zero_centered):
    """Output and the key / hidden / value / gamma / convolution-weight gradients.

    4096 tokens are 128 token tiles, each writing one partial per parameter gradient.
    """
    inputs, grad_outputs = _fused_dense_inputs()

    def fn(**tensors):
        return fused_dense.fused_engram_dense(
            **tensors, num_streams=STREAMS, eps=EPS, zero_centered=zero_centered, dilation=DILATION
        )

    _, grads = assert_replays_bit_exact(
        fn, inputs, replays=4, grad_outputs=grad_outputs, contention=True, what="fused_engram_dense"
    )
    assert len(grads) == len(inputs)


@needs_triton
def test_fused_engram_dense_packed_documents_replays_fwd_bwd():
    """Packed rows: the convolution restarts at every document, including a one-token one."""
    inputs, grad_outputs = _fused_dense_inputs()
    starts = torch.tensor([0, 700, 701, 2500], device="cuda")
    position = torch.arange(CONTENTION_TOKENS, device="cuda")
    segment_offsets = position - starts[torch.searchsorted(starts, position, right=True) - 1]

    def fn(**tensors):
        return fused_dense.fused_engram_dense(
            **tensors,
            num_streams=STREAMS,
            eps=EPS,
            zero_centered=True,
            dilation=DILATION,
            segment_offsets=segment_offsets,
        )

    _, grads = assert_replays_bit_exact(
        fn,
        inputs,
        replays=4,
        grad_outputs=grad_outputs,
        contention=True,
        what="fused_engram_dense (packed documents)",
    )
    assert len(grads) == len(inputs)


@pytest.mark.parametrize(
    "deterministic_mode", [True, False], ids=["sorted_segments", "F.embedding"]
)
def test_engram_table_lookup_replays_fwd_bwd(deterministic_mode):
    """Three tables of ~64 rows looked up by 32k tokens: ~500 gradient rows per table row.

    The sorted, segmented branch must replay its outputs and table gradients bit for bit. The
    ``F.embedding`` branch accumulates the repeated rows in an order that CUDA does not fix
    (which is why ``deterministic_mode`` has a branch of its own), so only its forward is
    replayed.
    """
    seeded()
    table_sizes, embedding_dim, tokens = (61, 67, 71), 256, 8 * CONTENTION_TOKENS
    config = SimpleNamespace(
        use_cpu_initialization=True,
        params_dtype=torch.bfloat16,
        perform_initialization=True,
        sequence_parallel=False,
        deterministic_mode=deterministic_mode,
    )
    embedding = EPShardedMultiTableEmbedding(
        config=config,
        table_sizes=table_sizes,
        embedding_dim=embedding_dim,
        init_method=lambda tensor: torch.nn.init.normal_(tensor, mean=0.0, std=0.02),
    ).cuda()
    assert embedding.ep_size == 1
    hash_ids = torch.stack(
        [torch.randint(0, size, (tokens,), device="cuda") for size in table_sizes], dim=-1
    )
    grad_output = torch.randn(
        tokens, len(table_sizes), embedding_dim, device="cuda", dtype=torch.bfloat16
    )

    outputs, grads = assert_module_replays_bit_exact(
        embedding,
        (hash_ids,),
        replays=6,
        backward=deterministic_mode,
        grad_output=grad_output,
        contention=True,
        what=f"Engram table lookup (deterministic_mode={deterministic_mode})",
    )
    assert outputs["out"].shape == grad_output.shape
    assert len(grads) == (len(table_sizes) if deterministic_mode else 0)
