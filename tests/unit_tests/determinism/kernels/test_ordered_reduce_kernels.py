# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay of the batch-invariant ordered TP reductions (tensor_parallel/ordered_reduce.py).

Covers the Triton fold, the symmetric-memory push (learner) and pull (inference) kernels, the
NCCL all-to-all fallback, the fixed-shard (``batch_invariant_tp_shards``) variants with one
and several partials per rank, and the forward reductions in ``tensor_parallel/mappings.py``
that route through them under batch-invariant mode. Multi-rank cases skip below their TP size.
"""

import pytest
import torch

from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
from megatron.core.tensor_parallel import mappings
from megatron.core.tensor_parallel import ordered_reduce as orr
from megatron.core.transformer.custom_layers.batch_invariant_kernels import set_batch_invariant_mode
from megatron.core.utils import get_tensor_model_parallel_group_if_none
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

TOKENS, HIDDEN, K = 4096, 2048, 4096


def _requires_tp(tp):
    return pytest.mark.skipif(
        Utils.world_size < tp or Utils.world_size % tp != 0, reason=f"needs a multiple of {tp} GPUs"
    )


def _mixed_magnitudes(shape, seed):
    """Values whose summation order changes the bits, distinct per TP rank."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(shape, device="cuda", generator=gen)
    return (x * torch.logspace(-4, 4, shape[-1], device="cuda")).bfloat16()


class TestOrderedReduceKernels:
    def setup_method(self, method):
        self._tp = None

    def teardown_method(self, method):
        torch.cuda.synchronize()
        SymmetricMemoryManager.destroy()
        orr._PEER_VIEWS.clear()
        if self._tp is not None:
            Utils.destroy_model_parallel()

    def _init(self, tp):
        Utils.initialize_model_parallel(tp, 1)
        self._tp = tp
        return get_tensor_model_parallel_group_if_none(tp_group=None)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("exact", [False, True], ids=["ordered", "multimem"])
    def test_ordered_fold_replays(self, dtype, exact):
        seeded()
        parts = _mixed_magnitudes((8, TOKENS, HIDDEN), 0).to(dtype)
        assert_replays_bit_exact(
            lambda p: orr.ordered_fold(p, exact=exact),
            (parts,),
            backward=False,
            contention=True,
            what="ordered_fold",
        )

    @pytest.mark.parametrize("num_shards", [2, 4])
    def test_local_shard_partials_and_fold_replay(self, num_shards):
        seeded()
        x = _mixed_magnitudes((TOKENS, K), 1)
        w = (torch.randn(HIDDEN, K, device="cuda") * 0.02).bfloat16()
        assert_replays_bit_exact(
            lambda x, w: orr.ordered_fold(orr.row_parallel_shard_partials(x, w, num_shards)),
            (x, w),
            backward=False,
            contention=True,
            what="row_parallel_shard_partials + ordered_fold",
        )

    @_requires_tp(4)
    @pytest.mark.parametrize("sequence_parallel", [False, True], ids=["all_reduce", "rs"])
    def test_mappings_ordered_forward_replays(self, sequence_parallel):
        """mappings' forward reductions take the ordered path at TP >= 3 (backward is stock)."""
        group = self._init(4)
        seeded()
        x = _mixed_magnitudes((TOKENS, 2, HIDDEN), 2 + group.rank()).requires_grad_(True)
        fn = (
            mappings.reduce_scatter_to_sequence_parallel_region
            if sequence_parallel
            else mappings.reduce_from_tensor_model_parallel_region
        )
        with set_batch_invariant_mode(True):
            assert_replays_bit_exact(
                lambda x: fn(x, group=group),
                (x,),
                contention=True,
                what=f"mappings.{fn.__name__} (ordered)",
            )

    @_requires_tp(4)
    @pytest.mark.parametrize("path", ["push", "all_to_all"])
    def test_ordered_reduce_scatter_replays(self, path, monkeypatch):
        group = self._init(4)
        if path == "all_to_all":
            monkeypatch.setattr(orr, "_push_staging", lambda *args, **kwargs: None)
        seeded()
        x = _mixed_magnitudes((TOKENS, HIDDEN), 3 + group.rank())
        assert_replays_bit_exact(
            lambda x: orr.ordered_reduce_scatter(x, group),
            (x,),
            backward=False,
            contention=True,
            what=f"ordered_reduce_scatter ({path})",
        )

    @_requires_tp(2)
    @pytest.mark.parametrize("tp,shards", [(2, 4), (4, 8)])
    @pytest.mark.parametrize("path", ["push", "all_to_all"])
    def test_shard_reduce_scatter_replays(self, tp, shards, path, monkeypatch):
        if Utils.world_size < tp:
            pytest.skip(f"needs {tp} GPUs")
        group = self._init(tp)
        if path == "all_to_all":
            monkeypatch.setattr(orr, "_push_staging", lambda *args, **kwargs: None)
        seeded()
        kl = K * shards // 4 // tp
        x = _mixed_magnitudes((TOKENS, kl), 4 + group.rank())
        w = (torch.randn(HIDDEN, kl, device="cuda") * 0.02).bfloat16()
        c = shards // tp
        assert_replays_bit_exact(
            lambda x, w: orr.ordered_reduce_scatter_shards(
                orr.row_parallel_shard_partials(x, w, c), group
            ),
            (x, w),
            backward=False,
            contention=True,
            what=f"ordered_reduce_scatter_shards c={c} ({path})",
        )

    @_requires_tp(2)
    @pytest.mark.parametrize("tp,shards", [(2, 4), (4, 4), (4, 8)])
    def test_symm_pull_replays(self, tp, shards):
        """Inference: the GEMM writes into the symmetric buffer, one pull kernel folds."""
        if Utils.world_size < tp:
            pytest.skip(f"needs {tp} GPUs")
        if torch.cuda.get_device_capability()[0] < 9:
            pytest.skip("symmetric memory needs sm90+")
        group = self._init(tp)
        c = shards // tp
        kl = K * shards // 4 // tp
        seeded()
        x = _mixed_magnitudes((TOKENS // 4, kl), 5 + group.rank())
        w = (torch.randn(HIDDEN, kl, device="cuda") * 0.02).bfloat16()
        buf = SymmetricMemoryManager.get_buffer("tp", process_group=group)
        symm = buf.maybe_get_tensor([c, x.shape[0], HIDDEN], dtype=torch.bfloat16)
        if symm["handle"] is None:
            pytest.skip("no symmetric memory")
        out = torch.empty((x.shape[0] // tp, HIDDEN), dtype=torch.bfloat16, device="cuda")

        def pull(x, w):
            parts = orr.row_parallel_shard_partials(x, w, c, out=symm["tensor"])
            if c == 1:
                return orr.symm_ordered_reduce_scatter(out, parts[0], symm["handle"])
            return orr.symm_ordered_reduce_scatter_shards(out, parts, symm["handle"])

        assert_replays_bit_exact(
            pull, (x, w), backward=False, contention=True, what=f"symm pull c={c}"
        )

    @_requires_tp(2)
    @pytest.mark.parametrize("sequence_parallel", [False, True], ids=["all_reduce", "rs"])
    def test_tp_shard_row_linear_replays_fwd_bwd(self, sequence_parallel):
        """Several shards per rank: shard forward plus the stock row-parallel backward."""
        group = self._init(2)
        seeded()
        x = _mixed_magnitudes((TOKENS, 2, K // 2), 6 + group.rank()).requires_grad_(True)
        w = (torch.randn(HIDDEN, K // 2, device="cuda") * 0.02).bfloat16().requires_grad_(True)
        assert_replays_bit_exact(
            lambda x, w: orr.tp_shard_row_linear(x, w, group, 2, sequence_parallel),
            (x, w),
            contention=True,
            what="tp_shard_row_linear",
        )
