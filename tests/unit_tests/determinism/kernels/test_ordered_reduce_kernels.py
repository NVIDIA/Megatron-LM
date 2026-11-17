# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay of the batch-invariant ordered TP reductions (ops/collectives/ordered_reduce.py).

Covers the Triton fold, the direct-buffer symmetric-memory pull kernels, the
NCCL all-to-all fallback, the fixed-shard (``batch_invariant_tp_shards``) variants with one
and several partials per rank, and the forward reductions in ``tensor_parallel/mappings.py``
that route through them under batch-invariant mode. Multi-rank cases skip below their TP size.
"""

import pytest
import torch

from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
from megatron.core.ops.collectives import ordered_reduce as orr
from megatron.core.ops.linear import ordered as orl
from megatron.core.ops.loss.vocab_sum import batch_invariant_vocab_sum
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import mappings
from megatron.core.tensor_parallel.gtp_api import HAVE_GTP
from megatron.core.transformer.custom_layers.batch_invariant_kernels import set_batch_invariant_mode
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_tensor_model_parallel_group_if_none
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded
from tests.unit_tests.test_utilities import Utils

if HAVE_GTP:
    from megatron.core.extensions.transformer_engine import TERowParallelLinear
    from megatron.core.tensor_parallel import generalized_tensor_parallelism as gtp

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
        # Peers may still read this rank's buffer until the group syncs.
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        SymmetricMemoryManager.destroy()
        if self._tp is not None:
            Utils.destroy_model_parallel()

    def _init(self, tp):
        Utils.initialize_model_parallel(tp, 1)
        self._tp = tp
        return get_tensor_model_parallel_group_if_none(tp_group=None)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_ordered_fold_replays(self, dtype):
        seeded()
        parts = _mixed_magnitudes((8, TOKENS, HIDDEN), 0).to(dtype)
        assert_replays_bit_exact(
            lambda p: orr.ordered_fold(p),
            (parts,),
            backward=False,
            contention=True,
            what="ordered_fold",
        )

    @_requires_tp(4)
    @pytest.mark.parametrize(
        "local_shards,sequence_parallel,fused_wgrad",
        [
            (2, False, False),
            (2, True, False),
            (2, False, True),
            (2, True, True),
            (1, False, False),
            (1, True, True),
        ],
    )
    def test_gtp_fixed_shard_linear_replays(
        self, local_shards, sequence_parallel, fused_wgrad, monkeypatch
    ):
        """Real GTP weights: fixed-shard output, rematerialized dgrad and accumulated wgrad."""
        if not HAVE_GTP:
            pytest.skip("GTP requires TransformerEngine >= 2.19")

        Utils.initialize_model_parallel(2, 1, gtp_remat_size=2)
        self._tp = 2
        groups = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "gtp_remat"])
        monkeypatch.setattr(gtp.GTP_CONFIG, "calculate_per_token_loss", True)
        cfg = TransformerConfig(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=8,
            tensor_model_parallel_size=2,
            gtp_weight_remat_size=2,
            sequence_parallel=sequence_parallel,
            params_dtype=torch.bfloat16,
            batch_invariant_mode=True,
            batch_invariant_tp_shards=4,
            gradient_accumulation_fusion=fused_wgrad,
            attention_backend=AttnBackend.flash,
            flash_attention_version=4,
            attention_dropout=0.0,
        )
        try:
            layer = TERowParallelLinear(
                512,
                256,
                config=cfg,
                init_method=lambda w: None,
                bias=False,
                input_is_parallel=True,
                skip_bias_add=True,
                is_expert=False,
                pg_collection=groups,
            ).cuda()
            seeded()
            full_weight = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16) * 0.02
            tp_rank, gtp_rank = groups.tp.rank(), groups.gtp_remat.rank()
            weight = full_weight[:, tp_rank * 256 : (tp_rank + 1) * 256].contiguous()
            with torch.no_grad():
                layer.weight.copy_(weight[gtp_rank * 128 : (gtp_rank + 1) * 128])
            layer.weight.main_grad = torch.zeros_like(layer.weight, dtype=torch.float32)
            gen = torch.Generator(device="cuda").manual_seed(51 + gtp_rank)
            full_x = torch.randn(32, 1, 512, device="cuda", dtype=torch.bfloat16, generator=gen)
            x = full_x[..., tp_rank * 256 : (tp_rank + 1) * 256].contiguous()
            dy = torch.randn(32, 1, 256, device="cuda", dtype=torch.bfloat16, generator=gen)
            local_dy = dy.chunk(2, dim=0)[tp_rank].contiguous() if sequence_parallel else dy
            reference = orr.ordered_fold(
                orl.row_parallel_shard_partials(full_x, full_weight, 2 * local_shards)
            )
            if sequence_parallel:
                reference = reference.chunk(2, dim=0)[tp_rank].contiguous()
            if local_shards == 1:
                dx = orl.te_cpp_extensions.general_gemm(
                    weight, dy.reshape(-1, 256), layout="NN", out_dtype=x.dtype, grad=True
                )[0].view_as(x)
            else:
                dx = dy.reshape(-1, 256).matmul(weight).view_as(x)
            if fused_wgrad:
                dw = orl.te_cpp_extensions.general_gemm(
                    x.reshape(-1, 256),
                    dy.reshape(-1, 256),
                    layout="NT",
                    out_dtype=torch.float32,
                    grad=True,
                )[0]
            else:
                dw = dy.reshape(-1, 256).t().matmul(x.reshape(-1, 256))
            torch.distributed.all_reduce(dw, group=groups.gtp_remat)
            dw = dw[gtp_rank * 128 : (gtp_rank + 1) * 128].float()

            def run(value):
                layer.weight.main_grad.zero_()
                outputs, inputs = [], []
                for _ in range(2):
                    leaf = value.detach().requires_grad_()
                    if local_shards > 1:
                        out, _ = layer(leaf)
                    else:
                        out = orl.tp_shard_row_linear(
                            leaf,
                            layer.weight,
                            groups.tp,
                            1,
                            sequence_parallel,
                            wgrad_into_main_grad=fused_wgrad,
                            use_te_gemm=True,
                        )
                    out.backward(local_dy)
                    outputs.append(out.detach())
                    inputs.append(leaf.grad)
                return outputs, inputs, layer.weight.main_grad.clone()

            with set_batch_invariant_mode(True, backend="te_native"):
                outputs, inputs, main_grad = run(x)
                for out, grad in zip(outputs, inputs):
                    assert torch.equal(out, reference)
                    torch.testing.assert_close(grad, dx, rtol=0, atol=0)
                torch.testing.assert_close(main_grad, dw * 2, rtol=0, atol=0)
                assert_replays_bit_exact(
                    run,
                    (x,),
                    backward=False,
                    contention=True,
                    what="GTP fixed-shard row linear forward/dgrad/wgrad",
                )
        finally:
            torch.cuda.synchronize()
            torch.distributed.barrier()
            gtp.reset_gtp_state()
            gtp._GTP_PARAMS.clear()

    @pytest.mark.parametrize("num_shards", [2, 4])
    def test_local_shard_partials_and_fold_replay(self, num_shards):
        seeded()
        x = _mixed_magnitudes((TOKENS, K), 1)
        w = (torch.randn(HIDDEN, K, device="cuda") * 0.02).bfloat16()
        assert_replays_bit_exact(
            lambda x, w: orr.ordered_fold(orl.row_parallel_shard_partials(x, w, num_shards)),
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
    def test_vocab_sum_replays(self):
        group = self._init(4)
        seeded()
        x = torch.rand((8, 4096), device="cuda")
        assert_replays_bit_exact(
            lambda x: batch_invariant_vocab_sum(x, group, 2),
            (x,),
            backward=False,
            contention=True,
            what="batch_invariant_vocab_sum",
        )
        # A row's denominator must not depend on the other rows in the batch.
        full = batch_invariant_vocab_sum(x, group, 2)
        single = batch_invariant_vocab_sum(x[:1], group, 2)
        assert torch.equal(full[:1], single)

    @_requires_tp(4)
    def test_ordered_reduce_scatter_replays(self):
        group = self._init(4)
        seeded()
        x = _mixed_magnitudes((TOKENS, HIDDEN), 3 + group.rank())
        assert_replays_bit_exact(
            lambda x: orr.ordered_reduce_scatter(x, group),
            (x,),
            backward=False,
            contention=True,
            what="ordered_reduce_scatter (all-to-all)",
        )

    @_requires_tp(2)
    @pytest.mark.parametrize("tp,shards", [(2, 4), (4, 8)])
    def test_shard_reduce_scatter_replays(self, tp, shards):
        if Utils.world_size < tp:
            pytest.skip(f"needs {tp} GPUs")
        group = self._init(tp)
        seeded()
        kl = K * shards // 4 // tp
        x = _mixed_magnitudes((TOKENS, kl), 4 + group.rank())
        w = (torch.randn(HIDDEN, kl, device="cuda") * 0.02).bfloat16()
        c = shards // tp
        assert_replays_bit_exact(
            lambda x, w: orr.ordered_reduce_scatter_shards(
                orl.row_parallel_shard_partials(x, w, c), group
            ),
            (x, w),
            backward=False,
            contention=True,
            what=f"ordered_reduce_scatter_shards c={c} (all-to-all)",
        )

    @_requires_tp(2)
    @pytest.mark.parametrize("tp,shards", [(2, 4), (4, 4), (4, 8)])
    @pytest.mark.parametrize("sync_after", [False, True])
    def test_symm_pull_replays(self, tp, shards, sync_after):
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

        gathered = torch.empty((x.shape[0], HIDDEN), dtype=torch.bfloat16, device="cuda")

        def pull(x, w):
            # The engine's sequence: GEMM into the symmetric buffer, pull reduce-scatter, then
            # an all-gather, which orders the next replay's GEMM after every peer's reads.
            parts = orl.row_parallel_shard_partials(x, w, c, out=symm["tensor"])
            if sync_after:
                # Training reuses scratch without an intervening all-gather.
                return orr.symm_ordered_reduce_scatter_shards(
                    parts, symm["handle"], out, sync_after=True
                )
            orr.symm_ordered_reduce_scatter_shards(parts, symm["handle"], out)
            torch.distributed.all_gather_into_tensor(gathered, out, group=group)
            return gathered

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
            lambda x, w: orl.tp_shard_row_linear(x, w, group, 2, sequence_parallel),
            (x, w),
            contention=True,
            what="tp_shard_row_linear",
        )
