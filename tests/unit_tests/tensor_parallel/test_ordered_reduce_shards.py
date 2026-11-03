# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""TP-size-invariant row-parallel forward (``batch_invariant_tp_shards``, ordered_reduce.py)."""

import pytest
import torch
import torch.distributed as dist

from megatron.core.extensions.transformer_engine import TERowParallelLinear
from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy
from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
from megatron.core.tensor_parallel import ordered_reduce as orr
from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
from megatron.core.transformer.custom_layers.batch_invariant_kernels import set_batch_invariant_mode
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_tensor_model_parallel_group_if_none
from tests.unit_tests.test_utilities import Utils

requires_4 = pytest.mark.skipif(Utils.world_size < 4, reason="needs 4 GPUs")


def _bits(t):
    return t.contiguous().view(torch.uint8)


def _canonical(x_full, w_full, num_shards, exact=False):
    """The defined value: shard GEMM outputs folded in shard order, one rounding."""
    parts = orr.row_parallel_shard_partials(x_full, w_full, num_shards)
    return orr.ordered_fold(parts.contiguous(), exact=exact)


def _inputs(m, k, n, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, device="cuda", generator=gen).bfloat16()
    # Mixed magnitudes along K so the shard grouping changes the bits.
    x = (x.float() * torch.logspace(-3, 3, k, device="cuda")).bfloat16()
    w = (torch.randn(n, k, device="cuda", generator=gen) * 0.02).bfloat16()
    return x, w


@requires_4
class TestShardReductions:
    def teardown_method(self, method):

        torch.cuda.synchronize()
        SymmetricMemoryManager.destroy()
        Utils.destroy_model_parallel()

    def test_shard_partials_match_per_shard_gemm(self):
        Utils.initialize_model_parallel(1, 1)
        x, w = _inputs(37, 1024, 384, 0)
        parts = orr.row_parallel_shard_partials(x, w, 4)
        for j in range(4):
            sl = slice(j * 256, (j + 1) * 256)
            assert torch.equal(parts[j], torch.matmul(x[:, sl].contiguous(), w[:, sl].t()))

    @pytest.mark.parametrize("tp,shards", [(2, 4), (2, 8), (4, 8), (4, 4), (2, 2)])
    @pytest.mark.parametrize("path", ["push", "all_to_all"])
    def test_reduce_scatter_shards_is_the_canonical_value(self, tp, shards, path, monkeypatch):
        Utils.initialize_model_parallel(tp, 1)
        group = get_tensor_model_parallel_group_if_none(tp_group=None)
        if path == "all_to_all":  # no staging -> grouped NCCL send/recv + fold
            monkeypatch.setattr(orr, "_push_staging", lambda *args, **kwargs: None)
        m, k, n = 64 * tp, 512 * shards, 448
        x, w = _inputs(m, k, n, 1)  # identical on every rank
        r, kl = group.rank(), k // tp
        parts = orr.row_parallel_shard_partials(
            x[:, r * kl : (r + 1) * kl], w[:, r * kl : (r + 1) * kl], shards // tp
        )
        with set_batch_invariant_mode(True):
            out = orr.ordered_reduce_scatter_shards(parts, group)
        ref = _canonical(x, w, shards)[r * (m // tp) : (r + 1) * (m // tp)]
        assert torch.equal(_bits(out), _bits(ref))

    @pytest.mark.parametrize("tp,shards", [(2, 4), (4, 8)])
    def test_all_reduce_shards_is_the_canonical_value(self, tp, shards):
        Utils.initialize_model_parallel(tp, 1)
        group = get_tensor_model_parallel_group_if_none(tp_group=None)
        m, k, n = 5, 256 * shards, 200  # odd sizes: padding path
        x, w = _inputs(m, k, n, 2)
        r, kl = group.rank(), k // tp
        parts = orr.row_parallel_shard_partials(
            x[:, r * kl : (r + 1) * kl], w[:, r * kl : (r + 1) * kl], shards // tp
        )
        with set_batch_invariant_mode(True):
            out = orr.ordered_all_reduce_shards(parts, group)
        assert torch.equal(_bits(out), _bits(_canonical(x, w, shards)))

    @pytest.mark.parametrize("tp,shards", [(2, 4), (4, 8)])
    def test_symm_pull_shards_is_the_canonical_value(self, tp, shards):
        Utils.initialize_model_parallel(tp, 1)
        group = get_tensor_model_parallel_group_if_none(tp_group=None)

        if torch.cuda.get_device_properties(0).major < 9:
            pytest.skip("symmetric memory needs sm90+")
        c = shards // tp
        m, k, n = 16 * tp, 256 * shards, 2688
        x, w = _inputs(m, k, n, 3)
        r, kl = group.rank(), k // tp
        buf = SymmetricMemoryManager.get_buffer("tp", process_group=group)
        symm = buf.maybe_get_tensor([c, m, n], dtype=torch.bfloat16)
        if symm["handle"] is None:
            pytest.skip("no symmetric memory")
        parts = orr.row_parallel_shard_partials(
            x[:, r * kl : (r + 1) * kl], w[:, r * kl : (r + 1) * kl], c, out=symm["tensor"]
        )
        out = orr.symm_ordered_reduce_scatter_shards(parts, symm["handle"])
        ref = _canonical(x, w, shards)[r * (m // tp) : (r + 1) * (m // tp)]
        assert torch.equal(_bits(out), _bits(ref))


@requires_4
class TestTERowParallelShards:
    """TERowParallelLinear output is the canonical value at every TP size dividing the shards."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("tp", [1, 2, 4])
    @pytest.mark.parametrize("sequence_parallel", [False, True])
    def test_te_row_parallel_tp_invariant(self, tp, sequence_parallel):

        Utils.initialize_model_parallel(tp, 1)
        group = get_tensor_model_parallel_group_if_none(tp_group=None)
        shards, k, n, m = 4, 2048, 512, 64
        cfg = TransformerConfig(
            num_layers=1,
            hidden_size=n,
            num_attention_heads=4,
            tensor_model_parallel_size=tp,
            sequence_parallel=sequence_parallel and tp > 1,
            bf16=True,
            params_dtype=torch.bfloat16,
            add_bias_linear=False,
            batch_invariant_mode=True,
            batch_invariant_tp_shards=shards,
            attention_backend=AttnBackend.flash,
            flash_attention_version=4,
            attention_dropout=0.0,
            hidden_dropout=0.0,
        )
        layer = TERowParallelLinear(
            k,
            n,
            config=cfg,
            init_method=lambda w: None,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
        ).cuda()
        x, w = _inputs(m, k, n, 5)
        r, kl = group.rank(), k // tp
        with torch.no_grad():
            layer.weight.copy_(w[:, r * kl : (r + 1) * kl])
        with set_batch_invariant_mode(True):
            xin = x[:, r * kl : (r + 1) * kl].unsqueeze(1).contiguous()
            for grad in (False, True):
                xg = xin.clone().requires_grad_(grad)
                with torch.set_grad_enabled(grad):
                    out, _ = layer(xg)
                ref = _canonical(x, w, shards).unsqueeze(1)
                if cfg.sequence_parallel:
                    ref = ref[r * (m // tp) : (r + 1) * (m // tp)]
                assert torch.equal(_bits(out.detach()), _bits(ref)), f"grad={grad}"
                if grad:
                    out.float().square().sum().backward()
                    assert xg.grad is not None and layer.weight.grad is not None

    def test_expert_row_parallel_keeps_its_reduction(self):
        """Routed-expert linears are not split into shards (their reduction is not covered)."""

        Utils.initialize_model_parallel(2, 1, expert_tensor_parallel_size=1)
        cfg = TransformerConfig(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=4,
            tensor_model_parallel_size=2,
            sequence_parallel=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            add_bias_linear=False,
            batch_invariant_mode=True,
            batch_invariant_tp_shards=4,
            attention_backend=AttnBackend.flash,
            flash_attention_version=4,
            attention_dropout=0.0,
            hidden_dropout=0.0,
        )
        kw = dict(
            config=cfg,
            init_method=lambda w: None,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
        )
        assert TERowParallelLinear(1024, 256, is_expert=False, **kw)._bi_tp_local_shards == 2
        assert TERowParallelLinear(1024, 256, is_expert=True, **kw)._bi_tp_local_shards == 1


@requires_4
def test_cross_entropy_is_tp_and_batch_invariant():
    """Batch-invariant vocab-parallel cross entropy with batch_invariant_tp_shards = 4: the
    same loss bits at TP 1, 2 and 4 (fixed vocab shards folded in order), and for a token in
    a smaller batch at another position. Both native implementations agree."""

    shards, vocab, tokens = 4, 4096, 64
    Utils.initialize_model_parallel(1, 1)
    groups = {tp: dist.new_group(list(range(tp))) for tp in (2, 4)}  # every rank creates them
    try:
        gen = torch.Generator(device="cuda").manual_seed(9)
        logits = torch.randn(tokens, 1, vocab, device="cuda", generator=gen) * 8
        target = torch.randint(0, vocab, (tokens, 1), device="cuda", generator=gen)
        losses = {}
        with set_batch_invariant_mode(True):
            for tp in (1, 2, 4):
                group = groups.get(tp)
                rank = dist.get_rank()
                if tp > 1 and rank >= tp:
                    continue
                local = logits.chunk(tp, dim=-1)[rank if tp > 1 else 0].contiguous()
                for name, ce in (
                    (
                        "unfused",
                        lambda l, t, g: vocab_parallel_cross_entropy(
                            l, t, tp_group=g, batch_invariant_tp_shards=shards
                        ),
                    ),
                    (
                        "fused",
                        lambda l, t, g: fused_vocab_parallel_cross_entropy(
                            l, t, g, batch_invariant_tp_shards=shards
                        ),
                    ),
                ):
                    g = group if tp > 1 else get_tensor_model_parallel_group_if_none(tp_group=None)
                    losses[tp, name] = ce(local.clone(), target, g)
                    small = ce(local[8:24].clone(), target[8:24], g)
                    assert torch.equal(_bits(small), _bits(losses[tp, name][8:24])), (tp, name)
        if dist.get_rank() == 0:
            ref = losses[1, "unfused"]
            for key, loss in losses.items():
                assert torch.equal(_bits(loss), _bits(ref)), key
    finally:
        Utils.destroy_model_parallel()


def test_config_rejects_unsupported_multi_shard_combinations():

    base = dict(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        batch_invariant_mode=True,
        batch_invariant_tp_shards=4,
        tensor_model_parallel_size=2,
        params_dtype=torch.bfloat16,
        bf16=True,
    )
    with pytest.raises(ValueError, match="not supported with fp8 yet"):
        TransformerConfig(**base, fp8="e4m3")
    with pytest.raises(ValueError, match="native cross entropy"):
        TransformerConfig(**base, cross_entropy_loss_fusion=True, cross_entropy_fusion_impl="te")
