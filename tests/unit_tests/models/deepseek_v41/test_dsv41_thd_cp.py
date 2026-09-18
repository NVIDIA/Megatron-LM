# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""GPU tests: packed (THD) sequences and context parallelism for the tiny DeepSeek-V4.1 model.

Single process: THD with one segment must reproduce the SBHD forward, a two-segment pack must
reproduce two independent SBHD sequences, and a *padded* pack (physical starts differ from the
valid lengths) must reproduce the independent sequences on its valid rows. Multi process
(``torchrun``): the CP-n forward of a packed batch must reproduce the matching row block of
the CP-1 forward, and the summed per-rank gradients must equal the CP-1 gradients, for
aligned, unaligned (compression groups split across ranks), zero-owner (ranks without any
compressed entry), padded and recomputed configurations.
"""

import pytest
import torch

from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.models.deepseek_v41.layer_specs import (
    build_dsv41_hybrid_layer_pattern,
    hybrid_dsv41_stack_spec,
)
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.models.deepseek_v41.test_dsv41_model import (
    SEQ,
    TINY_RATIOS,
    VOCAB,
    make_tiny_config,
)
from tests.unit_tests.test_utilities import Utils


def _cumsum_int32(lens, device):
    bounds = [0] + torch.tensor(lens).cumsum(0).tolist()
    return torch.tensor(bounds, dtype=torch.int32, device=device)


def _packed_params(seg_lens, device, pad_to=None):
    """Packed-sequence metadata; ``pad_to`` gives the physical length of every segment when
    the pack is padded (``cu_seqlens_q`` stays the valid cumulative lengths)."""
    cu = _cumsum_int32(seg_lens, device)
    cu_padded = cu if pad_to is None else _cumsum_int32(pad_to, device)
    max_len = max(seg_lens if pad_to is None else pad_to)
    return PackedSeqParams(
        qkv_format='thd',
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=cu_padded,
        cu_seqlens_kv_padded=cu_padded,
        max_seqlen_q=max_len,
        max_seqlen_kv=max_len,
        cp_partition_mode="contiguous",
    )


def _positions(seg_lens, device, pad_to=None):
    phys = seg_lens if pad_to is None else pad_to
    return torch.cat([torch.arange(n, device=device) for n in phys]).unsqueeze(0)


def _valid_rows(seg_lens, device, pad_to=None):
    """Bool ``[1, total_rows]`` mask of the valid rows of a (possibly padded) pack."""
    if pad_to is None:
        return torch.ones(1, sum(seg_lens), dtype=torch.bool, device=device)
    parts = [
        torch.cat([torch.ones(n, dtype=torch.bool), torch.zeros(p - n, dtype=torch.bool)])
        for n, p in zip(seg_lens, pad_to)
    ]
    return torch.cat(parts).unsqueeze(0).to(device)


def _build(config, pg_collection=None):
    return HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_dsv41_stack_spec(config),
        vocab_size=VOCAB,
        max_sequence_length=SEQ,
        hybrid_layer_pattern=build_dsv41_hybrid_layer_pattern(TINY_RATIOS),
        position_embedding_type="none",
        pg_collection=pg_collection,
    ).cuda()


def _assert_close(a, b, what, max_rel=5e-2, min_cos=1 - 1e-3):
    a, b = a.float(), b.float()
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    max_abs = (a - b).abs().max().item()
    scale = b.abs().max().item()
    assert cos > min_cos and max_abs < max_rel * max(
        scale, 1.0
    ), f"{what}: cosine {cos:.6f}, max abs diff {max_abs:.4f} (scale {scale:.3f})"


def _has_dsa_backward():
    try:
        from cudnn import DSA  # noqa: F401  (cudnn-frontend with the DSA namespace)
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
class TestTHDSingleProcess:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        if Utils.world_size != 1:
            pytest.skip("single-process test")
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(321)
        yield
        Utils.destroy_model_parallel()

    def test_single_segment_matches_sbhd(self):
        model = _build(make_tiny_config(engram=True)).eval()
        ids = torch.randint(0, VOCAB, (1, SEQ), device="cuda")
        pos = torch.arange(SEQ, device="cuda").unsqueeze(0)
        with torch.no_grad():
            sbhd = model(ids, pos, attention_mask=None)
            thd = model(
                ids, pos, attention_mask=None, packed_seq_params=_packed_params([SEQ], "cuda")
            )
        _assert_close(thd, sbhd, "THD single segment vs SBHD")

    def test_two_segments_match_independent_sequences(self):
        model = _build(make_tiny_config(engram=True)).eval()
        lens = [SEQ, SEQ - 9]
        seqs = [torch.randint(0, VOCAB, (1, n), device="cuda") for n in lens]
        with torch.no_grad():
            refs = [
                model(s, torch.arange(s.size(1), device="cuda").unsqueeze(0), attention_mask=None)
                for s in seqs
            ]
            packed = model(
                torch.cat(seqs, dim=1),
                _positions(lens, "cuda"),
                attention_mask=None,
                packed_seq_params=_packed_params(lens, "cuda"),
            )
        _assert_close(packed[:, : lens[0]], refs[0], "segment 0")
        _assert_close(packed[:, lens[0] :], refs[1], "segment 1")

    def test_padded_pack_matches_independent_sequences(self):
        """Padding rows (physical start != valid end) are not tokens: valid rows must match the
        unpadded sequences and no padding row may be compressed, hashed or attended."""
        model = _build(make_tiny_config(engram=True)).eval()
        lens = [SEQ - 3, SEQ - 9]
        pad_to = [SEQ, SEQ]  # 3 and 9 trailing padding rows
        seqs = [torch.randint(0, VOCAB, (1, n), device="cuda") for n in lens]
        pads = [torch.randint(0, VOCAB, (1, p - n), device="cuda") for n, p in zip(lens, pad_to)]
        ids = torch.cat([seqs[0], pads[0], seqs[1], pads[1]], dim=1)
        with torch.no_grad():
            refs = [
                model(s, torch.arange(s.size(1), device="cuda").unsqueeze(0), attention_mask=None)
                for s in seqs
            ]
            packed = model(
                ids,
                _positions(lens, "cuda", pad_to),
                attention_mask=None,
                packed_seq_params=_packed_params(lens, "cuda", pad_to),
            )
        assert torch.isfinite(packed).all()
        _assert_close(packed[:, : lens[0]], refs[0], "padded pack segment 0")
        _assert_close(packed[:, SEQ : SEQ + lens[1]], refs[1], "padded pack segment 1")

    def test_thd_backward_finite(self):
        model = _build(make_tiny_config(engram=True))
        lens = [SEQ, SEQ]
        ids = torch.randint(0, VOCAB, (1, sum(lens)), device="cuda")
        labels = torch.randint(0, VOCAB, (1, sum(lens)), device="cuda")
        loss = model(
            ids,
            _positions(lens, "cuda"),
            attention_mask=None,
            labels=labels,
            packed_seq_params=_packed_params(lens, "cuda"),
        )
        assert torch.isfinite(loss).all()
        loss.mean().backward()
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                assert torch.isfinite(p.grad).all(), name

    def test_explicit_mask_rejected_for_thd(self):
        model = _build(make_tiny_config(engram=True)).eval()
        ids = torch.randint(0, VOCAB, (1, SEQ), device="cuda")
        pos = torch.arange(SEQ, device="cuda").unsqueeze(0)
        mask = torch.ones(1, 1, SEQ, SEQ, dtype=torch.bool, device="cuda")
        with pytest.raises(NotImplementedError, match="attention masks"):
            with torch.no_grad():
                model(
                    ids, pos, attention_mask=mask, packed_seq_params=_packed_params([SEQ], "cuda")
                )


class _ReferenceCPGroup:
    """Stub process group so the same ranks can run a CP-1 reference forward."""

    def size(self):
        return 1

    def rank(self):
        return 0


def _unaligned_lens(total):
    """Segment lengths with odd starts so ratio-2 groups straddle 32-row rank boundaries."""
    base = [13, 8, 21, 9, 5, 3, 11, 17, 2, 1, 6]
    lens = []
    while sum(lens) < total:
        lens.append(base[len(lens) % len(base)])
    lens[-1] -= sum(lens) - total
    return [n for n in lens if n > 0]


def _raise_if_any_rank_failed(error):
    """Exchange per-rank assertion messages so a failing rank never leaves the others waiting
    in a collective (the run reports every rank's message instead of hanging)."""
    messages = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(messages, error)
    failures = [m for m in messages if m]
    assert not failures, "\n".join(failures)


FUSED_KWARGS = dict(
    hidden_size=128,
    num_attention_heads=64,
    v_head_dim=512,
    qk_pos_emb_head_dim=64,
    q_lora_rank=64,
    o_groups=8,
    o_lora_rank=64,
    dsa_indexer_n_heads=4,
    dsa_indexer_head_dim=128,
    # No discrete selections. The two sides of every fused comparison differ by bf16 kernel
    # accumulation order, and a near-tie in a discrete choice turns that into an O(1) change
    # of a few rows: top-2 of 4 experts (expert gradients cosine 0.96-0.99) and
    # indexer top-8 of up to 64 compressed keys (logit rows 15% off).
    # Routing to every expert and selecting every reachable compressed key / candidate block
    # (both are clamped to the available count) keeps the model differentiable end to end,
    # so the comparison measures the kernels and the CP arrangement, not the selectors.
    moe_router_topk=4,
    dsa_indexer_topk=256,
    csa2_candidate_topk_blocks=128,  # x block size 2 covers the top-k (config rule)
)


def make_fused_config(**overrides):
    """DSA-shaped heads (64 heads, v_head_dim 512, rope 64) so the FlashMLA sparse kernel
    applies (the kernel supports the released head counts only); dense MoE routing."""
    return make_tiny_config(engram=True, **FUSED_KWARGS, **overrides)


def _compare_grads(ref, model, tag, fused):
    """Summed per-rank gradients vs the CP-1 gradients; returns the mismatch list."""
    mismatches = []
    for (name, p_ref), (_, p_cp) in zip(ref.named_parameters(), model.named_parameters()):
        if p_ref.grad is None:
            continue
        if p_cp.grad is not None:
            g = p_cp.grad.detach().clone().float()
        else:
            g = torch.zeros_like(p_ref, dtype=torch.float32)
        torch.distributed.all_reduce(g, group=torch.distributed.group.WORLD)
        g_ref = p_ref.grad.float()
        scale = g_ref.abs().max().item()
        diff = (g - g_ref).abs().max().item()
        if not torch.isfinite(g).all():
            mismatches.append(f"{name}: non-finite gradient")
        elif fused:
            # bf16 kernels on both sides: near-zero gradients (mHC gates / biases) are
            # cancellation residues, compared on an absolute band; the rest on cosine + band.
            if scale < 1e-2:
                # Absolute band 1e-2: the noise floor of bf16 kernel accumulations relative
                # to the O(0.1-1) gradients of the model (mHC bias 4.1e-3 at scale 5.8e-3 and
                # alpha_pre 1.7e-3 at scale 8.9e-5; alpha_post 5.0e-3 at scale 6.6e-3).
                if diff > 1e-2:
                    mismatches.append(f"{name}: abs diff {diff:.2e} (scale {scale:.2e})")
            else:
                cos = torch.nn.functional.cosine_similarity(
                    g.flatten(), g_ref.flatten(), dim=0
                ).item()
                if cos < 0.99 or diff > 0.1 * max(scale, 1.0):
                    mismatches.append(f"{name}: cosine {cos:.5f}, abs diff {diff:.2e}")
        elif diff > 5e-2 * max(scale, 1e-6) + 1e-3:
            # Near-zero gradients (scalar mHC gates summing O(1e-2) bf16 contributions to
            # ~1e-5) differ by cancellation noise across ranks; the absolute floor covers that.
            mismatches.append(f"{name}: abs diff {diff:.2e} (scale {scale:.2e})")
    return [f"{tag} grad {m}" for m in mismatches]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
class TestContextParallel:
    """CP-n vs CP-1 on the reference (framework-free) sparse attention."""

    ROWS_PER_RANK = 32  # >= window (8) so every rank holds the halo
    FUSED = False

    @pytest.fixture(autouse=True)
    def setup_method(self):
        if Utils.world_size < 2:
            pytest.skip("needs torchrun with >= 2 processes")
        if self.FUSED:
            pytest.importorskip("flash_mla")
        Utils.initialize_model_parallel(1, 1, context_parallel_size=Utils.world_size)
        model_parallel_cuda_manual_seed(321)
        yield
        Utils.destroy_model_parallel()

    def _run_case(self, lens, pad_to=None, config_kwargs=None, seed=11):
        """CP-n forward and (summed) gradients vs the CP-1 reference on the same pack.

        The reference model runs the same implementation (reference or fused kernels) at CP 1;
        ``config_kwargs`` (e.g. recomputation) apply to the CP-n model only.
        """
        cp_size = Utils.world_size
        cp_rank = torch.distributed.get_rank()
        total = sum(lens if pad_to is None else pad_to)
        assert total % cp_size == 0
        local = total // cp_size
        base_kwargs = dict(FUSED_KWARGS, csa2_sparse_attention_impl="fused") if self.FUSED else {}
        config = make_tiny_config(engram=True, **base_kwargs, **(config_kwargs or {}))
        config.context_parallel_size = cp_size

        # reference model: identical weights, CP group replaced by a size-1 stub
        ref_pg = ProcessGroupCollection.use_mpu_process_groups()
        ref_pg.cp = _ReferenceCPGroup()
        torch.manual_seed(seed)
        model_parallel_cuda_manual_seed(seed)
        ref = _build(make_tiny_config(engram=True, **base_kwargs), ref_pg)
        torch.manual_seed(seed)
        model_parallel_cuda_manual_seed(seed)
        model = _build(config)
        model.load_state_dict(ref.state_dict())

        gen = torch.Generator(device="cuda").manual_seed(seed + 100)
        ids = torch.randint(0, VOCAB, (1, total), device="cuda", generator=gen)
        labels = torch.randint(0, VOCAB, (1, total), device="cuda", generator=gen)
        pos = _positions(lens, "cuda", pad_to)
        valid = _valid_rows(lens, "cuda", pad_to)
        sl = slice(cp_rank * local, (cp_rank + 1) * local)
        packed_full = _packed_params(lens, "cuda", pad_to)
        packed_local = _packed_params(lens, "cuda", pad_to)
        packed_local.cp_group = torch.distributed.group.WORLD
        kind = "fused" if self.FUSED else "reference"
        tag = f"{kind} CP{cp_size} rank {cp_rank} lens={lens[:6]}{'...' if len(lens) > 6 else ''}"

        # forward equivalence on the valid rows of this rank's block
        error = None
        with torch.no_grad():
            full = ref(ids, pos, attention_mask=None, packed_seq_params=packed_full)
            out = model(ids[:, sl], pos[:, sl], attention_mask=None, packed_seq_params=packed_local)
        try:
            assert torch.isfinite(out).all(), f"{tag}: non-finite logits"
            _assert_close(
                out[valid[:, sl]],
                full[:, sl][valid[:, sl]],
                tag,
                max_rel=0.1 if self.FUSED else 5e-2,
            )
        except AssertionError as exc:
            error = str(exc)
        _raise_if_any_rank_failed(error)

        if self.FUSED and not _has_dsa_backward():
            return  # FlashMLA forward only; the cuDNN DSA backward is not available

        # CP-1 reference: sum of per-token losses over the valid rows of the whole pack
        ref_loss = ref(ids, pos, attention_mask=None, labels=labels, packed_seq_params=packed_full)
        (ref_loss * valid).sum().backward()
        loss = model(
            ids[:, sl],
            pos[:, sl],
            attention_mask=None,
            labels=labels[:, sl],
            packed_seq_params=packed_local,
        )
        (loss * valid[:, sl]).sum().backward()
        mismatches = _compare_grads(ref, model, tag, self.FUSED)
        _raise_if_any_rank_failed("\n".join(mismatches[:8]) if mismatches else None)

    def test_cp_aligned_segments(self):
        """Segment lengths that keep ratio-2 groups inside rank blocks (the M1 baseline)."""
        total = self.ROWS_PER_RANK * Utils.world_size
        self._run_case([total // 2, total // 4, total // 4])

    def test_cp_unaligned_segments(self):
        """Compression groups straddling rank boundaries, segments shorter than the halo and
        than the ratio."""
        self._run_case(_unaligned_lens(self.ROWS_PER_RANK * Utils.world_size), seed=13)

    def test_cp_zero_owner_ranks(self):
        """The upper half of the ranks holds only length-1 segments: no ratio-2 compressed
        entry ends there, so those ranks contribute nothing to the ratio-2 all-gather but must
        still take part in its backward."""
        half = self.ROWS_PER_RANK * Utils.world_size // 2
        self._run_case([half] + [1] * half, seed=17)

    def test_cp_padded_pack(self):
        """Physical segment starts differ from the valid ends; padding rows sit inside rank
        blocks and at their boundaries."""
        total = self.ROWS_PER_RANK * Utils.world_size
        pad_to = [total // 2, total // 4, total // 4]
        lens = [pad_to[0] - 5, pad_to[1] - 3, pad_to[2] - 7]
        self._run_case(lens, pad_to=pad_to, seed=19)

    def test_cp_with_full_recompute(self):
        """Shared state crosses checkpoint boundaries under CP (uniform chunks of 2)."""
        self._run_case(
            _unaligned_lens(self.ROWS_PER_RANK * Utils.world_size),
            config_kwargs=dict(
                recompute_granularity="full", recompute_method="uniform", recompute_num_layers=2
            ),
            seed=23,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
class TestFusedContextParallel(TestContextParallel):
    """Same cases with the fused kernels (FlashMLA forward / cuDNN DSA backward) on both
    sides, DSA-shaped heads."""

    FUSED = True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
class TestFusedSparseAttention:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        if Utils.world_size != 1:
            pytest.skip("single-process test")
        pytest.importorskip("flash_mla")
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(77)
        yield
        Utils.destroy_model_parallel()

    def test_fused_rejects_unsupported_geometry(self):
        with pytest.raises(ValueError, match="fused.*geometry"):
            _build(make_tiny_config(engram=True, csa2_sparse_attention_impl="fused"))

    def test_fused_matches_reference_forward(self):
        torch.manual_seed(1)
        model_parallel_cuda_manual_seed(1)
        ref = _build(make_fused_config()).eval()
        fused = _build(make_fused_config(csa2_sparse_attention_impl="fused")).eval()
        fused.load_state_dict(ref.state_dict())
        lens = [SEQ, SEQ - 5]
        ids = torch.randint(0, VOCAB, (1, sum(lens)), device="cuda")
        pos = _positions(lens, "cuda")
        with torch.no_grad():
            a = ref(ids, pos, attention_mask=None, packed_seq_params=_packed_params(lens, "cuda"))
            b = fused(ids, pos, attention_mask=None, packed_seq_params=_packed_params(lens, "cuda"))
        # bf16 kernel accumulation vs the fp32 reference: allow a wider absolute band
        _assert_close(b, a, "fused sparse attention vs reference (forward)", max_rel=0.1)

    def test_fused_backward_matches_reference(self):
        if not _has_dsa_backward():
            pytest.skip("cudnn-frontend DSA namespace not available (sparse attention backward)")
        torch.manual_seed(2)
        model_parallel_cuda_manual_seed(2)
        ref = _build(make_fused_config())
        fused = _build(make_fused_config(csa2_sparse_attention_impl="fused"))
        fused.load_state_dict(ref.state_dict())
        lens = [SEQ, SEQ]
        ids = torch.randint(0, VOCAB, (1, sum(lens)), device="cuda")
        labels = torch.randint(0, VOCAB, (1, sum(lens)), device="cuda")
        pos = _positions(lens, "cuda")
        losses = []
        for m in (ref, fused):
            loss = m(
                ids,
                pos,
                attention_mask=None,
                labels=labels,
                packed_seq_params=_packed_params(lens, "cuda"),
            ).sum()
            loss.backward()
            losses.append(loss.detach())
        _assert_close(losses[1].view(1), losses[0].view(1), "loss", max_rel=0.1)
        for (name, p_ref), (_, p_f) in zip(ref.named_parameters(), fused.named_parameters()):
            if p_ref.grad is None:
                continue
            scale = p_ref.grad.abs().max().item()
            if scale < 1e-8:
                # e.g. mHC alpha_res of the first sub-layer: numerically zero in both paths
                assert p_f.grad.abs().max().item() < 1e-6, name
                continue
            if scale < 1e-2:
                # Tiny gradients (mHC biases / gates, O(1e-3)) are cancellation residues of
                # bf16 kernel outputs; their direction is noise (cosine 0.98-0.99 across
                # runs), so only an absolute band (same as _compare_grads) is
                # meaningful.
                assert (p_f.grad - p_ref.grad).abs().max().item() < 1e-2, f"grad {name}"
                continue
            # Gradients accumulate the bf16 kernel differences of every layer.
            _assert_close(p_f.grad, p_ref.grad, f"grad {name}", max_rel=0.1, min_cos=0.99)
