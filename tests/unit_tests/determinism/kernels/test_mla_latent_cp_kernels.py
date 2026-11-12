# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay of the single-rank kernels of the latent-KV context-parallel MLA.

``megatron/core/transformer/experimental_attention_variant/mla_latent_cp.py`` runs MLA context
parallelism as a ring over the latent KV (PR #8182). Everything of it that runs on one GPU is
replayed here with the real entry points:

* ``_merge``: the ``jit_fuser`` (``torch.compile``) online-softmax merge of one ring step into
  the fp32 accumulators, in place.
* ``_BuildKV``: strided key/value assembly from the up-projected latent and the positional
  channels broadcast over heads; its backward sums the positional key gradient over heads.
* ``_build_step_plan`` / ``_a2a_permutation`` / ``_Permute``: the integer ring metadata (cumsum,
  repeat_interleave, argsort) and the ``index_select`` permutation of the Ulysses stage, forward
  and backward.
* ``_cudnn_fwd`` / ``_cudnn_bwd`` (and the FlashAttention-4 shims where installed): the attention
  kernels of every ring step kind the planner produces -- causal (step 0), half-KV zero copy
  (owner before this rank) and local second chunk (owner after this rank) -- with the backward's
  ``deterministic`` flag that ``--deterministic-mode`` selects.
* ``_LatentCPRingAttention`` as one rank of a CP2 / CP4 ring with its transport stubbed: the
  step schedule, the per-step KV recompute, the merges, the second-chunk ``index_select`` /
  ``index_copy_`` / ``index_add_`` bookkeeping and the padding masks run as in production; only
  the values arriving from peers are synthetic. ``MLAWithLatentCP`` end to end at CP1 (RoPE,
  projections, output gate, cuDNN attention) is replayed as a module.

The NCCL exchange of the ring (CP > 1) is not qualified here; the multi-rank parity tests in
``tests/unit_tests/transformer/experimental_attention_variant/test_mla_latent_cp.py`` cover it
numerically.
"""

import functools
from types import SimpleNamespace

import pytest
import torch

from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.experimental_attention_variant import mla_latent_cp
from megatron.core.transformer.experimental_attention_variant.mla_latent_cp import (
    MLAWithLatentCP,
    _a2a_permutation,
    _build_step_plan,
    _BuildKV,
    _LatentCPRingAttention,
    _merge,
    _Permute,
    _Ring,
    apply_mla_latent_cp_spec,
)
from megatron.core.utils import is_te_min_version
from tests.unit_tests.determinism.kernels.harness import (
    CONTENTION_TOKENS,
    assert_module_replays_bit_exact,
    assert_replays_bit_exact,
    bytes_equal,
    seeded,
)
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

HIDDEN, HEADS = 64, 8
QK_HEAD_DIM, POS_DIM, V_HEAD_DIM = 128, 64, 128
Q_LORA_RANK, KV_LORA_RANK = 32, 32
# Four packed sequences; every padded length is divisible by 2 * CP up to CP4, so the zigzag
# chunks are whole. The actual lengths (the proportions of the module's parity tests) leave
# padding rows behind every sequence, a second chunk that is padding entirely on CP2 rank 0, and
# a whole local sequence of padding on CP2 rank 1.
CU_SEQLENS_PADDED = [0, 1024, 3072, 4096, 8192]
CU_SEQLENS_ACTUAL = [0, 768, 2560, 3200, 4800]
TOTAL_TOKENS = CU_SEQLENS_PADDED[-1]
MAX_SEQLEN = max(b - a for a, b in zip(CU_SEQLENS_PADDED, CU_SEQLENS_PADDED[1:]))


def _packed(padded):
    """THD metadata as the data pipeline builds it: actual and padded cumulative lengths."""
    cu = torch.tensor(CU_SEQLENS_ACTUAL if padded else CU_SEQLENS_PADDED, dtype=torch.int32)
    cup = torch.tensor(CU_SEQLENS_PADDED, dtype=torch.int32).cuda() if padded else None
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu.cuda(),
        cu_seqlens_kv=cu.cuda(),
        cu_seqlens_q_padded=cup,
        cu_seqlens_kv_padded=cup,
        max_seqlen_q=MAX_SEQLEN,
        max_seqlen_kv=MAX_SEQLEN,
    )


def _plan(cp_size, cp_rank, padded):
    return _build_step_plan(_packed(padded), cp_size, cp_rank, TOTAL_TOKENS // cp_size)


def _rank_actual_tokens(cp_size, cp_rank):
    """Actual (non-padding) tokens of ``cp_rank``'s two zigzag chunks of every sequence."""
    total = 0
    for s0, s1, p0, p1 in zip(
        CU_SEQLENS_ACTUAL, CU_SEQLENS_ACTUAL[1:], CU_SEQLENS_PADDED, CU_SEQLENS_PADDED[1:]
    ):
        actual, half = s1 - s0, (p1 - p0) // (2 * cp_size)
        for chunk in (cp_rank, 2 * cp_size - 1 - cp_rank):
            total += min(max(actual - chunk * half, 0), half)
    return total


def _step_kind(step):
    return "causal" if step.causal else ("second_chunk" if step.upper else "half_kv")


def _step_kinds(padded):
    """The three ring step kinds from the real planner on a CP2 ring, with the query rows and
    the query validity mask each one attends with: rank 0 runs the causal step 0 and then its
    local second chunk against owner 1; rank 1 runs step 0 and the half-KV step against owner 0."""
    rank0, rank1 = _plan(2, 0, padded), _plan(2, 1, padded)
    assert [_step_kind(s) for s in rank0.steps] == ["causal", "second_chunk"]
    assert [_step_kind(s) for s in rank1.steps] == ["causal", "half_kv"]
    second_valid = None if rank0.valid is None else rank0.valid.index_select(0, rank0.rows)
    return {
        "causal": (rank0.steps[0], None, rank0.valid),
        "half_kv": (rank1.steps[1], None, rank1.valid),
        "second_chunk": (rank0.steps[1], rank0.rows, second_valid),
    }


def _have_fa4():
    if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10):
        return False
    try:
        mla_latent_cp._fa4()
    except Exception:  # pylint: disable=broad-except
        return False
    return True


def _backends():
    """``(attn_fwd, attn_bwd)`` pairs the module can select: cuDNN always, FlashAttention-4 and
    the mixed directions where the pinned FA4 build is installed (Blackwell)."""
    cudnn = (mla_latent_cp._cudnn_fwd, mla_latent_cp._cudnn_bwd)
    params = [pytest.param(cudnn, id="cudnn")]
    if _have_fa4():
        fa4 = (mla_latent_cp._fa4_fwd, mla_latent_cp._fa4_bwd)
        params += [
            pytest.param(fa4, id="fa4"),
            pytest.param((cudnn[0], fa4[1]), id="cudnn_fwd-fa4_bwd"),
            pytest.param((fa4[0], cudnn[1]), id="fa4_fwd-cudnn_bwd"),
        ]
    return params


@functools.lru_cache(maxsize=None)
def _cudnn_unavailable_reason():
    """Why the cuDNN THD shims cannot run here (``None`` when they can), from one probe call."""
    if not HAVE_TE:
        return "needs Transformer Engine"
    if not is_te_min_version("2.5.0", check_equality=True):
        return "MLAWithLatentCP needs Transformer Engine >= 2.5.0"
    if not mla_latent_cp._FUSED_ATTN_API_OK:
        return "needs Transformer Engine fused_attn_fwd/bwd with the THD layout keywords"
    n = 64
    q = torch.randn(n, 2, QK_HEAD_DIM + POS_DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(n, 2, V_HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0, n], dtype=torch.int32, device="cuda")
    step = mla_latent_cp._Step(False, cu, cu, cu, cu, n, n, True)
    try:
        out, lse, rng = mla_latent_cp._cudnn_fwd(q, k, v, step, 0.1)
        mla_latent_cp._cudnn_bwd(q, k, v, out, torch.ones_like(out), lse, rng, step, 0.1, True)
        torch.cuda.synchronize()
    except RuntimeError as error:
        message = str(error).lower()
        if any(w in message for w in ("support", "cudnn", "backend", "avail", "softmax stats")):
            return f"cuDNN fused attention cannot run the latent-CP THD shims here: {error}"
        raise
    return None


def _skip_unless_cudnn_shims():
    reason = _cudnn_unavailable_reason()
    if reason:
        pytest.skip(reason)


def _config(**overrides):
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.transformer_config import MLATransformerConfig

    kwargs = dict(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        q_lora_rank=Q_LORA_RANK,
        kv_lora_rank=KV_LORA_RANK,
        qk_head_dim=QK_HEAD_DIM,
        qk_pos_emb_head_dim=POS_DIM,
        v_head_dim=V_HEAD_DIM,
        qk_layernorm=True,
        attention_output_gate=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        attention_backend=AttnBackend.fused,
        mla_latent_cp=True,
        deterministic_mode=True,
    )
    kwargs.update(overrides)
    return MLATransformerConfig(**kwargs)


def _build_module():
    """``MLAWithLatentCP`` from the TE MLA layer spec through the production spec helper."""
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.spec_utils import build_module

    model_parallel_cuda_manual_seed(123)
    spec = get_gpt_layer_with_transformer_engine_spec(
        multi_latent_attention=True, qk_layernorm=True
    ).submodules.self_attention
    module = build_module(apply_mla_latent_cp_spec(spec), config=_config(), layer_number=1).cuda()
    assert isinstance(module, MLAWithLatentCP)
    assert module._attn_fwd is mla_latent_cp._cudnn_fwd
    assert module._attn_bwd is mla_latent_cp._cudnn_bwd
    return module


# --- jit_fuser merge and key/value assembly ----------------------------------------------------


def test_merge_replays_and_matches_reference():
    """The compiled in-place online-softmax merge of one ring step into the fp32 accumulators."""
    seeded()
    n, h, d = CONTENTION_TOKENS, HEADS, V_HEAD_DIM
    out = torch.randn(n, h, d, device="cuda")
    lse = torch.randn(n, h, device="cuda")
    out_i = torch.randn(n, h, d, device="cuda", dtype=torch.bfloat16)
    lse_i = torch.randn(n, h, device="cuda")

    def merge(out, lse, out_i, lse_i):
        _merge(out, lse, out_i, lse_i)
        return out, lse

    outputs, _ = assert_replays_bit_exact(
        merge,
        (out, lse, out_i, lse_i),
        backward=False,
        contention=True,
        what="latent-CP online-softmax merge",
    )
    lse_new = torch.logaddexp(lse, lse_i)
    out_new = out * torch.exp(lse - lse_new).unsqueeze(-1) + out_i * torch.exp(
        lse_i - lse_new
    ).unsqueeze(-1)
    torch.testing.assert_close(outputs["out[0]"], out_new, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(outputs["out[1]"], lse_new, rtol=1e-6, atol=1e-6)


def test_build_kv_replays_fwd_bwd():
    """Strided key/value copies forward; the positional key gradient is summed over heads."""
    seeded()
    n = CONTENTION_TOKENS
    kv = torch.randn(
        n, HEADS, QK_HEAD_DIM + V_HEAD_DIM, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    k_pos = torch.randn(n, POS_DIM, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    grad_outputs = {
        "out[0]": torch.randn(n, HEADS, QK_HEAD_DIM + POS_DIM, device="cuda", dtype=torch.bfloat16),
        "out[1]": torch.randn(n, HEADS, V_HEAD_DIM, device="cuda", dtype=torch.bfloat16),
    }
    outputs, grads = assert_replays_bit_exact(
        lambda kv, k_pos: _BuildKV.apply(kv, k_pos, QK_HEAD_DIM),
        (kv, k_pos),
        grad_outputs=grad_outputs,
        contention=True,
        what="latent-CP key/value assembly",
    )
    key, value = outputs["out[0]"], outputs["out[1]"]
    assert bytes_equal(key[..., :QK_HEAD_DIM], kv[..., :QK_HEAD_DIM])
    assert bytes_equal(key[..., QK_HEAD_DIM:], k_pos.unsqueeze(1).expand(-1, HEADS, -1))
    assert bytes_equal(value, kv[..., QK_HEAD_DIM:])
    assert set(grads) == {"in[0]", "in[1]"}
    assert bytes_equal(grads["in[0]"][..., QK_HEAD_DIM:], grad_outputs["out[1]"])
    assert grads["in[1]"].shape == k_pos.shape


# --- ring metadata and the a2a permutation ------------------------------------------------------


@pytest.mark.parametrize(("cp_size", "cp_rank", "a2a_size"), ((2, 0, 2), (4, 1, 2), (4, 3, 4)))
@pytest.mark.parametrize("padded", [False, True], ids=["packed", "padded"])
def test_ring_metadata_replays(cp_size, cp_rank, a2a_size, padded):
    """Step plan (per-step cu_seqlens, second-chunk rows, validity mask) and a2a permutation."""
    psp = _packed(padded)
    tokens = TOTAL_TOKENS // cp_size

    def build(cu, cup):
        params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            cu_seqlens_q_padded=cup,
            cu_seqlens_kv_padded=cup,
            max_seqlen_q=MAX_SEQLEN,
            max_seqlen_kv=MAX_SEQLEN,
        )
        plan = _build_step_plan(params, cp_size, cp_rank, tokens)
        perm, inv = _a2a_permutation(params, cp_size, a2a_size, tokens)
        return plan, perm, inv

    outputs, _ = assert_replays_bit_exact(
        build,
        (psp.cu_seqlens_q, psp.cu_seqlens_q_padded),
        backward=False,
        contention=True,
        what=f"latent-CP ring metadata (rank {cp_rank}/{cp_size}, a2a {a2a_size})",
    )
    perm, inv = outputs["out[1]"], outputs["out[2]"]
    assert torch.equal(perm.index_select(0, inv), torch.arange(a2a_size * tokens, device="cuda"))
    if cp_rank < cp_size - 1:
        rows = outputs["out[0][1]"]
        assert rows.numel() == tokens // 2 and rows.unique().numel() == rows.numel()
        assert 0 <= int(rows.min()) and int(rows.max()) < tokens
    else:
        assert "out[0][1]" not in outputs
    if padded:
        valid = outputs["out[0][2]"]
        assert valid.shape == (tokens,)
        assert int(valid.sum()) == _rank_actual_tokens(cp_size, cp_rank)
    else:
        assert "out[0][2]" not in outputs


def test_a2a_permute_replays_fwd_bwd():
    """``index_select`` in both directions through the a2a permutation of the Ulysses stage."""
    seeded()
    cp_size, a2a_size = 4, 2
    tokens = TOTAL_TOKENS // cp_size
    perm, inv = _a2a_permutation(_packed(True), cp_size, a2a_size, tokens)
    x = torch.randn(
        a2a_size * tokens,
        HEADS // a2a_size,
        QK_HEAD_DIM + POS_DIM,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    grad = torch.randn_like(x)
    outputs, grads = assert_replays_bit_exact(
        lambda t: _Permute.apply(t, perm, inv),
        (x,),
        grad_outputs={"out": grad},
        contention=True,
        what="latent-CP a2a permutation",
    )
    assert bytes_equal(outputs["out"].index_select(0, inv), x)
    assert bytes_equal(grads["in[0]"], grad.index_select(0, inv))


# --- attention kernels of one ring step ---------------------------------------------------------


@pytest.mark.parametrize("kind", ["causal", "half_kv", "second_chunk"])
@pytest.mark.parametrize("padded", [False, True], ids=["packed", "padded"])
@pytest.mark.parametrize("backend", _backends())
def test_ring_step_attention_replays_fwd_bwd(backend, kind, padded):
    """One ring step on the planner's THD metadata: forward, then backward with deterministic=True.

    The softmax stats of padding rows are undefined; the module zeroes them before merging, and
    the replay does the same before comparing. Outputs and gradients of rows the kernels never
    visit are zero-filled by Transformer Engine / the FA4 shims.
    """
    _skip_unless_cudnn_shims()
    fwd, bwd = backend
    seeded()
    step, rows, valid = _step_kinds(padded)[kind]
    tokens = TOTAL_TOKENS // 2
    q = torch.randn(tokens, HEADS, QK_HEAD_DIM + POS_DIM, device="cuda", dtype=torch.bfloat16)
    if rows is not None:
        q = q.index_select(0, rows)
    k = torch.randn(tokens, HEADS, QK_HEAD_DIM + POS_DIM, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(tokens, HEADS, V_HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    dout = torch.randn(q.shape[0], HEADS, V_HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    pad = None if valid is None else (~valid).unsqueeze(-1)
    scale = (QK_HEAD_DIM + POS_DIM) ** -0.5

    def run(q, k, v):
        out, lse, rng = fwd(q, k, v, step, scale)
        if pad is not None:
            lse = lse.masked_fill(pad, 0.0)
        dq, dk, dv = bwd(q, k, v, out, dout, lse, rng, step, scale, True)
        return out, lse, dq, dk, dv

    outputs, _ = assert_replays_bit_exact(
        run,
        (q, k, v),
        backward=False,
        contention=True,
        what=f"latent-CP ring step [{kind}, padded={padded}]",
    )
    assert outputs["out[0]"].shape == dout.shape and outputs["out[1]"].shape == q.shape[:2]
    assert all(torch.isfinite(t).all() for t in outputs.values())
    if kind == "half_kv":
        # The step attends to owner 0's first chunks only (half lengths through the full offsets):
        # the keys of its second chunks are never read and their gradients stay zero-filled.
        unread = _plan(2, 0, padded).rows
        assert not outputs["out[3]"].index_select(0, unread).any()
        assert not outputs["out[4]"].index_select(0, unread).any()


# --- the ring on one rank, and the module at CP1 ------------------------------------------------


class _StubTransport:
    """Stands in for ``_p2p`` on one GPU.

    Every receive buffer is filled with what the peer would send: the latent of the owner the
    next step attends to (latents travel forward in the forward ring and backward in the backward
    ring) and a zero d-latent, so the local arithmetic of the ring runs unchanged without NCCL.
    """

    def __init__(self, latents, rank, size):
        self.latents, self.rank, self.size = latents, rank, size
        self.reset()

    def reset(self):
        self.fwd_calls = self.bwd_calls = 0

    def __call__(self, ring, sends, recvs, forward):
        assert ring.rank == self.rank and ring.size == self.size
        last = self.size - 1
        if forward:
            self.fwd_calls += 1
            assert len(sends) == len(recvs) == 1
            recvs[0].copy_(self.latents[(self.rank - self.fwd_calls) % self.size])
        else:
            i, self.bwd_calls = self.bwd_calls, self.bwd_calls + 1
            if i < last:
                assert len(sends) == len(recvs) == 2
                recvs[0].copy_(self.latents[(self.rank - (last - i - 1)) % self.size])
                recvs[1].zero_()
            else:
                assert len(sends) == len(recvs) == 1
                recvs[0].zero_()
        return [SimpleNamespace(wait=lambda: None)]


_RING_STEP_KINDS = {
    (2, 0): ["causal", "second_chunk"],
    (2, 1): ["causal", "half_kv"],
    (4, 1): ["causal", "half_kv", "second_chunk", "second_chunk"],
}


@pytest.mark.parametrize(("cp_size", "cp_rank"), sorted(_RING_STEP_KINDS))
@pytest.mark.parametrize("padded", [False, True], ids=["packed", "padded"])
def test_latent_ring_replays_on_one_rank(cp_size, cp_rank, padded, monkeypatch):
    """``_LatentCPRingAttention`` as rank ``cp_rank`` of a CP ``cp_size`` ring with the transport
    stubbed: per-step KV recompute through the TE up-projection, cuDNN attention of every step
    kind, the fp32 merges, the second-chunk ``index_select`` / ``index_copy_`` / ``index_add_``
    bookkeeping, the padding masks, and the latent / up-projection parameter gradients."""
    _skip_unless_cudnn_shims()
    Utils.initialize_model_parallel()
    try:
        module = _build_module()
        seeded()
        tokens = TOTAL_TOKENS // cp_size
        plan = _plan(cp_size, cp_rank, padded)
        assert [_step_kind(s) for s in plan.steps] == _RING_STEP_KINDS[(cp_size, cp_rank)]
        ring = _Ring(None, cp_rank, cp_size, (cp_rank + 1) % cp_size, (cp_rank - 1) % cp_size)
        latents = [
            torch.randn(tokens, KV_LORA_RANK + POS_DIM, device="cuda", dtype=torch.bfloat16)
            for _ in range(cp_size)
        ]
        transport = _StubTransport(latents, cp_rank, cp_size)
        monkeypatch.setattr(mla_latent_cp, "_p2p", transport)
        q = torch.randn(tokens, HEADS, QK_HEAD_DIM + POS_DIM, device="cuda", dtype=torch.bfloat16)
        dout = torch.randn(tokens, HEADS, V_HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        up_params = [
            p
            for m in (module.kv_layernorm, module.linear_kv_up_proj)
            for p in m.parameters()
            if p.requires_grad
        ]
        assert up_params

        def run(q, z):
            transport.reset()
            q = q.detach().requires_grad_()
            z = z.detach().requires_grad_()
            out = _LatentCPRingAttention.apply(
                q,
                z,
                module,
                plan,
                ring,
                module._attn_fwd,
                module._attn_bwd,
                module.softmax_scale,
                *up_params,
            )
            grads = torch.autograd.grad(out, (q, z, *up_params), dout)
            assert transport.fwd_calls == cp_size - 1 and transport.bwd_calls == cp_size
            return (out, *grads)

        outputs, _ = assert_replays_bit_exact(
            run,
            (q, latents[cp_rank]),
            backward=False,
            contention=True,
            what=f"latent-CP ring on rank {cp_rank} of {cp_size} (padded={padded})",
        )
        assert len(outputs) == 3 + len(up_params)
        assert outputs["out[0]"].shape == dout.shape and outputs["out[1]"].shape == q.shape
        assert outputs["out[2]"].shape == latents[cp_rank].shape
        assert all(torch.isfinite(t).all() for t in outputs.values())
        if plan.valid is not None:  # padding queries attend to nothing: zero output and gradient
            pad = ~plan.valid
            assert not outputs["out[0]"][pad].any() and not outputs["out[1]"][pad].any()
    finally:
        Utils.destroy_model_parallel()


def test_mla_latent_cp_module_replays_at_cp1():
    """``MLAWithLatentCP`` forward and backward on one rank (ring of size 1): RoPE on the query
    and the latent, the TE down / up projections, the cuDNN ring step with ``deterministic_mode``,
    the output gate and the output projection; every parameter and the input gradient compared."""
    _skip_unless_cudnn_shims()
    Utils.initialize_model_parallel()
    try:
        module = _build_module()
        seeded()
        hidden = torch.randn(
            TOTAL_TOKENS, 1, HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        inputs = dict(hidden_states=hidden, attention_mask=None, packed_seq_params=_packed(False))
        outputs, grads = assert_module_replays_bit_exact(
            module, inputs, replays=3, contention=True, what="MLAWithLatentCP at CP1"
        )
        assert outputs["out[0]"].shape == hidden.shape
        assert "in.hidden_states" in grads
        for name in ("linear_kv_down_proj", "linear_kv_up_proj", "linear_q_up_proj", "linear_proj"):
            assert any(name in key for key in grads), name
        assert all(torch.isfinite(t).all() for t in outputs.values())
        assert all(torch.isfinite(t).all() for t in grads.values())
    finally:
        Utils.destroy_model_parallel()
