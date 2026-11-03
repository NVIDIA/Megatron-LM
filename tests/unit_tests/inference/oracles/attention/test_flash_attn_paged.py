# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for the paged FlashAttention call sites of dynamic inference.

Call site: ``Attention.flash_decode_and_prefill`` (megatron/core/transformer/attention.py), which
dispatches to FA2 ``flash_attn_varlen_func(block_table=)`` (prefill / mixed) and
``flash_attn_with_kvcache(block_table=)`` (decode), FA3 ``_flash_attn_forward(page_table=)`` and
``flash_attn_with_kvcache(page_table=)``, and FA4 ``flash_attn_varlen_func(page_table=)`` (both
phases), then applies the sink correction from the kernel's log-sum-exp.

Contract (from the call site's arguments, the dynamic context's metadata and the flash-attn API):

* Inputs. ``q``: prefill/mixed ``[T, 1, H, D]`` (rows of the live requests' new tokens in request
  order, then padding rows), decode ``[R * S, 1, H, D]`` (S tokens for each of R requests, padding
  requests included). ``k``/``v``: the paged pool ``[pages, P, Hk, D]``. ``block_table [R, W]``
  int32. ``cu_seqlens_q``/``cu_seqlens_k``/``seqlens_k`` from the MHA metadata: padding requests
  come last with zero query and key lengths, and every block-table entry they own, like every
  entry past a live request's pages, is the dummy page.
* Computes, for live request ``r`` with ``S_r`` new tokens and ``L_r`` keys: key ``j`` is
  ``pool[block_table[r, j // P], j % P]``; query ``i`` sits at position ``L_r - S_r + i`` and
  attends keys ``j <= L_r - S_r + i`` (bottom-right causal), and with the layer's window
  ``(W, 0)`` only keys ``j >= L_r - S_r + i - W``; scores ``s = scale * q.k`` with
  ``scale = D ** -0.5`` (the layers leave ``TransformerConfig.softmax_scale`` unset, so the scale
  is the definition's, from the configured head dim ``kv_channels = D``; the reference takes it
  from the configuration, never from the layer); softcap ``c * tanh(s / c)``; a sink adds
  ``exp(sink_h)`` to head ``h``'s softmax denominator; GQA head ``h`` reads KV head
  ``h // (H / Hk)``.
* Outputs: prefill ``[T, 1, H, Dv]``, decode ``[R * S, 1, H, Dv]``, in q's row order. Rows past
  the live rows (padding) are unspecified: no zeroing contract outside quantized runs.
* Memory: the pool is read only. The kernel reads only the first ``ceil(L_r / P)`` block-table
  entries of a request and only rows ``< L_r``. Unused pages and the dummy page hold NaN here;
  the key rows past ``L_r`` in a request's last page hold NaN (the context never initializes key
  pages); its value rows there hold zero, because the context zeroes value pages when it acquires
  them (``DynamicInferenceContext._initialize_mha_value_pages``). FA2 also masks NaN value tails
  (it zero-fills out-of-bounds value rows), which ``v_tail="nan"`` checks.
* Batch-invariant mode (FA3/FA4, ``num_splits=1``): a request's rows are bitwise independent of
  the other requests in the batch.

Reference: fp64 attention over the KV gathered through the block table from the definition
(``attention_reference.paged_attention_reference``). Tolerance class: floating point, bounded by
the trusted yardstick ``attention_lowp`` (SDPA math-backend order in the kernel dtype) on the same
gathered KV, per token row (``dim=-1``). Batch-invariance: exact (property).

Variants covered: FA2, FA3, FA4 (each where it is installed and runs: FA2 and FA3 on Hopper, FA4
on Hopper and Blackwell); prefill/mixed and decode-only; page size 256 (FA2's only legal size) and
64/128/256 for FA3/FA4 (FA4 uses TMA paging only at 128); lengths ``1, P-1, P, P+1, 2P-1, 2P+1,
4097`` in one batch (rows of different lengths, the last crossing split-KV chunks); chunked
prefill continuing across a page boundary; decode with ``S`` in 1, 2, 4 (MTP); GQA 1, 4, 8;
head dim 64 and 128; window ``P-1``, ``P``, ``P+1``; softcap; learnable sink; zero-length padding
requests; FA4 ``max_seqlen_k`` at the block-table capacity (as CUDA graphs pass it); FA3/FA4
batch-invariant mode. Factors combined pairwise. Decode whose packed query rows ``S * H / Hk``
(FA3/FA4 pack a KV head's query heads with the request's tokens into the rows of one M tile) sit
below, at and past the 128-row M tile and past FA4's two-stage 256-row tile on SM100, where FA4
sizes its schedule from ``max_seqlen_q * H / Hk`` (one tile per request when that fits):
``test_decode_query_rows_cross_the_m_tile``.

Open PRs (written against main):
* #8584 (n2): on main, FA4 on Hopper asks for split-KV (``num_splits=0``), which FA4's SM90
  forward rejects ("SplitKV not supported on SM 9.0") once the KV is long (the 4097-key requests,
  and the capacity ``max_seqlen_k``): the FA4 cases fail on H100 on main and pass with n2.
  ``test_fa2_refuses_pages_it_cannot_read`` fails on main (FA2 raises a RuntimeError inside
  flash-attn) and passes with n2 (a ValueError from Megatron).
* #8860 (t3-g1): on main, FA4's SM100 forward cannot compile in the CI image (G1), so the FA4
  cases fail on GB200 on main and pass with t3-g1's FA4 lock.
* #8878 (t3-flashdecode) moves the window lookup into ``_flash_attn_window_size``; the window
  cases here must pass unchanged with it.
"""

from typing import Optional

import pytest
import torch

from megatron.core.transformer import attention as attention_module
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.attention import attention_reference as ar
from tests.unit_tests.test_utilities import Utils

HEADS = 8
LONG_KV = 4097  # crosses several split-KV chunks and 128-token tiles
FA_PAGE_SIZES = {2: (256,), 3: (64, 128, 256), 4: (64, 128, 256)}
SOFTCAP = 8.0
SOFTCAP_Q_SCALE = 4.0  # makes |scores| reach the cap, so a dropped softcap is visible
# FA generations the CI image ships: flash-attn 2 and its FA4 cute interface, not FA3.
IN_CI_IMAGE = {2, 4}


def _installed(version: int) -> bool:
    if version == 2:
        return (
            attention_module.flash_attn_varlen_func is not None
            and attention_module.flash_attn_with_kvcache is not None
        )
    if version == 3:
        return attention_module.HAVE_FA3
    return attention_module.HAVE_FA4


def _require_flash(version: int) -> None:
    """Skip when this FA generation is not installed. A run that targets the hardware
    (``MCORE_ORACLE_TARGET_SM``, which CI sets) fails instead for a generation the CI image
    ships, so its cases never pass by skipping; FA3's cases skip wherever FA3 is missing."""
    if _installed(version):
        return
    reason = f"FlashAttention-{version} is not installed"
    if version in IN_CI_IMAGE and ou.session_target_sm() is not None:
        pytest.fail(reason + " on a run that targets this hardware", pytrace=False)
    pytest.skip(reason)


def _version_marks(version: int):
    if version == 4:
        return [ou.requires_sm(ou.HOPPER, ou.BLACKWELL), pytest.mark.launch_on_gb200]
    return [ou.requires_sm(ou.HOPPER)]


@pytest.fixture
def model_parallel():
    Utils.initialize_model_parallel(1, 1)
    yield
    Utils.destroy_model_parallel()


def _attention(
    version: int,
    *,
    kv_heads: int,
    head_dim: int,
    window: Optional[int] = None,
    softcap: Optional[float] = None,
    batch_invariant: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    heads: int = HEADS,
):
    """A ``SelfAttention`` layer (TE spec) pinned to FlashAttention ``version``."""
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_submodules,
    )
    from megatron.core.transformer.attention import SelfAttention
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.transformer_config import TransformerConfig

    config = TransformerConfig(
        num_layers=1,
        hidden_size=heads * head_dim,
        num_attention_heads=heads,
        num_query_groups=kv_heads,
        kv_channels=head_dim,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        params_dtype=dtype,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        add_bias_linear=False,
        flash_attention_version=version,
        window_size=None if window is None else (window, 0),
        attn_logit_softcapping=softcap,
    )
    submodules = get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules
    module = SelfAttention(config, submodules, layer_number=1, attn_mask_type=AttnMaskType.causal)
    module = module.cuda().eval()
    # Batch-invariant mode as the dispatch reads it (num_splits=1); the config-level switch also
    # needs the training attention backend, which this test does not use.
    module.batch_invariant_mode = batch_invariant
    return module


def _prefill_requests(page: int):
    """(q_len, kv_len) per live request of a prefill/mixed step: rows of different lengths around
    the page size, a chunked-prefill chunk crossing a page boundary, and a decode row over a long
    KV."""
    return [
        (1, 1),
        (page - 1, page - 1),
        (page, page),
        (page + 1, page + 1),
        (3, 2 * page - 1),
        (page + 2, 2 * page + 1),  # chunk starting at page - 1 cached tokens
        (1, LONG_KV),
    ]


def _decode_requests(page: int, tokens: int):
    lengths = sorted(
        {max(tokens, 1), page - 1, page, page + 1, 2 * page - 1, 2 * page + 1, LONG_KV}
    )
    return [(tokens, n) for n in lengths]


def _run(
    module,
    *,
    phase: str,
    requests,
    padding: int,
    page: int,
    kv_heads: int,
    head_dim: int,
    tokens: int = 1,
    softcap: Optional[float] = None,
    sink: Optional[torch.Tensor] = None,
    v_tail: str = "zero",
    capacity_max_seqlen_k: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    heads: int = HEADS,
):
    """Run the call site on a poisoned paged batch; return (out, q, batch, live_rows)."""
    g = ou.make_generator(device="cuda")
    q_lens = [s for s, _ in requests] + [0] * padding
    kv_lens = [n for _, n in requests] + [0] * padding
    batch = ar.make_paged_batch(
        q_lens,
        kv_lens,
        page_size=page,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        generator=g,
        tail_value=0.0 if v_tail == "zero" else float("nan"),
    )
    live = sum(q_lens)
    if phase == "decode":
        rows = len(q_lens) * tokens  # padding requests own query rows too
        max_seqlen_q = tokens
    else:
        rows = -(-(live + 1) // 64) * 64  # padded token count, as CUDA-graph buckets pad it
        max_seqlen_q = max(q_lens)
    q = ou.poisoned_empty((rows, 1, heads, head_dim), dtype)
    q_live = torch.randn(live, 1, heads, head_dim, device="cuda", generator=g)
    if softcap is not None:
        q_live = q_live * SOFTCAP_Q_SCALE
    q[:live] = q_live.to(dtype)
    q_before = q.clone()
    max_seqlen_k = max(kv_lens)
    if capacity_max_seqlen_k:
        max_seqlen_k = batch.block_table.shape[1] * page
    with torch.no_grad():
        out = module.flash_decode_and_prefill(
            q,
            batch.k,
            batch.v,
            max_seqlen_q,
            max_seqlen_k,
            batch.cu(q_lens),
            batch.cu(kv_lens),
            batch.int_tensor(kv_lens),
            batch.block_table,
            phase == "decode",
            softmax_offset=sink,
        )
    torch.cuda.synchronize()
    batch.check_pool_unchanged()
    ou.assert_exact(q, q_before, what="q (read-only)")
    assert out.shape == (rows, 1, heads, head_dim), f"output shape {tuple(out.shape)}"
    return out, q, batch, live


def _check(out, q, batch, live, *, head_dim, softcap, sink, window, what):
    # The definition's scale for the configured head dim (``kv_channels``; no softmax_scale in the
    # config), independent of the layer's own lookup (``get_softmax_scale``).
    scale = head_dim**-0.5
    q_rows = q.squeeze(1)
    ref, _ = ar.paged_attention_reference(
        q_rows, batch, scale=scale, window=window, softcap=softcap, sink=sink
    )
    trusted, _ = ar.paged_attention_reference(
        q_rows, batch, scale=scale, window=window, softcap=softcap, sink=sink, lowp=True
    )
    flat = lambda t: t.reshape(t.shape[0], -1)  # noqa: E731  (per token row: [T, H * Dv])
    ou.assert_close_to_reference(flat(out[:live, 0]), flat(ref), flat(trusted), dim=-1, what=what)


def _sink(version_dtype: torch.dtype, kind: Optional[str]):
    if kind is None:
        return None
    g = ou.make_generator(seed=ou.DEFAULT_SEED + 1, device="cuda")
    # Per-head logits around the scale of the real ones, so the sink takes a visible share.
    return (torch.randn(HEADS, device="cuda", generator=g) * 2 + 1).to(version_dtype)


def _cases(version: int, phase: str):
    factors = {
        "page": list(FA_PAGE_SIZES[version]),
        "kv_heads": [HEADS, HEADS // 4, HEADS // 8],  # GQA 1, 4, 8
        "head_dim": [64, 128],
        "window": ["none", "P-1", "P", "P+1"],
        "softcap": [None, SOFTCAP],
        "sink": [None, "learnable"],
        "padding": [0, 2],
    }
    if phase == "decode":
        factors["tokens"] = [1, 2, 4]
    if version == 4:
        factors["capacity_max_seqlen_k"] = [False, True]
    if version in (3, 4):
        factors["batch_invariant"] = [False, True]
    if version == 2:
        factors["v_tail"] = ["zero", "nan"]
    params = []
    for case in ou.pairwise_cases(factors):
        # "fa4bi-..." for batch-invariant cases, so the ids can be selected by prefix.
        tag = f"fa{version}" + ("bi" if case.get("batch_invariant") else "")
        case_id = f"{tag}-" + "-".join(
            f"{k}={v}"
            for k, v in case.items()
            if k != "batch_invariant" and v not in (None, False, "none", "zero")
        )
        params.append(pytest.param(version, case, id=case_id, marks=_version_marks(version)))
    return params


def _window(spec: str, page: int) -> Optional[int]:
    return {"none": None, "P-1": page - 1, "P": page, "P+1": page + 1}[spec]


def _run_case(version, phase, case, dtype=torch.bfloat16):
    _require_flash(version)
    page = case["page"]
    window = _window(case["window"], page)
    module = _attention(
        version,
        kv_heads=case["kv_heads"],
        head_dim=case["head_dim"],
        window=window,
        softcap=case["softcap"],
        batch_invariant=case.get("batch_invariant", False),
        dtype=dtype,
    )
    tokens = case.get("tokens", 1)
    requests = _decode_requests(page, tokens) if phase == "decode" else _prefill_requests(page)
    sink = _sink(dtype, case["sink"])
    out, q, batch, live = _run(
        module,
        phase=phase,
        requests=requests,
        padding=case["padding"],
        page=page,
        kv_heads=case["kv_heads"],
        head_dim=case["head_dim"],
        tokens=tokens,
        softcap=case["softcap"],
        sink=sink,
        v_tail=case.get("v_tail", "zero"),
        capacity_max_seqlen_k=case.get("capacity_max_seqlen_k", False),
        dtype=dtype,
    )
    _check(
        out,
        q,
        batch,
        live,
        head_dim=case["head_dim"],
        softcap=case["softcap"],
        sink=sink,
        window=window,
        what=f"FA{version} {phase} {case}",
    )


@pytest.mark.usefixtures("model_parallel")
@pytest.mark.parametrize(
    "version, case", _cases(2, "prefill") + _cases(3, "prefill") + _cases(4, "prefill")
)
def test_prefill_matches_reference(version, case):
    """Prefill/mixed steps (varlen kernels over the paged pool) match the fp64 reference."""
    _run_case(version, "prefill", case)


@pytest.mark.usefixtures("model_parallel")
@pytest.mark.parametrize(
    "version, case", _cases(2, "decode") + _cases(3, "decode") + _cases(4, "decode")
)
def test_decode_matches_reference(version, case):
    """Decode-only steps (kvcache kernels, FA4 varlen) match the fp64 reference."""
    _run_case(version, "decode", case)


@pytest.mark.usefixtures("model_parallel")
@pytest.mark.parametrize("phase", ["prefill", "decode"])
@ou.requires_sm(ou.HOPPER)
def test_fp16_matches_reference(phase):
    """fp16 (FA2): the same contract in the other half-precision dtype."""
    case = dict(
        page=256,
        kv_heads=HEADS // 4,
        head_dim=128,
        window="none",
        softcap=None,
        sink="learnable",
        padding=2,
        tokens=2,
    )
    _run_case(2, phase, case, dtype=torch.float16)


@pytest.mark.usefixtures("model_parallel")
@pytest.mark.sanitizer
@pytest.mark.parametrize(
    "version", [pytest.param(v, id=f"fa{v}", marks=_version_marks(v)) for v in (2, 3, 4)]
)
def test_small_batch_for_sanitizer(version):
    """A small poisoned prefill and decode (compute-sanitizer memcheck case)."""
    _require_flash(version)
    page = FA_PAGE_SIZES[version][0]
    module = _attention(version, kv_heads=2, head_dim=64)
    for phase, requests in (
        ("prefill", [(3, 3), (2, page + 1)]),
        ("decode", [(1, 1), (1, page + 1)]),
    ):
        out, q, batch, live = _run(
            module, phase=phase, requests=requests, padding=1, page=page, kv_heads=2, head_dim=64
        )
        _check(
            out,
            q,
            batch,
            live,
            head_dim=64,
            softcap=None,
            sink=None,
            window=None,
            what=f"FA{version} {phase} sanitizer case",
        )


@pytest.mark.usefixtures("model_parallel")
@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize(
    "version", [pytest.param(v, id=f"fa{v}", marks=_version_marks(v)) for v in (3, 4)]
)
def test_batch_invariant_rows_do_not_depend_on_the_batch(version, phase):
    """Batch-invariant mode: each request's rows are bitwise the same alone and in the batch."""
    _require_flash(version)
    page = FA_PAGE_SIZES[version][1]
    module = _attention(version, kv_heads=2, head_dim=128, batch_invariant=True)
    requests = _decode_requests(page, 1) if phase == "decode" else _prefill_requests(page)
    out, q, batch, live = _run(
        module,
        phase=phase,
        requests=requests,
        padding=2,
        page=page,
        kv_heads=2,
        head_dim=128,
        capacity_max_seqlen_k=True,
    )
    # The batch's max_seqlen_q and max_seqlen_k (block-table capacity) are kept, as a CUDA graph
    # keeps them, so only the co-batched requests change.
    capacity = batch.block_table.shape[1] * page
    max_seqlen_q = 1 if phase == "decode" else max(s for s, _ in requests)
    start = 0
    for r, (q_len, kv_len) in enumerate(requests):
        rows = slice(start, start + q_len)
        start += q_len
        q_alone = q[rows].clone()
        with torch.no_grad():
            alone = module.flash_decode_and_prefill(
                q_alone,
                batch.k,
                batch.v,
                max_seqlen_q,
                capacity,
                batch.cu([q_len]),
                batch.cu([kv_len]),
                batch.int_tensor([kv_len]),
                batch.block_table[r : r + 1],
                phase == "decode",
            )
        ou.assert_exact(alone[:q_len], out[rows], what=f"request {r} alone vs in the batch")


# FA3/FA4 pack the H / Hk query heads that share a KV head with a request's S query tokens into
# the rows of one M tile: 128 rows for FA4 on SM90 at head dim 128 and for FA4 on SM100, which
# also runs a two-stage 256-row tile once the packed rows pass 128. FA4 sizes its schedule from
# max_seqlen_q * H / Hk; on SM100 it gives each request a single tile while that product fits
# (flash_attn/cute/interface.py, max_m_blocks_leq_one and q_stage, read at 4.0.0b27 and b34). With
# MQA (Hk = 1) and 32 query heads, S = 3, 4, 5, 9 give 96, 128, 160 and 288 packed rows: below,
# at and past the 128-row tile, and past the 256-row one.
M_TILE_HEADS, M_TILE_KV_HEADS, M_TILE_HEAD_DIM = 32, 1, 128
M_TILE_TOKENS = (3, 4, 5, 9)


def _m_tile_cases():
    params = []
    for tag, version, batch_invariant in (
        ("fa2", 2, False),
        ("fa3", 3, False),
        ("fa4", 4, False),
        ("fa4bi", 4, True),
    ):
        for tokens in M_TILE_TOKENS:
            params.append(
                pytest.param(
                    version,
                    batch_invariant,
                    tokens,
                    id=f"{tag}-tokens={tokens}",
                    marks=_version_marks(version),
                )
            )
    return params


@pytest.mark.usefixtures("model_parallel")
@pytest.mark.parametrize("version, batch_invariant, tokens", _m_tile_cases())
def test_decode_query_rows_cross_the_m_tile(version, batch_invariant, tokens):
    """Multi-token decode (MTP, speculative) whose packed query rows ``S * H / Hk`` sit below, at
    and past the 128-row M tile (and FA4's 256-row two-stage tile): every query row of every
    request matches the fp64 reference, so the kernel must be told ``S`` query tokens per request
    (``max_seqlen_q``) and must compute every M tile of a request."""
    _require_flash(version)
    page = 256 if version == 2 else 128  # FA2 reads only 256-token pages
    module = _attention(
        version,
        kv_heads=M_TILE_KV_HEADS,
        head_dim=M_TILE_HEAD_DIM,
        batch_invariant=batch_invariant,
        heads=M_TILE_HEADS,
    )
    out, q, batch, live = _run(
        module,
        phase="decode",
        requests=_decode_requests(page, tokens),
        padding=1,
        page=page,
        kv_heads=M_TILE_KV_HEADS,
        head_dim=M_TILE_HEAD_DIM,
        tokens=tokens,
        heads=M_TILE_HEADS,
    )
    _check(
        out,
        q,
        batch,
        live,
        head_dim=M_TILE_HEAD_DIM,
        softcap=None,
        sink=None,
        window=None,
        what=f"FA{version} decode, {tokens} tokens x {M_TILE_HEADS} query heads per KV head",
    )


@pytest.mark.usefixtures("model_parallel")
@pytest.mark.large
@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize(
    "version", [pytest.param(v, id=f"fa{v}", marks=_version_marks(v)) for v in (2, 4)]
)
def test_pool_offsets_past_int32(version, phase):
    """The live pages sit past 2**31 elements into each pool tensor (production pools are tens
    of GB): page offsets must be computed in 64 bits."""
    _require_flash(version)
    page, kv_heads, head_dim = 256, 8, 128
    page_elems = page * kv_heads * head_dim
    offset = ou.INT32_LIMIT // page_elems + 1
    ou.require_free_memory(4 * (offset + 16) * page_elems * 2)  # K, V and their copies, bf16
    module = _attention(version, kv_heads=kv_heads, head_dim=head_dim)
    requests = [(3, 3), (2, page + 1)] if phase == "prefill" else [(1, 1), (1, page + 1)]
    g = ou.make_generator(device="cuda")
    q_lens = [s for s, _ in requests]
    kv_lens = [n for _, n in requests]
    batch = ar.make_paged_batch(
        q_lens,
        kv_lens,
        page_size=page,
        kv_heads=kv_heads,
        head_dim=head_dim,
        generator=g,
        live_page_offset=offset,
    )
    assert int(batch.block_table[:, 0].min()) * page_elems >= ou.INT32_LIMIT
    live = sum(q_lens)
    q = ou.poisoned_empty((live if phase == "decode" else 64, 1, HEADS, head_dim), torch.bfloat16)
    q[:live] = torch.randn(live, 1, HEADS, head_dim, device="cuda", generator=g).to(q.dtype)
    with torch.no_grad():
        out = module.flash_decode_and_prefill(
            q,
            batch.k,
            batch.v,
            max(q_lens),
            max(kv_lens),
            batch.cu(q_lens),
            batch.cu(kv_lens),
            batch.int_tensor(kv_lens),
            batch.block_table,
            phase == "decode",
        )
    batch.check_pool_unchanged()
    _check(
        out,
        q,
        batch,
        live,
        head_dim=head_dim,
        softcap=None,
        sink=None,
        window=None,
        what=f"FA{version} {phase}, pages past 2**31 elements",
    )


@pytest.mark.usefixtures("model_parallel")
@ou.requires_sm(ou.HOPPER)
def test_fa2_refuses_pages_it_cannot_read():
    """FA2 reads a paged KV cache only in pages of a multiple of 256 tokens. The call site must
    refuse other page sizes with a ValueError (#8584, n2), not run into flash-attn's own check.

    Fails on main: flash-attn raises a RuntimeError ("Paged KV cache block size must be divisible
    by 256"). Passes with n2."""
    _require_flash(2)
    module = _attention(2, kv_heads=2, head_dim=64)
    with pytest.raises(ValueError):
        _run(
            module,
            phase="prefill",
            requests=[(5, 5), (2, 70)],
            padding=0,
            page=64,
            kv_heads=2,
            head_dim=64,
        )


# ---------------------------------------------------------------------------------------------
# CPU checks of the reference itself (no kernel): it gathers through the block table and
# matches torch's fp64 SDPA where SDPA applies.
# ---------------------------------------------------------------------------------------------


def test_reference_gathers_rows_through_the_block_table():
    g = torch.Generator().manual_seed(0)
    batch = ar.make_paged_batch(
        [3, 2, 0],
        [130, 64, 0],
        page_size=64,
        kv_heads=1,
        head_dim=8,
        dtype=torch.float32,
        device="cpu",
        generator=g,
    )
    # Tag each pool row with (page, row) and check the gathered order.
    pages, rows = batch.k.shape[:2]
    tags = torch.arange(pages * rows, dtype=torch.float32).view(pages, rows, 1, 1)
    pool = tags.expand(pages, rows, 1, 8)
    for r, length in enumerate(batch.kv_lens):
        got = ar.gather_request_kv(pool, batch.block_table, r, length)[:, 0, 0]
        expected = [int(batch.block_table[r, j // rows]) * rows + j % rows for j in range(length)]
        assert got.tolist() == expected
    for r, length in enumerate(batch.kv_lens):
        k = ar.gather_request_kv(batch.k, batch.block_table, r, length)
        assert torch.isfinite(k).all(), "a live row was poisoned"


@pytest.mark.parametrize("window", [None, 5])
@pytest.mark.parametrize("kv_heads", [4, 1])
def test_reference_matches_torch_sdpa_in_fp64(kv_heads, window):
    g = torch.Generator().manual_seed(1)
    q_len, kv_len, heads, dim = 4, 11, 4, 16
    q = torch.randn(q_len, heads, dim, generator=g, dtype=torch.float64)
    k = torch.randn(kv_len, kv_heads, dim, generator=g, dtype=torch.float64)
    v = torch.randn(kv_len, kv_heads, dim, generator=g, dtype=torch.float64)
    allowed = ar.allowed_keys(q_len, kv_len, window, "cpu")
    out, lse = ar.attention_fp64(q, k, v, allowed, scale=0.3)
    group = heads // kv_heads
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1),
        k.repeat_interleave(group, 1).transpose(0, 1),
        v.repeat_interleave(group, 1).transpose(0, 1),
        attn_mask=allowed,
        scale=0.3,
    ).transpose(0, 1)
    torch.testing.assert_close(out, expected)
    # The sink adds exp(sink) to the denominator: out_sink = out * sigmoid(lse - sink).
    sink = torch.linspace(-1.0, 2.0, heads, dtype=torch.float64)
    out_sink, _ = ar.attention_fp64(q, k, v, allowed, scale=0.3, sink=sink)
    torch.testing.assert_close(out_sink, out * torch.sigmoid(lse - sink)[..., None])
    # Query i sits at position kv_len - q_len + i and sees min(pos + 1, W + 1) keys.
    pos = kv_len - q_len + torch.arange(q_len)
    visible = pos + 1 if window is None else torch.clamp(pos + 1, max=window + 1)
    assert allowed.sum(-1).tolist() == visible.tolist()
