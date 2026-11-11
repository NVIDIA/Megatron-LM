# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GPU tests of ``dsa_indexer_kpool_scorer_backend='cudnn'`` on the real cuDNN indexer kernel.

The fused selection (cuDNN indexer forward at ``ratio=4`` on the pooled keys, then the indexer
top-k) must pick the same pools as the PyTorch FP32 KPool scorer, except pools whose exact
scores tie the selection boundary, at GLM-5.3-Flash's indexer shape (32 heads x 128) on
causal zigzag CP layouts, including runs that start off the pool grid.

The selector tests call ``fused_qk_topk_kpool`` on seeded scoring inputs. The layer test runs a
real ``DSAttention`` layer (Transformer Engine indexer projections, cuDNN kernels) with both
scorers, so the kernel is reached through ``DSAttention.forward`` and the real dispatch, on the
tensors the layer's own indexer hands it.
"""

import os
from types import SimpleNamespace

import pytest
import torch

from megatron.core import utils as core_utils
from megatron.core.extensions.transformer_engine import HAVE_TE, TELinear, TENorm
from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.experimental_attention_variant import dsa as dsa_module
from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels, dsa_kernels
from megatron.core.transformer.forward_sharing import forward_sharing_lifetime
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import init_method_normal, scaled_init_method_normal
from tests.unit_tests.transformer.experimental_attention_variant import kpool_selection_utils as ks

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

_HEADS, _DIM, _TOPK = 32, 128, 2048
# The layer test: one sequence without CP. Rows from position 259 on choose 256 // 4 = 64 of up
# to 256 pools, far above the cuDNN top-k's small-K guard (at most K < 14, on BF16 scores),
# which has its own tests; earlier rows keep every visible pool.
_LAYER_ROWS, _LAYER_TOPK = 1024, 256
# Limits on the layer output over the rows that select the same tokens with both scorers. The
# scorers return a row's pools in different slot orders (the PyTorch scorer by score, the cuDNN
# backend in the top-k kernel's order). The fused sparse attention compacts the selected
# tokens and sorts them by index, so on that path such rows feed the kernel the same inputs
# and are expected to agree much more closely than the limits. The limits cover the unfused
# fallback, where the slot order changes the accumulation order of the sparse attention by a
# few BF16 ulps; a wrong selection moves whole rows by O(1). These are the production-mode
# limits of test_dsa_cudnn_skip_score_fill_layer for the same effect.
_LAYER_REL_L2 = 1e-2  # ||candidate - reference|| / ||reference||
_LAYER_PEAK = 2e-2  # max |candidate - reference| / max |reference|


@pytest.fixture(autouse=True)
def cudnn_dsa_and_exact_reference():
    """Skip without the cuDNN DSA frontend; score the PyTorch reference in true FP32.

    Each torchrun rank of the unit-test launcher runs these single-GPU tests on its own device.
    Container images may set ``TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1``, which would run the
    reference's FP32 matmuls as TF32.
    """
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("the cuDNN DSA indexer needs SM90+")
    try:
        dsa_cudnn_kernels._ensure_dsa_namespace()
    except ImportError as exc:
        pytest.skip(f"cuDNN frontend DSA unavailable: {exc}")
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        assert not torch.backends.cuda.matmul.allow_tf32
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def _compare(seq, cp_size, cp_rank, *, batch, hf_scoring, deterministic=False, config=None):
    x = ks.kpool_inputs(
        seq=seq,
        cp_size=cp_size,
        cp_rank=cp_rank,
        batch=batch,
        heads=_HEADS,
        dim=_DIM,
        device="cuda",
        seed=seq + cp_rank,
    )
    reference = ks.select(x, topk=_TOPK, hf_scoring=hf_scoring)
    calls = []
    selector = ks.cudnn_pool_topk_fn(
        ks.query_runs(seq, cp_size, cp_rank),
        deterministic=deterministic,
        config=config,
        calls=calls,
    )
    candidate = ks.select(x, topk=_TOPK, hf_scoring=hf_scoring, pool_topk_fn=selector)
    assert calls == [True], "the cuDNN KPool selector declined the call"
    differing = ks.assert_same_selection_up_to_boundary_ties(
        x, reference, candidate, topk=_TOPK, rel_tol=1e-5
    )
    rows = batch * (seq // cp_size)
    assert differing <= max(4, rows // 1000), f"{differing} of {rows} rows differ"
    return candidate


@pytest.mark.parametrize(
    "seq, cp_size, cp_rank",
    [
        (8192, 1, 0),  # no CP: one run from position 0
        (8193, 1, 0),  # more rows than ratio x complete pools: an invisible pool is padded
        (16384, 8, 0),  # zigzag: the first and last chunks
        (16384, 8, 5),  # zigzag: two middle chunks
        (8200, 4, 1),  # zigzag chunks of 1025 tokens start off the pool grid
    ],
)
@pytest.mark.parametrize("hf_scoring", [True, False])
@pytest.mark.parametrize("batch", [1, 2])
def test_cudnn_kpool_selection_matches_torch_scorer(seq, cp_size, cp_rank, hf_scoring, batch):
    if (seq // (2 * cp_size)) % 4 and not dsa_cudnn_kernels._supports_indexer_q_causal_offsets():
        # Off-grid runs need q_causal_offsets (cuDNN frontend >= 1.26); without it the call
        # keeps the PyTorch scorer, which the CPU tests check.
        pytest.skip("this cuDNN frontend has no q_causal_offsets")
    _compare(seq, cp_size, cp_rank, batch=batch, hf_scoring=hf_scoring)


def test_cudnn_kpool_selection_deterministic_mode_repeats():
    first = _compare(16384, 8, 3, batch=1, hf_scoring=True, deterministic=True)
    second = _compare(16384, 8, 3, batch=1, hf_scoring=True, deterministic=True)
    assert torch.equal(first, second)


def test_cudnn_kpool_selection_with_deepselect_top_k(monkeypatch):
    """K=512 pool selection on DeepSelect, where it is installed and the row is wide enough."""
    from megatron.core.transformer.experimental_attention_variant import dsa_deepselect_topk

    # CP64 rank 0 of 262,144 tokens: the back run's rows see up to 65,536 pools (2,048 rows),
    # DeepSelect's FP32 width threshold.
    reason = dsa_deepselect_topk.static_fallback_reason(
        n_rows=2048,
        width=65536,
        topk_k=_TOPK // ks.POOL,
        dtype=torch.float32,
        device=torch.device("cuda"),
        deterministic=False,
    )
    if reason is not None:
        pytest.skip(f"DeepSelect does not take this call here: {reason}")
    ran = []
    deepselect_topk = dsa_deepselect_topk.deepselect_topk

    def spy(*args, **kwargs):
        ran.append(True)
        return deepselect_topk(*args, **kwargs)

    monkeypatch.setattr(dsa_deepselect_topk, "deepselect_topk", spy)
    _compare(
        1048576 // 4,
        64,
        0,
        batch=1,
        hf_scoring=True,
        config=SimpleNamespace(dsa_indexer_topk_backend="deepselect"),
    )
    assert ran, "DeepSelect never ran"


@pytest.mark.parametrize("batch", [1, 2])
def test_cudnn_kpool_selection_across_row_chunks(monkeypatch, batch):
    """The 1M CP64 shape splits each run into row chunks; force that at a smaller shape."""
    monkeypatch.setattr(dsa_cudnn_kernels, "_indexer_score_chunk_rows", lambda b, sq, sk: 512)
    _compare(16384, 8, 3, batch=batch, hf_scoring=True)
    if dsa_cudnn_kernels._supports_indexer_q_causal_offsets():
        _compare(8200, 4, 1, batch=batch, hf_scoring=False)  # off-grid chunks


class _OneRankGroup:
    """Stands in for a process group of this rank alone: the layer needs no distributed state."""

    def size(self):
        return 1

    def rank(self):
        return 0


def _layer_config(hf_scoring):
    """The 64-head DSA geometry of test_dsa_cudnn_skip_score_fill_layer, with KPool selection.

    Built with the ``'cudnn'`` scorer, so the validation covers the combination under test.
    ``hf_scoring`` also selects the key-dtype pool compression (the profile HF scoring is used
    with); both profiles hand the kernel BF16 pooled keys. No activation rotation: the selector
    then scores the indexer's q and pooled keys as the KPool call receives them, which is what
    the FP64 tie check scores.
    """
    return MLATransformerConfig(
        multi_latent_attention=True,
        experimental_attention_variant="dsa",
        num_layers=4,
        hidden_size=2048,
        num_attention_heads=64,
        num_query_groups=64,
        q_lora_rank=2048,
        kv_lora_rank=512,
        qk_head_dim=192,
        qk_pos_emb_head_dim=64,
        v_head_dim=256,
        kv_channels=192,
        dsa_indexer_n_heads=_HEADS,
        dsa_indexer_head_dim=_DIM,
        dsa_indexer_topk=_LAYER_TOPK,
        dsa_indexer_topk_freq=4,
        dsa_indexer_skip_topk_offset=3,
        dsa_indexer_loss_coeff=0.0,
        dsa_indexer_use_sparse_loss=True,
        dsa_indexer_kpool=ks.POOL,
        dsa_indexer_kpool_scorer_backend="cudnn",
        dsa_indexer_kpool_key_dtype_compression=hf_scoring,
        dsa_indexer_hf_scoring=hf_scoring,
        dsa_indexer_scoring_relu=True,
        dsa_indexer_rotate_activation=False,
        dsa_indexer_weights_proj_use_quantization=False,
        dsa_indexer_weights_proj_output_dtype="bf16",
        dsa_indexer_k_norm_epsilon=1e-6,
        dsa_kernel_backend="cudnn",
        attention_backend=AttnBackend.auto,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        layernorm_epsilon=1e-6,
        normalization="RMSNorm",
        qk_layernorm=True,
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        cp_comm_type="allgather",
        apply_rope_fusion=False,
        rope_type="rope",
        rotary_base=10000,
        gradient_accumulation_fusion=False,
        init_method=init_method_normal(0.02),
        output_layer_init_method=scaled_init_method_normal(0.02, 1, multiplier=2.0),
        use_cpu_initialization=False,
        perform_initialization=True,
    )


def _layer(config):
    """A real ``DSAttention`` layer that selects its own top-k, with seeded matrices."""
    indexer_spec = ModuleSpec(
        module=dsa_module.DSAIndexer,
        submodules=dsa_module.DSAIndexerSubmodules(
            linear_wq_b=ModuleSpec(module=TELinear),
            linear_wk=ModuleSpec(module=TELinear),
            k_norm=ModuleSpec(module=TENorm),
            linear_weights_proj=ModuleSpec(module=TELinear),
        ),
    )
    attention = dsa_module.DSAttention(
        config=config,
        submodules=dsa_module.DSAttentionSubmodules(indexer=indexer_spec),
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        cp_comm_type="allgather",
        pg_collection=SimpleNamespace(tp=None, cp=_OneRankGroup()),
    ).cuda()
    assert not attention.skip_topk, "layer 1 must select its own top-k"
    # Every matrix from one host generator, the same on every rank and independent of the RNG
    # state earlier tests left; this also gives the KPool gate and its zero-initialized
    # positional term real values. The key norm's scale and shift keep their initialization.
    generator = torch.Generator(device="cpu").manual_seed(2026)
    with torch.no_grad():
        for parameter in attention.parameters():
            if parameter.ndim >= 2:
                parameter.copy_(0.02 * torch.randn(parameter.shape, generator=generator))
    return attention.eval()


def _layer_inputs(config):
    """Absorbed-MLA query/key and indexer activations of one sequence, BF16 as in production."""
    heads = config.num_attention_heads
    absorbed_dim = config.kv_lora_rank + config.qk_pos_emb_head_dim
    generator = torch.Generator(device="cpu").manual_seed(2027)

    def randn(*shape, scale=1.0):
        return (torch.randn(shape, generator=generator) * scale).to(torch.bfloat16).cuda()

    return {
        "query": randn(_LAYER_ROWS, 1, heads, absorbed_dim),
        "key": randn(_LAYER_ROWS, 1, 1, absorbed_dim),
        "x": randn(_LAYER_ROWS, 1, config.hidden_size, scale=8.0),
        "qr": randn(_LAYER_ROWS, 1, config.q_lora_rank, scale=8.0),
        "up_v_weight": randn(heads, config.v_head_dim, config.kv_lora_rank),
    }


def _layer_forward(attention, inputs):
    """One forward; returns the layer output and the top-k selection the layer published."""
    with torch.no_grad(), forward_sharing_lifetime(None, None, attention.config):
        output = attention(
            query=inputs["query"].clone(),
            key=inputs["key"].clone(),
            value=None,
            attention_mask=None,
            x=inputs["x"].clone(),
            qr=inputs["qr"].clone(),
            attn_mask_type=AttnMaskType.causal,
            packed_seq_params=None,
            up_v_weight=inputs["up_v_weight"].clone(),
        )
        payload = attention._get_dsa_index_sharing_payload(None, None)
        published = payload.topk_by_layer[attention.layer_number].clone()
    torch.cuda.synchronize()
    return output, published


@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
@pytest.mark.parametrize("hf_scoring", [True, False])
def test_cudnn_kpool_scorer_in_the_real_layer_matches_torch_scorer(monkeypatch, hf_scoring):
    """A real layer reaches the kernel through the real dispatch and selects what ``'torch'`` does.

    One layer runs the same inputs with ``dsa_indexer_kpool_scorer_backend`` ``'cudnn'`` and
    then ``'torch'`` (the field is read on every forward, so both runs share the weights). The
    dispatch is wrapped, not replaced: the cuDNN run must get a selection from it on every
    call, the ``'torch'`` run must not reach it. The published top-k tokens agree up to pools
    tying at a row's boundary, and the layer outputs agree on the rows with the same tokens.
    """
    missing = core_utils._missing_cudnn_dsa_kernel_dependencies()
    if missing:
        pytest.skip(f"cuDNN DSA kernel dependencies are missing: {', '.join(missing)}")
    kpool_calls, dispatched = [], []
    real_kpool = dsa_module.fused_qk_topk_kpool
    real_dispatch = dsa_kernels.run_kpool_pool_topk

    def kpool(*args, **kwargs):
        kpool_calls.append((args, kwargs))
        return real_kpool(*args, **kwargs)

    def dispatch(*args, **kwargs):
        result = real_dispatch(*args, **kwargs)
        dispatched.append(result is not None)
        return result

    monkeypatch.setattr(dsa_module, "fused_qk_topk_kpool", kpool)
    monkeypatch.setattr(dsa_kernels, "run_kpool_pool_topk", dispatch)

    config = _layer_config(hf_scoring)
    attention = _layer(config)
    inputs = _layer_inputs(config)

    candidate_output, candidate = _layer_forward(attention, inputs)
    assert len(kpool_calls) == 1, f"{len(kpool_calls)} KPool selections in one forward"
    args, kwargs = kpool_calls[0]
    q, k, weights, _index_topk, _pool_size, gate, ape = args
    received = {
        name: (str(tensor.dtype), tuple(tensor.shape))
        for name, tensor in (("q", q), ("k", k), ("weights", weights), ("gate", gate))
    }
    assert dispatched, f"the layer built no cuDNN pool selector for {received}"
    assert all(dispatched), f"the cuDNN KPool backend declined the layer's call: {received}"
    # What the real indexer hands the selector: BF16 q and keys, and the head weights in the
    # dtype of the scoring convention.
    assert tuple(q.shape) == (_LAYER_ROWS, 1, _HEADS, _DIM), received
    assert tuple(k.shape) == (_LAYER_ROWS, 1, _DIM), received
    assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16, received
    assert weights.dtype == (torch.float32 if hf_scoring else torch.bfloat16), received
    assert kwargs["hf_scoring"] is hf_scoring
    assert kwargs["kpool_key_dtype_compression"] is hf_scoring

    config.dsa_indexer_kpool_scorer_backend = "torch"
    reference_output, reference = _layer_forward(attention, inputs)
    assert len(kpool_calls) == 2, f"{len(kpool_calls) - 1} KPool selections in one forward"
    assert len(dispatched) == 1, "the 'torch' scorer reached the pool-selection dispatch"
    reference_args, reference_kwargs = kpool_calls[1]
    assert reference_kwargs.get("pool_topk_fn") is None
    # The tie check below scores the candidate run's tensors: the indexer must hand both runs
    # the same ones.
    for name, mine, theirs in zip(
        ("q", "k", "weights", "gate"),
        (q, k, weights, gate),
        reference_args[:3] + reference_args[5:6],
    ):
        assert torch.equal(mine, theirs), f"the indexer's {name} differs between the two runs"

    assert tuple(candidate.shape) == (1, _LAYER_ROWS, _LAYER_TOPK + ks.POOL - 1)
    # The published selections agree up to pools whose FP64 scores tie at a row's boundary.
    # Without HF scoring the weights carry head_dim**-0.5, which the HF formula of the check
    # applies to the logits; ReLU is positively homogeneous, so the scores are the same.
    weights_hf = weights.double() * (1.0 if hf_scoring else q.size(-1) ** 0.5)
    scoring = SimpleNamespace(
        q=q, k=k, gate=gate, ape=ape, weights_hf=weights_hf, ends=kwargs["varlen_ends"]
    )
    differing = ks.assert_same_selection_up_to_boundary_ties(
        scoring,
        reference,
        candidate,
        topk=_LAYER_TOPK,
        rel_tol=1e-5,
        kpool_key_dtype_compression=kwargs["kpool_key_dtype_compression"],
    )
    assert differing <= max(4, _LAYER_ROWS // 1000), f"{differing} of {_LAYER_ROWS} rows differ"

    # The layer output, on the rows whose selected tokens are the same (in any slot order).
    same = (reference.sort(dim=-1).values == candidate.sort(dim=-1).values).all(dim=-1)[0]
    assert int(same.sum()) == _LAYER_ROWS - differing
    assert candidate_output.shape == reference_output.shape
    assert candidate_output.size(0) == _LAYER_ROWS
    expected, actual = reference_output[same].double(), candidate_output[same].double()
    assert torch.isfinite(expected).all() and torch.isfinite(actual).all()
    diff = (actual - expected).abs()
    rel_l2 = float(diff.norm() / expected.norm())
    peak = float(diff.max() / expected.abs().max())
    assert rel_l2 <= _LAYER_REL_L2, f"layer output rel L2 {rel_l2:.3e} (limit {_LAYER_REL_L2:.0e})"
    assert peak <= _LAYER_PEAK, f"layer output peak {peak:.3e} (limit {_LAYER_PEAK:.0e})"
