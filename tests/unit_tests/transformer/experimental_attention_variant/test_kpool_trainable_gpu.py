# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GPU and distributed tests of the trainable KPool indexer (sparse pool-level KL).

A positive ``dsa_indexer_loss_coeff`` on a KPool layer trains the whole indexer from a KL
between the main attention's mass on each selected pool and the selector's own pool score;
the selection itself stays discrete and gradient-free. These tests run real ``DSAttention``
layers (Transformer Engine indexer projections) on CUDA:

- every indexer parameter (``linear_wq_b``, ``linear_wk``, ``k_norm``,
  ``linear_weights_proj``, the pool gate and the slot bias) gets a gradient and the trunk
  does not: loss on and loss off give bitwise-equal attention outputs and query, key and
  value-projection gradients;
- no fall-through: a KPool layer with the loss on never reaches a token-level loss hook;
- the trainable call selects bitwise what the frozen call selects, in both compression
  profiles, in training, in ``eval()`` and under ``no_grad``;
- the loss and the indexer gradients match the independent reference
  (``kpool_trainable_reference.py``);
- context parallelism (CP2, CP4) gives the CP1 loss and gradients, and pool-then-gather pools
  the keys bit for bit as gather-then-pool does;
- tensor parallelism (TP2): both ranks use TP rank 0's selection and compute one loss, also
  when their own selections disagree or tie at the selection boundary;
- a stack with IndexShare (``topk_freq=2``) and an MTP DSA layer, on one rank and under
  PP2: losses and tail masses logged only at computing layers (the MTP layer at
  ``layer_number + num_layers``), skip layers attending their source's published top-k;
- the runtime layout guards, a frozen checkpoint loaded into a trainable layer (in memory,
  and through ``dist_checkpointing`` with the distributed optimizer, with and without its
  state), the distributed optimizer with ``overlap_param_gather``, bitwise repeats at the 4K
  and 128K-rank shapes and a 4K-shape Transformer Engine graph capture.

The default layers run ``dsa_kernel_backend='none'`` (the PyTorch sparse attention) so they
run on any CUDA GPU; one test runs the cuDNN backend where it is installed. Multi-rank tests
skip below their world size. Launch from the Megatron-LM root, e.g.::

    torchrun --nproc-per-node 4 -m pytest -q \
        tests/unit_tests/transformer/experimental_attention_variant/test_kpool_trainable_gpu.py
"""

import functools
import os
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core import utils as core_utils
from megatron.core.extensions.transformer_engine import HAVE_TE, TELinear, TENorm
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.experimental_attention_variant import dsa as dsa_module
from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels, dsa_kernels
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
)
from megatron.core.transformer.forward_sharing import forward_sharing_lifetime
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import (
    PARAM_READY_CALLBACK_ATTR,
    init_method_normal,
    scaled_init_method_normal,
)
from tests.unit_tests.test_utilities import Utils

try:
    from tests.unit_tests.transformer.experimental_attention_variant import (
        kpool_trainable_reference as reference,
    )
except ImportError:
    reference = None

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available"),
]

_POOL = 4
_COEFF = 0.1
_INDEX_HEADS, _INDEX_DIM = 32, 128  # GLM-5.3-Flash's indexer: 32 heads x 128
# Small layers: 256 rows select 64 // 4 = 16 of up to 64 visible pools.
_ROWS, _TOPK = 256, 64
# CP: 512 rows; under CP4 each rank holds two zigzag halves of 64 rows (16 whole pools).
_CP_ROWS = 512
# Packed THD without CP: three documents whose lengths are not multiples of the pool size,
# so pools restart off the global grid and the packed capacity holds invalid pool slots.
_PACKED_CU_SEQLENS = (0, 37, 150, _ROWS)
_PADDED_ROWS = 21  # trailing query rows marked as padding in the padded packed case
# The 4K shapes of the GLM-5.3-Flash 4K recipe: 4096 rows choose 2048 // 4 = 512 pools.
_GRAPH_ROWS, _GRAPH_TOPK = 4096, 2048
# One rank of the 128K CP8 recipe holds 16,384 query rows (top-k 2048, as at 4K). The dense
# PyTorch sparse attention of that layer is O(rows^2 x heads), so it runs with four main
# heads: about 4.3 GB per FP32 score tensor instead of 17 GB with sixteen.
_RANK_128K_ROWS = 16384
_RANK_128K_MIN_GPU_GIB = 64
# The cuDNN-backend layer: the geometry of test_kpool_cudnn_selection_gpu's layer test.
_CUDNN_ROWS, _CUDNN_TOPK = 1024, 256

# Absorbed NoPE MLA, as GLM-5.3-Flash's main attention, at a reduced width.
_SMALL_GEOMETRY = dict(
    hidden_size=512,
    num_attention_heads=16,
    q_lora_rank=256,
    kv_lora_rank=128,
    qk_head_dim=64,
    qk_pos_emb_head_dim=0,
    v_head_dim=64,
    kv_channels=64,
)
# The small geometry with four main heads, for the 128K-rank determinism shape.
_RANK_128K_GEOMETRY = dict(_SMALL_GEOMETRY, num_attention_heads=4)
# FlashMLA's sparse attention takes absorbed rows of 512 + 64 channels.
_CUDNN_GEOMETRY = dict(
    hidden_size=2048,
    num_attention_heads=64,
    q_lora_rank=2048,
    kv_lora_rank=512,
    qk_head_dim=192,
    qk_pos_emb_head_dim=64,
    v_head_dim=256,
    kv_channels=192,
)

_TRUNK_INPUTS = ("query", "key", "up_v_weight")
_LAYER_INPUTS = ("query", "key", "x", "qr", "up_v_weight")
_INDEXER_OUTPUTS = ("q", "k", "weights", "gate_score")
_INDEXER_COMPONENTS = (
    "linear_wq_b",
    "linear_wk",
    "k_norm",
    "linear_weights_proj",
    "index_kpool_compress_gate",
    "index_kpool_compress_ape",
)
_GATE = "indexer.index_kpool_compress_gate"
_APE = "indexer.index_kpool_compress_ape"
_PROFILES = [
    pytest.param(True, True, id="hf-key-dtype"),
    pytest.param(False, False, id="legacy-fp32"),
]
_REFERENCE_PROFILE = {True: "key_dtype", False: "fp32"}
# Token-level loss and top-k hooks that a KPool layer must never reach.
_TOKEN_LEVEL_HOOKS = (
    "run_fused_dsa_attention",
    "run_fused_qk_topk",
    "run_fused_qk_topk_with_scores",
    "run_fused_qk_topk_with_loss",
    "run_sparse_indexer_loss_from_topk",
)


@pytest.fixture(autouse=True)
def exact_cuda_math():
    """Bind this rank's device; true FP32 GEMMs and FP32 reductions in BF16 GEMMs.

    Each torchrun rank runs the single-GPU tests on its own device. The reference comparisons
    need true FP32 (container images may enable TF32 through
    ``TORCH_ALLOW_TF32_CUBLAS_OVERRIDE``), and the CP and TP comparisons need GEMM results
    that do not depend on reduced-precision split-K reductions.
    """
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    previous = (
        torch.get_float32_matmul_precision(),
        torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
    )
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    try:
        assert not torch.backends.cuda.matmul.allow_tf32
        yield
    finally:
        torch.set_float32_matmul_precision(previous[0])
        torch.backends.cudnn.allow_tf32 = previous[1]
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous[2]


@pytest.fixture(autouse=True)
def unit_indexer_loss_scale(monkeypatch):
    """The indexer loss enters the backward with scale 1; the tracker starts and ends empty."""
    monkeypatch.setattr(
        DSAIndexerLossAutoScaler, "main_loss_backward_scale", torch.ones((), device="cuda")
    )
    DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
    yield
    DSAIndexerLossLoggingHelper.clean_loss_in_tracker()


# ---------------------------------------------------------------------------------------
# Layers, inputs and runs
# ---------------------------------------------------------------------------------------


class _OneRankGroup:
    """Stands in for a process group of this rank alone: the layer needs no collective."""

    def size(self):
        return 1

    def rank(self):
        return 0


def _single_rank_groups():
    return SimpleNamespace(tp=None, cp=_OneRankGroup())


def _require_ranks(world_size):
    if Utils.world_size < world_size or Utils.world_size % world_size:
        pytest.skip(f"needs a multiple of {world_size} distributed ranks")


def _require_gpu_memory_gib(gib):
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    if total < gib * 1024**3:
        pytest.skip(f"needs a GPU with at least {gib} GiB, this one has {total / 1024**3:.0f}")


def _require_cudnn_dsa():
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("the cuDNN DSA kernels need SM90+")
    missing = core_utils._missing_cudnn_dsa_kernel_dependencies()
    if missing:
        pytest.skip(f"cuDNN DSA kernel dependencies are missing: {', '.join(missing)}")
    try:
        dsa_cudnn_kernels._ensure_dsa_namespace()
    except ImportError as exc:
        pytest.skip(f"cuDNN frontend DSA unavailable: {exc}")


def _config(
    *,
    loss_coeff,
    hf_scoring=True,
    key_dtype_compression=True,
    tp=1,
    cp=1,
    topk=_TOPK,
    topk_freq=4,
    per_token=False,
    deterministic=False,
    kernel_backend="none",
    scorer="torch",
    geometry=None,
    num_layers=4,
    mtp_num_layers=None,
    skip_topk_offset=None,
    pp=1,
):
    """A KPool DSA layer config; ``loss_coeff > 0`` makes its indexer trainable.

    ``topk_freq > 1`` makes layer 1 a computing index-share layer, so it publishes the token
    top-k its sparse attention consumed. ``hf_scoring`` and ``key_dtype_compression`` pick
    the scoring convention and the pool compression profile independently. ``skip_topk_offset``
    defaults to 3 with index sharing (0 without); ``mtp_num_layers`` and ``pp`` describe the
    model a stack of layers belongs to.
    """
    geometry = dict(_SMALL_GEOMETRY if geometry is None else geometry)
    if skip_topk_offset is None:
        skip_topk_offset = 3 if topk_freq > 1 else 0
    config = MLATransformerConfig(
        multi_latent_attention=True,
        experimental_attention_variant="dsa",
        num_layers=num_layers,
        mtp_num_layers=mtp_num_layers,
        pipeline_model_parallel_size=pp,
        pipeline_dtype=torch.bfloat16 if pp > 1 else None,
        num_query_groups=geometry["num_attention_heads"],
        **geometry,
        dsa_indexer_n_heads=_INDEX_HEADS,
        dsa_indexer_head_dim=_INDEX_DIM,
        dsa_indexer_topk=topk,
        dsa_indexer_topk_freq=topk_freq,
        dsa_indexer_skip_topk_offset=skip_topk_offset,
        dsa_indexer_loss_coeff=loss_coeff,
        dsa_indexer_use_sparse_loss=True,
        dsa_indexer_kpool=_POOL,
        dsa_indexer_kpool_scorer_backend=scorer,
        dsa_indexer_kpool_key_dtype_compression=key_dtype_compression,
        dsa_indexer_hf_scoring=hf_scoring,
        dsa_indexer_scoring_relu=True,
        dsa_indexer_rotate_activation=False,
        dsa_indexer_weights_proj_use_quantization=False,
        dsa_indexer_weights_proj_output_dtype="bf16",
        dsa_indexer_k_norm_epsilon=1e-6,
        dsa_kernel_backend=kernel_backend,
        attention_backend=AttnBackend.auto,
        calculate_per_token_loss=per_token,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        layernorm_epsilon=1e-6,
        normalization="RMSNorm",
        qk_layernorm=True,
        tensor_model_parallel_size=tp,
        sequence_parallel=False,
        context_parallel_size=cp,
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
    # Set after validation, which otherwise asserts launcher-level determinism settings.
    config.deterministic_mode = deterministic
    return config


def _seed_parameters(module, seed):
    """Every matrix from one host generator: the same on every rank and for every build.

    This also gives the pool gate and the (zero-initialized) slot bias real values. The key
    norm's scale and shift keep their initialization.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            if parameter.ndim >= 2:
                parameter.copy_(0.02 * torch.randn(parameter.shape, generator=generator))


def _indexer_spec():
    """The Transformer Engine KPool indexer of the test layers."""
    return ModuleSpec(
        module=dsa_module.DSAIndexer,
        submodules=dsa_module.DSAIndexerSubmodules(
            linear_wq_b=ModuleSpec(module=TELinear),
            linear_wk=ModuleSpec(module=TELinear),
            k_norm=ModuleSpec(module=TENorm),
            linear_weights_proj=ModuleSpec(module=TELinear),
        ),
    )


def _build_layer(config, pg_collection, *, seed=2026):
    """A real ``DSAttention`` layer 1 (Transformer Engine indexer), in training mode."""
    attention = dsa_module.DSAttention(
        config=config,
        submodules=dsa_module.DSAttentionSubmodules(indexer=_indexer_spec()),
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        cp_comm_type="allgather",
        pg_collection=pg_collection,
    ).cuda()
    assert attention.indexer is not None and attention.indexer.index_kpool == _POOL
    _seed_parameters(attention, seed)
    return attention.train()


def _make_inputs(config, rows, *, batch, seed):
    """Absorbed-MLA query/key, indexer activations and an output gradient, BF16 as in production.

    Large indexer inputs keep the selected pool scores away from boundary ties.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    heads = config.num_attention_heads
    absorbed = config.kv_lora_rank + config.qk_pos_emb_head_dim

    def randn(*shape, scale=1.0):
        return (torch.randn(shape, generator=generator) * scale).to(torch.bfloat16).cuda()

    return {
        "query": randn(rows, batch, heads, absorbed),
        "key": randn(rows, batch, 1, absorbed),
        "x": randn(rows, batch, config.hidden_size, scale=8.0),
        "qr": randn(rows, batch, config.q_lora_rank, scale=8.0),
        "up_v_weight": randn(heads, config.v_head_dim, config.kv_lora_rank),
        "grad_output": randn(rows, batch, heads, config.v_head_dim),
    }


def _zigzag_rows(seq_len, cp_size, cp_rank):
    """The global rows one rank holds under all-gather CP: two zigzag halves."""
    chunk = seq_len // (2 * cp_size)
    return torch.cat(
        (
            torch.arange(cp_rank * chunk, (cp_rank + 1) * chunk),
            torch.arange((2 * cp_size - cp_rank - 1) * chunk, (2 * cp_size - cp_rank) * chunk),
        )
    )


def _slice_rows(inputs, rows):
    index = rows.to("cuda")
    local = dict(inputs)
    for name in ("query", "key", "x", "qr", "grad_output"):
        local[name] = inputs[name].index_select(0, index).contiguous()
    return local


def _slice_heads(inputs, tp_rank, tp_size):
    per_rank = inputs["query"].size(2) // tp_size
    heads = slice(tp_rank * per_rank, (tp_rank + 1) * per_rank)
    local = dict(inputs)
    local["query"] = inputs["query"][:, :, heads].contiguous()
    local["up_v_weight"] = inputs["up_v_weight"][heads].contiguous()
    local["grad_output"] = inputs["grad_output"][:, :, heads].contiguous()
    return local


def _packed_seq_params(cu_seqlens):
    """Packed THD metadata of one pack without CP (batch 1, every row a real token)."""
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device="cuda")
    longest = max(end - start for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]))
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=cu.clone(),
        cu_seqlens_kv_padded=cu.clone(),
        max_seqlen_q=longest,
        max_seqlen_kv=longest,
    )


def _run_layer(
    attention,
    inputs,
    *,
    packed_seq_params=None,
    attention_mask=None,
    attn_mask_type=AttnMaskType.causal,
    grad=True,
    backward=True,
):
    """One forward (and backward) on private leaf copies of ``inputs``.

    Returns the output, the token top-k the layer published (index-share layers), the
    gradients of the leaves (``x`` and ``qr`` included, which the layer must detach) and the
    parameter gradients.
    """
    attention.zero_grad(set_to_none=True)
    DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
    leaves = {name: inputs[name].detach().clone().requires_grad_(grad) for name in _LAYER_INPUTS}
    with forward_sharing_lifetime(packed_seq_params, attention_mask, attention.config):
        with nullcontext() if grad else torch.no_grad():
            output = attention(
                query=leaves["query"],
                key=leaves["key"],
                value=None,
                attention_mask=attention_mask,
                x=leaves["x"],
                qr=leaves["qr"],
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
                up_v_weight=leaves["up_v_weight"],
            )
        published = None
        if attention.index_share:
            payload = attention._get_dsa_index_sharing_payload(packed_seq_params, attention_mask)
            topk = payload.topk_by_layer.get(attention.layer_number)
            published = None if topk is None else topk.detach().clone()
    if grad and backward:
        output.backward(inputs["grad_output"].view_as(output))
    torch.cuda.synchronize()
    return SimpleNamespace(
        output=output.detach().clone(),
        published=published,
        input_grads={
            name: None if leaf.grad is None else leaf.grad.detach().clone()
            for name, leaf in leaves.items()
        },
        param_grads={
            name: parameter.grad.detach().clone()
            for name, parameter in attention.named_parameters()
            if parameter.grad is not None
        },
    )


def _install_recorders(monkeypatch, *, perturb=None):
    """Wrap, without replacing, the KPool selector, the loss tracker and the KPool loss.

    ``seen.kpool`` holds every ``fused_qk_topk_kpool`` call (arguments and the result the
    layer received; ``perturb(args, kwargs, result)`` may replace that result),
    ``seen.losses`` every tracker entry and ``seen.loss_function_calls`` the number of
    ``KPoolSparseIndexerLoss`` applications.
    """
    seen = SimpleNamespace(kpool=[], losses=[], loss_function_calls=0)
    real_kpool = dsa_module.fused_qk_topk_kpool

    def kpool(*args, **kwargs):
        result = real_kpool(*args, **kwargs)
        if perturb is not None:
            result = perturb(args, kwargs, result)
        seen.kpool.append(SimpleNamespace(args=args, kwargs=kwargs, result=result))
        return result

    monkeypatch.setattr(dsa_module, "fused_qk_topk_kpool", kpool)
    real_save = DSAIndexerLossLoggingHelper.save_loss_to_tracker

    def save_loss_to_tracker(*args, **kwargs):
        record = dict(kwargs)
        for name in ("loss", "raw_loss"):
            if isinstance(record.get(name), torch.Tensor):
                record[name] = record[name].detach().clone()
        seen.losses.append(record)
        return real_save(*args, **kwargs)

    monkeypatch.setattr(
        DSAIndexerLossLoggingHelper, "save_loss_to_tracker", staticmethod(save_loss_to_tracker)
    )
    loss_function = getattr(dsa_module, "KPoolSparseIndexerLoss", None)
    if loss_function is not None:
        real_apply = loss_function.apply

        def apply(*args, **kwargs):
            seen.loss_function_calls += 1
            return real_apply(*args, **kwargs)

        monkeypatch.setattr(loss_function, "apply", apply)
    return seen


def _layer_losses(seen, layer_number=1):
    return [record for record in seen.losses if record.get("layer_number") == layer_number]


def _keep_grad(store, name, grad):
    store[name] = grad.detach().clone()


def _spy_indexer(attention):
    """Record the indexer tensors of each forward and, through hooks, their gradients."""
    calls = []
    real = attention.indexer.forward_before_topk

    def forward_before_topk(*args, **kwargs):
        result = real(*args, **kwargs)
        record = SimpleNamespace(outputs=tuple(result), grads={})
        for name, tensor in zip(_INDEXER_OUTPUTS, record.outputs):
            if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                tensor.register_hook(functools.partial(_keep_grad, record.grads, name))
        calls.append(record)
        return result

    attention.indexer.forward_before_topk = forward_before_topk
    return calls


def _forbid_token_level_loss_hooks(monkeypatch):
    """Make every token-level loss (and token top-k) hook raise if a layer reaches it."""

    def forbidden(name):
        def hook(*_args, **_kwargs):
            raise AssertionError(f"a trainable KPool layer reached {name}")

        return hook

    for name in _TOKEN_LEVEL_HOOKS:
        if hasattr(dsa_kernels, name):
            monkeypatch.setattr(dsa_kernels, name, forbidden(f"dsa_kernels.{name}"))
    for name in ("run_sparse_indexer_loss_from_topk", "run_fused_qk_topk_with_loss"):
        if hasattr(dsa_cudnn_kernels, name):
            monkeypatch.setattr(dsa_cudnn_kernels, name, forbidden(f"dsa_cudnn_kernels.{name}"))

    class _ForbiddenTokenLoss:
        apply = staticmethod(forbidden("FusedDSAIndexerLoss"))

    monkeypatch.setattr(dsa_module, "FusedDSAIndexerLoss", _ForbiddenTokenLoss)
    if hasattr(dsa_module, "compute_dsa_indexer_loss"):
        monkeypatch.setattr(
            dsa_module, "compute_dsa_indexer_loss", forbidden("compute_dsa_indexer_loss")
        )


def _call_arg(call, index, name):
    return call.kwargs[name] if name in call.kwargs else call.args[index]


def _is_trainable_call(call):
    return bool(call.kwargs.get("return_selection")) and call.kwargs.get("pooled_keys") is not None


def _all_gather(tensor, group=None):
    tensor = tensor.contiguous()
    gathered = [torch.empty_like(tensor) for _ in range(torch.distributed.get_world_size(group))]
    torch.distributed.all_gather(gathered, tensor, group=group)
    return gathered


def _assert_similar(actual, expected, *, label, eps):
    """Cosine similarity and norm ratio within ``eps`` of 1 (an exactly zero reference: zero)."""
    actual = actual.detach().double().cpu().flatten()
    expected = expected.detach().double().cpu().flatten()
    assert actual.shape == expected.shape, f"{label}: {actual.shape} != {expected.shape}"
    assert torch.isfinite(actual).all(), f"{label} has non-finite values"
    expected_norm = float(expected.norm())
    if expected_norm == 0:
        assert float(actual.abs().max()) == 0, f"{label}: expected zero, got {actual.abs().max()}"
        return
    cosine = float(F.cosine_similarity(actual, expected, dim=0))
    ratio = float(actual.norm()) / expected_norm
    assert cosine > 1 - eps, f"{label}: cosine {cosine:.8f}"
    assert abs(ratio - 1) < eps, f"{label}: norm ratio {ratio:.8f}"


def _assert_rank_sum_close(actual, magnitude, expected, *, label, rel):
    """A mean of per-rank gradients against one reference gradient.

    Each rank rounds its BF16 gradient before the cross-rank sum, so the error scales with
    the ranks' gradient magnitudes (``magnitude``: the mean of their absolute values), not
    with the result, which can cancel (a norm's shift, for instance). A missing or doubled
    rank contribution is an error of that same scale.
    """
    actual = actual.detach().double().cpu().flatten()
    magnitude = magnitude.detach().double().cpu().flatten()
    expected = expected.detach().double().cpu().flatten()
    assert actual.shape == expected.shape, f"{label}: {actual.shape} != {expected.shape}"
    assert torch.isfinite(actual).all(), f"{label} has non-finite values"
    error = float((actual - expected).norm())
    scale = float(magnitude.norm()) + float(expected.norm())
    assert error <= rel * scale, f"{label}: error {error:.3e}, scale {scale:.3e}"


def _assert_indexer_trained(attention, run):
    """Every parameter is the indexer's, trainable, and got a finite gradient (zeros allowed)."""
    names = []
    for name, parameter in attention.named_parameters():
        names.append(name)
        assert name.startswith("indexer."), f"unexpected parameter {name} outside the indexer"
        assert parameter.requires_grad, f"{name} is frozen in a trainable layer"
        assert name in run.param_grads, f"{name} got no gradient"
        assert torch.isfinite(run.param_grads[name]).all(), f"{name} has a non-finite gradient"
    for component in _INDEXER_COMPONENTS:
        assert any(
            name.startswith(f"indexer.{component}") for name in names
        ), f"the indexer has no {component} parameter"
    total = sum(float(grad.float().norm()) for grad in run.param_grads.values())
    assert total > 0, "every indexer gradient is zero"


@contextmanager
def _deterministic_algorithms(enabled):
    if not enabled:
        yield
        return
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])


# ---------------------------------------------------------------------------------------
# The reference loss: kpool_trainable_reference.py, or a dense restatement of DESIGN §1
# ---------------------------------------------------------------------------------------


def _document_starts(seq_len, cu_seqlens=None):
    """The first token of each row's document, int64 ``[seq_len]``."""
    starts = torch.zeros(seq_len, dtype=torch.int64)
    if cu_seqlens is not None:
        for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
            starts[start:end] = start
    return starts


def _pool_token_base(seq_len, pool_size, cu_seqlens=None):
    """Each pool slot's first token, ``-1`` for unused packed capacity, int64 ``[slots]``."""
    slots = seq_len // pool_size
    if cu_seqlens is None:
        return torch.arange(slots, dtype=torch.int64) * pool_size
    bases = []
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        bases.extend(range(start, end - pool_size + 1, pool_size))
    return torch.tensor(bases + [-1] * (slots - len(bases)), dtype=torch.int64)


def _expected_tail(seq_len, pool_size, cu_seqlens=None):
    """Each row's query-local tail: its ``(t - s + 1) % P`` newest tokens, ``-1`` padded."""
    rows = torch.arange(seq_len)
    count = (rows - _document_starts(seq_len, cu_seqlens) + 1) % pool_size
    offsets = torch.arange(pool_size - 1)
    tail = rows[:, None] - count[:, None] + 1 + offsets
    return torch.where(offsets < count[:, None], tail, torch.full_like(tail, -1))


def _pool_ids_from_tokens(token_topk, base, *, index_topk):
    """Pool slot ids ``[b, sq, G]`` of a published token top-k, checking its pool layout."""
    seq_len = token_topk.size(1)
    tokens = token_topk[..., :index_topk].detach().to(device="cpu", dtype=torch.int64)
    batch, rows, _ = tokens.shape
    slots = tokens.reshape(batch, rows, index_topk // _POOL, _POOL)
    first = slots[..., 0]
    valid = first >= 0
    expected = torch.where(
        valid.unsqueeze(-1), first.unsqueeze(-1) + torch.arange(_POOL), torch.full_like(slots, -1)
    )
    assert torch.equal(slots, expected), "a pool slot does not hold P consecutive tokens"
    lookup = torch.full((seq_len,), -1, dtype=torch.int64)
    lookup[base[base >= 0]] = torch.nonzero(base >= 0).flatten()
    ids = torch.where(valid, lookup[first.clamp(0, seq_len - 1)], torch.full_like(first, -1))
    assert bool((ids[valid] >= 0).all()), "a selected pool does not start on the pool grid"
    return ids


def _dense_sparse_indexer_loss(
    *,
    q,
    weights,
    k,
    gate_score,
    ape,
    query,
    key,
    pool_ids,
    pool_size,
    softmax_scale,
    hf_scoring,
    use_relu,
    profile,
    loss_coeff,
    row_count,
    cu_seqlens=None,
    row_mask=None,
    dtype=torch.float64,
):
    """DESIGN §1 restated with dense tensors, for runs without ``kpool_trainable_reference``.

    Same keywords and semantics as its ``sparse_indexer_loss``: the teacher is the absolute
    head-summed main-attention mass of each selected pool, with the tail in every head's
    denominator and an exact-zero safe L1 normalization; the student is the selector's pool
    score; the loss is ``coeff * sum_rows KL / row_count``. Differentiable in ``q``,
    ``weights``, ``k``, ``gate_score`` and ``ape``.
    """
    seq_len, batch, _heads, dim = q.shape
    base = _pool_token_base(seq_len, pool_size, cu_seqlens)
    offsets = torch.arange(pool_size)

    # Compression of every pool slot, [slots, batch, D]; unused capacity slots are zero.
    positions = (base.clamp_min(0)[:, None] + offsets).clamp_max(seq_len - 1)
    keys, gates = k[positions], gate_score[positions]
    if profile == "exact":
        logits = gates.to(dtype) + ape.to(dtype)[None, :, None]
        pooled = (logits.softmax(dim=1) * keys.to(dtype)).sum(dim=1)
    elif profile == "key_dtype":
        logits = gates.float() + ape.float()[None, :, None]
        probabilities = torch.nan_to_num(logits.softmax(dim=1)).to(keys.dtype)
        pooled = (probabilities * keys).sum(dim=1)
    elif profile == "fp32":
        logits = gates.float() + ape.float()[None, :, None]
        shifted = torch.exp(logits - logits.amax(dim=1, keepdim=True).detach())
        pooled = (shifted * keys.float()).sum(dim=1) / shifted.sum(dim=1).clamp_min(1e-12)
        pooled = pooled.to(torch.bfloat16)
    else:
        raise ValueError(f"unknown compression profile {profile!r}")
    pooled = torch.where((base >= 0).view(-1, 1, 1), pooled, torch.zeros_like(pooled))

    # Student: the selector's score of each selected pool.
    valid = pool_ids >= 0
    batch_index = torch.arange(batch).view(batch, 1, 1)
    selected = pooled.to(dtype).permute(1, 0, 2)[batch_index, pool_ids.clamp_min(0)]
    dots = torch.einsum("sbhd,bsgd->bshg", q.to(dtype), selected)
    if hf_scoring:
        dots = dots * dim**-0.5
    if use_relu:
        dots = torch.relu(dots)
    scores = torch.einsum("sbh,bshg->bsg", weights.to(dtype), dots)

    # Teacher, from the detached main query and key over the attended tokens A(t).
    with torch.no_grad():
        rows = torch.arange(seq_len)
        pool_tokens = (base[pool_ids.clamp_min(0)].unsqueeze(-1) + offsets).flatten(-2)
        pool_valid = valid.unsqueeze(-1).expand(-1, -1, -1, pool_size).flatten(-2)
        tail = _expected_tail(seq_len, pool_size, cu_seqlens)
        tokens = torch.cat((pool_tokens, tail.expand(batch, -1, -1)), dim=-1)
        attended = torch.cat((pool_valid, (tail >= 0).expand(batch, -1, -1)), dim=-1)
        main_query = query.detach().to(dtype)
        main_key = key.detach().to(dtype)
        if main_key.size(2) == 1:
            main_key = main_key.expand(-1, -1, main_query.size(2), -1)
        logits = softmax_scale * torch.einsum("sbhd,jbhd->bshj", main_query, main_key)
        index = tokens.clamp(0, seq_len - 1).unsqueeze(2).expand(-1, -1, logits.size(2), -1)
        picked = logits.gather(-1, index).masked_fill(~attended.unsqueeze(2), float("-inf"))
        lse = torch.logsumexp(picked, dim=-1, keepdim=True)
        probabilities = torch.where(
            attended.unsqueeze(2), torch.exp(picked - lse), torch.zeros_like(picked)
        )
        budget = pool_ids.size(-1)
        mass = probabilities.sum(dim=2)[..., : budget * pool_size]
        pool_mass = mass.reshape(batch, seq_len, budget, pool_size).sum(-1).masked_fill(~valid, 0)
        row_mass = pool_mass.sum(dim=-1, keepdim=True)
        target = pool_mass / torch.where(row_mass > 0, row_mass, torch.ones_like(row_mass))
        has_mass = row_mass.squeeze(-1) > 0
        if row_mask is not None:
            has_mass = has_mass & row_mask
        del rows

    # Forward KL over the valid selected slots.
    any_valid = valid.any(dim=-1, keepdim=True)
    masked = scores.masked_fill(~valid, float("-inf"))
    masked = torch.where(any_valid, masked, torch.zeros_like(masked))
    log_predict = torch.log_softmax(masked, dim=-1).masked_fill(~valid, 0.0)
    positive = valid & (target > 0)
    safe_target = torch.where(positive, target, torch.ones_like(target))
    terms = torch.where(
        positive, target * (torch.log(safe_target) - log_predict), torch.zeros_like(target)
    )
    per_row = terms.sum(dim=-1) * has_mass.to(dtype)
    if isinstance(row_count, torch.Tensor):
        row_count = row_count.to(dtype)
    return SimpleNamespace(loss=loss_coeff * per_row.sum() / row_count)


def _oracle(**kwargs):
    """``kpool_trainable_reference.sparse_indexer_loss``, or the dense restatement above."""
    function = getattr(reference, "sparse_indexer_loss", None) if reference is not None else None
    if function is None:
        function = _dense_sparse_indexer_loss
    return function(**kwargs)


def _reference_case(attention, inputs, run, indexer_call, *, cu_seqlens=None):
    """The reference's inputs for one layer run, on the host.

    The indexer tensors are the ones the layer's own indexer produced; the selection is the
    token top-k the layer published (and its sparse attention consumed).
    """
    config = attention.config
    q, k, weights = indexer_call.outputs[:3]
    gate_score = indexer_call.outputs[3] if len(indexer_call.outputs) > 3 else None
    if gate_score is None:
        with torch.no_grad():
            gate_score = F.linear(inputs["x"], attention.indexer.index_kpool_compress_gate)
    seq_len = k.size(0)
    tail = run.published[..., config.dsa_indexer_topk :].to(device="cpu", dtype=torch.int64)
    expected_tail = _expected_tail(seq_len, _POOL, cu_seqlens).expand_as(tail)
    assert torch.equal(tail, expected_tail), "the published tail is not each row's query tail"
    base = _pool_token_base(seq_len, _POOL, cu_seqlens)
    pool_ids = _pool_ids_from_tokens(run.published, base, index_topk=config.dsa_indexer_topk)
    tensors = dict(
        q=q.detach().cpu(),
        weights=weights.detach().cpu(),
        k=k.detach().cpu(),
        gate_score=gate_score.detach().cpu(),
        ape=attention.indexer.index_kpool_compress_ape.detach().cpu(),
    )
    fixed = dict(
        query=inputs["query"].detach().cpu(),
        key=inputs["key"].detach().cpu(),
        pool_ids=pool_ids,
        pool_size=_POOL,
        softmax_scale=attention.softmax_scale,
        hf_scoring=config.dsa_indexer_hf_scoring,
        use_relu=config.dsa_indexer_scoring_relu,
        cu_seqlens=None if cu_seqlens is None else list(cu_seqlens),
    )
    return SimpleNamespace(tensors=tensors, fixed=fixed, x=inputs["x"].detach().cpu())


# ---------------------------------------------------------------------------------------
# Single GPU: training, isolation, selection and the reference
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(("hf_scoring", "key_dtype_compression"), _PROFILES)
def test_trainable_layer_trains_the_whole_indexer_and_leaves_the_trunk_unchanged(
    monkeypatch, hf_scoring, key_dtype_compression
):
    """Loss on: every indexer parameter gets a gradient; the trunk sees no difference.

    The same layer with the loss off (coefficient 0, and unset) is today's frozen selector:
    no indexer parameter is trainable or gets a gradient. The attention output and the
    gradients of the main query, key and value projection are bitwise equal with the loss on
    and off, and neither run sends a gradient to ``x`` or ``qr``.
    """
    assert hasattr(dsa_module, "KPoolSparseIndexerLoss"), "the KPool loss Function is missing"
    seen = _install_recorders(monkeypatch)
    profile = dict(hf_scoring=hf_scoring, key_dtype_compression=key_dtype_compression)
    trainable = _build_layer(_config(loss_coeff=_COEFF, **profile), _single_rank_groups())
    inputs = _make_inputs(trainable.config, _ROWS, batch=2, seed=3)

    on = _run_layer(trainable, inputs)
    losses = _layer_losses(seen)
    assert len(losses) == 1, f"{len(losses)} indexer losses logged for one forward"
    loss = losses[0]["loss"]
    assert torch.isfinite(loss) and float(loss) > 0, f"indexer loss {float(loss)}"
    assert seen.loss_function_calls == 1
    _assert_indexer_trained(trainable, on)

    for frozen_coeff in (0.0, None):
        seen.losses.clear()
        frozen = _build_layer(_config(loss_coeff=frozen_coeff, **profile), _single_rank_groups())
        off = _run_layer(frozen, inputs)
        assert not _layer_losses(seen), f"coefficient {frozen_coeff} logged an indexer loss"
        for name, parameter in frozen.named_parameters():
            assert not parameter.requires_grad, f"frozen {name} requires grad"
        assert off.param_grads == {}, f"a frozen indexer got gradients: {sorted(off.param_grads)}"
        assert torch.equal(on.published, off.published), "loss on and off select differently"
        assert torch.equal(on.output, off.output), "the indexer loss changed the attention output"
        for name in _TRUNK_INPUTS:
            assert torch.equal(on.input_grads[name], off.input_grads[name]), f"d{name} differs"
        for name in ("x", "qr"):
            assert on.input_grads[name] is None, f"the indexer loss reached {name}"
            assert off.input_grads[name] is None, f"the frozen layer reached {name}"
    assert seen.loss_function_calls == 1


@pytest.mark.parametrize("layout", ["plain", "packed"])
def test_kpool_loss_never_reaches_the_token_level_loss_hooks(monkeypatch, layout):
    """The fused token-level hooks are eligible, yet a trainable KPool layer never calls them.

    The fused DSA dispatch is enabled (the configured backend then declines every hook, so
    the PyTorch sparse attention still runs) and every token-level loss and top-k hook, and
    the reference ``FusedDSAIndexerLoss``, raises when reached. The loss comes from
    ``KPoolSparseIndexerLoss`` alone.
    """
    seen = _install_recorders(monkeypatch)
    trainable = _build_layer(_config(loss_coeff=_COEFF), _single_rank_groups())
    monkeypatch.setattr(dsa_kernels, "use_fused_dsa_kernels", lambda _config: True)
    _forbid_token_level_loss_hooks(monkeypatch)
    batch = 2 if layout == "plain" else 1
    inputs = _make_inputs(trainable.config, _ROWS, batch=batch, seed=5)
    packed = _packed_seq_params(_PACKED_CU_SEQLENS) if layout == "packed" else None

    run = _run_layer(trainable, inputs, packed_seq_params=packed)
    assert len(_layer_losses(seen)) == 1
    assert seen.loss_function_calls == 1
    assert [_is_trainable_call(call) for call in seen.kpool] == [True]
    _assert_indexer_trained(trainable, run)


@pytest.mark.parametrize("layout", ["plain", "packed"])
@pytest.mark.parametrize("key_dtype_compression", [True, False], ids=["key-dtype", "fp32"])
@pytest.mark.parametrize("hf_scoring", [True, False], ids=["hf", "legacy"])
def test_trainable_selection_equals_frozen_selection_in_train_eval_and_no_grad(
    monkeypatch, hf_scoring, key_dtype_compression, layout
):
    """One selection, whatever the mode: trainable (train, ``no_grad``, ``eval``) and frozen.

    The trainable layer selects through its own path (pooled keys built outside the
    selector, the selected pool ids returned) in every forward; only the training forward
    computes the loss. Its published token top-k equals the frozen layer's bit for bit.
    """
    seen = _install_recorders(monkeypatch)
    profile = dict(hf_scoring=hf_scoring, key_dtype_compression=key_dtype_compression)
    trainable = _build_layer(_config(loss_coeff=_COEFF, **profile), _single_rank_groups())
    frozen = _build_layer(_config(loss_coeff=0.0, **profile), _single_rank_groups())
    batch = 2 if layout == "plain" else 1
    inputs = _make_inputs(trainable.config, _ROWS, batch=batch, seed=7)
    cu_seqlens = _PACKED_CU_SEQLENS if layout == "packed" else None
    packed = _packed_seq_params(cu_seqlens) if cu_seqlens is not None else None

    runs = {}
    for name, layer, grad, training in (
        ("trainable-train", trainable, True, True),
        ("trainable-no-grad", trainable, False, True),
        ("trainable-eval", trainable, True, False),
        ("trainable-eval-no-grad", trainable, False, False),
        ("frozen-train", frozen, True, True),
        ("frozen-eval-no-grad", frozen, False, False),
    ):
        layer.train(training)
        seen.kpool.clear()
        seen.losses.clear()
        run = _run_layer(layer, inputs, packed_seq_params=packed, grad=grad, backward=False)
        assert len(seen.kpool) == 1, f"{name}: {len(seen.kpool)} KPool selections"
        runs[name] = SimpleNamespace(
            published=run.published, call=seen.kpool[0], losses=len(_layer_losses(seen))
        )

    assert runs["trainable-train"].losses == 1
    for name, run in runs.items():
        if name != "trainable-train":
            assert run.losses == 0, f"{name} logged an indexer loss"
        assert torch.equal(run.published, runs["frozen-train"].published), f"{name} differs"
        call = run.call
        if name.startswith("frozen"):
            assert not call.kwargs.get("return_selection"), f"{name}: frozen call changed"
            assert call.kwargs.get("pooled_keys") is None, f"{name}: frozen call changed"
            continue
        assert _is_trainable_call(call), f"{name} did not take the trainable selection path"
        q, weights = _call_arg(call, 0, "q"), _call_arg(call, 2, "weights")
        pooled, base = call.kwargs["pooled_keys"]
        for label, tensor in (("q", q), ("weights", weights), ("pooled keys", pooled)):
            assert not tensor.requires_grad, f"{name}: the selector received a graph on {label}"
        # The third value: the selected pool ids, in the slot order of the token top-k.
        assert len(call.result) == 3
        _, tokens, pool_ids = call.result
        assert pool_ids.dtype == torch.int64
        assert tuple(pool_ids.shape) == (batch, _ROWS, _TOPK // _POOL)
        expected_base = _pool_token_base(_ROWS, _POOL, cu_seqlens)
        assert torch.equal(base.cpu(), expected_base), f"{name}: pool_token_base differs"
        starts = expected_base[pool_ids.clamp_min(0).cpu()]
        expanded = starts.unsqueeze(-1) + torch.arange(_POOL)
        expanded = torch.where((pool_ids >= 0).cpu().unsqueeze(-1), expanded, -1).flatten(-2)
        assert torch.equal(tokens[..., :_TOPK].cpu().long(), expanded), f"{name}: pool ids"


@pytest.mark.parametrize("layout", ["plain", "packed", "packed-padded"])
@pytest.mark.parametrize("per_token", [False, True], ids=["mean", "per-token"])
@pytest.mark.parametrize(("hf_scoring", "key_dtype_compression"), _PROFILES)
def test_trainable_loss_and_indexer_gradients_match_the_reference(
    monkeypatch, hf_scoring, key_dtype_compression, per_token, layout
):
    """The logged loss and the indexer gradients match the independent reference.

    The reference rebuilds the loss from the indexer's own tensors and the selection the
    layer published, in the rounding profile of the layer's compression. The loss and, by
    autograd through the reference, the gradients of ``q``, ``weights``, ``k``, the gate
    score, the pool gate and the slot bias must match. Rows without a pool, rows with fewer
    pools than the budget and, packed, invalid capacity pools and padded rows are included.
    """
    seen = _install_recorders(monkeypatch)
    profile = dict(hf_scoring=hf_scoring, key_dtype_compression=key_dtype_compression)
    config = _config(loss_coeff=_COEFF, per_token=per_token, **profile)
    trainable = _build_layer(config, _single_rank_groups())
    indexer_calls = _spy_indexer(trainable)
    batch = 2 if layout == "plain" else 1
    inputs = _make_inputs(config, _ROWS, batch=batch, seed=11)
    cu_seqlens = packed = row_mask = None
    if layout != "plain":
        cu_seqlens = _PACKED_CU_SEQLENS
        packed = _packed_seq_params(cu_seqlens)
    if layout == "packed-padded":
        row_mask = torch.ones(1, _ROWS, dtype=torch.bool)
        row_mask[:, -_PADDED_ROWS:] = False
        packed.real_token_mask_q = row_mask.cuda()

    run = _run_layer(trainable, inputs, packed_seq_params=packed)
    (record,) = _layer_losses(seen)
    (indexer_call,) = indexer_calls
    _assert_indexer_trained(trainable, run)
    torch.testing.assert_close(record["raw_loss"] * _COEFF, record["loss"], rtol=1e-6, atol=0)

    if per_token:
        row_count = 1.0
    elif row_mask is not None:
        row_count = row_mask.sum().double()
    else:
        row_count = float(batch * _ROWS)
    case = _reference_case(trainable, inputs, run, indexer_call, cu_seqlens=cu_seqlens)
    common = dict(case.fixed, loss_coeff=_COEFF, row_count=row_count, row_mask=row_mask)
    expected = _oracle(**case.tensors, **common, profile=_REFERENCE_PROFILE[key_dtype_compression])
    torch.testing.assert_close(
        record["loss"].double().cpu(), expected.loss.detach().double(), rtol=1e-4, atol=1e-8
    )

    # The gradient oracle differentiates the layer's own rounding profile: FP64 leaves, except
    # that the key-dtype profile keeps the BF16 keys and gate scores, whose dtype it rounds
    # the pool probabilities and products to.
    key_dtype_leaves = ("k", "gate_score") if key_dtype_compression else ()
    leaves = {
        name: (tensor.clone() if name in key_dtype_leaves else tensor.double()).requires_grad_()
        for name, tensor in case.tensors.items()
    }
    oracle = _oracle(**leaves, **common, profile=_REFERENCE_PROFILE[key_dtype_compression])
    oracle.loss.backward()
    expected = {name: leaf.grad.double() for name, leaf in leaves.items()}
    for name in ("q", "weights", "k", "gate_score"):
        assert name in indexer_call.grads, f"no gradient reached the indexer's {name}"
        _assert_similar(indexer_call.grads[name], expected[name], label=f"d{name}", eps=1e-3)
    expected_gate = torch.einsum("sbd,sbh->dh", expected["gate_score"], case.x.double())
    _assert_similar(run.param_grads[_GATE], expected_gate, label="dgate", eps=1e-3)
    _assert_similar(run.param_grads[_APE], expected["ape"], label="dape", eps=1e-3)


def test_trainable_explicit_mask_raises_and_frozen_accepts_it(monkeypatch):
    """An explicit float mask is outside the trainable path; the frozen selector keeps it."""
    rows, batch = 64, 1
    inputs = _make_inputs(_config(loss_coeff=0.0), rows, batch=batch, seed=13)
    masked = ~torch.ones(rows, rows, dtype=torch.bool, device="cuda").tril()
    attention_mask = masked.view(1, 1, rows, rows).expand(batch, 1, rows, rows).contiguous()
    frozen = _build_layer(_config(loss_coeff=0.0), _single_rank_groups())
    _run_layer(frozen, inputs, attention_mask=attention_mask, attn_mask_type=None)
    trainable = _build_layer(_config(loss_coeff=_COEFF), _single_rank_groups())
    with pytest.raises(ValueError, match="explicit attention mask"):
        _run_layer(trainable, inputs, attention_mask=attention_mask, attn_mask_type=None)


def test_frozen_checkpoint_loads_into_a_trainable_layer(monkeypatch):
    """Model keys are the same with the loss on and off; a frozen checkpoint trains on load.

    The plain and the sharded state dict have the same keys, shapes and dtypes in both
    modes. Only the parameter flags differ: every indexer parameter is trainable (TP-averaged
    gradients; the slot bias in FP32, outside Muon) or frozen. A frozen state dict loads
    strictly into a trainable layer, which then trains; a trainable one loads into a frozen
    layer, which stays frozen.
    """
    from megatron.core.dist_checkpointing.mapping import ShardedTensor

    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
    try:
        model_parallel_cuda_manual_seed(1234)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        frozen = _build_layer(_config(loss_coeff=0.0), pg_collection, seed=2026)
        trainable = _build_layer(_config(loss_coeff=_COEFF), pg_collection, seed=7)
        with torch.no_grad():
            for parameter in trainable.parameters():
                if parameter.ndim < 2:
                    parameter.add_(0.5)  # every tensor differs before the load

        for name, parameter in frozen.named_parameters():
            assert not parameter.requires_grad, f"frozen {name} requires grad"
        for name, parameter in trainable.named_parameters():
            assert parameter.requires_grad, f"trainable {name} is frozen"
            assert getattr(parameter, "average_gradients_across_tp_domain", False), name
        for layer in (frozen, trainable):
            assert layer.indexer.index_kpool_compress_gate.dtype == torch.bfloat16
            assert layer.indexer.index_kpool_compress_ape.dtype == torch.float32
        assert getattr(trainable.indexer.index_kpool_compress_ape, "use_muon", True) is False

        saved = {
            key: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for key, value in frozen.state_dict().items()
        }
        assert set(saved) == set(trainable.state_dict())
        prefix = "decoder.layers.0.self_attention.core_attention."
        frozen_sharded = frozen.sharded_state_dict(prefix=prefix)
        trainable_sharded = trainable.sharded_state_dict(prefix=prefix)
        assert set(frozen_sharded) == set(trainable_sharded)
        for key, entry in frozen_sharded.items():
            other = trainable_sharded[key]
            assert type(entry) is type(other), key
            if isinstance(entry, ShardedTensor):
                assert entry.global_shape == other.global_shape, key
                assert entry.local_shape == other.local_shape, key
                assert entry.dtype == other.dtype, key

        result = trainable.load_state_dict(saved, strict=True)
        assert not result.missing_keys and not result.unexpected_keys
        for key, value in trainable.state_dict().items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                assert torch.equal(value, saved[key]), f"{key} did not load"
        for name, parameter in trainable.named_parameters():
            assert parameter.requires_grad, f"loading froze {name}"

        seen = _install_recorders(monkeypatch)
        inputs = _make_inputs(trainable.config, _ROWS, batch=1, seed=17)
        run = _run_layer(trainable, inputs)
        assert len(_layer_losses(seen)) == 1
        _assert_indexer_trained(trainable, run)

        frozen_again = _build_layer(_config(loss_coeff=0.0), pg_collection, seed=99)
        result = frozen_again.load_state_dict(trainable.state_dict(), strict=True)
        assert not result.missing_keys and not result.unexpected_keys
        for name, parameter in frozen_again.named_parameters():
            assert not parameter.requires_grad, f"loading a trainable checkpoint unfroze {name}"
    finally:
        Utils.destroy_model_parallel()


class _TrunkAndCoreAttention(MegatronModule):
    """A trunk weight (the value up-projection) feeding a ``DSAttention`` core.

    The trunk trains in both modes, so a frozen-indexer model still has an optimizer, as a
    frozen run does; its optimizer state holds the trunk and no indexer parameter.
    """

    def __init__(self, attention, *, seed):
        super().__init__(config=attention.config)
        config = attention.config
        generator = torch.Generator(device="cpu").manual_seed(seed)
        weight = 0.02 * torch.randn(
            config.num_attention_heads, config.v_head_dim, config.kv_lora_rank, generator=generator
        )
        # Registered before the core, so the trunk's BF16 gradient buffer comes first in both
        # modes, as the decoder's weights precede an indexer in a model.
        self.up_v_weight = torch.nn.Parameter(weight.to(device="cuda", dtype=torch.bfloat16))
        self.core_attention = attention

    def forward(self, query, key, x, qr):
        return self.core_attention(
            query=query,
            key=key,
            value=None,
            attention_mask=None,
            x=x,
            qr=qr,
            attn_mask_type=AttnMaskType.causal,
            packed_seq_params=None,
            up_v_weight=self.up_v_weight,
        )


def _trunk_model_and_optimizer(config, pg_collection, *, seed):
    """The trunk-and-core module under DDP with the distributed optimizer (Adam)."""
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer

    module = _TrunkAndCoreAttention(_build_layer(config, pg_collection, seed=seed), seed=seed + 1)
    ddp_model = DistributedDataParallel(
        config, DistributedDataParallelConfig(use_distributed_optimizer=True), module
    )
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer="adam",
            lr=1e-2,
            weight_decay=0.0,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_distributed_optimizer=True,
        ),
        [ddp_model],
    )
    return module, ddp_model, optimizer


def _trunk_step(ddp_model, optimizer, inputs):
    """One forward, backward and distributed-optimizer step; returns whether it updated."""
    ddp_model.zero_grad_buffer()
    optimizer.zero_grad()
    leaves = {
        name: inputs[name].detach().clone().requires_grad_(name in _TRUNK_INPUTS)
        for name in ("query", "key", "x", "qr")
    }
    with forward_sharing_lifetime(None, None, ddp_model.module.config):
        output = ddp_model(leaves["query"], leaves["key"], leaves["x"], leaves["qr"])
    output.backward(inputs["grad_output"].view_as(output))
    ddp_model.finish_grad_sync()
    update_successful, _grad_norm, _zeros = optimizer.step()
    return update_successful


def _float_state(module):
    return {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    }


def _checkpoint_failures(error):
    """The exceptions behind a load error: a DCP ``CheckpointException`` wraps each rank's."""
    from torch.distributed.checkpoint.api import CheckpointException

    if isinstance(error, CheckpointException):
        return [failure[0] for failure in error.failures.values()]
    return [error]


@pytest.mark.parametrize("strict", ["assume_ok_unexpected", "log_all"])
def test_frozen_dist_checkpoint_loads_into_a_trainable_layer_without_optimizer_state(
    tmp_path_dist_ckpt, strict
):
    """A frozen run's distributed checkpoint resumes a trainable run only without its optimizer.

    The frozen model (trunk plus a frozen KPool indexer) trains one step under DDP with the
    distributed optimizer, then saves its model and optimizer state with
    ``dist_checkpointing.save`` in the default ``dp_reshardable`` format. A trainable model:

    - (a) loading the optimizer state too fails: the frozen run's BF16 gradient bucket holds
      the trunk alone, the trainable run's also the indexer, so the load stops on a global
      shape mismatch of that bucket's optimizer state before reading anything. The error
      names the bucket, not the indexer. ``log_all`` is the GLM-5.3-Flash recipes'
      ``--dist-ckpt-strictness``: it drops the trainable run's extra FP32 bucket (the slot
      bias) as unexpected and stops on the same mismatch.
    - (b) loading the model alone (``--no-load-optim``) gives the saved weights, and one step
      with a fresh optimizer updates the pool gate and the slot bias.
    """
    from torch.distributed.checkpoint.api import CheckpointException

    from megatron.core import dist_checkpointing
    from megatron.core.dist_checkpointing.core import CheckpointingException
    from tests.unit_tests.dist_checkpointing import TempNamedDir

    metadata = {"distrib_optim_sharding_type": "dp_reshardable"}
    gate, ape = f"core_attention.{_GATE}", f"core_attention.{_APE}"
    load_error = None
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
    try:
        model_parallel_cuda_manual_seed(1234)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        frozen_config = _config(loss_coeff=0.0, topk_freq=1)
        trainable_config = _config(loss_coeff=_COEFF, topk_freq=1)
        inputs = _make_inputs(frozen_config, _ROWS, batch=1, seed=59)

        frozen, frozen_ddp, frozen_optimizer = _trunk_model_and_optimizer(
            frozen_config, pg_collection, seed=2026
        )
        trained = {name for name, p in frozen.named_parameters() if p.requires_grad}
        assert trained == {"up_v_weight"}, f"the frozen model trains {sorted(trained)}"
        frozen_stepped = _trunk_step(frozen_ddp, frozen_optimizer, inputs)
        saved = _float_state(frozen)

        with TempNamedDir(
            tmp_path_dist_ckpt / f"kpool_frozen_to_trainable_{strict}", sync=True
        ) as ckpt_dir:
            # As load_checkpoint and save_checkpoint do, the unwrapped model gives the keys.
            model_sd = frozen.sharded_state_dict()
            dist_checkpointing.save(
                {
                    "model": model_sd,
                    "optimizer": frozen_optimizer.sharded_state_dict(model_sd, metadata=metadata),
                },
                ckpt_dir,
            )

            # (a) The model and the optimizer state, as a resume without --no-load-optim.
            trainable, _, trainable_optimizer = _trunk_model_and_optimizer(
                trainable_config, pg_collection, seed=7
            )
            template = trainable.sharded_state_dict()
            load_template = {
                "model": template,
                "optimizer": trainable_optimizer.sharded_state_dict(
                    template, is_loading=True, metadata=metadata
                ),
            }
            try:
                dist_checkpointing.load(load_template, ckpt_dir, strict=strict)
            except (Exception, CheckpointException) as error:
                load_error = error

            # (b) The model alone, as --no-load-optim: a fresh optimizer, main parameters
            # reloaded from the loaded weights as load_checkpoint does.
            fresh, fresh_ddp, fresh_optimizer = _trunk_model_and_optimizer(
                trainable_config, pg_collection, seed=11
            )
            with torch.no_grad():
                for parameter in fresh.parameters():
                    if parameter.ndim < 2:
                        parameter.add_(0.5)  # every tensor differs before the load
            before_load = _float_state(fresh)
            loaded = dist_checkpointing.load(
                {"model": fresh.sharded_state_dict()}, ckpt_dir, strict=strict
            )
            result = fresh.load_state_dict(loaded["model"], strict=True)
            fresh_optimizer.reload_model_params()
            after_load = _float_state(fresh)
            fresh_stepped = _trunk_step(fresh_ddp, fresh_optimizer, inputs)
            after_step = _float_state(fresh)
    finally:
        Utils.destroy_model_parallel()

    assert frozen_stepped, "the frozen run's optimizer step was skipped"
    # (a)
    assert load_error is not None, "the frozen optimizer state loaded into a trainable run"
    failures = _checkpoint_failures(load_error)
    assert any(
        isinstance(failure, CheckpointingException)
        and "Global shape mismatch" in str(failure)
        and "optimizer.distributed.dp_group_idx_0.gbuf_idx_0." in str(failure)
        for failure in failures
    ), f"unexpected optimizer-load failure: {[repr(failure) for failure in failures]}"
    # (b)
    assert not result.missing_keys and not result.unexpected_keys
    assert set(after_load) == set(saved)
    for name, value in saved.items():
        assert torch.equal(after_load[name], value), f"{name} did not load"
        if name in (gate, ape):
            assert not torch.equal(before_load[name], value), f"{name} equals before the load"
    for name, parameter in fresh.named_parameters():
        assert parameter.requires_grad, f"loading froze {name}"
    assert fresh_stepped, "the trainable run's optimizer step was skipped"
    for name in (gate, ape):
        assert torch.isfinite(after_step[name]).all(), f"{name} is not finite after the step"
        assert not torch.equal(after_step[name], after_load[name]), f"the step left {name}"


@pytest.mark.parametrize(
    ("deterministic", "rows"),
    [
        pytest.param(False, _GRAPH_ROWS, id="default"),
        pytest.param(True, _GRAPH_ROWS, id="deterministic"),
        pytest.param(True, _RANK_128K_ROWS, id="128k-rank-deterministic"),
    ],
)
def test_trainable_4k_step_repeats_bitwise(monkeypatch, deterministic, rows):
    """Two steps on the same inputs give bitwise-equal losses and indexer gradients.

    In deterministic mode every indexer gradient repeats bit for bit, at the 4K shape and at
    the rank shape of the 128K CP8 recipe (16,384 local query rows, top-k 2048, the
    production indexer heads), where the default 1 GiB budget splits the teacher and the
    student into dozens of row chunks. That shape runs as a CP1 layer, with fewer main
    attention heads to bound the dense PyTorch sparse attention. The KPool loss itself
    accumulates the pooled-key gradient without floating-point atomics in either mode
    (``index_add_`` under deterministic algorithms, a sorted segment sum otherwise), so in
    the default mode the loss and the gradients that reach the indexer only through the
    pooled keys (the pool gate and the slot bias) repeat bit for bit as well.
    """
    geometry = None
    if rows == _RANK_128K_ROWS:
        _require_gpu_memory_gib(_RANK_128K_MIN_GPU_GIB)
        geometry = _RANK_128K_GEOMETRY
        budget = _GRAPH_TOPK // _POOL
        student_rows = dsa_module._kpool_loss_chunk_rows(
            rows,
            dsa_module._kpool_student_bytes_per_row(1, _INDEX_HEADS, _INDEX_DIM, budget),
            dsa_module._KPOOL_LOSS_CHUNK_BYTES,
        )
        assert -(-rows // student_rows) >= 30, "the 128K-rank shape must span many row chunks"
    seen = _install_recorders(monkeypatch)
    config = _config(
        loss_coeff=_COEFF,
        topk=_GRAPH_TOPK,
        topk_freq=1,
        deterministic=deterministic,
        geometry=geometry,
    )
    trainable = _build_layer(config, _single_rank_groups())
    inputs = _make_inputs(config, rows, batch=1, seed=19)
    with _deterministic_algorithms(deterministic):
        first = _run_layer(trainable, inputs)
        second = _run_layer(trainable, inputs)
    losses = _layer_losses(seen)
    assert len(losses) == 2
    assert torch.equal(losses[0]["loss"], losses[1]["loss"]), "the loss differs between steps"
    _assert_indexer_trained(trainable, first)
    assert first.param_grads.keys() == second.param_grads.keys()
    for name in first.param_grads if deterministic else (_GATE, _APE):
        grad = first.param_grads[name]
        assert torch.equal(grad, second.param_grads[name]), f"{name} differs between steps"


class _PositionalLayer(torch.nn.Module):
    """``DSAttention`` behind the positional-tensor signature that graph capture takes."""

    def __init__(self, attention):
        super().__init__()
        self.attention = attention

    def forward(self, query, key, x, qr, up_v_weight):
        return self.attention(
            query=query,
            key=key,
            value=None,
            attention_mask=None,
            x=x,
            qr=qr,
            attn_mask_type=AttnMaskType.causal,
            packed_seq_params=None,
            up_v_weight=up_v_weight,
        )


def test_te_graph_capture_of_the_trainable_layer_replays_its_eager_step(record_property):
    """A 4K Transformer Engine graph of the trainable layer replays its eager step.

    The layer is captured as production captures ``attn`` (TE ``make_graphed_callables``
    with Megatron's capture flag set), forward and backward, loss included: the loss path
    has static shapes and no host synchronization. The replay's output, logged loss, input
    gradients and indexer gradients match an eager step. Peak allocated memory during
    capture and replay is recorded as a test property.
    """
    from transformer_engine.pytorch.graph import make_graphed_callables

    from megatron.core.transformer.cuda_graphs import _set_capture_end, _set_capture_start

    config = _config(loss_coeff=_COEFF, topk=_GRAPH_TOPK, topk_freq=1)
    trainable = _build_layer(config, _single_rank_groups())
    inputs = _make_inputs(config, _GRAPH_ROWS, batch=1, seed=23)
    eager = _run_layer(trainable, inputs)
    eager_loss = DSAIndexerLossLoggingHelper.tracker["values"][0].clone()
    assert torch.isfinite(eager_loss) and float(eager_loss) > 0
    _assert_indexer_trained(trainable, eager)

    def leaves():
        return tuple(
            inputs[name].detach().clone().requires_grad_(name in _TRUNK_INPUTS)
            for name in _LAYER_INPUTS
        )

    graphed_layer = _PositionalLayer(trainable)
    trainable.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    _set_capture_start()
    try:
        graphed = make_graphed_callables(
            graphed_layer, leaves(), num_warmup_iters=2, allow_unused_input=True
        )
    finally:
        _set_capture_end()
    torch.cuda.synchronize()
    record_property("capture_max_allocated_bytes", torch.cuda.max_memory_allocated())

    trainable.zero_grad(set_to_none=True)
    DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
    replay_inputs = leaves()
    torch.cuda.reset_peak_memory_stats()
    output = graphed(*replay_inputs)
    output.backward(inputs["grad_output"].view_as(output))
    torch.cuda.synchronize()
    record_property("replay_max_allocated_bytes", torch.cuda.max_memory_allocated())
    replay_loss = DSAIndexerLossLoggingHelper.tracker["values"][0].clone()

    torch.testing.assert_close(replay_loss, eager_loss, rtol=1e-5, atol=1e-8)
    _assert_similar(output, eager.output, label="output", eps=1e-3)
    for name, leaf in zip(_LAYER_INPUTS, replay_inputs):
        if name in _TRUNK_INPUTS:
            _assert_similar(leaf.grad, eager.input_grads[name], label=f"d{name}", eps=1e-3)
    for name, parameter in trainable.named_parameters():
        assert parameter.grad is not None, f"the graph gave {name} no gradient"
        _assert_similar(parameter.grad, eager.param_grads[name], label=name, eps=1e-3)


@pytest.mark.parametrize("scorer", ["torch", "cudnn"])
def test_cudnn_backend_trainable_layer_matches_the_reference(monkeypatch, scorer):
    """On the cuDNN DSA backend (FlashMLA sparse attention), the trainable layer still trains
    through the KPool loss alone and its loss matches the reference.

    The real token-level hooks of that backend raise when reached. With the cuDNN KPool
    scorer, the layer's selector is the cuDNN pool top-k, and with cuDNN frontend >= 1.31 the
    loss runs on the cuDNN frontend kernels instead of ``KPoolSparseIndexerLoss``
    (``test_kpool_trainable_cudnn_gpu.py`` covers that path).
    """
    _require_cudnn_dsa()
    cudnn_loss = scorer == "cudnn" and dsa_cudnn_kernels._cudnn_frontend_at_least(
        dsa_cudnn_kernels._KPOOL_SPARSE_LOSS_MIN_FE_VERSION
    )
    seen = _install_recorders(monkeypatch)
    config = _config(
        loss_coeff=_COEFF,
        topk=_CUDNN_TOPK,
        kernel_backend="cudnn",
        scorer=scorer,
        geometry=_CUDNN_GEOMETRY,
    )
    trainable = _build_layer(config, _single_rank_groups())
    _forbid_token_level_loss_hooks(monkeypatch)
    indexer_calls = _spy_indexer(trainable)
    inputs = _make_inputs(config, _CUDNN_ROWS, batch=1, seed=29)

    run = _run_layer(trainable, inputs)
    (record,) = _layer_losses(seen)
    (indexer_call,) = indexer_calls
    (call,) = seen.kpool
    assert seen.loss_function_calls == (0 if cudnn_loss else 1)
    assert _is_trainable_call(call)
    assert (call.kwargs.get("pool_topk_fn") is not None) == (scorer == "cudnn")
    _assert_indexer_trained(trainable, run)
    case = _reference_case(trainable, inputs, run, indexer_call)
    expected = _oracle(
        **case.tensors,
        **case.fixed,
        profile=_REFERENCE_PROFILE[True],
        loss_coeff=_COEFF,
        row_count=float(_CUDNN_ROWS),
    )
    torch.testing.assert_close(
        record["loss"].double().cpu(), expected.loss.detach().double(), rtol=1e-4, atol=1e-8
    )


# ---------------------------------------------------------------------------------------
# A layer stack: IndexShare (topk_freq=2), an MTP DSA layer and PP2
# ---------------------------------------------------------------------------------------

# Four decoder layers and one MTP layer with dsa_indexer_topk_freq=2 and skip offset 0:
# layers 1 and 3 compute their top-k, layers 2 and 4 reuse it, and the MTP layer, number
# num_layers + 1 = 5, computes its own. Under PP2, stage 0 holds layers 1-2 and stage 1
# layers 3-4 plus the MTP layer: each stage starts on a computing layer.
_STACK_LAYERS, _STACK_TOPK_FREQ = 4, 2
_STACK_COMPUTING = (1, 3, 5)
_STACK_SOURCES = {2: 1, 4: 3}


def _stack_config(pp=1):
    return _config(
        loss_coeff=_COEFF,
        topk_freq=_STACK_TOPK_FREQ,
        skip_topk_offset=0,
        num_layers=_STACK_LAYERS,
        mtp_num_layers=1,
        pp=pp,
    )


def _stack_stage_layers(config, pp_rank):
    """This stage's DSA layers in run order: ``(layer number, is_mtp, constructor number)``.

    The decoder layers come from Megatron's PP split, which must keep every skip layer with
    its source; the MTP layers follow on the last stage, numbered after the decoder.
    """
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        _validate_dsa_index_share_pipeline_split,
    )
    from megatron.core.transformer.multi_token_prediction import get_mtp_num_layers_to_build
    from megatron.core.transformer.transformer_block import get_num_layers_to_build
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

    offset = get_transformer_layer_offset(config, pp_rank=pp_rank)
    decoder = [offset + 1 + i for i in range(get_num_layers_to_build(config, pp_rank=pp_rank))]
    _validate_dsa_index_share_pipeline_split(config, [number - 1 for number in decoder])
    stage = [(number, False, number) for number in decoder]
    for depth in range(1, get_mtp_num_layers_to_build(config, pp_rank=pp_rank) + 1):
        stage.append((config.num_layers + depth, True, depth))
    return stage


def _build_stack_layer(config, pg_collection, layer_number, *, is_mtp, seed):
    """A real ``DSAttention`` of the stack; skip layers have no indexer."""
    attention = dsa_module.DSAttention(
        config=config,
        submodules=dsa_module.DSAttentionSubmodules(indexer=_indexer_spec()),
        layer_number=layer_number,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        cp_comm_type="allgather",
        pg_collection=pg_collection,
        is_mtp_layer=is_mtp,
    ).cuda()
    _seed_parameters(attention, seed)
    return attention.train()


def _run_stack_stage(monkeypatch, config, pg_collection, stage_layers, *, seed):
    """One training step of a PP stage's DSA layers: forward in order in one forward, backward.

    Records which layer made each tracker entry and each ``KPoolSparseIndexerLoss``
    application, the top-k each layer's sparse attention consumed (its storage and a copy)
    and the top-k each layer published, and snapshots the local metric tracker.
    """
    seen = _install_recorders(monkeypatch)
    current = {"layer": None}
    loss_layers, consumed = [], {}
    counted_apply = dsa_module.KPoolSparseIndexerLoss.apply

    def apply(*args, **kwargs):
        loss_layers.append(current["layer"])
        return counted_apply(*args, **kwargs)

    monkeypatch.setattr(dsa_module.KPoolSparseIndexerLoss, "apply", apply)
    real_sparse_attention = dsa_module._run_sparse_attention

    def sparse_attention(*args, **kwargs):
        topk = kwargs["topk_indices"]
        consumed[current["layer"]] = SimpleNamespace(
            data_ptr=topk.data_ptr(), value=topk.detach().clone()
        )
        return real_sparse_attention(*args, **kwargs)

    monkeypatch.setattr(dsa_module, "_run_sparse_attention", sparse_attention)

    layers = {}
    for number, is_mtp, constructor_number in stage_layers:
        layer = _build_stack_layer(
            config, pg_collection, constructor_number, is_mtp=is_mtp, seed=seed + number
        )
        assert layer.layer_number == number, f"layer {number} built as {layer.layer_number}"
        layers[number] = layer
    outputs, output_grads = [], []
    with forward_sharing_lifetime(None, None, config):
        for number, layer in layers.items():
            inputs = _make_inputs(config, _ROWS, batch=1, seed=seed + 100 + number)
            leaves = {
                name: inputs[name].detach().clone().requires_grad_(name in _TRUNK_INPUTS)
                for name in _LAYER_INPUTS
            }
            current["layer"] = number
            output = layer(
                query=leaves["query"],
                key=leaves["key"],
                value=None,
                attention_mask=None,
                x=leaves["x"],
                qr=leaves["qr"],
                attn_mask_type=AttnMaskType.causal,
                packed_seq_params=None,
                up_v_weight=leaves["up_v_weight"],
            )
            current["layer"] = None
            outputs.append(output)
            output_grads.append(inputs["grad_output"].view_as(output))
        payload = next(iter(layers.values()))._get_dsa_index_sharing_payload(None, None)
        published = {
            number: SimpleNamespace(data_ptr=topk.data_ptr(), value=topk.detach().clone())
            for number, topk in payload.topk_by_layer.items()
        }
    torch.autograd.backward(outputs, output_grads)
    torch.cuda.synchronize()
    tracker = DSAIndexerLossLoggingHelper.tracker
    return SimpleNamespace(
        layers=layers,
        losses=list(seen.losses),
        selections=list(seen.kpool),
        loss_layers=loss_layers,
        consumed=consumed,
        published=published,
        tracker={
            key: tracker[key].detach().clone()
            for key in ("values", "tail_mass_sums", "tail_mass_rows")
            if key in tracker
        },
        param_grads={
            number: {
                name: parameter.grad.detach().clone()
                for name, parameter in layer.named_parameters()
                if parameter.grad is not None
            }
            for number, layer in layers.items()
        },
    )


def _layer_slots(values):
    """1-based layer numbers whose tracker slot is non-zero."""
    return sorted(int(index) + 1 for index in torch.nonzero(values).flatten().tolist())


def _check_stack_stage(stage, stage_layers, num_layers):
    """Per-layer losses at computing layers only; skip layers attend their source's top-k."""
    numbers = [number for number, _, _ in stage_layers]
    computing = [number for number in numbers if number in _STACK_COMPUTING]
    skipping = [number for number in numbers if number in _STACK_SOURCES]
    assert sorted(computing + skipping) == sorted(numbers), f"unexpected layers {numbers}"
    for number in computing:
        layer = stage.layers[number]
        assert not layer.skip_topk and layer.indexer is not None, f"layer {number} skips"
    for number in skipping:
        layer = stage.layers[number]
        assert layer.skip_topk and layer.indexer is None, f"layer {number} computes"
        assert layer.source_layer == _STACK_SOURCES[number]
        assert not list(layer.parameters()), f"skip layer {number} owns parameters"

    # One loss per computing layer, logged at its own number (the MTP layer after the
    # decoder); skip layers neither apply the KPool loss nor log anything.
    assert sorted(record["layer_number"] for record in stage.losses) == computing
    assert all(record["num_layers"] == num_layers for record in stage.losses)
    assert sorted(stage.loss_layers) == computing, f"KPool losses ran in {stage.loss_layers}"
    assert _layer_slots(stage.tracker["values"]) == computing
    assert _layer_slots(stage.tracker["tail_mass_rows"]) == computing
    sums, rows = stage.tracker["tail_mass_sums"], stage.tracker["tail_mass_rows"]
    for number in computing:
        mean = float(sums[number - 1] / rows[number - 1])
        assert 0.0 < mean < 1.0, f"layer {number} logs a tail mass of {mean}"

    # Only computing layers select (through the trainable path) and publish a top-k; each
    # sparse attention consumes the published tensor of its source, bit for bit.
    assert len(stage.selections) == len(computing)
    assert all(_is_trainable_call(call) for call in stage.selections)
    assert sorted(stage.published) == computing, f"published {sorted(stage.published)}"
    assert sorted(stage.consumed) == sorted(numbers)
    for number in numbers:
        source = _STACK_SOURCES.get(number, number)
        used, published = stage.consumed[number], stage.published[source]
        assert used.data_ptr == published.data_ptr, f"layer {number} attends another tensor"
        assert torch.equal(used.value, published.value), f"layer {number} attends other ids"
    for number in skipping:
        source_used = stage.consumed[_STACK_SOURCES[number]].value
        assert torch.equal(stage.consumed[number].value, source_used)

    for number in computing:
        _assert_indexer_trained(
            stage.layers[number], SimpleNamespace(param_grads=stage.param_grads[number])
        )


def test_index_share_and_mtp_stack_logs_only_computing_layers(monkeypatch):
    """One rank, no PP: a trainable KPool stack with IndexShare (topk_freq=2) and MTP.

    Layers 1 and 3 and the MTP layer (number 5) select, train and log their loss and tail
    mass at their own numbers; layers 2 and 4 own no indexer, run no KPool loss, log nothing
    and attend bit for bit the top-k their source published.
    """
    monkeypatch.setattr(DSAIndexerLossLoggingHelper, "tracker", {})
    config = _stack_config()
    stage_layers = _stack_stage_layers(config, pp_rank=0)
    assert [number for number, _, _ in stage_layers] == [1, 2, 3, 4, 5]
    stage = _run_stack_stage(monkeypatch, config, _single_rank_groups(), stage_layers, seed=47)
    _check_stack_stage(stage, stage_layers, num_layers=_STACK_LAYERS + 1)


def test_pp2_index_share_and_mtp_stack_logs_only_computing_layers(monkeypatch):
    """PP2: each stage runs its own computing and skip layers, the MTP layer on the last.

    Stage 0 holds layers 1-2 and stage 1 layers 3-4 and the MTP layer 5. Each stage logs
    only its computing layers, at their global numbers, and its skip layer attends the top-k
    of its source on the same stage. The PP reduction of the tracker then holds the loss and
    the tail mass of layers 1, 3 and 5 on every rank, and nothing at layers 2 and 4.
    """
    _require_ranks(2)
    monkeypatch.setattr(DSAIndexerLossLoggingHelper, "tracker", {})
    config = _stack_config(pp=2)
    num_layers = _STACK_LAYERS + 1
    stage = failure = None
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=2)
    try:
        model_parallel_cuda_manual_seed(1234)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=["tp", "cp", "pp", "dp"]
        )
        pp_rank = pg_collection.pp.rank()
        stage_layers = _stack_stage_layers(config, pp_rank=pp_rank)
        try:
            stage = _run_stack_stage(monkeypatch, config, pg_collection, stage_layers, seed=53)
        except Exception as error:  # still join the collectives below, then fail
            failure = error
        DSAIndexerLossLoggingHelper.reduce_loss_in_tracker(
            num_layers=num_layers, pp_group=pg_collection.pp, dp_group=pg_collection.dp
        )
        tracker = DSAIndexerLossLoggingHelper.tracker
        reduced = {
            key: tracker[key].detach().clone()
            for key in ("values", "tail_mass_sums", "tail_mass_rows")
            if key in tracker
        }
    finally:
        Utils.destroy_model_parallel()
    if failure is not None:
        raise failure

    expected = {0: [(1, False, 1), (2, False, 2)], 1: [(3, False, 3), (4, False, 4), (5, True, 1)]}
    assert stage_layers == expected[pp_rank], f"PP rank {pp_rank} holds {stage_layers}"
    _check_stack_stage(stage, stage_layers, num_layers=num_layers)
    assert set(reduced) == {"values", "tail_mass_sums", "tail_mass_rows"}
    assert _layer_slots(reduced["values"]) == list(_STACK_COMPUTING)
    assert _layer_slots(reduced["tail_mass_rows"]) == list(_STACK_COMPUTING)
    sums, rows = reduced["tail_mass_sums"], reduced["tail_mass_rows"]
    for number in _STACK_COMPUTING:
        mean = float(sums[number - 1] / rows[number - 1])
        assert 0.0 < mean < 1.0, f"layer {number} reduces to a tail mass of {mean}"
    for number in (number for number, _, _ in stage_layers if number in _STACK_COMPUTING):
        # This stage's own layers keep their values through the PP sum (DP ranks agree).
        torch.testing.assert_close(
            reduced["values"][number - 1], stage.tracker["values"][number - 1], rtol=1e-5, atol=0
        )


# ---------------------------------------------------------------------------------------
# Distributed: CP, TP, layout guards and the distributed optimizer
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(("hf_scoring", "key_dtype_compression"), _PROFILES)
@pytest.mark.parametrize("cp_size", [2, 4])
def test_cp_loss_and_gradients_equal_cp1(monkeypatch, cp_size, hf_scoring, key_dtype_compression):
    """Under CP the trainable layer selects, trains and logs as one CP1 layer does.

    Each rank holds two zigzag halves. The ranks' mean loss is the CP1 loss, the ranks' mean
    indexer gradient (what the DP x CP gradient average applies) is the CP1 gradient, and
    every rank publishes the CP1 selection of its rows. The pooled keys the trainable layer
    builds (pool each rank's tokens, then gather) equal, bit for bit, the pools of the
    frozen layer's gathered keys, and the trainable and frozen layers select the same pools.
    """
    _require_ranks(cp_size)
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=cp_size)
    try:
        model_parallel_cuda_manual_seed(1234)
        seen = _install_recorders(monkeypatch)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        cp_group = pg_collection.cp
        profile = dict(hf_scoring=hf_scoring, key_dtype_compression=key_dtype_compression)

        # CP1 on the whole sequence, run by every rank on its own.
        cp1_layer = _build_layer(_config(loss_coeff=_COEFF, **profile), _single_rank_groups())
        cp1_indexer = _spy_indexer(cp1_layer)
        inputs = _make_inputs(cp1_layer.config, _CP_ROWS, batch=1, seed=31)
        cp1 = _run_layer(cp1_layer, inputs)
        cp1_losses = _layer_losses(seen)

        rows = _zigzag_rows(_CP_ROWS, cp_size, cp_group.rank())
        local = _slice_rows(inputs, rows)
        seen.losses.clear()
        seen.kpool.clear()
        layer = _build_layer(_config(loss_coeff=_COEFF, cp=cp_size, **profile), pg_collection)
        cpn_indexer = _spy_indexer(layer)
        cpn = _run_layer(layer, local)
        losses = _layer_losses(seen)
        trainable_calls = list(seen.kpool)

        seen.kpool.clear()
        frozen = _build_layer(_config(loss_coeff=0.0, cp=cp_size, **profile), pg_collection)
        frozen_run = _run_layer(frozen, local)
        frozen_calls = list(seen.kpool)

        # Every collective runs before the first assertion, so that a rank that fails one
        # cannot leave its peers waiting in a collective.
        nan = float("nan")
        mean_loss = (
            losses[0]["loss"].float().clone()
            if len(losses) == 1
            else torch.full((), nan, device="cuda")
        )
        torch.distributed.all_reduce(mean_loss, group=cp_group)
        mean_loss /= cp_size
        mean_grads, mean_magnitudes = {}, {}
        for name, parameter in layer.named_parameters():
            grad = cpn.param_grads.get(name)
            rank_grad = (
                grad.float()
                if grad is not None
                else torch.full(parameter.shape, nan, dtype=torch.float32, device="cuda")
            )
            total, magnitude = rank_grad.clone(), rank_grad.abs()
            torch.distributed.all_reduce(total, group=cp_group)
            torch.distributed.all_reduce(magnitude, group=cp_group)
            mean_grads[name] = total / cp_size
            mean_magnitudes[name] = magnitude / cp_size
    finally:
        Utils.destroy_model_parallel()

    assert len(cp1_losses) == 1 and len(losses) == 1
    assert losses[0]["avg_group"] is cp_group, "the CP loss is not logged as a CP average"
    _assert_indexer_trained(layer, cpn)
    torch.testing.assert_close(mean_loss, cp1_losses[0]["loss"].float(), rtol=1e-4, atol=1e-8)
    # This rank's loss averages over its own rows, and the pooled-key gather's backward sums
    # every rank's pooled-key gradient: each indexer tensor of this rank's tokens gets
    # cp_size times its CP1 gradient.
    (cp1_call,), (cpn_call,) = cp1_indexer, cpn_indexer
    for name in ("q", "weights", "k", "gate_score"):
        expected = cp_size * cp1_call.grads[name].index_select(0, rows.cuda())
        _assert_similar(cpn_call.grads[name], expected, label=f"CP{cp_size} d{name}", eps=5e-3)
    # Parameter gradients sum rounded per-token terms over each rank's tokens, some with heavy
    # cancellation (the key norm's shift); the bound still catches a missing or doubled rank.
    for name, grad in mean_grads.items():
        _assert_rank_sum_close(
            grad,
            mean_magnitudes[name],
            cp1.param_grads[name],
            label=f"CP{cp_size} {name}",
            rel=5e-2,
        )

    # Selection: this rank's CP1 rows, and the frozen layer's selection, bit for bit.
    cp1_rows = cp1.published[:, rows.cuda()]
    assert torch.equal(cpn.published, cp1_rows), "CP and CP1 select differently"
    assert torch.equal(cpn.published, frozen_run.published), "trainable and frozen differ"
    (trainable_call,) = trainable_calls
    (frozen_call,) = frozen_calls
    assert _is_trainable_call(trainable_call) and not _is_trainable_call(frozen_call)
    pooled, base = trainable_call.kwargs["pooled_keys"]
    expected_pooled = dsa_module._kpool_compress_keys(
        _call_arg(frozen_call, 1, "k"),
        _call_arg(frozen_call, 5, "gate_score"),
        _call_arg(frozen_call, 6, "ape"),
        _POOL,
        kpool_key_dtype_compression=key_dtype_compression,
    )
    assert pooled.shape == expected_pooled.shape == (_CP_ROWS // _POOL, 1, _INDEX_DIM)
    assert torch.equal(pooled.float(), expected_pooled.float()), "pool-then-gather differs"
    assert torch.equal(base.cpu(), _pool_token_base(_CP_ROWS, _POOL))


def _rank1_lowest_pools(tp_rank, state):
    """On TP rank 1 only, replace the selection with each row's lowest visible pools.

    Rows with more visible pools than the budget then differ from what the scores choose.
    ``state['on']`` arms it; the replaced selection is stored in ``state``.
    """

    def perturb(_args, kwargs, result):
        if not state["on"] or tp_rank != 1 or not kwargs.get("return_selection"):
            return result
        index_scores, tokens, pool_ids = result
        batch, rows, budget = pool_ids.shape
        device = pool_ids.device
        visible = torch.arange(1, rows + 1, device=device) // _POOL
        ids = torch.arange(budget, device=device)
        lowest = torch.where(ids < visible[:, None], ids, torch.full_like(ids, -1))
        lowest = lowest.expand(batch, rows, budget)
        expanded = lowest.unsqueeze(-1) * _POOL + torch.arange(_POOL, device=device)
        expanded = torch.where(lowest.unsqueeze(-1) >= 0, expanded, -1).flatten(-2)
        index_topk = budget * _POOL
        replaced = torch.cat((expanded.to(tokens.dtype), tokens[..., index_topk:]), dim=-1)
        state["original"], state["replaced"] = tokens.clone(), replaced.clone()
        return index_scores, replaced.contiguous(), lowest.to(pool_ids.dtype).contiguous()

    return perturb


@pytest.mark.parametrize("case", ["agreeing", "rank1-disagrees", "tied-zero-scores"])
def test_tp2_ranks_use_tp_rank0_selection_and_compute_one_loss(monkeypatch, case):
    """Every TP rank attends TP rank 0's selection and computes the same loss.

    The ranks hold different heads, so their tail masses differ, and the head-summed target
    is reduced over TP. ``rank1-disagrees`` hands TP rank 1 another selection than rank 0's;
    ``tied-zero-scores`` zeroes the head weights, so every pool ties at the boundary. Either
    way both ranks publish rank 0's selection and log bitwise the same loss, which is the
    TP1 loss; every rank's indexer gradients are the TP1 gradients.
    """
    tp_size = 2
    _require_ranks(tp_size)
    state = {"on": False}
    Utils.initialize_model_parallel(tensor_model_parallel_size=tp_size)
    try:
        model_parallel_cuda_manual_seed(1234)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        tp_group = pg_collection.tp
        tp_rank = tp_group.rank()
        perturb = _rank1_lowest_pools(tp_rank, state) if case == "rank1-disagrees" else None
        seen = _install_recorders(monkeypatch, perturb=perturb)

        def build(config, groups):
            layer = _build_layer(config, groups)
            if case == "tied-zero-scores":
                with torch.no_grad():
                    layer.indexer.linear_weights_proj.weight.zero_()
            return layer

        tp1_layer = build(_config(loss_coeff=_COEFF), _single_rank_groups())
        inputs = _make_inputs(tp1_layer.config, _ROWS, batch=1, seed=37)
        tp1 = _run_layer(tp1_layer, inputs)
        tp1_losses = _layer_losses(seen)

        seen.losses.clear()
        layer = build(_config(loss_coeff=_COEFF, tp=tp_size), pg_collection)
        state["on"] = True
        run = _run_layer(layer, _slice_heads(inputs, tp_rank, tp_size))
        state["on"] = False
        losses = _layer_losses(seen)

        nan = torch.full((1,), float("nan"), device="cuda")
        local_loss = losses[0]["loss"].float().reshape(1) if len(losses) == 1 else nan
        all_losses = _all_gather(local_loss, tp_group)
        all_published = _all_gather(run.published, tp_group)
    finally:
        Utils.destroy_model_parallel()

    assert len(tp1_losses) == 1 and len(losses) == 1
    _assert_indexer_trained(layer, run)
    assert torch.equal(all_losses[0], all_losses[1]), f"TP ranks log {all_losses}"
    assert torch.equal(all_published[0], all_published[1]), "TP ranks attend different tokens"
    assert torch.equal(run.published, tp1.published), "TP rank 0 does not select as TP1"
    if case == "rank1-disagrees" and tp_rank == 1:
        assert not torch.equal(state["original"], state["replaced"]), "nothing was replaced"
        assert not torch.equal(run.published, state["replaced"]), "rank 1 kept its own pools"
    torch.testing.assert_close(local_loss[0], tp1_losses[0]["loss"].float(), rtol=1e-5, atol=1e-8)
    for name, grad in run.param_grads.items():
        _assert_similar(grad, tp1.param_grads[name], label=f"TP{tp_size} {name}", eps=1e-3)


@pytest.mark.parametrize("layout", ["misaligned-cp-length", "packed-thd"])
def test_trainable_cp_layout_guards_raise(monkeypatch, layout):
    """Under CP2, packed THD and local lengths that split a pool raise ``ValueError``.

    Every rank raises at the same point. The frozen selector runs the misaligned layout, so
    that error is the trainable path's guard, not a layout the layer cannot run.
    """
    cp_size = 2
    _require_ranks(cp_size)
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=cp_size)
    try:
        model_parallel_cuda_manual_seed(1234)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        if layout == "misaligned-cp-length":
            # 24 rows: each rank holds two halves of 6 rows, which split the pools.
            global_rows, topk, packed, match = 24, 8, None, "multiple of 2"
        else:
            global_rows, topk, match = 64, _TOPK, "packed THD"
            packed = _packed_seq_params((0, global_rows))
        configs = {
            coeff: _config(loss_coeff=coeff, cp=cp_size, topk=topk, topk_freq=1)
            for coeff in (0.0, _COEFF)
        }
        inputs = _make_inputs(configs[0.0], global_rows, batch=1, seed=41)
        rows = _zigzag_rows(global_rows, cp_size, pg_collection.cp.rank())
        local = _slice_rows(inputs, rows)
        if packed is None:
            frozen = _build_layer(configs[0.0], pg_collection)
            _run_layer(frozen, local)
        trainable = _build_layer(configs[_COEFF], pg_collection)
        with pytest.raises(ValueError, match=match):
            _run_layer(trainable, local, packed_seq_params=packed)
    finally:
        Utils.destroy_model_parallel()


def _ready_and_record(callback, readied, name):
    readied.append(name)
    return callback()


def _ddp_step(ddp_model, optimizer, inputs):
    ddp_model.zero_grad_buffer()
    optimizer.zero_grad()
    leaves = {
        name: inputs[name].detach().clone().requires_grad_(name in _TRUNK_INPUTS)
        for name in _LAYER_INPUTS
    }
    with forward_sharing_lifetime(None, None, ddp_model.module.config):
        output = ddp_model(
            query=leaves["query"],
            key=leaves["key"],
            value=None,
            attention_mask=None,
            x=leaves["x"],
            qr=leaves["qr"],
            attn_mask_type=AttnMaskType.causal,
            packed_seq_params=None,
            up_v_weight=leaves["up_v_weight"],
        )
    output.backward(inputs["grad_output"].view_as(output))
    ddp_model.finish_grad_sync()
    update_successful, _grad_norm, _zeros = optimizer.step()
    return update_successful


def test_distributed_optimizer_with_overlap_param_gather_publishes_gate_and_ape_before_use():
    """The gate and the slot bias are all-gathered before the indexer reads them.

    ``DSAttention`` calls the indexer's ``forward_before_topk`` directly, so DDP's forward
    pre-hook on the indexer, which waits for an overlapped parameter all-gather, never runs.
    With the distributed optimizer, ``overlap_param_gather`` and one bucket per parameter,
    the second step's indexer must publish both parameters itself: their readiness
    callbacks run, and the values it reads are the ones the first step's update gave every
    data-parallel rank.
    """
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer

    _require_ranks(2)
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
    try:
        model_parallel_cuda_manual_seed(1234)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        config = _config(loss_coeff=_COEFF, topk_freq=1)
        trainable = _build_layer(config, pg_collection)
        indexer = trainable.indexer
        watched = {
            "gate": indexer.index_kpool_compress_gate,
            "ape": indexer.index_kpool_compress_ape,
        }
        ddp_config = DistributedDataParallelConfig(
            use_distributed_optimizer=True,
            overlap_param_gather=True,
            overlap_grad_reduce=True,
            bucket_size=1,
        )
        ddp_model = DistributedDataParallel(config, ddp_config, trainable)
        for name, parameter in watched.items():
            assert hasattr(parameter, PARAM_READY_CALLBACK_ATTR), f"DDP did not bucket the {name}"
        optimizer = get_megatron_optimizer(
            OptimizerConfig(
                optimizer="adam",
                lr=1e-2,
                weight_decay=0.0,
                bf16=True,
                params_dtype=torch.bfloat16,
                use_distributed_optimizer=True,
            ),
            [ddp_model],
        )
        rank = torch.distributed.get_rank()
        inputs = _make_inputs(config, _ROWS, batch=1, seed=43 + rank)
        initial = {name: parameter.detach().clone() for name, parameter in watched.items()}
        first_step = _ddp_step(ddp_model, optimizer, inputs)

        readied, read = [], {}
        for name, parameter in watched.items():
            callback = getattr(parameter, PARAM_READY_CALLBACK_ATTR)
            setattr(
                parameter,
                PARAM_READY_CALLBACK_ATTR,
                functools.partial(_ready_and_record, callback, readied, name),
            )
        real_forward_before_topk = indexer.forward_before_topk

        def forward_before_topk(*args, **kwargs):
            result = real_forward_before_topk(*args, **kwargs)
            read.update({name: parameter.detach().clone() for name, parameter in watched.items()})
            return result

        indexer.forward_before_topk = forward_before_topk
        second_step = _ddp_step(ddp_model, optimizer, inputs)
        gathered = {name: _all_gather(value) for name, value in read.items()}
    finally:
        Utils.destroy_model_parallel()

    assert first_step and second_step, "an optimizer step was skipped"
    assert set(readied) == set(watched), f"readiness callbacks run for {sorted(set(readied))}"
    for name in watched:
        values = gathered[name]
        for other in values[1:]:
            assert torch.equal(values[0], other), f"DP ranks read different {name} values"
        assert not torch.equal(read[name], initial[name]), f"the first step did not update {name}"
