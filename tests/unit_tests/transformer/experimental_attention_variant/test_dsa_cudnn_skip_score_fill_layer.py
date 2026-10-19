# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""``dsa_indexer_skip_masked_score_fill`` on an MLA ``DSAttention`` layer (real kernels).

With the switch on, the cuDNN packed-THD indexer scorer is called with ``clean_logits=False``
and leaves every score column at or past a row's causal limit unspecified. The unspecified
columns are additionally poisoned (NaN, +inf) through a caller-owned score buffer, which is
the worst case a reader past the limit could observe. Each layout runs in two modes:

* deterministic mode: a full forward/backward of the layer must be bitwise identical to the
  run with the ``-inf`` prefill: top-k indices, attention output, indexer loss and every
  gradient.
* production (non-deterministic) mode: the radix top-k writes its selection in an
  unspecified slot order and the sparse backward accumulates with atomics, so two prefilled
  runs already differ. Each row's top-k index set, sorted by key index, and the score of
  every selected key must still be exactly equal; the output, indexer loss and gradients
  must agree within a tolerance that the prefilled rerun itself has to meet.

Layouts follow packed THD training: multi-sequence packs with a dummy tail sequence and
fixed-capacity ``cu_seqlens`` (``thd_max_packed_sequences``), at CP1 and zigzag CP4, and
dynamic-CP static slots where a microbatch runs on a CP2 or CP1 subgroup of a CP4 world with
one real sequence plus the dummy tail. The metadata is produced by ``pad_sequence_for_thd``,
the same helper the data path uses. Run with 4 ranks.
"""

import dataclasses
import functools
import math
from typing import Optional

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import TELinear, TENorm
from megatron.core.packed_seq_params import PackedSeqParams, pad_sequence_for_thd
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexer,
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
    DSAIndexerSubmodules,
    DSAttention,
    DSAttentionSubmodules,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import init_method_normal, scaled_init_method_normal
from tests.unit_tests.test_utilities import Utils

_WORLD_SIZE = 4


@dataclasses.dataclass(frozen=True)
class _Layout:
    """One packed THD microbatch as every rank of the realized CP group sees it."""

    model_cp: int  # CP size the model parallel state is initialized with
    realized_cp: int  # CP size of the group that runs this microbatch
    dynamic: bool  # realized group is a dynamic-CP subgroup (local_cp_size/cp_group set)
    local_rows: int  # static per-rank slot length (pad target)
    seq_lens: tuple  # real sequences; the dummy tail fills the slot
    max_num_seqs: int  # thd_max_packed_sequences, including the dummy sequence
    topk: int


_LAYOUTS = [
    pytest.param(_Layout(1, 1, False, 512, (200, 120, 136), 5, 128), id="cp1-3seq-dummy"),
    # topk == max_seqlen_k: the scorer's full-cover shortcut instead of the ranked top-k.
    pytest.param(_Layout(1, 1, False, 256, (152, 64), 4, 256), id="cp1-2seq-dummy-fullcover"),
    pytest.param(_Layout(4, 4, False, 256, (416, 336, 160), 5, 128), id="cp4-3seq-dummy"),
    pytest.param(_Layout(4, 2, True, 512, (920,), 2, 256), id="dcp2of4-slot"),
    pytest.param(_Layout(4, 1, True, 512, (488,), 2, 256), id="dcp1of4-slot"),
]


def _skip_unless_supported() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the cuDNN DSA backend")
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("the cuDNN frontend clean_logits scorer path needs SM100+")
    if Utils.world_size != _WORLD_SIZE:
        pytest.skip(f"requires exactly {_WORLD_SIZE} distributed ranks")
    try:
        from cudnn import DSA  # noqa: F401
        from flash_mla import flash_mla_sparse_fwd  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"cuDNN fused DSA dependencies are unavailable: {exc}")
    dsa_cudnn_kernels._ensure_dsa_namespace()
    if not dsa_cudnn_kernels._supports_indexer_clean_logits():
        pytest.skip("the installed cuDNN frontend has no indexer clean_logits switch")


def _make_config(layout: _Layout, *, deterministic: bool) -> MLATransformerConfig:
    """64-head DSA attention geometry (512+64 latent, 32x128 indexer), small hidden."""
    config = MLATransformerConfig(
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
        dsa_indexer_n_heads=32,
        dsa_indexer_head_dim=128,
        dsa_indexer_topk=layout.topk,
        dsa_indexer_topk_freq=4,
        dsa_indexer_skip_topk_offset=3,
        dsa_indexer_loss_coeff=0.01,
        dsa_indexer_use_sparse_loss=True,
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
        context_parallel_size=layout.model_cp,
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
    # Set after validation: deterministic mode otherwise asserts launcher-level settings.
    # Bitwise comparisons need the deterministic sparse backward and top-k tie-breaking;
    # production runs leave it off.
    config.deterministic_mode = deterministic
    return config


def _make_attention(config: MLATransformerConfig, pg_collection: ProcessGroupCollection):
    indexer_spec = ModuleSpec(
        module=DSAIndexer,
        submodules=DSAIndexerSubmodules(
            linear_wq_b=ModuleSpec(module=TELinear),
            linear_wk=ModuleSpec(module=TELinear),
            k_norm=ModuleSpec(module=TENorm),
            linear_weights_proj=ModuleSpec(module=TELinear),
        ),
    )
    attention = DSAttention(
        config=config,
        submodules=DSAttentionSubmodules(indexer=indexer_spec),
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        cp_comm_type="allgather",
        pg_collection=pg_collection,
    ).cuda()
    assert not attention.skip_topk, "layer 1 must select its own top-k"
    return attention.train()


def _packed_seq_params(layout: _Layout, cp_group, cp_rank: int) -> PackedSeqParams:
    """Static-slot THD metadata exactly as ``pad_sequence_for_thd`` emits it."""
    lengths = torch.tensor((0, *layout.seq_lens), dtype=torch.int32, device="cuda")
    cu_seqlens = lengths.cumsum(0, dtype=torch.int32)
    real_rows = int(cu_seqlens[-1])
    assert real_rows % layout.realized_cp == 0
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens,
        cu_seqlens_kv_padded=cu_seqlens,
        max_seqlen_q=max(layout.seq_lens),
        max_seqlen_kv=max(layout.seq_lens),
        local_cp_size=layout.realized_cp if layout.dynamic else None,
        cp_group=cp_group if layout.dynamic else None,
        cp_partition_mode="zigzag",
    )
    tokens = torch.zeros((1, real_rows // layout.realized_cp), dtype=torch.long, device="cuda")
    padded = pad_sequence_for_thd(
        tokens,
        None,
        None,
        None,
        params,
        target_len=layout.local_rows,
        max_num_seqs=layout.max_num_seqs,
        tail_padding_policy="append_dummy_seq",
        cp_size=layout.realized_cp,
        cp_rank=cp_rank,
    )[4]
    global_rows = layout.local_rows * layout.realized_cp
    expected = [0, *torch.tensor(layout.seq_lens).cumsum(0).tolist(), global_rows]
    expected += [global_rows] * (layout.max_num_seqs + 1 - len(expected))
    assert padded.cu_seqlens_q.tolist() == expected, padded.cu_seqlens_q.tolist()
    assert padded.max_seqlen_q == padded.max_seqlen_kv == global_rows
    if layout.dynamic:
        assert padded.local_cp_size == layout.realized_cp and padded.cp_group is cp_group
    return padded


def _make_inputs(config: MLATransformerConfig, layout: _Layout, rank: int) -> dict:
    """Rank-local absorbed-MLA query/key and indexer activations for this slot."""
    rows = layout.local_rows
    heads = config.num_attention_heads
    absorbed_dim = config.kv_lora_rank + config.qk_pos_emb_head_dim
    generator = torch.Generator(device="cuda")
    generator.manual_seed(2026 + rank)

    def randn(*shape, scale=1.0):
        return torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator) * scale

    return {
        "query": randn(rows, 1, heads, absorbed_dim),
        "key": randn(rows, 1, 1, absorbed_dim),
        "x": randn(rows, 1, config.hidden_size, scale=8.0),
        "qr": randn(rows, 1, config.q_lora_rank, scale=8.0),
        "up_v_weight": randn(heads, config.v_head_dim, config.kv_lora_rank),
    }


class _ScorerSpy:
    """Proxy for the cuDNN DSA namespace that records ``clean_logits`` and can poison.

    With a poison value, every ``clean_logits=False`` indexer call gets a caller-owned
    score buffer prefilled with that value, so each column the kernel leaves unwritten
    reads back as the poison.
    """

    def __init__(self, namespace):
        self._namespace = namespace
        self.poison: Optional[float] = None
        self.clean_logits_calls = []
        self.poison_visible = []
        real = namespace.indexer_forward_wrapper

        @functools.wraps(real)
        def indexer_forward_wrapper(q, k, w, *args, **kwargs):
            clean_logits = kwargs.get("clean_logits", True)
            self.clean_logits_calls.append(clean_logits)
            if clean_logits or self.poison is None:
                return real(q, k, w, *args, **kwargs)
            assert not args and kwargs.get("out") is None and q.ndim == 3, "THD scorer expected"
            columns = int(kwargs["max_seqlen_k"])
            # An unaligned caller buffer is staged by the frontend; keep the layouts aligned.
            assert columns % 4 == 0, f"max_seqlen_k={columns} must be a multiple of 4"
            kwargs["out"] = torch.full(
                (q.size(0), columns), self.poison, dtype=torch.float32, device=q.device
            )
            result = real(q, k, w, *args, **kwargs)
            scores = result["scores"]
            assert scores.data_ptr() == kwargs["out"].data_ptr()
            hit = torch.isnan(scores) if math.isnan(self.poison) else scores == self.poison
            self.poison_visible.append(bool(hit.any()))
            return result

        self.indexer_forward_wrapper = indexer_forward_wrapper

    def __getattr__(self, name):
        return getattr(self._namespace, name)


def _step(attention, inputs, packed_seq_params, topk_log, *, skip_fill: bool):
    """One forward/backward of the layer; returns top-k, output, indexer loss and grads."""
    attention.config.dsa_indexer_skip_masked_score_fill = skip_fill
    DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
    attention.zero_grad(set_to_none=True)
    topk_log.clear()
    # The indexer reads x and qr detached (its loss trains only the indexer), so only the
    # attention operands are leaves; the indexer gradients are checked on its parameters.
    leaves = {
        name: inputs[name].detach().clone().requires_grad_(True)
        for name in ("query", "key", "up_v_weight")
    }
    output = attention(
        query=leaves["query"],
        key=leaves["key"],
        value=None,
        attention_mask=None,
        x=inputs["x"],
        qr=inputs["qr"],
        attn_mask_type=AttnMaskType.causal,
        packed_seq_params=packed_seq_params,
        up_v_weight=leaves["up_v_weight"],
    )
    generator = torch.Generator(device="cuda")
    generator.manual_seed(7 + torch.distributed.get_rank())
    grad_output = torch.randn(output.shape, device="cuda", dtype=output.dtype, generator=generator)
    output.backward(grad_output)
    torch.cuda.synchronize()
    result = {"output": output.detach(), "indexer_loss": None}
    result["indexer_loss"] = DSAIndexerLossLoggingHelper.tracker["values"][0].detach().clone()
    for call, returned in enumerate(topk_log):
        for position, tensor in enumerate(returned):
            if tensor is not None:
                result[f"topk.call{call}.{position}"] = tensor
    for name, leaf in leaves.items():
        assert leaf.grad is not None, f"input {name} has no gradient"
        result[f"grad.{name}"] = leaf.grad.detach().clone()
    for name, parameter in attention.named_parameters():
        assert parameter.grad is not None, f"parameter {name} has no gradient"
        result[f"grad.{name}"] = parameter.grad.detach().clone()
    return result


# Production-mode tolerances. The top-k slot order changes the accumulation order of the
# sparse attention, and the sparse backward reduces with atomics, so two identical runs
# differ by a few bf16 ulps. A wrong or poisoned selection moves whole rows by O(1).
_PRODUCTION_REL_L2 = 1e-2  # ||actual - expected|| / ||expected|| per tensor
_PRODUCTION_PEAK = 2e-2  # max |actual - expected| / max |expected| per tensor
_PRODUCTION_LOSS_RTOL = 1e-3  # the indexer loss is an fp32 scalar


def _bitwise_equal(value: torch.Tensor, reference: torch.Tensor) -> bool:
    return torch.equal(value.reshape(-1).view(torch.uint8), reference.reshape(-1).view(torch.uint8))


def _assert_bitwise_equal(actual: dict, expected: dict, label: str) -> None:
    assert actual.keys() == expected.keys(), f"{label}: {sorted(actual)} != {sorted(expected)}"
    for name, reference in expected.items():
        value = actual[name]
        assert value.dtype == reference.dtype and value.shape == reference.shape, (
            f"{label} {name}: {value.dtype}{tuple(value.shape)} != "
            f"{reference.dtype}{tuple(reference.shape)}"
        )
        if not _bitwise_equal(value, reference):
            diff = (value.double() - reference.double()).abs()
            pytest.fail(
                f"{label}: {name} differs bitwise in "
                f"{int((value != reference).sum())}/{value.numel()} elements "
                f"(max |diff| {diff.nan_to_num(nan=float('inf')).max().item():.3e})"
            )


def _is_topk_indices(name: str) -> bool:
    return name.startswith("topk.") and name.endswith(".0")


def _sort_topk_by_key(result: dict) -> dict:
    """Sort each row's top-k selection by key index and carry the selected scores along.

    Without deterministic mode the cuDNN radix top-k writes its selection in an
    unspecified slot order. The selected set, its length and the score of every selected
    key are still reproducible. Invalid slots (``-1``) sort last.
    """
    canonical = dict(result)
    for name, indices in result.items():
        if not _is_topk_indices(name):
            continue
        invalid = torch.full_like(indices, torch.iinfo(indices.dtype).max)
        order = torch.where(indices >= 0, indices, invalid).argsort(dim=-1, stable=True)
        canonical[name] = torch.gather(indices, -1, order)
        scores_name = name.rsplit(".", 1)[0] + ".2"
        if scores_name in result:
            scores = result[scores_name]
            assert scores.shape == indices.shape, f"{scores_name} is not a top-k score payload"
            canonical[scores_name] = torch.gather(scores, -1, order)
    return canonical


def _slot_order_diffs(actual: dict, expected: dict) -> int:
    """Raw top-k slots that differ, i.e. how much the unspecified slot order moved."""
    return sum(
        int((actual[name] != expected[name]).sum()) for name in expected if _is_topk_indices(name)
    )


def _ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    numerator, denominator = float(numerator), float(denominator)
    if denominator > 0:
        return numerator / denominator
    return 0.0 if numerator == 0 else math.inf


def _production_mismatches(actual: dict, expected: dict, label: str):
    """Compare two production-mode runs whose top-k selections are sorted by key.

    Top-k indices, lengths and scores must be bitwise equal. Every other tensor must be
    finite and within the production tolerances. Returns the failures and, per compared
    tensor, ``(relative L2 error, peak error relative to the peak magnitude)``.
    """
    if actual.keys() != expected.keys():
        return [f"{label}: {sorted(actual)} != {sorted(expected)}"], {}
    failures, errors = [], {}
    for name, reference in expected.items():
        value = actual[name]
        if value.dtype != reference.dtype or value.shape != reference.shape:
            failures.append(
                f"{label} {name}: {value.dtype}{tuple(value.shape)} != "
                f"{reference.dtype}{tuple(reference.shape)}"
            )
            continue
        if name.startswith("topk."):
            if not _bitwise_equal(value, reference):
                failures.append(
                    f"{label}: sorted {name} differs in "
                    f"{int((value != reference).sum())}/{value.numel()} elements"
                )
            continue
        value64, reference64 = value.double(), reference.double()
        if not (torch.isfinite(value64).all() and torch.isfinite(reference64).all()):
            failures.append(f"{label}: {name} has non-finite values")
            continue
        diff = (value64 - reference64).abs()
        rel_l2 = _ratio(diff.norm(), reference64.norm())
        peak = _ratio(diff.max(), reference64.abs().max())
        errors[name] = (rel_l2, peak)
        if name == "indexer_loss":
            limits = (_PRODUCTION_LOSS_RTOL, _PRODUCTION_LOSS_RTOL)
        else:
            limits = (_PRODUCTION_REL_L2, _PRODUCTION_PEAK)
        if rel_l2 > limits[0] or peak > limits[1]:
            failures.append(
                f"{label}: {name} rel L2 {rel_l2:.3e} (limit {limits[0]:.0e}), "
                f"peak {peak:.3e} (limit {limits[1]:.0e})"
            )
    return failures, errors


def _worst(errors: dict) -> str:
    if not errors:
        return "no float tensors"
    l2_name = max(errors, key=lambda name: errors[name][0])
    peak_name = max(errors, key=lambda name: errors[name][1])
    loss = errors.get("indexer_loss", (math.nan,))[0]
    return (
        f"max rel L2 {errors[l2_name][0]:.2e} ({l2_name}), "
        f"max peak {errors[peak_name][1]:.2e} ({peak_name}), loss rel {loss:.2e}"
    )


def _initialize_model_parallel(layout: _Layout) -> None:
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        context_parallel_size=layout.model_cp,
        dynamic_context_parallel=layout.dynamic,
        min_dynamic_context_parallel_size=1,
    )


def _build_step(layout: _Layout, monkeypatch, *, deterministic: bool):
    """Build the layer and inputs for ``layout``; returns the step function and scorer spy.

    The model parallel state must already be initialized for ``layout``.
    """
    model_parallel_cuda_manual_seed(1234)
    torch.manual_seed(1234)
    DSAIndexerLossAutoScaler.set_loss_scale(torch.ones((), device="cuda"))
    pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
    if layout.dynamic:
        # What MLA installs for a dynamic-CP microbatch (packed_seq_params.cp_group).
        pg_collection.cp = parallel_state.get_dynamic_data_context_parallel_groups(
            group_size=layout.realized_cp
        )
    cp_group = pg_collection.cp
    assert cp_group.size() == layout.realized_cp
    config = _make_config(layout, deterministic=deterministic)
    attention = _make_attention(config, pg_collection)
    packed_seq_params = _packed_seq_params(layout, cp_group, cp_group.rank())
    inputs = _make_inputs(config, layout, torch.distributed.get_rank())

    spy = _ScorerSpy(dsa_cudnn_kernels._cudnn_dsa)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", spy)
    topk_log = []
    indexer_topk = dsa_cudnn_kernels._indexer_topk_bshd

    def recording_indexer_topk(*args, **kwargs):
        returned = indexer_topk(*args, **kwargs)
        topk_log.append(tuple(None if t is None else t.detach().clone() for t in returned))
        return returned

    monkeypatch.setattr(dsa_cudnn_kernels, "_indexer_topk_bshd", recording_indexer_topk)
    return functools.partial(_step, attention, inputs, packed_seq_params, topk_log), spy


def _run_skipped(step, spy, poison: Optional[float]) -> dict:
    """One step with the prefill skipped and the unspecified columns poisoned."""
    spy.poison = poison
    spy.clean_logits_calls.clear()
    spy.poison_visible.clear()
    skipped = step(skip_fill=True)
    # The switch reached the packed THD scorer, and only as clean_logits=False.
    assert spy.clean_logits_calls and not any(spy.clean_logits_calls)
    if poison is not None:
        # Masked columns kept the poison, so the run really read a dirty buffer.
        assert any(spy.poison_visible), f"poison {poison} never survived the scorer"
    return skipped


_POISONS = (None, float("nan"), float("inf"))


@pytest.mark.parametrize("layout", _LAYOUTS)
def test_skip_masked_score_fill_is_bitwise_identical(layout, monkeypatch):
    _skip_unless_supported()
    _initialize_model_parallel(layout)
    try:
        step, spy = _build_step(layout, monkeypatch, deterministic=True)
        filled = step(skip_fill=False)
        assert spy.clean_logits_calls and all(spy.clean_logits_calls)
        assert any(name.startswith("topk.") for name in filled), "no top-k was selected"
        # The comparison is only meaningful if the reference itself is reproducible.
        _assert_bitwise_equal(step(skip_fill=False), filled, "prefill rerun")

        for poison in _POISONS:
            skipped = _run_skipped(step, spy, poison)
            _assert_bitwise_equal(skipped, filled, f"skip fill, poison={poison}")
        print(
            f"[skip-score-fill] layout={layout} rank={torch.distributed.get_rank()} "
            f"scorer_calls={len(spy.clean_logits_calls)} compared={len(filled)} tensors"
        )
    finally:
        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("layout", _LAYOUTS)
def test_skip_masked_score_fill_matches_in_production_mode(layout, monkeypatch):
    """Without deterministic mode: exact sorted top-k, values within run-to-run noise."""
    _skip_unless_supported()
    _initialize_model_parallel(layout)
    try:
        step, spy = _build_step(layout, monkeypatch, deterministic=False)
        raw_filled = step(skip_fill=False)
        assert spy.clean_logits_calls and all(spy.clean_logits_calls)
        assert any(name.startswith("topk.") for name in raw_filled), "no top-k was selected"
        filled = _sort_topk_by_key(raw_filled)
        # The tolerances must hold between two prefilled runs, or they do not describe the
        # run-to-run noise of the non-deterministic kernels.
        raw_rerun = step(skip_fill=False)
        failures, errors = _production_mismatches(
            _sort_topk_by_key(raw_rerun), filled, "prefill rerun"
        )
        report = [
            f"prefill rerun: slot diffs {_slot_order_diffs(raw_rerun, raw_filled)}, "
            f"{_worst(errors)}"
        ]

        for poison in _POISONS:
            raw_skipped = _run_skipped(step, spy, poison)
            label = f"skip fill, poison={poison}"
            mismatches, errors = _production_mismatches(
                _sort_topk_by_key(raw_skipped), filled, label
            )
            failures += mismatches
            report.append(
                f"{label}: slot diffs {_slot_order_diffs(raw_skipped, raw_filled)}, "
                f"{_worst(errors)}"
            )
        print(
            f"[skip-score-fill-production] layout={layout} rank={torch.distributed.get_rank()} "
            f"compared={len(filled)} tensors; " + "; ".join(report)
        )
        assert not failures, "\n".join(failures)
    finally:
        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
        Utils.destroy_model_parallel()
