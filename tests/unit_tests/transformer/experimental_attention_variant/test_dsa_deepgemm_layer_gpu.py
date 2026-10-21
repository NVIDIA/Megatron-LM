# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""``dsa_indexer_scorer_backend='deepgemm'`` on GLM-like ``DSAttention`` layers (real kernels).

Needs SM 10.3 GPUs (SM 10.0 with ``MCORE_DEEPGEMM_VALIDATE_SM100=1``), a ``deep_gemm >= 2.8``
build, Transformer Engine MXFP8, the cuDNN frontend DSA (>= 1.29, else the scorer falls back)
and FlashMLA; each test states its rank count. The layers run the cuDNN fused DSA path with ``dsa_indexer_precision='mxfp8'``:

* checkpoint replay with the indexer loss (``dsa_indexer_reuse_checkpoint_topk`` off and on,
  1 rank): the replay that re-selects and the replay that reuses the first call's stash give
  the same output, loss and gradients, so the first call (indices only) and the replay (with
  scores) score bitwise-identical logits, and the stash protocol takes the BF16-recompute
  payload. Every run selects through a pinned top-k (the first call's selection, checked
  against each later call's logits): cuDNN's BF16 radix top-k does not repeat its choice
  among tied logits, which is why the config requires the reuse for a recompute that replays
  the indexer;
* frozen indexer (loss off, 1 rank, the same pinned top-k): eager, replay-rerun and
  replay-reuse agree;
* production mode against the cuDNN MXFP8 scorer (4 ranks; CP1, zigzag CP4 and dynamic-CP
  layouts of the skip-fill layer test, plus a wide CP4 layout where DeepSelect takes the BF16
  top-k): DeepGEMM ran with no fallback, both scorers fed the indexer loss the FP32
  recompute (the rule follows the MXFP8 precision, not the backend), the selections overlap,
  the indexer loss and gradients track, and a rerun selects nearly the same keys (overlap
  >= 0.999: cuDNN's BF16 radix top-k may keep other tied keys; the kernel-level behaviour,
  bitwise repeats for DeepSelect and tie-only differences for cuDNN, is checked in
  ``test_dsa_deepgemm_scorer_gpu.py``);
* CUDA graph capture/replay of a CP2 layer fed by the prebuilt packed-CP layout tables (2
  ranks): DeepGEMM ran inside the capture; after an in-place refresh to another packing too,
  the replay scores bitwise the logits eager scored and its own cuDNN BF16 radix selection
  equals eager's tie-aware (with no key taken twice); given eager's selection (fed through
  buffers allocated before the capture) its output, dquery and dup_v_weight equal eager's
  bitwise. dkey, which cuDNN's sparse-attention backward sums with FP32 atomics, is checked per
  element: each rank's dkey partial (the gradient at the key's CP all-gather) within that
  summation order's bound, each run's dkey as one BF16 rounding of the summed partials, and
  dkey within the partials' differences. The captured step runs with
  ``set_sync_debug_mode('error')``. Every check is collected and raised on both ranks at the
  next synchronization point, before any rank enters a captured collective, and the graph is
  released before the process groups go.

Every test warms DeepGEMM up after model-parallel init has set the rank's device (DeepGEMM
runs on one device per process), and the 1-rank tests alias the padded cu_seqlens of Q and K
as the THD schedulers do (the MXFP8 scorer needs that proof).

Lines starting with ``[deepgemm-layer]`` report what each rank compared.
"""

import functools
import os

import pytest
import torch

from megatron.core.context_parallel_layout import prebuild_thd_cp_partition_routes
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.quantization import indexer_quantization
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant import (
    dsa,
    dsa_cudnn_kernels,
    dsa_deepgemm_scorer,
    dsa_deepselect_topk,
)
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
)
from megatron.core.transformer.experimental_attention_variant.dsa_layout import (
    copy_prebuilt_packed_cp_layout_,
    prebuild_packed_cp_layout,
)
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant import (
    test_dsa_checkpoint_topk_reuse as reuse,
)
from tests.unit_tests.transformer.experimental_attention_variant import (
    test_dsa_cudnn_skip_score_fill_layer as layer_helpers,
)

_VALIDATE_SM100_ENV = "MCORE_DEEPGEMM_VALIDATE_SM100"


def _report(message: str) -> None:
    print(f"[deepgemm-layer] {message}", flush=True)


def _require_deepgemm_layer(world_size: int, monkeypatch) -> None:
    reuse._skip_if_cudnn_dsa_unavailable(world_size)
    sm = tuple(torch.cuda.get_device_capability())
    if sm not in dsa_deepgemm_scorer.KNOWN_SM:
        pytest.skip("DeepGEMM's MQA logits run on SM 10.x only")
    if sm not in dsa_deepgemm_scorer.SUPPORTED_SM:
        if os.environ.get(_VALIDATE_SM100_ENV) != "1":
            pytest.skip(f"SM {sm} is not validated; set {_VALIDATE_SM100_ENV}=1 to validate it")
        monkeypatch.setattr(
            dsa_deepgemm_scorer, "SUPPORTED_SM", dsa_deepgemm_scorer.SUPPORTED_SM | {sm}
        )
    if not indexer_quantization.HAVE_TE_MXFP8:
        pytest.skip("Transformer Engine MXFP8 quantization is unavailable")
    if dsa_deepgemm_scorer._load_provider() is None:
        pytest.skip(dsa_deepgemm_scorer._IMPORT_STATE["error"])
    dsa_cudnn_kernels._ensure_dsa_namespace()
    if not dsa_cudnn_kernels._cudnn_frontend_at_least(
        dsa_cudnn_kernels._TOPK_OOB_LANE_GUARD_FE_VERSION
    ):
        pytest.skip("cuDNN frontend < 1.29: the DeepGEMM scorer falls back there")


def _warm_up_this_rank(indexer_heads: int) -> None:
    """Warm DeepGEMM up on this rank's device, once model-parallel init has set it.

    DeepGEMM runs on one device per process, the first one that calls it (deep_jit keeps one
    kernel handle per kernel and process): a warm-up before ``torch.cuda.set_device`` would
    claim ``cuda:0`` on every rank and leave ranks 1.. on the cuDNN MXFP8 scorer.
    """
    device = torch.device("cuda", torch.cuda.current_device())
    dsa_deepgemm_scorer.warmup(device, indexer_heads)
    owner = dsa_deepgemm_scorer._PROCESS_DEVICE.get("index")
    assert owner == device.index, f"DeepGEMM runs on cuda:{owner}, this rank on {device}"


def _alias_padded_cu_seqlens(inputs: dict) -> dict:
    """Give Q and K one padded cu_seqlens tensor, as the THD schedulers do.

    Only the general packed THD scorer runs MXFP8, and it proves that Q and K share their
    boundaries by aliasing (no host read). ``reuse._make_inputs`` clones the padded tensors,
    which sends its one-sequence pack to the packed_single_segments plan, where MXFP8 raises.
    """
    params = inputs["packed_seq_params"]
    params.cu_seqlens_q_padded = params.cu_seqlens_kv_padded = params.cu_seqlens_q
    return inputs


class _DeepGemmSpy:
    """Counts DeepGEMM logits launches, backend log outcomes and selected-score recomputes."""

    def __init__(self, monkeypatch):
        self.launches = 0
        self.captured_launches = 0
        self.reasons = []
        self.recomputes = 0
        real_logits = dsa_deepgemm_scorer.mqa_logits
        real_log = dsa_deepgemm_scorer.log_scorer_backend_once
        real_recompute = dsa_cudnn_kernels._recompute_selected_indexer_log_probs

        @functools.wraps(real_logits)
        def mqa_logits(*args, **kwargs):
            self.launches += 1
            self.captured_launches += int(torch.cuda.is_current_stream_capturing())
            return real_logits(*args, **kwargs)

        def log_once(reason):
            self.reasons.append(reason)
            return real_log(reason)

        @functools.wraps(real_recompute)
        def recompute(*args, **kwargs):
            self.recomputes += 1
            return real_recompute(*args, **kwargs)

        monkeypatch.setattr(dsa_deepgemm_scorer, "mqa_logits", mqa_logits)
        monkeypatch.setattr(dsa_deepgemm_scorer, "log_scorer_backend_once", log_once)
        monkeypatch.setattr(dsa_cudnn_kernels, "_recompute_selected_indexer_log_probs", recompute)

    def reset(self):
        self.launches = self.captured_launches = self.recomputes = 0
        self.reasons.clear()


class _PinnedTopk:
    """Every DeepGEMM-path top-k of a test returns the selection of its first call.

    The runs below select on bitwise-identical DeepGEMM logits, but cuDNN's BF16 radix top-k
    does not repeat its choice among tied BF16 logits (GB300, Slurm job 317672; the reason the
    config requires ``dsa_indexer_reuse_checkpoint_topk`` for a recompute that replays the
    indexer). Pinning the selection lets the checkpoint and stash protocols compare
    bitwise; every later call checks that its logits equal the first call's on each in-row
    column, so the pin never hides a scoring difference.
    """

    def __init__(self, monkeypatch):
        self.calls = 0
        self.first = None
        real = dsa_cudnn_kernels._deepgemm_logits_topk

        def pinned(logits, lengths, topk_k, topk_backend):
            self.calls += 1
            columns = torch.arange(logits.size(1), device=logits.device).view(1, -1)
            seen = torch.where(
                columns < lengths.view(-1, 1).long(), logits, torch.zeros_like(logits)
            )
            if self.first is None:
                indices, values = real(logits, lengths, topk_k, topk_backend)
                self.first = (seen.clone(), lengths.clone(), indices.clone(), values.clone())
                return indices, values
            first_seen, first_lengths, indices, values = self.first
            assert torch.equal(lengths, first_lengths), "the pinned top-k saw other row lengths"
            assert torch.equal(seen, first_seen), "the pinned top-k saw other logits"
            return indices.clone(), values.clone()

        monkeypatch.setattr(dsa_cudnn_kernels, "_deepgemm_logits_topk", pinned)


class _GraphTopkRecorder:
    """Records every DeepGEMM-path top-k; inside a capture it feeds the layer a static selection.

    cuDNN's BF16 radix top-k does not repeat its choice among tied BF16 logits (G5's documented
    negative), so an eager step and a graph replay of the same inputs may keep other keys tied
    at a row's K-th value, and on this layer one swapped key moves the output and gradients by
    far more than the comparison tolerance. Outside capture the real selection passes through
    and the in-row logits, lengths, indices and values are recorded (cloned). Inside capture
    the real top-k still runs and its in-row logits, lengths and selection are kept in static
    tensors that every replay refreshes, but the layer gets ``pinned``, buffers the test
    fills with the matching eager step's selection before each replay. So a replay is checked
    three ways: its logits bitwise against eager's, its own selection tie-aware against eager's,
    and its output and gradients against eager's given eager's selection.

    ``pinned`` is allocated before the capture (``allocate_pinned``) and never inside it: a
    capture allocation comes from the graph's private pool, which hands out memory that
    tensors freed earlier in the same capture used, so every replay rewrites it before the
    layer reads it (GB300, oci-jhb job 1171499: after a replay the pinned indices read 0 in
    12131-14592 of 16384 slots, and the output was off by up to 135).
    """

    def __init__(self, monkeypatch):
        self.eager = []
        self.captured = None
        self.pinned = None
        real = dsa_cudnn_kernels._deepgemm_logits_topk

        def recorded(logits, lengths, topk_k, topk_backend):
            indices, values = real(logits, lengths, topk_k, topk_backend)
            columns = torch.arange(logits.size(1), device=logits.device).view(1, -1)
            seen = torch.where(
                columns < lengths.view(-1, 1).long(), logits, torch.zeros_like(logits)
            )
            if torch.cuda.is_current_stream_capturing():
                assert self.captured is None, "one DeepGEMM top-k per captured step"
                assert self.pinned is not None, "allocate the pinned selection before the capture"
                for pinned, produced in zip(self.pinned, (indices, values)):
                    assert (pinned.shape, pinned.dtype) == (produced.shape, produced.dtype), (
                        f"pinned {tuple(pinned.shape)} {pinned.dtype}, the top-k gave "
                        f"{tuple(produced.shape)} {produced.dtype}"
                    )
                self.captured = (seen, lengths.clone(), indices, values)
                return self.pinned
            self.eager.append((seen.clone(), lengths.clone(), indices.clone(), values.clone()))
            return indices, values

        monkeypatch.setattr(dsa_cudnn_kernels, "_deepgemm_logits_topk", recorded)

    def take_eager(self):
        """The one selection the last eager step recorded."""
        assert len(self.eager) == 1, f"{len(self.eager)} DeepGEMM top-k calls in one step"
        return self.eager.pop()

    def allocate_pinned(self, selection) -> None:
        """Allocate the layer's static selection, shaped like ``selection``, before the capture."""
        assert not torch.cuda.is_current_stream_capturing(), "allocate pinned before the capture"
        self.pinned = tuple(torch.empty_like(source) for source in selection[2:])

    def pin(self, selection) -> None:
        """Feed the next replays ``selection`` (an eager record) as the layer's top-k."""
        for target, source in zip(self.pinned, selection[2:]):
            assert target.shape == source.shape, (tuple(target.shape), tuple(source.shape))
            target.copy_(source)

    def compare_replay(self, selection, label) -> int:
        """The replay's logits bitwise and its own selection tie-aware against ``selection``.

        Returns the number of rows whose replay selection kept other keys (tied ones only).
        """
        seen, lengths, indices, _values = self.captured
        want_seen, want_lengths, want_indices, _ = selection
        assert torch.equal(lengths, want_lengths), f"{label}: row lengths differ"
        differ = seen.view(torch.int16) != want_seen.view(torch.int16)
        assert not bool(differ.any()), (
            f"{label}: the replay scored other logits on {int(differ.sum())} in-row elements, "
            f"max |diff| {(seen.float() - want_seen.float()).abs().max().item():.3e}"
        )
        # Counts, not slot positions: the cuDNN radix need not keep its slot order.
        counts, want_counts = (indices >= 0).sum(dim=-1), (want_indices >= 0).sum(dim=-1)
        assert torch.equal(counts, want_counts), (
            f"{label}: {int((counts != want_counts).sum())} of {counts.numel()} rows selected "
            "another number of keys"
        )
        # The value multisets below cannot tell one tied key taken twice from two tied keys.
        valid_slots = lengths.clamp(max=indices.size(-1))
        for name, chosen in (("replay", indices), ("eager", want_indices)):
            repeated = _repeated_key_rows(chosen, valid_slots)
            assert not repeated, f"{label}: {repeated} rows of the {name} selection repeat a key"

        def selected(chosen):
            values = torch.gather(seen.float(), 1, chosen.long().clamp(min=0))
            return torch.where(chosen >= 0, values, float("-inf")).sort(dim=-1).values

        assert torch.equal(
            selected(indices), selected(want_indices)
        ), f"{label}: the replay selected keys that are not ties at a row's cutoff"
        keys, want_keys = indices.sort(dim=-1).values, want_indices.sort(dim=-1).values
        return int((keys != want_keys).any(dim=-1).sum())


def _repeated_key_rows(indices, valid_slots) -> int:
    """Rows of ``indices`` (``[rows, K]``) whose valid slots name some key twice.

    A slot is valid below ``valid_slots[row]`` and when its index is not negative. A tie-aware
    comparison compares the selected values as a multiset, which a top-k that took one tied
    key twice instead of two tied keys would pass; this check does not.
    """
    slots = torch.arange(indices.size(-1), device=indices.device).view(1, -1)
    valid = (slots < valid_slots.view(-1, 1).long()) & (indices >= 0)
    keys = torch.where(valid, indices.long(), -1).sort(dim=-1).values
    return int(((keys[:, 1:] == keys[:, :-1]) & (keys[:, 1:] >= 0)).any(dim=-1).sum())


class _KeyPartials:
    """Records this rank's dkey partial: the gradient at the layer's CP all-gather of the key.

    The gathered key (rank-major rows) is permuted into the global key order and read by
    cuDNN's sparse attention, whose backward returns dKV in BF16 from an FP32 atomic
    accumulator; the permutation's backward gathers it back exactly. So the hooked gradient is
    that BF16 dKV in gathered order, the inverse key reorder included, and the gather's
    backward, a BF16 CP reduce-scatter, sums the ranks' partials into each rank's dkey: rank r
    gets the gathered rows ``[r * _GRAPH_LOCAL_ROWS, (r + 1) * _GRAPH_LOCAL_ROWS)``. Outside
    capture each step appends a copy to ``eager``; inside capture ``captured`` is a static copy
    that every replay refreshes.
    """

    def __init__(self, monkeypatch, key_dim: int):
        self.eager = []
        self.captured = None
        real = dsa.gather_from_sequence_parallel_region

        def gather(input_, *args, **kwargs):
            output = real(input_, *args, **kwargs)
            if (
                output.requires_grad
                and output.size(0) == _GRAPH_GLOBAL_ROWS
                and output.size(-1) == key_dim
            ):
                output.register_hook(self._record)
            return output

        monkeypatch.setattr(dsa, "gather_from_sequence_parallel_region", gather)

    def _record(self, grad) -> None:
        partial = grad.detach().clone()
        if torch.cuda.is_current_stream_capturing():
            assert self.captured is None, "one key gather per captured step"
            self.captured = partial
        else:
            self.eager.append(partial)

    def take_eager(self):
        """The partial the last eager step recorded."""
        assert len(self.eager) == 1, f"{len(self.eager)} key gathers in one step"
        return self.eager.pop()


def _bf16_ulp(values: torch.Tensor) -> torch.Tensor:
    """Spacing of BF16 numbers at ``|values|``: ``2**(floor(log2|v|) - 7)``, and ``2**-133``
    (the subnormal spacing) below ``2**-126``, zero included. Built from the exponent bits, so
    it is exact on any device."""
    magnitude = values.float().abs()
    power = (magnitude.view(torch.int32) & 0x7F800000).view(torch.float32)
    return torch.where(
        magnitude < 2.0**-126, torch.full_like(magnitude, 2.0**-133), power * 2.0**-7
    )


# FP32 reorder term of a dkey partial element. cuDNN's backward (FE 1.31 dsa_bwd_sm100) runs one
# CTA per (query row, block of 64 heads) and atomically adds its FP32 dKV tile, so with 64 heads an
# element takes at most one add per local query row (_GRAPH_LOCAL_ROWS = 256). Two orders of n
# adds differ by at most 2 gamma_(n-1) sum|c| < 2**-15 sum|c| here. Assumed, not proven: the
# contributions to each element of a key row add up in magnitude to at most 64 times that row's
# largest |partial| M_row, so the term is at most 2**-9 M_row.
_PARTIAL_REORDER_PER_ROW_SCALE = 2.0**-9


def _partial_order_bound(replay, eager):
    """Elementwise bound on ``|replay - eager|`` for one rank's dkey partial (``[rows, ..., D]``).

    Both are the BF16 rounding of an FP32 sum of the same contributions in two atomic orders:
    ``|bf16(s) - bf16(s')| <= |s - s'| + (ulp(s) + ulp(s')) / 2``, each ulp at most that of the
    larger BF16 result, and ``|s - s'|`` at most ``_PARTIAL_REORDER_PER_ROW_SCALE`` times the
    largest ``|partial|`` of either run in that key row.
    """
    larger = torch.maximum(replay.float().abs(), eager.float().abs())
    row_scale = larger.reshape(larger.size(0), -1).amax(dim=1)
    row_scale = row_scale.view(-1, *([1] * (larger.dim() - 1)))
    return _bf16_ulp(larger) + _PARTIAL_REORDER_PER_ROW_SCALE * row_scale


def _reduce_scatter_bound(dkey, summed):
    """Bound on ``|dkey - summed|`` within one run, ``summed`` the FP32 sum of the partials.

    The reduce-scatter of two ranks rounds the sum of two BF16 partials to BF16 once: half a
    BF16 ulp. ``summed`` is exact unless the partials' exponents differ by more than 16, and a
    sum taken through FP32 first rounds twice; ``2**-16`` ulp covers both.
    """
    assert _GRAPH_CP == 2, "one rounding models the reduce-scatter of two ranks only"
    larger = torch.maximum(dkey.float().abs(), summed.abs())
    return _bf16_ulp(larger) * (0.5 + 2.0**-16)


def _dkey_bound(replay, eager, partial_diff):
    """Bound on ``|replay - eager|`` for this rank's dkey rows, given ``partial_diff``: the summed
    ``|replay - eager|`` of every rank's partial on those rows. Each run's dkey lies within
    ``_reduce_scatter_bound`` of its summed partials."""
    larger = torch.maximum(replay.float().abs(), eager.float().abs())
    return partial_diff + _bf16_ulp(larger) * (1 + 2.0**-15)


def _bounded_mismatch(actual, expected, bound, label):
    """(failure or ``None``, statistics) for ``|actual - expected| <= bound`` elementwise.

    A non-finite value on either side fails.
    """
    diff = (actual.float() - expected.float()).abs()
    over = ~(diff <= bound) | ~torch.isfinite(actual) | ~torch.isfinite(expected)
    stats = (
        f"{label} {int((diff > 0).sum())}/{diff.numel()} differ, max |diff| "
        f"{diff.max().item():.3e}, max |diff|/bound {(diff / bound).max().item():.3f}"
    )
    if bool(over.any()):
        return f"{label}: {int(over.sum())} elements past the bound; {stats}", stats
    return None, stats


def _mismatch(actual, expected, label):
    """``None`` when ``actual`` equals ``expected`` exactly, else the mismatch statistics."""
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return (
            f"{label}: {tuple(actual.shape)} {actual.dtype} against "
            f"{tuple(expected.shape)} {expected.dtype}"
        )
    if torch.equal(actual, expected):
        return None
    diff = (actual.float() - expected.float()).abs()
    return (
        f"{label}: {int((actual != expected).sum())} of {actual.numel()} elements differ, max "
        f"|diff| {diff.max().item():.3e}, max |expected| {expected.float().abs().max().item():.3e}"
    )


class _CollectedFailures:
    """The graph test's failed checks, raised on every rank of ``group`` at once.

    A rank that raised alone would leave the others waiting in their next collective, a captured
    one inside a replay included (the original "F, then a hang"). So the test collects every
    check and raises only at its synchronization points, where all ranks reduce the count.
    """

    def __init__(self, group, device="cuda"):
        self.group = group
        self.device = device
        self.messages = []
        self.cause = None

    def add(self, message) -> None:
        """Collect ``message`` unless it is ``None``."""
        if message:
            self.messages.append(message)

    def check(self, condition, message) -> None:
        """Collect ``message`` when ``condition`` is false."""
        if not condition:
            self.messages.append(message)

    def run(self, check, *args):
        """``check(*args)``, with an AssertionError collected (then ``None`` is returned)."""
        try:
            return check(*args)
        except AssertionError as error:
            self.messages.append(str(error) or repr(error))
            self.cause = self.cause or error
            return None

    def raise_on_any_rank(self, where) -> None:
        """A collective: raise on every rank when any rank collected a failure."""
        count = torch.tensor([len(self.messages)], dtype=torch.int64, device=self.device)
        torch.distributed.all_reduce(count, group=self.group)
        total = int(count.item())
        if total:
            here = "; ".join(self.messages) if self.messages else "none"
            raise AssertionError(
                f"{where}: {total} failed checks on the CP ranks; on this rank: {here}"
            ) from self.cause


def _use_deepgemm(config, backend: str = "deepgemm") -> None:
    """Switch a validated config to the MXFP8 scorer of ``backend`` (heads/dim/loss fit)."""
    config.dsa_indexer_precision = "mxfp8"
    config.dsa_indexer_scorer_backend = backend


# ------------------------------------------------------------------- checkpoint replay (1 rank)
@pytest.mark.parametrize("stash", ["used", "declined"])
def test_checkpoint_replay_with_the_loss_reselects_or_reuses_the_same_keys(stash, monkeypatch):
    _require_deepgemm_layer(1, monkeypatch)
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
    try:
        _warm_up_this_rank(indexer_heads=64)
        model_parallel_cuda_manual_seed(1234)
        torch.manual_seed(1234)
        DSAIndexerLossAutoScaler.set_loss_scale(torch.ones((), device="cuda"))
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        config = reuse._make_config(tp=1, cp=1, deterministic=False)
        _use_deepgemm(config)
        attention = reuse._make_attention(config, pg_collection)
        inputs = _alias_padded_cu_seqlens(reuse._make_inputs(config, tp_rank=0, cp_rank=0, cp=1))
        spy = _DeepGemmSpy(monkeypatch)
        pin = _PinnedTopk(monkeypatch)

        # A re-selecting loss replay (rejected by the config for deepgemm, built here on a
        # validated config) is the reference, given the pinned selection.
        config.dsa_indexer_reuse_checkpoint_topk = False
        baseline = reuse._checkpointed_step(attention, inputs, grad_output_seed=7)
        # First call (indices only) and replay (with scores) each ran DeepGEMM once; only the
        # replay recomputed the selected scores for the loss.
        assert spy.launches == 2 and spy.recomputes == 1 and set(spy.reasons) == {None}

        spy.reset()
        if stash == "declined":
            monkeypatch.setattr(
                dsa_cudnn_kernels, "run_fused_qk_topk_with_scores", lambda **_kwargs: None
            )
        config.dsa_indexer_reuse_checkpoint_topk = True
        reused = reuse._checkpointed_step(attention, inputs, grad_output_seed=7)
        if stash == "used":
            assert spy.launches == 1 and spy.recomputes == 1
        else:
            assert spy.launches == 2 and spy.recomputes == 1

        assert pin.calls == spy.launches + 2  # every DeepGEMM call ran the (pinned) top-k
        baseline_output, baseline_loss, baseline_grads = baseline
        output, loss, grads = reused
        torch.testing.assert_close(output, baseline_output, rtol=0, atol=0)
        torch.testing.assert_close(loss, baseline_loss, rtol=1e-4, atol=1e-7)
        for name, grad in grads.items():
            reuse._assert_similar(grad, baseline_grads[name], label=name, eps=1e-4)
        _report(f"checkpoint replay stash={stash}: loss {loss.item():.6e} matches")
    finally:
        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
        Utils.destroy_model_parallel()


def test_frozen_checkpoint_replay_reuses_the_first_call_indices(monkeypatch):
    _require_deepgemm_layer(1, monkeypatch)
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
    try:
        _warm_up_this_rank(indexer_heads=64)
        model_parallel_cuda_manual_seed(1234)
        torch.manual_seed(1234)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        config = reuse._make_config(tp=1, cp=1, deterministic=False)
        config.dsa_indexer_loss_coeff = 0.0
        _use_deepgemm(config)
        attention = reuse._make_attention(config, pg_collection)
        inputs = _alias_padded_cu_seqlens(reuse._make_inputs(config, tp_rank=0, cp_rank=0, cp=1))
        spy = _DeepGemmSpy(monkeypatch)
        pin = _PinnedTopk(monkeypatch)
        runs = {}
        for name, checkpoint, reuse_topk in (
            ("eager", False, True),
            ("replay-rerun", True, False),
            ("replay-reuse", True, True),
        ):
            config.dsa_indexer_reuse_checkpoint_topk = reuse_topk
            spy.reset()
            output, grads = reuse._frozen_step(
                attention, inputs, grad_output_seed=7, checkpoint=checkpoint
            )
            runs[name] = (output, grads, spy.launches, spy.recomputes)
        assert [runs[name][2:] for name in runs] == [(1, 0), (2, 0), (1, 0)]
        assert pin.calls == 4
        for name in ("replay-rerun", "replay-reuse"):
            torch.testing.assert_close(runs[name][0], runs["eager"][0], rtol=0, atol=0)
            for grad_name, grad in runs[name][1].items():
                reuse._assert_similar(
                    grad, runs["eager"][1][grad_name], label=f"{name} {grad_name}", eps=1e-4
                )
        _report("frozen replay: rerun and reuse match eager")
    finally:
        Utils.destroy_model_parallel()


# --------------------------------------------------------- production against FE MXFP8 (4 ranks)
_WIDE = layer_helpers._Layout(4, 4, False, 32768, (65536, 40000, 16000), 5, 2048)
_PRODUCTION_LAYOUTS = [
    pytest.param(param.values[0], id=param.id) for param in layer_helpers._LAYOUTS
] + [pytest.param(_WIDE, id="cp4-3seq-dummy-wide")]


def _overlap(actual: dict, expected: dict) -> list:
    """Per top-k call: mean |A ∩ B| / |B| over rows, with equal lengths required."""
    means = []
    for name in sorted(n for n in expected if layer_helpers._is_topk_indices(n)):
        call = name.rsplit(".", 1)[0]
        lengths = expected[f"{call}.1"].reshape(-1)
        assert torch.equal(actual[f"{call}.1"].reshape(-1), lengths), f"{call} lengths differ"
        a = actual[name].reshape(lengths.numel(), -1).long()
        b = expected[name].reshape(lengths.numel(), -1).long()
        keys = int(max(a.max(), b.max())) + 2
        member = torch.zeros((a.size(0), keys), dtype=torch.bool, device=a.device)
        member.scatter_(1, torch.where(b >= 0, b, keys - 1), True)
        member[:, keys - 1] = False
        hits = torch.gather(member, 1, torch.where(a >= 0, a, keys - 1)).sum(dim=1)
        keep = lengths > 0
        means.append((hits[keep].float() / lengths[keep].float()).mean().item())
    return means


@pytest.mark.parametrize("layout", _PRODUCTION_LAYOUTS)
def test_deepgemm_layer_tracks_the_fe_mxfp8_scorer(layout, monkeypatch):
    _require_deepgemm_layer(4, monkeypatch)
    if layout is _WIDE and torch.cuda.get_device_properties(0).total_memory < 200 * 1024**3:
        pytest.skip("the wide layout needs a GPU with at least 200 GiB (GB300)")
    layer_helpers._initialize_model_parallel(layout)
    try:
        _warm_up_this_rank(indexer_heads=32)
        step, _scorer_spy = layer_helpers._build_step(layout, monkeypatch, deterministic=False)
        attention = step.args[0]
        if layout is _WIDE and dsa_deepselect_topk._load_provider() is not None:
            attention.config.dsa_indexer_topk_backend = "deepselect"
        spy = _DeepGemmSpy(monkeypatch)

        _use_deepgemm(attention.config, "cudnn")
        fe_mxfp8 = layer_helpers._sort_topk_by_key(step(skip_fill=True))
        assert spy.launches == 0
        # MXFP8 compute: the FE scorer's loss takes the same FP32 recompute as DeepGEMM's, so
        # the two arms differ only in which keys they select.
        assert spy.recomputes > 0, "the FE MXFP8 indexer loss never took the BF16 recompute"
        spy.reset()
        _use_deepgemm(attention.config, "deepgemm")
        raw = step(skip_fill=True)
        dense_calls = sum(1 for name in raw if layer_helpers._is_topk_indices(name))
        # A call whose rows all fit in the top-k and keeps no scores skips DeepGEMM.
        assert 0 < spy.launches <= dense_calls and set(spy.reasons) == {None}, spy.reasons
        assert spy.recomputes > 0, "the indexer loss never took the BF16 recompute"
        deepgemm = layer_helpers._sort_topk_by_key(raw)
        overlaps = _overlap(deepgemm, fe_mxfp8)
        assert min(overlaps) >= 0.98, overlaps

        loss_rel = (
            (deepgemm["indexer_loss"] - fe_mxfp8["indexer_loss"]).abs()
            / fe_mxfp8["indexer_loss"].abs()
        ).item()
        assert loss_rel < 0.1, loss_rel
        for name, value in deepgemm.items():
            if not name.startswith("grad."):
                continue
            assert torch.isfinite(value).all(), name
            cosine = torch.nn.functional.cosine_similarity(
                value.double().flatten(), fe_mxfp8[name].double().flatten(), dim=0
            ).item()
            assert cosine > 0.9, (name, cosine)

        # The FP32 recompute payload separates keys that tied in BF16, so compare the rerun
        # by the selected keys, not by the payload.
        rerun_overlap = _overlap(layer_helpers._sort_topk_by_key(step(skip_fill=True)), deepgemm)
        assert min(rerun_overlap) >= 0.999, rerun_overlap
        _report(
            f"layout={layout} rank={torch.distributed.get_rank()} top-k calls={dense_calls} "
            f"overlap with FE MXFP8 min {min(overlaps):.4f}, loss rel {loss_rel:.3e}, "
            f"rerun overlap min {min(rerun_overlap):.5f}"
        )
    finally:
        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
        Utils.destroy_model_parallel()


# -------------------------------------------------------- CUDA graph with prebuilt layout (2 ranks)
_GRAPH_CP = 2
_GRAPH_LOCAL_ROWS = 256
_GRAPH_GLOBAL_ROWS = _GRAPH_LOCAL_ROWS * _GRAPH_CP
_PACK_A = (0, 128, 384, _GRAPH_GLOBAL_ROWS)
_PACK_B = (0, 256, 256, _GRAPH_GLOBAL_ROWS)
_GRAD_INPUTS = ("query", "key", "up_v_weight")


def _graph_params(boundaries, cp_group) -> PackedSeqParams:
    cu_seqlens = torch.tensor(boundaries, dtype=torch.int32, device="cuda")
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=_GRAPH_GLOBAL_ROWS,
        max_seqlen_kv=_GRAPH_GLOBAL_ROWS,
        cp_group=cp_group,
    )
    prebuild_thd_cp_partition_routes(params, cp_group)
    assert prebuild_packed_cp_layout(params) is not None
    return params


def _graph_inputs(config, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    absorbed_dim = config.kv_lora_rank + config.qk_pos_emb_head_dim

    def randn(*shape, scale=1.0):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16, generator=generator) * scale

    heads = config.num_attention_heads
    return {
        "query": randn(_GRAPH_LOCAL_ROWS, 1, heads, absorbed_dim),
        "key": randn(_GRAPH_LOCAL_ROWS, 1, 1, absorbed_dim),
        "x": randn(_GRAPH_LOCAL_ROWS, 1, config.hidden_size, scale=8.0),
        "qr": randn(_GRAPH_LOCAL_ROWS, 1, config.q_lora_rank, scale=8.0),
        "up_v_weight": randn(heads, config.v_head_dim, config.kv_lora_rank),
        "grad_output": randn(_GRAPH_LOCAL_ROWS, 1, heads * config.v_head_dim),
    }


def _graph_step(attention, inputs, params):
    output = attention(
        query=inputs["query"],
        key=inputs["key"],
        value=None,
        attention_mask=None,
        x=inputs["x"],
        qr=inputs["qr"],
        attn_mask_type=AttnMaskType.causal,
        packed_seq_params=params,
        up_v_weight=inputs["up_v_weight"],
    )
    output.backward(inputs["grad_output"].view_as(output))
    return output


def _graph_eager(attention, inputs, params):
    leaves = {name: value.detach().clone() for name, value in inputs.items()}
    for name in _GRAD_INPUTS:
        leaves[name].requires_grad_(True)
    output = _graph_step(attention, leaves, params)
    torch.cuda.synchronize()
    return output.detach().clone(), {name: leaves[name].grad.clone() for name in _GRAD_INPUTS}


def test_graph_replay_with_the_prebuilt_layout_matches_eager(monkeypatch):
    _require_deepgemm_layer(_GRAPH_CP, monkeypatch)
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=_GRAPH_CP)
    graph = None
    try:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        cp_group = pg_collection.cp
        # Every check below is collected and raised on both ranks at the next synchronization
        # point, so that no rank waits alone in a collective (captured ones included).
        failures = _CollectedFailures(cp_group)
        failures.run(_warm_up_this_rank, 64)
        failures.raise_on_any_rank("DeepGEMM warm-up")
        model_parallel_cuda_manual_seed(1234)
        torch.manual_seed(1234)
        config = reuse._make_config(tp=1, cp=_GRAPH_CP, deterministic=False)
        config.dsa_indexer_loss_coeff = 0.0  # frozen indexer: the replay needs the selection only
        _use_deepgemm(config)
        attention = reuse._make_attention(config, pg_collection)
        spy = _DeepGemmSpy(monkeypatch)
        recorder = _GraphTopkRecorder(monkeypatch)
        key_partials = _KeyPartials(monkeypatch, config.kv_lora_rank + config.qk_pos_emb_head_dim)
        rank = cp_group.rank()
        inputs_a, inputs_b = _graph_inputs(config, 11 + rank), _graph_inputs(config, 111 + rank)
        expected_a = _graph_eager(attention, inputs_a, _graph_params(_PACK_A, cp_group))
        selection_a = failures.run(recorder.take_eager)
        partial_a = failures.run(key_partials.take_eager)
        expected_b = _graph_eager(attention, inputs_b, _graph_params(_PACK_B, cp_group))
        selection_b = failures.run(recorder.take_eager)
        partial_b = failures.run(key_partials.take_eager)
        failures.raise_on_any_rank("eager reference steps")

        static_params = _graph_params(_PACK_A, cp_group)
        static_inputs = {name: value.detach().clone() for name, value in inputs_a.items()}
        for name in _GRAD_INPUTS:
            static_inputs[name].requires_grad_(True)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                for name in _GRAD_INPUTS:
                    static_inputs[name].grad = None
                _graph_step(attention, static_inputs, static_params)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        failures.check(
            len(recorder.eager) == 3 and len(key_partials.eager) == 3,
            f"the 3 side-stream warm-up steps recorded {len(recorder.eager)} DeepGEMM top-k "
            f"calls and {len(key_partials.eager)} key gathers",
        )
        recorder.eager.clear()
        key_partials.eager.clear()
        for name in _GRAD_INPUTS:
            static_inputs[name].grad = None
        failures.run(recorder.allocate_pinned, selection_a)
        failures.raise_on_any_rank("side-stream warm-up")

        spy.reset()
        graph = torch.cuda.CUDAGraph()
        static_output = None
        try:
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                # A host sync anywhere in the captured step, autograd's device thread included,
                # raises here instead of depending on how the capture is invalidated.
                torch.cuda.set_sync_debug_mode("error")
                try:
                    static_output = _graph_step(attention, static_inputs, static_params)
                finally:
                    torch.cuda.set_sync_debug_mode("default")
        except Exception as error:  # pylint: disable=broad-except
            # The other rank's capture may have succeeded: fail both at the next collective.
            failures.add(f"capture: {type(error).__name__}: {error}")
            failures.cause = failures.cause or error
        torch.cuda.synchronize()
        if static_output is not None:
            failures.check(
                spy.captured_launches > 0 and set(spy.reasons) == {None},
                f"DeepGEMM in the capture: {spy.captured_launches} launches, reasons "
                f"{spy.reasons}",
            )
            failures.check(
                recorder.captured is not None and not recorder.eager,
                "the capture recorded no DeepGEMM top-k, or an eager one",
            )
            failures.check(
                key_partials.captured is not None and not key_partials.eager,
                "the capture recorded no key partial, or an eager one",
            )
        failures.raise_on_any_rank("capture")

        def replay_and_check(expected, selection, eager_partial, label):
            failures.run(recorder.pin, selection)
            failures.raise_on_any_rank(f"{label}, before the replay")
            graph.replay()
            torch.cuda.synchronize()
            # Every rank reaches this all-gather and the final count before any raises.
            pairs = [
                torch.empty(
                    (2, *eager_partial.shape),
                    dtype=eager_partial.dtype,
                    device=eager_partial.device,
                )
                for _ in range(_GRAPH_CP)
            ]
            torch.distributed.all_gather(
                pairs, torch.stack([key_partials.captured, eager_partial]), group=cp_group
            )
            swapped = failures.run(recorder.compare_replay, selection, label)
            # Given the same selection the output, dquery and dup_v_weight repeat bitwise
            # (GB300, oci-jhb job 1171499).
            failures.add(_mismatch(static_output, expected[0], f"{label} output"))
            for name in ("query", "up_v_weight"):
                failures.add(
                    _mismatch(static_inputs[name].grad, expected[1][name], f"{label} d{name}")
                )
            # dkey sums FP32 atomics (_partial_order_bound): this rank's partial within the
            # atomic-order bound, each run's dkey one rounding of its summed partials, and dkey
            # within the partials' differences plus those roundings.
            partial, partial_eager = pairs[rank]
            failure, partial_stats = _bounded_mismatch(
                partial, partial_eager, _partial_order_bound(partial, partial_eager), "dkey partial"
            )
            failures.add(failure and f"{label} {failure}")
            rows = slice(rank * _GRAPH_LOCAL_ROWS, (rank + 1) * _GRAPH_LOCAL_ROWS)
            dkey, dkey_eager = static_inputs["key"].grad, expected[1]["key"]
            for run, got, index in (("replay", dkey, 0), ("eager", dkey_eager, 1)):
                summed = sum(pair[index][rows].float() for pair in pairs).reshape(got.shape)
                failure, _stats = _bounded_mismatch(
                    got, summed, _reduce_scatter_bound(got, summed), f"{run} dkey - partials"
                )
                failures.add(failure and f"{label} {failure}")
            partial_diff = sum(
                (pair[0][rows].float() - pair[1][rows].float()).abs() for pair in pairs
            ).reshape(dkey.shape)
            failure, dkey_stats = _bounded_mismatch(
                dkey, dkey_eager, _dkey_bound(dkey, dkey_eager, partial_diff), "dkey"
            )
            failures.add(failure and f"{label} {failure}")
            failures.raise_on_any_rank(label)
            return swapped, f"{partial_stats}; {dkey_stats}"

        swapped_a, dkey_a = replay_and_check(expected_a, selection_a, partial_a, "pack A")
        source = _graph_params(_PACK_B, cp_group)
        with torch.no_grad():
            static_params.cu_seqlens_q.copy_(source.cu_seqlens_q)
            for name, value in inputs_b.items():
                static_inputs[name].copy_(value)
        copy_prebuilt_packed_cp_layout_(static_params, source)
        swapped_b, dkey_b = replay_and_check(expected_b, selection_b, partial_b, "pack B")
        _report(
            f"graph rank={rank}: replay with refreshed prebuilt layout matches eager (logits "
            f"bitwise, selection tie-aware: rows with other tied keys A={swapped_a} "
            f"B={swapped_b} of {selection_a[2].size(0)}; output, dquery and dup_v_weight "
            f"bitwise; A: {dkey_a}; B: {dkey_b})"
        )
    finally:
        # A failure's traceback keeps this frame, and so the graph, alive, and a live graph
        # holding the CP collectives blocks the process-group teardown (it hung there after
        # the failure in oci-jhb job 1171206): release it before the process groups go.
        if graph is not None:
            graph.reset()
        graph = None
        torch.cuda.synchronize()
        Utils.destroy_model_parallel()
