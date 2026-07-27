# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Model-family registry for the fsdp_dtensor -> torch_dist reverse-converter
end-to-end suite.

The single source of truth for the suite: each :class:`ModelFamily` is a
tiny-but-real GPT architecture that gates a distinct converter transform. Adding a
family is one entry here and nothing else — parametrization, all four checks, and
the known-limitation handling pick it up automatically.

Deliberately **stdlib-only** (no ``torch`` / ``megatron`` import) so that pytest's
default-collection gate check stays cheap.
"""

from dataclasses import dataclass, field
from typing import Mapping, Optional, Tuple


@dataclass(frozen=True)
class ReshardCase:
    """A load-side target parallel layout to reshard the converted checkpoint into.

    ``with_optimizer=False`` loads weights only (``--no-load-optim``); ``xfail``
    marks a layout that is a known, documented limitation (strict-xfail, so the
    day it is fixed the suite flips to failure and forces this flag to be removed).
    """

    layout: str  # "TP2" | "PP2" | "EP2" | "TP2SP"
    with_optimizer: bool = True
    xfail: Optional[str] = None


@dataclass(frozen=True)
class SourceShardCase:
    """A source-side FSDP training layout (train a sharded source, then convert).

    ``unsupported`` marks a layout whose source checkpoint cannot even be produced
    (e.g. PP2: Megatron-FSDP + pipeline-parallel training fails at model build), so
    the test is skipped rather than xfailed — an xfail would burn a full multi-GPU
    training run only to fail at build.
    """

    layout: str  # "DP2" | "TP2" | "PP2" | "EP2"
    unsupported: Optional[str] = None


@dataclass(frozen=True)
class ModelFamily:
    """One converter-transform family."""

    name: str  # registry key / results subdir / test id
    label: str  # human-readable name
    transform: str  # which converter transform this family gates
    num_layers: int
    arch: Tuple[str, ...]  # architecture CLI flags, pre-tokenized
    reshard_cases: Tuple[ReshardCase, ...] = ()  # load-side target layouts to sweep
    source_shard_cases: Tuple[SourceShardCase, ...] = (SourceShardCase("DP2"),)
    requires: Tuple[str, ...] = ()  # importable deps to gate on, e.g. ("fla",)
    loss_rtol: float = 1e-2  # resume-continuity tolerance on lm loss (bf16); FP8 loosens
    bitexact_xfail: Optional[str] = None  # reason if not bit-exact by design (FP8)
    extra_env: Mapping[str, str] = field(default_factory=dict)


# Reused reason strings so the intent reads once and stays consistent.
_EP_OPTIM_XFAIL = "EP>1 optimizer load: ChainedOptimizer entry-count mismatch (Expected 2, got 4)"
_PP2_SOURCE_NA = "Megatron-FSDP + pipeline-parallel training fails at model build (EinopsError)"
_FP8_NOT_BITEXACT = (
    "FP8 not bit-exact by design: the amax/scale _extra_state is dropped in the "
    "fsdp_dtensor checkpoint, so a few fp8 weight tensors re-quantize differently on re-save"
)


def _ep2_reshard_pair() -> Tuple[ReshardCase, ...]:
    """EP2 splits into two cases: weights-only (passes) and with-optimizer (xfail)."""
    return (
        ReshardCase("EP2", with_optimizer=False),
        ReshardCase("EP2", with_optimizer=True, xfail=_EP_OPTIM_XFAIL),
    )


MODELS = {
    "dense": ModelFamily(
        name="dense",
        label="Dense GPT (GELU)",
        transform="dense layer-stack",
        num_layers=12,
        arch=(),
        reshard_cases=(ReshardCase("TP2"), ReshardCase("PP2")),
        source_shard_cases=(
            SourceShardCase("DP2"),
            SourceShardCase("TP2"),
            SourceShardCase("PP2", unsupported=_PP2_SOURCE_NA),
        ),
    ),
    "dense_swiglu": ModelFamily(
        name="dense_swiglu",
        label="Dense GPT + SwiGLU",
        transform="SwiGLU fc1 _w/_v merge",
        num_layers=12,
        arch=("--swiglu",),
        reshard_cases=(ReshardCase("TP2"), ReshardCase("PP2")),
    ),
    "moe_grouped": ModelFamily(
        name="moe_grouped",
        label="MoE, grouped-GEMM (mixtral-like)",
        transform="grouped-expert restack",
        num_layers=12,
        arch=("--swiglu", "--num-experts", "8", "--moe-grouped-gemm", "--disable-bias-linear"),
        reshard_cases=_ep2_reshard_pair() + (ReshardCase("TP2SP"),),
        source_shard_cases=(SourceShardCase("DP2"), SourceShardCase("EP2")),
    ),
    "moe_gated": ModelFamily(
        name="moe_gated",
        label="MoE, non-grouped shared-expert + gate",
        transform="non-grouped local_experts restack",
        num_layers=12,
        arch=(
            "--swiglu",
            "--num-experts",
            "8",
            "--disable-bias-linear",
            "--moe-shared-expert-intermediate-size",
            "128",
            "--moe-shared-expert-gate",
            "--moe-router-load-balancing-type",
            "aux_loss",
            "--moe-router-topk",
            "2",
        ),
        reshard_cases=_ep2_reshard_pair(),
    ),
    "mtp": ModelFamily(
        name="mtp",
        label="GPT + Multi-Token Prediction",
        transform="MTP key-rename",
        num_layers=12,
        arch=(
            "--mtp-num-layers",
            "1",
            "--position-embedding-type",
            "rope",
            "--untie-embeddings-and-output-weights",
        ),
        reshard_cases=(ReshardCase("TP2"), ReshardCase("PP2")),
    ),
    "gdn_hybrid": ModelFamily(
        name="gdn_hybrid",
        label="Hybrid Gated-DeltaNet + MoE (Qwen-Next-like)",
        transform="GDN in_proj/conv1d factory split + non-grouped experts",
        num_layers=6,
        arch=(
            "--group-query-attention",
            "--num-query-groups",
            "2",
            "--swiglu",
            "--disable-bias-linear",
            "--rotary-percent",
            "0.5",
            "--no-rope-fusion",
            "--apply-layernorm-1p",
            "--apply-wd-to-qk-layernorm",
            "--attention-output-gate",
            "--experimental-attention-variant",
            "gated_delta_net",
            "--linear-attention-freq",
            "3",
            "--linear-conv-kernel-dim",
            "4",
            "--linear-key-head-dim",
            "64",
            "--linear-value-head-dim",
            "64",
            "--linear-num-key-heads",
            "4",
            "--linear-num-value-heads",
            "8",
            "--untie-embeddings-and-output-weights",
            "--num-experts",
            "32",
            "--moe-ffn-hidden-size",
            "64",
            "--moe-shared-expert-intermediate-size",
            "64",
            "--moe-shared-expert-gate",
            "--moe-router-load-balancing-type",
            "aux_loss",
            "--moe-router-topk",
            "8",
            "--moe-router-dtype",
            "fp32",
            "--attention-softmax-in-fp32",
            "--attention-backend",
            "unfused",
        ),
        requires=("fla",),  # flash-linear-attention; the dev image's `fla` stub is insufficient
    ),
    "moe_mla_mtp": ModelFamily(
        name="moe_mla_mtp",
        label="MoE + MLA + MTP (deepseek-like)",
        transform="MLA + MTP passthrough over grouped-expert restack",
        num_layers=12,
        arch=(
            "--swiglu",
            "--num-experts",
            "8",
            "--moe-grouped-gemm",
            "--disable-bias-linear",
            "--multi-latent-attention",
            "--q-lora-rank",
            "512",
            "--kv-lora-rank",
            "256",
            "--mtp-num-layers",
            "1",
            "--position-embedding-type",
            "rope",
        ),
    ),
    "dense_fp8": ModelFamily(
        name="dense_fp8",
        label="Dense + FP8 (llama3-like)",
        transform="FP8 _extra_state drop (loose ~1% resume, by design)",
        num_layers=12,
        arch=(
            "--swiglu",
            "--fp8-format",
            "hybrid",
            "--fp8-amax-history-len",
            "32",
            "--fp8-param-gather",
        ),
        loss_rtol=3e-2,  # FP8 re-inits amax on resume and tracks ~1% looser than bf16
        bitexact_xfail=_FP8_NOT_BITEXACT,
    ),
}


def all_families():
    """The full family set, in registry order."""
    return tuple(MODELS.values())
