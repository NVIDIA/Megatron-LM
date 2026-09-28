"""Local-SPMD type checks for the models in Megatron's functional-test configs.

Each test reads a functional test's ``MODEL_ARGS``, parses them with Megatron's
own argument parser, shrinks the sizes and forces TP2 without other parallelism,
and builds the model as its pretraining script would. It then runs that script's
forward step, loss and backward on rank 0 of a fake TP2 job under the strict
checker. Every
feature flag in the config is kept, so a TP type bug in any of them shows up as
a type error somewhere in the step. To cover another config, add its name, or
add a variant to rerun it with some flags changed.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from functools import partial
from pathlib import Path
from typing import Callable, NamedTuple
from unittest.mock import patch

import pytest
import spmd_types as spmd
import torch
import torch.nn.functional as F
import yaml
from torch.utils._pytree import tree_map_only

from gpt_builders import gpt_builder
from megatron.core import Timers
from megatron.core.spmd.annotations import annotate_model
from megatron.core.ssm.gated_delta_net import common as gdn_common
from megatron.core.ssm.gated_delta_net import gdn
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_block import HAVE_APEX
from megatron.core.utils import GlobalMemoryBuffer
from megatron.training import global_vars
from megatron.training.arguments import parse_args, validate_args
from megatron.training.global_vars import set_args

from test_tp_sweep import (  # noqa: F401  # isort: skip
    _default_attention_backend,
    tp_group,
    tp_type,
    typecheck,
    typed,
)

REPO = Path(__file__).parents[3]
TEST_CASES = REPO / "tests" / "functional_tests" / "test_cases"

CONFIGS = [
    "gpt/gpt3_mcore_te_tp2_pp2",
    "gpt/gpt3_mcore_te_tp2_pp2_mla",
    "gpt/gpt3_mcore_te_tp2_pp2_cross_entropy_loss_fusion",
    "gpt/gpt3_mcore_te_tp4_pp1_qk_layernorm_test_mode",
    "gpt/gpt3_mcore_tp2_pp2_uninstall_te",
    "gpt/gpt3_mcore_te_tp2_pp1_gdn",
    "gpt/gpt3_mcore_te_tp2_pp2_dsa",
    "moe/gpt3_mcore_te_tp2_pp1_te_8experts2parallel_top2router",
    "bert/bert_mcore_tp2_pp2",
    "t5/t5_mcore_te_tp2_pp1_vp1_sequence_parallel",
]

# Configs rerun with some flags changed.
VARIANTS = [
    # Megatron-LM#7452.11: GDN shared output-normalization weight misses the TP gradient sum.
    # MoE layers need sequence parallelism under TP, so run this one dense.
    pytest.param(
        "gpt/gpt3_mcore_te_tp2_pp1_gdn",
        {"--sequence-parallel": False, "--num-experts": None},
        id="gpt/gpt3_mcore_te_tp2_pp1_gdn-no_sequence_parallel",
        marks=pytest.mark.xfail(strict=True, reason="7452.11: out_norm weight typed I"),
    )
]

SEQ_LENGTH = 64
MICRO_BATCH = 2
VOCAB_SIZE = 128

# Flags that only matter to a real training run (data, checkpoints, logging).
IGNORED_PREFIXES = (
    "--data-",
    "--save",
    "--load",
    "--tensorboard",
    "--log-",
    "--vocab-file",
    "--merge-file",
    "--tokenizer",
    "--split",
    "--train-iters",
    "--eval",
    # Debug-only cross-rank comparisons of raw parameter data.
    "--test-mode",
)

# Sizes and parallelism for a fake TP2 job on one GPU; every other flag is kept.
OVERRIDES = {
    "--tensor-model-parallel-size": 2,
    "--pipeline-model-parallel-size": 1,
    "--expert-model-parallel-size": 1,
    "--seq-length": SEQ_LENGTH,
    "--max-position-embeddings": SEQ_LENGTH,
    "--dsa-indexer-topk": SEQ_LENGTH // 2,
    "--micro-batch-size": MICRO_BATCH,
    "--global-batch-size": MICRO_BATCH,
}


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def shrink(raw: dict, variant: dict) -> dict:
    """Shrink a config's sizes to a fake TP2 job, keeping one layer of each kind."""
    # Hybrid attention layouts need enough layers for one of each kind.
    layers = max(2, raw.get("--linear-attention-freq", 1))
    flags = {**raw, **variant, **OVERRIDES}
    # Encoder-decoder models size each stack separately.
    for flag in ("--num-layers", "--encoder-num-layers", "--decoder-num-layers"):
        if flag in raw:
            flags[flag] = min(raw[flag], layers)
    if "--encoder-seq-length" in raw:
        flags["--seq-length"] = None
        flags["--encoder-seq-length"] = flags["--decoder-seq-length"] = SEQ_LENGTH
    # A variant that drops ``--num-experts`` runs the model dense.
    if flags.get("--num-experts") is None:
        flags = {flag: value for flag, value in flags.items() if not flag.startswith("--moe-")}
    return flags


def with_available_kernels(flags: dict) -> dict:
    """Fall back from kernels this environment lacks; they don't change the layout."""
    flags = dict(flags)
    if not HAVE_APEX:
        flags["--no-persist-layer-norm"] = True
    if not _installed("fused_weight_gradient_mlp_cuda"):
        flags["--no-gradient-accumulation-fusion"] = True
    if not _installed("scaled_masked_softmax_cuda"):
        flags["--no-masked-softmax-fusion"] = True
    return flags


def to_argv(flags: dict) -> list[str]:
    argv = []
    for flag, value in flags.items():
        if value is None or value is False or flag.startswith(IGNORED_PREFIXES):
            continue
        argv.append(flag)
        if value is not True:
            argv.append(str(value))
    return argv


def model_args(raw: dict, variant: dict):
    """Parse and validate a functional test's ``MODEL_ARGS``, shrunk to a fake TP2 job."""
    argv = ["pretrain.py", *to_argv(with_available_kernels(shrink(raw, variant)))]
    with patch.object(sys, "argv", argv), patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "2"}):
        args = parse_args()
    validate_args(args)
    args.padded_vocab_size = VOCAB_SIZE
    set_args(args)
    return args


@pytest.fixture(autouse=True)
def _untyped_attention_scratch(monkeypatch):
    """Work around spmd_types checking the input of ``baddbmm`` with ``beta=0``.

    Unfused attention computes scores into an uninitialized ``torch.empty``
    buffer, which ``baddbmm(beta=0)`` never reads but the strict checker rejects
    as untyped. Compute those scores without the buffer, and allocate Megatron's
    ``GlobalMemoryBuffer`` slices unchecked. Remove once spmd_types handles it.
    """
    baddbmm = torch.baddbmm
    get_tensor = GlobalMemoryBuffer.get_tensor

    def without_unread_input(input, batch1, batch2, *, beta=1, alpha=1, out=None):
        if beta == 0 and out is None:
            return torch.bmm(batch1, batch2) * alpha
        return baddbmm(input, batch1, batch2, beta=beta, alpha=alpha, out=out)

    def unchecked_get_tensor(*args, **kwargs):
        with spmd.no_typecheck():
            return get_tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "baddbmm", without_unread_input)
    monkeypatch.setattr(GlobalMemoryBuffer, "get_tensor", unchecked_get_tensor)


def l2norm(x, dim=-1, eps=1e-6):
    return F.normalize(x, dim=dim, eps=eps)


def gpt_sample(shape):
    return {
        "tokens": torch.randint(VOCAB_SIZE, shape),
        "labels": torch.randint(VOCAB_SIZE, shape),
        "loss_mask": torch.ones(shape),
        "position_ids": torch.arange(shape[1]).expand(shape).contiguous(),
    }


def bert_sample(shape):
    return {
        "text": torch.randint(VOCAB_SIZE, shape),
        "types": torch.zeros(shape, dtype=torch.long),
        "labels": torch.randint(VOCAB_SIZE, shape),
        "is_random": torch.randint(2, shape[:1]),
        "loss_mask": torch.ones(shape, dtype=torch.long),
        "padding_mask": torch.ones(shape, dtype=torch.long),
    }


def t5_sample(shape):
    return {
        "text_enc": torch.randint(VOCAB_SIZE, shape),
        "text_dec": torch.randint(VOCAB_SIZE, shape),
        "labels": torch.randint(VOCAB_SIZE, shape),
        "loss_mask": torch.ones(shape, dtype=torch.long),
        "enc_mask": torch.zeros(shape, dtype=torch.long),
        "dec_mask": torch.zeros(shape, dtype=torch.long),
    }


class Family(NamedTuple):
    """How a functional-test family trains: its pretraining script, and what its loader yields."""

    # Path of the pretraining script, relative to the repository root.
    script: str
    sample: Callable
    # TP type of each ``get_batch`` output, ``None`` for ones left untyped. Every rank
    # loads the same batch; labels and loss masks are typed I only because the loss
    # rules expect the loss's type.
    batch_types: tuple
    # Builds the model like ``model_provider``; ``None`` calls the script's own.
    build: Callable | None = None


# ``pretrain_gpt.get_batch`` returns its fields in alphabetical order.
GPT_BATCH_TYPES = (spmd.R, None, None, None, spmd.I, None, spmd.I, None, spmd.I, spmd.R)
FAMILIES = {
    "gpt": Family("pretrain_gpt.py", gpt_sample, GPT_BATCH_TYPES, gpt_builder),
    "moe": Family("pretrain_gpt.py", gpt_sample, GPT_BATCH_TYPES, gpt_builder),
    "bert": Family(
        "examples/bert/pretrain_bert.py",
        bert_sample,
        (spmd.R, spmd.R, spmd.I, spmd.I, spmd.I, spmd.R),
    ),
    "t5": Family(
        "examples/t5/pretrain_t5.py",
        t5_sample,
        (spmd.R, spmd.R, spmd.I, spmd.I, spmd.R, spmd.R, spmd.R),
    ),
}


def typed_batches(get_batch, batch_types):
    """Load a batch unchecked, then type what the model receives.

    ``get_batch`` broadcasts each batch from TP rank 0 with raw
    ``torch.distributed``, which spmd_types doesn't type yet.
    """

    def wrapper(*args, **kwargs):
        with spmd.no_typecheck():
            batch = get_batch(*args, **kwargs)
        assert len(batch) == len(batch_types)
        return tuple(
            (
                item
                if tp_type is None
                else tree_map_only(torch.Tensor, partial(typed, tp_type=tp_type), item)
            )
            for item, tp_type in zip(batch, batch_types)
        )

    return wrapper


@pytest.mark.parametrize(
    "name, variant", [pytest.param(name, {}, id=name) for name in CONFIGS] + VARIANTS
)
def test_config_forward_backward(name, variant, tp_group, monkeypatch):
    config = yaml.safe_load((TEST_CASES / name / "model_config.yaml").read_text())
    for variable, value in config.get("ENV_VARS", {}).items():
        monkeypatch.setenv(variable, str(value))
    args = model_args(config["MODEL_ARGS"], variant)
    if args.experimental_attention_variant == "dsa":
        pytest.importorskip("fast_hadamard_transform")
    uses_apex_norm = args.transformer_impl == "local" or name.startswith("bert/")
    if uses_apex_norm and args.sequence_parallel and not HAVE_APEX:
        pytest.skip("this spec's layer norms need apex for sequence parallelism")
    if args.experimental_attention_variant == "gated_delta_net" and not gdn_common.HAVE_FLA:
        # Deterministic mode runs GDN's torch reference kernels; only l2norm is FLA's.
        assert args.deterministic_mode
        monkeypatch.setattr(gdn_common, "HAVE_FLA", True)
        for module in (gdn_common, gdn):
            monkeypatch.setattr(module, "l2norm", l2norm)

    family = FAMILIES[name.split("/")[0]]
    script_path = REPO / family.script
    monkeypatch.syspath_prepend(str(script_path.parent))
    script = importlib.import_module(script_path.stem)
    if family.build:
        model = family.build(args, pre_process=True, post_process=True)
    else:
        model = script.model_provider(pre_process=True, post_process=True)
    model = model.cuda()
    if args.fp16 or args.bf16:
        model = Float16Module(model.config, model)
    monkeypatch.setattr(global_vars, "_GLOBAL_TIMERS", Timers(args.timing_log_level, "minmax"))
    monkeypatch.setattr(script, "get_batch", typed_batches(script.get_batch, family.batch_types))
    loader = iter([family.sample((args.micro_batch_size, SEQ_LENGTH))])
    with typecheck(tp_group):
        annotate_model(model)
        output, loss_func = script.forward_step(loader, model)
        loss = loss_func(output)[0]
        assert tp_type(loss, tp_group) is spmd.I
        loss.backward()
