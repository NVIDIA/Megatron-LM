# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Bit-exact worker (subprocess) for the reverse-converter end-to-end suite.

Loads a *reverse-converted* torch_dist checkpoint into a real classic (non-FSDP)
mcore ``GPTModel`` + ``DistributedOptimizer`` — built through the exact
``parse_and_validate_args`` -> ``initialize_megatron`` ->
``setup_model_and_optimizer(partial(model_provider, gpt_builder))`` path
``pretrain_gpt.py`` uses — and compares what the classic job actually holds after
the load against the original Megatron-FSDP ``fsdp_dtensor`` checkpoint, tensor by
tensor, in the model's own parameter-name space:

* every model ``state_dict`` tensor (weights in the compute dtype, fp32 buffers);
* every fp32 master parameter and every optimizer moment (``exp_avg`` /
  ``exp_avg_sq``) held by the ``DistributedOptimizer``;
* the per-parameter optimizer hyperparameters (``step``, ``betas``, ...) of the
  param group each parameter landed in.

The oracle is the FSDP checkpoint itself, read with no converter code: its keys are
the model's ``named_parameters`` behind a ``model.``/``module.`` wrapper prefix,
except for the two renames Megatron-FSDP applies on save, undone here: SwiGLU
``linear_fc1`` is split into ``_w``/``_v`` halves (``handle_swiglu_in_state_dict``)
and a GPT MTP layer's ``mtp_model_layer`` is written as ``transformer_layer``
(``handle_mtp_in_state_dict``). Every
converter transform (layer / expert stacking, GDN / MTP key handling, master
synthesis, param-group rebuild) therefore has to round-trip through mcore's real
loader to pass. Coverage is checked in both directions, so a source tensor the load
silently skipped (``log_all`` strictness only logs it) is reported as missing.

Runs in its OWN process per family, because ``initialize_megatron`` sets megatron's
global args exactly once per process. Requires one free GPU. Launched by
``harness.run_bitexact_worker``; the verdict is printed as one JSON object between
sentinels.
"""

import argparse
import io
import json
import re
import sys
from functools import partial
from pathlib import Path

# tests/integration_tests/tools/checkpoint/fsdp_dtensor_to_torch_dist/_bitexact_worker.py
# -> repo root is five package levels up. Put it on sys.path so both the tests
# package and the repo-root pretrain_gpt entrypoints (gpt_builders / model_provider)
# import cleanly when this file is run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO_ROOT))

from tests.integration_tests.tools.checkpoint.fsdp_dtensor_to_torch_dist import (  # noqa: E402
    config,
    registry,
)

_BEGIN = "===BITEXACT_JSON_BEGIN==="
_END = "===BITEXACT_JSON_END==="

# Param-group hyperparameters that must survive the conversion. ``lr`` is left to
# the resume check: the scheduler owns it and rewrites it on the first step.
_GROUP_KEYS = ("step", "betas", "eps", "weight_decay", "bias_correction", "lr_mult", "wd_mult")
# Checkpoint entries that are not model / optimizer state (dropped by design).
_SKIPPED = ("_extra_state", "rng_state", "rerun_state_machine")


# GPT MTP layers are saved as ``mtp.layers.<i>.transformer_layer.*`` but the module
# attribute (and so ``named_parameters``) is ``mtp_model_layer``.
_MTP_ON_DISK_RE = re.compile(r"^(mtp\.layers\.\d+)\.transformer_layer\.")


def _strip_wrappers(name):
    """``model.module.module.<fqn>`` / ``module.module.<fqn>`` -> module ``<fqn>``."""
    if name.startswith("model."):
        name = name[len("model.") :]
    while name.startswith("module."):
        name = name[len("module.") :]
    return _MTP_ON_DISK_RE.sub(r"\1.mtp_model_layer.", name)


def _swiglu_base(name, flat):
    """``<fqn>`` if ``name`` is one half of a Megatron-FSDP ``<fqn>_w``/``<fqn>_v`` pair."""
    base, tag = name[:-2], name[-2:]
    other = {"_w": "_v", "_v": "_w"}.get(tag)
    return base if other is not None and base + other in flat else None


def _merge_swiglu_halves(flat):
    """Concatenate Megatron-FSDP's SwiGLU ``linear_fc1`` ``_w``/``_v`` halves into ``<fqn>``."""
    import torch

    out = {}
    for name, value in flat.items():
        base = _swiglu_base(name, flat)
        if base is None:
            out[name] = value
        elif name.endswith("_w"):
            out[base] = torch.cat([value, flat[base + "_v"]], dim=0)
    return out


def load_fsdp_source(fsdp_iter_dir):
    """Read the fsdp_dtensor checkpoint into the model's parameter-name space.

    Returns ``(model, optim, group_meta)``: ``{fqn: tensor}``, ``{subkey: {fqn: tensor}}``
    and ``{fqn: {attr: value}}``. Single-rank DCP loads return full global tensors
    regardless of how the source run was sharded.
    """
    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    reader = FileSystemReader(str(fsdp_iter_dir))
    md = reader.read_metadata().state_dict_metadata
    state = {}
    for key, m in md.items():
        if any(s in key for s in _SKIPPED):
            continue
        if isinstance(m, TensorStorageMetadata):
            state[key] = torch.empty(m.size, dtype=m.properties.dtype)
        elif key.startswith("optimizer.param_to_group_meta."):
            state[key] = io.BytesIO()
    dcp.load(state, storage_reader=reader)

    model, optim, group_meta = {}, {}, {}
    for key, value in state.items():
        if key.startswith("model."):
            model[_strip_wrappers(key)] = value
        elif key.startswith("optimizer.state."):
            fqn, subkey = _strip_wrappers(key[len("optimizer.state.") :]).rsplit(".", 1)
            optim.setdefault(subkey, {})[fqn] = value
        elif key.startswith("optimizer.param_to_group_meta."):
            if isinstance(value, io.BytesIO):
                value.seek(0)
                value = torch.load(value, weights_only=False)
            fqn, attr = _strip_wrappers(key[len("optimizer.param_to_group_meta.") :]).rsplit(".", 1)
            group_meta.setdefault(fqn, {})[attr] = value
    model = _merge_swiglu_halves(model)
    optim = {subkey: _merge_swiglu_halves(by_fqn) for subkey, by_fqn in optim.items()}
    # Both SwiGLU halves carry their parameter's group meta; key it by the merged fqn.
    group_meta = {(_swiglu_base(k, group_meta) or k): v for k, v in group_meta.items()}
    return model, optim, group_meta


def compare_to_source(model_chunk, optimizer, source):
    """Diff the loaded classic model + optimizer against the fsdp source.

    Returns ``(counts, mismatches, missing, unexpected)``: ``missing`` = source tensors
    the classic job does not hold; ``unexpected`` = classic tensors with no source.
    """
    import torch

    from megatron.core.utils import unwrap_model

    src_model, src_optim, src_meta = source
    module = unwrap_model(model_chunk)
    mismatches, missing, unexpected = [], [], []
    counts = {"model": 0, "optim": 0, "groups": 0}

    def check(label, got, want):
        # The DistributedOptimizer keeps masters / moments as flat shard views, so
        # compare flattened; element count still pins the shape up to a reshape.
        got = got.detach().cpu().reshape(-1)
        want = want.reshape(-1)
        if got.numel() != want.numel() or not torch.equal(got, want.to(got.dtype)):
            mismatches.append(label)

    # Model section: every state_dict tensor, in the dtype the classic model keeps it.
    loaded = {
        k: v
        for k, v in module.state_dict().items()
        if isinstance(v, torch.Tensor) and "_extra_state" not in k
    }
    for fqn in sorted(set(src_model) - set(loaded)):
        missing.append(f"model:{fqn}")
    for fqn in sorted(set(loaded) - set(src_model)):
        unexpected.append(f"model:{fqn}")
    for fqn in sorted(set(loaded) & set(src_model)):
        check(f"model:{fqn}", loaded[fqn], src_model[fqn])
        counts["model"] += 1

    # Optimizer section: fp32 masters (Megatron-FSDP's fp32 model weights are its
    # masters) and moments, per model parameter, as the DistributedOptimizer holds
    # them; plus the hyperparameters of the group each parameter was placed in.
    name_of = {id(p): n for n, p in module.named_parameters()}
    expected = dict(src_optim, param=src_model)
    seen = {subkey: set() for subkey in src_optim}
    for opt in getattr(optimizer, "chained_optimizers", [optimizer]):
        for param, (group_index, _) in opt.model_param_group_index_map.items():
            fqn = name_of[id(param)]
            for subkey, tensor in opt._get_main_param_and_optimizer_states(param).items():
                want = expected.get(subkey, {}).get(fqn)
                if want is None:
                    unexpected.append(f"optimizer.{subkey}:{fqn}")
                    continue
                seen.setdefault(subkey, set()).add(fqn)
                check(f"optimizer.{subkey}:{fqn}", tensor, want)
                counts["optim"] += 1
            group = opt.optimizer.param_groups[group_index]
            meta = src_meta.get(fqn)
            if meta is None:
                missing.append(f"param_group:{fqn}")
                continue
            for attr in _GROUP_KEYS:
                if attr in meta and group.get(attr) != meta[attr]:
                    mismatches.append(
                        f"param_group.{attr}:{fqn} ({group.get(attr)!r} != {meta[attr]!r})"
                    )
            counts["groups"] += 1
    for subkey, by_fqn in src_optim.items():
        for fqn in sorted(set(by_fqn) - seen.get(subkey, set())):
            missing.append(f"optimizer.{subkey}:{fqn}")
    return counts, mismatches, missing, unexpected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("family", help="registry family name")
    ap.add_argument("--iter", type=int, required=True, help="converted checkpoint iteration")
    ap.add_argument("--td", required=True, help="converted torch_dist dir (<results>/<fam>/tdNN)")
    ap.add_argument("--fsdp", required=True, help="source fsdp_dtensor dir (<results>/<fam>/fsdp)")
    cli = ap.parse_args()

    fam = registry.MODELS[cli.family]
    # Build the arg vector from the registry/config, then drive megatron's own
    # parser / init exactly like pretrain_gpt.py.
    sys.argv = [
        fam.entrypoint,
        *config.COMMON_ARGS,
        "--num-layers",
        str(fam.num_layers),
        *fam.arch,
        *config.CLASSIC_LOAD_FLAGS,
        "--load",
        cli.td,
        "--train-iters",
        str(config.TRAIN_ITERS),
    ]

    from megatron.core.enums import ModelType
    from megatron.training.arguments import parse_and_validate_args
    from megatron.training.checkpointing import load_checkpoint
    from megatron.training.global_vars import initialize_runtime_services
    from megatron.training.initialize import initialize_megatron
    from megatron.training.training import setup_model_and_optimizer
    from model_provider import model_provider

    args = parse_and_validate_args(args_defaults={"tokenizer_type": "NullTokenizer"})
    initialize_runtime_services(args)  # timers etc., as pretrain_gpt.py does before pretrain()
    initialize_megatron()
    if fam.entrypoint == "pretrain_hybrid.py":
        from hybrid_builders import hybrid_builder as builder
    else:
        from gpt_builders import gpt_builder as builder
    model, optimizer, opt_sched = setup_model_and_optimizer(
        ModelType.encoder_or_decoder, partial(model_provider, builder)
    )
    iteration, _ = load_checkpoint(model, optimizer, opt_sched)

    assert len(model) == 1, "single-rank, no virtual pipeline: expected one model chunk"
    source = load_fsdp_source(Path(cli.fsdp) / f"iter_{cli.iter:07d}")
    counts, mismatches, missing, unexpected = compare_to_source(model[0], optimizer, source)
    payload = {
        "family": cli.family,
        "loaded_iteration": int(iteration),
        "counts": counts,
        "mismatches": mismatches,
        "missing": missing,
        "unexpected": unexpected,
    }
    print(_BEGIN)
    print(json.dumps(payload))
    print(_END)


if __name__ == "__main__":
    main()
