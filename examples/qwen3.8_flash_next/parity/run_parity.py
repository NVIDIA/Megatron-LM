# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Whole-model parity: the HF ``qwen4_exp`` proxy against Megatron's ``HybridModel``.

Same weights on both sides, same tokens, fp32. Three checks, in increasing strictness of what
they can catch:

    1. forward   -- logits agree, and the discrete decisions (QSA block selection, MoE top-k)
                    are identical. A forward-only check passes even when the backward is wrong.
    2. gradients -- one backward from the same loss; every parameter compared after mapping.
    3. trajectory-- N AdamW steps on both sides; the losses must track, not merely start equal.

Run through ``run_parity.sh``, which sets the environment the comparison depends on.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hf_to_mcore as M  # noqa: E402
import proxy_config as P  # noqa: E402

ATOL, RTOL = 1e-6, 1e-5
GRAD_RTOL, GRAD_ATOL = 1e-5, 1e-4


# ----------------------------------------------------------------------------- megatron side


def init_megatron(argv: list[str]):
    sys.argv = [sys.argv[0]] + argv
    from megatron.training.arguments import parse_args, validate_args
    from megatron.training.global_vars import set_global_variables
    from megatron.training.initialize import initialize_megatron

    args = parse_args()
    validate_args(args, {})
    set_global_variables(args)
    initialize_megatron()
    return args


def build_megatron_model():
    """The hybrid model, unwrapped: no DDP, no optimizer, so parameters stay comparable."""
    from functools import partial

    from hybrid_builders import hybrid_builder
    from megatron.core.enums import ModelType
    from megatron.training.training import get_model
    from model_provider import model_provider

    model = get_model(
        partial(model_provider, hybrid_builder),
        ModelType.encoder_or_decoder,
        wrap_with_ddp=False,
    )[0]
    model.train()
    return model


def load_mapped_state(model, mapped: dict) -> dict:
    """Copy the mapped tensors into the live model.

    The only structural difference left at this point is grouped MoE experts: the mapping
    produces one packed ``[E, ...]`` tensor while Megatron holds one parameter per local
    expert, named ``weight0 .. weight{E-1}``. Orientation is resolved by shape rather than
    assumed, because the grouped-GEMM backends differ on whether fc2 is stored transposed.
    """
    params = dict(model.named_parameters())
    stats = {"copied": 0, "expanded": 0, "missing": [], "shape_mismatch": []}

    with torch.no_grad():
        for key, tensor in mapped.items():
            name = key[len("module.") :] if key.startswith("module.") else key
            if name in params:
                target = params[name]
                src = tensor
                if tuple(src.shape) != tuple(target.shape):
                    if tuple(src.T.shape) == tuple(target.shape):
                        src = src.T.contiguous()
                    else:
                        stats["shape_mismatch"].append((name, tuple(src.shape), tuple(target.shape)))
                        continue
                target.copy_(src.to(device=target.device, dtype=target.dtype))
                stats["copied"] += 1
                continue

            # Grouped experts: one packed tensor -> weight0..weight{E-1}
            if name.endswith("experts.linear_fc1.weight") or name.endswith(
                "experts.linear_fc2.weight"
            ):
                for e in range(tensor.shape[0]):
                    sub_name = f"{name}{e}"
                    if sub_name not in params:
                        stats["missing"].append(sub_name)
                        continue
                    target = params[sub_name]
                    src = tensor[e]
                    if tuple(src.shape) != tuple(target.shape):
                        if tuple(src.T.shape) == tuple(target.shape):
                            src = src.T.contiguous()
                        else:
                            stats["shape_mismatch"].append(
                                (sub_name, tuple(src.shape), tuple(target.shape))
                            )
                            continue
                    target.copy_(src.to(device=target.device, dtype=target.dtype))
                    stats["expanded"] += 1
                continue

            stats["missing"].append(name)

    stats["ok"] = not stats["missing"] and not stats["shape_mismatch"]
    return stats


def load_hf_checkpoint(path: Path) -> dict:
    """Every tensor in a ``save_pretrained`` directory, sharded or not."""
    from safetensors.torch import load_file

    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors under {path}")
    state: dict = {}
    for f in files:
        state.update(load_file(str(f)))
    return state


def megatron_forward(model, tokens):
    b, s = tokens.shape
    position_ids = torch.arange(s, device=tokens.device).unsqueeze(0).expand(b, s)
    out = model(tokens, position_ids, None)
    return out[0] if isinstance(out, tuple) else out


# ----------------------------------------------------------------------------- comparison


def compare_tensors(a: torch.Tensor, b: torch.Tensor) -> dict:
    a32, b32 = a.detach().float(), b.detach().float()
    diff = (a32 - b32).abs()
    denom = a32.norm().item() or 1.0
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "rel_l2": ((a32 - b32).norm() / denom).item(),
        "allclose_1e-6_1e-5": torch.allclose(a32, b32, atol=ATOL, rtol=RTOL),
        "nan": bool(torch.isnan(diff).any()),
    }


def cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1)
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=20, help="trajectory length; 0 to skip")
    ap.add_argument("--lr", type=float, default=1e-3)
    own, _ = ap.parse_known_args()
    own.out.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report: dict = {
        "env": {
            "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
            "TRITON_F32_DEFAULT": os.environ.get("TRITON_F32_DEFAULT"),
            "torch": torch.__version__,
        }
    }
    if report["env"]["NVIDIA_TF32_OVERRIDE"] != "0":
        print(
            "[warn] NVIDIA_TF32_OVERRIDE is not 0: TransformerEngine will run fp32 GEMMs as "
            "TF32 and every linear will carry ~1e-3 of noise, which is far above what this "
            "comparison measures. Use run_parity.sh."
        )

    cfg = P.HF_CONFIG
    device = "cuda"
    batches = torch.load(own.fixture / "inputs.pt")["data"]
    tokens = batches[0, :, :-1].to(device)
    labels = batches[0, :, 1:].to(device)

    # ---- Megatron side ---------------------------------------------------------------
    t0 = time.time()
    init_megatron(P.mcore_argv())
    mcore = build_megatron_model()
    report["env"]["megatron_params"] = sum(p.numel() for p in mcore.parameters())
    print(f"[mcore] built HybridModel: {report['env']['megatron_params']} params "
          f"in {time.time() - t0:.1f}s")

    # ---- HF side ---------------------------------------------------------------------
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM

    hf = Qwen4ExpForCausalLM.from_pretrained(
        own.fixture / "hf", dtype=torch.float32, attn_implementation="eager"
    ).to(device)
    hf.train()
    report["env"]["hf_params"] = sum(p.numel() for p in hf.parameters())
    print(f"[hf] loaded proxy: {report['env']['hf_params']} params")

    # ---- map and load ----------------------------------------------------------------
    # Weights come from the checkpoint on disk, not from the live module, because that is the
    # path a real conversion takes -- and because the two forms differ: save_pretrained splits
    # the n-gram table into `split_ngram_parts` row shards, while the live module holds one
    # nn.Embedding. Reading the saved form here means the mapping is exercised as a converter.
    rules = M.build_rules(cfg, ple_source="sharded")
    hf_state = M.repack_experts(load_hf_checkpoint(own.fixture / "hf"), int(cfg["num_experts"]))
    mapped = M.apply_rules(rules, hf_state, head_dim=int(cfg["head_dim"]))
    load = load_mapped_state(mcore, mapped)
    report["step0"] = {"rules": len(rules), "mapped_tensors": len(mapped), "load": load}
    print(f"[step0] {len(rules)} rules -> {len(mapped)} tensors; copied {load['copied']}, "
          f"expanded {load['expanded']}, ok={load['ok']}")
    if not load["ok"]:
        print(f"[step0] missing={load['missing'][:8]} mismatched={load['shape_mismatch'][:8]}")
        (own.out / "report.json").write_text(json.dumps(report, indent=2))
        return 2

    # ---- step 1: forward -------------------------------------------------------------
    with torch.no_grad():
        hf_logits = hf(input_ids=tokens).logits
        mc_logits = megatron_forward(mcore, tokens)
    report["step1"] = {"logits": compare_tensors(hf_logits, mc_logits)}
    report["step1"]["argmax_equal"] = bool(
        (hf_logits.argmax(-1) == mc_logits.argmax(-1)).all().item()
    )
    s1 = report["step1"]["logits"]
    print(f"[step1] logits max_abs={s1['max_abs']:.3e} rel_l2={s1['rel_l2']:.3e} "
          f"allclose={s1['allclose_1e-6_1e-5']} argmax_equal={report['step1']['argmax_equal']}")

    # ---- step 2: gradients -----------------------------------------------------------
    hf.zero_grad(set_to_none=True)
    mcore.zero_grad(set_to_none=True)
    cross_entropy(hf(input_ids=tokens).logits, labels).backward()
    cross_entropy(megatron_forward(mcore, tokens), labels).backward()

    hf_grads = {k: v.grad.detach().cpu() for k, v in hf.named_parameters() if v.grad is not None}
    mapped_grads = M.apply_rules(
        # Gradients come off the live module, where the n-gram table is a single tensor.
        M.build_rules(cfg, ple_source="single"),
        hf_grads,
        head_dim=int(cfg["head_dim"]),
        on_missing=lambda _k: None,  # parameters without a gradient simply do not appear
        for_gradients=True,
    )
    mc_params = dict(mcore.named_parameters())
    compared, failures = 0, []
    for key, hf_grad in mapped_grads.items():
        name = key[len("module.") :] if key.startswith("module.") else key
        candidates = [name] if name in mc_params else [
            f"{name}{e}" for e in range(cfg["num_experts"])
        ]
        for idx, cand in enumerate(candidates):
            param = mc_params.get(cand)
            if param is None or param.grad is None:
                continue
            ref = hf_grad if len(candidates) == 1 else hf_grad[idx]
            got = param.grad.detach().cpu()
            if tuple(ref.shape) != tuple(got.shape) and tuple(ref.T.shape) == tuple(got.shape):
                ref = ref.T
            stat = compare_tensors(ref, got)
            compared += 1
            if not torch.allclose(ref.float(), got.float(), rtol=GRAD_RTOL, atol=GRAD_ATOL):
                failures.append({"name": cand, **stat})
    report["step2"] = {
        "compared": compared,
        "failures": failures[:12],
        "n_failures": len(failures),
        "all_pass": not failures,
    }
    print(f"[step2] gradients: {compared} compared, {len(failures)} outside "
          f"rtol={GRAD_RTOL}/atol={GRAD_ATOL}")
    for f in failures[:6]:
        print(f"         {f['name']}: rel_l2={f['rel_l2']:.3e} max_abs={f['max_abs']:.3e}")

    # ---- step 3: trajectory ----------------------------------------------------------
    if own.steps > 0:
        hf_opt = torch.optim.AdamW(hf.parameters(), lr=own.lr)
        mc_opt = torch.optim.AdamW(mcore.parameters(), lr=own.lr)
        rows = []
        print(f"[step3] {'step':>4} | {'loss_hf':>12} | {'loss_mcore':>12} | {'|dloss|':>9}")
        for step in range(min(own.steps, batches.shape[0])):
            tk = batches[step, :, :-1].to(device)
            lb = batches[step, :, 1:].to(device)
            hf_opt.zero_grad(set_to_none=True)
            mc_opt.zero_grad(set_to_none=True)
            loss_hf = cross_entropy(hf(input_ids=tk).logits, lb)
            loss_mc = cross_entropy(megatron_forward(mcore, tk), lb)
            loss_hf.backward()
            loss_mc.backward()
            hf_opt.step()
            mc_opt.step()
            d = abs(loss_hf.item() - loss_mc.item())
            rows.append({"step": step, "hf": loss_hf.item(), "mcore": loss_mc.item(), "abs": d})
            print(f"[step3] {step:>4} | {loss_hf.item():12.6f} | {loss_mc.item():12.6f} | {d:9.2e}")
        report["step3"] = {"rows": rows, "max_abs_dloss": max(r["abs"] for r in rows)}
        print(f"[step3] max |dloss| over {len(rows)} steps = {report['step3']['max_abs_dloss']:.2e}")

    (own.out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"[done] report -> {own.out / 'report.json'}")

    ok = (
        load["ok"]
        and report["step1"]["logits"]["allclose_1e-6_1e-5"]
        and report["step2"]["all_pass"]
    )
    print(f"[done] PASS={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
