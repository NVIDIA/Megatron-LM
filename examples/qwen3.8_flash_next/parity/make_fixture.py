# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Create the parity fixture: one random-initialised HF proxy plus the token batches.

Weights are generated here and never committed; only this generator is, so the fixture is
reproducible from the seed alone.

Run inside the container, on a GPU (``fla``'s causal_conv1d rejects CPU tensors), with a
``transformers`` build that provides ``qwen4_exp`` first on ``PYTHONPATH``:

    python3 make_fixture.py --out <dir>

Writes:
    <dir>/hf/          save_pretrained output (config.json + safetensors)
    <dir>/inputs.pt    {"data": int64 [NUM_BATCHES, BATCH, SEQ + 1]}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import proxy_config as P  # noqa: E402


def build_hf_model(device: str = "cuda"):
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM

    torch.manual_seed(P.SEED)
    cfg = Qwen4ExpTextConfig(**P.HF_CONFIG)
    cfg._attn_implementation = "eager"
    return Qwen4ExpForCausalLM(cfg).to(device=device, dtype=torch.float32)


def perturb_zero_inits(model, std: float = 0.05) -> list[str]:
    """Give every zero-initialised tensor a deterministic non-zero value.

    HF zero-initialises each ``Qwen4ExpTextRMSNorm`` gamma and the PLE conv1d weight. Left at
    zero, a gamma-convention mistake (zero-centered vs plain) and the entire dilated-conv
    branch would both multiply out to zero and the comparison would pass while saying nothing.
    ``RMSNormGated`` (the GDN output norm) is ones-initialised by HF and is left alone.
    """
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextPLELayer,
        Qwen4ExpTextRMSNorm,
    )

    gen = torch.Generator(device="cpu").manual_seed(P.SEED + 1)
    touched: list[str] = []
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, Qwen4ExpTextRMSNorm):
                noise = torch.randn(module.weight.shape, generator=gen) * std
                module.weight.copy_(noise.to(module.weight.device))
                touched.append(f"{name}.weight")
            elif isinstance(module, Qwen4ExpTextPLELayer):
                conv = module.conv1d
                noise = torch.randn(conv.weight.shape, generator=gen) * std
                conv.weight.copy_(noise.to(conv.weight.device))
                touched.append(f"{name}.conv1d.weight")
    return touched


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    model = build_hf_model()
    touched = perturb_zero_inits(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[fixture] HF proxy: {n_params} parameters, perturbed {len(touched)} zero-inits")

    model.save_pretrained(args.out / "hf", safe_serialization=True)

    # Token batches. SEQ + 1 so the trajectory can use [:-1] as input and [1:] as labels.
    gen = torch.Generator(device="cpu").manual_seed(P.SEED + 2)
    data = torch.randint(
        0, P.VOCAB, (P.NUM_BATCHES, P.BATCH, P.SEQ + 1), generator=gen, dtype=torch.int64
    )
    torch.save({"data": data}, args.out / "inputs.pt")

    (args.out / "fixture.json").write_text(
        json.dumps(
            {
                "seed": P.SEED,
                "hf_parameters": n_params,
                "perturbed": touched,
                "batches": list(data.shape),
            },
            indent=2,
        )
    )
    print(f"[fixture] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
