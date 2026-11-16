# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compare prepared GDP recurrence throughput; no model projections or tokenization.

Run on one CUDA device, for example:
    uv run python tools/benchmark_gdp_batch_invariant.py --output results.json
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import triton
from fla.ops.gated_delta_product import chunk_gated_delta_product
from triton.testing import do_bench_cudagraph

from megatron.core.ssm.ops.gdp import batch_invariant as bi


def main() -> None:
    """Measure forward, forward/backward, and steady-state single-token append."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--updates", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    results = {
        "torch": torch.__version__,
        "triton": triton.__version__,
        "device": torch.cuda.get_device_name(),
        "source_sha256": hashlib.sha256(Path(bi.__file__).read_bytes()).hexdigest(),
        "measurement": "CUDA graph replay; prepared recurrence only; allocations captured",
        "cases": [],
    }
    for batch in args.batch:
        for updates in args.updates:
            tokens, heads, key, value = args.tokens, args.heads, 128, 64

            def normalized(shape: tuple[int, ...]) -> torch.Tensor:
                return torch.nn.functional.normalize(
                    torch.randn(shape, device="cuda"), dim=-1
                ).bfloat16()

            xs = (
                normalized((batch, tokens, heads, key)),
                normalized((batch, tokens, updates, heads, key)),
                torch.randn(
                    batch, tokens, updates, heads, value, device="cuda", dtype=torch.bfloat16
                ),
                -torch.rand(batch, tokens, heads, device="cuda") * 0.1,
                torch.rand(batch, tokens, updates, heads, device="cuda"),
            )
            for x in xs:
                x.requires_grad_()
            dy = torch.randn(batch, tokens, heads, value, device="cuda", dtype=torch.bfloat16)
            for backend in ("native", "C16", "C32"):
                block = 32 if backend == "C32" else 16

                def forward() -> torch.Tensor:
                    if backend == "native":
                        q, k, v, g, beta = xs
                        return chunk_gated_delta_product(
                            q, k.flatten(1, 2), v.flatten(1, 2), g, beta.flatten(1, 2), updates
                        )[0]
                    return bi.train(*xs, block_size=block)[0]

                def backward() -> tuple[torch.Tensor, ...]:
                    return torch.autograd.grad(forward(), xs, dy)

                forward_ms = do_bench_cudagraph(forward, rep=100)
                backward_ms = do_bench_cudagraph(backward, rep=100)
                row = {
                    "batch": batch,
                    "tokens": tokens,
                    "heads": heads,
                    "key": key,
                    "value": value,
                    "updates": updates,
                    "backend": backend,
                    "forward_ms": forward_ms,
                    "forward_backward_ms": backward_ms,
                }
                if backend != "native":
                    cache = bi.GDPCache.allocate(batch, heads, key, value, block_size=block)
                    one = tuple(x[:, :1].detach().contiguous() for x in xs)
                    detached = tuple(x.detach() for x in xs)
                    with torch.no_grad():
                        row["prefill_ms"] = do_bench_cudagraph(
                            lambda: bi.prefill(*detached, block_size=block), rep=100
                        )
                        row["append_prompt_ms"] = do_bench_cudagraph(
                            lambda: bi.append(*detached, cache), rep=100
                        )
                        row["decode_ms"] = do_bench_cudagraph(
                            lambda: bi.append(*one, cache), rep=100
                        )
                results["cases"].append(row)
                print(json.dumps(row), flush=True)
                args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
