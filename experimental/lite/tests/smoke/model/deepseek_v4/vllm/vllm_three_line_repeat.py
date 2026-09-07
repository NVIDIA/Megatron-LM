"""Repeat fixed-token DS4 scoring in native vLLM without the RL harness."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any

import torch
from vllm import LLM, SamplingParams
from vllm.utils.network_utils import get_open_port


def _load_rows(path: Path, indices: tuple[int, ...]) -> list[dict[str, Any]]:
    wanted = set(indices)
    rows = []
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index not in wanted:
                continue
            value = json.loads(line)
            rows.append(
                {
                    "row": index,
                    "prompt": list(map(int, value["prompt_token_ids"])),
                    "response": list(map(int, value["response_token_ids"])),
                    "rollout_bits": value.get("rollout_logprob_bits"),
                    "actor_bits": value.get("actor_logprob_bits"),
                }
            )
    if tuple(row["row"] for row in rows) != indices:
        raise RuntimeError(f"missing requested rows: expected={indices}")
    return rows


def _score(llm: LLM, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prompts = [
        {"prompt_token_ids": row["prompt"] + row["response"]} for row in rows
    ]
    params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=1,
    )
    outputs = llm.generate(prompts, params, use_tqdm=False)
    if len(outputs) != len(rows):
        raise RuntimeError(f"unexpected vLLM output count: {len(outputs)}")

    reports = []
    for row, output in zip(rows, outputs, strict=True):
        prompt_logprobs = output.prompt_logprobs
        if prompt_logprobs is None:
            raise RuntimeError("vLLM did not return prompt logprobs")
        values = []
        for offset, token_id in enumerate(row["response"]):
            position = len(row["prompt"]) + offset
            candidates = prompt_logprobs[position]
            if candidates is None or token_id not in candidates:
                raise RuntimeError(
                    f"missing scored token row={row['row']} position={position}"
                )
            values.append(float(candidates[token_id].logprob))
        tensor = torch.tensor(values, dtype=torch.float32)
        bits = tensor.contiguous().view(torch.int32).tolist()
        reports.append(
            {
                "row": row["row"],
                "values": values,
                "bits": bits,
                "sha256": hashlib.sha256(
                    tensor.contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
            }
        )
    return reports


def _compare(
    reference: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    reports = []
    for expected, actual in zip(reference, candidate, strict=True):
        mismatches = [
            index
            for index, (left, right) in enumerate(
                zip(expected["bits"], actual["bits"], strict=True)
            )
            if left != right
        ]
        reports.append(
            {
                "row": expected["row"],
                "mismatches": len(mismatches),
                "first": mismatches[0] if mismatches else None,
                "reference_sha256": expected["sha256"],
                "candidate_sha256": actual["sha256"],
            }
        )
    return reports


def _compare_oracle(
    rows: list[dict[str, Any]],
    scored: list[dict[str, Any]],
    field: str,
) -> list[dict[str, Any]] | None:
    if any(row[field] is None for row in rows):
        return None
    oracle = [
        {
            "row": row["row"],
            "bits": row[field],
            "sha256": hashlib.sha256(
                torch.tensor(row[field], dtype=torch.int32)
                .contiguous()
                .view(torch.uint8)
                .numpy()
                .tobytes()
            ).hexdigest(),
        }
        for row in rows
    ]
    return _compare(oracle, scored)


def _run_rank(
    rank: int,
    dp_size: int,
    master_port: int,
    engine_args: dict[str, Any],
    rows: list[dict[str, Any]],
    repeats: int,
    output: Path,
) -> None:
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = "127.0.0.1"
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)
    llm = LLM(**engine_args)
    scored = [_score(llm, rows) for _ in range(repeats)]
    singleton = [_score(llm, [row])[0] for row in rows]
    output.write_text(
        json.dumps({"packed": scored, "singleton": singleton}, sort_keys=True) + "\n"
    )
    del llm
    time.sleep(1)


def main() -> None:
    multiprocessing.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rollout-jsonl", type=Path, required=True)
    parser.add_argument("--rows", default="13,14,31")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--all2all-backend", default="deepep_low_latency")
    args = parser.parse_args()

    rows = tuple(map(int, args.rows.split(",")))
    selected = _load_rows(args.rollout_jsonl, rows)
    capture_sizes = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]
    engine_args = {
        "model": str(args.model),
        "trust_remote_code": True,
        "tensor_parallel_size": 1,
        "enable_expert_parallel": True,
        "all2all_backend": args.all2all_backend,
        "moe_backend": "deep_gemm",
        "linear_backend": "deep_gemm",
        "kv_cache_dtype": "fp8",
        "gpu_memory_utilization": 0.55,
        "max_model_len": 32768,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": 64,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": args.enable_prefix_caching,
        "enforce_eager": args.enforce_eager,
        "compilation_config": {
            "cudagraph_mode": "NONE" if args.enforce_eager else "PIECEWISE",
            "cudagraph_capture_sizes": [] if args.enforce_eager else capture_sizes,
        },
        "hf_overrides": {
            "expert_dtype": "fp8",
            "quantization_config": {
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "quant_method": "fp8",
                "scale_fmt": "ue8m0",
                "weight_block_size": [128, 128],
            },
        },
        "disable_custom_all_reduce": True,
        "seed": 42,
    }
    signature = inspect.signature(LLM)
    if not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        unknown = sorted(set(engine_args) - set(signature.parameters))
        if unknown:
            raise RuntimeError(f"native vLLM LLM API rejected required args: {unknown}")

    rank_dir = args.output.parent / f".{args.output.stem}-ranks"
    rank_dir.mkdir(parents=True, exist_ok=True)
    master_port = get_open_port()
    processes = []
    for rank in range(4):
        rank_output = rank_dir / f"rank{rank}.json"
        if rank_output.exists():
            rank_output.unlink()
        process = multiprocessing.Process(
            target=_run_rank,
            args=(
                rank,
                4,
                master_port,
                engine_args,
                selected,
                args.repeats,
                rank_output,
            ),
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join(timeout=3300)
        if process.exitcode is None:
            process.kill()
            raise RuntimeError(f"timed out waiting for vLLM rank pid={process.pid}")
        if process.exitcode:
            raise RuntimeError(
                f"native vLLM rank pid={process.pid} exited {process.exitcode}"
            )

    rank_payloads = [
        json.loads((rank_dir / f"rank{rank}.json").read_text()) for rank in range(4)
    ]
    rank_repeats = [payload["packed"] for payload in rank_payloads]
    rank_singletons = [payload["singleton"] for payload in rank_payloads]
    repeats = rank_repeats[0]
    comparisons = [
        _compare(repeats[0], candidate) for candidate in repeats[1:]
    ]
    rank_comparisons = [
        _compare(repeats[0], candidate[0]) for candidate in rank_repeats[1:]
    ]
    singleton_comparisons = [
        _compare(rank_repeats[rank][0], rank_singletons[rank]) for rank in range(4)
    ]
    rollout_comparison = _compare_oracle(selected, repeats[0], "rollout_bits")
    actor_comparison = _compare_oracle(selected, repeats[0], "actor_bits")
    repeat_exact = all(
        row["mismatches"] == 0
        for comparison in comparisons + rank_comparisons + singleton_comparisons
        for row in comparison
    )
    oracle_reports = [
        report
        for comparison in (rollout_comparison, actor_comparison)
        if comparison is not None
        for report in comparison
    ]
    result = {
        "schema": "ds4-native-vllm-three-line-repeat/v1",
        "vllm_origin": inspect.getfile(LLM),
        "model": str(args.model.resolve()),
        "rollout_jsonl": str(args.rollout_jsonl.resolve()),
        "rows": list(rows),
        "engine_args": engine_args,
        "environment": {
            name: os.environ.get(name)
            for name in (
                "VLLM_BATCH_INVARIANT",
                "VLLM_DS4_DECODE_KERNEL",
                "DS4_BI_TOPK_LIB",
                "VLLM_BATCH_INVARIANT_KERNEL_LIB",
                "NVSHMEM_REMOTE_TRANSPORT",
                "NVSHMEM_DISABLE_NCCL",
                "NVSHMEM_MAX_TEAMS",
            )
        },
        "repeats": repeats,
        "rank_repeats": rank_repeats,
        "rank_singletons": rank_singletons,
        "comparisons": comparisons,
        "rank_comparisons": rank_comparisons,
        "singleton_comparisons": singleton_comparisons,
        "rollout_comparison": rollout_comparison,
        "actor_comparison": actor_comparison,
        "repeat_exact": repeat_exact,
        "oracle_exact": bool(oracle_reports)
        and all(report["mismatches"] == 0 for report in oracle_reports),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if not result["repeat_exact"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
