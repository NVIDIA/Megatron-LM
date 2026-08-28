# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Deterministic correctness runner for MLite and reference backends."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

_EXPERIMENTAL_LITE_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[4]
for root in (str(_REPO_ROOT), str(_EXPERIMENTAL_LITE_ROOT)):
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)

from megatron.lite.primitive.deterministic import set_deterministic
from megatron.lite.runtime import create_runtime

from examples.bench.bench import (
    BenchCliConfig,
    _install_deepep_topk_layout_normalizer,
    build_runtime_config,
    build_session_config,
)
from examples.bench.results import compare_correctness_artifacts, load_result_artifact
from examples.bench.session import _global_grad_norm_without_step, _make_data_iter


def _distributed_rank() -> int:
    for name in ("RANK", "SLURM_PROCID"):
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            return int(raw)
        except ValueError:
            continue
    return 0


def _distributed_world_size() -> int:
    for name in ("WORLD_SIZE", "SLURM_NTASKS"):
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            return int(raw)
        except ValueError:
            continue
    return 1


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _scalar(value: float | int | torch.Tensor | None) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("scalar fingerprint requires a scalar tensor.")
        value = float(value.detach().cpu().float().item())
    value_f = float(0.0 if value is None else value)
    return {
        "value": value_f,
        "float_hex": value_f.hex(),
        "sha256_f64_be": hashlib.sha256(struct.pack(">d", value_f)).hexdigest(),
    }


def _hash_tensor(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
    if hasattr(tensor, "to_local"):
        tensor = tensor.to_local()
    t = tensor.detach().contiguous().cpu()
    raw = t.view(torch.uint8).numpy().tobytes()
    as_bf16 = t.to(torch.bfloat16).contiguous() if t.is_floating_point() else None
    summary = t.float() if t.is_floating_point() else None
    result = {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if as_bf16 is not None:
        result["sha256_as_bf16"] = hashlib.sha256(
            as_bf16.view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        if os.environ.get("MLITE_CORRECTNESS_INCLUDE_VALUES") == "1":
            result["values"] = [float(x) for x in t.float().reshape(-1).tolist()]
    if summary is not None:
        flat = summary.reshape(-1)
        result["summary"] = {
            "min": float(flat.min().item()) if flat.numel() else 0.0,
            "max": float(flat.max().item()) if flat.numel() else 0.0,
            "mean": float(flat.mean().item()) if flat.numel() else 0.0,
            "l2": float(flat.norm().item()) if flat.numel() else 0.0,
            "first8": [float(x) for x in flat[:8].tolist()],
        }
    return result


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _record_activation_probe(
    records: list[dict[str, Any]],
    name: str,
    output: Any,
    *,
    record_grad: bool = False,
    tensor_hooks: list[Any] | None = None,
) -> None:
    tensor = _first_tensor(output)
    record = {"name": name, "found": True, "tensor": _hash_tensor(tensor)}
    if record_grad:
        record["grad"] = None
        record["grad_found"] = isinstance(tensor, torch.Tensor) and tensor.requires_grad
        if isinstance(tensor, torch.Tensor) and tensor.requires_grad:

            def _grad_hook(grad, _record=record):
                _record["grad"] = _hash_tensor(grad)

            hook = tensor.register_hook(_grad_hook)
            if tensor_hooks is not None:
                tensor_hooks.append(hook)
    records.append(record)


def _resolve_probe_module(modules: dict[str, Any], name: str) -> tuple[str, Any] | None:
    module = modules.get(name)
    if module is not None:
        return name, module
    matches = [
        (candidate_name, candidate)
        for candidate_name, candidate in modules.items()
        if candidate_name.endswith(f".{name}")
    ]
    if len(matches) == 1:
        return matches[0]
    return None


@contextmanager
def _activation_probe_context(handle, probe_names: list[str], *, record_grad: bool = False):
    records: list[dict[str, Any]] = []
    hooks = []
    tensor_hooks = []
    patched_methods = []
    modules = dict(handle._model.named_modules())
    for name in probe_names:
        probe_name = name
        record_input = probe_name.endswith(":input")
        lookup_name = probe_name[:-6] if record_input else probe_name
        if "::" in lookup_name:
            module_name, method_name = lookup_name.split("::", 1)
            resolved = _resolve_probe_module(modules, module_name)
            if resolved is None or not hasattr(resolved[1], method_name):
                records.append({"name": name, "found": False})
                continue
            resolved_name, module = resolved
            original = getattr(module, method_name)

            def _wrapped(
                *args,
                _original=original,
                _probe_name=name,
                _resolved_name=resolved_name,
                _module_type=type(module).__module__ + "." + type(module).__qualname__,
                _record_input=record_input,
                _method_name=method_name,
                **kwargs,
            ):
                if _record_input:
                    _record_activation_probe(
                        records,
                        _probe_name,
                        args,
                        record_grad=record_grad,
                        tensor_hooks=tensor_hooks,
                    )
                    records[-1]["resolved_name"] = f"{_resolved_name}::{_method_name}:input"
                    records[-1]["module_type"] = _module_type
                    return _original(*args, **kwargs)
                output = _original(*args, **kwargs)
                _record_activation_probe(
                    records, _probe_name, output, record_grad=record_grad, tensor_hooks=tensor_hooks
                )
                records[-1]["resolved_name"] = f"{_resolved_name}::{_method_name}"
                records[-1]["module_type"] = _module_type
                return output

            setattr(module, method_name, _wrapped)
            patched_methods.append((module, method_name, original))
            continue

        resolved = _resolve_probe_module(modules, lookup_name)
        if resolved is None:
            records.append({"name": name, "found": False})
            continue
        resolved_name, module = resolved

        if record_input:

            def _pre_hook(_module, args, probe_name=name, resolved_probe_name=resolved_name):
                _record_activation_probe(
                    records, probe_name, args, record_grad=record_grad, tensor_hooks=tensor_hooks
                )
                records[-1]["resolved_name"] = f"{resolved_probe_name}:input"
                records[-1]["module_type"] = (
                    type(_module).__module__ + "." + type(_module).__qualname__
                )

            hooks.append(module.register_forward_pre_hook(_pre_hook))
            continue

        def _hook(_module, _args, output, probe_name=name, resolved_probe_name=resolved_name):
            _record_activation_probe(
                records, probe_name, output, record_grad=record_grad, tensor_hooks=tensor_hooks
            )
            records[-1]["resolved_name"] = resolved_probe_name
            records[-1]["module_type"] = type(_module).__module__ + "." + type(_module).__qualname__

        hooks.append(module.register_forward_hook(_hook))
    try:
        yield records
    finally:
        for hook in tensor_hooks:
            hook.remove()
        for hook in hooks:
            hook.remove()
        for module, method_name, original in patched_methods:
            setattr(module, method_name, original)


def _update_hash_with_tensor(h: Any, name: str, tensor: torch.Tensor) -> None:
    if hasattr(tensor, "to_local"):
        tensor = tensor.to_local()
    t = tensor.detach().contiguous().cpu()
    h.update(name.encode("utf-8"))
    h.update(b"\0")
    h.update(str(t.dtype).encode("ascii"))
    h.update(b"\0")
    h.update(json.dumps(list(t.shape), separators=(",", ":")).encode("ascii"))
    h.update(b"\0")
    h.update(t.view(torch.uint8).numpy().tobytes())
    h.update(b"\0")


def _model_chunks(handle) -> list[Any]:
    chunks = handle._extras.get("model_chunks")
    if chunks is None:
        chunks = handle._extras.get("model_list")
    if chunks is None:
        chunks = [handle._model]
    return list(chunks)


def _grad_fingerprint(handle) -> dict[str, Any]:
    h = hashlib.sha256()
    count = 0
    details = []
    include_details = os.environ.get("MLITE_CORRECTNESS_GRAD_DETAILS") == "1"
    for chunk_idx, chunk in enumerate(_model_chunks(handle)):
        for name, param in sorted(chunk.named_parameters(), key=lambda item: item[0]):
            grad = param.grad
            if grad is None:
                grad = getattr(param, "main_grad", None)
            if grad is None:
                continue
            fingerprint_name = f"{chunk_idx}:{name}"
            _update_hash_with_tensor(h, fingerprint_name, grad)
            if include_details:
                detail = _hash_tensor(grad)
                assert detail is not None
                detail["name"] = fingerprint_name
                details.append(detail)
            count += 1
    result = {"sha256": h.hexdigest(), "tensor_count": count}
    if include_details:
        result["details"] = details
    return result


def _weight_fingerprint(rt, handle) -> dict[str, Any]:
    h = hashlib.sha256()
    count = 0
    details = []
    include_details = os.environ.get("MLITE_CORRECTNESS_WEIGHT_DETAILS") == "1"
    for name, tensor in sorted(rt.export_weights(handle), key=lambda item: item[0]):
        _update_hash_with_tensor(h, str(name), tensor)
        if include_details:
            detail = _hash_tensor(tensor)
            assert detail is not None
            detail["name"] = str(name)
            details.append(detail)
        count += 1
    result = {"sha256": h.hexdigest(), "tensor_count": count}
    if include_details:
        result["details"] = details
    return result


def _forward_logits(rt, handle, batch: Any) -> torch.Tensor | None:
    result = rt.forward_backward(
        handle,
        iter([batch]),
        loss_fn=None,
        num_microbatches=1,
        forward_only=True,
        router_replay={"action": "replay"}
        if getattr(batch, "routed_experts", None) is not None
        else None,
    )
    output = result.model_output
    return output.vocab_parallel_logits if output.vocab_parallel_logits is not None else (-output.log_probs if output.log_probs is not None else None)


def _fixed_route_batches(data_iter, handle):
    model_cfg = handle._extras.get("model_cfg")
    num_layers = int(getattr(model_cfg, "num_hidden_layers"))
    topk = int(getattr(model_cfg, "num_experts_per_tok"))
    num_experts = int(getattr(model_cfg, "n_routed_experts"))
    for batch in data_iter:
        rows = []
        for length_tensor in batch.seq_lens:
            length = int(length_tensor.item())
            tokens = torch.arange(length, device=batch.input_ids.device)
            layers = torch.arange(num_layers, device=batch.input_ids.device)
            slots = torch.arange(topk, device=batch.input_ids.device)
            routes = (
                tokens[:, None, None]
                + layers[None, :, None] * topk
                + slots[None, None, :]
            ).remainder(num_experts)
            rows.append(routes)
        batch.routed_experts = torch.nested.as_nested_tensor(
            rows, layout=torch.jagged
        )
        batch.r3_replay_mask = torch.ones(
            batch.input_ids.numel(), dtype=torch.bool, device=batch.input_ids.device
        )
        yield batch


def _batch_fingerprint(batch: Any) -> dict[str, Any]:
    return {
        name: _hash_tensor(getattr(batch, name, None))
        for name in ("input_ids", "labels", "seq_lens", "loss_mask", "position_ids")
    }


def run_backend(
    cfg: BenchCliConfig,
    *,
    hash_weights: bool = True,
    activation_probe_names: list[str] | None = None,
    fixed_router_replay: bool = False,
) -> dict[str, Any]:
    os.environ["MEGATRON_LITE_DETERMINISTIC"] = "1"
    set_deterministic(cfg.seed)

    if cfg.normalize_deepep_topk_layout:
        _install_deepep_topk_layout_normalizer()
    rt_cfg = build_runtime_config(cfg)
    rt = create_runtime(rt_cfg)
    handle = rt.build_model()
    session_cfg = build_session_config(cfg)

    eval_iter = _make_data_iter(handle, session_cfg)
    if fixed_router_replay:
        eval_iter = _fixed_route_batches(eval_iter, handle)
    eval_batch = next(eval_iter)
    initial_weights = _weight_fingerprint(rt, handle) if hash_weights else None
    input_fingerprint = _batch_fingerprint(eval_batch)
    activation_probe_names = list(activation_probe_names or [])
    with _activation_probe_context(handle, activation_probe_names) as activation_probes:
        with rt.eval_mode(handle):
            eval_logits = _hash_tensor(_forward_logits(rt, handle, eval_batch))

    data_iter = _make_data_iter(handle, session_cfg)
    if fixed_router_replay:
        data_iter = _fixed_route_batches(data_iter, handle)
    steps: list[dict[str, Any]] = []
    with rt.train_mode(handle):
        for step in range(session_cfg.steps):
            with _activation_probe_context(
                handle, activation_probe_names, record_grad=True
            ) as train_activation_probes:
                rt.zero_grad(handle)
                _sync(session_cfg.device)
                result = rt.forward_backward(
                    handle,
                    data_iter,
                    loss_fn=None,
                    num_microbatches=session_cfg.num_microbatches,
                    router_replay={"action": "replay"} if fixed_router_replay else None,
                )
                _sync(session_cfg.device)
                output = result.model_output
                logits = _hash_tensor(output.vocab_parallel_logits if output.vocab_parallel_logits is not None else (-output.log_probs if output.log_probs is not None else None))
                grads = _grad_fingerprint(handle)

                if session_cfg.no_optimizer:
                    update_successful = True
                    grad_norm = _global_grad_norm_without_step(handle)
                    num_zeros = 0
                else:
                    update_successful, grad_norm, num_zeros = rt.optimizer_step(handle)
                    rt.lr_scheduler_step(handle)
                _sync(session_cfg.device)
                loss = result.metrics.get("loss")
                if loss is None:
                    loss = result.model_output.loss

            steps.append(
                {
                    "step": step,
                    "loss": _scalar(0.0 if loss is None else loss),
                    "logits": logits,
                    "grad_fingerprint": grads,
                    "grad_norm": _scalar(grad_norm),
                    "update_successful": bool(update_successful),
                    "num_zeros": None if num_zeros is None else int(num_zeros),
                    "post_step_weights": _weight_fingerprint(rt, handle) if hash_weights else None,
                    "train_activation_probes": train_activation_probes,
                }
            )

    artifact = {
        "kind": "mlite_bench_correctness",
        "backend": cfg.backend,
        "model_name": cfg.model_name,
        "impl": cfg.impl,
        "seed": cfg.seed,
        "seq_len": cfg.seq_len,
        "num_microbatches": cfg.num_microbatches,
        "steps": steps,
        "eval_logits": eval_logits,
        "initial_weights": initial_weights,
        "input_fingerprint": input_fingerprint,
        "activation_probes": activation_probes,
        "metadata": {
            "deterministic": os.environ.get("MEGATRON_LITE_DETERMINISTIC") == "1",
            "determinism_env": {
                name: os.environ.get(name)
                for name in (
                    "MEGATRON_LITE_DETERMINISTIC",
                    "VLLM_BATCH_INVARIANT",
                    "VERL_FULL_DETERMINISM",
                    "CUDA_DEVICE_MAX_CONNECTIONS",
                    "CUBLAS_WORKSPACE_CONFIG",
                    "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
                    "NCCL_ALGO",
                )
            },
            "provenance": {
                "source_sha": os.environ.get("MLITE_SOURCE_SHA"),
                "container_image": os.environ.get("MLITE_CONTAINER_IMAGE"),
            },
            "rank": _distributed_rank(),
            "world_size": _distributed_world_size(),
            "hash_weights": hash_weights,
            "same_data_across_dp": cfg.same_data_across_dp,
            "use_thd": cfg.use_thd,
            "load_hf_weights": not cfg.skip_load_hf_weights,
            "build_optimizer": not cfg.skip_optimizer_build,
            "parallel": {
                "tp": cfg.tp,
                "etp": cfg.etp,
                "ep": cfg.ep,
                "pp": cfg.pp,
                "vpp": cfg.vpp,
                "cp": cfg.cp,
            },
            "optimizer": {
                "disabled": cfg.no_optimizer,
                "lr": cfg.optimizer_lr,
                "weight_decay": cfg.optimizer_weight_decay,
                "clip_grad": cfg.optimizer_clip_grad,
            },
            "impl_cfg": json.loads(cfg.impl_cfg_json),
            "fp8": {
                "fused_weight_quant": os.environ.get(
                    "MLITE_VLLM_FUSED_WEIGHT_QUANT", "1"
                ),
                "fused_ue8m0_weight_quant": os.environ.get(
                    "MLITE_VLLM_FUSED_UE8M0_WEIGHT_QUANT", "1"
                ),
                "batched_grouped_weight_quant": os.environ.get(
                    "MLITE_VLLM_BATCHED_GROUPED_WEIGHT_QUANT", "1"
                ),
            },
            "fixed_router_replay": fixed_router_replay,
        },
    }
    close = getattr(rt, "close", None)
    if close is not None:
        close(handle)
    return artifact


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=["mlite", "bridge", "mbridge"], required=True)
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--model-name", default="qwen3_5")
    parser.add_argument("--impl", default="lite")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--etp", type=int, default=None)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--vpp", type=int, default=1)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--num-microbatches", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-thd", action="store_true")
    parser.add_argument("--same-data-across-dp", action="store_true")
    parser.add_argument("--no-optimizer", action="store_true")
    parser.add_argument("--skip-load-hf-weights", action="store_true")
    parser.add_argument("--skip-optimizer-build", action="store_true")
    parser.add_argument("--keep-experts", type=int, default=None)
    parser.add_argument("--truncate-layers", type=int, default=None)
    parser.add_argument("--disable-mtp", action="store_true")
    parser.add_argument("--optimizer-lr", type=float, default=1e-4)
    parser.add_argument("--optimizer-weight-decay", type=float, default=0.1)
    parser.add_argument("--optimizer-clip-grad", type=float, default=1.0)
    parser.add_argument("--override-ddp-json", default="{}")
    parser.add_argument("--override-transformer-json", default="{}")
    parser.add_argument("--override-optimizer-json", default="{}")
    parser.add_argument("--impl-cfg-json", default="{}")
    parser.add_argument(
        "--normalize-deepep-topk-layout",
        action="store_true",
        help="Benchmark-only compatibility shim for DeepEP builds requiring contiguous top-k indices.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--skip-weight-hash", action="store_true")
    parser.add_argument("--activation-probes-json", default="{}")
    parser.add_argument(
        "--fixed-router-replay",
        action="store_true",
        help="Replay deterministic per-token expert IDs while retaining live router scores.",
    )


def _activation_probe_names(raw: str, backend: str) -> list[str]:
    value = json.loads(raw)
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, dict):
        selected = value.get(backend, [])
        if not isinstance(selected, list):
            raise ValueError(f"activation probe list for {backend!r} must be a JSON list.")
        return [str(item) for item in selected]
    raise ValueError("activation_probes_json must be a JSON list or backend-to-list mapping.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="run one backend and write a correctness artifact")
    _add_run_args(run_p)

    cmp_p = sub.add_parser("compare", help="strictly compare two correctness artifacts")
    cmp_p.add_argument("baseline")
    cmp_p.add_argument("candidate")
    cmp_p.add_argument("--output-json", default=None)
    cmp_p.add_argument("--fail-on-mismatch", action="store_true")
    cmp_p.add_argument("--loss-atol", type=float, default=0.0)
    cmp_p.add_argument("--loss-rtol", type=float, default=0.0)
    cmp_p.add_argument("--grad-atol", type=float, default=0.0)
    cmp_p.add_argument("--grad-rtol", type=float, default=0.0)
    cmp_p.add_argument("--tensor-atol", type=float, default=0.0)
    cmp_p.add_argument("--tensor-rtol", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ns = _parser().parse_args(argv)
    if ns.command == "compare":
        result = compare_correctness_artifacts(
            load_result_artifact(ns.baseline),
            load_result_artifact(ns.candidate),
            loss_atol=ns.loss_atol,
            loss_rtol=ns.loss_rtol,
            grad_atol=ns.grad_atol,
            grad_rtol=ns.grad_rtol,
            tensor_atol=ns.tensor_atol,
            tensor_rtol=ns.tensor_rtol,
        )
        text = json.dumps(result, indent=2, sort_keys=True)
        print(text, flush=True)
        if ns.output_json:
            Path(ns.output_json).write_text(text + "\n", encoding="utf-8")
        if ns.fail_on_mismatch and not result["passed"]:
            raise SystemExit(1)
        return result

    cfg = BenchCliConfig(
        **{k: v for k, v in vars(ns).items() if k in BenchCliConfig.__dataclass_fields__}
    )
    artifact = run_backend(
        cfg,
        hash_weights=not ns.skip_weight_hash,
        activation_probe_names=_activation_probe_names(ns.activation_probes_json, cfg.backend),
        fixed_router_replay=ns.fixed_router_replay,
    )
    text = json.dumps(artifact, indent=2, sort_keys=True)
    output_path = Path(ns.output_json)
    rank = _distributed_rank()
    world_size = _distributed_world_size()
    if world_size > 1:
        rank_output_path = output_path.with_name(
            f"{output_path.stem}.rank{rank}{output_path.suffix}"
        )
        rank_output_path.parent.mkdir(parents=True, exist_ok=True)
        rank_output_path.write_text(text + "\n", encoding="utf-8")
    if rank == 0:
        print(text, flush=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    return artifact


if __name__ == "__main__":
    main()
