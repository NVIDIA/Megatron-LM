"""Benchmark MLite and Megatron-Bridge runtime backends.

Run from the Megatron-LM repo root after adding ``experimental/lite`` to
``PYTHONPATH``. ``--dry-run`` validates config construction without importing
Megatron-Bridge or initializing distributed state.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

_EXPERIMENTAL_LITE_ROOT = Path(__file__).resolve().parents[2]
if str(_EXPERIMENTAL_LITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTAL_LITE_ROOT))

from megatron.lite.runtime import RuntimeConfig, create_runtime
from megatron.lite.runtime.backends.bridge.config import BridgeConfig
from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
from megatron.lite.runtime.contracts.config import OptimizerConfig, ParallelConfig

from examples.bench.results import StepTrace
from examples.bench.session import PretrainSessionConfig, run_pretrain_session


@dataclass
class BenchCliConfig:
    backend: str = "mlite"
    hf_path: str = ""
    model_name: str = "qwen3_moe"
    impl: str = "lite"
    tp: int = 1
    etp: int | None = None
    ep: int = 1
    pp: int = 1
    vpp: int = 1
    cp: int = 1
    steps: int = 2
    warmup: int = 0
    num_microbatches: int = 1
    seq_len: int = 2048
    seed: int = 42
    device: str = "cuda"
    use_thd: bool = False
    same_data_across_dp: bool = False
    no_optimizer: bool = False
    skip_load_hf_weights: bool = False
    skip_optimizer_build: bool = False
    keep_experts: int | None = None
    truncate_layers: int | None = None
    disable_mtp: bool = False
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 0.1
    optimizer_clip_grad: float = 1.0
    override_ddp_json: str = "{}"
    override_transformer_json: str = "{}"
    override_optimizer_json: str = "{}"
    impl_cfg_json: str = "{}"
    dry_run: bool = False
    output_json: str | None = None


def _json_mapping(raw: str, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be a JSON object: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return value


def _parallel_config(cfg: BenchCliConfig) -> ParallelConfig:
    return ParallelConfig(tp=cfg.tp, etp=cfg.etp, ep=cfg.ep, pp=cfg.pp, vpp=cfg.vpp, cp=cfg.cp)


def _optimizer_config(cfg: BenchCliConfig) -> OptimizerConfig:
    return OptimizerConfig(
        lr=cfg.optimizer_lr,
        weight_decay=cfg.optimizer_weight_decay,
        clip_grad=cfg.optimizer_clip_grad,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        use_precision_aware_optimizer=True,
        decoupled_weight_decay=True,
    )


def _set_field(config_obj: Any, name: str, value: Any) -> Any:
    if is_dataclass(config_obj):
        return replace(config_obj, **{name: value})
    setattr(config_obj, name, value)
    return config_obj


def _get_model_config_root(config_obj: Any) -> Any:
    return getattr(config_obj, "text_config", config_obj)


def _disable_mtp(config_obj: Any) -> Any:
    root = _get_model_config_root(config_obj)
    for attr in ("mtp_num_hidden_layers", "num_nextn_predict_layers"):
        if hasattr(root, attr):
            root = _set_field(root, attr, 0)
    if hasattr(root, "mtp_layer_types"):
        root = _set_field(root, "mtp_layer_types", [])
    if root is not config_obj and hasattr(config_obj, "text_config"):
        return _set_field(config_obj, "text_config", root)
    return root


def _make_mlite_model_config_hook(cfg: BenchCliConfig):
    hooks = []
    if cfg.keep_experts is not None:
        keep_experts = cfg.keep_experts

        def keep_experts_hook(model_cfg):
            old_num = getattr(model_cfg, "num_experts", None)
            old_topk = getattr(model_cfg, "num_experts_per_tok", None)
            if old_num is None or old_topk is None:
                raise ValueError("keep_experts requires model config with MoE expert metadata.")
            if keep_experts <= 0 or keep_experts > old_num:
                raise ValueError(f"keep_experts must be in [1, {old_num}], got {keep_experts}.")
            return replace(
                model_cfg,
                num_experts=keep_experts,
                num_experts_per_tok=min(old_topk, keep_experts),
            )

        hooks.append(keep_experts_hook)

    if cfg.truncate_layers is not None:
        keep_layers = cfg.truncate_layers

        def truncate_layers_hook(model_cfg):
            old_layers = getattr(model_cfg, "num_hidden_layers", None)
            layer_types = getattr(model_cfg, "layer_types", None)
            if old_layers is None or layer_types is None:
                raise ValueError("truncate_layers requires num_hidden_layers and layer_types.")
            if keep_layers <= 0 or keep_layers > old_layers:
                raise ValueError(f"truncate_layers must be in [1, {old_layers}], got {keep_layers}.")
            return replace(
                model_cfg,
                num_hidden_layers=keep_layers,
                layer_types=list(layer_types[:keep_layers]),
            )

        hooks.append(truncate_layers_hook)

    if cfg.disable_mtp:
        hooks.append(_disable_mtp)

    if not hooks:
        return None

    def composed(model_cfg):
        for hook in hooks:
            model_cfg = hook(model_cfg)
        return model_cfg

    return composed


def _make_bridge_post_init_hook(cfg: BenchCliConfig):
    hooks = []
    if cfg.keep_experts is not None:
        keep_experts = cfg.keep_experts

        def keep_experts_hook(bridge) -> None:
            hf_cfg = _get_model_config_root(bridge.hf_config)
            old_num = getattr(hf_cfg, "num_experts", None)
            old_topk = getattr(hf_cfg, "num_experts_per_tok", None)
            if old_num is None or old_topk is None:
                raise ValueError("keep_experts requires HF config with MoE expert metadata.")
            if keep_experts <= 0 or keep_experts > old_num:
                raise ValueError(f"keep_experts must be in [1, {old_num}], got {keep_experts}.")
            hf_cfg.num_experts = keep_experts
            hf_cfg.num_experts_per_tok = min(old_topk, keep_experts)
            bridge.config = bridge._build_config()

            if hasattr(bridge, "_weight_to_mcore_format"):
                original = bridge._weight_to_mcore_format

                def patched(name: str, hf_weights: list):
                    if "mlp.router.weight" in name and len(hf_weights) == 1:
                        hf_weights = [hf_weights[0][:keep_experts].contiguous()]
                    return original(name, hf_weights)

                bridge._weight_to_mcore_format = patched

        hooks.append(keep_experts_hook)

    if cfg.truncate_layers is not None:
        keep_layers = cfg.truncate_layers

        def truncate_layers_hook(bridge) -> None:
            hf_cfg = _get_model_config_root(bridge.hf_config)
            old_layers = getattr(hf_cfg, "num_hidden_layers", None)
            if old_layers is None:
                raise ValueError("truncate_layers requires HF config with num_hidden_layers.")
            if keep_layers <= 0 or keep_layers > old_layers:
                raise ValueError(f"truncate_layers must be in [1, {old_layers}], got {keep_layers}.")
            hf_cfg.num_hidden_layers = keep_layers
            if hasattr(hf_cfg, "layer_types"):
                hf_cfg.layer_types = list(hf_cfg.layer_types[:keep_layers])
            bridge.config = bridge._build_config()

        hooks.append(truncate_layers_hook)

    if cfg.disable_mtp:

        def disable_mtp_hook(bridge) -> None:
            hf_cfg = _get_model_config_root(bridge.hf_config)
            for attr in ("mtp_num_hidden_layers", "num_nextn_predict_layers"):
                if hasattr(hf_cfg, attr):
                    setattr(hf_cfg, attr, 0)
            if hasattr(hf_cfg, "mtp_layer_types"):
                hf_cfg.mtp_layer_types = []
            bridge.config = bridge._build_config()

        hooks.append(disable_mtp_hook)

    if not hooks:
        return None

    def composed(bridge) -> None:
        for hook in hooks:
            hook(bridge)

    return composed


def build_runtime_config(cfg: BenchCliConfig) -> RuntimeConfig:
    parallel = _parallel_config(cfg)
    optimizer = _optimizer_config(cfg)
    optimizer_overrides = _json_mapping(cfg.override_optimizer_json, name="override_optimizer_json")

    if cfg.backend == "mlite":
        for key, value in optimizer_overrides.items():
            setattr(optimizer, key, value)
        impl_cfg = _json_mapping(cfg.impl_cfg_json, name="impl_cfg_json")
        impl_cfg.setdefault("use_thd", cfg.use_thd)
        backend_cfg = MegatronLiteConfig(
            model_name=cfg.model_name,
            impl=cfg.impl,
            hf_path=cfg.hf_path,
            parallel=parallel,
            optimizer=optimizer,
            load_hf_weights=not cfg.skip_load_hf_weights,
            impl_cfg=impl_cfg,
            model_config_hook=_make_mlite_model_config_hook(cfg),
        )
    elif cfg.backend == "bridge":
        impl_cfg = _json_mapping(cfg.impl_cfg_json, name="impl_cfg_json")
        if impl_cfg:
            raise ValueError("bridge backend does not accept impl_cfg_json.")
        backend_cfg = BridgeConfig(
            model_name=cfg.model_name,
            parallel=parallel,
            optimizer=optimizer,
            load_hf_weights=not cfg.skip_load_hf_weights,
            build_optimizer=not cfg.skip_optimizer_build,
            override_ddp_config=_json_mapping(cfg.override_ddp_json, name="override_ddp_json"),
            override_transformer_config=_json_mapping(
                cfg.override_transformer_json,
                name="override_transformer_json",
            ),
            override_optimizer_config=optimizer_overrides,
            bridge_post_init=_make_bridge_post_init_hook(cfg),
        )
    else:
        raise ValueError(f"backend must be 'mlite' or 'bridge', got {cfg.backend!r}.")

    return RuntimeConfig(backend=cfg.backend, hf_path=cfg.hf_path, backend_cfg=backend_cfg)


def build_session_config(cfg: BenchCliConfig) -> PretrainSessionConfig:
    return PretrainSessionConfig(
        steps=cfg.steps,
        warmup=cfg.warmup,
        num_microbatches=cfg.num_microbatches,
        seq_len=cfg.seq_len,
        seed=cfg.seed,
        device=cfg.device,
        use_thd=cfg.use_thd,
        same_data_across_dp=cfg.same_data_across_dp,
        no_optimizer=cfg.no_optimizer,
    )


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: _to_jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if callable(value):
        return f"<callable:{getattr(value, '__name__', type(value).__name__)}>"
    if isinstance(value, Path):
        return str(value)
    return value


def build_dry_run_plan(cfg: BenchCliConfig) -> dict[str, Any]:
    return {
        "dry_run": True,
        "runtime": _to_jsonable(build_runtime_config(cfg)),
        "session": _to_jsonable(build_session_config(cfg)),
        "notes": [
            "Dry-run validates config construction only.",
            "Run under torchrun for real benchmark execution.",
        ],
    }


def _step_reporter(trace: StepTrace) -> None:
    parts = [
        "[MLITE_BENCH_STEP]",
        f"step={trace.step}",
        f"loss={trace.loss:.6f}",
        f"grad_norm={trace.grad_norm:.6f}",
        f"step_ms={trace.step_ms:.3f}",
        f"peak_mem_gb={(trace.peak_mem_gb or 0.0):.3f}",
    ]
    if trace.tflops_per_gpu is not None:
        parts.append(f"tflops_per_gpu={trace.tflops_per_gpu:.3f}")
    print(" ".join(parts), flush=True)


def run(cfg: BenchCliConfig) -> dict[str, Any]:
    if cfg.dry_run:
        return build_dry_run_plan(cfg)

    rt_cfg = build_runtime_config(cfg)
    rt = create_runtime(rt_cfg)
    handle = rt.build_model()
    result = run_pretrain_session(
        rt,
        handle,
        build_session_config(cfg),
        step_reporter=_step_reporter,
    )
    return result.to_dict()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["mlite", "bridge"], default="mlite")
    parser.add_argument("--hf-path", default="")
    parser.add_argument("--model-name", default="qwen3_moe")
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
    parser.add_argument("--seq-len", type=int, default=2048)
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
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-json", default=None)
    return parser


def parse_args(argv: list[str] | None = None) -> BenchCliConfig:
    ns = _parser().parse_args(argv)
    return BenchCliConfig(**vars(ns))


def main(argv: list[str] | None = None) -> dict[str, Any]:
    cfg = parse_args(argv)
    artifact = run(cfg)
    text = json.dumps(artifact, indent=2, sort_keys=True)
    print(text, flush=True)
    if cfg.output_json is not None:
        output_path = Path(cfg.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    return artifact


if __name__ == "__main__":
    main()
