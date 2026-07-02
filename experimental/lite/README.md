# Megatron Lite

Megatron Lite is an experimental, agentic-native training runtime and native
model implementation layer for Megatron. It is designed for work that needs to
move quickly without giving up Megatron-Core performance: small composable
primitives, explicit model/runtime protocols, and validation recipes that make
changes easy to review and easy to reproduce.

The source lives under `experimental/lite/megatron/lite`, and the public import
path is `megatron.lite`.

Do not import `experimental.lite` from user code. Examples and public APIs should
refer to `megatron.lite`.

## Scope

This initial drop contains:

- A lightweight runtime API in `megatron.lite.runtime`.
- Common training primitives in `megatron.lite.primitive`.
- Lite-only native model implementations for Qwen3 MoE and Qwen3.5 MoE.
- Hugging Face safetensors load/export helpers for the included models.
- Megatron-Core optimizer wrapping for the lite runtime.
- FSDP2 optimizer primitives for supported lite model protocols.
- Reference runtime backends for comparison runs: `mbridge` for the legacy
  package and `bridge` for real Megatron-Bridge environments.
- A benchmark example that can dry-run or execute `mlite`, `mbridge`, and
  `bridge` backends.

This initial drop intentionally does not include:

- Hybrid model implementations.
- Dense Qwen3 model support. The included Qwen3-family path is Qwen3 MoE only.

## Why MLite

- **Agentic-native development surface.** Runtime, model, and primitive code are
  split into reviewable contracts so agents and humans can make targeted changes
  without touching unrelated Megatron subsystems.
- **Native MLite models, not wrapper models.** `backend="mlite"` builds native
  `megatron.lite` model code; reference backends are used only for comparison.
- **Megatron-Core distopt parity.** In deterministic correctness runs against the
  `mbridge` reference backend on the Megatron-Core distributed optimizer path,
  MLite matched loss and grad-norm exactly (`max_abs=0.0`) with no mismatches;
  post-step weights and eval logits were checked by SHA256 fingerprints.
- **Speed-aligned with the Core path.** On an 8x H100 Qwen3.5 MoE benchmark using
  `distopt`, MLite measured 309.433 ms/step and 105,896.935 tokens/s, compared
  with 332.201 ms/step and 98,639.089 tokens/s for `mbridge`, and 334.936
  ms/step and 97,833.496 tokens/s for the real `bridge` path.

The `mbridge` benchmark line is the validated Megatron-Core/distopt reference
used for this PR. The `bridge` line is a separate Megatron-Bridge environment
check and should not be confused with the Core/distopt parity claim.

## Layout

```text
experimental/lite/
  README.md
  docs/                       Design and usage notes
  examples/                   Optional integration and benchmark examples
  skills/                     Agent-agnostic maintenance skills
  megatron/
    lite/
      runtime/                Runtime API, config, and backend registry
      model/                  Model registry and Qwen model implementations
      primitive/              Parallel, checkpoint, optimizer, module, and op primitives
```

For local source-tree use:

```bash
export PYTHONPATH=/path/to/Megatron-LM/experimental/lite:$PYTHONPATH
```

## Public API

```python
from megatron.lite.runtime import MegatronLiteConfig, RuntimeConfig, create_runtime

cfg = RuntimeConfig(
    backend="mlite",
    hf_path="/path/to/hf-model",
    backend_cfg=MegatronLiteConfig(model_name="qwen3_moe", impl="lite"),
)
runtime = create_runtime(cfg)
handle = runtime.build_model()
```

`backend="mlite"` selects the Megatron Lite runtime backend. `impl="lite"`
selects the model implementation inside the registered model family.
`backend="mbridge"` selects the legacy `mbridge` reference backend used by the
validated benchmark example. `backend="bridge"` selects the Megatron-Bridge
runtime backend and requires an environment where `import megatron.bridge` works
when the model is built.

Canonical model names currently registered by default:

- `qwen3_moe`: Qwen3 MoE lite implementation. Use this name in new configs.
- `qwen3_5`: Qwen3.5 MoE lite implementation.

Compatibility names:

- `qwen3`: legacy alias for the Qwen3 MoE implementation only. It does not mean
  dense Qwen3 support. HF `model_type` values `qwen3_moe` and `qwen2_moe`
  currently resolve through this compatibility path.

## Benchmark And Correctness Signoff

The validated benchmark and correctness commands live in
[`examples/bench/README.md`](examples/bench/README.md). The signoff setup uses
Qwen3.5 MoE, `optimizer_backend=distopt`, deterministic mode for strict
correctness, and identical synthetic input streams for paired performance runs.

Reproduce the strict MLite vs Megatron-Core/distopt comparison with:

```bash
export MEGATRON_LITE_DETERMINISTIC=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

HF_PATH=/models/Qwen3.5-35B-A3B \
REFERENCE_BACKEND=mbridge \
DRY_RUN=0 \
bash experimental/lite/examples/bench/scripts/run_qwen35_correctness_pair.sh
```

## Docs

- [Architecture](docs/architecture.md)
- [Runtime](docs/runtime.md)
- [Models](docs/models.md)
- [Porting Notes](docs/porting.md)
- [Skills](skills/README.md)
- [Bench Example](examples/bench/README.md)

## Acknowledgements

The Qwen3 MoE LoRA adapter support follows Mind-Lab's PEFT/Mint-compatible
adapter work. Thanks to Mind-Lab for the reference implementation and guidance.
