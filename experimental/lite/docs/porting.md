# Porting Notes

This tree is prepared as an experimental Megatron package. The package code
lives under `experimental/lite/megatron/lite`, so users can add
`experimental/lite` to `PYTHONPATH` and import `megatron.lite`.

## Naming Rules

- Use `Megatron Lite` for the component name in docs and comments.
- Use `megatron.lite` for public and internal imports.
- Do not introduce project-specific legacy branding.
- Prefer `lite` for backend names and runtime keys.

## Included Surface

Keep the first PR focused on the lite path:

- Runtime backend: `lite`.
- Models: Qwen3 MoE and Qwen3.5 MoE.
- Model implementations: `lite` only.
- Optimizer primitive: Megatron-Core optimizer wrapping.

Keep these out of the first PR unless the scope changes:

- Hybrid model implementation packages.
- Bridge model/runtime implementation packages.
- FSDP2 optimizer primitives.
- Benchmark scripts and experiment-specific entrypoints.

## Package Integration

No repository-level packaging changes are made in this experimental drop. The
current layout is importable from source with:

```bash
export PYTHONPATH=/path/to/Megatron-LM/experimental/lite:$PYTHONPATH
```

A future integration step can decide whether to keep the experimental location
or move the tree into the final package location.
