# Porting Notes

This tree is prepared as an experimental Megatron package. The package code
lives under `experimental/lite/megatron/lite`, so users can add
`experimental/lite` to `PYTHONPATH` and import `megatron.lite`.

## Naming Rules

- Use `Megatron Lite` for the component name in docs and comments.
- Use `megatron.lite` for public and internal imports.
- Do not introduce project-specific legacy branding.
- Use `mlite` for the runtime backend key.
- Use `lite` for model implementation names.

## Included Surface

Keep the PR focused on the lite model implementation path:

- Runtime backend: `mlite`.
- Reference comparison backends: `mbridge` for the validated legacy
  Megatron-Core/distopt path and `bridge` for real Megatron-Bridge environments.
- Models: Qwen3 MoE and Qwen3.5 MoE. Dense Qwen3 is not included.
- Model implementations: `lite` only.
- Optimizer primitives: Megatron-Core optimizer wrapping and FSDP2.
- Optional examples: benchmark and VERL launchers under `experimental/lite/examples`.

Keep these out of the first PR unless the scope changes:

- Hybrid model implementation packages.
- Megatron-Bridge model implementation packages.
- Undocumented experiment-specific entrypoints.

## Package Integration

No repository-level packaging changes are made in this experimental drop. The
current layout is importable from source with:

```bash
export PYTHONPATH=/path/to/Megatron-LM/experimental/lite:$PYTHONPATH
```

A future integration step can decide whether to keep the experimental location
or move the tree into the final package location.
