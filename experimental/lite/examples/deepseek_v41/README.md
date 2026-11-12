# DeepSeek 4.1 standard training

This model composes unmodified Megatron-Core dev
`7b150acfacee9d1126025f6cbe0f49c588975c51`. Literal ratios 0/2 use native CSA;
ratio 1 composes Core scoring, sparse attention and CP mappings. Shared-source
adapters manage KV/index reuse and candidate filtering. No Core patch is needed.

Core config uses `context_parallel_size=1`, selector=None; real CP comes from
runtime process groups and contiguous THD metadata. The model checks these agree.
DS4.1 overrides Core's Q projection to omit per-head Q RMS, and compressor methods
to retrieve shared sources. Zero-loss indexer projections are frozen; main KV is
trainable. `FrozenIndexerCSA.train` selects one inference-layout forward without
disabling autograd; this also changes the grad-enabled CP1 layout.

## Example

Use a distributed launcher, native release weights and its compressed token map:

```python
from megatron.lite.runtime import create_runtime
from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
from megatron.lite.runtime.contracts import ParallelConfig, RuntimeConfig

backend = MegatronLiteConfig(model_name="deepseek_v41", impl="lite",
    hf_path="/path/to/model", parallel=ParallelConfig(ep=4),
    impl_cfg={"token_map": compressed_token_map, "optimizer": "adam"})
handle = create_runtime(RuntimeConfig(backend="mlite", backend_cfg=backend)).build_model()
```

## Tests and limits

Acceptance generates small random release-format safetensors (FP4 experts, FP8
matrices with E8M0 scales, and native unquantized roles), then uses the strict
loader. No preconverted weights are needed for training. P40E is report-only.
GPU attention tests use `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
`torch.use_deterministic_algorithms(True, warn_only=True)` and
`impl_cfg.deterministic_mode=True`. Ratio-0/1 attention CP2/CP1 forward is bitwise equal; ratio-2 forward
and all CP backward comparisons report rounding differences. Whole-model CP is
report-only. Main attention gradients must be nonzero and finite.
The 24-step random SFT gate freezes data, seed and calibrated Adam learning rate;
last-seven mean loss must be at most 0.8 times first-seven. DCP restores persistent
state and continues three same-batch steps within loss atol/rtol 0.05/0.005.
CPU checkpoint RNG adaptation is test-only; GPU tests use original Core RNG.

Ratio-one CP replicates full window/main KV per rank, limiting long-sequence
memory savings. Core CP needs at least the window's rows on every rank; shorter
packs fail. Report memory peaks: OOM is resource incomplete, not a numerical pass.

Candidate-oracle tests are opt-in and do not vendor official code. Download
`inference/model.py` from `deepseek-ai/DeepSeek-V4.1-Flash` revision
`dba1be0a40aa45a94ad051997016db3960a90277` outside the checkout. Its SHA256 must be
`4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65`.
Set `DS41_RUN_OFFICIAL_REFERENCE=1` and `DS41_REFERENCE_DIR` to its directory,
then run `pytest experimental/lite/tests/unit/deepseek_v41/test_ratio_one.py`.
Without opt-in these tests skip explicitly; once enabled, missing/wrong source
fails. The oracle checks candidate/selection AST semantics, not model numerics.

## Export

`protocol.export_hf_weights(chunks, config, parallel_state)` streams HF/BF16
weights by default; `target="release"` selects FP8 32x32 UE8M0 / MXFP4 storage.
`protocol.save_hf_weights(chunks, path, config, parallel_state, target="release")`
writes bounded HF shards. Every rank must participate, including EP/PP owners.
Frozen Engram tables retain FP8/E8M0 storage and stream bounded row chunks.
Online consumers may use `include_frozen=False` to omit unchanged tables.
Native export is canonical and reload-idempotent; its decoded values obey
quantization rounding bounds, but original release bytes need not be reproduced.
