# DeepSeek 4.1 standard path: validation contract

The standard model composes Megatron-Core dev components. It does not require
run-to-run determinism. The implementation is being validated; component tests
are not evidence of a complete trained model.

Core reference: `NVIDIA/Megatron-LM` dev
`7b150acfacee9d1126025f6cbe0f49c588975c51`. No Core patch is installed.
Literal compression ratios zero and two use native CSA. Ratio one has a
model-specific forward composition over native scoring, sparse attention and
context-parallel mappings. Cross-layer source state and candidate filtering
belong to the model. Its context-parallel path replicates the full window/main
KV on each rank; this is a memory limitation for long sequences.

Core configuration declares `context_parallel_size=1` and no attention selector.
The model records the real CP size and checks its runtime process group and
contiguous THD `PackedSeqParams.cp_group`. Core CSA/Hybrid use those runtime
objects; the stock MLA class is not in this attention's MRO. RoPE lengths come
from packed maximum lengths, without calling `get_rotary_seq_len`.

Index selection has no differentiable LP loss. With zero auxiliary indexer loss,
its projection parameters are explicitly frozen and listed by
`ds41_frozen_parameters`; the main KV remains trainable. This is the standard
post-training policy, not a claim about an official training implementation.

## External official selection oracle

Ordinary unit tests need no downloaded official source. Selection-oracle tests
skip with an explicit reason unless `DS41_RUN_OFFICIAL_REFERENCE=1` is set.
Once enabled, a missing file or a wrong hash is a failure. Acceptance runs must
enable the oracle; a skipped oracle is not a numerical pass.

Download the official source outside this checkout. It is not vendored:

```sh
mkdir -p /tmp/ds41-fixture-reference
curl --fail --location \
  https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py \
  --output /tmp/ds41-fixture-reference/model.py
printf '%s  %s\n' \
  4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65 \
  /tmp/ds41-fixture-reference/model.py | sha256sum --check
DS41_RUN_OFFICIAL_REFERENCE=1 DS41_REFERENCE_DIR=/tmp/ds41-fixture-reference \
  python -m pytest experimental/lite/tests/unit/deepseek_v41/test_ratio_one.py
```

The CPU oracle executes the pinned official candidate function and the exact
selection-stage AST with prepared tensor inputs. It covers candidate publication,
reuse, causal selection, ties and partial blocks; it is not a full official model
forward and does not emulate its quantization kernels.

## Numerical acceptance

Acceptance uses small randomly initialized pytest models covering every layer type,
Engram and shared sources; release-weight P40E and official comparisons are reports.
GPU attention tests use `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
`torch.use_deterministic_algorithms(True, warn_only=True)` and
`impl_cfg.deterministic_mode=True`. Attention CP2/CP1 outputs and input/KV/parameter
backward gradients must be bitwise equal after CP reduction; whole-model CP is
report-only. Each rank needs at least the attention window's rows; shorter packs
fail. Main attention gradients must be nonzero and finite.

The 24-step random SFT gate freezes its data, seed and calibrated learning rate
before acceptance: final-seven mean loss must be at most 0.8 times first-seven.
DCP restores model/optimizer/scheduler/RNG state and continues three same-batch
steps within loss absolute/relative tolerance 0.05/0.005. Report per-rank memory
peaks; OOM means resource-incomplete, not a numerical pass. CPU checkpoint RNG
adaptation is confined to a test fixture; GPU tests use unmodified Core RNG.

## Build for training

Run this from a distributed launcher with BF16 release weights and the release
compressed token map (substitute your paths):

```python
from megatron.lite.runtime import create_runtime
from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
from megatron.lite.runtime.contracts import ParallelConfig, RuntimeConfig

backend = MegatronLiteConfig(
    model_name="deepseek_v41", impl="lite", hf_path="/path/to/model",
    parallel=ParallelConfig(ep=4),
    impl_cfg={"token_map": compressed_token_map, "optimizer": "adam"},
)
runtime = create_runtime(RuntimeConfig(backend="mlite", backend_cfg=backend))
handle = runtime.build_model()
```
