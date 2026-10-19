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

Against official FP64 emulation, the standard model's per-token log-probability
errors at p50/p99/p99.9 must each be within 1.1 times the official BF16 model's
own error; maximum error is reported. Identical batches are required for parallel
comparisons: maximum absolute LP error 0.125 nat, p99 error 0.03 nat, mean loss
absolute/relative tolerance 0.02/0.005. All expected trainable gradients must be
present and finite. CUDA FlashMLA and CPU unfused paths need separate evidence.

The 24-step SFT test requires all steps applied and finite, with the final seven
steps' mean loss at most 0.8 times the initial seven. DCP must restore the saved
model/optimizer/scheduler/RNG/data state and continue for three same-batch steps
within loss absolute/relative tolerance 0.05/0.005. No cross-run bitwise result is
required. Checkpoint CPU RNG adaptation is confined to a test fixture; GPU tests
must use unmodified Core RNG handling.
