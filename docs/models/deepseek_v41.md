# DeepSeek-V4.1-Flash

This implementation targets Megatron-LM `main` and composes the V4.1 backbone with
`HybridModel`'s vocabulary embedding, output projection, optimizer and checkpoint
interfaces. The `V` hybrid symbol identifies CSA2; each logical block contains one
`V` attention branch and one `E` expert branch.

The architecture follows the [released configuration and inference code](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/main/inference)
and [technical report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf).
The port uses [#7224](https://github.com/NVIDIA/Megatron-LM/pull/7224),
[#7231](https://github.com/NVIDIA/Megatron-LM/pull/7231), and
[#7022](https://github.com/NVIDIA/Megatron-LM/pull/7022) as references. Original
CSA2 training math/oracles are by Hongxiao Bai; distributed Engram lookup and its
parity tests are by Li Tao. Adapted DeepSeek code carries its MIT notice in
`megatron/core/models/deepseek_v41/LICENSE.deepseek`.

## Architecture

- **CSA2:** complete, non-overlapping ratio-2 groups in the encoder and ratio-1
  global KV in the decoder. Full layers own KV and indexer K; Reindex layers
  own only indexer Q; Reuse layers share the latest indices. Window and global
  positions participate in one softmax, including the attention sink. Indexer
  distillation uses the complete teacher denominator, including window/sink mass.
- **Causal encoder-decoder:** the first decoder Full layer creates global KV from
  its encoder-boundary input. Every later decoder layer reuses that graph-connected
  tensor. This training implementation evaluates all positions at all layers.
- **Single-pass mHC:** branch input uses the preceding branch's mixing coefficients.
  Mixing and coefficient prediction retain FP32 arithmetic. The final FFN mix
  contracts the output; there is no independent learned contraction head. Fused
  Sinkhorn, aggregation and residual mixing reuse the main-branch kernels. The
  coefficient projection retains the V4.1 RMS definition with epsilon inside sqrt.
- **Engram:** compressed-token hashing, prime-sized buckets, image-boundary resets,
  and context-aware per-stream gates. Tables are trainable and row-sharded over EP.
  Variable-split all-to-all supports unequal request counts and empty peer splits.
  Checkpoint descriptors preserve exact global rows across uneven shards. V4.1
  omits the older Engram convolution.
- **Vision:** full per-image attention with 2D RoPE, a 3x3 pad-and-unfold projector,
  row-major image spans and learned boundary/newline embeddings. The V4-specific
  N-layout, image-padding tokens and hash-router layers in #7022 are not V4.1 features.
  Text and image routing use separate correction biases and batch counters.
- **DSpark:** separate sliding-window draft blocks, parallel noise-token inputs,
  teacher-forced Markov correction and a confidence head. Backbone features,
  embedding and output weights are detached from the draft objective. The CE,
  distribution-L1 and confidence objectives follow
  [DeepSpec](https://github.com/deepseek-ai/DeepSpec/blob/main/deepspec/modeling/dspark/loss.py).
  `select_verification_length` accepts measured verification costs and conditional
  acceptance probabilities; it does not invent a serving-throughput model.

## Shared layers

CSA2 uses the ordinary `HybridModel` / `HybridStack` with the static
`hybrid_csa2_stack_spec` from `megatron.core.models.hybrid.hybrid_layer_specs`.
The `V` symbol selects CSA2; it can be composed with dense MLPs (`V-`), experts
(`VE`), or other supported layers. `CSA2HybridAdapter` owns the forward-local
sharing lifecycle. It does not replace the stack or own parameters.

Single-pass mHC is independently available through `TransformerConfig` with
`enable_mhc_connections=True, mhc_single_pass=True`. `HyperConnectionModule`,
`HyperConnectionHybridLayer`, and `HyperConnectionTransformerLayer` share the
same `SinglePassMHCState` interface. Ordinary attention/MLP HybridModels can use
it with `hybrid_stack_spec`, without CSA2 or a DeepSeek model class.

Schedules use **zero-based stack layer indices**, including intervening MLPs
or experts. For `V-V-V-`, an example is ratios `[2, 0, 2, 0, 2, 0]`, KV sources
`[0]`, and index sources `[0, 4]`. Non-attention positions have ratio zero.
`DeepSeekV41Config.from_hf` translates the released logical-block schedule into
this indexing. `DeepSeekV41Model` composes the conditional components around this standard stack.
Pass explicit process groups and enable experimental APIs with
`megatron.core.config.ENABLE_EXPERIMENTAL = True`. Use reduced dimensions for
initial checks; the released model is large.

The standard training parser exposes `--mhc-single-pass`, `--mhc-epsilon`,
`--dsv4-version v4.1`, and the `--csa2-*` schedule fields. Select
`--spec megatron.core.models.hybrid.hybrid_layer_specs hybrid_csa2_stack_spec`
with `pretrain_hybrid.py` and an explicit `--hybrid-layer-pattern`.


## Library use

```python
import json
import torch
from megatron.core import config as core_config
from megatron.core.models.deepseek_v41.config import DeepSeekV41Config
from megatron.core.models.deepseek_v41.model import DeepSeekV41Model

core_config.ENABLE_EXPERIMENTAL = True
hf = json.load(open("examples/deepseek_v41/config_flash.json"))
config = DeepSeekV41Config.from_hf(
    hf, params_dtype=torch.bfloat16, bf16=True,
    expert_model_parallel_size=64, dsa_indexer_loss_coeff=0.01,
)
# pg_collection comes from the application's initialized MCore process groups.
# The released model is very large: use reduced shapes for initial validation.
model = DeepSeekV41Model(
    config, hf["text_config"]["vocab_size"], 4096,
    pg_collection=pg_collection, tokenizer=tokenizer,
)
```

An application may provide an explicit compressed `token_map` instead of a
tokenizer. Its vocabulary must match `EngramConfig.compressed_vocab_size`;
substituting raw IDs changes the n-gram addresses and is not a valid way to load
trained Engram tables. The small synthetic test harness deliberately uses its own
identity-map vocabulary and newly initialized tables.

`forward(input_ids, position_ids, labels=...)` returns per-token backbone losses.
Images are lists of `ImageInput` objects, one list per batch example. Their `start`
and `types` describe slots in the input sequence. `prepare_image` converts a
Pillow image to the released patch/normalization layout without network access.

Passing `draft_anchor_positions` additionally returns `DeepSeekV41TrainingOutput`:
`backbone` contains the usual losses/logits, `draft` contains draft outputs, and
`draft_loss` is a separate scalar. An anchor names a real seed token in `input_ids`;
the following `block_size` tokens must be present for draft supervision.
Use `result.backbone.mean() + result.draft_loss` for joint training. A draft-only
stage should call `model.freeze_backbone_for_draft_training()` **before constructing
the optimizer**, then backpropagate only `result.draft_loss`. This also freezes
backbone routing-bias updates and excludes its parameters from optimizer weight decay.

## Execution coverage and limits

This is BF16/FP32 training support with TP=CP=PP=1 and expert/data parallelism.
Native and fused sparse attention share the same architecture. The native indexer
is a correctness path: it materializes dense score/candidate tensors and is not
a million-token performance implementation. GPU tests and the example report
actual exercised configurations; architecture presence is not evidence of
released-scale convergence.

Packed sequences, pipeline/virtual pipeline parallelism, context/tensor parallel
attention, stack activation recomputation and CUDA graphs are rejected. Weight
FP8/FP4 QAT is rejected. Optimized serving KV caches, bounded SWA replay, online
speculative verification, the report's custom Sinkhorn-balanced optimizer, and
HF quantized-weight import are outside this training implementation. Ordinary
MCore distributed model checkpoints are supported; the example checks every
parameter after save, zeroing, and reload.

Vision and DSpark are explicitly composed with the V4.1 stack rather than injecting
V4-specific fields or mutable per-microbatch metadata into generic transformer
layers. Other HybridModel configurations retain their existing execution paths.

## Validation

Run the focused tests through the distributed test launcher:

```bash
NVIDIA_TF32_OVERRIDE=0 uv run python -m torch.distributed.run \
  --standalone --nproc-per-node=1 -m pytest --experimental -q \
  tests/unit_tests/transformer/experimental_attention_variant/test_csa2.py \
  tests/unit_tests/transformer/test_single_pass_mhc.py \
  tests/unit_tests/models/test_deepseek_v41.py \
  tests/unit_tests/models/test_csa2_hybrid.py \
  tests/unit_tests/models/test_deepseek_v41_components.py \
  tests/unit_tests/determinism/kernels/test_deepseek_v41_kernels.py

uv run python -m torch.distributed.run --standalone --nproc-per-node=2 \
  -m pytest --experimental -q tests/unit_tests/models/test_engram_distributed_embedding.py
```

The strict FP32 attention oracle disables TF32 across both torch and TE to avoid
comparing different GEMM precision policies. BF16 paths are tested independently.
See `examples/deepseek_v41/README.md` for the training/performance harness.


The backbone and DSpark both use the standard `HybridStack`. The drafter selects
a static attention spec with explicit context-window and parallel-block inputs;
its layer loop and single-pass mHC are shared with other models. This refactor
uses standard split-layer checkpoint keys (for example,
`decoder.layers.2.inner_layer.self_attention` for logical attention block 1).
Checkpoints from the earlier bespoke-stack prototype need key conversion before
loading into this layout.
