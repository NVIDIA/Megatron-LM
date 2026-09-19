# DeepSeek-V4.1-Flash training

This main-branch HybridModel composition implements CSA2 Full/Reindex/Reuse sharing,
hierarchical index selection, the causal encoder-decoder KV source schedule,
single-pass mHC, and DeepSeekMoE. It follows the released configuration in
`examples/deepseek_v41/config_flash.json` and the [technical report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf).

CSA2 training math and dense-reference tests are adapted from [#7224](https://github.com/NVIDIA/Megatron-LM/pull/7224), by Hongxiao Bai. Shared tensors and mHC coefficients belong to one forward; no module stores another microbatch's activations. The `VE` pattern contains one attention and one expert branch per logical block. V4.1 removes V4's per-head query RMS normalization and uses compressed-position YaRN for ratio-1 layers as well.

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
this indexing. `DeepSeekV41Model` is a thin recipe around the standard stack.
Pass explicit process groups and enable experimental APIs with
`megatron.core.config.ENABLE_EXPERIMENTAL = True`. Use reduced dimensions for
initial checks; the released model is large.

The standard training parser exposes `--mhc-single-pass`, `--mhc-epsilon`,
`--dsv4-version v4.1`, and the `--csa2-*` schedule fields. Select
`--spec megatron.core.models.hybrid.hybrid_layer_specs hybrid_csa2_stack_spec`
with `pretrain_hybrid.py` and an explicit `--hybrid-layer-pattern`.

Supported execution is FP32/BF16 training with TP=CP=PP=1 and expert/data parallelism. Native and fused attention are available. Packed sequences, stack recomputation, CUDA graphs, weight QAT and optimized serving caches are not supported. The native indexer materializes dense score tensors and is not a million-token performance implementation. Model checkpoints preserve layer and expert ownership through the standard HybridStack.

Run focused tests with `NVIDIA_TF32_OVERRIDE=0 uv run python -m torch.distributed.run --standalone --nproc-per-node=1 -m pytest --experimental -q tests/unit_tests/transformer/experimental_attention_variant/test_csa2.py tests/unit_tests/transformer/test_single_pass_mhc.py tests/unit_tests/models/test_deepseek_v41.py tests/unit_tests/determinism/kernels/test_deepseek_v41_kernels.py`.

This composition is text-backbone-only. Pass `engram_config=None`, `vision_config=None`, and `dspark_config=None` when importing the released HF config. Conditional components are separate extensions.
