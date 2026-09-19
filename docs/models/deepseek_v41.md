# DeepSeek-V4.1-Flash training

This main-branch HybridModel composition implements CSA2 Full/Reindex/Reuse sharing,
hierarchical index selection, the causal encoder-decoder KV source schedule,
single-pass mHC, and DeepSeekMoE. It follows the released configuration in
`examples/deepseek_v41/config_flash.json` and the [technical report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf).

CSA2 training math and dense-reference tests are adapted from [#7224](https://github.com/NVIDIA/Megatron-LM/pull/7224), by Hongxiao Bai. Shared tensors and mHC coefficients belong to one forward; no module stores another microbatch's activations. The `VE` pattern contains one attention and one expert branch per logical block. V4.1 removes V4's per-head query RMS normalization and uses compressed-position YaRN for ratio-1 layers as well.

Construct `DeepSeekV41Config.from_hf` and `DeepSeekV41Model` from the `megatron.core.models.deepseek_v41` package with explicit process groups. Enable experimental APIs with `megatron.core.config.ENABLE_EXPERIMENTAL = True`. The released model is large; use reduced dimensions for initial checks.

Supported execution is FP32/BF16 training with TP=CP=PP=1 and expert/data parallelism. Native and fused attention are available. Packed sequences, stack recomputation, CUDA graphs, weight QAT and optimized serving caches are not supported. The native indexer materializes dense score tensors and is not a million-token performance implementation. Model checkpoints preserve layer and expert ownership through the custom stack.

Run focused tests with `NVIDIA_TF32_OVERRIDE=0 uv run python -m torch.distributed.run --standalone --nproc-per-node=1 -m pytest --experimental -q tests/unit_tests/transformer/experimental_attention_variant/test_csa2.py tests/unit_tests/transformer/test_single_pass_mhc.py tests/unit_tests/models/test_deepseek_v41.py tests/unit_tests/determinism/kernels/test_deepseek_v41_kernels.py`.

This composition is text-backbone-only. Pass `engram_config=None`, `vision_config=None`, and `dspark_config=None` when importing the released HF config. Conditional components are separate extensions.
