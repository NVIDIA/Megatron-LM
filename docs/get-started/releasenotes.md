<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Release Notes

These notes summarize notable changes in stable Megatron Core releases. For
release metadata and the generated changelog, when available, refer to the linked
GitHub release. Known issues describe the state of a release when it was published
and might be resolved in a later release.

## Megatron Core 0.18.0

[View the complete GitHub release](https://github.com/NVIDIA/Megatron-LM/releases/tag/core_v0.18.0).

### Parallelism

- **A2A overlap for Megatron-FSDP:** Bridge the Megatron-FSDP hook-based lifecycle
  with the expert-parallel overlap schedule's direct submodule execution path
  ([#3797](https://github.com/NVIDIA/Megatron-LM/pull/3797)).
- **Packed sequence support for GDN:** Add packed sequence support for gated delta
  networks, following the earlier implementation work ([#2645](https://github.com/NVIDIA/Megatron-LM/pull/2645), [#2644](https://github.com/NVIDIA/Megatron-LM/pull/2644)).
- **Permute fusion in Hybrid EP:** Fuse permute and unpermute operations with
  dispatch and combine operations into a single kernel path ([#4089](https://github.com/NVIDIA/Megatron-LM/pull/4089)).

### Mixture of Experts

- Add the `sqrtsoftplus` score function to the MoE router ([#3673](https://github.com/NVIDIA/Megatron-LM/pull/3673)).
- Add a shard-aligned parameter layout for `LayerWiseDistributedOptimizer` that
  prevents parameters from being split across shard boundaries ([#4509](https://github.com/NVIDIA/Megatron-LM/pull/4509)).
- Refactor the CUDA Graph interface by replacing the overloaded
  `cuda_graph_scope` field with three semantically distinct concepts ([#4292](https://github.com/NVIDIA/Megatron-LM/pull/4292)).

### Performance and Memory

- Consolidate batch-processing utilities and enable pipeline-parallel SFT in THD
  paths ([#4103](https://github.com/NVIDIA/Megatron-LM/pull/4103)).
- Generalize the optimizer infrastructure beyond Muon to support additional
  emerging optimizers ([#4113](https://github.com/NVIDIA/Megatron-LM/pull/4113), [#4119](https://github.com/NVIDIA/Megatron-LM/pull/4119)).
- Enable chunked MLP during training, extending the existing inference-prefill
  support ([#3656](https://github.com/NVIDIA/Megatron-LM/pull/3656)).
- Add Transformer Engine fused grouped MLP support, grouped quantized tensor
  plumbing, checkpoint compatibility, operation fuser support, and related MLP
  infrastructure ([#4636](https://github.com/NVIDIA/Megatron-LM/pull/4636)).
- Decouple oversized compute buffers from correctly sized activation buffers for
  backward-pass storage in full-model CUDA Graphs with paged stashing ([#4247](https://github.com/NVIDIA/Megatron-LM/pull/4247)).

### Precision

- Improve MXFP8 with fine-grained parameter-gather configuration, fixes for forced
  parameter all-gather during evaluation, numerical fixes when data-parallel
  overlap is disabled, and FP8 parameter-gather improvements ([#2582](https://github.com/NVIDIA/Megatron-LM/issues/2582), [#4181](https://github.com/NVIDIA/Megatron-LM/pull/4181),
  [#4562](https://github.com/NVIDIA/Megatron-LM/pull/4562), [#4800](https://github.com/NVIDIA/Megatron-LM/pull/4800)).
- Enable NVFP4 parameter gather with mixed precision in the NVFP4 recipe ([#4358](https://github.com/NVIDIA/Megatron-LM/pull/4358)).

### Inference

- Add an AllGatherV dispatcher for MoE inference and simplify the previous
  dispatcher path ([#4258](https://github.com/NVIDIA/Megatron-LM/pull/4258)).
- Port the vLLM grouped GEMM backend to the inference-optimized MoE path ([#4566](https://github.com/NVIDIA/Megatron-LM/pull/4566)).
- Enable CUDA Graphs for MTP inference ([#4260](https://github.com/NVIDIA/Megatron-LM/pull/4260)).

### Model Architecture and Support

- Add Flextron, a post-training method for converting a parent LLM into a nested
  family of submodels at different parameter budgets ([#4429](https://github.com/NVIDIA/Megatron-LM/pull/4429)).

### Multimodal

- Add the MIMO core primitive for heterogeneous tensor- and data-parallel MIMO
  training, including colocated bridge communication. This is a step toward
  early-fusion multimodal architectures with modular vision, audio, and video
  encoders, independent nD parallelism per module, and colocated or non-colocated
  training layouts ([#1375](https://github.com/NVIDIA/Megatron-LM/issues/1375), [#4368](https://github.com/NVIDIA/Megatron-LM/pull/4368)).

## Megatron Core 0.17.1

[View the complete GitHub release](https://github.com/NVIDIA/Megatron-LM/releases/tag/core_v0.17.1).

## Megatron Core 0.17.0

[View the complete GitHub release](https://github.com/NVIDIA/Megatron-LM/releases/tag/core_v0.17.0).

### Highlights

- **Absorbed MLA and fused MLA projections:** Add absorbed multi-latent attention
  and fuse MLA down-projection GEMMs to improve MLA training and inference
  throughput ([#3198](https://github.com/NVIDIA/Megatron-LM/pull/3198), [#3039](https://github.com/NVIDIA/Megatron-LM/pull/3039)).
- **Maximal Update Parameterization (muP):** Add first-class support for muP,
  enabling hyperparameter transfer from small-scale proxy models to large models,
  with MuP-over-Muon scaling ([#3058](https://github.com/NVIDIA/Megatron-LM/pull/3058), [#3715](https://github.com/NVIDIA/Megatron-LM/pull/3715)).
- **KV prefix caching:** Add a complete KV prefix-caching pipeline for inference,
  including coordinator-level scheduling, hybrid model support, and cache-aware
  load balancing ([#3063](https://github.com/NVIDIA/Megatron-LM/pull/3063), [#3225](https://github.com/NVIDIA/Megatron-LM/pull/3225), [#3665](https://github.com/NVIDIA/Megatron-LM/pull/3665), [#3930](https://github.com/NVIDIA/Megatron-LM/pull/3930)).
- **Speculative decoding with MTP:** Add speculative decoding with Multi-Token
  Prediction layers ([#3594](https://github.com/NVIDIA/Megatron-LM/pull/3594)).
- **Inference-optimized MoE:** Add an inference-optimized MoE path with PyTorch
  grouped GEMM for BF16 and MXFP8 and CUDA Graph support ([#3496](https://github.com/NVIDIA/Megatron-LM/pull/3496), [#3858](https://github.com/NVIDIA/Megatron-LM/pull/3858)).
- **MIMO heterogeneous parallelism:** Add heterogeneous parallelism for MIMO
  models, including a dedicated optimizer, distributed checkpointing, multimodule
  1F1B pipelining, and context parallelism with sequence packing ([#3211](https://github.com/NVIDIA/Megatron-LM/pull/3211), [#4019](https://github.com/NVIDIA/Megatron-LM/pull/4019),
  [#4020](https://github.com/NVIDIA/Megatron-LM/pull/4020), [#3129](https://github.com/NVIDIA/Megatron-LM/pull/3129), [#2135](https://github.com/NVIDIA/Megatron-LM/pull/2135)).
- **Lion optimizer:** Add the Lion optimizer as an alternative to Adam and AdamW
  ([#3813](https://github.com/NVIDIA/Megatron-LM/pull/3813)).

### CUDA Graph Enhancements

- Add CUDA Graph support for the Adam optimizer step ([#3429](https://github.com/NVIDIA/Megatron-LM/pull/3429)).
- Add an expert-parallel overlap dynamic computation stream for full-iteration
  CUDA Graphs ([#3820](https://github.com/NVIDIA/Megatron-LM/pull/3820)).
- Add Transformer Engine CUDA Graph support for the vision encoder ([#3293](https://github.com/NVIDIA/Megatron-LM/pull/3293)).
- Add full-iteration CUDA Graphs for the Mamba inference block ([#3250](https://github.com/NVIDIA/Megatron-LM/pull/3250)).
- Improve fine-grained CUDA Graph coverage for smaller batch sizes ([#3527](https://github.com/NVIDIA/Megatron-LM/pull/3527)).
- Add CUDA Graph support for prefix caching on hybrid models ([#3922](https://github.com/NVIDIA/Megatron-LM/pull/3922)).

### Fused Kernels and Performance

- Fuse dLN and add operations in the backward pass ([#3384](https://github.com/NVIDIA/Megatron-LM/pull/3384)).
- Fuse permute with pad and unpermute with unpad for FP8 and FP4 training ([#2763](https://github.com/NVIDIA/Megatron-LM/pull/2763)).
- Add MXFP8 quantization for inference linear layers ([#3447](https://github.com/NVIDIA/Megatron-LM/pull/3447)).
- Add FP32 local gradient accumulation for selected model parameters ([#4028](https://github.com/NVIDIA/Megatron-LM/pull/4028)).

### Megatron-FSDP

- Add data type customization ([#3067](https://github.com/NVIDIA/Megatron-LM/pull/3067)).
- Add an MXFP8 transpose helper buffer for Hybrid FSDP ([#3918](https://github.com/NVIDIA/Megatron-LM/pull/3918)).
- Add expert parallelism with HSDP ([#2840](https://github.com/NVIDIA/Megatron-LM/pull/2840)).
- Add Qwen3-VL support ([#2841](https://github.com/NVIDIA/Megatron-LM/pull/2841)).
- Add FSDP DSv3 proxy support ([#3844](https://github.com/NVIDIA/Megatron-LM/pull/3844)).

### Reinforcement Learning

- Add CUDA Graphs for hybrid MoE training and fix the transition between training
  and inference ([#3373](https://github.com/NVIDIA/Megatron-LM/pull/3373)).
- Add forced lag and off-policy tracking to improve training stability ([#3517](https://github.com/NVIDIA/Megatron-LM/pull/3517),
  [#3030](https://github.com/NVIDIA/Megatron-LM/pull/3030)).
- Add inference-only mode with `--skip-train` ([#3744](https://github.com/NVIDIA/Megatron-LM/pull/3744)).

### Model Support

- Add gated delta networks for Mamba hybrid models ([#3535](https://github.com/NVIDIA/Megatron-LM/pull/3535)).
- Add flexible virtual pipeline parallelism for hybrid models ([#3377](https://github.com/NVIDIA/Megatron-LM/pull/3377)).
- Add MTP support for hybrid models ([#3207](https://github.com/NVIDIA/Megatron-LM/pull/3207)).
- Add mRoPE for MTP ([#3114](https://github.com/NVIDIA/Megatron-LM/pull/3114)).
- Add a GPT-OSS example with Megatron Bridge ([#3018](https://github.com/NVIDIA/Megatron-LM/pull/3018)).
- Add ModelOpt EAGLE training with context-parallel support ([#3147](https://github.com/NVIDIA/Megatron-LM/pull/3147)).
- Add an NVIDIA-Nemotron-3-Super-120B-A12B-BF16 ModelOpt example ([#3805](https://github.com/NVIDIA/Megatron-LM/pull/3805)).

### Checkpointing

- Add zero-copy storage sharing in `CheckpointWithoutOutput` ([#3649](https://github.com/NVIDIA/Megatron-LM/pull/3649)).
- Use a single process for checkpoint saves to avoid forked multiprocessing
  issues ([#3424](https://github.com/NVIDIA/Megatron-LM/pull/3424)).
- Add `CachedMetadataFileSystemReader` with a shared cache ([#3326](https://github.com/NVIDIA/Megatron-LM/pull/3326)).
- Optimize asynchronous save-process management ([#3262](https://github.com/NVIDIA/Megatron-LM/pull/3262)).
- Support quantized CUDA tensors in the asynchronous checkpoint writer ([#3845](https://github.com/NVIDIA/Megatron-LM/pull/3845)).

### Ease of Use

- Add NCCL flight recorder configuration support ([#3806](https://github.com/NVIDIA/Megatron-LM/pull/3806)).
- Add `DistributedInitConfig` for cleaner distributed initialization ([#3173](https://github.com/NVIDIA/Megatron-LM/pull/3173)).
- Replace `ModuleSpec` with protocols across MLP, LayerNorm, MoE, and model
  submodules ([#3084](https://github.com/NVIDIA/Megatron-LM/pull/3084), [#3090](https://github.com/NVIDIA/Megatron-LM/pull/3090), [#3426](https://github.com/NVIDIA/Megatron-LM/pull/3426)).
- Improve the PyTorch profiler and add execution-trace support ([#3273](https://github.com/NVIDIA/Megatron-LM/pull/3273)).
- Add Common Pile data-preparation scripts ([#3902](https://github.com/NVIDIA/Megatron-LM/pull/3902)).
- Add `--overlap-param-gather` support for the layer-wise optimizer ([#3524](https://github.com/NVIDIA/Megatron-LM/pull/3524)).

### Breaking Changes

- The minimum supported Python version is now Python 3.12 ([#3826](https://github.com/NVIDIA/Megatron-LM/pull/3826)).
- Legacy and deprecated APIs were removed, including the legacy data module,
  legacy MPU, `GroupedMLP`, the deprecated Transformer Engine module, deprecated
  Mamba parameters, `async_grad_allreduce`, `get_te_version`, and
  `SampleListWebdataset` ([#3853](https://github.com/NVIDIA/Megatron-LM/pull/3853), [#3854](https://github.com/NVIDIA/Megatron-LM/pull/3854), [#3410](https://github.com/NVIDIA/Megatron-LM/pull/3410), [#3409](https://github.com/NVIDIA/Megatron-LM/pull/3409), [#3411](https://github.com/NVIDIA/Megatron-LM/pull/3411), [#3412](https://github.com/NVIDIA/Megatron-LM/pull/3412), [#3413](https://github.com/NVIDIA/Megatron-LM/pull/3413),
  [#3407](https://github.com/NVIDIA/Megatron-LM/pull/3407)).
- `encoder_and_decoder` was removed from model enums and related code ([#3836](https://github.com/NVIDIA/Megatron-LM/pull/3836)).
- The `nv-grouped-gemm` dependency was replaced by PyTorch grouped GEMM ([#3770](https://github.com/NVIDIA/Megatron-LM/pull/3770)).
- The legacy tokenizer system was removed ([#2946](https://github.com/NVIDIA/Megatron-LM/pull/2946)).

## Megatron Core 0.16.1

[View the complete GitHub release](https://github.com/NVIDIA/Megatron-LM/releases/tag/core_v0.16.1).

## Megatron Core 0.16.0

[View the complete GitHub release](https://github.com/NVIDIA/Megatron-LM/releases/tag/core_v0.16.0).

- Introduce `AdamOptimizerConfig` and `SGDOptimizerConfig`.
- Support different hyperparameters for different parameter groups.
- Add optimizer-state offloading to move optimizer states and master weights to
  the CPU, reducing GPU memory use.
- Support MTP layers in standalone pipeline stages for improved virtual pipeline
  parallel balancing.
- Improve DeepSeek V3.2 performance.

### CUDA Graph Enhancements

- Refactor `cuda_graph_scope`.
- Add partial CUDA Graph support for expert-parallel overlap to reduce CPU
  pressure within selected scopes.
- Reuse static input-memory buffers between microbatches with Transformer Engine
  CUDA Graphs.
- Add CUDA Graph compatibility with tensorwise and blockwise FP8 parameters.
- Add NVFP4 MoE CUDA Graph support with 128-token zero padding.

### Additional Features

- Add router replay for deterministic routing during debugging and reinforcement
  learning.
- Add a fake distributed process group that skips distributed communication with
  `--fake-process-group` for profiling.
- Exclude padding tokens from MoE routing loss calculations.
- Add context-parallel support for the eager attention implementation.
- Add packed sequence support to the MTP module.
- Add fine-grained activation offloading.
- Improve Hybrid EP with kernel optimizations for EP64 and NVL8+IB.

For details about breaking changes, refer to
[PR #2047](https://github.com/NVIDIA/Megatron-LM/pull/2047).

## Megatron Core 0.15.3

[View the complete GitHub release](https://github.com/NVIDIA/Megatron-LM/releases/tag/core_v0.15.3).

This release addresses known security issues. For the latest NVIDIA vulnerability
disclosures, refer to the [NVIDIA Product Security page](https://www.nvidia.com/en-us/security/).

## Megatron Core 0.15.2

[View the complete GitHub release](https://github.com/NVIDIA/Megatron-LM/releases/tag/core_v0.15.2).

### Bug Fixes

- Fix several Megatron-FSDP issues
  ([PR #2346](https://github.com/NVIDIA/Megatron-LM/pull/2346)).
- Support both old and new `DeviceMesh` APIs
  ([PR #2575](https://github.com/NVIDIA/Megatron-LM/pull/2575)).
- Build a default FSDP `DeviceMesh` and remove the model argument from
  `fully_shard_optimizer()`
  ([PR #2471](https://github.com/NVIDIA/Megatron-LM/pull/2471)).

## Megatron Core 0.15.1

[View the complete GitHub release](https://github.com/NVIDIA/Megatron-LM/releases/tag/core_v0.15.1).

This release was yanked.

## Megatron Core 0.15.0

[View the complete GitHub release](https://github.com/NVIDIA/Megatron-LM/releases/tag/core_v0.15.0).

### Performance

- Add fused QKV preprocessing with precomputed RoPE caches, improving
  preprocessing by 3x and end-to-end performance by 10-14%.
- Use the new Transformer Engine interface for user buffers.
- Add CPU activation offloading through Transformer Engine.
- Add configurable double buffering.
- Add the Muon optimizer and distributed optimizer support.
- Add a setting to select Adam or AdamW.

### Mixture of Experts

- Add DTensor support for expert parallelism and DSv3 modules.
- Add the HybridEP backend to Flex Dispatcher.
- Support FP8 recomputation for MoE components.
- Add NVFP4 zero padding for MoE.
- Compute shared experts before the router.
- Enable bias in the expert MLP.

### Model Support

- Add YaRN support for GPT-OSS.
- Add Qwen3-Next arguments.
- Add FP8 initialization for MTP.
- Add the `fp8_dpa` option for FP8 scaling.
- Add RADIO-g support to the converter and tester.
- Add audio semantic-reasoning data for voice chat and speech instructions.

### Megatron-FSDP

- Enable joint training of parallel modules.
- Add multimodule communication.

### Inference

- Add a CUDA Graph runner lookup-table cache, improving end-to-end performance by
  up to 2x.
- Add an MoE dropping and padding router for CUDA Graphs and decode.
- Support dynamic audio shapes with variable sequence lengths, improving
  throughput by 2.5x.
- Integrate unified memory for dynamic inference context.

### Post-Training

- Add GPT-OSS ModelOpt support with quantization and import/export.
- Add knowledge-distillation support to the hybrid training loop.
- Add a ModelOpt pruning example.

### Reinforcement Learning

- Add importance sampling and partial rollouts to Megatron RL.
- Add sequence packing for reinforcement learning.

### Ease of Use

- Handle CUDA absence during import.
- Add Granary data-loader functionality.
- Enable sliding-window attention mixing with attention.

### Bug Fixes

- Fix MXFP8 parameter gradient-buffer reuse convergence.
- Clone the loss mask to prevent incorrect updates.
- Preserve checkpoint metadata.
- Fix FSDP gradient-accumulation fusion.
- Fix non-Transformer Engine optimizer checkpoints.
- Fix BERT virtual pipeline parallelism.
- Add `gc.collect()` on the last layer to avoid a `gc.freeze()` slowdown.
- Fix full-iteration CUDA Graph handling of non-tensor values.
- Fix `model_auto_sync` configuration and add a gradient assertion.
- Fix Hugging Face import data types and checkpoint loading.
- Fix missing `ProcessGroupCollection` initialization.
- Fix tensor parallelism with sink attention.
- Fix `num_microbatches` calculation.
- Fix 1F1B overlap unit tests for standalone MTP.
- Fix stale state-dictionary handling.
- Fix dataset divergence involving tokenizer padding.
- Fix parameter initialization.
- Set tensor-parallel attributes regardless of the initialization flag.

## Megatron Core 0.14.0

[View the complete GitHub release](https://github.com/NVIDIA/Megatron-LM/releases/tag/core_v0.14.0).

### Inference

- Add asynchronous support to `DynamicInferenceEngine`.
- Pad input tensors and enable FP8 weights for FP8 inference.
- Always gather inference logits when using tensor parallelism.
- Support multiple batch sizes with dynamic-inference CUDA Graphs.

### Post-Training

- Update ModelOpt integration.
- Add speculative-decoding autoregressive validation.
- Add DeepSeek and Qwen model configurations.

### Performance

- Integrate `ModelCommProcessGroup`.
- Add HyperCommGrid, an n-dimensional communication grid for flexible process
  group creation and management.
- Add Spike No More embedding initializations and weight-decay skipping.

### Mixture of Experts

This release continued large-scale fine-grained MoE optimization for the Blackwell
platform.

#### Features

- Add expert-parallel all-to-all overlap.
- Add context parallelism and recomputation for MTP.
- Add global auxiliary loss.

#### Memory Optimization

- Add recomputation for FP8 layer normalization, MoE activation, and shared
  experts.
- Add optimizer offloading for DSv3 FP8 training.

#### Performance Optimization

- Add MoE router fusion.
- Update MoE CUDA Graph support.

#### Bug Fixes

- Fix the router-input jitter data type.

### Model Support

- Add a MiMo video VLM training example.
- Add AVLM for MIMO.

### Ease of Use

- Add `uv` support for source installations.
- Add automated weekly prereleases.

### Bug Fixes

- Use `mscale_all_dim` for `softmax_factor`.
- Fix FP8 parameter blockwise-scaling unit tests.
- Optimize prefill for tokenless requests.
- Add default values for `Fp8Padding` and `Fp8Unpadding`.
- Fix CUDA Graph logic for flexible pipeline parallel layouts.
- Load FP8 models with `strict=False`.
- Skip the fused RoPE check for Transformer Engine versions earlier than 1.4.0.
- Disable unstable Apex tests.
- Fix a typo in expert parallelism state.
- Guard ModelOpt on macOS.
- Add retries for CUDA function failures.
- Fix NCCL memory-pool creation.
- Fix the `get_rotary_seq_len` return type.
- Fix an NCCL allocator attribute error.
- Fix multi-prompt inference.
- Fix MD5 usage on FIPS systems.
- Fix dynamic-context and inference issues.
- Fix the Transformer Engine version check for interleaved fused RoPE.
- Fix MTP with MoE and tensor-parallel logging.
- Guard Transformer Engine imports.
- Add an assertion for NCCL user buffers.
- Remove encoder pipeline-parallel functions.
- Fix test segmentation faults.
- Fix a Transformer Engine error in the distributed optimizer.
- Remove a redundant checkpoint-flow barrier.
- Support virtual pipeline parallelism with MTP and fix logging.
- Add retries for `free(): invalid pointer` errors.
- Fix `test_replication.py`.
- Fix a typo in parallel state.
- Fix CUDA Graph selection logic.
- Fix Transformer Engine installation.
- Use the correct sharding type in local tests.
- Fix backward-buffer reuse on the last CUDA-graphed layer.
- Set a default for `packed_seq_params` in `get_rotary_seq_len`.
- Fix dynamic example scripts.

### Breaking Changes

- `megatron.core.distributed.custom_fsdp` was refactored as
  `megatron.core.distributed.fsdp.src.megatron_fsdp`.

## Megatron Core 0.13.1

[View the complete GitHub release](https://github.com/NVIDIA/Megatron-LM/releases/tag/core_v0.13.1).

## Megatron Core 0.13.0

[View the complete GitHub release](https://github.com/NVIDIA/Megatron-LM/releases/tag/core_v0.13.0).

- Support BF16 optimizer states with the Transformer Engine precision-aware
  optimizer.

### Mixture of Experts

#### Features

- Add flexible asymmetric virtual pipeline parallelism with custom pipeline
  layouts (`--pipeline-model-parallel-layout`).
- Allow custom parallelism groups to be passed to MoE modules.
- Add hybrid-shard data parallelism for MoE models
  (`--num-distributed-optimizer-instances`).
- Support expert parallelism with custom FSDP training for DeepSeek-V3.
- Add FP8 support for Multi-Token Prediction.

#### Memory Optimization

- Add fine-grained recomputation with `--recompute-modules` and
  `--recompute-granularity selective`.
- Reduce token-permutation memory by moving probability multiplication from
  unpermutation to the `GroupedMLP` activation function.

#### Performance Optimization

- Add an MLA RoPE fusion kernel and YaRN embedding cache.
- Optimize FP8 padding by padding the MoE routing map.

#### Bug Fixes

- Fix auxiliary-loss calculation when expert bias or group-limited routing is
  used. This changes `load_balancing_loss` values from the previous release.
- Fix packed sequence support for MLA.

#### Known Issues

- MTP is not compatible with flexible pipeline layouts.
- MTP has a convergence issue with TP2.

These issues were fixed in Megatron Core 0.14.0.

## Megatron Core 0.12.3

[View the complete GitHub release](https://github.com/NVIDIA/Megatron-LM/releases/tag/core_v0.12.3).

## Megatron Core 0.12.2

[View the complete GitHub release](https://github.com/NVIDIA/Megatron-LM/releases/tag/core_v0.12.2).

## Megatron Core 0.12.1

[View the complete GitHub release](https://github.com/NVIDIA/Megatron-LM/releases/tag/core_v0.12.1).

## Megatron Core 0.12.0

[View the complete GitHub release](https://github.com/NVIDIA/Megatron-LM/releases/tag/core_v0.12.0).

- Add FP8 recipe selection with `--fp8-recipe`, `--first-last-layers-bf16`,
  `--num-layers-at-start-in-bf16`, and `--num-layers-at-end-in-bf16`.
- Fix loss scaling for context parallelism when
  `calculate_per_token_loss=True`.
- Make the number of data-parallel communication buckets configurable with
  `--ddp-num-buckets` and `--ddp-pad-buckets-for-high-nccl-busbw`.

### Inference

- Support in-flight batching and a chunked KV cache.
- Reduce memory use by avoiding a materialized full attention mask, materializing
  logits only for the last token during decode, and removing an obsolete tensor
  reference.

### Hybrid Models

- Add CUDA Graph support for inference.
- Update `tools/run_mamba_text_generation_server.py` to use
  `megatron.core.inference`.
- Fix a logits-shape issue for Mamba models.
- Improve Mamba-layer initialization.
- Add `--mamba-state-dim`, `--mamba-head-dim`, `--mamba-num-groups`, and
  `--is-hybrid-model`.
- Support hybrid models in `num_floating_point_operations`.
- Support Transformer Engine linear layers in `hybrid_conversion.py`.
- Add FP8 support.
- Fix Mamba `dt_bias` tensor parallelism.
- Support multimodal tokenizers.
- Improve data-parallel scaling.

### Mixture of Experts

#### Features

- Add DeepEP support compatible with other parallelism strategies and token-drop
  or dropless operation.
- Add FP32 and FP64 routing and unpermutation with `--moe-router-dtype`. FP32 is
  recommended for fine-grained MoE training.
- Add CUDA Graph support for MoE.
- Add Multi-Token Prediction support.
- Add a fused `indices_to_multihot` kernel for the DeepEP dispatcher.

#### Bug Fixes

- Fix a hang in mixed MoE and dense models.
- Update theoretical memory and TFLOPS estimation for MoE and MLA.
- Fix MoE auxiliary-loss scaling with per-token loss.
- Fix group-limited routing and expert bias, verified with DeepSeek-V3 end-to-end
  tests.

#### Known Issues

- Checkpoints trained with custom FSDP for MoE might not be compatible with 3D
  parallel training.

## Megatron Core 0.11.0

[View the complete GitHub release](https://github.com/NVIDIA/Megatron-LM/releases/tag/v0.11.0).

- Add multidatacenter training.
- Allow `net_name` to be set in YAML so NCCL can use north-south connections
  between data centers.
- Support more data-parallel last-rank mappings.
- Add MXFP16 optimizer states and master weights.
- Optimize CUDA Graph memory use.
- Add the UCC backend for pipeline-parallel communication.
- Add CPU offloading for optimizer memory savings.
- Reduce inference memory by avoiding a materialized full attention mask.

### Models

- Add the initial RADIO and CRADIO implementation.
- Add Llama 3.2 support.

### Mixture of Experts

#### Features

- Add DeepSeek-V3 fine-tuning.
- Add auxiliary-loss-free load balancing.
- Add node-limited and device-limited routing.
- Add tensor parallelism for MLA and sequence auxiliary loss.
- MTP with tensor and pipeline parallelism was planned for a later release.
- Add a permutation and unpermutation fusion kernel from Transformer Engine.
- Support uneven virtual pipeline-parallel splits in the first and last pipeline
  stages.

#### Bug Fixes

- Fix gradient scaling when tensor parallelism differs from expert tensor
  parallelism and `average_in_collective` is enabled in DDP.
- Fix Transformer Engine `GroupedMLP` distributed-checkpoint compatibility with
  FP8 padding and unpadding.

#### Known Issues

- Dense and MoE hybrid model training can hang when a pipeline-parallel rank has
  no expert parameters.

### Hybrid Models

- Add quantization through TensorRT Model Optimizer.

## Megatron Core 0.10.0

[View the complete GitHub release](https://github.com/NVIDIA/Megatron-LM/releases/tag/core_r0.10.0).

- Add multi-latent attention to Megatron Core.
- Enable FP8 for `GroupedMLP`.
- Add MoE parallel folding.
- Support MoE layer-frequency patterns and configurable MoE FFN hidden size.
- Add NVLM multimodal training and evaluation.
- Add Mamba hybrid models.
- Improve the performance and memory use of the Triton language and compiler
  distributed cache.
- Expand unit-test coverage and fix bugs.

## Megatron Core 0.9.0

[View the complete GitHub release](https://github.com/NVIDIA/Megatron-LM/releases/tag/core_r0.9.0).

### Parallelism

- Add uneven pipeline parallelism so the first and last ranks can have fewer
  transformer layers than intermediate ranks.
- Add per-layer CUDA Graph support for GPT training with Transformer Engine
  modules.
- Support different tensor-parallel sizes for the vision encoder.
- Add pipeline parallelism for T5 and LLaVA models.
- Support multi-tile, multi-image inputs in LLaVA models.

### Mixture of Experts

- Add FP8 support.
- Add runtime upcycling.
- Optimize dispatcher implementations.
- Add shared experts with overlap optimizations.
- Add Qwen model support.

### Mamba Hybrid

- Add distributed checkpointing.
- Fix inference issues.
- Add unit tests.

The `main` branch at the time of this release was not compatible with released
checkpoints; use the `ssm` branch for those checkpoints.

### Fault Tolerance

- Add fault and hang detection in addition to straggler detection.
- Add graceful exit and automatic restart.

### Known Issues

- With sequence parallelism enabled, dropout did not use the correct random-number
  generator context during the transformer-block forward pass.

## Megatron Core 0.8.0

[View the complete GitHub release](https://github.com/NVIDIA/Megatron-LM/releases/tag/core_r0.8.0).

### Multimodal

- Add initial support for training vision-language models with the LLaVA
  architecture.
- Add initial multimodal inference support.
- Add an end-to-end multimodal example covering data collection, training, and
  evaluation in `examples/multimodal`.
- Support different tensor-parallel sizes for the encoder and decoder.

### Mixture of Experts

- Add context-parallel support.
- Add distributed-checkpoint support for grouped GEMM.

### Mamba

- Add initial Mamba-2 training and inference support.
- Support hybrid models with Mamba-2, attention, and MLP layers.
- Add examples in `examples/mamba`.

## Megatron Core 0.7.0

[View the complete GitHub release](https://github.com/NVIDIA/Megatron-LM/releases/tag/core_v0.7.0).

### Mixture of Experts

- Add token dropping.
- Improve efficiency, model parallelism, and memory use.

### Distributed Checkpointing

- Enable distributed checkpointing for RETRO.
- Add asynchronous checkpoint saving.

- Add minor bug fixes, speed improvements, and memory optimizations.

## Megatron Core 0.6.0

[View the complete GitHub release](https://github.com/NVIDIA/Megatron-LM/releases/tag/core_v0.6.0).

### Mixture of Experts

- Optimize performance and communication for single- and multi-GPU execution.
- Improve Mixtral performance on Hopper BF16 by 23% over Megatron Core 0.5.0,
  reaching 323 TFLOPS per GPU.
- Enhance `GroupedMLP` for Hopper.
- Add data-parallel overlap between computation, gradient reduction, and parameter
  gathering.
- Add an all-to-all token dispatcher.
- Add layer-wise load-balancing-loss logging.
- Improve expert parallelism, including distributed optimizer support.

### Additional Features

- Add a distributed optimizer.
- Add RETRO data processing and BERT support.
- Add distributed checkpointing with a PyTorch-native backend and improved save
  and load speed.
- Add TensorRT-LLM export with TensorRT Model Optimizer post-training
  quantization, a text-generation driver for quantization, and examples that use
  the unified TensorRT-LLM build API with Llama 2 and Nemotron-3 8B.
- Add minor enhancements, bug fixes, documentation updates, and performance and
  memory improvements.

## Megatron Core 0.5.0

[View the complete GitHub release](https://github.com/NVIDIA/Megatron-LM/releases/tag/core_v0.5.0).

## Megatron Core 0.4.0

[View the complete GitHub release](https://github.com/NVIDIA/Megatron-LM/releases/tag/core_v0.4.0).

## Roadmaps

Stay up-to-date with the development roadmaps and planned features:

- **[MoE Q3-Q4 2025 Roadmap](https://github.com/NVIDIA/Megatron-LM/issues/1729)** - Comprehensive MoE feature development including DeepSeek-V3, Qwen3, advanced parallelism, FP8 optimizations, and Blackwell enhancements.
- **[GPT-OSS Implementation Tracker](https://github.com/NVIDIA/Megatron-LM/issues/1739)** - Advanced features including YaRN RoPE scaling, attention sinks, and custom activation functions.

