# Changelog

## NVIDIA Megatron Core 0.14.0

* Features  
  * Inference  
    * Add async support for DynamicInferenceEngine ([MR \!3187](https://github.com/NVIDIA/Megatron-LM/commit/05079d55a5bfcc7a43f4619e36a40a9e8db3f882))  
    * Pad input tensors and enable FP8 weights for FP8 inference ([MR \!3341](https://github.com/NVIDIA/Megatron-LM/commit/6a6cd478839d90cf09a837adf8c79cbc844bc920))  
    * Force inference to always gather logits with tensor parallelism ([MR \!3442](https://github.com/NVIDIA/Megatron-LM/commit/7c9cdcb794089968278c7272e0261a68edf5d369))  
    * Multi batch size CUDA Graphs for Dynamic Inference ([MR \!3402](https://github.com/NVIDIA/Megatron-LM/commit/30aabe5e3133c6d70aa55aaabad4ea8cb04ce63c))  
  * Post-training  
    * ModelOpt updates ([MR \!3268](https://github.com/NVIDIA/Megatron-LM/commit/550ed5243c3a18e39430c15e8918ee63e41d7eaf))  
      * Add speculative decoding AR validation feature  
      * Add DeepSeek and Qwen model configs  
  * Performance  
    * ModelCommProcessGroup integration ([MR \!3391](https://github.com/NVIDIA/Megatron-LM/commit/26adc2dfde53fbc2b063e2fdd1d9ed26578811a6))  
    * Add HyperCommGrid: N-Dimensional Communication Grid for Model Parallelism ([MR \!3398](https://github.com/NVIDIA/Megatron-LM/commit/45400df7da7fa23e3aff86804e5ac254d9a8d3c0))  
      * Flexible creation and management of communication groups  
    * Add support for Spike No More embedding initializations and weight decay skipping ([MR \!3500](https://github.com/NVIDIA/Megatron-LM/commit/ee74aa66a06b24e511270f285db475941ef63bfd))  
  * Model support  
    * Add MiMo video VLM train example (\[MR \!3543)  
    * Add AVLM for MIMO (\[MR \!3624)  
  * Ease of use  
    * Add uv support for source installs ([MR \!3615](https://github.com/NVIDIA/Megatron-LM/commit/164204cd7216e642bdef7299c569d95f02f9a79e))  
    * Automated weekly prereleases ([MR \!3574](https://github.com/NVIDIA/Megatron-LM/commit/7e59266c70ef34a246438640af690b55c7ecac28))  
* Bug fixes  
  * Use mscale\_all\_dim for softmax\_factor ([MR \!2800](https://github.com/NVIDIA/Megatron-LM/commit/e96a358f60c82b8ac8d965d91c3cc4ad0230a4e0))  
  * Fix FP8 param blockwise scaling unit test ([MR \!3480](https://github.com/NVIDIA/Megatron-LM/commit/57082f946a04c3390fcfc43634dc546ec3ded033))  
  * Fix unit test blockwise scaling ([MR \!3491](https://github.com/NVIDIA/Megatron-LM/commit/6d95fe63658f967e56a3fda88a9c30a424fcb520))  
  * Optimize prefill for token-less requests ([MR \!3499](https://github.com/NVIDIA/Megatron-LM/commit/daaa650a9ac4291d4027ca2fdeb4298ce024efd2))  
  * Add default values for Fp8Padding and Fp8Unpadding ([MR \!3501](https://github.com/NVIDIA/Megatron-LM/commit/42b2b1d10a9cb699b7e5aa40f6bfba9c2a1348aa))  
  * Fix CUDA graph logic for flexible pp layout ([MR \!3505](https://github.com/NVIDIA/Megatron-LM/commit/020d85e50ddf0f0282802002acb3662129a519c5))  
  * Load FP8 models with strict=False ([MR \!3508](https://github.com/NVIDIA/Megatron-LM/commit/1ab876ddc4c1893c76f26d775226a8d1dcdfb3d2))  
  * Skip rope check for torch \< 1.4.0 ([MR \!3528](https://github.com/NVIDIA/Megatron-LM/commit/d8180ef8ed0bb6f305dcdedf1b27d91304f361a3))  
  * Disable Apex tests for stability ([MR \!3539](https://github.com/NVIDIA/Megatron-LM/commit/d1256277fe378add0a2cfd7251f5a350b6d126ec))  
  * Fix typo in parallel\_state expert parallelism ([MR \!3548](https://github.com/NVIDIA/Megatron-LM/commit/5783ff32af759b8102cf0cb0bb82b30c48b9da26))  
  * Guard modelopt on macOS ([MR \!3549](https://github.com/NVIDIA/Megatron-LM/commit/76144fe1106e4fb0e69aa75b7a6ab66e71e8f37f))  
  * Retry on CUDA function failure ([MR \!3554](https://github.com/NVIDIA/Megatron-LM/commit/809aab68307a64c1386d68cc78ef70f8f4e12a80))  
  * Fix NCCL mem pool creation error ([MR \!3557](https://github.com/NVIDIA/Megatron-LM/commit/b61e21153146a563309b5d44cb5d7f7425806072))  
  * Fix get\_rotary\_seq\_len return type ([MR \!3559](https://github.com/NVIDIA/Megatron-LM/commit/1fa6bc83c7aeae95abc8e86ff0aac596985a01c3))  
  * Retry on CUDA function failure ([MR \!3560](https://github.com/NVIDIA/Megatron-LM/commit/7da88d74865c3f1a59894173246f26e7b3bf91b9))  
  * Fix NCCL allocator attribute error ([MR \!3565](https://github.com/NVIDIA/Megatron-LM/commit/6b656114795d74c3353cb007c59af49b1752f447))  
  * Ensure multi-prompt inference works ([MR \!3568](https://github.com/NVIDIA/Megatron-LM/commit/0fae48931000c9c7af06f7dcf037b5b7d96e0cd6))  
  * Fix MD5 on FIPS systems ([MR \!3577](https://github.com/NVIDIA/Megatron-LM/commit/83ee8c2848a3b1d42b40086a64da11e19f4b191f))  
  * Fixes dynamic context and inference bugs ([MR \!3582](https://github.com/NVIDIA/Megatron-LM/commit/e9c1da60a1ccc85376666d58568ed1d3e5a4f9db))  
  * Fix TE version for interleaved fused RoPE ([MR \!3586](https://github.com/NVIDIA/Megatron-LM/commit/b72b6cc161f5273b545bca09677382917cf20492))  
  * Fix MTP with MoE and TP logging ([MR \!3594](https://github.com/NVIDIA/Megatron-LM/commit/9af96623b66693e058f6bfce8d0094dc976792d8))  
  * Guard TE import fix ([MR \!3596](https://github.com/NVIDIA/Megatron-LM/commit/1bf946b1ec3f11e71459c7c0d06a97edbed96a1a))  
  * Add assertion for NCCL UB case ([MR \!3599](https://github.com/NVIDIA/Megatron-LM/commit/e11d28592f19c122859be764b7afe7c208d9acc1))  
  * Remove Encoder PP related Functions ([MR \!3604](https://github.com/NVIDIA/Megatron-LM/commit/9e49aa4446a58cc21c4dc0c5d0806551ad075ca7))  
  * Fix segfaults in tests ([MR \!3605](https://github.com/NVIDIA/Megatron-LM/commit/f6492fe8164fd5b9ad55007d435ccfc66cb98cc7))  
  * Fix TE error in distributed optimizer ([MR \!3625](https://github.com/NVIDIA/Megatron-LM/commit/e6c510ff3c1159f8955589b26f7c395bdf0607d9))  
  * Remove redundant barrier in checkpoint flow ([MR \!3626](https://github.com/NVIDIA/Megatron-LM/commit/26869feb6a3ac7f5616cb7253c37a4244d107d70))  
  * Support VPP MTP, fix logging ([MR \!3630](https://github.com/NVIDIA/Megatron-LM/commit/c351a473c7eedac2c43eab0815afb9759f4f8187))  
  * Retry mechanism for free(): invalid pointer errors ([MR \!3632](https://github.com/NVIDIA/Megatron-LM/commit/ec35b41b2df145a7ccb84afc48d94e0786e094da))  
  * Fix test\_replication.py issues ([MR \!3633](https://github.com/NVIDIA/Megatron-LM/commit/f7b50b271b2e0e396069e02551b21aa6fb374b43))  
  * Fix typo in parallel\_state ([MR \!3634](https://github.com/NVIDIA/Megatron-LM/commit/3c79a2c330290df58804c33e28e7c197fcc1f0b9))  
  * Fix CUDA graph logic determination ([MR \!3635](https://github.com/NVIDIA/Megatron-LM/commit/90efa3ef8a3c4f9e0f1db9f67ab9348bfa501387))  
  * Fix TE installation error ([MR \!3636](https://github.com/NVIDIA/Megatron-LM/commit/7e7322c01c9cb8ec254ecd9042700b22b70fe5c8))  
  * Ensure correct sharding type in local tests ([MR \!3643](https://github.com/NVIDIA/Megatron-LM/commit/946357f8dd7fdc12424b3a66bc999e6c0a02696c))  
  * Fix cudagraphed backward buffer reuse for last layer ([MR \!3645](https://github.com/NVIDIA/Megatron-LM/commit/ee61cf450d24760952e8995aab045ab6d55b986e))  
  * Set default for packed\_seq\_params in get\_rotary\_seq\_len ([MR \!3651](https://github.com/NVIDIA/Megatron-LM/commit/510d58c46664f44c556005ac928c5c531e12f761))  
  * Fix dynamic example script errors ([MR \!3653](https://github.com/NVIDIA/Megatron-LM/commit/72e290bf1f4bbf0c8047bb10a51da6ea6372e163))  
  * Guard TE import fix ([MR \!3666](https://github.com/NVIDIA/Megatron-LM/commit/ac198fc0d60a8c748597e01ca4c6887d3a7bcf3d))  
* Known issues

## NVIDIA Megatron Core 0.13.0

* Support bf16 dtype for optimizer states to use precision-aware optimizer in TransformerEngine  
* MoE
  * Features:  
    * Flexible Asymmetric Virtual Pipeline Parallelism with Custom Pipeline Layout (--pipeline-model-parallel-layout)  
    * Add support to pass custom parallelism groups to MoE modules.  
    * Add Hybrid Shard Data-Parallel support for MoE models (--num-distributed-optimizer-instances)  
    * Support EP \+ custom FSDP training for DeepSeek-V3  
    * FP8 support for Multi-Token-Prediction  
  * Memory Optimization  
    * Fine-grained recomputation to reduce activation memory. (--recompute-modules with \--recompute-granularity selective)  
    * Memory efficient token permutation by moving the probs multiplication from unpermutation to activation function of GroupedMLP.  
  * Performance Optimization  
    * MLA RoPE fusion kernel and YARN embedding cache.  
    * FP8 padding optimization of MoE models by padding the routing map.  
  * Bug fixes:  
    * Fix the aux loss calculation when expert\_bias or group limited routing is used. This leads to load\_balancing\_loss values change compared to the previous version.  
    * Fix packed sequence support for MLA  
  * Known Issues:  
    * MTP is not compatible with flexible pipeline layout, will be fixed at \!3594.  
    * MTP convergence issue with TP2, will be fixed at \!3594.

## NVIDIA Megatron Core 0.12.0

* Add FP8 recipe selection to arguments (--fp8-recipe, --first-last-layers-bf16, --num-layers-at-start-in-bf16, --num-layers-at-end-in-bf16)
* Context parallel: fix loss scaling when calculate_per_token_loss=True
* Make the number of data parallel communication buckets configurable (--ddp-num-buckets, --ddp-pad-buckets-for-high-nccl-busbw)
* Inference
  * Support in-flight batching and chunked KV cache
  * Reduce memory usage,
    * by not materializing full attention mask
    * by only materializing logits for the last token during decode
    * by removing an obsolete tensor reference
* Hybrid Model
  * Inference
    * Add CUDA graph support
    * Change tools/run_mamba_text_generation_server.py to use megatron.core.inference
    * Fix a shape issue when materializing logits for Mamba model
  * Improve initialization of Mamba layers
  * Add configuration switches (--mamba-state-dim, --mamba-head-dim, --mamba-num-groups, --is-hybrid-model)
  * Make num_floating_point_operations work with hybrid model
  * Make hybrid_conversion.py work with mixer that uses TE linear
  * Add FP8 support
  * Fix Mamba dt_bias tensor parallelism
  * Support multimodal tokenizer
  * Improve data parallelism scaling
* MoE
  * Features:
    * DeepEP support, compatible with all the parallelisms and token drop / dropless
    * Important precision improvement: Enable FP32/FP64 routing and unpermutation using –moe-router-dtype. FP32 is recommended for all fine-grained MoE training
    * CUDA Graph support for MoE
    * Multi-Token Prediction (MTP) Support
    * Fused indices_to_multihot kernel for DeepEP dispatcher
  * Bug fixes:
    * Fix Hang Issue with MoE+Dense Hybrid models
    * Update theoretical memory and tflops estimation for MoE and MLA
    * Fix MoE Aux loss scaling for per token loss
    * Fixes for group limited routing and expert bias. We verified these fixes through dsv3 e2e verifications
  * Known issues:
    * The ckpt trained with Custom FSDP for MoE may not be compatible with 3D parallel training.

## NVIDIA Megatron Core 0.11.0

* Add multi datacenter training support though N/S connection
* MoE
  * Features
    * Support DeepSeek-V3 fine-tuning
      * Aux-loss-free load balancing strategy
      * Node-limited routing and Device-limited routing support.
      * Tensor Parallelism support for MLA and Sequence Auxiliary Loss
      * MTP (with TP and PP support) is coming soon.
    * Permutation / Unpermutation fusion kernel from TransformerEngine.
    * Uneven virtual pipeline parallel split support in first and last PP stage.
  * Bug fixes:
    * Fix the grad scale when TP != expert-TP and average_in_collective is enabled in DDP.
    * Fix TEGroupedMLP distckpt compatibility issue with FP8 padding/unpadding.
  * Known Issues:
    * When training the Dense+MoE hybrid model, the process will hang if any PP rank does not have expert params.
* Add MX-FP16 support for optimizer and master weights
* CUDA Graph memory optimizations
* Enable UCC backend for PP communication
* Optimizer CPU offload support for memory savings
* Models
  * Initial RADIO/CRADIO implementation
  * llama3.2 support
* Hybrid Model
  * Support quantization via TensorRT Model Optimizer

## NVIDIA Megatron Core 0.10.0

* Adding MLA to MCore
* Enable FP8 for GroupedMLP
* MoE Parallel Folding
* Enhance MoE Architecture: Support MoE Layer Frequency Patterns and Configurable MoE FFN Hidden Size
* Multimodal: NVLM training and evaluation support in MCore
* Mamba Hybrid
  * Increase performance and reduce memory footprint of Triton language/compiler distributed caching
  * Add more unit testing and fix bugs

## NVIDIA Megatron Core 0.9.0

* Uneven pipeline parallelism
  * Enable pipeline parallelism where first and last ranks have fewer transformer layers than the intermediate ranks
* Per layer CUDAGraph support for GPT training with Transformer Engine modules
* Enable different TP sizes for the vision encoder
* Enable pipeline parallelism for T5 & Llava models
* Support multi-tile multi-image input in Llava models
* MoE
  * FP8 support
  * Runtime upcycling support
  * Dispatcher implementation optimizations
  * Shared expert support with overlapping optimizations
    * Qwen Model support
* Known Issues
  * When using sequence parallel, during the transformer block forward pass, dropout is not using the appropriate rng context.
* NVRx / Fault tolerance
  * fault and hang detection in addition to existing straggler detection
  * graceful exit and auto restart

## NVIDIA Megatron Core 0.8.0

* Multimodal
  * Added initial support for training vision language models using the LLaVA architecture
  * Added initial support for inference with multimodal inputs
  * End-to-end multimodal example from data collection to training to evaluation is provided in examples/multimodal
* MoE
  * Context Parallel support.
  * Distributed checkpoint support for grouped GEMM.
* Mamba

## NVIDIA Megatron Core 0.7.0

* MoE
  * Token drop support
  * Several efficiency optimizations
  * Improved model parallelism
  * Memory optimizations
* Distributed checkpointing
  * Enabled for Retro
  * Asynchronous checkpoint saving
* Several minor bug fixes, speed improvements, and memory optimizations

## NVIDIA Megatron Core 0.6.0

* MoE (Mixture of Experts)
  * Performance optimization
    * Communication optimization for multi GPU and Single GPU
    * 23% improvement (323 TFLOPS/GPU) over MCore 0.5.0 on Mixtral with Hopper BF16
    * GroupedMLP enhancement for Hopper
    * DP Overlapping. Support overlapping computation with gradient reduction and parameter gathering.
  * All-to-All based Token Dispatcher
  * Layer-wise logging for load balancing loss.
  * Improved expert parallel support including distributed optimizer.
* Distributed optimizer
* RETRO
  * Data processing
* BERT
  * Distributed checkpointing
* Dist checkpointing
  * PyTorch native distributed backend
  * Improved saving/loading speed
* TensorRT-LLM Export
  * Integration with TensorRT Model Optimizer Post-training quantization (PTQ)
  * Text generation driver to perform PTQ in Megatron-LM
  * Llama2 and Nemotron3-8b examples to use TensorRT-LLM unified build API to build engine after training.
* Several minor enhancements, bug fixes, and documentation updates

## NVIDIA Megatron Core 0.5.0

### Key Features and Enhancements

Megatron core documentation is now [live!](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/index.html#quick-start)

### Model Features

* MoE (Mixture of Experts)
  * Support for Z-loss, Load balancing and Sinkhorn
  * Layer and communications refactor
  * Richer parallelism mappings and EP can be combined with other model parallel techniques for larger MoE variants, e.g. EP + TP + DP + SP + PP
  * Token dropless architecture with Top-K routing
  * Performance optimization with with GroupedGEMM when number of local experts is > 1
  * Distributed checkpointing
* Interleaved rotary embedding

### Datasets

* Masked WordPiece datasets for BERT and T5
* Raw and mock datasets

### Parallelism

### Performance

* Activation offloading to CPU
* Rope and Swiglu fusion
* Sliding window attention (via Transformer Engine)

### General Improvements

* Timers

## NVIDIA Megatron Core 0.4.0

### Key Features and Enhancements

#### Models

* BERT
* RETRO
* T5

#### Parallelism

* Mixture of Experts support for GPT
* Model parallel efficient Distributed Data Parallel (DDP)
* Context Parallel (2D Tensor Parallel) support

#### Datasets

* GPT Dataset
* Blended Dataset
