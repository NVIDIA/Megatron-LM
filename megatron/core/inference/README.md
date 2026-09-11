# Megatron Inference

Megatron Inference is an MCore-native generation capability for RL rollouts,
evaluation, and debugging of MCore models. It is complementary to dedicated
serving engines such as vLLM, SGLang, and TensorRT-LLM: generation runs on the
same MCore model, parallelism, and kernels you train with, so there is no
Hugging Face conversion step and no cross-framework numerical gap between
training and rollout.

Use `MegatronLLM` (sync) or `MegatronAsyncLLM` (async) for typical inference
workflows. Both classes hide the underlying engine pipeline
(`DynamicInferenceContext`, `GPTInferenceWrapper`, `TextGenerationController`,
and `DynamicInferenceEngine`) and provide a vLLM-style
`generate(prompts, sampling_params)` API.

Features include dynamic (in-flight) batching, chunked prefill, paged attention
with prefix caching, CUDA graphs, speculative decoding, weight refit and
resharding between training and inference, batch-invariant kernels for
training/inference log-prob consistency, and an OpenAI-compatible HTTP server
(chat templates, tool calling, reasoning parsers, and streaming) exposed through
`MegatronAsyncLLM.serve()`.

For the full documentation, including supported features, basic and advanced
usage, direct compared to coordinator modes, the OpenAI-compatible server, known
limitations, and the roadmap, refer to the [Megatron Core Inference user guide](../../../docs/mcore-inference-user-guide.md).

## Additional Resources

- Examples: [`examples/inference/`](../../../examples/inference/)
- Low-level engine building blocks live in this directory:
  `DynamicInferenceEngine`, `DynamicInferenceContext`,
  `TextGenerationController`, and the model inference wrappers under
  `model_inference_wrappers/`.
