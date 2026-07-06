# Megatron Inference

Use `MegatronLLM` (sync) or `MegatronAsyncLLM` (async, with HTTP serving
through `serve()`) for typical inference workflows. Both classes hide the underlying
engine pipeline (`DynamicInferenceContext`, `GPTInferenceWrapper`,
`TextGenerationController`, and `DynamicInferenceEngine`) and provide a vLLM-style
`generate(prompts, sampling_params)` API.

For the full documentation, including supported features, basic and advanced
usage, direct compared to coordinator modes, the OpenAI-compatible server, known
limitations, and the roadmap, refer to the [Megatron Core Inference user guide](../../../docs/mcore-inference-user-guide.md).

## Additional Resources

- Examples: [`examples/inference/`](../../../examples/inference/)
- Low-level engine building blocks live in this directory:
  `DynamicInferenceEngine`, `DynamicInferenceContext`,
  `TextGenerationController`, and the model inference wrappers under
  `model_inference_wrappers/`.
