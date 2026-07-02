# Megatron Inference

Use `MegatronLLM` (sync) or `MegatronAsyncLLM` (async, with HTTP serving via
`serve()`) for typical inference workflows. Both classes hide the underlying
engine pipeline (`DynamicInferenceContext` + `GPTInferenceWrapper` +
`TextGenerationController` + `DynamicInferenceEngine`) and provide a vLLM-style
`generate(prompts, sampling_params)` API.

**For the full documentation — supported features, basic and advanced usage,
direct vs. coordinator (indirect) modes, the OpenAI-compatible server, known
limitations, and the roadmap — see the user guide:**

➡️ [`docs/mcore-inference-user-guide.md`](../../../docs/mcore-inference-user-guide.md)

## See also

- Examples: [`examples/inference/`](../../../examples/inference/)
- Low-level engine building blocks live in this directory:
  `DynamicInferenceEngine`, `DynamicInferenceContext`,
  `TextGenerationController`, and the model inference wrappers under
  `model_inference_wrappers/`.
