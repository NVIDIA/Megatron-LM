# Megatron Core Export

This module is used to export megatron core models to different inference frameworks. 
Currently we support TRTLLM export . In the future we will be adding support for VLLM etc. 

## PTQ AND EXPORT
Follow the examples of [TensorRT Model Optimizer](../post_training/modelopt) to perform post training quantization, followed by an export to a HF-like checkpoint for TensorRT-LLM, vLLM, and SGLang deployment. 

# TRTLLM EXPORT
Follow the instructions in [trtllm_export](./trtllm_export/) to do export to TRTLLM checkpoint format alone.
