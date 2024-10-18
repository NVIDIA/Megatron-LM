# Megatron Core Export

This module is used to export megatron core models to different inference frameworks. 
Currently we support TRTLLM export . In the future we will be adding support for VLLM etc. 

## PTQ AND EXPORT
Follow the instructions in [ptq_and_trtllm_export](./ptq_and_trtllm_export) to do post training quantization, followed by an export to TRTLLM format. 

# TRTLLM EXPORT
Follow the instructions in [trtllm_export](./trtllm_export/) to do export to TRTLLM checkpoint format alone.