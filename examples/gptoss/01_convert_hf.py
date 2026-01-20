#!/usr/bin/env python3
"""Convert HuggingFace GPT-OSS checkpoint to Megatron format."""

from megatron.bridge import AutoBridge

if __name__ == "__main__":
    HF_MODEL = "openai/gpt-oss-20b"
    SAVE_PATH = "./megatron_checkpoints/gpt_oss_20b"
    
    print(f"Converting {HF_MODEL} to Megatron format...")
    
    bridge = AutoBridge.from_hf_pretrained(HF_MODEL, trust_remote_code=True)
    provider = bridge.to_megatron_provider()
    provider.tensor_model_parallel_size = 2
    provider.pipeline_model_parallel_size = 4
    provider.finalize()
    
    model = provider.provide_distributed_model(wrap_with_ddp=False)
    
    bridge.save_megatron_model(
        model,
        SAVE_PATH,
        hf_tokenizer_path=HF_MODEL
    )
    
    print(f"Saved Megatron checkpoint to {SAVE_PATH}")
