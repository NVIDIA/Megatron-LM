set -e

az ml job create --subscription c7209a17-0d9f-41df-8e45-e0172343698d \
--resource-group LLM-Test \
--workspace-name Provisioning-Test \
--file "examples/pretrain-llama/training/llama_ve/llama70b_ve_init_emb_en_reasoning_ar_with_trans_from_scratch/llama70b_ve_init_emb_en_reasoning_ar_with_trans_from_scratch.yaml"
