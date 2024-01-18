set -e

az ml job create --subscription c7209a17-0d9f-41df-8e45-e0172343698d \
 --resource-group LLM-Test \
 --workspace-name Provisioning-Test \
 --file "examples/pretrain-llama/training/llama_ve/llama13b_ve_init_emb_en_reasoning_ar_with_trans_from_scratch/llama13b_ve_init_emb_en_reasoning_ar_with_trans_from_scratch_resume.yaml"
