az ml job create --subscription c7209a17-0d9f-41df-8e45-e0172343698d \
 --resource-group LLM-Test \
 --workspace-name Provisioning-Test \
 --file "examples/pretrain-llama/training/llama_ve/llama_ve_init_emb_ar_from_scratch/llama_ve_init_emb_ar_from_scratch.yaml"

#  az ml job create --subscription c7209a17-0d9f-41df-8e45-e0172343698d \
#  --resource-group LLM-Test \
#  --workspace-name Provisioning-Test \
#  --file "examples/pretrain-llama/training/llama_ve/llama_ve_init_emb_ar_from_scratch/llama_ve_init_emb_ar_from_scratch_resume.yaml"