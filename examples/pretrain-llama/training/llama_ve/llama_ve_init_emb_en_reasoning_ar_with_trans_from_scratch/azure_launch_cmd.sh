for LR in 1e-5 2e-5 3e-5 4e-5 5e-5 1e-6
do
    sed "s/\${{lr_rate}}/$LR/g" examples/pretrain-llama/training/llama_ve/llama_ve_init_emb_en_reasoning_ar_lr_hyp_tune/llama_ve_init_emb_en_reasoning_ar_lr_hyp_tune.yaml > examples/pretrain-llama/training/llama_ve/llama_ve_init_emb_en_reasoning_ar_lr_hyp_tune/temp.yaml
    cat examples/pretrain-llama/training/llama_ve/llama_ve_init_emb_en_reasoning_ar_lr_hyp_tune/temp.yaml
    az ml job create --subscription c7209a17-0d9f-41df-8e45-e0172343698d \
    --resource-group LLM-Test \
    --workspace-name Provisioning-Test \
    --file "examples/pretrain-llama/training/llama_ve/llama_ve_init_emb_en_reasoning_ar_lr_hyp_tune/temp.yaml"
    rm examples/pretrain-llama/training/llama_ve/llama_ve_init_emb_en_reasoning_ar_lr_hyp_tune/temp.yaml
done