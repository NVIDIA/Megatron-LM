for ENG_TOK in 0 1 2 3 4 5 6 7 8 9 10
do
    sed "s/\${{ENG_TOK}}/$ENG_TOK/g" examples/pretrain-llama/training/llama_en_reasoning_ar_data_mix_hyp_tune/llama_en_reasoning_ar_data_mix_hyp_tune.yaml > examples/pretrain-llama/training/llama_en_reasoning_ar_data_mix_hyp_tune/temp.yaml
    az ml job create --subscription c7209a17-0d9f-41df-8e45-e0172343698d --resource-group LLM-Test --workspace-name Provisioning-Test --file examples/pretrain-llama/training/llama_en_reasoning_ar_data_mix_hyp_tune/temp.yaml
    rm examples/pretrain-llama/training/llama_en_reasoning_ar_data_mix_hyp_tune/temp.yaml
done