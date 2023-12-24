set -e

TOTAL_NUM_TOKENS=20_000_000_000

for ENG in 5 8 10 12 14 16
do
   
    AR=$((20 - $ENG))

    export ENG_TOK=$ENG
    export AR_TOK=$AR
    echo "Training with ENG_TOK: $ENG_TOK"
    echo "Training with AR_TOK: $AR_TOK"

    echo "{\"en\": $ENG_TOK,\"ar\": $AR_TOK}" > examples/pretrain-llama/training/llama_ve/llama_ve_init_emb_en_reasoning_ar_data_mix_hyp_tune/lang_prob.json

    python examples/data-processing/data_ratio_from_file.py \
--prefix-paths-from-json "examples/pretrain-llama/training/llama_ve/llama_ve_init_emb_en_reasoning_ar_data_mix_hyp_tune/data.json" \
--domain-ratio-from-json "examples/pretrain-llama/training/llama_ve/llama_ve_init_emb_en_reasoning_ar_data_mix_hyp_tune/data_ratio.json" \
--lang-select-prob-json "examples/pretrain-llama/training/llama_ve/llama_ve_init_emb_en_reasoning_ar_data_mix_hyp_tune/lang_prob.json" \
--total-token $TOTAL_NUM_TOKENS \
--exclude-iterator-json "examples/pretrain-llama/training/llama_ve/llama_ve_init_emb_en_reasoning_ar_data_mix_hyp_tune/exclude_iterator.json" \
--prefix-for-file-path "\$BIN_IDX_PATH/" \
--export-script "examples/pretrain-llama/training/llama_ve/llama_ve_init_emb_en_reasoning_ar_data_mix_hyp_tune/iter_prob.sh"
    
    sleep 2

    az ml job create --subscription c7209a17-0d9f-41df-8e45-e0172343698d \
    --resource-group LLM-Test \
    --workspace-name Provisioning-Test \
    --file "examples/pretrain-llama/training/llama_ve/llama_ve_init_emb_en_reasoning_ar_data_mix_hyp_tune/llama_ve_en_reasoning_ar_data_mix_hyp_tune.yaml"

    sleep 2

    rm examples/pretrain-llama/training/llama_ve/llama_ve_init_emb_en_reasoning_ar_data_mix_hyp_tune/iter_prob.sh

done