set -e

# TP=$1
# PP=$2

# pairs=(
#     "1 8"
#     "2 4"
#     "2 8"
#     "4 2"
#     "4 4"
#     "4 8"
#     "8 1"
#     "8 2"
#     "8 4"
#     "8 8"
# )

pairs=(
    "8 8"
)
for VP in None; do 
    # currently $VP is manually deactivated inside the bash script
    # if you want to activate it, please add the --num-layers-per-virtual-pipeline-stage to the bash script linked in the yaml file.
    for MBS in 4; do
        for pair in "${pairs[@]}"; do
            # Split the pair into TP and PP
            read TP PP <<< "$pair"
            sed -e "s/\${{TP}}/$TP/g" \
            -e "s/\${{PP}}/$PP/g" \
            -e "s/\${{VP}}/$VP/g" \
            -e "s/\${{MBS}}/$MBS/g" examples/pretrain-llama/training/llama_ve/llama70b_ve_init_emb_en_reasoning_ar_with_trans_from_scratch_test/llama70b_ve_init_emb_en_reasoning_ar_with_trans_from_scratch_test_vp_mbs.yaml > examples/pretrain-llama/training/llama_ve/llama70b_ve_init_emb_en_reasoning_ar_with_trans_from_scratch_test/temp.yaml

            az ml job create --subscription c7209a17-0d9f-41df-8e45-e0172343698d \
            --resource-group LLM-Test \
            --workspace-name Provisioning-Test \
            --file "examples/pretrain-llama/training/llama_ve/llama70b_ve_init_emb_en_reasoning_ar_with_trans_from_scratch_test/temp.yaml"

            rm examples/pretrain-llama/training/llama_ve/llama70b_ve_init_emb_en_reasoning_ar_with_trans_from_scratch_test/temp.yaml
        done
    done
done