set -e

TP=$1
PP=$2
sed -e "s/\${{TP}}/$TP/g" -e "s/\${{PP}}/$PP/g" examples/pretrain-llama/training/llama_ve/llama70b_ve_init_emb_en_reasoning_ar_with_trans_from_scratch_test/llama70b_ve_init_emb_en_reasoning_ar_with_trans_from_scratch_test.yaml > examples/pretrain-llama/training/llama_ve/llama70b_ve_init_emb_en_reasoning_ar_with_trans_from_scratch_test/temp.yaml

az ml job create --subscription c7209a17-0d9f-41df-8e45-e0172343698d \
 --resource-group LLM-Test \
 --workspace-name Provisioning-Test \
 --file "examples/pretrain-llama/training/llama_ve/llama70b_ve_init_emb_en_reasoning_ar_with_trans_from_scratch_test/temp.yaml"

 rm examples/pretrain-llama/training/llama_ve/llama70b_ve_init_emb_en_reasoning_ar_with_trans_from_scratch_test/temp.yaml