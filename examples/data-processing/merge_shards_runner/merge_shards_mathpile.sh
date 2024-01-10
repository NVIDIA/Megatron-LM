bash scripts/envs/env_vars.sh

DATASET_PATH="../RAW_DATA_FOLDER/MathPile/"
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/GAIR/MathPile $DATASET_PATH
cd $DATASET_PATH
git lfs pull
find . -type f -name '*.gz' | sort | xargs -I {} gunzip {}

rm -rf custom_merged_shards
mkdir -p custom_merged_shards
find . -type f -name '*jsonl' | grep -v stackexchange | sort | xargs -I {} cat {} >> custom_merged_shards/mathpile-arxiv-cc-proofwiki-textbook-wiki.jsonl

azcopy copy . "https://allamllmuksstandard.blob.core.windows.net/llm-data/?sv=2023-01-03&ss=btqf&srt=sco&st=2024-01-09T16%3A06%3A06Z&se=2024-01-10T16%3A06%3A06Z&sp=rwdxftlacup&sig=NEWQ%2BWSRi9Wpmp72FxuF5vA07lwx7V9tJjVP3NZQAc4%3D" --recursive --overwrite=false 

cd ../../ALLaM-Megatron-LM/

SHARD_SIZE=80268055040
python examples/data-processing/merge_shards.py \
--input-folder-path "https://allamllmuksstandard.blob.core.windows.net/llm-data/MathPile/custom_merged_shards/" \
--output-folder-path "https://allamllmuksstandard.blob.core.windows.net/llm-data/MathPile/mathpile_merged_shards" \
--prefix-name "en_mathpile-arxiv-cc-proofwiki-textbook-wiki_" \
--az-configs "examples/configs/azure_login_configs.json" \
--az-sample-yaml-job-file "examples/data-processing/az_templates/template_merge_shard.yaml" \
--compute-target 'azure' \
--use-file-input \
--shard-size $SHARD_SIZE
