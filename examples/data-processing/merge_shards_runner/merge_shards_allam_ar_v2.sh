AZJOB_FILE="examples/data-processing/az_templates/template_merge_shard.yaml"
AZJOB_CONFIG="examples/configs/azure_login_configs.json"
AZ_OUTPUT_FOLDER="https://allamllmuksstandard.blob.core.windows.net/llm-data/allam-v2-data/merged_shards"
NUM_TOKEN=40486543872

python examples/data-processing/merge_shards.py \
--input-folder-path "https://allamllmuksstandard.blob.core.windows.net/llm-data/allam-v2-data/ar/encyclopedias/" \
--output-folder-path "$AZ_OUTPUT_FOLDER" \
--az-sample-yaml-job-file "$AZJOB_FILE" \
--prefix-name "ar_Allam-encyclopedias_" \
--shard-size $NUM_TOKEN \
--az-configs $AZJOB_CONFIG \
--use-file-input \
--compute-target 'azure'

python examples/data-processing/merge_shards.py \
--input-folder-path "https://allamllmuksstandard.blob.core.windows.net/llm-data/allam-v2-data/ar/news/" \
--output-folder-path "$AZ_OUTPUT_FOLDER" \
--az-sample-yaml-job-file "$AZJOB_FILE" \
--prefix-name "ar_Allam-news_" \
--shard-size $NUM_TOKEN \
--az-configs $AZJOB_CONFIG \
--use-file-input \
--compute-target 'azure'

python examples/data-processing/merge_shards.py \
--input-folder-path "https://allamllmuksstandard.blob.core.windows.net/llm-data/allam-v2-data/ar/others/" \
--output-folder-path "$AZ_OUTPUT_FOLDER" \
--az-sample-yaml-job-file "$AZJOB_FILE" \
--prefix-name "ar_Allam-others_" \
--shard-size $NUM_TOKEN \
--az-configs $AZJOB_CONFIG \
--use-file-input \
--compute-target 'azure'

python examples/data-processing/merge_shards.py \
--input-folder-path "https://allamllmuksstandard.blob.core.windows.net/llm-data/allam-v2-data/ar/transcribed/" \
--output-folder-path "$AZ_OUTPUT_FOLDER" \
--az-sample-yaml-job-file "$AZJOB_FILE" \
--prefix-name "ar_Allam-transcribed_" \
--shard-size $NUM_TOKEN \
--az-configs $AZJOB_CONFIG \
--use-file-input \
--compute-target 'azure'

python examples/data-processing/merge_shards.py \
--input-folder-path "https://allamllmuksstandard.blob.core.windows.net/llm-data/allam-v2-data/ar/web/" \
--output-folder-path "$AZ_OUTPUT_FOLDER" \
--az-sample-yaml-job-file "$AZJOB_FILE" \
--prefix-name "ar_Allam-web_" \
--shard-size $NUM_TOKEN \
--az-configs $AZJOB_CONFIG \
--use-file-input \
--compute-target 'azure'

python examples/data-processing/merge_shards.py \
--input-folder-path "https://allamllmuksstandard.blob.core.windows.net/llm-data/allam-v2-data/ar/translated/en2ar_books_corpus_formatted/" \
--output-folder-path "$AZ_OUTPUT_FOLDER" \
--az-sample-yaml-job-file "$AZJOB_FILE" \
--prefix-name "trans-ar_books3_" \
--shard-size $NUM_TOKEN \
--az-configs $AZJOB_CONFIG 

python examples/data-processing/merge_shards.py \
--input-folder-path "https://allamllmuksstandard.blob.core.windows.net/llm-data/allam-v2-data/ar/translated/en2ar_c4/" \
--output-folder-path "$AZ_OUTPUT_FOLDER" \
--az-sample-yaml-job-file "$AZJOB_FILE" \
--prefix-name "trans-ar_c4_" \
--shard-size $NUM_TOKEN \
--az-configs $AZJOB_CONFIG 

python examples/data-processing/merge_shards.py \
--input-folder-path "https://allamllmuksstandard.blob.core.windows.net/llm-data/allam-v2-data/ar/translated/en2ar_peS2o/" \
--output-folder-path "$AZ_OUTPUT_FOLDER" \
--az-sample-yaml-job-file "$AZJOB_FILE" \
--prefix-name "trans-ar_peS20_" \
--shard-size $NUM_TOKEN  \
--az-configs $AZJOB_CONFIG 

python examples/data-processing/merge_shards.py \
--input-folder-path "https://allamllmuksstandard.blob.core.windows.net/llm-data/allam-v2-data/ar/translated/en2ar_wikipedia/" \
--output-folder-path "$AZ_OUTPUT_FOLDER" \
--az-sample-yaml-job-file "$AZJOB_FILE" \
--prefix-name "trans-ar_wiki_" \
--shard-size $NUM_TOKEN \
--az-configs $AZJOB_CONFIG


# Process the books data locally. 
# This is because the file names are with newlines and spaces.
# azcopy does not support downloading files with newlines and spaces in the file names. (or I couldn't figure out how to do it)
azcopy copy "https://allamllmuksstandard.blob.core.windows.net/llm-data/allam-v2-data/ar/books/?sv=2023-01-03&ss=btqf&srt=sco&st=2023-12-03T21%3A43%3A17Z&se=2025-05-30T21%3A43%3A00Z&sp=rwdxftlacup&sig=51chDeU9Xqnk7GrTSA3u2gEfdgCUQIq9SDAJEQCPQxE%3D" "../RAW_DATA_FOLDER/allam-v2-data/ar/" --recursive

mkdir -p "../RAW_DATA_FOLDER/allam-v2-data/merged_shards/"

python examples/data-processing/merge_shards.py \
--input-folder-path "../RAW_DATA_FOLDER/allam-v2-data/ar/books/" \
--output-folder-path "../RAW_DATA_FOLDER/allam-v2-data/merged_shards/" \
--az-sample-yaml-job-file "$AZJOB_FILE" \
--prefix-name "ar_Allam-books_" \
--shard-size $NUM_TOKEN \
--az-configs $AZJOB_CONFIG \
--use-file-input \
--compute-target 'local'

azcopy copy "../RAW_DATA_FOLDER/allam-v2-data/merged_shards/" "https://allamllmuksstandard.blob.core.windows.net/llm-data/allam-v2-data/?sv=2023-01-03&ss=btqf&srt=sco&st=2023-12-03T21%3A43%3A17Z&se=2025-05-30T21%3A43%3A00Z&sp=rwdxftlacup&sig=51chDeU9Xqnk7GrTSA3u2gEfdgCUQIq9SDAJEQCPQxE%3D" --recursive