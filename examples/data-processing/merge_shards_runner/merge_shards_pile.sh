
PILE_PATH="../RAW_DATA_FOLDER/the_pile/"
python examples/data-processing/sample_json_data_from_meta.py \
--input-folder-path "$PILE_PATH/data/" \
--output-folder-path "$PILE_PATH/data_by_domain" \
--meta-keys meta pile_set_name

for file in $PILE_PATH/data_by_domain/*; do mv "$file" "${file// /_}"; done
cd $PILE_PATH/data_by_domain
for file in *; do mv "$file" "${file//_/-}"; done
cd -

ls -l $PILE_PATH/data_by_domain/* | awk '{print $9}' | xargs -I {}  wc -l "{}" | awk '{sum += $1} END {print sum}' 


split --line-bytes=85G --additional-suffix=.jsonl -d -a 4 RedPajamaCommonCrawl.jsonl ../merged_shards/RedPajamaC4_

# SIZE=53486543872
# python examples/data-processing/merge_shards.py \
# --input-folder-path "https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/the_pile/" \
# --output-folder-path "https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/the_pile/merged_shards" \
# --prefix-name "en_pile_" \
# --az-configs "examples/configs/azure_login_configs.json" \
# --az-sample-yaml-job-file "examples/data-processing/az_templates/template_merge_shard.yaml" \
# --compute-target 'azure' \
# --shard-size $SIZE



# SIZE=53486543872
# mkdir -p ../RAW_DATA_FOLDER/the_pile/merged_shards

# python examples/data-processing/merge_shards.py \
# --input-folder-path "../RAW_DATA_FOLDER/the_pile/" \
# --output-folder-path "../RAW_DATA_FOLDER/the_pile/merged_shards" \
# --prefix-name "en_pile_" \
# --compute-target 'local' \
# --shard-size $SIZE