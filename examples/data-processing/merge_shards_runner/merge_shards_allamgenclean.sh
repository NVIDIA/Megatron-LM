DATA_FOLDER="../dumped"
mkdir -p  $DATA_FOLDER
UK_SOUTH_SAS_TOKEN="sv=2023-01-03&ss=btqf&srt=sco&st=2024-02-12T10%3A52%3A04Z&se=2025-02-28T10%3A52%3A00Z&sp=rwdxftlacup&sig=L%2FrQa%2B7Inj9BzPimSIGwWsB%2FcEVWTD8evBp7Po5xS7M%3D"
AZURE_DATA_LINK="https://allamllmuksstandard.blob.core.windows.net/llm-data/allam_gen_cleaned"
azcopy copy "$AZURE_DATA_LINK/?$UK_SOUTH_SAS_TOKEN" $DATA_FOLDER/ --recursive --overwrite=false

ls -ld $DATA_FOLDER/allam_gen_cleaned/cleaned_filtered/* | awk '{print $9}' | xargs -I {} sh -c 'newname=$(echo "{}" | sed "s/\(.*\/\)\(.*\).jsonl\([0-9]*\)/\1ar_web-\2.\3.jsonl/"); mv {} "$newname"'
ls -ld $DATA_FOLDER/allam_gen_cleaned/cleaned_filtered/* | awk '{print $9}' | grep "CulturaX_ar" | xargs -I {} sh -c 'newname=$(echo "{}" | sed "s/CulturaX_ar/CulturaXar/"); mv {} "$newname"'
ls -ld $DATA_FOLDER/allam_gen_cleaned/cleaned_filtered/* | awk '{print $9}' | grep "ar_web-arabicweb16_v2" | xargs -I {} sh -c 'newname=$(echo "{}" | sed "s/ar_web-arabicweb16_v2/ar_web-arabicweb16v2/"); mv {} "$newname"'
ls -ld $DATA_FOLDER/allam_gen_cleaned/cleaned_filtered/* | awk '{print $9}' | grep "ar_" | xargs -I {} sh -c 'newname=$(echo "{}" | sed "s/ar_/ar_Allam-GC-/"); mv {} "$newname"'

mkdir -p $DATA_FOLDER/allam_gen_cleaned/data/arabicweb16v2
mkdir -p $DATA_FOLDER/allam_gen_cleaned/data/arabicweb22
mkdir -p $DATA_FOLDER/allam_gen_cleaned/data/CulturaXar
mkdir -p $DATA_FOLDER/allam_gen_cleaned/data/metadialog
mkdir -p $DATA_FOLDER/allam_gen_cleaned/data/oscar2301

mv $DATA_FOLDER/allam_gen_cleaned/cleaned_filtered/ar_Allam-GC-web-arabicweb16v2* $DATA_FOLDER/allam_gen_cleaned/data/arabicweb16v2/
mv $DATA_FOLDER/allam_gen_cleaned/cleaned_filtered/ar_Allam-GC-web-arabicweb22* $DATA_FOLDER/allam_gen_cleaned/data/arabicweb22/
mv $DATA_FOLDER/allam_gen_cleaned/cleaned_filtered/ar_Allam-GC-web-CulturaXar* $DATA_FOLDER/allam_gen_cleaned/data/CulturaXar/
mv $DATA_FOLDER/allam_gen_cleaned/cleaned_filtered/ar_Allam-GC-web-metadialog* $DATA_FOLDER/allam_gen_cleaned/data/metadialog/
mv $DATA_FOLDER/allam_gen_cleaned/cleaned_filtered/ar_Allam-GC-web-oscar2301* $DATA_FOLDER/allam_gen_cleaned/data/oscar2301/

azcopy copy "$DATA_FOLDER/allam_gen_cleaned/data" "$AZURE_DATA_LINK/?$UK_SOUTH_SAS_TOKEN" --recursive --overwrite=true

rm -rf "$DATA_FOLDER/allam_gen_cleaned/"
azcopy copy "$AZURE_DATA_LINK/data/?$UK_SOUTH_SAS_TOKEN" "$DATA_FOLDER/allam_gen_cleaned/data/" --recursive 

NUM_TOKEN=40268055040

python examples/data-processing/merge_shards.py \
--input-folder-path "https://allamllmuksstandard.blob.core.windows.net/llm-data/allam_gen_cleaned/data/arabicweb16v2/" \
--output-folder-path "https://allamllmuksstandard.blob.core.windows.net/llm-data/allam_gen_cleaned/Allam-GC_merged_shards" \
--prefix-name "ar_Allam-GC-web-arabicweb16v2_" \
--az-configs "examples/configs/azure_uk_south.json" \
--az-sample-yaml-job-file "examples/data-processing/az_templates/template_merge_shard.yaml" \
--compute-target 'azure' \
--shard-size $NUM_TOKEN 

python examples/data-processing/merge_shards.py \
--input-folder-path "https://allamllmuksstandard.blob.core.windows.net/llm-data/allam_gen_cleaned/data/arabicweb22/" \
--output-folder-path "https://allamllmuksstandard.blob.core.windows.net/llm-data/allam_gen_cleaned/Allam-GC_merged_shards" \
--prefix-name "ar_Allam-GC-web-arabicweb22_" \
--az-configs "examples/configs/azure_uk_south.json" \
--az-sample-yaml-job-file "examples/data-processing/az_templates/template_merge_shard.yaml" \
--compute-target 'azure' \
--shard-size $NUM_TOKEN 

python examples/data-processing/merge_shards.py \
--input-folder-path "https://allamllmuksstandard.blob.core.windows.net/llm-data/allam_gen_cleaned/data/CulturaXar/" \
--output-folder-path "https://allamllmuksstandard.blob.core.windows.net/llm-data/allam_gen_cleaned/Allam-GC_merged_shards" \
--prefix-name "ar_Allam-GC-web-CulturaXar_" \
--az-configs "examples/configs/azure_uk_south.json" \
--az-sample-yaml-job-file "examples/data-processing/az_templates/template_merge_shard.yaml" \
--compute-target 'azure' \
--shard-size $NUM_TOKEN 

python examples/data-processing/merge_shards.py \
--input-folder-path "https://allamllmuksstandard.blob.core.windows.net/llm-data/allam_gen_cleaned/data/metadialog/" \
--output-folder-path "https://allamllmuksstandard.blob.core.windows.net/llm-data/allam_gen_cleaned/Allam-GC_merged_shards" \
--prefix-name "ar_Allam-GC-web-metadialog_" \
--az-configs "examples/configs/azure_uk_south.json" \
--az-sample-yaml-job-file "examples/data-processing/az_templates/template_merge_shard.yaml" \
--compute-target 'azure' \
--shard-size $NUM_TOKEN 

python examples/data-processing/merge_shards.py \
--input-folder-path "https://allamllmuksstandard.blob.core.windows.net/llm-data/allam_gen_cleaned/data/oscar2301/" \
--output-folder-path "https://allamllmuksstandard.blob.core.windows.net/llm-data/allam_gen_cleaned/Allam-GC_merged_shards" \
--prefix-name "ar_Allam-GC-web-oscar2301_" \
--az-configs "examples/configs/azure_uk_south.json" \
--az-sample-yaml-job-file "examples/data-processing/az_templates/template_merge_shard.yaml" \
--compute-target 'azure' \
--shard-size $NUM_TOKEN 