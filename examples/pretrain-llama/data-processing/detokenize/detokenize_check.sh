#!/bin/bash
set -e

ar_file="allam_data_2-1_splits-llama2-indexed_data/llama2_bin_idx/ar_encyclopedias_split_00_text_document"
ar_bin="${ar_file}.bin"
ar_idx="${ar_file}.idx"

en_file="allam_data_2-1_splits-llama2-indexed_data/llama2_bin_idx/en_books_books_split_02_text_document"
en_bin="${en_file}.bin"
en_idx="${en_file}.idx"

oputput_path="outputs/data"
tok_path="outputs/tokenizer"
tok_file="${tok_path}/tokenizer.model"
final_output="outputs/results"

if [ ! -d $oputput_path ]; then
    mkdir -p ${oputput_path}
fi

if [ ! -d $tok_path ]; then
    mkdir -p ${tok_path}
fi

if [ ! -d $final_output ]; then
    mkdir -p ${final_output}
fi

if [ ! -f $tok_file ]; then
    azure_tok="allam_data_2-1_splits-llama2-indexed_data/tokenizer/tokenizer.model"
    azure_path="https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/${azure_tok}?sp=racwdl&st=2023-11-13T10:57:16Z&se=2025-04-17T18:57:16Z&spr=https&sv=2022-11-02&sr=c&sig=Gg2nHVydrJozaXGrcst0iF3mVgKkhRWO05Mt%2Bfp%2F69Y%3D"
    azcopy copy "${azure_path}" "${tok_file}"
fi

file_paths=($ar_bin $ar_idx $en_bin $en_idx)

for file_path in "${file_paths[@]}"
do
    output_path="${oputput_path}/${file_path}"
    if [ ! -f $output_path ]; then
        azure_path="https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/${file_path}?sp=racwdl&st=2023-11-13T10:57:16Z&se=2025-04-17T18:57:16Z&spr=https&sv=2022-11-02&sr=c&sig=Gg2nHVydrJozaXGrcst0iF3mVgKkhRWO05Mt%2Bfp%2F69Y%3D"
        azcopy copy "${azure_path}" "${output_path}"
    fi
done

python yz_detokenize.py --tokenizer_path "${tok_file}" --source_prefix_path "${oputput_path}/${en_file}"
