#!/bin/bash
set -e

output_data="outputs/data"
bin_idx_list_path="outputs/bin_idx_list.jsonl"
original_jsonl_path="outputs/original.jsonl"
tok_path="outputs/tokenizer"
tok_file="${tok_path}/tokenizer.model"
final_output="outputs/results"

base_azure_data_dir="allam_data_2-1_splits-llama2-indexed_data/llama2_bin_idx"
azure_path="https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/${base_azure_data_dir}/?sp=racwdl&st=2023-11-13T10:57:16Z&se=2025-04-17T18:57:16Z&spr=https&sv=2022-11-02&sr=c&sig=Gg2nHVydrJozaXGrcst0iF3mVgKkhRWO05Mt%2Bfp%2F69Y%3D"
azcopy list "${azure_path}" --output-type json > ${bin_idx_list_path}

azure_path="https://provisioningte0256624006.blob.core.windows.net/noufs/allam_data_2-1_splits/?sp=racwdl&st=2023-11-16T09:08:26Z&se=2025-04-28T17:08:26Z&spr=https&sv=2022-11-02&sr=c&sig=02JMB0Kku5GH542GWKkirzToMeEg%2FCF16PBf9Tfnei4%3D"
azcopy list "${azure_path}" --output-type json > ${original_jsonl_path}

previous_line=""

declare -a bin_idx_filenames
declare -a original_filenames

while IFS= read -r line; do
    # Extract MessageContent using jq
    message_content=$(echo "$line" | jq -r '.MessageContent')

    if echo "$message_content" | grep -q "\.bin"; then
        previous_line="$message_content"
    elif echo "$message_content" | grep -q "\.idx"; then
        if [ -n "$previous_line" ]; then
            bin_text=$(echo "$previous_line" | sed -n 's/INFO: \(.*\)\.bin.*/\1/p')
            idx_text=$(echo "$message_content" | sed -n 's/INFO: \(.*\)\.idx.*/\1/p')

            echo "Bin and idx file name: $bin_text"

            bin_idx_filenames+=("$bin_text")
        fi
        previous_line=""
    fi
done < "$bin_idx_list_path"

while IFS= read -r line; do
    message_content=$(echo "$line" | jq -r '.MessageContent')
    jsonl_text=$(echo "$message_content" | sed -n 's/INFO: \(.*\)\.jsonl.*/\1/p')

    echo "Original jsonl file name: $jsonl_text"
    
    original_filenames+=("$jsonl_text")
done < "$original_jsonl_path"

if [ ! -d $output_data ]; then
    mkdir -p ${output_data}
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


extract_pattern() {
    echo "$1" | sed 's/_text_document$//'
}

sort_jsonl() {
    jq -S . $1 | sort
}

for file_without_ext in "${bin_idx_filenames[@]}"; do
    pattern=$(extract_pattern "$file_without_ext")

    bin_file="${file_without_ext}.bin"
    idx_file="${file_without_ext}.idx"
    echo "Searching for original jsonl file for $file_without_ext"
    for original_file in "${original_filenames[@]}"; do
        if [[ $original_file == $pattern* ]]; then
            echo "Match found for $file_without_ext: $original_file"
            echo "Downloading files..."

            original_jsonl_file="${original_file}.jsonl"
            original_jsonl_file_output="${output_data}/${original_jsonl_file}"

            bin_file_output="${output_data}/${bin_file}"
            if [ ! -f $bin_file_output ]; then
                azure_path="https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/${base_azure_data_dir}/${bin_file}?sp=racwdl&st=2023-11-13T10:57:16Z&se=2025-04-17T18:57:16Z&spr=https&sv=2022-11-02&sr=c&sig=Gg2nHVydrJozaXGrcst0iF3mVgKkhRWO05Mt%2Bfp%2F69Y%3D"
                azcopy copy "${azure_path}" "${bin_file_output}"
            fi
            
            idx_file_output="${output_data}/${idx_file}"
            if [ ! -f $idx_file_output ]; then
                azure_path="https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/${base_azure_data_dir}/${idx_file}?sp=racwdl&st=2023-11-13T10:57:16Z&se=2025-04-17T18:57:16Z&spr=https&sv=2022-11-02&sr=c&sig=Gg2nHVydrJozaXGrcst0iF3mVgKkhRWO05Mt%2Bfp%2F69Y%3D"
                azcopy copy "${azure_path}" "${idx_file_output}"
            fi

            if [ ! -f $original_jsonl_file_output ]; then
                azure_path="https://provisioningte0256624006.blob.core.windows.net/noufs/allam_data_2-1_splits/${original_jsonl_file}?sp=racwdl&st=2023-11-16T09:08:26Z&se=2025-04-28T17:08:26Z&spr=https&sv=2022-11-02&sr=c&sig=02JMB0Kku5GH542GWKkirzToMeEg%2FCF16PBf9Tfnei4%3D"
                azcopy copy "${azure_path}" "${original_jsonl_file_output}"
            fi

            detokenized="${output_data}/${file_without_ext}"

            echo "Detokenizing ${file_without_ext} and comparing against ${original_jsonl_file}..."

            python yz_detokenize.py --tokenizer_path "${tok_file}" --source_prefix_path "${detokenized}"
            
            if ! cmp -s <(sort_jsonl "${detokenized}.jsonl") <(sort_jsonl "${original_jsonl_file_output}"); then
                echo "$detokenized and $file2 are different."
                echo "$detokenized $original_jsonl_file_output" >> "${output_data}/diff.txt"
                echo "Differences:"
                diff <(sort_jsonl "$detokenized") <(sort_jsonl "$original_jsonl_file_output")
            else
                echo "No difference"
            fi

            rm -rf "${bin_file_output}"
            rm -rf "${idx_file_output}"
            rm -rf "${original_jsonl_file_output}"

        fi
    done
done

