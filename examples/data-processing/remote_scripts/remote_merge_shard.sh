#!/bin/bash

# Validate the number of arguments
if [ "$#" -lt 5 ]; then
    echo "Usage: $0 <input_folder> <output_folder_name> <sas_token> <output_file> <compute_target> [additional_files...]"
    exit 1
fi

format_number() {
    echo $1 | sed ':a;s/\B[0-9]\{3\}\>/_&/;ta'
}

# Assigning input arguments to variables
input_folder=$1
output_folder_name=$2
sas_token=$3
compute_target=$4
output_file=$5

shift 5

echo "Compute target: $compute_target"

# Loop through additional arguments
for arg in "$@"; do
    if [ "$compute_target" = "azure" ]; then
        echo "Running azcopy for $arg"
        if ! azcopy copy "${input_folder}/${arg}?${sas_token}" "."; then
            echo "Error copying $arg from Azure Blob Storage. Exiting."
            exit 1
        fi
    fi
    if [ -f "$arg" ]; then
        cat "$arg" >> "$output_file"
    elif [ -f $(basename "$arg") ]; then
        filename=$(basename "$arg")
        cat "$filename" >> "$output_file"
    else
        echo "Warning: File $arg not found after copy."
    fi
done

word_count=$(jq -r '.text' $output_file | wc -w)
formatted_word_count=$(format_number $word_count)

new_output_file="${output_file%.jsonl}_wc=${formatted_word_count}.jsonl"
mv $output_file $new_output_file
output_file=$new_output_file

if [ "$compute_target" = "local" ]; then
    echo "Your file is merged as $output_file"
    echo "Moving $output_file to $output_folder_name"
    mv $output_file $output_folder_name
elif [ "$compute_target" = "azure" ]; then
    echo "Copying merged file $output_file to Azure Blob Storage."
    echo "Running azcopy copy \"$output_file\" \"${output_folder_name}/?${sas_token}\""
    if ! azcopy copy "$output_file" "${output_folder_name}/?${sas_token}"; then
        echo "Error copying merged file to Azure Blob Storage. Exiting."
        exit 1
    fi
else
    echo "Compute target not implemented."
    exit 1
fi
