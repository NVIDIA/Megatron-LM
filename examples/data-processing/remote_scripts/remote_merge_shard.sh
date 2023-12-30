#!/bin/bash
set -e

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
input_file=$6

# if there is an input file 

output_file_wo_ext="${output_file%.*}"
input_file_start="input_${output_file_wo_ext}.txt"


if [[ "$input_file" == "$input_file_start"* ]]; then
    input_files=()
    while IFS= read -r line; do
        if [[ -n "$line" ]]; then
            input_files+=("$line")
        fi
    done < "$input_file"
    IFS=$'\n' sorted_input_files=($(sort <<<"${input_files[*]}"))
    unset IFS
else
    shift 5
    input_files=("$@")
fi

echo "Output file: "$output_file
echo "Compute target: $compute_target"
# printf "%s\n" "${input_files[@]}"
echo "Number of input files: ${#input_files[@]}"

OLD_IFS=$IFS
# Set IFS to newline only
IFS=$'\n'
# Loop through additional arguments
for arg in ${input_files[@]}; do
    arg=$(echo "$arg" | tr -d '\n')
    echo "processing $arg"
    if [ "$compute_target" = "azure" ]; then
        echo "Running azcopy for $arg"
        if ! azcopy copy "${input_folder}/${arg}?${sas_token}" "."; then
            echo "Error copying $arg from Azure Blob Storage. Exiting."
            exit 1
        fi
    fi
    # Check if file exists and is readable
    if [ -f "$arg" ]; then
        cat "$arg" >> "$output_file"
    elif [ -f $(basename "$arg") ]; then
        filename=$(basename "$arg")
        cat "$filename" >> "$output_file"
    else
        echo "Warning: File $arg not found after copy."
    fi
done
IFS=$OLD_IFS

# word_count=$(python $output_file)
# formatted_word_count=$(format_number $word_count)

# new_output_file="${output_file%.jsonl}_wc=${formatted_word_count}.jsonl"
# mv $output_file $new_output_file
# output_file=$new_output_file

if [ "$compute_target" = "local" ]; then
    echo "Your file is merged as $output_file"
    echo "Moving $output_file to $output_folder_name/"
    mv $output_file $output_folder_name/
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
