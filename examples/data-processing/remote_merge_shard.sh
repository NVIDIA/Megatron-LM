az_blob_input_folder=$1
az_output_folder_name=$2
sas_token=$3
output_file=$4
shift
shift
shift
shift
for arg in "$@"; do
    azcopy copy "$az_blob_input_folder/$arg?$sas_token" "." --recursive
    cat $arg >> $output_file
done
azcopy copy $output_file "$az_output_folder_name/?$sas_token"