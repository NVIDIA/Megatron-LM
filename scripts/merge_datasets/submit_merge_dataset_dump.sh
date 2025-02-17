TOKENIZER_NAME=Llama-3.1-70B
DATASET_NAME=fineweb-2

MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM
PATH_TO_SLURM_LOGGING_DIR=$MEGATRON_LM_DIR/logs/slurm/merge-$TOKENIZER_NAME-$DATASET_NAME

PATH_TO_INPUT_FOLDER=/iopsstor/scratch/cscs/asolergi/Megatron-LM/datasets/$DATASET_NAME/tokenized-dir-link
PATH_TO_OUTPUT_FOLDER=/capstor/store/cscs/swissai/a06/datasets_tokenized/nemo # /iopsstor/scratch/cscs/$USER/datasets
DATASET_OUTPUT_FOLDER_NAME=$PATH_TO_OUTPUT_FOLDER/$TOKENIZER_NAME/$DATASET_NAME-merge

mkdir -p $PATH_TO_SLURM_LOGGING_DIR
mkdir -p $DATASET_OUTPUT_FOLDER_NAME

for dump_folder in "$PATH_TO_INPUT_FOLDER"/*; do
    dump_prefix=$(basename $dump_folder)
    sbatch --job-name=merge-$DATASET_NAME-$dump_prefix --output=$PATH_TO_SLURM_LOGGING_DIR/R-%x-%j.out --error=$PATH_TO_SLURM_LOGGING_DIR/R-%x-%j.err $MEGATRON_LM_DIR/scripts/merge_datasets/merge.sh $dump_folder $DATASET_OUTPUT_FOLDER_NAME/$dump_prefix-merged $MEGATRON_LM_DIR
done