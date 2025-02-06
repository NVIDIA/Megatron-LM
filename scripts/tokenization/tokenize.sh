#!/bin/bash

#SBATCH --account=a-a06
#SBATCH --time=07:59:59
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --environment=/capstor/scratch/cscs/asolergi/Megatron-LM/container/datatrove.toml # WARN(tj.solergibert) Modify path to your own file
#SBATCH --no-requeue

input_folder=$1
output_folder=$2
tokenizer=$3
logging_dir=$4
CSV_RESULTS_FILE=$5
paths_file=$6
number_of_tasks=$7
MEGATRON_LM_DIR=$8

# Setup ENV
export HF_HUB_ENABLE_HF_TRANSFER=0
# Setup directories
rm -rf $output_folder
mkdir -p $output_folder

echo "START TIME: $(date) | Preprocessing $paths_file with $number_of_tasks tasks per node with the $tokenizer tokenizer. Storing tokenized dataset in $output_folder"
start_s=`date`
start=`date +%s`
numactl --membind=0-3 python3 $MEGATRON_LM_DIR/scripts/tokenization/preprocess_megatron.py --tokenizer-name-or-path $tokenizer --output-folder $output_folder --logging-dir $logging_dir --n-tasks $number_of_tasks --dataset $input_folder --paths-file $paths_file
end=`date +%s`
end_s=`date`
echo "FINISH TIME: $(date) | Preprocessed $paths_file ! Stored in $output_folder"

# Stats
wc=$((end-start))
dataset_total_size=$(python3 $MEGATRON_LM_DIR/scripts/tokenization/compute_dump_size.py $paths_file)
processed_total_size=$(du -shLb $output_folder | cut -f1)
bw=$(python3 -c "print($dataset_total_size/$wc)")
total_tokens_processed=$(($(du -shLcb $output_folder/*.bin | tail -n1 | sed -r 's/([^0-9]*([0-9]*)){1}.*/\2/')/4))
throughput=$(python3 -c "print($total_tokens_processed/$wc)")
echo "$SLURM_JOB_ID,$(hostname),$start_s,$end_s,$paths_file,$output_folder,$dataset_total_size,$processed_total_size,$number_of_tasks,$wc,$bw,$total_tokens_processed,$throughput"
echo "$SLURM_JOB_ID,$(hostname),$start_s,$end_s,$paths_file,$output_folder,$dataset_total_size,$processed_total_size,$number_of_tasks,$wc,$bw,$total_tokens_processed,$throughput" >> $CSV_RESULTS_FILE

sleep 10
ls -lS $output_folder
mv $paths_file $(dirname $(dirname $paths_file))/completed_dumps
