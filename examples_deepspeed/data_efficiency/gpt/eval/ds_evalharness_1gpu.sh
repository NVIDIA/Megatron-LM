## CAUTION: first read Megatron-DeepSpeed/blob/main/examples_deepspeed/MoE/readme_evalharness.md
## and follow the steps of installation/data downloading.

## Code below only works when you run each evalharness task on a single GPU.
## For multi-GPU evalharness, check Megatron-DeepSpeed/blob/main/examples_deepspeed/MoE/ds_evalharness.sh
checkpoint_path=$1
config_path=$2
result_path=$3
rank=$4
tasks=$5
hostname=$6
master_port=$(( 12345 + ${rank} ))
batch_size=$7
num_fewshot=$8

mp_size=1
pp_size=1
no_pp="true"
ep_size=1

vocab_file="gpt2-vocab.json"
if [ ! -f "$vocab_file" ]; then
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
fi
merge_file="gpt2-merges.txt"
if [ ! -f "$merge_file" ]; then
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
fi

# export HF_DATASETS_OFFLINE=1

dir2=$(dirname "$checkpoint_path")
dirname=$(basename "$dir2")/$(basename "$checkpoint_path")
result_path="${result_path}/${dirname}"
mkdir -p $result_path
result_file="${result_path}/${tasks}_${num_fewshot}shot.json"

# Dummy arguments to make megatron happy. No need to configure them.
# The reason we don't need to configure them and many other arguments is
# because the eval framework will read the arguments from checkpoint file.
megatron_required_args="\
    --num-layers -1 \
    --hidden-size -1 \
    --num-attention-heads -1 \
    --seq-length -1 \
    --max-position-embeddings -1
"

command="../../../../tasks/eval_harness/evaluate.py \
    --load ${checkpoint_path} \
    --tensor-model-parallel-size ${mp_size} \
    --pipeline-model-parallel-size ${pp_size} \
    --moe-expert-parallel-size ${ep_size} \
    --vocab-file ${vocab_file} \
    --merge-file ${merge_file} \
    --micro-batch-size ${batch_size} \
    --no-load-optim \
    --no-load-rng \
    --inference \
    --disable-moe-token-dropping \
    --tokenizer-type GPT2BPETokenizer \
    --adaptive_seq_len \
    --eval_fp32 \
    --num_fewshot ${num_fewshot} \
    --task_list ${tasks} \
    --results_path ${result_file} \
    --deepspeed \
    --deepspeed_config ${config_path} \
    ${megatron_required_args} \
    "

if [[ "${no_pp}" = "true" ]]; then
command="${command} \
    --no-pipeline-parallel"
fi

launcher="deepspeed --include=$hostname:$rank --master_port=${master_port}"
$launcher $command &> "${result_path}/${tasks}_${num_fewshot}shot.log"