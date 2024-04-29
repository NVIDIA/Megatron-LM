#! /bin/bash
echo "------ARGUMENTS LIST --------"
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
   echo "$KEY=$VALUE"
done
echo "---------------------------------"

set -exo pipefail

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

command="export CUDA_DEVICE_MAX_CONNECTIONS=1; export HF_HOME=/workspace/huggingface/hub;"

set +x
# Runs the "126m" parameter model

build_run_cmd() {
  #DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NUM_NODES"
  [[ -n "$RUN_CMD" ]] && run_cmd=$RUN_CMD || run_cmd="python examples/nlp/language_modeling/megatron_gpt_pretraining.py"
  nemo_run_cmd="$run_cmd \
        trainer.num_nodes=$NUM_NODES \
        trainer.devices=$GPUS_PER_NODE \
        trainer.max_steps=$MAX_STEPS \
        trainer.val_check_interval=$MAX_STEPS \
        trainer.limit_val_batches=50 \
        trainer.max_epochs=null \
        trainer.precision=bf16 \
        model.num_layers=12 \
        model.hidden_size=768 \
        model.num_attention_heads=12 \
        model.micro_batch_size=$MBS \
        model.global_batch_size=$GBS \
        model.tensor_model_parallel_size=$TP_SIZE \
        model.pipeline_model_parallel_size=$PP_SIZE \
        model.virtual_pipeline_model_parallel_size=${VP_SIZE:-null} \
        model.encoder_seq_length=2048 \
        model.max_position_embeddings=2048 \
        model.ffn_hidden_size=3072 \
        model.mcore_gpt=True \
        model.apply_query_key_layer_scaling=True \
        model.megatron_amp_O2=True \
        model.data.data_prefix=[] \
        model.data.data_impl=mock \
        model.data.splits_string=[99990,8,2] \
        model.optim.name=distributed_fused_adam \
        model.optim.weight_decay=0.1 \
        exp_manager.create_checkpoint_callback=False \
        ${ADDITIONAL_PARAMS:+$ADDITIONAL_PARAMS}"
}

build_run_cmd
command="$command $nemo_run_cmd"
eval $command
