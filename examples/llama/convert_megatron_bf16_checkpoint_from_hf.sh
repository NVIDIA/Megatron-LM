#/usr/bin/bash
SCRIPT_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$SCRIPT_ROOT/../..

export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

TP=${TP:-1}
PP=${PP:-1}

MODEL=Llama-2-7b-hf
MODEL_TYPE=llama2-7B

# create soft links to /workspace/models
MODEL_DIR=/workspace/models

SAVER=megatron

dtype="bf16"

HF_MODEL_DIR=$MODEL_DIR/$MODEL
OUTPUT=$MODEL_DIR/$MODEL-to-$SAVER-tp$TP-pp$PP-$dtype

# old transformer use this path
# TOKENIZER_MODEL=$HF_MODEL_DIR/tokenizer.model

TOKENIZER_MODEL=$HF_MODEL_DIR

dtype_opt=(
  --$dtype
)

python $PROJECT_ROOT/tools/checkpoint/convert.py \
  --model-type GPT \
  --loader llama_mistral \
  --model-size $MODEL_TYPE \
  --saver $SAVER \
  --target-tensor-parallel-size ${TP} \
  --target-pipeline-parallel-size ${PP} \
  --checkpoint-type hf \
  --load-dir $HF_MODEL_DIR \
  --save-dir $OUTPUT \
  --tokenizer-model ${TOKENIZER_MODEL} \
  ${dtype_opt[@]}