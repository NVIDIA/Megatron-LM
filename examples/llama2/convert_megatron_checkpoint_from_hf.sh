#/usr/bin/bash
SCRIPT_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$SCRIPT_ROOT/../..

TP=${TP:-1}
PP=${PP:-1}

MODEL=Llama-2-7b-hf
MODEL_TYPE=llama2-7B

HF_MODEL_DIR=/workspace/models/$MODEL
OUTPUT=/workspace/models/$MODEL-to-megatron-tp$TP-pp$PP
TOKENIZER_MODEL=/workspace/models/$MODEL/tokenizer.model

python $PROJECT_ROOT/tools/checkpoint/convert.py \
  --model-type GPT \
  --loader llama_mistral \
  --model-size $MODEL_TYPE \
  --saver megatron \
  --target-tensor-parallel-size ${TP} \
  --target-pipeline-parallel-size ${PP} \
  --checkpoint-type hf \
  --load-dir $HF_MODEL_DIR \
  --save-dir $OUTPUT \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --bf16