# Minimal VLM training

This is a minimal visual language model training code in pure Megatron style.

## Step 1

Download LLaMA weights: https://huggingface.co/meta-llama/Llama-2-70b-chat-hf。

The EvaViT model weights will be automatically downloaded by [SwissArmyTransformer](https://github.com/THUDM/SwissArmyTransformer).

You can install SAT with:

```
git clone https://github.com/THUDM/SwissArmyTransformer
cd SwissArmyTransformer
pip install . --no-deps
```

Then, transform LLaMA and EvaViT model into megatron format:

```
python tools/checkpoint/convert.py --model-type GPT --loader llama2 --saver mcore --load-dir /path/to/Llama-2-70b-chat-hf --save-dir /path/to/save/llama2-70b-chat-megatron-mcore-tp8-pp2-first36 --tokenizer-model /path/to/Llama-2-70b-chat-hf/tokenizer.model --target-tensor-parallel-size 8 --target-pipeline-parallel-size 2 --model-size 70Bf --checkpoint-type hf --first-pipeline-num-layers 36
python tools/checkpoint/convert.py --model-type EVA --loader eva_sat --saver eva_mcore --load-dir eva-clip-4b-14-x-drop-last-layer --save-dir /path/to/save/eva2-clip-224-mcore-tp8-pp1 --target-tensor-parallel-size 8 --target-pipeline-parallel-size 1
```

## Step 2

Download LLaVA dataset [`metadata.json`](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md) and `images.zip`.

Run training: (You should change th /path/to into the real path of your file.)

```
bash pretrain_llama2_70b_tp8_pp2.sh
```

If you want to use sequence parallel and context parallel, you need to change --image-seq-length from 257 to 256, which will omit the [CLS] token of ViT.

## Step 3

Run inference: (You should change th /path/to into the real path of your file.)

```
bash run_inference_70b_tp8_pp2.sh # server
python llava_inference_cli.py 127.0.0.1:5000 # client
```