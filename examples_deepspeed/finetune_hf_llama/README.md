## Example of Finetuning LLAMA-7B from Hugging Face Weights

### Dataset
You can access the dataset from [here](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json).

### Pre-trained Weights
The pre-trained weights can be found at [Hugging Face - LLAMA-7B](https://huggingface.co/huggyllama/llama-7b).

### Usage:

#### 1. Converting Hugging Face Model Weights to Megatron-Deepspeed Model
```bash
bash examples_deepspeed/finetune_hf_llama/finetune_llama.sh convert_hf2mds
```
This command writes the Hugging Face model weights into the Megatron-Deepspeed model and saves it. You can adjust the parallel configuration in the script.```convert_mds2hf``` can convert a Megatron-Deepspeed model into the Hugging Face format

#### 2. Fine-tuning Process
```bash
bash examples_deepspeed/finetune_hf_llama/finetune_llama.sh
```
Execute this command to initiate the finetuning process. The task originates from [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca.git).



