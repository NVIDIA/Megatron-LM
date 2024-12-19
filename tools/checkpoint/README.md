# Megatron-LM/Hugging Face Transformers Checkpoint Converter

## Convert Megatron-LM to Hugging Face Checkpoint

This tool converts distributed Megatron-LM checkpoints to the Hugging Face format, allowing easier loading and deployment using the Hugging Face Transformers library.

### Supported Recipes

The following recipes are supported by the Checkpoint converter:
- LLaMA 2, 3 & 3.1,
- Mixtral.

### Checkpoint Conversion Process

1. Load the single/distributed MLM checkpoint (`model_optim_rng.pt`) for rank 0 as the starting point.
2. Use the tokenizer metadata from the checkpoint to build:
    - LLaMA 3 & 3.1: Utilize Llama3Converter.
    - LLaMA 2, Mixtral: Utilize LlamaTokenizerFast.
3. Add special tokens to preserve the state and structure of the final tokenizer.
4. Prepare the `state_dict` and initialize the target data type (`dtype`).
5. Populate `LlamaConfig` or `MixtralConfig` using metadata from the checkpoint.
6. Create an output state dictionary to store layer-specific details for Hugging Face LLaMA or Mixtral recipes.
7. Translate each layer's names and value type conversions from the Megatron format to Transformers format.
8. Collect all distributed layers from tensor parallel and pipeline parallel ranks (traverse through `mp_rank_*`).
9. Collect all Mixtral distributed layers from expert parallel and MoE extended tensor parallel ranks.
10. Merge tensors to prepare a unified tensor for tensor-parallel-supported layers within a transformer block.
11. Save the merged tensors in `output_state_dict`.
12. Store the `output_state_dict` as a `safe_tensors` file in the output directory using Hugging Face Hub (HF_hub) along with the configuration.
13. Save Megatron-LM specific capacity bins parameters to `capacity_bins.pt`.

### Prerequisites
Python >= 3.10

```bash
export MEGATRON_LM_ROOT=/path/to/Megatron-LM
pip install $MEGATRON_LM_ROOT
pip install -r $MEGATRON_LM_ROOT/tools/checkpoint/requirements.txt
```

### Usage
To convert the distributed Megatron-LM checkpoints to the Hugging Face format, run the following command:

```bash
python $MEGATRON_LM_ROOT/tools/checkpoint/convert_mlm_to_hf_checkpoint.py \
--ckpt-dir-name "iter_0000004" \
--target-params-dtype "bf16" \
--source-model-type "llama3.1" \
--load-path "/path/to/mlm_checkpoints/" \
--save-path "/path/to/save/hf_checkpoints/"
```

## [LLaMA] Convert Hugging Face to Megatron-LM checkpoint

This tool converts Hugging Face checkpoints to the Megatron-LM for LLaMA recipes.

### Checkpoint conversion process

- Hugging face format supports single-node checkpoint and doesn't have a notion of tensor parallel and pipeline parallel checkpoint shards. Hugging Face format ideally stores model weights and model configuration required for inference, fine-tuning and transfer learning. It doesn't store any details regarding distributed training regime and its configuration as it is not relevant.
- This process undergoes a loader and saver approach, `convert.py` is a connecting link between both the loader and saver module shared by torch multiprocessing queue.
- Queue is a shared object between both the process. loader enqueues the loaded checkpoint states and saver fetches, persists them in a Megatron GPT format.
- Loader expects tensor parallel and pipeline parallel size to be 1 and saver can have any megatron compatible tensor and pipeline parallel ranks. (Easier to go from TP1 PP1 → TP8 PP4)


### Prerequisites
Python >= 3.10

```bash
export MEGATRON_LM_ROOT=/path/to/Megatron-LM
export PT_HPU_GPU_MIGRATION=1

pip install $MEGATRON_LM_ROOT
pip install -r $MEGATRON_LM_ROOT/megatron/core/requirements.txt
pip install -r $MEGATRON_LM_ROOT/tools/checkpoint/requirements.txt
```

### Usage
To convert the distributed Hugging Face checkpoints into Megatron-LM format, run the following command:

```bash
# To get more details on supported arguments.
python $MEGATRON_LM_ROOT/tools/checkpoint/convert.py --help

python $MEGATRON_LM_ROOT/tools/checkpoint/convert.py \
--bf16 \
--model-type GPT \
--loader llama_mistral \
--saver mcore \
--loader-transformer-impl local \
--saver-transformer-impl transformer_engine \
--target-tensor-parallel-size $TP \
--target-pipeline-parallel-size $DP \
--checkpoint-type hf \
--source-margs-file "/path/to/hf/checkpoints/source_megatron_args.json" \
--load-dir "/path/to/hf/checkpoints" \
--save-dir "/path/to/save/mlm/checkpoint" \
--tokenizer-model "/path/to/tokenizer/model" \
--model-size llama3-70B
```
