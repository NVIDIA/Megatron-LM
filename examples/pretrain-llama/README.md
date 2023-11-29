# Continue Pretrain LLaMa

## Convert the checkpoint

At first convert the checkpoint from huggingface to Megatron format with propoer model parallel. Check out the script in `pretrain-llama/checkpointing/convert_checkpoint.sh`. 

## Raw data processing

When data comes from a vendor and/or the data team, it usually comes in various shard sizes. Use the following code to merge different shards:

```
bash examples/pretrain-llama/data-processing/merge_shard/merge_shard.sh
```

Please check the arguments of `examples/pretrain-llama/data-processing/merge_shard/merge_shard.py` for a detailed description. It takes an Azure Blob Storage folder path and outputs the merged dataset in Azure Blob Storage. Processing `7`-`8` TB of data can often be cumbersome. This script launches smaller CPU jobs to the CPU cluster. 

## Sharded data re-naming

Just a simple script to renaming your `jsonl` shards.

```
python pretrain-llama/data-processing/clean.py
```

Please add a `domain` and/or `sub-domain` name to each file name, unless it is already included in the shard names. At this stage, you may need to edit the code to fit your requirements. Note that this is only an example of the code. This step is crucial because later on, we will sample the data based on the domain (not `subdomain`). Therefore, ensure that the shard (`*.jsonl`) file names are correctly planned. Currently, we are following this naming structure.

```
{language}_{domain}_{subdomain}_split_{split_id}_text_document_dc={document_count}_sc={sentence_count}_tc={token_count}"
```

## Data Processing

```
bash examples/pretrain-llama/data-processing/tokenize/process_data.sh
```

The `process_data.sh` script will run a multi-processor job runner. The objective of this script is to tokenize each dataset as a separate process. Each separate process can utilize multiple CPUs to process that shard of data. Please check the arguments of the `pretrain-llama/data-processing/tokenize/multiprocess_runner.py` to become familiar with the options.

This is good if you have large CPU nodes. But sometimes even with 128 core cpu nodes, the tokenization becomes slower. 
Another good alternative is to use `pretrain-llama/data-processing/tokenize/az_batch_tokenize.sh` this script. Similar to `merge_shard.py`, this script automatically process each of the shards in a different CPU. If you have large cpu cluster in azure, you can use this scirpt to run the data processing faster. 

## `*.bin`, `*.idx` data naming

Once we are done with creating binary and index files, then we embed some simple metadata with the file names.

```
python examples/pretrain-llama/data-processing/count_token_and_rename_bin_idx.py --source_prefix_path "../DUMPED/*" 
```

Here `--source_prefix_path` is the folder where bin and index files are. This will add  `_dc={document_count}_sc={sentence_count}_tc={token_count}` with the file name. 

## Calculate probability of the shards.

Once you have the file names, please create a data signature (a list of name prefixes of the shards) from the names of the `*.bin`/`*.idx` files and save it as a JSON file for future use. Here is an example of two data signatures.

```
cat pretrain-llama/data-signatures/allam_data_2-1_splits-llama2-indexed_data.json
cat pretrain-llama/data-signatures/allam_data_2-1_splits-llama2-VE-indexed_data.json
```

Then, create a training folder (e.g., `pretrain-llama/training/llama_en_ar_v1`). All the contents of the training folder should be sufficient to reproduce the experiments. Here are the files in the training folder.

1. `DockerFile`: For Environment. 
2. `data_ratio.json`: The format would be either of the following, 

```
{
    "en" : {
        "books": 1, 
        "code": 2, 
        "reasoning": 3, 
        "scientific": 2
    },
    "ar" : {
        "books": 3, 
        "encyclopedias": 2, 
        "news": 1, 
        "others": 1, 
        "transcribed": 2.5, 
        "web": 1
    }
}
```
or 

```
 {
    "books": 1, 
    "code": 2, 
    "reasoning": 3, 
    "scientific": 2
}
```
Each of the keys are a domain name. Based on different domain you can set your sampling ratio. The valued of the dictionary keys defines relative sampling rate of that domain. It should be in between, `1 to x`.

3. Data signature file : Your data signature file. Check the example in `pretrain-llama/training/llama_en_reasoning_ar_lr_hyp_tune/en_reasoning_and_arabic_files.json`

4. Exclude iterator: List of iterator you want to remove from the training. Check the example in `pretrain-llama/training/llama_en_ar_v1/exclude_iterator.json`

5. Language selection probability: Check out the language selection probability. Check the example in `pretrain-llama/training/llama_en_reasoning_ar_lr_hyp_tune/lang_prob.json`

Once we have all the files, we can create the iterator probability. To calculate iterator selection probability check, 


```
python pretrain-llama/data-processing/data_ratio_from_file.py --help
```

## Training

## Bash script

We run our pretraining job from a bash script. 

Notably, you souldn't use `torchrun` while running your code in azure. `azureml` handles that. Just run your code in python.
For pretraining you may have to copy-paste the iterator selection probability from the [Calculate probability of the shards.](#calculate-probability-of-the-shards).

Check the example in `pretrain-llama/training/llama_en_reasoning_ar_lr_hyp_tune/llama_en_reasoning_ar_lr_hyp_tune.sh`



## Azure job file

Check the example in `pretrain-llama/training/llama_en_reasoning_ar_lr_hyp_tune/llama_en_reasoning_ar_lr_hyp_tune.yaml`.  `azureml` job file launch the experiment with the appropiate `input` and `output`.

TODO: tensorboard still doesn't work.