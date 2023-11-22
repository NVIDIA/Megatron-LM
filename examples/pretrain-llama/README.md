# Continue Pretrain LLaMa

## Sharded data re-naming

Just a simple script to renaming your `jsonl`` shards.

```
python examples/pretrain-llama/data-processing/clean.py
```

"Please add a domain or sub-domain name to each of the file names, if it's not already included in the shard names. At this stage, you may need to edit the code based on your requirements. This is just an example of the code. This step is important because later on based on domain (not subdomain) we will sample the data. So make sure you plan the shard (`*.jsonl`) file names are correct. Currently we are following the following naming structure. 

```
{language}_{domain}_{subdomain}_split_{split_id}_text_document_dc={document_count}_sc={sentence_count}_tc={token_count}"
```

## Data Processing

```
bash examples/pretrain-llama/data-processing/process_data.sh
```

The `process_data.sh`` script will run a multi-processor job runner. The objective of this script is to tokenize each dataset as a separate process. Each separate process can utilize multiple CPUs to process that shard of data. Please check the arguments of the 'examples/pretrain-llama/data-processing/multiprocess_runner.py' to become familiar with the options.

## `*.bin`, `*.idx` data naming

Once we are done with creating binary and index files, then we embed some simple metadata with the file names.

```
python examples/pretrain-llama/data-processing/count_token_and_rename_bin_idx.py --source_prefix_path "../DUMPED/*" 
```

Here `--source_prefix_path` is the folder where bin and index files are. This will add  `_dc={document_count}_sc={sentence_count}_tc={token_count}` with the file name. 

## Calculate probability of the shards.

Once you have the file names, please create a data signature (a list of name prefixes of the shards) from the names of the `*.bin`/`*.idx` files and save it as a JSON file for future use. Here is an example of two data signatures.

```
cat examples/pretrain-llama/data-signatures/allam_data_2-1_splits-llama2-indexed_data.json
cat examples/pretrain-llama/data-signatures/allam_data_2-1_splits-llama2-VE-indexed_data.json
```

Then, create a training folder (e.g., examples/pretrain-llama/training/llama_en_ar_v1). All the contents of the training folder should be sufficient to reproduce the experiments. Here are the files in the training folder.

1. `DockerFile`: For Environment. 
2. `data_ratio.json`: The format of the data_ration file would be either of the following, 

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

3. Data signature file : Your data signature file. Check the example in `examples/pretrain-llama/training/llama_en_reasoning_ar_lr_hyp_tune/en_reasoning_and_arabic_files.json`

4. Exclude iterator: List of iterator you want to remove from the training. Check the example in `examples/pretrain-llama/training/llama_en_ar_v1/exclude_iterator.json`

5. Language selection probability: Check out the language selection probability. Check the example in `examples/pretrain-llama/training/llama_en_reasoning_ar_lr_hyp_tune/lang_prob.json`

6. Bash script: A bash script to run the experiment. Check the example in `examples/pretrain-llama/training/llama_en_reasoning_ar_lr_hyp_tune/llama_en_reasoning_ar_lr_hyp_tune.sh`

7. Bash script: A bash script to run the experiment. Check the example in `examples/pretrain-llama/training/llama_en_reasoning_ar_lr_hyp_tune/llama_en_reasoning_ar_lr_hyp_tune.sh`

8. Azure job file: An azure job file. Check the example in `examples/pretrain-llama/training/llama_en_reasoning_ar_lr_hyp_tune/llama_en_reasoning_ar_lr_hyp_tune.yaml`

Once we have all the files, we can create the iterator probability. To calculate iterator selection probability check, 


```
python examples/pretrain-llama/data-processing/data_ratio_from_file.py --help
```


