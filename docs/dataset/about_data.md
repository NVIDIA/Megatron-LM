# ABEJA CC JA

This describes the format and gives simple examples for getting started with ABEJA CC JA on AWS.

## Data Format

The dataset is saved in jsonl format. This decision prioritizes the convenience and compatibility of the data.

In jsonl file, there are two keys.

- url
    - The URL is the source site of the text data found on Common Crawl.
- content
    - The Content column contains the relevant text. The data to be used for training is in this column, and the text has been preprocessed.

Each jsonl data size is about 53GB.

Therefore, please be careful when downloading to machines with small storage capacity.

## Example of use

The dataset format is simple, so there are many ways to read.

When you will use pandas such as checking data

```python
import pandas as pd

filepath = "abeja_cc_0001.jsonl"
df = pd.read_json(filepath, lines=True)

# df.head(1)
# | url | content |
# | - | - |
# | https://tech-blog.abeja.asia | このブログは.... |
```

Also Megatron-LM, this is major library for large language model training, it is easy to handle this format.

https://github.com/NVIDIA/Megatron-LM

This is sample script for preprocessing the jsonl data for megatron-lm

```python
#!/bin/bash
python tools/preprocess_data.py \
--input data/abeja_cc_0001.jsonl \
--output-prefix data/abeja_cc_ja \
--tokenizer-model tokenizers/xxxxx
--json-key content \
--append-eod
```

## Preprocess

About preprocessing, please refer to this blog.

https://tech-blog.abeja.asia/entry/abeja-nedo-project-part2-202405
