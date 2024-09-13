We will introduce the llm finetuning with huggngface.

## 0. Preparation

- Setting aws-cli

https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-welcome.html

- Python and huggingface environment

https://huggingface.co/docs/transformers/ja/installation

https://hub.docker.com/r/huggingface/transformers-pytorch-gpu

## 1. Download the dataset

At here, we use only 1 file.

```bash
aws s3 cp s3://abeja-cc-ja/common_crawl_01.jsonl data/
```

If you will use all data, it is recommended that using `aws s3 sync` command.

## 2.Check dataset

You can check the data after download.

```bash
head -n 1 data/common_crawl_01.jsonl
```

You can find such output.

```bash
{"url":"https:\/\/xxxxx/","content":"### 今日のご飯\n こんにちは、今日のご飯についてご紹介します。..."}
```

## 3. Training

```python
import json

import torch
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Function to load JSONL dataset
def load_jsonl_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data = [json.loads(line) for line in lines]
    return Dataset.from_list(data)

# Paths to the dataset files
# for small data check we sampled data
# by `head -n 100000 common_crawl_01.jsonl > sampled_common_crawl_01.jsonl`
train_file_path = 'sampled_common_crawl_01.jsonl'

# Load datasets
train_dataset = load_jsonl_dataset(train_file_path)

# Define tokenizer and model
model_name = 'abeja/gpt2-large-japanese'  # Example using GPT-2
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
model.to(device)
tokenizer.pad_token = tokenizer.eos_token

# Preprocess the dataset
def preprocess_function(examples):
    # Tokenize the text and truncate to max length
    return tokenizer(examples['content'], truncation=True, padding='max_length', max_length=128)

train_dataset = train_dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    dataloader_pin_memory=False
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Train the model
trainer.train()

```

In recent LLMs, distributed learning using multiple machine nodes has become common due to the large size of the models. When using distributed learning, libraries such as Megatron-LM and DeepSpeed are utilized.

However, even in such cases, this dataset should be just as easy to handle.
