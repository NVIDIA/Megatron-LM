# SGEAT: Detoxify Larger-scale Language Models

This is the official code base for our NeurIPS 2022 paper:

[Exploring the Limits of Domain-Adaptive Training for Detoxifying Large-Scale Language Models](https://arxiv.org/abs/2202.04173)

Boxin Wang, Wei Ping, Chaowei Xiao, Peng Xu, Mostofa Patwary, Mohammad Shoeybi, Bo Li, Anima Anandkumar, Bryan Catanzaro


## Citation

```
@article{WangExp2022,
  title={Exploring the Limits of Domain-Adaptive Training for Detoxifying Large-Scale Language Models},
  author={Wang, Boxin and Ping, Wei and Xiao, Chaowei and Xu, Peng and Patwary, Mostofa and Shoeybi, Mohammad and and Li, Bo and Anandkumar, Anima and Catanzaro, Bryan},
  journal={NeurIPS},
  year={2022}
}
```

## Usage

### Prepare your environment

The project environment is based on the standard [nvcr docker](nvcr.io/nvidia/pytorch:21.12-py3) of version `nvcr.io/nvidia/pytorch:21.12-py3`.

To run Perspective API, you need to install `google-api-python-client`
```bash
pip install --upgrade google-api-python-client
```

### Self Generation

#### SGEAT (Standard)
To perform unconditional generation for a Megatron LM, we provide an example script for 1.3B LM.

```bash
#                                                                              [num of samples]     [model checkpoint]          [random seed]
bash examples/detxoify_lm/self_generation/selfgenerate-1.3b-unconditional.sh       1000          checkpoints/gpt3/gpt3-1.3b/      2333
```
This will generate a jsonl file of  1000 generated text (as a toy example) at `selfgeneration/unconditional_generation_gpt3-1.3b/2333.out`. 

Note that you may want to set your own gpt2 vocab and merge file dir, as well as your output data dir in `selfgenerate-1.3b-unconditional.sh`.

### Annotation

We then use Perspective API to annotate the self generated corpus. Note that you need to fill in your own Perspective API key in the `examples/detoxify_lm/perspective_api_annotate.py`. 

```bash
python examples/detxoify_lm/perspective_api_annotate.py --data-path [input-data-path] --out-path [output-data-path] --workers 70
```

For example,

```bash
python examples/detxoify_lm/annotations/perspective_api_annotate.py --data-path  selfgeneration/unconditional_generation_gpt3-1.3b/2333.out --out-path  selfgeneration/unconditional_generation_gpt3-1.3b/2333.annotated.out --workers 70
```

### Filtering

We then filter the self annotated generated corpus to get the most nontoxic 50% of the corus.

For example,
```bash
python examples/detxoify_lm/annotations/filter-selfgeneration.py --data-path  selfgeneration/unconditional_generation_gpt3-1.3b/2333.annotated.out --out-path  selfgeneration/unconditional_generation_gpt3-1.3b/2333.annotated.nontoxic.out
```

This will generate a jsonl file of 500 text of the lowest toxicity (as a toy example) at `selfgeneration/unconditional_generation_gpt3-1.3b/2333.annotated.nontoxic.out`. 


### Preprocess

We then preprocess the dataset so that Megatron LM can use the dumped dataset to fine-tune.

```
bash examples/detxoify_lm/annotations/preprocess.sh selfgeneration/unconditional_generation_gpt3-1.3b/2333.annotated.nontoxic.out selfgeneration/unconditional_generation_gpt3-1.3b/2333.annotated.nontoxic
```

This will generate two files as follows
```bash
selfgeneration/unconditional_generation_gpt3-1.3b/2333.annotated.nontoxic_text_document.idx
selfgeneration/unconditional_generation_gpt3-1.3b/2333.annotated.nontoxic_text_document.bin
```
which will be used in the following domain-adative training step.

### Fine-tuning

We then use the preprocess dataset as input to fine-tune our Megatron-LM. 
```bash
#                                                                          [fine-tuning dataset]                                                                      [output-dir]                             [lr]    [bs]      [train-iters]                       [load checkpoint]
bash examples/detxoify_lm/finetune_gpt_distributed-1.3b.sh    selfgeneration/unconditional_generation_gpt3-1.3b/2333.annotated.nontoxic_text_document         gpt3-1.3b-toy-example-lr-2e-5-bs-512             2e-5     512            78                          checkpoints/gpt3/gpt3-1.3b
```

This will dump the final checkpoint in `$SHARE_DATA/gpt3-1.3b-toy-example-lr-2e-5-bs-512`. (`$SHARE_DATA` is your current work dir, default to `$PWD`)

### Evaluation

We then use the fine-tuned checkpoint to perform conditional generation given RealToxicityPrompts:

```bash
#                                                 [input-prompts]                          [model-checkpoint]
bash examples/detxoify_lm/generate-1.3b.sh     augmented_prompts.jsonl      $SHARE_DATA/gpt3-1.3b-toy-example-lr-2e-5-bs-512
```
For example, this will generate the continuations in the file `augmented_prompts.jsonl_output_gpt3-1.3b-toy-example-lr-2e-5-bs-512_seed_31846.jsonl` (seed is a random generated number).

Note that the input prompts are augmented so that each prompts appear 25 times to calculate the Expected Maximum Toxicity over 25 generations and Toxicity Probability,  

We then use Perspective API to evaluate the Expected Maximum Toxicity and Toxicity Probability.   

```bash
python examples/detxoify_lm/perspective_api.py --data-path "augmented_prompts.jsonl_output_gpt3-1.3b-toy-example-lr-2e-5-bs-512_seed_31846.jsonl" --prompt-path prompts.jsonl --workers 30
```