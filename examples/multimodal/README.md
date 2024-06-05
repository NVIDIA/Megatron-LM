# Multimodal Example

NOTE: This is work in progress and not fully functional yet.

## Setup

### Docker container

You can build a docker container using `examples/multimodal/Dockerfile` to run this example.

### Vision model

This example uses the OpenAI CLIP `ViT-L/14@336px` Vision model. To download the weights from OpenAI and convert them to a format that can be loaded in megatron, please run the following:

```
python examples/multimodal/clip_converter.py --download-root /some/download/folder --output /some/output/folder --tensor-parallel-size 4
```

## Training

### Pretraining

Run the following script:
```
examples/multimodal/pretrain_8b.sh
```

### SFT

Run the following script:
```
examples/multimodal/sft_8b.sh
```

## Evaluation

### Generation

Run the following script:

```
examples/multimodal/text_generation_8b.sh --input-image-path /path/to/input/images --output-path /some/output/directory --model-path /path/to/model.pt --tokenizer-path /path/to/tokenizer.model --gt-path /path/to/groundtruth/file --task generation-task-name
```

### COCO captioning

First, run text generation using `--task captioning`. Then, run the following command:

```
python examples/multimodal/evaluate_coco.py --input-path /output/directory/from/generation --groundtruth-path /path/to/groundtruth/file
```

### TextVQA

First, run text generation using `--task TextVQA`. Then, run the following command:

```
python examples/multimodal/evaluate_textvqa.py --input-path /output/directory/from/generation --groundtruth-path /path/to/groundtruth/file
```

### VQAv2

First, run text generation using `--task VQAv2`. Then, run the following command:

```
python examples/multimodal/evaluate_textvqa.py --input-path /output/directory/from/generation --groundtruth-path /path/to/groundtruth/file --question-path /path/to/question/file
```
